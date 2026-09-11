"""
Calibration 적용 최종 버전.

- calibrated_model.joblib: 사용자에게 표시할 보정 낙상확률
- rf_model_for_shap.joblib: Local SHAP / SHAP interaction
- base_model.joblib: SMOTE + RandomForest 전체 학습 파이프라인(검증/재사용용)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

ROOT = Path(__file__).resolve().parent
BASE_MODEL_PATH = ROOT / "artifacts/base_model.joblib"
SHAP_MODEL_PATH = ROOT / "artifacts/rf_model_for_shap.joblib"
CALIBRATED_MODEL_PATH = ROOT / "artifacts/calibrated_model.joblib"
SCHEMA_PATH = ROOT / "artifacts/feature_schema.json"
METADATA_PATH = ROOT / "artifacts/model_metadata.json"

PREFIX_RULES = [
    ("동작수행 어려움_", "동작수행저하"),
    ("의사진단 만성질환 유무_", "만성질환"),
    ("치료 여부_", "만성질환"),
    ("RES_MAR_", "결혼상태"),
    ("Q1_", "주거형태"),
]

EXACT_MAP = {
    "근력상태_의자나 침대에 앉았다가 일어나기 5회 반복": "근력저하",
    "Nutritional": "영양상태",
    "ADL": "ADL 저하",
    "IADL": "IADL 저하",
    "평소 운동 여부": "운동 부족",
    "GDS": "우울",
    "인지기능_총점": "인지기능저하",
    "노인이 생활하는데 편리함 정도": "생활환경",
    "현재 살고있는 주택에 대한 만족도": "생활환경",
    "지역사회 환경 만족도_지역사회 환경 전반": "생활환경",
    "노인 조사 대상자 만연령": "고령",
    "노인 조사 대상자 성별": "성별",
    "노인 조사 대상자 교육연수": "교육수준",
    "노인가구형태": "독거",
    "만족도_경제상태": "경제상태",
    "현재 경제활동 여부": "경제활동",
    "평소의 건강상태": "전반적건강상태",
    "만족도_건강상태": "전반적건강상태",
    "지난 1년 간 음주 빈도": "음주",
    "일상생활의 불편함_시력": "시력저하",
    "일상생활의 불편함_청력": "청력저하",
    "현재 3개월 이상 복용하고 있는 의사처방약(종류)": "복용약물",
    "의사진단 만성질환 총 수": "만성질환",
}

_EXPLAINER_CACHE: dict[int, Any] = {}


def map_feature_to_factor(feature_name: str) -> str:
    for prefix, factor in PREFIX_RULES:
        if feature_name.startswith(prefix):
            return factor
    return EXACT_MAP.get(feature_name, feature_name)


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def load_models_and_schema(
    base_model_path: str | Path = BASE_MODEL_PATH,
    shap_model_path: str | Path = SHAP_MODEL_PATH,
    calibrated_model_path: str | Path = CALIBRATED_MODEL_PATH,
    schema_path: str | Path | None = SCHEMA_PATH,
    metadata_path: str | Path = METADATA_PATH,
):
    base_model = joblib.load(base_model_path)
    shap_model = joblib.load(shap_model_path)
    calibrated_model = joblib.load(calibrated_model_path)
    metadata = _load_json(metadata_path)

    feature_order: list[str] | None = None
    if schema_path is not None and Path(schema_path).exists():
        feature_order = [str(x) for x in _load_json(schema_path)["feature_order"]]
    elif hasattr(shap_model, "feature_names_in_"):
        feature_order = [str(x) for x in shap_model.feature_names_in_]

    if not feature_order:
        raise ValueError(
            "feature_schema.json이 없고 SHAP 모델에도 feature_names_in_가 없습니다. "
            "학습 당시 52개 변수 순서가 필요합니다."
        )

    return base_model, shap_model, calibrated_model, feature_order, metadata


def get_explainer(model):
    key = id(model)
    if key not in _EXPLAINER_CACHE:
        _EXPLAINER_CACHE[key] = shap.TreeExplainer(model)
    return _EXPLAINER_CACHE[key]


def _positive_class_index(model, positive_label: int | str = 1) -> int:
    classes = list(getattr(model, "classes_", []))
    if classes and positive_label in classes:
        return classes.index(positive_label)
    if len(classes) == 2:
        return 1
    raise ValueError(f"양성 클래스 위치를 판단할 수 없습니다. classes_={classes}")


def _extract_class_vector(values: Any, class_index: int, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[class_index]).reshape(-1, n_features)[0]
    arr = np.asarray(values)
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 3:
        if arr.shape[1] == n_features:
            return arr[0, :, class_index]
        if arr.shape[2] == n_features:
            return arr[class_index, 0, :]
    raise ValueError(f"지원하지 않는 SHAP values shape: {arr.shape}")


def _extract_interaction_matrix(values: Any, class_index: int, n_features: int) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[class_index]).reshape(-1, n_features, n_features)[0]
    arr = np.asarray(values)
    if arr.ndim == 3 and arr.shape[1:] == (n_features, n_features):
        return arr[0]
    if arr.ndim == 4:
        if arr.shape[1] == n_features and arr.shape[2] == n_features:
            return arr[0, :, :, class_index]
        if arr.shape[2] == n_features and arr.shape[3] == n_features:
            return arr[class_index, 0, :, :]
    raise ValueError(f"지원하지 않는 SHAP interaction shape: {arr.shape}")


def _row_frame(user_row: dict, feature_order: list[str]) -> pd.DataFrame:
    missing = [name for name in feature_order if name not in user_row]
    if missing:
        raise ValueError(f"user_row에 다음 피처가 빠져 있습니다: {missing}")
    return pd.DataFrame([user_row], columns=feature_order)


def compute_local_shap_raw(user_row: dict, model, feature_order: list[str]) -> list[dict]:
    X = _row_frame(user_row, feature_order)
    class_index = _positive_class_index(model)
    values = _extract_class_vector(
        get_explainer(model).shap_values(X), class_index, len(feature_order)
    )
    result = []
    for feature_name, shap_value in zip(feature_order, values):
        value = X.iloc[0][feature_name]
        result.append({
            "feature": feature_name,
            "feature_value": None if pd.isna(value) else value.item() if hasattr(value, "item") else value,
            "legacy_factor": map_feature_to_factor(feature_name),
            "shap_value": float(shap_value),
            "positive_risk_contribution": float(max(shap_value, 0.0)),
        })
    result.sort(key=lambda item: abs(item["shap_value"]), reverse=True)
    return result


def compute_local_shap_interaction_pairs(user_row: dict, model, feature_order: list[str]):
    X = _row_frame(user_row, feature_order)
    try:
        raw = get_explainer(model).shap_interaction_values(X)
    except Exception as exc:
        return [], f"interaction_not_available: {type(exc).__name__}: {exc}"

    try:
        class_index = _positive_class_index(model)
        matrix = _extract_interaction_matrix(raw, class_index, len(feature_order))
    except Exception as exc:
        return [], f"interaction_shape_error: {type(exc).__name__}: {exc}"

    pairs = []
    for i in range(len(feature_order)):
        for j in range(i + 1, len(feature_order)):
            value = float(2.0 * matrix[i, j])
            if abs(value) <= 1e-12:
                continue
            pairs.append({
                "feature_1": feature_order[i],
                "feature_2": feature_order[j],
                "legacy_factor_1": map_feature_to_factor(feature_order[i]),
                "legacy_factor_2": map_feature_to_factor(feature_order[j]),
                "interaction_value": value,
                "positive_interaction_contribution": float(max(value, 0.0)),
            })
    pairs.sort(key=lambda item: abs(item["interaction_value"]), reverse=True)
    return pairs, None


def explain_user_row(
    user_row: dict,
    shap_model,
    calibrated_model,
    feature_order: list[str],
    metadata: dict,
    row_index: int | None = None,
) -> dict:
    X = _row_frame(user_row, feature_order)

    shap_class_index = _positive_class_index(shap_model)
    calibrated_class_index = _positive_class_index(calibrated_model)

    raw_probability = float(shap_model.predict_proba(X)[0, shap_class_index])
    calibrated_probability = float(
        calibrated_model.predict_proba(X)[0, calibrated_class_index]
    )
    threshold = float(metadata["selected_threshold"])
    predicted_class = int(calibrated_probability >= threshold)

    main_effects = compute_local_shap_raw(user_row, shap_model, feature_order)
    legacy: dict[str, float] = {}
    for item in main_effects:
        legacy[item["legacy_factor"]] = legacy.get(item["legacy_factor"], 0.0) + item["shap_value"]
    interactions, interaction_warning = compute_local_shap_interaction_pairs(
        user_row, shap_model, feature_order
    )

    return {
        "row_index": row_index,
        "model_structure": {
            "prediction_probability_model": "sigmoid CalibratedClassifierCV(SMOTE + RandomForest)",
            "shap_explanation_model": "RandomForestClassifier extracted from fitted pipeline",
            "calibration_affects_shap_priority": False,
        },
        "raw_random_forest_probability": raw_probability,
        "calibrated_probability": calibrated_probability,
        "predicted_probability": calibrated_probability,
        "selected_threshold": threshold,
        "predicted_class": predicted_class,
        "risk_label": "HIGH_RISK" if predicted_class == 1 else "LOW_RISK",
        "calibration": metadata.get("calibration", {}),
        "threshold_selection": metadata.get("threshold_selection", {}),
        "positive_class_index": shap_class_index,
        "feature_count": len(feature_order),
        "main_effects": main_effects,
        "legacy_factor_effects": [
            {
                "factor": key,
                "shap_value": float(value),
                "positive_risk_contribution": float(max(value, 0.0)),
            }
            for key, value in sorted(legacy.items(), key=lambda pair: abs(pair[1]), reverse=True)
        ],
        "interactions": interactions,
        "interaction_warning": interaction_warning,
        "interaction_priority_status": "ONLY_APPROVED_INTERACTION_MAPPING_IS_APPLIED",
    }

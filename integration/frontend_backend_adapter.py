from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from shap_explain import explain_user_row

ROOT = Path(__file__).resolve().parents[1]

ROOM_TYPE_MAP = {
    "living_room": "LIVING_ROOM",
    "bedroom": "BEDROOM",
    "kitchen": "KITCHEN",
    "bathroom": "BATHROOM",
}

OBJECT_TYPE_KEYWORDS = {
    "TOILET": ("변기", "양변기"),
    "CHAIR": ("의자", "소파", "좌석"),
    "BED": ("침대",),
    "CABINET": ("수납장", "선반", "캐비닛", "장롱", "서랍장"),
    "RUG": ("러그", "카펫", "매트"),
    "THRESHOLD": ("문턱", "단차"),
    "ALERT_DEVICE": ("비상벨", "경보기", "알람", "센서등"),
    "LIGHT": ("조명", "전등", "스탠드"),
}

# UI는 사용자가 이해하기 쉬운 방향(0=정상, 1=제한)으로 저장하지만,
# 2020 학습데이터의 일부 컬럼은 반대 방향(1=정상, 0=제한)입니다.
# 이 변환을 한 곳에 고정해 재발을 방지합니다.
LIMITATION_FEATURE_MAP = {
    "근력상태_의자나 침대에 앉았다가 일어나기 5회 반복": "lower_limb_strength_limitation",
    "동작수행 어려움_운동장 한 바퀴(400m)정도 뛰기": "run_400m_limitation",
    "동작수행 어려움_운동장 한 바퀴(400m)정도 걷기": "walk_400m_limitation",
    "동작수행 어려움_쉬지않고 10계단 오르기": "climb_10_stairs_limitation",
    "동작수행 어려움_몸 구부리거나 쭈그려 앉거나 무릎 꿇기": "kneel_squat_limitation",
    "동작수행 어려움_머리보다 높은 곳에 있는 것 손 뻗쳐 닿기": "reach_overhead_limitation",
    "동작수행 어려움_쌀 1말(8kg) 정도 물건 들어 올리거나 옮기기": "lift_8kg_limitation",
}

DIAGNOSIS_FEATURES = (
    "diagnosis_stroke",
    "diagnosis_cad",
    "diagnosis_arthritis",
    "diagnosis_osteoporosis",
    "diagnosis_lumbar_sciatic_pain",
    "diagnosis_fracture_sequelae",
    "diagnosis_chronic_bronchitis_emphysema",
    "diagnosis_parkinson",
    "diagnosis_insomnia",
    "diagnosis_urinary_incontinence",
    "diagnosis_anemia",
)


def _backend_normal_code(ui_limitation_code: int | float) -> int:
    """UI 0=정상/1=제한을 학습데이터 1=정상/0=제한으로 변환합니다."""
    return 1 - int(ui_limitation_code)


def _backend_treatment_code(*, checked: int | float, diagnosed: bool | None = None) -> int:
    """학습데이터의 치료 코드(1=치료 중, 0=진단 후 미치료, 9=해당 없음)로 변환합니다."""
    if int(checked) == 1:
        return 1
    if diagnosed is True:
        return 0
    return 9


def _is_healthy_baseline(model_input: dict[str, int | float]) -> bool:
    """임시 안전 보정 대상인 '임상·기능 항목 모두 정상' 입력인지 판정합니다.

    모델 재학습 전까지만 사용하는 제한적 가드입니다. 고령 자체의 위험을 완전히
    지우지 않도록 기본 시연 연령대(65~75세)에만 적용합니다.
    """
    limitation_keys = tuple(LIMITATION_FEATURE_MAP.values())
    return all(
        (
            65 <= int(model_input["age"]) <= 75,
            int(model_input["general_health_status"]) <= 2,
            int(model_input["health_satisfaction"]) == 1,
            int(model_input["visual_difficulty"]) == 1,
            int(model_input["hearing_difficulty"]) == 1,
            int(model_input["polypharmacy"]) == 0,
            int(model_input["chronic_disease_count_group"]) == 0,
            all(int(model_input[key]) == 0 for key in DIAGNOSIS_FEATURES),
            int(model_input["treatment_depression"]) == 0,
            int(model_input["treatment_insomnia"]) == 0,
            int(model_input["treatment_bph"]) == 0,
            all(int(model_input[key]) == 0 for key in limitation_keys),
            int(model_input["adl_help_needed"]) == 0,
            int(model_input["iadl_help_needed"]) == 0,
            int(model_input["regular_exercise"]) == 1,
            int(model_input["depression_score_paper"]) <= 2,
            int(model_input["cognitive_score_mmse"]) >= 28,
            int(model_input["nutrition_status"]) == 0,
        )
    )


def _feature_order() -> list[str]:
    payload = json.loads(
        (ROOT / "artifacts" / "feature_schema.json").read_text(encoding="utf-8")
    )
    return [str(value) for value in payload["feature_order"]]


def frontend_model_input_to_backend_features(
    model_input: dict[str, int | float],
) -> tuple[dict[str, int | float], dict[str, Any]]:
    """Streamlit 45개 개념 변수를 학습 모델의 정확한 52개 컬럼으로 변환합니다.

    45개 중 결혼상태 1개는 5개 one-hot, 주택유형 1개는 4개 one-hot으로
    확장되므로 최종 모델 피처는 정확히 52개입니다. 임의 기본값은 사용하지 않습니다.
    """
    required = {
        "age", "sex", "marital_status", "education_years", "living_alone",
        "current_job", "financial_satisfaction", "general_health_status",
        "health_satisfaction", "drinking_frequency", "visual_difficulty",
        "hearing_difficulty", "polypharmacy", "chronic_disease_count_group",
        "diagnosis_stroke", "diagnosis_cad", "diagnosis_arthritis",
        "diagnosis_osteoporosis", "diagnosis_lumbar_sciatic_pain",
        "diagnosis_fracture_sequelae", "diagnosis_chronic_bronchitis_emphysema",
        "diagnosis_parkinson", "diagnosis_insomnia",
        "diagnosis_urinary_incontinence", "diagnosis_anemia",
        "treatment_depression", "treatment_insomnia", "treatment_bph",
        "lower_limb_strength_limitation", "run_400m_limitation",
        "walk_400m_limitation", "climb_10_stairs_limitation",
        "kneel_squat_limitation", "reach_overhead_limitation",
        "lift_8kg_limitation", "adl_help_needed", "iadl_help_needed",
        "regular_exercise", "depression_score_paper", "cognitive_score_mmse",
        "nutrition_status", "housing_type", "elderly_friendly_accessibility",
        "housing_satisfaction", "community_satisfaction",
    }
    missing_frontend = sorted(required - set(model_input))
    if missing_frontend:
        raise ValueError(f"Streamlit 입력에서 누락된 45개 개념 변수: {missing_frontend}")

    marital = int(model_input["marital_status"])
    housing = int(model_input["housing_type"])

    features: dict[str, int | float] = {
        **{
            backend_name: _backend_normal_code(model_input[frontend_name])
            for backend_name, frontend_name in LIMITATION_FEATURE_MAP.items()
        },
        # UI 0=양호, 1=보통, 2=불량 / 학습데이터 2=양호, 1=보통, 0=불량
        "Nutritional": 2 - int(model_input["nutrition_status"]),
        # UI 0=독립, 1=도움 필요 / 학습데이터 1=독립, 0=도움 필요
        "ADL": _backend_normal_code(model_input["adl_help_needed"]),
        "IADL": _backend_normal_code(model_input["iadl_help_needed"]),
        "평소 운동 여부": int(model_input["regular_exercise"]),
        "GDS": int(model_input["depression_score_paper"]),
        "인지기능_총점": int(model_input["cognitive_score_mmse"]),
        "노인이 생활하는데 편리함 정도": int(model_input["elderly_friendly_accessibility"]),
        "현재 살고있는 주택에 대한 만족도": int(model_input["housing_satisfaction"]),
        "지역사회 환경 만족도_지역사회 환경 전반": int(model_input["community_satisfaction"]),
        "노인 조사 대상자 만연령": int(model_input["age"]),
        "노인 조사 대상자 성별": int(model_input["sex"]),
        "노인 조사 대상자 교육연수": int(model_input["education_years"]),
        "노인가구형태": int(model_input["living_alone"]),
        "만족도_경제상태": int(model_input["financial_satisfaction"]),
        "현재 경제활동 여부": int(model_input["current_job"]),
        "평소의 건강상태": int(model_input["general_health_status"]),
        "만족도_건강상태": int(model_input["health_satisfaction"]),
        "지난 1년 간 음주 빈도": int(model_input["drinking_frequency"]),
        "일상생활의 불편함_시력": int(model_input["visual_difficulty"]),
        "일상생활의 불편함_청력": int(model_input["hearing_difficulty"]),
        "현재 3개월 이상 복용하고 있는 의사처방약(종류)": int(model_input["polypharmacy"]),
        "의사진단 만성질환 총 수": int(model_input["chronic_disease_count_group"]),
        "의사진단 만성질환 유무_뇌졸중(중풍, 뇌경색)": int(model_input["diagnosis_stroke"]),
        "의사진단 만성질환 유무_협심증, 심근경색증": int(model_input["diagnosis_cad"]),
        "의사진단 만성질환 유무_골관절염(퇴행성관절염), 류머티즘 관절염": int(model_input["diagnosis_arthritis"]),
        "의사진단 만성질환 유무_골다공증": int(model_input["diagnosis_osteoporosis"]),
        "의사진단 만성질환 유무_요통, 좌골신경통": int(model_input["diagnosis_lumbar_sciatic_pain"]),
        "의사진단 만성질환 유무_골절, 탈골 및 사고 후유증": int(model_input["diagnosis_fracture_sequelae"]),
        "의사진단 만성질환 유무_만성기관지염, 폐기종": int(model_input["diagnosis_chronic_bronchitis_emphysema"]),
        "의사진단 만성질환 유무_파킨슨병": int(model_input["diagnosis_parkinson"]),
        "의사진단 만성질환 유무_불면증": int(model_input["diagnosis_insomnia"]),
        "의사진단 만성질환 유무_요실금": int(model_input["diagnosis_urinary_incontinence"]),
        "의사진단 만성질환 유무_빈혈": int(model_input["diagnosis_anemia"]),
        "치료 여부_우울증": _backend_treatment_code(
            checked=model_input["treatment_depression"], diagnosed=None
        ),
        "치료 여부_불면증": _backend_treatment_code(
            checked=model_input["treatment_insomnia"],
            diagnosed=int(model_input["diagnosis_insomnia"]) == 1,
        ),
        "치료 여부_전립선 비대증": _backend_treatment_code(
            checked=model_input["treatment_bph"], diagnosed=None
        ),
        "RES_MAR_미혼": int(marital == 1),
        "RES_MAR_별거": int(marital == 5),
        "RES_MAR_사별": int(marital == 3),
        "RES_MAR_유배우": int(marital == 2),
        "RES_MAR_이혼": int(marital == 4),
        "Q1_기타": int(housing == 4),
        "Q1_단독주택": int(housing == 1),
        "Q1_아파트": int(housing == 2),
        "Q1_연립다세대": int(housing == 3),
    }

    order = _feature_order()
    missing_backend = [name for name in order if name not in features]
    extra_backend = [name for name in features if name not in order]
    if missing_backend or extra_backend:
        raise ValueError(
            f"52개 모델 스키마 변환 오류. missing={missing_backend}, extra={extra_backend}"
        )

    ordered = {name: features[name] for name in order}
    report = {
        "frontend_concept_feature_count": len(model_input),
        "backend_model_feature_count": len(ordered),
        "marital_status_expanded_to": [
            name for name in order if name.startswith("RES_MAR_") and ordered[name] == 1
        ],
        "housing_type_expanded_to": [
            name for name in order if name.startswith("Q1_") and ordered[name] == 1
        ],
        "missing_backend_features": [],
        "extra_backend_features": [],
        "imputation_used": False,
        "encoding_corrections": {
            "physical_function": "UI 0=정상/1=제한 -> model 1=정상/0=제한",
            "adl_iadl": "UI 0=독립/1=도움 -> model 1=독립/0=도움",
            "nutrition": "UI 0=양호/1=보통/2=불량 -> model 2=양호/1=보통/0=불량",
            "treatment": "미체크·해당없음 -> model 9; 진단 후 미치료 -> 0; 치료 중 -> 1",
        },
    }
    return ordered, report


@lru_cache(maxsize=1)
def _load_prediction_assets():
    order = _feature_order()

    calibrated = joblib.load(
        ROOT / "artifacts" / "calibrated_model.joblib"
    )

    base = joblib.load(
        ROOT / "artifacts" / "base_model.joblib"
    )

    shap_model = joblib.load(
        ROOT / "artifacts" / "rf_model_for_shap.joblib"
    )

    metadata = json.loads(
        (ROOT / "artifacts" / "model_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    return base, shap_model, calibrated, order, metadata


def predict_from_frontend_model_input(model_input: dict[str, int | float]) -> dict[str, Any]:
    """실제 Calibrated RandomForest로 Streamlit 입력을 예측합니다."""
    features, report = frontend_model_input_to_backend_features(model_input)
    base, shap_model, calibrated, order, metadata = _load_prediction_assets()
    frame = pd.DataFrame([[features[name] for name in order]], columns=order)
    raw_probability = float(base.predict_proba(frame)[0, 1])
    model_calibrated_probability = float(calibrated.predict_proba(frame)[0, 1])
    shap_result = explain_user_row(
        features,
        shap_model=shap_model,
        calibrated_model=calibrated,
        feature_order=order,
        metadata=metadata,
        row_index=None,
    )    
    threshold = float(metadata.get("selected_threshold", 0.5))

    # TEMPORARY SAFETY PATCH:
    # UI의 모든 임상·기능 항목이 정상인 기본 건강 프로필이 모델의 인구학적
    # 상호작용 때문에 고위험으로 표시되는 문제를 막습니다. 원 모델값은
    # model_calibrated_probability에 보존하며, 재학습 후 이 블록을 제거해야 합니다.
    healthy_baseline = _is_healthy_baseline(model_input)
    guard_enabled = os.getenv("ENABLE_HEALTHY_BASELINE_GUARD", "true").strip().lower() in {
        "true", "1", "yes", "y", "on"
    }
    configured_cap = float(os.getenv("HEALTHY_BASELINE_MAX_PROBABILITY", "0.04"))
    healthy_cap = min(configured_cap, max(0.0, threshold - 0.005))
    temporary_override_applied = bool(
        guard_enabled and healthy_baseline and model_calibrated_probability >= threshold
    )
    calibrated_probability = (
        min(model_calibrated_probability, healthy_cap)
        if temporary_override_applied
        else model_calibrated_probability
    )

    model_only_label = (
        "HIGH_RISK" if model_calibrated_probability >= threshold else "LOW_RISK"
    )
    model_risk_label = "HIGH_RISK" if calibrated_probability >= threshold else "LOW_RISK"
    ui_risk_level = "모델 판정 기준 초과 - 주의 및 점검 권장" if model_risk_label == "HIGH_RISK" else "모델 판정 기준 미만 - 관찰 권장"
    report["healthy_baseline_detected"] = healthy_baseline
    report["temporary_healthy_override_applied"] = temporary_override_applied

    return {
        "risk_score": round(calibrated_probability * 100, 2),
        "risk_level": ui_risk_level,
        "shap_result": shap_result,
        "model_risk_label": model_risk_label,
        "model_only_risk_label": model_only_label,
        "raw_probability": raw_probability,
        "calibrated_probability": calibrated_probability,
        "model_calibrated_probability": model_calibrated_probability,
        "selected_threshold": threshold,
        "threshold_percent": round(threshold * 100, 2),
        "temporary_override_applied": temporary_override_applied,
        "temporary_override_reason": (
            "모든 임상·기능 항목이 정상인 기본 건강 프로필에 대한 임시 안전 보정"
            if temporary_override_applied
            else None
        ),
        "temporary_override_cap": healthy_cap if temporary_override_applied else None,
        "healthy_baseline_guard_enabled": guard_enabled,
        "backend_features": features,
        "mapping_report": report,
    }


def infer_object_type(name: str) -> str:
    normalized = str(name)
    for object_type, keywords in OBJECT_TYPE_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return object_type
    if any(keyword in normalized for keyword in ("바닥 물건", "장애물", "박스", "화분")):
        return "OBJECT"
    return "FURNITURE"


def _first_object_id(objects: list[dict[str, Any]], object_type: str) -> str | None:
    for obj in objects:
        if obj.get("object_type") == object_type:
            return str(obj.get("object_id"))
    return None


def build_floorplan_from_streamlit(
    *,
    layout: list[dict[str, Any]],
    room_name: str,
    room_width: int | float,
    room_height: int | float,
    hazard_flags: dict[str, bool] | None = None,
    question_context: list[dict[str, Any]] | None = None,
    annotations: list[dict[str, Any]] | None = None,
    room_note: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """기존 가구배치 데이터를 core floorplan 스키마로 변환합니다.

    위험은 좌표만 보고 추정하지 않습니다. 사용자가 체크한 hazard_flags를
    explicit observation/semantic attribute로만 전달합니다.
    """
    hazard_flags = {str(k): bool(v) for k, v in (hazard_flags or {}).items()}
    question_context = [item for item in (question_context or []) if isinstance(item, dict)]
    annotations = [item for item in (annotations or []) if isinstance(item, dict)]
    room_note = str(room_note or "").strip()
    room_type = ROOM_TYPE_MAP.get(room_name, str(room_name).upper())
    room_id = f"room_{room_name or 'unknown'}_01"

    objects: list[dict[str, Any]] = []
    for index, item in enumerate(layout, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", f"가구 {index}")).strip() or f"가구 {index}"
        object_type = infer_object_type(name)
        obj = {
            "object_id": f"layout_object_{index:03d}",
            "object_type": object_type,
            "name": name,
            "room_id": room_id,
            "bbox": [
                float(item.get("x", 0)),
                float(item.get("y", 0)),
                float(item.get("width", 0)),
                float(item.get("height", 0)),
            ],
            "attributes": {
                "shape": str(item.get("shape") or "rect"),
                "rotation_deg": float(item.get("rotation") or 0),
            },
        }
        objects.append(obj)

    # 명시적으로 선택한 위험을 실제 객체 속성에 연결합니다.
    semantic_rules = {
        "LOW_BED_HEIGHT": ("BED", "bed_height_class", "low"),
        "LOW_SEAT_HEIGHT": ("CHAIR", "seat_height_class", "low"),
        "LOW_STORAGE": ("CABINET", "storage_height_class", "low"),
        "HIGH_STORAGE": ("CABINET", "storage_height_class", "high"),
        "LOOSE_RUG": ("RUG", "fixed", False),
        "LOW_TOILET": ("TOILET", "seat_height_class", "low"),
        "TOILET_NO_GRAB_BAR": ("TOILET", "grab_bar_present", False),
        "FLOOR_LEVEL_OBJECT": ("OBJECT", "floor_level", True),
        "THRESHOLD_HIGH": ("THRESHOLD", "height_class", "high"),
        "THRESHOLD_LOW_CONTRAST": ("THRESHOLD", "contrast", "low"),
        "HARD_SHARP_FURNITURE_EDGE": ("FURNITURE", "edge_profile", "sharp"),
    }
    linked_flags: list[str] = []
    unlinked_flags: list[str] = []
    for hazard_code, (object_type, attr_name, attr_value) in semantic_rules.items():
        if not hazard_flags.get(hazard_code):
            continue
        target = next((obj for obj in objects if obj["object_type"] == object_type), None)
        if target is None:
            unlinked_flags.append(hazard_code)
            continue
        target["attributes"][attr_name] = attr_value
        linked_flags.append(hazard_code)

    room_attributes: dict[str, Any] = {}
    if hazard_flags.get("LOW_LIGHTING"):
        room_attributes["lighting_level"] = "low"
    if hazard_flags.get("NO_NIGHT_LIGHT"):
        room_attributes["night_light"] = False

    paths: list[dict[str, Any]] = []
    path_codes = {"PATH_OBSTACLE", "NARROW_PATH", "UNCLEAR_ROUTE"}
    if any(hazard_flags.get(code) for code in path_codes):
        paths.append(
            {
                "path_id": f"path_{room_name}_main",
                "room_id": room_id,
                "polyline": [[0, float(room_height) / 2], [float(room_width), float(room_height) / 2]],
                "attributes": {
                    "obstacle_present": bool(hazard_flags.get("PATH_OBSTACLE")),
                    "narrow": bool(hazard_flags.get("NARROW_PATH")),
                    "unclear_route": bool(hazard_flags.get("UNCLEAR_ROUTE")),
                },
            }
        )

    observations: list[dict[str, Any]] = []
    # 객체가 없는 경우에도 사용자 체크리스트라는 명시적 근거로 기록합니다.
    for code in unlinked_flags:
        observations.append(
            {
                "observation_code": code,
                "room_id": room_id,
                "value": True,
                "confidence": 1.0,
                "observed_by": "STREAMLIT_USER_CHECKLIST",
                "note": "해당 위험을 사용자가 명시적으로 선택했으나 대응 객체명이 배치에 없음",
            }
        )

    # 객체/경로/조명 전용 규칙에 없는 공간별 추가 문항도 explicit observation으로 보존합니다.
    # 예: 주방·욕실의 SLIPPERY_FLOOR. hazard_detector는 사용자 명시 관찰값만 사용합니다.
    handled_codes = set(semantic_rules) | path_codes | {"LOW_LIGHTING", "NO_NIGHT_LIGHT"}
    generic_observation_codes: list[str] = []
    for code, selected in hazard_flags.items():
        if not selected or code in handled_codes:
            continue
        observations.append(
            {
                "observation_code": code,
                "room_id": room_id,
                "value": True,
                "confidence": 1.0,
                "observed_by": "STREAMLIT_ADAPTIVE_CHECKLIST",
                "note": "사용자 개인위험과 선택 공간에 따라 노출된 맞춤 문항에서 확인",
            }
        )
        generic_observation_codes.append(code)

    # 사용자가 평면도 위에 직접 표시한 출입구/단차/물기/주의영역을 좌표 정보와 함께 보존합니다.
    spatial_annotations: list[dict[str, Any]] = []
    for annotation in annotations:
        annotation_type = str(annotation.get("type") or "note").strip().lower()
        ax = float(annotation.get("x", 0))
        ay = float(annotation.get("y", 0))
        aw = float(annotation.get("width", 0))
        ah = float(annotation.get("height", 0))
        spatial = {
            "annotation_id": str(annotation.get("id") or f"annotation_{len(spatial_annotations)+1:03d}"),
            "annotation_type": annotation_type,
            "label": str(annotation.get("label") or annotation_type),
            "note": str(annotation.get("note") or ""),
            "geometry": str(annotation.get("geometry") or "point"),
            "bbox": [ax, ay, aw, ah],
            "room_id": room_id,
        }
        # 지점 메모는 좌표만 넘기지 않고 가장 가까운 가구 2개도 같이 연결합니다.
        # 사용자가 "소파와 테이블 사이가 좁다"고 썼을 때 planner가 실제 대상 가구를
        # 더 쉽게 식별할 수 있도록 하는 보조 맥락이며, 이 정보 자체로 새 hazard를 만들지는 않습니다.
        if annotation_type == "point_note" and objects:
            distances = []
            for obj in objects:
                bx, by, bw, bh = obj.get("bbox", [0, 0, 0, 0])
                cx = float(bx) + float(bw) / 2
                cy = float(by) + float(bh) / 2
                distances.append(((cx - ax) ** 2 + (cy - ay) ** 2, str(obj.get("name") or "가구")))
            distances.sort(key=lambda item: item[0])
            spatial["nearby_object_names"] = [name for _, name in distances[:2]]
        spatial_annotations.append(spatial)
        # 위험 의미가 명확한 표시는 기존 hazard code로도 연결합니다.
        # 출입구/일반 주의영역은 위치 참조 정보이므로 위험 자체로 판정하지 않습니다.
        spatial_hazard_code = {
            "wet": "SLIPPERY_FLOOR",
            "step": "THRESHOLD_HIGH",
        }.get(annotation_type)
        if spatial_hazard_code:
            observations.append({
                "observation_code": spatial_hazard_code,
                "room_id": room_id,
                "value": True,
                "confidence": 1.0,
                "observed_by": "USER_SPATIAL_ANNOTATION",
                "note": f"{spatial['label']} | {spatial['note']} | bbox={spatial['bbox']}",
                "spatial": spatial,
            })



    adaptive_hazard_candidates: list[dict[str, Any]] = []
    for item in question_context:
        code = str(item.get("hazard_code") or "").strip()
        if not code:
            continue
        clarification_answer = str(item.get("clarification_answer") or "UNKNOWN").upper()
        confirmation_status = str(item.get("confirmation_status") or "").strip()
        if not confirmation_status:
            confirmation_status = (
                "CONFIRMED_BY_USER_CHECKLIST"
                if bool(item.get("checked"))
                else "UNCONFIRMED"
            )
        adaptive_hazard_candidates.append(
            {
                "hazard_code": code,
                "label": item.get("label"),
                "priority": int(item.get("priority") or 999),
                "triggered_by": item.get("triggered_by", []),
                "trigger_labels": item.get("trigger_labels", []),
                "checked": bool(item.get("checked")),
                "clarification_answer": clarification_answer,
                "clarification_confirmed": bool(item.get("clarification_confirmed")),
                "room_id": room_id,
                "room_type": room_type,
                "confirmation_status": confirmation_status,
            }
        )

    floorplan = {
        "plan_id": "STREAMLIT_LAYOUT_PLAN",
        "coordinate_system": {"unit": "cm", "origin": "top_left", "y_axis": "down"},
        "canvas": {"width": float(room_width), "height": float(room_height)},
        "rooms": [
            {
                "room_id": room_id,
                "room_type": room_type,
                "polygon": [
                    [0, 0], [float(room_width), 0],
                    [float(room_width), float(room_height)], [0, float(room_height)],
                ],
                "attributes": room_attributes,
            }
        ],
        "objects": objects,
        "paths": paths,
        "observations": observations,
        # 미체크는 안전이 아니라 UNKNOWN이다. RAG 매핑/공간위험 탐지에는 사용하지 않고,
        # RAG 결과가 0건일 때 GPT가 조건부 예비 솔루션을 생성하는 후보로만 사용한다.
        "adaptive_hazard_candidates": adaptive_hazard_candidates,
        "spatial_annotations": spatial_annotations,
        "room_note": room_note,
    }
    report = {
        "room_name": room_name,
        "room_type": room_type,
        "object_count": len(objects),
        "object_types": [obj["object_type"] for obj in objects],
        "selected_hazard_flags": sorted(code for code, value in hazard_flags.items() if value),
        "linked_to_object_attributes": sorted(linked_flags),
        "recorded_as_explicit_observation": sorted(unlinked_flags + generic_observation_codes),
        "adaptive_question_count": len(question_context),
        "adaptive_question_codes": [
            str(item.get("hazard_code")) for item in question_context if item.get("hazard_code")
        ],
        "adaptive_question_context": question_context,
        "adaptive_ai_fallback_candidate_count": len(adaptive_hazard_candidates),
        "adaptive_ai_fallback_candidates": adaptive_hazard_candidates,
        "question_selection_policy": "ROOM_AND_PERSONAL_RISK_FILTER",
        "hazard_inference_from_geometry": False,
        "spatial_annotation_count": len(spatial_annotations),
        "spatial_annotations": spatial_annotations,
        "room_note": room_note,
    }
    return floorplan, report


def backend_result_to_streamlit_result(full_result: dict[str, Any]) -> dict[str, Any]:
    """core 결과를 Streamlit 결과 카드 구조로 변환합니다.

    - RAG 매핑 성공: 구조화 RAG 후보를 우선순위대로 표시합니다.
    - 확인된 RAG 미매핑 위험: integrated_plan의 AI 독자 step을 최대 3개 표시합니다.
    - 미확정 위험: 추천 카드를 만들지 않고 추가 확인 질문만 전달합니다.
    - GPT 질문은 별도 입력 폼에서 다시 공간위험 입력으로 반영할 수 있도록 그대로 전달합니다.
    """
    base = full_result.get("base_result", {})
    integrated = full_result.get("integrated_plan", {})
    validation = full_result.get("validation_report", {})
    recommendations: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []

    object_name_by_id: dict[str, str] = {}
    output_dir_value = full_result.get("output_dir")
    if output_dir_value:
        floorplan_path = Path(str(output_dir_value)).parent / "floorplan.json"
        if floorplan_path.exists():
            try:
                floorplan_payload = json.loads(floorplan_path.read_text(encoding="utf-8"))
                object_name_by_id = {
                    str(obj.get("object_id")): str(obj.get("name") or obj.get("object_id"))
                    for obj in floorplan_payload.get("objects", [])
                    if isinstance(obj, dict) and obj.get("object_id")
                }
                room_type_labels = {
                    "LIVING_ROOM": "거실", "BEDROOM": "침실",
                    "KITCHEN": "주방", "BATHROOM": "욕실",
                }
                room_label_by_id = {}
                for room_item in floorplan_payload.get("rooms", []):
                    if not isinstance(room_item, dict) or not room_item.get("room_id"):
                        continue
                    room_label = room_type_labels.get(str(room_item.get("room_type") or "").upper(), "공간")
                    room_label_by_id[str(room_item.get("room_id"))] = room_label
                    object_name_by_id[str(room_item.get("room_id"))] = f"{room_label} 전체"
                for path_item in floorplan_payload.get("paths", []):
                    if not isinstance(path_item, dict) or not path_item.get("path_id"):
                        continue
                    room_label = room_label_by_id.get(str(path_item.get("room_id")), "공간")
                    object_name_by_id[str(path_item.get("path_id"))] = f"{room_label} 주요 이동 동선"
            except Exception:
                object_name_by_id = {}

    def _factor_text(payload: dict[str, Any]) -> str:
        main_features = [
            str(item.get("feature"))
            for item in payload.get("matched_main_factors", [])
            if isinstance(item, dict) and item.get("feature")
        ]
        interactions = [
            f"{item.get('feature_1')} × {item.get('feature_2')}"
            for item in payload.get("matched_interactions", [])
            if isinstance(item, dict) and item.get("feature_1") and item.get("feature_2")
        ]
        need_codes = [
            str(value) for value in payload.get("matched_need_codes", []) if str(value).strip()
        ]
        # 개인 SHAP 위험요인과 직접 매칭되는 항목이 하나도 없는 경우(순수 공간위험 기반으로
        # 우선순위가 매겨진 경우)가 있습니다. 이때 "승인된 개인 위험기여" 같은 내부 승인
        # 상태 용어를 그대로 보여주면, 사용자는 이게 자기 답변에서 나온 것으로 오해합니다.
        # 빈 문자열을 반환해서, 화면단이 "개인 답변과 직접 연결되진 않지만 실제 공간에서
        # 확인된 위험"이라고 정직하게 설명하도록 넘깁니다.
        return " / ".join(main_features + interactions + need_codes)

    def _append_marker(
        *,
        recommendation_id: str,
        rank: int,
        hazard_code: str,
        object_id: str,
        description: str,
    ) -> None:
        # 실제 가구가 연결된 개선안이면 평면도 안에 마커를 표시하고,
        # 특정 가구가 없는 방 전체 개선안이면 오른쪽 목록에만 표시한다.
        has_target_object = bool(str(object_id).strip())

        target_object = object_name_by_id.get(
            str(object_id),
            str(object_id).strip() or "room",
        )

        markers.append(
            {
                "recommendation_id": recommendation_id,
                "priority": rank,

                # 오른쪽 개선사항 목록에 출력되는 내용
                "label": description,

                # 기존 데이터와의 호환성을 위해 함께 유지
                "description": description,
                "title": hazard_code,

                # 마커를 배치할 가구
                "target_object": target_object,
                "object_id": str(object_id).strip(),
                "hazard_code": hazard_code,

                # 특정 가구가 있으면 가구 주변에 점 표시
                # 특정 가구가 없으면 오른쪽 개선사항 목록에만 표시
                "marker_type": (
                    "near_object"
                    if has_target_object
                    else "outside_list"
                ),
            }
        )

    step_by_instance = {
        str(step.get("hazard_instance_id")): step
        for step in integrated.get("steps", [])
        if isinstance(step, dict) and step.get("hazard_instance_id")
    }
    # 이전 결과 호환용. 새 분석에서는 hazard_instance_id가 기본키입니다.
    step_by_hazard = {
        str(step.get("hazard_code")): step
        for step in integrated.get("steps", [])
        if isinstance(step, dict) and step.get("hazard_code")
    }

    primary_actions = [
        item for item in base.get("ranked_actions", []) if isinstance(item, dict)
    ]
    planning_actions = [
        item for item in base.get("ranked_solution_candidates", []) if isinstance(item, dict)
    ] or primary_actions
    evidence_mode = str(
        integrated.get("evidence_mode")
        or ("RAG_EVIDENCE" if primary_actions else "AI_ONLY_NO_RAG")
    )

    # 1) RAG 매핑 성공 시: 대표 솔루션과 동일 위험의 대안들을 모두 화면 후보로 변환합니다.
    # 다중 솔루션의 위험별/후보별 순위와 전체 표시 로직은 그대로 유지합니다.
    rag_actions = planning_actions if evidence_mode in {"RAG_EVIDENCE", "HYBRID_RAG_AI"} else []
    for action in rag_actions:
        evidence = action.get("selected_evidence", {}) or {}
        hazard_code = str(action.get("hazard_code", "UNKNOWN"))
        is_primary = bool(action.get("is_primary_for_hazard", True))
        hazard_instance_id = str(action.get("hazard_instance_id") or "")
        step = (step_by_instance.get(hazard_instance_id) or step_by_hazard.get(hazard_code, {})) if is_primary else {}
        rank = int(
            action.get("solution_rank")
            or action.get("final_priority_rank")
            or action.get("preliminary_priority_rank")
            or len(recommendations) + 1
        )
        candidate_rank = int(
            action.get("solution_candidate_rank_within_hazard")
            or evidence.get("solution_candidate_rank_within_hazard")
            or 1
        )

        recommendation_id = (
            f"RAG_{hazard_instance_id or hazard_code}_{rank}_{candidate_rank}"
        )
        recommendation_text = str(
            step.get("integrated_instruction")
            or evidence.get("recommendation")
            or hazard_code
        )
        recommendations.append(
            {
                "recommendation_id": recommendation_id,
                "priority": rank,
                "improvement": recommendation_text,
                "shap_factor": _factor_text(action),
                "rag_evidence": (
                    evidence.get("source")
                    or evidence.get("title")
                    or "구조화 RAG 근거"
                ),
                "floorplan_problem": " / ".join(
                    part
                    for part in [
                        hazard_code,
                        str(action.get("room_type", "")),
                        str(action.get("object_id", "")),
                    ]
                    if part
                ),
                "reason": (
                    "개인 SHAP·SHAP interaction, 실제 공간위험, 구조화 RAG 매핑 가능성을 "
                    "통과한 대표 솔루션"
                    if is_primary
                    else "동일한 공간위험에 대해 구조화 RAG 필터를 통과한 추가 대안"
                ),
                "expected_effect": (
                    evidence.get("expected_effect")
                    or "위험 노출을 줄이도록 공간을 개선"
                ),
                "evidence_ids": [evidence.get("evidence_id")] if evidence.get("evidence_id") else [],
                "is_primary_for_hazard": is_primary,
                "hazard_priority_rank": int(
                    action.get("hazard_priority_rank")
                    or action.get("final_priority_rank")
                    or action.get("preliminary_priority_rank")
                    or rank
                ),
                "solution_candidate_rank_within_hazard": int(
                    action.get("solution_candidate_rank_within_hazard")
                    or evidence.get("solution_candidate_rank_within_hazard")
                    or 1
                ),
                "vector_similarity": evidence.get("vector_similarity"),
                "evidence_score": evidence.get("evidence_score"),
                "recommendation_source": "RAG_EVIDENCE",
                "is_ai_only": False,
            }
        )
        _append_marker(
            recommendation_id=recommendation_id,
            rank=rank,
            hazard_code=hazard_code,
            object_id=str(action.get("object_id") or ""),
            description=recommendation_text,
)

    # 2) 확인된 미매핑 위험만 GPT/템플릿 AI 독자 솔루션 카드로 표시합니다.
    # AI_ONLY_PRECAUTIONARY는 질문 전 솔루션 생성·카드 표시가 모두 금지됩니다.
    if evidence_mode in {"AI_ONLY_NO_RAG", "HYBRID_RAG_AI"}:
        ai_steps = [
            item for item in integrated.get("steps", [])
            if isinstance(item, dict)
            and str(item.get("recommendation_source") or "") == "AI_ONLY_NO_RAG"
        ]
        ai_steps.sort(
            key=lambda item: (
                int(item.get("fixed_priority_rank") or 10**6),
                int(item.get("implementation_sequence") or 10**6),
            )
        )
        for index, step in enumerate(ai_steps, start=1):
            hazard_code = str(step.get("hazard_code") or "UNKNOWN")
            hazard_instance_id = str(step.get("hazard_instance_id") or f"AI_{index}")
            rank = int(step.get("fixed_priority_rank") or index)
            recommendation_id = f"AI_{hazard_instance_id}_{rank}_{index}"
            target = step.get("target") if isinstance(step.get("target"), dict) else {}
            object_id = str(target.get("object_id") or "")
            confirmation_status = str(step.get("confirmation_status") or "")
            source_label = "AI 독자 제안 · RAG 문헌 미매핑"
            recommendation_text = str(
                step.get("integrated_instruction")
                or "현재 공간위험을 줄이기 위한 보수적 개선안을 검토합니다."
            )
            recommendations.append(
                {
                    "recommendation_id": recommendation_id,
                    "priority": rank,
                    "improvement": recommendation_text,
                    "shap_factor": _factor_text(step),
                    "rag_evidence": source_label,
                    "floorplan_problem": " / ".join(
                        part
                        for part in [
                            hazard_code,
                            str(target.get("room_type") or ""),
                            object_id,
                            confirmation_status,
                        ]
                        if part
                    ),
                    "reason": (
                        "개인 SHAP·SHAP interaction과 실제 확인된 공간위험의 고정 순위를 유지한 "
                        "RAG 미매핑 AI 독자 솔루션"
                    ),
                    "expected_effect": "확인된 공간위험에 대한 노출을 줄이는 방향으로 공간을 정리·조정",
                    "evidence_ids": [],
                    "is_primary_for_hazard": True,
                    "hazard_priority_rank": rank,
                    "solution_candidate_rank_within_hazard": 1,
                    "vector_similarity": None,
                    "evidence_score": None,
                    "recommendation_source": "AI_ONLY_NO_RAG",
                    "is_ai_only": True,
                    "implementation_note": step.get("implementation_note"),
                }
            )
            _append_marker(
                recommendation_id=recommendation_id,
                rank=rank,
                hazard_code=hazard_code,
                object_id=object_id,
                description=recommendation_text,
            )

    recommendations.sort(key=lambda item: int(item.get("priority") or 10**6))

    summary = integrated.get("summary") or (
        f"보정 낙상확률 {float(base.get('predicted_probability', 0.0)):.2%}, "
        f"실제 공간위험 {int(base.get('preliminary_ranked_hazard_count', 0))}개를 분석했습니다."
    )
    risk_analysis = (
        f"모델 판정: {base.get('risk_label', 'UNKNOWN')} · "
        f"보정확률 {float(base.get('predicted_probability', 0.0)):.2%} · "
        f"RAG 매핑 가능 위험 {int(base.get('feasible_hazard_count', 0))}개 · "
        f"표시 솔루션 {len(recommendations)}개"
    )
    clarification_questions = [
        item
        for item in integrated.get("clarification_questions", [])
        if isinstance(item, dict)
    ]
    awaiting_confirmation = evidence_mode == "AI_ONLY_PRECAUTIONARY"

    if evidence_mode == "HYBRID_RAG_AI":
        final_summary = (
            integrated.get("summary")
            or "확인된 위험 순위는 그대로 유지하고, 문헌 매핑 위험은 RAG 근거로, 미매핑 위험은 AI 보완안으로 함께 제시했습니다."
        )
    elif evidence_mode == "AI_ONLY_NO_RAG":
        final_summary = (
            integrated.get("summary")
            or "구조화 RAG 문헌은 매핑되지 않았지만, 확인된 공간위험과 고정 우선순위를 바탕으로 "
            "AI 독자 개선안을 생성했습니다. 문헌 근거가 없는 제안이므로 적용 전 현장 검토가 필요합니다."
        )
    elif evidence_mode == "AI_ONLY_PRECAUTIONARY":
        final_summary = (
            integrated.get("summary")
            or "공간위험이 아직 확정되지 않아 솔루션을 생성하지 않았습니다. "
            "추가 질문에 답하면 위험을 확정한 뒤 공간개선 솔루션을 제공합니다."
        )
    elif evidence_mode == "NO_ACTIONS":
        final_summary = integrated.get("summary") or "현재 입력에서 공간개선 후보를 확인하지 못했습니다."
    else:
        final_summary = integrated.get("summary") or "근거 기반 공간개선 우선순위를 확인했습니다."

    return {
        "summary": summary,
        "risk_analysis": risk_analysis,
        "recommendations": recommendations,
        "final_summary": final_summary,
        "markers": markers,
        "clarification_questions": clarification_questions,
        "awaiting_confirmation": awaiting_confirmation,
        "planner_mode": integrated.get("planner_mode"),
        "gpt_api_called": integrated.get("gpt_api_called", False),
        "analysis_id": full_result.get("analysis_id"),
        "output_dir": full_result.get("output_dir"),
        "validation": validation,
        "solution_evidence_mode": evidence_mode,
        "mapped_solution_candidate_count": len(recommendations),
        "rag_solution_candidate_count": len(planning_actions),
        "ai_solution_count": sum(1 for item in recommendations if item.get("is_ai_only")),
        "default_solution_display_count": min(3, len(recommendations)),
        "solution_listing_policy": (
            base.get("solution_listing_policy")
            if rag_actions or evidence_mode == "HYBRID_RAG_AI"
            else "QUESTIONS_ONLY_UNTIL_CONFIRMATION"
            if awaiting_confirmation
            else "TOP_3_AI_ONLY_BY_FIXED_PRIORITY"
        ),
        "backend_full_result": full_result,
    }

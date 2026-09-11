from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.gpt_integrated_planner import (
    generate_clarification_followup_plan,
    generate_integrated_plan,
)
from src.io_utils import read_json, write_json
from src.pipeline import run_pipeline
from src.plan_markdown import write_plan_markdown

ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "runtime" / "cases"


def _default_paths() -> dict[str, Path]:
    return {
        "base_model_path": ROOT / "artifacts/base_model.joblib",
        "shap_model_path": ROOT / "artifacts/rf_model_for_shap.joblib",
        "calibrated_model_path": ROOT / "artifacts/calibrated_model.joblib",
        "metadata_path": ROOT / "artifacts/model_metadata.json",
        "schema_path": ROOT / "artifacts/feature_schema.json",
        "data_path": ROOT / "data/fall_data_2020.csv",
        "ontology_path": ROOT / "data/factor_ontology.csv",
        "bridge_path": ROOT / "data/factor_hazard_bridge.csv",
        "interaction_bridge_path": ROOT / "data/interaction_hazard_bridge.csv",
        "rag_path": ROOT / "data/fall_space_rag_ready.csv",
        "target_alias_path": ROOT / "data/target_object_aliases.csv",
        "rag_hazard_override_path": ROOT / "data/rag_hazard_overrides.csv",
    }


def _load_feature_order() -> list[str]:
    schema = json.loads(
        (ROOT / "artifacts/feature_schema.json").read_text(encoding="utf-8")
    )
    return [str(value) for value in schema["feature_order"]]


def _write_user_csv(user_features: dict[str, Any], path: Path) -> None:
    feature_order = _load_feature_order()
    missing = [feature for feature in feature_order if feature not in user_features]
    extra = [feature for feature in user_features if feature not in feature_order]
    if missing:
        raise ValueError(f"입력에서 누락된 모델 피처: {missing}")
    if extra:
        raise ValueError(f"모델 스키마에 없는 추가 피처: {extra}")

    row = {feature: user_features[feature] for feature in feature_order}
    pd.DataFrame([row]).to_csv(path, index=False, encoding="utf-8-sig")


def run_base_analysis(
    *,
    floorplan: dict | None,
    row_index: int | None = None,
    user_features: dict[str, Any] | None = None,
    top_n: int = 7,
    include_review: bool = False,
    selected_room: str | None = None,
    case_id: str | None = None,
    prediction_context: dict[str, Any] | None = None,
    monitor_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[str, dict, dict, Path]:
    """HTTP 없이 v3 모델·SHAP·공간위험·RAG 파이프라인을 직접 실행한다."""
    if (row_index is None) == (user_features is None):
        raise ValueError("row_index 또는 user_features 중 하나만 제공해야 합니다.")

    case_id = case_id or f"case_{uuid.uuid4().hex[:12]}"
    case_dir = CASE_ROOT / case_id
    output_dir = case_dir / "outputs"
    case_dir.mkdir(parents=True, exist_ok=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    floorplan_payload = floorplan or read_json(ROOT / "runtime/floorplan.json")
    floorplan_path = case_dir / "floorplan.json"
    write_json(floorplan_path, floorplan_payload)

    paths = _default_paths()
    actual_row_index = int(row_index or 0)
    data_path = paths["data_path"]

    if user_features is not None:
        data_path = case_dir / "user_features.csv"
        _write_user_csv(user_features, data_path)
        actual_row_index = 0

    # ============================================================
    # DEBUG PIPELINE MONITOR START
    # REMOVE THIS BLOCK BEFORE PRODUCTION
    # ============================================================
    if monitor_callback is not None:
        try:
            monitor_callback(
                "input_ready",
                {
                    "case_id": case_id,
                    "feature_count": len(user_features or {}),
                    "floorplan": floorplan_payload,
                    "output_dir": str(output_dir),
                },
            )
        except Exception:
            pass
    # ============================================================
    # DEBUG PIPELINE MONITOR END
    # ============================================================

    result = run_pipeline(
        row_index=actual_row_index,
        top_n=top_n,
        base_model_path=paths["base_model_path"],
        shap_model_path=paths["shap_model_path"],
        calibrated_model_path=paths["calibrated_model_path"],
        metadata_path=paths["metadata_path"],
        schema_path=paths["schema_path"],
        data_path=data_path,
        floorplan_path=floorplan_path,
        ontology_path=paths["ontology_path"],
        bridge_path=paths["bridge_path"],
        interaction_bridge_path=paths["interaction_bridge_path"],
        rag_path=paths["rag_path"],
        target_alias_path=paths["target_alias_path"],
        rag_hazard_override_path=paths["rag_hazard_override_path"],
        output_dir=output_dir,
        include_review=include_review,
        selected_room=selected_room,
        prediction_context=prediction_context,
        monitor_callback=monitor_callback,
    )
    return case_id, result, floorplan_payload, output_dir


def add_gpt_integrated_plan(
    *,
    case_id: str,
    base_result: dict,
    floorplan: dict,
    output_dir: Path,
    use_gpt: bool = True,
    monitor_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[dict, dict]:
    """고정 우선순위와 선택된 RAG 근거를 유지한 GPT 통합계획을 생성한다."""
    plan, validation = generate_integrated_plan(
        base_result,
        floorplan,
        use_gpt=use_gpt,
    )
    plan["analysis_id"] = case_id
    write_json(output_dir / "07_gpt_integrated_plan.json", plan)
    write_json(output_dir / "08_gpt_plan_validation.json", validation)
    write_plan_markdown(plan, output_dir / "09_gpt_integrated_plan.md")
    # ============================================================
    # DEBUG PIPELINE MONITOR START
    # REMOVE THIS BLOCK BEFORE PRODUCTION
    # ============================================================
    if monitor_callback is not None:
        try:
            monitor_callback("gpt_completed", {"result": plan, "path": str(output_dir / "07_gpt_integrated_plan.json")})
            monitor_callback("validation_completed", {"result": validation, "path": str(output_dir / "08_gpt_plan_validation.json")})
        except Exception:
            pass
    # ============================================================
    # DEBUG PIPELINE MONITOR END
    # ============================================================
    return plan, validation


def run_full_analysis(
    *,
    floorplan: dict | None,
    row_index: int | None = None,
    user_features: dict[str, Any] | None = None,
    top_n: int = 7,
    include_review: bool = False,
    selected_room: str | None = None,
    use_gpt: bool = True,
    case_id: str | None = None,
    prediction_context: dict[str, Any] | None = None,
    monitor_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """HTTP 없이 전체 파이프라인을 직접 호출하고 결과 dict를 반환한다."""
    case_id, base_result, floorplan_payload, output_dir = run_base_analysis(
        floorplan=floorplan,
        row_index=row_index,
        user_features=user_features,
        top_n=top_n,
        include_review=include_review,
        selected_room=selected_room,
        case_id=case_id,
        prediction_context=prediction_context,
        monitor_callback=monitor_callback,
    )
    integrated_plan, validation_report = add_gpt_integrated_plan(
        case_id=case_id,
        base_result=base_result,
        floorplan=floorplan_payload,
        output_dir=output_dir,
        use_gpt=use_gpt,
        monitor_callback=monitor_callback,
    )
    result_payload = {
        "analysis_id": case_id,
        "base_result": base_result,
        "integrated_plan": integrated_plan,
        "validation_report": validation_report,
        "output_dir": str(output_dir),
    }
    # ============================================================
    # DEBUG PIPELINE MONITOR START
    # REMOVE THIS BLOCK BEFORE PRODUCTION
    # ============================================================
    if monitor_callback is not None:
        try:
            monitor_callback("pipeline_completed", {"result": result_payload, "output_dir": str(output_dir)})
        except Exception:
            pass
    # ============================================================
    # DEBUG PIPELINE MONITOR END
    # ============================================================
    return result_payload


def run_clarification_followup(
    *,
    previous_full_result: dict[str, Any],
    floorplan: dict,
    clarification_questions: list[dict[str, Any]],
    answers_by_question_id: dict[str, Any],
    use_gpt: bool = True,
) -> dict[str, Any]:
    """기존 분석 결과를 재사용해 추가 질문 답변용 AI 계획만 생성합니다.

    run_pipeline을 호출하지 않으므로 예측, SHAP, interaction, hazard detector,
    RAG 검색은 반복되지 않습니다.
    """
    base_result = previous_full_result.get("base_result", {})
    if not isinstance(base_result, dict):
        raise ValueError("이전 base_result가 없어 추가 질문 후속 계획을 생성할 수 없습니다.")

    plan, validation = generate_clarification_followup_plan(
        base_result,
        floorplan if isinstance(floorplan, dict) else {},
        clarification_questions,
        answers_by_question_id,
        use_gpt=use_gpt,
    )
    analysis_id = str(previous_full_result.get("analysis_id") or "clarification_followup")
    plan["analysis_id"] = analysis_id

    output_dir_value = previous_full_result.get("output_dir")
    if output_dir_value:
        output_dir = Path(str(output_dir_value))
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "10_clarification_followup_plan.json", plan)
        write_json(output_dir / "11_clarification_followup_validation.json", validation)
        write_plan_markdown(plan, output_dir / "12_clarification_followup_plan.md")

    return {
        **previous_full_result,
        "analysis_id": analysis_id,
        "integrated_plan": plan,
        "validation_report": validation,
        "clarification_followup": True,
        "reused_previous_analysis": True,
        "rerun_stages": [],
    }

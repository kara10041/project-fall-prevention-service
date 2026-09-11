from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd

from shap_explain import explain_user_row, load_models_and_schema
from src.hazard_detector import detect_hazards
from src.io_utils import read_csv_flexible, read_json, write_json
from src.overlay_renderer import render_svg
from src.rag_enricher import ensure_enriched_rag
from src.rag_retriever import evaluate_feasibility_and_retrieve
from src.spatial_mapper import rank_spatial_hazards


def run_pipeline(
    *,
    row_index: int,
    top_n: int,
    base_model_path: str | Path,
    shap_model_path: str | Path,
    calibrated_model_path: str | Path,
    metadata_path: str | Path,
    schema_path: str | Path | None,
    data_path: str | Path,
    floorplan_path: str | Path,
    ontology_path: str | Path,
    bridge_path: str | Path,
    interaction_bridge_path: str | Path | None,
    rag_path: str | Path,
    target_alias_path: str | Path,
    rag_hazard_override_path: str | Path | None,
    output_dir: str | Path,
    include_review: bool = False,
    selected_room: str | None = None,
    prediction_context: dict[str, Any] | None = None,
    monitor_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # DEBUG PIPELINE MONITOR START
    # REMOVE THIS BLOCK BEFORE PRODUCTION
    # ============================================================
    def emit(stage: str, payload: dict[str, Any]) -> None:
        if monitor_callback is None:
            return
        try:
            monitor_callback(stage, payload)
        except Exception:
            # 모니터링 실패가 실제 분석을 중단시키지 않도록 격리합니다.
            return
    # ============================================================
    # DEBUG PIPELINE MONITOR END
    # ============================================================

    # 1. 실제 2020 데이터의 지정 행을 그대로 모델에 투입한다. 랜덤 대체 없음.
    base_model, shap_model, calibrated_model, feature_order, model_metadata = load_models_and_schema(
        base_model_path=base_model_path,
        shap_model_path=shap_model_path,
        calibrated_model_path=calibrated_model_path,
        schema_path=schema_path,
        metadata_path=metadata_path,
    )
    data = read_csv_flexible(data_path)
    if row_index < 0 or row_index >= len(data):
        raise IndexError(f"row={row_index} 범위 오류. 사용 가능 범위: 0~{len(data)-1}")
    missing = [feature for feature in feature_order if feature not in data.columns]
    if missing:
        raise KeyError(f"모델 피처가 2020 CSV에 없습니다: {missing}")
    user_row = data.iloc[row_index][feature_order].to_dict()

    explanation = explain_user_row(
        user_row,
        shap_model=shap_model,
        calibrated_model=calibrated_model,
        feature_order=feature_order,
        metadata=model_metadata,
        row_index=row_index,
    )
    explanation["source_person_id"] = data.iloc[row_index].get("개인 ID")
    explanation["source_target"] = data.iloc[row_index].get("낙상유무")
    explanation["feature_input_policy"] = "DIRECT_ROW_FROM_FALL_DATA_2020_NO_RANDOM_IMPUTATION"
    if prediction_context and prediction_context.get("temporary_override_applied"):
        original_probability = float(
            explanation.get("predicted_probability", explanation.get("calibrated_probability", 0.0))
        )
        explanation["model_predicted_probability_before_temporary_override"] = original_probability
        explanation["predicted_probability"] = float(
            prediction_context.get("calibrated_probability", original_probability)
        )
        explanation["calibrated_probability"] = explanation["predicted_probability"]
        explanation["risk_label"] = prediction_context.get("model_risk_label", "LOW_RISK")
        explanation["temporary_override_applied"] = True
        explanation["temporary_override_reason"] = prediction_context.get(
            "temporary_override_reason"
        )
        explanation["shap_values_explain_original_model_output"] = True
    write_json(out / "01_shap_result.json", explanation)
    emit("shap_completed", {"result": explanation, "path": str(out / "01_shap_result.json")})

    # 2. semantic 공간 속성 또는 명시적 observation에서만 hazard를 판정한다.
    floorplan = read_json(floorplan_path)
    detected = detect_hazards(floorplan, selected_room=selected_room)
    normalized_floorplan = detected.pop("normalized_floorplan")
    write_json(out / "02_hazard_assessment.json", detected)
    emit("hazards_completed", {"result": detected, "path": str(out / "02_hazard_assessment.json")})

    # 3. ontology + bridge + 승인된 interaction으로 사전 우선순위를 계산한다.
    ontology = read_csv_flexible(ontology_path)
    bridge = read_csv_flexible(bridge_path)
    interaction_bridge = None
    if interaction_bridge_path and Path(interaction_bridge_path).exists():
        interaction_bridge = read_csv_flexible(interaction_bridge_path)
    preliminary = rank_spatial_hazards(
        explanation,
        detected,
        ontology,
        bridge,
        interaction_bridge=interaction_bridge,
        include_review=include_review,
    )
    write_json(out / "03_preliminary_hazard_priority.json", preliminary)
    emit("priority_completed", {"result": preliminary, "path": str(out / "03_preliminary_hazard_priority.json")})

    # 4. 원본 RAG는 보존하고 실행용 structured view를 만든다.
    rag_db = read_csv_flexible(rag_path)
    target_alias_table = read_csv_flexible(target_alias_path)
    hazard_overrides = None
    if rag_hazard_override_path and Path(rag_hazard_override_path).exists():
        hazard_overrides = read_csv_flexible(rag_hazard_override_path)
    enriched_path = out / "04_rag_enriched.csv"
    enriched_rag, enrichment_summary = ensure_enriched_rag(
        rag_db,
        ontology,
        bridge,
        target_alias_table,
        hazard_overrides=hazard_overrides,
        output_path=enriched_path,
    )
    write_json(out / "04_rag_enrichment_summary.json", enrichment_summary)
    emit("rag_enrichment_completed", {"result": enrichment_summary, "path": str(out / "04_rag_enrichment_summary.json")})

    review_mask = enriched_rag["structured_mapping_status"].astype(str).eq("AMBIGUOUS_HAZARD_MAPPING_REVIEW_REQUIRED")
    review_columns = [
        col for col in [
            "evidence_id", "factor_1", "factor_2", "factor_aliases", "need_codes",
            "candidate_hazard_codes", "space_categories", "target_object",
            "recommendation", "eligible", "review_status"
        ] if col in enriched_rag.columns
    ]
    enriched_rag.loc[review_mask, review_columns].to_csv(
        out / "04_rag_hazard_review_required.csv", index=False, encoding="utf-8-sig"
    )

    # 5. hazard별 RAG 매핑 가능성을 확인하되, 미매핑 hazard도 제거하지 않고 기존 위험순위를 유지한다.
    completed = evaluate_feasibility_and_retrieve(preliminary, enriched_rag, top_n=top_n)
    completed["rag_enrichment_summary"] = enrichment_summary
    for key in (
        "temporary_override_applied",
        "temporary_override_reason",
        "model_predicted_probability_before_temporary_override",
        "shap_values_explain_original_model_output",
    ):
        if key in explanation:
            completed[key] = explanation[key]
    mapped_count = int(completed.get("feasible_hazard_count", 0))
    unresolved_count = int(completed.get("unresolved_hazard_count", 0))
    if mapped_count and unresolved_count:
        completed["solution_generation_mode"] = "HYBRID_RAG_AI"
        completed["ai_only_candidate_count"] = unresolved_count
        completed["intervention_generation_allowed"] = True
    elif mapped_count:
        completed["solution_generation_mode"] = "RAG_EVIDENCE"
        completed["ai_only_candidate_count"] = 0
        completed["intervention_generation_allowed"] = True
    elif unresolved_count:
        completed["solution_generation_mode"] = "AI_ONLY_NO_RAG"
        completed["ai_only_candidate_count"] = unresolved_count
        completed["intervention_generation_allowed"] = True
    else:
        adaptive_candidates = [
            item
            for item in floorplan.get("adaptive_hazard_candidates", [])
            if isinstance(item, dict) and item.get("hazard_code")
        ]
        completed["solution_generation_mode"] = (
            "AI_ONLY_PRECAUTIONARY" if adaptive_candidates else "NO_ACTIONS"
        )
        completed["ai_only_candidate_count"] = len(adaptive_candidates)
        completed["intervention_generation_allowed"] = bool(adaptive_candidates)

    if completed.get("unresolved_hazards"):
        write_json(
            out / "05_ai_only_solution_candidates.json",
            {
                "mode": completed.get("solution_generation_mode"),
                "warning": (
                    "RAG 미매핑 hazard만 AI 독자 개선안 대상으로 보냅니다. "
                    "RAG 매핑 hazard와 함께 존재해도 원래 hazard priority rank를 유지하며, "
                    "AI 항목에는 evidence_id를 생성하지 않습니다."
                ),
                "candidates": completed["unresolved_hazards"],
            },
        )

    write_json(out / "05_final_ranked_action_plan.json", completed)
    emit("rag_completed", {"result": completed, "path": str(out / "05_final_ranked_action_plan.json")})

    rows = []
    solution_rows = completed.get("ranked_solution_candidates") or completed.get("ranked_actions", [])
    for action in solution_rows:
        evidence = action.get("selected_evidence", {})
        rows.append({
            "solution_rank": action.get("solution_rank") or action.get("final_priority_rank"),
            "hazard_priority_rank": action.get("hazard_priority_rank") or action.get("final_priority_rank"),
            "is_primary_for_hazard": action.get("is_primary_for_hazard", True),
            "solution_candidate_rank_within_hazard": action.get(
                "solution_candidate_rank_within_hazard", 1
            ),
            "final_priority_score": action.get("final_priority_score"),
            "final_priority_raw": action.get("final_priority_raw"),
            "preliminary_priority_rank": action.get("preliminary_priority_rank"),
            "personal_risk_contribution": action.get("personal_risk_contribution"),
            "hazard_presence_score": action.get("hazard_presence_score"),
            "detection_confidence": action.get("detection_confidence"),
            "main_effect_score": action.get("main_effect_score"),
            "interaction_score": action.get("interaction_score"),
            "need_codes": " | ".join(action.get("matched_need_codes", [])),
            "hazard_code": action.get("hazard_code"),
            "room_type": action.get("room_type"),
            "object_id": action.get("object_id"),
            "matched_features": " | ".join(x.get("feature", "") for x in action.get("matched_main_factors", [])),
            "evidence_id": evidence.get("evidence_id"),
            "recommendation": evidence.get("recommendation"),
            "expected_effect": evidence.get("expected_effect"),
            "vector_similarity": evidence.get("vector_similarity"),
            "evidence_score": evidence.get("evidence_score"),
            "sms": evidence.get("sms"),
            "tier": evidence.get("tier"),
            "source": evidence.get("source"),
            "doi": evidence.get("doi"),
        })
    pd.DataFrame(rows).to_csv(out / "05_selected_solutions.csv", index=False, encoding="utf-8-sig")

    unresolved_rows = []
    for item in completed.get("unresolved_hazards", []):
        unresolved_rows.append({
            "preliminary_priority_rank": item.get("preliminary_priority_rank"),
            "preliminary_priority_score": item.get("preliminary_priority_score"),
            "need_codes": " | ".join(item.get("matched_need_codes", [])),
            "hazard_code": item.get("hazard_code"),
            "room_type": item.get("room_type"),
            "object_id": item.get("object_id"),
            "reason": item.get("mapping_failure_reason"),
        })
    pd.DataFrame(unresolved_rows).to_csv(out / "05_unresolved_hazards.csv", index=False, encoding="utf-8-sig")

    render_svg(normalized_floorplan, completed, out / "06_overlay_result.svg")
    emit("overlay_completed", {"path": str(out / "06_overlay_result.svg"), "result": completed})
    return completed

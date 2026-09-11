from __future__ import annotations

from collections import defaultdict

import pandas as pd


def _split(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [token.strip() for token in str(value).replace(",", "|").split("|") if token.strip()]


def _approved_status(value: object, include_review: bool) -> bool:
    status = str(value).strip()
    if status.startswith("확정"):
        return True
    return include_review and status == "검토"


def rank_spatial_hazards(
    explanation: dict,
    detected_hazards: dict,
    ontology: pd.DataFrame,
    factor_hazard_bridge: pd.DataFrame,
    interaction_bridge: pd.DataFrame | None = None,
    include_review: bool = False,
) -> dict:
    """
    1차 위험 우선순위 계산.

    personal_risk_contribution
      = positive Local SHAP main effects mapped to the hazard
      + approved positive interaction effects mapped to the hazard exactly once

    preliminary_priority_raw
      = environmental_hazard_baseline × (1 + personal_risk_contribution)
        × hazard_presence_score × detection_confidence

    공간에서 실제로 확인된 hazard는 SHAP 매핑이 0이어도 제거하지 않는다.
    SHAP/interaction은 hazard의 존재 여부를 결정하는 gate가 아니라 개인화 가중치로만 사용한다.
    RAG/근거점수는 이 단계의 위험 우선순위에 관여하지 않는다.
    """
    ont = ontology[ontology["review_status"].map(lambda x: _approved_status(x, include_review))].copy()
    ont_by_feature = {str(row.model_feature): row for row in ont.itertuples(index=False)}
    factor_by_feature = {str(row.model_feature): str(row.factor_code) for row in ont.itertuples(index=False)}

    bridge_meta: dict[tuple[str, str], dict] = {}
    need_to_hazards: dict[str, set[str]] = defaultdict(set)
    for row in factor_hazard_bridge.itertuples(index=False):
        need_code = str(row.need_code)
        hazard_code = str(row.hazard_code)
        need_to_hazards[need_code].add(hazard_code)
        bridge_meta[(need_code, hazard_code)] = {
            "applicable_rooms": _split(row.applicable_rooms),
            "target_object_type": str(row.target_object_type),
            "rationale": str(row.rationale),
        }

    main_by_hazard: dict[str, list[dict]] = defaultdict(list)
    need_main_score: dict[str, float] = defaultdict(float)
    unmapped: list[dict] = []

    for effect in explanation.get("main_effects", []):
        contribution = float(effect.get("positive_risk_contribution", 0.0))
        if contribution <= 0:
            continue
        feature = str(effect["feature"])
        ont_row = ont_by_feature.get(feature)
        if ont_row is None:
            unmapped.append({
                "feature": feature,
                "shap_value": effect.get("shap_value"),
                "reason": "ONTOLOGY_NOT_APPROVED_OR_NOT_FOUND",
            })
            continue
        if int(ont_row.spatial_actionable) != 1 or str(ont_row.need_code) == "NON_SPATIAL_CONTEXT":
            unmapped.append({
                "feature": feature,
                "factor_code": str(ont_row.factor_code),
                "shap_value": effect.get("shap_value"),
                "reason": "NON_SPATIAL_CONTEXT",
            })
            continue

        need_code = str(ont_row.need_code)
        need_main_score[need_code] += contribution
        for hazard_code in need_to_hazards.get(need_code, set()):
            meta = bridge_meta[(need_code, hazard_code)]
            main_by_hazard[hazard_code].append({
                "feature": feature,
                "legacy_factor": effect.get("legacy_factor"),
                "factor_code": str(ont_row.factor_code),
                "need_code": need_code,
                "contribution": contribution,
                "bridge_rationale": meta["rationale"],
                "target_object_type": meta["target_object_type"],
            })

    # 승인된 interaction mapping만 사용하며 pair contribution은 hazard당 한 번만 더한다.
    interaction_by_hazard: dict[str, list[dict]] = defaultdict(list)
    need_interaction_score: dict[str, float] = defaultdict(float)
    interaction_mapping_status = "NO_APPROVED_INTERACTION_MAPPING"
    approved_interaction_count = 0

    interaction_lookup: dict[frozenset[str], list[dict]] = defaultdict(list)
    if interaction_bridge is not None and not interaction_bridge.empty:
        for row in interaction_bridge.itertuples(index=False):
            if not _approved_status(getattr(row, "review_status", ""), include_review=False):
                continue
            key = frozenset({str(row.factor_code_1), str(row.factor_code_2)})
            interaction_lookup[key].append({
                "need_code": str(row.interaction_need_code),
                "hazard_codes": _split(row.hazard_codes),
                "rationale": str(row.rationale),
            })

    if interaction_lookup:
        interaction_mapping_status = "APPROVED_MAPPING_AVAILABLE_NO_POSITIVE_MATCH_FOR_THIS_ROW"
        for pair in explanation.get("interactions", []):
            contribution = float(pair.get("positive_interaction_contribution", 0.0))
            if contribution <= 0:
                continue
            feature_1, feature_2 = str(pair["feature_1"]), str(pair["feature_2"])
            factor_1 = factor_by_feature.get(feature_1)
            factor_2 = factor_by_feature.get(feature_2)
            if not factor_1 or not factor_2:
                continue
            mappings = interaction_lookup.get(frozenset({factor_1, factor_2}), [])
            for mapping in mappings:
                need_code = mapping["need_code"]
                need_interaction_score[need_code] += contribution
                for hazard_code in mapping["hazard_codes"]:
                    interaction_by_hazard[hazard_code].append({
                        "feature_1": feature_1,
                        "feature_2": feature_2,
                        "factor_code_1": factor_1,
                        "factor_code_2": factor_2,
                        "need_code": need_code,
                        "contribution": contribution,
                        "rationale": mapping["rationale"],
                    })
                approved_interaction_count += 1
        if approved_interaction_count > 0:
            interaction_mapping_status = "APPROVED_POSITIVE_INTERACTIONS_APPLIED"

    ranked: list[dict] = []
    for hazard in detected_hazards.get("hazard_instances", []):
        code = str(hazard["hazard_code"])
        matched_main: list[dict] = []
        for factor in main_by_hazard.get(code, []):
            meta = bridge_meta.get((factor["need_code"], code), {})
            rooms = meta.get("applicable_rooms", [])
            if rooms and "ALL" not in rooms and hazard.get("room_type") not in rooms:
                continue
            matched_main.append(factor)

        matched_interactions = list(interaction_by_hazard.get(code, []))

        main_score = sum(float(item["contribution"]) for item in matched_main)
        interaction_score = sum(float(item["contribution"]) for item in matched_interactions)
        personal_score = main_score + interaction_score
        presence_score = float(hazard.get("presence_score", 1.0 if hazard.get("present") else 0.0))
        confidence = float(hazard.get("confidence", 0.0))

        # 실제 공간에서 확인된 위험 자체에 기본 위험값을 부여한다.
        # SHAP는 위험의 존재 여부를 gate하지 않고, 개인별 취약성을 더하는 multiplier로만 사용한다.
        environmental_baseline = 1.0
        personalization_multiplier = 1.0 + max(personal_score, 0.0)
        priority_raw = (
            environmental_baseline
            * personalization_multiplier
            * presence_score
            * confidence
        )
        if priority_raw <= 0:
            continue

        matched_need_codes = sorted(
            {item["need_code"] for item in matched_main}
            | {item["need_code"] for item in matched_interactions}
        )
        target_object_types = sorted({item["target_object_type"] for item in matched_main if item.get("target_object_type")})
        if not target_object_types:
            target_object_types = [str(hazard.get("object_type", ""))]

        ranked.append({
            **hazard,
            "matched_need_codes": matched_need_codes,
            "expected_target_object_types": target_object_types,
            "matched_main_factors": matched_main,
            "matched_interactions": matched_interactions,
            "main_effect_score": main_score,
            "interaction_score": interaction_score,
            "personal_risk_contribution": personal_score,
            "environmental_hazard_baseline": environmental_baseline,
            "personalization_multiplier": personalization_multiplier,
            "hazard_presence_score": presence_score,
            "detection_confidence": confidence,
            "priority_basis": "CONFIRMED_ENVIRONMENTAL_HAZARD_BASELINE_X_PERSONALIZATION",
            "preliminary_priority_raw": priority_raw,
        })

    ranked.sort(key=lambda item: item["preliminary_priority_raw"], reverse=True)
    max_score = max((item["preliminary_priority_raw"] for item in ranked), default=0.0)
    for index, item in enumerate(ranked, start=1):
        item["preliminary_priority_rank"] = index
        item["preliminary_priority_score"] = item["preliminary_priority_raw"] / max_score if max_score else 0.0

    all_needs = sorted(set(need_main_score) | set(need_interaction_score))
    need_profiles = [{
        "need_code": need,
        "positive_main_effect_score": float(need_main_score.get(need, 0.0)),
        "positive_interaction_score": float(need_interaction_score.get(need, 0.0)),
        "personal_need_score": float(need_main_score.get(need, 0.0) + need_interaction_score.get(need, 0.0)),
    } for need in all_needs]
    need_profiles.sort(key=lambda x: x["personal_need_score"], reverse=True)

    return {
        "row_index": explanation.get("row_index"),
        "raw_random_forest_probability": explanation.get("raw_random_forest_probability"),
        "calibrated_probability": explanation.get("calibrated_probability"),
        "predicted_probability": explanation.get("predicted_probability"),
        "selected_threshold": explanation.get("selected_threshold"),
        "predicted_class": explanation.get("predicted_class"),
        "risk_label": explanation.get("risk_label"),
        "calibration_affects_shap_priority": False,
        "priority_formula": "ENVIRONMENTAL_HAZARD_BASELINE * (1 + PERSONAL_RISK_CONTRIBUTION) * HAZARD_PRESENCE_SCORE * DETECTION_CONFIDENCE",
        "environmental_hazard_gate_policy": "CONFIRMED_SPATIAL_HAZARD_IS_RETAINED_EVEN_WHEN_SHAP_MAPPING_IS_ZERO",
        "personal_risk_formula": "SUM(POSITIVE_LOCAL_SHAP_MAIN) + SUM(APPROVED_POSITIVE_INTERACTIONS_ONCE)",
        "interaction_priority_status": interaction_mapping_status,
        "approved_positive_interaction_mapping_count": approved_interaction_count,
        "include_review_ontology": include_review,
        "need_profiles": need_profiles,
        "preliminary_ranked_hazard_count": len(ranked),
        "preliminary_ranked_hazards": ranked,
        "unmapped_positive_effects": unmapped,
    }

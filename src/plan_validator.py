from __future__ import annotations

from copy import deepcopy
from typing import Any


class PlanValidationError(ValueError):
    """GPT 계획이 고정된 위험/근거 정책을 위반한 경우."""


def _safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _instance_key(item: dict, fallback: str = "") -> str:
    """동일 hazard_code가 여러 개 있어도 충돌하지 않도록 instance id를 우선 사용합니다."""
    return _safe_text(item.get("hazard_instance_id")) or fallback


def _fixed_context(base_result: dict) -> tuple[dict[str, dict], str]:
    ranked = [item for item in base_result.get("ranked_actions", []) if isinstance(item, dict)]
    if ranked:
        fixed: dict[str, dict] = {}
        source_modes: set[str] = set()
        for index, item in enumerate(ranked, start=1):
            key = _instance_key(item, f"HZ_FALLBACK_{index:03d}")
            source_mode = _safe_text(item.get("recommendation_source")) or (
                "RAG_EVIDENCE" if item.get("selected_evidence") else "AI_ONLY_NO_RAG"
            )
            source_modes.add(source_mode)
            fixed[key] = {
                **item,
                "hazard_instance_id": key,
                "final_priority_rank": int(
                    item.get("final_priority_rank")
                    or item.get("preliminary_priority_rank")
                    or index
                ),
                "recommendation_source": source_mode,
                "selected_evidence": item.get("selected_evidence", {}) or {},
            }
        mode = "HYBRID_RAG_AI" if len(source_modes) > 1 else next(iter(source_modes), "NO_ACTIONS")
        return fixed, mode

    ai_candidates = [
        item for item in base_result.get("ai_only_candidates", [])
        if isinstance(item, dict) and item.get("hazard_code")
    ]
    if ai_candidates:
        mode = str(base_result.get("ai_only_evidence_mode") or "AI_ONLY_NO_RAG")
        fixed: dict[str, dict] = {}
        for index, item in enumerate(ai_candidates, start=1):
            key = _instance_key(item, f"AI_FALLBACK_{index:03d}")
            fixed[key] = {
                **item,
                "hazard_instance_id": key,
                "final_priority_rank": int(
                    item.get("final_priority_rank")
                    or item.get("preliminary_priority_rank")
                    or index
                ),
                "recommendation_source": mode,
                "selected_evidence": {},
            }
        return fixed, mode

    unresolved = [
        item for item in base_result.get("unresolved_hazards", [])
        if isinstance(item, dict) and item.get("hazard_code")
    ]
    unresolved.sort(
        key=lambda item: (
            int(item.get("preliminary_priority_rank") or 10**6),
            -float(item.get("preliminary_priority_raw", 0.0)),
        )
    )
    if unresolved:
        fixed = {}
        for index, item in enumerate(unresolved[:3], start=1):
            key = _instance_key(item, f"AI_FALLBACK_{index:03d}")
            fixed[key] = {
                **item,
                "hazard_instance_id": key,
                "final_priority_rank": int(item.get("preliminary_priority_rank") or index),
                "selected_evidence": {},
                "recommendation_source": "AI_ONLY_NO_RAG",
                "confirmation_status": "CONFIRMED_SPATIAL_HAZARD",
            }
        return fixed, "AI_ONLY_NO_RAG"

    return {}, "NO_ACTIONS"


def validate_and_normalize_plan(plan: dict, base_result: dict) -> tuple[dict, dict]:
    """GPT 출력을 hazard instance별 고정 순위·근거 경계에 맞게 검증합니다.

    - RAG_EVIDENCE item: 선택된 RAG evidence만 허용
    - AI_ONLY_NO_RAG item: evidence_id 금지
    - HYBRID_RAG_AI: 위 두 정책을 각 hazard instance별로 동시에 적용
    - AI_ONLY_PRECAUTIONARY: 질문만 허용
    - NO_ACTIONS: step과 질문 모두 금지
    """
    fixed_by_instance, evidence_mode = _fixed_context(base_result)
    errors: list[str] = []
    warnings: list[str] = []
    normalized = deepcopy(plan)

    steps = normalized.get("steps")
    if not isinstance(steps, list):
        raise PlanValidationError("GPT 계획의 steps가 list가 아닙니다.")

    normalized_steps: list[dict] = []
    seen_instances: set[str] = set()

    if evidence_mode == "NO_ACTIONS":
        if steps:
            errors.append("NO_ACTIONS 모드에서는 intervention step 생성이 금지됩니다.")
    elif evidence_mode == "AI_ONLY_PRECAUTIONARY":
        if steps:
            errors.append("AI_ONLY_PRECAUTIONARY 모드에서는 위험 확인 전 solution step 생성이 금지됩니다.")
    else:
        for step_index, raw_step in enumerate(steps, start=1):
            if not isinstance(raw_step, dict):
                errors.append("step이 object가 아닙니다.")
                continue

            instance_id = _safe_text(raw_step.get("hazard_instance_id"))
            hazard_code = _safe_text(raw_step.get("hazard_code"))
            if instance_id not in fixed_by_instance:
                # GPT가 AI_FALLBACK_001처럼 임시 ID를 만들어 반환하는 경우가 있습니다.
                # hazard 자체가 달라진 것은 아니므로, hazard_code + 고정순위로 원래 HZ_xxx를
                # 안전하게 복구합니다. 복구가 유일하게 결정되지 않을 때만 오류로 처리합니다.
                try:
                    provided_rank = int(raw_step.get("fixed_priority_rank") or step_index)
                except (TypeError, ValueError):
                    provided_rank = step_index
                candidates = [
                    (key, item) for key, item in fixed_by_instance.items()
                    if _safe_text(item.get("hazard_code")) == hazard_code
                    and key not in seen_instances
                ]
                rank_matches = [
                    (key, item) for key, item in candidates
                    if int(item.get("final_priority_rank") or 0) == provided_rank
                ]
                if len(rank_matches) == 1:
                    recovered_id = rank_matches[0][0]
                elif len(candidates) == 1:
                    recovered_id = candidates[0][0]
                else:
                    # hazard_code까지 GPT가 임시/오류 코드로 바꾼 경우에도 고정순위는 서버가
                    # 이미 알고 있으므로 같은 순위의 아직 사용되지 않은 hazard instance로 복구합니다.
                    rank_global = [
                        (key, item) for key, item in fixed_by_instance.items()
                        if key not in seen_instances
                        and int(item.get("final_priority_rank") or 0) == provided_rank
                    ]
                    if len(rank_global) == 1:
                        recovered_id = rank_global[0][0]
                    else:
                        ordered = sorted(
                            [(key, item) for key, item in fixed_by_instance.items() if key not in seen_instances],
                            key=lambda pair: int(pair[1].get("final_priority_rank") or 10**6),
                        )
                        recovered_id = ordered[step_index - 1][0] if step_index - 1 < len(ordered) else ""
                if recovered_id:
                    warnings.append(f"GPT 임시 hazard ID {instance_id or '(빈 값)'}를 {recovered_id}로 복구했습니다.")
                    instance_id = recovered_id
                else:
                    errors.append(f"허용되지 않은 hazard_instance_id: {instance_id}")
                    continue
            if instance_id in seen_instances:
                errors.append(f"중복 hazard instance step: {instance_id}")
                continue
            seen_instances.add(instance_id)

            fixed = fixed_by_instance[instance_id]
            expected_code = _safe_text(fixed.get("hazard_code"))
            if hazard_code != expected_code:
                warnings.append(f"{instance_id}: GPT hazard_code {hazard_code or '(빈 값)'}를 서버 고정값 {expected_code}로 복구했습니다.")
                hazard_code = expected_code

            expected_rank = int(fixed.get("final_priority_rank", 0))
            provided_rank = int(raw_step.get("fixed_priority_rank", expected_rank))
            if provided_rank != expected_rank:
                warnings.append(f"{instance_id}: GPT 순위 {provided_rank}를 서버 고정순위 {expected_rank}로 복구했습니다.")

            source_mode = _safe_text(fixed.get("recommendation_source")) or "AI_ONLY_NO_RAG"
            evidence = fixed.get("selected_evidence", {}) or {}
            expected_evidence_id = _safe_text(evidence.get("evidence_id"))
            provided_ids = [
                _safe_text(value) for value in raw_step.get("evidence_ids", []) if _safe_text(value)
            ]

            if source_mode == "RAG_EVIDENCE":
                if expected_evidence_id and expected_evidence_id not in provided_ids:
                    warnings.append(f"{instance_id}: 누락된 근거 ID를 서버 고정값 {expected_evidence_id}로 복구했습니다.")
                unexpected = [value for value in provided_ids if value != expected_evidence_id]
                if unexpected:
                    warnings.append(f"{instance_id}: GPT가 추가한 비허용 근거 ID를 제거했습니다: {unexpected}")
                normalized_evidence_ids = [expected_evidence_id] if expected_evidence_id else []
            else:
                if provided_ids:
                    warnings.append(f"{instance_id}: AI 독자 모드의 근거 ID를 제거했습니다: {provided_ids}")
                normalized_evidence_ids = []

            instruction = _safe_text(raw_step.get("integrated_instruction"))
            if not instruction:
                errors.append(f"{instance_id}: integrated_instruction이 비어 있음")

            target = raw_step.get("target") if isinstance(raw_step.get("target"), dict) else {}
            normalized_steps.append({
                "hazard_instance_id": instance_id,
                "hazard_code": expected_code,
                "fixed_priority_rank": expected_rank,
                "implementation_sequence": int(raw_step.get("implementation_sequence", expected_rank)),
                "evidence_ids": normalized_evidence_ids,
                "recommendation_source": source_mode,
                "source_recommendation": evidence.get("recommendation"),
                "source_expected_effect": evidence.get("expected_effect"),
                "source_title": evidence.get("source"),
                "source_doi": evidence.get("doi"),
                "integrated_instruction": instruction,
                "sequence_reason": _safe_text(raw_step.get("sequence_reason")),
                "target": {
                    "room_type": fixed.get("room_type"),
                    "room_id": fixed.get("room_id"),
                    "object_id": fixed.get("object_id"),
                    "object_type": fixed.get("object_type"),
                    "placement_zone": _safe_text(target.get("placement_zone")) or None,
                    "avoid_zones": [
                        _safe_text(value) for value in target.get("avoid_zones", []) if _safe_text(value)
                    ],
                },
                "dependencies": [
                    int(value) for value in raw_step.get("dependencies", []) if str(value).isdigit()
                ],
                "conflicts_considered": [
                    _safe_text(value) for value in raw_step.get("conflicts_considered", []) if _safe_text(value)
                ],
                "implementation_note": _safe_text(raw_step.get("implementation_note")),
                "matched_need_codes": fixed.get("matched_need_codes", []),
                "matched_main_factors": fixed.get("matched_main_factors", []),
                "matched_interactions": fixed.get("matched_interactions", []),
                "personal_risk_contribution": fixed.get("personal_risk_contribution"),
                "environmental_hazard_baseline": fixed.get("environmental_hazard_baseline"),
                "confirmation_status": fixed.get("confirmation_status"),
                "priority_basis": fixed.get("priority_basis"),
                "mapping_failure_reason": fixed.get("mapping_failure_reason"),
                "candidate_label": fixed.get("candidate_label"),
            })

        missing = set(fixed_by_instance) - seen_instances
        if missing:
            errors.append("통합계획에 누락된 hazard instance: " + ", ".join(sorted(missing)))

    normalized_steps.sort(key=lambda item: item["fixed_priority_rank"])
    normalized["steps"] = normalized_steps
    normalized["priority_policy"] = (
        "FIXED_FROM_PERSONAL_SHAP_AND_ROOM_CANDIDATE"
        if evidence_mode == "AI_ONLY_PRECAUTIONARY"
        else "FIXED_FROM_CONFIRMED_ENVIRONMENTAL_HAZARD_WITH_PERSONALIZATION"
    )
    normalized["priority_changed_by_gpt"] = False
    normalized["evidence_mode"] = evidence_mode

    allowed_codes = {_safe_text(item.get("hazard_code")) for item in fixed_by_instance.values()}
    questions = normalized.get("clarification_questions", [])
    valid_questions: list[dict] = []
    if not isinstance(questions, list):
        errors.append("clarification_questions가 list가 아닙니다.")
        questions = []
    for index, question in enumerate(questions[:3], start=1):
        if not isinstance(question, dict):
            continue
        related = []
        for code in question.get("related_hazard_codes", []):
            normalized_code = str(code)
            if normalized_code in allowed_codes and normalized_code not in related:
                related.append(normalized_code)
        if not related:
            warnings.append(f"질문 {index}: 유효한 관련 hazard가 없어 제외")
            continue
        question_text = _safe_text(question.get("question"))
        if not question_text:
            warnings.append(f"질문 {index}: 질문 문구가 없어 제외")
            continue
        valid_questions.append({
            "question_id": _safe_text(question.get("question_id")) or f"Q{index}",
            "question": question_text,
            "reason": _safe_text(question.get("reason")),
            "answer_type": "YES_NO_UNKNOWN",
            "related_hazard_codes": related,
        })
    if evidence_mode == "NO_ACTIONS" and valid_questions:
        errors.append("NO_ACTIONS 모드에서는 추가 질문 생성이 금지됩니다.")
    normalized["clarification_questions"] = valid_questions

    conflicts = normalized.get("cross_solution_conflicts", [])
    valid_conflicts: list[dict] = []
    if evidence_mode == "NO_ACTIONS" and conflicts:
        errors.append("NO_ACTIONS 모드에서는 solution conflict 생성이 금지됩니다.")
    elif evidence_mode == "AI_ONLY_PRECAUTIONARY" and conflicts:
        errors.append("AI_ONLY_PRECAUTIONARY 모드에서는 위험 확인 전 solution conflict 생성이 금지됩니다.")
    elif isinstance(conflicts, list):
        for conflict in conflicts:
            if not isinstance(conflict, dict):
                continue
            related = [str(code) for code in conflict.get("related_hazard_codes", []) if str(code) in allowed_codes]
            if len(related) < 2:
                continue
            valid_conflicts.append({
                "conflict_type": _safe_text(conflict.get("conflict_type")),
                "related_hazard_codes": related,
                "description": _safe_text(conflict.get("description")),
                "resolution": _safe_text(conflict.get("resolution")),
            })
    normalized["cross_solution_conflicts"] = valid_conflicts

    source_modes = sorted({_safe_text(item.get("recommendation_source")) for item in fixed_by_instance.values()})
    report = {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "fixed_hazard_count": len(fixed_by_instance),
        "validated_step_count": len(normalized_steps),
        "validated_question_count": len(valid_questions),
        "priority_policy": "GPT_CANNOT_CHANGE_FIXED_PRIORITY_RANK",
        "instance_identity_policy": "HAZARD_INSTANCE_ID_PRIMARY_KEY",
        "evidence_mode": evidence_mode,
        "item_source_modes": source_modes,
        "evidence_policy": "EVIDENCE_RULE_ENFORCED_PER_HAZARD_INSTANCE",
    }

    if errors:
        raise PlanValidationError("; ".join(errors))
    return normalized, report

from __future__ import annotations

import csv
import json
import os
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.plan_validator import PlanValidationError, validate_and_normalize_plan


ROOT = Path(__file__).resolve().parents[1]
MAX_AI_ONLY_STEPS = 3


PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "plan_title": {"type": "string"},
        "summary": {"type": "string"},
        "priority_policy": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hazard_instance_id": {"type": "string"},
                    "hazard_code": {"type": "string"},
                    "fixed_priority_rank": {"type": "integer"},
                    "implementation_sequence": {"type": "integer"},
                    "evidence_ids": {"type": "array", "items": {"type": "string"}},
                    "integrated_instruction": {"type": "string"},
                    "sequence_reason": {"type": "string"},
                    "target": {
                        "type": "object",
                        "properties": {
                            "placement_zone": {"type": ["string", "null"]},
                            "avoid_zones": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["placement_zone", "avoid_zones"],
                        "additionalProperties": False,
                    },
                    "dependencies": {"type": "array", "items": {"type": "integer"}},
                    "conflicts_considered": {"type": "array", "items": {"type": "string"}},
                    "implementation_note": {"type": "string"},
                },
                "required": [
                    "hazard_instance_id",
                    "hazard_code",
                    "fixed_priority_rank",
                    "implementation_sequence",
                    "evidence_ids",
                    "integrated_instruction",
                    "sequence_reason",
                    "target",
                    "dependencies",
                    "conflicts_considered",
                    "implementation_note",
                ],
                "additionalProperties": False,
            },
        },
        "cross_solution_conflicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "conflict_type": {"type": "string"},
                    "related_hazard_codes": {"type": "array", "items": {"type": "string"}},
                    "description": {"type": "string"},
                    "resolution": {"type": "string"},
                },
                "required": ["conflict_type", "related_hazard_codes", "description", "resolution"],
                "additionalProperties": False,
            },
        },
        "clarification_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "question": {"type": "string"},
                    "reason": {"type": "string"},
                    "answer_type": {"type": "string", "enum": ["YES_NO_UNKNOWN"]},
                    "related_hazard_codes": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "question_id",
                    "question",
                    "reason",
                    "answer_type",
                    "related_hazard_codes",
                ],
                "additionalProperties": False,
            },
        },
        "limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "plan_title",
        "summary",
        "priority_policy",
        "steps",
        "cross_solution_conflicts",
        "clarification_questions",
        "limitations",
    ],
    "additionalProperties": False,
}


HAZARD_QUESTION_TEMPLATES = {
    "LOW_STORAGE": "낮은 수납 위치의 물건이 실제로 자주 사용하는 물건인가요?",
    "HIGH_STORAGE": "높은 수납 위치의 물건을 꺼낼 때 발판이나 의자를 사용하나요?",
    "FLOOR_LEVEL_OBJECT": "바닥 가까운 물건이 주 이동동선 안에 놓여 있나요?",
    "LOW_TOILET": "변기에서 일어날 때 손으로 주변을 짚거나 다른 사람의 도움이 필요한가요?",
    "TOILET_NO_GRAB_BAR": "변기·샤워 구역에서 몸을 지지할 고정된 손잡이가 실제로 없나요?",
    "SLIPPERY_FLOOR": "해당 바닥은 물기가 생긴 뒤에도 미끄러운 상태가 자주 지속되나요?",
    "PATH_OBSTACLE": "확인된 장애물은 매일 사용하는 주 이동동선을 실제로 가로막나요?",
    "NARROW_PATH": "보행보조기 또는 양손 지지가 필요할 때 통과하기 어려운 폭인가요?",
    "LOOSE_RUG": "러그·매트의 가장자리가 들리거나 발에 걸린 적이 있나요?",
    "LOW_LIGHTING": "야간이나 흐린 날에 물체의 경계가 잘 보이지 않나요?",
    "NO_NIGHT_LIGHT": "야간 이동 시 별도의 조명을 켜기 전까지 이동 구간이 어두운가요?",
    "THRESHOLD_HIGH": "문턱·단차를 넘을 때 발이 걸리거나 보조가 필요한가요?",
    "THRESHOLD_LOW_CONTRAST": "문턱·단차가 주변 바닥과 비슷한 색이라 구분하기 어려운가요?",
    "UNCLEAR_ROUTE": "사용자가 목적지까지의 이동 방향을 자주 혼동하나요?",
    "HARD_SHARP_FURNITURE_EDGE": "날카로운 모서리가 주 이동동선 또는 넘어질 가능성이 있는 위치에 있나요?",
}


AI_SOLUTION_TEMPLATES = {
    "PATH_OBSTACLE": "주 이동동선을 가로막는 물건과 가구를 치우거나 재배치해 끊기지 않는 통로를 확보합니다.",
    "NARROW_PATH": "통로 양쪽의 가구 배치를 단순화하고 자주 이동하는 구간의 유효 폭을 우선 확보합니다.",
    "LOOSE_RUG": "미고정 러그·매트는 제거하거나 들뜨지 않도록 고정하고, 가장자리가 이동동선에 걸리지 않게 정리합니다.",
    "SLIPPERY_FLOOR": "물기가 자주 생기는 구역은 즉시 건조할 수 있게 관리하고 미끄럼을 줄이는 표면·매트 사용 여부를 점검합니다.",
    "LOW_LIGHTING": "주 이동동선과 작업 구역의 조명을 보강하고 그림자 때문에 경계가 사라지는 위치를 줄입니다.",
    "NO_NIGHT_LIGHT": "침대·출입문·화장실로 이어지는 야간 동선에 쉽게 켤 수 있는 보조 조명을 배치합니다.",
    "LOW_STORAGE": "자주 사용하는 물건을 허리와 어깨 사이의 손이 닿기 쉬운 수납 위치로 옮깁니다.",
    "HIGH_STORAGE": "자주 사용하는 물건을 높은 선반에서 내려 발판이나 의자 없이 꺼낼 수 있는 위치로 옮깁니다.",
    "FLOOR_LEVEL_OBJECT": "바닥 가까이에 놓인 물건을 선반이나 수납함으로 옮겨 발 주변의 걸림 요소를 제거합니다.",
    "LOW_SEAT_HEIGHT": "자주 사용하는 의자·소파는 앉고 일어날 때 과도하게 낮아지지 않는 좌면으로 조정하거나 교체를 검토합니다.",
    "LOW_BED_HEIGHT": "침대에서 일어설 때 무릎과 허리에 부담이 크지 않도록 침상 높이와 주변 지지 위치를 조정합니다.",
    "LOW_TOILET": "변기에서 앉고 일어나는 동작 부담을 줄일 수 있도록 좌면 높이 조정 가능성을 검토합니다.",
    "TOILET_NO_GRAB_BAR": "변기 또는 샤워 구역에서 몸을 안정적으로 지지할 수 있는 고정 지지점을 마련합니다.",
    "THRESHOLD_HIGH": "문턱·단차는 제거 또는 완화 가능성을 검토하고, 당장 변경하기 어렵다면 이동 시 걸림을 줄이는 방식으로 표시·정리합니다.",
    "THRESHOLD_LOW_CONTRAST": "문턱·단차의 경계를 바닥과 구분되는 방식으로 표시해 시각적으로 쉽게 인지되도록 합니다.",
    "UNCLEAR_ROUTE": "주요 목적지까지의 이동 방향이 한눈에 보이도록 가구 배치를 단순화하고 안내 단서를 정리합니다.",
    "HARD_SHARP_FURNITURE_EDGE": "주 이동동선 주변의 날카로운 모서리는 동선 밖으로 이동하거나 충격을 줄이는 보호 처리를 검토합니다.",
}


def _compact_floorplan(floorplan: dict) -> dict:
    return {
        "plan_id": floorplan.get("plan_id"),
        "canvas": floorplan.get("canvas"),
        "rooms": [
            {
                "room_id": room.get("room_id"),
                "room_type": room.get("room_type"),
                "polygon": room.get("polygon"),
                "attributes": room.get("attributes", {}),
            }
            for room in floorplan.get("rooms", [])
        ],
        "objects": [
            {
                "object_id": obj.get("object_id"),
                "object_type": obj.get("object_type"),
                "name": obj.get("name"),
                "room_id": obj.get("room_id"),
                "bbox": obj.get("bbox"),
                "attributes": obj.get("attributes", {}),
            }
            for obj in floorplan.get("objects", [])
        ],
        "paths": floorplan.get("paths", []),
        "observations": floorplan.get("observations", []),
        "adaptive_hazard_candidates": floorplan.get("adaptive_hazard_candidates", []),
        # 사용자가 직접 입력한 자유 메모는 hazard 판정을 새로 만들지는 않지만,
        # 이미 확정된 위험에 대한 공간 맥락/설명 생성에는 활용합니다.
        "room_note": floorplan.get("room_note", ""),
        "spatial_annotations": floorplan.get("spatial_annotations", []),
    }


def _ranked_actions(base_result: dict) -> list[dict]:
    return [item for item in base_result.get("ranked_actions", []) if isinstance(item, dict)]


def _unresolved_hazards(base_result: dict) -> list[dict]:
    unresolved = [
        item
        for item in base_result.get("unresolved_hazards", [])
        if isinstance(item, dict) and item.get("hazard_code")
    ]
    if not unresolved and not _ranked_actions(base_result):
        unresolved = [
            item
            for item in base_result.get("preliminary_ranked_hazards", [])
            if isinstance(item, dict) and item.get("hazard_code")
        ]
    unresolved.sort(
        key=lambda item: (
            int(item.get("preliminary_priority_rank") or 10**6),
            -float(item.get("preliminary_priority_raw", 0.0)),
        )
    )
    return unresolved


@lru_cache(maxsize=1)
def _hazard_need_map() -> dict[str, set[str]]:
    path = ROOT / "data" / "factor_hazard_bridge.csv"
    mapping: dict[str, set[str]] = {}
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            hazard = str(row.get("hazard_code") or "").strip()
            need = str(row.get("need_code") or "").strip()
            if hazard and need:
                mapping.setdefault(hazard, set()).add(need)
    return mapping


def _need_scores(base_result: dict) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in base_result.get("need_profiles", []):
        if not isinstance(item, dict) or not item.get("need_code"):
            continue
        result[str(item["need_code"])] = float(item.get("personal_need_score", 0.0))
    return result


def _adaptive_ai_candidates(base_result: dict, floorplan: dict) -> list[dict]:
    """공간위험이 아직 확정되지 않은 경우의 추가 확인 질문 후보를 만든다.

    후보는 현재 공간에서 사용자에게 실제로 노출된 맞춤 문항으로만 제한한다.
    우선순위는 Local SHAP/interaction으로 계산된 need profile을 먼저 사용하고,
    동점일 때 문항 우선순위를 사용한다. 존재 여부는 확정하지 않는다.
    """
    candidates = [
        item
        for item in floorplan.get("adaptive_hazard_candidates", [])
        if isinstance(item, dict) and item.get("hazard_code")
    ]
    if not candidates:
        return []

    needs_by_hazard = _hazard_need_map()
    need_scores = _need_scores(base_result)
    high_risk = str(base_result.get("risk_label") or "") == "HIGH_RISK"
    prepared: list[dict] = []
    seen: set[str] = set()

    for item in candidates:
        code = str(item.get("hazard_code"))
        if code in seen:
            continue
        seen.add(code)
        matched_needs = sorted(needs_by_hazard.get(code, set()) & set(need_scores))
        personal_score = sum(need_scores[need] for need in matched_needs)
        clarification_answer = str(item.get("clarification_answer") or "UNKNOWN").upper()
        # 사용자가 추가 질문에서 아니오로 확정한 위험은 이후 GPT 조건부 후보에서도 제외한다.
        if clarification_answer == "NO":
            continue
        checked = bool(item.get("checked")) or clarification_answer == "YES"
        # 체크된 위험은 공간 확인 신호가 있으므로 우선 포함한다. 미체크 후보는
        # SHAP need가 있거나 모델 고위험일 때만 조건부 후보로 사용한다.
        if not checked and personal_score <= 0 and not high_risk:
            continue
        question_priority = int(item.get("priority") or 999)
        fallback_score = personal_score if personal_score > 0 else 1.0 / max(question_priority, 1)
        prepared.append(
            {
                "hazard_code": code,
                "room_id": item.get("room_id"),
                "room_type": item.get("room_type"),
                "object_id": item.get("object_id"),
                "object_type": item.get("object_type"),
                "matched_need_codes": matched_needs,
                "matched_main_factors": [],
                "matched_interactions": [],
                "personal_risk_contribution": personal_score,
                "preliminary_priority_raw": fallback_score,
                "question_priority": question_priority,
                "candidate_label": item.get("label"),
                "trigger_labels": item.get("trigger_labels", []),
                "confirmation_status": (
                    str(item.get("confirmation_status"))
                    if item.get("confirmation_status")
                    else "CONFIRMED_BY_USER_CHECKLIST"
                    if checked
                    else "UNCONFIRMED_SPACE_CANDIDATE"
                ),
                "clarification_answer": clarification_answer,
                "priority_basis": (
                    "PERSONAL_SHAP_NEED_SCORE_PLUS_CONFIRMED_CHECKLIST"
                    if checked
                    else "PERSONAL_SHAP_NEED_SCORE_AND_ROOM_SPECIFIC_CANDIDATE"
                ),
                "mapping_failure_reason": "NO_CONFIRMED_RAG_MAPPABLE_SPATIAL_HAZARD",
            }
        )

    prepared.sort(
        key=lambda item: (
            item.get("confirmation_status") != "CONFIRMED_BY_USER_CHECKLIST",
            -float(item.get("preliminary_priority_raw", 0.0)),
            int(item.get("question_priority", 999)),
            str(item.get("hazard_code")),
        )
    )
    for index, item in enumerate(prepared[:MAX_AI_ONLY_STEPS], start=1):
        item["preliminary_priority_rank"] = index
        item["final_priority_rank"] = index
    return prepared[:MAX_AI_ONLY_STEPS]


def _planning_context(base_result: dict, floorplan: dict) -> tuple[str, list[dict]]:
    ranked = _ranked_actions(base_result)
    if ranked:
        modes = {
            str(item.get("recommendation_source") or ("RAG_EVIDENCE" if item.get("selected_evidence") else "AI_ONLY_NO_RAG"))
            for item in ranked
        }
        if "RAG_EVIDENCE" in modes and "AI_ONLY_NO_RAG" in modes:
            return "HYBRID_RAG_AI", ranked
        if modes == {"AI_ONLY_NO_RAG"}:
            return "AI_ONLY_NO_RAG", ranked
        return "RAG_EVIDENCE", ranked

    unresolved = _unresolved_hazards(base_result)
    if unresolved:
        prepared: list[dict] = []
        for index, item in enumerate(unresolved[:MAX_AI_ONLY_STEPS], start=1):
            prepared.append(
                {
                    **item,
                    "final_priority_rank": int(item.get("preliminary_priority_rank") or index),
                    "confirmation_status": "CONFIRMED_SPATIAL_HAZARD",
                    "priority_basis": "SHAP_INTERACTION_X_SPATIAL_PRESENCE_X_CONFIDENCE",
                }
            )
        return "AI_ONLY_NO_RAG", prepared

    adaptive = _adaptive_ai_candidates(base_result, floorplan)
    if adaptive:
        return "AI_ONLY_PRECAUTIONARY", adaptive
    return "NO_ACTIONS", []


def _validation_base(base_result: dict, evidence_mode: str, items: list[dict]) -> dict:
    prepared = deepcopy(base_result)
    if evidence_mode in {"AI_ONLY_NO_RAG", "AI_ONLY_PRECAUTIONARY"}:
        prepared["ai_only_candidates"] = deepcopy(items)
        prepared["ai_only_evidence_mode"] = evidence_mode
    return prepared


def build_planner_payload(base_result: dict, floorplan: dict) -> dict:
    evidence_mode, items = _planning_context(base_result, floorplan)
    payload: dict[str, Any] = {
        "fixed_priority_policy": (
            "RAG_EVIDENCE, AI_ONLY_NO_RAG, HYBRID_RAG_AI의 fixed_priority_rank는 "
            "실제 공간위험 baseline × (1+Local SHAP/interaction 개인화) × 탐지신뢰도 결과이며 변경하지 않는다. "
            "AI_ONLY_PRECAUTIONARY는 공간위험 미확정 상태이므로 개인 SHAP need와 현재 공간 후보 순서를 고정한다."
        ),
        "evidence_mode": evidence_mode,
        "fall_risk_context": {
            "calibrated_probability": base_result.get("calibrated_probability"),
            "selected_threshold": base_result.get("selected_threshold"),
            "risk_label": base_result.get("risk_label"),
        },
        "floorplan": _compact_floorplan(floorplan),
        "fixed_ranked_actions": [],
        "fixed_ai_candidates": [],
    }

    if evidence_mode in {"RAG_EVIDENCE", "HYBRID_RAG_AI"}:
        rag_actions: list[dict] = []
        ai_candidates: list[dict] = []
        for index, action in enumerate(items, start=1):
            evidence = action.get("selected_evidence", {}) or {}
            source_mode = str(
                action.get("recommendation_source")
                or ("RAG_EVIDENCE" if evidence else "AI_ONLY_NO_RAG")
            )
            common = {
                "hazard_instance_id": action.get("hazard_instance_id") or f"HZ_FALLBACK_{index:03d}",
                "hazard_code": action.get("hazard_code"),
                "fixed_priority_rank": int(action.get("final_priority_rank") or index),
                "fixed_priority_score": action.get("final_priority_score"),
                "room_id": action.get("room_id"),
                "room_type": action.get("room_type"),
                "object_id": action.get("object_id"),
                "object_type": action.get("object_type"),
                "anchor_geometry": action.get("anchor_geometry"),
                "detected_from": action.get("detected_from", []),
                "matched_need_codes": action.get("matched_need_codes", []),
                "matched_main_factors": action.get("matched_main_factors", []),
                "matched_interactions": action.get("matched_interactions", []),
                "personal_risk_contribution": action.get("personal_risk_contribution"),
                "environmental_hazard_baseline": action.get("environmental_hazard_baseline"),
                "priority_basis": action.get("priority_basis"),
                "confirmation_status": action.get("confirmation_status"),
            }
            if source_mode == "RAG_EVIDENCE":
                rag_actions.append({
                    **common,
                    "evidence": {
                        "evidence_id": evidence.get("evidence_id"),
                        "recommendation": evidence.get("recommendation"),
                        "expected_effect": evidence.get("expected_effect"),
                        "source": evidence.get("source"),
                        "doi": evidence.get("doi"),
                        "tier": evidence.get("tier"),
                    },
                })
            else:
                ai_candidates.append({
                    **common,
                    "candidate_label": action.get("candidate_label"),
                    "mapping_failure_reason": action.get("mapping_failure_reason"),
                })
        payload["fixed_ranked_actions"] = rag_actions
        payload["fixed_ai_candidates"] = ai_candidates
        payload["generation_scope"] = (
            "hazard별 source를 독립적으로 유지한다. RAG_EVIDENCE 항목은 제공된 문헌 recommendation 범위 안에서 통합하고, "
            "AI_ONLY_NO_RAG 항목은 evidence_ids=[]로 보수적 일반 공간개선안을 생성한다. "
            "두 종류 모두 fixed_priority_rank와 hazard_instance_id를 변경하지 않는다."
        )
    elif evidence_mode == "AI_ONLY_NO_RAG":
        payload["generation_scope"] = (
            "구조화 RAG evidence가 없는 확인 완료 hazard에 대해 evidence_ids=[]인 보수적 공간개선 step을 생성한다. "
            "fixed_priority_rank와 hazard_instance_id는 변경하지 않는다."
        )
        payload["fixed_ai_candidates"] = [
            {
                "hazard_instance_id": item.get("hazard_instance_id") or f"AI_FALLBACK_{index:03d}",
                "hazard_code": item.get("hazard_code"),
                "fixed_priority_rank": int(item.get("final_priority_rank") or index),
                "room_id": item.get("room_id"),
                "room_type": item.get("room_type"),
                "object_id": item.get("object_id"),
                "object_type": item.get("object_type"),
                "matched_need_codes": item.get("matched_need_codes", []),
                "matched_main_factors": item.get("matched_main_factors", []),
                "matched_interactions": item.get("matched_interactions", []),
                "personal_risk_contribution": item.get("personal_risk_contribution"),
                "environmental_hazard_baseline": item.get("environmental_hazard_baseline"),
                "confirmation_status": item.get("confirmation_status"),
                "priority_basis": item.get("priority_basis"),
                "candidate_label": item.get("candidate_label"),
                "detected_from": item.get("detected_from", []),
                "mapping_failure_reason": item.get("mapping_failure_reason"),
            }
            for index, item in enumerate(items, start=1)
        ]
    elif evidence_mode == "AI_ONLY_PRECAUTIONARY":
        payload["generation_scope"] = (
            "공간위험 존재가 아직 확정되지 않았으므로 솔루션을 만들지 않고 confirmation 질문만 생성한다."
        )
        payload["fixed_ai_candidates"] = [
            {
                "hazard_instance_id": item.get("hazard_instance_id") or f"PRECAUTION_{index:03d}",
                "hazard_code": item.get("hazard_code"),
                "fixed_priority_rank": int(item.get("final_priority_rank") or index),
                "room_id": item.get("room_id"),
                "room_type": item.get("room_type"),
                "object_id": item.get("object_id"),
                "object_type": item.get("object_type"),
                "matched_need_codes": item.get("matched_need_codes", []),
                "personal_risk_contribution": item.get("personal_risk_contribution"),
                "confirmation_status": item.get("confirmation_status"),
                "priority_basis": item.get("priority_basis"),
                "candidate_label": item.get("candidate_label"),
            }
            for index, item in enumerate(items, start=1)
        ]
    else:
        payload["generation_scope"] = "개인 위험과 연결할 공간 후보가 없어 계획을 생성하지 않는다."

    return payload

def _base_plan(*, title: str, summary: str, limitations: list[str]) -> dict:
    return {
        "plan_title": title,
        "summary": summary,
        "priority_policy": "FIXED_FROM_SHAP_SPATIAL_PRIORITY",
        "steps": [],
        "cross_solution_conflicts": [],
        "clarification_questions": [],
        "limitations": limitations,
    }


def _rag_template_plan(base_result: dict, reason: str) -> dict:
    plan = _base_plan(
        title="문헌 근거 기반 공간개선 통합계획",
        summary="확정된 위험 순위와 선택된 문헌 recommendation을 그대로 유지한 기본 계획입니다.",
        limitations=[reason],
    )
    for action in _ranked_actions(base_result):
        evidence = action.get("selected_evidence", {}) or {}
        rank = int(action.get("final_priority_rank", len(plan["steps"]) + 1))
        plan["steps"].append(
            {
                "hazard_instance_id": action.get("hazard_instance_id"),
                "hazard_code": action.get("hazard_code"),
                "fixed_priority_rank": rank,
                "implementation_sequence": rank,
                "evidence_ids": [evidence.get("evidence_id")] if evidence.get("evidence_id") else [],
                "integrated_instruction": str(evidence.get("recommendation") or ""),
                "sequence_reason": "고정된 사용자별 위험 우선순위를 따른 기본 실행 순서",
                "target": {
                    "placement_zone": action.get("object_id") or action.get("room_type"),
                    "avoid_zones": [],
                },
                "dependencies": [],
                "conflicts_considered": [],
                "implementation_note": "GPT 비활성 또는 호출 실패로 선택 문헌 recommendation을 그대로 유지",
            }
        )
    return plan


def _ai_only_template_plan(items: list[dict], evidence_mode: str, reason: str) -> dict:
    precautionary = evidence_mode == "AI_ONLY_PRECAUTIONARY"
    plan = _base_plan(
        title=(
            "공간위험 추가 확인"
            if precautionary
            else "RAG 미매핑 위험 AI 독자 공간개선안"
        ),
        summary=(
            "공간위험이 아직 확정되지 않아 솔루션을 생성하지 않고 추가 확인 질문만 제공합니다."
            if precautionary
            else "문헌 매핑은 없지만 확인된 공간위험과 고정 우선순위를 바탕으로 AI 독자 개선안을 제공합니다."
        ),
        limitations=[
            reason,
            "NO_SELECTED_RAG_EVIDENCE",
            *(
                ["SOLUTION_WITHHELD_UNTIL_SPATIAL_HAZARD_CONFIRMATION"]
                if precautionary
                else ["AI_GENERATED_SOLUTION_REQUIRES_HUMAN_REVIEW"]
            ),
        ],
    )
    for index, item in enumerate(items, start=1):
        code = str(item.get("hazard_code"))
        if precautionary:
            question = HAZARD_QUESTION_TEMPLATES.get(code)
            if question:
                plan["clarification_questions"].append(
                    {
                        "question_id": f"AQ{index}",
                        "question": question,
                        "reason": "솔루션 생성 전에 공간위험의 실제 존재 여부를 확정하기 위한 질문",
                        "answer_type": "YES_NO_UNKNOWN",
                        "related_hazard_codes": [code],
                    }
                )
            continue

        rank = int(item.get("final_priority_rank") or item.get("preliminary_priority_rank") or index)
        instruction = AI_SOLUTION_TEMPLATES.get(
            code,
            "현재 공간에서 해당 위험을 줄일 수 있도록 가구와 동선을 단순화하고 적용 전 현장 상태를 다시 확인합니다.",
        )
        plan["steps"].append(
            {
                "hazard_instance_id": item.get("hazard_instance_id") or f"AI_FALLBACK_{index:03d}",
                "hazard_code": code,
                "fixed_priority_rank": rank,
                "implementation_sequence": rank,
                "evidence_ids": [],
                "integrated_instruction": instruction,
                "sequence_reason": "개인 SHAP·interaction과 확인된 공간위험으로 계산된 순위를 유지",
                "target": {
                    "placement_zone": item.get("object_id") or item.get("room_type"),
                    "avoid_zones": [],
                },
                "dependencies": [],
                "conflicts_considered": [],
                "implementation_note": "RAG 문헌 근거가 없는 AI 독자 제안이므로 전문가 또는 현장 검토 후 적용",
            }
        )
    return plan


def _hybrid_template_plan(items: list[dict], reason: str) -> dict:
    """RAG 매핑/미매핑 hazard를 같은 고정 위험순위 안에서 함께 출력합니다."""
    plan = _base_plan(
        title="근거·AI 혼합 공간개선 통합계획",
        summary="확정된 위험 순위는 유지하고, hazard별로 RAG 문헌 또는 AI 보완안을 사용한 기본 계획입니다.",
        limitations=[reason, "AI_ONLY_ITEMS_HAVE_NO_RAG_EVIDENCE_AND_REQUIRE_HUMAN_REVIEW"],
    )
    for index, item in enumerate(items, start=1):
        rank = int(item.get("final_priority_rank") or item.get("preliminary_priority_rank") or index)
        code = str(item.get("hazard_code"))
        instance_id = item.get("hazard_instance_id") or f"HZ_FALLBACK_{index:03d}"
        source_mode = str(item.get("recommendation_source") or ("RAG_EVIDENCE" if item.get("selected_evidence") else "AI_ONLY_NO_RAG"))
        evidence = item.get("selected_evidence", {}) or {}
        if source_mode == "RAG_EVIDENCE":
            instruction = str(evidence.get("recommendation") or "")
            evidence_ids = [evidence.get("evidence_id")] if evidence.get("evidence_id") else []
            note = "GPT 비활성 또는 호출 실패로 선택 문헌 recommendation을 그대로 유지"
        else:
            instruction = AI_SOLUTION_TEMPLATES.get(
                code,
                "현재 공간에서 해당 위험을 줄일 수 있도록 가구와 동선을 단순화하고 적용 전 현장 상태를 다시 확인합니다.",
            )
            evidence_ids = []
            note = "RAG 문헌 근거가 없는 AI 독자 제안이므로 전문가 또는 현장 검토 후 적용"
        plan["steps"].append({
            "hazard_instance_id": instance_id,
            "hazard_code": code,
            "fixed_priority_rank": rank,
            "implementation_sequence": rank,
            "evidence_ids": evidence_ids,
            "integrated_instruction": instruction,
            "sequence_reason": "RAG coverage와 무관하게 고정된 hazard priority rank를 유지",
            "target": {
                "placement_zone": item.get("object_id") or item.get("room_type"),
                "avoid_zones": [],
            },
            "dependencies": [],
            "conflicts_considered": [],
            "implementation_note": note,
        })
    return plan


def _clarification_answers_by_hazard(
    clarification_questions: list[dict[str, Any]] | None,
    answers_by_question_id: dict[str, Any] | None,
) -> tuple[dict[str, str], bool]:
    """추가 질문 응답을 hazard 단위 YES/NO/UNKNOWN으로 정규화합니다."""
    questions = [item for item in (clarification_questions or []) if isinstance(item, dict)]
    raw_answers = dict(answers_by_question_id or {})
    priority = {"UNKNOWN": 0, "NO": 1, "YES": 2}
    answers_by_hazard: dict[str, str] = {}
    for index, question in enumerate(questions, start=1):
        question_id = str(question.get("question_id") or f"Q{index}")
        raw = str(raw_answers.get(question_id) or "UNKNOWN").strip().upper()
        answer = {
            "예": "YES",
            "네": "YES",
            "Y": "YES",
            "YES": "YES",
            "아니오": "NO",
            "아니요": "NO",
            "N": "NO",
            "NO": "NO",
            "모름": "UNKNOWN",
            "UNKNOWN": "UNKNOWN",
            "": "UNKNOWN",
        }.get(raw, raw)
        if answer not in {"YES", "NO", "UNKNOWN"}:
            answer = "UNKNOWN"
        for code in question.get("related_hazard_codes", []):
            hazard_code = str(code).strip()
            if not hazard_code:
                continue
            previous = answers_by_hazard.get(hazard_code, "UNKNOWN")
            if priority[answer] >= priority[previous]:
                answers_by_hazard[hazard_code] = answer

    return answers_by_hazard, any(value == "UNKNOWN" for value in answers_by_hazard.values())


def _clarification_followup_candidates(
    base_result: dict,
    floorplan: dict,
    clarification_questions: list[dict[str, Any]] | None,
    answers_by_question_id: dict[str, Any] | None,
) -> tuple[list[dict], list[dict]]:
    """기존 SHAP/interaction 결과와 질문 순서를 재사용해 확인된 후보만 추립니다.

    이 함수는 모델 예측, SHAP, 공간위험 탐지, RAG 검색을 다시 실행하지 않습니다.
    """
    answers_by_hazard, _ = _clarification_answers_by_hazard(
        clarification_questions,
        answers_by_question_id,
    )
    original_candidates = _adaptive_ai_candidates(base_result, floorplan)
    by_code = {
        str(item.get("hazard_code")): item
        for item in original_candidates
        if isinstance(item, dict) and item.get("hazard_code")
    }

    confirmed: list[dict] = []
    pending: list[dict] = []
    seen: set[str] = set()
    for question_index, question in enumerate(
        [item for item in (clarification_questions or []) if isinstance(item, dict)],
        start=1,
    ):
        for code_value in question.get("related_hazard_codes", []):
            code = str(code_value).strip()
            if not code or code in seen:
                continue
            answer = answers_by_hazard.get(code, "UNKNOWN")
            if answer == "NO":
                seen.add(code)
                continue
            source = deepcopy(by_code.get(code, {}))
            rank = int(
                source.get("final_priority_rank")
                or source.get("preliminary_priority_rank")
                or question_index
            )
            if answer == "UNKNOWN":
                pending.append(
                    {
                        **source,
                        "hazard_code": code,
                        "final_priority_rank": rank,
                        "preliminary_priority_rank": rank,
                    }
                )
                seen.add(code)
                continue

            seen.add(code)
            confirmed.append(
                {
                    **source,
                    "hazard_code": code,
                    "final_priority_rank": rank,
                    "preliminary_priority_rank": rank,
                    "confirmation_status": "CONFIRMED_PRESENT_BY_CLARIFICATION",
                    "clarification_answer": "YES",
                    "priority_basis": (
                        source.get("priority_basis")
                        or "REUSED_PERSONAL_SHAP_NEED_AND_CLARIFICATION_CONFIRMATION"
                    ),
                    "candidate_label": (
                        source.get("candidate_label")
                        or question.get("question")
                        or code
                    ),
                    "mapping_failure_reason": "NO_RAG_EVIDENCE_PREVIOUS_ANALYSIS",
                    "detected_from": [
                        *list(source.get("detected_from", [])),
                        f"clarification:{question.get('question_id') or question_index}=YES",
                    ],
                }
            )

    confirmed.sort(
        key=lambda item: (
            int(item.get("final_priority_rank") or 10**6),
            str(item.get("hazard_code") or ""),
        )
    )
    for new_rank, item in enumerate(confirmed[:MAX_AI_ONLY_STEPS], start=1):
        # 기존 질문/후보 우선순서를 유지하되, NO로 제외된 후보 때문에 생긴 빈 순위만 압축합니다.
        item["final_priority_rank"] = new_rank
        item["preliminary_priority_rank"] = new_rank
    pending.sort(
        key=lambda item: (
            int(item.get("final_priority_rank") or 10**6),
            str(item.get("hazard_code") or ""),
        )
    )
    return confirmed[:MAX_AI_ONLY_STEPS], pending[:MAX_AI_ONLY_STEPS]

def _call_openai(payload: dict) -> dict:
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai 패키지를 불러오지 못했습니다. requirements.txt 설치가 필요합니다.") from exc

    model = os.getenv("OPENAI_MODEL", "gpt-5.5")
    timeout_seconds = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90"))
    client = OpenAI(timeout=timeout_seconds)
    evidence_mode = str(payload.get("evidence_mode") or "RAG_EVIDENCE")
    generation_origin = str(payload.get("generation_origin") or "")

    if generation_origin == "CLARIFICATION_DIRECT_AI":
        instructions = f"""
당신은 추가 확인 질문의 답변을 반영해 공간개선 카드를 직접 생성하는 AI입니다.
이 요청은 이전 분석에서 이미 계산된 SHAP, SHAP interaction, 개인 기능위험 후보 순서를 재사용합니다.
모델 예측, SHAP 계산, 공간위험 탐지, RAG 검색을 다시 수행하거나 그 결과를 새로 추정하지 마십시오.
현재 fixed_ai_candidates는 사용자가 '예'라고 답해 실제 존재를 확인한 RAG 미매핑 공간위험입니다.

반드시 지킬 것:
- fixed_ai_candidates의 각 hazard_instance_id에 대해 정확히 1개의 step 생성
- fixed_priority_rank를 절대 변경하지 않음
- evidence_ids는 항상 빈 배열 []
- clarification_questions는 반드시 빈 배열 []
- 입력에 없는 hazard, need_code, evidence_id를 만들지 않음
- 특정 제품명, 가격, 법적 기준, 정밀 규격·높이·수치, 확정적 의학 효과를 만들지 않음
- 현재 floorplan에 없는 객체가 있다고 단정하지 않음
- 질문 답변으로 확인된 위험을 직접 줄이는 간단하고 실행 가능한 공간개선안을 작성
- floorplan.room_note와 floorplan.spatial_annotations에 사용자가 직접 적은 구체적 위치/상태 설명이 있으면 절대 무시하지 말고, 해당 hazard와 관련되는 내용을 integrated_instruction에 반영
- integrated_instruction은 가능하면 2~4개의 구체적 실행 요소를 포함: (현재 관찰된 문제) + (정확한 적용 위치/대상) + (무엇을 어떻게 바꿀지) + (왜 필요한지). 단, 입력에 없는 수치나 물건은 만들지 않음
- summary와 limitations에 '추가 질문 반영 AI 독자 제안'이며 RAG 문헌 근거가 없음을 명시

출력은 지정된 JSON Schema를 정확히 따르십시오.
""".strip()
    elif evidence_mode == "AI_ONLY_PRECAUTIONARY":
        instructions = f"""
당신은 고령자 낙상예방 주거환경의 미확정 위험을 확인하는 AI입니다.
현재 fixed_ai_candidates는 개인 SHAP과 관련된 확인 후보일 뿐, 실제 공간위험으로 확정되지 않았습니다.
따라서 질문에 답하기 전에는 어떠한 솔루션도 생성하면 안 됩니다.

반드시 지킬 것:
- steps는 반드시 빈 배열 []
- cross_solution_conflicts는 반드시 빈 배열 []
- 조건부 솔루션, 임시 조치, 대안, 권고 문장도 생성하지 않음
- clarification_questions만 최대 {MAX_AI_ONLY_STEPS}개 생성
- 질문은 fixed_ai_candidates의 hazard_code 존재 여부를 확인하는 예/아니오/모름 질문으로 작성
- related_hazard_codes에는 입력에 있는 hazard_code만 사용
- 질문의 의미는 예=위험 존재, 아니오=위험 부재, 모름=미확정으로 일관되게 작성
- summary에는 '위험 확정 전 솔루션을 보류하고 질문만 제공한다'고 명시
- 입력에 없는 hazard, need_code, evidence_id를 만들지 않음

출력은 지정된 JSON Schema를 정확히 따르십시오.
""".strip()
    elif evidence_mode == "HYBRID_RAG_AI":
        instructions = """
당신은 고령자 낙상예방 주거환경 개입계획을 통합하는 생성형 AI입니다.
현재 입력에는 문헌 근거가 있는 RAG_EVIDENCE hazard와 문헌 미매핑 AI_ONLY_NO_RAG hazard가 함께 있습니다.

반드시 지킬 것:
- fixed_ranked_actions와 fixed_ai_candidates의 모든 hazard_instance_id에 대해 정확히 1개의 step을 생성
- hazard_instance_id와 fixed_priority_rank를 절대 변경하지 않음
- fixed_ranked_actions는 제공된 evidence_id와 recommendation 범위만 사용
- fixed_ai_candidates는 evidence_ids를 반드시 []로 유지하고 보수적인 일반 공간개선안만 생성
- RAG coverage를 이유로 위험 순서를 재정렬하거나 미매핑 hazard를 누락하지 않음
- 입력에 없는 hazard, 객체, evidence_id, 제품명, 가격, 법적 기준, 정밀 규격, 확정적 의학 효과를 만들지 않음
- priority rank와 implementation sequence는 구분하며 priority는 변경하지 않음
- floorplan.room_note와 floorplan.spatial_annotations의 사용자 메모가 해당 hazard와 관련되면 integrated_instruction과 implementation_note에 구체적으로 반영
- 문헌 recommendation의 의미 범위는 유지하되, 사용자 메모에 나온 실제 위치·가구명·좁은 간격·낮은 높이·물기 등 관찰 사실을 이용해 적용 대상을 구체화. integrated_instruction은 한 문장짜리 추상 권고보다 2~4개의 실행 요소가 드러나게 작성

출력은 지정된 JSON Schema를 정확히 따르십시오.
""".strip()
    elif evidence_mode == "AI_ONLY_NO_RAG":
        instructions = f"""
당신은 고령자 낙상예방 주거환경 개선안을 생성하는 AI입니다.
현재 구조화 RAG evidence는 0건이지만 fixed_ai_candidates의 공간위험은 확인되었습니다.
입력의 fixed_ai_candidates에 한정하여 실용적이고 보수적인 공간개선 step을 생성해야 합니다.

반드시 지킬 것:
- fixed_ai_candidates의 각 hazard_instance_id에 대해 정확히 1개의 step 생성
- fixed_priority_rank를 절대 변경하지 않음
- evidence_ids는 항상 빈 배열 []
- 입력에 없는 hazard, need_code, evidence_id를 만들지 않음
- 특정 제품명, 가격, 법적 기준, 정밀 규격·높이·수치, 확정적 의학 효과를 만들지 않음
- 현재 floorplan에 없는 객체가 있다고 단정하지 않음
- 간단하고 실행 가능한 정리·재배치·조명·지지·표시 중심으로 제안
- summary와 limitations에 'RAG 문헌 미매핑 AI 독자 제안'임을 명시
- clarification_questions는 최대 3개
- 추가 질문은 related_hazard_codes의 적용 가능성을 묻는 예/아니오/모름 질문으로 작성
- 공간위험은 확인되었으므로 해당 위험을 직접 낮추는 실행안을 작성
- floorplan.room_note와 floorplan.spatial_annotations에 사용자가 직접 적은 문제를 해당 hazard와 연결해 반영. 특히 "A와 B 사이가 좁다", "가구가 낮다", "특정 위치가 미끄럽다" 같은 구체 메모는 integrated_instruction에서 위치와 조치가 분명히 보이도록 사용
- integrated_instruction은 가능하면 2~4개의 구체적 실행 요소를 포함하고, 단순히 "정리한다/설치한다" 한 줄로 끝내지 않음

출력은 지정된 JSON Schema를 정확히 따르십시오.
""".strip()
    else:
        instructions = """
당신은 고령자 낙상예방 주거환경 개입계획을 통합하는 생성형 AI입니다.
반드시 입력 JSON의 fixed_priority_rank를 그대로 유지하십시오.
각 step의 hazard_instance_id와 evidence_ids에는 입력에 제공된 값을 그대로 사용하십시오.
문헌 recommendation의 범위를 벗어나지 마십시오.

해야 할 일:
1) 고정된 각 위험을 한 공간에서 충돌하지 않도록 통합합니다.
2) priority rank와 implementation sequence를 구분합니다. priority는 절대 변경하지 않습니다.
3) 평면도에서 확인 가능한 room/object/path/attribute를 근거로 적용 위치를 설명합니다.
4) 솔루션끼리 충돌 가능성이 있으면 명시합니다.
5) 답에 따라 계획이 실제로 달라지는 경우에만 최대 3개의 질문을 생성합니다. 질문은 related_hazard_codes의 위험이 실제 공간에 존재하는지 또는 적용 가능한지를 묻고, 예=존재·적용 가능, 아니오=부재·적용 불가, 모름=미확정으로 해석되게 작성합니다.
6) floorplan.room_note와 floorplan.spatial_annotations의 사용자 직접 메모가 관련 hazard와 연결되면 반드시 적용 위치와 현재 상태를 구체적으로 반영합니다. integrated_instruction은 (관찰된 문제 → 적용 대상/위치 → 구체 조치 → 이유)가 보이게 작성합니다.

금지:
- 입력에 없는 hazard 생성
- SHAP 점수 또는 fixed_priority_rank 변경
- 입력에 없는 evidence_id 생성
- 평면도에 없는 객체가 있다고 단정
- 특정 제품명, 가격, 법적 기준, 정확한 규격·높이·수치, 의학적 효과 생성

출력은 지정된 JSON Schema를 정확히 따르십시오.
""".strip()

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "integrated_space_plan",
                "strict": True,
                "schema": PLAN_JSON_SCHEMA,
            }
        },
    )
    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("OpenAI 응답에 output_text가 없습니다.")
    result = json.loads(output_text)
    result["openai_model"] = model
    return result


def _finalize_report(
    plan: dict,
    report: dict,
    *,
    planner_mode: str,
    evidence_mode: str,
    gpt_api_called: bool,
) -> tuple[dict, dict]:
    plan["planner_mode"] = planner_mode
    plan["evidence_mode"] = evidence_mode
    plan["gpt_api_called"] = bool(gpt_api_called)
    report["planner_mode"] = planner_mode
    report["evidence_mode"] = evidence_mode
    report["gpt_api_called"] = bool(gpt_api_called)
    report["step_count"] = len(plan.get("steps", []))
    report["question_count"] = len(plan.get("clarification_questions", []))
    return plan, report


def generate_clarification_followup_plan(
    base_result: dict,
    floorplan: dict,
    clarification_questions: list[dict[str, Any]] | None,
    answers_by_question_id: dict[str, Any] | None,
    *,
    use_gpt: bool = True,
) -> tuple[dict, dict]:
    """추가 질문 답변만 반영해 AI 카드를 생성합니다.

    이전 분석의 SHAP/interaction 및 공간 후보 순서를 재사용하며,
    예측·SHAP·hazard detector·RAG 파이프라인은 다시 실행하지 않습니다.
    """
    load_dotenv()
    fail_mode = os.getenv("GPT_FAIL_MODE", "fallback").strip().lower()
    confirmed_items, pending_items = _clarification_followup_candidates(
        base_result,
        floorplan,
        clarification_questions,
        answers_by_question_id,
    )

    if not confirmed_items:
        if pending_items:
            # 사용자가 아직 모름으로 남긴 후보만 질문 전용 상태로 유지합니다.
            plan = _ai_only_template_plan(
                pending_items,
                "AI_ONLY_PRECAUTIONARY",
                "추가 질문에 미확정 응답이 남아 AI 솔루션 생성을 보류",
            )
            working_base = _validation_base(
                base_result,
                "AI_ONLY_PRECAUTIONARY",
                pending_items,
            )
            plan, report = validate_and_normalize_plan(plan, working_base)
            return _finalize_report(
                plan,
                report,
                planner_mode="CLARIFICATION_STILL_PENDING",
                evidence_mode="AI_ONLY_PRECAUTIONARY",
                gpt_api_called=False,
            )

        plan = _base_plan(
            title="추가 확인 결과 공간개선 대상 없음",
            summary="추가 질문에서 후보 공간위험이 존재하지 않는 것으로 확인되어 AI 솔루션 카드를 생성하지 않았습니다.",
            limitations=["CLARIFICATION_CONFIRMED_NO_TARGET_HAZARD"],
        )
        plan, report = validate_and_normalize_plan(plan, {})
        return _finalize_report(
            plan,
            report,
            planner_mode="CLARIFICATION_NO_ACTIONS",
            evidence_mode="NO_ACTIONS",
            gpt_api_called=False,
        )

    evidence_mode = "AI_ONLY_NO_RAG"
    working_base = _validation_base(base_result, evidence_mode, confirmed_items)
    payload = {
        "generation_origin": "CLARIFICATION_DIRECT_AI",
        "fixed_priority_policy": (
            "이전 분석의 Local SHAP·interaction 기반 후보 순서와 추가 질문 답변을 재사용한다. "
            "SHAP, 공간위험 탐지, RAG를 다시 실행하지 않고 fixed_priority_rank를 변경하지 않는다."
        ),
        "evidence_mode": evidence_mode,
        "fall_risk_context": {
            "calibrated_probability": base_result.get("calibrated_probability"),
            "selected_threshold": base_result.get("selected_threshold"),
            "risk_label": base_result.get("risk_label"),
        },
        "floorplan": _compact_floorplan(floorplan),
        "fixed_ranked_actions": [],
        "fixed_ai_candidates": [
            {
                "hazard_code": item.get("hazard_code"),
                "fixed_priority_rank": int(item.get("final_priority_rank") or index),
                "room_id": item.get("room_id"),
                "room_type": item.get("room_type"),
                "object_id": item.get("object_id"),
                "object_type": item.get("object_type"),
                "matched_need_codes": item.get("matched_need_codes", []),
                "matched_main_factors": item.get("matched_main_factors", []),
                "matched_interactions": item.get("matched_interactions", []),
                "personal_risk_contribution": item.get("personal_risk_contribution"),
                "confirmation_status": item.get("confirmation_status"),
                "priority_basis": item.get("priority_basis"),
                "candidate_label": item.get("candidate_label"),
                "detected_from": item.get("detected_from", []),
                "mapping_failure_reason": item.get("mapping_failure_reason"),
            }
            for index, item in enumerate(confirmed_items, start=1)
        ],
        "generation_scope": (
            "추가 질문에서 사용자가 존재한다고 확인한 RAG 미매핑 위험만 대상으로 "
            "AI 솔루션 카드를 직접 생성한다. 전체 분석은 반복하지 않는다."
        ),
    }

    def fallback(reason: str, *, api_called: bool) -> tuple[dict, dict]:
        plan = _ai_only_template_plan(confirmed_items, evidence_mode, reason)
        plan["clarification_questions"] = []
        plan["summary"] = (
            "추가 질문 답변으로 확인된 RAG 미매핑 공간위험에 대해 "
            "기존 분석 순서를 재사용하여 AI 독자 개선안을 생성했습니다."
        )
        plan, report = validate_and_normalize_plan(plan, working_base)
        return _finalize_report(
            plan,
            report,
            planner_mode=(
                "CLARIFICATION_DIRECT_AI_TEMPLATE_AFTER_GPT_ERROR"
                if api_called
                else "CLARIFICATION_DIRECT_AI_TEMPLATE"
            ),
            evidence_mode=evidence_mode,
            gpt_api_called=api_called,
        )

    if not use_gpt:
        return fallback("use_gpt=false · 전체 분석 재실행 없음", api_called=False)

    if not os.getenv("OPENAI_API_KEY"):
        if fail_mode == "error":
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
        return fallback("OPENAI_API_KEY 미설정 · 전체 분석 재실행 없음", api_called=False)

    try:
        raw_plan = _call_openai(payload)
        raw_plan["clarification_questions"] = []

        # Clarification GPT가 confirmed hazard 일부를 누락하거나 임시 HZ ID를 반환해도
        # 사용자에게 오류를 띄우지 않습니다. 서버의 confirmed_items를 source of truth로 두고
        # GPT step을 hazard별로 정렬한 뒤 누락분만 deterministic template로 보완합니다.
        raw_steps = [step for step in raw_plan.get("steps", []) if isinstance(step, dict)]
        template_plan = _ai_only_template_plan(
            confirmed_items, evidence_mode, "clarification GPT 누락 step 서버 자동 보완"
        )
        template_steps = [step for step in template_plan.get("steps", []) if isinstance(step, dict)]
        used_raw: set[int] = set()
        aligned_steps: list[dict] = []

        def _raw_rank(step: dict) -> int:
            try:
                return int(step.get("fixed_priority_rank") or 0)
            except (TypeError, ValueError):
                return 0

        for idx, expected in enumerate(confirmed_items, start=1):
            expected_id = str(expected.get("hazard_instance_id") or "").strip()
            expected_code = str(expected.get("hazard_code") or "").strip()
            expected_rank = int(
                expected.get("final_priority_rank")
                or expected.get("preliminary_priority_rank")
                or idx
            )

            chosen_index = None
            # 1) 정확한 instance ID
            for raw_idx, step in enumerate(raw_steps):
                if raw_idx in used_raw:
                    continue
                if expected_id and str(step.get("hazard_instance_id") or "").strip() == expected_id:
                    chosen_index = raw_idx
                    break
            # 2) hazard code + 고정 순위
            if chosen_index is None:
                for raw_idx, step in enumerate(raw_steps):
                    if raw_idx in used_raw:
                        continue
                    if (expected_code
                        and str(step.get("hazard_code") or "").strip() == expected_code
                        and _raw_rank(step) == expected_rank):
                        chosen_index = raw_idx
                        break
            # 3) 고정 순위만 일치
            if chosen_index is None:
                rank_matches = [
                    raw_idx for raw_idx, step in enumerate(raw_steps)
                    if raw_idx not in used_raw and _raw_rank(step) == expected_rank
                ]
                if len(rank_matches) == 1:
                    chosen_index = rank_matches[0]
            # 4) 아직 남은 GPT step을 순서대로 사용
            if chosen_index is None:
                remaining = [i for i in range(len(raw_steps)) if i not in used_raw]
                if remaining:
                    chosen_index = remaining[0]

            if chosen_index is not None:
                chosen = dict(raw_steps[chosen_index])
                used_raw.add(chosen_index)
            else:
                chosen = dict(template_steps[idx - 1]) if idx - 1 < len(template_steps) else {}

            # ID/code/rank는 GPT가 아니라 서버에서 고정합니다.
            chosen["hazard_instance_id"] = expected_id or f"AI_FALLBACK_{idx:03d}"
            chosen["hazard_code"] = expected_code
            chosen["fixed_priority_rank"] = expected_rank
            if not str(chosen.get("integrated_instruction") or "").strip():
                fallback = template_steps[idx - 1] if idx - 1 < len(template_steps) else {}
                chosen["integrated_instruction"] = fallback.get(
                    "integrated_instruction",
                    "확인된 공간위험을 줄이도록 현재 배치와 동선을 안전하게 조정합니다.",
                )
            aligned_steps.append(chosen)

        raw_plan["steps"] = aligned_steps
        plan, report = validate_and_normalize_plan(raw_plan, working_base)
        return _finalize_report(
            plan,
            report,
            planner_mode="OPENAI_CLARIFICATION_DIRECT_AI",
            evidence_mode=evidence_mode,
            gpt_api_called=True,
        )
    except (Exception, PlanValidationError) as exc:
        if fail_mode == "error":
            raise
        plan, report = fallback(
            f"GPT 호출 또는 검증 실패: {type(exc).__name__}: {exc}",
            api_called=True,
        )
        report["gpt_error"] = f"{type(exc).__name__}: {exc}"
        return plan, report

def generate_integrated_plan(
    base_result: dict,
    floorplan: dict,
    *,
    use_gpt: bool = True,
) -> tuple[dict, dict]:
    """RAG 매핑 성공 시 문헌 통합, 실패 시 AI 독자 솔루션을 생성합니다."""
    load_dotenv()
    fail_mode = os.getenv("GPT_FAIL_MODE", "fallback").strip().lower()
    evidence_mode, items = _planning_context(base_result, floorplan)
    working_base = _validation_base(base_result, evidence_mode, items)

    if evidence_mode == "NO_ACTIONS":
        plan = _base_plan(
            title="공간개선 후보 없음",
            summary="현재 입력에서 개인 SHAP 위험과 연결할 공간위험 또는 확인 후보를 찾지 못했습니다.",
            limitations=["NO_PERSONAL_SPATIAL_CANDIDATE_TO_PLAN"],
        )
        plan, report = validate_and_normalize_plan(plan, working_base)
        return _finalize_report(
            plan,
            report,
            planner_mode="NO_ACTIONS",
            evidence_mode="NO_ACTIONS",
            gpt_api_called=False,
        )

    if not use_gpt:
        plan = (
            _rag_template_plan(base_result, "use_gpt=false")
            if evidence_mode == "RAG_EVIDENCE"
            else _hybrid_template_plan(items, "use_gpt=false")
            if evidence_mode == "HYBRID_RAG_AI"
            else _ai_only_template_plan(items, evidence_mode, "use_gpt=false")
        )
        plan, report = validate_and_normalize_plan(plan, working_base)
        mode = "TEMPLATE_FALLBACK" if evidence_mode == "RAG_EVIDENCE" else f"{evidence_mode}_TEMPLATE"
        return _finalize_report(plan, report, planner_mode=mode, evidence_mode=evidence_mode, gpt_api_called=False)

    if not os.getenv("OPENAI_API_KEY"):
        if fail_mode == "error":
            raise RuntimeError("OPENAI_API_KEY가 설정되지 않았습니다.")
        plan = (
            _rag_template_plan(base_result, "OPENAI_API_KEY 미설정")
            if evidence_mode == "RAG_EVIDENCE"
            else _hybrid_template_plan(items, "OPENAI_API_KEY 미설정")
            if evidence_mode == "HYBRID_RAG_AI"
            else _ai_only_template_plan(items, evidence_mode, "OPENAI_API_KEY 미설정")
        )
        plan, report = validate_and_normalize_plan(plan, working_base)
        mode = (
            "TEMPLATE_FALLBACK_NO_API_KEY"
            if evidence_mode == "RAG_EVIDENCE"
            else f"{evidence_mode}_TEMPLATE_NO_API_KEY"
        )
        return _finalize_report(plan, report, planner_mode=mode, evidence_mode=evidence_mode, gpt_api_called=False)

    payload = build_planner_payload(base_result, floorplan)
    try:
        raw_plan = _call_openai(payload)
        plan, report = validate_and_normalize_plan(raw_plan, working_base)
        mode = (
            "OPENAI_RAG_INTEGRATION"
            if evidence_mode == "RAG_EVIDENCE"
            else f"OPENAI_{evidence_mode}"
        )
        return _finalize_report(plan, report, planner_mode=mode, evidence_mode=evidence_mode, gpt_api_called=True)
    except (Exception, PlanValidationError) as exc:
        if fail_mode == "error":
            raise
        plan = (
            _rag_template_plan(base_result, f"GPT 호출 또는 검증 실패: {type(exc).__name__}: {exc}")
            if evidence_mode == "RAG_EVIDENCE"
            else _hybrid_template_plan(items, f"GPT 호출 또는 검증 실패: {type(exc).__name__}: {exc}")
            if evidence_mode == "HYBRID_RAG_AI"
            else _ai_only_template_plan(
                items,
                evidence_mode,
                f"GPT 호출 또는 검증 실패: {type(exc).__name__}: {exc}",
            )
        )
        plan, report = validate_and_normalize_plan(plan, working_base)
        mode = (
            "TEMPLATE_FALLBACK_AFTER_GPT_ERROR"
            if evidence_mode == "RAG_EVIDENCE"
            else f"{evidence_mode}_TEMPLATE_AFTER_GPT_ERROR"
        )
        report["gpt_error"] = f"{type(exc).__name__}: {exc}"
        return _finalize_report(plan, report, planner_mode=mode, evidence_mode=evidence_mode, gpt_api_called=True)

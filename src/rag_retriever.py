from __future__ import annotations

from copy import deepcopy
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOM_TYPE_TO_KO = {
    "LIVING_ROOM": "거실",
    "BEDROOM": "침실",
    "KITCHEN": "주방",
    "BATHROOM": "욕실",
    "HALLWAY": "복도",
    "STAIR": "계단",
    "ENTRANCE": "현관",
    "OUTDOOR": "주거 외부",
}


DEFAULT_SOLUTION_DISPLAY_COUNT = 3


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _tokens(value) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    return {token.strip() for token in str(value).replace(",", "|").split("|") if token.strip()}


def _room_match(row: pd.Series, room_type: str | None) -> bool:
    room_ko = ROOM_TYPE_TO_KO.get(room_type or "")
    if not room_ko:
        return True
    categories = _tokens(row.get("space_categories"))
    if not categories:
        categories = _tokens(row.get("space"))
    return room_ko in categories or "주거공간 전반" in categories


def _vector_rank(query: str, candidates: pd.DataFrame) -> pd.DataFrame:
    docs = candidates["search_text"].fillna(candidates.get("evidence_text", "")).fillna("").astype(str).tolist()
    if len(docs) == 1:
        similarities = np.array([1.0])
    else:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        matrix = vectorizer.fit_transform([query] + docs)
        similarities = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    ranked = candidates.assign(_vector_similarity=similarities)
    ranked["_evidence_score"] = pd.to_numeric(ranked.get("evidence_score"), errors="coerce").fillna(0.0)
    ranked["_sms"] = pd.to_numeric(ranked.get("sms"), errors="coerce").fillna(0.0)
    return ranked.sort_values(
        ["_vector_similarity", "_evidence_score", "_sms"],
        ascending=[False, False, False],
        kind="mergesort",
    )


def _nullable(value):
    return None if value is None or pd.isna(value) else value


def _evidence_payload(row: pd.Series, *, candidate_rank: int) -> dict:
    """RAG 후보 행을 UI/GPT가 공통으로 사용할 구조로 변환합니다."""
    return {
        "solution_candidate_rank_within_hazard": int(candidate_rank),
        "evidence_id": _nullable(row.get("evidence_id")),
        "factor_1": _nullable(row.get("factor_1")),
        "factor_2": _nullable(row.get("factor_2")),
        "factor_codes": _nullable(row.get("factor_codes")),
        "need_codes": _nullable(row.get("need_codes")),
        "hazard_codes": _nullable(row.get("hazard_codes")),
        "space": _nullable(row.get("space")),
        "space_categories": _nullable(row.get("space_categories")),
        "target_object": _nullable(row.get("target_object")),
        "recommendation": _nullable(row.get("recommendation")),
        "expected_effect": _nullable(row.get("expected_effect")),
        "evidence_type_standard": _nullable(row.get("evidence_type_standard")),
        "evidence_score": float(row.get("_evidence_score", 0.0)),
        "vector_similarity": float(row.get("_vector_similarity", 0.0)),
        "sms": float(row.get("_sms", 0.0)),
        "tier": _nullable(row.get("tier")),
        "source": _nullable(row.get("source")),
        "doi": _nullable(row.get("doi")),
        "source_url": _nullable(row.get("source_url")),
        "review_status": _nullable(row.get("review_status")),
        "structured_mapping_status": _nullable(row.get("structured_mapping_status")),
    }


def _dedupe_key(row: pd.Series) -> str:
    """동일 문헌행 또는 사실상 같은 권고 문구의 중복 노출을 방지합니다."""
    evidence_id = _nullable(row.get("evidence_id"))
    if evidence_id:
        return f"ID::{evidence_id}"
    recommendation = re.sub(r"\s+", "", str(_nullable(row.get("recommendation")) or "")).lower()
    target = re.sub(r"\s+", "", str(_nullable(row.get("target_object")) or "")).lower()
    return f"TEXT::{recommendation}::{target}"


def _candidate_evidences(ranked_candidates: pd.DataFrame) -> list[dict]:
    evidences: list[dict] = []
    seen: set[str] = set()
    for _, row in ranked_candidates.iterrows():
        key = _dedupe_key(row)
        if key in seen:
            continue
        seen.add(key)
        evidences.append(_evidence_payload(row, candidate_rank=len(evidences) + 1))
    return evidences


def _solution_candidate_action(
    action: dict,
    evidence: dict,
    *,
    solution_rank: int,
    primary_for_hazard: bool,
) -> dict:
    """GPT용 주 행동을 건드리지 않고 화면용 솔루션 후보를 생성합니다."""
    payload = {
        key: value
        for key, value in action.items()
        if key not in {"solution_candidates", "selected_evidence"}
    }
    payload.update(
        {
            "solution_rank": int(solution_rank),
            "hazard_priority_rank": int(action.get("final_priority_rank", solution_rank)),
            "solution_candidate_rank_within_hazard": int(
                evidence.get("solution_candidate_rank_within_hazard", 1)
            ),
            "is_primary_for_hazard": bool(primary_for_hazard),
            "selected_evidence": evidence,
        }
    )
    return payload


def _build_ranked_solution_candidates(ranked_actions: list[dict]) -> list[dict]:
    """상위 위험별 대표안 우선, 이후 동일 위험의 대안을 순차 추가합니다.

    정책
    ----
    1. 각 hazard의 1순위 RAG 솔루션을 hazard 위험순위대로 먼저 배치합니다.
    2. 표시 후보가 더 필요하면 각 hazard의 2순위, 3순위 ... 대안을
       같은 hazard 위험순서로 순환하여 추가합니다.

    이 방식은 상위 3개 화면에서 서로 다른 핵심 위험을 우선 보여주면서,
    실제 매핑 위험이 1개뿐인 경우에는 해당 위험의 대안 솔루션으로 3개를 채웁니다.
    """
    ordered: list[dict] = []

    # 1차: 위험별 대표 솔루션 한 개씩
    for action in ranked_actions:
        candidates = action.get("solution_candidates", [])
        if not candidates:
            continue
        ordered.append(
            _solution_candidate_action(
                action,
                candidates[0],
                solution_rank=len(ordered) + 1,
                primary_for_hazard=True,
            )
        )

    # 2차: 위험별 추가 대안, 후보 차수별 round-robin
    max_candidates = max(
        (len(action.get("solution_candidates", [])) for action in ranked_actions),
        default=0,
    )
    for candidate_index in range(1, max_candidates):
        for action in ranked_actions:
            candidates = action.get("solution_candidates", [])
            if candidate_index >= len(candidates):
                continue
            ordered.append(
                _solution_candidate_action(
                    action,
                    candidates[candidate_index],
                    solution_rank=len(ordered) + 1,
                    primary_for_hazard=False,
                )
            )
    return ordered


def evaluate_feasibility_and_retrieve(
    preliminary_plan: dict,
    enriched_rag_db: pd.DataFrame,
    top_n: int = 7,
) -> dict:
    """각 hazard의 고정 위험순위를 유지한 채 해결안 출처만 hazard별로 결정합니다.

    핵심 정책
    ---------
    1. 공간위험 우선순위는 이 함수에 들어오기 전에 이미 확정되며 RAG가 변경하지 않습니다.
    2. 각 hazard는 독립적으로 RAG_EVIDENCE 또는 AI_ONLY_NO_RAG 출처를 가집니다.
    3. RAG 미매핑 hazard도 결과에서 제거하지 않으며 원래 hazard rank를 그대로 유지합니다.
    4. SHAP need가 없는 객관적 공간위험은 hazard_code + room으로 일반 RAG 근거를 검색합니다.
    """
    result = deepcopy(preliminary_plan)
    db = enriched_rag_db.copy()
    if "eligible" in db.columns:
        db = db[db["eligible"].map(_truthy)]

    all_actions: list[dict] = []
    unresolved: list[dict] = []
    feasible: list[dict] = []
    feasibility_records: list[dict] = []

    hazards = [
        item for item in result.get("preliminary_ranked_hazards", [])
        if isinstance(item, dict)
    ][:top_n]
    max_priority = max(
        (float(item.get("preliminary_priority_raw", 0.0)) for item in hazards),
        default=0.0,
    )

    for fallback_rank, hazard in enumerate(hazards, start=1):
        hazard_code = str(hazard["hazard_code"])
        need_codes = set(hazard.get("matched_need_codes", []))
        fixed_rank = int(hazard.get("preliminary_priority_rank") or fallback_rank)
        raw_score = float(hazard.get("preliminary_priority_raw", 0.0))
        normalized_score = raw_score / max_priority if max_priority else 0.0

        mask_hazard = db["hazard_codes"].map(lambda value: hazard_code in _tokens(value))
        # 개인 SHAP need가 없어도 실제 확인된 공간위험은 일반 hazard 근거를 검색할 수 있어야 한다.
        if need_codes:
            mask_need = db["need_codes"].map(lambda value: bool(_tokens(value) & need_codes))
        else:
            mask_need = pd.Series(True, index=db.index)
        mask_room = db.apply(lambda row: _room_match(row, hazard.get("room_type")), axis=1)
        candidates = db[mask_hazard & mask_need & mask_room].copy()

        feasibility = {
            "hazard_instance_id": hazard.get("hazard_instance_id"),
            "hazard_code": hazard_code,
            "room_type": hazard.get("room_type"),
            "matched_need_codes": sorted(need_codes),
            "preliminary_priority_rank": fixed_rank,
            "preliminary_priority_raw": raw_score,
            "mapping_feasible": not candidates.empty,
            "metadata_candidate_count": int(len(candidates)),
            "filter_policy": (
                "eligible + hazard_code + need_code + room"
                if need_codes
                else "eligible + hazard_code + room (no SHAP need gate)"
            ),
        }
        feasibility_records.append(feasibility)

        base_action = {
            **hazard,
            "recommendation_rank": fixed_rank,
            "final_priority_rank": fixed_rank,
            "hazard_priority_rank": fixed_rank,
            "final_priority_raw": raw_score,
            "final_priority_score": normalized_score,
            "priority_raw": raw_score,
            "priority_score": normalized_score,
        }

        if candidates.empty:
            unresolved_item = {
                **base_action,
                "mapping_feasible": False,
                "mapping_failure_reason": "NO_STRUCTURED_RAG_MATCH",
                "recommendation_source": "AI_ONLY_NO_RAG",
                "selected_evidence": {},
                "solution_candidates": [],
                "confirmation_status": hazard.get("confirmation_status") or "CONFIRMED_SPATIAL_HAZARD",
            }
            unresolved.append(unresolved_item)
            all_actions.append(unresolved_item)
            continue

        room_ko = ROOM_TYPE_TO_KO.get(hazard.get("room_type"), hazard.get("room_type", ""))
        factor_text = " ".join(
            str(item.get("feature", "")) for item in hazard.get("matched_main_factors", [])
        )
        interaction_text = " ".join(
            f"{item.get('feature_1', '')} × {item.get('feature_2', '')}"
            for item in hazard.get("matched_interactions", [])
        )
        query = (
            f"need={' '.join(sorted(need_codes))} hazard={hazard_code} room={room_ko} "
            f"factor={factor_text} interaction={interaction_text} 구체적인 공간 개선"
        )
        ranked_candidates = _vector_rank(query, candidates)
        candidate_evidences = _candidate_evidences(ranked_candidates)
        if not candidate_evidences:
            feasibility["mapping_feasible"] = False
            unresolved_item = {
                **base_action,
                "mapping_feasible": False,
                "mapping_failure_reason": "NO_UNIQUE_RAG_SOLUTION_AFTER_DEDUPLICATION",
                "recommendation_source": "AI_ONLY_NO_RAG",
                "selected_evidence": {},
                "solution_candidates": [],
                "confirmation_status": hazard.get("confirmation_status") or "CONFIRMED_SPATIAL_HAZARD",
            }
            unresolved.append(unresolved_item)
            all_actions.append(unresolved_item)
            continue

        best = candidate_evidences[0]
        mapped_item = {
            **base_action,
            "mapping_feasible": True,
            "recommendation_source": "RAG_EVIDENCE",
            "rag": {
                "retrieval_engine": "STRUCTURED_METADATA_FILTER_THEN_LOCAL_TFIDF",
                "metadata_candidate_count": int(len(ranked_candidates)),
                "unique_solution_candidate_count": int(len(candidate_evidences)),
                "query": query,
                "vector_similarity": float(best.get("vector_similarity", 0.0)),
                "risk_priority_influenced_by_rag": False,
                "solution_selection_tiebreak": "vector_similarity > evidence_score > sms",
            },
            "selected_evidence": best,
            "solution_candidates": candidate_evidences,
        }
        feasible.append(mapped_item)
        all_actions.append(mapped_item)

    # 원래 hazard rank를 끝까지 보존한다. RAG coverage 때문에 순위를 압축하거나 재부여하지 않는다.
    all_actions.sort(key=lambda item: int(item.get("final_priority_rank") or 10**6))
    feasible.sort(key=lambda item: int(item.get("final_priority_rank") or 10**6))
    unresolved.sort(key=lambda item: int(item.get("final_priority_rank") or 10**6))
    ranked_solution_candidates = _build_ranked_solution_candidates(feasible)

    if feasible and unresolved:
        generation_mode = "HYBRID_RAG_AI"
    elif feasible:
        generation_mode = "RAG_EVIDENCE"
    elif unresolved:
        generation_mode = "AI_ONLY_NO_RAG"
    else:
        generation_mode = "NO_ACTIONS"

    result.update({
        "solution_feasibility_policy": "HAZARD_LEVEL_SOURCE_SELECTION_WITHOUT_RISK_RERANK",
        "final_rerank_policy": "PRESERVE_ORIGINAL_HAZARD_PRIORITY_RANK_REGARDLESS_OF_RAG_COVERAGE",
        "solution_listing_policy": (
            "FIXED_HAZARD_PRIORITY; RAG_ALTERNATIVES_ONLY_WITHIN_MAPPED_HAZARDS; "
            "AI_FALLBACK_FOR_UNMAPPED_HAZARDS"
        ),
        "solution_mapping_feasibility": feasibility_records,
        "feasible_hazard_count": len(feasible),
        "unresolved_hazard_count": len(unresolved),
        "unresolved_hazards": unresolved,
        "requested_top_n": top_n,
        "solution_generation_mode": generation_mode,
        "selected_recommendation_count": len(all_actions),
        # 모든 확인된 hazard를 포함한다. 각 item의 recommendation_source가 근거 출처를 결정한다.
        "ranked_actions": all_actions,
        "mapped_primary_actions": feasible,
        "mapped_solution_candidate_count": len(ranked_solution_candidates),
        "default_solution_display_count": min(DEFAULT_SOLUTION_DISPLAY_COUNT, len(all_actions)),
        "ranked_solution_candidates": ranked_solution_candidates,
    })
    return result


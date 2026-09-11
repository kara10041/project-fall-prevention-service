from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOM_EN_TO_KO = {
    "LIVING_ROOM": "거실",
    "BEDROOM": "침실",
    "KITCHEN": "주방",
    "BATHROOM": "욕실",
    "HALLWAY": "복도",
    "STAIR": "계단",
    "ENTRANCE": "현관",
    "OUTDOOR": "주거 외부",
    "ALL": "주거공간 전반",
}


def _split(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [x.strip() for x in str(value).replace(",", "|").split("|") if x.strip()]


def normalize_factor_text(value) -> str:
    """공백·괄호·구두점 차이만 정규화한다. 의미 기반 fuzzy 추론은 하지 않는다."""
    if value is None or pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return re.sub(r"[\s_\-–—·ㆍ,./\\()\[\]{}:;]+", "", text)


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _ontology_index(ontology: pd.DataFrame) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = defaultdict(list)
    for row in ontology.itertuples(index=False):
        index[normalize_factor_text(row.model_feature)].append({
            "model_feature": str(row.model_feature),
            "factor_code": str(row.factor_code),
            "need_code": str(row.need_code),
        })
    return index


def _object_alias_index(alias_table: pd.DataFrame) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in alias_table.itertuples(index=False):
        result[str(row.target_object_type)].add(str(row.target_object_keyword).strip())
    return result


def _room_match_for_bridge(rag_row: pd.Series, bridge_rooms: list[str]) -> bool:
    rag_rooms = set(_split(rag_row.get("space_categories"))) or set(_split(rag_row.get("space")))
    if "주거공간 전반" in rag_rooms or "ALL" in bridge_rooms:
        return True
    bridge_ko = {ROOM_EN_TO_KO.get(room, room) for room in bridge_rooms}
    return bool(rag_rooms & bridge_ko)


def _target_match_for_bridge(target_text: str, target_type: str, aliases: dict[str, set[str]]) -> bool:
    keywords = aliases.get(target_type, set())
    return bool(keywords) and any(keyword and keyword in target_text for keyword in keywords)


def _override_index(overrides: pd.DataFrame | None) -> dict[str, str]:
    if overrides is None or overrides.empty:
        return {}
    result = {}
    for row in overrides.itertuples(index=False):
        evidence_id = str(getattr(row, "evidence_id", "")).strip()
        hazard_codes = "|".join(sorted(set(_split(getattr(row, "hazard_codes", "")))))
        review_status = str(getattr(row, "review_status", "")).strip()
        if evidence_id and hazard_codes and review_status.startswith("확정"):
            result[evidence_id] = hazard_codes
    return result


def enrich_rag_dataframe(
    rag: pd.DataFrame,
    ontology: pd.DataFrame,
    bridge: pd.DataFrame,
    target_alias_table: pd.DataFrame,
    hazard_overrides: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    원본 RAG에 구조화 코드를 부여한다.

    자동 사용 조건
    -------------
    - factor/alias가 ontology model_feature와 정규화 exact match
    - need + room + target object로 도출된 candidate hazard가 정확히 1개
    - 또는 rag_hazard_overrides.csv에 사람이 확정한 hazard_codes가 있음

    candidate hazard가 2개 이상이면 자동 retrieval에서 제외하고 review 대상으로 남긴다.
    """
    ontology_idx = _ontology_index(ontology)
    object_aliases = _object_alias_index(target_alias_table)
    overrides = _override_index(hazard_overrides)

    bridge_by_need: dict[str, list[dict]] = defaultdict(list)
    for row in bridge.itertuples(index=False):
        bridge_by_need[str(row.need_code)].append({
            "hazard_code": str(row.hazard_code),
            "rooms": _split(row.applicable_rooms),
            "target_object_type": str(row.target_object_type),
        })

    out = rag.copy()
    columns = {
        "ontology_model_features": [],
        "factor_codes": [],
        "need_codes": [],
        "factor_code_match_method": [],
        "candidate_hazard_codes": [],
        "hazard_codes": [],
        "hazard_mapping_method": [],
        "structured_mapping_status": [],
    }

    counts = defaultdict(int)

    for _, row in out.iterrows():
        source_values: list[tuple[str, str]] = []
        for col in ("factor_1", "factor_2"):
            value = row.get(col)
            if value is not None and not pd.isna(value) and str(value).strip():
                source_values.append((col, str(value).strip()))
        for alias in _split(row.get("factor_aliases")):
            source_values.append(("factor_aliases", alias))

        matched: dict[str, dict] = {}
        methods: set[str] = set()
        for source_col, value in source_values:
            for item in ontology_idx.get(normalize_factor_text(value), []):
                matched[item["model_feature"]] = item
                methods.add("ALIAS_EXACT_NORMALIZED" if source_col == "factor_aliases" else "FACTOR_EXACT_NORMALIZED")

        factor_codes = sorted({m["factor_code"] for m in matched.values()})
        need_codes = sorted({m["need_code"] for m in matched.values() if m["need_code"]})
        model_features = sorted(matched)
        if need_codes:
            counts["need_mapped_rows"] += 1
        if "FACTOR_EXACT_NORMALIZED" in methods:
            counts["factor_exact_rows"] += 1
        if "ALIAS_EXACT_NORMALIZED" in methods:
            counts["alias_exact_rows"] += 1

        target_text = str(row.get("target_object", ""))
        candidates: set[str] = set()
        for need_code in need_codes:
            for bridge_row in bridge_by_need.get(need_code, []):
                if not _room_match_for_bridge(row, bridge_row["rooms"]):
                    continue
                if not _target_match_for_bridge(target_text, bridge_row["target_object_type"], object_aliases):
                    continue
                candidates.add(bridge_row["hazard_code"])

        evidence_id = str(row.get("evidence_id", "")).strip()
        if evidence_id in overrides:
            resolved_hazards = overrides[evidence_id]
            hazard_method = "HUMAN_CONFIRMED_OVERRIDE"
            status = "FACTOR_NEED_HAZARD_CONFIRMED"
            counts["override_rows"] += 1
        elif len(candidates) == 1:
            resolved_hazards = next(iter(candidates))
            hazard_method = "UNIQUE_NEED_ROOM_TARGET_MATCH"
            status = "FACTOR_NEED_HAZARD_UNIQUE"
            counts["unique_hazard_rows"] += 1
        elif len(candidates) > 1:
            resolved_hazards = ""
            hazard_method = "AMBIGUOUS_REVIEW_REQUIRED"
            status = "AMBIGUOUS_HAZARD_MAPPING_REVIEW_REQUIRED"
            counts["ambiguous_hazard_rows"] += 1
        elif need_codes:
            resolved_hazards = ""
            hazard_method = "NO_HAZARD_CANDIDATE"
            status = "NEED_MAPPED_HAZARD_UNRESOLVED"
            counts["unresolved_hazard_rows"] += 1
        else:
            resolved_hazards = ""
            hazard_method = "FACTOR_UNMATCHED"
            status = "FACTOR_UNMATCHED"
            counts["factor_unmatched_rows"] += 1

        columns["ontology_model_features"].append("|".join(model_features))
        columns["factor_codes"].append("|".join(factor_codes))
        columns["need_codes"].append("|".join(need_codes))
        columns["factor_code_match_method"].append("|".join(sorted(methods)))
        columns["candidate_hazard_codes"].append("|".join(sorted(candidates)))
        columns["hazard_codes"].append(resolved_hazards)
        columns["hazard_mapping_method"].append(hazard_method)
        columns["structured_mapping_status"].append(status)

    for name, values in columns.items():
        out[name] = values

    eligible_mask = out["eligible"].map(_truthy) if "eligible" in out.columns else pd.Series(True, index=out.index)
    summary = {
        "rag_row_count": int(len(out)),
        "eligible_row_count": int(eligible_mask.sum()),
        "factor_or_alias_exact_mapped_rows": int(counts["need_mapped_rows"]),
        "factor_column_exact_rows": int(counts["factor_exact_rows"]),
        "alias_exact_rows": int(counts["alias_exact_rows"]),
        "unique_hazard_auto_mapped_rows": int(counts["unique_hazard_rows"]),
        "human_override_rows": int(counts["override_rows"]),
        "ambiguous_hazard_review_rows": int(counts["ambiguous_hazard_rows"]),
        "eligible_resolved_hazard_rows": int((eligible_mask & out["hazard_codes"].astype(str).ne("")).sum()),
        "mapping_policy": "NORMALIZED_EXACT_FACTOR_OR_ALIAS + UNIQUE_NEED_ROOM_TARGET; AMBIGUOUS_EXCLUDED_UNLESS_HUMAN_OVERRIDE",
    }
    return out, summary


def ensure_enriched_rag(
    rag: pd.DataFrame,
    ontology: pd.DataFrame,
    bridge: pd.DataFrame,
    target_alias_table: pd.DataFrame,
    hazard_overrides: pd.DataFrame | None = None,
    output_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    required = {"factor_codes", "need_codes", "hazard_codes", "structured_mapping_status"}
    if required.issubset(rag.columns):
        return rag.copy(), {
            "rag_row_count": int(len(rag)),
            "reused_existing_structured_columns": True,
            "mapping_policy": "PRE_ENRICHED_RAG",
        }

    enriched, summary = enrich_rag_dataframe(
        rag,
        ontology,
        bridge,
        target_alias_table,
        hazard_overrides=hazard_overrides,
    )
    summary["reused_existing_structured_columns"] = False
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        enriched.to_csv(path, index=False, encoding="utf-8-sig")
        summary["enriched_rag_path"] = str(path)
    return enriched, summary

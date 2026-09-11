from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.hazard_detector import detect_hazards
from src.spatial_mapper import rank_spatial_hazards
from src.adaptive_space_questions import QUESTION_BANK

ROOT = Path(__file__).resolve().parents[1]

ROOM_TYPE_MAP = {
    "living_room": "LIVING_ROOM",
    "bedroom": "BEDROOM",
    "kitchen": "KITCHEN",
    "bathroom": "BATHROOM",
}
ROOM_NAME_BY_TYPE = {value: key for key, value in ROOM_TYPE_MAP.items()}

# hazard_code는 여러 방에서 재사용되지만 방마다 문구(label)가 다르므로,
# (room_name, hazard_code) 튜플을 키로 써서 방별로 올바른 문구를 조회한다.
# 예: LOW_LIGHTING -> 거실 "조명이 어두움" / 화장실 "화장실 조명이 어두움"
HAZARD_LABELS_BY_ROOM: dict[tuple[str, str], str] = {
    (room_name, str(item["hazard_code"])): str(item["label"])
    for room_name, questions in QUESTION_BANK.items()
    for item in questions
}


def _hazard_label(room_name: str, hazard_code: str) -> str:
    """방별 hazard 문구를 조회한다. 해당 방의 QUESTION_BANK에 없는 코드라면
    (이론상 발생하지 않아야 하지만) 다른 방의 문구라도 안전하게 폴백한다."""
    label = HAZARD_LABELS_BY_ROOM.get((room_name, hazard_code))
    if label is not None:
        return label
    for (r, code), fallback_label in HAZARD_LABELS_BY_ROOM.items():
        if code == hazard_code:
            return fallback_label
    return hazard_code


@lru_cache(maxsize=1)
def _load_mapping_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ontology = pd.read_csv(ROOT / "data" / "factor_ontology.csv", encoding="utf-8-sig")
    bridge = pd.read_csv(ROOT / "data" / "factor_hazard_bridge.csv", encoding="utf-8-sig")
    interaction = pd.read_csv(ROOT / "data" / "interaction_hazard_bridge.csv", encoding="utf-8-sig")
    return ontology, bridge, interaction


def build_priority_floorplan(flags_by_room: dict[str, dict[str, bool]]) -> dict[str, Any]:
    """공간 우선순위 산출용 최소 floorplan을 만든다.

    이 단계에서는 가구 좌표를 추정하지 않는다. 사용자가 체크한 Hazard만
    explicit observation으로 넣어 기존 hazard_detector를 그대로 통과시킨다.
    """
    rooms: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []

    for room_name, room_type in ROOM_TYPE_MAP.items():
        room_id = f"room_{room_name}_priority"
        rooms.append(
            {
                "room_id": room_id,
                "room_type": room_type,
                "polygon": [],
                "attributes": {},
            }
        )
        for hazard_code, present in (flags_by_room.get(room_name) or {}).items():
            if not bool(present):
                continue
            observations.append(
                {
                    "observation_code": str(hazard_code),
                    "room_id": room_id,
                    "value": True,
                    "confidence": 1.0,
                    "observed_by": "PRE_PRIORITY_USER_CHECKLIST",
                    "note": "공간 개선 우선순위 산출 전 사용자 체크리스트에서 확인",
                }
            )

    return {
        "plan_id": "PRE_PRIORITY_HAZARD_CHECK",
        "coordinate_system": {"unit": "cm", "origin": "top_left", "y_axis": "down"},
        "canvas": {"width": 1, "height": 1},
        "rooms": rooms,
        "objects": [],
        "paths": [],
        "observations": observations,
    }


def aggregate_room_priorities(
    ranked_hazards: list[dict[str, Any]],
    *,
    secondary_weight: float = 0.30,
) -> list[dict[str, Any]]:
    """Hazard 우선순위를 공간 단위의 서비스용 권장 순서로 집계한다.

    room_raw_score = 가장 높은 Hazard raw score
                     + secondary_weight * 나머지 Hazard raw score 합

    이 점수는 임상 위험도가 아니라 서비스 내 '개선 권장 순서'용 집계값이다.
    """
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in ROOM_TYPE_MAP}
    for hazard in ranked_hazards:
        room_name = ROOM_NAME_BY_TYPE.get(str(hazard.get("room_type") or ""))
        if room_name:
            grouped.setdefault(room_name, []).append(hazard)

    rows: list[dict[str, Any]] = []
    for room_name in ROOM_TYPE_MAP:
        hazards = sorted(
            grouped.get(room_name, []),
            key=lambda item: float(item.get("preliminary_priority_raw", 0.0)),
            reverse=True,
        )
        hazards = [
            {**item, "hazard_label": _hazard_label(room_name, str(item.get("hazard_code", "")))}
            for item in hazards
        ]
        scores = [float(item.get("preliminary_priority_raw", 0.0)) for item in hazards]
        room_raw = scores[0] + secondary_weight * sum(scores[1:]) if scores else 0.0
        rows.append(
            {
                "room": room_name,
                "room_raw_score": room_raw,
                "hazard_count": len(hazards),
                "hazards": hazards,
            }
        )

    rows.sort(key=lambda item: (item["room_raw_score"], item["hazard_count"]), reverse=True)
    max_score = max((item["room_raw_score"] for item in rows), default=0.0)
    positive_rank = 0
    for item in rows:
        if item["room_raw_score"] > 0:
            positive_rank += 1
            item["rank"] = positive_rank
            item["room_priority_score"] = item["room_raw_score"] / max_score if max_score else 0.0
        else:
            item["rank"] = None
            item["room_priority_score"] = 0.0
    return rows


def compute_room_priorities(
    *,
    shap_explanation: dict[str, Any],
    flags_by_room: dict[str, dict[str, bool]],
    secondary_weight: float = 0.30,
) -> dict[str, Any]:
    floorplan = build_priority_floorplan(flags_by_room)
    detected = detect_hazards(floorplan, selected_room=None)
    detected.pop("normalized_floorplan", None)

    ontology, bridge, interaction = _load_mapping_tables()
    preliminary = rank_spatial_hazards(
        shap_explanation,
        detected,
        ontology,
        bridge,
        interaction_bridge=interaction,
        include_review=False,
    )
    ranked_hazards = preliminary.get("preliminary_ranked_hazards", [])
    ranked_rooms = aggregate_room_priorities(
        ranked_hazards,
        secondary_weight=secondary_weight,
    )

    return {
        "room_aggregation_formula": "MAX_HAZARD_RAW + 0.30 * SUM(OTHER_HAZARD_RAW)",
        "room_score_is_clinical_risk": False,
        "detected_hazard_count": len(detected.get("hazard_instances", [])),
        "ranked_hazard_count": len(ranked_hazards),
        "ranked_hazards": ranked_hazards,
        "ranked_rooms": ranked_rooms,
    }

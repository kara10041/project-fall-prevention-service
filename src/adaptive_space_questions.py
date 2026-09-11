from __future__ import annotations

from typing import Any


ROOM_LABELS = {
    "living_room": "거실",
    "bedroom": "침실",
    "kitchen": "주방",
    "bathroom": "화장실·욕실",
}

SIGNAL_LABELS = {
    "VISUAL_GUIDANCE": "시력 불편",
    "ALERT_SUPPORT": "청력 불편",
    "LOWER_LIMB_SUPPORT": "하지 근력·관절 지지 필요",
    "WALKING_SUPPORT": "보행·계단 이동 제한",
    "LOW_REACH_AVOIDANCE": "몸 굽힘·쪼그림 회피 필요",
    "HIGH_REACH_AVOIDANCE": "높은 곳에 손 뻗기 제한",
    "ACCESSIBILITY_SUPPORT": "ADL·IADL 도움 필요",
    "ORIENTATION_SUPPORT": "인지·경로 확인 필요",
    "NIGHTTIME_SAFETY": "야간 이동 확인 필요",
    "TOILET_ACCESS": "화장실 접근 확인 필요",
    "FALL_IMPACT_REDUCTION": "골절 충격 위험",
    "HIGH_MODEL_RISK": "모델 고위험 판정",
}

# 질문은 공간별로 분리하고, 각 질문에 연결된 개인 위험 신호가 있을 때만 노출합니다.
# priority가 작을수록 먼저 표시됩니다.
QUESTION_BANK: dict[str, list[dict[str, Any]]] = {
    "living_room": [
        {"hazard_code": "PATH_OBSTACLE", "label": "주 이동동선에 장애물이 있음", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT", "ORIENTATION_SUPPORT", "VISUAL_GUIDANCE", "HIGH_MODEL_RISK"}, "priority": 1},
        {"hazard_code": "LOOSE_RUG", "label": "러그·매트가 고정되지 않음", "signals": {"WALKING_SUPPORT", "VISUAL_GUIDANCE", "FALL_IMPACT_REDUCTION", "HIGH_MODEL_RISK"}, "priority": 2},
        {"hazard_code": "LOW_SEAT_HEIGHT", "label": "의자·소파 좌면이 낮음", "signals": {"LOWER_LIMB_SUPPORT", "ACCESSIBILITY_SUPPORT"}, "priority": 3},
        {"hazard_code": "FLOOR_LEVEL_OBJECT", "label": "바닥 가까이에 물건이 있음", "signals": {"LOW_REACH_AVOIDANCE", "WALKING_SUPPORT", "VISUAL_GUIDANCE"}, "priority": 4},
        {"hazard_code": "NARROW_PATH", "label": "가구 사이 통로가 좁음", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT"}, "priority": 5},
        {"hazard_code": "LOW_LIGHTING", "label": "조명이 어두움", "signals": {"VISUAL_GUIDANCE", "ORIENTATION_SUPPORT", "HIGH_MODEL_RISK"}, "priority": 6},
        {"hazard_code": "NO_NIGHT_LIGHT", "label": "야간 이동을 위한 조명이 없음", "signals": {"NIGHTTIME_SAFETY", "VISUAL_GUIDANCE"}, "priority": 7},
        {"hazard_code": "LOW_STORAGE", "label": "자주 쓰는 수납 위치가 무릎 아래에 있음", "signals": {"LOW_REACH_AVOIDANCE"}, "priority": 8},
        {"hazard_code": "HARD_SHARP_FURNITURE_EDGE", "label": "주 동선 주변 가구 모서리가 날카로움", "signals": {"FALL_IMPACT_REDUCTION", "WALKING_SUPPORT"}, "priority": 9},
        {"hazard_code": "THRESHOLD_HIGH", "label": "출입부 문턱·단차가 높음", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT"}, "priority": 10},
        {"hazard_code": "THRESHOLD_LOW_CONTRAST", "label": "문턱·단차의 색 대비가 낮음", "signals": {"VISUAL_GUIDANCE"}, "priority": 11},
        {"hazard_code": "UNCLEAR_ROUTE", "label": "주 이동 경로가 불명확함", "signals": {"ORIENTATION_SUPPORT"}, "priority": 12},
    ],
    "bedroom": [
        {"hazard_code": "NO_NIGHT_LIGHT", "label": "침대에서 출입문까지 야간 조명이 없음", "signals": {"NIGHTTIME_SAFETY", "TOILET_ACCESS", "VISUAL_GUIDANCE"}, "priority": 1},
        {"hazard_code": "PATH_OBSTACLE", "label": "침대에서 출입문까지 이동동선에 장애물이 있음", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT", "TOILET_ACCESS", "VISUAL_GUIDANCE", "HIGH_MODEL_RISK"}, "priority": 2},
        {"hazard_code": "LOW_BED_HEIGHT", "label": "침대가 낮아 앉고 일어나기 어려움", "signals": {"LOWER_LIMB_SUPPORT", "ACCESSIBILITY_SUPPORT"}, "priority": 3},
        {"hazard_code": "LOOSE_RUG", "label": "침대 주변 러그·매트가 고정되지 않음", "signals": {"WALKING_SUPPORT", "VISUAL_GUIDANCE", "FALL_IMPACT_REDUCTION", "HIGH_MODEL_RISK"}, "priority": 4},
        {"hazard_code": "FLOOR_LEVEL_OBJECT", "label": "침대 주변 바닥 가까이에 물건이 있음", "signals": {"LOW_REACH_AVOIDANCE", "WALKING_SUPPORT", "VISUAL_GUIDANCE"}, "priority": 5},
        {"hazard_code": "NARROW_PATH", "label": "침대 주변 통로가 좁음", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT"}, "priority": 6},
        {"hazard_code": "LOW_LIGHTING", "label": "침실 조명이 어두움", "signals": {"VISUAL_GUIDANCE", "ORIENTATION_SUPPORT", "HIGH_MODEL_RISK"}, "priority": 7},
        {"hazard_code": "LOW_STORAGE", "label": "자주 쓰는 옷·물건이 무릎 아래 수납칸에 있음", "signals": {"LOW_REACH_AVOIDANCE"}, "priority": 8},
        {"hazard_code": "HIGH_STORAGE", "label": "자주 쓰는 옷·물건이 어깨 위 수납칸에 있음", "signals": {"HIGH_REACH_AVOIDANCE"}, "priority": 9},
        {"hazard_code": "HARD_SHARP_FURNITURE_EDGE", "label": "침대 주변 가구 모서리가 날카로움", "signals": {"FALL_IMPACT_REDUCTION", "WALKING_SUPPORT"}, "priority": 10},
        {"hazard_code": "THRESHOLD_HIGH", "label": "침실 출입부 문턱·단차가 높음", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT"}, "priority": 11},
        {"hazard_code": "THRESHOLD_LOW_CONTRAST", "label": "침실 출입부 문턱·단차의 색 대비가 낮음", "signals": {"VISUAL_GUIDANCE"}, "priority": 12},
        {"hazard_code": "UNCLEAR_ROUTE", "label": "침대에서 출입문까지 경로가 불명확함", "signals": {"ORIENTATION_SUPPORT"}, "priority": 13},
    ],
    "kitchen": [
        {"hazard_code": "SLIPPERY_FLOOR", "label": "싱크대·조리대 주변 바닥이 자주 젖거나 미끄러움", "signals": {"LOWER_LIMB_SUPPORT", "WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT", "VISUAL_GUIDANCE", "FALL_IMPACT_REDUCTION", "HIGH_MODEL_RISK"}, "priority": 1},
        {"hazard_code": "LOW_STORAGE", "label": "자주 쓰는 조리도구·식재료가 무릎 아래에 있음", "signals": {"LOW_REACH_AVOIDANCE", "ACCESSIBILITY_SUPPORT"}, "priority": 2},
        {"hazard_code": "HIGH_STORAGE", "label": "자주 쓰는 조리도구·식재료가 어깨 위에 있음", "signals": {"HIGH_REACH_AVOIDANCE", "ACCESSIBILITY_SUPPORT"}, "priority": 3},
        {"hazard_code": "PATH_OBSTACLE", "label": "냉장고·싱크대·조리대 사이 동선에 장애물이 있음", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT", "ORIENTATION_SUPPORT", "VISUAL_GUIDANCE", "HIGH_MODEL_RISK"}, "priority": 4},
        {"hazard_code": "NARROW_PATH", "label": "조리 공간의 이동 통로가 좁음", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT"}, "priority": 5},
        {"hazard_code": "FLOOR_LEVEL_OBJECT", "label": "바닥 가까이에 생수·냄비·식재료 등이 놓여 있음", "signals": {"LOW_REACH_AVOIDANCE", "WALKING_SUPPORT", "VISUAL_GUIDANCE"}, "priority": 6},
        {"hazard_code": "LOW_LIGHTING", "label": "조리대·싱크대 주변 조명이 어두움", "signals": {"VISUAL_GUIDANCE", "ORIENTATION_SUPPORT", "HIGH_MODEL_RISK"}, "priority": 7},
        {"hazard_code": "LOOSE_RUG", "label": "주방 매트가 고정되지 않음", "signals": {"WALKING_SUPPORT", "VISUAL_GUIDANCE", "FALL_IMPACT_REDUCTION"}, "priority": 8},
        {"hazard_code": "HARD_SHARP_FURNITURE_EDGE", "label": "주 동선 주변 가구·조리대 모서리가 날카로움", "signals": {"FALL_IMPACT_REDUCTION", "WALKING_SUPPORT"}, "priority": 9},
        {"hazard_code": "THRESHOLD_HIGH", "label": "주방 출입부 문턱·단차가 높음", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT"}, "priority": 10},
        {"hazard_code": "THRESHOLD_LOW_CONTRAST", "label": "주방 출입부 문턱·단차의 색 대비가 낮음", "signals": {"VISUAL_GUIDANCE"}, "priority": 11},
        {"hazard_code": "UNCLEAR_ROUTE", "label": "주방 내 이동 경로가 불명확함", "signals": {"ORIENTATION_SUPPORT"}, "priority": 12},
    ],
    "bathroom": [
        {"hazard_code": "SLIPPERY_FLOOR", "label": "세면·샤워 후 바닥이 젖거나 미끄러움", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT", "LOWER_LIMB_SUPPORT", "VISUAL_GUIDANCE", "FALL_IMPACT_REDUCTION", "HIGH_MODEL_RISK"}, "priority": 1},
        {"hazard_code": "TOILET_NO_GRAB_BAR", "label": "변기 또는 샤워 구역 주변에 잡을 손잡이가 없음", "signals": {"LOWER_LIMB_SUPPORT", "ACCESSIBILITY_SUPPORT", "TOILET_ACCESS"}, "priority": 2},
        {"hazard_code": "LOW_TOILET", "label": "변기 좌면이 낮아 앉고 일어나기 어려움", "signals": {"LOWER_LIMB_SUPPORT", "LOW_REACH_AVOIDANCE", "ACCESSIBILITY_SUPPORT", "TOILET_ACCESS"}, "priority": 3},
        {"hazard_code": "NO_NIGHT_LIGHT", "label": "침실에서 화장실까지 야간 조명이 없음", "signals": {"NIGHTTIME_SAFETY", "TOILET_ACCESS", "VISUAL_GUIDANCE"}, "priority": 4},
        {"hazard_code": "PATH_OBSTACLE", "label": "화장실 출입·변기·세면대 사이 동선에 장애물이 있음", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT", "TOILET_ACCESS", "VISUAL_GUIDANCE", "HIGH_MODEL_RISK"}, "priority": 5},
        {"hazard_code": "LOOSE_RUG", "label": "욕실 매트가 고정되지 않음", "signals": {"WALKING_SUPPORT", "VISUAL_GUIDANCE", "FALL_IMPACT_REDUCTION"}, "priority": 6},
        {"hazard_code": "THRESHOLD_HIGH", "label": "화장실 출입부 문턱·단차가 높음", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT", "TOILET_ACCESS"}, "priority": 7},
        {"hazard_code": "THRESHOLD_LOW_CONTRAST", "label": "화장실 출입부 문턱·단차의 색 대비가 낮음", "signals": {"VISUAL_GUIDANCE"}, "priority": 8},
        {"hazard_code": "NARROW_PATH", "label": "변기·세면대 주변 이동 공간이 좁음", "signals": {"WALKING_SUPPORT", "ACCESSIBILITY_SUPPORT", "TOILET_ACCESS"}, "priority": 9},
        {"hazard_code": "LOW_LIGHTING", "label": "화장실 조명이 어두움", "signals": {"VISUAL_GUIDANCE", "ORIENTATION_SUPPORT", "HIGH_MODEL_RISK"}, "priority": 10},
        {"hazard_code": "FLOOR_LEVEL_OBJECT", "label": "화장실 바닥 가까이에 세면용품·물건이 놓여 있음", "signals": {"LOW_REACH_AVOIDANCE", "WALKING_SUPPORT", "VISUAL_GUIDANCE"}, "priority": 11},
        {"hazard_code": "UNCLEAR_ROUTE", "label": "화장실 출입 경로가 불명확함", "signals": {"ORIENTATION_SUPPORT"}, "priority": 12},
    ],
}


def derive_personal_space_signals(
    model_input: dict[str, int | float] | None,
    prediction: dict[str, Any] | None = None,
) -> set[str]:
    """45개 설문값에서 공간 검증에 직접 필요한 기능 신호만 추출합니다."""
    payload = model_input or {}
    signals: set[str] = set()

    def truthy(key: str) -> bool:
        try:
            return int(payload.get(key, 0)) == 1
        except (TypeError, ValueError):
            return False

    def number(key: str, default: float = 0.0) -> float:
        try:
            return float(payload.get(key, default))
        except (TypeError, ValueError):
            return default

    if number("visual_difficulty", 1) >= 2:
        signals.add("VISUAL_GUIDANCE")
    if number("hearing_difficulty", 1) >= 2:
        signals.add("ALERT_SUPPORT")

    if any(
        truthy(key)
        for key in (
            "lower_limb_strength_limitation",
            "diagnosis_arthritis",
            "diagnosis_fracture_sequelae",
            "diagnosis_parkinson",
        )
    ):
        signals.add("LOWER_LIMB_SUPPORT")

    if any(
        truthy(key)
        for key in (
            "walk_400m_limitation",
            "climb_10_stairs_limitation",
            "diagnosis_stroke",
            "diagnosis_parkinson",
        )
    ):
        signals.add("WALKING_SUPPORT")

    if any(
        truthy(key)
        for key in (
            "kneel_squat_limitation",
            "diagnosis_arthritis",
            "diagnosis_lumbar_sciatic_pain",
        )
    ):
        signals.add("LOW_REACH_AVOIDANCE")

    if truthy("reach_overhead_limitation") or truthy("lift_8kg_limitation"):
        signals.add("HIGH_REACH_AVOIDANCE")

    if truthy("adl_help_needed") or truthy("iadl_help_needed"):
        signals.update({"ACCESSIBILITY_SUPPORT", "WALKING_SUPPORT"})

    if number("cognitive_score_mmse", 30) <= 24:
        signals.add("ORIENTATION_SUPPORT")

    if any(
        truthy(key)
        for key in (
            "diagnosis_insomnia",
            "treatment_insomnia",
            "diagnosis_urinary_incontinence",
            "treatment_bph",
        )
    ):
        signals.add("NIGHTTIME_SAFETY")

    if any(
        truthy(key)
        for key in (
            "diagnosis_urinary_incontinence",
            "treatment_bph",
            "adl_help_needed",
            "walk_400m_limitation",
        )
    ):
        signals.add("TOILET_ACCESS")

    if truthy("diagnosis_osteoporosis") or truthy("diagnosis_fracture_sequelae"):
        signals.add("FALL_IMPACT_REDUCTION")

    if prediction and str(prediction.get("model_risk_label")) == "HIGH_RISK":
        signals.add("HIGH_MODEL_RISK")

    return signals


def select_adaptive_space_questions(
    model_input: dict[str, int | float] | None,
    room_name: str,
    *,
    prediction: dict[str, Any] | None = None,
    max_questions: int = 8,
) -> list[dict[str, Any]]:
    """사용자 기능위험과 선택 공간에 관련된 문항만 반환합니다."""
    signals = derive_personal_space_signals(model_input, prediction)
    questions: list[dict[str, Any]] = []

    for item in QUESTION_BANK.get(room_name, []):
        required = set(item.get("signals", set()))
        matched = sorted(required & signals)
        if not matched:
            continue
        questions.append(
            {
                "hazard_code": str(item["hazard_code"]),
                "label": str(item["label"]),
                "room_name": room_name,
                "room_label": ROOM_LABELS.get(room_name, room_name),
                "priority": int(item.get("priority", 999)),
                "triggered_by": matched,
                "trigger_labels": [SIGNAL_LABELS.get(code, code) for code in matched],
                "selection_policy": "ROOM_AND_PERSONAL_RISK_FILTER",
            }
        )

    questions.sort(
        key=lambda item: (
            item["priority"],
            -len(item["triggered_by"]),
            item["hazard_code"],
        )
    )
    return questions[: max(0, int(max_questions))]

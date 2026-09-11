"""
web_helper.py
=============
utils/helper.py 중 Streamlit에 의존하지 않는 로직만 그대로 옮긴 모듈입니다.
(색상 상수, 공간 메타데이터, 마커 좌표 계산, 평면도 HTML 렌더링 등)

predict_fall_risk / run_full_analysis 같은 실제 백엔드 계산은 그대로
integration.frontend_backend_adapter / src.integration_pipeline 을 사용합니다.
"""

from __future__ import annotations

import html as html_lib
import math
import re

# ----------------------------------------------------------------------------
# 브랜드 컬러 (기존 utils/helper.py 와 동일)
# ----------------------------------------------------------------------------
BG = "#EEF2ED"
PAPER = "#F8FAF7"
INK = "#1E2B26"
INK_SOFT = "#4B5C54"
LINE = "#C7D3CB"
LINE_STRONG = "#8FA399"
TEAL = "#2F6F62"
CORAL = "#C1543C"
AMBER = "#D98E3B"
SAGE = "#6E9C86"
CORAL_SOFT = "#F1DAD3"
AMBER_SOFT = "#F5E5CC"
SAGE_SOFT = "#DDE9E1"

SUCCESS_GREEN = SAGE
WARNING_AMBER = AMBER
DANGER_RED = CORAL

RISK_COLOR_MAP = {
    "저위험": SUCCESS_GREEN,
    "중위험": WARNING_AMBER,
    "고위험": DANGER_RED,
    "모델 판정 기준 미만 - 관찰 권장": SUCCESS_GREEN,
    "모델 판정 기준 초과 - 주의 및 점검 권장": DANGER_RED,
}


def get_risk_level(score: float) -> str:
    if score < 40:
        return "저위험"
    elif score < 70:
        return "중위험"
    return "고위험"


# ----------------------------------------------------------------------------
# 페이지 진행 단계
# ----------------------------------------------------------------------------
PAGE_ORDER = ["home", "input", "result", "floorplan", "room_detail"]
PAGE_LABELS = {
    "home": "홈",
    "input": "정보 입력",
    "result": "예측 결과",
    "floorplan": "공간 추천",
    "room_detail": "상세 추천 · AI 분석",
}


# ----------------------------------------------------------------------------
# 공간 및 추천 인테리어 샘플 데이터
# ----------------------------------------------------------------------------
ROOM_INFO = {
    "living_room": {
        "label": "거실",
        "icon": "🛋️",
        "desc": "가족이 가장 오래 머무는 공간으로, 가구 배치와 바닥 상태 점검이 중요합니다.",
        "priority": "medium",
        "priority_label": "점검하면 좋음",
    },
    "bedroom": {
        "label": "침실",
        "icon": "🛏️",
        "desc": "야간 기상 시 낙상 위험이 높은 공간으로, 조명과 침대 높이가 핵심입니다.",
        "priority": "habit",
        "priority_label": "습관으로 관리",
    },
    "bathroom": {
        "label": "화장실",
        "icon": "🚿",
        "desc": "물기로 인한 미끄러짐 사고가 가장 빈번한 고위험 공간입니다.",
        "priority": "high",
        "priority_label": "우선 조치 필요",
    },
    "kitchen": {
        "label": "주방",
        "icon": "🍳",
        "desc": "기름/물기와 조리 기구로 인해 주의가 필요한 공간입니다.",
        "priority": "medium",
        "priority_label": "점검하면 좋음",
    },
}

RECOMMENDATION_DATA = {
    "living_room": [
        {"icon": "🪑", "title": "가구 동선 확보", "desc": "이동 경로에 놓인 낮은 탁자를 정리해 걸려 넘어질 위험을 줄입니다."},
        {"icon": "🧵", "title": "카펫 제거 또는 고정", "desc": "미끄러지거나 발이 걸리기 쉬운 러그를 제거하거나 미끄럼 방지 패드로 고정합니다."},
        {"icon": "💡", "title": "조명 밝기 향상", "desc": "그림자 지는 구역에 보조 조명을 추가해 야간 시야를 확보합니다."},
    ],
    "bedroom": [
        {"icon": "🛏️", "title": "침대 높이 조절", "desc": "무릎 높이에 맞춰 침대 높이를 조절해 일어날 때의 부담을 줄입니다."},
        {"icon": "💡", "title": "취침등 설치", "desc": "야간 화장실 이동 동선에 센서등을 설치합니다."},
        {"icon": "🧦", "title": "미끄럼 방지 슬리퍼 비치", "desc": "침대 옆에 미끄럼 방지 처리된 실내화를 비치합니다."},
    ],
    "bathroom": [
        {"icon": "🧴", "title": "미끄럼 방지 바닥재 설치", "desc": "물기가 많은 바닥에 미끄럼 방지 매트나 타일을 설치합니다."},
        {"icon": "🖐️", "title": "안전 손잡이 설치", "desc": "변기와 샤워 부스 주변에 손잡이를 설치해 이동을 보조합니다."},
        {"icon": "🪑", "title": "샤워 의자 비치", "desc": "장시간 서 있는 부담을 줄이는 샤워용 의자를 비치합니다."},
    ],
    "kitchen": [
        {"icon": "🧯", "title": "미끄럼 방지 매트 설치", "desc": "싱크대 앞 조리 구역에 미끄럼 방지 매트를 설치합니다."},
        {"icon": "📦", "title": "자주 쓰는 물건 낮은 위치 배치", "desc": "높은 곳에 올라가지 않도록 자주 쓰는 물건을 손 닿는 위치로 재배치합니다."},
        {"icon": "💡", "title": "조리대 조명 보강", "desc": "조리 구역의 그림자를 줄이는 직하 조명을 추가합니다."},
    ],
}


def build_main_factor_cards(shap_result: dict, top_n: int = 12) -> list:
    """SHAP main_effects(개별 피처 단위)를 결과 화면 카드/레이더 차트용으로 가공합니다.
    result_visualizer.py에 이미 있는 피처명 -> 자연스러운 한글 라벨 매핑을 그대로 재사용해서,
    "동작수행 어려움_쉬지않고 10계단 오르기" 같은 원시 컬럼명 대신 "계단 오르기 어려움"처럼 보여줍니다."""
    from result_visualizer import _pretty_feature_name

    cards = []
    for item in (shap_result.get("main_effects") or [])[:top_n]:
        if not isinstance(item, dict):
            continue
        cards.append({
            "label": _pretty_feature_name(str(item.get("feature", ""))),
            "shap_value": float(item.get("shap_value", 0.0)),
        })
    return cards


def build_interaction_cards(shap_result: dict, top_n: int = 12) -> list:
    """SHAP interaction(두 피처 간 상호작용) 상위 top_n개를 카드용으로 가공합니다.
    45개 변수 기준 최대 990개 조합이 계산되므로, 3개보다 훨씬 더 많이 보여줄 데이터는 항상 충분합니다."""
    from result_visualizer import _pretty_feature_name

    cards = []
    for item in (shap_result.get("interactions") or [])[:top_n]:
        if not isinstance(item, dict):
            continue
        cards.append({
            "label_1": _pretty_feature_name(str(item.get("feature_1", ""))),
            "label_2": _pretty_feature_name(str(item.get("feature_2", ""))),
            "interaction_value": float(item.get("interaction_value", 0.0)),
        })
    return cards


def build_comparison_legend(hazard_groups: list) -> list:
    """hazard_groups(위험요인별 대표 해결책 + 대안 목록 + 현재 선택 인덱스)로부터
    화면에 보여줄 범례를 만듭니다. 현재 선택된 해결책은 improvement로,
    나머지는 alternatives로 분리합니다.

    선택된 대안이 가구 높이를 바꾸는 것이면 height_note를 같이 붙이는데, 문구가 두 가지로
    갈립니다:
    - 대안 문구 자체(문헌 근거)에 이미 구체적인 높이 숫자가 있었던 경우
      (height_source == "literature_text") → "문헌에 명시된 목표 높이" 라고 표시.
      이 경우엔 AI가 새로 추정한 게 아니라 원래 문헌 문구에 있던 숫자를 그대로 옮긴
      것이므로 "AI 추정치"라고 붙이지 않습니다.
    - 문구에 숫자가 없어서 GPT가 새로 어림잡은 경우(height_source == "ai_estimate")만
      "AI 추정치, 문헌 근거 아님"이라고 명확히 구분해서 표시합니다."""
    legend = []
    for group in hazard_groups:
        options = group.get("options", [])
        selected = group.get("selected", 0)
        if not (0 <= selected < len(options)):
            selected = 0

        height_note = ""
        action_plans = group.get("action_plans") or []
        if 0 <= selected < len(action_plans):
            plan = action_plans[selected]
            if plan.get("affects_furniture_height") and plan.get("target_height_cm"):
                target = int(plan["target_height_cm"])
                if plan.get("height_source") == "literature_text":
                    height_note = f"목표 높이 약 {target}cm (문헌에 명시된 값)"
                else:
                    height_note = f"약 {target}cm로 (AI 추정치, 문헌 근거 아님)"

        selected_improvement = options[selected] if options else ""
        legend.append({
            "priority": group.get("priority"),
            "display_number": group.get("display_number", group.get("priority")),
            "improvement": selected_improvement,
            "floorplan_problem": group.get("floorplan_problem", ""),
            "target_label": _related_category_label(
                selected_improvement,
                str(group.get("target_object") or "room"),
                [],
            ),
            "alternatives": [opt for i, opt in enumerate(options) if i != selected],
            "height_note": height_note,
        })
    legend.sort(key=lambda item: (item.get("display_number") is None, item.get("display_number") if isinstance(item.get("display_number"), (int, float)) else 0))
    return legend



_GARBLED_TILDE_RE = re.compile(r"(\d)\?(\d+\s*(?:cm|mm|m|kg|세|개|회))")


def _humanize_internal_id_token(token: str) -> str:
    raw = str(token or "").strip()
    low = raw.lower()
    room_match = re.search(r"(?:room|path)_((?:living_room)|bedroom|kitchen|bathroom)(?:_|$)", low)
    if room_match:
        room_key = room_match.group(1)
        room_label = _ROOM_INTERNAL_LABELS.get(room_key, "공간")
        if low.startswith("path_"):
            return f"{room_label} 주요 이동 동선"
        return f"{room_label} 전체"
    if low.startswith("path_"):
        return "주요 이동 동선"
    if low.startswith(("room_", "layout_object_", "object_", "hz_")):
        return "공간 항목"
    return raw

_INTERNAL_ID_RE = re.compile(
    r"\b(?:room|path)_(?:living_room|bedroom|kitchen|bathroom)(?:_[A-Za-z0-9]+)*\b|"
    r"\b(?:layout_object|object|hz)_[A-Za-z0-9_]+\b",
    re.IGNORECASE,
)

def clean_display_text(text: str) -> str:
    """화면/PDF용 텍스트 정리.

    1) CSV의 깨진 범위표기(45?50cm -> 45~50cm)를 복구하고,
    2) room_living_room_01, path_living_room_main 같은 내부 ID가 사용자 화면에
       노출되지 않도록 한글 라벨로 치환합니다.
    """
    if not text:
        return text
    cleaned = _GARBLED_TILDE_RE.sub(r"\1~\2", str(text))
    return _INTERNAL_ID_RE.sub(lambda m: _humanize_internal_id_token(m.group(0)), cleaned)


_SHAP_FACTOR_CODE_SUFFIX_RE = re.compile(r"\s*/\s*[A-Z][A-Z0-9_]{2,}\s*$")


def humanize_shap_factor(raw: str) -> str:
    """SHAP 개인 위험요인 원본 문자열(예: "근력상태_의자나 침대에 앉았다가
    일어나기 5회 반복 / SIT_TO_STAND_SUPPORT")을 화면에 보여줄 수 있는 자연스러운
    한글 문구("근력상태: 의자나 침대에 앉았다가 일어나기 5회 반복")로 정리합니다.
    내부 feature 코드(대문자 스네이크케이스 접미사)는 사용자에게 의미가 없어서 뗍니다."""
    text = clean_display_text(str(raw or "").strip())
    if not text:
        return ""
    text = _SHAP_FACTOR_CODE_SUFFIX_RE.sub("", text).strip()
    if "_" in text:
        head, _, rest = text.partition("_")
        head = head.strip()
        rest = rest.strip()
        if head and rest:
            text = f"{head}: {rest}"
    return text


def dedupe_recommendations_by_hazard(recommendations: list) -> list:
    """추천(recommendation) 목록에는 같은 위험요인(hazard_code)에 대한 대안 해결책이
    여러 개 들어있을 수 있습니다(예: "변기가 낮음" 하나에 대해 "높임변기 설치" /
    "손잡이 설치" / "샤워의자 설치"가 각각 우선순위 1·2·3위로 나열되는 식).

    이미지에는 위험요인당 대표 해결책 1개만 그릴 수 있어서 번호는 위험요인 단위로
    매기지만, 나머지 대안들을 그냥 버리지는 않습니다 — 각 대표 항목에
    "alternative_improvements"로 같이 담아서, 화면에서 "이 방법 말고 이런 대안도
    있다"는 걸 사용자가 직접 비교하고 고를 수 있게 합니다.
    (list는 이미 priority 오름차순으로 정렬돼 들어온다고 가정하고, 각 hazard_code의
    첫 등장 = 가장 우선순위 높은 항목을 대표로 삼습니다.)
    """
    groups: dict[str, dict] = {}
    order: list[str] = []
    for rec in recommendations:
        if not isinstance(rec, dict):
            continue
        raw_problem = str(rec.get("floorplan_problem", ""))
        hazard_code = raw_problem.split(" / ", 1)[0].strip() or rec.get("recommendation_id", "")
        if hazard_code not in groups:
            groups[hazard_code] = {"primary": rec, "alternatives": []}
            order.append(hazard_code)
        else:
            groups[hazard_code]["alternatives"].append(rec)

    # 중복 제거로 번호가 1, 4처럼 건너뛰지 않도록 화면/이미지 표시용 우선순위를 다시 매깁니다.
    renumbered = []
    for idx, hazard_code in enumerate(order, start=1):
        group = groups[hazard_code]
        primary = dict(group["primary"])
        primary["priority"] = idx
        primary["alternative_improvements"] = [
            text
            for alt in group["alternatives"]
            if (text := str(alt.get("improvement", "")).strip())
        ]
        renumbered.append(primary)
    return renumbered


def describe_floorplan_problem(room_name: str, raw_floorplan_problem: str) -> str:
    """추천(recommendation)의 floorplan_problem 필드는 원래
    "HAZARD_CODE / ROOM_TYPE / object_id" 형태의 내부 디버그용 원시 문자열이라
    화면에 그대로 보여주면 사용자가 이해할 수 없습니다.
    이미 방별 hazard 문구 매핑(room_priority.HAZARD_LABELS_BY_ROOM)이 있으므로
    hazard_code만 뽑아 자연스러운 한글 문구로 바꿔 반환합니다.
    매핑이 없는 코드라면(신규/미등록 hazard) 원본 문자열을 그대로 돌려줍니다."""
    raw = str(raw_floorplan_problem or "").strip()
    if not raw:
        return raw

    hazard_code = raw.split(" / ", 1)[0].strip()
    if not hazard_code:
        return raw

    try:
        from src.room_priority import _hazard_label
    except Exception:
        return raw

    label = _hazard_label(room_name, hazard_code)
    if label != hazard_code:
        return label

    # 신규 hazard가 아직 한글 매핑에 등록되지 않았더라도 내부 room/object ID를
    # 사용자 화면에 그대로 노출하지 않습니다.
    room_label = ROOM_INFO.get(room_name, {}).get("label") or _humanize_target_object(room_name)
    return f"{room_label}에서 확인된 공간 위험" if room_label else "확인된 공간 위험"


def estimate_risk_factors(user_input: dict) -> list:
    """SHAP 유사 값을 이용한 개인별 위험요인 우선순위 추정(간이 버전)."""
    contributions: dict[str, float] = {}

    age = user_input.get("age", 65)
    if age >= 80:
        contributions["고령"] = 0.25
    elif age >= 70:
        contributions["고령"] = 0.15

    if user_input.get("vision_trouble") == "예":
        contributions["시력저하"] = 0.22

    adl_score = user_input.get("adl_score", 100)
    if adl_score < 60:
        contributions["ADL 저하"] = 0.27
    elif adl_score < 85:
        contributions["ADL 저하"] = 0.15

    depression_score = user_input.get("depression_score", 0)
    if depression_score >= 15:
        contributions["우울"] = 0.20
    elif depression_score >= 8:
        contributions["우울"] = 0.10

    if user_input.get("fall_history") == "있음":
        contributions["낙상 경험"] = 0.28

    if user_input.get("exercise") == "안 함":
        contributions["운동 부족"] = 0.18
    elif user_input.get("exercise") == "가끔":
        contributions["운동 부족"] = 0.08

    if user_input.get("living_alone") == "예":
        contributions["독거"] = 0.12

    if not contributions:
        contributions = {"운동 부족": 0.1, "고령": 0.1}

    sorted_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    return [{"factor": f, "shap_importance": round(v, 2)} for f, v in sorted_factors]




_ROOM_INTERNAL_LABELS = {
    "living_room": "거실",
    "bedroom": "침실",
    "kitchen": "주방",
    "bathroom": "욕실",
}

_RELATED_OBJECT_ALIASES = [
    ("변기", ("변기", "양변기", "toilet")),
    ("샤워시설", ("샤워시설", "샤워 시설", "샤워공간", "샤워 공간", "샤워기", "샤워")),
    ("욕조", ("욕조", "bathtub")),
    ("세면대", ("세면대", "washbasin", "sink basin")),
    ("침대", ("침대", "bed")),
    ("의자", ("의자", "chair")),
    ("소파", ("소파", "sofa")),
    ("러그", ("러그", "카펫", "매트", "rug", "carpet")),
    ("싱크대", ("싱크대", "sink")),
    ("조리대", ("조리대", "counter")),
    ("식탁", ("식탁", "dining table")),
    ("수납장", ("수납장", "서랍장", "옷장", "cabinet", "storage")),
    ("출입구", ("출입구", "문", "doorway", "door")),
]

def _humanize_target_object(target_object: str) -> str:
    """내부 object/room ID를 사용자용 한글 라벨로 바꿉니다."""
    raw = str(target_object or "").strip()
    if not raw or raw.lower() == "room":
        return "방 전체"
    low = raw.lower()
    for room_key, label in _ROOM_INTERNAL_LABELS.items():
        if room_key in low and low.startswith("room_"):
            return f"{label} 전체"
        if room_key in low and low.startswith("path_"):
            return f"{label} 주요 이동 동선"
    if low.startswith("path_"):
        return "주요 이동 동선"
    if low.startswith(("room_", "layout_object_", "object_", "hz_")):
        return "방 전체"
    return clean_display_text(raw)

def _related_category_label(label: str, target_object: str, layout: list) -> str:
    """개선 문구가 여러 설비를 동시에 지칭하면 모두 표시합니다.

    예: '변기 및 샤워시설 옆에 안전바 설치' -> '변기 · 샤워시설'
    내부 room_living_room_01 같은 ID는 절대 화면에 노출하지 않습니다.
    """
    source = str(label or "").strip()
    source_low = source.lower()
    found: list[str] = []

    # 실제 배치 가구명이 개선 문구에 직접 등장하면 우선 수집합니다.
    for obj in layout or []:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name") or "").strip()
        if name and name.lower() in source_low and name not in found:
            found.append(name)

    # '샤워시설' vs '샤워공간'처럼 표현이 달라도 같은 대상을 인식합니다.
    for display, aliases in _RELATED_OBJECT_ALIASES:
        if any(alias.lower() in source_low for alias in aliases):
            # 샤워공간/샤워시설처럼 실제 가구명과 의미상 중복이면 대표 라벨로 통일
            found = [item for item in found if not (display == "샤워시설" and "샤워" in item)]
            if display not in found:
                found.append(display)

    if found:
        return " · ".join(found[:4])
    return _humanize_target_object(target_object)


def _target_alias_candidates(target_object: str, label: str = "") -> list[str]:
    raw = clean_display_text(str(target_object or "")).strip().lower()
    label_low = clean_display_text(str(label or "")).strip().lower()
    combined = f"{raw} {label_low}"
    candidates: list[str] = []

    def add(*values: str):
        for value in values:
            val = str(value or "").strip().lower()
            if val and val not in candidates:
                candidates.append(val)

    add(raw)
    if any(key in combined for key in ("문", "출입구", "door")):
        add("문", "출입구", "door")
    if any(key in combined for key in ("창문", "window")):
        add("창문", "window")
    if any(key in combined for key in ("변기", "양변기", "toilet")):
        add("변기", "양변기", "toilet")
    if any(key in combined for key in ("세면대", "washbasin", "sink basin")):
        add("세면대", "washbasin")
    if any(key in combined for key in ("욕조", "bathtub")):
        add("욕조", "bathtub")
    if any(key in combined for key in ("샤워", "shower")):
        add("샤워", "샤워시설", "샤워공간", "shower")
    return candidates


def _find_layout_target(layout: list, target_object: str, label: str = ""):
    if not target_object or str(target_object).strip().lower() == "room":
        target_object = ""
    candidates = _target_alias_candidates(target_object, label)
    if not candidates:
        return None
    for cand in candidates:
        for obj in layout:
            if not isinstance(obj, dict):
                continue
            obj_name = clean_display_text(str(obj.get("name", ""))).strip().lower()
            if obj_name and (cand in obj_name or obj_name in cand):
                return obj
    return None


def calculate_marker_positions(
    layout: list,
    markers: list,
    room_width: int = 720,
    room_height: int = 460,
) -> list:
    """개선 마커를 실제 평면도 좌표로 변환합니다.

    같은 가구에 마커가 여러 개 몰리는 경우(예: 수납장 관련 대안이 3~4개), 예전에는
    고정된 8방향만 돌려썼기 때문에 개수가 많아지면 서로 겹쳤습니다. 이제는 8개를 다 쓰면
    반지름을 키워 바깥쪽으로 한 바퀴 더 도는 방식(spiral)으로 확장해서, 마커가 몇 개든
    서로 겹치지 않게 계속 퍼집니다.

    각 마커에는 어떤 가구에 붙은 것인지 사람이 읽을 수 있는 "category" 라벨도 같이
    반환합니다("수납장", "방 전체" 등) — 화면에서 "[수납장 관련]"처럼 접두어로 써서,
    번호가 많아져도 어떤 위험요인끼리 묶이는지 헷갈리지 않게 하기 위함입니다.
    """
    positioned = []
    room_corners = [
        {"x": room_width - 50, "y": 20},
        {"x": 20, "y": 20},
        {"x": room_width - 50, "y": room_height - 50},
        {"x": 20, "y": room_height - 50},
        {"x": room_width // 2, "y": 20},
    ]
    corner_idx = 0
    target_marker_counts: dict[str, int] = {}

    def find_target(target_name, marker_label=""):
        return _find_layout_target(layout, target_name, marker_label)

    def clamp_position(x, y):
        x = max(4, min(x, room_width - 40))
        y = max(4, min(y, room_height - 40))
        return x, y

    def spiral_offset(marker_index, obj_width, obj_height):
        """8방향을 다 쓰면 반지름을 키워 바깥으로 확장하는 나선형 오프셋.
        몇 개가 몰리든 서로 겹치지 않고 계속 퍼집니다."""
        base_radius = max(obj_width, obj_height) / 2 + 20
        ring = marker_index // 8
        angle_deg = (marker_index % 8) * 45
        radius = base_radius + ring * 36
        angle_rad = math.radians(angle_deg)
        return radius * math.cos(angle_rad), radius * math.sin(angle_rad)

    for marker in markers:
        marker_type = marker.get("marker_type", "outside_list")
        target_object = str(marker.get("target_object", "room")).strip()
        label = marker.get("label") or marker.get("description") or marker.get("title") or "개선사항"
        icon_type = marker.get("icon_type", "none")

        obj = find_target(target_object, label)
        room_wide_target = (not target_object or target_object == "room" or str(target_object).lower().startswith("path_")) and obj is None

        if marker_type == "outside_list" and room_wide_target:
            positioned.append({
                "priority": marker.get("priority"),
                "display_number": marker.get("display_number", marker.get("priority")),
                "label": label,
                "category": _related_category_label(label, target_object, layout),
                "x": None,
                "y": None,
                "outside": True,
                "icon_type": icon_type,
            })
            continue

        if obj:
            category = _related_category_label(label, target_object, layout)
            marker_index = target_marker_counts.get(target_object, 0)
            target_marker_counts[target_object] = marker_index + 1

            obj_x = float(obj.get("x", 0))
            obj_y = float(obj.get("y", 0))
            obj_width = float(obj.get("width", 100))
            obj_height = float(obj.get("height", 50))

            center_x = obj_x + obj_width / 2
            center_y = obj_y + obj_height / 2

            if marker_type == "object_center" and marker_index == 0:
                x = center_x - 16
                y = center_y - 16
            else:
                offset_x, offset_y = spiral_offset(marker_index, obj_width, obj_height)
                x = center_x + offset_x - 16
                y = center_y + offset_y - 16

            x, y = clamp_position(x, y)
        else:
            category = _related_category_label(label, target_object, layout)
            corner = room_corners[corner_idx % len(room_corners)]
            x = corner["x"]
            y = corner["y"]
            corner_idx += 1

        positioned.append({
            "priority": marker.get("priority"),
            "display_number": marker.get("display_number", marker.get("priority")),
            "label": label,
            "category": category,
            "x": int(x),
            "y": int(y),
            "outside": False,
            "icon_type": icon_type,
        })

    return positioned


def render_annotated_floorplan_html(
    layout: list,
    positioned_markers: list,
    room_width: int = 720,
    room_height: int = 460,
) -> str:
    """분석 결과용 평면도 HTML. 공간 안에는 가구와 위험 위치 번호만 표시합니다."""

    def _furniture_class(name: str, shape: str) -> str:
        n = str(name or "").lower()
        if shape == "door" or "문" in n:
            return " furniture-door"
        if any(k in n for k in ("침대", "소파", "의자")):
            return " furniture-soft"
        if any(k in n for k in ("테이블", "식탁", "책상", "협탁", "조리대", "싱크")):
            return " furniture-work"
        if any(k in n for k in ("수납", "서랍", "옷장", "tv장", "냉장고", "장")):
            return " furniture-storage"
        if any(k in n for k in ("러그", "매트")):
            return " furniture-floor"
        return " furniture-default"

    objects_html = ""
    for obj in layout:
        raw_name = str(obj.get("name", ""))
        obj_name = html_lib.escape(raw_name)
        shape = str(obj.get("shape", "rect") or "rect").strip().lower()
        shape_class = f" shape-{shape}" if shape in ("circle", "door") else ""
        furniture_class = _furniture_class(raw_name, shape)
        rotation = float(obj.get("rotation", 0) or 0)
        objects_html += f"""
        <div class="placed{shape_class}{furniture_class}" style="left:{obj.get('x', 0)}px; top:{obj.get('y', 0)}px;
            width:{obj.get('width', 100)}px; height:{obj.get('height', 50)}px;
            transform:rotate({rotation}deg);">
            <span>{obj_name}</span>
        </div>
        """

    inside_markers_html = ""
    for marker in positioned_markers:
        if marker.get("outside", False):
            continue
        priority = html_lib.escape(str(marker.get("display_number", marker.get("priority", ""))))
        label = html_lib.escape(clean_display_text(str(marker.get("label", ""))))
        category = html_lib.escape(clean_display_text(str(marker.get("category", "")).strip()))
        prefixed_label = f"[{category} 관련] {label}" if category else label
        inside_markers_html += f"""
        <div class="room-marker" title="{prefixed_label}" style="left:{marker.get('x', 0)}px; top:{marker.get('y', 0)}px;">
            {priority}
        </div>
        """

    major_grid = 100
    minor_grid = 50
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="UTF-8">
    <style>
        html, body {{ margin:0; padding:0; background:transparent;
            font-family:'Inter', -apple-system, 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif; }}
        * {{ box-sizing:border-box; }}
        .canvas {{ display:inline-grid; grid-template-columns:36px auto; grid-template-rows:32px auto;
            padding:8px 10px 12px 4px; }}
        .top-dimension {{ grid-column:2; grid-row:1; display:flex; align-items:center; justify-content:center;
            color:#66736e; font-size:12px; font-weight:700; letter-spacing:.01em; position:relative; }}
        .top-dimension::before, .top-dimension::after {{ content:""; position:absolute; top:15px; width:34%; height:1px; background:#aebdb7; }}
        .top-dimension::before {{ left:3%; }} .top-dimension::after {{ right:3%; }}
        .left-dimension {{ grid-column:1; grid-row:2; display:flex; align-items:center; justify-content:center;
            color:#66736e; font-size:12px; font-weight:700; writing-mode:vertical-rl; transform:rotate(180deg); }}
        .room-shell {{ grid-column:2; grid-row:2; padding:7px; border-radius:14px;
            background:#f7faf8; border:1px solid #dce6e1; box-shadow:0 8px 24px rgba(43,72,61,.08); }}
        .room {{ position:relative; width:{room_width}px; height:{room_height}px;
            border:2px solid #9fb2aa; border-radius:10px; overflow:hidden;
            background-color:#fbfdfc;
            background-image:
              linear-gradient(rgba(75,109,95,.10) 1px, transparent 1px),
              linear-gradient(90deg, rgba(75,109,95,.10) 1px, transparent 1px),
              linear-gradient(rgba(75,109,95,.045) 1px, transparent 1px),
              linear-gradient(90deg, rgba(75,109,95,.045) 1px, transparent 1px);
            background-size:{major_grid}px {major_grid}px, {major_grid}px {major_grid}px,
                            {minor_grid}px {minor_grid}px, {minor_grid}px {minor_grid}px; }}
        .placed {{ position:absolute; display:flex; align-items:center; justify-content:center;
            padding:7px; text-align:center; border:1.5px solid #9fb2aa; border-radius:9px;
            overflow:hidden; font-size:13px; font-weight:650; color:#2d3b36;
            transform-origin:center center; box-shadow:0 2px 7px rgba(45,69,60,.08); }}
        .placed span {{ position:relative; z-index:1; max-width:100%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
        .furniture-soft {{ background:#eef4f1; border-color:#8aa99d; }}
        .furniture-work {{ background:#f6f3eb; border-color:#b6aa8c; }}
        .furniture-storage {{ background:#eef1f4; border-color:#9eaab2; }}
        .furniture-floor {{ background:rgba(217,229,221,.58); border-style:dashed; border-color:#9bb4a9; }}
        .furniture-default {{ background:#ffffff; }}
        .placed.shape-circle {{ border-radius:50%; }}
        .placed.shape-door, .furniture-door {{ border-radius:100% 0 0 0; border-width:2px;
            background:rgba(255,255,255,.88); border-color:#7f978d; box-shadow:none; }}
        .room-marker {{ position:absolute; width:34px; height:34px; border-radius:50%;
            background:{CORAL}; color:#fff; font-weight:800; font-size:14px;
            display:flex; align-items:center; justify-content:center; border:3px solid #fff;
            box-shadow:0 3px 10px rgba(40,55,49,.28); z-index:20; cursor:help;
            transform:translate(-50%,-50%); }}
        .grid-note {{ grid-column:2; grid-row:3; margin-top:7px; color:#87928e; font-size:11px; text-align:right; }}
    </style>
    </head>
    <body>
        <div class="canvas">
            <div class="top-dimension">가로 {room_width} cm</div>
            <div class="left-dimension">세로 {room_height} cm</div>
            <div class="room-shell"><div class="room">{objects_html}{inside_markers_html}</div></div>
            <div class="grid-note">가는 선 50 cm · 진한 선 100 cm</div>
        </div>
    </body>
    </html>
    """

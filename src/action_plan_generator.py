"""
action_plan_generator.py
=========================
위험요인(hazard) 하나당, 그 안의 모든 대안 해결책에 대해 "실행 계획" 메타데이터
(예상 소요 시간, 난이도, 준비물, 도우미 필요 여부, 실행 방법 단계, 기대 효과)를
GPT로 생성합니다.

- 어떤 해결책을 제안할지는 이미 RAG 문헌 근거 + SHAP 개인 위험요인 분석으로 확정되어
  있습니다. 이 모듈은 그 문구를 바꾸지 않고, 문헌 DB에 없는 실무적·주관적 정보만 채웁니다.
- 같은 위험요인의 대안들을 "한 번에 같이" GPT에 물어봐서, 대안끼리 상대적으로 비교된
  난이도·시간이 매겨지도록 합니다. 대안 하나씩 따로 호출하면 기준이 흔들리고
  (예: 두 대안 다 "보통"으로 나와서 실제 난이도 차이가 안 드러남) 호출 비용도 커집니다.
  → 호출 횟수는 위험요인 개수만큼만 발생하고, 대안 개수와는 무관합니다.
"""

from __future__ import annotations

import json
import os

ACTION_PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "option_index": {"type": "integer"},
                    "estimated_minutes": {"type": "integer"},
                    "difficulty": {"type": "integer", "enum": [1, 2, 3]},
                    "required_items": {"type": "array", "items": {"type": "string"}},
                    "requires_helper": {"type": "boolean"},
                    "execution_steps": {"type": "array", "items": {"type": "string"}},
                    "expected_effects": {"type": "array", "items": {"type": "string"}},
                    "affects_furniture_height": {"type": "boolean"},
                    "target_height_cm": {"type": "integer"},
                    "height_source": {"type": "string", "enum": ["literature_text", "ai_estimate", "not_applicable"]},
                    "visual_addition_type": {
                        "type": "string",
                        "enum": ["grab_bar", "handrail", "non_slip_mat", "night_light", "ramp", "shower_chair", "none"],
                    },
                    "estimated_price_krw_min": {"type": "integer"},
                    "estimated_price_krw_max": {"type": "integer"},
                },
                "required": [
                    "option_index", "estimated_minutes", "difficulty", "required_items",
                    "requires_helper", "execution_steps", "expected_effects",
                    "affects_furniture_height", "target_height_cm", "height_source",
                    "visual_addition_type", "estimated_price_krw_min", "estimated_price_krw_max",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["alternatives"],
    "additionalProperties": False,
}

_INSTRUCTIONS_TEMPLATE = """
당신은 고령자 낙상예방 주거환경 개선안의 "실행 계획" 정보를 채우는 AI입니다.
아래 대안들은 이미 문헌 근거(RAG)와 개인 위험요인(SHAP) 분석으로 확정된 해결책입니다 —
어떤 해결책을 제안할지는 이미 정해져 있으므로 대안 문구 자체를 새로 만들거나 바꾸지 마십시오.
당신의 역할은 각 대안에 대해, 문헌에는 없는 실무적 정보(예상 소요 시간, 난이도, 준비물,
도우미 필요 여부, 실행 방법 단계, 기대 효과, 가구 높이 변경 여부)를 채우는 것뿐입니다.

반드시 지킬 것:
- 입력된 대안 순서(option_index)를 그대로 유지합니다.
- 난이도(difficulty)는 1=쉬움, 2=보통, 3=어려움 중 하나이며, **같은 위험요인 안의
  대안들끼리 상대적으로 비교**해서 일관되게 매깁니다 (모든 대안을 다 "보통"으로 매기지 않음).
- estimated_minutes는 그 대안 하나를 혼자 또는 도움을 받아 실행하는 데 걸리는 대략적인
  시간(분)입니다.
- execution_steps는 실제로 따라할 수 있는 2~4단계의 구체적인 행동으로 작성합니다.
- expected_effects는 1~2개의 기대 효과를 짧은 문장으로 작성합니다.
- 특정 제품명, 가격, 법적 기준, 정밀 치수·규격을 만들지 않습니다.
- required_items가 없으면 빈 배열 []을 사용합니다.
- affects_furniture_height: 이 대안이 가구(침대·수납장·의자 등)의 높이를 바꾸는 대안이면
  true, 위치 이동·정리·손잡이 설치처럼 높이와 무관하면 false로 표시합니다.
- target_height_cm과 height_source: affects_furniture_height가 true일 때만 채웁니다.
  **가장 먼저 확인할 것**: 대안 문구(text) 안에 이미 구체적인 높이 숫자나 범위가 적혀
  있는지 보십시오(예: "45~50cm로 조정", "40cm 이상"). 이미 적혀 있으면 그 숫자를 그대로
  가져다 쓰고(범위면 가운데 값을 정수로 반올림) height_source="literature_text"로
  표시하십시오 — 이 경우 당신은 새로운 숫자를 만드는 게 아니라 이미 문헌 근거에 있는
  값을 그대로 옮기는 것뿐입니다. 문구에 숫자가 전혀 없는데 높이를 바꾸는 대안이라면,
  일반적인 노인 친화 가구 높이 범위 감각으로 보수적으로 추정하고
  height_source="ai_estimate"로 표시하십시오. affects_furniture_height가 false이면
  target_height_cm=0, height_source="not_applicable"로 둡니다.
- visual_addition_type: 이 대안이 "기존 가구 높이를 바꾸는 게 아니라, 지금 없는 새로운
  고정물/보조기구를 새로 설치하는" 대안이면(그랩바·손잡이·미끄럼방지매트·야간조명·경사로·
  샤워의자 등), 가장 가까운 종류를 아래 중 하나로 표시하십시오: grab_bar(그랩바·짧은 손잡이),
  handrail(긴 안전 손잡이·난간), non_slip_mat(미끄럼방지 매트), night_light(야간 조명·
  센서등), ramp(문턱 완화용 경사로), shower_chair(샤워의자·좌식 보조기구). 해당하는 게
  없거나 단순 정리·이동·높이 조정 대안이면 none으로 둡니다. 이 값은 미리보기 그림에
  "새로 생기는 물건"을 표시하기 위한 용도이며, 실제 제품 사양을 뜻하지 않습니다.
- estimated_price_krw_min / estimated_price_krw_max: 이 대안을 실행하는 데 필요한
  준비물(required_items) 전체를 기준으로, 한국 기준 대략적인 총 비용 범위(원)를
  정수로 추정합니다. 이 값은 실제 시세 조사가 아니라 일반적인 감각의 근사치이므로
  보수적인 범위로 넓게 잡으십시오. 비용이 전혀 들지 않는 대안(가구 재배치, 정리 등)은
  둘 다 0으로 둡니다.

공간: {room_label}
현재 문제: {floorplan_problem}

출력은 지정된 JSON Schema를 정확히 따르십시오.
""".strip()


def _build_client():
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai 패키지를 불러오지 못했습니다. requirements.txt 설치가 필요합니다.") from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않아 실행 계획을 생성할 수 없습니다.")

    # Action Plan은 사용자가 화면에서 기다리는 동기 요청이므로 일반 분석보다 짧게 제한합니다.
    # 네트워크/API가 느릴 때 전체 Flask 요청이 오래 붙잡혀 브라우저에서 Failed to fetch가
    # 발생하지 않도록 하며, 호출 실패 시 app.py에서 결정론적 fallback 계획을 사용합니다.
    timeout_seconds = float(os.getenv("OPENAI_ACTION_PLAN_TIMEOUT_SECONDS", "18"))
    return OpenAI(timeout=timeout_seconds, max_retries=0)


def generate_action_plan_for_hazard(room_label: str, floorplan_problem: str, options: list[str]) -> list[dict]:
    """위험요인 하나(floorplan_problem)에 딸린 모든 대안(options)에 대해,
    options와 같은 순서·같은 길이의 실행 계획 dict 리스트를 반환합니다."""
    if not options:
        return []

    client = _build_client()
    model = os.getenv("OPENAI_MODEL", "gpt-5.5")

    payload = {
        "room_label": room_label,
        "floorplan_problem": floorplan_problem,
        "alternatives": [{"option_index": i, "text": opt} for i, opt in enumerate(options)],
    }
    instructions = _INSTRUCTIONS_TEMPLATE.format(room_label=room_label, floorplan_problem=floorplan_problem)

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "action_plan_alternatives",
                "strict": True,
                "schema": ACTION_PLAN_JSON_SCHEMA,
            }
        },
    )
    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("OpenAI 응답에 output_text가 없습니다.")

    result = json.loads(output_text)
    raw_alternatives = result.get("alternatives", [])
    by_index = {}
    for item in raw_alternatives:
        try:
            idx = int(item.get("option_index"))
        except (TypeError, ValueError):
            continue
        by_index[idx] = item

    ordered: list[dict] = []
    for i in range(len(options)):
        item = by_index.get(i, {})
        ordered.append({
            "estimated_minutes": item.get("estimated_minutes"),
            "difficulty": item.get("difficulty"),
            "required_items": item.get("required_items") or [],
            "requires_helper": bool(item.get("requires_helper", False)),
            "execution_steps": item.get("execution_steps") or [],
            "expected_effects": item.get("expected_effects") or [],
            "affects_furniture_height": bool(item.get("affects_furniture_height", False)),
            "target_height_cm": int(item.get("target_height_cm") or 0),
            "height_source": item.get("height_source", "not_applicable"),
            "visual_addition_type": item.get("visual_addition_type", "none"),
            "estimated_price_krw_min": int(item.get("estimated_price_krw_min") or 0),
            "estimated_price_krw_max": int(item.get("estimated_price_krw_max") or 0),
        })
    return ordered


# -----------------------------------------------------------------------------
# RAG 문헌 미매핑 위험요인에 대한 AI 독자 제안 (fallback)
# -----------------------------------------------------------------------------
# 배경: 백엔드 파이프라인(src/pipeline.py)은 위험요인마다 RAG 문헌 매칭을 시도하고,
# 실패한 것들을 completed["unresolved_hazards"]에 기록합니다. 그런데 같은 분석 안에
# 다른 위험요인이 RAG 매칭에 성공하면 전체 결과가 "RAG_EVIDENCE" 모드로 넘어가면서,
# 이 unresolved_hazards는 화면에 전혀 노출되지 않고 조용히 버려졌습니다(사용자가
# 체크한 위험요인인데 개선안이 하나도 안 나오는 문제). 이 함수는 그 미매핑 위험요인들에
# 대해 GPT가 독자적으로 개선안을 제안하게 하고, evidence_id 없이 명확히 "AI 독자 제안"
# 으로 표시합니다(RAG 근거로 오인되지 않도록).

AI_ONLY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hazard_index": {"type": "integer"},
                    "already_covered": {"type": "boolean"},
                    "improvement": {"type": "string"},
                },
                "required": ["hazard_index", "already_covered", "improvement"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["recommendations"],
    "additionalProperties": False,
}

_AI_ONLY_INSTRUCTIONS = """
당신은 고령자 낙상예방 주거환경 개선 전문가입니다. 아래 위험요인들은 사용자가 직접
확인·체크한 실제 위험요인이지만, 문헌 근거(RAG) 데이터베이스에서 딱 맞는 해결책을 찾지
못했습니다. 그렇다고 이 위험요인을 화면에서 빼면 사용자는 체크한 위험요인에 대한 개선안을
전혀 못 받게 되므로, 당신이 일반적인 낙상예방 원칙에 기반해 독자적으로 개선안을 제안해
주십시오.

**가장 먼저 확인할 것**: 이 공간에는 이미 문헌 근거로 추천된 해결책들이 따로 있습니다
(아래 "이미 추천된 해결책" 목록). 새 위험요인에 대해 떠오르는 해결책이 그 목록 중 하나와
사실상 같은 조치라면(예: 목록에 이미 "그랩바 설치"가 있는데 새 위험요인도 그랩바 설치로
해결된다면), **새로 만들지 말고** already_covered=true로 표시하고 improvement는 빈
문자열로 두십시오 — 이미 추천된 항목이 이 위험요인도 함께 해결한다는 뜻입니다. 이렇게
해야 사용자에게 사실상 같은 내용이 "문헌 근거 기반"과 "AI 독자 제안"으로 중복해서
보이는 걸 막을 수 있습니다. 정말 다른 조치가 필요한 경우에만 already_covered=false로
새 개선안을 작성하십시오.

반드시 지킬 것:
- 새로 작성하는 개선안은 위험요인마다 정확히 1개씩, 한국어 한 문장으로 작성합니다.
- 특정 제품명, 브랜드, 가격, 법적 기준, 정밀 치수·규격을 만들지 않습니다.
- 이미 상식적으로 잘 알려진 낙상예방 원칙(동선 확보, 미끄럼 방지, 문턱 낮추기, 조명 개선,
  손잡이 설치 등)에 근거해서만 제안하고, 근거 없는 의학적 주장은 하지 않습니다.
- hazard_index는 입력된 순서(0부터 시작)를 그대로 사용합니다.

공간: {room_label}

이미 추천된 해결책 (문헌 근거 기반, 중복 제안 금지):
{existing_recommendations}

출력은 지정된 JSON Schema를 정확히 따르십시오.
""".strip()


def generate_ai_only_recommendations(
    room_label: str, hazard_descriptions: list[str], existing_recommendations: list[str] | None = None,
) -> list[str]:
    """RAG 미매핑 위험요인 설명 리스트를 받아, 같은 순서·같은 길이의 개선안 문장 리스트를
    반환합니다. 위험요인 개수와 무관하게 호출은 1번만 발생합니다(배치 처리).
    이미 문헌 근거로 추천된 해결책(existing_recommendations)과 사실상 같은 내용이면
    GPT가 already_covered=true로 표시하고, 이 함수는 그 항목에 빈 문자열을 돌려줘서
    호출자가 "중복이니 새로 추가하지 말라"는 신호로 쓸 수 있게 합니다."""
    if not hazard_descriptions:
        return []

    client = _build_client()
    model = os.getenv("OPENAI_MODEL", "gpt-5.5")

    existing_text = "\n".join(f"- {r}" for r in (existing_recommendations or [])) or "(없음)"
    payload = {
        "room_label": room_label,
        "hazards": [{"hazard_index": i, "description": d} for i, d in enumerate(hazard_descriptions)],
    }
    instructions = _AI_ONLY_INSTRUCTIONS.format(room_label=room_label, existing_recommendations=existing_text)

    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "ai_only_recommendations",
                "strict": True,
                "schema": AI_ONLY_JSON_SCHEMA,
            }
        },
    )
    output_text = getattr(response, "output_text", None)
    if not output_text:
        raise RuntimeError("OpenAI 응답에 output_text가 없습니다.")

    result = json.loads(output_text)
    by_index = {}
    for item in result.get("recommendations", []):
        try:
            idx = int(item.get("hazard_index"))
        except (TypeError, ValueError):
            continue
        if item.get("already_covered"):
            by_index[idx] = ""
        else:
            by_index[idx] = str(item.get("improvement", "")).strip()

    return [by_index.get(i, "") for i in range(len(hazard_descriptions))]


def build_shopping_links(item_name: str) -> dict:
    """준비물 이름으로 쇼핑 검색 링크를 만듭니다. GPT에게 URL을 만들어 달라고 하면
    존재하지 않는 링크를 지어낼 위험이 있어서, 검색 쿼리 URL은 코드로 직접 조립합니다
    (실제 상품 링크가 아니라 "이 이름으로 검색한 결과" 페이지라 항상 유효합니다)."""
    from urllib.parse import quote

    q = quote(str(item_name).strip())
    if not q:
        return {}
    return {
        "coupang": f"https://www.coupang.com/np/search?q={q}",
        "naver": f"https://search.shopping.naver.com/search/all?query={q}",
    }


def generate_fallback_action_plan_for_hazard(
    room_label: str, floorplan_problem: str, options: list[str]
) -> list[dict]:
    """OpenAI 연결 실패/타임아웃 때 화면을 살리기 위한 보수적 실행계획 fallback.

    새로운 해결책을 만들지 않고 이미 확정된 option 문구만 실행 가능한 형태로 정리합니다.
    가격/정밀 규격은 추정하지 않으며 비용은 0으로 둡니다.
    """
    plans: list[dict] = []
    for option in options:
        text = str(option or "").strip()
        lower = text.lower()

        install_words = ("설치", "고정", "부착", "교체", "시공", "안전바", "손잡이", "조명", "매트", "경사로")
        move_words = ("이동", "치우", "정리", "재배치", "확보", "줄이", "옮기")
        helper_words = ("안전바", "손잡이", "시공", "고정", "설치", "교체")

        if any(w in text for w in install_words):
            difficulty, minutes = 2, 45
        elif any(w in text for w in move_words):
            difficulty, minutes = 1, 20
        else:
            difficulty, minutes = 1, 30

        requires_helper = any(w in text for w in helper_words)
        required_items: list[str] = []
        for keyword, item in (
            ("미끄럼", "미끄럼 방지 용품"),
            ("조명", "조명 또는 센서등"),
            ("안전바", "안전바"),
            ("손잡이", "보조 손잡이"),
            ("매트", "고정용 패드 또는 매트"),
        ):
            if keyword in text and item not in required_items:
                required_items.append(item)

        visual = "none"
        if "안전바" in text or "그랩바" in text:
            visual = "grab_bar"
        elif "손잡이" in text or "난간" in text:
            visual = "handrail"
        elif "미끄럼" in text or "매트" in text:
            visual = "non_slip_mat"
        elif "조명" in text or "센서등" in text:
            visual = "night_light"
        elif "경사로" in text:
            visual = "ramp"
        elif "샤워의자" in text or "샤워 의자" in text:
            visual = "shower_chair"

        plans.append({
            "estimated_minutes": minutes,
            "difficulty": difficulty,
            "required_items": required_items,
            "requires_helper": requires_helper,
            "execution_steps": [
                f"{room_label}에서 해당 문제 위치와 주변 통로를 먼저 확인합니다.",
                f"선택한 개선안인 ‘{text}’을(를) 주변 이동을 방해하지 않도록 적용합니다.",
                "적용 후 실제 보행 동선과 걸림·미끄럼 요소가 줄었는지 다시 확인합니다.",
            ],
            "expected_effects": [
                f"{floorplan_problem}과 관련된 생활공간 위험요인을 줄이는 데 도움을 줄 수 있습니다."
            ],
            "affects_furniture_height": any(w in text for w in ("높이", "낮은 의자", "낮은 침대")),
            "target_height_cm": 0,
            "height_source": "not_applicable",
            "visual_addition_type": visual,
            "estimated_price_krw_min": 0,
            "estimated_price_krw_max": 0,
            "fallback_generated": True,
        })
    return plans

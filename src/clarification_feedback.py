from __future__ import annotations

from copy import deepcopy
from typing import Any


VALID_ANSWERS = {"YES", "NO", "UNKNOWN"}


def normalize_answer(value: Any) -> str:
    """UI 응답을 백엔드 공통 코드 YES/NO/UNKNOWN으로 변환합니다."""
    text = str(value or "").strip().upper()
    aliases = {
        "예": "YES",
        "네": "YES",
        "Y": "YES",
        "YES": "YES",
        "아니오": "NO",
        "아니요": "NO",
        "N": "NO",
        "NO": "NO",
        "모름": "UNKNOWN",
        "모르겠음": "UNKNOWN",
        "확인 필요": "UNKNOWN",
        "UNKNOWN": "UNKNOWN",
        "": "UNKNOWN",
    }
    normalized = aliases.get(text, text)
    return normalized if normalized in VALID_ANSWERS else "UNKNOWN"


def merge_clarification_answers(
    *,
    existing_flags: dict[str, bool] | None,
    existing_question_context: list[dict[str, Any]] | None,
    clarification_questions: list[dict[str, Any]] | None,
    answers_by_question_id: dict[str, Any] | None,
    room_name: str,
) -> tuple[dict[str, bool], list[dict[str, Any]], dict[str, str]]:
    """GPT 추가 질문 응답을 다음 분석의 공간 입력으로 병합합니다.

    - YES: 해당 hazard를 사용자 확인 공간위험으로 전달합니다.
    - NO: 해당 hazard를 명시적 부재로 기록하고 AI 조건부 후보에서도 제외합니다.
    - UNKNOWN: 기존 체크값을 임의로 뒤집지 않고 미확정 상태로 유지합니다.

    같은 hazard에 질문이 여러 개면 YES > NO > UNKNOWN 순으로 병합합니다.
    """
    flags = dict(existing_flags or {})
    context = [deepcopy(item) for item in (existing_question_context or []) if isinstance(item, dict)]
    questions = [item for item in (clarification_questions or []) if isinstance(item, dict)]
    raw_answers = dict(answers_by_question_id or {})

    context_by_hazard: dict[str, dict[str, Any]] = {}
    context_order: list[str] = []
    for item in context:
        code = str(item.get("hazard_code") or "").strip()
        if not code:
            continue
        if code not in context_by_hazard:
            context_order.append(code)
            context_by_hazard[code] = item

    normalized_by_question: dict[str, str] = {}
    answer_priority = {"UNKNOWN": 0, "NO": 1, "YES": 2}
    answer_by_hazard: dict[str, str] = {}

    for index, question in enumerate(questions, start=1):
        question_id = str(question.get("question_id") or f"Q{index}")
        answer = normalize_answer(raw_answers.get(question_id))
        normalized_by_question[question_id] = answer
        related = [
            str(code).strip()
            for code in question.get("related_hazard_codes", [])
            if str(code).strip()
        ]
        for code in related:
            previous = answer_by_hazard.get(code, "UNKNOWN")
            if answer_priority[answer] >= answer_priority[previous]:
                answer_by_hazard[code] = answer

            if code not in context_by_hazard:
                context_by_hazard[code] = {
                    "hazard_code": code,
                    "label": question.get("question") or code,
                    "room_name": room_name,
                    "room_label": room_name,
                    "priority": index,
                    "triggered_by": ["GPT_CLARIFICATION"],
                    "trigger_labels": ["GPT 추가 확인 질문"],
                    "selection_policy": "GPT_CLARIFICATION_FEEDBACK",
                }
                context_order.append(code)

    for code, answer in answer_by_hazard.items():
        item = context_by_hazard[code]
        previous_checked = bool(item.get("checked") or flags.get(code, False))
        item["clarification_answer"] = answer
        item["clarification_confirmed"] = answer in {"YES", "NO"}

        if answer == "YES":
            flags[code] = True
            item["checked"] = True
            item["confirmation_status"] = "CONFIRMED_PRESENT_BY_CLARIFICATION"
        elif answer == "NO":
            flags[code] = False
            item["checked"] = False
            item["confirmation_status"] = "CONFIRMED_ABSENT_BY_CLARIFICATION"
        else:
            # '모름'은 기존에 사용자가 직접 체크한 위험을 지우지 않습니다.
            flags[code] = previous_checked
            item["checked"] = previous_checked
            item["confirmation_status"] = (
                "CONFIRMED_BY_EXISTING_CHECKLIST"
                if previous_checked
                else "UNKNOWN_AFTER_CLARIFICATION"
            )

    merged_context = [context_by_hazard[code] for code in context_order]
    return flags, merged_context, normalized_by_question

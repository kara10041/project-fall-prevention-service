from __future__ import annotations

from pathlib import Path


def write_plan_markdown(plan: dict, output_path: str | Path) -> None:
    lines = [
        f"# {plan.get('plan_title', 'GPT 통합 공간개선 계획')}",
        "",
        str(plan.get("summary", "")),
        "",
        "## 고정된 사용자별 우선순위",
        "",
    ]
    for step in plan.get("steps", []):
        lines.extend([
            f"### {step.get('fixed_priority_rank')}순위 · {step.get('hazard_code')}",
            "",
            f"- 구현 순서: {step.get('implementation_sequence')}",
            f"- 문헌 근거: {', '.join(step.get('evidence_ids', []))}",
            f"- 원문 솔루션: {step.get('source_recommendation')}",
            f"- 통합 적용안: {step.get('integrated_instruction')}",
            f"- 대상: {step.get('target', {}).get('room_type')} / {step.get('target', {}).get('object_id')}",
            f"- 순서 근거: {step.get('sequence_reason')}",
            "",
        ])

    conflicts = plan.get("cross_solution_conflicts", [])
    if conflicts:
        lines.extend(["## 솔루션 간 충돌 검토", ""])
        for conflict in conflicts:
            lines.append(
                f"- **{conflict.get('conflict_type')}**: {conflict.get('description')} "
                f"→ {conflict.get('resolution')}"
            )
        lines.append("")

    questions = plan.get("clarification_questions", [])
    if questions:
        lines.extend(["## 추가 확인 질문", ""])
        for q in questions:
            lines.append(f"- {q.get('question')} ({q.get('reason')})")
        lines.append("")

    if plan.get("limitations"):
        lines.extend(["## 제한사항", ""])
        for item in plan["limitations"]:
            lines.append(f"- {item}")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")

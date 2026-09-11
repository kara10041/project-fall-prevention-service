from __future__ import annotations

from html import escape
from pathlib import Path
import textwrap


def _center(geometry: dict) -> tuple[float, float]:
    coords = geometry.get("coordinates", [])
    if geometry.get("type") == "bbox" and len(coords) == 4:
        x, y, w, h = coords
        return x + w / 2, y + h / 2
    points = coords
    if points and isinstance(points[0], (list, tuple)):
        return sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points)
    return 30.0, 30.0


def _shape(geometry: dict, css_class: str) -> str:
    kind = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if kind == "bbox" and len(coords) == 4:
        x, y, w, h = coords
        return f'<rect class="{css_class}" x="{x}" y="{y}" width="{w}" height="{h}" rx="8"/>'
    if kind in {"polygon", "polyline"} and coords:
        points = " ".join(f"{p[0]},{p[1]}" for p in coords)
        tag = "polygon" if kind == "polygon" else "polyline"
        return f'<{tag} class="{css_class}" points="{points}"/>'
    return ""


def render_svg(floorplan: dict, plan: dict, output_path: str | Path) -> None:
    canvas = floorplan.get("canvas", {"width": 1200, "height": 800})
    plan_width, height = int(canvas["width"]), int(canvas["height"])
    panel_width = 520
    width = plan_width + panel_width
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        """<style>
        text{font-family:'NanumSquare','Noto Sans CJK KR','UnDotum',sans-serif}.room{fill:#f7f7f7;stroke:#777;stroke-width:2}
        .object{fill:#ececec;stroke:#999}.hazard{fill:#ff6b6b33;stroke:#c62828;stroke-width:4}
        .rank{fill:#c62828}.ranktext{fill:white;font-weight:700;text-anchor:middle;dominant-baseline:middle}
        .title{font-size:23px;font-weight:700}.item{font-size:16px;font-weight:700}.small{font-size:12px;fill:#555}
        </style>""",
        f'<rect width="{width}" height="{height}" fill="white"/>',
    ]
    for room in floorplan.get("rooms", []):
        parts.append(_shape({"type": "polygon", "coordinates": room.get("polygon", [])}, "room"))
        if room.get("polygon"):
            x, y = room["polygon"][0]
            parts.append(f'<text x="{x+10}" y="{y+24}" class="small">{escape(room.get("room_type", ""))}</text>')
    for obj in floorplan.get("objects", []):
        parts.append(_shape({"type": "bbox", "coordinates": obj.get("bbox", [])}, "object"))
    for action in plan.get("ranked_actions", []):
        geometry = action.get("anchor_geometry", {})
        parts.append(_shape(geometry, "hazard"))
        x, y = _center(geometry)
        rank = action.get("recommendation_rank", "?")
        parts.append(f'<circle class="rank" cx="{x}" cy="{y}" r="16"/>')
        parts.append(f'<text class="ranktext" x="{x}" y="{y+1}">{rank}</text>')
    parts.append(f'<rect x="{plan_width}" y="0" width="{panel_width}" height="{height}" fill="#fafafa" stroke="#ddd"/>')
    px = plan_width + 22
    parts.append(f'<text x="{px}" y="38" class="title">공간 개선 우선순위</text>')
    parts.append(f'<text x="{px}" y="60" class="small">개인위험(SHAP) × 실제위험 × 신뢰도 → RAG</text>')
    y = 96
    for action in plan.get("ranked_actions", []):
        evidence = action.get("selected_evidence", {})
        rank = action.get("recommendation_rank", "?")
        title = str(evidence.get("recommendation", action.get("hazard_code", "")))
        parts.append(f'<circle class="rank" cx="{px+14}" cy="{y-5}" r="13"/>')
        parts.append(f'<text class="ranktext" x="{px+14}" y="{y-4}">{rank}</text>')
        lines = textwrap.wrap(title, width=34)[:4]
        for index, line in enumerate(lines):
            parts.append(f'<text x="{px+38}" y="{y+index*18}" class="item">{escape(line)}</text>')
        dy = y + len(lines) * 18 + 4
        parts.append(f'<text x="{px+38}" y="{dy}" class="small">{escape(action.get("room_type", ""))} · {escape(action.get("hazard_code", ""))}</text>')
        parts.append(f'<text x="{px+38}" y="{dy+17}" class="small">priority={float(action.get("priority_score",0)):.3f} · {escape(str(evidence.get("tier", "")))}</text>')
        y = dy + 54
        if y > height - 50:
            break
    parts.append("</svg>")
    Path(output_path).write_text("\n".join(part for part in parts if part), encoding="utf-8")

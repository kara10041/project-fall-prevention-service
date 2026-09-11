"""
deterministic_floorplan.py
===========================
AI 이미지 생성에 의존하지 않고, 사용자가 실제로 배치한 가구 좌표(x, y, width, height)와
백엔드가 이미 계산해 둔 마커 좌표(calculate_marker_positions)를 그대로 사용해
평면도 이미지를 정확하게 그립니다.

배경: 처음에는 OpenAI 이미지 생성/편집 API로 Before/After 평면도를 만들었는데, 두 가지
문제가 있었습니다.
1) AI가 텍스트 프롬프트로 받은 좌표·글자 배치 지시를 100% 지키지 못해 레이아웃이 깨짐
   (배지 텍스트가 방 밖으로 삐져나오는 등)
2) 대안(다른 해결책)을 선택할 때마다 이미지 편집 API를 다시 호출해야 해서 수십 초씩 걸림

이 모듈은 두 문제를 구조적으로 없앱니다 — 이미지를 "생성"하지 않고 이미 갖고 있는 정확한
좌표로 "그리기"만 하므로, 레이아웃이 깨질 수 없고 렌더링도 즉시 끝납니다. 다만 이 방식은
"가구를 어디로 옮겨야 하는지" 같은 새로운 목표 좌표까지는 계산하지 않으므로, 가구 이동
화살표(after 쪽에서 가구 위치 자체가 바뀌는 연출)는 만들지 않습니다 — 지금 배치는 그대로
두고, 그 위에 위험/개선 마커만 얹어서 보여줍니다.
"""

from __future__ import annotations

import io
import math
import os
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# static/css/style.css의 디자인 토큰과 맞춘 색상
COLOR_CANVAS_BG = (238, 242, 237)      # --bg
COLOR_ROOM_BG = (251, 252, 250)        # --paper 근처
COLOR_LINE_STRONG = (143, 163, 153)    # --line-strong
COLOR_FURNITURE_BG = (255, 255, 255)
COLOR_TEAL = (47, 111, 98)             # --teal
COLOR_INK = (30, 43, 38)               # --ink
COLOR_CORAL = (193, 84, 60)            # --coral (위험 마커)
COLOR_SAGE = (90, 138, 111)            # --sage 계열 (개선 완료 마커)

_FONT_PATH_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "C:/Windows/Fonts/malgun.ttf",
)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATH_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def pick_canvas_size(room_width: float, room_height: float) -> tuple[tuple[int, int], str]:
    """방 실측 비율에 맞춰 보기 좋은 캔버스 크기를 고릅니다."""
    ratio = (room_width or 1) / (room_height or 1)
    if ratio >= 1.2:
        return (1536, 1024), "1536x1024"
    if ratio <= 1 / 1.2:
        return (1024, 1536), "1024x1536"
    return (1024, 1024), "1024x1024"


def _fit_text(draw: ImageDraw.ImageDraw, text: str, max_width: float, base_size: int) -> tuple[ImageFont.FreeTypeFont, str]:
    """박스 너비 안에 들어가도록 폰트 크기를 줄이거나, 그래도 안 들어가면 말줄임표로 자릅니다."""
    size = base_size
    while size > 9:
        font = _load_font(size)
        if draw.textlength(text, font=font) <= max_width:
            return font, text
        size -= 1

    font = _load_font(10)
    truncated = text
    while len(truncated) > 1 and draw.textlength(truncated + "…", font=font) > max_width:
        truncated = truncated[:-1]
    return font, (truncated + "…") if truncated != text else text


def _room_geometry(room_width: float, room_height: float, canvas_px: tuple[int, int]) -> tuple[float, float, float]:
    """캔버스 안에서 방을 그릴 배율(scale)과 좌상단 오프셋(ox, oy)을 계산합니다."""
    canvas_w, canvas_h = canvas_px
    margin = int(min(canvas_w, canvas_h) * 0.09)
    avail_w = canvas_w - margin * 2
    avail_h = canvas_h - margin * 2
    scale = max(0.01, min(avail_w / room_width, avail_h / room_height))
    room_px_w = room_width * scale
    room_px_h = room_height * scale
    ox = (canvas_w - room_px_w) / 2
    oy = (canvas_h - room_px_h) / 2
    return scale, ox, oy


def _draw_room_and_furniture(
    draw: ImageDraw.ImageDraw, layout: list, room_width: float, room_height: float,
    canvas_px: tuple[int, int], scale: float, ox: float, oy: float,
    show_handles: bool = True,
) -> None:
    room_px_w = room_width * scale
    room_px_h = room_height * scale
    draw.rounded_rectangle(
        [ox, oy, ox + room_px_w, oy + room_px_h],
        radius=12, fill=COLOR_ROOM_BG, outline=COLOR_LINE_STRONG, width=3,
    )

    for item in layout:
        if not isinstance(item, dict):
            continue
        try:
            x = ox + float(item.get("x", 0)) * scale
            y = oy + float(item.get("y", 0)) * scale
            w = max(6.0, float(item.get("width", 60)) * scale)
            h = max(6.0, float(item.get("height", 40)) * scale)
        except (TypeError, ValueError):
            continue
        name = str(item.get("name", "가구")).strip() or "가구"

        draw.rounded_rectangle([x, y, x + w, y + h], radius=7, fill=COLOR_FURNITURE_BG, outline=COLOR_LINE_STRONG, width=2)

        pad = 6
        base_size = max(11, int(min(w, h) * 0.22))
        font, label = _fit_text(draw, name, max(4.0, w - pad * 2), base_size)
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + w / 2 - text_w / 2, y + h / 2 - text_h / 2 - bbox[1]), label, font=font, fill=COLOR_INK)

        if show_handles:
            handle = max(7, min(16, int(min(w, h) * 0.16)))
            draw.rounded_rectangle(
                [x + w - handle - 3, y + h - handle - 3, x + w - 3, y + h - 3],
                radius=2, fill=COLOR_TEAL,
            )


def render_floorplan_png(
    room_width: float,
    room_height: float,
    layout: list,
    canvas_px: Optional[tuple[int, int]] = None,
    show_handles: bool = True,
) -> bytes:
    """실제 배치(layout: [{name, x, y, width, height}, ...])를 좌표 그대로 그린 PNG 바이트를 반환합니다."""
    room_width = float(room_width or 720)
    room_height = float(room_height or 460)
    if canvas_px is None:
        canvas_px, _ = pick_canvas_size(room_width, room_height)

    img = Image.new("RGB", canvas_px, COLOR_CANVAS_BG)
    draw = ImageDraw.Draw(img)
    scale, ox, oy = _room_geometry(room_width, room_height, canvas_px)
    _draw_room_and_furniture(draw, layout, room_width, room_height, canvas_px, scale, ox, oy, show_handles=show_handles)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _draw_checkmark(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, color) -> None:
    w = max(2, int(r * 0.18))
    p1 = (cx - r * 0.45, cy + r * 0.02)
    p2 = (cx - r * 0.1, cy + r * 0.4)
    p3 = (cx + r * 0.5, cy - r * 0.35)
    draw.line([p1, p2, p3], fill=color, width=w, joint="curve")


def _draw_addition_icon(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, icon_type: str) -> None:
    """지금 배치에는 없지만 새로 추가되는 고정물(그랩바·매트·조명 등)을 간단한 아이콘으로
    그립니다. "개선 후" 이미지에서 체크마크만으로는 뭐가 달라졌는지 안 보이는 문제
    (예: 그랩바 설치처럼 기존 가구의 위치·높이가 안 바뀌는 개선안)를 보완하기 위함입니다."""
    color = COLOR_TEAL
    w = max(2, int(r * 0.2))
    if icon_type == "grab_bar":
        draw.line([(cx - r, cy), (cx + r, cy)], fill=color, width=w)
        draw.line([(cx - r, cy - r * 0.5), (cx - r, cy + r * 0.5)], fill=color, width=w)
        draw.line([(cx + r, cy - r * 0.5), (cx + r, cy + r * 0.5)], fill=color, width=w)
    elif icon_type == "handrail":
        draw.line([(cx - r * 1.2, cy), (cx + r * 1.2, cy)], fill=color, width=max(3, int(r * 0.26)))
        for dx in (-r * 1.2, 0, r * 1.2):
            draw.ellipse([cx + dx - r * 0.14, cy - r * 0.14, cx + dx + r * 0.14, cy + r * 0.14], fill=color)
    elif icon_type == "non_slip_mat":
        draw.rounded_rectangle([cx - r, cy - r * 0.6, cx + r, cy + r * 0.6], radius=r * 0.18, outline=color, width=w)
        for i in (-1, 0, 1):
            draw.line(
                [(cx + i * r * 0.55 - r * 0.25, cy + r * 0.45), (cx + i * r * 0.55 + r * 0.25, cy - r * 0.45)],
                fill=color, width=max(1, int(r * 0.09)),
            )
    elif icon_type == "night_light":
        draw.ellipse([cx - r * 0.45, cy - r * 0.45, cx + r * 0.45, cy + r * 0.45], fill=(240, 200, 90))
        for ang_deg in range(0, 360, 45):
            ang = math.radians(ang_deg)
            x1, y1 = cx + math.cos(ang) * r * 0.55, cy + math.sin(ang) * r * 0.55
            x2, y2 = cx + math.cos(ang) * r * 0.95, cy + math.sin(ang) * r * 0.95
            draw.line([(x1, y1), (x2, y2)], fill=(240, 200, 90), width=max(1, int(r * 0.08)))
    elif icon_type == "ramp":
        draw.polygon(
            [(cx - r, cy + r * 0.55), (cx + r, cy + r * 0.55), (cx + r, cy - r * 0.55)],
            outline=color, width=w,
        )
    elif icon_type == "shower_chair":
        draw.rounded_rectangle([cx - r * 0.65, cy - r * 0.5, cx + r * 0.65, cy + r * 0.05], radius=r * 0.14, outline=color, width=w)
        draw.line([(cx - r * 0.45, cy + r * 0.05), (cx - r * 0.45, cy + r * 0.65)], fill=color, width=max(2, int(r * 0.1)))
        draw.line([(cx + r * 0.45, cy + r * 0.05), (cx + r * 0.45, cy + r * 0.65)], fill=color, width=max(2, int(r * 0.1)))


def render_floorplan_with_markers(
    room_width: float,
    room_height: float,
    layout: list,
    positioned_markers: list,
    canvas_px: Optional[tuple[int, int]] = None,
    marker_style: str = "risk",
    show_handles: bool = False,
) -> bytes:
    """평면도 + 번호/체크 마커를 함께 그립니다.
    marker_style="risk": 코랄색 원 안에 우선순위 번호 (개선 전, 위험 요소)
    marker_style="done": 세이지색 원 안에 체크 표시 + 작은 번호 (개선 후, 개선 포인트)
    positioned_markers는 web_helper.calculate_marker_positions()의 반환값과 같은 형식이며,
    x/y는 layout과 동일한 단위(방 실측 cm)의 32(cm 상당) 정사각형 마커 좌상단 좌표입니다."""
    room_width = float(room_width or 720)
    room_height = float(room_height or 460)
    if canvas_px is None:
        canvas_px, _ = pick_canvas_size(room_width, room_height)

    img = Image.new("RGB", canvas_px, COLOR_CANVAS_BG)
    draw = ImageDraw.Draw(img)
    scale, ox, oy = _room_geometry(room_width, room_height, canvas_px)
    _draw_room_and_furniture(draw, layout, room_width, room_height, canvas_px, scale, ox, oy, show_handles=show_handles)

    marker_color = COLOR_CORAL if marker_style == "risk" else COLOR_SAGE
    num_font = _load_font(max(11, int(16 * scale)))

    for marker in positioned_markers:
        if marker.get("outside") or marker.get("x") is None or marker.get("y") is None:
            continue
        r = 16 * scale
        cx = ox + (float(marker["x"]) + 16) * scale
        cy = oy + (float(marker["y"]) + 16) * scale

        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=marker_color, outline="white", width=max(2, int(r * 0.14)))

        label = str(marker.get("display_number", marker.get("priority", "")))
        bbox = draw.textbbox((0, 0), label, font=num_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw / 2, cy - th / 2 - bbox[1]), label, font=num_font, fill="white")

        if marker_style == "done":
            badge_r = max(5, r * 0.38)
            bx, by = cx + r * 0.72, cy - r * 0.72
            draw.ellipse([bx-badge_r, by-badge_r, bx+badge_r, by+badge_r], fill="white", outline=COLOR_SAGE, width=max(1, int(badge_r*0.18)))
            _draw_checkmark(draw, bx, by, badge_r * 0.7, COLOR_SAGE)
            icon_type = marker.get("icon_type", "none")
            if icon_type and icon_type != "none":
                _draw_addition_icon(draw, cx + r * 2.6, cy, r * 1.1, icon_type)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# -----------------------------------------------------------------------------
# 아이소메트릭 3D 스타일 렌더링
# -----------------------------------------------------------------------------
# 2D 평면도는 "위에서 내려다본" 그림이라 가구 높이를 표현할 수 없습니다(예: "침대를
# 높여라" 같은 개선안은 2D에서는 눈에 보이는 변화가 없음). 이 섹션은 같은 좌표 데이터를
# 아이소메트릭(등각 투영) 박스로 다시 그려서, 가구 높이 차이까지 시각적으로 보여줍니다.
# 여전히 AI 이미지 생성이 아니라 좌표 계산이라 빠르고 레이아웃이 깨지지 않습니다.

# 가구 이름에 이 키워드가 포함되면 기본 높이(cm)로 사용 (실측 데이터가 없을 때의 근사치)
DEFAULT_FURNITURE_HEIGHT_CM = {
    "침대": 45, "협탁": 55, "화장대": 75, "옷장": 180, "책상": 72,
    "수납장": 90, "식탁": 72, "의자": 45, "소파": 40, "러그": 2,
    "냉장고": 170, "싱크대": 85, "조리대": 90, "TV장": 45, "TV": 100,
    "변기": 40, "세면대": 80, "욕조": 55, "샤워": 5,
}
DEFAULT_HEIGHT_FALLBACK_CM = 45
WALL_HEIGHT_CM = 220
_ZU_RATIO = 0.5  # 세로(높이) 축척을 가로 축척의 몇 배로 할지

COLOR_WALL_LEFT = (233, 229, 216)
COLOR_WALL_BACK = (253, 252, 249)
COLOR_FURN_TOP = (255, 255, 255)
COLOR_FURN_LEFT = (216, 219, 210)
COLOR_FURN_RIGHT = (191, 196, 184)
COLOR_FURN_TOP_HL = (223, 234, 227)     # 높이가 바뀐(강조) 가구 - sage 톤
COLOR_FURN_LEFT_HL = (183, 204, 190)
COLOR_FURN_RIGHT_HL = (146, 174, 158)


def _furniture_height_cm(name: str, override: Optional[float] = None) -> float:
    if override:
        return float(override)
    name = str(name or "")
    for key, h in DEFAULT_FURNITURE_HEIGHT_CM.items():
        if key in name:
            return float(h)
    return float(DEFAULT_HEIGHT_FALLBACK_CM)


def pick_iso_canvas_size(room_width: float, room_height: float) -> tuple[tuple[int, int], str]:
    """아이소메트릭 장면 전용 캔버스 크기 선택.

    pick_canvas_size()는 2D 평면도(윗면만, 벽 높이 없음) 기준 방 가로세로 비율로 캔버스를
    고르는데, 아이소메트릭은 여기에 "벽 높이만큼 위로 솟은 부분"이 추가로 필요해서 실제
    그림의 가로세로 비율이 달라집니다. 2D용 함수를 그대로 재사용하면(특히 세로로 긴 방일
    때) 그림이 캔버스 위쪽에 몰리고 아래쪽에 큰 빈 공간이 남는 문제가 있어서, 여기서는
    "다이아몬드 폭 : (다이아몬드 반높이 + 벽높이)" 비율을 직접 계산해서 그 비율에 맞는
    캔버스를 고릅니다."""
    diamond_w_units = (room_width or 1) + (room_height or 1)
    diamond_h_units = diamond_w_units / 2 + WALL_HEIGHT_CM * _ZU_RATIO
    ratio = diamond_w_units / diamond_h_units if diamond_h_units else 1
    if ratio >= 1.2:
        return (1536, 1024), "1536x1024"
    if ratio <= 1 / 1.2:
        return (1024, 1536), "1024x1536"
    return (1024, 1024), "1024x1024"


def _pick_iso_scale(room_width: float, room_height: float, canvas_px: tuple[int, int]) -> tuple[float, float]:
    canvas_w, canvas_h = canvas_px
    margin_ratio = 0.1
    avail_w = canvas_w * (1 - margin_ratio * 2)
    avail_h = canvas_h * (1 - margin_ratio * 2)
    diamond_w_units = room_width + room_height
    diamond_h_units = diamond_w_units / 2 + WALL_HEIGHT_CM * _ZU_RATIO
    u_from_w = avail_w / diamond_w_units if diamond_w_units else 1
    u_from_h = avail_h / diamond_h_units if diamond_h_units else 1
    U = max(0.4, min(u_from_w, u_from_h))
    return U, U * _ZU_RATIO


def render_isometric_scene(
    room_width: float,
    room_height: float,
    layout: list,
    positioned_markers: list,
    canvas_px: Optional[tuple[int, int]] = None,
    marker_style: str = "risk",
    height_overrides: Optional[dict] = None,
) -> bytes:
    """같은 가구 좌표를 아이소메트릭 3D 박스로 그립니다.
    height_overrides: {가구 id 또는 이름: 목표 높이(cm)} — Action Plan에서 GPT가 추정한
    "가구 높이를 이렇게 바꾸면 좋겠다"는 값입니다(문헌 근거 아님, AI 추정치)."""
    room_width = float(room_width or 720)
    room_height = float(room_height or 460)
    height_overrides = height_overrides or {}
    if canvas_px is None:
        canvas_px, _ = pick_iso_canvas_size(room_width, room_height)
    canvas_w, canvas_h = canvas_px

    U, Zu = _pick_iso_scale(room_width, room_height, canvas_px)

    def iso(x, y, z, ox, oy):
        return (ox + (x - y) * U, oy + (x + y) * U / 2 - z * Zu)

    probe_ox, probe_oy = 0.0, 0.0
    probe_points = [
        iso(0, 0, WALL_HEIGHT_CM, probe_ox, probe_oy),
        iso(room_width, 0, 0, probe_ox, probe_oy),
        iso(room_width, room_height, 0, probe_ox, probe_oy),
        iso(0, room_height, 0, probe_ox, probe_oy),
    ]
    min_x = min(p[0] for p in probe_points)
    max_x = max(p[0] for p in probe_points)
    min_y = min(p[1] for p in probe_points)
    max_y = max(p[1] for p in probe_points)
    ox = canvas_w / 2 - (min_x + max_x) / 2
    oy = canvas_h * 0.12 - min_y

    img = Image.new("RGB", canvas_px, COLOR_CANVAS_BG)
    draw = ImageDraw.Draw(img)

    def P(x, y, z=0.0):
        return iso(x, y, z, ox, oy)

    draw.polygon([P(0, 0, 0), P(0, room_height, 0), P(0, room_height, WALL_HEIGHT_CM), P(0, 0, WALL_HEIGHT_CM)],
                 fill=COLOR_WALL_LEFT, outline=COLOR_LINE_STRONG)
    draw.polygon([P(0, 0, 0), P(room_width, 0, 0), P(room_width, 0, WALL_HEIGHT_CM), P(0, 0, WALL_HEIGHT_CM)],
                 fill=COLOR_WALL_BACK, outline=COLOR_LINE_STRONG)

    draw.polygon([P(0, 0, 0), P(room_width, 0, 0), P(room_width, room_height, 0), P(0, room_height, 0)],
                 fill=COLOR_ROOM_BG, outline=COLOR_LINE_STRONG)

    font_size = max(9, int(11 * max(U / 3.2, 0.7)))
    font = _load_font(font_size)

    items = []
    for item in layout:
        if not isinstance(item, dict):
            continue
        try:
            x0 = float(item.get("x", 0)); y0 = float(item.get("y", 0))
            w = float(item.get("width", 60)); h = float(item.get("height", 40))
        except (TypeError, ValueError):
            continue
        items.append((x0, y0, w, h, item))
    items.sort(key=lambda t: t[0] + t[1] + t[2] + t[3])

    for x0, y0, w, h, item in items:
        name = str(item.get("name", "가구")).strip() or "가구"
        item_id = str(item.get("id", ""))
        override = height_overrides.get(item_id) or height_overrides.get(name)
        height_cm = _furniture_height_cm(name, override)
        highlighted = override is not None

        x1, y1 = x0 + w, y0 + h
        top_color = COLOR_FURN_TOP_HL if highlighted else COLOR_FURN_TOP
        left_color = COLOR_FURN_LEFT_HL if highlighted else COLOR_FURN_LEFT
        right_color = COLOR_FURN_RIGHT_HL if highlighted else COLOR_FURN_RIGHT

        draw.polygon([P(x0, y0, 0), P(x0, y1, 0), P(x0, y1, height_cm), P(x0, y0, height_cm)],
                     fill=left_color, outline=COLOR_LINE_STRONG)
        draw.polygon([P(x0, y1, 0), P(x1, y1, 0), P(x1, y1, height_cm), P(x0, y1, height_cm)],
                     fill=right_color, outline=COLOR_LINE_STRONG)
        draw.polygon([P(x0, y0, height_cm), P(x1, y0, height_cm), P(x1, y1, height_cm), P(x0, y1, height_cm)],
                     fill=top_color, outline=COLOR_LINE_STRONG)

        # 3D 안의 작은 가구명/수치 텍스트는 축소 시 읽기 어렵고 오히려 혼란을 줍니다.
        # 가구명과 개선 설명은 바로 위 2D/범례에서 확인하고, 3D는 위치·높낮이 관계와
        # 큰 번호 마커만 보여주는 보조 시각화로 사용합니다.

    marker_color = COLOR_CORAL if marker_style == "risk" else COLOR_SAGE
    marker_r = max(16, U * 0.78)
    num_font = _load_font(max(16, int(marker_r * 0.95)))

    for marker in positioned_markers:
        if marker.get("outside") or marker.get("x") is None or marker.get("y") is None:
            continue
        mx = float(marker["x"]) + 16
        my = float(marker["y"]) + 16
        nearest_h = 0.0
        best_dist = None
        for x0, y0, w, h, item in items:
            cx, cy = x0 + w / 2, y0 + h / 2
            dist = (cx - mx) ** 2 + (cy - my) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                name = str(item.get("name", ""))
                item_id = str(item.get("id", ""))
                override = height_overrides.get(item_id) or height_overrides.get(name)
                nearest_h = _furniture_height_cm(name, override)

        cx, cy = P(mx, my, nearest_h + 22)
        draw.ellipse([cx - marker_r, cy - marker_r, cx + marker_r, cy + marker_r],
                     fill=marker_color, outline="white", width=max(2, int(marker_r * 0.14)))
        label = str(marker.get("priority", ""))
        bbox = draw.textbbox((0, 0), label, font=num_font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((cx - tw / 2, cy - th / 2 - bbox[1]), label, font=num_font, fill="white")

        if marker_style == "done":
            badge_r = max(5, marker_r * 0.38)
            bx, by = cx + marker_r * 0.72, cy - marker_r * 0.72
            draw.ellipse([bx-badge_r, by-badge_r, bx+badge_r, by+badge_r], fill="white", outline=COLOR_SAGE, width=max(1, int(badge_r*0.18)))
            _draw_checkmark(draw, bx, by, badge_r * 0.7, COLOR_SAGE)
            icon_type = marker.get("icon_type", "none")
            if icon_type and icon_type != "none":
                _draw_addition_icon(draw, cx + marker_r * 2.6, cy, marker_r * 1.1, icon_type)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

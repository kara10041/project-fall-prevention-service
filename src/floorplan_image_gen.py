"""
floorplan_image_gen.py
=======================
"참고용 실사(포토리얼리스틱) 이미지"를 만듭니다.

- 개선 전/후 비교의 "정확성"은 더 이상 이 모듈이 책임지지 않습니다 — 그건 이제
  src/deterministic_floorplan.py(좌표 기반 2D/3D 렌더링)가 담당합니다.
- 이 모듈은 딱 하나, "AI가 상상해서 그린 실사 느낌 참고 이미지"만 만듭니다.
  정확한 가구 배치·개수·치수를 보장하지 않으며, 화면에도 항상 "참고용, AI 생성"이라고
  명시해서 보여줘야 합니다. 좌표 기반 렌더링이 신뢰할 수 있는 "사실"이라면, 이 이미지는
  "이런 느낌일 수도 있겠다"는 감을 잡기 위한 보조 자료입니다.
- OPENAI_API_KEY가 없거나 호출이 실패하면 예외를 던져 호출자가 사용자에게 명확히 안내하게 합니다.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
import uuid
from pathlib import Path

logger = logging.getLogger("floorplan_image_gen")

GENERATED_DIR = Path(__file__).resolve().parent.parent / "static" / "generated"


def _furniture_lines(layout: list, room_width: float, room_height: float) -> str:
    """실사 생성 모델이 평면도와 최대한 비슷한 토폴로지를 유지하도록
    가구의 실제 bbox와 방 안 상대 위치까지 텍스트로 전달합니다."""
    lines = []
    rw = max(float(room_width or 1), 1.0)
    rh = max(float(room_height or 1), 1.0)
    for index, item in enumerate(layout, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "가구")).strip() or "가구"
        try:
            x = float(item.get("x", 0)); y = float(item.get("y", 0))
            w = float(item.get("width", 0)); h = float(item.get("height", 0))
            rotation = float(item.get("rotation", 0) or 0)
        except (TypeError, ValueError):
            continue
        cx = x + w / 2; cy = y + h / 2
        horizontal = "왼쪽" if cx < rw/3 else "오른쪽" if cx > rw*2/3 else "가운데"
        vertical = "위쪽" if cy < rh/3 else "아래쪽" if cy > rh*2/3 else "중앙"
        lines.append(
            f"- {index}. {name}: bbox x={round(x)}cm, y={round(y)}cm, "
            f"w={round(w)}cm, h={round(h)}cm, rotation={round(rotation)}deg; "
            f"방의 {vertical}-{horizontal} 영역"
        )
    return "\n".join(lines) if lines else "- (배치된 가구 없음)"



def _spatial_lock_lines(layout: list, room_width: float, room_height: float) -> str:
    """이미지 모델이 평면도를 '비슷한 방'으로 재해석하지 못하도록 각 객체의
    정규화 좌표, 벽과의 거리, 중심점, 방향을 명시합니다."""
    rw = max(float(room_width or 1), 1.0)
    rh = max(float(room_height or 1), 1.0)
    lines = []
    for index, item in enumerate(layout or [], start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "가구").strip() or "가구"
        try:
            x = float(item.get("x", 0)); y = float(item.get("y", 0))
            w = max(float(item.get("width", 0)), 1.0); h = max(float(item.get("height", 0)), 1.0)
            rotation = float(item.get("rotation", 0) or 0)
        except (TypeError, ValueError):
            continue
        cx, cy = x + w/2, y + h/2
        left, top = x, y
        right, bottom = max(0.0, rw-(x+w)), max(0.0, rh-(y+h))
        orientation = "세로로 긴 형태" if h > w*1.25 else "가로로 긴 형태" if w > h*1.25 else "거의 정사각/원형 비율"
        wall_notes=[]
        if left <= max(20, rw*0.05): wall_notes.append("왼쪽 벽에 붙거나 매우 가까움")
        if right <= max(20, rw*0.05): wall_notes.append("오른쪽 벽에 붙거나 매우 가까움")
        if top <= max(20, rh*0.05): wall_notes.append("위쪽 벽에 붙거나 매우 가까움")
        if bottom <= max(20, rh*0.05): wall_notes.append("아래쪽 벽에 붙거나 매우 가까움")
        wall_text = ", ".join(wall_notes) if wall_notes else "벽에서 떨어진 내부 배치"
        lines.append(
            f"- LOCK {index} [{name}]: center=({cx/rw:.3f}W,{cy/rh:.3f}H), "
            f"bbox=({x/rw:.3f}W,{y/rh:.3f}H,{w/rw:.3f}W,{h/rh:.3f}H), "
            f"rotation={rotation:.0f}deg, {orientation}, {wall_text}. "
            f"이 객체를 다른 벽/다른 구역으로 옮기거나 다른 객체와 합치지 마세요."
        )
    return "\n".join(lines) if lines else "- 배치 객체 없음"


def _relationship_lock_lines(layout: list, room_width: float, room_height: float) -> str:
    """각 객체에서 가장 가까운 다른 객체와의 상대관계를 고정합니다."""
    rw = max(float(room_width or 1), 1.0)
    rh = max(float(room_height or 1), 1.0)
    objs=[]
    for item in layout or []:
        if not isinstance(item, dict):
            continue
        try:
            x=float(item.get('x',0)); y=float(item.get('y',0)); w=max(float(item.get('width',0)),1); h=max(float(item.get('height',0)),1)
        except (TypeError,ValueError):
            continue
        objs.append((str(item.get('name') or '가구').strip() or '가구', x+w/2, y+h/2))
    lines=[]
    for i,(name,cx,cy) in enumerate(objs[:16]):
        others=[(j,n,ox,oy,(ox-cx)**2+(oy-cy)**2) for j,(n,ox,oy) in enumerate(objs) if j!=i]
        if not others: continue
        _, near_name, ox, oy, _ = min(others, key=lambda t:t[4])
        dx=(ox-cx)/rw; dy=(oy-cy)/rh
        horiz = "오른쪽" if dx > 0.04 else "왼쪽" if dx < -0.04 else "거의 같은 세로선"
        vert = "아래쪽" if dy > 0.04 else "위쪽" if dy < -0.04 else "거의 같은 가로선"
        lines.append(f"- {name}의 가장 가까운 이웃은 {near_name}; {near_name}은 {name} 기준 {vert}, {horiz}에 있어야 합니다.")
    return "\n".join(lines) if lines else "- 별도 상대관계 없음"

def _recommendation_lines(recommendations: list) -> str:
    lines = []
    for rec in recommendations:
        if not isinstance(rec, dict):
            continue
        priority = rec.get("priority", "?")
        improvement = str(rec.get("improvement", "")).strip()
        if improvement:
            lines.append(f"{priority}. {improvement}")
    return "\n".join(lines) if lines else "- (제안된 개선사항 없음)"


def _issue_lines(issues: list[dict] | None) -> str:
    lines = []
    for idx, item in enumerate(issues or [], start=1):
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "사용자 확인").strip()
        label = str(item.get("label") or item.get("text") or "").strip()
        location = str(item.get("location") or "").strip()
        if not label:
            continue
        suffix = f"; 위치={location}" if location else ""
        lines.append(f"- {idx}. [{source}] {label}{suffix}")
    return "\n".join(lines) if lines else "- 사용자가 별도로 확인한 문제 없음"


def build_before_from_user_floorplan_prompt(
    room_label: str, room_width: int, room_height: int, layout: list, user_issues: list[dict] | None = None,
) -> str:
    """사용자 평면도를 절대 기준으로 현재 상태를 보수적으로 재현합니다."""
    return f"""Create ONE realistic high-oblique / bird's-eye visualization of the CURRENT {room_label} BEFORE any improvement.

ABSOLUTE PRIORITY: the supplied user floorplan image AND the coordinate locks below are the authoritative spatial source. Reconstruct THAT exact topology. Do not reinterpret it as a conventional room layout. If a normal interior-design convention conflicts with the supplied coordinates, FOLLOW THE SUPPLIED COORDINATES. Do not invent a more dramatic, cluttered, stylish, or generic room.

Room size: approximately {room_width}cm x {room_height}cm.
Objects explicitly present in the user's floorplan:
{_furniture_lines(layout, room_width, room_height)}

COORDINATE LOCKS — these are hard constraints, not suggestions:
{_spatial_lock_lines(layout, room_width, room_height)}

NEAREST-NEIGHBOR / TOPOLOGY LOCKS — preserve these relationships:
{_relationship_lock_lines(layout, room_width, room_height)}

User-confirmed current issues (secondary context only):
{_issue_lines(user_issues)}

STRICT SPATIAL PRESERVATION RULES:
1. Preserve room outline, wall boundaries, door/window locations when visible, furniture count, furniture identity, relative size, orientation, adjacency, and left/right/top/bottom relationships as closely as possible.
2. Do NOT add furniture, bags, clothes, boxes, trash, plants, decorations, appliances, storage, safety products, or other objects unless they are explicitly present in the floorplan or explicitly named by the user's issue text.
3. Do NOT remove or relocate existing furniture. This is BEFORE. Do not improve anything yet.
4. A generic issue such as 'floor obstruction', 'clutter', 'narrow path', or 'hard to access' is NOT permission to fill the room with invented clutter. If the exact obstructing object is not specified, preserve the existing layout and express the issue only through the existing geometry/clearance.
5. If an issue explicitly names an object, show only a small, plausible amount of that named object in the relevant area. Never exaggerate quantity.
6. Checklist and note text is contextual evidence, not a request to redesign or stage the room.
7. Preserve empty/open areas from the floorplan. Do not cover empty floor merely to make the problem look obvious.
8. Keep the whole room visible from a consistent high ceiling-corner / bird's-eye viewpoint. Spatial fidelity is more important than decoration or cinematic realism.
9. No people, text, captions, labels, arrows, numbered badges, highlighted circles, dashed paths, or watermark.
10. The final image must be immediately recognizable as the user's own floorplan translated into a realistic room.
11. HARD FAILURE CONDITIONS: do not move an island/center counter to a wall; do not merge two separate furniture items into one run; do not swap upper/lower or left/right positions; do not change a vertical object into a horizontal wall run; do not relocate the refrigerator, sink, bed, table, cabinet, door, or other named item to a different zone.
12. Treat x/y coordinates and relative topology as higher priority than realism, symmetry, standard kitchen/bedroom design conventions, or visual neatness.

Goal: faithful reconstruction first. Current-problem context is secondary and must never distort the user's layout.
""".strip()


def build_after_improved_floorplan_prompt(
    room_label: str, commands: list[dict], recommendations: list[dict] | None = None,
) -> str:
    command_lines = []
    for item in commands or []:
        if not isinstance(item, dict):
            continue
        priority = item.get("priority", "?")
        target = str(item.get("target") or "공간").strip()
        instruction = str(item.get("instruction") or "").strip()
        preserve = str(item.get("preserve") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if instruction:
            command_lines.append(f"{priority}. target={target}; action={instruction}; reason={reason}; preserve={preserve}")
    joined = "\n".join(command_lines) if command_lines else _recommendation_lines(recommendations or [])
    return f"""Edit the supplied BEFORE image into the IMPROVED AFTER image of the SAME {room_label}.

The BEFORE image is the authoritative visual base. This is a MINIMAL-CHANGE edit, not a redesign. Preserve the room identity and change ONLY what the explicit improvement commands require.

### Improvement commands
{joined}

STRICT MINIMAL-CHANGE RULES:
1. Preserve the same room geometry, wall/door/window positions, camera angle, perspective, lighting style, furniture identities, furniture design, and all unrelated furniture positions.
2. Apply every explicit command, but modify only the smallest relevant region needed to implement it.
3. Do NOT redecorate, restyle, clean, declutter, replace furniture, move furniture, add storage, add safety equipment, or change colors unless that exact change is requested by a command.
4. If a command removes a floor obstacle, remove only the relevant obstacle(s); do not automatically make the entire room pristine.
5. If a command changes storage accessibility, alter only the relevant storage/use position. Keep unrelated cabinets and furniture untouched.
6. If a command adds a safety product such as a handrail, add only that product in the requested location and preserve everything else.
7. If exact application is structurally impossible, make the smallest reasonable adaptation near the requested target rather than redesigning the room.
8. Maintain the same crop and high-oblique / bird's-eye viewpoint so BEFORE and AFTER can be compared directly.
9. Do NOT draw any text, Korean labels, numbers, badges, arrows, leader lines, circles, colored outlines, dashed walking paths, legends, or watermark. The web interface will add annotations separately.
10. The result must look like the SAME photograph/visualization with only the required safety changes applied.

Goal: BEFORE and AFTER should differ only at the explicit improvement points.
""".strip()

def generate_ai_before_from_floorplan_once(
    room_label: str, room_width: int, room_height: int, layout: list,
    reference_floorplan_png: bytes, user_issues: list[dict] | None = None, size: str = "1024x1024",
) -> bytes:
    if not reference_floorplan_png:
        raise RuntimeError("개선 전 이미지 생성을 위해 사용자 평면도 이미지가 필요합니다.")
    client = _build_client()
    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
    prompt = build_before_from_user_floorplan_prompt(room_label, room_width, room_height, layout, user_issues)
    ref = io.BytesIO(reference_floorplan_png)
    ref.name = "user_floorplan.png"
    before_fidelity = os.getenv("OPENAI_IMAGE_BEFORE_INPUT_FIDELITY", "high")
    before_quality = os.getenv("OPENAI_IMAGE_BEFORE_QUALITY", "medium")
    response = _call_edit_once(
        client, model=model, image_file=ref, prompt=prompt, size=size,
        fidelity_override=before_fidelity, quality_override=before_quality,
    )
    return _extract_image_bytes(response, "개선 전 이미지 생성")


def build_photoreal_reference_prompt(
    room_label: str, room_width: int, room_height: int, layout: list, recommendations: list,
) -> str:
    """정확한 배치 재현이 목적이 아니라, "이런 느낌일 수 있겠다"는 참고용 실사 이미지를
    위한 프롬프트입니다. 좌표를 픽셀 단위로 지키라고 강제하지 않습니다 - 어차피 이 이미지는
    참고용이라고 화면에 명시할 것이기 때문입니다."""
    circle_count = len([r for r in recommendations if isinstance(r, dict) and str(r.get("improvement", "")).strip()])

    return f"""Create one photorealistic bird's-eye / high oblique interior visualization of a {room_label} in an elderly
person's home, approximately {room_width}cm x {room_height}cm. Use realistic materials, textures,
and daylight, but choose a high corner or ceiling-level camera angle so the ENTIRE room layout is visible
in one frame, similar to an architectural cutaway photograph. The goal is not cinematic realism; the goal
is one-to-one visual mapping with the supplied measured floorplan. Preserve the same furniture count,
the same named furniture, the same left/right/top/bottom relationships, wall-side placement, relative
adjacency, door position, and open walking spaces. Do not invent or omit any major object. Do not move
furniture for composition. The resulting room must be recognizably the same floorplan when compared
side-by-side, even if this makes the image look more like a realistic architectural visualization than a
normal eye-level photograph.

Coordinate convention: x increases left-to-right, y increases top-to-bottom in the plan.
The room contains these exact placed items:
{_furniture_lines(layout, room_width, room_height)}

Show the room AFTER these elderly fall-prevention safety improvements have been applied:
{_recommendation_lines(recommendations)}

Highlight callouts: draw EXACTLY {circle_count} red circle/oval outline(s) on the photo -
no more, no fewer - one for each numbered improvement listed above, and only for those.
Each circle must tightly frame the specific fixture or spot that improvement changes.
Attach a small, clear red circular number badge to each callout: improvement 1 gets badge
"1", improvement 2 gets badge "2", and so on in the exact same order as the list above.
Do not add any words or captions inside the image; only the numeric callout badges are
allowed.

Strict rules:
- do NOT circle any furniture, cabinet, fixture, or object that is not explicitly one of the
  {circle_count} improvement(s) listed above.
- preserve the listed furniture count and spatial relationships before adding the improvements.
- if an improvement affects a path or gap between two objects, circle that GAP/PATH rather than
  moving unrelated furniture to create a different room.
Before finishing, verify that there are exactly {circle_count} callouts and that each callout
number maps to the correspondingly numbered improvement.

Style: natural daylight, realistic textures, a safe and elderly-friendly looking home
interior, no words or captions anywhere in the image other than the numeric callout badges, no people, no watermarks.
""".strip()


def generate_photoreal_reference_image(
    room_label: str, room_width: int, room_height: int, layout: list, recommendations: list,
    size: str = "1024x1024", reference_floorplan_png: bytes | None = None,
    visual_commands: list[dict] | None = None,
) -> bytes:
    client = _build_client()
    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
    prompt = (
        build_after_visual_command_prompt(room_label, visual_commands or [])
        if visual_commands else build_photoreal_reference_prompt(room_label, room_width, room_height, layout, recommendations)
    )

    # 가능하면 실제 2D 평면도 PNG를 이미지 입력으로 함께 전달합니다. 텍스트 좌표만 넣는 것보다
    # 가구 개수와 상대 배치를 보존하는 데 유리합니다. 설치된 OpenAI SDK/모델이 edit 입력을
    # 지원하지 않는 환경에서는 자동으로 기존 text-to-image 방식으로 fallback 합니다.
    if reference_floorplan_png:
        ref = io.BytesIO(reference_floorplan_png)
        ref.name = "floorplan_reference.png"
        response = _call_edit_once(
            client,
            model=model,
            image_file=ref,
            prompt=(
                prompt
                + "\n\nThe supplied image is the authoritative measured floorplan reference. "
                  "Preserve its furniture count, adjacency, wall-side placement, and open paths as closely as possible. "
                  "Transform the diagram into a realistic interior visualization rather than redesigning the room."
            ),
            size=size,
        )
        return _extract_image_bytes(response, "참조 평면도 기반 생성")

    response = client.images.generate(
        model=model, prompt=prompt, size=size, n=1,
        quality=os.getenv("OPENAI_IMAGE_QUALITY", "low"),
    )
    return _extract_image_bytes(response, "생성")




VISUAL_COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "commands": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "priority": {"type": "integer"},
                    "target": {"type": "string"},
                    "instruction": {"type": "string"},
                    "preserve": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["priority", "target", "instruction", "preserve", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["commands"],
    "additionalProperties": False,
}


def generate_concrete_visual_commands(
    room_label: str, room_width: int, room_height: int, layout: list, selected_items: list[dict],
) -> list[dict]:
    """Use a fast text model first to turn abstract safety advice into concrete visual-edit commands.

    The image model should not decide *how* to realize phrases such as '동선 확보' or
    '가구 배치 수정'. This step resolves those phrases into explicit target/action instructions.
    If the text model fails, a deterministic fallback built from the existing action-plan steps is used.
    """
    fallback: list[dict] = []
    for index, item in enumerate(selected_items, start=1):
        priority = int(item.get("priority") or index)
        target = str(item.get("target_object") or item.get("target_label") or "공간").strip() or "공간"
        improvement = str(item.get("improvement") or "").strip()
        steps = [str(s).strip() for s in (item.get("execution_steps") or []) if str(s).strip()]
        concrete = " ".join(steps[:3]).strip() or improvement
        fallback.append({
            "priority": priority,
            "target": target,
            "instruction": concrete,
            "preserve": "이 조치와 관계없는 가구·문·벽·통로는 변경하지 않음",
            "reason": str(item.get("floorplan_problem") or improvement or "안전 위험을 줄이기 위해").strip(),
        })

    if not selected_items:
        return fallback

    try:
        from openai import OpenAI
        if not os.getenv("OPENAI_API_KEY"):
            return fallback
        client = OpenAI(
            timeout=float(os.getenv("OPENAI_VISUAL_COMMAND_TIMEOUT_SECONDS", "10")),
            max_retries=0,
        )
        model = os.getenv("OPENAI_MODEL", "gpt-5.5")
        payload = {
            "room": room_label,
            "room_size_cm": {"width": room_width, "height": room_height},
            "layout": [
                {
                    "name": str(obj.get("name") or "가구"),
                    "x": obj.get("x"), "y": obj.get("y"),
                    "width": obj.get("width"), "height": obj.get("height"),
                    "rotation": obj.get("rotation", 0),
                }
                for obj in layout if isinstance(obj, dict)
            ],
            "selected_improvements": selected_items,
        }
        instructions = (
            "당신은 이미지 생성 전에 공간개선 명령을 구체화하는 백엔드 AI입니다. "
            "선택된 개선안을 새로 제안하지 말고, 각 항목을 이미지 모델이 그대로 실행할 수 있는 구체적인 시각 수정 명령으로 바꾸세요. "
            "특히 '동선 확보', '가구 배치 수정', '정리' 같은 추상 표현은 어떤 대상의 어느 쪽을 비우거나 옮기고 무엇을 유지해야 하는지 명확히 쓰세요. "
            "입력에 없는 정확한 cm 수치, 새 가구, 문/창문 위치는 만들지 마세요. 이미 execution_steps에 위치나 행동이 있으면 우선 사용하세요. "
            "가구를 옮겨야 하는 경우에는 현재 layout의 상대 위치를 참고해 '침대 오른쪽 통로를 비움', '협탁을 침대 머리맡 쪽으로 옮김'처럼 상대 위치로 표현하세요. "
            "각 명령은 한두 문장으로 짧게 쓰고, 관련 없는 물체는 유지한다는 preserve 지침도 포함하세요. "
            "reason에는 사용자가 확인한 현재 문제와 이 변경이 필요한 이유를 짧게 쓰세요."
        )
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=json.dumps(payload, ensure_ascii=False),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "visual_change_commands",
                    "strict": True,
                    "schema": VISUAL_COMMAND_SCHEMA,
                }
            },
        )
        output_text = getattr(response, "output_text", None)
        if not output_text:
            return fallback
        parsed = json.loads(output_text)
        commands = [c for c in parsed.get("commands", []) if isinstance(c, dict) and str(c.get("instruction") or "").strip()]
        return commands or fallback
    except Exception:
        return fallback


def build_after_visual_command_prompt(room_label: str, commands: list[dict]) -> str:
    return build_after_improved_floorplan_prompt(room_label, commands, [])

def _call_edit_once(client, *, model: str, image_file, prompt: str, size: str, fidelity_override: str | None = None, quality_override: str | None = None):
    """이미지 edit 네트워크 호출은 최대 한 번만 수행합니다.

    속도 우선 UI이므로 기본값은 quality=low, input_fidelity=low 입니다.
    구버전 SDK가 새 인자를 모르는 경우에만 HTTP 요청 전 TypeError를 받아
    호환 가능한 최소 인자로 재구성합니다. 네트워크 오류 자체는 재시도하지 않습니다.

    "1시간이 지나도 안 된다" 같은 문제를 다음에 또 겪지 않도록, 실제 네트워크 호출의
    시작·종료·소요시간·실패 여부를 서버 콘솔에 항상 남깁니다. 타임아웃(client의
    timeout 설정)이 실제로 걸리고 있는지, 아니면 그보다 훨씬 전에/후에 뭔가 막히고
    있는지를 로그만 보고 바로 판단할 수 있게 하기 위함입니다.
    """
    quality = (quality_override or os.getenv("OPENAI_IMAGE_QUALITY", "low")).strip().lower() or "low"
    if quality not in {"low", "medium", "high", "auto"}:
        quality = "low"
    fidelity = (fidelity_override or os.getenv("OPENAI_IMAGE_INPUT_FIDELITY", "low")).strip().lower() or "low"
    if fidelity not in {"low", "high"}:
        fidelity = "low"

    kwargs = {
        "model": model,
        "image": image_file,
        "prompt": prompt,
        "size": size,
        "n": 1,
        "quality": quality,
        "input_fidelity": fidelity,
        "output_format": "png",
    }
    started = time.monotonic()
    logger.info("[image_gen] images.edit 호출 시작 (model=%s, size=%s, timeout=%ss)", model, size, client.timeout)
    try:
        result = client.images.edit(**kwargs)
        logger.info("[image_gen] images.edit 성공 (%.1f초 소요)", time.monotonic() - started)
        return result
    except TypeError:
        # SDK 스키마 호환성 문제는 HTTP 요청 전에 발생합니다. 이때만 최소 인자로 다시 구성합니다.
        logger.warning("[image_gen] SDK가 quality/input_fidelity를 지원하지 않아 최소 인자로 재시도합니다.")
        try:
            image_file.seek(0)
        except Exception:
            pass
        return client.images.edit(
            model=model, image=image_file, prompt=prompt, size=size, n=1
        )


def generate_ai_after_from_floorplan_once(
    room_label: str, room_width: int, room_height: int, layout: list, recommendations: list,
    size: str = "1024x1024", reference_floorplan_png: bytes | None = None,
    visual_commands: list[dict] | None = None,
) -> bytes:
    """Generate the AFTER image by editing the already-generated BEFORE image in one image API call.

    The caller should pass the BEFORE image bytes as reference_floorplan_png so the same room/camera/layout
    are preserved while the concrete improvements are applied.
    """
    if not reference_floorplan_png:
        raise RuntimeError("개선 후 이미지 생성을 위해 기준 평면도 이미지가 필요합니다.")
    client = _build_client()
    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1.5")
    # 중요: 1단계 텍스트 AI가 확정한 짧은 명령을 실제 이미지 프롬프트에 사용합니다.
    # visual_commands가 없을 때만 기존 recommendation 프롬프트로 fallback 합니다.
    prompt = (
        build_after_visual_command_prompt(room_label, visual_commands or [])
        if visual_commands
        else build_after_improved_floorplan_prompt(room_label, [], recommendations)
    )
    ref = io.BytesIO(reference_floorplan_png)
    ref.name = "measured_floorplan.png"
    try:
        response = _call_edit_once(
            client,
            model=model,
            image_file=ref,
            prompt=(
                prompt
                + "\n\nAFTER image only. Use the supplied BEFORE image as the exact visual reference. "
                  "Apply only the explicit change commands and keep unrelated objects unchanged. "
                  "Do not add any text, labels, numbers, arrows, circles, paths, or annotations inside the image."
            ),
            size=size,
            fidelity_override=os.getenv("OPENAI_IMAGE_AFTER_INPUT_FIDELITY", "high"),
        )
    except Exception as exc:
        # 여기서 예외의 실제 타입(APITimeoutError/APIConnectionError/기타)을 서버 콘솔에
        # 남겨야, "정말 타임아웃인지" "네트워크 자체가 안 되는지"를 로그만 보고 구분할 수
        # 있습니다. 이 예외는 그대로 다시 던져서 /api/generate_ai_before_after가
        # 502로 사용자에게 명확히 전달하게 합니다.
        logger.error("[image_gen] AFTER 이미지 생성 실패: %s: %s", type(exc).__name__, exc)
        raise
    return _extract_image_bytes(response, "개선 후 이미지 생성")

def _build_client():
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("openai 패키지를 불러오지 못했습니다. requirements.txt 설치가 필요합니다.") from exc

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 설정되지 않아 이미지를 생성할 수 없습니다.")

    configured_timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90"))
    timeout_seconds = min(configured_timeout, float(os.getenv("OPENAI_IMAGE_MAX_SECONDS", "55")))
    # OpenAI SDK 기본 max_retries=2는, 이 파일에 이미 있는 코드 레벨 fallback(edit 실패 시
    # 다른 방식으로 재시도)과 겹치면 "60초 타임아웃 x SDK 재시도 x 코드 fallback 단계"로
    # 곱연산이 되어 최악의 경우 요청 하나가 수십 분까지 걸릴 수 있습니다(실제로 사용자가
    # 1시간 넘게 로딩만 보는 문제로 이어졌습니다). SDK 자체 재시도는 꺼서, 시간 상한을
    # "60초 x 코드 fallback 단계 수"로만 예측 가능하게 만듭니다.
    return OpenAI(timeout=timeout_seconds, max_retries=0)


def _extract_image_bytes(response, action_label: str) -> bytes:
    if not response.data:
        raise RuntimeError(f"이미지 {action_label} 응답이 비어 있습니다.")
    b64 = getattr(response.data[0], "b64_json", None)
    if not b64:
        raise RuntimeError(f"이미지 {action_label} 응답에 이미지 데이터(b64_json)가 없습니다.")
    return base64.b64decode(b64)


def save_generated_image(image_bytes: bytes, prefix: str) -> str:
    """static/generated/ 아래에 저장하고, static 폴더 기준 상대 경로를 반환합니다."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
    (GENERATED_DIR / filename).write_bytes(image_bytes)
    return f"generated/{filename}"

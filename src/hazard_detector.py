from __future__ import annotations

from typing import Any

OBJECT_NAME_RULES = {
    "TOILET": ("변기", "양변기"),
    "CHAIR": ("의자", "소파", "좌석"),
    "BED": ("침대",),
    "CABINET": ("수납장", "선반", "캐비닛", "장롱", "서랍장"),
    "RUG": ("러그", "카펫", "매트"),
    "THRESHOLD": ("문턱", "단차"),
    "ALERT_DEVICE": ("비상벨", "경보기", "알람", "센서등"),
    "LIGHT": ("조명", "전등", "스탠드"),
}


def _anchor_for(item: dict) -> dict:
    for key, geometry_type in (("bbox", "bbox"), ("polygon", "polygon"), ("polyline", "polyline")):
        if key in item:
            return {"type": geometry_type, "coordinates": item[key]}
    if all(key in item for key in ("x", "y", "width", "height")):
        return {
            "type": "bbox",
            "coordinates": [item["x"], item["y"], item["width"], item["height"]],
        }
    return {"type": "none", "coordinates": []}


def _infer_object_type(name: str) -> str:
    for object_type, keywords in OBJECT_NAME_RULES.items():
        if any(keyword in name for keyword in keywords):
            return object_type
    return "FURNITURE"


def normalize_layout_plan(plan: dict) -> dict:
    """
    두 형식을 지원한다.

    A. 권장 구조: rooms / objects / paths / observations
    B. layout_editor 호환 구조:
       {"room_id": ..., "room_type": ..., "layout": [{name,x,y,width,height,attributes?}, ...]}

    B 형식의 x/y/width/height는 화면 도형 크기일 뿐 실제 가구 높이가 아니다.
    따라서 hazard 판정은 attributes가 있을 때만 수행한다.
    """
    if "rooms" in plan or "objects" in plan:
        normalized = dict(plan)
        normalized.setdefault("rooms", [])
        normalized.setdefault("objects", [])
        normalized.setdefault("paths", [])
        normalized.setdefault("observations", [])
        return normalized

    layout = plan.get("layout") or plan.get("current_room_layout") or []
    room_id = str(plan.get("room_id", "room_01"))
    room_type = str(plan.get("room_type", "UNKNOWN")).upper()
    room = {
        "room_id": room_id,
        "room_type": room_type,
        "polygon": plan.get("polygon", []),
        "attributes": plan.get("room_attributes", {}),
    }
    objects = []
    for index, item in enumerate(layout, start=1):
        name = str(item.get("name", f"object_{index}"))
        object_type = str(item.get("object_type") or _infer_object_type(name)).upper()
        obj = {
            "object_id": str(item.get("object_id", f"layout_object_{index:03d}")),
            "object_type": object_type,
            "name": name,
            "room_id": room_id,
            "bbox": item.get("bbox")
            or [item.get("x", 0), item.get("y", 0), item.get("width", 0), item.get("height", 0)],
            "attributes": item.get("attributes", {}),
        }
        objects.append(obj)
    return {
        "plan_id": plan.get("plan_id", "LAYOUT_EDITOR_PLAN"),
        "canvas": plan.get("canvas", {"width": 1000, "height": 700}),
        "rooms": [room],
        "objects": objects,
        "paths": plan.get("paths", []),
        "observations": plan.get("observations", []),
    }


def detect_hazards(plan: dict, selected_room: str | None = None) -> dict:
    """
    hazard를 임의 추정하지 않고 semantic attributes 또는 explicit observation으로만 판정한다.

    상태
    ----
    present: 실제 위험 존재 확인
    absent: 해당 속성이 안전한 상태로 확인
    unknown_schema_limited: 객체는 있으나 판정 속성이 없음
    not_applicable: 대상 객체가 없음
    """
    plan = normalize_layout_plan(plan)
    rooms = {room["room_id"]: room for room in plan.get("rooms", [])}
    objects = {obj["object_id"]: obj for obj in plan.get("objects", [])}
    paths = {path["path_id"]: path for path in plan.get("paths", [])}
    assessments: list[dict[str, Any]] = []
    assessment_index: dict[tuple[str, str | None, str | None], int] = {}

    def record(
        code: str,
        source: dict,
        status: str,
        evidence: list[str],
        confidence: float = 1.0,
        room_id: str | None = None,
        object_id: str | None = None,
    ) -> None:
        rid = room_id or source.get("room_id")
        room_type = rooms.get(rid, {}).get("room_type", "UNKNOWN")
        oid = object_id or source.get("object_id") or source.get("path_id") or source.get("room_id")
        if selected_room and selected_room not in {rid, room_type}:
            return
        key = (code, rid, oid)
        payload = {
            "hazard_code": code,
            "room_id": rid,
            "room_type": room_type,
            "object_id": oid,
            "object_type": source.get("object_type", "PATH" if "path_id" in source else "ROOM"),
            "status": status,
            "present": status == "present",
            "presence_score": 1.0 if status == "present" else 0.0,
            "confidence": float(confidence if status == "present" else 0.0),
            "anchor_geometry": _anchor_for(source),
            "detected_from": [e for e in evidence if e],
        }
        if key in assessment_index:
            # explicit observation이 나중/먼저 오더라도 present/absent 판정을 unknown보다 우선한다.
            old_idx = assessment_index[key]
            old = assessments[old_idx]
            priority = {"unknown_schema_limited": 0, "not_applicable": 0, "absent": 1, "present": 2}
            if priority.get(status, 0) >= priority.get(old["status"], 0):
                assessments[old_idx] = payload
            return
        assessment_index[key] = len(assessments)
        assessments.append(payload)

    # 1. explicit observation: 체크리스트·사진분석·센서 결과
    for obs in plan.get("observations", []):
        code = obs.get("hazard_code") or obs.get("observation_code")
        if not code:
            continue
        value = obs.get("present", obs.get("value"))
        if value is True:
            status = "present"
        elif value is False:
            status = "absent"
        else:
            status = "unknown_schema_limited"
        source = (
            objects.get(obs.get("object_id"))
            or paths.get(obs.get("path_id"))
            or rooms.get(obs.get("room_id"))
            or obs
        )
        record(
            code,
            source,
            status,
            [f"explicit_observation:{obs.get('observed_by', 'UNKNOWN')}", str(obs.get("note", ""))],
            float(obs.get("confidence", 1.0)),
            room_id=obs.get("room_id"),
            object_id=obs.get("object_id") or obs.get("path_id"),
        )

    # 2. semantic object attributes
    for obj in plan.get("objects", []):
        typ = str(obj.get("object_type", "")).upper()
        attr = obj.get("attributes") or {}

        if typ == "TOILET":
            if "grab_bar_present" in attr:
                record("TOILET_NO_GRAB_BAR", obj, "absent" if attr["grab_bar_present"] else "present", [f"grab_bar_present={attr['grab_bar_present']}"])
            elif "grab_bar_left" in attr or "grab_bar_right" in attr:
                left, right = attr.get("grab_bar_left"), attr.get("grab_bar_right")
                if left is True or right is True:
                    record("TOILET_NO_GRAB_BAR", obj, "absent", [f"grab_bar_left={left}", f"grab_bar_right={right}"])
                elif left is False and right is False:
                    record("TOILET_NO_GRAB_BAR", obj, "present", ["grab_bar_left=false", "grab_bar_right=false"])
                else:
                    record("TOILET_NO_GRAB_BAR", obj, "unknown_schema_limited", ["grab bar side attributes incomplete"])
            else:
                record("TOILET_NO_GRAB_BAR", obj, "unknown_schema_limited", ["required: grab_bar_present or grab_bar_left/right"])

            height_class = str(attr.get("seat_height_class", "")).lower()
            if attr.get("low_toilet") is True or height_class in {"low", "낮음"}:
                record("LOW_TOILET", obj, "present", [f"seat_height_class={height_class or 'legacy_low_toilet=true'}"])
            elif height_class in {"normal", "high", "보통", "높음"} or attr.get("low_toilet") is False:
                record("LOW_TOILET", obj, "absent", [f"seat_height_class={height_class or 'legacy_low_toilet=false'}"])
            else:
                record("LOW_TOILET", obj, "unknown_schema_limited", ["required: seat_height_class(low/normal/high)"])

        elif typ == "CHAIR":
            height_class = str(attr.get("seat_height_class", "")).lower()
            if attr.get("low_seat") is True or height_class in {"low", "낮음"}:
                record("LOW_SEAT_HEIGHT", obj, "present", [f"seat_height_class={height_class or 'legacy_low_seat=true'}"])
            elif height_class in {"normal", "high", "보통", "높음"} or attr.get("low_seat") is False:
                record("LOW_SEAT_HEIGHT", obj, "absent", [f"seat_height_class={height_class or 'legacy_low_seat=false'}"])
            else:
                record("LOW_SEAT_HEIGHT", obj, "unknown_schema_limited", ["required: seat_height_class"])

        elif typ == "BED":
            height_class = str(attr.get("bed_height_class", "")).lower()
            if attr.get("low_bed") is True or height_class in {"low", "낮음"}:
                record("LOW_BED_HEIGHT", obj, "present", [f"bed_height_class={height_class or 'legacy_low_bed=true'}"])
            elif height_class in {"normal", "high", "보통", "높음"} or attr.get("low_bed") is False:
                record("LOW_BED_HEIGHT", obj, "absent", [f"bed_height_class={height_class or 'legacy_low_bed=false'}"])
            else:
                record("LOW_BED_HEIGHT", obj, "unknown_schema_limited", ["required: bed_height_class"])

        elif typ == "CABINET":
            storage_class = str(attr.get("storage_height_class", "")).lower()
            if attr.get("low_storage") is True or storage_class in {"low", "낮음"}:
                record("LOW_STORAGE", obj, "present", [f"storage_height_class={storage_class or 'legacy_low_storage=true'}"])
            elif storage_class in {"normal", "high", "보통", "높음"} or attr.get("low_storage") is False:
                record("LOW_STORAGE", obj, "absent", [f"storage_height_class={storage_class or 'legacy_low_storage=false'}"])
            else:
                record("LOW_STORAGE", obj, "unknown_schema_limited", ["required: storage_height_class"])

            if attr.get("high_storage") is True or storage_class in {"high", "높음"}:
                record("HIGH_STORAGE", obj, "present", [f"storage_height_class={storage_class or 'legacy_high_storage=true'}"])
            elif storage_class in {"normal", "low", "보통", "낮음"} or attr.get("high_storage") is False:
                record("HIGH_STORAGE", obj, "absent", [f"storage_height_class={storage_class or 'legacy_high_storage=false'}"])
            else:
                record("HIGH_STORAGE", obj, "unknown_schema_limited", ["required: storage_height_class"])

        elif typ == "RUG":
            if attr.get("fixed") is False or attr.get("loose") is True:
                record("LOOSE_RUG", obj, "present", [f"fixed={attr.get('fixed')}", f"loose={attr.get('loose')}"])
            elif attr.get("fixed") is True or attr.get("loose") is False:
                record("LOOSE_RUG", obj, "absent", [f"fixed={attr.get('fixed')}", f"loose={attr.get('loose')}"])
            else:
                record("LOOSE_RUG", obj, "unknown_schema_limited", ["required: fixed or loose"])

        elif typ == "OBJECT":
            if attr.get("floor_level") is True:
                record("FLOOR_LEVEL_OBJECT", obj, "present", ["floor_level=true"])
            elif attr.get("floor_level") is False:
                record("FLOOR_LEVEL_OBJECT", obj, "absent", ["floor_level=false"])
            else:
                record("FLOOR_LEVEL_OBJECT", obj, "unknown_schema_limited", ["required: floor_level"])

        elif typ == "THRESHOLD":
            height_class = str(attr.get("height_class", "")).lower()
            if attr.get("high_threshold") is True or height_class in {"high", "높음"}:
                record("THRESHOLD_HIGH", obj, "present", [f"height_class={height_class or 'legacy_high_threshold=true'}"])
            elif attr.get("high_threshold") is False or height_class in {"low", "normal", "낮음", "보통"}:
                record("THRESHOLD_HIGH", obj, "absent", [f"height_class={height_class or 'legacy_high_threshold=false'}"])
            else:
                record("THRESHOLD_HIGH", obj, "unknown_schema_limited", ["required: height_class"])

            contrast = str(attr.get("contrast", "")).lower()
            if contrast in {"low", "poor", "낮음"}:
                record("THRESHOLD_LOW_CONTRAST", obj, "present", [f"contrast={contrast}"])
            elif contrast in {"normal", "high", "good", "보통", "높음"}:
                record("THRESHOLD_LOW_CONTRAST", obj, "absent", [f"contrast={contrast}"])
            else:
                record("THRESHOLD_LOW_CONTRAST", obj, "unknown_schema_limited", ["required: contrast"])

        elif typ == "FURNITURE":
            edge = str(attr.get("edge_profile", "")).lower()
            if edge in {"sharp", "hard_sharp", "날카로움"}:
                record("HARD_SHARP_FURNITURE_EDGE", obj, "present", [f"edge_profile={edge}"])
            elif edge in {"rounded", "soft", "둥근", "부드러움"}:
                record("HARD_SHARP_FURNITURE_EDGE", obj, "absent", [f"edge_profile={edge}"])
            else:
                record("HARD_SHARP_FURNITURE_EDGE", obj, "unknown_schema_limited", ["required: edge_profile"])

        elif typ == "ALERT_DEVICE":
            if attr.get("visual_alert") is False:
                record("NO_VISUAL_ALERT", obj, "present", ["visual_alert=false"])
            elif attr.get("visual_alert") is True:
                record("NO_VISUAL_ALERT", obj, "absent", ["visual_alert=true"])
            else:
                record("NO_VISUAL_ALERT", obj, "unknown_schema_limited", ["required: visual_alert"])

    # 3. path attributes
    for path in plan.get("paths", []):
        attr = path.get("attributes") or {}
        if path.get("obstacle_ids") or attr.get("obstacle_present") is True:
            record("PATH_OBSTACLE", path, "present", [f"obstacle_ids={path.get('obstacle_ids', [])}"])
        elif attr.get("obstacle_present") is False:
            record("PATH_OBSTACLE", path, "absent", ["obstacle_present=false"])
        else:
            record("PATH_OBSTACLE", path, "unknown_schema_limited", ["required: obstacle_present or obstacle_ids"])

        for code, attr_name in (("NARROW_PATH", "narrow"), ("UNCLEAR_ROUTE", "unclear_route")):
            if attr.get(attr_name) is True:
                record(code, path, "present", [f"{attr_name}=true"])
            elif attr.get(attr_name) is False:
                record(code, path, "absent", [f"{attr_name}=false"])
            else:
                record(code, path, "unknown_schema_limited", [f"required: {attr_name}"])

    # 4. room attributes. 조명 객체 유무로 조도를 추정하지 않는다.
    for room in plan.get("rooms", []):
        attr = room.get("attributes") or {}
        lighting = str(attr.get("lighting_level", "")).lower()
        if attr.get("low_lighting") is True or lighting in {"low", "poor", "낮음"}:
            record("LOW_LIGHTING", room, "present", [f"lighting_level={lighting or 'legacy_low_lighting=true'}"])
        elif attr.get("low_lighting") is False or lighting in {"normal", "high", "good", "보통", "높음"}:
            record("LOW_LIGHTING", room, "absent", [f"lighting_level={lighting or 'legacy_low_lighting=false'}"])
        else:
            record("LOW_LIGHTING", room, "unknown_schema_limited", ["required: lighting_level or explicit observation"])

        if attr.get("night_light") is False:
            record("NO_NIGHT_LIGHT", room, "present", ["night_light=false"])
        elif attr.get("night_light") is True:
            record("NO_NIGHT_LIGHT", room, "absent", ["night_light=true"])
        else:
            record("NO_NIGHT_LIGHT", room, "unknown_schema_limited", ["required: night_light"])

    present = []
    unknown = []
    absent = []
    for item in assessments:
        if item["status"] == "present":
            present.append({**item, "hazard_instance_id": f"HZ_{len(present) + 1:03d}"})
        elif item["status"] == "unknown_schema_limited":
            unknown.append(item)
        elif item["status"] == "absent":
            absent.append(item)

    return {
        "plan_id": plan.get("plan_id"),
        "selected_room": selected_room,
        "detection_policy": "EXPLICIT_SEMANTIC_ATTRIBUTES_AND_OBSERVATIONS_ONLY",
        "hazard_count": len(present),
        "unknown_count": len(unknown),
        "absent_count": len(absent),
        "hazard_instances": present,
        "unknown_schema_limited": unknown,
        "confirmed_absent": absent,
        "normalized_floorplan": plan,
    }

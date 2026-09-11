"""
app.py (Flask 버전)
===================
기존 Streamlit UI(app.py + app_pages/*)를 순수 HTML/CSS/JS + Flask API로 교체한 버전입니다.

- 백엔드 로직(ML 예측, SHAP, RAG, GPT 플래너, 공간위험 판정)은 전혀 수정하지 않고
  기존 integration/frontend_backend_adapter.py, src/integration_pipeline.py,
  src/adaptive_space_questions.py 를 그대로 호출합니다.
- session_state 대신 서버 메모리의 SESSIONS 딕셔너리 + 서명된 쿠키(sid)로
  사용자별 상태를 유지합니다. (다중 프로세스로 스케일할 경우 Redis 등으로 교체 필요)
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for

load_dotenv()

ROOT = Path(__file__).resolve().parent

from integration.frontend_backend_adapter import (
    backend_result_to_streamlit_result,
    build_floorplan_from_streamlit,
    frontend_model_input_to_backend_features,
    predict_from_frontend_model_input,
)
from src.adaptive_space_questions import ROOM_LABELS, select_adaptive_space_questions
from src.room_priority import compute_room_priorities
from src.floorplan_image_gen import save_generated_image, generate_photoreal_reference_image, generate_ai_before_from_floorplan_once, generate_ai_after_from_floorplan_once, generate_concrete_visual_commands
from src.deterministic_floorplan import (
    pick_canvas_size, pick_iso_canvas_size, render_floorplan_png, render_floorplan_with_markers, render_isometric_scene,
)
from src.action_plan_generator import (
    generate_action_plan_for_hazard, generate_fallback_action_plan_for_hazard,
    generate_ai_only_recommendations, build_shopping_links,
)
from src.report_generator import build_report_pdf
from src.integration_pipeline import run_clarification_followup, run_full_analysis
from src.clarification_feedback import merge_clarification_answers
from auth import create_user, verify_user

import web_helper as wh

# AI 이미지 생성이 "1시간 지나도 안 끝난다"처럼 원인을 알 수 없는 상태에 빠지는 걸
# 막기 위해, 각 단계(특히 OpenAI images.edit 호출)의 시작·종료·소요시간·실패 원인을
# 서버 콘솔에 항상 남깁니다. 실제로 55초 타임아웃이 걸리고 있는지, 아니면 그보다 훨씬
# 전에/후에 다른 지점에서 멈추는지를 로그만 보고 바로 구분할 수 있게 하기 위함입니다.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "an-neomeojib-dev-secret")

# 서버 메모리 세션 저장소. { sid: {model_input, prediction, ...} }
SESSIONS: dict[str, dict] = {}

GENERATED_DIR = ROOT / "static" / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def _ai_before_cache_key(room_label: str, room_width: int, room_height: int, layout: list) -> str:
    """Stable key for the physical room/layout. Improvement choices are deliberately excluded."""
    payload = {
        "room_label": room_label,
        "room_width": int(room_width),
        "room_height": int(room_height),
        "layout": layout,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cached_before_path(state: dict, room_label: str, room_width: int, room_height: int, layout: list) -> str | None:
    cache = state.get("ai_before_cache") or {}
    expected = _ai_before_cache_key(room_label, room_width, room_height, layout)
    rel = cache.get("path") if cache.get("key") == expected else None
    if not rel:
        return None
    full = ROOT / "static" / rel
    return rel if full.exists() else None


def get_sid() -> str:
    sid = session.get("sid")
    if not sid or sid not in SESSIONS:
        sid = uuid.uuid4().hex
        session["sid"] = sid
        SESSIONS[sid] = {}
    return sid


def get_state() -> dict:
    return SESSIONS[get_sid()]


# ----------------------------------------------------------------------------
# 진행 단계 사이드바 컨텍스트
# ----------------------------------------------------------------------------
def progress_context(current_page: str) -> dict:
    progress_page = "room_detail" if current_page == "layout_editor" else current_page
    idx = wh.PAGE_ORDER.index(progress_page) if progress_page in wh.PAGE_ORDER else 0
    steps = []
    for i, key in enumerate(wh.PAGE_ORDER):
        status = "done" if i < idx else "active" if i == idx else "todo"
        steps.append({"label": wh.PAGE_LABELS[key], "status": status, "num": i + 1})
    return {"steps": steps}


@app.context_processor
def inject_globals():
    return {"page": request.endpoint or "", "current_user": get_state().get("user")}


# ----------------------------------------------------------------------------
# 인증 (회원가입 / 로그인 / 로그아웃)
# ----------------------------------------------------------------------------
@app.route("/login")
def login_page():
    if get_state().get("user"):
        return redirect(url_for("home"))
    return render_template("login.html")


@app.route("/signup")
def signup_page():
    if get_state().get("user"):
        return redirect(url_for("home"))
    return render_template("signup.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    payload = request.get_json(force=True)
    ok, message, user = verify_user(payload.get("email", ""), payload.get("password", ""))
    if not ok:
        return jsonify({"error": message}), 400
    get_state()["user"] = user
    return jsonify({"ok": True, "user": user})


@app.route("/api/signup", methods=["POST"])
def api_signup():
    payload = request.get_json(force=True)
    ok, message, user = create_user(
        payload.get("email", ""), payload.get("password", ""), payload.get("name", ""),
    )
    if not ok:
        return jsonify({"error": message}), 400
    # 가입과 동시에 로그인 처리
    get_state()["user"] = user
    return jsonify({"ok": True, "user": user})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    get_state().pop("user", None)
    return jsonify({"ok": True})


# ----------------------------------------------------------------------------
# 페이지 라우트
# ----------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template(
        "home.html",
        progress=progress_context("home"),
        show_login_required=bool(request.args.get("login_required")),
    )


@app.route("/input")
def input_page():
    if not get_state().get("user"):
        return redirect(url_for("home", login_required=1))
    state = get_state()
    saved = state.get("model_input", {})
    return render_template(
        "input.html",
        progress=progress_context("input"),
        saved=saved,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    state = get_state()
    payload = request.get_json(force=True)
    model_input = payload.get("model_input")
    if not isinstance(model_input, dict):
        return jsonify({"error": "설문 입력 내용을 찾을 수 없습니다. 설문부터 다시 진행해 주세요."}), 400

    try:
        model_input = {k: int(v) for k, v in model_input.items()}
        prediction = predict_from_frontend_model_input(model_input)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"예측에 실패했습니다: {exc}"}), 400

    state["model_input"] = model_input
    state["prediction"] = prediction
    state["user_features"] = prediction["backend_features"]
    state["feature_mapping_report"] = prediction["mapping_report"]
    # 새 설문이 들어오면 이전 분석 결과는 초기화
    for key in ("ai_result", "backend_analysis_result", "floorplan_json", "selected_room",
                "backend_hazard_flags_by_room", "backend_hazard_context_by_room", "room_priority_result", "current_room_layout"):
        state.pop(key, None)

    return jsonify({"ok": True, "risk_score": prediction["risk_score"], "risk_level": prediction["risk_level"]})


@app.route("/result")
def result_page():
    state = get_state()
    prediction = state.get("prediction")
    if not prediction:
        return redirect(url_for("input_page"))

    risk_score = prediction["risk_score"]
    risk_level = prediction["risk_level"]
    risk_color = wh.RISK_COLOR_MAP.get(risk_level, wh.TEAL)
    threshold_percent = float(prediction.get("threshold_percent", 5.3))
    gauge_max = max(12.0, risk_score * 1.2)

    model_calibrated_percent = float(prediction.get("model_calibrated_probability", risk_score / 100.0)) * 100

    messages = {
        "모델 판정 기준 미만 - 관찰 권장":
            "현재 예측 위험도는 기준보다 낮습니다. 그래도 꾸준한 운동과 집 안 안전 점검을 권장드립니다.",
        "모델 판정 기준 초과 - 주의 및 점검 권장":
            "현재 예측 위험도가 기준 이상입니다. 낙상 예방을 위해 집 안의 위험한 부분을 먼저 살펴보는 것을 권장드립니다.",
    }

    # SHAP 결과를 matplotlib 이미지 대신 구조화된 데이터로 넘겨서,
    # 화면에서 레이더 차트(SVG)와 번호 매긴 카드 목록으로 직접 그립니다.
    shap_result = prediction.get("shap_result") or {}
    main_factors = wh.build_main_factor_cards(shap_result, top_n=12) if shap_result else []
    interaction_cards = wh.build_interaction_cards(shap_result, top_n=12) if shap_result else []

    model_input = state.get("model_input", {})

    return render_template(
        "result.html",
        progress=progress_context("result"),
        prediction=prediction,
        risk_score=risk_score,
        risk_level=risk_level,
        risk_color=risk_color,
        threshold_percent=threshold_percent,
        gauge_max=gauge_max,
        model_calibrated_percent=model_calibrated_percent,
        risk_message=messages.get(risk_level, ""),
        main_factors=main_factors,
        interaction_cards=interaction_cards,
        model_input=model_input,
    )


@app.route("/space_check")
def space_check_page():
    state = get_state()
    prediction = state.get("prediction")
    if not prediction:
        return redirect(url_for("input_page"))

    model_input = state.get("model_input", {})
    saved_by_room = state.get("backend_hazard_flags_by_room", {})
    questions_by_room = {
        room: select_adaptive_space_questions(
            model_input,
            room,
            prediction=prediction if isinstance(prediction, dict) else None,
            max_questions=8,
        )
        for room in wh.ROOM_INFO
    }
    return render_template(
        "space_check.html",
        progress=progress_context("floorplan"),
        rooms=wh.ROOM_INFO,
        questions_by_room=questions_by_room,
        saved_by_room=saved_by_room,
    )


@app.route("/api/space_hazards", methods=["POST"])
def api_space_hazards():
    state = get_state()
    prediction = state.get("prediction")
    if not isinstance(prediction, dict):
        return jsonify({"error": "예측 결과가 없습니다. 설문부터 다시 진행해 주세요."}), 400

    payload = request.get_json(force=True)
    flags_by_room = payload.get("flags_by_room", {})
    context_by_room = payload.get("context_by_room", {})
    if not isinstance(flags_by_room, dict):
        return jsonify({"error": "공간 점검 내용을 읽지 못했습니다. 다시 선택해 주세요."}), 400

    cleaned_flags: dict[str, dict[str, bool]] = {}
    for room in wh.ROOM_INFO:
        raw_flags = flags_by_room.get(room, {})
        cleaned_flags[room] = {
            str(code): bool(value)
            for code, value in raw_flags.items()
        } if isinstance(raw_flags, dict) else {}

    shap_explanation = prediction.get("shap_result")
    if not isinstance(shap_explanation, dict):
        return jsonify({"error": "이전 단계의 낙상 위험 분석 결과가 없습니다. 건강·생활 설문부터 다시 진행해 주세요."}), 400

    try:
        priority_result = compute_room_priorities(
            shap_explanation=shap_explanation,
            flags_by_room=cleaned_flags,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"공간 점검 결과를 정리하지 못했습니다: {exc}"}), 500

    state["backend_hazard_flags_by_room"] = cleaned_flags
    state["backend_hazard_context_by_room"] = context_by_room if isinstance(context_by_room, dict) else {}
    state["room_priority_result"] = priority_result
    return jsonify({
        "ok": True,
        "ranked_rooms": priority_result.get("ranked_rooms", []),
        "ranked_hazard_count": priority_result.get("ranked_hazard_count", 0),
    })


@app.route("/floorplan")
def floorplan_page():
    state = get_state()
    if not state.get("prediction"):
        return redirect(url_for("input_page"))
    priority_result = state.get("room_priority_result")
    if not isinstance(priority_result, dict):
        return redirect(url_for("space_check_page"))

    ranked_rooms = []
    for item in priority_result.get("ranked_rooms", []):
        room = item.get("room")
        if room not in wh.ROOM_INFO:
            continue
        info = dict(wh.ROOM_INFO[room])
        info.update(item)
        info["key"] = room
        ranked_rooms.append(info)

    return render_template(
        "floorplan.html",
        progress=progress_context("floorplan"),
        ranked_rooms=ranked_rooms,
    )


@app.route("/api/select_room", methods=["POST"])
def api_select_room():
    state = get_state()
    room = request.get_json(force=True).get("room")
    if room not in wh.ROOM_INFO:
        return jsonify({"error": "알 수 없는 공간입니다."}), 400
    state["selected_room"] = room
    for key in ("current_room_layout", "ai_layout", "ai_result", "last_layout_request_id",
                "backend_hazard_flags", "backend_hazard_question_context", "comparison_images", "current_room_annotations", "current_room_note", "optimal_layout_result"):
        state.pop(key, None)
    return jsonify({"ok": True})


@app.route("/layout_editor")
def layout_editor_page():
    state = get_state()
    room = state.get("selected_room")
    if not room:
        return redirect(url_for("floorplan_page"))

    model_input = state.get("model_input", {})
    prediction = state.get("prediction", {})
    questions = select_adaptive_space_questions(
        model_input, room, prediction=prediction if isinstance(prediction, dict) else None, max_questions=8,
    )
    flags_by_room = state.get("backend_hazard_flags_by_room", {})
    saved_flags = flags_by_room.get(room, {})

    ai_result = state.get("ai_result")

    return render_template(
        "layout_editor.html",
        progress=progress_context("layout_editor"),
        room=room,
        room_info=wh.ROOM_INFO[room],
        room_label=ROOM_LABELS.get(room, room),
        questions=questions,
        saved_flags=saved_flags,
        ai_result=ai_result,
        saved_layout=state.get("current_room_layout", []),
        saved_room_width=state.get("ai_room_width", 720),
        saved_room_height=state.get("ai_room_height", 460),
        saved_room_note=state.get("current_room_note", ""),
        saved_point_notes=state.get("current_room_annotations", []),
    )


@app.route("/action_plan")
def action_plan_page():
    state = get_state()
    room = state.get("selected_room")
    if not room:
        return redirect(url_for("floorplan_page"))
    if not state.get("ai_result"):
        return redirect(url_for("layout_editor_page"))

    return render_template(
        "action_plan.html",
        progress=progress_context("room_detail"),
        room=room,
        room_label=ROOM_LABELS.get(room, room),
    )


def _run_layout_analysis_payload(state: dict, payload: dict):
    layout = payload.get("layout", [])
    room_width = float(payload.get("room_width", 720))
    room_height = float(payload.get("room_height", 460))
    hazard_flags = payload.get("hazard_flags", {})
    question_context = payload.get("question_context", [])
    room_note = str(payload.get("room_note") or "").strip()
    annotations = [item for item in (payload.get("point_notes") or []) if isinstance(item, dict)]

    if not layout:
        raise ValueError("가구를 한 개 이상 배치한 뒤 분석해 주세요.")

    model_input = state.get("model_input")
    if not model_input:
        raise ValueError("설문 입력이 없습니다. 설문부터 다시 진행해 주세요.")

    room = state.get("selected_room", "living_room")
    state.pop("comparison_images", None)

    state["current_room_layout"] = layout
    state["ai_layout"] = layout
    state["ai_room_width"] = room_width
    state["ai_room_height"] = room_height
    flags_by_room = state.get("backend_hazard_flags_by_room", {})
    flags_by_room[room] = hazard_flags
    state["backend_hazard_flags_by_room"] = flags_by_room
    state["backend_hazard_flags"] = hazard_flags
    state["backend_hazard_question_context"] = question_context
    state["current_room_annotations"] = annotations
    state["current_room_note"] = room_note

    user_features = state.get("user_features")
    feature_report = state.get("feature_mapping_report")
    if not isinstance(user_features, dict) or len(user_features) != 52:
        user_features, feature_report = frontend_model_input_to_backend_features(model_input)
        state["user_features"] = user_features
        state["feature_mapping_report"] = feature_report

    floorplan, floorplan_report = build_floorplan_from_streamlit(
        layout=layout,
        room_name=room,
        room_width=room_width,
        room_height=room_height,
        hazard_flags=hazard_flags,
        question_context=question_context,
        annotations=annotations,
        room_note=room_note,
    )
    state["floorplan_json"] = floorplan
    state["floorplan_mapping_report"] = floorplan_report

    full_result = run_full_analysis(
        user_features=user_features,
        row_index=None,
        floorplan=floorplan,
        top_n=7,
        selected_room=floorplan_report["room_type"],
        use_gpt=True,
        prediction_context=state.get("prediction"),
    )

    legacy_result = backend_result_to_streamlit_result(full_result)
    state["backend_analysis_result"] = full_result
    state["analysis_id"] = full_result.get("analysis_id")
    state["ai_result"] = legacy_result
    state.pop("pending_layout_analysis_payload", None)
    return build_ai_result_payload(state, layout, room_width, room_height)


@app.route("/api/prepare_layout_analysis", methods=["POST"])
def api_prepare_layout_analysis():
    state = get_state()
    payload = request.get_json(force=True) or {}
    if not payload.get("layout"):
        return jsonify({"error": "가구를 한 개 이상 배치한 뒤 분석해 주세요."}), 400
    # 분석은 다음 화면에서 실행합니다. 여기서는 사용자가 만든 배치만 안전하게 저장합니다.
    state["pending_layout_analysis_payload"] = payload
    state["current_room_layout"] = payload.get("layout", [])
    state["ai_layout"] = payload.get("layout", [])
    state["ai_room_width"] = float(payload.get("room_width", 720))
    state["ai_room_height"] = float(payload.get("room_height", 460))
    state["current_room_annotations"] = [i for i in (payload.get("point_notes") or []) if isinstance(i, dict)]
    state["current_room_note"] = str(payload.get("room_note") or "").strip()
    return jsonify({"ok": True, "redirect_url": url_for("layout_analysis_page")})


@app.route("/layout_analysis")
def layout_analysis_page():
    state = get_state()
    room = state.get("selected_room")
    if not room:
        return redirect(url_for("floorplan_page"))
    pending = bool(state.get("pending_layout_analysis_payload"))
    initial_data = None
    if not pending and state.get("ai_result"):
        initial_data = build_ai_result_payload(
            state,
            state.get("ai_layout") or state.get("current_room_layout") or [],
            state.get("ai_room_width", 720),
            state.get("ai_room_height", 460),
        )
    if not pending and not initial_data:
        return redirect(url_for("layout_editor_page"))
    return render_template(
        "layout_analysis.html",
        progress=progress_context("room_detail"),
        room=room,
        room_label=ROOM_LABELS.get(room, room),
        should_run=pending,
        initial_data=initial_data,
    )


@app.route("/api/run_layout_analysis", methods=["POST"])
def api_run_layout_analysis():
    state = get_state()
    payload = state.get("pending_layout_analysis_payload")
    if not isinstance(payload, dict):
        if state.get("ai_result"):
            return jsonify(build_ai_result_payload(
                state,
                state.get("ai_layout") or state.get("current_room_layout") or [],
                state.get("ai_room_width", 720),
                state.get("ai_room_height", 460),
            ))
        return jsonify({"error": "분석할 배치 정보를 찾을 수 없습니다. 배치 화면에서 다시 시작해 주세요."}), 400
    try:
        return jsonify(_run_layout_analysis_payload(state, payload))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc).strip() or exc.__class__.__name__}), 500


@app.route("/api/analyze_layout", methods=["POST"])
def api_analyze_layout():
    # 기존 API 호환용. 새 UI에서는 prepare -> layout_analysis -> run 순서를 사용합니다.
    state = get_state()
    payload = request.get_json(force=True) or {}
    try:
        return jsonify(_run_layout_analysis_payload(state, payload))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc).strip() or exc.__class__.__name__}), 500


def _augment_with_ai_only_fallback(state: dict, room: str) -> None:
    """RAG 문헌 매칭에 실패한 위험요인은 파이프라인이 이미 계산해서
    backend_analysis_result.base_result["unresolved_hazards"]에 담아 두는데, 같은 분석
    안에서 다른 위험요인이 RAG 매칭에 성공하면 결과 전체가 "RAG_EVIDENCE" 모드로 넘어가면서
    이 미매핑 위험요인들은 화면에 전혀 노출되지 않고 조용히 버려집니다(사용자가 체크리스트에서
    분명히 확인한 위험요인인데 개선안이 하나도 안 나오는 문제).

    여기서는 그 unresolved_hazards를 찾아서, GPT에게 "문헌 근거는 없지만 독자적으로
    합리적인 개선안을 제안해 달라"고 요청하고, evidence_id 없이 recommendation_source를
    "AI_ONLY_NO_RAG"로 명확히 표시한 채로 state["ai_result"]에 병합합니다. 한 세션에서
    한 번만 계산되도록 캐싱합니다(재호출 시 GPT를 다시 부르지 않음)."""
    ai_result = state.get("ai_result")
    if not isinstance(ai_result, dict) or ai_result.get("ai_only_fallback_applied"):
        return
    # v6 hybrid pipeline에서는 미매핑 hazard가 이미 같은 planner 결과에 포함됩니다.
    # 기존 보조 fallback을 다시 실행하면 AI 추천이 중복되므로 새 결과에서는 건너뜁니다.
    if str(ai_result.get("solution_evidence_mode") or "") == "HYBRID_RAG_AI":
        ai_result["ai_only_fallback_applied"] = True
        return
    if any(
        isinstance(item, dict) and item.get("is_ai_only")
        for item in ai_result.get("recommendations", [])
    ):
        ai_result["ai_only_fallback_applied"] = True
        return

    backend_result = state.get("backend_analysis_result") or {}
    base_result = backend_result.get("base_result") or {}
    unresolved = [
        h for h in (base_result.get("unresolved_hazards") or [])
        if isinstance(h, dict) and str(h.get("room_type", "")).strip().lower() == str(room).lower()
    ]
    ai_result["ai_only_fallback_applied"] = True
    if not unresolved:
        return

    layout = state.get("ai_layout") or state.get("current_room_layout") or []
    layout_by_id = {str(item.get("id")): item for item in layout if isinstance(item, dict)}

    descriptions = []
    for h in unresolved:
        raw = f"{h.get('hazard_code', '')} / {room.upper()} / {h.get('object_id', '') or 'room'}"
        descriptions.append(wh.describe_floorplan_problem(room, raw))

    existing = [r for r in ai_result.get("recommendations", []) if isinstance(r, dict)]
    # 대표 해결책뿐 아니라 그 안에 담긴 대안들(alternative_improvements)까지 전부 넘겨서,
    # "이미 그랩바 설치가 대안으로 제안돼 있는데 AI가 그걸 모르고 또 그랩바를 제안"하는
    # 중복을 방지합니다.
    existing_texts = []
    for r in existing:
        text = wh.clean_display_text(str(r.get("improvement", "")).strip())
        if text:
            existing_texts.append(text)
        for alt in r.get("alternative_improvements", []) or []:
            alt_text = wh.clean_display_text(str(alt).strip())
            if alt_text:
                existing_texts.append(alt_text)

    room_label = ROOM_LABELS.get(room, room)
    try:
        ai_texts = generate_ai_only_recommendations(room_label, descriptions, existing_texts)
    except Exception:
        return  # 실패해도 기존 RAG 추천은 그대로 유지 (전체가 죽지 않게)

    next_priority = max([r.get("priority") or 0 for r in existing], default=0) + 1
    new_recs, new_markers = [], []

    for i, hazard in enumerate(unresolved):
        text = wh.clean_display_text((ai_texts[i] if i < len(ai_texts) else "").strip())
        if not text:
            continue
        object_id = str(hazard.get("object_id", "") or "")
        target_item = layout_by_id.get(object_id)
        target_object = target_item.get("name") if target_item else "room"
        rec_id = f"ai_only_{room}_{i}"

        new_recs.append({
            "recommendation_id": rec_id,
            "priority": next_priority,
            "hazard_priority_rank": next_priority,
            "solution_candidate_rank_within_hazard": 1,
            "improvement": text,
            "floorplan_problem": f"{hazard.get('hazard_code', '')} / {room.upper()} / {object_id or 'room'}",
            "recommendation_source": "AI_ONLY_NO_RAG",
            "is_ai_only": True,
            "shap_factor": "",
            "reason": "RAG 문헌 근거 미매핑 — GPT가 일반적인 낙상예방 원칙에 따라 독자 제안",
            "rag_evidence": "",
            "expected_effect": "",
            "evidence_ids": [],
        })
        new_markers.append({
            "recommendation_id": rec_id,
            "priority": next_priority,
            "label": text,
            "target_object": target_object,
            "marker_type": "near_object" if target_item else "outside_list",
        })
        next_priority += 1

    if new_recs:
        ai_result["recommendations"] = existing + new_recs
        ai_result["markers"] = [m for m in (ai_result.get("markers") or []) if isinstance(m, dict)] + new_markers


def build_ai_result_payload(state: dict, layout: list, room_width: float, room_height: float) -> dict:
    room = state.get("selected_room") or ""
    if room:
        _augment_with_ai_only_fallback(state, room)
    result = state.get("ai_result", {})

    safe_width = max(float(room_width or 0), 1.0)
    safe_height = max(float(room_height or 0), 1.0)
    room_area_cm2 = safe_width * safe_height
    furniture_area_cm2 = 0.0
    valid_count = 0
    for item in layout:
        if not isinstance(item, dict):
            continue
        try:
            w = max(float(item.get("width", 0)), 0.0)
            h = max(float(item.get("height", 0)), 0.0)
        except (TypeError, ValueError):
            continue
        furniture_area_cm2 += w * h
        valid_count += 1
    occupancy = furniture_area_cm2 / room_area_cm2 * 100 if room_area_cm2 > 0 else 0.0
    metrics = {
        "room_size": f"{int(safe_width):,} × {int(safe_height):,} cm",
        "room_area": f"{room_area_cm2 / 10000:.1f}㎡",
        "furniture_count": f"{valid_count}개",
        "occupancy": f"{occupancy:.1f}%",
    }

    awaiting_confirmation = bool(result.get("awaiting_confirmation")) or (
        str(result.get("solution_evidence_mode") or "") == "AI_ONLY_PRECAUTIONARY"
    )

    all_recommendations = [r for r in result.get("recommendations", []) if isinstance(r, dict)]
    deduped_recommendations = wh.dedupe_recommendations_by_hazard(all_recommendations)
    # 사용자 화면으로 나가는 모든 추천 텍스트에서 내부 room/path/object ID를 제거하고,
    # 위험요인 단위로 1,2,3... 번호를 다시 매겨 중복 번호가 보이지 않게 합니다.
    safe_recommendations = []
    for idx, rec in enumerate(deduped_recommendations, start=1):
        safe = dict(rec)
        safe["priority"] = idx
        safe["display_number"] = idx
        safe["improvement"] = wh.clean_display_text(str(safe.get("improvement") or ""))
        safe["reason"] = wh.clean_display_text(str(safe.get("reason") or ""))
        safe["expected_effect"] = wh.clean_display_text(str(safe.get("expected_effect") or ""))
        safe["floorplan_problem"] = wh.describe_floorplan_problem(room, str(safe.get("floorplan_problem") or ""))
        safe_recommendations.append(safe)
    # 추가 확인 질문은 "추가 후보"를 확인하기 위한 것이며, 이미 확정되어 표시되던
    # 기존 RAG/AI 개선안을 숨기면 안 됩니다. 기존 추천은 항상 유지하고 질문은 별도로 표시합니다.
    displayed_recommendations = safe_recommendations

    # "내 방의 위험 지점" 평면도 + 번호 마커. 특정 가구와 연결된 개선안은 그 가구 위에,
    # 방 전체에 적용되는 개선안(조명·바닥·동선 등)은 옆 목록에만 표시됩니다.
    floorplan_html = None
    if layout and displayed_recommendations:
        # 옆의 "개선사항" 텍스트 목록은 위에서 dedup 후 1,2,3...으로 재번호했는데,
        # 원본 markers는 재번호 이전의 priority를 그대로 갖고 있어서 그대로 쓰면
        # 평면도 배지 번호와 텍스트 번호가 서로 어긋납니다(서로 다른 두 위험요인의
        # 원본 priority가 우연히 같아서 둘 다 "1"로 찍히는 문제). recommendation_id로
        # display_number를 다시 매핑해서 항상 텍스트 목록과 같은 번호가 찍히게 합니다.
        display_number_by_id = {
            str(r.get("recommendation_id")): r.get("display_number")
            for r in displayed_recommendations if r.get("recommendation_id")
        }
        all_markers = [m for m in result.get("markers", []) if isinstance(m, dict)]
        displayed_ids = set(display_number_by_id.keys())
        raw_markers = (
            [m for m in all_markers if str(m.get("recommendation_id")) in displayed_ids]
            if displayed_ids else all_markers[: len(displayed_recommendations)]
        )
        markers = []
        for m in raw_markers:
            renumbered = dict(m)
            new_number = display_number_by_id.get(str(m.get("recommendation_id")))
            if new_number is not None:
                renumbered["priority"] = new_number
                renumbered["display_number"] = new_number
            markers.append(renumbered)
        if markers:
            positioned_markers = wh.calculate_marker_positions(
                layout=layout, markers=markers, room_width=int(room_width), room_height=int(room_height),
            )
            floorplan_html = wh.render_annotated_floorplan_html(
                layout=layout, positioned_markers=positioned_markers,
                room_width=int(room_width), room_height=int(room_height),
            )

    return {
        "ok": True,
        "metrics": metrics,
        "summary": wh.clean_display_text(str(result.get("summary", ""))),
        "risk_analysis": wh.clean_display_text(str(result.get("risk_analysis", ""))),
        "final_summary": wh.clean_display_text(str(result.get("final_summary", ""))),
        "recommendations": displayed_recommendations,
        "clarification_questions": result.get("clarification_questions", []),
        "awaiting_confirmation": awaiting_confirmation,
        "analysis_id": result.get("analysis_id"),
        "floorplan_html": floorplan_html,
        "planner_mode": result.get("planner_mode"),
        "gpt_api_called": result.get("gpt_api_called"),
    }


@app.route("/api/ai_result")
def api_ai_result():
    state = get_state()
    layout = state.get("ai_layout") or state.get("current_room_layout") or []
    room_width = state.get("ai_room_width", 720)
    room_height = state.get("ai_room_height", 460)
    if not state.get("ai_result"):
        return jsonify({"error": "저장된 분석 결과가 없습니다."}), 404
    return jsonify(build_ai_result_payload(state, layout, room_width, room_height))


def _merge_clarification_result_preserving_existing(previous_ai_result: dict, followup_ai_result: dict) -> dict:
    """추가 질문은 기존 공간개선안을 대체하지 않고 보완만 합니다.

    후속 clarification 플래너는 질문으로 확인된 RAG 미매핑 후보만 다시 계획하므로,
    그대로 state["ai_result"]를 덮어쓰면 이전에 이미 화면에 보였던 RAG 개선안이
    사라질 수 있습니다. 기존 recommendation/marker는 보존하고, 후속 결과에서 새로
    생긴 AI 개선안만 중복 없이 합칩니다.
    """
    previous = previous_ai_result if isinstance(previous_ai_result, dict) else {}
    followup = followup_ai_result if isinstance(followup_ai_result, dict) else {}
    merged = dict(followup)

    old_recs = [dict(r) for r in previous.get("recommendations", []) if isinstance(r, dict)]
    new_recs = [dict(r) for r in followup.get("recommendations", []) if isinstance(r, dict)]

    def rec_key(rec: dict) -> tuple[str, str]:
        rec_id = str(rec.get("recommendation_id") or "").strip()
        improvement = wh.clean_display_text(str(rec.get("improvement") or "")).strip().lower()
        return rec_id, improvement

    combined_recs = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    for rec in old_recs + new_recs:
        rec_id, improvement = rec_key(rec)
        if rec_id and rec_id in seen_ids:
            continue
        if improvement and improvement in seen_texts:
            continue
        combined_recs.append(rec)
        if rec_id:
            seen_ids.add(rec_id)
        if improvement:
            seen_texts.add(improvement)

    # 기존 순서를 유지하되 화면 표시 번호는 다시 1..N으로 정규화합니다.
    for idx, rec in enumerate(combined_recs, start=1):
        rec["priority"] = idx
        rec["display_number"] = idx
    merged["recommendations"] = combined_recs

    old_markers = [dict(m) for m in previous.get("markers", []) if isinstance(m, dict)]
    new_markers = [dict(m) for m in followup.get("markers", []) if isinstance(m, dict)]
    marker_by_rec_id: dict[str, dict] = {}
    loose_markers: list[dict] = []
    for marker in old_markers + new_markers:
        rec_id = str(marker.get("recommendation_id") or "").strip()
        if rec_id:
            marker_by_rec_id.setdefault(rec_id, marker)
        else:
            loose_markers.append(marker)

    merged_markers = []
    for idx, rec in enumerate(combined_recs, start=1):
        rec_id = str(rec.get("recommendation_id") or "").strip()
        marker = marker_by_rec_id.get(rec_id)
        if marker:
            marker = dict(marker)
            marker["priority"] = idx
            marker["display_number"] = idx
            merged_markers.append(marker)
    # recommendation_id가 없는 레거시 마커는 잃지 않되 뒤에 유지합니다.
    merged_markers.extend(loose_markers)
    merged["markers"] = merged_markers

    # 후속 질문이 NO여서 새 AI 개선안이 0개여도 기존 개선안이 있으면 NO_ACTIONS가 아닙니다.
    if combined_recs:
        has_rag = any(str(r.get("recommendation_source") or "").startswith("RAG") for r in combined_recs)
        has_ai = any(str(r.get("recommendation_source") or "").startswith("AI") or r.get("is_ai_only") for r in combined_recs)
        if has_rag and has_ai:
            merged["solution_evidence_mode"] = "HYBRID_RAG_AI"
        elif has_rag:
            merged["solution_evidence_mode"] = "RAG_EVIDENCE"
        elif has_ai:
            merged["solution_evidence_mode"] = "AI_ONLY_NO_RAG"
        merged["mapped_solution_candidate_count"] = len(combined_recs)
        merged["default_solution_display_count"] = min(3, len(combined_recs))
        merged["awaiting_confirmation"] = bool(merged.get("clarification_questions"))

        if not new_recs and old_recs:
            merged["summary"] = previous.get("summary") or merged.get("summary")
            merged["risk_analysis"] = previous.get("risk_analysis") or merged.get("risk_analysis")
            merged["final_summary"] = (
                "추가 확인 답변을 반영했습니다. 새로 확정된 추가 위험은 없으며, "
                "기존에 확인된 공간개선안은 그대로 유지합니다."
            )
    return merged


@app.route("/api/clarification_answer", methods=["POST"])
def api_clarification_answer():
    """추가 확인 질문 답변을 반영해 SHAP/RAG 재실행 없이 AI가 솔루션 카드를 다시 생성합니다."""
    state = get_state()
    payload = request.get_json(force=True)
    # {question_id: "YES" | "NO" | "UNKNOWN"}
    answers_by_question_id = payload.get("answers", {})

    previous_full_result = state.get("backend_analysis_result")
    ai_result = state.get("ai_result", {})
    questions = [q for q in ai_result.get("clarification_questions", []) if isinstance(q, dict)]
    room = state.get("selected_room", "living_room")
    floorplan = state.get("floorplan_json", {})

    if not previous_full_result or not questions:
        return jsonify({"error": "이전 공간 점검 결과를 찾을 수 없습니다. 배치 안전 점검을 다시 진행해 주세요."}), 400

    try:
        existing_flags = state.get("backend_hazard_flags", {})
        existing_context = state.get("backend_hazard_question_context", [])
        merged_flags, merged_context, normalized_answers = merge_clarification_answers(
            existing_flags=existing_flags if isinstance(existing_flags, dict) else {},
            existing_question_context=existing_context if isinstance(existing_context, list) else [],
            clarification_questions=questions,
            answers_by_question_id=answers_by_question_id,
            room_name=room,
        )
        state["backend_hazard_flags"] = merged_flags
        state["backend_hazard_question_context"] = merged_context
        flags_by_room = state.get("backend_hazard_flags_by_room", {})
        flags_by_room[room] = merged_flags
        state["backend_hazard_flags_by_room"] = flags_by_room

        # clarification은 기존 분석을 "교체"하지 않고 "보완"해야 합니다.
        # 이전에 이미 표시되던 RAG 개선안을 보존한 채, 질문으로 새로 확인된 AI 개선안만 합칩니다.
        previous_ai_result = dict(ai_result)
        followup_result = run_clarification_followup(
            previous_full_result=previous_full_result,
            floorplan=floorplan if isinstance(floorplan, dict) else {},
            clarification_questions=questions,
            answers_by_question_id=normalized_answers,
            use_gpt=True,
        )
        legacy_result = backend_result_to_streamlit_result(followup_result)
        merged_ai_result = _merge_clarification_result_preserving_existing(previous_ai_result, legacy_result)
        # 원래 전체 분석 결과는 이후 추가 질문에서도 계속 기준으로 사용해야 합니다.
        # follow-up 결과로 덮어쓰면 다음 질문에서 기존 HZ_xxx context가 사라져
        # "통합계획에 누락된 hazard instance" 검증 오류가 발생할 수 있습니다.
        state["backend_last_clarification_result"] = followup_result
        state["ai_result"] = merged_ai_result
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500

    layout = state.get("ai_layout", [])
    return jsonify(build_ai_result_payload(state, layout, state.get("ai_room_width", 720), state.get("ai_room_height", 460)))


@app.route("/room_detail")
def room_detail_page():
    state = get_state()
    room = state.get("selected_room")
    if not room:
        return redirect(url_for("floorplan_page"))

    info = wh.ROOM_INFO[room]
    return render_template(
        "room_detail.html",
        progress=progress_context("room_detail"),
        room=room,
        info=info,
        recommendations=wh.RECOMMENDATION_DATA[room],
        has_ai_result=bool(state.get("ai_result")),
    )


@app.route("/api/reset_ai_result", methods=["POST"])
def api_reset_ai_result():
    state = get_state()
    state.pop("ai_result", None)
    state.pop("last_layout_request_id", None)
    state.pop("comparison_images", None)
    return jsonify({"ok": True})


def _build_comparison_images(layout: list, room_width: int, room_height: int, hazard_groups: list) -> dict:
    """사용자가 실제 배치한 평면도 좌표를 그대로 사용한 단일 2D 개선 위치 지도를 생성합니다.

    3D는 작은 글자와 원근 때문에 정보 전달력이 낮아 제거했습니다. 이 2D 지도를 실사형 참고 이미지와
    번호로 직접 대응시키는 것이 현재 서비스 목적에 더 적합합니다.
    """
    markers = []
    for group in hazard_groups:
        options = group.get("options", [])
        selected = group.get("selected", 0)
        selected = selected if 0 <= selected < len(options) else 0
        label = options[selected] if options else group.get("floorplan_problem", "")
        action_plans = group.get("action_plans") or []
        icon_type = "none"
        if 0 <= selected < len(action_plans):
            icon_type = action_plans[selected].get("visual_addition_type", "none") or "none"
        display_number = group.get("display_number", group.get("priority"))
        markers.append({
            "priority": group.get("priority"),
            "display_number": display_number,
            "label": label,
            "target_object": group.get("target_object", "room"),
            "marker_type": group.get("marker_type", "outside_list"),
            "icon_type": icon_type,
        })
    positioned = wh.calculate_marker_positions(layout, markers, room_width=room_width, room_height=room_height)
    canvas_px_2d, _ = pick_canvas_size(room_width, room_height)
    before_2d = render_floorplan_png(
        room_width, room_height, layout, canvas_px=canvas_px_2d, show_handles=False,
    )
    after_2d = render_floorplan_with_markers(
        room_width, room_height, layout, positioned, canvas_px=canvas_px_2d, marker_style="done",
    )
    return {
        "before_2d_path": save_generated_image(before_2d, "before2d"),
        "after_2d_path": save_generated_image(after_2d, "after2d"),
    }



def _build_web_overlay_markers(layout: list, hazard_groups: list, room_width: int, room_height: int) -> list[dict]:
    """AI 이미지 자체에는 글자/번호를 그리지 않고 웹에서 표시할 마커 좌표를 만듭니다.

    좌표는 사용자가 만든 평면도 기준의 상대 비율(%)입니다. AI AFTER가 같은 구도를 유지하도록
    프롬프트를 강하게 제한했기 때문에, 웹 오버레이는 근사 위치 안내로 사용합니다.
    """
    defs = []
    for group in hazard_groups or []:
        options = group.get("options") or []
        selected = group.get("selected", 0)
        selected = selected if isinstance(selected, int) and 0 <= selected < len(options) else 0
        improvement = options[selected] if options else group.get("floorplan_problem", "")
        defs.append({
            "priority": group.get("priority"),
            "display_number": group.get("display_number", group.get("priority")),
            "label": wh.clean_display_text(str(improvement or "개선 위치")),
            "target_object": group.get("target_object", "room"),
            "marker_type": group.get("marker_type", "outside_list"),
            "icon_type": "none",
        })
    positioned = wh.calculate_marker_positions(layout, defs, room_width=room_width, room_height=room_height)
    overlays = []
    outside_idx = 0
    for marker in positioned:
        if marker.get("outside") or marker.get("x") is None or marker.get("y") is None:
            # 방 전체/동선 같은 항목은 이미지 상단 가장자리에서 겹치지 않게 안내합니다.
            left_pct = min(82.0, 8.0 + outside_idx * 17.0)
            top_pct = 8.0
            outside_idx += 1
        else:
            left_pct = max(4.0, min(94.0, (float(marker.get("x", 0)) + 16.0) / max(room_width, 1) * 100.0))
            top_pct = max(5.0, min(92.0, (float(marker.get("y", 0)) + 16.0) / max(room_height, 1) * 100.0))
        overlays.append({
            "display_number": marker.get("display_number", marker.get("priority")),
            "label": wh.clean_display_text(str(marker.get("label") or "개선 위치")),
            "left_pct": round(left_pct, 1),
            "top_pct": round(top_pct, 1),
        })
    return overlays

def _get_dedup_hazard_groups(state: dict, room: str) -> tuple[list, str]:
    """ai_result의 recommendations를 위험요인당 대표 1개로 dedup하고,
    각 대표에 원본 마커(target_object 등)를 붙여 hazard_groups로 만듭니다.
    generate_comparison_images / generate_action_plan 양쪽에서 공통으로 씁니다.
    실패 시 (빈 리스트, 에러메시지) 를 반환합니다."""
    ai_result = state.get("ai_result")
    if not ai_result:
        return [], "먼저 'AI 분석하기'를 실행해 개선 우선순위를 계산해 주세요."

    recommendations = [r for r in ai_result.get("recommendations", []) if isinstance(r, dict)]
    if not recommendations:
        return [], "표시할 개선 우선순위가 없습니다. 분석 결과를 먼저 확인해 주세요."

    # PDF 리포트에서 하던 것과 동일하게, 개인 위험요인(shap_factor)·근거·이유를 원본
    # recommendations에서 미리 찾아둡니다 — Action Plan 카드에서 "왜 이 개선안이 뜨는지"를
    # 사용자의 설문 응답과 연결해 보여주기 위함입니다(요청: 설문 답변이 이 개선방안을
    # 뜨게 한 이유를 강조 표시).
    all_recs_by_id = {str(r.get("recommendation_id")): r for r in recommendations}

    recommendations = wh.dedupe_recommendations_by_hazard(recommendations)

    all_markers = [m for m in (ai_result.get("markers") or []) if isinstance(m, dict)]
    markers_by_rec_id = {str(m.get("recommendation_id")): m for m in all_markers}

    hazard_groups = []
    for rec in recommendations:
        options = [str(rec.get("improvement", "")).strip()] + list(rec.get("alternative_improvements", []))
        options = [wh.clean_display_text(opt) for opt in options if opt]
        rec_id = str(rec.get("recommendation_id"))
        marker = markers_by_rec_id.get(rec_id)
        rec_detail = all_recs_by_id.get(rec_id, {})
        hazard_groups.append({
            "priority": rec.get("priority"),
            "display_number": rec.get("priority"),
            "recommendation_id": rec_id,
            "floorplan_problem": wh.describe_floorplan_problem(room, rec.get("floorplan_problem", "")),
            "options": options,
            "selected": 0,
            "target_object": marker.get("target_object", "room") if marker else "room",
            "marker_type": marker.get("marker_type", "outside_list") if marker else "outside_list",
            "shap_factor": wh.humanize_shap_factor(rec_detail.get("shap_factor", "")),
            "rag_evidence": wh.clean_display_text(rec_detail.get("rag_evidence", "")),
            "reason": wh.clean_display_text(rec_detail.get("reason", "")),
            "recommendation_source": rec_detail.get("recommendation_source", ""),
        })
    return hazard_groups, ""


@app.route("/api/generate_comparison_images", methods=["POST"])
def api_generate_comparison_images():
    """현재 가구 배치(Before)와, 이미 계산된 개선 우선순위(recommendations)를 반영한
    개선 후(After) 평면도를 실제 좌표 기반으로 즉시 그려서 비교해서 보여줍니다.
    우선순위 자체는 새로 계산하지 않고, 이미 /api/analyze_layout 에서 계산된 결과를
    그대로 마커로 표현합니다. (예전에는 OpenAI 이미지 생성/편집을 썼지만, 텍스트 프롬프트로
    좌표를 완벽히 지키지 못해 레이아웃이 깨지고 대안 전환마다 오래 걸리는 문제가 있어
    정확한 좌표 렌더링 방식으로 교체했습니다.)"""
    state = get_state()
    room = state.get("selected_room")
    if not room:
        return jsonify({"error": "선택된 공간이 없습니다."}), 400

    layout = state.get("ai_layout") or state.get("current_room_layout") or []
    if not layout:
        return jsonify({"error": "배치된 가구가 없습니다. 먼저 가구를 배치하고 분석해 주세요."}), 400

    hazard_groups, error = _get_dedup_hazard_groups(state, room)
    if error:
        return jsonify({"error": error}), 400

    room_width = int(state.get("ai_room_width", 720))
    room_height = int(state.get("ai_room_height", 460))
    room_label = ROOM_LABELS.get(room, room)
    ai_result = state.get("ai_result", {})
    ai_comment = str(ai_result.get("final_summary") or ai_result.get("summary") or "").strip()

    images = _build_comparison_images(layout, room_width, room_height, hazard_groups)
    overlay_markers = _build_web_overlay_markers(layout, hazard_groups, room_width, room_height)

    cached_before = _cached_before_path(state, room_label, room_width, room_height, layout)
    state["comparison_images"] = {
        "room_label": room_label,
        "room_width": room_width,
        "room_height": room_height,
        "ai_comment": ai_comment,
        "hazard_groups": hazard_groups,
        "before_2d_path": images["before_2d_path"],
        "after_2d_path": images["after_2d_path"],
        # 코드 렌더링 평면도는 AI 입력용 reference일 뿐, 사용자에게 보여줄 BEFORE가 아닙니다.
        "photoreal_before_image_path": cached_before if cached_before else None,
        "web_overlay_markers": overlay_markers,
    }

    return jsonify({
        "ok": True,
        "before_image_url": url_for("static", filename=images["before_2d_path"]),
        "after_image_url": url_for("static", filename=images["after_2d_path"]),
        "before_ai_image_url": url_for("static", filename=cached_before) if cached_before else None,
        "room_label": room_label,
        "ai_comment": ai_comment,
        "legend": wh.build_comparison_legend(hazard_groups),
    })


def _collect_user_space_requests(state: dict) -> list[dict]:
    requests = []
    layout = [i for i in (state.get("ai_layout") or state.get("current_room_layout") or []) if isinstance(i, dict)]
    room_note = str(state.get("current_room_note") or "").strip()
    if room_note:
        requests.append({"type": "room", "label": "방 전체 메모", "text": room_note, "nearby_objects": []})
    for idx, item in enumerate(state.get("current_room_annotations") or [], start=1):
        if not isinstance(item, dict):
            continue
        note = str(item.get("note") or "").strip()
        if not note:
            continue
        nearby = list(item.get("nearby_object_names") or [])
        if not nearby and layout:
            try:
                px = float(item.get("x", 0)); py = float(item.get("y", 0))
                ranked = []
                for obj in layout:
                    ox = float(obj.get("x", 0)); oy = float(obj.get("y", 0))
                    ow = float(obj.get("width", 0)); oh = float(obj.get("height", 0))
                    cx, cy = ox + ow / 2, oy + oh / 2
                    ranked.append(((cx-px)**2 + (cy-py)**2, str(obj.get("name") or "가구")))
                ranked.sort(key=lambda t: t[0])
                nearby = [name for _, name in ranked[:2]]
            except (TypeError, ValueError):
                nearby = []
        requests.append({
            "type": "point",
            "label": f"지점 메모 {idx}",
            "text": note,
            "nearby_objects": nearby,
            "x": item.get("x"),
            "y": item.get("y"),
        })
    return requests


def _generate_supplementary_space_advice(state: dict, room_label: str) -> dict:
    """낙상 위험 우선순위와 별개인 자유 메모를 절대 버리지 않고 별도 상담으로 반환합니다.

    자유 메모를 억지로 hazard로 승격하지는 않습니다. 배치 요청처럼 위험모델의 범위를 벗어나는
    내용은 '추가 공간 요청사항'으로 분리해 답하고, 좌표만으로 확정하기 어려운 경우 한계를 명시합니다.
    """
    requests = _collect_user_space_requests(state)
    if not requests:
        return {"requests": [], "advice": "", "scope_note": ""}

    layout = state.get("ai_layout") or state.get("current_room_layout") or []
    payload = {
        "room": room_label,
        "room_width_cm": state.get("ai_room_width", 720),
        "room_height_cm": state.get("ai_room_height", 460),
        "furniture": [
            {
                "name": str(i.get("name") or "가구"),
                "x": i.get("x"), "y": i.get("y"),
                "width": i.get("width"), "height": i.get("height"),
                "rotation": i.get("rotation", 0),
            }
            for i in layout if isinstance(i, dict)
        ],
        "user_requests": requests,
    }
    fallback = (
        "추가 메모는 낙상 위험 우선순위와 별도로 보존했습니다. "
        "현재 입력만으로 가구의 실제 깊이, 문 열림 범위, 콘센트·창문·수납 사용 빈도까지 알 수 없어 "
        "새 가구의 최적 위치를 확정할 수는 없습니다. 평면도에서 통로와 문 여닫힘을 막지 않는 후보 위치를 먼저 검토하고, "
        "실제 설치 전 현장에서 여유 폭을 확인해 주세요."
    )
    try:
        from openai import OpenAI
        client = OpenAI(timeout=float(os.getenv("OPENAI_SUPPLEMENTARY_TIMEOUT_SECONDS", "12")), max_retries=0)
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            instructions=(
                "당신은 고령자 주거공간 배치 보조 AI입니다. 입력의 user_requests는 낙상 위험모델의 hazard가 아닐 수도 있습니다. "
                "따라서 위험 우선순위를 새로 만들지 말고, 사용자가 직접 적은 요청을 하나도 빠뜨리지 않은 채 별도의 '추가 공간 요청사항'으로 답하세요. "
                "가구를 어디 둘지 묻는 요청은 제공된 방 크기와 가구 bbox만 근거로 후보 위치를 제안하되, 실제 문 열림 범위·창문·콘센트·가구 깊이 정보가 없으면 확정 표현을 피하세요. "
                "기존 낙상 개선안과 충돌할 수 있는 배치라면 통로 확보를 우선하라고 설명하세요. 3~6문장으로 구체적으로 작성하고, 입력에 없는 치수나 객체를 만들지 마세요."
            ),
            input=json.dumps(payload, ensure_ascii=False),
        )
        advice = str(getattr(response, "output_text", "") or "").strip() or fallback
    except Exception:
        advice = fallback
    return {
        "requests": requests,
        "advice": advice,
        "scope_note": "이 항목은 낙상 위험 우선순위와 별도의 공간 활용 상담입니다. 안전 우선순위 자체를 변경하지 않습니다.",
    }


@app.route("/api/generate_action_plan", methods=["POST"])
def api_generate_action_plan():
    """위험요인별로 '모든 대안'에 대한 실행 계획(난이도·예상 소요 시간·준비물·
    도우미 필요 여부·실행 방법·기대 효과)을 GPT로 생성합니다.
    같은 위험요인의 대안들을 한 번에 같이 물어봐서 상대적으로 비교 가능한 값이 나오도록
    하며, 호출 횟수는 위험요인 개수만큼만 발생합니다(대안 개수와 무관)."""
    state = get_state()
    room = state.get("selected_room")
    if not room:
        return jsonify({"error": "선택된 공간이 없습니다."}), 400

    hazard_groups, error = _get_dedup_hazard_groups(state, room)
    if error:
        return jsonify({"error": error}), 400

    room_label = ROOM_LABELS.get(room, room)
    warnings: list[str] = []

    # 각 hazard의 GPT 호출을 순차 실행하면 2~4개만 있어도 호스팅 요청 제한을 넘길 수 있습니다.
    # 독립적인 호출이므로 병렬로 처리하고, 개별 호출이 실패하면 이미 확정된 개선안 문구를
    # 그대로 사용한 결정론적 fallback 실행계획으로 대체합니다. 따라서 OpenAI 장애가 Action
    # Plan 화면 전체의 Failed to fetch로 이어지지 않습니다.
    def _build_one(index_group):
        index, group = index_group
        try:
            plans = generate_action_plan_for_hazard(
                room_label, group["floorplan_problem"], group["options"],
            )
            return index, plans, None
        except Exception as exc:  # noqa: BLE001
            plans = generate_fallback_action_plan_for_hazard(
                room_label, group["floorplan_problem"], group["options"],
            )
            return index, plans, str(exc)

    results: dict[int, tuple[list[dict], str | None]] = {}
    max_workers = max(1, min(4, len(hazard_groups)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_build_one, pair) for pair in enumerate(hazard_groups)]
        for future in as_completed(futures):
            index, plans, err = future.result()
            results[index] = (plans, err)

    for index, group in enumerate(hazard_groups):
        plans, err = results.get(index, ([], "실행계획 생성 결과 누락"))
        if err:
            warnings.append(
                f"{group.get('priority', index + 1)}번 개선안은 AI 연결 지연으로 기본 실행계획을 사용했습니다."
            )
        for plan in plans:
            plan["shopping_links"] = [
                {"item": item, "links": build_shopping_links(item)}
                for item in (plan.get("required_items") or [])
            ]
        group["action_plans"] = plans

    supplementary = _generate_supplementary_space_advice(state, room_label)
    state["supplementary_space_advice"] = supplementary
    state["action_plan_groups"] = hazard_groups
    return jsonify({
        "ok": True,
        "groups": hazard_groups,
        "supplementary": supplementary,
        "warnings": warnings,
    })


@app.route("/api/confirm_action_plan", methods=["POST"])
def api_confirm_action_plan():
    """Action Plan 화면에서 사용자가 위험요인마다 고른 대안을 확정하고,
    그 확정된 선택으로 Before/After 평면도를 한 번만 그립니다."""
    state = get_state()
    groups = state.get("action_plan_groups")
    if not groups:
        # Action Plan 화면에는 이미 개선안 카드가 렌더링되어 있는데 서버 메모리 상태가
        # 재시작/개발 리로더 등으로 유실된 경우, 이미지 생성 자체를 막지 않습니다.
        # 현재 분석 결과에서 동일한 hazard/options를 즉시 복원하고 실행계획 메타데이터는
        # 결정론적 fallback으로 채웁니다. 이미지 생성의 필수 조건은 "선택된 개선안"이지
        # GPT 실행계획 캐시의 존재가 아닙니다.
        room_for_recovery = state.get("selected_room")
        if not room_for_recovery:
            return jsonify({"error": "선택된 공간 정보를 찾을 수 없습니다. 이전 단계에서 공간을 다시 선택해 주세요."}), 400
        recovered_groups, recovery_error = _get_dedup_hazard_groups(state, room_for_recovery)
        if recovery_error or not recovered_groups:
            return jsonify({"error": recovery_error or "현재 분석 결과에서 개선안을 복원할 수 없습니다."}), 400
        room_label_for_recovery = ROOM_LABELS.get(room_for_recovery, room_for_recovery)
        for recovered in recovered_groups:
            options = recovered.get("options") or []
            recovered["action_plans"] = generate_fallback_action_plan_for_hazard(
                room_label_for_recovery,
                recovered.get("floorplan_problem") or "",
                options,
            )
        groups = recovered_groups
        state["action_plan_groups"] = groups

    payload = request.get_json(force=True)
    selections = payload.get("selections") or {}

    layout = state.get("ai_layout") or state.get("current_room_layout") or []
    if not layout:
        return jsonify({"error": "배치 정보를 찾을 수 없습니다. 배치를 다시 확인해 주세요."}), 400

    hazard_groups = []
    for group in groups:
        options = group.get("options", [])
        # 각 개선 지점은 최대 1개만 선택하며, 선택하지 않은 지점은 이번 실행 대상에서 제외합니다.
        # 프론트는 미선택을 null로 보내며, 서버에서도 기본 0번 대안으로 강제 선택하지 않습니다.
        raw_sel = selections.get(str(group.get("priority")), None)
        if raw_sel is None:
            continue
        if not isinstance(raw_sel, int) or not (0 <= raw_sel < len(options)):
            return jsonify({"error": "선택한 개선 방법 정보를 확인할 수 없습니다. 다시 선택해 주세요."}), 400
        hazard_groups.append({**group, "selected": raw_sel})

    if not hazard_groups:
        return jsonify({"error": "실천할 개선 방법을 최소 1개 선택해 주세요."}), 400

    # PDF/비교 이미지에서도 실제로 선택한 항목만 사용하도록 별도 보관합니다.
    state["selected_action_plan_groups"] = hazard_groups

    room = state.get("selected_room")
    room_width = int(state.get("ai_room_width", 720))
    room_height = int(state.get("ai_room_height", 460))
    room_label = ROOM_LABELS.get(room, room)
    ai_result = state.get("ai_result", {})
    ai_comment = str(ai_result.get("final_summary") or ai_result.get("summary") or "").strip()

    images = _build_comparison_images(layout, room_width, room_height, hazard_groups)
    # 웹 주석(번호/설명)은 이미지 AI가 그리는 것이 아니라, 사용자 평면도 좌표를
    # 기준으로 브라우저에서 별도 overlay 합니다. confirm 단계에서도 반드시 생성해야
    # 응답의 overlay_markers와 세션 상태가 일치합니다.
    overlay_markers = _build_web_overlay_markers(
        layout, hazard_groups, room_width, room_height
    )

    cached_before = _cached_before_path(state, room_label, room_width, room_height, layout)
    state["comparison_images"] = {
        "room_label": room_label,
        "room_width": room_width,
        "room_height": room_height,
        "ai_comment": ai_comment,
        "hazard_groups": hazard_groups,
        "before_2d_path": images["before_2d_path"],
        "after_2d_path": images["after_2d_path"],
        # 코드 렌더링 평면도는 AI 입력용 reference일 뿐, 사용자에게 보여줄 BEFORE가 아닙니다.
        "photoreal_before_image_path": cached_before if cached_before else None,
        "web_overlay_markers": overlay_markers,
    }

    return jsonify({
        "ok": True,
        "before_image_url": url_for("static", filename=images["before_2d_path"]),
        "after_image_url": url_for("static", filename=images["after_2d_path"]),
        "before_ai_image_url": url_for("static", filename=cached_before) if cached_before else None,
        "room_label": room_label,
        "ai_comment": ai_comment,
        "legend": wh.build_comparison_legend(hazard_groups),
        "overlay_markers": overlay_markers,
    })


@app.route("/api/swap_comparison_solution", methods=["POST"])
def api_swap_comparison_solution():
    """After 이미지에서 특정 우선순위 번호가 나타내는 해결책을, 사용자가 고른
    다른 대안으로 바꿔서 즉시 다시 그립니다. 실제 좌표 렌더링이라 AI 호출이 없고,
    이전에 수십 초 걸리던 것이 거의 즉시 끝납니다."""
    state = get_state()
    comparison = state.get("comparison_images")
    if not comparison:
        return jsonify({"error": "먼저 개선 전후 이미지를 생성해 주세요."}), 400

    payload = request.get_json(force=True)
    try:
        priority = int(payload.get("priority"))
    except (TypeError, ValueError):
        return jsonify({"error": "잘못된 요청입니다."}), 400
    improvement = str(payload.get("improvement", "")).strip()

    hazard_groups = comparison.get("hazard_groups", [])
    group = next((g for g in hazard_groups if g.get("priority") == priority), None)
    if not group:
        return jsonify({"error": "해당 항목을 찾을 수 없습니다."}), 400
    options = group.get("options", [])
    if improvement not in options:
        return jsonify({"error": "선택할 수 없는 항목입니다."}), 400
    group["selected"] = options.index(improvement)

    layout = state.get("ai_layout") or state.get("current_room_layout") or []
    if not layout:
        return jsonify({"error": "배치 정보를 찾을 수 없습니다. 이미지를 다시 생성해 주세요."}), 400

    images = _build_comparison_images(
        layout,
        comparison.get("room_width", 720),
        comparison.get("room_height", 460),
        hazard_groups,
    )
    comparison["last_images"] = images
    comparison["after_2d_path"] = images["after_2d_path"]
    # 개선안이 바뀌어도 BEFORE는 같은 방/배치이므로 재사용합니다. AFTER만 무효화합니다.
    comparison.pop("photoreal_after_image_path", None)
    comparison.pop("photoreal_image_path", None)
    comparison.pop("visual_change_commands", None)

    return jsonify({
        "ok": True,
        "after_image_url": url_for("static", filename=images["after_2d_path"]),
        "legend": wh.build_comparison_legend(hazard_groups),
    })



def _collect_before_image_user_issues(state: dict) -> list[dict]:
    """개선 전 AI 이미지가 현재 문제 상태를 재현할 수 있도록 체크리스트/직접 메모를 하나의 목록으로 정리합니다."""
    issues: list[dict] = []
    flags = state.get("backend_hazard_flags") or {}
    context = state.get("backend_hazard_question_context") or []
    context_by_code = {
        str(item.get("hazard_code")): item
        for item in context if isinstance(item, dict) and item.get("hazard_code")
    }
    for code, checked in flags.items():
        if not checked:
            continue
        item = context_by_code.get(str(code), {})
        label = str(item.get("label") or code).strip()
        issues.append({"source": "체크리스트", "label": label})

    room_note = str(state.get("current_room_note") or "").strip()
    if room_note:
        issues.append({"source": "방 전체 메모", "label": room_note})

    layout = [x for x in (state.get("ai_layout") or state.get("current_room_layout") or []) if isinstance(x, dict)]
    for idx, note in enumerate(state.get("current_room_annotations") or [], start=1):
        if not isinstance(note, dict):
            continue
        text_value = str(note.get("note") or "").strip()
        if not text_value:
            continue
        nearby = note.get("nearby_object_names") or []
        location = " · ".join(str(x) for x in nearby if str(x).strip())
        if not location and layout:
            try:
                px, py = float(note.get("x", 0)), float(note.get("y", 0))
                ranked = []
                for obj in layout:
                    cx = float(obj.get("x", 0)) + float(obj.get("width", 0)) / 2
                    cy = float(obj.get("y", 0)) + float(obj.get("height", 0)) / 2
                    ranked.append(((cx-px)**2 + (cy-py)**2, str(obj.get("name") or "가구")))
                ranked.sort(key=lambda t: t[0])
                location = " · ".join(name for _, name in ranked[:2])
            except (TypeError, ValueError):
                location = ""
        issues.append({"source": f"지점 메모 {idx}", "label": text_value, "location": location})
    return issues


@app.route("/api/prepare_visual_commands", methods=["POST"])
def api_prepare_visual_commands():
    """Fast first stage: resolve abstract improvement text into concrete visual commands."""
    _t0 = time.monotonic()
    state = get_state()
    comparison = state.get("comparison_images")
    if not comparison:
        return jsonify({"error": "먼저 개선안을 확정해 주세요."}), 400
    layout = state.get("ai_layout") or state.get("current_room_layout") or []
    if not layout:
        return jsonify({"error": "배치 정보를 찾을 수 없습니다."}), 400

    hazard_groups = comparison.get("hazard_groups", [])
    selected_items = []
    for group in hazard_groups:
        options = group.get("options", [])
        selected = group.get("selected", 0)
        selected = selected if isinstance(selected, int) and 0 <= selected < len(options) else 0
        action_plans = group.get("action_plans") or []
        selected_plan = action_plans[selected] if 0 <= selected < len(action_plans) else {}
        selected_items.append({
            "priority": group.get("display_number", group.get("priority")),
            "improvement": options[selected] if options else "",
            "floorplan_problem": group.get("floorplan_problem", ""),
            "target_object": group.get("target_object", "room"),
            "execution_steps": selected_plan.get("execution_steps") or [],
            "visual_addition_type": selected_plan.get("visual_addition_type", "none"),
        })

    commands = generate_concrete_visual_commands(
        room_label=comparison.get("room_label", "공간"),
        room_width=int(comparison.get("room_width", 720)),
        room_height=int(comparison.get("room_height", 460)),
        layout=layout,
        selected_items=selected_items,
    )
    comparison["visual_change_commands"] = commands
    logging.getLogger("app").info("[prepare_visual_commands] 총 %.1f초 소요", time.monotonic() - _t0)
    return jsonify({"ok": True, "visual_commands": commands})


@app.route("/api/generate_ai_before_after", methods=["POST"])
def api_generate_ai_before_after():
    """즉시 응답하고, 실제 이미지 생성은 백그라운드 스레드에서 진행합니다.

    이전에는 이 라우트가 이미지 생성이 끝날 때까지(최대 55초+) 응답을 안 주고 붙잡고
    있었는데, 브라우저 fetch가 그보다 먼저(58초) 포기해버리면 "서버는 결국 성공해서
    저장했지만(그래서 PDF엔 보임) 화면은 그 성공 응답을 못 받아 계속 로딩 중"인
    불일치가 생겼습니다. 이제는 요청을 받자마자 생성을 백그라운드로 던지고 즉시
    반환하며, 화면은 /api/check_ai_after_status를 가볍게 반복 확인(polling)해서
    "얼마나 오래 걸리든" 완료되는 즉시 반영합니다.
    """
    state = get_state()
    comparison = state.get("comparison_images")
    if not comparison:
        return jsonify({"error": "먼저 개선안을 확정해 주세요."}), 400

    layout = state.get("ai_layout") or state.get("current_room_layout") or []
    if not layout:
        return jsonify({"error": "배치 정보를 찾을 수 없습니다. 배치를 다시 확인해 주세요."}), 400

    hazard_groups = comparison.get("hazard_groups", [])
    recommendations = []
    selected_items = []
    for group in hazard_groups:
        options = group.get("options", [])
        selected = group.get("selected", 0)
        selected = selected if isinstance(selected, int) and 0 <= selected < len(options) else 0
        improvement = options[selected] if options else ""
        action_plans = group.get("action_plans") or []
        selected_plan = action_plans[selected] if 0 <= selected < len(action_plans) else {}
        priority = group.get("display_number", group.get("priority"))
        recommendations.append({"priority": priority, "improvement": improvement})
        selected_items.append({
            "priority": priority,
            "improvement": improvement,
            "floorplan_problem": group.get("floorplan_problem", ""),
            "target_object": group.get("target_object", "room"),
            "execution_steps": selected_plan.get("execution_steps") or [],
            "visual_addition_type": selected_plan.get("visual_addition_type", "none"),
        })

    room_width = int(comparison.get("room_width", 720))
    room_height = int(comparison.get("room_height", 460))
    room_label = comparison.get("room_label", "공간")
    payload = request.get_json(silent=True) or {}
    force_after = bool(payload.get("force"))

    before_2d_path = comparison.get("before_2d_path")
    if not before_2d_path or not (ROOT / "static" / before_2d_path).exists():
        canvas_px_2d, _ = pick_canvas_size(room_width, room_height)
        before_2d = render_floorplan_png(room_width, room_height, layout, canvas_px=canvas_px_2d, show_handles=False)
        before_2d_path = save_generated_image(before_2d, "before2d")
        comparison["before_2d_path"] = before_2d_path

    # BEFORE는 사용자 평면도를 바탕으로 AI가 생성한 실제 BEFORE 이미지만 화면에 표시합니다.
    # 코드 렌더링 평면도(before_2d)는 이미지 모델의 reference로만 사용하고 UI placeholder로 노출하지 않습니다.
    ai_before_path = comparison.get("photoreal_before_image_path")
    ai_before_ready = bool(ai_before_path and ai_before_path != before_2d_path and (ROOT / "static" / ai_before_path).exists())
    before_url = url_for("static", filename=ai_before_path) if ai_before_ready else None

    after_path = comparison.get("photoreal_after_image_path")
    if after_path and not force_after and (ROOT / "static" / after_path).exists():
        return jsonify({
            "ok": True, "status": "done",
            "before_ai_image_url": before_url,
            "after_ai_image_url": url_for("static", filename=after_path),
            "legend": wh.build_comparison_legend(hazard_groups),
            "before_mode": "deterministic_floorplan",
            "visual_commands": comparison.get("visual_change_commands") or [],
        })

    if comparison.get("photoreal_after_generating"):
        return jsonify({
            "ok": True, "status": "generating",
            "before_ai_image_url": before_url,
            "legend": wh.build_comparison_legend(hazard_groups),
            "before_mode": "deterministic_floorplan",
        })

    comparison["photoreal_after_generating"] = True
    comparison["photoreal_after_error"] = None
    if force_after:
        comparison["photoreal_before_image_path"] = None
        comparison["photoreal_after_image_path"] = None
        ai_before_path = None
        ai_before_ready = False

    reference_floorplan_png = render_floorplan_png(room_width, room_height, layout, show_handles=False)
    before_user_issues = _collect_before_image_user_issues(state)
    comparison["before_user_issues"] = before_user_issues
    visual_commands = comparison.get("visual_change_commands") or generate_concrete_visual_commands(
        room_label=room_label, room_width=room_width, room_height=room_height,
        layout=layout, selected_items=selected_items,
    )
    comparison["visual_change_commands"] = visual_commands

    sid = get_sid()
    logger = logging.getLogger("app")

    def _worker():
        started = time.monotonic()
        logger.info("[ai_after worker] 시작 (sid=%s)", sid[:8])
        try:
            # 1) 사용자가 직접 만든 평면도 + 체크리스트/직접 메모를 반영한 개선 전 공간을 먼저 생성합니다.
            live_at_start = SESSIONS.get(sid, {}).get("comparison_images") or comparison
            existing_before_path = live_at_start.get("photoreal_before_image_path")
            before_bytes = None
            if existing_before_path and existing_before_path != before_2d_path and (ROOT / "static" / existing_before_path).exists():
                before_bytes = (ROOT / "static" / existing_before_path).read_bytes()
                saved_before_path = existing_before_path
            else:
                before_bytes = generate_ai_before_from_floorplan_once(
                    room_label=room_label, room_width=room_width, room_height=room_height,
                    layout=layout, reference_floorplan_png=reference_floorplan_png,
                    user_issues=before_user_issues,
                    size=os.getenv("OPENAI_IMAGE_BEFORE_SIZE", "1024x1024"),
                )
                saved_before_path = save_generated_image(before_bytes, "ai_before")
                live = SESSIONS.get(sid, {}).get("comparison_images")
                if live is not None:
                    live["photoreal_before_image_path"] = saved_before_path

            # 2) 방금 생성한 BEFORE를 그대로 기준 이미지로 사용해 개선안을 실제 적용한 AFTER를 생성합니다.
            after_bytes = generate_ai_after_from_floorplan_once(
                room_label=room_label, room_width=room_width, room_height=room_height,
                layout=layout, recommendations=recommendations,
                size=os.getenv("OPENAI_IMAGE_AFTER_SIZE", "1024x1024"),
                reference_floorplan_png=before_bytes,
                visual_commands=visual_commands,
            )
            saved_path = save_generated_image(after_bytes, "ai_after")
            # 처음에는 "이 워커를 시작했을 때의 comparison 객체와 지금 세션에 살아있는
            # comparison_images가 정말 같은 객체인지"(is 비교)를 확인하고 나서만 결과를
            # 반영했는데, 이게 실제로 문제를 일으켰습니다: 사용자가 새로고침하거나 다시
            # 확정하면 comparison_images가 새 dict로 교체되고, 워커가 들고 있던 예전
            # comparison 객체는 아무도 참조하지 않는 고아 객체가 됩니다. 그러면 워커가
            # 성공해도 이 identity 비교가 항상 실패해서 결과를 어디에도 못 쓰고,
            # 폴링(/api/check_ai_after_status)은 영원히 "generating"만 보게 됩니다.
            # 이제는 객체가 같은지 따지지 않고, "쓰는 시점에 세션에 실제로 살아있는
            # comparison_images"를 다시 조회해서 거기에 무조건 최신 결과를 반영합니다.
            live = SESSIONS.get(sid, {}).get("comparison_images")
            if live is not None:
                live["photoreal_after_image_path"] = saved_path
                live["photoreal_image_path"] = saved_path
                live["photoreal_after_generating"] = False
                live["photoreal_after_error"] = None
            logger.info("[ai_after worker] 성공 (%.1f초 소요, sid=%s)", time.monotonic() - started, sid[:8])
        except Exception as exc:
            logger.error("[ai_after worker] 실패 (%.1f초 후, sid=%s): %s: %s",
                         time.monotonic() - started, sid[:8], type(exc).__name__, exc)
            live = SESSIONS.get(sid, {}).get("comparison_images")
            if live is not None:
                live["photoreal_after_error"] = str(exc)
                live["photoreal_after_generating"] = False

    threading.Thread(target=_worker, daemon=True).start()

    return jsonify({
        "ok": True, "status": "started",
        "before_ai_image_url": before_url,
        "legend": wh.build_comparison_legend(hazard_groups),
        "before_mode": "deterministic_floorplan",
        "visual_commands": visual_commands,
    })


@app.route("/api/check_ai_after_status", methods=["GET"])
def api_check_ai_after_status():
    """AFTER 이미지가 완성됐는지 가볍게(즉시 응답) 확인합니다. 화면은 이 엔드포인트를
    몇 초 간격으로 반복 호출해서, 백그라운드 생성이 얼마나 오래 걸리든 완료되는 즉시
    반영합니다 — "몇 분째 화면이 안 바뀐다"는 문제를 근본적으로 없애기 위한 구조입니다."""
    state = get_state()
    comparison = state.get("comparison_images")
    if not comparison:
        return jsonify({"error": "먼저 개선안을 확정해 주세요."}), 400

    before_2d_path = comparison.get("before_2d_path")
    before_path = comparison.get("photoreal_before_image_path")
    before_ready = bool(
        before_path
        and before_path != before_2d_path
        and (ROOT / "static" / before_path).exists()
    )
    before_url = url_for("static", filename=before_path) if before_ready else None

    after_path = comparison.get("photoreal_after_image_path")
    if after_path and (ROOT / "static" / after_path).exists():
        return jsonify({
            "ok": True, "status": "done",
            "before_ai_image_url": before_url,
            "after_ai_image_url": url_for("static", filename=after_path),
            "overlay_markers": comparison.get("web_overlay_markers") or [],
        })

    error = comparison.get("photoreal_after_error")
    if error:
        return jsonify({"ok": True, "status": "error", "error": error, "before_ai_image_url": before_url, "overlay_markers": comparison.get("web_overlay_markers") or []})

    if comparison.get("photoreal_after_generating"):
        # BEFORE가 먼저 완성되면 AFTER를 기다리는 동안 실제 AI BEFORE만 먼저 보여줄 수 있습니다.
        return jsonify({"ok": True, "status": "generating", "before_ai_image_url": before_url, "overlay_markers": comparison.get("web_overlay_markers") or []})

    return jsonify({"ok": True, "status": "idle"})


@app.route("/api/generate_photoreal_reference", methods=["POST"])
def api_generate_photoreal_reference():
    """개선 후(After) 모습을 AI 이미지 생성으로 "참고용" 실사 이미지 1장으로 만듭니다.
    좌표 기반 2D/3D 렌더링과 달리 정확한 배치를 보장하지 않는, 순수 참고/무드 이미지입니다.
    선택 사항이며, 사용자가 명시적으로 버튼을 눌렀을 때만 호출됩니다(비용·시간이 들기 때문)."""
    state = get_state()
    comparison = state.get("comparison_images")
    if not comparison:
        return jsonify({"error": "먼저 공간 비교를 생성해 주세요."}), 400

    layout = state.get("ai_layout") or state.get("current_room_layout") or []
    if not layout:
        return jsonify({"error": "배치 정보를 찾을 수 없습니다."}), 400

    hazard_groups = comparison.get("hazard_groups", [])
    recommendations = []
    for group in hazard_groups:
        options = group.get("options", [])
        selected = group.get("selected", 0)
        selected = selected if 0 <= selected < len(options) else 0
        recommendations.append({
            "priority": group.get("display_number", group.get("priority")),
            "improvement": options[selected] if options else "",
        })

    try:
        # 실사 참고 이미지가 평면도와 크게 달라지는 문제를 줄이기 위해, 현재 좌표 기반
        # 2D 평면도를 이미지 참조로 함께 전달합니다. SDK가 image edit를 지원하지 않으면
        # floorplan_image_gen 내부에서 자동으로 텍스트 기반 생성으로 fallback 합니다.
        marker_defs = []
        for group in hazard_groups:
            options = group.get("options", [])
            selected = group.get("selected", 0)
            selected = selected if 0 <= selected < len(options) else 0
            marker_defs.append({
                "priority": group.get("priority"),
                "display_number": group.get("display_number", group.get("priority")),
                "label": options[selected] if options else group.get("floorplan_problem", ""),
                "target_object": group.get("target_object", "room"),
                "marker_type": group.get("marker_type", "outside_list"),
                "icon_type": "none",
            })
        positioned = wh.calculate_marker_positions(
            layout, marker_defs,
            room_width=comparison.get("room_width", 720),
            room_height=comparison.get("room_height", 460),
        )
        reference_floorplan_png = render_floorplan_with_markers(
            comparison.get("room_width", 720),
            comparison.get("room_height", 460),
            layout, positioned, marker_style="done",
        )
        image_bytes = generate_photoreal_reference_image(
            room_label=comparison.get("room_label", ""),
            room_width=comparison.get("room_width", 720),
            room_height=comparison.get("room_height", 460),
            layout=layout,
            recommendations=recommendations,
            reference_floorplan_png=reference_floorplan_png,
            visual_commands=comparison.get("visual_change_commands") or [],
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    image_path = save_generated_image(image_bytes, "photoreal")
    comparison["photoreal_image_path"] = image_path
    return jsonify({
        "ok": True,
        "photoreal_image_url": url_for("static", filename=image_path),
        "legend": wh.build_comparison_legend(hazard_groups),
    })


@app.route("/api/download_report")
def api_download_report():
    """지금까지 계산된 위험요인 분석·RAG 근거·Action Plan 선택·생성된 이미지를 전부
    PDF 리포트 하나로 묶어서 다운로드합니다. 2D/3D 이미지는 결정론적 렌더링이라 즉시 다시
    그려서 최신 상태로 포함하고, 실사 참고 이미지는 이미 생성해 둔 게 있으면 그것만 포함합니다
    (새로 만들진 않음 — AI 호출 비용/시간 때문에 리포트 다운로드 시점에 강제로 만들지 않습니다)."""
    state = get_state()
    comparison = state.get("comparison_images")
    if not comparison:
        return jsonify({"error": "먼저 공간 비교를 생성해 주세요."}), 400

    before_photo_path = comparison.get("photoreal_before_image_path") or comparison.get("before_2d_path")
    after_photo_path = comparison.get("photoreal_after_image_path")
    before_ready = bool(before_photo_path and (ROOT / "static" / before_photo_path).exists())
    after_ready = bool(after_photo_path and (ROOT / "static" / after_photo_path).exists())
    if not after_ready:
        return jsonify({
            "error": "개선 후 AI 이미지 생성이 완료된 뒤 PDF 리포트를 다운로드할 수 있습니다. 화면의 이미지 생성 완료를 기다려 주세요."
        }), 409

    room = state.get("selected_room")
    layout = state.get("ai_layout") or state.get("current_room_layout") or []
    hazard_groups = comparison.get("hazard_groups", [])

    # 화면에 보여줬던 recommendation 상세(RAG 근거·개인 위험요인 등)를 다시 매칭합니다.
    ai_result = state.get("ai_result") or {}
    all_recs_by_id = {
        str(r.get("recommendation_id")): r
        for r in (ai_result.get("recommendations") or [])
        if isinstance(r, dict)
    }

    DIFFICULTY_LABELS = {1: "쉬움", 2: "보통", 3: "어려움"}

    hazard_sections = []
    for group in hazard_groups:
        options = group.get("options", [])
        selected = group.get("selected", 0)
        selected = selected if 0 <= selected < len(options) else 0
        improvement_text = options[selected] if options else ""
        action_plans = group.get("action_plans") or []

        # recommendation_id는 "대표(옵션 0)" 것만 hazard_groups에 저장돼 있으므로,
        # 다른 대안을 선택했을 때는 텍스트로 매칭을 시도합니다(정확한 매칭은 대표 옵션일 때만 보장).
        # improvement_text는 이미 clean_display_text를 거쳤으므로, 비교 대상도 똑같이 정제해야
        # "글자는 같은데 깨진 문자 하나 때문에 매칭 실패" 하는 일이 없습니다.
        rec_detail = all_recs_by_id.get(str(group.get("recommendation_id")), {})
        if wh.clean_display_text(rec_detail.get("improvement", "")) != improvement_text:
            rec_detail = next(
                (r for r in all_recs_by_id.values() if wh.clean_display_text(r.get("improvement", "")) == improvement_text),
                rec_detail,
            )

        # 대안 전부(선택한 것 포함)를 실행 계획 상세와 함께 담습니다 — "이 문제에 대해
        # 검토된 해결책이 여러 개였고, 그중 이걸 선택했다"는 걸 리포트에서 보여주기 위함입니다.
        alt_entries = []
        for i, opt_text in enumerate(options):
            plan = action_plans[i] if i < len(action_plans) else {}
            difficulty = int(plan.get("difficulty") or 0)

            height_note = ""
            if plan.get("affects_furniture_height") and plan.get("target_height_cm"):
                target = int(plan["target_height_cm"])
                height_note = (
                    f"목표 높이 약 {target}cm (문헌에 명시된 값)"
                    if plan.get("height_source") == "literature_text"
                    else f"약 {target}cm로 (AI 추정치, 문헌 근거 아님)"
                )

            price_min = int(plan.get("estimated_price_krw_min") or 0)
            price_max = int(plan.get("estimated_price_krw_max") or 0)
            price_note = f"약 {price_min:,}~{price_max:,}원 (AI 추정치)" if price_max else "추가 비용 없음"

            alt_entries.append({
                "text": opt_text,
                "selected": i == selected,
                "difficulty": difficulty,
                "difficulty_label": DIFFICULTY_LABELS.get(difficulty, "-"),
                "estimated_minutes": plan.get("estimated_minutes"),
                "requires_helper": bool(plan.get("requires_helper", False)),
                "required_items": plan.get("required_items") or [],
                "execution_steps": plan.get("execution_steps") or [],
                "expected_effects": plan.get("expected_effects") or [],
                "height_note": height_note,
                "price_note": price_note,
            })

        hazard_sections.append({
            "priority": group.get("priority"),
            "floorplan_problem": group.get("floorplan_problem", ""),
            "recommendation_source": rec_detail.get("recommendation_source", ""),
            "shap_factor": wh.humanize_shap_factor(rec_detail.get("shap_factor", "")),
            "reason": wh.clean_display_text(rec_detail.get("reason", "")),
            "rag_evidence": wh.clean_display_text(rec_detail.get("rag_evidence", "")),
            "alternatives": alt_entries,
        })

    # 리포트는 현재 서비스에서 실제 계산/입력된 값만 사용합니다.
    # 별도의 공간 위험등급이나 임의의 위험 감소율은 생성하지 않습니다.
    built = {}
    if layout:
        built = _build_comparison_images(
            layout, comparison.get("room_width", 720), comparison.get("room_height", 460), hazard_groups,
        )

    prediction = state.get("prediction") or {}
    shap_result = prediction.get("shap_result") or {}
    main_factors = wh.build_main_factor_cards(shap_result, top_n=12) if shap_result else []

    pdf_bytes = build_report_pdf({
        "room_label": comparison.get("room_label", ROOM_LABELS.get(room, room or "")),
        "ai_comment": comparison.get("ai_comment", ""),
        "generated_at": datetime.now(),
        "risk_score": prediction.get("risk_score"),
        "risk_level": prediction.get("risk_level"),
        "threshold_percent": prediction.get("threshold_percent"),
        "main_factors": main_factors,
        "hazard_sections": hazard_sections,
        "room_note": state.get("current_room_note", ""),
        "after_2d_path": built.get("after_2d_path"),
        "photoreal_before_image_path": before_photo_path,
        "photoreal_after_image_path": after_photo_path,
        "photoreal_image_path": after_photo_path,
        "supplementary_space_advice": state.get("supplementary_space_advice", {}),
    })

    filename = f"anneomeojib_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)

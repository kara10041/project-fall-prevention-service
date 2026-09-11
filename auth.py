"""
auth.py
=======
아주 단순한 파일 기반(JSON) 회원가입/로그인 모듈입니다.

- 비밀번호는 werkzeug의 salted hash로 저장합니다(평문 저장 없음).
- 세션 자체는 기존 app.py의 SESSIONS(서버 메모리) + 서명 쿠키(sid) 구조를 그대로 사용합니다.
  이 모듈은 "이메일/비밀번호가 맞는가"만 판단하고, 로그인 상태 유지는 app.py가 담당합니다.
- 프로덕션에서는 JSON 파일 대신 실제 DB(+ 레이트리밋, 이메일 인증 등)로 교체가 필요합니다.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

USERS_FILE = Path(__file__).resolve().parent / "data" / "users.json"
_LOCK = threading.Lock()

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _load_users() -> dict[str, dict]:
    if not USERS_FILE.exists():
        return {}
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_users(users: dict[str, dict]) -> None:
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def create_user(email: str, password: str, name: str) -> tuple[bool, str, dict | None]:
    """회원가입. 성공 시 (True, 메시지, 공개용 유저 정보) 반환."""
    email = (email or "").strip().lower()
    name = (name or "").strip()
    password = password or ""

    if not name:
        return False, "이름을 입력해 주세요.", None
    if not email or not EMAIL_RE.match(email):
        return False, "올바른 이메일 형식을 입력해 주세요.", None
    if len(password) < 6:
        return False, "비밀번호는 6자 이상이어야 합니다.", None

    with _LOCK:
        users = _load_users()
        if email in users:
            return False, "이미 가입된 이메일입니다.", None
        users[email] = {
            "name": name,
            "password_hash": generate_password_hash(password),
        }
        _save_users(users)

    return True, "회원가입이 완료되었습니다.", {"email": email, "name": name}


def verify_user(email: str, password: str) -> tuple[bool, str, dict | None]:
    """로그인. 성공 시 (True, 메시지, 공개용 유저 정보) 반환."""
    email = (email or "").strip().lower()
    password = password or ""

    if not email or not password:
        return False, "이메일과 비밀번호를 입력해 주세요.", None

    with _LOCK:
        users = _load_users()

    user = users.get(email)
    if not user or not check_password_hash(user.get("password_hash", ""), password):
        return False, "이메일 또는 비밀번호가 올바르지 않습니다.", None

    return True, "로그인되었습니다.", {"email": email, "name": user.get("name", "")}

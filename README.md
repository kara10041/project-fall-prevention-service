Team project archive

#### 안넘어집 (Flask)

Streamlit 화면을 Flask + 순수 HTML/CSS/JS로 마이그레이션한 버전. 백엔드 로직(`src/`, `integration/`)은 그대로 유지

#### 실행

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
python app.py

```

#### 구조

* `app.py`: Flask 라우팅 및 백엔드 연동
* `templates/`: Jinja2 UI 템플릿
* `static/`: 스타일 및 가구 배치 에디터(Vanilla JS)
* `integration/`, `src/`: 기존 모델·RAG·GPT 파이프라인

#### 주요 변경점

* 세션: 서명된 쿠키(`sid`) 기반 서버 메모리 관리
* UI: Plotly → 순수 SVG 게이지, iframe 에디터 → 단일 페이지 JS 통합
* 2026-08 기능: 공간별 Hazard 체크리스트 기반 개선 우선순위 산정 알고리즘 추가장 순서를 만들기 위한 서비스용 집계값임. 
* `/floorplan`에서는 우선순위 순서로 공간을 보여주며 사용자가 원하는 공간을 선택하면 기존 `/layout_editor` 흐름을 그대로 사용함

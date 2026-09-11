Team project archive

#### 안넘어집 (Flask)
노인/취약계층 낙상 위험 예측 및 공간 개선 플래너 서비스

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

* UI 변경 (- Figma:https://www.figma.com/proto/o1zAtHisev5TrBhTaOkVN9/ICT-%EC%9C%B5%ED%95%A9-%EA%B3%B5%EB%AA%A8%EC%A0%84?node-id=0-1&t=duIyzG0VWalNhKaW-1) )
* Streamlit 화면을 Flask + 순수 HTML/CSS/JS로 마이그레이션한 버전. 백엔드 로직(`src/`, `integration/`)은 그대로 유지
* 서비스 수정: `/floorplan`에서는 우선순위 순서로 공간을 보여주며 사용자가 원하는 공간을 선택하면 기존 `/layout_editor` 흐름을 그대로 사용함

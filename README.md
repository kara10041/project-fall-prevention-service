# 안넘어집 — Flask 버전 (HTML/CSS/JS)

Streamlit 기반이었던 서비스 화면을 **순수 HTML/CSS/JS + Flask**로 교체한 버전입니다.
백엔드 로직(모델 예측, SHAP, RAG 검색, GPT 통합 플래너, 공간위험 판정)은 **전혀 수정하지 않고**
기존 `integration/`, `src/` 코드를 그대로 호출합니다. 바뀐 것은 화면(UI)뿐입니다.

## 실행 방법

```bash
python -m venv .venv
source .venv/bin/activate      # Windows는 .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env   # 이미 .env 가 있다면 생략, OPENAI_API_KEY 등을 채워주세요
python app.py
```

브라우저에서 http://127.0.0.1:5000 접속.

- `OPENAI_API_KEY`가 없어도 앱은 정상 동작합니다. 다만 GPT 통합 플래너 단계는
  `GPT_FAIL_MODE=fallback` 설정에 따라 자동으로 대체(fallback) 로직으로 동작합니다
  (기존 프로젝트와 동일한 동작입니다).
- 개발 서버(`app.run(debug=True)`)는 로컬 테스트용입니다. 운영 배포 시에는
  gunicorn 등 WSGI 서버를 사용하세요.

## 구조

```
app.py                     Flask 라우팅 + API 엔드포인트 (실제 백엔드 파이프라인 호출)
web_helper.py               utils/helper.py 중 Streamlit에 의존하지 않는 부분만 이식
templates/                  Jinja2 HTML 템플릿 (기존 각 app_pages/*.py 1:1 대응)
  home.html                  홈
  input.html                 45개 변수 설문
  result.html                예측 결과 + SHAP 시각화 (SVG 게이지 + matplotlib 이미지)
  floorplan.html              공간 선택
  layout_editor.html          가구 배치 + 백엔드 확인질문 + AI 분석 결과
  room_detail.html            개선 전/후 placeholder + 추천 카드
static/css/style.css        기존 inject_custom_css()의 브랜드 스타일 이식
static/js/layout_editor.js  가구 드래그·리사이즈 편집기
  (기존 app_pages/room_layout_component/index.html 의 vanilla JS 로직을
   Streamlit postMessage 프로토콜 없이 같은 페이지에서 바로 쓰도록 이식)

src/, integration/, artifacts/, data/, config/   ← 기존 백엔드 코드/모델/데이터 그대로
```

## 세션 처리

Streamlit의 `st.session_state` 대신, 서명된 쿠키(`sid`)로 사용자를 식별하고
서버 메모리의 `SESSIONS` 딕셔너리에 사용자별 상태(설문값, 예측 결과, 배치, AI 분석 결과 등)를
저장합니다. 여러 프로세스로 스케일아웃하려면 Redis 등 외부 저장소로 교체가 필요합니다.

## 원본 대비 달라진 점

- Plotly 게이지 차트 → 순수 SVG로 다시 그렸습니다(디자인은 동일).
- Streamlit 컴포넌트(iframe + postMessage)로 만들어졌던 가구 배치 에디터를
  같은 페이지 안에서 바로 동작하는 JS로 이식했습니다(로직은 100% 동일).
- 3D 뷰어(GLB/glTF)는 원본과 동일하게 아직 placeholder입니다. `<model-viewer>`
  웹 컴포넌트를 붙이면 되는 위치는 `templates/room_detail.html` 안에 주석으로 남겨뒀습니다.
- `utils/helper.py`(Streamlit 전용 코드)는 제거하고 `web_helper.py`로 대체했습니다.
  ML/RAG/GPT 관련 함수는 전부 원본 파일(`integration/`, `src/`) 그대로입니다.

## 2026-08 공간 개선 우선순위 기능

예측 결과 뒤에 `/space_check` 단계가 추가되었습니다. 사용자가 거실·침실·주방·욕실의 Hazard 체크리스트를 입력하면, 기존 SHAP/interaction-Hazard bridge를 사용해 Hazard별 `preliminary_priority_raw`를 계산합니다. 이후 공간별 서비스 우선순위는 `최고 Hazard raw + 0.30 × 나머지 Hazard raw 합`으로 집계합니다. 이 공간 점수는 임상 위험도가 아니라 개선 권장 순서를 만들기 위한 서비스용 집계값입니다. `/floorplan`에서는 우선순위 순서로 공간을 보여주며 사용자가 원하는 공간을 선택하면 기존 `/layout_editor` 흐름을 그대로 사용합니다.

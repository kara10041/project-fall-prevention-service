# AI 이미지 생성 안정화 수정 (2026-08-19)

- 1단계 텍스트 AI의 `visual_commands`가 실제 AFTER 이미지 프롬프트에 전달되지 않던 연결 오류 수정.
- `generate_photoreal_reference_image()`에서 정의되지 않은 `visual_commands`를 참조하던 NameError 가능성 수정.
- 이미지 edit는 1회만 호출하고 네트워크 오류에 대한 자동 재시도/다중 fallback을 하지 않도록 유지.
- 속도 우선 기본값: `quality=low`, `input_fidelity=low`, `1024x1024`.
- 서버 이미지 요청 상한 55초 / 브라우저 요청 상한 58초로 정렬.
- SDK가 quality/input_fidelity 인자를 지원하지 않는 경우, HTTP 요청 전 TypeError에 한해 최소 인자로 호환 처리.
- Before는 좌표 기반 코드 렌더링, After만 AI 이미지 생성.
- PDF는 After AI 이미지가 생성된 뒤 활성화되며 Before 도면 + After AI 이미지를 포함.

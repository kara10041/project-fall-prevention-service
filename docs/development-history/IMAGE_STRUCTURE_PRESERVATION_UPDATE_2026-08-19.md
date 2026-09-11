# 이미지 구조 보존 / 최소 수정 업데이트

- BEFORE 프롬프트를 사용자 평면도 구조 복원 우선으로 재작성.
- 체크리스트/메모는 보조 맥락으로만 사용하고, 명시되지 않은 잡동사니/가구/장식을 임의 생성하지 않도록 제한.
- AFTER 프롬프트를 BEFORE 기반 최소 수정 방식으로 변경. 개선 명령과 무관한 가구/구조/스타일 변경 금지.
- AI 이미지 내부 번호/한글/화살표/경로/원형 강조 생성 금지.
- 번호와 짧은 개선 설명은 웹 UI overlay로 별도 표시.
- overlay 위치는 사용자 평면도 좌표에서 계산한 상대 위치를 사용.
- 구조 보존을 위해 BEFORE/AFTER image edit input fidelity 기본값을 high로 분리 설정 가능:
  - OPENAI_IMAGE_BEFORE_INPUT_FIDELITY
  - OPENAI_IMAGE_AFTER_INPUT_FIDELITY

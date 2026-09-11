# Clarification hazard + AI BEFORE placeholder fix

## 1. 추가 질문 후 `통합계획에 누락된 hazard instance` 오류
- 원래 전체 backend 분석 결과를 clarification follow-up 결과로 덮어쓰지 않도록 수정했습니다.
- clarification GPT가 일부 hazard step을 누락하거나 임시 ID를 반환해도 서버의 confirmed hazard ID/code/rank를 기준으로 정렬합니다.
- 누락된 step만 deterministic AI template로 보완한 후 기존 validator를 통과시킵니다.
- 따라서 HZ_002 같은 기존 hazard가 GPT 출력에서 빠져도 사용자 alert로 전체 흐름이 중단되지 않습니다.

## 2. AI 이미지 생성 전 코드 평면도 placeholder 제거
- deterministic floorplan은 이미지 모델 reference 용도로만 유지합니다.
- 메인 BEFORE 카드에는 코드 평면도를 먼저 표시하지 않습니다.
- 실제 AI BEFORE 파일이 생성된 경우에만 브라우저에 표시합니다.
- BEFORE가 AFTER보다 먼저 완성되면 상태 polling을 통해 실제 AI BEFORE만 먼저 표시합니다.
- PDF 활성화 로직은 실제 AFTER 이미지 로드 완료 이후를 유지합니다.

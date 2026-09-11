# 공간디자인 로직 수정 사항 (2026-08-18)

## 반영한 핵심 수정

1. **실제 공간위험을 SHAP=0 때문에 제거하지 않음**
   - 이전: `personal_risk_contribution × presence × confidence`
   - 변경: `environmental baseline × (1 + personal risk contribution) × presence × confidence`
   - 공간에서 확인된 위험은 유지하고 SHAP/interaction은 개인화 가중치로 사용합니다.

2. **RAG/AI를 전체 분석 단위가 아니라 hazard 단위로 선택**
   - RAG 매핑 hazard: `RAG_EVIDENCE`
   - RAG 미매핑 hazard: `AI_ONLY_NO_RAG`
   - 두 종류가 동시에 있으면 `HYBRID_RAG_AI`

3. **RAG 미매핑 위험 제거 및 재랭킹 금지**
   - 원래 `preliminary_priority_rank`를 `final_priority_rank`로 그대로 유지합니다.
   - RAG coverage가 위험 우선순위를 바꾸지 않습니다.

4. **SHAP need가 없는 객관적 공간위험도 RAG 검색 허용**
   - need가 없을 때 `hazard_code + room` 기준으로 일반 근거를 검색합니다.

5. **동일 hazard_code 다중 instance 충돌 방지**
   - GPT step/validator의 기본 키를 `hazard_instance_id`로 변경했습니다.
   - 같은 `LOOSE_RUG`가 여러 위치에 있어도 각각 독립적으로 유지됩니다.

6. **프론트엔드 adapter도 hybrid 결과를 동시에 표시**
   - RAG 카드와 AI-only 카드를 같은 고정 위험순위 안에서 함께 표시합니다.
   - 기존 app-level AI fallback은 hybrid 결과에서 중복 실행하지 않습니다.

## 확인한 테스트

- Python 전체 compile 통과
- SHAP=0 + 실제 hazard 유지 테스트 통과
- RAG 1개 + 미매핑 1개 hybrid 출력 테스트 통과
- 원래 hazard rank 유지 테스트 통과
- 동일 hazard_code의 서로 다른 `hazard_instance_id` validator 테스트 통과
- frontend adapter의 RAG + AI hybrid 카드 병합 테스트 통과

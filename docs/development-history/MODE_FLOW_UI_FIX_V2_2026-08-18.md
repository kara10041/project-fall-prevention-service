# Mode flow UI fix v2

- Manual and AI modes now use the same primary-action position below the floorplan.
- Manual mode: `현재 배치 안전성 점검하기`.
- AI mode: `AI에게 안전한 배치 추천받기`.
- Removed the post-generation safety-analysis step from AI mode.
- AI generation now returns deterministic validation for room-boundary violations and furniture overlap in the same request.
- AI result card shows the validation summary alongside SHAP/RAG-based layout rationale.
- AI mode flow is now: room/fixed structure -> requested furniture -> AI safe-layout recommendation -> review/edit.

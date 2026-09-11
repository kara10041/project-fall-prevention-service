# Action Plan Failed-to-fetch fix

- Action Plan GPT requests are now run in parallel instead of sequentially.
- Added a separate short OpenAI timeout (`OPENAI_ACTION_PLAN_TIMEOUT_SECONDS`, default 18s) and disabled retries for this synchronous UI request.
- If one GPT request times out/fails, the whole page no longer returns 502. A deterministic fallback action plan is generated from the already-approved improvement text.
- Supplementary free-note advice uses a shorter 12s timeout and already has a local fallback.
- The UI shows a warning when fallback action-plan metadata was used instead of failing the page.

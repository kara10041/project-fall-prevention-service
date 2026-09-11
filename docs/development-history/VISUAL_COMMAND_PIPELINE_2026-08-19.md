# Visual command pipeline update (2026-08-19)

- Added a first-stage text AI pass that converts abstract safety advice (e.g. 동선 확보, 가구 배치 수정) into short, concrete visual-edit commands.
- Added `/api/prepare_visual_commands` so the browser receives and displays those commands before image generation starts.
- AFTER image generation now receives only the pre-resolved visual commands plus the measured floorplan reference; the image model no longer has to decide the safety intervention itself.
- BEFORE remains deterministic/code-rendered and instant.
- Text-stage timeout defaults to 10 seconds; image-stage timeout defaults to 42 seconds, with browser stage timeouts of 12s and 46s respectively.
- Changing an alternative invalidates both the previous AFTER image and the cached visual commands.
- The concrete commands are shown in the Action Plan UI before/during AFTER rendering so vague phrases are translated into actionable changes.

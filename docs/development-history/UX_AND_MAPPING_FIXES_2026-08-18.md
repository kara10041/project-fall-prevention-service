# UX and mapping fixes - 2026-08-18

1. Layout editor usability
- Added 3-column editor: furniture controls / room canvas / edit & notes.
- Removed square furniture option; standard furniture presets now use rectangles.
- Fixed circle resizing so horizontal or vertical drag both resize the circle.
- Enlarged and separated the rotation handle; furniture drag no longer steals rotation-handle pointer events.
- Added Undo button and Ctrl+Z / Cmd+Z for add, move, resize, rotate, delete, reset and point-note moves/add/delete.

2. Spatial notes
- Added point-specific notes: enter text -> click location -> numbered pin.
- Pins can be dragged and deleted.
- Kept a separate room-wide free-text memo.
- Both note types are sent to the planner as context without creating artificial hazard ranks.

3. Human-readable hazard target labels
- Internal IDs such as room_living_room_01 are converted to Korean labels such as "거실 전체".
- Multi-target recommendations infer all related fixtures from recommendation text, e.g. "변기 및 샤워시설..." -> "변기 · 샤워시설 관련".
- Unknown hazard fallbacks no longer expose internal room/object IDs.

4. Improvement visualization readability
- 2D and 3D views are now stacked at full width instead of squeezed side-by-side.
- Improvement cards are shown before the diagrams and include target location labels.
- Deterministic 2D/3D improvement markers now retain the priority number and use a separate check badge.

5. Photoreal reference mapping
- Image prompt now requests numbered red callouts matching improvement priority numbers.
- The UI shows a numbered mapping legend directly below the photoreal image.

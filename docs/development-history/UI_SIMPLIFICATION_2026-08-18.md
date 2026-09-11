# UI simplification update (2026-08-18)

## What changed
- Removed the complex spatial annotation UI (doorway/step/wet/area markers).
- Added a simple `공간 메모` textarea instead.
- Kept the locked backend hazard checklist at the top.
- Switched furniture interaction to direct manipulation:
  - drag to move
  - drag bottom-right handle to resize
  - drag top-right circular handle to rotate
- Preserved room-specific quick-add furniture buttons.
- Stored `room_note` in session state and forwarded it into the floorplan payload as `ROOM_USER_NOTE`.

## Files changed
- `templates/layout_editor.html`
- `static/js/layout_editor.js`
- `static/css/style.css`
- `app.py`
- `integration/frontend_backend_adapter.py`

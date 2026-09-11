# AI Before image cache update

- BEFORE AI image is now cached by room label + room dimensions + physical layout.
- Changing improvement selections no longer invalidates BEFORE.
- The regenerate button regenerates AFTER only.
- BEFORE is regenerated only when the room/layout itself changes.
- The confirm-action-plan response returns a cached BEFORE URL immediately when available, so the UI can keep showing it while AFTER is generated.

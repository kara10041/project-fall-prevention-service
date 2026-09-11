// 안넘어집 공간 배치 편집기
// - 가구 드래그 이동 / 손잡이 크기조절 / 큰 회전 핸들
// - Ctrl+Z / 되돌리기
// - 지점 메모 + 방 전체 메모
(function () {
  const room = document.getElementById("room");
  const roomStage = document.getElementById("roomStage");
  const nameInput = document.getElementById("furnitureName");
  const shapeInput = document.getElementById("furnitureShape");
  const addButton = document.getElementById("addFurniture");
  const deleteButton = document.getElementById("deleteSelected");
  const resetButton = document.getElementById("resetLayout");
  const undoButton = document.getElementById("undoLayout");
  const itemCount = document.getElementById("itemCount");
  const emptyHint = document.getElementById("emptyHint");
  const message = document.getElementById("message");
  const pointNoteText = document.getElementById("pointNoteText");
  const startPointNoteButton = document.getElementById("startPointNote");
  const pointNoteList = document.getElementById("pointNoteList");
  const workspaceTitle = document.getElementById("workspaceTitle");
  const workspaceHelp = document.getElementById("workspaceHelp");
  const analyzeLayoutButton = document.getElementById("analyzeLayout");

  let selectedElement = null;
  let dragState = null;
  let resizeState = null;
  let rotateState = null;
  let noteDragState = null;
  let pointNoteMode = null;
  let sequence = 0;
  let noteSequence = 0;
  let logicalWidth = 720;
  let logicalHeight = 460;
  let historyStack = [];
  let restoring = false;

  const clamp = (v, min, max) => Math.max(min, Math.min(v, max));
  const displayScale = () => room.clientWidth > 0 ? room.clientWidth / logicalWidth : 1;

  function escapeText(value) {
    const d = document.createElement("div");
    d.textContent = String(value ?? "");
    return d.innerHTML;
  }

  function setRotation(el, deg) {
    const value = Number.isFinite(Number(deg)) ? Number(deg) : 0;
    el.dataset.rotation = String(value);
    el.style.transform = `rotate(${value}deg)`;
  }

  function applyShape(box, shape) {
    const normalized = ["rect", "circle", "door"].includes(shape) ? shape : "rect";
    box.dataset.shape = normalized;
    box.classList.remove("shape-circle", "shape-door");
    if (normalized === "circle") box.classList.add("shape-circle");
    if (normalized === "door") box.classList.add("shape-door");
  }

  function logicalRect(el) {
    const scaleX = logicalWidth / Math.max(1, room.clientWidth);
    const scaleY = logicalHeight / Math.max(1, room.clientHeight);
    return {
      x: Math.round(parseFloat(el.style.left || "0") * scaleX),
      y: Math.round(parseFloat(el.style.top || "0") * scaleY),
      width: Math.round(el.offsetWidth * scaleX),
      height: Math.round(el.offsetHeight * scaleY),
    };
  }

  function collectLayout() {
    return Array.from(room.querySelectorAll(".placed")).map((box) => ({
      id: box.dataset.itemId,
      name: box.dataset.name,
      ...logicalRect(box),
      shape: box.dataset.shape || "rect",
      rotation: Number(box.dataset.rotation || 0),
      kind: box.dataset.kind || "furniture",
      generated_by: box.dataset.generatedBy || "",
    }));
  }

  function collectPointNotes() {
    return Array.from(room.querySelectorAll(".point-note-marker")).map((pin) => {
      const r = logicalRect(pin);
      return {
        id: pin.dataset.noteId,
        type: "point_note",
        label: "지점 메모",
        note: pin.dataset.note || "",
        x: r.x + Math.round(r.width / 2),
        y: r.y + Math.round(r.height / 2),
        width: 0,
        height: 0,
        geometry: "point",
      };
    });
  }

  function snapshot() {
    return {
      layout: collectLayout(),
      pointNotes: collectPointNotes(),
    };
  }

  function pushHistory() {
    if (restoring) return;
    const snap = JSON.stringify(snapshot());
    if (historyStack[historyStack.length - 1] !== snap) {
      historyStack.push(snap);
      if (historyStack.length > 60) historyStack.shift();
    }
    updateUndoButton();
  }

  function updateUndoButton() {
    if (!undoButton) return;
    undoButton.disabled = historyStack.length === 0;
  }

  function undo() {
    if (!historyStack.length) {
      message.textContent = "되돌릴 작업이 없습니다.";
      return;
    }
    const previous = JSON.parse(historyStack.pop());
    restoreState(previous);
    updateUndoButton();
    message.textContent = "이전 배치로 되돌렸습니다.";
  }

  function selectBox(box) {
    room.querySelectorAll(".placed").forEach((el) => el.classList.remove("selected"));
    selectedElement = box && box.classList.contains("placed") ? box : null;
    if (selectedElement) selectedElement.classList.add("selected");
  }

  function updateCount() {
    const count = room.querySelectorAll(".placed").length;
    itemCount.textContent = `가구 ${count}개`;
    const noteCount = room.querySelectorAll(".point-note-marker").length;
    emptyHint.style.display = count || noteCount ? "none" : "flex";
  }

  function renderPointNoteList() {
    if (!pointNoteList) return;
    const notes = Array.from(room.querySelectorAll(".point-note-marker"));
    if (!notes.length) {
      pointNoteList.innerHTML = '<p class="caption" style="margin:6px 0 0;">추가된 지점 메모가 없습니다.</p>';
      return;
    }
    pointNoteList.innerHTML = notes.map((pin, idx) => `
      <div class="point-note-row">
        <span class="point-note-num">${idx + 1}</span>
        <span class="point-note-copy">${escapeText(pin.dataset.note || "")}</span>
        <button type="button" class="point-note-delete" data-note-id="${escapeText(pin.dataset.noteId)}" aria-label="지점 메모 삭제">×</button>
      </div>`).join("");
    pointNoteList.querySelectorAll(".point-note-delete").forEach((btn) => {
      btn.addEventListener("click", () => {
        const pin = room.querySelector(`.point-note-marker[data-note-id="${CSS.escape(btn.dataset.noteId)}"]`);
        if (!pin) return;
        pushHistory();
        pin.remove();
        renumberPointNotes();
        updateCount();
        renderPointNoteList();
      });
    });
  }

  function renumberPointNotes() {
    room.querySelectorAll(".point-note-marker").forEach((pin, idx) => {
      pin.textContent = String(idx + 1);
      pin.title = pin.dataset.note || "지점 메모";
    });
  }

  function applyRoomSize(nextWidth, nextHeight) {
    const prevScale = displayScale();
    logicalWidth = Math.max(200, Number(nextWidth || 720));
    logicalHeight = Math.max(200, Number(nextHeight || 460));
    const availableWidth = Math.max(300, roomStage.clientWidth - 8);
    const nextScale = Math.min(1, availableWidth / logicalWidth);
    room.style.width = `${Math.round(logicalWidth * nextScale)}px`;
    room.style.height = `${Math.round(logicalHeight * nextScale)}px`;

    updateGrid();

    if (prevScale > 0 && Math.abs(nextScale - prevScale) > 0.0001) {
      const ratio = nextScale / prevScale;
      room.querySelectorAll(".placed,.point-note-marker").forEach((el) => {
        el.style.left = `${parseFloat(el.style.left || 0) * ratio}px`;
        el.style.top = `${parseFloat(el.style.top || 0) * ratio}px`;
        if (el.classList.contains("placed")) {
          el.style.width = `${Math.max(12, el.offsetWidth * ratio)}px`;
          el.style.height = `${Math.max(12, el.offsetHeight * ratio)}px`;
        }
      });
    }
  }


  function updateGrid() {
    if (!room) return;
    const scale = displayScale();
    const minor = Math.max(8, 50 * scale);
    const major = Math.max(minor * 2, 100 * scale);
    room.style.setProperty("--grid-minor", `${minor}px`);
    room.style.setProperty("--grid-major", `${major}px`);
  }
  function pointerAngleFromCenter(el, clientX, clientY) {
    const left = parseFloat(el.style.left || 0);
    const top = parseFloat(el.style.top || 0);
    const rr = room.getBoundingClientRect();
    const cx = rr.left + left + el.offsetWidth / 2;
    const cy = rr.top + top + el.offsetHeight / 2;
    return Math.atan2(clientY - cy, clientX - cx);
  }

  function snapRotation(deg) {
    return Math.round(deg / 15) * 15;
  }

  function placePointNoteAtEvent(event) {
    if (!pointNoteMode) return false;
    const rr = room.getBoundingClientRect();
    const scale = displayScale();
    pushHistory();
    createPointNote(
      pointNoteMode.note,
      (event.clientX - rr.left) / scale,
      (event.clientY - rr.top) / scale,
    );
    pointNoteMode = null;
    startPointNoteButton?.classList.remove("active");
    if (startPointNoteButton) startPointNoteButton.textContent = "설명 입력 후 지점 찍기";
    if (pointNoteText) pointNoteText.value = "";
    selectBox(null);
    message.textContent = "지점 메모를 추가했습니다. 번호 핀을 드래그해 위치를 옮길 수 있습니다.";
    return true;
  }

  // 지점 메모 모드에서는 가구 위를 클릭해도 가구 선택/이동보다 메모 찍기가 먼저 처리되어야 합니다.
  // capture=true로 방 내부의 pointerdown을 가장 먼저 가로챕니다.
  room.addEventListener("pointerdown", (event) => {
    if (!pointNoteMode) return;
    if (!room.contains(event.target)) return;
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
    placePointNoteAtEvent(event);
  }, true);

  function bindFurniture(box, resizeHandle, rotateHandle) {
    box.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".resize-handle") || event.target.closest(".rotate-handle")) return;
      event.preventDefault();
      event.stopPropagation();
      pushHistory();
      selectBox(box);
      const rr = room.getBoundingClientRect();
      const left = parseFloat(box.style.left || 0);
      const top = parseFloat(box.style.top || 0);
      dragState = {
        element: box,
        pointerId: event.pointerId,
        offsetX: event.clientX - (rr.left + left),
        offsetY: event.clientY - (rr.top + top),
      };
      box.setPointerCapture(event.pointerId);
    });

    resizeHandle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      pushHistory();
      selectBox(box);
      resizeState = {
        element: box,
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        startWidth: box.offsetWidth,
        startHeight: box.offsetHeight,
      };
      resizeHandle.setPointerCapture(event.pointerId);
    });

    rotateHandle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      pushHistory();
      selectBox(box);
      rotateState = {
        element: box,
        pointerId: event.pointerId,
        startRotation: Number(box.dataset.rotation || 0),
        startAngle: pointerAngleFromCenter(box, event.clientX, event.clientY),
      };
      rotateHandle.setPointerCapture(event.pointerId);
    });
  }

  function createFurniture(name, initial = {}, shape = "rect", {select = true} = {}) {
    const cleanName = String(name || "").trim();
    if (!cleanName) {
      message.textContent = "가구 이름을 입력해 주세요.";
      return null;
    }

    const normalizedShape = ["rect", "circle", "door"].includes(shape) ? shape : "rect";
    const box = document.createElement("div");
    box.className = "placed";
    box.dataset.itemId = initial.id || `furniture-${Date.now()}-${sequence++}`;
    box.dataset.name = cleanName;
    box.dataset.kind = initial.kind || "furniture";
    box.dataset.generatedBy = initial.generated_by || "";
    if (box.dataset.kind === "fixed") box.classList.add("fixed-structure");

    const label = document.createElement("span");
    label.className = "furniture-label";
    label.textContent = box.dataset.kind === "fixed" ? `고정 · ${cleanName}` : cleanName;

    const resizeHandle = document.createElement("span");
    resizeHandle.className = "resize-handle";
    resizeHandle.title = "드래그해서 크기 조절";

    const rotateHandle = document.createElement("span");
    rotateHandle.className = "rotate-handle";
    rotateHandle.title = "드래그해서 회전";
    rotateHandle.textContent = "↻";

    box.append(label, resizeHandle, rotateHandle);

    const count = room.querySelectorAll(".placed").length;
    const offset = (count % 7) * 24;
    box.style.left = `${initial.x ?? 36 + offset}px`;
    box.style.top = `${initial.y ?? 36 + offset}px`;

    let w = Number(initial.width ?? (normalizedShape === "door" ? 82 : normalizedShape === "circle" ? 80 : 120));
    let h = Number(initial.height ?? (normalizedShape === "door" ? 82 : normalizedShape === "circle" ? 80 : 68));
    if (normalizedShape === "circle") {
      const side = Math.max(w, h);
      w = side;
      h = side;
    }
    box.style.width = `${w}px`;
    box.style.height = `${h}px`;

    applyShape(box, initial.shape || normalizedShape);
    setRotation(box, Number(initial.rotation || 0));
    bindFurniture(box, resizeHandle, rotateHandle);
    room.appendChild(box);
    if (select) selectBox(box);
    updateCount();
    message.textContent = "";
    return box;
  }

  function createPointNote(note, logicalX, logicalY, id = null) {
    const text = String(note || "").trim();
    if (!text) return null;
    const scale = displayScale();
    const pin = document.createElement("div");
    pin.className = "point-note-marker";
    pin.dataset.noteId = id || `point-note-${Date.now()}-${noteSequence++}`;
    pin.dataset.note = text;
    const size = 30;
    pin.style.left = `${clamp(logicalX * scale - size / 2, 0, Math.max(0, room.clientWidth - size))}px`;
    pin.style.top = `${clamp(logicalY * scale - size / 2, 0, Math.max(0, room.clientHeight - size))}px`;
    pin.style.width = `${size}px`;
    pin.style.height = `${size}px`;
    pin.title = text;
    pin.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      pushHistory();
      const rr = room.getBoundingClientRect();
      const left = parseFloat(pin.style.left || 0);
      const top = parseFloat(pin.style.top || 0);
      noteDragState = {
        element: pin,
        pointerId: event.pointerId,
        offsetX: event.clientX - (rr.left + left),
        offsetY: event.clientY - (rr.top + top),
      };
      pin.setPointerCapture(event.pointerId);
    });
    room.appendChild(pin);
    renumberPointNotes();
    updateCount();
    renderPointNoteList();
    return pin;
  }

  function restoreState(state) {
    restoring = true;
    room.querySelectorAll(".placed,.point-note-marker").forEach((el) => el.remove());
    const scale = displayScale();
    (state?.layout || []).forEach((item) => createFurniture(item.name, {
      id: item.id,
      x: (Number(item.x) || 0) * scale,
      y: (Number(item.y) || 0) * scale,
      width: (Number(item.width) || 120) * scale,
      height: (Number(item.height) || 68) * scale,
      shape: item.shape,
      rotation: item.rotation,
      kind: item.kind || "furniture",
      generated_by: item.generated_by || "",
    }, item.shape || "rect", {select: false}));
    (state?.pointNotes || []).forEach((note) => createPointNote(
      note.note,
      Number(note.x) || 0,
      Number(note.y) || 0,
      note.id,
    ));
    selectBox(null);
    renumberPointNotes();
    updateCount();
    renderPointNoteList();
    restoring = false;
  }

  addButton.addEventListener("click", () => {
    pushHistory();
    createFurniture(nameInput.value, {}, shapeInput.value);
    nameInput.value = "";
  });

  nameInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      addButton.click();
    }
  });

  document.querySelectorAll(".preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      pushHistory();
      createFurniture(btn.dataset.name, {}, btn.dataset.shape || "rect");
    });
  });

  if (startPointNoteButton) {
    startPointNoteButton.addEventListener("click", () => {
      const note = String(pointNoteText?.value || "").trim();
      if (!note) {
        message.textContent = "먼저 지점에 대한 설명을 입력해 주세요.";
        pointNoteText?.focus();
        return;
      }
      pointNoteMode = {note};
      startPointNoteButton.classList.add("active");
      startPointNoteButton.textContent = "방 안에서 위치를 클릭하세요";
      message.textContent = "메모할 지점을 방 안에서 한 번 클릭해 주세요.";
    });
  }

  deleteButton.addEventListener("click", () => {
    if (!selectedElement) {
      message.textContent = "삭제할 가구를 먼저 선택해 주세요.";
      return;
    }
    pushHistory();
    selectedElement.remove();
    selectedElement = null;
    updateCount();
    message.textContent = "";
  });

  resetButton.addEventListener("click", () => {
    if (!room.querySelector(".placed,.point-note-marker")) return;
    pushHistory();
    room.querySelectorAll(".placed,.point-note-marker").forEach((el) => el.remove());
    selectedElement = null;
    updateCount();
    renderPointNoteList();
    message.textContent = "배치와 지점 메모를 초기화했습니다.";
  });

  undoButton?.addEventListener("click", undo);

  document.addEventListener("keydown", (event) => {
    const isUndo = (event.ctrlKey || event.metaKey) && !event.shiftKey && event.key.toLowerCase() === "z";
    if (!isUndo) return;
    const tag = document.activeElement?.tagName?.toLowerCase();
    if (["input", "textarea"].includes(tag)) return;
    event.preventDefault();
    undo();
  });

  room.addEventListener("click", (event) => {
    if (event.target !== room && event.target !== emptyHint) return;
    selectBox(null);
  });

  window.addEventListener("pointermove", (event) => {
    if (dragState && dragState.pointerId === event.pointerId) {
      const el = dragState.element;
      const rr = room.getBoundingClientRect();
      let x = event.clientX - rr.left - dragState.offsetX;
      let y = event.clientY - rr.top - dragState.offsetY;
      x = clamp(x, 0, Math.max(0, room.clientWidth - el.offsetWidth));
      y = clamp(y, 0, Math.max(0, room.clientHeight - el.offsetHeight));
      el.style.left = `${x}px`;
      el.style.top = `${y}px`;
    }

    if (noteDragState && noteDragState.pointerId === event.pointerId) {
      const el = noteDragState.element;
      const rr = room.getBoundingClientRect();
      let x = event.clientX - rr.left - noteDragState.offsetX;
      let y = event.clientY - rr.top - noteDragState.offsetY;
      x = clamp(x, 0, Math.max(0, room.clientWidth - el.offsetWidth));
      y = clamp(y, 0, Math.max(0, room.clientHeight - el.offsetHeight));
      el.style.left = `${x}px`;
      el.style.top = `${y}px`;
    }

    if (resizeState && resizeState.pointerId === event.pointerId) {
      const el = resizeState.element;
      const dx = event.clientX - resizeState.startX;
      const dy = event.clientY - resizeState.startY;
      const left = parseFloat(el.style.left || 0);
      const top = parseFloat(el.style.top || 0);
      const maxWidth = Math.max(20, room.clientWidth - left);
      const maxHeight = Math.max(20, room.clientHeight - top);

      if (el.dataset.shape === "circle") {
        // 원형은 가로/세로 어느 방향으로만 끌어도 크기가 바뀌도록 dominant delta를 사용합니다.
        const delta = Math.abs(dx) >= Math.abs(dy) ? dx : dy;
        const side = clamp(resizeState.startWidth + delta, 24, Math.min(maxWidth, maxHeight));
        el.style.width = `${side}px`;
        el.style.height = `${side}px`;
      } else {
        const nextWidth = clamp(resizeState.startWidth + dx, 20, maxWidth);
        const nextHeight = clamp(resizeState.startHeight + dy, 20, maxHeight);
        el.style.width = `${nextWidth}px`;
        el.style.height = `${nextHeight}px`;
      }
    }

    if (rotateState && rotateState.pointerId === event.pointerId) {
      const el = rotateState.element;
      const currentAngle = pointerAngleFromCenter(el, event.clientX, event.clientY);
      const deltaDeg = ((currentAngle - rotateState.startAngle) * 180) / Math.PI;
      setRotation(el, snapRotation(rotateState.startRotation + deltaDeg));
    }
  });

  function finishPointer(event) {
    if (dragState && dragState.pointerId === event.pointerId) dragState = null;
    if (resizeState && resizeState.pointerId === event.pointerId) resizeState = null;
    if (rotateState && rotateState.pointerId === event.pointerId) rotateState = null;
    if (noteDragState && noteDragState.pointerId === event.pointerId) noteDragState = null;
  }

  window.addEventListener("pointerup", finishPointer);
  window.addEventListener("pointercancel", finishPointer);
  window.addEventListener("resize", () => applyRoomSize(logicalWidth, logicalHeight));

  window.RoomLayoutEditor = {
    applyRoomSize,
    updateGrid,
    collectLayout,
    collectPointNotes,
    getLogicalSize: () => ({width: logicalWidth, height: logicalHeight}),
    checkpoint: pushHistory,
    undo,
    restore(saved) {
      restoreState({layout: saved || [], pointNotes: collectPointNotes()});
    },
    restorePointNotes(saved) {
      restoreState({layout: collectLayout(), pointNotes: saved || []});
    },
    clearHistory() {
      historyStack = [];
      updateUndoButton();
    },
    replaceLayout(layout) {
      restoreState({layout: layout || [], pointNotes: []});
    },
  };

  applyRoomSize(720, 460);
  updateCount();
  renderPointNoteList();
  updateUndoButton();
})();

/*
 * AMI Hybrid Vision System — Group 1
 * File: app-utils.js
 *
 * Shared utility functions used across the modules
 *
 * Utilities (added incrementally):
 *   getConf()        — read the current confidence threshold (0–1)
 *   renderBboxes()   — paint YOLO bounding boxes over a frame image
 *   bindSlider()     — wire a range slider with drag-tracking; returns isDragging()
 *   fetchLiveFrame() — fetch a single-frame YOLO result from the microservice
 */

/* ── Live single-frame detection ─────────────────────────────────────────── */

// Fetches detections for one frame from /api/detect_frame and calls onResult
// with the boxes array. Discards the response if isCurrent(seq, n) returns
// false — meaning the user has already moved to a different frame while the
// request was in flight (race-condition guard).
async function fetchLiveFrame(seq, n, model, isCurrent, onResult) {
  try {
    const r = await fetch('/api/detect_frame/' + seq + '/' + n + '?model=' + model);
    if (!r.ok) return;
    const data = await r.json();
    if (!isCurrent(seq, n)) return;
    onResult(data.boxes || []);
  } catch(e) {}
}

/* ── Confidence threshold ─────────────────────────────────────────────────── */

// Returns the shared confidence threshold as a 0–1 float.
// sharedConfPct is the integer 0–100 slider value defined in app.js.
function getConf() {
  return sharedConfPct / 100;
}

/* ── Slider drag tracking ─────────────────────────────────────────────────── */

// Wires a range <input> so dragging it doesn't cause the play timer to fight
// the user's position. Returns { isDragging() } so setFrame() and play timers
// can skip their slider update while the user is actively dragging.
//   onDrag(value)   — called on every 'input' event (update labels only)
//   onCommit(value) — called on pointerup (update labels + trigger image load)
function bindSlider(slId, onDrag, onCommit) {
  const sl = document.getElementById(slId);
  let dragging = false;
  sl.addEventListener('pointerdown', () => { dragging = true; });
  document.addEventListener('pointerup', () => {
    if (!dragging) return;
    dragging = false;
    onCommit(+sl.value);
  });
  sl.addEventListener('input', function() { onDrag(+this.value); });
  return { isDragging: () => dragging };
}

/* ── Bounding box rendering ───────────────────────────────────────────────── */

// Renders an array of already-filtered detection boxes as absolutely-positioned
// divs inside containerEl, scaled to the natural pixel dimensions of imgEl.
// boxes: [{bbox:[x,y,w,h], class, confidence}, ...]
function renderBboxes(boxes, imgEl, containerEl) {
  const imgW = imgEl.naturalWidth || 1280, imgH = imgEl.naturalHeight || 720;
  containerEl.innerHTML = boxes.map(d => {
    const [x, y, w, h] = d.bbox;
    const px = (x/imgW*100).toFixed(2)+'%', py = (y/imgH*100).toFixed(2)+'%';
    const pw = (w/imgW*100).toFixed(2)+'%', ph = (h/imgH*100).toFixed(2)+'%';
    return `<div class="bbox" style="left:${px};top:${py};width:${pw};height:${ph}"><span class="bbox-lbl">${d.class} ${d.confidence.toFixed(2)}</span></div>`;
  }).join('');
}

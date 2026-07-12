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

/* ── Confidence timeline ──────────────────────────────────────────────────── */

// Builds a Float32Array[N] of max confidence per e2vid frame index.
// opts.secondaryCount + slope/offset/calibrated: maps secondary frame space
//   to e2vid space. inverse=true iterates detections (hyper→e2vid);
//   inverse=false forward-scans e2vid frames (fusion→e2vid).
function buildConfArray(dets, N, opts = {}) {
  if (!dets || !N) return null;
  const arr = new Float32Array(N);
  const { slope, offset, calibrated, secondaryCount, inverse } = opts;
  if (secondaryCount && secondaryCount !== N) {
    if (inverse) {
      dets.forEach(d => {
        const ef = (calibrated && slope)
          ? Math.max(0, Math.min(N-1, Math.round((d.frame - offset) / slope)))
          : Math.round(d.frame * N / secondaryCount);
        if (ef >= 0 && ef < N && d.confidence > arr[ef]) arr[ef] = d.confidence;
      });
    } else {
      const map = new Map();
      dets.forEach(d => { if (!map.has(d.frame) || d.confidence > map.get(d.frame)) map.set(d.frame, d.confidence); });
      for (let ef = 0; ef < N; ef++) {
        const sf = calibrated
          ? Math.max(0, Math.min(secondaryCount-1, Math.round(slope*ef+offset)))
          : Math.round(ef * secondaryCount / N);
        const c = map.get(sf) || 0;
        if (c > arr[ef]) arr[ef] = c;
      }
    }
  } else {
    dets.forEach(d => { if (d.frame < N && d.confidence > arr[d.frame]) arr[d.frame] = d.confidence; });
  }
  return arr;
}

// Draws a multi-line confidence sparkline with a current-frame marker and legend.
// lines: [{arr: Float32Array|null, color: string, label: string}, ...]
function drawConfTimeline(canvasId, frame, max, lines) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const W = canvas.width = canvas.offsetWidth;
  const H = canvas.height = canvas.offsetHeight;
  if (!W || !H) return;
  const ctx = canvas.getContext('2d');
  const dark = document.documentElement.dataset.theme === 'dark' ||
    (!document.documentElement.dataset.theme && window.matchMedia('(prefers-color-scheme:dark)').matches);
  ctx.fillStyle = dark ? '#1e2530' : '#f1f3f5';
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = dark ? 'rgba(200,200,200,0.15)' : 'rgba(0,0,0,0.1)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, H*0.5); ctx.lineTo(W, H*0.5); ctx.stroke();
  const N = max + 1;
  lines.forEach(({arr, color}) => {
    if (!arr) return;
    ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.5;
    let first = true;
    for (let i = 0; i < N; i++) {
      const x = i / (N-1||1) * W, y = H - arr[i] * H;
      if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
  if (max > 0) {
    const x = frame / max * W;
    ctx.strokeStyle = 'rgba(220,38,38,0.75)'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  // Legend
  ctx.font = '9px sans-serif';
  let lx = 6;
  lines.forEach(({arr, color, label}) => {
    if (!arr || !label) return;
    ctx.fillStyle = color;
    ctx.fillRect(lx, 4, 10, 7);
    ctx.fillStyle = dark ? 'rgba(255,255,255,0.7)' : 'rgba(0,0,0,0.55)';
    ctx.fillText(label, lx+13, 11);
    lx += 13 + ctx.measureText(label).width + 8;
  });
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

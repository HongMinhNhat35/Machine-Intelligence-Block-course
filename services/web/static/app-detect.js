/*
 * AMI Hybrid Vision System — Group 1
 * File: app-detect.js
 *
 * Detection screen — loads cached or live YOLO detections for the selected
 * sequence and model, overlays bounding boxes on reconstructed frames,
 * and provides play/pause/scrub controls.
 *
 * Load order: after app-utils.js, app-dataset.js, and app-recon.js, before app.js
 * Globals consumed: selSeq, sharedConfPct (app.js), getConf, fetchLiveFrame,
 *                   renderBboxes, bindSlider (app-utils.js)
 * Globals exported: dtRenderBboxes, dtLoadImage, dtStop, detLoad
 *                   — called from app.js (go() and tab navigation hook)
 */

/* ── State ────────────────────────────────────────────────────────────────── */

let dtPlaying=false, dtTimer=null, dtFrame=0, dtMax=0, dtLoadedSeq=null;
let dtDetections=null, dtImgDebounce=null, dtLiveMode=false, dtLiveBoxes=[];

/* ── Rendering ────────────────────────────────────────────────────────────── */

// Filters stored or live boxes by confidence and draws them over det-img2.
function dtRenderBboxes(){
  const bboxDiv=document.getElementById('det-bboxes');
  if(!bboxDiv) return;
  const conf=getConf();
  let boxes;
  if(dtLiveMode){
    boxes=(dtLiveBoxes||[]).filter(d=>d.confidence>=conf);
  } else {
    if(!dtDetections){ bboxDiv.innerHTML=''; document.getElementById('det-count').textContent='—'; return; }
    boxes=dtDetections.filter(d=>d.frame===dtFrame && d.confidence>=conf);
  }
  document.getElementById('det-count').textContent=boxes.length ? boxes.length+' detection'+(boxes.length===1?'':'s') : 'no detections';
  renderBboxes(boxes, document.getElementById('det-img2'), bboxDiv);
}

// Returns the dataset-m value of the currently selected model radio button.
function dtGetModel(){
  const c=document.querySelector('[name="dm"]:checked'); return c?c.dataset.m:'e2vid';
}

/* ── Image loading ────────────────────────────────────────────────────────── */

// Debounced frame loader — waits 60 ms before setting img.src to avoid
// flooding the server during rapid scrubbing. In live mode, triggers a
// YOLO inference request after the image loads.
function dtLoadImage(){
  const model=dtGetModel();
  if(model==='hypere2vid' && selSeq && !selSeq.hypere2vid_done){
    document.getElementById('det-img').src='';
    document.getElementById('det-img2').src='';
    document.getElementById('det-bboxes').innerHTML='';
    document.getElementById('det-count').textContent='—';
    return;
  }
  clearTimeout(dtImgDebounce);
  const seq=selSeq.id, n=dtFrame;
  const isFusion=model==='fusion';
  if(dtLiveMode) dtLiveBoxes=[];
  dtImgDebounce=setTimeout(()=>{
    if(dtFrame!==n) return;
    const src=isFusion ? '/frames_rgb/'+seq+'/'+n : '/frames/'+seq+'/'+n+'?model='+model;
    document.getElementById('det-img').src=src;
    const img2=document.getElementById('det-img2');
    img2.onload=()=>{ dtRenderBboxes(); if(dtLiveMode) fetchLiveFrame(seq, n, model, (s,f)=>dtFrame===f&&dtLoadedSeq===s, boxes=>{dtLiveBoxes=boxes;dtRenderBboxes();}); };
    img2.src=src;
  }, 60);
}

/* ── Frame navigation ─────────────────────────────────────────────────────── */

// Clamps n to [0, dtMax], updates pill and slider, and triggers dtLoadImage
// unless the user is actively dragging the slider.
function dtSetFrame(n){
  dtFrame=Math.max(0,Math.min(n,dtMax));
  document.getElementById('det-pill').textContent='frame '+dtFrame+' / '+dtMax;
  const gi=document.getElementById('det-goto'); if(gi) gi.value=dtFrame;
  if(!dtSlider.isDragging()){
    document.getElementById('det-sl').value=dtFrame;
    dtLoadImage();
  }
}

// Stops the play timer and resets the Play button label.
function dtStop(){
  dtPlaying=false;
  clearInterval(dtTimer);
  dtTimer=null;
  const btn=document.getElementById('det-play');
  if(btn) btn.innerHTML='<i class="ti ti-player-play"></i> Play';
}

/* ── Detection cache ──────────────────────────────────────────────────────── */

// Fetches cached detections from the API; falls back to live mode on 404.
// Sets dtDetections, dtLiveMode, and dtLiveBoxes, then calls dtRenderBboxes.
// In live mode also calls dtLoadImage to trigger the first inference.
async function dtLoadDetections(seqId, model){
  if(!model){ const c=document.querySelector('[name="dm"]:checked'); model=c?c.dataset.m:'e2vid'; }
  if(model==='hypere2vid' && selSeq && !selSeq.hypere2vid_done){
    dtDetections=null; dtLiveMode=false; dtLiveBoxes=[];
    document.getElementById('det-status').textContent='No HyperE2VID reconstruction frames for this sequence';
    dtRenderBboxes();
    return;
  }
  const sbKeys={e2vid:'sb-e2vid-dets',hypere2vid:'sb-hyper-dets',fusion:'sb-fusion-dets'};
  try{
    const r=await fetch('/api/detections/'+seqId+'?model='+model);
    if(r.ok){
      const data=await r.json();
      dtDetections=data.detections;
      dtLiveMode=false; dtLiveBoxes=[];
      document.getElementById('det-status').textContent='Cached — '+dtDetections.length+' detections across all frames';
      const sbD=document.getElementById(sbKeys[model]||'sb-e2vid-dets'); if(sbD) sbD.textContent=dtDetections.length;
      dtRenderBboxes();
    } else {
      if(model==='fusion'){
        // Fusion cannot do live frame-by-frame detection — requires full-sequence inference
        dtDetections=null; dtLiveMode=false; dtLiveBoxes=[];
        document.getElementById('det-status').textContent='No fusion cache — run detection from Upload screen first';
        dtRenderBboxes();
        dtLoadImage();
        return;
      }
      dtDetections=null;
      dtLiveMode=true; dtLiveBoxes=[];
      document.getElementById('det-status').textContent='Live detection — running frame by frame (no cache)';
      const sbD=document.getElementById('sb-dets'); if(sbD) sbD.textContent='live';
      dtLoadImage();
    }
  } catch(e){}
}

/* ── Screen load ──────────────────────────────────────────────────────────── */

// Called by the tab navigation hook (app.js) when the Detection screen becomes
// active. Resets state on sequence change and kicks off detection loading.
function detLoad(){
  const noSeq=document.getElementById('det-no-seq');
  const noFrames=document.getElementById('det-no-frames');
  const panel=document.getElementById('det-panel');
  noSeq.style.display='none'; noFrames.style.display='none'; panel.style.display='none';
  if(!selSeq){ noSeq.style.display='block'; return; }
  if(!selSeq.e2vid_done){ dtStop(); noFrames.style.display='block'; return; }
  const seqChanged=dtLoadedSeq!==selSeq.id;
  if(seqChanged){
    dtStop();
    dtLoadedSeq=selSeq.id;
    dtMax=selSeq.frame_count-1;
    dtDetections=null;
    document.getElementById('det-sl').max=dtMax;
    document.getElementById('det-max').textContent=dtMax;
    document.getElementById('det-max2').textContent=dtMax;
    const gi=document.getElementById('det-goto'); if(gi){ gi.max=dtMax; gi.value=0; }
    document.getElementById('det-status').textContent='Loading…';
    document.getElementById('det-count').textContent='—';
    document.getElementById('det-bboxes').innerHTML='';
    const sbDets=document.getElementById('sb-dets'); if(sbDets) sbDets.textContent='—';
    dtLoadDetections(selSeq.id);
  }
  panel.style.display='block';
  if(seqChanged) dtSetFrame(0);
}

/* ── Controls ─────────────────────────────────────────────────────────────── */

// Slider drag-tracking — onDrag updates pill only; onCommit also loads the frame.
const dtSlider = bindSlider('det-sl',
  v => { dtFrame=Math.max(0,Math.min(v,dtMax)); document.getElementById('det-pill').textContent='frame '+dtFrame+' / '+dtMax; },
  v => { dtFrame=Math.max(0,Math.min(v,dtMax)); document.getElementById('det-pill').textContent='frame '+dtFrame+' / '+dtMax; dtLoadImage(); }
);

document.getElementById('det-play').addEventListener('click',function(){
  if(!selSeq||!selSeq.e2vid_done) return;
  dtPlaying=!dtPlaying;
  clearInterval(dtTimer); dtTimer=null;
  if(dtPlaying){
    this.innerHTML='<i class="ti ti-player-pause"></i> Pause';
    dtTimer=setInterval(()=>{ if(!dtSlider.isDragging()) dtSetFrame(dtFrame>=dtMax ? 0 : dtFrame+1); }, dtLiveMode ? 500 : 150);
  } else {
    this.innerHTML='<i class="ti ti-player-play"></i> Play';
  }
});

document.getElementById('det-stop').addEventListener('click',()=>{ dtStop(); dtSetFrame(0); });
document.getElementById('det-prev').addEventListener('click',()=>{ dtStop(); dtSetFrame(dtFrame-1); });
document.getElementById('det-next').addEventListener('click',()=>{ dtStop(); dtSetFrame(dtFrame+1); });
document.getElementById('det-next-det').addEventListener('click',()=>{
  if(!dtDetections||dtLiveMode) return;
  const conf=getConf();
  const frames=[...new Set(dtDetections.filter(d=>d.confidence>=conf).map(d=>d.frame))].sort((a,b)=>a-b);
  const next=frames.find(f=>f>dtFrame) ?? frames[0];
  if(next!=null){ dtStop(); dtSetFrame(next); }
});
document.getElementById('det-goto').addEventListener('change',function(){ dtStop(); dtSetFrame(parseInt(this.value)||0); });

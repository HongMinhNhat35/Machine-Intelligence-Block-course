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
let dtMode='e2vid'; // 'e2vid' | 'hypere2vid' | 'fusion_rgb' | 'fusion_event' | 'fusion'
let dtAllFusionDets=null; // raw fusion detections, filtered per mode in dtRenderBboxes
let dtConfArr=null; // Float32Array for confidence timeline
let dtTrailOn=false;
const dtTrail=[];

/* ── Rendering ────────────────────────────────────────────────────────────── */

// Filters stored or live boxes by confidence and draws them over det-img2.
// For fusion_rgb/fusion_event modes, filters the fusion cache by source field.
function dtRenderBboxes(){
  const bboxDiv=document.getElementById('det-bboxes');
  if(!bboxDiv) return;
  const conf=getConf();
  const imgEl=document.getElementById('det-img2');
  const trailCanvas=document.getElementById('det-trail');
  let boxes;
  if(dtLiveMode){
    boxes=(dtLiveBoxes||[]).filter(d=>d.confidence>=conf);
  } else {
    const pool=(dtMode==='fusion_rgb'||dtMode==='fusion_event') ? dtAllFusionDets : dtDetections;
    if(!pool){
      bboxDiv.innerHTML=''; document.getElementById('det-count').textContent='—';
      if(dtTrailOn) trailPush(dtTrail, [], imgEl);
      drawTrail(trailCanvas, dtTrailOn ? dtTrail : []);
      return;
    }
    const src=dtMode==='fusion_event'?'event':dtMode==='fusion_rgb'?'rgb':null;
    boxes=pool.filter(d=>d.frame===dtFrame && d.confidence>=conf && (!src||d.source===src));
  }
  document.getElementById('det-count').textContent=boxes.length ? boxes.length+' detection'+(boxes.length===1?'':'s') : 'no detections';
  renderBboxes(boxes, imgEl, bboxDiv);
  if(dtTrailOn) trailPush(dtTrail, boxes, imgEl);
  drawTrail(trailCanvas, dtTrailOn ? dtTrail : []);
}

// Returns the active mode string.
function dtGetModel(){ return dtMode; }

/* ── Image loading ────────────────────────────────────────────────────────── */

// Returns the frame URL for the current mode.
function dtFrameSrc(seq, n){
  if(dtMode==='hypere2vid')  return '/frames/'+seq+'/'+n+'?model=hypere2vid';
  if(dtMode==='fusion_rgb' || dtMode==='fusion') return '/frames_rgb/'+seq+'/'+n;
  if(dtMode==='fusion_event') return '/frames_event/'+seq+'/'+n;
  return '/frames/'+seq+'/'+n+'?model=e2vid';
}

// Debounced frame loader — waits 60 ms before setting img.src to avoid
// flooding the server during rapid scrubbing. In live mode, triggers a
// YOLO inference request after the image loads.
function dtLoadImage(){
  if(dtMode==='hypere2vid' && selSeq && !selSeq.hypere2vid_done){
    document.getElementById('det-img').src='';
    document.getElementById('det-img2').src='';
    document.getElementById('det-bboxes').innerHTML='';
    document.getElementById('det-count').textContent='—';
    return;
  }
  clearTimeout(dtImgDebounce);
  const seq=selSeq.id, n=dtFrame;
  if(dtLiveMode) dtLiveBoxes=[];
  dtImgDebounce=setTimeout(()=>{
    if(dtFrame!==n) return;
    const src=dtFrameSrc(seq, n);
    document.getElementById('det-img').src=src;
    const img2=document.getElementById('det-img2');
    const liveModel=dtMode==='fusion'||dtMode==='fusion_rgb'||dtMode==='fusion_event'?'fusion':dtMode;
    img2.onload=()=>{ dtRenderBboxes(); if(dtLiveMode) fetchLiveFrame(seq, n, liveModel, (s,f)=>dtFrame===f&&dtLoadedSeq===s, boxes=>{dtLiveBoxes=boxes;dtRenderBboxes();}); };
    img2.src=src;
  }, 60);
}

/* ── Frame navigation ─────────────────────────────────────────────────────── */

// Clamps n to [0, dtMax], updates pill and slider, and triggers dtLoadImage
// unless the user is actively dragging the slider.
const DT_MODEL_COLORS={e2vid:'#3b82f6',hypere2vid:'#8b5cf6',fusion:'#f97316',fusion_rgb:'#22c55e',fusion_event:'#14b8a6'};

function dtDrawConfTimeline(){
  drawConfTimeline('det-conf-timeline', dtFrame, dtMax, [
    {arr:dtConfArr, color:DT_MODEL_COLORS[dtMode]||'#888', label:MODE_LABELS[dtMode]||dtMode},
  ]);
}

// Returns the correct frame max for the active mode, using per-mode frame counts.
function dtFrameMax(){
  if(!selSeq) return 0;
  if(dtMode==='fusion_event') return Math.max(0,(selSeq.fred_event_count||1)-1);
  if(dtMode==='hypere2vid')   return Math.max(0,(selSeq.hyper_frame_count||selSeq.frame_count||1)-1);
  return Math.max(0,(selSeq.frame_count||1)-1);
}

function dtSetFrame(n){
  dtFrame=Math.max(0,Math.min(n,dtMax));
  document.getElementById('det-pill').textContent='frame '+dtFrame+' / '+dtMax;
  const gi=document.getElementById('det-goto'); if(gi) gi.value=dtFrame;
  dtDrawConfTimeline();
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

// Fetches cached detections for the current mode; falls back to live for e2vid/hyper.
// fusion_rgb and fusion_event use the fusion cache filtered by source.
async function dtLoadDetections(seqId, mode){
  if(mode) dtMode=mode;
  dtDetections=null; dtAllFusionDets=null; dtLiveMode=false; dtLiveBoxes=[];
  dtTrail.length=0; drawTrail(document.getElementById('det-trail'),[]);
  const isFusionVariant=dtMode==='fusion'||dtMode==='fusion_rgb'||dtMode==='fusion_event';
  if(dtMode==='hypere2vid' && selSeq && !selSeq.hypere2vid_done){
    document.getElementById('det-status').textContent='No HyperE2VID reconstruction frames for this sequence';
    dtRenderBboxes(); return;
  }
  const sbKeys={e2vid:'sb-e2vid-dets',hypere2vid:'sb-hyper-dets',fusion:'sb-fusion-dets'};
  try{
    const apiModel=isFusionVariant?'fusion':dtMode;
    const r=await fetch('/api/detections/'+seqId+'?model='+apiModel);
    if(r.ok){
      const data=await r.json();
      if(isFusionVariant){
        dtAllFusionDets=data.detections;
        dtDetections=data.detections; // also set for compat
      } else {
        dtDetections=data.detections;
      }
      dtLiveMode=false;
      document.getElementById('det-status').textContent='Cached — '+data.detections.length+' detections across all frames';
      const sbD=document.getElementById(sbKeys[apiModel]||'sb-e2vid-dets'); if(sbD) sbD.textContent=data.detections.length;
      const srcFilter=dtMode==='fusion_event'?'event':dtMode==='fusion_rgb'?'rgb':null;
      const confPool=isFusionVariant?(srcFilter?dtAllFusionDets.filter(d=>d.source===srcFilter):dtAllFusionDets):dtDetections;
      dtConfArr=buildConfArray(confPool, dtMax+1);
      dtDrawConfTimeline();
      dtRenderBboxes();
    } else {
      if(isFusionVariant){
        document.getElementById('det-status').textContent='No fusion cache — run detection from Upload screen first';
        dtRenderBboxes(); dtLoadImage(); return;
      }
      dtLiveMode=true;
      document.getElementById('det-status').textContent='Live detection — running frame by frame (no cache)';
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
    dtMax=dtFrameMax();
    dtDetections=null; dtConfArr=null;
    dtTrail.length=0; drawTrail(document.getElementById('det-trail'),[]);
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
  v => { dtFrame=Math.max(0,Math.min(v,dtMax)); document.getElementById('det-pill').textContent='frame '+dtFrame+' / '+dtMax; dtDrawConfTimeline(); },
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

/* ── Mode buttons ────────────────────────────────────────────────────────── */

const MODE_LABELS={
  e2vid:'e2vid',hypere2vid:'HyperE2VID',
  fusion_rgb:'RGB',fusion_event:'Event',fusion:'Late Fusion'
};

function dtSetMode(m){
  dtMode=m;
  document.querySelectorAll('#det-mode-btns button').forEach(b=>{
    const active=b.dataset.dm===m;
    b.style.background=active?'#1a3a60':'';
    b.style.color=active?'#fff':'';
    b.style.borderColor=active?'#1a3a60':'';
  });
  const lbl=document.getElementById('det-left-lbl');
  const pill=document.getElementById('det-pill');
  if(lbl){ lbl.textContent=(MODE_LABELS[m]||m)+' '; if(pill) lbl.appendChild(pill); }
  document.getElementById('tb').textContent=MODE_LABELS[m]||m;
  if(selSeq){
    const newMax=dtFrameMax();
    if(newMax!==dtMax){
      dtMax=newMax;
      document.getElementById('det-sl').max=dtMax;
      document.getElementById('det-max').textContent=dtMax;
      document.getElementById('det-max2').textContent=dtMax;
      const gi=document.getElementById('det-goto'); if(gi) gi.max=dtMax;
      if(dtFrame>dtMax){ dtFrame=dtMax; }
    }
    dtLoadDetections(selSeq.id); dtLoadImage();
  }
}

document.querySelectorAll('#det-mode-btns button').forEach(b=>{
  b.addEventListener('click',()=>dtSetMode(b.dataset.dm));
});
document.getElementById('det-trail-btn').addEventListener('click', function(){
  dtTrailOn=!dtTrailOn;
  this.style.background=dtTrailOn?'#1a3a60':'';
  this.style.color=dtTrailOn?'#fff':'';
  this.style.borderColor=dtTrailOn?'#1a3a60':'';
  if(!dtTrailOn){ dtTrail.length=0; drawTrail(document.getElementById('det-trail'),[]); }
});

dtSetMode('e2vid'); // initialise

/*
 * AMI Hybrid Vision System — Group 1
 * File: app-compare.js
 *
 * Comparison screen — shows e2vid and HyperE2VID frames side by side with
 * detection overlays from cache or live YOLO inference. Loads KPI metrics
 * from the API and populates the mAP/precision/recall table.
 *
 * Load order: after app-utils.js, app-dataset.js, app-recon.js, and
 *             app-detect.js, before app.js
 * Globals consumed: selSeq, kpiCache (app.js), getConf, fetchLiveFrame,
 *                   renderBboxes, bindSlider (app-utils.js),
 *                   getKpis, updateSidebarMap50 (app-dataset.js)
 * Globals exported: cpRenderBboxes, cpStop, compLoad
 *                   — called from app.js (go() and tab navigation hook)
 */

/* ── State ────────────────────────────────────────────────────────────────── */

let cpPlaying=false, cpTimer=null, cpFrame=0, cpMax=0, cpLoadedSeq=null;
let cpDetections=null, cpHyperDetections=null, cpFusionDetections=null, cpImgDebounce=null;
let cpLiveE2vid=false, cpLiveHyper=false, cpLiveBoxes=[], cpLiveHyperBoxes=[];
let cpFusionFrameCount=0;  // total FRED RGB frames; used to map e2vid index → fusion index
let cpHyperFrameCount=0;   // total hypere2vid frames; used to map e2vid index → hyper index

// Two-anchor linear calibration: slope + offset per secondary stream.
// Computed from the first/last SUSTAINED detection frames, not first/last
// single detections (which are noise). Falls back to ratio when unavailable.
let cpFusionSlope=0, cpFusionOffset=0, cpFusionCalibrated=false;
let cpHyperSlope=0,  cpHyperOffset=0,  cpHyperCalibrated=false;

// Returns the first frame where at least minCount of the next `window` frames
// have a detection — i.e., the start of a sustained detection cluster.
function cpFirstSustained(frames, window=10, minCount=5){
  const s=new Set(frames);
  for(const f of frames){
    let c=0; for(let x=f;x<f+window;x++) if(s.has(x)) c++;
    if(c>=minCount) return f;
  }
  return frames[0];
}

// Returns the last frame where at least minCount of the previous `window`
// frames have a detection — i.e., the end of a sustained detection cluster.
function cpLastSustained(frames, window=10, minCount=5){
  const s=new Set(frames);
  for(let i=frames.length-1;i>=0;i--){
    const f=frames[i];
    let c=0; for(let x=f-window+1;x<=f;x++) if(s.has(x)) c++;
    if(c>=minCount) return f;
  }
  return frames[frames.length-1];
}

// Calibrates slope+offset for each secondary stream using sustained detection
// anchors. These mark the true drone appearing/disappearing and correspond
// to the same real-world event across all streams (unlike first/last single
// detections, which are often noise or reflect different model sensitivities).
function cpCalibrate(){
  const CAL_CONF=0.05;
  const detFrames=dets=>[...new Set(dets.filter(d=>d.confidence>=CAL_CONF).map(d=>d.frame))].sort((a,b)=>a-b);
  const eF=cpDetections ? detFrames(cpDetections) : [];
  cpFusionCalibrated=false; cpHyperCalibrated=false;
  if(eF.length<10) return;
  const eFirst=cpFirstSustained(eF), eLast=cpLastSustained(eF);
  if(eLast<=eFirst) return;
  if(cpFusionDetections && cpFusionFrameCount){
    const fF=detFrames(cpFusionDetections);
    if(fF.length>=10){
      const fFirst=cpFirstSustained(fF), fLast=cpLastSustained(fF);
      if(fLast>fFirst){
        cpFusionSlope=(fLast-fFirst)/(eLast-eFirst);
        cpFusionOffset=fFirst-cpFusionSlope*eFirst;
        cpFusionCalibrated=true;
      }
    }
  }
  if(cpHyperDetections && cpHyperFrameCount){
    const hF=detFrames(cpHyperDetections);
    if(hF.length>=10){
      const hFirst=cpFirstSustained(hF), hLast=cpLastSustained(hF);
      if(hLast>hFirst){
        cpHyperSlope=(hLast-hFirst)/(eLast-eFirst);
        cpHyperOffset=hFirst-cpHyperSlope*eFirst;
        cpHyperCalibrated=true;
      }
    }
  }
}

/* ── Rendering ────────────────────────────────────────────────────────────── */

// Filters stored or live e2vid boxes by confidence and draws them over cp-e2vid-img.
function cpRenderBboxes(){
  const bboxDiv=document.getElementById('cp-e2vid-bboxes');
  if(!bboxDiv) return;
  const conf=getConf();
  let boxes;
  if(cpLiveE2vid){
    boxes=(cpLiveBoxes||[]).filter(d=>d.confidence>=conf);
  } else {
    if(!cpDetections){ bboxDiv.innerHTML=''; return; }
    boxes=cpDetections.filter(d=>d.frame===cpFrame && d.confidence>=conf);
  }
  renderBboxes(boxes, document.getElementById('cp-e2vid-img'), bboxDiv);
}

// Filters stored or live HyperE2VID boxes by confidence and draws them over cp-hyper-img.
function cpRenderHyperBboxes(){
  const bboxDiv=document.getElementById('cp-hyper-bboxes');
  if(!bboxDiv) return;
  const conf=getConf();
  let boxes;
  if(cpLiveHyper){
    boxes=(cpLiveHyperBoxes||[]).filter(d=>d.confidence>=conf);
  } else {
    if(!cpHyperDetections){ bboxDiv.innerHTML=''; return; }
    boxes=cpHyperDetections.filter(d=>d.frame===cpHyperFrame() && d.confidence>=conf);
  }
  renderBboxes(boxes, document.getElementById('cp-hyper-img'), bboxDiv);
}

// Maps the current e2vid frame index to the corresponding hypere2vid frame index.
function cpHyperFrame(){
  if(!cpHyperFrameCount || !cpMax) return cpFrame;
  if(cpHyperCalibrated)
    return Math.max(0,Math.min(cpHyperFrameCount-1, Math.round(cpHyperSlope*cpFrame+cpHyperOffset)));
  return Math.round(cpFrame * cpHyperFrameCount / (cpMax + 1));
}

// Maps the current e2vid frame index to the corresponding fusion (FRED RGB) frame index.
function cpFusionFrame(){
  if(!cpFusionFrameCount || !cpMax) return cpFrame;
  if(cpFusionCalibrated)
    return Math.max(0,Math.min(cpFusionFrameCount-1, Math.round(cpFusionSlope*cpFrame+cpFusionOffset)));
  return Math.round(cpFrame * cpFusionFrameCount / (cpMax + 1));
}

// Filters cached fusion boxes by confidence and draws them over cp-fusion-img.
function cpRenderFusionBboxes(){
  const bboxDiv=document.getElementById('cp-fusion-bboxes');
  if(!bboxDiv) return;
  if(!cpFusionDetections){ bboxDiv.innerHTML=''; return; }
  const conf=getConf();
  const ff=cpFusionFrame();
  const boxes=cpFusionDetections.filter(d=>d.frame===ff && d.confidence>=conf);
  renderBboxes(boxes, document.getElementById('cp-fusion-img'), bboxDiv);
}

/* ── Image loading ────────────────────────────────────────────────────────── */

// Debounced frame loader — waits 60 ms before setting img.src. After each
// image loads, renders bounding boxes and, in live mode, fires a YOLO request.
function cpLoadImage(){
  clearTimeout(cpImgDebounce);
  const seq=selSeq.id, n=cpFrame;
  if(cpLiveE2vid) cpLiveBoxes=[];
  if(cpLiveHyper) cpLiveHyperBoxes=[];
  cpImgDebounce=setTimeout(()=>{
    if(cpFrame!==n) return;
    const img=document.getElementById('cp-e2vid-img');
    const e2vidOnload=()=>{ cpRenderBboxes(); if(cpLiveE2vid) fetchLiveFrame(seq, n, 'e2vid', (s,f)=>cpFrame===f&&cpLoadedSeq===s, boxes=>{cpLiveBoxes=boxes;cpRenderBboxes();}); };
    img.onload=e2vidOnload;
    img.src='/frames/'+seq+'/'+n;
    if(img.complete) e2vidOnload();
    if(selSeq.hypere2vid_done){
      const hyperImg=document.getElementById('cp-hyper-img');
      const hyperOnload=()=>{ cpRenderHyperBboxes(); if(cpLiveHyper) fetchLiveFrame(seq, n, 'hypere2vid', (s,f)=>cpFrame===f&&cpLoadedSeq===s, boxes=>{cpLiveHyperBoxes=boxes;cpRenderHyperBboxes();}); };
      hyperImg.onload=hyperOnload;
      hyperImg.src='/frames/'+seq+'/'+cpHyperFrame()+'?model=hypere2vid';
      if(hyperImg.complete) hyperOnload();
    }
    // Fusion column: show FRED RGB frame (colour source for fusion model)
    if(cpFusionDetections){
      const fusionImg=document.getElementById('cp-fusion-img');
      const fusionOnload=()=>cpRenderFusionBboxes();
      fusionImg.onload=fusionOnload;
      fusionImg.src='/frames_rgb/'+seq+'/'+cpFusionFrame();
      if(fusionImg.complete) fusionOnload();
    }
  }, 60);
}

/* ── Frame navigation ─────────────────────────────────────────────────────── */

// Clamps n to [0, cpMax], updates goto input and slider, and triggers
// cpLoadImage unless the user is actively dragging the slider.
function cpSetFrame(n){
  cpFrame=Math.max(0,Math.min(n,cpMax));
  const gi=document.getElementById('cp-goto'); if(gi) gi.value=cpFrame;
  if(!cpSlider.isDragging()){
    document.getElementById('cp-sl').value=cpFrame;
    cpLoadImage();
  }
}

// Stops the play timer and resets the Play button label.
function cpStop(){
  cpPlaying=false;
  clearInterval(cpTimer); cpTimer=null;
  const btn=document.getElementById('cp-play');
  if(btn) btn.innerHTML='<i class="ti ti-player-play"></i> Play';
}

/* ── Metrics ──────────────────────────────────────────────────────────────── */

// Populates the e2vid mAP/precision/recall cells from kpiCache and refreshes
// the sidebar mAP50 badge.
function cpUpdateMetrics(){
  if(!kpiCache) return;
  const runs=kpiCache.filter(r=>r.model==='e2vid');
  if(!runs.length) return;
  const run=runs.reduce((b,r)=>((r.detection?.canonical?.map50||0)>(b.detection?.canonical?.map50||0)?r:b),runs[0]);
  const m=run.detection?.canonical||{};
  const pct=v=>(v===null||v===undefined)?'—':(Math.round(+v*1000)/10).toFixed(1)+'%';
  document.getElementById('cp-map50').textContent=pct(m.map50);
  document.getElementById('cp-map5095').textContent=pct(m.map50_95);
  document.getElementById('cp-prec').textContent=pct(m.precision);
  document.getElementById('cp-rec').textContent=pct(m.recall);
  updateSidebarMap50();
}

// Populates the HyperE2VID mAP/precision/recall cells from kpiCache.
function cpUpdateMetricsHyper(){
  if(!kpiCache) return;
  const runs=kpiCache.filter(r=>r.model==='hypere2vid');
  if(!runs.length) return;
  const run=runs.reduce((b,r)=>((r.detection?.canonical?.map50||0)>(b.detection?.canonical?.map50||0)?r:b),runs[0]);
  const m=run.detection?.canonical||{};
  const pct=v=>(v===null||v===undefined)?'—':(Math.round(+v*1000)/10).toFixed(1)+'%';
  document.getElementById('cp-hyper-map50').textContent=pct(m.map50);
  document.getElementById('cp-hyper-map5095').textContent=pct(m.map50_95);
  document.getElementById('cp-hyper-prec').textContent=pct(m.precision);
  document.getElementById('cp-hyper-rec').textContent=pct(m.recall);
}

// Populates the fusion mAP cells from kpiCache (shows — until a fusion KPI file exists).
function cpUpdateMetricsFusion(){
  if(!kpiCache) return;
  const runs=kpiCache.filter(r=>r.model==='fusion');
  const pct=v=>(v===null||v===undefined)?'—':(Math.round(+v*1000)/10).toFixed(1)+'%';
  if(!runs.length){
    ['cp-fusion-map50','cp-fusion-map5095','cp-fusion-prec','cp-fusion-rec'].forEach(id=>{
      const el=document.getElementById(id); if(el) el.textContent='—';
    });
    return;
  }
  const run=runs.reduce((b,r)=>((r.detection?.canonical?.map50||0)>(b.detection?.canonical?.map50||0)?r:b),runs[0]);
  const m=run.detection?.canonical||{};
  document.getElementById('cp-fusion-map50').textContent=pct(m.map50);
  document.getElementById('cp-fusion-map5095').textContent=pct(m.map50_95);
  document.getElementById('cp-fusion-prec').textContent=pct(m.precision);
  document.getElementById('cp-fusion-rec').textContent=pct(m.recall);
}

/* ── Screen load ──────────────────────────────────────────────────────────── */

// Called by the tab navigation hook (app.js) when the Comparison screen becomes
// active. Awaits both detection cache checks before calling cpSetFrame(0) so
// live-mode flags are guaranteed to be set before the first image loads.
async function compLoad(){
  const noSeq=document.getElementById('comp-no-seq');
  const panel=document.getElementById('comp-panel');
  noSeq.style.display='none'; panel.style.display='none';
  if(!selSeq){ noSeq.style.display='block'; return; }
  const seqChanged=cpLoadedSeq!==selSeq.id;
  if(seqChanged){
    cpStop();
    cpLoadedSeq=selSeq.id;
    cpMax=selSeq.frame_count-1;
    cpDetections=null; cpHyperDetections=null; cpFusionDetections=null;
    cpFusionFrameCount=selSeq.fred_rgb_count||0;
    cpHyperFrameCount=selSeq.hyper_frame_count||0;
    cpFusionCalibrated=false; cpHyperCalibrated=false;
    cpLiveE2vid=false; cpLiveHyper=false; cpLiveBoxes=[]; cpLiveHyperBoxes=[];
    document.getElementById('cp-sl').max=cpMax;
    document.getElementById('cp-max').textContent=cpMax;
    document.getElementById('cp-max2').textContent=cpMax;
    const gi=document.getElementById('cp-goto'); if(gi){ gi.max=cpMax; gi.value=0; }
    const cpHyperImg=document.getElementById('cp-hyper-img');
    const cpHyperNoFrames=document.getElementById('cp-hyper-no-frames');
    if(selSeq.hypere2vid_done){
      cpHyperImg.style.display='block'; if(cpHyperNoFrames) cpHyperNoFrames.style.display='none';
    } else {
      cpHyperImg.src=''; cpHyperImg.style.display='none';
      if(cpHyperNoFrames) cpHyperNoFrames.style.display='flex';
    }
    const cpFusionNoCache=document.getElementById('cp-fusion-no-cache');
    const cpFusionImg=document.getElementById('cp-fusion-img');
    const [r1, r2, r3] = await Promise.all([
      fetch('/api/detections/'+selSeq.id).catch(()=>null),
      selSeq.hypere2vid_done
        ? fetch('/api/detections/'+selSeq.id+'?model=hypere2vid').catch(()=>null)
        : Promise.resolve(null),
      fetch('/api/detections/'+selSeq.id+'?model=fusion').catch(()=>null),
    ]);
    const d1 = r1?.ok ? await r1.json().catch(()=>null) : null;
    if(d1){ cpDetections=d1.detections; cpLiveE2vid=false; }
    else { cpLiveE2vid=true; cpLiveBoxes=[]; }
    if(r2){
      const d2 = r2.ok ? await r2.json().catch(()=>null) : null;
      if(d2){ cpHyperDetections=d2.detections; cpLiveHyper=false; }
      else { cpLiveHyper=true; cpLiveHyperBoxes=[]; }
    }
    const d3 = r3?.ok ? await r3.json().catch(()=>null) : null;
    if(d3){
      cpFusionDetections=d3.detections;
      if(cpFusionImg) cpFusionImg.style.display='block';
      if(cpFusionNoCache) cpFusionNoCache.style.display='none';
    } else {
      cpFusionDetections=null;
      if(cpFusionImg){ cpFusionImg.src=''; cpFusionImg.style.display='none'; }
      if(cpFusionNoCache) cpFusionNoCache.style.display='flex';
    }
    cpCalibrate();
  }
  panel.style.display='block';
  if(seqChanged) cpSetFrame(0);
  await getKpis();
  cpUpdateMetrics();
  cpUpdateMetricsHyper();
  cpUpdateMetricsFusion();
}

/* ── Controls ─────────────────────────────────────────────────────────────── */

// Slider drag-tracking — onDrag updates goto input only; onCommit also loads the frame.
const cpSlider = bindSlider('cp-sl',
  v => { cpFrame=Math.max(0,Math.min(v,cpMax)); const gi=document.getElementById('cp-goto'); if(gi) gi.value=cpFrame; },
  v => { cpFrame=Math.max(0,Math.min(v,cpMax)); const gi=document.getElementById('cp-goto'); if(gi) gi.value=cpFrame; cpLoadImage(); }
);

document.getElementById('cp-play').addEventListener('click',function(){
  if(!selSeq||!selSeq.e2vid_done) return;
  cpPlaying=!cpPlaying;
  clearInterval(cpTimer); cpTimer=null;
  if(cpPlaying){
    this.innerHTML='<i class="ti ti-player-pause"></i> Pause';
    cpTimer=setInterval(()=>{ if(!cpSlider.isDragging()) cpSetFrame(cpFrame>=cpMax?0:cpFrame+1); },150);
  } else {
    this.innerHTML='<i class="ti ti-player-play"></i> Play';
  }
});
document.getElementById('cp-stop').addEventListener('click',()=>{ cpStop(); cpSetFrame(0); });
document.getElementById('cp-prev').addEventListener('click',()=>{ cpStop(); cpSetFrame(cpFrame-1); });
document.getElementById('cp-next').addEventListener('click',()=>{ cpStop(); cpSetFrame(cpFrame+1); });
document.getElementById('cp-goto').addEventListener('change',function(){ cpStop(); cpSetFrame(parseInt(this.value)||0); });
document.getElementById('cp-next-det').addEventListener('click',()=>{
  const conf=getConf();
  const frameSet=new Set();
  if(cpDetections) cpDetections.filter(d=>d.confidence>=conf).forEach(d=>frameSet.add(d.frame));
  // Hyper and fusion frames are on different axes — convert back to e2vid frame index
  if(cpHyperDetections && cpHyperFrameCount && cpMax)
    cpHyperDetections.filter(d=>d.confidence>=conf).forEach(d=>{
      const e=Math.round(d.frame*(cpMax+1)/cpHyperFrameCount); if(e>=0&&e<=cpMax) frameSet.add(e);
    });
  if(cpFusionDetections && cpFusionFrameCount && cpMax)
    cpFusionDetections.filter(d=>d.confidence>=conf).forEach(d=>{
      const e=Math.round(d.frame*(cpMax+1)/cpFusionFrameCount); if(e>=0&&e<=cpMax) frameSet.add(e);
    });
  if(!frameSet.size) return;
  const frames=[...frameSet].sort((a,b)=>a-b);
  const next=frames.find(f=>f>cpFrame) ?? frames[0];
  if(next!=null){ cpStop(); cpSetFrame(next); }
});

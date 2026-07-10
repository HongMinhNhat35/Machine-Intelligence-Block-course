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

// Detection-based sync calibration: piecewise linear map built from N percentile
// anchor pairs sampled from the sorted detection-frame lists of each stream.
// Falls back to ratio formula when calibration is unavailable.
let cpFusionPairs=null, cpFusionCalibrated=false;
let cpHyperPairs=null,  cpHyperCalibrated=false;

// Builds anchor pairs by dividing the e2vid frame range into N equal temporal
// bins, then finding the median detection frame in each bin for both streams.
// The corresponding fusion bin is located using the global frame-count ratio,
// so pairs reflect real-world time position rather than detection-count rank.
// This avoids systematic offset when one stream detects more densely than the
// other at certain points in the sequence.
function cpBuildPairs(eFrames, fFrames, eTotal, fTotal, N=16){
  if(eFrames.length<4 || fFrames.length<4 || !eTotal || !fTotal) return null;
  const ratio=fTotal/eTotal;
  const pairs=[];
  for(let k=0;k<N;k++){
    const eMin=k*eTotal/N, eMax=(k+1)*eTotal/N;
    const eBin=eFrames.filter(f=>f>=eMin&&f<eMax);
    if(!eBin.length) continue;
    const eAnchor=eBin[Math.floor(eBin.length/2)];
    const fMin=eMin*ratio, fMax=eMax*ratio;
    const fBin=fFrames.filter(f=>f>=fMin&&f<fMax);
    if(!fBin.length) continue;
    const fAnchor=fBin[Math.floor(fBin.length/2)];
    pairs.push([eAnchor,fAnchor]);
  }
  const mono=[]; let last=-1;
  for(const p of pairs){ if(p[0]>last){mono.push(p);last=p[0];} }
  return mono.length>=2 ? mono : null;
}

// Piecewise-linear lookup: given sorted anchor pairs, interpolate sFrame for eFrame.
// Extrapolates beyond the first/last anchor using that terminal segment's slope
// so frames past the last detection stay in sync rather than freezing.
function cpInterpolate(pairs, x){
  if(!pairs||!pairs.length) return x;
  const n=pairs.length;
  if(x<=pairs[0][0]){
    if(n<2) return pairs[0][1];
    const slope=(pairs[1][1]-pairs[0][1])/(pairs[1][0]-pairs[0][0]);
    return Math.round(pairs[0][1]+slope*(x-pairs[0][0]));
  }
  if(x>=pairs[n-1][0]){
    if(n<2) return pairs[n-1][1];
    const slope=(pairs[n-1][1]-pairs[n-2][1])/(pairs[n-1][0]-pairs[n-2][0]);
    return Math.round(pairs[n-1][1]+slope*(x-pairs[n-1][0]));
  }
  for(let i=0;i<n-1;i++){
    if(x>=pairs[i][0]&&x<=pairs[i+1][0]){
      const t=(x-pairs[i][0])/(pairs[i+1][0]-pairs[i][0]);
      return Math.round(pairs[i][1]+t*(pairs[i+1][1]-pairs[i][1]));
    }
  }
  return pairs[n-1][1];
}

// Inverse lookup: given sFrame, find the e2vid frame via the anchor pairs.
// Also extrapolates beyond the anchor range.
function cpInterpolateInv(pairs, y){
  if(!pairs||!pairs.length) return y;
  const n=pairs.length;
  if(y<=pairs[0][1]){
    if(n<2) return pairs[0][0];
    const slope=(pairs[1][0]-pairs[0][0])/(pairs[1][1]-pairs[0][1]);
    return Math.round(pairs[0][0]+slope*(y-pairs[0][1]));
  }
  if(y>=pairs[n-1][1]){
    if(n<2) return pairs[n-1][0];
    const slope=(pairs[n-1][0]-pairs[n-2][0])/(pairs[n-1][1]-pairs[n-2][1]);
    return Math.round(pairs[n-1][0]+slope*(y-pairs[n-1][1]));
  }
  for(let i=0;i<n-1;i++){
    if(y>=pairs[i][1]&&y<=pairs[i+1][1]){
      const t=(y-pairs[i][1])/(pairs[i+1][1]-pairs[i][1]);
      return Math.round(pairs[i][0]+t*(pairs[i+1][0]-pairs[i][0]));
    }
  }
  return pairs[n-1][0];
}

// Recomputes piecewise calibration from detection frames at a low confidence
// threshold, independent of the display conf slider.
function cpCalibrate(){
  const CAL_CONF=0.05;
  const detFrames=dets=>[...new Set(dets.filter(d=>d.confidence>=CAL_CONF).map(d=>d.frame))].sort((a,b)=>a-b);
  const eF=cpDetections ? detFrames(cpDetections) : [];
  const eTotal=cpMax+1;
  cpFusionCalibrated=false; cpFusionPairs=null;
  if(eF.length>=4 && cpFusionDetections && cpFusionFrameCount){
    const fF=detFrames(cpFusionDetections);
    cpFusionPairs=cpBuildPairs(eF,fF,eTotal,cpFusionFrameCount);
    cpFusionCalibrated=cpFusionPairs!==null;
  }
  cpHyperCalibrated=false; cpHyperPairs=null;
  if(eF.length>=4 && cpHyperDetections && cpHyperFrameCount){
    const hF=detFrames(cpHyperDetections);
    cpHyperPairs=cpBuildPairs(eF,hF,eTotal,cpHyperFrameCount);
    cpHyperCalibrated=cpHyperPairs!==null;
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
    return Math.max(0,Math.min(cpHyperFrameCount-1, cpInterpolate(cpHyperPairs, cpFrame)));
  return Math.round(cpFrame * cpHyperFrameCount / (cpMax + 1));
}

// Maps the current e2vid frame index to the corresponding fusion (FRED RGB) frame index.
function cpFusionFrame(){
  if(!cpFusionFrameCount || !cpMax) return cpFrame;
  if(cpFusionCalibrated)
    return Math.max(0,Math.min(cpFusionFrameCount-1, cpInterpolate(cpFusionPairs, cpFrame)));
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
    cpFusionCalibrated=false; cpFusionPairs=null;
    cpHyperCalibrated=false;  cpHyperPairs=null;
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
  if(cpHyperDetections && cpHyperFrameCount && cpMax){
    const toE2vid=cpHyperCalibrated
      ? f=>Math.max(0,Math.min(cpMax, cpInterpolateInv(cpHyperPairs, f)))
      : f=>Math.round(f*(cpMax+1)/cpHyperFrameCount);
    cpHyperDetections.filter(d=>d.confidence>=conf).forEach(d=>{
      const e=toE2vid(d.frame); if(e>=0&&e<=cpMax) frameSet.add(e);
    });
  }
  if(cpFusionDetections && cpFusionFrameCount && cpMax){
    const toE2vid=cpFusionCalibrated
      ? f=>Math.max(0,Math.min(cpMax, cpInterpolateInv(cpFusionPairs, f)))
      : f=>Math.round(f*(cpMax+1)/cpFusionFrameCount);
    cpFusionDetections.filter(d=>d.confidence>=conf).forEach(d=>{
      const e=toE2vid(d.frame); if(e>=0&&e<=cpMax) frameSet.add(e);
    });
  }
  if(!frameSet.size) return;
  const frames=[...frameSet].sort((a,b)=>a-b);
  const next=frames.find(f=>f>cpFrame) ?? frames[0];
  if(next!=null){ cpStop(); cpSetFrame(next); }
});

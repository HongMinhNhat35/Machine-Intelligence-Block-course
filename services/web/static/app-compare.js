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
let cpSync=null;           // {e2vid_ts, rgb_ts} from /api/sync — enables exact timestamp sync

// Two-anchor linear calibration: slope + offset per secondary stream.
// Computed from the first/last SUSTAINED detection frames, not first/last
// single detections (which are noise). Falls back to ratio when unavailable.
let cpFusionSlope=0, cpFusionOffset=0, cpFusionCalibrated=false;
let cpHyperSlope=0,  cpHyperOffset=0,  cpHyperCalibrated=false;

// Returns {start, end} for each sustained detection cluster that spans at least
// minSpan frames. Brief noise detections (e.g. a 6-frame blip) are filtered out
// so they don't corrupt rank-based cluster matching across streams.
function cpFindClusters(frames, window=10, minCount=5, minSpan=30){
  const s=new Set(frames);
  const clusters=[];
  let inCluster=false, clusterStart=null, lastSustained=null;
  for(const f of frames){
    let c=0; for(let x=f;x<f+window;x++) if(s.has(x)) c++;
    const sustained=c>=minCount;
    if(sustained){
      if(!inCluster){ clusterStart=f; inCluster=true; }
      lastSustained=f;
    } else if(inCluster){
      if(lastSustained-clusterStart>=minSpan) clusters.push({start:clusterStart,end:lastSustained});
      inCluster=false;
    }
  }
  if(inCluster && lastSustained-clusterStart>=minSpan) clusters.push({start:clusterStart,end:lastSustained});
  return clusters;
}

// Calibrates slope+offset for a secondary stream against primary detection frames.
// Step 1 – slope from detection span: uses first/last significant cluster endpoints
//          (avoids brief noise blips corrupting the ratio).
// Step 2 – offset via 1-D cross-correlation: scans ±300 frames around the
//          span-derived anchor and picks the offset that maximises the number of
//          primary detection frames that land on a secondary detection frame.
//          This removes progressive drift caused by a slightly wrong anchor.
function cpCalibrateStream(primaryFrames, primaryClusters, secondaryDets, secondaryFrameCount, primaryMax){
  if(!secondaryDets || !secondaryFrameCount || !primaryClusters.length || !primaryFrames.length) return null;
  const CAL_CONF=0.05;
  const sF=[...new Set(secondaryDets.filter(d=>d.confidence>=CAL_CONF).map(d=>d.frame))].sort((a,b)=>a-b);
  if(sF.length<10) return null;
  const sClusters=cpFindClusters(sF);
  if(!sClusters.length) return null;

  // Slope from detection span ratio
  const pFirst=primaryClusters[0].start, pLast=primaryClusters[primaryClusters.length-1].end;
  const sFirst=sClusters[0].start,       sLast=sClusters[sClusters.length-1].end;
  const slope=(pLast>pFirst && sLast>sFirst)
    ? (sLast-sFirst)/(pLast-pFirst)
    : secondaryFrameCount/(primaryMax+1);

  // 1-D offset search: maximise overlap between mapped primary frames and secondary detection set
  const sSet=new Set(sF);
  const pDet=primaryFrames;  // array of primary frames that have detections
  const baseOffset=sFirst-slope*pFirst;
  let bestOffset=baseOffset, bestScore=-1;
  for(let delta=-300; delta<=300; delta++){
    const off=baseOffset+delta;
    let score=0;
    for(const e of pDet){ if(sSet.has(Math.round(e*slope+off))) score++; }
    if(score>bestScore){ bestScore=score; bestOffset=off; }
  }
  return {slope, offset:bestOffset};
}

function cpCalibrate(){
  const CAL_CONF=0.05;
  const detFrames=dets=>[...new Set(dets.filter(d=>d.confidence>=CAL_CONF).map(d=>d.frame))].sort((a,b)=>a-b);
  const eF=cpDetections ? detFrames(cpDetections) : [];
  cpFusionCalibrated=false; cpHyperCalibrated=false;
  if(eF.length<10) return;
  const eClusters=cpFindClusters(eF);
  if(!eClusters.length) return;
  const fr=cpCalibrateStream(eF, eClusters, cpFusionDetections, cpFusionFrameCount, cpMax);
  if(fr){ cpFusionSlope=fr.slope; cpFusionOffset=fr.offset; cpFusionCalibrated=true; }
  const hr=cpCalibrateStream(eF, eClusters, cpHyperDetections, cpHyperFrameCount, cpMax);
  if(hr){ cpHyperSlope=hr.slope; cpHyperOffset=hr.offset; cpHyperCalibrated=true; }
}

// Binary search: index of value in sorted array ts closest to target t.
function cpFindClosest(ts, t){
  if(!ts||!ts.length) return 0;
  let lo=0, hi=ts.length-1;
  while(lo<hi){ const mid=(lo+hi)>>1; if(ts[mid]<t) lo=mid+1; else hi=mid; }
  if(lo>0 && Math.abs(ts[lo-1]-t)<Math.abs(ts[lo]-t)) return lo-1;
  return lo;
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
// Uses exact event timestamps when /api/sync data is available; falls back to calibration.
function cpFusionFrame(){
  if(cpSync?.e2vid_ts && cpSync?.rgb_ts && cpFrame<cpSync.e2vid_ts.length)
    return cpFindClosest(cpSync.rgb_ts, cpSync.e2vid_ts[cpFrame]);
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
    const [r1, r2, r3, r4] = await Promise.all([
      fetch('/api/detections/'+selSeq.id).catch(()=>null),
      selSeq.hypere2vid_done
        ? fetch('/api/detections/'+selSeq.id+'?model=hypere2vid').catch(()=>null)
        : Promise.resolve(null),
      fetch('/api/detections/'+selSeq.id+'?model=fusion').catch(()=>null),
      fetch('/api/sync/'+selSeq.id).catch(()=>null),
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
    cpSync = r4?.ok ? await r4.json().catch(()=>null) : null;
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

/* ══ NEW COMPARISON SCREEN (compare2) ══════════════════════════════════════ */

/* ── State ──────────────────────────────────────────────────────────────── */

let cp2Playing=false, cp2Timer=null, cp2Frame=0, cp2Max=0, cp2LoadedSeq=null;
let cp2E2vidDets=null, cp2HyperDets=null, cp2FusionDets=null, cp2ImgDebounce=null;
let cp2FusionFrameCount=0, cp2HyperFrameCount=0;
let cp2LeftMode='e2vid';
let cp2Sync=null;
let cp2FusionSlope=0, cp2FusionOffset=0, cp2FusionCalibrated=false;
let cp2HyperSlope=0,  cp2HyperOffset=0,  cp2HyperCalibrated=false;

/* ── Calibration ─────────────────────────────────────────────────────── */

function cp2Calibrate(){
  const CAL_CONF=0.05;
  const detFrames=dets=>[...new Set(dets.filter(d=>d.confidence>=CAL_CONF).map(d=>d.frame))].sort((a,b)=>a-b);
  const eF=cp2E2vidDets ? detFrames(cp2E2vidDets) : [];
  cp2FusionCalibrated=false; cp2HyperCalibrated=false;
  if(eF.length<10) return;
  const eClusters=cpFindClusters(eF);
  if(!eClusters.length) return;
  const fr=cpCalibrateStream(eF, eClusters, cp2FusionDets, cp2FusionFrameCount, cp2Max);
  if(fr){ cp2FusionSlope=fr.slope; cp2FusionOffset=fr.offset; cp2FusionCalibrated=true; }
  const hr=cpCalibrateStream(eF, eClusters, cp2HyperDets, cp2HyperFrameCount, cp2Max);
  if(hr){ cp2HyperSlope=hr.slope; cp2HyperOffset=hr.offset; cp2HyperCalibrated=true; }
}

/* ── Frame mapping ─────────────────────────────────────────────────── */

function cp2GetHyperFrame(){
  if(!cp2HyperFrameCount || !cp2Max) return cp2Frame;
  if(cp2HyperCalibrated)
    return Math.max(0,Math.min(cp2HyperFrameCount-1, Math.round(cp2HyperSlope*cp2Frame+cp2HyperOffset)));
  return Math.round(cp2Frame * cp2HyperFrameCount / (cp2Max + 1));
}

function cp2GetFusionFrame(){
  if(cp2Sync?.e2vid_ts && cp2Sync?.rgb_ts && cp2Frame<cp2Sync.e2vid_ts.length)
    return cpFindClosest(cp2Sync.rgb_ts, cp2Sync.e2vid_ts[cp2Frame]);
  if(!cp2FusionFrameCount || !cp2Max) return cp2Frame;
  if(cp2FusionCalibrated)
    return Math.max(0,Math.min(cp2FusionFrameCount-1, Math.round(cp2FusionSlope*cp2Frame+cp2FusionOffset)));
  return Math.round(cp2Frame * cp2FusionFrameCount / (cp2Max + 1));
}

/* ── Rendering ─────────────────────────────────────────────────────── */

function cp2RenderLeftBboxes(){
  const bboxDiv=document.getElementById('cp2-left-bboxes');
  if(!bboxDiv) return;
  const conf=getConf();
  const dets=cp2LeftMode==='hypere2vid' ? cp2HyperDets : cp2E2vidDets;
  const f=cp2LeftMode==='hypere2vid' ? cp2GetHyperFrame() : cp2Frame;
  if(!dets){ bboxDiv.innerHTML=''; return; }
  renderBboxes(dets.filter(d=>d.frame===f && d.confidence>=conf), document.getElementById('cp2-left-img'), bboxDiv);
}

function cp2RenderRgbBboxes(){
  const bboxDiv=document.getElementById('cp2-rgb-bboxes');
  if(!bboxDiv) return;
  if(!cp2FusionDets){ bboxDiv.innerHTML=''; return; }
  const conf=getConf(), ff=cp2GetFusionFrame();
  renderBboxes(cp2FusionDets.filter(d=>d.frame===ff && d.confidence>=conf), document.getElementById('cp2-rgb-img'), bboxDiv);
}

function cp2RenderFusionBboxes(){
  const bboxDiv=document.getElementById('cp2-fusion-bboxes');
  if(!bboxDiv) return;
  if(!cp2FusionDets){ bboxDiv.innerHTML=''; return; }
  const conf=getConf(), ff=cp2GetFusionFrame();
  renderBboxes(cp2FusionDets.filter(d=>d.frame===ff && d.confidence>=conf), document.getElementById('cp2-fusion-img'), bboxDiv);
}

function cp2RenderAllBboxes(){
  cp2RenderLeftBboxes(); cp2RenderRgbBboxes(); cp2RenderFusionBboxes();
}

/* ── Image loading ─────────────────────────────────────────────────── */

function cp2LoadImage(){
  clearTimeout(cp2ImgDebounce);
  const seq=selSeq.id, n=cp2Frame;
  cp2ImgDebounce=setTimeout(()=>{
    if(cp2Frame!==n) return;
    const leftImg=document.getElementById('cp2-left-img');
    const noHyper=document.getElementById('cp2-left-no-frames');
    if(cp2LeftMode==='hypere2vid' && !selSeq.hypere2vid_done){
      if(leftImg){ leftImg.src=''; leftImg.style.display='none'; }
      if(noHyper) noHyper.style.display='flex';
    } else {
      if(noHyper) noHyper.style.display='none';
      if(leftImg){
        leftImg.style.display='block';
        const lf=cp2LeftMode==='hypere2vid' ? cp2GetHyperFrame() : cp2Frame;
        const lm=cp2LeftMode==='hypere2vid' ? 'hypere2vid' : 'e2vid';
        leftImg.onload=()=>cp2RenderLeftBboxes();
        leftImg.src='/frames/'+seq+'/'+lf+'?model='+lm;
        if(leftImg.complete) cp2RenderLeftBboxes();
      }
    }
    if(cp2FusionDets){
      const ff=cp2GetFusionFrame();
      const rgbImg=document.getElementById('cp2-rgb-img');
      if(rgbImg){
        rgbImg.onload=()=>cp2RenderRgbBboxes();
        rgbImg.src='/frames_rgb/'+seq+'/'+ff;
        if(rgbImg.complete) cp2RenderRgbBboxes();
      }
      const fusionImg=document.getElementById('cp2-fusion-img');
      if(fusionImg){
        fusionImg.onload=()=>cp2RenderFusionBboxes();
        fusionImg.src='/frames_rgb/'+seq+'/'+ff;
        if(fusionImg.complete) cp2RenderFusionBboxes();
      }
    }
  }, 60);
}

/* ── Frame navigation ──────────────────────────────────────────────── */

function cp2SetFrame(n){
  cp2Frame=Math.max(0,Math.min(n,cp2Max));
  const gi=document.getElementById('cp2-goto'); if(gi) gi.value=cp2Frame;
  if(!cp2Slider.isDragging()){
    document.getElementById('cp2-sl').value=cp2Frame;
    cp2LoadImage();
  }
}

function cp2Stop(){
  cp2Playing=false;
  clearInterval(cp2Timer); cp2Timer=null;
  const btn=document.getElementById('cp2-play');
  if(btn) btn.innerHTML='<i class="ti ti-player-play"></i> Play';
}

/* ── Mode buttons ──────────────────────────────────────────────────── */

function cp2SetMode(m){
  cp2LeftMode=m;
  document.querySelectorAll('#cp2-mode-btns button').forEach(b=>{
    const active=b.dataset.cp2m===m;
    b.style.background=active?'#1a3a60':'';
    b.style.color=active?'#fff':'';
    b.style.borderColor=active?'#1a3a60':'';
  });
  const hdr=document.getElementById('cp2-left-hdr');
  if(hdr) hdr.textContent=m==='hypere2vid'?'HyperE2VID + YOLO':'e2vid + YOLO';
  if(typeof kpiCache!=='undefined' && kpiCache) cp2UpdateMetrics();
  if(typeof selSeq!=='undefined' && selSeq && cp2LoadedSeq===selSeq.id) cp2LoadImage();
}

document.querySelectorAll('#cp2-mode-btns button').forEach(b=>{
  b.addEventListener('click',()=>cp2SetMode(b.dataset.cp2m));
});

/* ── Metrics ───────────────────────────────────────────────────────── */

function cp2UpdateMetrics(){
  if(!kpiCache) return;
  const pct=v=>(v===null||v===undefined)?'—':(Math.round(+v*1000)/10).toFixed(1)+'%';
  const best=(runs,model)=>{
    const r=runs.filter(x=>x.model===model);
    return r.length ? r.reduce((b,x)=>((x.detection?.canonical?.map50||0)>(b.detection?.canonical?.map50||0)?x:b),r[0]).detection?.canonical||{} : {};
  };
  const leftModel=cp2LeftMode==='hypere2vid'?'hypere2vid':'e2vid';
  const lm=best(kpiCache,leftModel);
  document.getElementById('cp2-left-map50').textContent=pct(lm.map50);
  document.getElementById('cp2-left-map5095').textContent=pct(lm.map50_95);
  document.getElementById('cp2-left-prec').textContent=pct(lm.precision);
  document.getElementById('cp2-left-rec').textContent=pct(lm.recall);
  const fm=best(kpiCache,'fusion');
  ['cp2-fusion-map50','cp2-fusion-map5095','cp2-fusion-prec','cp2-fusion-rec',
   'cp2-fusion2-map50','cp2-fusion2-map5095','cp2-fusion2-prec','cp2-fusion2-rec'].forEach((id,i)=>{
    const el=document.getElementById(id); if(!el) return;
    const keys=['map50','map50_95','precision','recall'];
    el.textContent=pct(fm[keys[i%4]]);
  });
}

/* ── Screen load ───────────────────────────────────────────────────── */

async function comp2Load(){
  const noSeq=document.getElementById('comp2-no-seq');
  const panel=document.getElementById('comp2-panel');
  noSeq.style.display='none'; panel.style.display='none';
  if(!selSeq){ noSeq.style.display='block'; return; }
  const seqChanged=cp2LoadedSeq!==selSeq.id;
  if(seqChanged){
    cp2Stop();
    cp2LoadedSeq=selSeq.id;
    cp2Max=selSeq.frame_count-1;
    cp2E2vidDets=null; cp2HyperDets=null; cp2FusionDets=null;
    cp2FusionFrameCount=selSeq.fred_rgb_count||0;
    cp2HyperFrameCount=selSeq.hyper_frame_count||0;
    cp2FusionCalibrated=false; cp2HyperCalibrated=false;
    document.getElementById('cp2-sl').max=cp2Max;
    document.getElementById('cp2-max').textContent=cp2Max;
    document.getElementById('cp2-max2').textContent=cp2Max;
    const gi=document.getElementById('cp2-goto'); if(gi){ gi.max=cp2Max; gi.value=0; }
    const [r1,r2,r3,r4]=await Promise.all([
      fetch('/api/detections/'+selSeq.id).catch(()=>null),
      selSeq.hypere2vid_done
        ? fetch('/api/detections/'+selSeq.id+'?model=hypere2vid').catch(()=>null)
        : Promise.resolve(null),
      fetch('/api/detections/'+selSeq.id+'?model=fusion').catch(()=>null),
      fetch('/api/sync/'+selSeq.id).catch(()=>null),
    ]);
    const d1=r1?.ok ? await r1.json().catch(()=>null) : null;
    if(d1) cp2E2vidDets=d1.detections;
    const d2=r2?.ok ? await r2.json().catch(()=>null) : null;
    if(d2) cp2HyperDets=d2.detections;
    const d3=r3?.ok ? await r3.json().catch(()=>null) : null;
    const noRgb=document.getElementById('cp2-rgb-no-cache');
    const noFusion=document.getElementById('cp2-fusion-no-cache2');
    if(d3){
      cp2FusionDets=d3.detections;
      if(noRgb) noRgb.style.display='none';
      if(noFusion) noFusion.style.display='none';
    } else {
      cp2FusionDets=null;
      if(noRgb) noRgb.style.display='flex';
      if(noFusion) noFusion.style.display='flex';
    }
    cp2Sync = r4?.ok ? await r4.json().catch(()=>null) : null;
    cp2Calibrate();
  }
  panel.style.display='block';
  if(seqChanged) cp2SetFrame(0);
  await getKpis();
  cp2UpdateMetrics();
}

/* ── Controls ──────────────────────────────────────────────────────── */

const cp2Slider=bindSlider('cp2-sl',
  v=>{ cp2Frame=Math.max(0,Math.min(v,cp2Max)); const gi=document.getElementById('cp2-goto'); if(gi) gi.value=cp2Frame; },
  v=>{ cp2Frame=Math.max(0,Math.min(v,cp2Max)); const gi=document.getElementById('cp2-goto'); if(gi) gi.value=cp2Frame; cp2LoadImage(); }
);

document.getElementById('cp2-play').addEventListener('click',function(){
  if(!selSeq||!selSeq.e2vid_done) return;
  cp2Playing=!cp2Playing;
  clearInterval(cp2Timer); cp2Timer=null;
  if(cp2Playing){
    this.innerHTML='<i class="ti ti-player-pause"></i> Pause';
    cp2Timer=setInterval(()=>{ if(!cp2Slider.isDragging()) cp2SetFrame(cp2Frame>=cp2Max?0:cp2Frame+1); },150);
  } else {
    this.innerHTML='<i class="ti ti-player-play"></i> Play';
  }
});
document.getElementById('cp2-stop').addEventListener('click',()=>{ cp2Stop(); cp2SetFrame(0); });
document.getElementById('cp2-prev').addEventListener('click',()=>{ cp2Stop(); cp2SetFrame(cp2Frame-1); });
document.getElementById('cp2-next').addEventListener('click',()=>{ cp2Stop(); cp2SetFrame(cp2Frame+1); });
document.getElementById('cp2-goto').addEventListener('change',function(){ cp2Stop(); cp2SetFrame(parseInt(this.value)||0); });
document.getElementById('cp2-next-det').addEventListener('click',()=>{
  const conf=getConf();
  const dets=cp2LeftMode==='hypere2vid' ? cp2HyperDets : cp2E2vidDets;
  if(!dets) return;
  let frames;
  if(cp2LeftMode==='hypere2vid' && cp2HyperFrameCount && cp2Max){
    const fs=new Set();
    dets.filter(d=>d.confidence>=conf).forEach(d=>{
      const e=Math.round(d.frame*(cp2Max+1)/cp2HyperFrameCount); if(e>=0&&e<=cp2Max) fs.add(e);
    });
    frames=[...fs].sort((a,b)=>a-b);
  } else {
    frames=[...new Set(dets.filter(d=>d.confidence>=conf).map(d=>d.frame))].sort((a,b)=>a-b);
  }
  const next=frames.find(f=>f>cp2Frame)??frames[0];
  if(next!=null){ cp2Stop(); cp2SetFrame(next); }
});
cp2SetMode('e2vid'); // initialise button highlight after all listeners are wired

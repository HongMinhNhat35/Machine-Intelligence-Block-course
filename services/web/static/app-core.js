const detModels={
  'e2vid':{lbl:'Reconstruction (e2vid)',sub:'event recon. frame'},
  'hyper':{lbl:'Reconstruction (HyperE2VID)',sub:'event recon. frame'},
  'fusion':{lbl:'Fusion input',sub:'RGB + event frames'}
};
const cfg={
  home:{title:'Hybrid Vision System',badge:'Group 1 · TU Munich',extra:''},
  upload:{title:'Upload sequence',badge:'no sequence selected',extra:''},
  recon:{title:'Reconstruction viewer',badge:'seq_0 · 871 frames',extra:''},
  detect:{title:'Detection',badge:'e2vid + YOLO',extra:`<div class="sb-lbl">Confidence</div><div class="conf-wrap"><div class="conf-top"><span>Threshold</span><span id="cval">0.20</span></div><input type="range" style="width:100%;accent-color:#1a3a60" min="0" max="100" value="20" id="csl"></div>`},
  compare2:{title:'Comparison',badge:'',extra:`<div class="sb-lbl">Confidence</div><div class="conf-wrap"><div class="conf-top"><span>Threshold</span><span id="cval3">0.20</span></div><input type="range" style="width:100%;accent-color:#1a3a60" min="0" max="100" value="20" id="csl3"></div>`},
  kpis:{title:'KPIs',badge:'',extra:''},
  dataset:{title:'Dataset',badge:'5 sequences · FRED',extra:''},
  admin:{title:'Admin',badge:'',extra:''}
};
function go(k){
  document.querySelectorAll('.screen').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.tab,.nav-item').forEach(el=>el.classList.toggle('active',el.dataset.s===k));
  document.getElementById('s-'+k).classList.add('active');
  document.getElementById('tt').textContent=cfg[k].title;
  document.getElementById('tb').textContent=cfg[k].badge;
  document.getElementById('sb-extra').innerHTML=cfg[k].extra;
  if(k==='kpis')  loadKpis();
  if(k==='detect'){
    const cs=document.getElementById('csl'),cv=document.getElementById('cval');
    if(cs){
      cs.value=sharedConfPct;
      if(cv) cv.textContent=(sharedConfPct/100).toFixed(2);
      cs.addEventListener('input',()=>{
        sharedConfPct=+cs.value;
        if(cv) cv.textContent=(sharedConfPct/100).toFixed(2);
        dtRenderBboxes();
      });
    }
    // tb label is kept in sync by dtSetMode()
  }
  if(k==='compare2'){
    const cs3=document.getElementById('csl3'),cv3=document.getElementById('cval3');
    if(cs3){
      cs3.value=sharedConfPct;
      if(cv3) cv3.textContent=(sharedConfPct/100).toFixed(2);
      cs3.addEventListener('input',()=>{
        sharedConfPct=+cs3.value;
        if(cv3) cv3.textContent=(sharedConfPct/100).toFixed(2);
        cp2RenderAllBboxes();
      });
    }
  }
}
document.querySelectorAll('.tab,.nav-item').forEach(el=>el.addEventListener('click',()=>go(el.dataset.s)));
// Mode buttons are handled in app-detect.js (dtSetMode)
let allSeqs=[], selSeq=null;
let sharedConfPct=20, kpiCache=null;

/* ── Shared KPI helpers ───────────────────────────────────────────────────── */

// Fetches /api/kpis and stores the result in kpiCache. Returns kpiCache so
// callers can chain .then(). Safe to call multiple times — on failure leaves
// any previously loaded cache intact.
async function getKpis(){
  try{ kpiCache=await fetch('/api/kpis').then(r=>r.json()); }
  catch(e){ kpiCache=kpiCache||[]; }
  return kpiCache;
}

// Refreshes the mAP@0.5 sidebar badges for both models from kpiCache.
// Called on startup and from cpUpdateMetrics after the comparison screen loads.
function updateSidebarMap50(){
  if(!kpiCache) return;
  const pct=v=>(v===null||v===undefined)?'—':(Math.round(+v*1000)/10).toFixed(1)+'%';
  const bestRun=runs=>runs.find(r=>r.featured||r.deployed)||runs.reduce((b,r)=>((r.detection?.canonical?.map50||0)>(b.detection?.canonical?.map50||0)?r:b),runs[0]);
  const e2vidRuns=kpiCache.filter(r=>r.model==='e2vid');
  const e2vidEl=document.getElementById('sb-e2vid-map50');
  if(e2vidEl){
    const v=e2vidRuns.length ? bestRun(e2vidRuns).detection?.canonical?.map50 : undefined;
    e2vidEl.textContent=pct(v);
  }
  const hyperRuns=kpiCache.filter(r=>r.model==='hypere2vid');
  const hyperEl=document.getElementById('sb-hyper-map50');
  if(hyperEl){
    const v=hyperRuns.length ? bestRun(hyperRuns).detection?.canonical?.map50 : undefined;
    hyperEl.textContent=pct(v);
  }
  const fusionRuns=kpiCache.filter(r=>r.model==='fusion');
  const fusionEl=document.getElementById('sb-fusion-map50');
  if(fusionEl){
    const v=fusionRuns.length ? bestRun(fusionRuns).detection?.canonical?.map50 : undefined;
    fusionEl.textContent=v!==undefined ? '>'+pct(v) : '—';
  }
}

// Screen-specific hooks on navigation
document.querySelectorAll('.tab,.nav-item').forEach(el=>el.addEventListener('click',()=>{
  if(el.dataset.s!=='recon')    pbStop();
  if(el.dataset.s!=='detect')   dtStop();
  if(el.dataset.s!=='compare2') cp2Stop();
  if(el.dataset.s==='recon')    reconLoad();
  if(el.dataset.s==='detect')   detLoad();
  if(el.dataset.s==='compare2') comp2Load();
  if(el.dataset.s==='admin')    loadAdmin();
}));

getKpis().then(updateSidebarMap50).then(()=>{
  lpPopulateTable();
});

function lpPopulateTable(){
  const pct=v=>(v===null||v===undefined)?'—':(Math.round(+v*1000)/10).toFixed(1)+'%';
  const bestRun=runs=>runs.find(r=>r.featured||r.deployed)||runs.reduce((b,r)=>((r.detection?.canonical?.map50||0)>(b.detection?.canonical?.map50||0)?r:b),runs[0]);
  const body=document.getElementById('lp-kpi-body');
  const map50El=document.getElementById('lp-map50');
  if(!kpiCache||!body) return;
  const modelOrder=['e2vid','hypere2vid','fusion'];
  const modelLabel={'e2vid':'E2VID + YOLO','hypere2vid':'HyperE2VID + YOLO','fusion':'Late Fusion'};
  let fusionVal=null;
  const rows=modelOrder.map(m=>{
    const runs=kpiCache.filter(r=>r.model===m);
    if(!runs.length) return '';
    const r=bestRun(runs);
    const v=r.detection?.canonical?.map50;
    const isFusion=m==='fusion';
    if(isFusion && v!=null) fusionVal=v;
    const cls=isFusion?' class="lp-fusion"':'';
    return `<tr${cls}><td>${modelLabel[m]}</td><td>${r.run_id||'—'}</td><td>${pct(v)}</td><td>${pct(r.detection?.canonical?.precision)}</td><td>${pct(r.detection?.canonical?.recall)}</td></tr>`;
  }).join('');
  const paperRows=`<tr><td style="color:#888">YOLOv11 (FRED paper)</td><td>—</td><td>87.7%</td><td>—</td><td>—</td></tr>
  <tr><td style="color:#888">ER-DETR (FRED paper)</td><td>—</td><td>78.6%</td><td>—</td><td>—</td></tr>`;
  body.innerHTML=rows+paperRows;
  if(map50El && fusionVal!=null) map50El.textContent=pct(fusionVal);
}

async function lpStartPlayer(){
  const SEQ='sequence_201', START=147;
  const cvEv=document.getElementById('lp-cv-ev');
  const cvRgb=document.getElementById('lp-cv-rgb');
  const cvFus=document.getElementById('lp-cv-fus');
  if(!cvEv||!cvRgb||!cvFus) return;

  let frameCount=800;
  try{
    const seqs=allSeqs.length?allSeqs:await fetch('/api/sequences').then(r=>r.json());
    const s=seqs.find(x=>x.id===SEQ);
    if(s?.frame_count) frameCount=s.frame_count;
  }catch(e){}

  let syncE=[], syncR=[];
  try{
    const sd=await fetch('/api/sync/'+SEQ).then(r=>r.json());
    syncE=sd.e2vid_ts||[]; syncR=sd.rgb_ts||[];
  }catch(e){}
  function bisectRight(arr,v){let lo=0,hi=arr.length;while(lo<hi){const m=(lo+hi)>>1;arr[m]<=v?lo=m+1:hi=m;}return lo;}
  function toRgb(ef){
    if(syncE.length&&syncR.length&&ef<syncE.length){
      const t=syncE[ef], idx=bisectRight(syncR,t);
      return Math.min(Math.max(idx,0),syncR.length-1);
    }
    return Math.round(ef*syncR.length/Math.max(syncE.length,1));
  }

  const evMap={}, fusMap={};
  try{
    const ed=await fetch('/api/detections/'+SEQ+'?model=e2vid').then(r=>r.json());
    (ed.detections||[]).forEach(d=>{(evMap[d.frame]=evMap[d.frame]||[]).push(d);});
  }catch(e){}
  try{
    const fd=await fetch('/api/detections/'+SEQ+'?model=fusion').then(r=>r.json());
    (fd.detections||[]).forEach(d=>{(fusMap[d.frame]=fusMap[d.frame]||[]).push(d);});
  }catch(e){}

  const pillEv=document.getElementById('lp-pill-ev');
  const pillFus=document.getElementById('lp-pill-fus');
  if(pillEv) pillEv.textContent='e2vid boxes';
  if(pillFus) pillFus.textContent='fusion boxes';

  // bbox is [cx,cy,w,h] in absolute pixels; coords are in natural img dimensions
  function drawBoxes(ctx,boxes,color){
    if(!boxes||!boxes.length) return;
    ctx.strokeStyle=color; ctx.lineWidth=8;
    boxes.forEach(d=>{
      if(d.confidence<0.2) return;
      const [cx,cy,w,h]=d.bbox;
      ctx.strokeRect(cx-w/2, cy-h/2, w, h);
    });
  }

  function loadImg(src){
    return new Promise(res=>{
      const img=new Image(); img.onload=()=>res(img); img.onerror=()=>res(null); img.src=src;
    });
  }

  let frame=START;
  async function tick(){
    const rgbF=toRgb(frame);
    const [imgEv,imgRgb]=await Promise.all([
      loadImg('/frames/'+SEQ+'/'+frame),
      loadImg('/frames_rgb/'+SEQ+'/'+rgbF)
    ]);
    if(!imgEv){ frame=START; setTimeout(tick,160); return; }
    cvEv.width=imgEv.naturalWidth; cvEv.height=imgEv.naturalHeight;
    const ctxEv=cvEv.getContext('2d'); ctxEv.drawImage(imgEv,0,0);
    drawBoxes(ctxEv,evMap[frame],'#38bdf8');
    if(imgRgb){
      cvRgb.width=imgRgb.naturalWidth; cvRgb.height=imgRgb.naturalHeight;
      cvRgb.getContext('2d').drawImage(imgRgb,0,0);
      cvFus.width=imgRgb.naturalWidth; cvFus.height=imgRgb.naturalHeight;
      const ctxFus=cvFus.getContext('2d'); ctxFus.drawImage(imgRgb,0,0);
      drawBoxes(ctxFus,fusMap[rgbF],'#fb923c');
    }
    frame++;
    if(frame>=frameCount) frame=START;
    setTimeout(tick,160);
  }
  tick();
}

go('home');

const detModels={
  'e2vid':{lbl:'Reconstruction (e2vid)',sub:'event recon. frame'},
  'hyper':{lbl:'Reconstruction (HyperE2VID)',sub:'event recon. frame'},
  'fusion':{lbl:'Fusion input',sub:'RGB + event frames'}
};
const cfg={
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
  if(el.dataset.s==='kpis')     loadKpis();
  if(el.dataset.s==='admin')    loadAdmin();
}));

getKpis().then(updateSidebarMap50);

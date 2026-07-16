/*
 * AMI Hybrid Vision System — Group 1
 * File: app-admin.js
 *
 * Admin screen — service health, data paths, sequence readiness, model weights.
 * KPIs screen — detection, reconstruction, and training metric tables.
 * Dataset screen — static content, no dynamic JS required.
 *
 * Load order: after app-upload.js, before app-recon.js and app-core.js
 * Globals consumed: kpiCache (app-core.js), getKpis (app-core.js)
 * Globals exported: loadAdmin, loadKpis
 *                   — called from app-core.js tab navigation hook and at startup
 */

/* ── Admin screen ─────────────────────────────────────────────────────────── */

// Fetches /api/admin and populates the Admin screen: service health dots,
// data path display, sequence list with e2vid/hyper readiness badges,
// model weight paths, and sidebar service status dots.
async function loadAdmin(){
  try{
    const d=await fetch('/api/admin').then(r=>r.json());

    const dotCls={ok:'dot-g',warn:'dot-y',error:'dot-x'};
    document.getElementById('svc-health').innerHTML=d.services.map(s=>`
      <div class="svc-adm">
        <div class="svc-left">
          <div class="dot ${dotCls[s.status]||'dot-x'}"></div>
          <span>${s.name}</span>
          <span class="svc-port">:${s.port}</span>
          <span class="svc-detail">· ${s.detail}</span>
        </div>
      </div>`).join('');

    document.getElementById('path-fred').textContent=d.paths.FRED_DATA_PATH;
    document.getElementById('path-recon').textContent=d.paths.RECON_DATA_PATH;

    const n=d.sequences.length;
    const volEl=document.getElementById('vol-count');
    const icon=n>0?'<i class="ti ti-check vol-ok"></i>':'<i class="ti ti-alert-triangle vol-warn"></i>';
    volEl.innerHTML=`${icon} FRED raw data — ${n} sequence${n!==1?'s':''} found`;
    document.getElementById('seq-badges').innerHTML=d.sequences.map(s=>
      `<span class="seq-badge ${s.e2vid?'sb-ok':'sb-warn'}">${s.id}</span>`).join('');

    const e2vidReady=d.sequences.filter(s=>s.e2vid).map(s=>s.id);
    const e2vidEl=document.getElementById('e2vid-recon');
    if(e2vidReady.length){
      e2vidEl.innerHTML=`<i class="ti ti-check vol-ok"></i> e2vid — ${e2vidReady.join(', ')} ready`;
    }else{
      e2vidEl.innerHTML=`<i class="ti ti-alert-triangle vol-warn"></i> e2vid — no frames found`;
    }
    const hyperReady=d.sequences.filter(s=>s.hypere2vid).map(s=>s.id);
    const hyperEl=document.getElementById('hyper-recon');
    if(hyperReady.length){
      hyperEl.innerHTML=`<i class="ti ti-check vol-ok"></i> HyperE2VID — ${hyperReady.join(', ')} ready`;
    }else{
      hyperEl.innerHTML=`<i class="ti ti-alert-triangle vol-warn"></i> HyperE2VID — no frames found`;
    }

    const ms=d.model_settings;
    document.getElementById('ms-e2vid').textContent=ms.e2vid_weights;
    document.getElementById('ms-hyper').textContent=ms.hypere2vid_weights;
    document.getElementById('ms-fusion').textContent=ms.fusion_weights;

    const sbSvcIds={e2vid:'sb-svc-e2vid',hypere2vid:'sb-svc-hyper',fusion:'sb-svc-fusion'};
    const sbSvcLabels={e2vid:'e2vid',hypere2vid:'HyperE2VID',fusion:'Fusion'};
    d.services.forEach(s=>{
      const row=document.getElementById(sbSvcIds[s.name]);
      if(!row) return;
      const cls='dot '+(dotCls[s.status]||'dot-x');
      const label=sbSvcLabels[s.name]||s.name;
      row.innerHTML=`<div class="${cls}"></div>${label} · ${s.detail}`;
    });

  }catch(e){
    document.getElementById('svc-health').innerHTML=
      '<div class="adm-row" style="color:#d00;font-size:11px">Failed to load — is the backend running?</div>';
  }
}

/* ── KPIs screen ──────────────────────────────────────────────────────────── */

// Format helpers — all return '—' for null/undefined values.
function kpiFmt(v, decimals=1, suffix=''){ return (v===null||v===undefined) ? '—' : (+v).toFixed(decimals)+suffix; }
function kpiPct(v){ return (v===null||v===undefined) ? '—' : (Math.round(+v*1000)/10).toFixed(1)+'%'; }
function kpiHrs(s){ return (s===null||s===undefined) ? '—' : (+s/3600).toFixed(1)+' h'; }
function kpiNum(v){ return (v===null||v===undefined) ? '—' : (+v).toLocaleString(); }
function kpiRun(id){ return id ? id.replace(/^run_?/i,'') : '—'; }
function kpiSeqs(seqs){ return Array.isArray(seqs) && seqs.length ? seqs.map(s=>s.replace('sequence_','')).join(', ') : '—'; }

const MODEL_LABELS = {
  'e2vid':        'e2vid + YOLOv8s',
  'hypere2vid':   'HyperE2VID + YOLOv8n',
  'fusion':       'Late Fusion — Combined',
  'fusion_event': 'Late Fusion — Event',
  'fusion_rgb':   'Late Fusion — RGB',
};
function modelLabel(r){
  if (r.detector) return (r.model||'') + ' + ' + r.detector;
  return MODEL_LABELS[r.model] || r.model || '—';
}

// Fetches /api/kpis and populates the summary table and three detail tables.
async function loadKpis(){
  try{
    const runs = (await fetch('/api/kpis').then(r=>r.json())).filter(r=>r.model);

    // ── Summary: prefer featured/deployed run; fall back to highest mAP ────────
    const modelOrder = ['e2vid','hypere2vid','fusion','fusion_event','fusion_rgb'];
    const gt = s => `<span style="color:var(--c-muted,#888);font-weight:600">&gt;</span>${s}`;
    const summaryRows = modelOrder.map(model => {
      const modelRuns = runs.filter(r => r.model === model);
      if (!modelRuns.length) return '';
      const featured = modelRuns.find(r => r.featured || r.deployed);
      const best = featured || modelRuns.reduce((b,r) => ((r.detection?.canonical?.map50||0) > (b.detection?.canonical?.map50||0) ? r : b));
      const c = best.detection?.canonical || {};
      const rc = best.reconstruction || {};
      const tr = best.training || {};
      const lbl = modelLabel(best);
      const valSeqs = kpiSeqs(tr.val_sequences);
      let totalS = (rc.total_runtime_s||0) + (tr.runtime_s||0);
      if (model === 'fusion') {
        const compRuntime = m => {
          const rs = runs.filter(r=>r.model===m);
          const b = rs.find(r=>r.featured||r.deployed)||rs.reduce((a,r)=>((r.detection?.canonical?.map50||0)>(a.detection?.canonical?.map50||0)?r:a),rs[0]);
          return b?.training?.runtime_s||0;
        };
        totalS += compRuntime('fusion_rgb') + compRuntime('fusion_event');
      }
      const runtime = totalS > 0 ? kpiHrs(totalS) : '—';
      const isCombined = model === 'fusion';
      const cls = isCombined ? 'best ours' : 'ours';
      const wrap = v => isCombined ? gt(v) : v;
      return `<tr class="${cls}"><td>${lbl}</td><td>${kpiRun(best.run_id)}</td><td>${valSeqs}</td>
        <td>${wrap(kpiPct(c.map50))}</td><td>${wrap(kpiPct(c.map50_95))}</td>
        <td>${wrap(kpiPct(c.precision))}</td><td>${wrap(kpiPct(c.recall))}</td>
        <td>${runtime}</td></tr>`;
    });
    document.getElementById('kpi-summary-body').innerHTML = summaryRows.join('');

    // ── Detection detail table ────────────────────────────────────────────────
    const detOrder = ['e2vid','hypere2vid','fusion','fusion_event','fusion_rgb'];
    const runNum = r => parseInt((r.run_id||'').replace(/\D/g,''))||0;
    const byModelRun = (a,b) => {
      const ai=detOrder.indexOf(a.model), bi=detOrder.indexOf(b.model);
      if (ai !== bi) return (ai<0?99:ai)-(bi<0?99:bi);
      return runNum(a)-runNum(b);
    };
    const orderedRuns = [...runs].sort(byModelRun);
    const isCombined = r => r.model === 'fusion';
    document.getElementById('kpi-det-body').innerHTML = orderedRuns.length ? orderedRuns.map(r=>{
      const c=r.detection?.canonical||{};
      const rc=r.reconstruction||{}, tr=r.training||{};
      const lbl=modelLabel(r);
      const valSeqs=kpiSeqs(tr.val_sequences);
      let totalS=(rc.total_runtime_s||0)+(tr.runtime_s||0);
      if(isCombined(r)){
        const compRuntime=m=>{
          const rs=runs.filter(x=>x.model===m);
          const b=rs.find(x=>x.featured||x.deployed)||rs.reduce((a,x)=>((x.detection?.canonical?.map50||0)>(a.detection?.canonical?.map50||0)?x:a),rs[0]);
          return b?.training?.runtime_s||0;
        };
        totalS+=compRuntime('fusion_rgb')+compRuntime('fusion_event');
      }
      const runtime=totalS>0?kpiHrs(totalS):'—';
      const cls=isCombined(r)?'best ours':'ours';
      const gt2 = s => isCombined(r) ? `<span style="color:var(--c-muted,#888);font-weight:600">&gt;</span>${s}` : s;
      return `<tr class="${cls}"><td>${lbl}</td><td>${kpiRun(r.run_id)}</td><td>${valSeqs}</td>
        <td>${gt2(kpiPct(c.map50))}</td><td>${gt2(kpiPct(c.map50_95))}</td>
        <td>${gt2(kpiPct(c.precision))}</td><td>${gt2(kpiPct(c.recall))}</td>
        <td>${runtime}</td></tr>`;
    }).join('') : '<tr><td colspan="8" style="color:#aaa">No KPI files found.</td></tr>';

    document.getElementById('kpi-rec-body').innerHTML = runs.length ? [...runs].sort(byModelRun).map(r=>{
      const rc=r.reconstruction||{};
      const lbl=(r.model||'')+(r.detector?' + '+r.detector:'');
      return `<tr class="ours"><td>${lbl}</td><td>${kpiRun(r.run_id)}</td><td>${kpiSeqs(rc.sequences)}</td>
        <td>${rc.events_per_pixel!==null&&rc.events_per_pixel!==undefined?rc.events_per_pixel:'—'}</td>
        <td>${kpiNum(rc.total_frames)}</td><td>${kpiHrs(rc.total_runtime_s)}</td>
        <td>${kpiFmt(rc.avg_fps,2,' fps')}</td><td>${rc.gpu||'—'}</td></tr>`;
    }).join('') : '<tr><td colspan="8" style="color:#aaa">No KPI files found.</td></tr>';

    document.getElementById('kpi-tr-body').innerHTML = runs.length ? [...runs].sort(byModelRun).map(r=>{
      const tr=r.training||{};
      const lbl=(r.model||'')+(r.detector?' + '+r.detector:'');
      const epochs=(tr.epochs_completed!==null&&tr.epochs_completed!==undefined)
        ? `${tr.epochs_completed} / ${tr.epochs_requested||'—'}` : '—';
      return `<tr class="ours"><td>${lbl}</td><td>${kpiRun(r.run_id)}</td><td>${kpiSeqs(tr.train_sequences)}</td>
        <td>${kpiNum(tr.n_train_images)}</td><td>${kpiHrs(tr.runtime_s)}</td>
        <td>${epochs}</td><td>${tr.best_epoch!==null&&tr.best_epoch!==undefined?tr.best_epoch:'—'}</td>
        <td>${tr.lr0!==null&&tr.lr0!==undefined?tr.lr0:'—'}</td>
        <td>${tr.effective_batch!==null&&tr.effective_batch!==undefined?tr.effective_batch:'—'}</td>
        <td>${tr.gpu||'—'}</td></tr>`;
    }).join('') : '<tr><td colspan="10" style="color:#aaa">No KPI files found.</td></tr>';

  }catch(e){
    ['kpi-det-body','kpi-rec-body','kpi-tr-body'].forEach(id=>{
      const el=document.getElementById(id);
      if(el) el.innerHTML='<tr><td colspan="10" style="color:#d00">Failed to load.</td></tr>';
    });
  }
}

/* ── Startup ──────────────────────────────────────────────────────────────── */

loadAdmin();

/*
 * AMI Hybrid Vision System — Group 1
 * File: app-upload.js
 *
 * Upload screen — sequence browser, file-status rows, and precompute-to-cache
 * controls. Also owns onSeqSelected which updates the sidebar and triggers
 * screen refreshes across all player modules when the active sequence changes.
 *
 * Load order: after app-utils.js, before app-admin.js, app-recon.js, and app-core.js
 * Globals consumed: allSeqs, selSeq, kpiCache (app-core.js),
 *                   getKpis, updateSidebarMap50 (app-core.js),
 *                   reconLoad (app-recon.js), dtLoadDetections, dtLoadedSeq (app-detect.js)
 * Globals exported: onSeqSelected — called by initBrowser click handlers
 *                   checkPrecomputeStatus — called by onSeqSelected and runPrecompute
 */

/* ── File status rows (Upload panel) ─────────────────────────────────────── */

// Renders a single file-status row in the Upload panel.
// tagOk colours the tag badge green; false leaves it grey.
function fileRow(id, icon, text, tag, tagOk){
  const el=document.getElementById(id);
  el.innerHTML=`<i class="ti ${icon}" style="color:#1a3a60"></i> <span>${text}</span> <span class="file-tag${tagOk?' ok':''}">${tag}</span>`;
  el.style.display='flex';
}

/* ── Sequence selection ───────────────────────────────────────────────────── */

// Central handler called whenever the user clicks a sequence in the browser.
// Updates selSeq, the topbar badge, the sidebar info block, the Upload panel
// file-status rows, the precompute section visibility, and triggers reconLoad()
// so the reconstruction player reflects the new sequence immediately.
function onSeqSelected(seq){
  selSeq=seq;
  document.getElementById('tb').textContent=seq ? seq.id : 'no sequence selected';
  const block=document.getElementById('sb-seq-block');
  if(block) block.style.display=seq ? '' : 'none';
  if(seq){
    const es=document.getElementById('sb-seq'); if(es) es.textContent=seq.id;
    const set=(id,v)=>{const el=document.getElementById(id);if(el)el.textContent=v;};
    set('sb-e2vid-frames', seq.e2vid_done ? seq.frame_count : '—');
    set('sb-e2vid-dets', '—');
    set('sb-hyper-frames', seq.hypere2vid_done ? seq.hyper_frame_count : '—');
    set('sb-hyper-dets', '—');
    set('sb-fusion-frames', seq.fred_rgb_count > 0 ? seq.fred_rgb_count : '—');
    set('sb-fusion-dets', '—');
    getKpis().then(()=>{
      const e2vidRun=kpiCache?kpiCache.filter(r=>r.model==='e2vid').slice(-1)[0]:null;
      const hyperRun=kpiCache?kpiCache.filter(r=>r.model==='hypere2vid').slice(-1)[0]:null;
      const pct=v=>(v===null||v===undefined)?'—':(Math.round(+v*1000)/10).toFixed(1)+'%';
      set('sb-e2vid-map50', e2vidRun ? pct(e2vidRun.detection?.canonical?.map50) : '—');
      set('sb-hyper-map50', hyperRun ? pct(hyperRun.detection?.canonical?.map50) : '—');
    });
  }
  if(seq){
    fileRow('df-events',
      seq.has_events_zip ? 'ti-file-zip' : seq.has_events_h5 ? 'ti-file-code' : seq.has_raw_events ? 'ti-file-code' : 'ti-alert-triangle',
      seq.has_events_zip ? 'events.zip' : seq.has_events_h5 ? 'events.h5' : seq.has_raw_events ? 'Event/ (FRED raw)' : 'no events file found',
      seq.has_events_zip||seq.has_events_h5||seq.has_raw_events ? 'events' : 'missing',
      seq.has_events_zip||seq.has_events_h5||seq.has_raw_events);
    if(seq.has_coordinates) fileRow('df-coords','ti-map-pin','coordinates.txt','annotations',true);
    else document.getElementById('df-coords').style.display='none';
    document.getElementById('detected-files').style.display='flex';
    const pcSection=document.getElementById('det-precompute-section');
    if(seq.e2vid_done){
      pcSection.style.display='';
      document.getElementById('precomp-progress').textContent='';
    } else {
      pcSection.style.display='none';
    }
    checkPrecomputeStatus(seq.id);
  }
  reconLoad();
}

/* ── Sequence browser (Dataset panel) ────────────────────────────────────── */

// Fetches all sequences from /api/sequences, renders the browser list, and
// attaches click handlers that call onSeqSelected. Called once at startup.
async function initBrowser(){
  try{
    const resp=await fetch('/api/sequences');
    allSeqs=await resp.json();
  } catch(e){
    document.getElementById('browser-body').innerHTML='<div style="padding:14px;color:#d00;font-size:11px">Failed to load sequences.</div>';
    return;
  }
  document.getElementById('seq-count').textContent=allSeqs.length+' sequences';
  document.getElementById('browser-body').innerHTML=allSeqs.length===0
    ? '<div style="padding:14px;color:#aaa;font-size:11px">No sequences found.</div>'
    : allSeqs.map(s=>{
        const e2vidTag=s.e2vid_done
          ? `<span class="b-tag ok">e2vid ${s.frame_count}</span>`
          : `<span class="b-tag">e2vid —</span>`;
        const hyperTag=s.hypere2vid_done
          ? `<span class="b-tag ok">hyper ${s.hyper_frame_count}</span>`
          : `<span class="b-tag">hyper —</span>`;
        const fusionTag=s.fred_rgb_count>0
          ? `<span class="b-tag ok">fusion ${s.fred_rgb_count}</span>`
          : `<span class="b-tag">fusion —</span>`;
        return `<div class="b-row" data-val="${s.id}"><i class="ti ti-folder-open" style="color:#f59e0b"></i> ${s.id}${e2vidTag}${hyperTag}${fusionTag}</div>`;
      }).join('');
  document.querySelectorAll('#browser-body .b-row').forEach(el=>{
    el.addEventListener('click',()=>{
      document.querySelectorAll('#browser-body .b-row').forEach(r=>r.classList.remove('selected'));
      el.classList.add('selected');
      onSeqSelected(allSeqs.find(s=>s.id===el.dataset.val));
    });
  });
}

/* ── Pre-compute detection cache ──────────────────────────────────────────── */

// Checks /api/detections for both e2vid and hypere2vid and updates the
// precompute status line and sidebar detection-count badges.
// Called when a sequence is selected and after runPrecompute finishes.
async function checkPrecomputeStatus(seqId){
  const el=document.getElementById('precomp-cache-status');
  const parts=[];
  const sbKeys={e2vid:'sb-e2vid-dets',hypere2vid:'sb-hyper-dets',fusion:'sb-fusion-dets'};
  for(const model of ['e2vid','hypere2vid','fusion']){
    try{
      const r=await fetch('/api/detections/'+seqId+'?model='+model);
      if(r.ok){
        const data=await r.json();
        const n=data.detections?.length||0;
        parts.push(model+': '+n+' det');
        const sbD=document.getElementById(sbKeys[model]); if(sbD) sbD.textContent=n;
      } else {
        parts.push(model+': not cached');
        const sbD=document.getElementById(sbKeys[model]); if(sbD) sbD.textContent='—';
      }
    } catch(e){ parts.push(model+': —'); }
  }
  el.textContent=parts.join(' · ');
  el.style.color='#555';
}

// Streams the SSE response from /api/detect to run full-sequence YOLO detection
// and cache the results. Updates the progress label line-by-line. When done,
// refreshes the cache status and reloads detections in the detection screen if
// the same sequence is still loaded there.
async function runPrecompute(btn, model){
  if(!selSeq) return;
  const frameCount=model==='fusion'     ? selSeq.fred_rgb_count :
                   model==='hypere2vid' ? selSeq.hyper_frame_count : selSeq.frame_count;
  if(!frameCount){ alert('No '+model+' frames available for this sequence.'); return; }
  const prog=document.getElementById('precomp-progress');
  const statusEl=document.getElementById('precomp-cache-status');
  btn.disabled=true;
  prog.textContent='['+model+'] Starting…';
  statusEl.textContent='Running '+model+'…';
  statusEl.style.color='#888';
  try{
    const r=await fetch('/api/detect',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({sequence_id:selSeq.id,frame_count:frameCount,model})});
    const reader=r.body.getReader(), dec=new TextDecoder();
    let buf='', eventType=null;
    while(true){
      const {done,value}=await reader.read();
      if(done) break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\n'); buf=lines.pop();
      for(const line of lines){
        if(line.startsWith('event: ')) eventType=line.slice(7).trim();
        else if(line.startsWith('data: ')){
          const data=line.slice(6).trim();
          if(eventType==='log'||eventType==='error') prog.textContent='['+model+'] '+data;
          if(eventType==='done'&&data==='ok'){
            prog.textContent='['+model+'] Done!';
            await checkPrecomputeStatus(selSeq.id);
            if(dtLoadedSeq===selSeq.id) await dtLoadDetections(selSeq.id, model);
          }
          eventType=null;
        }
      }
    }
  } catch(e){ prog.textContent='['+model+'] Failed: '+e.message; }
  btn.disabled=false;
}

document.getElementById('precomp-e2vid-btn').addEventListener('click',function(){ runPrecompute(this,'e2vid'); });
document.getElementById('precomp-hyper-btn').addEventListener('click',function(){ runPrecompute(this,'hypere2vid'); });
document.getElementById('precomp-fusion-btn').addEventListener('click',function(){ runPrecompute(this,'fusion'); });

/* ── Startup ──────────────────────────────────────────────────────────────── */

initBrowser();

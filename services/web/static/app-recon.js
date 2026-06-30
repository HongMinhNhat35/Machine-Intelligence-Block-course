/*
 * AMI Hybrid Vision System — Group 1
 * File: app-recon.js
 *
 * Reconstruction viewer screen — plays back e2vid and HyperE2VID frames
 * for the selected sequence with a scrubable slider and play/pause controls.
 *
 * Load order: after app-utils.js and app-dataset.js, before app.js
 * Globals consumed: selSeq (app.js), bindSlider (app-utils.js)
 * Globals exported: reconLoad() — called by onSeqSelected (app-dataset.js)
 *                                  and the tab navigation hook (app.js)
 */

/* ── State ────────────────────────────────────────────────────────────────── */

let pbPlaying=false, pbTimer=null, pbFrame=0, pbMax=0, pbLoadedSeq=null;
let pbImgDebounce=null;

/* ── Image loading ────────────────────────────────────────────────────────── */

// Debounced frame loader — waits 60 ms after the last call before setting
// img.src, so rapid scrubbing doesn't flood the server with requests.
// Loads both the e2vid frame and, if the hyper panel is active, the
// HyperE2VID frame for the same index.
function pbLoadImage(){
  clearTimeout(pbImgDebounce);
  const seq=selSeq.id, n=pbFrame;
  pbImgDebounce=setTimeout(()=>{
    if(pbFrame===n){
      document.getElementById('recon-img').src='/frames/'+seq+'/'+n;
      const hyperImg=document.getElementById('recon-img-hyper');
      if(hyperImg.dataset.active==='1') hyperImg.src='/frames/'+seq+'/'+n+'?model=hypere2vid';
    }
  }, 60);
}

/* ── Frame navigation ─────────────────────────────────────────────────────── */

// Clamps n to [0, pbMax], updates the frame label and slider, and triggers
// pbLoadImage unless the user is actively dragging the slider.
function pbSetFrame(n){
  pbFrame=Math.max(0,Math.min(n,pbMax));
  document.getElementById('pb-lbl').textContent='Frame '+pbFrame;
  document.getElementById('r-pill').textContent='frame '+pbFrame+' / '+pbMax;
  const hp=document.getElementById('r-pill-hyper');
  if(hp && document.getElementById('recon-img-hyper').dataset.active==='1')
    hp.textContent='frame '+pbFrame;
  const gi=document.getElementById('pb-goto'); if(gi) gi.value=pbFrame;
  if(!pbSlider.isDragging()){
    document.getElementById('pb-sl').value=pbFrame;
    pbLoadImage();
  }
}

// Stops the play timer and resets the Play button label.
function pbStop(){
  pbPlaying=false;
  clearInterval(pbTimer);
  pbTimer=null;
  document.getElementById('pb-play').innerHTML='<i class="ti ti-player-play"></i> Play';
}

/* ── Screen load ──────────────────────────────────────────────────────────── */

// Called by onSeqSelected and by the tab navigation hook whenever the
// Reconstruction screen becomes active. Resets state on sequence change,
// shows/hides the HyperE2VID panel based on selSeq.hypere2vid_done, and
// jumps to frame 0.
function reconLoad(){
  document.getElementById('recon-no-seq').style.display='none';
  document.getElementById('recon-no-frames').style.display='none';
  document.getElementById('recon-player').style.display='none';

  if(!selSeq){ document.getElementById('recon-no-seq').style.display='block'; return; }
  if(!selSeq.e2vid_done){ pbStop(); document.getElementById('recon-no-frames').style.display='block'; return; }

  const seqChanged=pbLoadedSeq!==selSeq.id;
  if(seqChanged){
    pbStop();
    pbLoadedSeq=selSeq.id;
    pbMax=selSeq.frame_count-1;
    document.getElementById('pb-sl').max=pbMax;
    document.getElementById('pb-max').textContent=pbMax;
    const gi=document.getElementById('pb-goto'); if(gi){ gi.max=pbMax; gi.value=0; }
    // Show the HyperE2VID column only when frames for this sequence exist.
    const ph=document.getElementById('hyper-placeholder');
    const hi=document.getElementById('recon-img-hyper');
    if(selSeq.hypere2vid_done){
      hi.dataset.active='1'; hi.style.display='block'; ph.style.display='none';
      document.getElementById('r-pill-hyper').textContent='frame 0';
    } else {
      hi.dataset.active='0'; hi.style.display='none'; ph.style.display='flex';
      hi.src='';
    }
  }
  document.getElementById('recon-player').style.display='block';
  if(seqChanged) pbSetFrame(0);
}

/* ── Controls ─────────────────────────────────────────────────────────────── */

// Slider drag-tracking — onDrag updates labels only; onCommit also loads the frame.
const pbSlider = bindSlider('pb-sl',
  v => { pbFrame=Math.max(0,Math.min(v,pbMax)); document.getElementById('pb-lbl').textContent='Frame '+pbFrame; document.getElementById('r-pill').textContent='frame '+pbFrame+' / '+pbMax; },
  v => { pbFrame=Math.max(0,Math.min(v,pbMax)); document.getElementById('pb-lbl').textContent='Frame '+pbFrame; document.getElementById('r-pill').textContent='frame '+pbFrame+' / '+pbMax; pbLoadImage(); }
);

document.getElementById('pb-play').addEventListener('click',function(){
  if(!selSeq||!selSeq.e2vid_done) return;
  pbPlaying=!pbPlaying;
  clearInterval(pbTimer);
  pbTimer=null;
  if(pbPlaying){
    this.innerHTML='<i class="ti ti-player-pause"></i> Pause';
    pbTimer=setInterval(()=>{
      if(!pbSlider.isDragging()) pbSetFrame(pbFrame>=pbMax ? 0 : pbFrame+1);
    }, 150);
  } else {
    this.innerHTML='<i class="ti ti-player-play"></i> Play';
  }
});

document.getElementById('pb-stop').addEventListener('click',()=>{pbStop();pbSetFrame(0);});
document.getElementById('pb-prev').addEventListener('click',()=>{ pbStop(); pbSetFrame(pbFrame-1); });
document.getElementById('pb-next').addEventListener('click',()=>{ pbStop(); pbSetFrame(pbFrame+1); });
document.getElementById('pb-goto').addEventListener('change',function(){ pbStop(); pbSetFrame(parseInt(this.value)||0); });

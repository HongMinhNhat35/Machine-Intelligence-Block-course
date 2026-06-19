"""
reconstruct.py — Convert FRED events → e2vid grayscale frames.

Accepts either a pre-built event zip (preferred on Colab — no OpenEB needed)
or a raw OpenEB HDF5 file (requires ECF codec plugin, available on the local laptop).

Usage with pre-built zip (Colab):
    python scripts/reconstruct.py \
        --zip_path  data/processed/sequence_0/events.zip \
        --out_dir   data/processed/sequence_0/reconstruction_e2vid

Usage with HDF5 (local, OpenEB installed):
    python scripts/reconstruct.py \
        --h5_path   data/processed/sequence_0/events.h5 \
        --out_dir   data/processed/sequence_0/reconstruction_e2vid \
        --start_s   5.0
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='FRED e2vid reconstruction pipeline')

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--zip_path', type=Path,
                     help='Pre-built event zip (no OpenEB needed — use this on Colab)')
    src.add_argument('--h5_path',  type=Path,
                     help='OpenEB events.h5 (requires ECF codec, use this locally)')

    p.add_argument('--out_dir',  required=True,  type=Path,
                   help='Output directory for reconstructed frames')
    p.add_argument('--work_dir', default=None,   type=Path,
                   help='Working directory for rpg_e2vid clone + weights')
    p.add_argument('--start_s',  default=5.0,    type=float,
                   help='(h5 only) Skip events before this offset in seconds (default: 5.0)')
    p.add_argument('--duration', default=None,   type=float,
                   help='(h5 only) Duration in seconds (default: full sequence)')
    p.add_argument('--events_per_pixel', default=0.01, type=float,
                   help='Event window size as fraction of pixels (default: 0.01)')
    p.add_argument('--max_events', default=None, type=int,
                   help='Limit to first N events — use for smoke tests '
                        '(e.g. --max_events 100000 → ~10 frames)')
    p.add_argument('--width',    default=1280,   type=int)
    p.add_argument('--height',   default=720,    type=int)
    p.add_argument('--compress_jpeg', action='store_true',
                   help='Convert PNG frames to JPEG after reconstruction (~5x smaller). '
                        'Use on Kaggle to avoid disk exhaustion.')
    return p.parse_args()


# ── HDF5 → zip ───────────────────────────────────────────────────────────────

def convert_h5_to_zip(h5_path: Path, zip_path: Path,
                      width: int, height: int,
                      start_s: float, duration_s,
                      chunk_size: int = 5_000_000):
    """Convert HDF5 event stream to rpg_e2vid zip format.

    Reads in chunks to avoid loading hundreds of millions of events into RAM at once.
    chunk_size controls how many events are processed per iteration (default 5M).
    """
    # ECF codec ships with OpenEB/Metavision SDK, not with hdf5plugin.
    # If OpenEB is installed, the filter plugin lives at the path below.
    plugin_path = '/usr/lib/x86_64-linux-gnu/hdf5/plugins'
    if Path(plugin_path).exists():
        os.environ.setdefault('HDF5_PLUGIN_PATH', plugin_path)

    import h5py
    import numpy as np

    print(f'Reading {h5_path} ...')
    try:
        _f = h5py.File(h5_path, 'r')
        _ = _f['CD/events'][0]['t']   # probe: triggers codec load
        _f.close()
    except OSError as _e:
        if 'find plugin' in str(_e) or 'filter' in str(_e).lower():
            sys.exit(
                f'\nERROR: HDF5 ECF codec plugin not found.\n'
                f'The FRED events.h5 files use Prophesee\'s ECF compression codec,\n'
                f'which is NOT part of hdf5plugin — it requires OpenEB.\n\n'
                f'Fix: install OpenEB / Metavision SDK on this machine:\n'
                f'  https://docs.prophesee.ai/stable/installation/linux.html\n\n'
                f'Workaround (no OpenEB needed): obtain the pre-converted events.zip\n'
                f'from a teammate and place it at {h5_path.parent}/events.zip.\n'
                f'prepare_sequences.sh skips h5 conversion when events.zip already exists.\n'
            )
        raise

    with h5py.File(h5_path, 'r') as f:
        ev      = f['CD/events']
        n_total = len(ev)
        t0_us   = int(ev[0]['t'])
        t0_s    = t0_us / 1e6
        t_start = t0_s + start_s
        t_end   = (t0_s + start_s + duration_s) if duration_s is not None else None

        # Find start index via binary search on timestamps
        lo, hi = 0, n_total
        while lo < hi:
            mid = (lo + hi) // 2
            if ev[mid]['t'] / 1e6 < t_start:
                lo = mid + 1
            else:
                hi = mid
        start_idx = lo

        header     = f'{width} {height}\n'
        n_written  = 0
        t_first = t_last = None

        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
            with zf.open('events.txt', 'w', force_zip64=True) as out:
                out.write(header.encode())
                idx = start_idx
                while idx < n_total:
                    chunk = ev[idx: idx + chunk_size]
                    t_s   = chunk['t'].astype(np.float64) / 1e6
                    if t_end is not None:
                        chunk = chunk[t_s <= t_end]
                        t_s   = t_s[t_s <= t_end]
                        if len(chunk) == 0:
                            break
                    x = chunk['x'].astype(np.int32)
                    y = chunk['y'].astype(np.int32)
                    p = (chunk['p'] > 0).astype(np.int32)
                    lines = '\n'.join(
                        f'{ti:.9f} {xi} {yi} {pi}'
                        for ti, xi, yi, pi in zip(t_s, x, y, p)
                    ) + '\n'
                    out.write(lines.encode())
                    n_written += len(chunk)
                    if t_first is None:
                        t_first = t_s[0]
                    t_last = t_s[-1]
                    idx += chunk_size
                    print(f'  {n_written:,} events written...', end='\r', flush=True)

    print(f'Events in window : {n_written:,}')
    if t_first is not None:
        print(f'Time range       : {t_first:.3f}s → {t_last:.3f}s  ({t_last-t_first:.1f}s)')
        print(f'Expected frames  : ~{n_written // int(width * height * 0.01)}')
    print(f'Written : {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)')


# ── rpg_e2vid setup + patches ─────────────────────────────────────────────────

def setup_e2vid(work_dir: Path) -> tuple[Path, Path]:
    """Clone rpg_e2vid if needed, apply all compatibility patches, return (e2vid_dir, weights_path)."""
    e2vid_dir    = work_dir / 'rpg_e2vid'
    weights_path = work_dir / 'E2VID.pth.tar'

    if not e2vid_dir.exists():
        print('Cloning rpg_e2vid ...')
        subprocess.run(
            ['git', 'clone', 'https://github.com/uzh-rpg/rpg_e2vid.git', str(e2vid_dir)],
            check=True
        )
    else:
        print('rpg_e2vid already cloned')

    if not weights_path.exists():
        print('Downloading pretrained weights ...')
        subprocess.run([
            'wget', '-q',
            'http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID.pth.tar',
            '-O', str(weights_path),
        ], check=True)
        print('Weights downloaded')
    else:
        print('Weights already present')

    _apply_patches(e2vid_dir)
    return e2vid_dir, weights_path


def _apply_patches(e2vid_dir: Path):
    """Apply all compatibility patches to rpg_e2vid for modern Python/NumPy/PyTorch/pandas."""

    # Patch 1 — torch._six.string_classes removed in PyTorch 1.10
    import torch
    if not hasattr(torch, '_six'):
        torch._six = types.ModuleType('torch._six')
    if not hasattr(torch._six, 'string_classes'):
        torch._six.string_classes = (str, bytes)

    # Patch 2 — np.int removed in NumPy 1.24
    for rel in ['run_reconstruction.py', 'utils/inference_utils.py']:
        _patch_file(e2vid_dir / rel,
                    lambda s: re.sub(r'np\.int([^0-9_e8])', r'int\1', s),
                    'np.int → int')

    # Patch 3 — pandas removed delim_whitespace in 2.0
    for rel in ['run_reconstruction.py', 'utils/event_readers.py']:
        _patch_file(e2vid_dir / rel,
                    lambda s: s.replace("delim_whitespace=True", "sep=r'\\s+'"),
                    'delim_whitespace → sep')

    # Patch 4 — torch.load needs map_location='cpu' for GPU-saved weights
    _patch_file(e2vid_dir / 'utils/loading_utils.py',
                lambda s: s.replace(
                    "torch.load(path_to_model)",
                    "torch.load(path_to_model, map_location='cpu')"),
                'map_location=cpu')

    # Patch 5 — auto-detect GPU instead of hardcoded True/False
    _patch_file(e2vid_dir / 'options/inference_options.py',
                lambda s: s.replace(
                    "parser.set_defaults(use_gpu=True)",
                    "import torch as _torch; parser.set_defaults(use_gpu=_torch.cuda.is_available())"),
                'use_gpu=auto')

    # Patch 6 — CudaTimer uses torch.cuda.Event at __init__, crashes on CPU-only builds
    _patch_cuda_timer(e2vid_dir / 'utils/timers.py')

    print('All patches applied.')


def _patch_file(path: Path, transform, label: str):
    if not path.exists():
        return
    original = path.read_text()
    patched  = transform(original)
    if patched != original:
        path.write_text(patched)
        print(f'  Patched ({label}): {path.name}')


def _patch_cuda_timer(timers_path: Path):
    if not timers_path.exists():
        return
    new_class = '''\
class CudaTimer:
    def __init__(self, timer_name=''):
        self.timer_name = timer_name
        self.enabled = torch.cuda.is_available()
        if not self.enabled:
            return
        if self.timer_name not in cuda_timers:
            cuda_timers[self.timer_name] = []
        self.start = torch.cuda.Event(enable_timing=True)
        self.end   = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.enabled:
            self.start.record()
        return self

    def __exit__(self, *args):
        if self.enabled:
            self.end.record()
            torch.cuda.synchronize()
            cuda_timers[self.timer_name].append(self.start.elapsed_time(self.end))
'''
    src     = timers_path.read_text()
    patched = re.sub(r'class CudaTimer:.*?(?=\nclass |\Z)',
                     new_class + '\n\n', src, flags=re.DOTALL)
    if patched != src:
        timers_path.write_text(patched)
        print(f'  Patched (CudaTimer): {timers_path.name}')


# ── Run reconstruction ────────────────────────────────────────────────────────

def _png_staging_dir(out_dir: Path, n_events: int, events_per_pixel: float,
                     width: int, height: int) -> Path:
    """Return /dev/shm staging dir if RAM disk has enough space, else fall back to out_dir.

    Keeping intermediate PNGs on a RAM disk avoids exhausting the 20 GB
    /kaggle/working quota with ~11 GB of temporary PNG files for large sequences.
    /dev/shm is a tmpfs mounted in RAM on all Linux hosts (Kaggle included).
    """
    devshm = Path('/dev/shm')
    if devshm.exists():
        estimated_frames = n_events // max(1, int(width * height * events_per_pixel))
        estimated_png_bytes = estimated_frames * width * height * 0.15  # ~0.15 bytes/pixel for grayscale PNG
        try:
            free = shutil.disk_usage(devshm).free
            if free > estimated_png_bytes * 1.2:
                staging = devshm / f'e2vid_{out_dir.parent.name}'
                print(f'PNG staging: {staging}  '
                      f'(RAM disk, {free / 1e9:.1f} GB free, '
                      f'~{estimated_png_bytes / 1e9:.1f} GB needed)')
                return staging
            else:
                print(f'RAM disk too small ({free / 1e9:.1f} GB free, '
                      f'~{estimated_png_bytes / 1e9:.1f} GB needed) — using out_dir')
        except Exception:
            pass
    return out_dir


def run_reconstruction(e2vid_dir: Path, weights_path: Path,
                       zip_path: Path, out_dir: Path,
                       events_per_pixel: float,
                       n_events: int, width: int, height: int) -> Path:
    """Run RPG E2VID and return the staging directory where PNGs were written."""
    import time

    staging = _png_staging_dir(out_dir, n_events, events_per_pixel, width, height)

    # Clear stale frames from any previous interrupted run
    recon_subdir = staging / 'reconstruction'
    if recon_subdir.exists():
        shutil.rmtree(recon_subdir)

    staging.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(e2vid_dir / 'run_reconstruction.py'),
        '-c', str(weights_path),
        '-i', str(zip_path),
        '--output_folder', str(staging),
        '--auto_hdr',
        '--num_events_per_pixel', str(events_per_pixel),
    ]
    print('\nRunning:', ' '.join(cmd), '\n')

    expected = n_events // max(1, int(width * height * events_per_pixel))
    process = subprocess.Popen(cmd)
    t0 = time.time()

    while process.poll() is None:
        time.sleep(30)
        n_done = len(list(recon_subdir.glob('frame_*.png'))) if recon_subdir.exists() else 0
        elapsed = time.time() - t0
        if n_done > 0:
            fps  = n_done / elapsed
            eta  = (expected - n_done) / fps
            print(f'  {n_done}/{expected} frames  '
                  f'{n_done/expected*100:.0f}%  '
                  f'{fps:.2f} fps  ETA {eta/60:.1f} min', flush=True)
        else:
            print(f'  Waiting for first frame... ({elapsed:.0f}s elapsed)', flush=True)

    if process.returncode != 0:
        sys.exit(f'e2vid failed with return code {process.returncode}')

    return staging


# ── Post-process: rename frames + keep timestamps.txt ─────────────────────────

def postprocess(out_dir: Path, staging: Path, compress_jpeg: bool = False, quality: int = 90):
    """Rename frames from staging/reconstruction/ into out_dir.

    If staging != out_dir (i.e. PNG staging was on /dev/shm), files are moved
    cross-filesystem.  When compress_jpeg=True, each PNG is converted to JPEG
    immediately after being read, so at most one PNG exists on-disk at a time.
    The staging directory is removed after all frames are processed.
    """
    from PIL import Image

    recon_subdir = staging / 'reconstruction'
    frames = sorted(recon_subdir.glob('frame_*.png'))
    if not frames:
        sys.exit(f'No frames found in {recon_subdir}')

    before_mb = sum(p.stat().st_size for p in frames) / 1e6
    after_mb  = 0.0

    for i, fp in enumerate(frames):
        if compress_jpeg:
            img = Image.open(fp)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            dst = out_dir / f'frame_{i:06d}.jpg'
            img.save(str(dst), 'JPEG', quality=quality)
            after_mb += dst.stat().st_size / 1e6
            fp.unlink()
        else:
            dst = out_dir / f'frame_{i:06d}.png'
            if staging == out_dir:
                fp.rename(dst)
            else:
                shutil.move(str(fp), str(dst))

    ts_src = recon_subdir / 'timestamps.txt'
    if ts_src.exists():
        shutil.copy(ts_src, out_dir / 'timestamps.txt')
        print(f'timestamps.txt preserved in {out_dir}')

    if staging != out_dir and staging.exists():
        shutil.rmtree(staging)
        print(f'Staging dir removed: {staging}')

    if compress_jpeg:
        print(f'Compressed {len(frames)} frames: {before_mb:.0f} MB PNG → {after_mb:.0f} MB JPEG  (q={quality})')

    exts   = ('*.jpg',) if compress_jpeg else ('*.png',)
    final  = [f for e in exts for f in sorted(out_dir.glob(e))]
    print(f'Frames: {len(final)}  ({final[0].name} → {final[-1].name})')
    print(f'Output: {out_dir}')


# ── Smoke trimming ────────────────────────────────────────────────────────────

def trim_events(src: Path, max_events: int, work_dir: Path) -> Path:
    """Return a trimmed copy (zip or txt) containing only the first max_events events."""
    if src.suffix == '.zip':
        dst = work_dir / f'{src.stem}_smoke{max_events}.zip'
        with zipfile.ZipFile(src) as zin, zin.open('events.txt') as f:
            header = f.readline()
            lines  = [f.readline() for _ in range(max_events)]
        with zipfile.ZipFile(dst, 'w', compression=zipfile.ZIP_DEFLATED) as zout:
            zout.writestr('events.txt', header.decode() + b''.join(lines).decode())
    else:
        dst = work_dir / f'{src.stem}_smoke{max_events}.txt'
        with open(src, 'rb') as f:
            header = f.readline()
            lines  = [f.readline() for _ in range(max_events)]
        dst.write_bytes(header + b''.join(lines))
    print(f'Smoke events: {max_events:,} → {dst}  ({dst.stat().st_size/1e6:.1f} MB)')
    return dst


# ── KPI logging ───────────────────────────────────────────────────────────────

def count_events(path: Path) -> int:
    """Count event lines in a zip or plain txt (excludes the header line)."""
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z, z.open('events.txt') as f:
            return sum(1 for _ in f) - 1
    else:
        with open(path, 'rb') as f:
            return sum(1 for _ in f) - 1


def device_info() -> dict:
    import platform
    info = {'cpu': platform.processor() or platform.machine()}
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            info['gpu_memory_total_mb'] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e6)
    except Exception:
        pass
    return info


def peak_ram_mb() -> float:
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:
        return -1


def save_kpis(path: Path, kpis: dict):
    import json, datetime
    kpis['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kpis, indent=2))
    print(f'\nKPIs written → {path}')
    for k, v in kpis.items():
        print(f'  {k}: {v}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import time
    args = parse_args()

    work_dir = args.work_dir or Path(__file__).parent.parent / 'notebooks'
    work_dir.mkdir(parents=True, exist_ok=True)

    if args.zip_path:
        events_path = args.zip_path
        if not events_path.exists():
            # Kaggle auto-extracts zip files: events.zip → events/events.txt
            # Pass the txt directly — rpg_e2vid's event reader handles both formats.
            txt_candidate = events_path.parent / events_path.stem / 'events.txt'
            if txt_candidate.exists():
                print(f'=== Kaggle layout: using {txt_candidate} directly ===')
                events_path = txt_candidate
            else:
                raise FileNotFoundError(f'Neither {events_path} nor {txt_candidate} found')
        else:
            print(f'=== Using pre-built zip: {events_path} ===')
    else:
        events_path = work_dir / f'{args.h5_path.parent.name}_events.zip'
        print('=== Step 1: Convert HDF5 → event zip ===')
        convert_h5_to_zip(args.h5_path, events_path,
                          args.width, args.height,
                          args.start_s, args.duration)

    if args.max_events:
        print(f'\n=== Smoke test: trimming to {args.max_events:,} events ===')
        events_path = trim_events(events_path, args.max_events, work_dir)

    n_events = count_events(events_path)
    print(f'Events: {n_events:,}')

    print('\n=== Setup rpg_e2vid ===')
    e2vid_dir, weights_path = setup_e2vid(work_dir)

    print('\n=== Run e2vid reconstruction ===')
    t0 = time.time()
    staging = run_reconstruction(e2vid_dir, weights_path, events_path,
                                 args.out_dir, args.events_per_pixel,
                                 n_events, args.width, args.height)
    runtime = time.time() - t0

    print('\n=== Rename frames (compress_jpeg={}) ==='.format(args.compress_jpeg))
    postprocess(args.out_dir, staging, compress_jpeg=args.compress_jpeg)

    n_frames = len(sorted(args.out_dir.glob('frame_*.png'))) or \
               len(sorted(args.out_dir.glob('frame_*.jpg')))

    seq_id = (args.zip_path or args.h5_path).parent.name
    kpis = {
        'stage':                  'reconstruction',
        'sequence':               seq_id,
        **device_info(),
        'n_events':               n_events,
        'n_frames':               n_frames,
        'events_per_frame':       int(args.width * args.height * args.events_per_pixel),
        'input_size_mb':          round(events_path.stat().st_size / 1e6, 1),
        'runtime_s':              round(runtime, 1),
        'seconds_per_frame':      round(runtime / n_frames, 2) if n_frames else -1,
        'events_per_second':      round(n_events / runtime) if runtime else -1,
        'frames_per_second':      round(n_frames / runtime, 2) if runtime else -1,
        'peak_ram_mb':            peak_ram_mb(),
    }
    save_kpis(args.out_dir.parent / 'kpis' / f'reconstruct_{seq_id}.json', kpis)

    print('\nDone.')


if __name__ == '__main__':
    main()

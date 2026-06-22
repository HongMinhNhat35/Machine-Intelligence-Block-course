"""
reconstruct.py — Convert FRED events → HyperE2VID grayscale frames.

Streams events directly through the model with zero intermediate disk use.
No memmap, no extracted txt copy needed — events are read line-by-line and
voxel grids are built in RAM one window at a time, then immediately fed to
the model and saved as JPEG.

Disk budget on Kaggle (20 GB /kaggle/working):
  - EVREAL + HyperE2VID repos  : ~200 MB
  - Output JPEGs (11 k frames) : ~1–2 GB
  - Everything else             : negligible

Accepted input formats
-----------------------
  --zip_path   Pre-built event zip  (Kaggle auto-extracts → <stem>/events.txt)
  --h5_path    OpenEB HDF5 file     (local, requires ECF codec plugin)

Usage (Kaggle):
    python scripts/reconstruct.py \\
        --zip_path    /kaggle/input/.../events.zip \\
        --out_dir     /kaggle/working/data/processed/sequence_84/reconstruction_e2vid \\
        --work_dir    /kaggle/working/work \\
        --weights_path /kaggle/input/datasets/kevinhong54385/hypere2vidmodel/model.pth \\
        --compress_jpeg
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='FRED HyperE2VID reconstruction pipeline')

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--zip_path', type=Path,
                     help='Pre-built event zip (Kaggle layout: events.txt inside zip)')
    src.add_argument('--h5_path',  type=Path,
                     help='OpenEB events.h5 (requires ECF codec, use locally)')

    p.add_argument('--out_dir',       required=True, type=Path)
    p.add_argument('--work_dir',      default=None,  type=Path,
                   help='Working directory for repo clones (default: <script>/../notebooks)')
    p.add_argument('--weights_path',  default=None,  type=Path,
                   help='Pre-downloaded HyperE2VID .pth file (skips Google Drive download)')
    p.add_argument('--start_s',       default=5.0,   type=float,
                   help='(h5 only) Skip events before this many seconds (default: 5.0)')
    p.add_argument('--duration',      default=None,  type=float,
                   help='(h5 only) Duration in seconds (default: full sequence)')
    p.add_argument('--num_bins',      default=5,     type=int,
                   help='Temporal bins for voxel grid — must match model training (default: 5)')
    p.add_argument('--events_per_pixel', default=0.01, type=float,
                   help='Events per pixel per voxel window (default: 0.01)')
    p.add_argument('--max_events',    default=None,  type=int,
                   help='Smoke test: stop after N events (e.g. 100000 → ~10 frames)')
    p.add_argument('--width',         default=1280,  type=int)
    p.add_argument('--height',        default=720,   type=int)
    p.add_argument('--compress_jpeg', action='store_true',
                   help='Save frames as JPEG instead of PNG (~5x smaller, use on Kaggle)')
    return p.parse_args()


# ── HDF5 → generator ─────────────────────────────────────────────────────────

def h5_event_generator(h5_path: Path, start_s: float, duration_s,
                        chunk_size: int = 5_000_000):
    """Yield (t_s, x, y, p) numpy arrays one chunk at a time from an HDF5 file."""
    plugin_path = '/usr/lib/x86_64-linux-gnu/hdf5/plugins'
    if Path(plugin_path).exists():
        os.environ.setdefault('HDF5_PLUGIN_PATH', plugin_path)
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass

    import h5py
    import numpy as np

    with h5py.File(h5_path, 'r') as f:
        ev      = f['CD/events']
        n_total = len(ev)
        t0_s    = int(ev[0]['t']) / 1e6
        t_start = t0_s + start_s
        t_end   = (t0_s + start_s + duration_s) if duration_s is not None else None

        lo, hi = 0, n_total
        while lo < hi:
            mid = (lo + hi) // 2
            if ev[mid]['t'] / 1e6 < t_start:
                lo = mid + 1
            else:
                hi = mid

        idx = lo
        while idx < n_total:
            chunk = ev[idx: idx + chunk_size]
            t_s   = chunk['t'].astype(np.float64) / 1e6
            if t_end is not None:
                mask  = t_s <= t_end
                chunk = chunk[mask]
                t_s   = t_s[mask]
                if len(chunk) == 0:
                    break
            yield (t_s,
                   chunk['x'].astype(np.int32),
                   chunk['y'].astype(np.int32),
                   (chunk['p'] > 0).astype(np.float32) * 2 - 1)
            idx += chunk_size


# ── Resolve events source → open file handle ─────────────────────────────────

def resolve_events_txt(zip_path: Path) -> Path:
    """Return the path to events.txt, handling both zip-present and auto-extracted layouts."""
    if zip_path.exists():
        # Peek inside — if it's truly a zip, Kaggle did NOT auto-extract it.
        # In that case we need to extract events.txt to a temp location.
        tmp = zip_path.parent / '_evtmp_events.txt'
        if not tmp.exists():
            print(f'Extracting events.txt from {zip_path} ...')
            with zipfile.ZipFile(zip_path, 'r') as z:
                with z.open('events.txt') as src, open(tmp, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
            print(f'Extracted ({tmp.stat().st_size / 1e6:.0f} MB)')
        return tmp

    # Kaggle auto-extracted: <zip_stem>/events.txt
    candidate = zip_path.parent / zip_path.stem / 'events.txt'
    if candidate.exists():
        print(f'Kaggle auto-extracted layout: {candidate}')
        return candidate

    raise FileNotFoundError(
        f'Neither {zip_path} nor {candidate} found. '
        f'Make sure the dataset is attached to the notebook.'
    )


# ── Streaming voxel-grid builder ─────────────────────────────────────────────

class StreamingVoxelBuilder:
    """
    Reads events.txt line-by-line (or accepts (t,x,y,p) numpy chunks) and
    yields one voxel grid tensor per window without ever holding more than
    one window's worth of events in memory.

    Voxel grid: bilinear temporal binning, shape (num_bins, H, W), float32.
    """

    def __init__(self, txt_path: Path, width: int, height: int,
                 num_bins: int, events_per_pixel: float,
                 max_events: int | None = None):
        self.txt_path       = txt_path
        self.width          = width
        self.height         = height
        self.num_bins       = num_bins
        self.N              = int(width * height * events_per_pixel)
        self.max_events     = max_events

        import numpy as np
        self._np = np

        print(f'Voxel window     : {self.N:,} events/frame  '
              f'({num_bins} bins, {width}×{height})')

    def _build_voxel(self, buf_t, buf_x, buf_y, buf_p):
        np = self._np
        N  = len(buf_t)
        t_norm = (buf_t - buf_t[0]) / (buf_t[-1] - buf_t[0] + 1e-9) * (self.num_bins - 1)
        voxel  = np.zeros((self.num_bins, self.height, self.width), dtype=np.float32)
        for b in range(self.num_bins):
            w = np.maximum(0.0, 1.0 - np.abs(t_norm - b))
            np.add.at(voxel[b], (buf_y, buf_x), buf_p * w)
        return voxel, float(buf_t[0])

    def __iter__(self):
        np = self._np
        N  = self.N

        buf_t = np.empty(N, dtype=np.float64)
        buf_x = np.empty(N, dtype=np.int32)
        buf_y = np.empty(N, dtype=np.int32)
        buf_p = np.empty(N, dtype=np.float32)
        fill  = 0
        total = 0

        with open(self.txt_path) as f:
            f.readline()  # skip header line  "<width> <height>"
            for line in f:
                if self.max_events is not None and total >= self.max_events:
                    break
                parts = line.split()
                if len(parts) < 4:
                    continue
                idx = fill % N
                buf_t[idx] = float(parts[0])
                buf_x[idx] = int(parts[1])
                buf_y[idx] = int(parts[2])
                buf_p[idx] = 1.0 if int(parts[3]) > 0 else -1.0
                fill  += 1
                total += 1

                if fill == N:
                    yield self._build_voxel(buf_t, buf_x, buf_y, buf_p)
                    fill = 0

    def count_windows(self) -> int:
        """Fast line count to know how many frames to expect."""
        print('Counting events (fast pass)...')
        with open(self.txt_path) as f:
            f.readline()
            n = sum(1 for _ in f)
        if self.max_events is not None:
            n = min(n, self.max_events)
        w = n // self.N
        print(f'Events: {n:,}  →  {w:,} frames')
        return n, w


# ── HyperE2VID / EVREAL setup ─────────────────────────────────────────────────

def setup_hypere2vid(work_dir: Path,
                     provided_weights: Path | None = None) -> tuple[Path, Path]:
    """Clone EVREAL + HyperE2VID, install deps, return (evreal_dir, weights_path)."""
    evreal_dir     = work_dir / 'EVREAL'
    hypere2vid_dir = work_dir / 'HyperE2VID'
    weights_path   = work_dir / 'HyperE2VID.pth'

    if provided_weights is not None:
        if not provided_weights.exists():
            sys.exit(f'ERROR: --weights_path {provided_weights} does not exist.')
        weights_path = provided_weights
        print(f'Using provided weights: {weights_path}')

    if not evreal_dir.exists():
        print('Cloning EVREAL ...')
        subprocess.run(
            ['git', 'clone', '--depth', '1',
             'https://github.com/ercanburak/EVREAL.git', str(evreal_dir)],
            check=True)
    else:
        print('EVREAL already cloned')

    if not hypere2vid_dir.exists():
        print('Cloning HyperE2VID ...')
        subprocess.run(
            ['git', 'clone', '--depth', '1',
             'https://github.com/ercanburak/HyperE2VID.git', str(hypere2vid_dir)],
            check=True)
    else:
        print('HyperE2VID already cloned')

    # Copy HyperE2VID model package into EVREAL so its imports resolve
    model_dst = evreal_dir / 'model'
    if not model_dst.exists():
        shutil.copytree(str(hypere2vid_dir / 'model'), str(model_dst))
        print('HyperE2VID model package copied into EVREAL')

    req = evreal_dir / 'requirements.txt'
    if req.exists():
        print('Installing EVREAL requirements ...')
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-q', '-r', str(req)],
            check=True)

    # Download weights only if not provided and not already present
    if provided_weights is None and not weights_path.exists():
        GDRIVE_FILE_ID = '1_kiTCoyZM2LjgRFjECCnGMnWGSoFJkGe'
        print('Downloading HyperE2VID pretrained weights ...')
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
            subprocess.run(
                [sys.executable, '-m', 'gdown',
                 f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}',
                 '-O', str(weights_path)],
                check=True)
        except subprocess.CalledProcessError:
            sys.exit(
                'ERROR: Could not download weights.\n'
                'Please supply --weights_path or upload the .pth to a Kaggle dataset.\n'
                'Download from: https://drive.google.com/drive/folders/'
                '1UuGnKwSz5C9di-cVH1QzSFjgTRNqpYep'
            )
    elif provided_weights is None:
        print(f'Weights already present: {weights_path}')

    return evreal_dir, weights_path


# ── Load HyperE2VID model ─────────────────────────────────────────────────────

def load_model(evreal_dir: Path, weights_path: Path, num_bins: int, device):
    """
    Build and load the HyperE2VID model from an EVREAL checkpoint.

    The checkpoint stores a ConfigParser object under 'config' that knows the
    architecture type and kwargs.  We use config.init_obj('arch', model_module)
    to build the *inner* recurrent model (e.g. E2VIDRecurrent), then wrap it
    in ColorNet exactly as EVREAL's trainer does.
    """
    evreal_str = str(evreal_dir)
    if evreal_str not in sys.path:
        sys.path.insert(0, evreal_str)

    import torch
    import model as model_module          # EVREAL/model/__init__.py
    from model.model import ColorNet

    print('Loading checkpoint ...')
    # weights_only=False: checkpoint contains a ConfigParser object, not just tensors
    checkpoint = torch.load(str(weights_path), map_location='cpu', weights_only=False)

    config = checkpoint['config']         # ConfigParser instance
    # config.init_obj('arch', module) reads config['arch']['type'] and
    # config['arch']['args'] then calls module.<type>(**args)
    inner_model = config.init_obj('arch', model_module)

    # ColorNet wraps the inner recurrent model to handle RGBW demosaicing
    model = ColorNet(inner_model)

    state = checkpoint['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    print(f'Model loaded → {device}  (inner: {type(inner_model).__name__})')
    return model


# ── Streaming inference ───────────────────────────────────────────────────────

def run_streaming_inference(model, voxel_iter, n_events: int, n_windows: int,
                             out_dir: Path, width: int, height: int,
                             compress_jpeg: bool):
    """
    Iterate voxel windows one at a time:
      build voxel → GPU → model → CPU → save JPEG/PNG
    Peak RAM: one voxel grid (~18 MB for 5×720×1280 float32) + model weights.
    Peak disk: only the output frames.
    """
    import time
    import numpy as np
    import torch
    from PIL import Image

    device = next(model.parameters()).device
    ext    = '.jpg' if compress_jpeg else '.png'
    out_dir.mkdir(parents=True, exist_ok=True)

    # ColorNet handles padding internally via CropParameters — no manual padding needed.
    prev_states = None
    prev_image  = None
    ts_lines    = []
    frame_idx   = 0
    t0          = time.time()

    with torch.no_grad():
        for voxel_np, ts in voxel_iter:
            # ColorNet.forward takes a raw (1, num_bins, H, W) tensor.
            # It handles padding internally via CropParameters, so we do NOT
            # pad here.  It returns {'image': tensor} where image is (3, H, W)
            # float32 in [0,1] — BGR channel order (grayscale copied to all 3).
            voxel = torch.from_numpy(voxel_np).float().unsqueeze(0).to(device)

            out     = model.forward(voxel)          # ColorNet.forward(tensor)
            pred    = out['image']                   # (3, H, W) float32 in [0,1]

            # Convert to uint8 grayscale (all channels are identical for grayscale mode)
            pred_np = pred[0].cpu().numpy()          # take first channel (R≡G≡B here)
            pred_u8 = np.clip(pred_np * 255, 0, 255).astype(np.uint8)

            frame_path = out_dir / f'frame_{frame_idx:06d}{ext}'
            img = Image.fromarray(pred_u8, mode='L')
            if compress_jpeg:
                img.save(str(frame_path), 'JPEG', quality=90)
            else:
                img.save(str(frame_path))

            ts_lines.append(f'{ts:.9f}')
            frame_idx += 1

            if frame_idx % 100 == 0 or frame_idx == n_windows:
                elapsed = time.time() - t0
                fps     = frame_idx / elapsed
                eta     = (n_windows - frame_idx) / fps if fps > 0 else 0
                pct     = frame_idx / n_windows * 100 if n_windows else 0
                print(f'  {frame_idx}/{n_windows}  {pct:.0f}%  '
                      f'{fps:.1f} fps  ETA {eta/60:.1f} min', flush=True)

    (out_dir / 'timestamps.txt').write_text('\n'.join(ts_lines) + '\n')

    runtime = time.time() - t0
    print(f'\nInference done: {frame_idx} frames in {runtime:.0f}s  '
          f'({frame_idx/runtime:.1f} fps)')
    return frame_idx, runtime


# ── HDF5 → streaming inference ────────────────────────────────────────────────

def h5_streaming_iter(h5_path, start_s, duration_s, width, height,
                       num_bins, events_per_pixel, max_events):
    """
    Generator that yields (voxel_np, timestamp) directly from HDF5,
    without writing events.txt at all.
    """
    import numpy as np

    N = int(width * height * events_per_pixel)

    buf_t = np.empty(N, dtype=np.float64)
    buf_x = np.empty(N, dtype=np.int32)
    buf_y = np.empty(N, dtype=np.int32)
    buf_p = np.empty(N, dtype=np.float32)
    fill  = 0
    total = 0

    for t_s, x, y, p in h5_event_generator(h5_path, start_s, duration_s):
        for i in range(len(t_s)):
            if max_events is not None and total >= max_events:
                return
            idx          = fill % N
            buf_t[idx]   = t_s[i]
            buf_x[idx]   = x[i]
            buf_y[idx]   = y[i]
            buf_p[idx]   = p[i]
            fill  += 1
            total += 1
            if fill == N:
                yield _build_voxel_np(buf_t, buf_x, buf_y, buf_p, num_bins, height, width)
                fill = 0


def _build_voxel_np(buf_t, buf_x, buf_y, buf_p, num_bins, height, width):
    import numpy as np
    t_norm = (buf_t - buf_t[0]) / (buf_t[-1] - buf_t[0] + 1e-9) * (num_bins - 1)
    voxel  = np.zeros((num_bins, height, width), dtype=np.float32)
    for b in range(num_bins):
        w = np.maximum(0.0, 1.0 - np.abs(t_norm - b))
        np.add.at(voxel[b], (buf_y, buf_x), buf_p * w)
    return voxel, float(buf_t[0])


# ── KPI utilities ─────────────────────────────────────────────────────────────

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
    import datetime
    kpis['timestamp'] = datetime.datetime.now().isoformat(timespec='seconds')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kpis, indent=2))
    print(f'\nKPIs → {path}')
    for k, v in kpis.items():
        print(f'  {k}: {v}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import time
    import torch

    args     = parse_args()
    work_dir = args.work_dir or Path(__file__).parent.parent / 'notebooks'
    work_dir.mkdir(parents=True, exist_ok=True)
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_name = (args.zip_path or args.h5_path).parent.name

    # ── Setup model (clone repos, load weights) ───────────────────────────────
    print('\n=== Step 1: Setup HyperE2VID + EVREAL ===')
    evreal_dir, weights_path = setup_hypere2vid(work_dir, args.weights_path)
    model = load_model(evreal_dir, weights_path, args.num_bins, device)

    # ── Build event source iterator ───────────────────────────────────────────
    if args.zip_path:
        events_txt = resolve_events_txt(args.zip_path)

        builder  = StreamingVoxelBuilder(
            events_txt, args.width, args.height,
            args.num_bins, args.events_per_pixel, args.max_events,
        )
        n_events, n_windows = builder.count_windows()
        voxel_iter = builder
    else:
        # HDF5: stream directly, no txt needed
        import numpy as np
        print(f'Reading {args.h5_path} ...')
        voxel_iter = h5_streaming_iter(
            args.h5_path, args.start_s, args.duration,
            args.width, args.height,
            args.num_bins, args.events_per_pixel, args.max_events,
        )
        # Approximate window count for progress display
        n_events  = -1
        n_windows = -1

    # ── Stream inference → frames ─────────────────────────────────────────────
    print(f'\n=== Step 2: Streaming inference → {args.out_dir} ===')
    t0 = time.time()
    n_frames, runtime = run_streaming_inference(
        model, voxel_iter, n_events, n_windows,
        args.out_dir, args.width, args.height,
        args.compress_jpeg,
    )

    # ── Verify output ─────────────────────────────────────────────────────────
    ext    = '.jpg' if args.compress_jpeg else '.png'
    frames = sorted(args.out_dir.glob(f'frame_*{ext}'))
    if not frames:
        sys.exit(f'ERROR: No frames found in {args.out_dir}')
    print(f'Frames: {len(frames)}  ({frames[0].name} → {frames[-1].name})')

    # ── KPIs ──────────────────────────────────────────────────────────────────
    kpis = {
        'stage':              'reconstruction',
        'model':              'HyperE2VID',
        'sequence':           seq_name,
        **device_info(),
        'n_events':           n_events,
        'n_frames':           n_frames,
        'num_bins':           args.num_bins,
        'events_per_frame':   int(args.width * args.height * args.events_per_pixel),
        'runtime_s':          round(runtime, 1),
        'seconds_per_frame':  round(runtime / n_frames, 2) if n_frames else -1,
        'events_per_second':  round(n_events / runtime) if n_events > 0 and runtime else -1,
        'frames_per_second':  round(n_frames / runtime, 2) if runtime else -1,
        'peak_ram_mb':        peak_ram_mb(),
    }
    save_kpis(args.out_dir.parent / 'kpis' / f'reconstruct_{seq_name}.json', kpis)

    print('\nDone.')


if __name__ == '__main__':
    main()
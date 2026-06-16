"""
train_yolo.py — Build YOLO dataset from e2vid frames + FRED annotations, then train.

Usage (frame-level split, smoke test):
    python scripts/train_yolo.py \
        --sequences  sequence_0 \
        --raw_root   data/raw \
        --recon_root data/processed \
        --out_dir    data/yolo_e2vid \
        --weights    data/yolo_e2vid.pt \
        --epochs     3

Usage (sequence-level split, full training on Kaggle — Run 5):
    python scripts/train_yolo.py \
        --sequences     sequence_44 sequence_45 sequence_46 sequence_47 sequence_146 \
        --val_sequences sequence_146 \
        --raw_root      data/raw \
        --recon_root    data/processed \
        --out_dir       data/yolo_e2vid \
        --weights       data/yolo_e2vid.pt \
        --model         yolov8s.pt \
        --epochs        100 \
        --batch         16

Usage (resume after Colab disconnect):
    python scripts/train_yolo.py \
        --sequences     sequence_0 sequence_1 sequence_2 sequence_3 sequence_8 \
        --val_sequences sequence_8 \
        --raw_root      data/raw \
        --recon_root    data/processed \
        --out_dir       data/yolo_e2vid \
        --weights       data/yolo_e2vid.pt \
        --epochs        100 \
        --batch         16 \
        --resume
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='YOLO training on FRED e2vid frames')
    p.add_argument('--sequences',   nargs='+', required=True,
                   help='Sequence IDs to include, e.g. sequence_0 sequence_1')
    p.add_argument('--raw_root',    required=True, type=Path,
                   help='Root of raw FRED data (contains sequence_N/coordinates.txt)')
    p.add_argument('--recon_root',  required=True, type=Path,
                   help='Root of processed data (contains sequence_N/reconstruction_e2vid/)')
    p.add_argument('--out_dir',     required=True, type=Path,
                   help='Where to write the YOLO dataset')
    p.add_argument('--weights',     required=True, type=Path,
                   help='Output path for best.pt (e.g. data/yolo_e2vid.pt)')
    p.add_argument('--runs_dir',    default=None,  type=Path,
                   help='Directory for training run logs (default: out_dir/../yolo_runs)')
    p.add_argument('--model',       default='yolov8s.pt',
                   help='Base YOLO model (default: yolov8s.pt)')
    p.add_argument('--epochs',      default=100,   type=int)
    p.add_argument('--batch',       default=8,     type=int)
    p.add_argument('--imgsz',       default=640,   type=int)
    p.add_argument('--width',       default=1280,  type=int)
    p.add_argument('--height',      default=720,   type=int)
    p.add_argument('--match_threshold', default=0.5, type=float,
                   help='Max gap (s) between e2vid frame and nearest annotation (default: 0.5)')
    p.add_argument('--val_sequences', nargs='*',  default=None,
                   help='Sequences to use entirely as val (sequence-level split). '
                        'If omitted, a random frame-level split is used instead.')
    p.add_argument('--split',       default=0.8,   type=float,
                   help='Train fraction for frame-level split (default: 0.8, ignored if --val_sequences set)')
    p.add_argument('--seed',        default=42,    type=int)
    p.add_argument('--resume',      action='store_true',
                   help='Resume training from last.pt in the runs directory')
    return p.parse_args()


# ── Annotations ───────────────────────────────────────────────────────────────

def load_annotations(seq_dir: Path):
    """Parse coordinates.txt → (timestamps_s, boxes_pixels arrays)."""
    ann_file = seq_dir / 'coordinates.txt'
    ts_list, box_list = [], []
    for line in ann_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        ts_str, box_str = line.split(':')
        ts_list.append(float(ts_str.strip()))
        box_list.append([float(v) for v in box_str.split(',')[:4]])
    ts  = np.array(ts_list)
    box = np.array(box_list)
    print(f'  {ann_file.parent.name}: {len(ts)} annotations  '
          f't={ts[0]:.2f}s..{ts[-1]:.2f}s')
    return ts, box


# ── Frame → annotation matching ───────────────────────────────────────────────

def match_frames(seq_id: str, recon_root: Path,
                 ann_t: np.ndarray, ann_boxes: np.ndarray,
                 match_threshold: float, W: int, H: int):
    """
    Returns list of (image_path, yolo_label_str_or_None).
    yolo_label_str: '0 xc yc w h' normalised; None = background frame.
    """
    recon_dir = recon_root / seq_id / 'reconstruction_e2vid'
    ts_file   = recon_dir / 'timestamps.txt'

    if not ts_file.exists():
        print(f'  ⚠  {seq_id}: timestamps.txt not found — skipping')
        return []

    frame_t = np.array([float(l) for l in ts_file.read_text().splitlines() if l.strip()])
    frames  = sorted(recon_dir.glob('frame_*.png')) or sorted(recon_dir.glob('frame_*.jpg'))

    if len(frame_t) != len(frames):
        n = min(len(frame_t), len(frames))
        print(f'  ⚠  {seq_id}: timestamp/frame count mismatch — using first {n}')
        frame_t, frames = frame_t[:n], frames[:n]

    matched, results = 0, []
    for fp, ft in zip(frames, frame_t):
        j    = np.argmin(np.abs(ann_t - ft))
        dist = np.abs(ann_t[j] - ft)
        if dist <= match_threshold:
            x1, y1, x2, y2 = ann_boxes[j]
            xc = (x1 + x2) / 2 / W
            yc = (y1 + y2) / 2 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            results.append((fp, f'0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}'))
            matched += 1
        else:
            results.append((fp, None))

    print(f'  {seq_id}: {len(results)} frames  '
          f'matched={matched}  background={len(results)-matched}')
    return results


# ── YOLO dataset ──────────────────────────────────────────────────────────────

def build_dataset(train_matches, val_matches, out_dir: Path):
    """Write YOLO dataset: labels placed alongside frames, images referenced via txt manifests.

    No image copies or symlinks — avoids disk exhaustion on space-constrained environments.
    Ultralytics label discovery: when path has no /images/, it looks for .txt in the same dir.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    train_paths, val_paths = [], []
    for matches, path_list in [(train_matches, train_paths), (val_matches, val_paths)]:
        for fp, label in matches:
            fp.with_suffix('.txt').write_text(label if label else '')
            path_list.append(str(fp.resolve()))

    (out_dir / 'train.txt').write_text('\n'.join(train_paths))
    (out_dir / 'val.txt').write_text('\n'.join(val_paths))

    n_tr, n_va = len(train_matches), len(val_matches)
    print(f'Dataset: {n_tr} train / {n_va} val  →  {out_dir}')

    yaml = f"""path: {out_dir.resolve()}
train: train.txt
val:   val.txt

nc: 1
names:
  0: drone
"""
    (out_dir / 'dataset.yaml').write_text(yaml)
    print('dataset.yaml written')


# ── KPI logging ───────────────────────────────────────────────────────────────

def device_info() -> dict:
    import platform
    info = {'cpu': platform.processor() or platform.machine()}
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            info['gpu']                = f"{name} x{n}" if n > 1 else name
            info['n_gpus']             = n
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


# ── Training ──────────────────────────────────────────────────────────────────

def train(out_dir: Path, weights_out: Path, runs_dir: Path, args):
    import time, torch
    from ultralytics import YOLO

    # PyTorch 2.10 tightened DDP gradient reduction checks; ultralytics 8.4.x
    # (Python 3.12 build) doesn't set find_unused_parameters=True, causing a
    # crash when some detection head params don't receive gradients in a batch.
    # Patch the installed trainer.py before launching DDP.
    import importlib, re as _re
    _spec = importlib.util.find_spec('ultralytics')
    if _spec:
        _trainer = Path(_spec.submodule_search_locations[0]) / 'engine' / 'trainer.py'
        if _trainer.exists():
            _src = _trainer.read_text()
            if 'find_unused_parameters' not in _src:
                _src = _src.replace(
                    'static_graph=bool(self.args.compile),\n            )',
                    'static_graph=bool(self.args.compile),\n                find_unused_parameters=True,\n            )',
                )
                _trainer.write_text(_src)
                print('Patched ultralytics DDP: find_unused_parameters=True')

    if torch.cuda.device_count() > 1:
        device = ','.join(str(i) for i in range(torch.cuda.device_count()))
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'\nTraining on {device}  ({torch.cuda.device_count()} GPU(s))  epochs={args.epochs}  batch={args.batch}')

    train_txt = out_dir / 'train.txt'
    val_txt   = out_dir / 'val.txt'
    n_train = sum(1 for _ in train_txt.open()) if train_txt.exists() else 0
    n_val   = sum(1 for _ in val_txt.open())   if val_txt.exists()   else 0

    last_pt = runs_dir / 'e2vid' / 'weights' / 'last.pt'
    if args.resume:
        if not last_pt.exists():
            raise SystemExit(f'--resume requested but last.pt not found at {last_pt}')
        print(f'Resuming from {last_pt}')
        model = YOLO(str(last_pt))
    else:
        model = YOLO(args.model)

    t0    = time.time()
    results = model.train(
        data      = str(out_dir / 'dataset.yaml'),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = device,
        project   = str(runs_dir),
        name      = 'e2vid',
        exist_ok  = True,
        patience  = 20,
        optimizer = 'SGD',
        lr0       = 0.01,
        verbose   = True,
        resume    = args.resume,
        mosaic    = 1.0,   # tiles 4 images — helps small object detection
        mixup     = 0.0,   # disabled — blends images and can erase tiny drones
    )
    runtime = time.time() - t0

    # save_dir is deterministic: project / name (we pass both explicitly).
    # results is None in DDP multi-GPU mode — don't rely on it.
    save_dir = runs_dir / 'e2vid'
    best = save_dir / 'weights' / 'best.pt'
    if not best.exists():
        raise FileNotFoundError(f'best.pt not found at {best}')
    shutil.copy(best, weights_out)
    print(f'\nWeights saved → {weights_out}  ({weights_out.stat().st_size / 1e6:.1f} MB)')

    # Metrics: results is None under DDP; fall back to results.csv written by ultralytics.
    import pandas as pd
    metrics = {}
    best_metrics = {}
    csv_path = save_dir / 'results.csv'
    epochs_completed = args.epochs  # fallback

    if results is not None and hasattr(results, 'results_dict'):
        metrics = results.results_dict
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        epochs_completed = len(df)  # actual rows = epochs run (handles early stopping)
        if not metrics:
            metrics = dict(df.iloc[-1])  # last epoch for general metrics
        # best epoch by mAP50 for KPI reporting
        map_col = 'metrics/mAP50(B)'
        if map_col in df.columns:
            best_row = df.loc[df[map_col].idxmax()]
            best_metrics = dict(best_row)
    elif results is not None and hasattr(model, 'trainer') and model.trainer:
        epochs_completed = model.trainer.epoch + 1

    gpu_mem_mb = -1
    if torch.cuda.is_available():
        gpu_mem_mb = round(torch.cuda.max_memory_allocated() / 1e6)

    src = best_metrics if best_metrics else metrics
    dev = device_info()

    kpis = {
        'stage':               'training',
        'split_mode':          'sequence-level' if args.val_sequences else 'frame-level',
        'train_sequences':     [s for s in args.sequences if not args.val_sequences or s not in args.val_sequences],
        'val_sequences':       args.val_sequences or [],
        **dev,
        'gpu_memory_peak_mb':  gpu_mem_mb,
        'n_train_images':      n_train,
        'n_val_images':        n_val,
        'epochs_requested':    args.epochs,
        'epochs_completed':    epochs_completed,
        'best_epoch':          int(float(src.get('epoch', -1))) if src else -1,
        'batch_per_gpu':       args.batch,
        'effective_batch':     args.batch * dev.get('n_gpus', 1),
        'imgsz':               args.imgsz,
        'runtime_s':           round(runtime, 1),
        'runtime_per_epoch_s': round(runtime / max(epochs_completed, 1), 1),
        'map50':               round(float(src.get('metrics/mAP50(B)', -1)), 4),
        'map50_95':            round(float(src.get('metrics/mAP50-95(B)', -1)), 4),
        'precision':           round(float(src.get('metrics/precision(B)', -1)), 4),
        'recall':              round(float(src.get('metrics/recall(B)', -1)), 4),
        'peak_ram_mb':         peak_ram_mb(),
    }
    save_kpis(weights_out.parent / 'kpis' / 'train_yolo.json', kpis)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    runs_dir = args.runs_dir or args.out_dir.parent / 'yolo_runs'
    runs_dir.mkdir(parents=True, exist_ok=True)
    args.weights.parent.mkdir(parents=True, exist_ok=True)

    val_set = set(args.val_sequences) if args.val_sequences else set()

    if True:  # dataset always rebuilt — it lives on local SSD, not Drive
        print('=== Step 1: Load annotations ===')
        if args.val_sequences:
            print(f'Split mode : sequence-level')
            print(f'  Train    : {[s for s in args.sequences if s not in val_set]}')
            print(f'  Val      : {args.val_sequences}')
        else:
            print(f'Split mode : frame-level  (train fraction={args.split})')

        train_matches, val_matches = [], []
        for sid in args.sequences:
            ann_t, ann_boxes = load_annotations(args.raw_root / sid)
            matches = match_frames(sid, args.recon_root, ann_t, ann_boxes,
                                   args.match_threshold, args.width, args.height)
            if val_set:
                if sid in val_set:
                    val_matches.extend(matches)
                else:
                    train_matches.extend(matches)
            else:
                train_matches.extend(matches)

        if not val_set:
            # Frame-level random split
            random.seed(args.seed)
            all_matches = train_matches
            random.shuffle(all_matches)
            n_train = int(args.split * len(all_matches))
            train_matches = all_matches[:n_train]
            val_matches   = all_matches[n_train:]

        if not train_matches:
            raise SystemExit('No train frames found — check --recon_root and sequence IDs')
        if not val_matches:
            raise SystemExit('No val frames found — check --val_sequences or --split')
        print(f'Total: {len(train_matches)} train / {len(val_matches)} val frames')

        print('\n=== Step 2: Build YOLO dataset ===')
        build_dataset(train_matches, val_matches, args.out_dir)

    print('\n=== Step 3: Train YOLO ===')
    train(args.out_dir, args.weights, runs_dir, args)

    print('\nDone.')


if __name__ == '__main__':
    main()

# FRED Sequence Selection — Analysis and Recommendations

Principled dataset selection for training and validation, based on
annotation-level analysis of all 231 FRED sequences without downloading the
event data (~35 MB total vs ~250 GB for the full dataset).

**Script:** `scripts/analyze_fred_sequences.py`  
**Outputs:** `data/kpis/fred_sequence_analysis.csv`, `data/kpis/fred_analysis.png`

---

## Why random selection is not enough

The FRED dataset has 231 sequences (184 train, 47 test) that vary enormously:

| Property | Min | Max |
|---|---|---|
| Annotation count | 235 | 6,900 |
| Mean drone area (% of frame) | 0.076% | 1.425% |
| Drone size regime | 100% tiny | 24% tiny / 72% small |

A randomly chosen sequence can be annotation-poor, spatially homogeneous, or
contain an unusually large drone. All three hurt training and make validation
scores unreliable.

---

## How the script works

It downloads only `coordinates.txt` (~150 KB) from each HuggingFace zip using
HTTP Range requests — no event data is needed. For 231 sequences this takes
~10 minutes and ~35 MB.

```bash
/home/bani/.local/share/pipx/venvs/kaggle/bin/python3 \
    scripts/analyze_fred_sequences.py --plot

# Lower the event density threshold if you want to inspect sparse sequences too:
python scripts/analyze_fred_sequences.py --min-event-mb 200
```

---

## Parameters measured per sequence

### 1. Annotation count (`n_ann`)
Total bounding-box annotations after the 5-second noise-burst skip.
More annotations → more positive training examples for YOLO.
Range across FRED: 235 – 6,900.

### 2. Usable duration (`duration_s`)
Seconds of recording with drone present (first to last annotation).
Short sequences give the E2VID LSTM less warmup time and fewer frames.

### 3. Annotation density (`ann_per_s`)
Annotations per second. Consistent across most sequences (~28–30/s) because
FRED uses a fixed annotation rate. Outliers (< 20/s) tend to be slower-moving
drones — a different flight regime worth being aware of.

### 4. Mean drone area (`mean_area_pct`)
Average bounding-box area as a percentage of the 1280×720 frame.
- < 0.15% → tiny drone (hardest to detect)
- 0.15–0.40% → small-to-medium drone
- \> 0.40% → large / close drone

Training on large-drone sequences then validating on tiny-drone sequences
inflates the apparent performance gap. The current seq_201 (0.325%) → seq_127
(0.145%) mismatch is one reason for the low mAP50=0.29.

### 5. Size distribution (`tiny_pct`, `small_pct`, `medium_pct`)
Fraction of annotations in each size class:
- **Tiny** (< 0.003% of frame, < ~6×6 px at 1280×720): hardest class
- **Small** (0.003–0.01%): typical FRED drone at mid-range
- **Medium** (> 0.01%): close or large drone

A training set dominated by tiny annotations is harder but more representative
of real deployment. A mixed distribution (60–75% tiny, 25–40% small) gives
the model exposure to multiple scales.

### 6. Event data size (`event_data_mb`)  ← **added after Run 8**

Uncompressed size of `events.hdf5` inside the FRED sequence zip, read from the
ZIP Central Directory in the same HTTP request as `coordinates.txt` — no extra download.
Cached alongside coordinates.txt so re-runs are instant.

This is the **critical filter** the original analysis was missing. Sequences 44–47 scored
top-4 for training by annotation quality but yielded only 736 training frames at epp=0.1
because their event streams are 10–18× sparser than the Run 7 sequences:

| Sequence | events.hdf5 (est.) | Train frames at epp=0.1 |
|---|---|---|
| sequence_84  | ~2.2 GB | 1,539 |
| sequence_201 | ~2.0 GB | 1,268 |
| sequence_44  | ~0.12 GB | 193 |
| sequence_47  | ~0.10 GB | 160 |

**Hard filter:** sequences with `event_data_mb < 500` are excluded from the
recommended split. The threshold can be adjusted with `--min-event-mb`.

A minimum of ~500 MB (≈ 1,500+ frames at epp=0.1) ensures YOLO has enough examples
to converge. Run 7 sequences ranged from 1.2–6.0 GB.

### 7. Spatial spread (`cx_std + cy_std`)
Standard deviation of drone centroid position (x and y separately, normalised
to [0,1]). Low spread means the drone always appears in the same region of the
frame — the model can cheat by using position rather than appearance.

Ideal training sequences have spread > 0.35 (drone moves across most of the
frame). Ideal validation sequences can have lower spread as long as the drone
characteristics differ meaningfully from training.

---

## Scoring

### Training score (higher = better)
```
train_score = 0.30 × norm_ann + 0.30 × norm_spread + 0.15 × size_diversity + 0.25 × norm_events
```
- `norm_ann`    = min(n_ann / 3000, 1)
- `norm_spread` = min(spatial_spread / 0.4, 1)
- `size_diversity` = min((small_pct + medium_pct) / 30, 1) — rewards mixed sizes
- `norm_events` = min(event_data_mb / 500, 1) — **new**: penalises event-sparse sequences

Sequences below `--min-event-mb` (default 500 MB) are excluded from recommendations
entirely, regardless of score.

### Validation score (higher = better)
```
val_score = 0.4 × norm_ann + 0.3 × area_diff + 0.3 × hardness
```
- `area_diff` = how different the mean area is from the training mean (larger
  gap = more informative test of generalisation)
- `hardness` = reward for smaller (harder) drones in validation

---

## Results across all 231 sequences

Last run: 2026-06-30, all 231 sequences, `--min-event-mb 500`.

### Top training candidates (event-dense, ≥ 500 MB)

| Seq | N_ann | EvMB | Area% | Tiny% | Sm% | Spread | Score |
|---|---|---|---|---|---|---|---|
| sequence_201 | 2,846 | 2,114 | 0.325 | 55 | 45 | 0.338 | 0.938 |
| sequence_215 | 5,790 |   612 | 0.661 | 24 | 65 | 0.316 | 0.937 |
| sequence_212 | 6,769 |   553 | 0.405 | 28 | 72 | 0.311 | 0.933 |
| sequence_214 | 6,288 |   622 | 0.527 | 26 | 68 | 0.293 | 0.920 |
| sequence_213 | 6,900 |   571 | 0.497 | 22 | 74 | 0.264 | 0.898 |

Sequences 212–216 form a new cluster with the highest annotation counts in the
dataset (5,790–6,900 per sequence). All are event-dense (500–650 MB).

Run 7 sequences for comparison: sequence_84 now ranks 17th (score 0.832),
sequence_85 9th-best validation. The new top-4 offers ~2–3× more annotations
per sequence than the Run 7 training set.

### Excluded — event-sparse (< 500 MB)

| Seq | N_ann | EvMB | Score | Notes |
|---|---|---|---|---|
| sequence_45 | 4,989 | 469 | 0.964 | Just below threshold; would rank #1 without filter |
| sequence_44 | 2,925 | 356 | 0.921 | |
| sequence_47 | 5,066 | 263 | 0.881 | |
| sequence_46 | 5,073 | 272 | 0.864 | |

These were selected for Run 8 (no event filter). At epp=0.1 they produced only
736 training frames total → mAP50=0.022 (noise level). See `kpis/e2vid_run8.json`.

### Top validation candidates

| Seq | N_ann | EvMB | Area% | Tiny% | Spread | Score |
|---|---|---|---|---|---|---|
| sequence_146 | 2,910 | 3,234 | 0.076 | 98 | 0.171 | 0.876 |
| sequence_147 | 3,369 |   528 | 0.088 | 98 | 0.225 | 0.860 |
| sequence_151 | 3,059 |   810 | 0.089 | 98 | 0.178 | 0.858 |
| sequence_148 | 2,862 |   584 | 0.090 | 98 | 0.172 | 0.836 |
| sequence_131 | 3,021 |   544 | 0.106 | 97 | 0.276 | 0.819 |

sequence_146 is confirmed as the best validation choice: 3,234 MB event data
(most dense of all val candidates), 98% tiny drone — hardest challenge set.

### Recommended split — Run 9

```
Train : sequence_201, sequence_215, sequence_212, sequence_214
Val   : sequence_146
```

Estimated training frames at epp=0.1: ~10,000–15,000 (vs 6,264 in Run 7).

---

## Run history — where selections stand

| Run | Train seqs | Val | mAP50 | Notes |
|---|---|---|---|---|
| Run 5–7 | 84, 85, 201, 124 | 127 | 0.936 (R7) | seq_127 worst val choice (rank 201/231); seq_84/85 redundant |
| Run 8 | 44, 45, 46, 47 | 146 | 0.022 | Failed — seqs 44–47 event-sparse (263–469 MB), only 736 train frames |
| Run 9 (planned) | 201, 212, 214, 215 | 146 | — | New event-dense sequences; ~10–15k estimated train frames |

**seq_201** is the only overlap between Run 7 and Run 9 — it remains the top-ranked
sequence even with the event density filter (2,114 MB).

**seqs 212, 214, 215** are newly identified — the highest annotation counts in the
dataset (5,790–6,769 per sequence), not previously downloaded.

---

## Note on FRED original split vs. our split

The HuggingFace repository organises sequences into `train/` and `test/`
folders — this is the **original FRED dataset split** defined by the dataset
authors, not our training/validation split.

Our selection is based purely on statistical properties (annotation count,
spatial spread, drone size) and deliberately crosses the original split:

| Sequence | FRED folder | Our role |
|---|---|---|
| sequence_44 | `train/` | Train |
| sequence_45 | `train/` | Train |
| sequence_46 | `test/`  | Train |
| sequence_47 | `train/` | Train |
| sequence_146 | `train/` | Val |

This is intentional — sequence_46 scored 0.971 for training and sequence_146
scored 0.876 for validation regardless of their original designation.

---

## Practical constraints

The recommended sequences (44–47, 146) are on HuggingFace but not yet
downloaded. Each zip is ~1 GB (dominated by events.hdf5). To prepare them:

```bash
curl -L -o data/raw/zips/44.zip  "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/44.zip"
curl -L -o data/raw/zips/45.zip  "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/45.zip"
curl -L -o data/raw/zips/46.zip  "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/test/46.zip"
curl -L -o data/raw/zips/47.zip  "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/47.zip"
curl -L -o data/raw/zips/146.zip "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/146.zip"
bash scripts/prepare_sequences.sh 44 45 46 47 146
```

Total download: ~5 GB. Reconstruction on Kaggle T4: ~7 hours for the four
training sequences combined (~50k frames estimated). Fits within the 9-hour
limit.

---

## Caveats

- **Cluster risk:** sequences 44–47 share a number cluster and likely similar
  scene conditions. If the validation sequence has very different lighting or
  background, training on a single scene cluster may overfit to it. Consider
  adding one sequence from a different range (e.g. sequence_20, score=0.963)
  for diversity.

- **Validation spread:** the top val sequences (146–151) all have very low
  spatial spread (0.17–0.23). The drone appears in a limited region of the
  frame. This is fine for measuring detection accuracy but means the model is
  not tested on all spatial positions.

- **Drone type:** `coordinates.txt` includes a drone type label in some entries
  (e.g. "DJI Mini 3"). The analysis currently ignores this. A future improvement
  would filter out sequences with different drone types from the training set
  (as was done for sequence_124, which had a DJI Tello EDU).

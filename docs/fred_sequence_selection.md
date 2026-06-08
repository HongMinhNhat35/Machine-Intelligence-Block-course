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

### 6. Spatial spread (`cx_std + cy_std`)
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
train_score = 0.4 × norm_ann + 0.4 × norm_spread + 0.2 × size_diversity
```
- `norm_ann` = min(n_ann / 3000, 1)
- `norm_spread` = min(spatial_spread / 0.4, 1)
- `size_diversity` = min((small_pct + medium_pct) / 30, 1) — rewards mixed sizes

### Validation score (higher = better)
```
val_score = 0.4 × norm_ann + 0.3 × area_diff + 0.3 × hardness
```
- `area_diff` = how different the mean area is from the training mean (larger
  gap = more informative test of generalisation)
- `hardness` = reward for smaller (harder) drones in validation

---

## Results across all 231 sequences

### Top training candidates

| Seq | N_ann | Area% | Tiny% | Sm% | Spread | Score |
|---|---|---|---|---|---|---|
| sequence_47 | 5,066 | 0.323 | 62 | 36 | 0.431 | 1.000 |
| sequence_44 | 2,925 | 0.299 | 68 | 30 | 0.428 | 0.990 |
| sequence_45 | 4,989 | 0.247 | 74 | 26 | 0.428 | 0.973 |
| sequence_46 | 5,073 | 0.293 | 72 | 27 | 0.381 | 0.971 |
| sequence_20 | 2,993 | 0.355 | 59 | 38 | 0.364 | 0.963 |

Sequences 44–47 form a cluster: high annotation counts, good spatial spread,
and mixed tiny/small distributions. They were likely recorded in the same
session (similar scene/conditions), which is a mild concern for diversity but
outweighed by their label richness.

### Top validation candidates

| Seq | N_ann | Area% | Tiny% | Spread | Score |
|---|---|---|---|---|---|
| sequence_146 | 2,910 | 0.076 | 98 | 0.171 | 0.876 |
| sequence_147 | 3,369 | 0.088 | 98 | 0.225 | 0.860 |
| sequence_151 | 3,059 | 0.089 | 98 | 0.178 | 0.858 |
| sequence_220 | 3,281 | 0.093 | 100 | 0.201 | 0.848 |
| sequence_148 | 2,862 | 0.090 | 98 | 0.172 | 0.836 |

These sequences have the smallest drones in the dataset (area 0.076–0.106%,
97–100% tiny). They form a principled challenge set: the model must generalise
to much harder detections than it saw in training.

### Recommended split

```
Train : sequence_47, sequence_44, sequence_45, sequence_46
Val   : sequence_146
```

---

## Current team selection — where it stands

| Seq | Role | Train rank | Val rank | Notes |
|---|---|---|---|---|
| sequence_84 | train | 108 / 231 | 44 | Mediocre for training; actually decent for val |
| sequence_85 | train | 143 / 231 | 32 | Very similar to seq_84; redundant |
| sequence_201 | train | **14** / 231 | 149 | Best of the current train set; large drone (0.325%) |
| sequence_127 | val | 224 / 231 | 201 | One of the worst choices for both purposes |

**seq_127 is the most problematic.** It is the shortest sequence (40s usable),
has the fewest annotations (1,181), the worst spatial spread (0.236), and ranks
201st out of 231 as a validation sequence. The mAP50=0.29 from the current run
reflects both genuine model limitations *and* a poor measurement setup.

**seq_84 and seq_85 are redundant.** Both have nearly identical area (0.141 vs
0.132%), size distribution (95–98% tiny), and spatial spread. Using both in
training provides minimal additional diversity.

**seq_201 is the only solid choice** in the current set (train rank 14), but its
large drone (0.325%) relative to the validation drone (0.145%) creates a
scale mismatch that artificially deflates mAP on validation.

---

## Practical constraints

The recommended sequences (44–47, 146) are on HuggingFace but not yet
downloaded. Each zip is ~1 GB (dominated by events.hdf5). To prepare them:

```bash
curl -L -o data/raw/zips/44.zip  "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/44.zip"
curl -L -o data/raw/zips/45.zip  "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/45.zip"
curl -L -o data/raw/zips/46.zip  "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/46.zip"
curl -L -o data/raw/zips/47.zip  "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/train/47.zip"
curl -L -o data/raw/zips/146.zip "https://huggingface.co/datasets/GabrieleMagrini/FRED/resolve/main/test/146.zip"
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

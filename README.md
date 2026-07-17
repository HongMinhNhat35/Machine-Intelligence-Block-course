# AMI Group 01

Hybrid drone detection system combining event-camera reconstruction with RGB-based detection and late fusion.

## Getting started

- [Quickstart](https://gitlab.lrz.de/ldv/teaching/ami/ami2026/group01/-/blob/gui/g1_QUICKSTART.md) — installation and first run
- [GUI user guide](https://gitlab.lrz.de/ldv/teaching/ami/ami2026/group01/-/blob/gui/docs/gui_user_guide.md) — how to use the web interface

## Repository structure

This is a **monorepo**. All services share the same codebase; branches reflect each team member's area of contribution.

### Branches

| Branch | Content |
|---|---|
| `e2vid` | E2VID reconstruction service + YOLO detection (Klaus) |
| `hypere2vid` | HyperE2VID reconstruction service + YOLO detection |
| `fusion` | Late-fusion layer combining event and RGB detections (Kutay) |
| `gui` | Web frontend — separate git history from `ami-gui` repo |

### Directory layout

```
services/
  e2vid/          ← E2VID reconstruction + YOLOv8 detection service
  hypere2vid/     ← HyperE2VID reconstruction + YOLOv8 detection service
  fusion/         ← Late-fusion service (event + RGB detections)
notebooks/        ← Kaggle/Colab training notebooks per team member
scripts/          ← Utility and tuning scripts
docs/             ← Pipeline documentation
docker-compose.yml
```

### Adding notebooks or scripts

Team members push their training notebooks and analysis scripts to their respective branch:

```bash
git clone git@gitlab.lrz.de:ldv/teaching/ami/ami2026/group01.git
cd group01
git checkout fusion        # or hypere2vid
git add notebooks/my_notebook.ipynb scripts/my_script.py
git commit -m "add training notebook"
git push origin fusion
```

## Documentation

Detailed pipeline and setup guides are in [`docs/`](docs/):

- [`docs/e2vid_pipeline.md`](docs/e2vid_pipeline.md) — end-to-end E2VID pipeline
- [`docs/e2vid_service.md`](docs/e2vid_service.md) — Docker service setup
- [`docs/kpi.md`](docs/kpi.md) — KPI schema and results overview
- [`docs/fred_sequence_selection.md`](docs/fred_sequence_selection.md) — FRED dataset sequence selection


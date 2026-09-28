# ClearPath

Cascade hybrid next-activity recommender on the Open University Learning Analytics Dataset (OULAD). ItemKNN generates candidates; LambdaMART ranks them; TreeSHAP explains each recommendation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_oulad.py
```

On macOS, LightGBM needs OpenMP:

```bash
brew install libomp
```

Use the helper wrapper so `libomp` is on the library path:

```bash
chmod +x scripts/run.sh
./scripts/run.sh python -m clearpath.data_prep
```

## Pipeline (phase order)

```bash
./scripts/run.sh python -m clearpath.data_prep
./scripts/run.sh python -m clearpath.splits
./scripts/run.sh python -m clearpath.candidates
./scripts/run.sh python -m clearpath.features
./scripts/run.sh python -m clearpath.ranker
./scripts/run.sh python -m clearpath.evaluate
./scripts/run.sh python -m clearpath.explain
./scripts/run.sh python -m clearpath.figures
./scripts/run.sh python -m clearpath.app
./scripts/run.sh python -m pytest tests/ -q
```

## Live demo

```bash
./scripts/run.sh python -m clearpath.app
# → http://127.0.0.1:5050
```

Pick a course and student in the demo UI, compare ClearPath vs Most popular, then expand a row for TreeSHAP reasons. Latency is shown on every recommend call.

## Layout

- `clearpath/` - pipeline modules
- `data/raw/` - OULAD CSVs (gitignored)
- `data/processed/` - matrices, metadata, split indices
- `artifacts/` - models, SHAP cache, metrics, figures
- `tests/` - requirement checks

## Scope

Modules BBB/DDD/FFF x presentations 2013B/2013J/2014B/2014J. Offline evaluation only; no user study.

## Submission packaging

`data/raw/` and most of `data/processed/` are regenerable. The submission zip is **code + demo artefacts only**.

```bash
./scripts/package_submission.sh
# -> ClearPath_submission.zip
```

Included: `clearpath/`, `scripts/run.sh`, `scripts/download_oulad.py`, `tests/`, `requirements.txt`, `README.md`, models, metrics, figures, SHAP cache, demo rankings.

Excluded: raw/processed data dumps, logs, local notes.

Marker setup after unzip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional full rebuild from OULAD:
# python scripts/download_oulad.py && ./scripts/run.sh python -m clearpath.data_prep ...
./scripts/run.sh python -m clearpath.app
```

`requirements.txt` uses pinned versions. On macOS install `libomp` before LightGBM.

Agent Handoff (Mac)
====================

Target path on Mac:
- Code: `/Users/gilraitses/INDYsim`
- Data (validated H5s): `/Users/gilraitses/INDYsim/data/h5_validated/`

What to download from OneDrive
- 10 validated H5 files (same names as lab PC) into `data/h5_validated/`
- `manifest.json` into `data/h5_validated/`

Key scripts to run
1) Analyze all validated H5s
```bash
mkdir -p "/Users/gilraitses/INDYsim/data/h5_validated/analysis"
cd "/Users/gilraitses/INDYsim/scripts/2025-12-04/platform_liberation"
python3 engineer_dataset_from_h5.py "/Users/gilraitses/INDYsim/data/h5_validated" \
  -o "/Users/gilraitses/INDYsim/data/h5_validated/analysis"
```
Outputs: per-file JSONs + `combined_analysis.json` in `/data/h5_validated/analysis/`.

2) Export master H5 for simulator (optional)
```bash
cd "/Users/gilraitses/INDYsim/scripts/2025-12-04/platform_liberation/agent/worktree"
python3 export_master_h5.py \
  --combined "/Users/gilraitses/INDYsim/data/h5_validated/analysis/combined_analysis.json" \
  --output "/Users/gilraitses/INDYsim/data/h5_validated/master_sim_input.h5"
```
Also writes a manifest JSON alongside the master H5.

What the enhanced analysis includes
- Per track: reversals (count/duration/timing), turns (count/rate), mean SpeedRunVel, %negative SpeedRunVel, total duration.
- Per stimulus window (LED-derived ton/toff): per-track window stats, population window aggregates.
- Concurrency estimate: active tracks per time bin.
- Uses `derived_quantities/sloc`, `/lengthPerPixel` root, SpeedRunVel<0 for >3s reversals, turn threshold 45°.

Files of interest
- `scripts/2025-12-04/platform_liberation/engineer_dataset_from_h5.py` (enhanced)
- Originals kept as:
  - `scripts/engineer_dataset_from_h5_OG.py`
  - `scripts/2025-12-04/platform_liberation/engineer_dataset_from_h5_backup.py`
- Export helper: `agent/worktree/export_master_h5.py`

Notes
- If Python on Mac is `python` instead of `python3`, adjust commands.
- No MATLAB engine needed for this step; all analysis is Python-only.


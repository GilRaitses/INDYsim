Worktree Notes (Agent)
======================

Purpose
- Run enhanced analysis on validated H5s and optionally export a master H5 for the simulator.

Run analysis
```bash
mkdir -p "/Users/gilraitses/INDYsim/data/h5_validated/analysis"
cd "/Users/gilraitses/INDYsim/scripts/2025-12-04/platform_liberation"
python3 engineer_dataset_from_h5.py "/Users/gilraitses/INDYsim/data/h5_validated" \
  -o "/Users/gilraitses/INDYsim/data/h5_validated/analysis"
```

Export master H5 (optional)
```bash
cd "/Users/gilraitses/INDYsim/scripts/2025-12-04/platform_liberation/agent/worktree"
python3 export_master_h5.py \
  --combined "/Users/gilraitses/INDYsim/data/h5_validated/analysis/combined_analysis.json" \
  --output "/Users/gilraitses/INDYsim/data/h5_validated/master_sim_input.h5"
```

Outputs
- Per-file JSONs and `combined_analysis.json` in `/data/h5_validated/analysis/`
- Optional `master_sim_input.h5` plus manifest (JSON)

Inputs expected
- 10 validated H5s + `manifest.json` in `/data/h5_validated/`

Key script
- `export_master_h5.py` (in this folder)


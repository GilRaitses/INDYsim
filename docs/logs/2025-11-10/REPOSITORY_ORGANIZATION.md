# Repository Organization - November 10, 2025

## Structure Created

### New Folders
```
D:\INDYsim\
├── docs/
│   ├── logs/
│   │   └── 2025-11-10/
│   │       └── REPOSITORY_ORGANIZATION.md (this file)
│   └── backups/
│       └── scripts_backup_2025-11-10/
│           └── [backup of all scripts]
└── scripts/
    └── 2025-11-10/
        └── agent-handoffs/
            ├── DAILY_PROTOCOL.md
            └── composer_larry_20251110-160000_fix-turn-rate-calculation-matlab-match.md
```

## Purpose

### docs/logs/
- Daily logs documenting work progress
- Organized by date: `YYYY-MM-DD/`
- Each day can have subfolders for specific work items

### docs/backups/
- Backups of critical files before major changes
- Named with date: `scripts_backup_YYYY-MM-DD/`
- Created before fixing turn rate calculation

### scripts/YYYY-MM-DD/
- Daily work folders following mechanosensation repo pattern
- Contains scripts and agent handoffs for that day's work
- Agent handoffs follow naming convention: `<author>_<recipient>_<timestamp>_<subject>.md`

## Daily Protocol

Copied from `d:\mechanosensation\scripts\2025-11-10\agent-handoffs\DAILY_PROTOCOL.md`

Key points:
- All handoff files follow strict naming convention
- Daily logs required in `docs/logs/YYYY-MM-DD.md`
- Status tracking in `agent-handoffs/STATUS.md`

## Today's Work

### Turn Rate Calculation Fix
- **Issue:** Python implementation didn't match MATLAB reference
- **Fix:** Changed from summing boolean values to counting reorientation times directly
- **File:** `scripts/create_eda_figures.py`
- **Documentation:** See handoff document in `scripts/2025-11-10/agent-handoffs/`

### Backup Created
- All scripts backed up to `docs/backups/scripts_backup_2025-11-10/`
- Created before making changes to `create_eda_figures.py`

## Path Notes

### Current Script Paths
- Scripts are in `D:\INDYsim\scripts/`
- Data files in `D:\INDYsim\data/`
- Output in `D:\INDYsim\output/`

### Relative Paths Recommended
- Scripts should use relative paths where possible
- Example: `Path("data/engineered")` instead of `Path(r"D:\INDYsim\data\engineered")`
- Some scripts may have hardcoded paths that need updating

## Next Steps

1. Review scripts for hardcoded paths and update to relative paths
2. Continue daily logging in `docs/logs/`
3. Follow daily protocol for future work
4. Test turn rate fix with actual data

---

**Created:** 2025-11-10  
**Purpose:** Document repository organization and today's work


# Path Cleanup Notes - November 10, 2025

## Hardcoded Paths Found

The following scripts contain hardcoded paths that should be updated to use relative paths or configuration:

### Scripts with Hardcoded Paths

1. **create_eda_figures.py** (lines 634, 637)
   - `/Users/gilraitses/mechanosensation/h5tests/...` (macOS path)
   - `/Users/gilraitses/ecs630/labs/termprojectproposal/output/...` (macOS path)
   - **Status:** Default arguments in argparse, should use relative paths

2. **inspect_h5_files.py** (line 190)
   - `/Users/gilraitses/mechanosensation/h5tests` (macOS path)
   - **Status:** Hardcoded default, should use relative path

3. **analyze_h5_stimulus.py** (lines 182-183)
   - `/Users/gilraitses/mechanosensation/h5tests/...` (macOS paths)
   - **Status:** Example file paths, should use relative paths

4. **check_actual_pulse_duration.py** (line 89)
   - `/Users/gilraitses/mechanosensation/h5tests/...` (macOS path)
   - **Status:** Hardcoded test file, should use relative path

5. **visualize_behavioral_events_stepwise.py** (line 392)
   - `/Users/gilraitses/mechanosensation/h5tests/...` (macOS path)
   - **Status:** Default argument, should use relative path

6. **visualize_behavioral_events.py** (line 492)
   - `/Users/gilraitses/mechanosensation/h5tests/...` (macOS path)
   - **Status:** Default argument, should use relative path

## Recommended Fixes

### Pattern to Use
```python
from pathlib import Path

# Instead of:
h5_file = Path('/Users/gilraitses/mechanosensation/h5tests/file.h5')

# Use:
script_dir = Path(__file__).parent.parent
h5_file = script_dir / 'data' / 'file.h5'
```

### For Default Arguments
```python
# Instead of:
parser.add_argument('--h5-file', default='/Users/.../file.h5')

# Use:
default_h5 = Path(__file__).parent.parent / 'data' / 'file.h5'
parser.add_argument('--h5-file', default=str(default_h5))
```

## Priority

**Low Priority** - These are mostly default arguments that can be overridden via command line. However, fixing them would make the scripts more portable and easier to use.

## Notes

- Most scripts already use `Path` objects, which is good
- The hardcoded paths are primarily in default arguments
- Scripts work when paths are provided via command line arguments
- Consider creating a `config.py` file for common paths

---

**Created:** 2025-11-10  
**Status:** Documented for future cleanup


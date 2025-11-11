# LED Value Timecode Alignment Issue - November 11, 2025

## Problem Statement

**Critical Issue:** Biologically implausible stimulus-response event dynamics due to misaligned LED value timecode with maggot track timecode.

**Impact:** Cannot analyze relationships with stimulus phase at relevant temporal resolution. All stimulus-response analysis is compromised until this is fixed.

## Root Cause

LED values are combined into an MMF (Memory-Mapped File) in LabVIEW and written to a bin file through the Track Extraction Software (`D:\magniphyq\codebase\Track-Extraction-Software`). The timecode alignment between LED values and maggot tracks is not properly handled in the current Python implementation.

## Current Implementation

**File:** `scripts/engineer_dataset_from_h5.py`

**Current Approach:**
- Direct reading from `global_quantities/led1Val/yData` in H5 files
- Assumes LED values are already aligned with track timecode
- Does not account for potential timecode offsets

**Issue:** This approach may not properly align LED value timecode with track timecode, leading to:
- Stimulus-response dynamics appearing at incorrect times
- Biologically implausible event timing
- Inability to analyze stimulus phase relationships

## Reference Implementation

**MATLAB Codebase:** `D:\magniphyq\codebase\Matlab-Track-Analysis-SkanataLab`

**Key Files:**
- `@GlobalLookupTable/createLedTableFromBitFile.m` - Creates LED lookup table from bit files
- `@GlobalLookupTable/GlobalLookupTable.m` - Handles LED value alignment

**Key Concepts:**
1. **ledFrame and bufnum mapping:** Maps LED values to track frames using buffer numbers
2. **bytes/bits per frame calculation:** Determines how LED values map to track frames
3. **Buffer axis to bit axis mapping:** Maps buffer numbers to positions in LED value bin files
4. **Interpolation to ETI:** Interpolates LED values to elapsed time index (ETI) for alignment

**MATLAB Alignment Method:**
```matlab
% Calculate bytes per frame
bytesperframe = round((ds.ledNBytesOut(ind))/(ds.bufnum(ind) - ds.bufnum(firstFrame)));

% Map buffer numbers to LED value positions
bufaxis = (ds.bufnum - ds.bufnum(firstFrame))*bytesperframe;

% Map to bit file positions
bitaxis = min(bufaxis):(min(length(bits)-1,max(bufaxis)));

% Interpolate to elapsed time
timaxis = interp1(bufaxis, et, bitaxis);
```

## Investigation Plan

### Task 0.1: Track Extraction Software Investigation (osito-tender)
- Review Track Extraction Software documentation
- Understand MMF creation process in LabVIEW
- Document LED value bin file format
- Map LED value timecode to track timecode

### Task 0.2: MATLAB Alignment Implementation (conejo-code)
- Study MATLAB GlobalLookupTable alignment method
- Implement ledFrame/bufnum mapping in Python
- Fix temporal alignment in H5 processing
- Validate with biologically plausible results

## Success Criteria

- [ ] LED values properly aligned with track timecode
- [ ] Stimulus-response dynamics are biologically plausible
- [ ] Turn rate spikes appear at correct times relative to stimulus
- [ ] Alignment matches MATLAB reference implementation
- [ ] Can analyze stimulus phase relationships at relevant temporal resolution

## References

- Track Extraction Software: `D:\magniphyq\codebase\Track-Extraction-Software`
- MATLAB Track Analysis: `D:\magniphyq\codebase\Matlab-Track-Analysis-SkanataLab`
- Work Tree: `docs/work-trees/2025-11-11-work-tree.md`

---

**Status:** Critical - Blocking all stimulus-response analysis  
**Priority:** P0  
**Created:** 2025-11-11


# Lab Experiment Set (ESET) Naming Convention

**Date:** 2025-11-11  
**Prepared by:** larry  
**Source:** User clarification - Lab naming convention

## Directory Structure

**Parent Folder:** Optogenetic variant of larva (e.g., `GMR61@GMR61`)
- Contains engineered sensory activation by red light
- Represents the neural pathway variant

**ESET Folders:** Inside parent folder, named with lab convention

## Lab ESET Naming Format

**Pattern:** `T_{LED1_COLOR}_{WAVEFORM}_{MIN}to{MAX}PWM_{REST}#{LED2_TYPE}_{LED2_COLOR}_{LED2_VALUES}PWM`

**Example:** `T_Re_Sq_0to250PWM_30#C_Bl_7PWM`

## Format Breakdown

### Prefix: `T_`
- **T** = Time (indicates time-based stimulus)

### LED1 Section: `{COLOR}_{WAVEFORM}_{MIN}to{MAX}PWM_{REST}`

**Example:** `Re_Sq_0to250PWM_30`

- **COLOR:** `Re` = Red, `Bl` = Blue
- **WAVEFORM:** `Sq` = Square wave
- **MINtoMAXPWM:** PWM range (e.g., `0to250`, `50to250`)
- **REST:** LED off duration in seconds (e.g., `30`)

**Translation:** `Re_Sq_0to250PWM_30`
- LED1: Red, Square wave
- PWM: 0 to 250
- 30 second LED off duration (resting interval)

### Separator: `#`
- Separates LED1 and LED2 specifications

### LED2 Section: `{TYPE}_{COLOR}_{VALUES}PWM`

**Example 1:** `C_Bl_7PWM` (Constant)
- **C** = Constant
- **Bl** = Blue
- **7PWM** = Constant at 7 PWM

**Example 2:** `T_Bl_Sq_5to15PWM_30` (Time-varying)
- **T** = Time (time-varying)
- **Bl** = Blue
- **Sq** = Square wave
- **5to15PWM** = PWM range 5 to 15
- **30** = LED off duration

## Actual ESET Folders Found

**Location:** `data/matlab_data/GMR61@GMR61/`

1. **`T_Re_Sq_0to250PWM_30#C_Bl_7PWM`**
   - **LED1:** Red, Square wave, 0-250 PWM, 30s LED off
   - **LED2:** Constant Blue at 7 PWM
   - **Stimulus period:** 10 seconds (validate using pattern recognition)

2. **`T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30`**
   - **LED1:** Red, Square wave, 0-250 PWM, 30s LED off
   - **LED2:** Blue, Square wave, 5-15 PWM, 30s LED off
   - **Stimulus period:** 10 seconds (validate using pattern recognition)

3. **`T_Re_Sq_50to250PMW_30#T_Re_Sq_5to15PMW_30`** ⚠️ **LIKELY HAS TWO TYPOS**
   - **LED1:** Red, Square wave, 50-250 PWM, 30s LED off
   - **LED2:** Likely Blue (not Red) - folder name has typo
   - **Typo 1:** `PMW` instead of `PWM` (in both LED1 and LED2 sections)
   - **Typo 2:** `T_Re_Sq` (Red) instead of `T_Bl_Sq` (Blue) for LED2
   - **Expected correct name:** `T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30`
   - **Stimulus period:** 10 seconds (validate using pattern recognition)

4. **`T_Re_Sq_50to250PWM_30#C_Bl_7PWM`**
   - **LED1:** Red, Square wave, 50-250 PWM, 30s LED off
   - **LED2:** Constant Blue at 7 PWM
   - **Stimulus period:** 10 seconds (validate using pattern recognition)

## Mapping to Simplified Names

**Lab Format → Simplified Format:**

1. `T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30` → `R_0_250_30_B_5_15_30`
2. `T_Re_Sq_50to250PMW_30#T_Re_Sq_5to15PMW_30` → `R_50_250_30_R_5_15_30` ⚠️ (LED2 is Red, not Blue)
3. `T_Re_Sq_0to250PWM_30#C_Bl_7PWM` → `R_0_250_30_B_7`
4. `T_Re_Sq_50to250PWM_30#C_Bl_7PWM` → `R_50_250_30_B_7`

**Note:** The simplified names (`R_0_250_30_B_5_15_30`) are user-created shortcuts. The actual lab folders use the full format (`T_Re_Sq_0to250PWM_30#T_Bl_Sq_5to15PWM_30`).

**Important:** Folder #3 (`T_Re_Sq_50to250PMW_30#T_Re_Sq_5to15PMW_30`) likely has **two typos**:
- Typo 1: `PMW` instead of `PWM` (appears twice)
- Typo 2: `T_Re_Sq` (Red) instead of `T_Bl_Sq` (Blue) for LED2
- Expected correct name: `T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30`
- Status: Likely correct (user hypothesis), pending confirmation with Devindi

## Key Differences

**Lab Format:**
- Uses full words: `Re` (Red), `Bl` (Blue), `Sq` (Square)
- Includes waveform type: `Sq` for square wave
- Uses `#` separator
- LED2 type: `C` (Constant) or `T` (Time-varying)
- Includes `PWM` suffix

**Simplified Format:**
- Uses single letters: `R` (Red), `B` (Blue)
- No waveform specification
- Uses `_` separator throughout
- LED2 values: Multiple values (`5_15_30`) or single constant (`7`)
- No `PWM` suffix

## Stimulus Period Validation

**Important:** All stimulus periods are **10 seconds**, but must be **validated** using pattern recognition variables:

**Validation Variables:**
- `led1Val_ton` - LED1 ON transition times (ETI)
- `led1Val_toff` - LED1 OFF transition times (ETI)
- `led2Val_ton` - LED2 ON transition times (ETI)
- `led2Val_toff` - LED2 OFF transition times (ETI)

**Validation Process:**
1. Detect LED ON/OFF transitions using threshold
2. Calculate intervals between transitions
3. Verify period is ~10 seconds
4. Log start and end ETIs for each stimulus pulse
5. Calculate duty cycle for each pulse

## Pattern Recognition Implementation

**Current Implementation:**
- Period detection from LED ON/OFF pattern
- Uses threshold (10% of max) to detect transitions
- Calculates median of ON transition intervals
- Filters to reasonable range (5-30 seconds)
- Validates consistency with ETI

**Required for Validation:**
- Log start ETI for each stimulus pulse
- Log end ETI for each stimulus pulse
- Calculate duty cycle for each pulse
- Verify period is ~10 seconds
- Document any deviations

## Issues Identified

### Folder #3: Multiple Typos Likely

**Folder:** `T_Re_Sq_50to250PMW_30#T_Re_Sq_5to15PMW_30`

**Hypothesis:** Folder has **two typos**:
1. **Typo 1:** `PMW` instead of `PWM` (appears twice in folder name)
2. **Typo 2:** `T_Re_Sq` (Red) instead of `T_Bl_Sq` (Blue) for LED2

**Rationale:**
- All other folders have Blue LED2 (`T_Bl_Sq` or `C_Bl`)
- If someone made one typo (`PMW`), they likely made another (`Re` instead of `Bl`)
- Pattern consistency suggests LED2 should be Blue

**Expected Correct Name:** `T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30`

**Status:** ⏳ **LIKELY CORRECT** - User hypothesis suggests two typos, pending confirmation with Devindi

**Action Required:**
- If confirmed: Rename folder to `T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30`
- Verify LED2 is actually Blue in the data files

---

**Status:** Lab naming convention documented, issues identified


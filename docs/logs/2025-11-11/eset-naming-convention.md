# Experiment Set (ESET) Naming Convention

**Date:** 2025-11-11  
**Prepared by:** larry  
**Source:** User clarification

**Note:** This document describes the **simplified naming convention** used in `data/matlab_data/` folder names. For the **actual lab naming convention** used in experiment files, see `docs/logs/2025-11-11/lab-eset-naming-convention.md`.

## Naming Format

**Pattern:** `{LED1_COLOR}_{LED1_MIN}_{LED1_MAX}_{LED1_REST}_{LED2_COLOR}_{LED2_VALUES}`

**Example:** `R_0_250_30_B_5_15_30`

## Format Breakdown

### LED1 (First LED - Red in examples)

**Format:** `{COLOR}_{MIN_PWM}_{MAX_PWM}_{REST_INTERVAL}`

- **COLOR:** `R` = Red, `B` = Blue
- **MIN_PWM:** Minimum PWM value (e.g., 0, 50)
- **MAX_PWM:** Maximum PWM value (e.g., 250)
- **REST_INTERVAL:** Resting interval in seconds (e.g., 30)

**Example:** `R_0_250_30`
- LED1 is Red
- PWM oscillates between 0 and 250
- 30 second resting interval between cycles

### LED2 (Second LED)

**Format:** `{COLOR}_{VALUES}`

- **COLOR:** `R` = Red, `B` = Blue
- **VALUES:** Either:
  - Multiple PWM values: `5_15_30` (three different PWM levels)
  - Single constant PWM: `7` (constant at 7 PWM)

**Example 1:** `B_5_15_30`
- LED2 is Blue
- PWM cycles through values: 5, 15, 30

**Example 2:** `B_7`
- LED2 is Blue
- Constant at 7 PWM

## Complete Examples

### Example 1: `R_0_250_30_B_5_15_30`
- **LED1:** Red, PWM 0-250, 30s rest
- **LED2:** Blue, PWM values 5, 15, 30
- **Stimulus period:** 10 seconds (validate using pattern recognition)

### Example 2: `R_50_250_30_B_5_15_30`
- **LED1:** Red, PWM 50-250, 30s rest
- **LED2:** Blue, PWM values 5, 15, 30
- **Stimulus period:** 10 seconds (validate using pattern recognition)

### Example 3: `R_0_250_30_B_7`
- **LED1:** Red, PWM 0-250, 30s rest
- **LED2:** Blue, constant 7 PWM
- **Stimulus period:** 10 seconds (validate using pattern recognition)

### Example 4: `R_50_250_30_B_7`
- **LED1:** Red, PWM 50-250, 30s rest
- **LED2:** Blue, constant 7 PWM
- **Stimulus period:** 10 seconds (validate using pattern recognition)

## ⚠️ Potential Issue Identified

### ESET: `R_50_250_30_R_5_15_30`

**Issue:** LED2 is listed as `R` (Red), but this seems incorrect.

**Expected:** Should probably be `B` (Blue) based on pattern:
- All other esets have Blue LED2 (`B`)
- Format suggests: `R_50_250_30_B_5_15_30`
- User notes: "if this is right i'll ask you to change it back but i think its supposed to be blue in the second led color element position"

**Status:** ⏳ **PENDING CONFIRMATION** - User will confirm with Devindi

**Action Required:**
- If confirmed incorrect: Change folder name from `R_50_250_30_R_5_15_30` to `R_50_250_30_B_5_15_30`
- If confirmed correct: Document as special case (both LEDs are Red)

**Location:** `data/matlab_data/R_50_250_30_R_5_15_30/`

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
5. Document duty cycle for each pulse

## All 4 Experiment Conditions

Based on user description:

1. **`R_0_250_30_B_5_15_30`**
   - LED1: Red, 0-250 PWM, 30s rest
   - LED2: Blue, PWM 5, 15, 30

2. **`R_50_250_30_B_5_15_30`**
   - LED1: Red, 50-250 PWM, 30s rest
   - LED2: Blue, PWM 5, 15, 30

3. **`R_0_250_30_B_7`**
   - LED1: Red, 0-250 PWM, 30s rest
   - LED2: Blue, constant 7 PWM

4. **`R_50_250_30_B_7`**
   - LED1: Red, 50-250 PWM, 30s rest
   - LED2: Blue, constant 7 PWM

**Note:** There is a folder `R_50_250_30_R_5_15_30` (lab format: `T_Re_Sq_50to250PMW_30#T_Re_Sq_5to15PMW_30`) that likely has **two typos**:
- Typo 1: `PMW` instead of `PWM`
- Typo 2: `R` (Red) instead of `B` (Blue) for LED2
- Expected correct name: `R_50_250_30_B_5_15_30` (simplified) or `T_Re_Sq_50to250PWM_30#T_Bl_Sq_5to15PWM_30` (lab format)
- Status: Likely correct (user hypothesis), pending confirmation with Devindi

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

---

**Status:** Naming convention documented, pending confirmation on `R_50_250_30_R_5_15_30`


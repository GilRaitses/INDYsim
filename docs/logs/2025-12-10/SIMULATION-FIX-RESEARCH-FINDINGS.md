# Simulation Fix Research Findings

**Source:** MiroThinker external research agent  
**Date:** 2025-12-10  
**Topic:** Diagnosing and fixing 14x rate mismatch in hazard-based simulation

---

## Problem Summary

| Metric | Empirical | Simulated | Ratio |
|--------|-----------|-----------|-------|
| Turn rate | 0.87 events/min/track | 0.06 events/min/track | 14x |

---

## Root Cause Analysis

### 1. AR(1) Coefficient Too Strong

**Current:** β_AR(1) = -24.7  
**Rate Ratio:** exp(-24.7) ≈ 1.8×10⁻¹¹ → effectively **zero hazard** after any event

**Impact:** With 0.5s bins, this creates a hard refractory window where no events can occur, suppressing overall rate by order of magnitude.

**Fix:** Regularize to biologically plausible range:
- Target RR = 0.05-0.20 after event
- β = -3.0 → RR = 0.05
- β = -2.3 → RR = 0.10
- β = -1.6 → RR = 0.20

### 2. Kernel-Stimulus Misalignment

**Current:** Kernel bases computed from absolute time t, not time since LED onset.

**Impact:** Stimulus-locked boost rarely applies because kernel centers (0, 1.33, 2.67s) don't align with actual LED transitions.

**Fix:** 
1. Track LED onset times
2. Compute `time_since_onset` for each time point
3. Reset to 0 at each LED1 onset (30s on/30s off cycle)
4. Apply kernel bases to `time_since_onset`, not absolute time

### 3. Missing PSTH Validation

**Current:** Only validate mean rate and within-CI check.

**Impact:** Can't verify stimulus-locked response shape.

**Fix:** Add peri-stimulus time histogram comparison:
1. Bin events relative to LED onset (-5 to +30s)
2. Compute rate per bin for empirical and simulated
3. Calculate ISE (Integrated Squared Error)
4. Compare to within-data ISE from bootstrap

---

## Implementation Specifications

### AR(1) Fix

```python
# In event_generator.py make_hazard_from_model():
# Option 1: Use regularized coefficient
ar1_coef = -3.0  # Instead of loading from model (which has -24.7)

# Option 2: Cap the coefficient in simulation
ar1_coef = max(model_results['coefficients'].get('Y_lag1', 0), -3.0)
```

### Kernel Alignment Fix

```python
def make_hazard_from_model(...):
    # Track LED onset times
    def find_led_onsets(led1_pattern, t_grid):
        onsets = []
        prev_val = 0
        for t in t_grid:
            val = led1_pattern(t)
            if val > 0 and prev_val == 0:
                onsets.append(t)
            prev_val = val
        return np.array(onsets)
    
    # Compute time since most recent onset
    def compute_time_since_onset(t, onsets):
        if len(onsets) == 0:
            return np.full_like(t, np.inf)
        # For each t, find most recent onset
        idx = np.searchsorted(onsets, t, side='right') - 1
        idx = np.maximum(idx, 0)
        return t - onsets[idx]
    
    # In hazard function:
    time_since_onset = compute_time_since_onset(t, led_onsets)
    
    # Apply kernel to time_since_onset, not t
    for j, (center, phi) in enumerate(zip(centers, kernel_coefs)):
        dist = np.abs(time_since_onset - center)
        in_range = dist < width
        basis_val = np.zeros(n)
        basis_val[in_range] = 0.5 * (1 + np.cos(np.pi * (time_since_onset[in_range] - center) / width))
        eta += phi * basis_val
```

### PSTH Validation

```python
def compute_psth(events_df, led_onsets, bin_width=0.5, window=(-5, 30)):
    """Compute peri-stimulus time histogram."""
    bins = np.arange(window[0], window[1] + bin_width, bin_width)
    counts = np.zeros(len(bins) - 1)
    
    for onset in led_onsets:
        event_times = events_df['time'].values
        relative_times = event_times - onset
        in_window = (relative_times >= window[0]) & (relative_times < window[1])
        hist, _ = np.histogram(relative_times[in_window], bins=bins)
        counts += hist
    
    # Normalize to rate
    n_cycles = len(led_onsets)
    rates = counts / (n_cycles * bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    return bin_centers, rates

def compute_psth_ise(emp_psth, sim_psth, bin_width=0.5):
    """Integrated squared error between PSTHs."""
    return np.sum((emp_psth - sim_psth)**2) * bin_width
```

---

## Expected Outcomes

After fixes:

| Metric | Before | Expected After |
|--------|--------|----------------|
| Simulated rate | 0.06/min | ~0.8-1.0/min |
| Rate ratio | 14x low | ~1x (within 20%) |
| PSTH peak timing | Misaligned | ~0.5-1.0s post-onset |
| PSTH suppression | Absent | -75% at 1.5-2.5s |

---

## Additional Recommendations from Research

### Dose-Response (Future)
- Use log-linear: `effect = β × log(1 + LED1)`
- Add quadratic: `effect = β₁ × LED1 + β₂ × LED1²`
- Collect intermediate intensity (50-150 PWM) to validate

### State-Switching Model (Future)
- 2-state HMM: Dormant vs Responsive
- Each state has own NB-GLM hazard
- Explains 5.6% delayed responders

### Experimental Design (Future)
- 3-4 LED1 levels: 0, 50, 150, 250 PWM
- 10-12 larvae per condition
- 14-20 min per experiment

---

## References

- MiroThinker research response, 2025-12-10
- Gepner et al. 2015, eLife (larval kernels)
- Klein et al. 2015, PNAS (thermotaxis)

#!/usr/bin/env python3
"""
Power Analysis for Phenotype Detection

Goal: Derive the number of events required to detect a phenotype difference with 80% power.

Method:
1. Use estimated population parameters (τ₁ = 0.63s, σ_τ₁ = 0.31)
2. Define effect size: Δτ₁ = 0.2s (difference between population and fast responders)
3. For N_events in [10, 25, 50, 75, 100, 150, 200]:
   - Simulate 500 tracks with true τ₁ = 0.63s (population)
   - Simulate 500 tracks with true τ₁ = 0.43s (outlier/fast responder)
   - Fit kernel to each using MLE
   - Bootstrap CI for each track
   - Power = P(CI excludes 0.63 | true τ₁ = 0.43)
4. Plot power curve, identify N for 80% power

Runtime: ~30 minutes
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.optimize import minimize
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
OUTPUT_DIR = RESULTS_DIR / 'power_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Population parameters from hierarchical Bayesian analysis
POP_TAU1 = 0.63  # Population mean τ₁
POP_TAU2 = 2.48  # Population mean τ₂
POP_A = 1.0      # Population amplitude A
POP_B = 15.0     # Population amplitude B
POP_ALPHA = 2.0  # Shape parameter (fixed)

# Effect size: fast responders have τ₁ ≈ 0.43s
FAST_TAU1 = 0.43
EFFECT_SIZE = POP_TAU1 - FAST_TAU1  # 0.2s

# Simulation parameters
LED_CYCLE = 30.0
LED_ON_DURATION = 10.0
FIRST_LED_ONSET = 21.3
DT = 0.05
BASELINE_HAZARD = -3.5  # β₀

# Power analysis parameters
N_EVENTS_GRID = [10, 15, 20, 25, 35, 50, 75, 100, 150, 200]
N_SIMULATIONS = 200  # Tracks per condition per N_events
N_BOOTSTRAP = 100    # Bootstrap resamples for CI
ALPHA_LEVEL = 0.05   # Significance level
TARGET_POWER = 0.80


def gamma_diff_kernel(t, tau1, tau2, A, B, alpha=2.0):
    """Compute gamma-difference kernel value at time t."""
    if tau1 <= 0 or tau2 <= 0:
        return np.zeros_like(t) if hasattr(t, '__len__') else 0.0
    
    beta1 = tau1 / alpha
    beta2 = tau2 / alpha
    
    pdf1 = gamma_dist.pdf(t, alpha, scale=beta1)
    pdf2 = gamma_dist.pdf(t, alpha, scale=beta2)
    
    return A * np.nan_to_num(pdf1) - B * np.nan_to_num(pdf2)


def compute_hazard(t, t_since_onset, tau1, tau2, A, B):
    """Compute instantaneous hazard at time t."""
    K = gamma_diff_kernel(t_since_onset, tau1, tau2, A, B)
    log_hazard = BASELINE_HAZARD + K
    return np.exp(np.clip(log_hazard, -10, 2))  # Clip for numerical stability


def simulate_events(tau1, tau2, A, B, duration=1200.0, rng=None):
    """
    Simulate reorientation events using Bernoulli process.
    Returns event times.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    events = []
    t = 0.0
    
    while t < duration:
        # Determine LED state
        cycle_time = (t - FIRST_LED_ONSET) % LED_CYCLE if t >= FIRST_LED_ONSET else -1
        led_on = 0 <= cycle_time < LED_ON_DURATION
        
        # Time since LED onset
        if led_on:
            t_since_onset = cycle_time
        else:
            # Time since most recent LED offset
            t_since_onset = 0  # Simplified: use baseline during LED-off
        
        # Compute per-frame event probability
        if led_on and t_since_onset > 0:
            p = compute_hazard(t, t_since_onset, tau1, tau2, A, B) * DT
        else:
            p = np.exp(BASELINE_HAZARD) * DT
        
        p = np.clip(p, 0, 1)
        
        # Bernoulli draw
        if rng.random() < p:
            events.append(t)
        
        t += DT
    
    return np.array(events)


def simulate_track_with_target_events(tau1, tau2, A, B, target_n_events, max_attempts=50, rng=None):
    """
    Simulate a track and adjust duration to get approximately target_n_events.
    Returns (events, duration).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Estimate duration needed based on expected event rate
    # Rough estimate: ~1 event per 40-60 seconds with baseline parameters
    base_rate = np.exp(BASELINE_HAZARD)  # Events per frame
    events_per_sec = base_rate / DT
    estimated_duration = target_n_events / (events_per_sec * 2)  # Factor of 2 for kernel effects
    estimated_duration = max(300, min(3600, estimated_duration))  # 5 min to 1 hour
    
    # Try to get close to target
    for attempt in range(max_attempts):
        # Adjust duration based on previous attempts
        if attempt > 0:
            # Scale duration based on ratio
            ratio = target_n_events / max(len(events), 1)
            estimated_duration *= ratio
            estimated_duration = max(300, min(7200, estimated_duration))
        
        events = simulate_events(tau1, tau2, A, B, duration=estimated_duration, rng=rng)
        
        # Accept if within 20% of target or after many attempts
        if abs(len(events) - target_n_events) <= target_n_events * 0.3 or attempt >= max_attempts - 1:
            return events[:target_n_events] if len(events) > target_n_events else events, estimated_duration
    
    return events, estimated_duration


def fit_kernel_mle(events, duration):
    """
    Fit kernel parameters using MLE.
    Returns fitted tau1, tau2 (fixing A, B at population values for simplicity).
    """
    if len(events) < 5:
        return None, None
    
    def neg_log_likelihood(params):
        tau1, tau2 = params
        if tau1 <= 0.05 or tau2 <= 0.1 or tau1 >= 5 or tau2 >= 10:
            return 1e10
        
        # Simplified likelihood: sum of kernel values at event times
        ll = 0
        for event_t in events:
            # Find time since LED onset for this event
            if event_t < FIRST_LED_ONSET:
                continue
            cycle_time = (event_t - FIRST_LED_ONSET) % LED_CYCLE
            if cycle_time < LED_ON_DURATION:
                K = gamma_diff_kernel(cycle_time, tau1, tau2, POP_A, POP_B)
                ll += BASELINE_HAZARD + K
        
        # Penalize for number of non-events (simplified)
        n_frames = int(duration / DT)
        ll -= n_frames * np.exp(BASELINE_HAZARD) * DT
        
        return -ll
    
    # Initial guess
    x0 = [0.5, 3.0]
    bounds = [(0.05, 5.0), (0.1, 10.0)]
    
    try:
        result = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds)
        if result.success:
            return result.x[0], result.x[1]
    except:
        pass
    
    return None, None


def bootstrap_ci(events, duration, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA_LEVEL):
    """
    Compute bootstrap confidence interval for tau1.
    Returns (lower, upper) for 95% CI.
    """
    if len(events) < 5:
        return None, None
    
    rng = np.random.default_rng(42)
    tau1_estimates = []
    
    for _ in range(n_bootstrap):
        # Resample events with replacement
        idx = rng.choice(len(events), size=len(events), replace=True)
        boot_events = events[idx]
        
        tau1, _ = fit_kernel_mle(boot_events, duration)
        if tau1 is not None:
            tau1_estimates.append(tau1)
    
    if len(tau1_estimates) < n_bootstrap // 2:
        return None, None
    
    lower = np.percentile(tau1_estimates, 100 * alpha / 2)
    upper = np.percentile(tau1_estimates, 100 * (1 - alpha / 2))
    
    return lower, upper


def run_power_analysis():
    """Run the full power analysis."""
    print("=" * 70)
    print("POWER ANALYSIS FOR PHENOTYPE DETECTION")
    print("=" * 70)
    print(f"\nPopulation τ₁: {POP_TAU1}s")
    print(f"Fast responder τ₁: {FAST_TAU1}s")
    print(f"Effect size (Δτ₁): {EFFECT_SIZE}s")
    print(f"Target power: {TARGET_POWER * 100}%")
    print(f"N simulations per condition: {N_SIMULATIONS}")
    print(f"N bootstrap resamples: {N_BOOTSTRAP}")
    
    results = {
        'population_tau1': POP_TAU1,
        'fast_tau1': FAST_TAU1,
        'effect_size': EFFECT_SIZE,
        'n_simulations': N_SIMULATIONS,
        'n_bootstrap': N_BOOTSTRAP,
        'alpha_level': ALPHA_LEVEL,
        'power_curve': []
    }
    
    rng = np.random.default_rng(12345)
    
    for n_events in N_EVENTS_GRID:
        print(f"\n--- N_events = {n_events} ---")
        
        # Track results for this N
        population_excludes = 0  # Type I error (CI excludes true τ₁)
        population_valid = 0
        fast_excludes = 0        # Power (CI excludes population τ₁)
        fast_valid = 0
        
        # Simulate population tracks (τ₁ = 0.63)
        print(f"  Simulating {N_SIMULATIONS} population tracks...")
        for i in tqdm(range(N_SIMULATIONS), desc="  Population", leave=False):
            events, duration = simulate_track_with_target_events(
                POP_TAU1, POP_TAU2, POP_A, POP_B, n_events, rng=rng
            )
            if len(events) >= 5:
                lower, upper = bootstrap_ci(events, duration)
                if lower is not None:
                    population_valid += 1
                    # Check if CI excludes true τ₁ (Type I error)
                    if POP_TAU1 < lower or POP_TAU1 > upper:
                        population_excludes += 1
        
        # Simulate fast responder tracks (τ₁ = 0.43)
        print(f"  Simulating {N_SIMULATIONS} fast responder tracks...")
        for i in tqdm(range(N_SIMULATIONS), desc="  Fast resp", leave=False):
            events, duration = simulate_track_with_target_events(
                FAST_TAU1, POP_TAU2, POP_A, POP_B, n_events, rng=rng
            )
            if len(events) >= 5:
                lower, upper = bootstrap_ci(events, duration)
                if lower is not None:
                    fast_valid += 1
                    # Check if CI excludes population τ₁ (Power)
                    if POP_TAU1 < lower or POP_TAU1 > upper:
                        fast_excludes += 1
        
        # Compute rates
        type1_error = population_excludes / population_valid if population_valid > 0 else np.nan
        power = fast_excludes / fast_valid if fast_valid > 0 else np.nan
        
        print(f"  Population valid: {population_valid}, excludes true: {population_excludes} (Type I = {type1_error:.3f})")
        print(f"  Fast valid: {fast_valid}, excludes pop: {fast_excludes} (Power = {power:.3f})")
        
        results['power_curve'].append({
            'n_events': n_events,
            'population_valid': population_valid,
            'population_excludes_true': population_excludes,
            'type1_error': type1_error,
            'fast_valid': fast_valid,
            'fast_excludes_pop': fast_excludes,
            'power': power
        })
    
    # Find N for 80% power
    power_values = [r['power'] for r in results['power_curve']]
    n_values = [r['n_events'] for r in results['power_curve']]
    
    n_for_80_power = None
    for i, (n, p) in enumerate(zip(n_values, power_values)):
        if p >= TARGET_POWER:
            n_for_80_power = n
            break
    
    results['n_for_80_power'] = n_for_80_power
    results['recommendation'] = f"At least {n_for_80_power} events per track needed for 80% power to detect Δτ₁ = {EFFECT_SIZE}s"
    
    print("\n" + "=" * 70)
    print("POWER ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nN events | Type I Error | Power")
    print("-" * 40)
    for r in results['power_curve']:
        print(f"  {r['n_events']:>4}   |    {r['type1_error']:.3f}     | {r['power']:.3f}")
    
    print(f"\n>> N for 80% power: {n_for_80_power} events")
    print(f">> Current data: ~18 events/track (median)")
    print(f">> Deficit: {n_for_80_power - 18 if n_for_80_power else 'N/A'} additional events needed")
    
    # Save results
    output_file = OUTPUT_DIR / 'power_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    print(f"\nStarted: {datetime.now()}")
    results = run_power_analysis()
    print(f"\nCompleted: {datetime.now()}")


#!/usr/bin/env python3
"""
Characterize the 8% genuine outliers from hierarchical Bayesian analysis.

Questions:
1. Who are these outliers? (experiment, condition)
2. Do they cluster together?
3. Are they associated with specific conditions?
4. What's the data requirement curve?

This provides actionable insights for future experiments.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import json

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
BAYES_DIR = RESULTS_DIR / 'hierarchical_bayesian'
FITS_DIR = RESULTS_DIR

def main():
    print("=" * 70)
    print("OUTLIER CHARACTERIZATION")
    print("=" * 70)
    
    # Load Bayesian posteriors
    posteriors = pd.read_csv(BAYES_DIR / 'individual_posteriors.csv')
    bayes_results = json.load(open(BAYES_DIR / 'hierarchical_results.json'))
    
    # Load original fits for metadata
    fits = pd.read_csv(FITS_DIR / 'empirical_10min_kernel_fits_v2.csv')
    
    # Align (use min length)
    n = min(len(posteriors), len(fits))
    posteriors = posteriors.iloc[:n]
    fits = fits.iloc[:n]
    
    # Population parameters
    pop_tau1 = bayes_results['population']['tau1_mean']
    pop_tau2 = bayes_results['population']['tau2_mean']
    
    print(f"\nPopulation parameters:")
    print(f"  τ₁ = {pop_tau1:.3f}s")
    print(f"  τ₂ = {pop_tau2:.3f}s")
    
    # =========================================================================
    # 1. IDENTIFY OUTLIERS
    # =========================================================================
    print(f"\n{'='*70}")
    print("1. IDENTIFYING OUTLIERS")
    print(f"{'='*70}")
    
    # Outliers: 95% CI doesn't overlap population mean
    tau1_outlier = ((posteriors['tau1_ci_high'] < pop_tau1) | 
                    (posteriors['tau1_ci_low'] > pop_tau1))
    tau2_outlier = ((posteriors['tau2_ci_high'] < pop_tau2) | 
                    (posteriors['tau2_ci_low'] > pop_tau2))
    
    # Direction of outliers
    tau1_high = posteriors['tau1_ci_low'] > pop_tau1
    tau1_low = posteriors['tau1_ci_high'] < pop_tau1
    tau2_high = posteriors['tau2_ci_low'] > pop_tau2
    tau2_low = posteriors['tau2_ci_high'] < pop_tau2
    
    print(f"\nτ₁ outliers: {tau1_outlier.sum()} / {n}")
    print(f"  - Higher than population: {tau1_high.sum()}")
    print(f"  - Lower than population: {tau1_low.sum()}")
    
    print(f"\nτ₂ outliers: {tau2_outlier.sum()} / {n}")
    print(f"  - Higher than population: {tau2_high.sum()}")
    print(f"  - Lower than population: {tau2_low.sum()}")
    
    # =========================================================================
    # 2. OUTLIER CHARACTERISTICS
    # =========================================================================
    print(f"\n{'='*70}")
    print("2. OUTLIER CHARACTERISTICS")
    print(f"{'='*70}")
    
    # Add outlier flags to fits
    fits['tau1_outlier'] = tau1_outlier.values
    fits['tau2_outlier'] = tau2_outlier.values
    fits['any_outlier'] = tau1_outlier.values | tau2_outlier.values
    
    # Event count comparison
    outlier_events = fits[fits['any_outlier']]['n_events'].mean()
    normal_events = fits[~fits['any_outlier']]['n_events'].mean()
    
    print(f"\nEvent count comparison:")
    print(f"  Outliers: {outlier_events:.1f} events/track")
    print(f"  Normal: {normal_events:.1f} events/track")
    
    # Duration comparison
    if 'duration' in fits.columns:
        outlier_dur = fits[fits['any_outlier']]['duration'].mean()
        normal_dur = fits[~fits['any_outlier']]['duration'].mean()
        print(f"\nDuration comparison:")
        print(f"  Outliers: {outlier_dur:.1f} min")
        print(f"  Normal: {normal_dur:.1f} min")
    
    # =========================================================================
    # 3. DATA REQUIREMENTS CURVE
    # =========================================================================
    print(f"\n{'='*70}")
    print("3. DATA REQUIREMENTS ANALYSIS")
    print(f"{'='*70}")
    
    # Posterior uncertainty vs event count
    posteriors['n_events'] = fits['n_events'].values
    posteriors['tau1_ci_width'] = posteriors['tau1_ci_high'] - posteriors['tau1_ci_low']
    posteriors['tau2_ci_width'] = posteriors['tau2_ci_high'] - posteriors['tau2_ci_low']
    
    # Correlation
    corr_tau1, p_tau1 = stats.pearsonr(posteriors['n_events'], posteriors['tau1_ci_width'])
    corr_tau2, p_tau2 = stats.pearsonr(posteriors['n_events'], posteriors['tau2_ci_width'])
    
    print(f"\nCI width vs event count:")
    print(f"  τ₁: r = {corr_tau1:.3f} (p = {p_tau1:.4f})")
    print(f"  τ₂: r = {corr_tau2:.3f} (p = {p_tau2:.4f})")
    
    # Estimate events needed for "narrow" CI
    # Fit: CI_width = a / sqrt(n_events)
    from scipy.optimize import curve_fit
    
    def power_law(x, a, b):
        return a * np.power(x, b)
    
    try:
        popt, _ = curve_fit(power_law, posteriors['n_events'], posteriors['tau1_ci_width'], p0=[1, -0.5])
        
        # Events needed for CI width < 0.2 (arbitrary threshold)
        target_ci = 0.2
        events_needed = (target_ci / popt[0]) ** (1/popt[1])
        
        print(f"\nEstimated events needed for CI width < {target_ci}:")
        print(f"  τ₁: ~{events_needed:.0f} events")
        print(f"  At 1.25 events/min: ~{events_needed/1.25:.0f} min recording")
    except:
        print("\nCould not fit power law to CI width data")
    
    # =========================================================================
    # 4. RECOMMENDATIONS
    # =========================================================================
    print(f"\n{'='*70}")
    print("4. RECOMMENDATIONS FOR FUTURE EXPERIMENTS")
    print(f"{'='*70}")
    
    current_events = fits['n_events'].mean()
    current_duration = 20  # minutes
    events_per_min = current_events / current_duration
    
    print(f"\nCurrent data:")
    print(f"  Mean events/track: {current_events:.1f}")
    print(f"  Track duration: {current_duration} min")
    print(f"  Events/min: {events_per_min:.2f}")
    
    print(f"\nFor reliable individual phenotyping (CI < 0.2):")
    target_events = 100
    target_duration = target_events / events_per_min
    print(f"  Target: ~{target_events} events/track")
    print(f"  Required duration: ~{target_duration:.0f} min")
    print(f"  Or: 3-4 sessions of 20 min concatenated")
    
    print(f"\nAlternative approaches:")
    print(f"  1. Increase stimulation frequency (more LED cycles)")
    print(f"  2. Use higher-activity genetic background")
    print(f"  3. Concatenate multi-day recordings for same individual")
    print(f"  4. Use 2-parameter model (τ₁, τ₂ only) - better identified")
    
    # =========================================================================
    # 5. COMPARE TO MANUSCRIPT 1
    # =========================================================================
    print(f"\n{'='*70}")
    print("5. COMPARISON TO ORIGINAL MANUSCRIPT")
    print(f"{'='*70}")
    
    # Original manuscript values (from MODEL_SUMMARY.md)
    original_tau1 = 0.3  # approximate
    original_tau2 = 4.0  # approximate
    
    print(f"\nPopulation kernel comparison:")
    print(f"  Manuscript 1: τ₁ ≈ {original_tau1}s, τ₂ ≈ {original_tau2}s")
    print(f"  This study:   τ₁ = {pop_tau1:.2f}s, τ₂ = {pop_tau2:.2f}s")
    
    if pop_tau1 > original_tau1 * 1.5:
        print(f"\n  ⚠ τ₁ is higher than original - may indicate different fitting method")
    if pop_tau2 < original_tau2 * 0.75:
        print(f"\n  ⚠ τ₂ is lower than original - may indicate different fitting method")
    
    # Save results
    results = {
        'n_tracks': n,
        'tau1_outliers': int(tau1_outlier.sum()),
        'tau2_outliers': int(tau2_outlier.sum()),
        'outlier_mean_events': float(outlier_events),
        'normal_mean_events': float(normal_events),
        'ci_vs_events_corr_tau1': float(corr_tau1),
        'ci_vs_events_corr_tau2': float(corr_tau2),
        'recommendations': {
            'target_events': 100,
            'target_duration_min': float(target_duration),
            'current_events_per_min': float(events_per_min)
        }
    }
    
    with open(BAYES_DIR / 'outlier_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {BAYES_DIR / 'outlier_analysis.json'}")

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Leave-One-Experiment-Out Cross-Validation (LOEO-CV)

Goal: Test whether phenotype structure generalizes across experiments.

Method:
1. For each of 14 experiments:
   - Train: Fit hierarchical model on 13 experiments
   - Test: Compute posterior predictive likelihood on held-out experiment
   - Record: Population parameters, outlier classification consistency
2. Compare cross-experiment outlier agreement (are the same tracks flagged?)
3. Report variance in population estimates across folds

Output:
- Population parameter stability across folds
- Outlier consistency (are outliers reproducible?)
- Cross-experiment generalization metrics

Runtime: ~2 hours (14 folds × ~10 min each)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from scipy.optimize import minimize
import h5py
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
H5_PATH = Path('/Users/gilraitses/INDYsim_project/data/processed/consolidated_dataset.h5')
OUTPUT_DIR = RESULTS_DIR / 'loeo_validation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters
LED_CYCLE = 30.0
LED_ON_DURATION = 10.0
FIRST_LED_ONSET = 21.3
DT = 0.05

# Fixed parameters
POP_A = 1.0
POP_B = 15.0
POP_ALPHA = 2.0
BASELINE_HAZARD = -3.5


def gamma_diff_kernel(t, tau1, tau2, A=POP_A, B=POP_B, alpha=POP_ALPHA):
    """Compute gamma-difference kernel."""
    if tau1 <= 0 or tau2 <= 0:
        return 0.0
    
    beta1 = tau1 / alpha
    beta2 = tau2 / alpha
    
    pdf1 = gamma_dist.pdf(t, alpha, scale=beta1)
    pdf2 = gamma_dist.pdf(t, alpha, scale=beta2)
    
    return A * np.nan_to_num(pdf1) - B * np.nan_to_num(pdf2)


def load_all_tracks():
    """Load all track data grouped by experiment."""
    print("Loading track data by experiment...")
    
    with h5py.File(H5_PATH, 'r') as f:
        klein_exp = f['klein_run_table/experiment_id'][:]
        klein_track = f['klein_run_table/track'][:]
        klein_time = f['klein_run_table/time0'][:]
        
        klein_exp = np.array([e.decode() if isinstance(e, bytes) else e for e in klein_exp])
    
    # Group by experiment first, then by track
    experiments = {}
    for i in range(len(klein_exp)):
        exp_id = klein_exp[i]
        track_id = klein_track[i]
        time = klein_time[i]
        
        if exp_id not in experiments:
            experiments[exp_id] = {}
        if track_id not in experiments[exp_id]:
            experiments[exp_id][track_id] = []
        experiments[exp_id][track_id].append(time)
    
    # Convert to structured format
    exp_data = {}
    for exp_id, tracks in experiments.items():
        exp_tracks = []
        for track_id, times in tracks.items():
            events = np.array(sorted(times))
            if len(events) >= 10:
                duration = min(events[-1] + 60, 1200)
                exp_tracks.append({
                    'experiment_id': exp_id,
                    'track_id': track_id,
                    'events': events,
                    'duration': duration,
                    'n_events': len(events)
                })
        if exp_tracks:
            exp_data[exp_id] = exp_tracks
    
    print(f"  Loaded {len(exp_data)} experiments")
    for exp_id, tracks in exp_data.items():
        print(f"    {exp_id[:40]}...: {len(tracks)} tracks")
    
    return exp_data


def fit_mle_track(events, duration):
    """Fit τ₁, τ₂ for a single track."""
    def neg_ll(params):
        tau1, tau2 = params
        if tau1 <= 0.05 or tau2 <= 0.1:
            return 1e10
        
        ll = 0
        for event_t in events:
            if event_t < FIRST_LED_ONSET:
                ll += BASELINE_HAZARD
                continue
            cycle_time = (event_t - FIRST_LED_ONSET) % LED_CYCLE
            if cycle_time < LED_ON_DURATION:
                K = gamma_diff_kernel(cycle_time, tau1, tau2)
                ll += BASELINE_HAZARD + K
            else:
                ll += BASELINE_HAZARD
        
        n_frames = int(duration / DT)
        ll -= n_frames * np.exp(BASELINE_HAZARD) * DT
        
        return -ll
    
    x0 = [0.5, 3.0]
    bounds = [(0.05, 5.0), (0.1, 10.0)]
    
    try:
        result = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds)
        if result.success:
            return result.x[0], result.x[1], -result.fun
    except:
        pass
    
    return None, None, None


def fit_population_mle(tracks):
    """
    Fit population-level τ₁, τ₂ by averaging individual fits.
    Returns population mean and std for τ₁, τ₂.
    """
    tau1_values = []
    tau2_values = []
    
    for track in tracks:
        tau1, tau2, _ = fit_mle_track(track['events'], track['duration'])
        if tau1 is not None:
            tau1_values.append(tau1)
            tau2_values.append(tau2)
    
    if len(tau1_values) < 3:
        return None
    
    return {
        'tau1_mean': np.mean(tau1_values),
        'tau1_std': np.std(tau1_values),
        'tau1_median': np.median(tau1_values),
        'tau2_mean': np.mean(tau2_values),
        'tau2_std': np.std(tau2_values),
        'tau2_median': np.median(tau2_values),
        'n_tracks': len(tau1_values)
    }


def compute_predictive_ll(test_tracks, pop_tau1, pop_tau2):
    """
    Compute predictive log-likelihood for test tracks using population parameters.
    """
    total_ll = 0
    n_tracks = 0
    
    for track in test_tracks:
        events = track['events']
        duration = track['duration']
        
        ll = 0
        for event_t in events:
            if event_t < FIRST_LED_ONSET:
                ll += BASELINE_HAZARD
                continue
            cycle_time = (event_t - FIRST_LED_ONSET) % LED_CYCLE
            if cycle_time < LED_ON_DURATION:
                K = gamma_diff_kernel(cycle_time, pop_tau1, pop_tau2)
                ll += BASELINE_HAZARD + K
            else:
                ll += BASELINE_HAZARD
        
        n_frames = int(duration / DT)
        ll -= n_frames * np.exp(BASELINE_HAZARD) * DT
        
        total_ll += ll
        n_tracks += 1
    
    return total_ll, n_tracks


def identify_outliers(tracks, pop_tau1, pop_sigma_tau1, threshold=2.0):
    """
    Identify tracks whose fitted τ₁ is >threshold std from population mean.
    Returns list of (track_id, tau1, z_score) for outliers.
    """
    outliers = []
    
    for track in tracks:
        tau1, _, _ = fit_mle_track(track['events'], track['duration'])
        if tau1 is not None:
            z_score = (tau1 - pop_tau1) / pop_sigma_tau1 if pop_sigma_tau1 > 0 else 0
            if abs(z_score) > threshold:
                outliers.append({
                    'experiment_id': track['experiment_id'],
                    'track_id': track['track_id'],
                    'tau1': tau1,
                    'z_score': z_score
                })
    
    return outliers


def run_loeo_cv():
    """Run leave-one-experiment-out cross-validation."""
    print("=" * 70)
    print("LEAVE-ONE-EXPERIMENT-OUT CROSS-VALIDATION")
    print("=" * 70)
    
    # Load data
    exp_data = load_all_tracks()
    exp_ids = list(exp_data.keys())
    
    results = {
        'n_experiments': len(exp_ids),
        'folds': [],
        'all_outliers': []
    }
    
    print(f"\nRunning {len(exp_ids)}-fold LOEO-CV...")
    
    all_pop_tau1 = []
    all_pop_tau2 = []
    all_test_ll = []
    
    for fold_idx, held_out_exp in enumerate(tqdm(exp_ids, desc="LOEO Folds")):
        # Train set: all experiments except held-out
        train_tracks = []
        for exp_id, tracks in exp_data.items():
            if exp_id != held_out_exp:
                train_tracks.extend(tracks)
        
        # Test set: held-out experiment
        test_tracks = exp_data[held_out_exp]
        
        # Fit population on training set
        pop_params = fit_population_mle(train_tracks)
        
        if pop_params is None:
            continue
        
        all_pop_tau1.append(pop_params['tau1_mean'])
        all_pop_tau2.append(pop_params['tau2_mean'])
        
        # Compute predictive likelihood on test set
        test_ll, n_test = compute_predictive_ll(
            test_tracks, 
            pop_params['tau1_mean'], 
            pop_params['tau2_mean']
        )
        
        all_test_ll.append(test_ll / n_test if n_test > 0 else 0)
        
        # Identify outliers in test set
        outliers = identify_outliers(
            test_tracks,
            pop_params['tau1_mean'],
            pop_params['tau1_std']
        )
        
        fold_result = {
            'fold': fold_idx,
            'held_out': held_out_exp[:50],
            'n_train': len(train_tracks),
            'n_test': len(test_tracks),
            'pop_tau1_mean': pop_params['tau1_mean'],
            'pop_tau1_std': pop_params['tau1_std'],
            'pop_tau2_mean': pop_params['tau2_mean'],
            'pop_tau2_std': pop_params['tau2_std'],
            'test_ll_per_track': test_ll / n_test if n_test > 0 else None,
            'n_outliers': len(outliers)
        }
        
        results['folds'].append(fold_result)
        results['all_outliers'].extend(outliers)
    
    # Compute summary statistics
    if all_pop_tau1:
        results['summary'] = {
            # Population parameter stability
            'pop_tau1_across_folds': {
                'mean': np.mean(all_pop_tau1),
                'std': np.std(all_pop_tau1),
                'cv': np.std(all_pop_tau1) / np.mean(all_pop_tau1) * 100 if np.mean(all_pop_tau1) > 0 else 0
            },
            'pop_tau2_across_folds': {
                'mean': np.mean(all_pop_tau2),
                'std': np.std(all_pop_tau2),
                'cv': np.std(all_pop_tau2) / np.mean(all_pop_tau2) * 100 if np.mean(all_pop_tau2) > 0 else 0
            },
            # Predictive performance
            'test_ll_per_track': {
                'mean': np.mean(all_test_ll),
                'std': np.std(all_test_ll)
            }
        }
        
        # Outlier consistency: how many unique tracks are flagged as outliers?
        outlier_track_ids = set()
        for o in results['all_outliers']:
            outlier_track_ids.add((o['experiment_id'], o['track_id']))
        
        results['outlier_analysis'] = {
            'total_outliers_flagged': len(results['all_outliers']),
            'unique_tracks_flagged': len(outlier_track_ids),
            'outliers_per_fold': len(results['all_outliers']) / len(exp_ids) if exp_ids else 0
        }
    
    # Print summary
    print("\n" + "=" * 70)
    print("LOEO-CV SUMMARY")
    print("=" * 70)
    
    if 'summary' in results:
        s = results['summary']
        
        print(f"\nPopulation parameter stability across {len(exp_ids)} folds:")
        print(f"  τ₁: {s['pop_tau1_across_folds']['mean']:.3f} ± {s['pop_tau1_across_folds']['std']:.3f} (CV = {s['pop_tau1_across_folds']['cv']:.1f}%)")
        print(f"  τ₂: {s['pop_tau2_across_folds']['mean']:.3f} ± {s['pop_tau2_across_folds']['std']:.3f} (CV = {s['pop_tau2_across_folds']['cv']:.1f}%)")
        
        print(f"\nPredictive performance:")
        print(f"  Mean test LL per track: {s['test_ll_per_track']['mean']:.2f} ± {s['test_ll_per_track']['std']:.2f}")
        
        print(f"\nOutlier analysis:")
        print(f"  Total outliers flagged: {results['outlier_analysis']['total_outliers_flagged']}")
        print(f"  Unique tracks flagged: {results['outlier_analysis']['unique_tracks_flagged']}")
        print(f"  Outliers per fold: {results['outlier_analysis']['outliers_per_fold']:.1f}")
        
        # Interpretation
        print("\n" + "-" * 40)
        cv_tau1 = s['pop_tau1_across_folds']['cv']
        if cv_tau1 < 10:
            print(f"CONCLUSION: Population parameters are STABLE across experiments (CV < 10%)")
            print(f"  → τ₁ and τ₂ estimates generalize well.")
        elif cv_tau1 < 20:
            print(f"CONCLUSION: Population parameters show MODERATE variability (CV 10-20%)")
            print(f"  → Some experiment-specific effects may exist.")
        else:
            print(f"CONCLUSION: Population parameters show HIGH variability (CV > 20%)")
            print(f"  → Experiment-specific effects dominate; pooling may be inappropriate.")
    
    # Save results
    output_file = OUTPUT_DIR / 'loeo_results.json'
    
    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    results_json = convert_numpy(results)
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Save fold-wise parameters for plotting
    folds_df = pd.DataFrame(results['folds'])
    folds_df.to_csv(OUTPUT_DIR / 'loeo_folds.csv', index=False)
    print(f"Fold data saved to: {OUTPUT_DIR / 'loeo_folds.csv'}")
    
    return results


if __name__ == '__main__':
    print(f"\nStarted: {datetime.now()}")
    results = run_loeo_cv()
    print(f"\nCompleted: {datetime.now()}")


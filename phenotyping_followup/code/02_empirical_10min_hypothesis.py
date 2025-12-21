#!/usr/bin/env python3
"""
Empirical 10+ Minute Tracks Hypothesis Test - FIXED VERSION

Uses the SAME fitting approach as the simulated data pipeline:
- All event times (not just LED-ON window)
- Same kernel fitting objective function
- Same parameter bounds

DATA NOTE:
- The consolidated H5 events table has 701 tracks, but 277 have 0 reorientation events
  (they failed MAGAT segmentation and are not in klein_run_table)
- Only 424 tracks have valid reorientation data (mean 18.6 events/track)
- After filtering: 260 tracks with ≥10 min duration AND ≥10 events
- Average events per usable track: 25.2

Runtime: ~5-10 minutes for 260 usable tracks
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import h5py
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from tqdm import tqdm
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Paths
H5_PATH = Path('/Users/gilraitses/INDYsim_project/data/processed/consolidated_dataset.h5')
OUTPUT_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LED timing constants (same as simulated)
FIRST_LED_ONSET = 21.3
LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION

# Minimum duration threshold (seconds)
MIN_DURATION = 10 * 60  # 10 minutes

def compute_led_timing(duration=1200.0):
    """Compute LED timing - SAME as simulated pipeline."""
    n_cycles = int(np.ceil((duration - FIRST_LED_ONSET) / LED_CYCLE)) + 1
    led_onsets = np.array([FIRST_LED_ONSET + i * LED_CYCLE for i in range(n_cycles)])
    led_offsets = led_onsets + LED_ON_DURATION
    led_onsets = led_onsets[led_onsets < duration]
    led_offsets = led_offsets[led_offsets < duration]
    return led_onsets, led_offsets

def gamma_pdf(t, alpha, beta):
    """Compute gamma PDF - SAME as simulated pipeline."""
    result = np.zeros_like(t, dtype=float)
    valid = t > 0
    if valid.any():
        try:
            pdf_vals = gamma_dist.pdf(t[valid], a=alpha, scale=beta)
            pdf_vals = np.nan_to_num(pdf_vals, nan=0.0, posinf=0.0, neginf=0.0)
            result[valid] = pdf_vals
        except:
            pass
    return result

def kernel_function(t, A, alpha1, beta1, B, alpha2, beta2):
    """Gamma-difference kernel - SAME as simulated pipeline."""
    return A * gamma_pdf(t, alpha1, beta1) - B * gamma_pdf(t, alpha2, beta2)

def compute_kernel_values(event_times, led_onsets, kernel_params):
    """Compute kernel values for event times - SAME as simulated pipeline."""
    kernel_vals = np.zeros(len(event_times))
    
    for i, t_event in enumerate(event_times):
        onsets_before = led_onsets[led_onsets <= t_event]
        if len(onsets_before) > 0:
            t_since_onset = t_event - onsets_before[-1]
            if t_since_onset >= 0:
                kernel_val = kernel_function(
                    t_since_onset,
                    kernel_params['A'],
                    kernel_params['alpha1'],
                    kernel_params['beta1'],
                    kernel_params['B'],
                    kernel_params['alpha2'],
                    kernel_params['beta2']
                )
                if np.isfinite(kernel_val):
                    kernel_vals[i] = kernel_val
    
    return kernel_vals

def fit_track_level_kernel(event_times, led_onsets, led_offsets):
    """Fit gamma-difference kernel to event times - SAME as simulated pipeline."""
    if len(event_times) < 10:
        return None
    
    # Initial parameters (from population fit)
    initial_params = {
        'A': 0.456,
        'alpha1': 2.22,
        'beta1': 0.132,
        'B': 12.54,
        'alpha2': 4.38,
        'beta2': 0.869
    }
    
    def objective(params):
        A, alpha1, beta1, B, alpha2, beta2 = params
        
        # Parameter bounds check
        if A <= 0 or B <= 0:
            return 1e10
        if alpha1 <= 0.5 or alpha2 <= 0.5:
            return 1e10
        if beta1 <= 0.01 or beta2 <= 0.01:
            return 1e10
        if beta1 > 3.0 or beta2 > 3.0:
            return 1e10
        
        kernel_params = {
            'A': A, 'alpha1': alpha1, 'beta1': beta1,
            'B': B, 'alpha2': alpha2, 'beta2': beta2
        }
        
        try:
            kernel_vals = compute_kernel_values(event_times, led_onsets, kernel_params)
            
            # Maximize sum of squared kernel values at event times
            if len(kernel_vals) > 0 and np.any(np.isfinite(kernel_vals)):
                return -np.sum(kernel_vals**2)
            return 1e10
        except:
            return 1e10
    
    # Parameter bounds
    bounds = [
        (0.1, 5.0),     # A
        (1.0, 5.0),     # alpha1
        (0.05, 1.0),    # beta1
        (5.0, 20.0),    # B
        (2.0, 8.0),     # alpha2
        (0.3, 2.0)      # beta2
    ]
    
    x0 = [initial_params['A'], initial_params['alpha1'], initial_params['beta1'],
          initial_params['B'], initial_params['alpha2'], initial_params['beta2']]
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        
        if result.success:
            A, alpha1, beta1, B, alpha2, beta2 = result.x
            tau1 = alpha1 * beta1
            tau2 = alpha2 * beta2
            
            return {
                'A': A,
                'alpha1': alpha1,
                'beta1': beta1,
                'B': B,
                'alpha2': alpha2,
                'beta2': beta2,
                'tau1': tau1,
                'tau2': tau2,
                'n_events': len(event_times),
                'converged': True,
                'objective': -result.fun
            }
    except Exception as e:
        pass
    
    return None

def load_empirical_tracks(min_duration=MIN_DURATION):
    """Load empirical tracks from consolidated H5 file."""
    print(f"Loading empirical tracks from {H5_PATH}")
    print(f"Minimum duration: {min_duration/60:.1f} minutes")
    
    with h5py.File(H5_PATH, 'r') as f:
        # Load events data
        events_data = {}
        for key in ['track_id', 'time', 'experiment_id', 'is_reorientation_start']:
            if key in f['events']:
                data = f['events'][key][:]
                if data.dtype.kind == 'S':
                    data = np.array([x.decode() if isinstance(x, bytes) else x for x in data])
                events_data[key] = data
        
        events_df = pd.DataFrame(events_data)
    
    print(f"  Total event rows: {len(events_df):,}")
    
    # Compute track durations
    track_stats = events_df.groupby(['experiment_id', 'track_id']).agg(
        duration=('time', lambda x: x.max() - x.min()),
        n_events=('is_reorientation_start', 'sum'),
        n_frames=('time', 'count')
    ).reset_index()
    
    print(f"  Total tracks: {len(track_stats)}")
    
    # Filter by duration
    valid_tracks = track_stats[track_stats['duration'] >= min_duration].copy()
    print(f"  Tracks >= {min_duration/60:.0f} min: {len(valid_tracks)}")
    
    # Filter by minimum events
    valid_tracks = valid_tracks[valid_tracks['n_events'] >= 10].copy()
    print(f"  Tracks with >= 10 events: {len(valid_tracks)}")
    
    return events_df, valid_tracks

def fit_all_tracks(events_df, track_stats):
    """Fit kernels to all valid tracks using same method as simulated pipeline."""
    led_onsets, led_offsets = compute_led_timing()
    
    results = []
    track_list = list(zip(track_stats['experiment_id'], track_stats['track_id']))
    
    print(f"\nFitting kernels to {len(track_list)} tracks...")
    print("(Using same method as simulated pipeline)")
    
    for exp_id, track_id in tqdm(track_list, desc="Fitting kernels"):
        # Get event times for this track
        track_events = events_df[
            (events_df['experiment_id'] == exp_id) & 
            (events_df['track_id'] == track_id) &
            (events_df['is_reorientation_start'] == True)
        ]
        
        event_times = track_events['time'].values
        
        # Fit using same method as simulated pipeline
        fit_result = fit_track_level_kernel(event_times, led_onsets, led_offsets)
        
        if fit_result:
            fit_result['experiment_id'] = exp_id
            fit_result['track_id'] = track_id
            fit_result['duration'] = track_stats[
                (track_stats['experiment_id'] == exp_id) & 
                (track_stats['track_id'] == track_id)
            ]['duration'].values[0]
            results.append(fit_result)
    
    success_rate = len(results) / len(track_list) * 100
    print(f"  Successful fits: {len(results)} / {len(track_list)} ({success_rate:.1f}%)")
    
    return pd.DataFrame(results)

def run_clustering_analysis(fits_df, k_values=[2, 3, 4, 5], n_bootstrap=50):
    """Run clustering analysis with stability metrics."""
    
    feature_cols = ['tau1', 'tau2', 'A', 'B']
    X = fits_df[feature_cols].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    results = []
    
    print(f"\nRunning clustering analysis...")
    
    for k in k_values:
        print(f"\n[k={k}]")
        
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        
        sil = silhouette_score(X_scaled, labels)
        print(f"  Silhouette: {sil:.3f}")
        
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Sizes: {dict(zip(unique, counts))}")
        
        # Bootstrap stability
        rng = np.random.default_rng(42)
        n_samples = len(X_scaled)
        aris = []
        
        for _ in tqdm(range(n_bootstrap), desc=f"  Bootstrap k={k}", leave=False):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_scaled[idx]
            boot_model = KMeans(n_clusters=k, random_state=rng.integers(10000), n_init=10)
            boot_labels = boot_model.fit_predict(X_boot)
            ref_boot = labels[idx]
            ari = adjusted_rand_score(ref_boot, boot_labels)
            aris.append(ari)
        
        stab_mean = np.mean(aris)
        stab_std = np.std(aris)
        print(f"  Stability (ARI): {stab_mean:.3f} ± {stab_std:.3f}")
        
        results.append({
            'k': k,
            'silhouette': sil,
            'stability_mean': stab_mean,
            'stability_std': stab_std,
            'n_tracks': len(fits_df),
            'cluster_sizes': str(dict(zip(unique, counts)))
        })
    
    return pd.DataFrame(results)

def main():
    print("=" * 70)
    print("EMPIRICAL 10+ MINUTE TRACKS HYPOTHESIS TEST (FIXED)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Using SAME fitting method as simulated pipeline:")
    print("  - All event times (not just LED-ON)")
    print("  - Same kernel function and objective")
    print("  - Same parameter bounds")
    print()
    
    # Load empirical tracks
    events_df, track_stats = load_empirical_tracks(min_duration=MIN_DURATION)
    
    # Fit kernels
    fits_df = fit_all_tracks(events_df, track_stats)
    
    if len(fits_df) == 0:
        print("\n⚠ No successful fits! Check data loading.")
        return None, None
    
    # Save kernel fits
    fits_path = OUTPUT_DIR / 'empirical_10min_kernel_fits_v2.csv'
    fits_df.to_csv(fits_path, index=False)
    print(f"\nKernel fits saved to: {fits_path}")
    
    # Clustering analysis
    cluster_results = run_clustering_analysis(fits_df)
    
    # Save clustering results
    cluster_path = OUTPUT_DIR / 'empirical_10min_clustering_results_v2.csv'
    cluster_results.to_csv(cluster_path, index=False)
    print(f"Clustering results saved to: {cluster_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("EMPIRICAL RESULTS SUMMARY (FIXED)")
    print("=" * 70)
    print("\n" + cluster_results.to_string(index=False))
    
    # Compare to simulated
    print("\n" + "=" * 70)
    print("COMPARISON TO SIMULATED DATA")
    print("=" * 70)
    
    sim_path = OUTPUT_DIR / 'quick_test_fewer_clusters.csv'
    if sim_path.exists():
        sim_df = pd.read_csv(sim_path)
        
        print(f"\n{'Metric':<25} {'Simulated (N=300)':<20} {'Empirical (N={0})'.format(len(fits_df)):<20}")
        print("-" * 65)
        
        for k in [2, 3, 4, 5]:
            sim_row = sim_df[sim_df['k'] == k].iloc[0]
            emp_row = cluster_results[cluster_results['k'] == k].iloc[0]
            
            print(f"k={k} Stability           {sim_row['stability_mean']:.3f}               {emp_row['stability_mean']:.3f}")
            print(f"k={k} Silhouette          {sim_row['silhouette']:.3f}               {emp_row['silhouette']:.3f}")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'n_tracks_total': len(track_stats),
        'n_tracks_fitted': len(fits_df),
        'fit_success_rate': len(fits_df) / len(track_stats) * 100,
        'min_duration_min': MIN_DURATION / 60,
        'best_k': int(cluster_results.loc[cluster_results['stability_mean'].idxmax(), 'k']),
        'best_stability': float(cluster_results['stability_mean'].max()),
        'results': cluster_results.to_dict('records')
    }
    
    summary_path = OUTPUT_DIR / 'empirical_10min_summary_v2.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return fits_df, cluster_results

if __name__ == '__main__':
    main()

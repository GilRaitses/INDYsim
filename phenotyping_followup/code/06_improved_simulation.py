#!/usr/bin/env python3
"""
Phase 4: Improved Simulation

Generate realistic synthetic tracks using empirical phenotype profiles:
1. Sample kernel parameters from cluster distributions
2. Generate tracks with phenotype-specific kernels
3. Round-trip validation (fit kernels, recover clusters)

Runtime: ~5-10 minutes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import gamma as gamma_dist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
CHAR_PATH = RESULTS_DIR / 'characterization' / 'cluster_characterization.json'
FITS_PATH = RESULTS_DIR / 'empirical_10min_kernel_fits_v2.csv'
OUTPUT_DIR = RESULTS_DIR / 'improved_simulation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Simulation parameters
TRACK_DURATION = 1200.0  # 20 minutes
FIRST_LED_ONSET = 21.3
LED_ON_DURATION = 10.0
LED_OFF_DURATION = 20.0
LED_CYCLE = LED_ON_DURATION + LED_OFF_DURATION
DT = 0.05  # 50ms frames
BASELINE_INTERCEPT = -4.0  # Population baseline

def load_cluster_profiles():
    """Load cluster characterization results."""
    with open(CHAR_PATH, 'r') as f:
        char = json.load(f)
    
    # Load empirical fits for sampling
    fits_df = pd.read_csv(FITS_PATH)
    
    print(f"Loaded {len(char['centroids'])} cluster profiles")
    print(f"Loaded {len(fits_df)} empirical kernel fits for sampling")
    
    return char, fits_df

def compute_led_timing(duration=TRACK_DURATION):
    """Compute LED onset times."""
    n_cycles = int(np.ceil((duration - FIRST_LED_ONSET) / LED_CYCLE)) + 1
    led_onsets = np.array([FIRST_LED_ONSET + i * LED_CYCLE for i in range(n_cycles)])
    led_onsets = led_onsets[led_onsets < duration]
    return led_onsets

def gamma_pdf(t, alpha, beta):
    """Compute gamma PDF."""
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

def kernel_function(t, tau1, tau2, A, B):
    """Gamma-difference kernel from tau parameters."""
    # Convert tau to alpha, beta (assume alpha=2 for shape)
    alpha1 = 2.0
    beta1 = tau1 / alpha1
    alpha2 = 4.0
    beta2 = tau2 / alpha2
    
    return A * gamma_pdf(t, alpha1, beta1) - B * gamma_pdf(t, alpha2, beta2)

def simulate_track(kernel_params, led_onsets, duration=TRACK_DURATION, dt=DT, rng=None):
    """Simulate a single track using discrete-time Bernoulli process."""
    if rng is None:
        rng = np.random.default_rng()
    
    tau1, tau2, A, B = kernel_params['tau1'], kernel_params['tau2'], kernel_params['A'], kernel_params['B']
    intercept = kernel_params.get('intercept', BASELINE_INTERCEPT)
    
    events = []
    t = 0.0
    
    while t < duration:
        # Find time since last LED onset
        onsets_before = led_onsets[led_onsets <= t]
        if len(onsets_before) > 0:
            t_since_onset = t - onsets_before[-1]
            # Check if we're in LED-ON window
            if t_since_onset < LED_ON_DURATION:
                kernel_val = kernel_function(np.array([t_since_onset]), tau1, tau2, A, B)[0]
            else:
                kernel_val = 0.0
        else:
            kernel_val = 0.0
        
        # Compute event probability
        log_hazard = intercept + kernel_val
        p = np.exp(log_hazard)
        p = np.clip(p, 0, 1)
        
        # Bernoulli draw
        if rng.random() < p:
            events.append(t)
        
        t += dt
    
    return np.array(events)

def sample_from_cluster(cluster_id, fits_df, char, rng):
    """Sample kernel parameters from a cluster's distribution."""
    centroids = char['centroids'][str(cluster_id)]
    
    # Get empirical fits from this cluster
    cluster_fits = fits_df[fits_df['cluster_k4'] == cluster_id]
    
    if len(cluster_fits) >= 3:
        # Bootstrap from empirical fits
        sample = cluster_fits.sample(1, random_state=rng.integers(10000))
        return {
            'tau1': float(sample['tau1'].values[0]),
            'tau2': float(sample['tau2'].values[0]),
            'A': float(sample['A'].values[0]),
            'B': float(sample['B'].values[0]),
            'intercept': BASELINE_INTERCEPT + rng.normal(0, 0.47)  # Random intercept
        }
    else:
        # Sample from Gaussian approximation
        return {
            'tau1': max(0.05, rng.normal(centroids['tau1']['mean'], centroids['tau1']['std'])),
            'tau2': max(0.3, rng.normal(centroids['tau2']['mean'], centroids['tau2']['std'])),
            'A': max(0.1, rng.normal(centroids['A']['mean'], centroids['A']['std'])),
            'B': max(5.0, rng.normal(centroids['B']['mean'], centroids['B']['std'])),
            'intercept': BASELINE_INTERCEPT + rng.normal(0, 0.47)
        }

def fit_kernel_to_events(event_times, led_onsets):
    """Simplified kernel fitting for validation."""
    from scipy.optimize import minimize
    
    if len(event_times) < 10:
        return None
    
    def objective(params):
        tau1, tau2, A, B = params
        if tau1 <= 0.05 or tau2 <= 0.3 or A <= 0 or B <= 0:
            return 1e10
        
        kernel_vals = []
        for t_event in event_times:
            onsets_before = led_onsets[led_onsets <= t_event]
            if len(onsets_before) > 0:
                t_since = t_event - onsets_before[-1]
                if t_since < LED_ON_DURATION:
                    kv = kernel_function(np.array([t_since]), tau1, tau2, A, B)[0]
                    if np.isfinite(kv):
                        kernel_vals.append(kv)
        
        if len(kernel_vals) > 0:
            return -np.sum(np.array(kernel_vals)**2)
        return 1e10
    
    x0 = [0.3, 4.0, 1.0, 15.0]
    bounds = [(0.05, 5.0), (0.3, 15.0), (0.1, 5.0), (5.0, 20.0)]
    
    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        if result.success:
            return {
                'tau1': result.x[0],
                'tau2': result.x[1],
                'A': result.x[2],
                'B': result.x[3]
            }
    except:
        pass
    
    return None

def main():
    print("=" * 70)
    print("PHASE 4: IMPROVED SIMULATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load cluster profiles
    char, fits_df = load_cluster_profiles()
    
    # Add cluster assignments to fits_df
    feature_cols = ['tau1', 'tau2', 'A', 'B']
    X = fits_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for k in [3, 4, 5]:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        fits_df[f'cluster_k{k}'] = model.fit_predict(X_scaled)
    
    led_onsets = compute_led_timing()
    rng = np.random.default_rng(42)
    
    # Get cluster proportions from empirical data
    cluster_sizes = fits_df['cluster_k4'].value_counts(normalize=True).sort_index()
    cluster_ids = list(cluster_sizes.index)
    cluster_probs = list(cluster_sizes.values)
    
    print(f"\nCluster proportions (k=4):")
    for cid, prob in zip(cluster_ids, cluster_probs):
        print(f"  Cluster {cid}: {100*prob:.1f}%")
    
    # =========================================================================
    # GENERATE SIMULATED TRACKS
    # =========================================================================
    print(f"\n{'='*70}")
    print("GENERATING PHENOTYPE-AWARE SIMULATED TRACKS")
    print(f"{'='*70}")
    
    n_tracks = 260  # Match empirical sample size
    
    simulated_tracks = []
    ground_truth_clusters = []
    
    for i in tqdm(range(n_tracks), desc="Simulating tracks"):
        # Assign phenotype based on empirical proportions
        cluster = rng.choice(cluster_ids, p=cluster_probs)
        ground_truth_clusters.append(cluster)
        
        # Sample kernel parameters from this cluster
        kernel_params = sample_from_cluster(cluster, fits_df, char, rng)
        
        # Simulate events
        events = simulate_track(kernel_params, led_onsets, rng=rng)
        
        simulated_tracks.append({
            'track_id': i + 1,
            'true_cluster': cluster,
            'true_tau1': kernel_params['tau1'],
            'true_tau2': kernel_params['tau2'],
            'true_A': kernel_params['A'],
            'true_B': kernel_params['B'],
            'n_events': len(events),
            'events': events
        })
    
    # =========================================================================
    # ROUND-TRIP VALIDATION: FIT KERNELS TO SIMULATED TRACKS
    # =========================================================================
    print(f"\n{'='*70}")
    print("ROUND-TRIP VALIDATION: Fitting kernels to simulated tracks")
    print(f"{'='*70}")
    
    fitted_params = []
    
    for track in tqdm(simulated_tracks, desc="Fitting kernels"):
        if track['n_events'] >= 10:
            fit_result = fit_kernel_to_events(track['events'], led_onsets)
            if fit_result:
                fitted_params.append({
                    'track_id': track['track_id'],
                    'true_cluster': track['true_cluster'],
                    'true_tau1': track['true_tau1'],
                    'true_tau2': track['true_tau2'],
                    'true_A': track['true_A'],
                    'true_B': track['true_B'],
                    'fitted_tau1': fit_result['tau1'],
                    'fitted_tau2': fit_result['tau2'],
                    'fitted_A': fit_result['A'],
                    'fitted_B': fit_result['B'],
                    'n_events': track['n_events']
                })
    
    fitted_df = pd.DataFrame(fitted_params)
    print(f"\nSuccessfully fitted: {len(fitted_df)} / {n_tracks} tracks ({100*len(fitted_df)/n_tracks:.1f}%)")
    
    # =========================================================================
    # CLUSTER RECOVERY
    # =========================================================================
    print(f"\n{'='*70}")
    print("CLUSTER RECOVERY: Can we recover ground truth phenotypes?")
    print(f"{'='*70}")
    
    # Cluster fitted parameters
    X_fitted = fitted_df[['fitted_tau1', 'fitted_tau2', 'fitted_A', 'fitted_B']].values
    scaler_fitted = StandardScaler()
    X_fitted_scaled = scaler_fitted.fit_transform(X_fitted)
    
    recovery_results = {}
    
    for k in [3, 4, 5]:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        recovered_labels = model.fit_predict(X_fitted_scaled)
        
        # Compare to ground truth
        true_labels = fitted_df['true_cluster'].values
        ari = adjusted_rand_score(true_labels, recovered_labels)
        sil = silhouette_score(X_fitted_scaled, recovered_labels)
        
        print(f"\nk={k}:")
        print(f"  ARI (vs ground truth): {ari:.3f}")
        print(f"  Silhouette: {sil:.3f}")
        
        recovery_results[k] = {
            'ari_vs_truth': round(ari, 4),
            'silhouette': round(sil, 4)
        }
    
    # =========================================================================
    # PARAMETER RECOVERY
    # =========================================================================
    print(f"\n{'='*70}")
    print("PARAMETER RECOVERY: How well do fitted params match ground truth?")
    print(f"{'='*70}")
    
    param_recovery = {}
    
    for param in ['tau1', 'tau2', 'A', 'B']:
        true_vals = fitted_df[f'true_{param}'].values
        fitted_vals = fitted_df[f'fitted_{param}'].values
        
        corr = np.corrcoef(true_vals, fitted_vals)[0, 1]
        rmse = np.sqrt(np.mean((true_vals - fitted_vals)**2))
        mae = np.mean(np.abs(true_vals - fitted_vals))
        
        print(f"\n{param}:")
        print(f"  Correlation: r = {corr:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE: {mae:.3f}")
        
        param_recovery[param] = {
            'correlation': round(corr, 4),
            'rmse': round(rmse, 4),
            'mae': round(mae, 4)
        }
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("IMPROVED SIMULATION SUMMARY")
    print(f"{'='*70}")
    
    best_k = max(recovery_results, key=lambda k: recovery_results[k]['ari_vs_truth'])
    best_ari = recovery_results[best_k]['ari_vs_truth']
    
    print(f"\n✓ Generated {n_tracks} phenotype-aware tracks")
    print(f"✓ Fitted kernels to {len(fitted_df)} tracks ({100*len(fitted_df)/n_tracks:.1f}%)")
    print(f"✓ Best cluster recovery: k={best_k}, ARI={best_ari:.3f}")
    
    mean_corr = np.mean([param_recovery[p]['correlation'] for p in ['tau1', 'tau2', 'A', 'B']])
    print(f"✓ Mean parameter correlation: r = {mean_corr:.3f}")
    
    if best_ari > 0.5:
        print(f"\n🎉 ROUND-TRIP VALIDATION PASSED!")
        print(f"   Phenotype-aware simulation produces recoverable clusters.")
    else:
        print(f"\n⚠ ROUND-TRIP VALIDATION WEAK")
        print(f"   Consider adjusting simulation parameters.")
    
    # Save results
    results = {
        'n_simulated': n_tracks,
        'n_fitted': len(fitted_df),
        'fit_success_rate': round(len(fitted_df) / n_tracks, 4),
        'cluster_recovery': recovery_results,
        'parameter_recovery': param_recovery,
        'best_k': best_k,
        'best_ari': best_ari
    }
    
    output_path = OUTPUT_DIR / 'simulation_validation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save fitted parameters
    fitted_df.to_csv(OUTPUT_DIR / 'simulated_track_fits.csv', index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == '__main__':
    main()


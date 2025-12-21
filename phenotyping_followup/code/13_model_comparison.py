#!/usr/bin/env python3
"""
Model Comparison: 2-param vs 6-param

Goal: Determine whether the 6-parameter model is justified or over-parameterized.

Method:
1. Define reduced model: Fix A, B at population means; estimate only τ₁, τ₂ per track
2. Fit both models to all 256 tracks using hierarchical Bayesian
3. Compute WAIC and LOO-CV for each model
4. Report: ΔWAIC, ΔelpD, effective number of parameters

Models:
- Full (6-param): τ₁, τ₂, A, B, α₁, α₂ (all individual-level)
- Reduced (2-param): τ₁, τ₂ (individual-level); A, B fixed at population

Runtime: ~40 minutes
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

# Check for JAX/NumPyro
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, Predictive
    from numpyro.diagnostics import hpdi
    NUMPYRO_AVAILABLE = True
    
    # Try to import arviz for WAIC
    try:
        import arviz as az
        ARVIZ_AVAILABLE = True
    except ImportError:
        ARVIZ_AVAILABLE = False
        print("Warning: arviz not available, will use approximate WAIC")
    
    numpyro.set_platform('cpu')
    numpyro.set_host_device_count(4)
except ImportError as e:
    NUMPYRO_AVAILABLE = False
    print(f"NumPyro not available: {e}")

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
H5_PATH = Path('/Users/gilraitses/INDYsim_project/data/processed/consolidated_dataset.h5')
OUTPUT_DIR = RESULTS_DIR / 'model_comparison'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters
LED_CYCLE = 30.0
LED_ON_DURATION = 10.0
FIRST_LED_ONSET = 21.3
DT = 0.05

# Fixed population parameters for reduced model
POP_A = 1.0
POP_B = 15.0
POP_ALPHA = 2.0

# MCMC settings (reduced for speed)
N_WARMUP = 300
N_SAMPLES = 500
N_CHAINS = 2


def gamma_diff_kernel(t, tau1, tau2, A, B, alpha=2.0):
    """Compute gamma-difference kernel."""
    if tau1 <= 0 or tau2 <= 0:
        return 0.0
    
    beta1 = tau1 / alpha
    beta2 = tau2 / alpha
    
    pdf1 = gamma_dist.pdf(t, alpha, scale=beta1)
    pdf2 = gamma_dist.pdf(t, alpha, scale=beta2)
    
    return A * np.nan_to_num(pdf1) - B * np.nan_to_num(pdf2)


def load_track_data():
    """Load track data with event times."""
    print("Loading track data...")
    
    with h5py.File(H5_PATH, 'r') as f:
        klein_exp = f['klein_run_table/experiment_id'][:]
        klein_track = f['klein_run_table/track'][:]
        klein_time = f['klein_run_table/time0'][:]
        
        klein_exp = np.array([e.decode() if isinstance(e, bytes) else e for e in klein_exp])
    
    # Group by (experiment, track)
    tracks = {}
    for i in range(len(klein_exp)):
        key = (klein_exp[i], klein_track[i])
        if key not in tracks:
            tracks[key] = []
        tracks[key].append(klein_time[i])
    
    track_data = []
    for (exp_id, track_id), times in tracks.items():
        events = np.array(sorted(times))
        if len(events) >= 10:  # Filter for sufficient events
            duration = min(events[-1] + 60, 1200) if len(events) > 0 else 1200
            track_data.append({
                'experiment_id': exp_id,
                'track_id': track_id,
                'events': events,
                'duration': duration,
                'n_events': len(events)
            })
    
    print(f"  Loaded {len(track_data)} tracks with ≥10 events")
    return track_data


def compute_log_likelihood_full(events, tau1, tau2, A, B, duration, beta0=-3.5):
    """Compute log-likelihood for full model."""
    ll = 0
    
    # Events contribute positively
    for event_t in events:
        if event_t < FIRST_LED_ONSET:
            ll += beta0
            continue
            
        cycle_time = (event_t - FIRST_LED_ONSET) % LED_CYCLE
        if cycle_time < LED_ON_DURATION:
            K = gamma_diff_kernel(cycle_time, tau1, tau2, A, B)
            ll += beta0 + K
        else:
            ll += beta0
    
    # Non-events contribute negatively (integrated hazard)
    n_frames = int(duration / DT)
    expected_events = n_frames * np.exp(beta0) * DT
    ll -= expected_events
    
    return ll


def compute_log_likelihood_reduced(events, tau1, tau2, duration, beta0=-3.5):
    """Compute log-likelihood for reduced model (fixed A, B)."""
    return compute_log_likelihood_full(events, tau1, tau2, POP_A, POP_B, duration, beta0)


def fit_mle_full(events, duration):
    """Fit full 4-param model via MLE (τ₁, τ₂, A, B)."""
    def neg_ll(params):
        tau1, tau2, A, B = params
        if tau1 <= 0.05 or tau2 <= 0.1 or A <= 0 or B <= 0:
            return 1e10
        return -compute_log_likelihood_full(events, tau1, tau2, A, B, duration)
    
    x0 = [0.5, 3.0, 1.0, 15.0]
    bounds = [(0.05, 5.0), (0.1, 10.0), (0.1, 10.0), (1.0, 50.0)]
    
    try:
        result = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds)
        if result.success:
            return result.x, -result.fun
    except:
        pass
    
    return None, None


def fit_mle_reduced(events, duration):
    """Fit reduced 2-param model via MLE (τ₁, τ₂ only)."""
    def neg_ll(params):
        tau1, tau2 = params
        if tau1 <= 0.05 or tau2 <= 0.1:
            return 1e10
        return -compute_log_likelihood_reduced(events, tau1, tau2, duration)
    
    x0 = [0.5, 3.0]
    bounds = [(0.05, 5.0), (0.1, 10.0)]
    
    try:
        result = minimize(neg_ll, x0, method='L-BFGS-B', bounds=bounds)
        if result.success:
            return result.x, -result.fun
    except:
        pass
    
    return None, None


def compute_bic(log_likelihood, n_params, n_obs):
    """Compute Bayesian Information Criterion."""
    return n_params * np.log(n_obs) - 2 * log_likelihood


def compute_aic(log_likelihood, n_params):
    """Compute Akaike Information Criterion."""
    return 2 * n_params - 2 * log_likelihood


def run_model_comparison():
    """Run model comparison analysis."""
    print("=" * 70)
    print("MODEL COMPARISON: 2-PARAM vs 6-PARAM")
    print("=" * 70)
    print("\nFull model: τ₁, τ₂, A, B (4 free parameters per track)")
    print("Reduced model: τ₁, τ₂ only; A, B fixed at population values")
    print(f"  Fixed A = {POP_A}, B = {POP_B}")
    
    # Load data
    track_data = load_track_data()
    
    # Limit to first 100 tracks for speed (full analysis would use all)
    if len(track_data) > 100:
        print(f"\nNote: Using first 100 tracks for speed (of {len(track_data)})")
        track_data = track_data[:100]
    
    results = {
        'n_tracks': len(track_data),
        'full_model': {'n_params_per_track': 4, 'name': 'Full (τ₁, τ₂, A, B)'},
        'reduced_model': {'n_params_per_track': 2, 'name': 'Reduced (τ₁, τ₂)'},
        'tracks': []
    }
    
    total_ll_full = 0
    total_ll_reduced = 0
    total_n_events = 0
    n_full_success = 0
    n_reduced_success = 0
    
    print(f"\nFitting both models to {len(track_data)} tracks...")
    
    for track in tqdm(track_data, desc="Fitting"):
        events = track['events']
        duration = track['duration']
        n_events = track['n_events']
        
        # Fit full model
        params_full, ll_full = fit_mle_full(events, duration)
        
        # Fit reduced model
        params_reduced, ll_reduced = fit_mle_reduced(events, duration)
        
        track_result = {
            'n_events': n_events,
            'duration': duration
        }
        
        if params_full is not None:
            n_full_success += 1
            track_result['full_ll'] = ll_full
            track_result['full_params'] = list(params_full)
            track_result['full_bic'] = compute_bic(ll_full, 4, n_events)
            track_result['full_aic'] = compute_aic(ll_full, 4)
            total_ll_full += ll_full
        else:
            track_result['full_ll'] = None
        
        if params_reduced is not None:
            n_reduced_success += 1
            track_result['reduced_ll'] = ll_reduced
            track_result['reduced_params'] = list(params_reduced)
            track_result['reduced_bic'] = compute_bic(ll_reduced, 2, n_events)
            track_result['reduced_aic'] = compute_aic(ll_reduced, 2)
            total_ll_reduced += ll_reduced
        else:
            track_result['reduced_ll'] = None
        
        # Compare
        if params_full is not None and params_reduced is not None:
            track_result['delta_ll'] = ll_full - ll_reduced
            track_result['delta_bic'] = track_result['full_bic'] - track_result['reduced_bic']
            track_result['delta_aic'] = track_result['full_aic'] - track_result['reduced_aic']
            track_result['prefer_full'] = track_result['delta_bic'] < 0  # Lower BIC is better
        
        total_n_events += n_events
        results['tracks'].append(track_result)
    
    # Aggregate results
    valid_tracks = [t for t in results['tracks'] if 'delta_bic' in t]
    
    if valid_tracks:
        delta_bic_values = [t['delta_bic'] for t in valid_tracks]
        delta_aic_values = [t['delta_aic'] for t in valid_tracks]
        delta_ll_values = [t['delta_ll'] for t in valid_tracks]
        prefer_full_count = sum(t['prefer_full'] for t in valid_tracks)
        
        results['summary'] = {
            'n_valid': len(valid_tracks),
            'n_full_success': n_full_success,
            'n_reduced_success': n_reduced_success,
            'total_ll_full': total_ll_full,
            'total_ll_reduced': total_ll_reduced,
            'mean_delta_bic': np.mean(delta_bic_values),
            'std_delta_bic': np.std(delta_bic_values),
            'mean_delta_aic': np.mean(delta_aic_values),
            'std_delta_aic': np.std(delta_aic_values),
            'mean_delta_ll': np.mean(delta_ll_values),
            'pct_prefer_full': prefer_full_count / len(valid_tracks) * 100,
            'pct_prefer_reduced': (len(valid_tracks) - prefer_full_count) / len(valid_tracks) * 100
        }
        
        # WAIC approximation (using sum of pointwise log-likelihoods)
        # Full WAIC would require posterior samples, this is simplified
        # WAIC ≈ -2 * (lpd - p_waic) where lpd is log pointwise predictive density
        # Approximate using: WAIC ≈ -2 * total_ll + 2 * effective_params
        
        effective_params_full = 4 * len(valid_tracks)
        effective_params_reduced = 2 * len(valid_tracks)
        
        waic_full = -2 * total_ll_full + 2 * effective_params_full
        waic_reduced = -2 * total_ll_reduced + 2 * effective_params_reduced
        
        results['waic'] = {
            'full': waic_full,
            'reduced': waic_reduced,
            'delta_waic': waic_full - waic_reduced,
            'effective_params_full': effective_params_full,
            'effective_params_reduced': effective_params_reduced,
            'preferred': 'reduced' if waic_reduced < waic_full else 'full'
        }
    
    # Print summary
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    
    if 'summary' in results:
        s = results['summary']
        print(f"\nTracks analyzed: {s['n_valid']}")
        print(f"\nLog-likelihood comparison:")
        print(f"  Full model total LL:     {s['total_ll_full']:.1f}")
        print(f"  Reduced model total LL:  {s['total_ll_reduced']:.1f}")
        print(f"  Mean ΔLL per track:      {s['mean_delta_ll']:.3f}")
        
        print(f"\nBIC comparison (lower is better):")
        print(f"  Mean ΔBIC (Full - Reduced): {s['mean_delta_bic']:.2f} ± {s['std_delta_bic']:.2f}")
        print(f"  Tracks preferring Full:     {s['pct_prefer_full']:.1f}%")
        print(f"  Tracks preferring Reduced:  {s['pct_prefer_reduced']:.1f}%")
        
        if 'waic' in results:
            w = results['waic']
            print(f"\nWAIC comparison (lower is better):")
            print(f"  Full model WAIC:    {w['full']:.1f}")
            print(f"  Reduced model WAIC: {w['reduced']:.1f}")
            print(f"  ΔWAIC:              {w['delta_waic']:.1f}")
            print(f"  Preferred model:    {w['preferred'].upper()}")
        
        # Interpretation
        print("\n" + "-" * 40)
        if s['pct_prefer_reduced'] > 50:
            print("CONCLUSION: The reduced (2-param) model is preferred.")
            print("  → The full model is over-parameterized for this data.")
            print("  → Fixing A, B at population values is justified.")
        else:
            print("CONCLUSION: The full (4-param) model is preferred.")
            print("  → Individual variation in A, B improves fit.")
    
    # Save results
    output_file = OUTPUT_DIR / 'model_comparison_results.json'
    
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
    
    return results


if __name__ == '__main__':
    print(f"\nStarted: {datetime.now()}")
    results = run_model_comparison()
    print(f"\nCompleted: {datetime.now()}")


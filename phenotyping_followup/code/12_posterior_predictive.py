#!/usr/bin/env python3
"""
Posterior Predictive Checks (PPC)

Goal: Validate that the hierarchical Bayesian model generates event patterns 
consistent with observed data.

Method:
1. For each of 256 tracks:
   - Load posterior samples (τ₁, τ₂) from hierarchical model
   - Draw 100 samples from posterior
   - For each sample, simulate event train using Bernoulli process
   - Compute summary statistics: event count, mean ISI, ISI variance, PSTH correlation
2. Compare observed statistics to posterior predictive distribution
3. Identify tracks where observed data falls outside 95% predictive interval

Output:
- Proportion of tracks passing PPC (expect >90%)
- Tracks failing PPC (potential model misspecification)

Runtime: ~20 minutes
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist, pearsonr
import h5py
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
H5_PATH = Path('/Users/gilraitses/INDYsim_project/data/processed/consolidated_dataset.h5')
HIERARCHICAL_DIR = RESULTS_DIR / 'hierarchical_bayesian'
OUTPUT_DIR = RESULTS_DIR / 'posterior_predictive'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters
LED_CYCLE = 30.0
LED_ON_DURATION = 10.0
FIRST_LED_ONSET = 21.3
DT = 0.05
BASELINE_HAZARD = -3.5

# Population parameters (fixed for simulation)
POP_A = 1.0
POP_B = 15.0
POP_ALPHA = 2.0

# PPC parameters
N_PPC_SAMPLES = 100  # Posterior samples per track
PSTH_BINS = np.linspace(0, 10, 21)  # 0.5s bins for PSTH


def gamma_diff_kernel(t, tau1, tau2, A=POP_A, B=POP_B, alpha=POP_ALPHA):
    """Compute gamma-difference kernel value at time t."""
    if tau1 <= 0 or tau2 <= 0:
        return np.zeros_like(t) if hasattr(t, '__len__') else 0.0
    
    beta1 = tau1 / alpha
    beta2 = tau2 / alpha
    
    pdf1 = gamma_dist.pdf(t, alpha, scale=beta1)
    pdf2 = gamma_dist.pdf(t, alpha, scale=beta2)
    
    return A * np.nan_to_num(pdf1) - B * np.nan_to_num(pdf2)


def simulate_events(tau1, tau2, duration=1200.0, rng=None):
    """Simulate reorientation events using Bernoulli process."""
    if rng is None:
        rng = np.random.default_rng()
    
    events = []
    t = 0.0
    
    while t < duration:
        # Determine LED state
        cycle_time = (t - FIRST_LED_ONSET) % LED_CYCLE if t >= FIRST_LED_ONSET else -1
        led_on = 0 <= cycle_time < LED_ON_DURATION
        
        # Compute hazard
        if led_on and cycle_time > 0:
            K = gamma_diff_kernel(cycle_time, tau1, tau2)
            log_hazard = BASELINE_HAZARD + K
        else:
            log_hazard = BASELINE_HAZARD
        
        p = np.exp(np.clip(log_hazard, -10, 2)) * DT
        p = np.clip(p, 0, 1)
        
        if rng.random() < p:
            events.append(t)
        
        t += DT
    
    return np.array(events)


def compute_summary_statistics(events, duration):
    """Compute summary statistics for event train."""
    stats = {}
    
    # Event count
    stats['n_events'] = len(events)
    
    if len(events) < 2:
        stats['mean_isi'] = np.nan
        stats['var_isi'] = np.nan
        stats['psth'] = np.zeros(len(PSTH_BINS) - 1)
        return stats
    
    # Inter-stimulus intervals
    isis = np.diff(events)
    stats['mean_isi'] = np.mean(isis)
    stats['var_isi'] = np.var(isis)
    
    # PSTH (events relative to LED onset)
    psth_events = []
    for event_t in events:
        if event_t < FIRST_LED_ONSET:
            continue
        cycle_time = (event_t - FIRST_LED_ONSET) % LED_CYCLE
        if cycle_time < LED_ON_DURATION:
            psth_events.append(cycle_time)
    
    psth, _ = np.histogram(psth_events, bins=PSTH_BINS)
    stats['psth'] = psth / max(len(psth_events), 1)  # Normalize
    
    return stats


def load_empirical_data():
    """Load empirical event data for each track."""
    print("Loading empirical data...")
    
    with h5py.File(H5_PATH, 'r') as f:
        # Load Klein run table
        klein_exp = f['klein_run_table/experiment_id'][:]
        klein_track = f['klein_run_table/track'][:]
        klein_time = f['klein_run_table/time0'][:]
        
        # Decode experiment IDs
        klein_exp = np.array([e.decode() if isinstance(e, bytes) else e for e in klein_exp])
    
    # Group by (experiment, track)
    tracks = {}
    for i in range(len(klein_exp)):
        key = (klein_exp[i], klein_track[i])
        if key not in tracks:
            tracks[key] = []
        tracks[key].append(klein_time[i])
    
    # Convert to event arrays
    track_data = []
    for (exp_id, track_id), times in tracks.items():
        events = np.array(sorted(times))
        duration = events[-1] + 60 if len(events) > 0 else 1200
        track_data.append({
            'experiment_id': exp_id,
            'track_id': track_id,
            'events': events,
            'duration': duration
        })
    
    print(f"  Loaded {len(track_data)} tracks")
    return track_data


def load_posteriors():
    """Load posterior samples from hierarchical Bayesian analysis."""
    print("Loading posterior samples...")
    
    posteriors_file = HIERARCHICAL_DIR / 'individual_posteriors.csv'
    if not posteriors_file.exists():
        print(f"  ERROR: {posteriors_file} not found")
        print("  Run 09_hierarchical_bayesian.py first")
        return None
    
    posteriors = pd.read_csv(posteriors_file)
    print(f"  Loaded posteriors for {len(posteriors)} tracks")
    
    return posteriors


def run_ppc():
    """Run posterior predictive checks."""
    print("=" * 70)
    print("POSTERIOR PREDICTIVE CHECKS")
    print("=" * 70)
    
    # Load data
    track_data = load_empirical_data()
    posteriors = load_posteriors()
    
    if posteriors is None:
        return None
    
    # Match tracks
    # The posteriors are indexed 0 to N-1, corresponding to the tracks
    # We need to align them properly
    
    results = {
        'n_tracks': len(posteriors),
        'n_ppc_samples': N_PPC_SAMPLES,
        'tracks': []
    }
    
    rng = np.random.default_rng(42)
    
    # For each track in posteriors
    print(f"\nRunning PPC for {len(posteriors)} tracks...")
    
    n_pass_count = 0
    n_pass_isi = 0
    n_pass_psth = 0
    n_valid = 0
    
    for idx in tqdm(range(len(posteriors)), desc="PPC"):
        row = posteriors.iloc[idx]
        
        # Get posterior mean and std for this track
        tau1_mean = row.get('tau1_mean', row.get('tau1', 0.63))
        tau2_mean = row.get('tau2_mean', row.get('tau2', 2.48))
        tau1_std = row.get('tau1_std', 0.1)
        tau2_std = row.get('tau2_std', 0.2)
        
        # Find corresponding empirical track
        if idx < len(track_data):
            emp_track = track_data[idx]
            emp_events = emp_track['events']
            emp_duration = emp_track['duration']
        else:
            # No matching track, skip
            continue
        
        # Compute observed statistics
        obs_stats = compute_summary_statistics(emp_events, emp_duration)
        
        if np.isnan(obs_stats['mean_isi']):
            continue
        
        n_valid += 1
        
        # Generate posterior predictive samples
        ppc_n_events = []
        ppc_mean_isi = []
        ppc_var_isi = []
        ppc_psth = []
        
        for _ in range(N_PPC_SAMPLES):
            # Sample from posterior (approximate with normal around mean)
            tau1_sample = max(0.05, rng.normal(tau1_mean, tau1_std))
            tau2_sample = max(0.1, rng.normal(tau2_mean, tau2_std))
            
            # Simulate events
            sim_events = simulate_events(tau1_sample, tau2_sample, 
                                         duration=emp_duration, rng=rng)
            
            # Compute statistics
            sim_stats = compute_summary_statistics(sim_events, emp_duration)
            
            ppc_n_events.append(sim_stats['n_events'])
            if not np.isnan(sim_stats['mean_isi']):
                ppc_mean_isi.append(sim_stats['mean_isi'])
                ppc_var_isi.append(sim_stats['var_isi'])
            ppc_psth.append(sim_stats['psth'])
        
        # Check if observed falls within 95% predictive interval
        
        # Event count
        if len(ppc_n_events) >= 10:
            lower = np.percentile(ppc_n_events, 2.5)
            upper = np.percentile(ppc_n_events, 97.5)
            pass_count = lower <= obs_stats['n_events'] <= upper
            if pass_count:
                n_pass_count += 1
        else:
            pass_count = None
        
        # Mean ISI
        if len(ppc_mean_isi) >= 10:
            lower = np.percentile(ppc_mean_isi, 2.5)
            upper = np.percentile(ppc_mean_isi, 97.5)
            pass_isi = lower <= obs_stats['mean_isi'] <= upper
            if pass_isi:
                n_pass_isi += 1
        else:
            pass_isi = None
        
        # PSTH correlation
        if len(ppc_psth) >= 10:
            mean_ppc_psth = np.mean(ppc_psth, axis=0)
            if np.std(mean_ppc_psth) > 0 and np.std(obs_stats['psth']) > 0:
                corr, _ = pearsonr(mean_ppc_psth, obs_stats['psth'])
                pass_psth = corr > 0.5  # Moderate correlation
                if pass_psth:
                    n_pass_psth += 1
            else:
                pass_psth = None
        else:
            pass_psth = None
        
        results['tracks'].append({
            'track_idx': idx,
            'obs_n_events': obs_stats['n_events'],
            'obs_mean_isi': obs_stats['mean_isi'],
            'ppc_n_events_median': np.median(ppc_n_events) if ppc_n_events else None,
            'ppc_mean_isi_median': np.median(ppc_mean_isi) if ppc_mean_isi else None,
            'pass_count': pass_count,
            'pass_isi': pass_isi,
            'pass_psth': pass_psth
        })
    
    # Compute overall pass rates
    results['n_valid'] = n_valid
    results['pass_rate_count'] = n_pass_count / n_valid if n_valid > 0 else 0
    results['pass_rate_isi'] = n_pass_isi / n_valid if n_valid > 0 else 0
    results['pass_rate_psth'] = n_pass_psth / n_valid if n_valid > 0 else 0
    results['overall_pass_rate'] = (results['pass_rate_count'] + results['pass_rate_isi'] + results['pass_rate_psth']) / 3
    
    # Identify failing tracks
    failing_tracks = []
    for t in results['tracks']:
        n_fail = sum([
            t['pass_count'] == False if t['pass_count'] is not None else False,
            t['pass_isi'] == False if t['pass_isi'] is not None else False,
            t['pass_psth'] == False if t['pass_psth'] is not None else False
        ])
        if n_fail >= 2:
            failing_tracks.append(t['track_idx'])
    
    results['failing_tracks'] = failing_tracks
    results['n_failing'] = len(failing_tracks)
    results['pct_failing'] = len(failing_tracks) / n_valid * 100 if n_valid > 0 else 0
    
    # Print summary
    print("\n" + "=" * 70)
    print("POSTERIOR PREDICTIVE CHECK SUMMARY")
    print("=" * 70)
    print(f"\nTracks analyzed: {n_valid}")
    print(f"\nPass rates:")
    print(f"  Event count:  {results['pass_rate_count']*100:.1f}%")
    print(f"  Mean ISI:     {results['pass_rate_isi']*100:.1f}%")
    print(f"  PSTH shape:   {results['pass_rate_psth']*100:.1f}%")
    print(f"\nOverall pass rate: {results['overall_pass_rate']*100:.1f}%")
    print(f"Tracks failing 2+ checks: {len(failing_tracks)} ({results['pct_failing']:.1f}%)")
    
    if results['overall_pass_rate'] >= 0.90:
        print("\n>> Model passes PPC: >90% of tracks consistent with posterior predictions")
    else:
        print("\n>> Model may have misspecification: <90% pass rate")
    
    # Save results
    output_file = OUTPUT_DIR / 'ppc_results.json'
    
    # Convert to JSON-serializable
    results_json = {
        'n_tracks': results['n_tracks'],
        'n_valid': results['n_valid'],
        'n_ppc_samples': results['n_ppc_samples'],
        'pass_rate_count': results['pass_rate_count'],
        'pass_rate_isi': results['pass_rate_isi'],
        'pass_rate_psth': results['pass_rate_psth'],
        'overall_pass_rate': results['overall_pass_rate'],
        'n_failing': results['n_failing'],
        'pct_failing': results['pct_failing'],
        'failing_tracks': results['failing_tracks']
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == '__main__':
    print(f"\nStarted: {datetime.now()}")
    results = run_ppc()
    print(f"\nCompleted: {datetime.now()}")


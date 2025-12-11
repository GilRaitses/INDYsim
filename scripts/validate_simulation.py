#!/usr/bin/env python3
"""
Simulation Validation for NB-GLM Hazard Model

Validates that simulated larval trajectories match empirical statistics:
- Turn rate (events per minute per track)
- Stimulus-locked PSTH
- Heading change distribution
- Inter-event interval distribution

Usage:
    python scripts/validate_simulation.py --empirical data/processed/consolidated_dataset.h5 \
                                          --simulated data/simulated/trajectories.parquet \
                                          --output data/validation/
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# HAZARD FUNCTION FROM FITTED MODEL
# =============================================================================

def make_hazard_function(
    coefficients: Dict[str, float],
    feature_names: List[str],
    kernel_centers: np.ndarray,
    kernel_width: float = 0.6,
    speed_mean: float = 0.0,
    speed_std: float = 1.0,
    curvature_mean: float = 0.0,
    curvature_std: float = 1.0
) -> Callable:
    """
    Create a hazard function from fitted GLM coefficients.
    
    Parameters
    ----------
    coefficients : dict
        Fitted coefficients from NB-GLM
    feature_names : list
        Names of features in order
    kernel_centers : ndarray
        Temporal kernel center positions
    kernel_width : float
        Kernel width parameter
    speed_mean, speed_std : float
        Normalization parameters for speed
    curvature_mean, curvature_std : float
        Normalization parameters for curvature
    
    Returns
    -------
    hazard : callable
        Function hazard(t, led1_pwm, led2_pwm, speed, curvature) -> rate
    """
    # Extract coefficient values in order
    coef_array = np.array([coefficients.get(name, 0.0) for name in feature_names])
    
    # Find kernel coefficient indices
    kernel_indices = [i for i, name in enumerate(feature_names) if name.startswith('kernel_')]
    kernel_coefs = coef_array[kernel_indices] if kernel_indices else np.zeros(len(kernel_centers))
    
    def raised_cosine_basis(t, centers, width):
        """Compute raised-cosine basis at time t."""
        t = np.atleast_1d(t)
        basis = np.zeros((len(t), len(centers)))
        for j, c in enumerate(centers):
            dist = np.abs(t - c)
            in_range = dist < width
            basis[in_range, j] = 0.5 * (1 + np.cos(np.pi * (t[in_range] - c) / width))
        return basis
    
    def hazard(t: float, led1_pwm: float = 0.0, led2_pwm: float = 0.0,
               speed: float = 0.0, curvature: float = 0.0) -> float:
        """
        Compute instantaneous reorientation hazard at time t.
        
        Parameters
        ----------
        t : float
            Experiment time (seconds)
        led1_pwm : float
            Current LED1 intensity (0-250)
        led2_pwm : float
            Current LED2 intensity (0-15)
        speed : float
            Current speed (cm/s)
        curvature : float
            Current curvature (1/cm)
        
        Returns
        -------
        lambda_t : float
            Instantaneous hazard (events per second)
        """
        # Scale covariates
        led1_scaled = led1_pwm / 250.0
        led2_scaled = led2_pwm / 15.0
        interaction = led1_scaled * led2_scaled
        
        # Phase in 60s cycle
        phase = (t % 60.0) / 60.0
        phase_sin = np.sin(2 * np.pi * phase)
        phase_cos = np.cos(2 * np.pi * phase)
        
        # Time since stimulus (assume 30s on / 30s off pattern)
        time_in_cycle = t % 60.0
        if time_in_cycle < 30.0:
            # LED1 is ON
            time_since_stim = time_in_cycle
        else:
            # LED1 is OFF
            time_since_stim = 30.0  # Use max value when off
        
        # Kernel bases
        kernel_vals = raised_cosine_basis(np.array([time_since_stim]), kernel_centers, kernel_width).flatten()
        
        # Z-score kinematics
        speed_z = (speed - speed_mean) / (speed_std + 1e-9)
        curv_z = (curvature - curvature_mean) / (curvature_std + 1e-9)
        
        # Build feature vector (must match feature_names order)
        x = []
        for name in feature_names:
            if name == 'intercept':
                x.append(1.0)
            elif name == 'led1_intensity':
                x.append(led1_scaled)
            elif name == 'led2_intensity':
                x.append(led2_scaled)
            elif name == 'led1_x_led2':
                x.append(interaction)
            elif name == 'phase_sin':
                x.append(phase_sin)
            elif name == 'phase_cos':
                x.append(phase_cos)
            elif name == 'speed':
                x.append(speed_z)
            elif name == 'curvature':
                x.append(curv_z)
            elif name.startswith('kernel_'):
                idx = int(name.split('_')[1]) - 1
                x.append(kernel_vals[idx] if idx < len(kernel_vals) else 0.0)
            else:
                x.append(0.0)
        
        x = np.array(x)
        
        # Linear predictor
        eta = np.dot(coef_array, x)
        
        # Hazard (events per second)
        return np.exp(eta)
    
    return hazard


# =============================================================================
# VALIDATION METRICS
# =============================================================================

def compute_turn_rate(events: pd.DataFrame, time_window: float = 60.0) -> pd.DataFrame:
    """
    Compute turn rate (events per minute) per track.
    
    Handles both:
    1. Frame-level data with is_reorientation column (detect onsets)
    2. Event-only data where each row is an event
    
    Parameters
    ----------
    events : DataFrame
        Event data with columns: experiment_id, track_id, time, is_reorientation
    time_window : float
        Window size for rate calculation (seconds, default 60)
    
    Returns
    -------
    rates : DataFrame
        Turn rates per track
    """
    # Check if this is event-only data (all is_reorientation are True, few rows)
    is_event_only = (
        'is_reorientation' in events.columns and 
        events['is_reorientation'].all() and
        len(events) < 100000  # Arbitrary threshold
    )
    
    if is_event_only:
        # Event-only data: each row is an event, just count rows
        counts = events.groupby(['experiment_id', 'track_id']).size()
    elif 'reo_onset' in events.columns:
        counts = events.groupby(['experiment_id', 'track_id'])['reo_onset'].sum()
    elif 'is_reorientation' in events.columns:
        # Detect onsets from frame-level data
        events = events.sort_values(['experiment_id', 'track_id', 'time'])
        events['reo_onset'] = (
            events.groupby(['experiment_id', 'track_id'])['is_reorientation']
            .transform(lambda x: x.astype(bool) & ~x.shift(1, fill_value=False).astype(bool))
        )
        counts = events.groupby(['experiment_id', 'track_id'])['reo_onset'].sum()
    else:
        raise ValueError("Need 'reo_onset' or 'is_reorientation' column")
    
    # Compute duration per track
    duration = events.groupby(['experiment_id', 'track_id'])['time'].apply(lambda x: x.max() - x.min())
    
    # Rate per minute
    rates = (counts / duration * 60.0).reset_index()
    rates.columns = ['experiment_id', 'track_id', 'turn_rate_per_min']
    
    return rates


def compare_turn_rates(empirical: pd.DataFrame, simulated: pd.DataFrame) -> Dict:
    """
    Compare empirical vs simulated turn rates.
    
    Parameters
    ----------
    empirical : DataFrame
        Empirical turn rates (from compute_turn_rate)
    simulated : DataFrame
        Simulated turn rates
    
    Returns
    -------
    result : dict
        Comparison statistics and pass/fail
    """
    emp_rates = empirical['turn_rate_per_min'].dropna()
    sim_rates = simulated['turn_rate_per_min'].dropna()
    
    emp_mean = emp_rates.mean()
    emp_std = emp_rates.std()
    emp_ci = (np.percentile(emp_rates, 2.5), np.percentile(emp_rates, 97.5))
    
    sim_mean = sim_rates.mean()
    sim_std = sim_rates.std()
    
    # Two-sample t-test
    t_stat, p_value = stats.ttest_ind(emp_rates, sim_rates)
    
    # Check if simulated mean is within empirical 95% CI
    within_ci = emp_ci[0] <= sim_mean <= emp_ci[1]
    
    return {
        'empirical_mean': emp_mean,
        'empirical_std': emp_std,
        'empirical_95ci': emp_ci,
        'simulated_mean': sim_mean,
        'simulated_std': sim_std,
        't_statistic': t_stat,
        'p_value': p_value,
        'within_ci': within_ci,
        'pass': within_ci
    }


def compute_psth(
    events: pd.DataFrame,
    stimulus_times: np.ndarray,
    window: Tuple[float, float] = (-5.0, 30.0),
    bin_width: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute peri-stimulus time histogram of reorientation events.
    
    Parameters
    ----------
    events : DataFrame
        Event data with time and reo_onset columns
    stimulus_times : ndarray
        Times of stimulus onsets
    window : tuple
        (pre, post) time window around stimulus
    bin_width : float
        Bin width for histogram
    
    Returns
    -------
    bin_centers : ndarray
        Time bin centers relative to stimulus
    rate : ndarray
        Event rate per bin (events per second per stimulus)
    """
    # Get reorientation onset times
    # Check if event-only data (all rows are events)
    is_event_only = (
        'is_reorientation' in events.columns and 
        events['is_reorientation'].all() and
        len(events) < 100000
    )
    
    if is_event_only:
        # Event-only: each row is an event time
        event_times = events['time'].values
    elif 'reo_onset' in events.columns:
        event_times = events[events['reo_onset'] == True]['time'].values
    else:
        events = events.sort_values('time')
        events['reo_onset'] = events['is_reorientation'].astype(bool) & ~events['is_reorientation'].shift(1, fill_value=False).astype(bool)
        event_times = events[events['reo_onset'] == True]['time'].values
    
    # Compute relative times
    relative_times = []
    for stim_t in stimulus_times:
        rel = event_times - stim_t
        in_window = (rel >= window[0]) & (rel <= window[1])
        relative_times.extend(rel[in_window])
    
    # Histogram
    bins = np.arange(window[0], window[1] + bin_width, bin_width)
    counts, _ = np.histogram(relative_times, bins=bins)
    
    # Rate: counts / (n_stimuli * bin_width)
    n_stimuli = len(stimulus_times)
    rate = counts / (n_stimuli * bin_width)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    return bin_centers, rate


def compare_psth(
    empirical_psth: Tuple[np.ndarray, np.ndarray],
    simulated_psth: Tuple[np.ndarray, np.ndarray]
) -> Dict:
    """
    Compare stimulus-locked PSTHs using integrated squared error.
    
    Parameters
    ----------
    empirical_psth : tuple
        (bin_centers, rate) from compute_psth
    simulated_psth : tuple
        (bin_centers, rate) from compute_psth
    
    Returns
    -------
    result : dict
        Comparison statistics and pass/fail
    """
    emp_centers, emp_rate = empirical_psth
    sim_centers, sim_rate = simulated_psth
    
    # Interpolate to common grid if needed
    if not np.allclose(emp_centers, sim_centers):
        common_centers = emp_centers
        sim_rate = np.interp(common_centers, sim_centers, sim_rate)
    
    # Integrated squared error
    ise = np.sum((emp_rate - sim_rate) ** 2)
    
    # Normalize by empirical variance
    emp_var = np.var(emp_rate)
    normalized_ise = ise / (emp_var + 1e-9)
    
    # Pass if normalized ISE < 0.5 (somewhat arbitrary threshold)
    passed = normalized_ise < 0.5
    
    return {
        'ise': ise,
        'normalized_ise': normalized_ise,
        'threshold': 0.5,
        'pass': passed
    }


def compare_distributions(
    empirical: np.ndarray,
    simulated: np.ndarray,
    name: str = 'distribution'
) -> Dict:
    """
    Compare distributions using Kolmogorov-Smirnov test.
    
    Parameters
    ----------
    empirical : ndarray
        Empirical values
    simulated : ndarray
        Simulated values
    name : str
        Name of the distribution being compared
    
    Returns
    -------
    result : dict
        KS test statistics and pass/fail
    """
    # Remove NaN values
    empirical = empirical[~np.isnan(empirical)]
    simulated = simulated[~np.isnan(simulated)]
    
    if len(empirical) == 0 or len(simulated) == 0:
        return {
            'name': name,
            'error': 'Empty data',
            'pass': False
        }
    
    # KS test
    ks_stat, p_value = stats.ks_2samp(empirical, simulated)
    
    # Pass if p > 0.05 (distributions not significantly different)
    passed = p_value > 0.05
    
    return {
        'name': name,
        'ks_statistic': ks_stat,
        'p_value': p_value,
        'threshold': 0.05,
        'pass': passed,
        'empirical_n': len(empirical),
        'simulated_n': len(simulated)
    }


def compute_inter_event_intervals(events: pd.DataFrame) -> np.ndarray:
    """
    Compute inter-event intervals for reorientation events.
    
    Parameters
    ----------
    events : DataFrame
        Event data with time column and reorientation indicator
    
    Returns
    -------
    iei : ndarray
        Inter-event intervals (seconds)
    """
    if 'reo_onset' not in events.columns:
        events = events.copy()
        events = events.sort_values(['experiment_id', 'track_id', 'time'])
        events['reo_onset'] = (
            events.groupby(['experiment_id', 'track_id'])['is_reorientation']
            .transform(lambda x: x & ~x.shift(1, fill_value=False))
        )
    
    # Get event times per track
    iei_list = []
    for (exp, track), group in events[events['reo_onset'] == True].groupby(['experiment_id', 'track_id']):
        times = group['time'].sort_values().values
        if len(times) > 1:
            intervals = np.diff(times)
            iei_list.extend(intervals)
    
    return np.array(iei_list)


def run_validation(
    empirical_data: pd.DataFrame,
    simulated_data: pd.DataFrame,
    stimulus_times: np.ndarray = None
) -> Dict:
    """
    Run all validation comparisons.
    
    Parameters
    ----------
    empirical_data : DataFrame
        Empirical event data
    simulated_data : DataFrame
        Simulated event data
    stimulus_times : ndarray, optional
        Times of stimulus onsets for PSTH
    
    Returns
    -------
    results : dict
        All validation results
    """
    results = {}
    
    # 1. Turn rate comparison
    print("Comparing turn rates...")
    emp_rates = compute_turn_rate(empirical_data)
    sim_rates = compute_turn_rate(simulated_data)
    results['turn_rate'] = compare_turn_rates(emp_rates, sim_rates)
    print(f"  Empirical: {results['turn_rate']['empirical_mean']:.2f} +/- {results['turn_rate']['empirical_std']:.2f}")
    print(f"  Simulated: {results['turn_rate']['simulated_mean']:.2f} +/- {results['turn_rate']['simulated_std']:.2f}")
    print(f"  Pass: {results['turn_rate']['pass']}")
    
    # 2. PSTH comparison (if stimulus times provided)
    if stimulus_times is not None and len(stimulus_times) > 0:
        print("\nComparing stimulus-locked PSTH...")
        emp_psth = compute_psth(empirical_data, stimulus_times)
        sim_psth = compute_psth(simulated_data, stimulus_times)
        results['psth'] = compare_psth(emp_psth, sim_psth)
        print(f"  Normalized ISE: {results['psth']['normalized_ise']:.4f}")
        print(f"  Pass: {results['psth']['pass']}")
    
    # 3. Heading change distribution
    if 'reo_dtheta' in empirical_data.columns and 'reo_dtheta' in simulated_data.columns:
        print("\nComparing heading change distribution...")
        results['heading_change'] = compare_distributions(
            empirical_data['reo_dtheta'].values,
            simulated_data['reo_dtheta'].values,
            name='heading_change'
        )
        print(f"  KS statistic: {results['heading_change']['ks_statistic']:.4f}")
        print(f"  p-value: {results['heading_change']['p_value']:.4f}")
        print(f"  Pass: {results['heading_change']['pass']}")
    
    # 4. Inter-event interval distribution
    print("\nComparing inter-event intervals...")
    emp_iei = compute_inter_event_intervals(empirical_data)
    sim_iei = compute_inter_event_intervals(simulated_data)
    if len(emp_iei) > 0 and len(sim_iei) > 0:
        results['iei'] = compare_distributions(emp_iei, sim_iei, name='iei')
        print(f"  KS statistic: {results['iei']['ks_statistic']:.4f}")
        print(f"  p-value: {results['iei']['p_value']:.4f}")
        print(f"  Pass: {results['iei']['pass']}")
    
    # Overall pass/fail
    all_passed = all(
        r.get('pass', True) 
        for r in results.values() 
        if isinstance(r, dict)
    )
    results['overall'] = 'PASS' if all_passed else 'FAIL'
    print(f"\nOverall: {results['overall']}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validate simulated vs empirical data')
    parser.add_argument('--empirical', type=str, required=True,
                        help='Path to empirical data (parquet or h5)')
    parser.add_argument('--simulated', type=str, required=True,
                        help='Path to simulated data (parquet)')
    parser.add_argument('--output', type=str, default='data/validation/',
                        help='Output directory for validation results')
    
    args = parser.parse_args()
    
    emp_path = Path(args.empirical)
    sim_path = Path(args.simulated)
    output_dir = Path(args.output)
    
    # Load data
    print(f"Loading empirical data from {emp_path}...")
    if emp_path.suffix == '.h5':
        import h5py
        with h5py.File(emp_path, 'r') as f:
            if 'events' in f:
                grp = f['events']
                data = {k: grp[k][:] for k in grp.keys()}
                empirical = pd.DataFrame(data)
            else:
                raise ValueError("No 'events' group in H5 file")
    else:
        empirical = pd.read_parquet(emp_path)
    
    print(f"Loading simulated data from {sim_path}...")
    simulated = pd.read_parquet(sim_path)
    
    # Extract LED onset times for PSTH validation
    # For 20-min experiments with 30s on/30s off, expect ~20 onsets per experiment
    stimulus_times = None
    if 'led1Val' in empirical.columns and 'time' in empirical.columns:
        print("Extracting LED onset times for PSTH...")
        
        # Approach: sample at 1s resolution, detect major transitions
        all_onsets = []
        for exp_id in empirical['experiment_id'].unique():
            exp_df = empirical[empirical['experiment_id'] == exp_id].sort_values('time')
            
            # Sample to ~1s resolution to avoid detecting ramp steps
            t = exp_df['time'].values
            led = exp_df['led1Val'].values
            
            # Bin to 1s resolution
            t_bins = np.arange(t.min(), t.max() + 1, 1.0)
            led_binned = np.zeros(len(t_bins) - 1)
            for i in range(len(t_bins) - 1):
                mask = (t >= t_bins[i]) & (t < t_bins[i+1])
                if mask.any():
                    led_binned[i] = led[mask].max()
            
            # Detect transitions from low to high (>200 PWM)
            led_high = led_binned > 200
            led_high_prev = np.roll(led_high, 1)
            led_high_prev[0] = False
            transitions = led_high & ~led_high_prev
            
            onset_times = t_bins[:-1][transitions]
            all_onsets.extend(onset_times)
        
        if len(all_onsets) > 0:
            stimulus_times = np.array(all_onsets)
            print(f"  Found {len(stimulus_times)} LED onsets (~{len(stimulus_times)/14:.1f} per experiment)")
    
    # Run validation
    results = run_validation(empirical, simulated, stimulus_times=stimulus_times)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    import json
    with open(output_dir / 'validation_results.json', 'w') as f:
        # Convert numpy types to Python types
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'validation_results.json'}")


if __name__ == '__main__':
    main()

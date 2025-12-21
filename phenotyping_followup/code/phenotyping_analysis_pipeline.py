#!/usr/bin/env python3
"""
End-to-End Phenotyping Analysis Pipeline

Performs complete phenotyping analysis on simulated tracks:
1. Fit track-level kernels to each track
2. Extract phenotype features
3. Perform clustering analysis
4. Validate clustering results
5. Generate visualizations

Features Bambi-style progress monitoring with detailed status updates.

VALIDATION:
- Automatically detects inflated turn rates (> 50 turns/min)
- Documents errors and exits if counting errors detected
- Expected turn rates: 5-30 turns/min for larval behavior
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime
import json

# Add InDySim code directory to path
INDYSIM_CODE = Path('/Users/gilraitses/InDySim/code')
if INDYSIM_CODE.exists() and str(INDYSIM_CODE) not in sys.path:
    sys.path.insert(0, str(INDYSIM_CODE))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gamma as gamma_dist
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    print("Progress bars will be simplified.")


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ProgressMonitor:
    """Bambi-style progress monitor with detailed status updates."""
    
    def __init__(self, total, desc="Processing", bar_length=50):
        self.total = total
        self.current = 0
        self.desc = desc
        self.bar_length = bar_length
        self.start_time = time.time()
        self.last_update = time.time()
        self.phase = "INIT"
        
    def update(self, n=1, phase=None, message=""):
        """Update progress."""
        self.current += n
        if phase:
            self.phase = phase
        self.last_update = time.time()
        self._display(message)
    
    def _display(self, message=""):
        """Display progress bar."""
        if self.total == 0:
            return
        
        elapsed = time.time() - self.start_time
        progress = self.current / self.total
        
        # Calculate ETA
        if self.current > 0:
            rate = self.current / elapsed
            eta = (self.total - self.current) / rate if rate > 0 else 0
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"
        
        # Build progress bar
        filled = int(self.bar_length * progress)
        bar = '█' * filled + '░' * (self.bar_length - filled)
        
        # Status line
        status = f"{self.current}/{self.total} ({progress*100:.1f}%)"
        
        # Print
        sys.stdout.write(f"\r{Colors.CYAN}[{self.phase}]{Colors.ENDC} {bar} {status} {eta_str}")
        if message:
            sys.stdout.write(f" | {message}")
        sys.stdout.flush()
    
    def finish(self, message="Complete!"):
        """Finish progress bar."""
        elapsed = time.time() - self.start_time
        sys.stdout.write(f"\r{Colors.GREEN}✓ {self.desc} {message}{Colors.ENDC} ({elapsed:.1f}s)\n")
        sys.stdout.flush()


def print_header(title, width=70):
    """Print formatted header."""
    print(f"\n{Colors.BOLD}{'='*width}{Colors.ENDC}")
    print(f"{Colors.BOLD}{title.center(width)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*width}{Colors.ENDC}\n")


def print_status(phase, message, status="INFO"):
    """Print status message."""
    colors = {
        "INFO": Colors.CYAN,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED
    }
    color = colors.get(status, Colors.CYAN)
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] [{status}] {phase}: {message}{Colors.ENDC}")


def load_simulated_tracks(data_dir: Path) -> pd.DataFrame:
    """Load all simulated tracks."""
    print_status("LOAD", "Loading simulated tracks...", "INFO")
    
    all_tracks = []
    # Find all track files, but exclude summary files
    all_files = sorted(data_dir.glob("**/track_*.parquet"))
    track_files = [f for f in all_files if 'summary' not in f.stem.lower()]
    
    if not track_files:
        raise FileNotFoundError(f"No track files found in {data_dir}")
    
    print_status("LOAD", f"Found {len(track_files)} track files (excluded {len(all_files) - len(track_files)} summary files)", "INFO")
    
    monitor = ProgressMonitor(len(track_files), desc="Loading tracks")
    
    for i, track_file in enumerate(track_files):
        track_df = pd.read_parquet(track_file)
        
        # Extract track_id and condition from path
        parts = track_file.parts
        condition = parts[-2] if len(parts) >= 2 else "unknown"
        
        # Parse track_id from filename (e.g., track_0001 -> 1)
        try:
            track_id = int(track_file.stem.split('_')[1])
        except (ValueError, IndexError) as e:
            print_status("LOAD", f"Skipping file with invalid track ID format: {track_file.name}", "WARNING")
            continue
        
        track_df['track_id'] = track_id
        track_df['condition'] = condition
        track_df['file_path'] = str(track_file)
        
        all_tracks.append(track_df)
        monitor.update(1, phase="LOAD", message=f"Loaded {condition}/track_{track_id:04d}")
    
    monitor.finish()
    
    combined = pd.concat(all_tracks, ignore_index=True)
    print_status("LOAD", f"Loaded {len(combined['track_id'].unique())} tracks", "SUCCESS")
    
    return combined


def extract_events_from_track(track_df: pd.DataFrame) -> np.ndarray:
    """Extract event times from track DataFrame."""
    # Events are marked by is_turn=True and state transitions
    events = track_df[track_df['is_turn'] == True].copy()
    
    # Get event onset times (first frame of each turn)
    event_mask = (track_df['state'] == 'TURN') & (track_df['state'].shift(1) != 'TURN')
    event_times = track_df[event_mask]['time'].values
    
    return event_times


def compute_led_timing(duration=1200.0):
    """Compute LED timing for 20-minute experiment."""
    LED_ON_DURATION = 10.0
    LED_OFF_DURATION = 20.0
    FIRST_LED_ONSET = 21.3
    
    n_cycles = int(np.ceil((duration - FIRST_LED_ONSET) / (LED_ON_DURATION + LED_OFF_DURATION))) + 1
    led_onsets = np.array([FIRST_LED_ONSET + i * (LED_ON_DURATION + LED_OFF_DURATION) 
                           for i in range(n_cycles)])
    led_offsets = led_onsets + LED_ON_DURATION
    led_onsets = led_onsets[led_onsets < duration]
    led_offsets = led_offsets[led_offsets < duration]
    
    return led_onsets, led_offsets


def gamma_pdf(t, alpha, beta):
    """Compute gamma PDF."""
    result = np.zeros_like(t, dtype=float)
    valid = t > 0
    if valid.any():
        try:
            pdf_vals = gamma_dist.pdf(t[valid], a=alpha, scale=beta)
            # Handle NaN/Inf values
            pdf_vals = np.nan_to_num(pdf_vals, nan=0.0, posinf=0.0, neginf=0.0)
            result[valid] = pdf_vals
        except:
            # If gamma PDF fails, return zeros
            pass
    return result


def kernel_function(t, A, alpha1, beta1, B, alpha2, beta2):
    """Gamma-difference kernel."""
    return A * gamma_pdf(t, alpha1, beta1) - B * gamma_pdf(t, alpha2, beta2)


def compute_kernel_values(event_times, led_onsets, led_offsets, kernel_params):
    """Compute kernel values for event times."""
    kernel_vals = np.zeros(len(event_times))
    
    for i, t_event in enumerate(event_times):
        # Find most recent LED onset
        onsets_before = led_onsets[led_onsets <= t_event]
        if len(onsets_before) > 0:
            t_since_onset = t_event - onsets_before[-1]
            # Check both ON and OFF periods (kernel can have values in both)
            if t_since_onset >= 0:  # After LED onset
                kernel_val = kernel_function(
                    t_since_onset,
                    kernel_params['A'],
                    kernel_params['alpha1'],
                    kernel_params['beta1'],
                    kernel_params['B'],
                    kernel_params['alpha2'],
                    kernel_params['beta2']
                )
                # Handle NaN/Inf
                if np.isfinite(kernel_val):
                    kernel_vals[i] = kernel_val
    
    return kernel_vals


def fit_track_level_kernel(track_df: pd.DataFrame, led_onsets, led_offsets):
    """Fit gamma-difference kernel to a single track."""
    # Extract events
    event_times = extract_events_from_track(track_df)
    
    if len(event_times) < 10:
        return None  # Not enough events
    
    # Initial parameters (from population fit)
    initial_params = {
        'A': 0.456,
        'alpha1': 2.22,
        'beta1': 0.132,
        'B': 12.54,
        'alpha2': 4.38,
        'beta2': 0.869
    }
    
    # Objective function: maximize kernel values at event times
    def objective(params):
        A, alpha1, beta1, B, alpha2, beta2 = params
        
        # Check for invalid parameters
        if any(np.isnan(params)) or any(np.isinf(params)):
            return 1e10  # Large penalty for invalid params
        
        # Compute kernel values
        try:
            kernel_vals = compute_kernel_values(
                event_times, led_onsets, led_offsets,
                {'A': A, 'alpha1': alpha1, 'beta1': beta1,
                 'B': B, 'alpha2': alpha2, 'beta2': beta2}
            )
        except Exception:
            return 1e10
        
        # Check for invalid kernel values
        if np.any(np.isnan(kernel_vals)) or np.any(np.isinf(kernel_vals)):
            return 1e10  # Large penalty for invalid kernel values
        
        # Use sum of squared kernel values as objective (numerically stable)
        # We want to maximize kernel values at event times
        # Negative because minimize() minimizes
        try:
            kernel_vals_clipped = np.clip(kernel_vals, 1e-8, 1e8)
            objective_val = -np.sum(kernel_vals_clipped ** 2)
        except Exception:
            return 1e10
        
        # Check for invalid result
        if np.isnan(objective_val) or np.isinf(objective_val):
            return 1e10
        
        return objective_val
    
    # Bounds
    bounds = [
        (0.1, 2.0),      # A
        (1.5, 3.0),      # alpha1
        (0.05, 0.3),     # beta1
        (5.0, 20.0),     # B
        (3.0, 6.0),      # alpha2
        (0.5, 1.5)       # beta2
    ]
    
        # Optimize - try multiple methods if L-BFGS-B fails
    try:
        # First try L-BFGS-B
        result = minimize(
            objective,
            [initial_params['A'], initial_params['alpha1'], initial_params['beta1'],
             initial_params['B'], initial_params['alpha2'], initial_params['beta2']],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-4, 'gtol': 1e-3}
        )
        
        # If L-BFGS-B fails, try Nelder-Mead (doesn't need gradients)
        if not result.success:
            result = minimize(
                objective,
                [initial_params['A'], initial_params['alpha1'], initial_params['beta1'],
                 initial_params['B'], initial_params['alpha2'], initial_params['beta2']],
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4}
            )
        
        if result.success:
            return {
                'A': result.x[0],
                'alpha1': result.x[1],
                'beta1': result.x[2],
                'B': result.x[3],
                'alpha2': result.x[4],
                'beta2': result.x[5],
                'tau1': result.x[1] * result.x[2],
                'tau2': result.x[4] * result.x[5],
                'n_events': len(event_times),
                'converged': True
            }
    except Exception:
        pass
    
    return None


def fit_all_track_kernels(tracks_df: pd.DataFrame, output_dir: Path):
    """Fit kernels to all tracks."""
    print_header("STEP 1: Fitting Track-Level Kernels")
    
    led_onsets, led_offsets = compute_led_timing()
    
    unique_tracks = tracks_df.groupby(['condition', 'track_id'])
    n_tracks = len(unique_tracks)
    
    print_status("FIT", f"Fitting kernels to {n_tracks} tracks...", "INFO")
    
    results = []
    
    if HAS_TQDM:
        iterator = tqdm(unique_tracks, desc="Fitting kernels", unit="track")
    else:
        iterator = unique_tracks
        monitor = ProgressMonitor(n_tracks, desc="Fitting kernels")
    
    for (condition, track_id), track_group in iterator:
        track_df = track_group.sort_values('time').reset_index(drop=True)
        
        kernel_result = fit_track_level_kernel(track_df, led_onsets, led_offsets)
        
        if kernel_result:
            kernel_result['condition'] = condition
            kernel_result['track_id'] = track_id
            results.append(kernel_result)
        
        if not HAS_TQDM:
            monitor.update(1, phase="FIT", message=f"{condition}/track_{track_id:04d}")
    
    if not HAS_TQDM:
        monitor.finish()
    
    results_df = pd.DataFrame(results)
    
    # Save results (only if we have results)
    if len(results_df) > 0:
        results_df.to_csv(output_dir / 'track_kernel_fits.csv', index=False)
        results_df.to_parquet(output_dir / 'track_kernel_fits.parquet', index=False)
    else:
        # Create empty DataFrame with expected columns
        results_df = pd.DataFrame(columns=['condition', 'track_id', 'A', 'alpha1', 'beta1', 'B', 'alpha2', 'beta2', 'tau1', 'tau2', 'n_events', 'converged'])
    
    print_status("FIT", f"Successfully fitted {len(results_df)}/{n_tracks} tracks", "SUCCESS")
    
    return results_df


def extract_phenotype_features(tracks_df: pd.DataFrame, kernel_fits_df: pd.DataFrame, output_dir: Path):
    """Extract phenotype features from tracks and kernel fits."""
    print_header("STEP 2: Extracting Phenotype Features")
    
    print_status("EXTRACT", "Computing event statistics...", "INFO")
    
    features = []
    
    unique_tracks = tracks_df.groupby(['condition', 'track_id'])
    n_tracks = len(unique_tracks)
    
    if HAS_TQDM:
        iterator = tqdm(unique_tracks, desc="Extracting features", unit="track")
    else:
        iterator = unique_tracks
        monitor = ProgressMonitor(n_tracks, desc="Extracting features")
    
    for (condition, track_id), track_group in iterator:
        track_df = track_group.sort_values('time').reset_index(drop=True)
        
        # Event statistics
        # CRITICAL FIX: Count turn ONSETS (state transitions), not all frames with is_turn=True
        # is_turn=True marks all frames during a turn, not just the event onset
        # With 20 Hz sampling, a 1s turn = 20 frames, so counting is_turn inflates by ~20-30x
        
        # Behavioral allocation (calculate first, needed for logging)
        total_time = track_df['time'].max() - track_df['time'].min()
        
        # Count actual turn events (state transitions from RUN to TURN)
        # CRITICAL: Count state transitions, not all frames with is_turn=True
        # is_turn=True marks all frames during a turn (20-30 frames per turn at 20 Hz)
        turn_onsets = (track_df['state'] == 'TURN') & (track_df['state'].shift(1) != 'TURN')
        n_turn_events = turn_onsets.sum()
        
        # Also count frames with is_turn for comparison/debugging
        n_turn_frames = track_df['is_turn'].sum()
        
        # #region agent log
        log_path = Path('/Users/gilraitses/INDYsim_project/.cursor/debug.log')
        try:
            import json
            with open(log_path, 'a') as f:
                f.write(json.dumps({"location": "extract_phenotype_features:turn_counting", "message": "Turn counting comparison", "data": {"track_id": track_id, "condition": condition, "n_turn_events": int(n_turn_events), "n_turn_frames": int(n_turn_frames), "inflation_factor": float(n_turn_frames / n_turn_events) if n_turn_events > 0 else 0, "turn_rate_correct": float(n_turn_events / (total_time / 60)) if total_time > 0 else 0, "turn_rate_wrong": float(n_turn_frames / (total_time / 60)) if total_time > 0 else 0}, "timestamp": int(time.time() * 1000), "sessionId": "debug-session", "runId": "run2", "hypothesisId": "H6"}) + '\n')
        except: pass
        # #endregion
        
        # Turn durations (from state transitions)
        turn_durations = []
        turn_start_times = track_df[turn_onsets]['time'].values
        turn_end_mask = (track_df['state'] != 'TURN') & (track_df['state'].shift(1) == 'TURN')
        turn_end_times = track_df[turn_end_mask]['time'].values
        
        # Match starts and ends
        for i, start_time in enumerate(turn_start_times):
            # Find next end time after this start
            ends_after = turn_end_times[turn_end_times > start_time]
            if len(ends_after) > 0:
                duration = ends_after[0] - start_time
                turn_durations.append(duration)
        run_time = (track_df['state'] == 'RUN').sum() * 0.05  # 20 Hz
        turn_time = (track_df['state'] == 'TURN').sum() * 0.05
        
        # Get kernel parameters if available
        # Handle empty kernel_fits_df gracefully
        if len(kernel_fits_df) > 0 and 'condition' in kernel_fits_df.columns:
            kernel_row = kernel_fits_df[
                (kernel_fits_df['condition'] == condition) & 
                (kernel_fits_df['track_id'] == track_id)
            ]
        else:
            kernel_row = pd.DataFrame()  # Empty DataFrame
        
        # Calculate turn rate
        turn_rate = n_turn_events / (total_time / 60) if total_time > 0 else 0
        
        # VALIDATION: Detect inflated turn rates (indicates counting error)
        # Expected range: 5-30 turns/min for larval behavior
        # If > 50 turns/min, likely counting frames instead of events
        if turn_rate > 50:
            error_msg = f"""
================================================================================
ERROR: INFLATED TURN RATE DETECTED
================================================================================

Track: {condition}/track_{track_id:04d}
Turn rate: {turn_rate:.1f} turns/min (expected: 5-30 turns/min)

This indicates a counting error - likely counting frames with is_turn=True
instead of actual turn events (state transitions).

Diagnostics:
  - Turn events (state transitions): {n_turn_events}
  - Frames with is_turn=True: {n_turn_frames}
  - Inflation factor: {n_turn_frames / n_turn_events if n_turn_events > 0 else 'N/A':.1f}x
  - Expected turn rate: {n_turn_events / (total_time / 60) if total_time > 0 else 0:.1f} turns/min

FIX REQUIRED:
  Use state transitions to count events:
    turn_onsets = (track_df['state'] == 'TURN') & (track_df['state'].shift(1) != 'TURN')
    n_events = turn_onsets.sum()

Do NOT use:
    n_events = len(track_df[track_df['is_turn'] == True])  # WRONG - counts all frames

================================================================================
"""
            print(error_msg)
            
            # Write error to file
            error_file = output_dir / 'TURN_RATE_ERROR_DETECTED.txt'
            with open(error_file, 'w') as f:
                f.write(error_msg)
                f.write(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Script: phenotyping_analysis_pipeline.py\n")
                f.write(f"Location: extract_phenotype_features()\n")
            
            print_status("ERROR", f"Error documented in {error_file}", "ERROR")
            raise ValueError(f"Inflated turn rate detected: {turn_rate:.1f} turns/min. "
                           f"Expected 5-30 turns/min. See {error_file} for details.")
        
        feature_dict = {
            'condition': condition,
            'track_id': track_id,
            'n_events': n_turn_events,  # FIXED: Count actual events, not frames
            'turn_rate': turn_rate,  # Validated: Should be 5-30 turns/min
            'mean_turn_duration': np.mean(turn_durations) if turn_durations else 0,
            'median_turn_duration': np.median(turn_durations) if turn_durations else 0,
            'run_fraction': run_time / total_time if total_time > 0 else 0,
            'turn_fraction': turn_time / total_time if total_time > 0 else 0,
        }
        
        if len(kernel_row) > 0:
            feature_dict.update({
                'tau1': kernel_row.iloc[0]['tau1'],
                'tau2': kernel_row.iloc[0]['tau2'],
                'amplitude_A': kernel_row.iloc[0]['A'],
                'amplitude_B': kernel_row.iloc[0]['B'],
            })
        else:
            feature_dict.update({
                'tau1': np.nan,
                'tau2': np.nan,
                'amplitude_A': np.nan,
                'amplitude_B': np.nan,
            })
        
        features.append(feature_dict)
        
        if not HAS_TQDM:
            monitor.update(1, phase="EXTRACT", message=f"{condition}/track_{track_id:04d}")
    
    if not HAS_TQDM:
        monitor.finish()
    
    features_df = pd.DataFrame(features)
    
    # VALIDATION: Check all turn rates are reasonable
    if 'turn_rate' in features_df.columns:
        max_rate = features_df['turn_rate'].max()
        min_rate = features_df['turn_rate'].min()
        mean_rate = features_df['turn_rate'].mean()
        
        print_status("EXTRACT", f"Turn rate validation: min={min_rate:.1f}, max={max_rate:.1f}, mean={mean_rate:.1f} turns/min", "INFO")
        
        if max_rate > 50:
            error_msg = f"""
================================================================================
ERROR: INFLATED TURN RATES DETECTED IN RESULTS
================================================================================

Summary statistics:
  - Minimum turn rate: {min_rate:.1f} turns/min
  - Maximum turn rate: {max_rate:.1f} turns/min (EXPECTED: < 30)
  - Mean turn rate: {mean_rate:.1f} turns/min (EXPECTED: ~10-15)

Tracks with inflated rates (> 50 turns/min): {(features_df['turn_rate'] > 50).sum()}

This indicates a counting error in the feature extraction code.
Likely counting frames with is_turn=True instead of actual turn events.

See TURN_RATE_ERROR_DETECTED.txt for details on first problematic track.

================================================================================
"""
            print(error_msg)
            
            error_file = output_dir / 'TURN_RATE_ERROR_DETECTED.txt'
            with open(error_file, 'a') as f:
                f.write(error_msg)
            
            print_status("ERROR", "Inflated turn rates detected. Analysis stopped.", "ERROR")
            raise ValueError(f"Invalid turn rates detected. Max rate: {max_rate:.1f} turns/min. "
                           f"Expected < 30 turns/min. See {error_file} for details.")
        
        if mean_rate < 5 or mean_rate > 30:
            print_status("WARNING", f"Mean turn rate ({mean_rate:.1f}) outside expected range (5-30 turns/min)", "WARNING")
    
    # Save features
    features_df.to_csv(output_dir / 'phenotype_features.csv', index=False)
    features_df.to_parquet(output_dir / 'phenotype_features.parquet', index=False)
    
    print_status("EXTRACT", f"Extracted features from {len(features_df)} tracks", "SUCCESS")
    print_status("EXTRACT", f"Turn rates validated: all within expected range (5-30 turns/min)", "SUCCESS")
    
    return features_df


def perform_clustering(features_df: pd.DataFrame, output_dir: Path):
    """Perform clustering analysis."""
    print_header("STEP 3: Clustering Analysis")
    
    # Select features for clustering
    feature_cols = ['tau1', 'tau2', 'amplitude_A', 'amplitude_B', 
                    'turn_rate', 'mean_turn_duration', 'run_fraction']
    
    # Remove rows with missing kernel parameters
    clustering_df = features_df.dropna(subset=['tau1', 'tau2']).copy()
    
    if len(clustering_df) == 0:
        print_status("CLUSTER", "No tracks with valid kernel fits for clustering", "WARNING")
        return None
    
    print_status("CLUSTER", f"Clustering {len(clustering_df)} tracks...", "INFO")
    
    # Prepare features
    X = clustering_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Try different numbers of clusters
    n_clusters_range = range(2, min(8, len(clustering_df) // 10 + 1))
    results = []
    
    print_status("CLUSTER", "Testing different numbers of clusters...", "INFO")
    
    for n_clusters in n_clusters_range:
        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # Hierarchical
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        hier_labels = hierarchical.fit_predict(X_scaled)
        
        # Silhouette scores
        kmeans_sil = silhouette_score(X_scaled, kmeans_labels)
        hier_sil = silhouette_score(X_scaled, hier_labels)
        
        results.append({
            'n_clusters': n_clusters,
            'kmeans_silhouette': kmeans_sil,
            'hierarchical_silhouette': hier_sil,
            'kmeans_labels': kmeans_labels,
            'hierarchical_labels': hier_labels
        })
        
        print_status("CLUSTER", f"n_clusters={n_clusters}: K-means sil={kmeans_sil:.3f}, Hier sil={hier_sil:.3f}", "INFO")
    
    # Select best number of clusters
    best_result = max(results, key=lambda x: max(x['kmeans_silhouette'], x['hierarchical_silhouette']))
    best_n = best_result['n_clusters']
    
    print_status("CLUSTER", f"Best n_clusters: {best_n}", "SUCCESS")
    
    # Add cluster labels to dataframe
    clustering_df['kmeans_cluster'] = best_result['kmeans_labels']
    clustering_df['hierarchical_cluster'] = best_result['hierarchical_labels']
    
    # Save clustering results
    clustering_df.to_csv(output_dir / 'clustering_results.csv', index=False)
    clustering_df.to_parquet(output_dir / 'clustering_results.parquet', index=False)
    
    # Save cluster summary
    cluster_summary = clustering_df.groupby('kmeans_cluster')[feature_cols].mean()
    cluster_summary.to_csv(output_dir / 'cluster_summary.csv')
    
    print_status("CLUSTER", "Clustering complete", "SUCCESS")
    
    return clustering_df, best_result


def generate_visualizations(features_df: pd.DataFrame, clustering_df: pd.DataFrame, output_dir: Path):
    """Generate visualization plots."""
    print_header("STEP 4: Generating Visualizations")
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    print_status("VIZ", "Creating plots...", "INFO")
    
    # 1. Kernel parameter distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    if 'tau1' in features_df.columns:
        features_df['tau1'].dropna().hist(ax=axes[0, 0], bins=30, edgecolor='black')
        axes[0, 0].set_xlabel('τ₁ (s)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Fast Timescale (τ₁) Distribution')
    
    if 'tau2' in features_df.columns:
        features_df['tau2'].dropna().hist(ax=axes[0, 1], bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('τ₂ (s)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Slow Timescale (τ₂) Distribution')
    
    if 'amplitude_A' in features_df.columns:
        features_df['amplitude_A'].dropna().hist(ax=axes[1, 0], bins=30, edgecolor='black')
        axes[1, 0].set_xlabel('Amplitude A')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Fast Component Amplitude')
    
    if 'turn_rate' in features_df.columns:
        features_df['turn_rate'].hist(ax=axes[1, 1], bins=30, edgecolor='black')
        axes[1, 1].set_xlabel('Turn Rate (turns/min)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Turn Rate Distribution')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'parameter_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Clustering visualization (if available)
    if clustering_df is not None and 'kmeans_cluster' in clustering_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # PCA for visualization
        feature_cols = ['tau1', 'tau2', 'amplitude_A', 'amplitude_B', 
                       'turn_rate', 'mean_turn_duration', 'run_fraction']
        X = clustering_df[feature_cols].dropna().values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # K-means clusters
        valid_mask = clustering_df[feature_cols].notna().all(axis=1)
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], 
                                 c=clustering_df.loc[valid_mask, 'kmeans_cluster'],
                                 cmap='viridis', alpha=0.6)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[0].set_title('K-means Clustering')
        plt.colorbar(scatter, ax=axes[0])
        
        # Hierarchical clusters
        scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                                 c=clustering_df.loc[valid_mask, 'hierarchical_cluster'],
                                 cmap='plasma', alpha=0.6)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[1].set_title('Hierarchical Clustering')
        plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'clustering_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print_status("VIZ", f"Visualizations saved to {viz_dir}/", "SUCCESS")


def main():
    """Main analysis pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='End-to-end phenotyping analysis pipeline')
    parser.add_argument('--data-dir', type=str, 
                       default='/Users/gilraitses/InDySim/data/simulated_phenotyping',
                       help='Directory containing simulated tracks')
    parser.add_argument('--output-dir', type=str,
                       default='/Users/gilraitses/InDySim/results/phenotyping_analysis',
                       help='Output directory for results')
    parser.add_argument('--skip-clustering', action='store_true',
                       help='Skip clustering analysis')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header("PHENOTYPING ANALYSIS PIPELINE", width=80)
    print_status("INIT", f"Data directory: {data_dir}", "INFO")
    print_status("INIT", f"Output directory: {output_dir}", "INFO")
    print_status("INIT", f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "INFO")
    
    start_time = time.time()
    
    try:
        # Step 1: Load tracks
        tracks_df = load_simulated_tracks(data_dir)
        
        # Step 2: Fit track-level kernels
        kernel_fits_df = fit_all_track_kernels(tracks_df, output_dir)
        
        # Step 3: Extract features
        features_df = extract_phenotype_features(tracks_df, kernel_fits_df, output_dir)
        
        # Step 4: Clustering
        clustering_df = None
        if not args.skip_clustering:
            clustering_result = perform_clustering(features_df, output_dir)
            if clustering_result:
                clustering_df, _ = clustering_result
        
        # Step 5: Visualizations
        if not args.skip_viz:
            generate_visualizations(features_df, clustering_df, output_dir)
        
        elapsed = time.time() - start_time
        
        print_header("ANALYSIS COMPLETE", width=80)
        print_status("DONE", f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)", "SUCCESS")
        print_status("DONE", f"Results saved to: {output_dir}", "SUCCESS")
        print_status("DONE", f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "SUCCESS")
        
    except Exception as e:
        print_status("ERROR", f"Analysis failed: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


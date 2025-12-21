#!/usr/bin/env python3
"""
Assess Incomplete Tracks for Phenotyping

Analyzes incomplete tracks (10+ minutes) to determine if they should be included
in phenotyping analysis.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add InDySim code directory to path
INDYSIM_CODE = Path('/Users/gilraitses/InDySim/code')
if INDYSIM_CODE.exists() and str(INDYSIM_CODE) not in sys.path:
    sys.path.insert(0, str(INDYSIM_CODE))

from phenotyping_analysis_pipeline import (
    load_simulated_tracks,
    fit_track_level_kernel,
    compute_led_timing,
    extract_events_from_track
)

def assess_track_length_requirements(tracks_df, min_durations=[5, 7, 10, 15, 20]):
    """
    Assess minimum track length needed for reliable kernel fitting.
    
    Parameters
    ----------
    tracks_df : DataFrame
        All tracks
    min_durations : list
        Minimum durations to test (minutes)
    
    Returns
    -------
    results : dict
        Fitting success rates by duration
    """
    print("=" * 80)
    print("ASSESSING MINIMUM TRACK LENGTH REQUIREMENTS")
    print("=" * 80)
    
    led_onsets, led_offsets = compute_led_timing()
    
    results = {}
    
    for min_dur_min in min_durations:
        min_dur_sec = min_dur_min * 60
        
        # Filter tracks by duration
        unique_tracks = tracks_df.groupby(['condition', 'track_id'])
        filtered_tracks = []
        
        for (condition, track_id), track_group in unique_tracks:
            track_df = track_group.sort_values('time').reset_index(drop=True)
            duration = track_df['time'].max()
            
            if duration >= min_dur_sec:
                filtered_tracks.append((condition, track_id, track_df))
        
        print(f"\nTesting minimum duration: {min_dur_min} minutes")
        print(f"  Tracks meeting criteria: {len(filtered_tracks)}")
        
        # Try to fit kernels
        successful_fits = 0
        n_events_list = []
        
        for condition, track_id, track_df in filtered_tracks[:50]:  # Sample first 50
            event_times = extract_events_from_track(track_df)
            n_events_list.append(len(event_times))
            
            if len(event_times) >= 10:
                result = fit_track_level_kernel(track_df, led_onsets, led_offsets)
                if result and result.get('converged'):
                    successful_fits += 1
        
        success_rate = successful_fits / len(filtered_tracks[:50]) if filtered_tracks else 0
        mean_events = np.mean(n_events_list) if n_events_list else 0
        
        results[min_dur_min] = {
            'n_tracks': len(filtered_tracks),
            'success_rate': success_rate,
            'mean_events': mean_events,
            'successful_fits': successful_fits
        }
        
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Mean events: {mean_events:.1f}")
    
    return results


def compare_complete_vs_incomplete(complete_tracks_df, incomplete_tracks_df):
    """
    Compare parameter distributions between complete and incomplete tracks.
    
    Parameters
    ----------
    complete_tracks_df : DataFrame
        Complete tracks (20 min)
    incomplete_tracks_df : DataFrame
        Incomplete tracks (10-19 min)
    
    Returns
    -------
    comparison : dict
        Statistical comparison results
    """
    print("\n" + "=" * 80)
    print("COMPARING COMPLETE vs INCOMPLETE TRACKS")
    print("=" * 80)
    
    # Fit kernels to both
    led_onsets, led_offsets = compute_led_timing()
    
    complete_params = []
    incomplete_params = []
    
    print("\nFitting kernels to complete tracks...")
    complete_unique = complete_tracks_df.groupby(['condition', 'track_id'])
    for (condition, track_id), track_group in list(complete_unique)[:50]:  # Sample
        track_df = track_group.sort_values('time').reset_index(drop=True)
        result = fit_track_level_kernel(track_df, led_onsets, led_offsets)
        if result and result.get('converged'):
            complete_params.append(result)
    
    print(f"  Successful fits: {len(complete_params)}")
    
    print("\nFitting kernels to incomplete tracks (≥10 min)...")
    incomplete_unique = incomplete_tracks_df.groupby(['condition', 'track_id'])
    incomplete_filtered = []
    for (condition, track_id), track_group in incomplete_unique:
        track_df = track_group.sort_values('time').reset_index(drop=True)
        duration = track_df['time'].max()
        if duration >= 600:  # 10 minutes
            incomplete_filtered.append((condition, track_id, track_df))
    
    for condition, track_id, track_df in incomplete_filtered[:50]:  # Sample
        result = fit_track_level_kernel(track_df, led_onsets, led_offsets)
        if result and result.get('converged'):
            incomplete_params.append(result)
    
    print(f"  Successful fits: {len(incomplete_params)}")
    
    if len(complete_params) == 0 or len(incomplete_params) == 0:
        print("\n⚠️  Insufficient data for comparison")
        return None
    
    # Compare distributions
    from scipy import stats
    
    complete_df = pd.DataFrame(complete_params)
    incomplete_df = pd.DataFrame(incomplete_params)
    
    comparison = {}
    
    for param in ['tau1', 'tau2', 'A', 'B']:
        if param in complete_df.columns and param in incomplete_df.columns:
            complete_vals = complete_df[param].dropna()
            incomplete_vals = incomplete_df[param].dropna()
            
            if len(complete_vals) > 0 and len(incomplete_vals) > 0:
                # Mann-Whitney U test
                stat, pval = stats.mannwhitneyu(complete_vals, incomplete_vals, alternative='two-sided')
                
                comparison[param] = {
                    'complete_mean': complete_vals.mean(),
                    'incomplete_mean': incomplete_vals.mean(),
                    'p_value': pval,
                    'different': pval < 0.05
                }
                
                print(f"\n{param}:")
                print(f"  Complete: {complete_vals.mean():.3f} ± {complete_vals.std():.3f}")
                print(f"  Incomplete: {incomplete_vals.mean():.3f} ± {incomplete_vals.std():.3f}")
                print(f"  p-value: {pval:.4f} {'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}")
    
    return comparison


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Assess incomplete tracks for phenotyping')
    parser.add_argument('--complete-tracks-dir', type=str,
                       default='/Users/gilraitses/InDySim/data/simulated_phenotyping',
                       help='Directory with complete tracks')
    parser.add_argument('--incomplete-tracks-dir', type=str,
                       default=None,
                       help='Directory with incomplete tracks (if separate)')
    parser.add_argument('--output-dir', type=str,
                       default='/Users/gilraitses/InDySim/results/incomplete_tracks_analysis',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("INCOMPLETE TRACKS ASSESSMENT")
    print("=" * 80)
    
    # Load complete tracks
    print("\nLoading complete tracks...")
    complete_tracks = load_simulated_tracks(Path(args.complete_tracks_dir))
    
    # Assess minimum duration requirements
    duration_results = assess_track_length_requirements(complete_tracks)
    
    # Save results
    duration_df = pd.DataFrame(duration_results).T
    duration_df.to_csv(output_dir / 'minimum_duration_requirements.csv')
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Find minimum duration with >80% success rate
    for min_dur, results in sorted(duration_results.items()):
        if results['success_rate'] >= 0.8:
            print(f"\n✓ Minimum reliable duration: {min_dur} minutes")
            print(f"  Success rate: {results['success_rate']:.1%}")
            print(f"  Mean events: {results['mean_events']:.1f}")
            break
    
    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()


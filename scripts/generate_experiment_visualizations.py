#!/usr/bin/env python3
"""
Generate visualization plots for INDYsim experiments.

Creates folder structure and generates:
- ESET-level LED value plots
- ESET-level summary statistics
- Experiment-level exact reference plots (MATLAB style)
- Individual track composite plots
- Cycle-based turn rate bin analysis

Author: larry
Date: 2025-11-11
"""

import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Optional, Tuple

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from cinnamoroll_palette import set_cinnamoroll_style
    HAS_CINNAMOROLL = True
except ImportError:
    HAS_CINNAMOROLL = False
    print("Warning: cinnamoroll_palette not found, using default matplotlib style")


def create_figure_structure(eset_name: str, experiment_timestamps: List[str], 
                           base_dir: Path = Path("data/figures")) -> Dict[str, Path]:
    """
    Create folder structure for experiment visualizations.
    
    Args:
        eset_name: ESET folder name (e.g., T_Re_Sq_0to250PWM_30#C_Bl_7PWM)
        experiment_timestamps: List of experiment timestamps
        base_dir: Base directory for figures
    
    Returns:
        dict with paths to created directories
    """
    eset_fig_dir = base_dir / eset_name
    
    # Create main ESET directory
    eset_fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    ledvals_dir = eset_fig_dir / "ledVals"
    summary_dir = eset_fig_dir / "summary"
    
    ledvals_dir.mkdir(exist_ok=True)
    summary_dir.mkdir(exist_ok=True)
    
    # Create experiment directories
    experiment_dirs = {}
    for timestamp in experiment_timestamps:
        exp_dir = eset_fig_dir / timestamp
        exp_dir.mkdir(exist_ok=True)
        
        exact_ref_dir = exp_dir / "exact_reference_plots_matlab"
        cycle_analysis_dir = exp_dir / "cycle_analysis"
        
        exact_ref_dir.mkdir(exist_ok=True)
        cycle_analysis_dir.mkdir(exist_ok=True)
        
        experiment_dirs[timestamp] = {
            'root': exp_dir,
            'exact_reference': exact_ref_dir,
            'cycle_analysis': cycle_analysis_dir
        }
    
    return {
        'eset_root': eset_fig_dir,
        'ledvals': ledvals_dir,
        'summary': summary_dir,
        'experiments': experiment_dirs
    }


def plot_led_values(h5_file: Path, output_path: Path, 
                   frame_rate: float = 10.0) -> None:
    """
    Plot LED values over time for a single experiment.
    
    Args:
        h5_file: Path to H5 file
        output_path: Path to save plot
        frame_rate: Frame rate in Hz (default 10 fps)
    """
    if HAS_CINNAMOROLL:
        set_cinnamoroll_style()
    
    with h5py.File(h5_file, 'r') as f:
        # Get LED values
        led1_vals = f['global_quantities/led1Val/yData'][:]
        led2_vals = None
        if 'global_quantities/led2Val/yData' in f:
            led2_vals = f['global_quantities/led2Val/yData'][:]
        
        # Get ETI for time axis
        if 'eti' in f:
            time_axis = f['eti'][:]
        else:
            # Fallback: use frame numbers
            time_axis = np.arange(len(led1_vals)) / frame_rate
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot LED1 (red)
    ax.plot(time_axis, led1_vals, 'r-', linewidth=1.5, label='LED1 (Red)')
    ax.fill_between(time_axis, 0, led1_vals, alpha=0.3, color='red')
    
    # Plot LED2 (blue) if available
    if led2_vals is not None:
        ax.plot(time_axis, led2_vals, 'b-', linewidth=1.5, label='LED2 (Blue)')
        ax.fill_between(time_axis, 0, led2_vals, alpha=0.3, color='blue')
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('LED Value (PWM)', fontsize=12)
    ax.set_title(f'LED Values: {h5_file.stem}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_eset_summary_ledvals(eset_dir: Path, output_path: Path,
                              h5_files: List[Path]) -> None:
    """
    Plot aggregate LED value summary for an ESET.
    
    Args:
        eset_dir: ESET directory path
        output_path: Path to save plot
        h5_files: List of H5 file paths for this ESET
    """
    if HAS_CINNAMOROLL:
        set_cinnamoroll_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Collect LED values from all experiments
    all_led1_vals = []
    all_led2_vals = []
    pulse_intensities_led1 = []
    pulse_intensities_led2 = []
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            led1 = f['global_quantities/led1Val/yData'][:]
            all_led1_vals.append(led1)
            
            # Extract pulse intensities (max values during ON periods)
            # Simple threshold-based detection
            threshold = np.max(led1) * 0.1
            pulse_intensities_led1.extend(led1[led1 > threshold])
            
            if 'global_quantities/led2Val/yData' in f:
                led2 = f['global_quantities/led2Val/yData'][:]
                all_led2_vals.append(led2)
                pulse_intensities_led2.extend(led2[led2 > threshold])
    
    # Plot 1: Overlay all LED1 values
    ax1 = axes[0, 0]
    for i, led1_vals in enumerate(all_led1_vals):
        time_axis = np.arange(len(led1_vals)) / 10.0  # 10 fps
        ax1.plot(time_axis, led1_vals, alpha=0.3, linewidth=0.5)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('LED1 Value (PWM)')
    ax1.set_title('LED1 Values - All Experiments')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of pulse intensities (LED1)
    ax2 = axes[0, 1]
    if pulse_intensities_led1:
        ax2.hist(pulse_intensities_led1, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax2.axvline(np.mean(pulse_intensities_led1), color='darkred', 
                    linestyle='--', linewidth=2, label=f'Mean: {np.mean(pulse_intensities_led1):.1f}')
        ax2.axvline(np.min(pulse_intensities_led1), color='blue', 
                    linestyle='--', linewidth=1, label=f'Min: {np.min(pulse_intensities_led1):.1f}')
        ax2.axvline(np.max(pulse_intensities_led1), color='green', 
                    linestyle='--', linewidth=1, label=f'Max: {np.max(pulse_intensities_led1):.1f}')
    ax2.set_xlabel('LED1 Pulse Intensity (PWM)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('LED1 Pulse Intensity Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Overlay all LED2 values (if available)
    ax3 = axes[1, 0]
    if all_led2_vals:
        for led2_vals in all_led2_vals:
            time_axis = np.arange(len(led2_vals)) / 10.0
            ax3.plot(time_axis, led2_vals, alpha=0.3, linewidth=0.5, color='blue')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('LED2 Value (PWM)')
    ax3.set_title('LED2 Values - All Experiments')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution of pulse intensities (LED2)
    ax4 = axes[1, 1]
    if pulse_intensities_led2:
        ax4.hist(pulse_intensities_led2, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(np.mean(pulse_intensities_led2), color='darkblue', 
                    linestyle='--', linewidth=2, label=f'Mean: {np.mean(pulse_intensities_led2):.1f}')
    ax4.set_xlabel('LED2 Pulse Intensity (PWM)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('LED2 Pulse Intensity Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'ESET LED Values Summary: {eset_dir.name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_eset_statistics(eset_dir: Path, output_path: Path,
                        h5_files: List[Path]) -> None:
    """
    Plot ESET-level summary statistics.
    
    Args:
        eset_dir: ESET directory path
        output_path: Path to save plot
        h5_files: List of H5 file paths for this ESET
    """
    if HAS_CINNAMOROLL:
        set_cinnamoroll_style()
    
    # Collect statistics from all experiments
    track_counts = []
    experiment_names = []
    
    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            if 'tracks' in f:
                num_tracks = len([k for k in f['tracks'].keys() if k.startswith('track_')])
                track_counts.append(num_tracks)
                experiment_names.append(h5_file.stem)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Track count distribution
    ax1 = axes[0, 0]
    ax1.hist(track_counts, bins=10, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(track_counts), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(track_counts):.1f}')
    ax1.set_xlabel('Number of Tracks')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Track Count Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Track counts by experiment
    ax2 = axes[0, 1]
    x_pos = np.arange(len(experiment_names))
    ax2.bar(x_pos, track_counts, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Experiment')
    ax2.set_ylabel('Number of Tracks')
    ax2.set_title('Track Counts by Experiment')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.split('_')[-1] for name in experiment_names], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Statistics table (placeholder - would need processed data)
    ax3 = axes[1, 0]
    ax3.axis('off')
    stats_text = f"""
    ESET Statistics Summary
    
    Total Experiments: {len(h5_files)}
    Total Tracks: {sum(track_counts)}
    Mean Tracks/Experiment: {np.mean(track_counts):.1f}
    Std Tracks/Experiment: {np.std(track_counts):.1f}
    Min Tracks: {np.min(track_counts)}
    Max Tracks: {np.max(track_counts)}
    
    Note: Additional statistics (turn rate, latency, etc.)
    require processed trajectory data.
    """
    ax3.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             family='monospace')
    
    # Plot 4: Box plot of track counts
    ax4 = axes[1, 1]
    ax4.boxplot(track_counts, vert=True)
    ax4.set_ylabel('Number of Tracks')
    ax4.set_title('Track Count Distribution (Box Plot)')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'ESET Statistics: {eset_dir.name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_experiment_composite(h5_file: Path, output_path: Path,
                              frame_rate: float = 10.0) -> None:
    """
    Plot experiment-level composite (MATLAB reference style).
    
    Top panel: LED stimulus profile
    Bottom panel: Mean turn rate with variability
    
    Args:
        h5_file: Path to H5 file
        output_path: Path to save plot
        frame_rate: Frame rate in Hz
    """
    if HAS_CINNAMOROLL:
        set_cinnamoroll_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    with h5py.File(h5_file, 'r') as f:
        # Get LED values
        led1_vals = f['global_quantities/led1Val/yData'][:]
        
        # Get ETI
        if 'eti' in f:
            time_axis = f['eti'][:]
        else:
            time_axis = np.arange(len(led1_vals)) / frame_rate
        
        # Top panel: LED stimulus (Fictive CO2 equivalent)
        ax1 = axes[0]
        ax1.fill_between(time_axis, 0, led1_vals / np.max(led1_vals), 
                         alpha=0.7, color='red', label='Fictive CO2')
        ax1.set_ylabel('Fictive CO2', fontsize=12)
        ax1.set_ylim(0, 1.1)
        ax1.set_title(f'Experiment: {h5_file.stem}', fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Bottom panel: Turn rate (placeholder - requires processed data)
        ax2 = axes[1]
        ax2.text(0.5, 0.5, 'Turn rate data requires processed trajectory data.\n'
                           'This will be populated after data processing.',
                 transform=ax2.transAxes, ha='center', va='center',
                 fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_xlabel('Time in cycle (s)', fontsize=12)
        ax2.set_ylabel('Turn rate (min⁻¹)', fontsize=12)
        ax2.set_ylim(0, 6)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to generate all visualizations."""
    print("Visualization generation script")
    print("=" * 60)
    print("\nNote: This script requires:")
    print("  1. H5 files with aligned LED values")
    print("  2. Processed trajectory data for turn rate calculations")
    print("  3. LED alignment integration (currently P0 blocking task)")
    print("\nStructure will be created, but plots requiring processed data")
    print("will be placeholders until data processing is complete.")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()


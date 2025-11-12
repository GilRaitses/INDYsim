#!/usr/bin/env python3
"""Analyze LED pattern to understand period structure."""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_led_pattern(h5_file: Path, frame_rate: float = 10.0):
    """Analyze LED pattern to detect pulse duration and period."""
    with h5py.File(h5_file, 'r') as f:
        gq = f['global_quantities']
        led1 = gq['led1Val']['yData'][:]
        
        # Create time axis
        if 'eti' in f:
            times = f['eti'][:]
        else:
            times = np.arange(len(led1)) / frame_rate
    
    # Detect ON/OFF transitions
    threshold = np.max(led1) * 0.1
    is_on = led1 > threshold
    
    # Find transitions
    on_transitions = np.where(np.diff(is_on.astype(int)) > 0)[0]  # OFF->ON
    off_transitions = np.where(np.diff(is_on.astype(int)) < 0)[0]  # ON->OFF
    
    print(f"LED Pattern Analysis: {h5_file.name}")
    print("="*70)
    print(f"Total frames: {len(led1)}")
    print(f"Time range: {times[0]:.2f} to {times[-1]:.2f} seconds")
    print(f"ON transitions: {len(on_transitions)}")
    print(f"OFF transitions: {len(off_transitions)}")
    
    if len(on_transitions) >= 2:
        # Calculate intervals between ON transitions (full period)
        on_intervals = np.diff(times[on_transitions])
        print(f"\nON-to-ON intervals (full period):")
        print(f"  Mean: {np.mean(on_intervals):.2f}s")
        print(f"  Median: {np.median(on_intervals):.2f}s")
        print(f"  Std: {np.std(on_intervals):.2f}s")
        print(f"  Range: [{np.min(on_intervals):.2f}, {np.max(on_intervals):.2f}]s")
    
    if len(on_transitions) > 0 and len(off_transitions) > 0:
        # Calculate pulse durations (ON time)
        pulse_durations = []
        for i, on_idx in enumerate(on_transitions[:10]):  # First 10 pulses
            off_after = off_transitions[off_transitions > on_idx]
            if len(off_after) > 0:
                pulse_dur = times[off_after[0]] - times[on_idx]
                pulse_durations.append(pulse_dur)
        
        if pulse_durations:
            print(f"\nPulse durations (ON time):")
            print(f"  Mean: {np.mean(pulse_durations):.2f}s")
            print(f"  Median: {np.median(pulse_durations):.2f}s")
            print(f"  Range: [{np.min(pulse_durations):.2f}, {np.max(pulse_durations):.2f}]s")
        
        # Calculate inter-pulse intervals (OFF time)
        inter_pulse_intervals = []
        for i, off_idx in enumerate(off_transitions[:10]):  # First 10 OFF transitions
            on_after = on_transitions[on_transitions > off_idx]
            if len(on_after) > 0:
                interval = times[on_after[0]] - times[off_idx]
                inter_pulse_intervals.append(interval)
        
        if inter_pulse_intervals:
            print(f"\nInter-pulse intervals (OFF time):")
            print(f"  Mean: {np.mean(inter_pulse_intervals):.2f}s")
            print(f"  Median: {np.median(inter_pulse_intervals):.2f}s")
            print(f"  Range: [{np.min(inter_pulse_intervals):.2f}, {np.max(inter_pulse_intervals):.2f}]s")
    
    # Show first few transitions
    print(f"\nFirst 5 ON transitions (times):")
    for i, idx in enumerate(on_transitions[:5]):
        print(f"  {i+1}: Frame {idx}, Time {times[idx]:.2f}s, LED value {led1[idx]:.1f}")
    
    print(f"\nFirst 5 OFF transitions (times):")
    for i, idx in enumerate(off_transitions[:5]):
        print(f"  {i+1}: Frame {idx}, Time {times[idx]:.2f}s, LED value {led1[idx]:.1f}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        h5_file = Path(sys.argv[1])
    else:
        # Test File 3 that failed period detection
        h5_file = Path('data/h5_files/GMR61@GMR61_T_Re_Sq_50to250PWM_30#C_Bl_7PWM_202510291435.h5')
    analyze_led_pattern(h5_file)


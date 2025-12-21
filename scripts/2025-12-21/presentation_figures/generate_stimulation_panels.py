#!/usr/bin/env python3
"""
Generate Individual Panels for Stimulation Protocol Figure - PRESENTATION VERSION

Outputs 4 separate PDFs, one per design, with larger fonts for slides.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("/Users/gilraitses/INDYsim_project/phenotyping_followup/presentation/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Larger fonts for presentation
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Avenir', 'Helvetica Neue', 'Arial']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.linewidth'] = 1.5

# Time axis - full cycle
dt = 0.01
t = np.arange(-2, 32, dt)

# Colors
RED = '#DC2626'
RED_FILL = '#FCA5A5'
GRAY = '#6B7280'
GREEN = '#22C55E'

# LED1 patterns
def led1_current_design(t):
    signal = np.zeros_like(t)
    signal[(t >= 0) & (t < 10)] = 1
    return signal

def led1_burst_design(t):
    signal = np.zeros_like(t)
    for i in range(10):
        start = i * 1.0
        end = start + 0.5
        signal[(t >= start) & (t < end)] = 1
    return signal

def led1_medium_design(t):
    signal = np.zeros_like(t)
    pulse_times = [(0, 1), (2.5, 3.5), (5, 6), (7.5, 8.5)]
    for start, end in pulse_times:
        signal[(t >= start) & (t < end)] = 1
    return signal

def led1_long_design(t):
    signal = np.zeros_like(t)
    pulse_times = [(0, 2), (5, 7)]
    for start, end in pulse_times:
        signal[(t >= start) & (t < end)] = 1
    return signal


def generate_single_panel(panel_id, title, led_func, stats, is_recommended=False):
    """Generate a single stimulation protocol panel."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    signal = led_func(t)
    
    # Plot signal
    ax.fill_between(t, 0, signal * 0.8, where=signal > 0, 
                    color=RED_FILL, alpha=0.9, step='pre')
    ax.step(t, signal * 0.8, where='pre', color=RED, linewidth=3)
    
    # Shade ON and OFF periods
    ax.axvspan(0, 10, alpha=0.08, color=RED, zorder=0)
    ax.axvspan(10, 30, alpha=0.04, color='gray', zorder=0)
    
    # Period labels
    ax.text(5, 0.95, 'ON PERIOD (10s)', ha='center', fontsize=12, 
            color=RED, fontweight='bold')
    ax.text(20, 0.95, 'OFF PERIOD (20s)', ha='center', fontsize=12, 
            color=GRAY, fontweight='bold')
    
    # Vertical markers
    for x in [0, 10, 30]:
        ax.axvline(x, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Title
    ax.set_title(title, loc='left', fontweight='bold', fontsize=18)
    
    # Stats box
    stats_text = f"Events: {stats['events']}\nRMSE: {stats['rmse']}\nPower: {stats['power']}\nDuty: {stats['duty']}"
    ax.text(31.5, 0.4, stats_text, fontsize=12, ha='left', va='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F3F4F6', edgecolor='#D1D5DB'))
    
    # Note
    note_color = GREEN if is_recommended else GRAY
    ax.text(0, -0.25, stats['note'], fontsize=12, color=note_color, style='italic')
    
    # Formatting
    ax.set_xlim(-1, 31)
    ax.set_ylim(-0.4, 1.15)
    ax.set_yticks([0, 0.8])
    ax.set_yticklabels(['OFF', 'ON'], fontsize=14)
    ax.set_xlabel('Time from Cycle Start (seconds)', fontsize=14, fontweight='bold')
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / f'stimulation_panel_{panel_id}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / f'stimulation_panel_{panel_id}.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / f'stimulation_panel_{panel_id}.pdf'}")
    plt.close()


def main():
    designs = [
        ('A', 'Current Design: Continuous 10s', led1_current_design, 
         {'events': '1.9/track', 'rmse': '0.108s', 'power': '42%', 'duty': '100%',
          'note': 'Low power, high bias. Most time in inhibitory regime.'}, False),
        ('B', 'Recommended: Burst 10×0.5s', led1_burst_design,
         {'events': '14.9/track', 'rmse': '0.036s', 'power': '100%', 'duty': '50%',
          'note': '8× more events from repeated excitatory onset sampling.'}, True),
        ('C', 'Alternative: Medium 4×1s', led1_medium_design,
         {'events': '6.0/track', 'rmse': '0.048s', 'power': '100%', 'duty': '40%',
          'note': 'Moderate improvement. Longer pulses, fewer onsets.'}, False),
        ('D', 'Alternative: Long 2×2s', led1_long_design,
         {'events': '3.1/track', 'rmse': '0.077s', 'power': '100%', 'duty': '40%',
          'note': 'Minimal improvement. Too few onsets to sample kernel.'}, False),
    ]
    
    print("Generating individual stimulation panels...")
    for panel_id, title, led_func, stats, is_recommended in designs:
        generate_single_panel(panel_id, title, led_func, stats, is_recommended)
    
    print(f"\nAll panels saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Stimulation Protocol Schematic - RED LED ONLY
==============================================
Shows the full 30s cycle (10s ON + 20s OFF) with:
- LED1 (red): Optogenetic stimulation designs compared

Blue LED is NOT shown because the analysis focused exclusively on 
red LED pulse timing. Blue LED impact was not analyzed.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("/Users/gilraitses/INDYsim_project/phenotyping_followup/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set up fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Avenir', 'Helvetica Neue', 'DejaVu Sans', 'Arial']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2

# Time axis - full cycle
dt = 0.01
t = np.arange(-2, 32, dt)

# LED1 patterns (red optogenetic)
def led1_current_design(t):
    """Current design: 10s continuous ON, 20s OFF"""
    signal = np.zeros_like(t)
    signal[(t >= 0) & (t < 10)] = 1
    return signal

def led1_burst_design(t):
    """Burst: 10 x 0.5s pulses with 0.5s gaps within 10s window"""
    signal = np.zeros_like(t)
    for i in range(10):
        start = i * 1.0
        end = start + 0.5
        signal[(t >= start) & (t < end)] = 1
    return signal

def led1_medium_design(t):
    """Medium: 4 x 1s pulses with 1.5s gaps"""
    signal = np.zeros_like(t)
    pulse_times = [(0, 1), (2.5, 3.5), (5, 6), (7.5, 8.5)]
    for start, end in pulse_times:
        signal[(t >= start) & (t < end)] = 1
    return signal

def led1_long_design(t):
    """Long: 2 x 2s pulses with 3s gaps"""
    signal = np.zeros_like(t)
    pulse_times = [(0, 2), (5, 7)]
    for start, end in pulse_times:
        signal[(t >= start) & (t < end)] = 1
    return signal

# Design info with statistics
designs = [
    ('A. Current: Continuous 10s', led1_current_design, 
     {'events': '1.9/track', 'rmse': '0.108s', 'power': '42%', 'duty': '100%',
      'note': 'Low power, high bias. Most time in inhibitory regime.'}),
    ('B. Recommended: Burst 10×0.5s', led1_burst_design,
     {'events': '14.9/track', 'rmse': '0.036s', 'power': '100%', 'duty': '50%',
      'note': '8× more events from repeated excitatory onset sampling.'}),
    ('C. Alternative: Medium 4×1s', led1_medium_design,
     {'events': '6.0/track', 'rmse': '0.048s', 'power': '100%', 'duty': '40%',
      'note': 'Moderate improvement. Longer pulses, fewer onsets.'}),
    ('D. Alternative: Long 2×2s', led1_long_design,
     {'events': '3.1/track', 'rmse': '0.077s', 'power': '100%', 'duty': '40%',
      'note': 'Minimal improvement. Too few onsets to sample kernel.'}),
]

# Colors
RED = '#DC2626'
RED_FILL = '#FEE2E2'
GRAY = '#6B7280'

# Create figure
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

for ax_idx, (ax, (title, led_func, stats)) in enumerate(zip(axes, designs)):
    # Get signal
    signal = led_func(t)
    
    # Plot signal as filled region
    ax.fill_between(t, 0, signal * 0.8, where=signal > 0, 
                    color=RED_FILL, alpha=0.8, step='pre')
    ax.step(t, signal * 0.8, where='pre', color=RED, linewidth=2.5)
    
    # Shade ON and OFF periods
    ax.axvspan(0, 10, alpha=0.06, color=RED, zorder=0)
    ax.axvspan(10, 30, alpha=0.03, color='gray', zorder=0)
    
    # Period labels (only on first panel)
    if ax_idx == 0:
        ax.text(5, 0.95, 'ON PERIOD (10s)', ha='center', fontsize=10, 
                color=RED, fontweight='bold')
        ax.text(20, 0.95, 'OFF PERIOD (20s)', ha='center', fontsize=10, 
                color=GRAY, fontweight='bold')
    
    # Vertical markers
    for x in [0, 10, 30]:
        ax.axvline(x, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    # Title
    ax.set_title(title, loc='left', fontweight='bold', fontsize=12)
    
    # Stats box on right
    stats_text = f"Events: {stats['events']}\nRMSE: {stats['rmse']}\nPower: {stats['power']}\nDuty: {stats['duty']}"
    ax.text(31.5, 0.4, stats_text, fontsize=9, ha='left', va='center',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F3F4F6', edgecolor='#D1D5DB'))
    
    # Note below signal
    note_color = '#059669' if 'Recommended' in title else '#6B7280'
    ax.text(0, -0.2, stats['note'], fontsize=9, color=note_color, style='italic')
    
    # Formatting
    ax.set_xlim(-1, 31)
    ax.set_ylim(-0.35, 1.1)
    ax.set_yticks([0, 0.8])
    ax.set_yticklabels(['OFF', 'ON'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# X-axis
axes[-1].set_xlabel('Time from Cycle Start (seconds)', fontsize=12, fontweight='bold')
axes[-1].set_xticks([0, 5, 10, 15, 20, 25, 30])

# Title
fig.suptitle('Red LED Stimulation Designs: Full 30-Second Cycle', 
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.02, 0.88, 0.96])

# Save
output_path = OUTPUT_DIR / 'fig_stimulation_schematic.png'
plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

output_pdf = OUTPUT_DIR / 'fig_stimulation_schematic.pdf'
plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_pdf}")

plt.close()
print("\nFigure shows RED LED timing only. Blue LED not shown.")

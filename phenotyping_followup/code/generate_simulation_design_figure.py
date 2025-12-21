#!/usr/bin/env python3
"""
Generate figure for section 2.1: Simulated Trajectory Generation
Shows trajectory visualizations and simulation parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from cinnamoroll_palette import COLORS

# Output directory
OUTPUT_DIR = Path("/Users/gilraitses/INDYsim_project/phenotyping_followup/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configure matplotlib
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.weight'] = 'ultralight'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['text.color'] = COLORS['text']
plt.rcParams['axes.labelcolor'] = COLORS['text']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5

def create_figure():
    """Create the simulation design figure."""
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3, left=0.06, right=0.95, top=0.90, bottom=0.10)
    
    # Conditions
    conditions = {
        '0-250 Constant': {'intensity': 'Low (0-250)', 'pattern': 'Constant', 'n': 75, 'color': COLORS['primary']},
        '0-250 Cycling': {'intensity': 'Low (0-250)', 'pattern': 'Cycling', 'n': 75, 'color': COLORS['primary_light']},
        '50-250 Constant': {'intensity': 'High (50-250)', 'pattern': 'Constant', 'n': 75, 'color': COLORS['secondary']},
        '50-250 Cycling': {'intensity': 'High (50-250)', 'pattern': 'Cycling', 'n': 75, 'color': COLORS['accent']}
    }
    
    # Panel A: Example Trajectory Paths (one track per condition)
    ax1 = fig.add_subplot(gs[0])
    ax1.text(-0.12, 1.08, 'A', transform=ax1.transAxes,
            fontsize=14, fontweight='ultralight', fontfamily='Avenir', 
            color=COLORS['text'], va='top', ha='left')
    
    # Generate example trajectory paths (simplified - random walk with turns)
    np.random.seed(42)
    duration = 600  # 10 minutes
    dt = 0.05  # 20 Hz
    n_points = int(duration / dt)
    
    # Create 4 example trajectories, one per condition
    for idx, (cond_name, cond) in enumerate(conditions.items()):
        # Simple random walk model with occasional turns
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        angle = 0
        
        # Simulate LED cycles
        led_onsets = np.arange(0, duration, 30)
        
        for i in range(1, n_points):
            t = i * dt
            
            # Check if LED is ON (affects turn probability)
            led_on = any((t >= onset) and (t < onset + 10) for onset in led_onsets)
            
            # Turn probability higher during LED ON for some conditions
            if led_on and cond_name in ['0-250 Constant', '0-250 Cycling']:
                turn_prob = 0.02
            else:
                turn_prob = 0.01
            
            # Random turn
            if np.random.rand() < turn_prob:
                angle += np.random.uniform(-np.pi/3, np.pi/3)
            
            # Move forward
            speed = 0.1  # mm/s
            x[i] = x[i-1] + speed * dt * np.cos(angle)
            y[i] = y[i-1] + speed * dt * np.sin(angle)
        
        # Plot trajectory (sample every 10th point for clarity)
        sample = slice(0, None, 10)
        ax1.plot(x[sample], y[sample], color=cond['color'], alpha=0.7, 
                linewidth=1.5, label=cond_name)
        
        # Mark start
        ax1.scatter(x[0], y[0], color=cond['color'], s=40, marker='o', 
                   edgecolors='white', linewidth=1, zorder=10)
        # Mark end
        ax1.scatter(x[-1], y[-1], color=cond['color'], s=40, marker='s', 
                   edgecolors='white', linewidth=1, zorder=10)
    
    ax1.set_xlabel('X position (mm)', fontfamily='Avenir', fontweight='ultralight')
    ax1.set_ylabel('Y position (mm)', fontfamily='Avenir', fontweight='ultralight')
    ax1.set_title('Example Simulated Trajectories', fontfamily='Avenir', fontweight='ultralight', pad=10)
    ax1.legend(loc='upper right', prop={'family': 'Avenir', 'weight': 'ultralight'}, fontsize=8)
    ax1.set_aspect('equal')
    ax1.set_facecolor('#D0D0D0')  # Darker gray background
    ax1.grid(True, alpha=0.4, linestyle='--', color='white')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel B: Trajectory Density Heatmap (showing many tracks overlaid)
    ax2 = fig.add_subplot(gs[1])
    ax2.text(-0.12, 1.08, 'B', transform=ax2.transAxes,
            fontsize=14, fontweight='ultralight', fontfamily='Avenir', 
            color=COLORS['text'], va='top', ha='left')
    
    # Generate multiple trajectories and create density heatmap
    np.random.seed(42)
    n_tracks = 20  # Sample of tracks
    duration = 600
    dt = 0.05
    n_points = int(duration / dt)
    
    all_x = []
    all_y = []
    
    for track_id in range(n_tracks):
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        angle = np.random.uniform(0, 2*np.pi)
        
        led_onsets = np.arange(0, duration, 30)
        
        for i in range(1, n_points):
            t = i * dt
            led_on = any((t >= onset) and (t < onset + 10) for onset in led_onsets)
            
            if led_on:
                turn_prob = 0.02
            else:
                turn_prob = 0.01
            
            if np.random.rand() < turn_prob:
                angle += np.random.uniform(-np.pi/3, np.pi/3)
            
            speed = 0.1
            x[i] = x[i-1] + speed * dt * np.cos(angle)
            y[i] = y[i-1] + speed * dt * np.sin(angle)
        
        # Sample points for density
        sample = slice(0, None, 50)
        all_x.extend(x[sample])
        all_y.extend(y[sample])
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(all_x, all_y, bins=30)
    H = H.T  # Transpose for imshow
    
    # Plot density heatmap with darker background
    im = ax2.imshow(H, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                    origin='lower', cmap='Blues', alpha=0.8, aspect='auto')
    ax2.set_facecolor('#D0D0D0')  # Darker gray background
    
    ax2.set_xlabel('X position (mm)', fontfamily='Avenir', fontweight='ultralight')
    ax2.set_ylabel('Y position (mm)', fontfamily='Avenir', fontweight='ultralight')
    ax2.set_title('Trajectory Density (20 tracks)', fontfamily='Avenir', fontweight='ultralight', pad=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, linestyle='--', color='white')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Track density', fontfamily='Avenir', fontweight='ultralight', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    for label in cbar.ax.get_yticklabels():
        label.set_fontfamily('Avenir')
        label.set_fontweight('ultralight')
    
    # Panel C: Condition Comparison
    ax3 = fig.add_subplot(gs[2])
    ax3.text(-0.12, 1.08, 'C', transform=ax3.transAxes,
            fontsize=14, fontweight='ultralight', fontfamily='Avenir', 
            color=COLORS['text'], va='top', ha='left')
    
    # Show condition differences (simplified representation)
    cond_names = list(conditions.keys())
    cond_colors = [conditions[c]['color'] for c in cond_names]
    
    # Create a simple bar chart showing relative amplitude (from code)
    amplitudes = [1.005, 1.157, 0.340, 0.492]  # From generate_simulated_tracks_for_phenotyping.py
    bars = ax3.bar(range(len(cond_names)), amplitudes, color=cond_colors,
                   edgecolor=COLORS['border'], linewidth=0.5, alpha=0.7)
    
    # Add value labels
    for i, (bar, amp) in enumerate(zip(bars, amplitudes)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{amp:.3f}', ha='center', va='bottom',
                fontsize=9, fontfamily='Avenir', fontweight='ultralight')
    
    ax3.set_xticks(range(len(cond_names)))
    ax3.set_xticklabels([c.replace(' ', '\n') for c in cond_names],
                        fontfamily='Avenir', fontweight='ultralight', fontsize=9)
    ax3.set_ylabel('Relative Amplitude', fontfamily='Avenir', fontweight='ultralight')
    ax3.set_title('Condition Amplitude Modulation', fontfamily='Avenir', fontweight='ultralight', pad=10)
    ax3.set_ylim(0, max(amplitudes) * 1.2)
    ax3.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Add figure title
    fig.suptitle('Simulated Trajectory Generation', fontsize=14, fontweight='ultralight',
                 fontfamily='Avenir', y=0.98, color=COLORS['text'])
    
    plt.savefig(OUTPUT_DIR / 'fig_simulation_design.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none', dpi=300)
    plt.savefig(OUTPUT_DIR / 'fig_simulation_design.png', bbox_inches='tight',
                facecolor='white', edgecolor='none', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {OUTPUT_DIR / 'fig_simulation_design.pdf'}")


if __name__ == '__main__':
    create_figure()

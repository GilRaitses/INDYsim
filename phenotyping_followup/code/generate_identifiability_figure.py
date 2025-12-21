#!/usr/bin/env python3
"""
Generate Figure 2: The Data Sparsity Problem (v3 - with empirical sweep data)

Updated to use actual empirical findings from the design × kernel sweep:
- Replaces heuristic "100 events required" with design-specific empirical thresholds
- Shows that burst design achieves 4× lower bias with same event count
- Explains 6:1 ratio in terms of Fisher Information, not just count
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from pathlib import Path
import pandas as pd
import json

# Output
OUTPUT_DIR = Path("/Users/gilraitses/INDYsim_project/phenotyping_followup/figures")
RESULTS_DIR = Path("/Users/gilraitses/INDYsim_project/phenotyping_followup/results/phenotyping_analysis_v2")
SWEEP_FILE = Path("/Users/gilraitses/INDYsim_project/scripts/2025-12-19/phenotyping_experiments/15_identification_analysis/results/design_kernel_sweep/sweep_results.json")

# Font setup
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Avenir', 'Helvetica Neue', 'Arial']
plt.rcParams['font.size'] = 11
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Cinnamoroll color palette (soft pastels)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cinnamoroll_palette import COLORS as CINNAMOROLL_COLORS
# Map semantic names to cinnamoroll palette
COLORS = {
    'data': CINNAMOROLL_COLORS['primary'],
    'data_light': CINNAMOROLL_COLORS['primary_light'],
    'success': CINNAMOROLL_COLORS['success'],
    'failure': CINNAMOROLL_COLORS['failure'],
    'population': CINNAMOROLL_COLORS['population'],
    'text': CINNAMOROLL_COLORS['text'],
    'ci': CINNAMOROLL_COLORS['ci'],
    'burst': CINNAMOROLL_COLORS['recommended'],
}

def load_sweep_data():
    """Load the design × kernel sweep results."""
    if SWEEP_FILE.exists():
        with open(SWEEP_FILE) as f:
            sweep = json.load(f)
        # Filter for our kernel (A/B = 0.125)
        our_kernel = [r for r in sweep if r['ab_ratio'] == 0.125]
        return our_kernel
    return None

def load_data():
    """Load analysis results."""
    data = {}
    
    fits_path = RESULTS_DIR / 'individual_fits.csv'
    if fits_path.exists():
        data['fits'] = pd.read_csv(fits_path)
        print(f"Loaded {len(data['fits'])} individual fits")
    
    posteriors_path = RESULTS_DIR / 'posteriors.csv'
    if posteriors_path.exists():
        data['posteriors'] = pd.read_csv(posteriors_path)
        print(f"Loaded {len(data['posteriors'])} posteriors")
    
    data['sweep'] = load_sweep_data()
    if data['sweep']:
        print(f"Loaded sweep data for {len(data['sweep'])} designs")
    
    return data

def generate_figure(data):
    """Generate the 4-panel Data Sparsity Problem figure with empirical sweep data."""
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('The Identifiability Problem', fontsize=16, fontweight='bold', y=0.98)
    
    fits = data.get('fits', None)
    posteriors = data.get('posteriors', None)
    sweep = data.get('sweep', None)
    
    # Use simulated data if real data not available
    if fits is None:
        np.random.seed(42)
        n_tracks = 256
        fits = pd.DataFrame({
            'n_events': np.random.negative_binomial(5, 0.2, n_tracks) + 10,
            'tau1': np.abs(np.random.normal(0.63, 0.8, n_tracks)) + 0.1,
        })
        fits['tau1'] = np.clip(fits['tau1'], 0.1, 5.0)
    
    if posteriors is None:
        np.random.seed(42)
        n_tracks = len(fits)
        posteriors = pd.DataFrame({
            'tau1_mean': np.random.normal(0.63, 0.15, n_tracks),
            'tau1_ci_low': np.random.normal(0.63, 0.15, n_tracks) - 0.3,
            'tau1_ci_high': np.random.normal(0.63, 0.15, n_tracks) + 0.3,
        })
        posteriors['tau1_ci_width'] = posteriors['tau1_ci_high'] - posteriors['tau1_ci_low']
    
    pop_tau1 = 0.63
    
    # Get empirical values from sweep
    if sweep:
        continuous = next((r for r in sweep if 'Continuous' in r['design']), None)
        burst = next((r for r in sweep if 'Burst' in r['design']), None)
    else:
        # Fallback to hardcoded values from sweep
        continuous = {'bias': 0.61, 'rmse': 0.71, 'mean_events': 16.3, 'fisher': 0.29}
        burst = {'bias': 0.14, 'rmse': 0.38, 'mean_events': 16.9, 'fisher': 2.88}
    
    # ========== Panel A: Design Comparison (replaces simple event histogram) ==========
    ax1 = axes[0, 0]
    ax1.text(-0.12, 1.05, 'A', transform=ax1.transAxes, fontsize=14, 
             fontweight='bold', va='top')
    
    # Bar chart comparing designs
    designs = ['Continuous\n10s ON', 'Burst\n10×0.5s']
    bias_vals = [continuous['bias'], burst['bias']]
    rmse_vals = [continuous['rmse'], burst['rmse']]
    
    x = np.arange(len(designs))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, bias_vals, width, label='Bias', color=COLORS['failure'], alpha=0.8)
    bars2 = ax1.bar(x + width/2, rmse_vals, width, label='RMSE', color=COLORS['data'], alpha=0.8)
    
    ax1.set_ylabel(r'Error in $\tau_1$ (seconds)', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(designs)
    ax1.set_title('Same Events, Different Information\n(~17 events each)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, val in zip(bars1, bias_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'+{val:.2f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, rmse_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_ylim(0, 0.95)
    
    # ========== Panel B: The Information Problem (replaces simple ratio) ==========
    ax2 = axes[0, 1]
    ax2.text(-0.12, 1.05, 'B', transform=ax2.transAxes, fontsize=14,
             fontweight='bold', va='top')
    ax2.axis('off')
    
    ax2.set_title('The Information Problem', fontsize=12, fontweight='bold')
    
    # Show Fisher Information comparison
    ax2.text(0.5, 0.88, r'Fisher Information for $\tau_1$', fontsize=14, ha='center',
             fontweight='bold', color=COLORS['text'], transform=ax2.transAxes)
    
    ax2.text(0.5, 0.72, f'Continuous: {continuous["fisher"]:.2f}', fontsize=14, ha='center',
             color=COLORS['failure'], transform=ax2.transAxes)
    
    # Use darker green for better visibility
    green_color = COLORS.get('success_dark', '#4A9E4A')  # Darker green
    ax2.text(0.5, 0.58, f'Burst: {burst["fisher"]:.2f}', fontsize=14, ha='center',
             color=green_color, transform=ax2.transAxes)
    
    ax2.text(0.5, 0.42, '_______________', fontsize=14, ha='center',
             color=COLORS['text'], transform=ax2.transAxes)
    
    ratio = burst["fisher"] / continuous["fisher"]
    ax2.text(0.5, 0.28, f'Burst extracts {ratio:.0f}× more info', fontsize=14, ha='center',
             color=green_color, fontweight='bold', transform=ax2.transAxes)
    
    ax2.text(0.5, 0.12, 'from the same number of events', fontsize=12, ha='center',
             color=COLORS['text'], transform=ax2.transAxes)
    
    # ========== Panel C: Bias by Design (replaces shrinkage scatter) ==========
    ax3 = axes[1, 0]
    ax3.text(-0.12, 1.05, 'C', transform=ax3.transAxes, fontsize=14,
             fontweight='bold', va='top')
    
    # Show what MLE returns for each design
    true_tau1 = 0.63
    
    ax3.axhline(y=true_tau1, color=COLORS['success'], linewidth=2, linestyle='--', label=r'True $\tau_1$')
    
    x_pos = [0, 1]
    fitted_means = [continuous['fitted_mean'], burst['fitted_mean']]
    fitted_stds = [continuous.get('fitted_std', 0.37), burst.get('fitted_std', 0.35)]
    
    ax3.bar(x_pos, fitted_means, yerr=fitted_stds, width=0.6, 
            color=[COLORS['failure'], COLORS['burst']], alpha=0.8, capsize=5,
            error_kw={'elinewidth': 2, 'capthick': 2})
    
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['Continuous', 'Burst'])
    ax3.set_ylabel(r'Fitted $\tau_1$ (seconds)', fontsize=11)
    ax3.set_title('MLE Recovery by Design', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', framealpha=0.95)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.set_ylim(0, 1.8)
    
    # Add bias annotations
    for i, (fm, bias) in enumerate(zip(fitted_means, bias_vals)):
        ax3.annotate(f'Bias: +{bias:.2f}s', (x_pos[i], fm + fitted_stds[i] + 0.08),
                    ha='center', fontsize=10, color=COLORS['text'])
    
    # ========== Panel D: Why Continuous Fails ==========
    ax4 = axes[1, 1]
    ax4.text(-0.12, 1.05, 'D', transform=ax4.transAxes, fontsize=14,
             fontweight='bold', va='top')
    ax4.axis('off')
    
    ax4.set_title('Why Continuous Design Fails', fontsize=12, fontweight='bold')
    
    # Explanation text - use ASCII arrows and spell out tau
    explanations = [
        ("Kernel is inhibition-dominated (B/A = 8)", COLORS['text'], 'normal'),
        ("", COLORS['text'], 'normal'),
        ("~80% of events occur during LED-OFF", COLORS['text'], 'normal'),
        ("  -> No tau1 information", COLORS['failure'], 'normal'),
        ("", COLORS['text'], 'normal'),
        ("Remaining ~20% mostly after t > 0.5s", COLORS['text'], 'normal'),
        ("  -> Inhibition dominates, tau1 unidentifiable", COLORS['failure'], 'normal'),
        ("", COLORS['text'], 'normal'),
        ("Burst design samples multiple", COLORS['success'], 'bold'),
        ("early excitatory windows", COLORS['success'], 'normal'),
    ]
    
    green_color = COLORS.get('success_dark', '#4A9E4A')  # Darker green for visibility
    y_start = 0.90
    for i, (text, color, weight) in enumerate(explanations):
        if color == COLORS['success']:
            # Replace with darker green
            color = green_color
        ax4.text(0.1, y_start - i * 0.085, text, fontsize=11, ha='left',
                color=color, fontweight=weight, transform=ax4.transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    fig.savefig(OUTPUT_DIR / 'fig2_identifiability_v3.png', dpi=200, bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'fig2_identifiability_v3.pdf', bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'fig2_identifiability_v3.png'}")
    
    plt.close()

if __name__ == '__main__':
    data = load_data()
    generate_figure(data)


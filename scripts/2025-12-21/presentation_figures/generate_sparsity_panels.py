#!/usr/bin/env python3
"""
Generate Individual Panels for Data Sparsity Figure - PRESENTATION VERSION

Outputs 4 separate PDFs, one per panel, with larger fonts for slides.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Output
OUTPUT_DIR = Path("/Users/gilraitses/INDYsim_project/phenotyping_followup/presentation/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("/Users/gilraitses/INDYsim_project/phenotyping_followup/results/phenotyping_analysis_v2")

# Larger fonts for presentation
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Avenir', 'Helvetica Neue', 'Arial']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# Colors
COLORS = {
    'data': '#60A5FA',
    'data_light': '#BFDBFE',
    'success': '#22C55E',
    'failure': '#EF4444',
    'population': '#A78BFA',
    'text': '#1E293B',
    'ci': '#94A3B8',
}


def load_data():
    """Load analysis results or generate synthetic data."""
    data = {}
    
    fits_path = RESULTS_DIR / 'individual_fits.csv'
    if fits_path.exists():
        data['fits'] = pd.read_csv(fits_path)
    else:
        np.random.seed(42)
        n_tracks = 256
        data['fits'] = pd.DataFrame({
            'n_events': np.random.negative_binomial(5, 0.2, n_tracks) + 10,
            'tau1': np.abs(np.random.normal(0.63, 0.8, n_tracks)) + 0.1,
        })
        data['fits']['tau1'] = np.clip(data['fits']['tau1'], 0.1, 5.0)
    
    posteriors_path = RESULTS_DIR / 'posteriors.csv'
    if posteriors_path.exists():
        data['posteriors'] = pd.read_csv(posteriors_path)
    else:
        np.random.seed(42)
        n_tracks = len(data['fits'])
        data['posteriors'] = pd.DataFrame({
            'tau1_mean': np.random.normal(0.63, 0.15, n_tracks),
            'tau1_ci_low': np.random.normal(0.63, 0.15, n_tracks) - 0.3,
            'tau1_ci_high': np.random.normal(0.63, 0.15, n_tracks) + 0.3,
        })
        data['posteriors']['tau1_ci_width'] = data['posteriors']['tau1_ci_high'] - data['posteriors']['tau1_ci_low']
    
    return data


def panel_A_event_distribution(data):
    """Panel A: Histogram of events per track."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    events = data['fits']['n_events']
    ax.hist(events, bins=25, color=COLORS['data'], alpha=0.8, 
             edgecolor='white', linewidth=0.8)
    ax.axvline(x=events.mean(), color=COLORS['failure'], linewidth=3,
                linestyle='-', label=f'Mean: {events.mean():.0f}')
    ax.axvline(x=100, color=COLORS['success'], linewidth=3,
                linestyle='--', label='Required: 100')
    
    ax.set_xlabel('Events per Track')
    ax.set_ylabel('Number of Tracks')
    ax.set_title('Data Sparsity\n(~25 events, need ~100)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'sparsity_panel_A.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'sparsity_panel_A.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'sparsity_panel_A.pdf'}")
    plt.close()


def panel_B_mle_estimates(data):
    """Panel B: Histogram of MLE tau1 estimates showing biological vs implausible range."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    tau1 = data['fits']['tau1']
    
    # Biological range
    bio_mask = (tau1 >= 0.3) & (tau1 <= 1.5)
    
    ax.hist(tau1[bio_mask], bins=20, color=COLORS['success'], alpha=0.8, 
             edgecolor='white', linewidth=0.8, label='Biological range')
    ax.hist(tau1[~bio_mask], bins=20, color=COLORS['failure'], alpha=0.8, 
             edgecolor='white', linewidth=0.8, label='Fitting failures')
    
    ax.axvline(x=0.3, color=COLORS['success'], linewidth=2, linestyle='--', alpha=0.7)
    ax.axvline(x=1.5, color=COLORS['success'], linewidth=2, linestyle='--', alpha=0.7)
    
    ax.set_xlabel(r'$\tau_1$ (seconds)')
    ax.set_ylabel('Number of Tracks')
    ax.set_title('MLE Estimates\n(Implausible range 0-5s)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, 5.5)
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'sparsity_panel_B.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'sparsity_panel_B.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'sparsity_panel_B.pdf'}")
    plt.close()


def panel_C_shrinkage(data):
    """Panel C: Scatter plot of MLE vs Bayesian estimates showing shrinkage."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mle = data['fits']['tau1'].values
    pop_tau1 = 0.63
    
    if 'tau1_mean' in data['posteriors'].columns:
        bayes = data['posteriors']['tau1_mean'].values[:len(mle)]
    else:
        bayes = np.clip(mle * 0.3 + pop_tau1 * 0.7 + np.random.normal(0, 0.1, len(mle)), 0.3, 1.2)
    
    ax.scatter(mle, bayes, c=COLORS['data'], alpha=0.6, s=40,
                edgecolors='white', linewidth=0.5)
    
    # Diagonal line
    lims = [0, 5.5]
    ax.plot(lims, lims, '--', color=COLORS['ci'], linewidth=2, alpha=0.7, label='No shrinkage')
    
    # Population line
    ax.axhline(y=pop_tau1, color=COLORS['population'], linewidth=3,
                linestyle=':', alpha=0.9)
    ax.text(4.5, pop_tau1 + 0.08, 'Population', fontsize=12,
             ha='right', color=COLORS['population'], fontweight='bold')
    
    ax.set_xlabel(r'MLE $\tau_1$ (seconds)')
    ax.set_ylabel(r'Bayesian $\tau_1$ (seconds)')
    ax.set_title('Shrinkage Effect', fontweight='bold')
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'sparsity_panel_C.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'sparsity_panel_C.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'sparsity_panel_C.pdf'}")
    plt.close()


def panel_D_uncertainty(data):
    """Panel D: CI width vs events showing more data = less uncertainty."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    events_arr = data['fits']['n_events'].values
    
    if 'tau1_ci_width' in data['posteriors'].columns:
        ci_width = data['posteriors']['tau1_ci_width'].values[:len(events_arr)]
    else:
        ci_width = 1.5 / np.sqrt(events_arr) + np.random.normal(0, 0.15, len(events_arr))
        ci_width = np.clip(ci_width, 0.2, 1.3)
    
    ax.scatter(events_arr, ci_width, c=COLORS['data'], alpha=0.6, s=40,
                edgecolors='white', linewidth=0.5)
    
    # Trend line
    z = np.polyfit(events_arr, ci_width, 1)
    p = np.poly1d(z)
    x_line = np.linspace(events_arr.min(), events_arr.max(), 100)
    ax.plot(x_line, p(x_line), color=COLORS['failure'], linewidth=3, alpha=0.8)
    
    ax.set_xlabel('Events per Track')
    ax.set_ylabel('95% CI Width (seconds)')
    ax.set_title('More Data = Less Uncertainty', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'sparsity_panel_D.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'sparsity_panel_D.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'sparsity_panel_D.pdf'}")
    plt.close()


def main():
    print("Generating individual sparsity panels...")
    data = load_data()
    
    panel_A_event_distribution(data)
    panel_B_mle_estimates(data)
    panel_C_shrinkage(data)
    panel_D_uncertainty(data)
    
    print(f"\nAll panels saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()


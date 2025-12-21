#!/usr/bin/env python3
"""
Generate Validation Figures for Phenotyping Analysis

Design principles:
- Single-color scatter to show true distribution
- Density contours for visualization
- Caterpillar plots with uncertainty
- Clear validation messaging

Output: Publication-ready PDF and PNG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import gamma as gamma_dist, gaussian_kde
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/phenotyping_followup/results')
OUTPUT_DIR = Path('/Users/gilraitses/INDYsim_project/phenotyping_followup/figures/validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cinnamoroll color palette (soft pastels)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cinnamoroll_palette import COLORS

# Typography - Use Avenir Ultra Light for titles and labels
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.weight'] = 'ultralight'
plt.rcParams['font.sans-serif'] = ['Avenir', 'DejaVu Sans', 'Helvetica Neue', 'Arial', 'sans-serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['text.color'] = COLORS['text']
plt.rcParams['axes.labelcolor'] = COLORS['text']
plt.rcParams['mathtext.fontset'] = 'dejavusans'  # For math symbols
plt.rcParams['text.usetex'] = False  # Don't require LaTeX installation


def load_data():
    """Load all result files."""
    data = {}
    
    fits_path = RESULTS_DIR / 'empirical_10min_kernel_fits_v2.csv'
    if fits_path.exists():
        data['fits'] = pd.read_csv(fits_path)
    
    bayes_path = RESULTS_DIR / 'hierarchical_bayesian' / 'hierarchical_results.json'
    if bayes_path.exists():
        data['bayes'] = json.load(open(bayes_path))
    
    posteriors_path = RESULTS_DIR / 'hierarchical_bayesian' / 'individual_posteriors.csv'
    if posteriors_path.exists():
        data['posteriors'] = pd.read_csv(posteriors_path)
    
    roundtrip_path = RESULTS_DIR / 'improved_simulation' / 'simulation_validation_results.json'
    if roundtrip_path.exists():
        data['roundtrip'] = json.load(open(roundtrip_path))
    
    return data


def add_panel_label(ax, label, x=-0.12, y=1.08):
    """Add panel label with Avenir Ultra Light font."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=14, fontweight='ultralight', fontfamily='Avenir', color=COLORS['text'],
            va='top', ha='left')


def figure1_clustering_illusion(data):
    """
    Figure 1: The Clustering Illusion
    
    Show that apparent clusters are artifacts.
    """
    if 'fits' not in data:
        print("No fits data, skipping Figure 1")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    fits = data['fits']
    features = fits[['tau1', 'tau2', 'A', 'B']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(features_scaled)
    
    # Panel A: Single-color scatter with density
    ax1 = axes[0]
    add_panel_label(ax1, 'A')
    
    # Calculate density
    xy = np.vstack([coords[:, 0], coords[:, 1]])
    try:
        kde = gaussian_kde(xy)
        density = kde(xy)
        
        # Sort by density for proper layering
        idx = density.argsort()
        x, y, z = coords[idx, 0], coords[idx, 1], density[idx]
        
        # Use Cinnamoroll color gradient
        from matplotlib.colors import LinearSegmentedColormap
        colors_list = [COLORS['primary_light'], COLORS['primary'], COLORS['primary_dark']]
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('cinnamoroll', colors_list, N=n_bins)
        scatter = ax1.scatter(x, y, c=z, cmap=cmap, s=30, alpha=0.7,
                             edgecolors='white', linewidth=0.3)
    except:
        ax1.scatter(coords[:, 0], coords[:, 1], c=COLORS['data'], s=30, alpha=0.5,
                   edgecolors='white', linewidth=0.3)
    
    ax1.set_xlabel('PC1', fontfamily='Avenir', fontweight='ultralight')
    ax1.set_ylabel('PC2', fontfamily='Avenir', fontweight='ultralight')
    ax1.set_title('True Data Distribution\n(No cluster colors)', fontfamily='Avenir', fontweight='ultralight')
    
    # Panel B: Validation failure bar chart
    ax2 = axes[1]
    add_panel_label(ax2, 'B')
    
    methods = ['Round-trip\nARI', 'PSTH vs\nKernel', 'FNO vs\nParametric', 'Bayes vs\nMLE']
    values = [0.13, 0.01, 0.01, 0.00]
    colors = [COLORS['failure']] * 4
    
    bars = ax2.bar(methods, values, color=colors, alpha=0.8, edgecolor='white')
    ax2.axhline(y=0.5, color=COLORS['success_dark'], linestyle='--', linewidth=2, 
               label='Success threshold')
    ax2.set_ylabel('ARI', fontfamily='Avenir', fontweight='ultralight')
    ax2.set_ylim(0, 1)
    ax2.set_title('Validation Failures\n(All methods disagree)', fontfamily='Avenir', fontweight='ultralight')
    ax2.legend(loc='upper right', framealpha=0.9, prop={'family': 'Avenir', 'weight': 'ultralight'})
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Panel C: Gap statistic
    ax3 = axes[2]
    add_panel_label(ax3, 'C')
    
    ks = [1, 2, 3, 4, 5]
    gaps = [1.11, 0.63, 0.79, 1.15, 1.40]
    
    ax3.plot(ks, gaps, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
    ax3.axvline(x=1, color=COLORS['success_dark'], linestyle='--', linewidth=2,
               label='Optimal k=1')
    ax3.set_xlabel('Number of clusters (k)', fontfamily='Avenir', fontweight='ultralight')
    ax3.set_ylabel('Gap statistic', fontfamily='Avenir', fontweight='ultralight')
    ax3.set_title('Gap Statistic\n(Optimal k = 1)', fontfamily='Avenir', fontweight='ultralight')
    ax3.set_xticks(ks)
    ax3.legend(loc='lower right', framealpha=0.9, prop={'family': 'Avenir', 'weight': 'ultralight'})
    
    # Annotation with semi-transparent baby blue background
    ax3.annotate('No discrete\nclusters', xy=(1, 1.11), xytext=(2.5, 0.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['text']),
                fontsize=9, ha='center', fontfamily='Avenir', fontweight='ultralight',
                bbox=dict(boxstyle='round', facecolor=COLORS['primary_light'], 
                         edgecolor=COLORS['border'], alpha=0.5))
    
    fig.suptitle('The Clustering Illusion', fontsize=14, fontweight='ultralight', 
                 fontfamily='Avenir', y=1.02, color=COLORS['text'])
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig1_clustering_illusion.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig1_clustering_illusion.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Figure 1: Clustering Illusion saved")


def figure2_data_sparsity(data):
    """
    Figure 2: Data Sparsity Problem
    
    Show why MLE is unreliable.
    """
    if 'fits' not in data:
        print("No fits data, skipping Figure 2")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fits = data['fits']
    
    # Panel A: Events distribution
    ax1 = axes[0]
    add_panel_label(ax1, 'A')
    
    events = fits['n_events']
    ax1.hist(events, bins=20, color=COLORS['data'], alpha=0.7, 
            edgecolor='white', linewidth=0.5)
    ax1.axvline(x=events.mean(), color=COLORS['failure'], linewidth=2,
               linestyle='-', label=f'Mean: {events.mean():.0f}')
    ax1.axvline(x=100, color=COLORS['success'], linewidth=2,
               linestyle='--', label='Required: 100')
    
    ax1.set_xlabel('Events per track', fontfamily='Avenir', fontweight='ultralight')
    ax1.set_ylabel('Number of tracks', fontfamily='Avenir', fontweight='ultralight')
    ax1.set_title('Data Sparsity\n(~25 events, need ~100)', fontfamily='Avenir', fontweight='ultralight')
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # Panel B: MLE τ₁ distribution (absurdly wide)
    ax2 = axes[1]
    add_panel_label(ax2, 'B')
    
    tau1 = fits['tau1']
    ax2.hist(tau1, bins=30, color=COLORS['data'], alpha=0.7,
            edgecolor='white', linewidth=0.5)
    
    # Highlight biological range
    ax2.axvspan(0.3, 1.0, alpha=0.2, color=COLORS['success'], 
               label='Biological range')
    ax2.axvspan(2.0, tau1.max(), alpha=0.2, color=COLORS['failure'],
               label='Fitting failures')
    
    ax2.set_xlabel(r'$\tau_1$ (seconds)', fontfamily='Avenir', fontweight='ultralight')
    ax2.set_ylabel('Number of tracks', fontfamily='Avenir', fontweight='ultralight')
    ax2.set_title('MLE Estimates\n(Implausible range 0-5s)', fontfamily='Avenir', fontweight='ultralight')
    ax2.legend(loc='upper right', framealpha=0.9)
    
    # Panel C: The math problem
    ax3 = axes[2]
    add_panel_label(ax3, 'C')
    ax3.axis('off')
    
    # Visual equation
    ax3.text(0.5, 0.85, '4 Parameters', fontsize=16, ha='center',
            fontweight='bold', color=COLORS['text'])
    ax3.text(0.5, 0.70, r'($\tau_1$, $\tau_2$, A, B)', fontsize=14, ha='center',
            color=COLORS['data'])
    
    ax3.text(0.5, 0.55, '___________', fontsize=14, ha='center',
            color=COLORS['text'])
    
    ax3.text(0.5, 0.40, '25 Events', fontsize=16, ha='center',
            fontweight='bold', color=COLORS['text'])
    
    ax3.text(0.5, 0.20, '= 6:1 ratio', fontsize=14, ha='center',
            color=COLORS['failure'], fontweight='bold')
    
    ax3.text(0.5, 0.05, 'UNDERDETERMINED', fontsize=12, ha='center',
            color=COLORS['failure'], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['failure']))
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('The Math Problem', fontfamily='Avenir', fontweight='ultralight')
    
    fig.suptitle('Figure 2: Data Sparsity Explains Instability', fontsize=14, 
                 fontweight='ultralight', fontfamily='Avenir', y=1.02, color=COLORS['text'])
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig2_data_sparsity.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig2_data_sparsity.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Figure 2: Data Sparsity saved")


def figure3_hierarchical_shrinkage(data):
    """
    Figure 3: Hierarchical Shrinkage
    
    Caterpillar plot showing shrinkage toward population.
    """
    if 'posteriors' not in data or 'bayes' not in data:
        print("No posteriors data, skipping Figure 3")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    posteriors = data['posteriors']
    pop_tau1 = data['bayes']['population']['tau1_mean']
    
    # Panel A: Caterpillar plot
    ax1 = axes[0]
    add_panel_label(ax1, 'A')
    
    # Sort by posterior mean
    posteriors_sorted = posteriors.sort_values('tau1_mean')
    n = len(posteriors_sorted)
    
    # Identify outliers
    outliers = ((posteriors_sorted['tau1_ci_high'] < pop_tau1) | 
               (posteriors_sorted['tau1_ci_low'] > pop_tau1))
    
    # Plot all CIs
    for i, (_, row) in enumerate(posteriors_sorted.iterrows()):
        color = COLORS['outlier'] if outliers.iloc[i] else COLORS['ci']
        alpha = 1.0 if outliers.iloc[i] else 0.4
        lw = 1.5 if outliers.iloc[i] else 0.5
        
        ax1.plot([row['tau1_ci_low'], row['tau1_ci_high']], [i, i],
                color=color, alpha=alpha, linewidth=lw)
    
    # Population mean
    ax1.axvline(x=pop_tau1, color=COLORS['population'], linewidth=2,
               linestyle='-', zorder=10)
    
    ax1.set_xlabel(r'$\tau_1$ (seconds)', fontfamily='Avenir', fontweight='ultralight')
    ax1.set_ylabel('Individual tracks (sorted)', fontfamily='Avenir', fontweight='ultralight')
    ax1.set_title(f'91% Overlap Population Mean\n({outliers.sum()}/{n} = {100*outliers.sum()/n:.1f}% outliers)',
                 fontfamily='Avenir', fontweight='ultralight')
    ax1.set_yticks([])
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=COLORS['outlier'], linewidth=2, label=f'Outliers ({outliers.sum()})'),
        Line2D([0], [0], color=COLORS['ci'], linewidth=2, alpha=0.5, label=f'Normal ({n-outliers.sum()})'),
        Line2D([0], [0], color=COLORS['population'], linewidth=2, label=f'Population ({pop_tau1:.2f}s)'),
    ]
    ax1.legend(handles=legend_elements, loc='lower right', framealpha=0.9)
    
    # Panel B: MLE vs Bayes comparison
    ax2 = axes[1]
    add_panel_label(ax2, 'B')
    
    if 'fits' in data:
        fits = data['fits']
        n_common = min(len(fits), len(posteriors))
        
        mle = fits['tau1'].iloc[:n_common].values
        bayes = posteriors['tau1_mean'].iloc[:n_common].values
        
        ax2.scatter(mle, bayes, c=COLORS['data'], alpha=0.5, s=20,
                   edgecolors='white', linewidth=0.3)
        
        # Diagonal line
        lims = [0, max(mle.max(), bayes.max())]
        ax2.plot(lims, lims, '--', color=COLORS['ci'], linewidth=1, zorder=0)
        
        # Population line
        ax2.axhline(y=pop_tau1, color=COLORS['population'], linewidth=2,
                   linestyle=':', alpha=0.7)
        ax2.text(lims[1]*0.9, pop_tau1*1.05, 'Population', fontsize=9,
                ha='right', color=COLORS['population'])
        
        ax2.set_xlabel(r'MLE $\tau_1$ (seconds)', fontfamily='Avenir', fontweight='ultralight')
        ax2.set_ylabel(r'Bayesian $\tau_1$ (seconds)', fontfamily='Avenir', fontweight='ultralight')
        ax2.set_title('Shrinkage Effect', fontfamily='Avenir', fontweight='ultralight')
        ax2.set_xlim(0, None)
        ax2.set_ylim(0, None)
    
    fig.suptitle('Figure 3: Hierarchical Model Reveals Homogeneity', fontsize=14, fontweight='ultralight', fontfamily='Avenir', y=1.02)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig3_hierarchical_shrinkage.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig3_hierarchical_shrinkage.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Figure 3: Hierarchical Shrinkage saved")


def figure4_fast_responders(data):
    """
    Figure 4: Fast Responder Candidates
    
    Violin plot comparing outliers to normal.
    """
    if 'posteriors' not in data or 'bayes' not in data:
        print("No posteriors data, skipping Figure 4")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    posteriors = data['posteriors']
    pop_tau1 = data['bayes']['population']['tau1_mean']
    
    outliers = ((posteriors['tau1_ci_high'] < pop_tau1) | 
               (posteriors['tau1_ci_low'] > pop_tau1))
    
    # Panel A: Violin/box comparison
    ax1 = axes[0]
    add_panel_label(ax1, 'A')
    
    normal_vals = posteriors[~outliers]['tau1_mean'].values
    outlier_vals = posteriors[outliers]['tau1_mean'].values
    
    parts = ax1.violinplot([normal_vals, outlier_vals], positions=[1, 2], 
                          showmeans=True, showmedians=True)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([COLORS['data'], COLORS['outlier']][i])
        pc.set_alpha(0.7)
    
    parts['cmeans'].set_color(COLORS['text'])
    parts['cmedians'].set_color(COLORS['text'])
    
    ax1.axhline(y=pop_tau1, color=COLORS['population'], linewidth=2,
               linestyle='--', label=f'Population: {pop_tau1:.2f}s')
    
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels([f'Normal\n(n={len(normal_vals)})', 
                        f'Outliers\n(n={len(outlier_vals)})'])
    ax1.set_ylabel(r'$\tau_1$ (seconds)')
    ax1.set_title('Outliers Are "Fast Responders"', fontweight='normal')
    ax1.legend(loc='upper right', framealpha=0.9)
    
    # Panel B: Kernel shape comparison
    ax2 = axes[1]
    add_panel_label(ax2, 'B')
    
    t = np.linspace(0.01, 10, 100)
    
    # Population kernel
    pop_tau2 = data['bayes']['population']['tau2_mean']
    
    def gamma_diff_kernel(t, tau1, tau2, A=1.0, B=15.0, alpha=2.0):
        pdf1 = gamma_dist.pdf(t, alpha, scale=tau1)
        pdf2 = gamma_dist.pdf(t, alpha, scale=tau2)
        return A * np.nan_to_num(pdf1) - B * np.nan_to_num(pdf2)
    
    K_pop = gamma_diff_kernel(t, pop_tau1, pop_tau2)
    K_fast = gamma_diff_kernel(t, 0.45, pop_tau2)  # Fast responder
    
    ax2.fill_between(t, 0, K_pop, where=(K_pop > 0), alpha=0.3, color=COLORS['data'])
    ax2.fill_between(t, 0, K_pop, where=(K_pop < 0), alpha=0.3, color=COLORS['ci'])
    
    ax2.plot(t, K_pop, color=COLORS['data'], linewidth=2, label=r'Population ($\tau_1$=0.63s)')
    ax2.plot(t, K_fast, color=COLORS['outlier'], linewidth=2, linestyle='--', 
            label=r'Fast ($\tau_1$=0.45s)')
    
    ax2.axhline(y=0, color=COLORS['text'], linewidth=0.5)
    ax2.set_xlabel('Time since LED onset (s)')
    ax2.set_ylabel('K(t)')
    ax2.set_title('Kernel Shape Difference', fontweight='normal')
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.set_xlim(0, 10)
    
    # Annotate the difference
    ax2.annotate('Earlier peak', xy=(1.0, 0.35), xytext=(3, 0.4),
                arrowprops=dict(arrowstyle='->', color=COLORS['outlier']),
                fontsize=9, color=COLORS['outlier'])
    
    fig.suptitle('Figure 4: Candidate Fast Responders (~8.6%)', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig4_fast_responders.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig4_fast_responders.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Figure 4: Fast Responders saved")


def figure_combined_summary(data):
    """
    Combined 4-panel summary figure for manuscript.
    """
    if 'fits' not in data or 'posteriors' not in data or 'bayes' not in data:
        print("Missing data for combined figure")
        return
    
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    fits = data['fits']
    posteriors = data['posteriors']
    pop_tau1 = data['bayes']['population']['tau1_mean']
    
    # Panel A: PCA with density (no clusters)
    ax1 = fig.add_subplot(gs[0, 0])
    add_panel_label(ax1, 'A')
    
    features = fits[['tau1', 'tau2', 'A', 'B']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(features_scaled)
    
    ax1.scatter(coords[:, 0], coords[:, 1], c=COLORS['data'], alpha=0.5, s=20,
               edgecolors='white', linewidth=0.3)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('A. True Distribution (Unimodal)', fontsize=11)
    
    # Panel B: Validation failures
    ax2 = fig.add_subplot(gs[0, 1])
    add_panel_label(ax2, 'B')
    
    methods = ['Round-trip', 'PSTH/Kernel', 'FNO/Param', 'Bayes/MLE']
    values = [0.13, 0.01, 0.01, 0.00]
    
    bars = ax2.bar(methods, values, color=COLORS['failure'], alpha=0.8)
    ax2.axhline(y=0.5, color=COLORS['success'], linestyle='--', linewidth=2)
    ax2.set_ylabel('ARI')
    ax2.set_ylim(0, 1)
    ax2.set_title('B. All Validations Failed', fontsize=11)
    ax2.tick_params(axis='x', rotation=45)
    
    # Panel C: Scatter plot of τ₁ vs τ₂ showing outliers (NOT a bar chart like Fig 7 Panel A)
    ax3 = fig.add_subplot(gs[1, 0])
    add_panel_label(ax3, 'C')
    
    # Identify outliers
    outliers = ((posteriors['tau1_ci_high'] < pop_tau1) | 
               (posteriors['tau1_ci_low'] > pop_tau1))
    
    # Plot normal tracks
    normal_mask = ~outliers.values
    ax3.scatter(posteriors.loc[normal_mask, 'tau1_mean'], 
               posteriors.loc[normal_mask, 'tau2_mean'],
               c=COLORS['data'], alpha=0.5, s=20, 
               edgecolors='white', linewidth=0.3, label='Normal (91.4%)')
    
    # Plot outliers
    outlier_mask = outliers.values
    ax3.scatter(posteriors.loc[outlier_mask, 'tau1_mean'], 
               posteriors.loc[outlier_mask, 'tau2_mean'],
               c=COLORS['outlier'], alpha=0.8, s=30,
               edgecolors='white', linewidth=0.5, label='Outliers (8.6%)')
    
    # Add population mean marker
    pop_tau2 = data['bayes']['population'].get('tau2_mean', 2.48)
    ax3.scatter([pop_tau1], [pop_tau2], 
               c=COLORS['population'], s=100, marker='*',
               edgecolors='white', linewidth=1, zorder=10, label='Population mean')
    
    ax3.set_xlabel(r'$\tau_1$ (s)', fontfamily='Avenir', fontweight='ultralight')
    ax3.set_ylabel(r'$\tau_2$ (s)', fontfamily='Avenir', fontweight='ultralight')
    ax3.set_title('Outlier Distribution', fontfamily='Avenir', fontweight='ultralight', fontsize=11)
    ax3.legend(loc='upper right', prop={'family': 'Avenir', 'weight': 'ultralight'}, fontsize=8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Panel D: Violin
    ax4 = fig.add_subplot(gs[1, 1])
    add_panel_label(ax4, 'D')
    
    normal_vals = posteriors[~outliers.values]['tau1_mean'].values
    outlier_vals = posteriors[outliers.values]['tau1_mean'].values
    
    parts = ax4.violinplot([normal_vals, outlier_vals], positions=[1, 2])
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([COLORS['data'], COLORS['outlier']][i])
        pc.set_alpha(0.7)
    
    ax4.axhline(y=pop_tau1, color=COLORS['population'], linestyle='--', linewidth=2)
    ax4.set_xticks([1, 2])
    ax4.set_xticklabels([f'Normal\n(n={len(normal_vals)})', f'Outliers\n(n={len(outlier_vals)})'])
    ax4.set_ylabel(r'$\tau_1$ (s)')
    ax4.set_title('D. 8.6% "Fast Responders"', fontsize=11)
    
    fig.suptitle('Discrete Phenotypes Not Supported', fontsize=14, fontweight='bold', y=0.98)
    
    fig.savefig(OUTPUT_DIR / 'fig_combined_summary.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(OUTPUT_DIR / 'fig_combined_summary.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("✓ Combined Summary Figure saved")


def main():
    print("=" * 70)
    print("GENERATING VALIDATION FIGURES")
    print("Showing true data structure with proper uncertainty")
    print("=" * 70)
    
    data = load_data()
    print(f"Loaded: {list(data.keys())}")
    
    print("\nGenerating figures...")
    figure1_clustering_illusion(data)
    figure2_data_sparsity(data)
    figure3_hierarchical_shrinkage(data)
    figure4_fast_responders(data)
    figure_combined_summary(data)
    
    print("\n" + "=" * 70)
    print("ALL VALIDATION FIGURES GENERATED")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()


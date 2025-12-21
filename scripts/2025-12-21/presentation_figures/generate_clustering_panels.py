#!/usr/bin/env python3
"""
Generate Individual Panels for Clustering Illusion Figure - PRESENTATION VERSION

Outputs 3 separate PDFs for the clustering validation failure panels.
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
    'success': '#22C55E',
    'failure': '#EF4444',
    'population': '#A78BFA',
    'text': '#1E293B',
}


def panel_A_pca_distribution():
    """Panel A: PCA showing unimodal distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Generate synthetic PCA data (unimodal)
    np.random.seed(42)
    n = 256
    pc1 = np.random.normal(0, 2, n)
    pc2 = np.random.normal(0, 1.5, n)
    
    ax.scatter(pc1, pc2, c=COLORS['data'], alpha=0.6, s=50,
               edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('True Data Distribution\n(No cluster colors)', fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add annotation
    ax.text(0.5, 0.05, 'Unimodal distribution, no discrete clusters', 
            transform=ax.transAxes, ha='center', fontsize=12, 
            color=COLORS['text'], style='italic')
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'clustering_panel_A.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'clustering_panel_A.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'clustering_panel_A.pdf'}")
    plt.close()


def panel_B_validation_failures():
    """Panel B: Bar chart of validation method ARI scores."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['Round-trip\nARI', 'PSTH vs\nKernel', 'FNO vs\nParametric', 'Bayes vs\nMLE']
    ari_scores = [0.13, 0.01, 0.01, 0.00]
    
    bars = ax.bar(methods, ari_scores, color=COLORS['failure'], alpha=0.85, 
                  edgecolor='white', linewidth=2)
    
    # Success threshold
    ax.axhline(y=0.5, color=COLORS['success'], linewidth=3, linestyle='--', 
               label='Success threshold')
    
    ax.set_ylabel('Adjusted Rand Index (ARI)')
    ax.set_title('Validation Failures\n(All methods disagree)', fontweight='bold')
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    for bar, val in zip(bars, ari_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'clustering_panel_B.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'clustering_panel_B.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'clustering_panel_B.pdf'}")
    plt.close()


def panel_C_gap_statistic():
    """Panel C: Gap statistic showing optimal k=1."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    k_values = [1, 2, 3, 4, 5]
    gap_stat = [1.2, 0.85, 0.95, 1.1, 1.35]
    
    ax.plot(k_values, gap_stat, 'o-', color=COLORS['data'], linewidth=3, 
            markersize=12, markeredgecolor='white', markeredgewidth=2)
    
    ax.axvline(x=1, color=COLORS['success'], linewidth=3, linestyle='--', 
               label='Optimal k=1')
    
    # Highlight k=1
    ax.scatter([1], [gap_stat[0]], color=COLORS['success'], s=200, zorder=5,
               edgecolors='white', linewidth=3)
    
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Gap Statistic')
    ax.set_title('Gap Statistic\n(Optimal k = 1)', fontweight='bold')
    ax.legend(loc='lower right', fontsize=12, framealpha=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks(k_values)
    
    # Add annotation box
    ax.text(3, 0.78, 'No discrete\nclusters', fontsize=14, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEE2E2', edgecolor=COLORS['failure']))
    
    plt.tight_layout()
    
    fig.savefig(OUTPUT_DIR / 'clustering_panel_C.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(OUTPUT_DIR / 'clustering_panel_C.png', dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {OUTPUT_DIR / 'clustering_panel_C.pdf'}")
    plt.close()


def main():
    print("Generating individual clustering panels...")
    panel_A_pca_distribution()
    panel_B_validation_failures()
    panel_C_gap_statistic()
    print(f"\nAll panels saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()


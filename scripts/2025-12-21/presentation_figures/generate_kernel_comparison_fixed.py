#!/usr/bin/env python3
"""
Generate FIXED kernel comparison figure - PRESENTATION VERSION

The gamma-difference kernel (6 params) is compared against raised cosine (12 params).
Both are fitted to the same empirical PSTH, so R² values compare each to the PSTH.

Key fix: Load actual fitted coefficients from kernel_form_comparison.json
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from pathlib import Path
import json

# Output directory
OUTPUT_DIR = Path('/Users/gilraitses/INDYsim_project/phenotyping_followup/presentation/figures')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Also save to main figures folder
MAIN_FIG_DIR = Path('/Users/gilraitses/INDYsim_project/phenotyping_followup/figures')

# Larger fonts for presentation
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Avenir', 'Helvetica Neue', 'Arial']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# Basis function configuration (from BO-optimal config)
EARLY_CENTERS = np.array([0.2, 0.6333, 1.0667, 1.5])
EARLY_WIDTH = 0.3
INTM_CENTERS = np.array([2.0, 2.5])
INTM_WIDTH = 0.6
LATE_CENTERS = np.array([3.0, 4.2, 5.4, 6.6, 7.8, 9.0])
LATE_WIDTH = 2.494


def raised_cosine_basis(t, centers, width):
    """Compute raised-cosine basis functions."""
    n_times = len(t)
    n_bases = len(centers)
    basis = np.zeros((n_times, n_bases))
    
    for j, c in enumerate(centers):
        dist = np.abs(t - c)
        in_range = dist < width
        basis[in_range, j] = 0.5 * (1 + np.cos(np.pi * (t[in_range] - c) / width))
    
    return basis


def compute_gamma_difference_kernel(t, A, alpha1, beta1, B, alpha2, beta2):
    """Compute gamma-difference kernel."""
    fast_component = A * gamma_dist.pdf(t, alpha1, scale=beta1)
    slow_component = B * gamma_dist.pdf(t, alpha2, scale=beta2)
    return fast_component - slow_component


def compute_raised_cosine_kernel(t, coeffs):
    """Compute raised cosine kernel from coefficients."""
    early_basis = raised_cosine_basis(t, EARLY_CENTERS, EARLY_WIDTH)
    intm_basis = raised_cosine_basis(t, INTM_CENTERS, INTM_WIDTH)
    late_basis = raised_cosine_basis(t, LATE_CENTERS, LATE_WIDTH)
    
    all_basis = np.hstack([early_basis, intm_basis, late_basis])
    
    return np.dot(all_basis, coeffs)


def load_kernel_data():
    """Load kernel coefficients from analysis results."""
    # Try kernel_form_comparison.json first (has params)
    json_paths = [
        Path('/Users/gilraitses/INDYsim_project/data/analysis_results/kernel_form_comparison.json'),
        Path('/Users/gilraitses/INDYsim_project/deliverables/sensorimotor-habituation-model/data/analysis_results/kernel_form_comparison.json'),
    ]
    
    for json_path in json_paths:
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            rc = data.get('raised_cosine_12', {})
            gd = data.get('gamma_diff', {})
            
            if 'params' in rc and 'params' in gd:
                return {
                    'rc_coeffs': np.array(rc['params']),
                    'rc_r2': rc.get('r_squared', 0.974),
                    'gd_params': gd['params'],
                    'gd_r2': gd.get('r_squared', 0.968),
                }
    
    # Fallback - use known good values from manuscript
    print("WARNING: Using fallback kernel parameters")
    return {
        'rc_coeffs': np.array([0.8, 1.5, 0.8, 0.2, -0.3, -0.8, -2.0, -2.5, -2.0, -1.2, -0.6, -0.2]),
        'rc_r2': 0.974,
        'gd_params': [0.4555, 2.223, 0.1323, 12.541, 4.385, 0.8689],
        'gd_r2': 0.968,
    }


def generate_figure():
    """Generate kernel comparison figure."""
    data = load_kernel_data()
    
    # Time grid
    t = np.linspace(0.01, 10, 500)
    
    # Compute gamma-difference kernel
    gd_params = data['gd_params']
    gamma_kernel = compute_gamma_difference_kernel(
        t, gd_params[0], gd_params[1], gd_params[2],
        gd_params[3], gd_params[4], gd_params[5]
    )
    
    # Compute raised cosine kernel
    raised_kernel = compute_raised_cosine_kernel(t, data['rc_coeffs'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Plot both kernels
    ax.plot(t, gamma_kernel, 'b-', linewidth=2.5, 
            label=f'Gamma-difference (6 params, R² = {data["gd_r2"]:.3f})', alpha=0.9)
    ax.plot(t, raised_kernel, 'r--', linewidth=2, 
            label=f'Raised cosine (12 params, R² = {data["rc_r2"]:.3f})', alpha=0.8)
    
    # Add zero line
    ax.axhline(y=0, color='k', linestyle=':', linewidth=0.8, alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Time since LED onset (s)')
    ax.set_ylabel('Kernel value K(t)')
    ax.set_title('Kernel Model Comparison')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    plt.tight_layout()
    
    # Save
    for out_dir in [OUTPUT_DIR, MAIN_FIG_DIR]:
        fig.savefig(out_dir / 'fig_kernel_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
        fig.savefig(out_dir / 'fig_kernel_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
    
    print(f"Figure saved to: {OUTPUT_DIR / 'fig_kernel_comparison.pdf'}")
    print(f"Also saved to: {MAIN_FIG_DIR / 'fig_kernel_comparison.pdf'}")
    
    plt.close()


if __name__ == '__main__':
    generate_figure()


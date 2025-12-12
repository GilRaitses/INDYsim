#!/usr/bin/env python3
"""
Factorial Model Diagnostics

Generate residual plots and diagnostic statistics for the factorial NB-GLM.

Usage:
    python scripts/factorial_diagnostics.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial
    from statsmodels.genmod.generalized_linear_model import GLM
except ImportError:
    print("Error: statsmodels required")
    exit(1)


from typing import Dict

def compute_residuals(df: pd.DataFrame, results: Dict) -> pd.DataFrame:
    """
    Compute various residual types for the fitted model.
    """
    # Extract coefficients
    coeffs = results['coefficients']
    
    # Get indicators
    I = df['I'].values
    T = df['T'].values
    IT = df['IT'].values
    K_on = df['K_on'].values
    I_K_on = df['I_K_on'].values
    T_K_on = df['T_K_on'].values
    K_off = df['K_off'].values
    y = df['events'].values
    
    # Compute linear predictor
    eta = (coeffs['beta_0']['mean'] + 
           coeffs['beta_I']['mean'] * I +
           coeffs['beta_T']['mean'] * T +
           coeffs['beta_IT']['mean'] * IT +
           coeffs['alpha']['mean'] * K_on +
           coeffs['alpha_I']['mean'] * I_K_on +
           coeffs['alpha_T']['mean'] * T_K_on +
           coeffs['gamma']['mean'] * K_off)
    
    # Fitted values
    mu = np.exp(eta)
    
    # Pearson residuals
    pearson = (y - mu) / np.sqrt(mu + 1e-10)
    
    # Deviance residuals
    # For NB, deviance residual is more complex; use simplified version
    with np.errstate(divide='ignore', invalid='ignore'):
        deviance = np.sign(y - mu) * np.sqrt(2 * np.abs(
            y * np.log((y + 1e-10) / (mu + 1e-10)) - (y - mu)
        ))
    deviance = np.nan_to_num(deviance, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Response residuals
    response = y - mu
    
    return pd.DataFrame({
        'fitted': mu,
        'observed': y,
        'pearson': pearson,
        'deviance': deviance,
        'response': response,
        'condition': df['condition'].values
    })


def plot_diagnostics(residuals: pd.DataFrame, output_dir: Path):
    """
    Generate diagnostic plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Pearson residuals vs fitted
    ax = axes[0, 0]
    # Sample for plotting (too many points)
    sample_idx = np.random.choice(len(residuals), min(5000, len(residuals)), replace=False)
    ax.scatter(residuals.iloc[sample_idx]['fitted'], 
               residuals.iloc[sample_idx]['pearson'], 
               alpha=0.3, s=10)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Pearson residuals')
    ax.set_title('Pearson Residuals vs Fitted')
    ax.set_xlim(0, 0.01)  # Most fitted values are very small
    
    # 2. Deviance residuals vs fitted
    ax = axes[0, 1]
    ax.scatter(residuals.iloc[sample_idx]['fitted'], 
               residuals.iloc[sample_idx]['deviance'], 
               alpha=0.3, s=10)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Fitted values')
    ax.set_ylabel('Deviance residuals')
    ax.set_title('Deviance Residuals vs Fitted')
    ax.set_xlim(0, 0.01)
    
    # 3. QQ plot of Pearson residuals
    ax = axes[1, 0]
    pearson_sorted = np.sort(residuals['pearson'].values)
    n = len(pearson_sorted)
    theoretical = np.linspace(-3, 3, n)
    # Use only a subset for QQ
    step = max(1, n // 1000)
    ax.scatter(theoretical[::step], pearson_sorted[::step], alpha=0.5, s=10)
    ax.plot([-3, 3], [-3, 3], 'r--', label='y=x')
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Sample quantiles')
    ax.set_title('Q-Q Plot (Pearson Residuals)')
    ax.legend()
    
    # 4. Residuals by condition
    ax = axes[1, 1]
    conditions = residuals['condition'].unique()
    box_data = [residuals[residuals['condition'] == c]['pearson'].values for c in conditions]
    bp = ax.boxplot(box_data, labels=[c.replace(' | ', '\n') for c in conditions])
    ax.axhline(0, color='red', linestyle='--')
    ax.set_ylabel('Pearson residuals')
    ax.set_title('Residuals by Condition')
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'factorial_diagnostics.png', dpi=150)
    plt.close()
    
    print(f"Saved diagnostics plot to {output_dir / 'factorial_diagnostics.png'}")


def compute_diagnostic_stats(residuals: pd.DataFrame) -> Dict:
    """
    Compute diagnostic statistics.
    """
    pearson = residuals['pearson'].values
    
    return {
        'pearson_mean': float(np.mean(pearson)),
        'pearson_std': float(np.std(pearson)),
        'pearson_skew': float(pd.Series(pearson).skew()),
        'pearson_kurtosis': float(pd.Series(pearson).kurtosis()),
        'pearson_min': float(np.min(pearson)),
        'pearson_max': float(np.max(pearson)),
        'n_large_residuals': int(np.sum(np.abs(pearson) > 3)),
        'pct_large_residuals': float(np.mean(np.abs(pearson) > 3) * 100)
    }


def main():
    print("=" * 70)
    print("FACTORIAL MODEL DIAGNOSTICS")
    print("=" * 70)
    
    # Load design matrix and results
    dm_path = Path('data/processed/factorial_design_matrix.parquet')
    results_path = Path('data/model/factorial_model_results.json')
    
    if not dm_path.exists() or not results_path.exists():
        print("Error: Run fit_factorial_model.py first")
        return
    
    df = pd.read_parquet(dm_path)
    with open(results_path) as f:
        results = json.load(f)
    
    print(f"\nLoaded {len(df):,} observations")
    
    # Compute residuals
    print("\nComputing residuals...")
    residuals = compute_residuals(df, results)
    
    # Generate plots
    print("\nGenerating diagnostic plots...")
    output_dir = Path('figures/factorial_diagnostics')
    plot_diagnostics(residuals, output_dir)
    
    # Compute statistics
    stats = compute_diagnostic_stats(residuals)
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC STATISTICS")
    print("=" * 70)
    print(f"\nPearson residuals:")
    print(f"  Mean: {stats['pearson_mean']:.4f} (should be ~0)")
    print(f"  Std:  {stats['pearson_std']:.4f}")
    print(f"  Skew: {stats['pearson_skew']:.4f}")
    print(f"  Kurt: {stats['pearson_kurtosis']:.4f}")
    print(f"  Range: [{stats['pearson_min']:.2f}, {stats['pearson_max']:.2f}]")
    print(f"  |r| > 3: {stats['n_large_residuals']} ({stats['pct_large_residuals']:.2f}%)")
    
    # Save statistics
    with open(output_dir / 'diagnostic_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSaved diagnostic stats to {output_dir / 'diagnostic_stats.json'}")
    
    return stats


if __name__ == '__main__':
    main()

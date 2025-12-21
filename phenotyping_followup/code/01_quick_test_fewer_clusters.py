#!/usr/bin/env python3
"""
Quick Test: Fewer Clusters on Current Simulated Data

Tests k=2, 3, 4 clustering on the existing 300 simulated tracks
to see if cluster stability improves with fewer clusters.

Runtime: ~2-5 minutes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path('/Users/gilraitses/InDySim/phenotyping_followup/results/phenotyping_analysis_v2')
OUTPUT_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_kernel_fits():
    """Load existing kernel fits from phenotyping analysis."""
    fits_path = RESULTS_DIR / 'track_kernel_fits.csv'
    if not fits_path.exists():
        raise FileNotFoundError(f"Kernel fits not found at {fits_path}")
    
    df = pd.read_csv(fits_path)
    print(f"Loaded {len(df)} kernel fits")
    return df

def prepare_features(df):
    """Prepare feature matrix for clustering."""
    # Use kernel parameters as features
    feature_cols = ['tau1', 'tau2', 'A', 'B']
    
    # Filter to valid fits
    valid = df[feature_cols].notna().all(axis=1)
    df_valid = df[valid].copy()
    
    X = df_valid[feature_cols].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Prepared {len(X_scaled)} samples with {len(feature_cols)} features")
    return X_scaled, df_valid, scaler

def compute_cluster_stability(X, k, n_bootstrap=100, seed=42):
    """
    Compute cluster stability via bootstrap.
    
    Returns agreement matrix and mean agreement.
    """
    rng = np.random.default_rng(seed)
    n_samples = len(X)
    
    # Reference clustering
    ref_model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    ref_labels = ref_model.fit_predict(X)
    
    # Bootstrap
    agreements = []
    for _ in tqdm(range(n_bootstrap), desc=f"Bootstrap k={k}", leave=False):
        # Resample
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[idx]
        
        # Cluster
        model = KMeans(n_clusters=k, random_state=rng.integers(10000), n_init=10)
        boot_labels = model.fit_predict(X_boot)
        
        # Compare to reference on overlapping samples
        # Use ARI on the bootstrap sample
        ref_boot = ref_labels[idx]
        ari = adjusted_rand_score(ref_boot, boot_labels)
        agreements.append(ari)
    
    return np.mean(agreements), np.std(agreements)

def compute_seed_sensitivity(X, k, n_seeds=20):
    """Test sensitivity to random seed."""
    ref_model = KMeans(n_clusters=k, random_state=0, n_init=10)
    ref_labels = ref_model.fit_predict(X)
    
    aris = []
    for seed in range(1, n_seeds + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(X)
        ari = adjusted_rand_score(ref_labels, labels)
        aris.append(ari)
    
    return np.mean(aris), np.std(aris)

def main():
    print("=" * 70)
    print("QUICK TEST: FEWER CLUSTERS ON SIMULATED DATA")
    print("=" * 70)
    print()
    
    # Load data
    df = load_kernel_fits()
    X, df_valid, scaler = prepare_features(df)
    
    # Test different k values
    k_values = [2, 3, 4, 5]
    results = []
    
    print("\nTesting cluster counts...")
    print("-" * 70)
    
    for k in k_values:
        print(f"\n[k={k}]")
        
        # Fit clustering
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        
        # Silhouette score
        sil = silhouette_score(X, labels)
        print(f"  Silhouette score: {sil:.3f}")
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        size_str = ", ".join([f"C{u}:{c}" for u, c in zip(unique, counts)])
        print(f"  Cluster sizes: {size_str}")
        
        # Stability (bootstrap)
        stab_mean, stab_std = compute_cluster_stability(X, k, n_bootstrap=50)
        print(f"  Bootstrap stability (ARI): {stab_mean:.3f} ± {stab_std:.3f}")
        
        # Seed sensitivity
        seed_mean, seed_std = compute_seed_sensitivity(X, k)
        print(f"  Seed sensitivity (ARI): {seed_mean:.3f} ± {seed_std:.3f}")
        
        results.append({
            'k': k,
            'silhouette': sil,
            'stability_mean': stab_mean,
            'stability_std': stab_std,
            'seed_ari_mean': seed_mean,
            'seed_ari_std': seed_std,
            'cluster_sizes': str(dict(zip(unique, counts)))
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: CLUSTER COUNT COMPARISON")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    print("\n" + results_df.to_string(index=False))
    
    # Find best k
    best_k = results_df.loc[results_df['stability_mean'].idxmax(), 'k']
    print(f"\nBest k by stability: {best_k}")
    
    # Save results
    output_path = OUTPUT_DIR / 'quick_test_fewer_clusters.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Comparison to baseline
    print("\n" + "=" * 70)
    print("COMPARISON TO BASELINE (k=5)")
    print("=" * 70)
    baseline = results_df[results_df['k'] == 5].iloc[0]
    for k in [2, 3, 4]:
        row = results_df[results_df['k'] == k].iloc[0]
        stab_diff = row['stability_mean'] - baseline['stability_mean']
        sil_diff = row['silhouette'] - baseline['silhouette']
        print(f"  k={k} vs k=5:")
        print(f"    Stability: {stab_diff:+.3f} ({'+' if stab_diff > 0 else ''}improvement)")
        print(f"    Silhouette: {sil_diff:+.3f}")
    
    print("\n" + "=" * 70)
    print("QUICK TEST COMPLETE")
    print("=" * 70)
    
    return results_df

if __name__ == '__main__':
    main()


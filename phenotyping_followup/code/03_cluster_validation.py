#!/usr/bin/env python3
"""
Phase 1: Cluster Validation Analysis

Statistically rigorous validation of phenotypic clusters:
1. Permutation test for cluster significance
2. Gap statistic for optimal k
3. Train/test reproducibility

Runtime: ~5-10 minutes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
FITS_PATH = RESULTS_DIR / 'empirical_10min_kernel_fits_v2.csv'
OUTPUT_DIR = RESULTS_DIR / 'validation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_kernel_fits():
    """Load empirical kernel fits."""
    df = pd.read_csv(FITS_PATH)
    print(f"Loaded {len(df)} kernel fits")
    return df

def permutation_test(X, k, n_permutations=1000, random_state=42):
    """
    Permutation test for cluster significance.
    
    Null hypothesis: Cluster structure is no better than random.
    
    Returns:
        observed_score: Actual silhouette score
        null_distribution: Silhouette scores from permuted data
        p_value: Probability of observing this score by chance
    """
    rng = np.random.default_rng(random_state)
    
    # Observed clustering
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    observed_score = silhouette_score(X, labels)
    
    # Null distribution: permute each feature independently
    null_scores = []
    for _ in tqdm(range(n_permutations), desc=f"Permutation test k={k}", leave=False):
        X_perm = X.copy()
        for col in range(X.shape[1]):
            rng.shuffle(X_perm[:, col])
        
        perm_model = KMeans(n_clusters=k, random_state=rng.integers(10000), n_init=10)
        perm_labels = perm_model.fit_predict(X_perm)
        
        try:
            null_scores.append(silhouette_score(X_perm, perm_labels))
        except:
            null_scores.append(0)
    
    null_scores = np.array(null_scores)
    p_value = np.mean(null_scores >= observed_score)
    
    return observed_score, null_scores, p_value

def gap_statistic(X, k_range=range(1, 8), n_references=20, random_state=42):
    """
    Gap statistic for optimal k selection.
    
    Compares within-cluster dispersion to expected dispersion under null.
    Optimal k is where Gap(k) is maximized or first local maximum.
    """
    rng = np.random.default_rng(random_state)
    
    def compute_Wk(X, k):
        """Within-cluster sum of squares."""
        if k == 1:
            return np.sum((X - X.mean(axis=0))**2)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        Wk = 0
        for label in range(k):
            cluster_points = X[labels == label]
            if len(cluster_points) > 0:
                Wk += np.sum((cluster_points - cluster_points.mean(axis=0))**2)
        return Wk
    
    # Observed Wk for each k
    Wks = []
    for k in tqdm(k_range, desc="Computing Wk", leave=False):
        Wks.append(np.log(compute_Wk(X, k)))
    Wks = np.array(Wks)
    
    # Reference distribution (uniform over bounding box)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    
    ref_Wks = np.zeros((n_references, len(k_range)))
    for b in tqdm(range(n_references), desc="Reference samples", leave=False):
        X_ref = rng.uniform(mins, maxs, size=X.shape)
        for i, k in enumerate(k_range):
            ref_Wks[b, i] = np.log(compute_Wk(X_ref, k))
    
    # Gap statistic
    gaps = ref_Wks.mean(axis=0) - Wks
    gap_sds = ref_Wks.std(axis=0) * np.sqrt(1 + 1/n_references)
    
    # Optimal k: first k where Gap(k) >= Gap(k+1) - s(k+1)
    optimal_k = list(k_range)[0]
    for i in range(len(k_range) - 1):
        if gaps[i] >= gaps[i+1] - gap_sds[i+1]:
            optimal_k = list(k_range)[i]
            break
    else:
        optimal_k = list(k_range)[np.argmax(gaps)]
    
    return {
        'k_range': list(k_range),
        'gaps': gaps.tolist(),
        'gap_sds': gap_sds.tolist(),
        'optimal_k': optimal_k
    }

def train_test_reproducibility(X, k, n_splits=20, test_size=0.2, random_state=42):
    """
    Test cluster reproducibility on held-out data.
    
    1. Fit clusters on training set (80%)
    2. Assign test set to nearest cluster
    3. Compare to clustering test set independently
    4. Measure agreement with ARI
    """
    rng = np.random.default_rng(random_state)
    
    aris = []
    for i in tqdm(range(n_splits), desc=f"Train/test splits k={k}", leave=False):
        X_train, X_test = train_test_split(X, test_size=test_size, 
                                            random_state=rng.integers(10000))
        
        # Train on training set
        train_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        train_model.fit(X_train)
        
        # Predict on test set using trained model
        test_labels_from_train = train_model.predict(X_test)
        
        # Independent clustering on test set
        test_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        test_labels_independent = test_model.fit_predict(X_test)
        
        # Agreement
        ari = adjusted_rand_score(test_labels_from_train, test_labels_independent)
        aris.append(ari)
    
    return {
        'mean_ari': np.mean(aris),
        'std_ari': np.std(aris),
        'min_ari': np.min(aris),
        'max_ari': np.max(aris)
    }

def main():
    print("=" * 70)
    print("PHASE 1: CLUSTER VALIDATION ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load data
    fits_df = load_kernel_fits()
    
    # Prepare features
    feature_cols = ['tau1', 'tau2', 'A', 'B']
    X = fits_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nFeatures: {feature_cols}")
    print(f"N samples: {len(X_scaled)}")
    print()
    
    results = {}
    
    # =========================================================================
    # 1. PERMUTATION TEST
    # =========================================================================
    print("=" * 70)
    print("1. PERMUTATION TEST FOR CLUSTER SIGNIFICANCE")
    print("=" * 70)
    print("H0: Cluster structure is no better than random")
    print()
    
    perm_results = {}
    for k in [2, 3, 4, 5]:
        observed, null_dist, p_value = permutation_test(X_scaled, k, n_permutations=500)
        
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        
        print(f"k={k}: Silhouette={observed:.3f}, p={p_value:.4f} {significance}")
        print(f"      Null distribution: mean={null_dist.mean():.3f}, std={null_dist.std():.3f}")
        
        perm_results[k] = {
            'observed_silhouette': observed,
            'null_mean': null_dist.mean(),
            'null_std': null_dist.std(),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    results['permutation_test'] = perm_results
    print()
    
    # =========================================================================
    # 2. GAP STATISTIC
    # =========================================================================
    print("=" * 70)
    print("2. GAP STATISTIC FOR OPTIMAL K")
    print("=" * 70)
    
    gap_results = gap_statistic(X_scaled, k_range=range(1, 8), n_references=20)
    
    print(f"\nGap statistic by k:")
    for i, k in enumerate(gap_results['k_range']):
        gap = gap_results['gaps'][i]
        sd = gap_results['gap_sds'][i]
        marker = " <-- OPTIMAL" if k == gap_results['optimal_k'] else ""
        print(f"  k={k}: Gap={gap:.3f} ± {sd:.3f}{marker}")
    
    print(f"\nOptimal k (Gap statistic): {gap_results['optimal_k']}")
    
    results['gap_statistic'] = gap_results
    print()
    
    # =========================================================================
    # 3. TRAIN/TEST REPRODUCIBILITY
    # =========================================================================
    print("=" * 70)
    print("3. TRAIN/TEST REPRODUCIBILITY")
    print("=" * 70)
    print("Testing if clusters replicate on held-out data")
    print()
    
    repro_results = {}
    for k in [2, 3, 4, 5]:
        repro = train_test_reproducibility(X_scaled, k, n_splits=20)
        
        quality = "Excellent" if repro['mean_ari'] > 0.8 else "Good" if repro['mean_ari'] > 0.6 else "Moderate" if repro['mean_ari'] > 0.4 else "Poor"
        
        print(f"k={k}: ARI={repro['mean_ari']:.3f} ± {repro['std_ari']:.3f} [{quality}]")
        print(f"      Range: [{repro['min_ari']:.3f}, {repro['max_ari']:.3f}]")
        
        repro_results[k] = repro
    
    results['reproducibility'] = repro_results
    print()
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    # Determine best k based on all criteria
    k_scores = {}
    for k in [2, 3, 4, 5]:
        score = 0
        
        # Permutation test significant?
        if perm_results[k]['significant']:
            score += 1
        
        # Close to gap optimal?
        if k == gap_results['optimal_k']:
            score += 2
        elif abs(k - gap_results['optimal_k']) == 1:
            score += 1
        
        # Good reproducibility?
        if repro_results[k]['mean_ari'] > 0.6:
            score += 2
        elif repro_results[k]['mean_ari'] > 0.4:
            score += 1
        
        k_scores[k] = score
    
    best_k = max(k_scores, key=k_scores.get)
    
    print(f"\n{'k':<5} {'Perm.Test':<12} {'Gap Optimal':<12} {'Reproducibility':<15} {'Score':<8}")
    print("-" * 55)
    for k in [2, 3, 4, 5]:
        perm = "✓" if perm_results[k]['significant'] else "✗"
        gap = "✓" if k == gap_results['optimal_k'] else "~" if abs(k - gap_results['optimal_k']) == 1 else "✗"
        repro = f"{repro_results[k]['mean_ari']:.2f}"
        score = k_scores[k]
        marker = " <-- BEST" if k == best_k else ""
        print(f"k={k:<3} {perm:<12} {gap:<12} {repro:<15} {score:<8}{marker}")
    
    results['summary'] = {
        'best_k': best_k,
        'k_scores': k_scores,
        'gap_optimal_k': gap_results['optimal_k']
    }
    
    # Conclusion
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    if perm_results[best_k]['significant'] and repro_results[best_k]['mean_ari'] > 0.5:
        print(f"\n✓ CLUSTERS ARE STATISTICALLY VALIDATED")
        print(f"  - Best k = {best_k} (supported by multiple criteria)")
        print(f"  - Permutation test p < 0.05: clusters are non-random")
        print(f"  - Reproducibility ARI = {repro_results[best_k]['mean_ari']:.2f}: clusters replicate on held-out data")
        print(f"\n  → PROCEED TO PHASE 2: Cluster characterization")
    else:
        print(f"\n⚠ CLUSTER VALIDATION INCONCLUSIVE")
        print(f"  - Consider alternative clustering methods or k values")
    
    # Save results
    output_path = OUTPUT_DIR / 'cluster_validation_results.json'
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        return obj
    
    results = convert_numpy(results)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == '__main__':
    main()


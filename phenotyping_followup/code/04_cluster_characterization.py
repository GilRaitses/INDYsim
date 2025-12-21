#!/usr/bin/env python3
"""
Phase 2: Cluster Characterization

Characterize validated phenotypic clusters:
1. Extract cluster centroids (mean kernel params per cluster)
2. ANOVA/Kruskal-Wallis for each parameter
3. Discriminant analysis (which params separate clusters?)
4. Visualize cluster profiles

Runtime: ~2 minutes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
FITS_PATH = RESULTS_DIR / 'empirical_10min_kernel_fits_v2.csv'
OUTPUT_DIR = RESULTS_DIR / 'characterization'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load kernel fits and assign clusters."""
    df = pd.read_csv(FITS_PATH)
    print(f"Loaded {len(df)} kernel fits")
    
    # Prepare features
    feature_cols = ['tau1', 'tau2', 'A', 'B']
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Assign clusters for k=4 (best validated)
    for k in [3, 4, 5]:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        df[f'cluster_k{k}'] = model.fit_predict(X_scaled)
    
    return df, feature_cols, scaler

def extract_centroids(df, feature_cols, k=4):
    """Extract cluster centroids."""
    cluster_col = f'cluster_k{k}'
    
    centroids = df.groupby(cluster_col)[feature_cols].agg(['mean', 'std', 'count'])
    
    print(f"\n{'='*70}")
    print(f"CLUSTER CENTROIDS (k={k})")
    print(f"{'='*70}")
    
    # Flatten column names
    centroid_flat = {}
    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]
        n = len(cluster_data)
        print(f"\nCluster {cluster_id} (n={n}, {100*n/len(df):.1f}%):")
        
        centroid_flat[int(cluster_id)] = {'n': n, 'pct': round(100*n/len(df), 1)}
        
        for col in feature_cols:
            mean = cluster_data[col].mean()
            std = cluster_data[col].std()
            print(f"  {col:>6}: {mean:>8.3f} ± {std:.3f}")
            centroid_flat[int(cluster_id)][col] = {'mean': round(mean, 4), 'std': round(std, 4)}
    
    return centroid_flat

def statistical_tests(df, feature_cols, k=4):
    """Run ANOVA/Kruskal-Wallis for each parameter."""
    cluster_col = f'cluster_k{k}'
    
    print(f"\n{'='*70}")
    print(f"STATISTICAL TESTS: Do parameters differ across clusters?")
    print(f"{'='*70}")
    
    results = {}
    
    for col in feature_cols:
        groups = [df[df[cluster_col] == c][col].values for c in sorted(df[cluster_col].unique())]
        
        # Kruskal-Wallis (non-parametric)
        kw_stat, kw_p = stats.kruskal(*groups)
        
        # One-way ANOVA (parametric)
        f_stat, anova_p = stats.f_oneway(*groups)
        
        # Effect size (eta-squared)
        grand_mean = df[col].mean()
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum((df[col] - grand_mean)**2)
        eta_sq = ss_between / ss_total
        
        sig = "***" if kw_p < 0.001 else "**" if kw_p < 0.01 else "*" if kw_p < 0.05 else "n.s."
        effect = "Large" if eta_sq > 0.14 else "Medium" if eta_sq > 0.06 else "Small"
        
        print(f"\n{col}:")
        print(f"  Kruskal-Wallis: H={kw_stat:.2f}, p={kw_p:.4f} {sig}")
        print(f"  ANOVA: F={f_stat:.2f}, p={anova_p:.4f}")
        print(f"  Effect size (η²): {eta_sq:.3f} [{effect}]")
        
        results[col] = {
            'kruskal_wallis_H': round(kw_stat, 3),
            'kruskal_wallis_p': round(kw_p, 6),
            'anova_F': round(f_stat, 3),
            'anova_p': round(anova_p, 6),
            'eta_squared': round(eta_sq, 4),
            'significant': kw_p < 0.05,
            'effect_size': effect
        }
    
    return results

def discriminant_analysis(df, feature_cols, k=4):
    """Linear Discriminant Analysis to find separating dimensions."""
    cluster_col = f'cluster_k{k}'
    
    print(f"\n{'='*70}")
    print(f"DISCRIMINANT ANALYSIS: Which parameters best separate clusters?")
    print(f"{'='*70}")
    
    X = df[feature_cols].values
    y = df[cluster_col].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit LDA
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_scaled, y)
    
    # Get coefficients (loadings)
    n_components = min(k-1, len(feature_cols))
    
    print(f"\nDiscriminant function loadings (standardized):")
    print(f"Higher absolute values = more discriminating power")
    print()
    
    results = {'loadings': {}, 'explained_variance_ratio': []}
    
    for i in range(n_components):
        print(f"LD{i+1} (explains {100*lda.explained_variance_ratio_[i]:.1f}% of between-class variance):")
        results['explained_variance_ratio'].append(round(lda.explained_variance_ratio_[i], 4))
        
        loadings = lda.scalings_[:, i]
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        
        for j in sorted_idx:
            col = feature_cols[j]
            loading = loadings[j]
            bar = "█" * int(abs(loading) * 10)
            sign = "+" if loading > 0 else "-"
            print(f"  {col:>6}: {sign}{abs(loading):.3f} {bar}")
            
            if f'LD{i+1}' not in results['loadings']:
                results['loadings'][f'LD{i+1}'] = {}
            results['loadings'][f'LD{i+1}'][col] = round(loading, 4)
        print()
    
    # Classification accuracy (leave-one-out)
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(lda, X_scaled, y, cv=10)
    print(f"Classification accuracy (10-fold CV): {100*cv_scores.mean():.1f}% ± {100*cv_scores.std():.1f}%")
    
    results['cv_accuracy_mean'] = round(cv_scores.mean(), 4)
    results['cv_accuracy_std'] = round(cv_scores.std(), 4)
    
    return results

def rare_phenotype_analysis(df, feature_cols, k=4):
    """Analyze the small distinct cluster (rare phenotype)."""
    cluster_col = f'cluster_k{k}'
    
    print(f"\n{'='*70}")
    print(f"RARE PHENOTYPE ANALYSIS")
    print(f"{'='*70}")
    
    # Find smallest cluster
    cluster_sizes = df[cluster_col].value_counts()
    rare_cluster = cluster_sizes.idxmin()
    rare_n = cluster_sizes.min()
    
    print(f"\nSmallest cluster: {rare_cluster} (n={rare_n}, {100*rare_n/len(df):.1f}%)")
    
    rare_df = df[df[cluster_col] == rare_cluster]
    other_df = df[df[cluster_col] != rare_cluster]
    
    results = {'cluster_id': int(rare_cluster), 'n': int(rare_n), 'comparisons': {}}
    
    print(f"\nComparing rare phenotype to all others:")
    print(f"{'Parameter':<10} {'Rare Mean':>12} {'Others Mean':>12} {'Diff':>10} {'p-value':>10}")
    print("-" * 60)
    
    for col in feature_cols:
        rare_vals = rare_df[col].values
        other_vals = other_df[col].values
        
        rare_mean = np.mean(rare_vals)
        other_mean = np.mean(other_vals)
        diff = rare_mean - other_mean
        
        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(rare_vals, other_vals, alternative='two-sided')
        
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        
        print(f"{col:<10} {rare_mean:>12.3f} {other_mean:>12.3f} {diff:>+10.3f} {p:>10.4f} {sig}")
        
        results['comparisons'][col] = {
            'rare_mean': round(rare_mean, 4),
            'others_mean': round(other_mean, 4),
            'difference': round(diff, 4),
            'mann_whitney_p': round(p, 6),
            'significant': p < 0.05
        }
    
    # Check if rare phenotype tracks come from specific experiments
    print(f"\nRare phenotype by experiment:")
    exp_counts = rare_df['experiment_id'].value_counts()
    for exp, count in exp_counts.items():
        exp_total = len(df[df['experiment_id'] == exp])
        print(f"  {exp}: {count}/{exp_total} ({100*count/exp_total:.1f}%)")
    
    return results

def main():
    print("=" * 70)
    print("PHASE 2: CLUSTER CHARACTERIZATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df, feature_cols, scaler = load_data()
    
    # Use k=4 as primary (highest reproducibility)
    k = 4
    
    results = {'k': k, 'n_tracks': len(df)}
    
    # 1. Extract centroids
    results['centroids'] = extract_centroids(df, feature_cols, k=k)
    
    # 2. Statistical tests
    results['statistical_tests'] = statistical_tests(df, feature_cols, k=k)
    
    # 3. Discriminant analysis
    results['discriminant_analysis'] = discriminant_analysis(df, feature_cols, k=k)
    
    # 4. Rare phenotype analysis
    results['rare_phenotype'] = rare_phenotype_analysis(df, feature_cols, k=k)
    
    # Summary
    print(f"\n{'='*70}")
    print("CHARACTERIZATION SUMMARY")
    print(f"{'='*70}")
    
    sig_params = [p for p, r in results['statistical_tests'].items() if r['significant']]
    print(f"\nParameters that significantly differ across clusters:")
    for p in sig_params:
        r = results['statistical_tests'][p]
        print(f"  {p}: η² = {r['eta_squared']:.3f} [{r['effect_size']}]")
    
    print(f"\nDiscriminant analysis classification accuracy: {100*results['discriminant_analysis']['cv_accuracy_mean']:.1f}%")
    
    rare = results['rare_phenotype']
    print(f"\nRare phenotype (cluster {rare['cluster_id']}, n={rare['n']}):")
    for param, comp in rare['comparisons'].items():
        if comp['significant']:
            direction = "higher" if comp['difference'] > 0 else "lower"
            print(f"  {param}: {direction} than others (p < 0.05)")
    
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
    
    # Save results
    output_path = OUTPUT_DIR / 'cluster_characterization.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save cluster assignments
    cluster_path = OUTPUT_DIR / 'track_cluster_assignments.csv'
    df[['experiment_id', 'track_id', 'tau1', 'tau2', 'A', 'B', 'cluster_k3', 'cluster_k4', 'cluster_k5']].to_csv(cluster_path, index=False)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Cluster assignments saved to: {cluster_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == '__main__':
    main()


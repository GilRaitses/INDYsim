#!/usr/bin/env python3
"""
Cross-Validation of Clustering

Validates clustering by:
1. Bootstrap stability testing
2. Different random seeds
3. Silhouette analysis per cluster
4. Cluster assignment reproducibility
"""

import sys
import os
from pathlib import Path
import time
from datetime import datetime

# Add InDySim code directory to path
INDYSIM_CODE = Path('/Users/gilraitses/InDySim/code')
if INDYSIM_CODE.exists() and str(INDYSIM_CODE) not in sys.path:
    sys.path.insert(0, str(INDYSIM_CODE))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from phenotyping_analysis_pipeline import ProgressMonitor, print_header, print_status


def bootstrap_clustering(features_df, n_bootstrap=100, n_clusters=5, seed=42):
    """
    Bootstrap clustering stability test.
    
    Parameters
    ----------
    features_df : DataFrame
        Feature matrix
    n_bootstrap : int
        Number of bootstrap samples
    n_clusters : int
        Number of clusters
    seed : int
        Random seed
    
    Returns
    -------
    stability_results : dict
        Cluster stability statistics
    """
    print_header("CLUSTER STABILITY: Bootstrap Testing")
    
    # Prepare features
    feature_cols = ['tau1', 'tau2', 'amplitude_A', 'amplitude_B', 
                    'turn_rate', 'mean_turn_duration', 'run_fraction']
    
    clustering_df = features_df.dropna(subset=['tau1', 'tau2']).copy()
    X = clustering_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print_status("STABILITY", f"Testing stability with {n_bootstrap} bootstrap samples...", "INFO")
    
    # Store cluster assignments for each bootstrap
    bootstrap_assignments = []
    
    rng = np.random.default_rng(seed)
    
    if HAS_TQDM:
        iterator = tqdm(range(n_bootstrap), desc="Bootstrap", unit="sample")
    else:
        iterator = range(n_bootstrap)
        monitor = ProgressMonitor(n_bootstrap, desc="Bootstrap")
    
    for i in iterator:
        # Bootstrap sample (with replacement)
        boot_indices = rng.choice(len(X_scaled), size=len(X_scaled), replace=True)
        X_boot = X_scaled[boot_indices]
        
        # Cluster bootstrap sample
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed+i, n_init=10)
        labels_boot = kmeans.fit_predict(X_boot)
        
        # Map back to original indices
        labels_full = np.full(len(X_scaled), -1)
        for j, orig_idx in enumerate(boot_indices):
            if labels_full[orig_idx] == -1:
                labels_full[orig_idx] = labels_boot[j]
        
        bootstrap_assignments.append(labels_full)
        
        if not HAS_TQDM:
            monitor.update(1, phase="STABILITY")
    
    if not HAS_TQDM:
        monitor.finish()
    
    # Compute stability metrics
    bootstrap_assignments = np.array(bootstrap_assignments)
    
    # Agreement matrix: how often do pairs of tracks cluster together?
    n_tracks = len(X_scaled)
    agreement_matrix = np.zeros((n_tracks, n_tracks))
    
    for i in range(n_tracks):
        for j in range(i+1, n_tracks):
            # Count how often i and j are in same cluster
            same_cluster = (bootstrap_assignments[:, i] == bootstrap_assignments[:, j]).sum()
            agreement_matrix[i, j] = same_cluster / n_bootstrap
            agreement_matrix[j, i] = agreement_matrix[i, j]
    
    # Mean agreement
    mean_agreement = np.mean(agreement_matrix[np.triu_indices(n_tracks, k=1)])
    
    # Cluster assignment consistency
    # For each track, find most common cluster assignment
    most_common_clusters = []
    for i in range(n_tracks):
        cluster_counts = np.bincount(bootstrap_assignments[:, i] + 1)  # +1 to handle -1
        most_common = np.argmax(cluster_counts) - 1  # -1 to convert back
        consistency = cluster_counts[most_common] / n_bootstrap
        most_common_clusters.append((most_common, consistency))
    
    mean_consistency = np.mean([c[1] for c in most_common_clusters])
    
    stability_results = {
        'mean_agreement': mean_agreement,
        'mean_consistency': mean_consistency,
        'agreement_matrix': agreement_matrix,
        'most_common_clusters': most_common_clusters
    }
    
    print_status("STABILITY", f"Mean agreement: {mean_agreement:.3f}", "INFO")
    print_status("STABILITY", f"Mean consistency: {mean_consistency:.3f}", "INFO")
    
    return stability_results


def test_seed_sensitivity(features_df, n_seeds=20, n_clusters=5):
    """
    Test clustering sensitivity to random seed.
    
    Parameters
    ----------
    features_df : DataFrame
        Feature matrix
    n_seeds : int
        Number of different seeds to test
    n_clusters : int
        Number of clusters
    
    Returns
    -------
    seed_results : dict
        Results for each seed
    """
    print_header("SEED SENSITIVITY: Random Seed Testing")
    
    feature_cols = ['tau1', 'tau2', 'amplitude_A', 'amplitude_B', 
                    'turn_rate', 'mean_turn_duration', 'run_fraction']
    
    clustering_df = features_df.dropna(subset=['tau1', 'tau2']).copy()
    X = clustering_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print_status("SEED", f"Testing {n_seeds} different random seeds...", "INFO")
    
    seed_results = []
    
    if HAS_TQDM:
        iterator = tqdm(range(n_seeds), desc="Seed test", unit="seed")
    else:
        iterator = range(n_seeds)
        monitor = ProgressMonitor(n_seeds, desc="Seed test")
    
    for seed in iterator:
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        silhouette = silhouette_score(X_scaled, labels)
        
        seed_results.append({
            'seed': seed,
            'silhouette_score': silhouette,
            'labels': labels
        })
        
        if not HAS_TQDM:
            monitor.update(1, phase="SEED")
    
    if not HAS_TQDM:
        monitor.finish()
    
    # Compare cluster assignments across seeds
    seed_labels = np.array([r['labels'] for r in seed_results])
    
    # Adjusted Rand Index between first seed and others
    ari_scores = []
    for i in range(1, n_seeds):
        ari = adjusted_rand_score(seed_labels[0], seed_labels[i])
        ari_scores.append(ari)
    
    mean_ari = np.mean(ari_scores)
    
    print_status("SEED", f"Mean ARI vs seed 0: {mean_ari:.3f}", "INFO")
    
    return {
        'seed_results': seed_results,
        'mean_ari': mean_ari,
        'ari_scores': ari_scores
    }


def per_cluster_silhouette(features_df, n_clusters=5, seed=42):
    """
    Compute silhouette scores per cluster.
    
    Parameters
    ----------
    features_df : DataFrame
        Feature matrix
    n_clusters : int
        Number of clusters
    seed : int
        Random seed
    
    Returns
    -------
    silhouette_results : dict
        Per-cluster silhouette scores
    """
    print_header("SILHOUETTE: Per-Cluster Analysis")
    
    feature_cols = ['tau1', 'tau2', 'amplitude_A', 'amplitude_B', 
                    'turn_rate', 'mean_turn_duration', 'run_fraction']
    
    clustering_df = features_df.dropna(subset=['tau1', 'tau2']).copy()
    X = clustering_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Compute per-cluster silhouette
    from sklearn.metrics import silhouette_samples
    silhouette_samples_vals = silhouette_samples(X_scaled, labels)
    
    # Per-cluster statistics
    cluster_silhouettes = {}
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_sil = silhouette_samples_vals[cluster_mask]
        
        cluster_silhouettes[cluster_id] = {
            'mean': np.mean(cluster_sil),
            'std': np.std(cluster_sil),
            'min': np.min(cluster_sil),
            'max': np.max(cluster_sil),
            'n_tracks': cluster_mask.sum()
        }
        
        print_status("SILHOUETTE", 
            f"Cluster {cluster_id}: mean={np.mean(cluster_sil):.3f}, n={cluster_mask.sum()}",
            "INFO" if np.mean(cluster_sil) > 0.2 else "WARNING")
    
    overall_silhouette = silhouette_score(X_scaled, labels)
    print_status("SILHOUETTE", f"Overall silhouette: {overall_silhouette:.3f}", "INFO")
    
    return {
        'per_cluster': cluster_silhouettes,
        'overall': overall_silhouette,
        'labels': labels
    }


def main():
    """Main clustering CV pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-validation of clustering')
    parser.add_argument('--features', type=str,
                       default='/Users/gilraitses/InDySim/results/phenotyping_analysis_v2/phenotype_features.csv',
                       help='Path to features CSV')
    parser.add_argument('--output-dir', type=str,
                       default='/Users/gilraitses/InDySim/results/phenotyping_analysis_v2/cv',
                       help='Output directory for CV results')
    parser.add_argument('--n-clusters', type=int, default=5,
                       help='Number of clusters')
    parser.add_argument('--n-bootstrap', type=int, default=100,
                       help='Number of bootstrap samples')
    parser.add_argument('--n-seeds', type=int, default=20,
                       help='Number of random seeds to test')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print_header("CLUSTERING CROSS-VALIDATION PIPELINE", width=80)
    
    # Load features
    features_df = pd.read_csv(args.features)
    print_status("LOAD", f"Loaded {len(features_df)} tracks", "INFO")
    
    # 1. Bootstrap stability
    stability_results = bootstrap_clustering(
        features_df, n_bootstrap=args.n_bootstrap, n_clusters=args.n_clusters
    )
    
    # Save stability results
    stability_df = pd.DataFrame([{
        'mean_agreement': stability_results['mean_agreement'],
        'mean_consistency': stability_results['mean_consistency']
    }])
    stability_df.to_csv(output_dir / 'cluster_stability.csv', index=False)
    
    # 2. Seed sensitivity
    seed_results = test_seed_sensitivity(
        features_df, n_seeds=args.n_seeds, n_clusters=args.n_clusters
    )
    
    # Save seed results
    seed_df = pd.DataFrame([{
        'mean_ari': seed_results['mean_ari'],
        'min_ari': np.min(seed_results['ari_scores']),
        'max_ari': np.max(seed_results['ari_scores'])
    }])
    seed_df.to_csv(output_dir / 'seed_sensitivity.csv', index=False)
    
    # 3. Per-cluster silhouette
    silhouette_results = per_cluster_silhouette(
        features_df, n_clusters=args.n_clusters
    )
    
    # Save silhouette results
    silhouette_df = pd.DataFrame(silhouette_results['per_cluster']).T
    silhouette_df.to_csv(output_dir / 'per_cluster_silhouette.csv')
    
    print_header("CLUSTERING CV COMPLETE", width=80)
    print_status("DONE", f"Results saved to: {output_dir}", "SUCCESS")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  Cluster stability (agreement): {stability_results['mean_agreement']:.3f}")
    print(f"  Seed sensitivity (mean ARI): {seed_results['mean_ari']:.3f}")
    print(f"  Overall silhouette: {silhouette_results['overall']:.3f}")


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Deep EDA: PCA and Feature Analysis for Improved Phenotyping

Explore what features actually distinguish tracks:
1. PSTH-based PCA (raw event patterns)
2. Kernel parameter PCA (current approach)
3. Cross-correlation analysis
4. Information content in different representations

Runtime: ~2 minutes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
FITS_PATH = RESULTS_DIR / 'empirical_10min_kernel_fits_v2.csv'
H5_PATH = Path('/Users/gilraitses/INDYsim_project/data/processed/consolidated_dataset.h5')
OUTPUT_DIR = RESULTS_DIR / 'deep_eda'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# PSTH parameters
PSTH_BINS = np.linspace(0, 10, 21)  # 0-10s in 0.5s bins
LED_CYCLE = 30.0
FIRST_LED_ONSET = 21.3

def load_events():
    """Load event data from H5."""
    import h5py
    
    with h5py.File(H5_PATH, 'r') as f:
        exp_id = f['events']['experiment_id'][:]
        track_id = f['events']['track_id'][:]
        time = f['events']['time'][:]
        is_reo = f['events']['is_reorientation_start'][:]
        
        if exp_id.dtype.kind == 'S':
            exp_id = np.array([x.decode() for x in exp_id])
    
    df = pd.DataFrame({
        'experiment_id': exp_id,
        'track_id': track_id,
        'time': time,
        'is_reo_start': is_reo
    })
    
    return df

def compute_track_psth(events_df, exp_id, track_id, bins=PSTH_BINS):
    """Compute PSTH for a single track."""
    track_events = events_df[
        (events_df['experiment_id'] == exp_id) & 
        (events_df['track_id'] == track_id) &
        (events_df['is_reo_start'] == True)
    ]['time'].values
    
    # Compute LED onsets
    led_onsets = np.arange(FIRST_LED_ONSET, 1200, LED_CYCLE)
    
    # Align events to LED onsets
    aligned_times = []
    for event_time in track_events:
        onsets_before = led_onsets[led_onsets <= event_time]
        if len(onsets_before) > 0:
            t_since = event_time - onsets_before[-1]
            if t_since < 10:  # LED-ON window
                aligned_times.append(t_since)
    
    # Bin into PSTH
    psth, _ = np.histogram(aligned_times, bins=bins)
    
    # Normalize by number of LED cycles
    n_cycles = len(led_onsets)
    psth_rate = psth / (n_cycles * np.diff(bins))  # Events per second per cycle
    
    return psth_rate

def main():
    print("=" * 70)
    print("DEEP EDA: PCA AND FEATURE ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load kernel fits
    fits_df = pd.read_csv(FITS_PATH)
    print(f"Loaded {len(fits_df)} kernel fits")
    
    # Load events
    print("Loading event data...")
    events_df = load_events()
    print(f"Loaded {len(events_df)} event rows")
    
    # =========================================================================
    # 1. COMPUTE PSTH FOR EACH TRACK
    # =========================================================================
    print(f"\n{'='*70}")
    print("1. COMPUTING TRACK-LEVEL PSTH")
    print(f"{'='*70}")
    
    psth_matrix = []
    valid_tracks = []
    
    from tqdm import tqdm
    for _, row in tqdm(fits_df.iterrows(), total=len(fits_df), desc="Computing PSTH"):
        psth = compute_track_psth(events_df, row['experiment_id'], row['track_id'])
        if np.sum(psth) > 0:  # Has events in LED-ON window
            psth_matrix.append(psth)
            valid_tracks.append((row['experiment_id'], row['track_id']))
    
    psth_matrix = np.array(psth_matrix)
    print(f"Computed PSTH for {len(psth_matrix)} tracks")
    print(f"PSTH shape: {psth_matrix.shape} (tracks × time bins)")
    
    # =========================================================================
    # 2. PCA ON PSTH
    # =========================================================================
    print(f"\n{'='*70}")
    print("2. PCA ON PSTH (Raw Event Patterns)")
    print(f"{'='*70}")
    
    # Normalize PSTH
    psth_scaler = StandardScaler()
    psth_scaled = psth_scaler.fit_transform(psth_matrix)
    
    # PCA
    pca_psth = PCA()
    psth_pcs = pca_psth.fit_transform(psth_scaled)
    
    print(f"\nVariance explained by each PC:")
    cumvar = np.cumsum(pca_psth.explained_variance_ratio_)
    for i in range(min(10, len(pca_psth.explained_variance_ratio_))):
        print(f"  PC{i+1}: {100*pca_psth.explained_variance_ratio_[i]:.1f}% (cumulative: {100*cumvar[i]:.1f}%)")
    
    # Find how many PCs for 80%, 90%, 95%
    n_80 = np.searchsorted(cumvar, 0.80) + 1
    n_90 = np.searchsorted(cumvar, 0.90) + 1
    n_95 = np.searchsorted(cumvar, 0.95) + 1
    print(f"\nPCs needed for: 80% → {n_80}, 90% → {n_90}, 95% → {n_95}")
    
    # =========================================================================
    # 3. PCA ON KERNEL PARAMETERS
    # =========================================================================
    print(f"\n{'='*70}")
    print("3. PCA ON KERNEL PARAMETERS (Current Approach)")
    print(f"{'='*70}")
    
    # Filter to valid tracks
    valid_exp_track = set(valid_tracks)
    fits_valid = fits_df[fits_df.apply(lambda r: (r['experiment_id'], r['track_id']) in valid_exp_track, axis=1)]
    
    kernel_params = fits_valid[['tau1', 'tau2', 'A', 'B']].values
    kernel_scaler = StandardScaler()
    kernel_scaled = kernel_scaler.fit_transform(kernel_params)
    
    pca_kernel = PCA()
    kernel_pcs = pca_kernel.fit_transform(kernel_scaled)
    
    print(f"\nVariance explained by each PC:")
    cumvar_k = np.cumsum(pca_kernel.explained_variance_ratio_)
    for i in range(4):
        print(f"  PC{i+1}: {100*pca_kernel.explained_variance_ratio_[i]:.1f}% (cumulative: {100*cumvar_k[i]:.1f}%)")
    
    # =========================================================================
    # 4. COMPARE CLUSTERING ON DIFFERENT REPRESENTATIONS
    # =========================================================================
    print(f"\n{'='*70}")
    print("4. CLUSTERING COMPARISON: PSTH vs KERNEL PARAMS")
    print(f"{'='*70}")
    
    # Cluster on PSTH PCs (first 3)
    psth_features = psth_pcs[:, :3]
    
    # Cluster on kernel params
    kernel_features = kernel_scaled
    
    print(f"\n{'Method':<25} {'k':<5} {'Silhouette':<12} {'Notes'}")
    print("-" * 55)
    
    results = {'psth_clustering': {}, 'kernel_clustering': {}}
    
    for k in [3, 4, 5]:
        # PSTH clustering
        psth_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(psth_features)
        psth_sil = silhouette_score(psth_features, psth_labels)
        
        # Kernel clustering
        kernel_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(kernel_features)
        kernel_sil = silhouette_score(kernel_features, kernel_labels)
        
        # Agreement between methods
        ari = adjusted_rand_score(psth_labels, kernel_labels)
        
        print(f"PSTH (3 PCs)             k={k}   {psth_sil:.3f}")
        print(f"Kernel (4 params)        k={k}   {kernel_sil:.3f}        ARI vs PSTH: {ari:.3f}")
        print()
        
        results['psth_clustering'][k] = {'silhouette': round(psth_sil, 4)}
        results['kernel_clustering'][k] = {'silhouette': round(kernel_sil, 4), 'ari_vs_psth': round(ari, 4)}
    
    # =========================================================================
    # 5. WHAT DO PSTH PCs MEAN?
    # =========================================================================
    print(f"\n{'='*70}")
    print("5. INTERPRETING PSTH PRINCIPAL COMPONENTS")
    print(f"{'='*70}")
    
    bin_centers = (PSTH_BINS[:-1] + PSTH_BINS[1:]) / 2
    
    print(f"\nPC loadings (what each PC captures):")
    for i in range(3):
        loadings = pca_psth.components_[i]
        peak_bin = bin_centers[np.argmax(np.abs(loadings))]
        peak_sign = "+" if loadings[np.argmax(np.abs(loadings))] > 0 else "-"
        
        # Interpret
        if i == 0:
            interp = "Overall event rate (baseline)"
        elif i == 1:
            if peak_bin < 3:
                interp = "Early response (excitation timing)"
            else:
                interp = "Late response (suppression timing)"
        else:
            interp = "Response shape detail"
        
        print(f"\n  PC{i+1} ({100*pca_psth.explained_variance_ratio_[i]:.1f}% variance):")
        print(f"    Peak at {peak_bin:.1f}s ({peak_sign})")
        print(f"    Interpretation: {interp}")
    
    # =========================================================================
    # 6. CORRELATION BETWEEN REPRESENTATIONS
    # =========================================================================
    print(f"\n{'='*70}")
    print("6. CORRELATION: PSTH PCs vs KERNEL PARAMS")
    print(f"{'='*70}")
    
    print(f"\n{'':>15} {'PC1':>10} {'PC2':>10} {'PC3':>10}")
    print("-" * 50)
    
    param_names = ['tau1', 'tau2', 'A', 'B']
    for i, param in enumerate(param_names):
        corrs = []
        for j in range(3):
            r, _ = stats.pearsonr(kernel_params[:, i], psth_pcs[:, j])
            corrs.append(r)
        print(f"{param:>15} {corrs[0]:>10.3f} {corrs[1]:>10.3f} {corrs[2]:>10.3f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("DEEP EDA SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n1. PSTH dimensionality: {n_90} PCs explain 90% of variance")
    print(f"   → Event patterns are relatively low-dimensional")
    
    print(f"\n2. PSTH vs Kernel clustering:")
    ari_k4 = results['kernel_clustering'][4]['ari_vs_psth']
    if ari_k4 > 0.5:
        print(f"   → High agreement (ARI={ari_k4:.2f}): Both capture similar structure")
    else:
        print(f"   → Low agreement (ARI={ari_k4:.2f}): Different aspects of behavior")
    
    print(f"\n3. Recommendation:")
    if results['psth_clustering'][4]['silhouette'] > results['kernel_clustering'][4]['silhouette']:
        print(f"   → PSTH-based clustering has BETTER separation")
        print(f"   → Consider using PSTH PCs for phenotyping")
    else:
        print(f"   → Kernel-based clustering has better separation")
        print(f"   → But round-trip validation failed, so interpret cautiously")
    
    # Save results
    results['psth_pca'] = {
        'variance_explained': pca_psth.explained_variance_ratio_.tolist(),
        'n_pcs_80': int(n_80),
        'n_pcs_90': int(n_90),
        'n_pcs_95': int(n_95)
    }
    results['kernel_pca'] = {
        'variance_explained': pca_kernel.explained_variance_ratio_.tolist()
    }
    
    output_path = OUTPUT_DIR / 'deep_eda_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save PSTH matrix for potential FNO training
    np.save(OUTPUT_DIR / 'psth_matrix.npy', psth_matrix)
    np.save(OUTPUT_DIR / 'psth_pcs.npy', psth_pcs)
    
    print(f"\nResults saved to: {output_path}")
    print(f"PSTH matrix saved for potential FNO training")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == '__main__':
    main()


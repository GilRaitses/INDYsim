#!/usr/bin/env python3
"""
Phase 3: External Validation

Link phenotypic clusters to independent variables:
1. Cluster × Experiment association (chi-square)
2. Cluster × Track duration (bias check)
3. Cluster × Experimental condition (0-250 vs 50-250, Constant vs Cycling)
4. Rare phenotype deep-dive

Runtime: ~1 minute
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
CHAR_DIR = RESULTS_DIR / 'characterization'
FITS_PATH = RESULTS_DIR / 'empirical_10min_kernel_fits_v2.csv'
ASSIGNMENTS_PATH = CHAR_DIR / 'track_cluster_assignments.csv'
OUTPUT_DIR = RESULTS_DIR / 'external_validation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load kernel fits with cluster assignments."""
    df = pd.read_csv(ASSIGNMENTS_PATH)
    print(f"Loaded {len(df)} tracks with cluster assignments")
    return df

def parse_experiment_condition(exp_id):
    """Extract experimental condition from experiment ID."""
    # Example: GMR61@GMR61_T_Re_Sq_0to250PWM_30#C_Bl_7PWM_202510291652
    # Conditions: 0to250 vs 50to250, and C_Bl (Constant Blue) vs T_Bl_Sq (pulsing)
    
    if '0to250' in exp_id:
        intensity = '0-250'
    elif '50to250' in exp_id:
        intensity = '50-250'
    else:
        intensity = 'Unknown'
    
    if '#C_Bl' in exp_id or 'C_Bl' in exp_id:
        pattern = 'Constant'
    elif '#T_Bl' in exp_id or 'T_Re_Sq' in exp_id:
        pattern = 'Cycling'
    else:
        pattern = 'Unknown'
    
    return intensity, pattern

def cluster_experiment_association(df, k=4):
    """Test if clusters are associated with specific experiments."""
    cluster_col = f'cluster_k{k}'
    
    print(f"\n{'='*70}")
    print("1. CLUSTER × EXPERIMENT ASSOCIATION")
    print(f"{'='*70}")
    
    # Create contingency table
    contingency = pd.crosstab(df['experiment_id'], df[cluster_col])
    
    # Chi-square test
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    
    # Cramér's V (effect size)
    n = len(df)
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    
    effect = "Large" if cramers_v > 0.5 else "Medium" if cramers_v > 0.3 else "Small" if cramers_v > 0.1 else "Negligible"
    
    print(f"\nChi-square test: χ² = {chi2:.2f}, df = {dof}, p = {p:.4f}")
    print(f"Cramér's V = {cramers_v:.3f} [{effect}]")
    
    if p < 0.05:
        print("\n⚠ WARNING: Clusters ARE associated with specific experiments!")
        print("This could indicate batch effects rather than true phenotypes.")
    else:
        print("\n✓ Clusters are NOT significantly associated with experiments.")
        print("Phenotypes are distributed across experiments (good!).")
    
    # Show distribution
    print(f"\nCluster distribution by experiment:")
    print(contingency.to_string())
    
    return {
        'chi2': round(chi2, 3),
        'p_value': round(p, 6),
        'dof': dof,
        'cramers_v': round(cramers_v, 4),
        'effect_size': effect,
        'significant': p < 0.05
    }

def cluster_condition_association(df, k=4):
    """Test if clusters are associated with experimental conditions."""
    cluster_col = f'cluster_k{k}'
    
    print(f"\n{'='*70}")
    print("2. CLUSTER × EXPERIMENTAL CONDITION")
    print(f"{'='*70}")
    
    # Parse conditions
    df['intensity'], df['pattern'] = zip(*df['experiment_id'].apply(parse_experiment_condition))
    
    print(f"\nConditions found:")
    print(f"  Intensity: {df['intensity'].value_counts().to_dict()}")
    print(f"  Pattern: {df['pattern'].value_counts().to_dict()}")
    
    results = {}
    
    # Test intensity association
    print(f"\n--- Intensity (0-250 vs 50-250) ---")
    contingency_int = pd.crosstab(df['intensity'], df[cluster_col])
    chi2_int, p_int, dof_int, _ = stats.chi2_contingency(contingency_int)
    
    n = len(df)
    min_dim = min(contingency_int.shape) - 1
    cramers_v_int = np.sqrt(chi2_int / (n * min_dim)) if min_dim > 0 else 0
    
    print(f"Chi-square: χ² = {chi2_int:.2f}, p = {p_int:.4f}")
    print(f"Cramér's V = {cramers_v_int:.3f}")
    print(contingency_int.to_string())
    
    results['intensity'] = {
        'chi2': round(chi2_int, 3),
        'p_value': round(p_int, 6),
        'cramers_v': round(cramers_v_int, 4),
        'significant': p_int < 0.05
    }
    
    # Test pattern association
    print(f"\n--- Pattern (Constant vs Cycling) ---")
    contingency_pat = pd.crosstab(df['pattern'], df[cluster_col])
    chi2_pat, p_pat, dof_pat, _ = stats.chi2_contingency(contingency_pat)
    
    min_dim = min(contingency_pat.shape) - 1
    cramers_v_pat = np.sqrt(chi2_pat / (n * min_dim)) if min_dim > 0 else 0
    
    print(f"Chi-square: χ² = {chi2_pat:.2f}, p = {p_pat:.4f}")
    print(f"Cramér's V = {cramers_v_pat:.3f}")
    print(contingency_pat.to_string())
    
    results['pattern'] = {
        'chi2': round(chi2_pat, 3),
        'p_value': round(p_pat, 6),
        'cramers_v': round(cramers_v_pat, 4),
        'significant': p_pat < 0.05
    }
    
    return results

def cluster_duration_bias(df, k=4):
    """Check if clusters are biased by track duration."""
    cluster_col = f'cluster_k{k}'
    
    print(f"\n{'='*70}")
    print("3. CLUSTER × TRACK DURATION (BIAS CHECK)")
    print(f"{'='*70}")
    
    # Load full fits to get duration
    fits_df = pd.read_csv(FITS_PATH)
    df = df.merge(fits_df[['experiment_id', 'track_id', 'duration', 'n_events']], 
                  on=['experiment_id', 'track_id'], how='left')
    
    # Kruskal-Wallis test for duration across clusters
    groups = [df[df[cluster_col] == c]['duration'].dropna().values for c in sorted(df[cluster_col].unique())]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) >= 2:
        kw_stat, kw_p = stats.kruskal(*groups)
    else:
        kw_stat, kw_p = 0, 1.0
    
    print(f"\nDuration by cluster:")
    for c in sorted(df[cluster_col].unique()):
        cluster_dur = df[df[cluster_col] == c]['duration'].dropna()
        if len(cluster_dur) > 0:
            print(f"  Cluster {c}: {cluster_dur.mean()/60:.1f} ± {cluster_dur.std()/60:.1f} min (n={len(cluster_dur)})")
    
    print(f"\nKruskal-Wallis: H = {kw_stat:.2f}, p = {kw_p:.4f}")
    
    if kw_p < 0.05:
        print("⚠ WARNING: Duration differs significantly across clusters!")
        print("Some phenotypes may be artifacts of track length.")
    else:
        print("✓ Duration does NOT differ across clusters (good!).")
    
    # Same for n_events
    groups_events = [df[df[cluster_col] == c]['n_events'].dropna().values for c in sorted(df[cluster_col].unique())]
    groups_events = [g for g in groups_events if len(g) > 0]
    
    if len(groups_events) >= 2:
        kw_stat_ev, kw_p_ev = stats.kruskal(*groups_events)
    else:
        kw_stat_ev, kw_p_ev = 0, 1.0
    
    print(f"\nEvents by cluster:")
    for c in sorted(df[cluster_col].unique()):
        cluster_ev = df[df[cluster_col] == c]['n_events'].dropna()
        if len(cluster_ev) > 0:
            print(f"  Cluster {c}: {cluster_ev.mean():.1f} ± {cluster_ev.std():.1f} events")
    
    print(f"\nKruskal-Wallis: H = {kw_stat_ev:.2f}, p = {kw_p_ev:.4f}")
    
    return {
        'duration': {
            'kruskal_wallis_H': round(kw_stat, 3),
            'p_value': round(kw_p, 6),
            'significant': kw_p < 0.05
        },
        'n_events': {
            'kruskal_wallis_H': round(kw_stat_ev, 3),
            'p_value': round(kw_p_ev, 6),
            'significant': kw_p_ev < 0.05
        }
    }

def rare_phenotype_deepdive(df, k=4):
    """Deep-dive into the rare phenotype cluster."""
    cluster_col = f'cluster_k{k}'
    
    print(f"\n{'='*70}")
    print("4. RARE PHENOTYPE DEEP-DIVE")
    print(f"{'='*70}")
    
    # Find smallest cluster
    cluster_sizes = df[cluster_col].value_counts()
    rare_cluster = cluster_sizes.idxmin()
    rare_n = cluster_sizes.min()
    
    rare_df = df[df[cluster_col] == rare_cluster]
    
    print(f"\nRare phenotype: Cluster {rare_cluster} (n={rare_n})")
    
    # List all rare phenotype tracks
    print(f"\nTracks in rare phenotype:")
    for _, row in rare_df.iterrows():
        exp_short = row['experiment_id'].split('_')[-1] if '_' in row['experiment_id'] else row['experiment_id']
        print(f"  Experiment: ...{exp_short[-12:]}, Track {row['track_id']}")
    
    # Check experiment distribution
    print(f"\nExperiment distribution:")
    exp_counts = rare_df['experiment_id'].value_counts()
    for exp, count in exp_counts.items():
        total_in_exp = len(df[df['experiment_id'] == exp])
        print(f"  {exp[-20:]}: {count}/{total_in_exp} ({100*count/total_in_exp:.1f}%)")
    
    # Parse conditions for rare phenotype
    rare_df = rare_df.copy()
    rare_df['intensity'], rare_df['pattern'] = zip(*rare_df['experiment_id'].apply(parse_experiment_condition))
    
    print(f"\nCondition distribution in rare phenotype:")
    print(f"  Intensity: {rare_df['intensity'].value_counts().to_dict()}")
    print(f"  Pattern: {rare_df['pattern'].value_counts().to_dict()}")
    
    # Compare to overall distribution
    df_copy = df.copy()
    df_copy['intensity'], df_copy['pattern'] = zip(*df_copy['experiment_id'].apply(parse_experiment_condition))
    
    print(f"\nOverall distribution (for comparison):")
    print(f"  Intensity: {df_copy['intensity'].value_counts(normalize=True).to_dict()}")
    print(f"  Pattern: {df_copy['pattern'].value_counts(normalize=True).to_dict()}")
    
    return {
        'cluster_id': int(rare_cluster),
        'n': int(rare_n),
        'experiments': exp_counts.to_dict(),
        'conditions': {
            'intensity': rare_df['intensity'].value_counts().to_dict(),
            'pattern': rare_df['pattern'].value_counts().to_dict()
        }
    }

def main():
    print("=" * 70)
    print("PHASE 3: EXTERNAL VALIDATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = load_data()
    
    k = 4  # Use validated k
    
    results = {'k': k, 'n_tracks': len(df)}
    
    # 1. Cluster × Experiment
    results['experiment_association'] = cluster_experiment_association(df, k)
    
    # 2. Cluster × Condition
    results['condition_association'] = cluster_condition_association(df, k)
    
    # 3. Duration bias
    results['duration_bias'] = cluster_duration_bias(df, k)
    
    # 4. Rare phenotype
    results['rare_phenotype'] = rare_phenotype_deepdive(df, k)
    
    # Summary
    print(f"\n{'='*70}")
    print("EXTERNAL VALIDATION SUMMARY")
    print(f"{'='*70}")
    
    issues = []
    
    if results['experiment_association']['significant']:
        issues.append("Clusters associated with specific experiments (batch effect?)")
    
    if results['condition_association']['intensity']['significant']:
        issues.append("Clusters associated with LED intensity")
    
    if results['condition_association']['pattern']['significant']:
        issues.append("Clusters associated with stimulation pattern")
    
    if results['duration_bias']['duration']['significant']:
        issues.append("Clusters biased by track duration")
    
    if results['duration_bias']['n_events']['significant']:
        issues.append("Clusters biased by event count")
    
    if len(issues) == 0:
        print("\n✓ ALL VALIDATION CHECKS PASSED")
        print("  - Phenotypes are distributed across experiments")
        print("  - Phenotypes are not biased by track characteristics")
        print("  - Clusters represent genuine individual differences")
    else:
        print(f"\n⚠ {len(issues)} POTENTIAL ISSUES IDENTIFIED:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Convert numpy types
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
    output_path = OUTPUT_DIR / 'external_validation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results

if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Hierarchical Bayesian Model for Individual Kernel Estimation

This is the gold standard approach for individual differences with sparse data:
1. Fit population + individual deviations jointly
2. Shrinkage toward population mean regularizes sparse tracks
3. Posterior uncertainty quantifies confidence in individual estimates

Model Structure:
  Population: θ ~ Prior(hyperparams)
  Individual: θᵢ ~ Normal(θ_pop, σ_pop)  
  Likelihood: events ~ Bernoulli(exp(β₀ + K(t; θᵢ)))

Runtime: ~15-30 min for MCMC sampling
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Check for JAX/NumPyro
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, Predictive
    print(f"JAX version: {jax.__version__}")
    print(f"NumPyro version: {numpyro.__version__}")
    
    # Set CPU (more stable for this model size)
    numpyro.set_platform('cpu')
    numpyro.set_host_device_count(4)  # Use multiple cores
except ImportError as e:
    print(f"JAX/NumPyro not available: {e}")
    print("Install with: pip install numpyro jax jaxlib")
    sys.exit(1)

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
H5_PATH = Path('/Users/gilraitses/INDYsim_project/data/processed/consolidated_dataset.h5')
OUTPUT_DIR = RESULTS_DIR / 'hierarchical_bayesian'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model parameters
LED_CYCLE = 30.0
LED_ON_DURATION = 10.0
FIRST_LED_ONSET = 21.3
TRACK_DURATION = 1200.0
DT = 0.05

# MCMC parameters
N_WARMUP = 500
N_SAMPLES = 1000
N_CHAINS = 2

# Time grid for kernel evaluation
KERNEL_T = jnp.linspace(0.1, 10.0, 50)


def gamma_kernel_jax(t, tau1, tau2, A, B, alpha=2.0):
    """
    Gamma-difference kernel in JAX.
    K(t) = A * Gamma(t; alpha, tau1) - B * Gamma(t; alpha, tau2)
    """
    # Gamma PDF: t^(alpha-1) * exp(-t/beta) / (beta^alpha * Gamma(alpha))
    def gamma_pdf(t, alpha, beta):
        log_pdf = (alpha - 1) * jnp.log(t + 1e-10) - t / beta - alpha * jnp.log(beta) - jax.scipy.special.gammaln(alpha)
        return jnp.exp(log_pdf)
    
    pdf1 = gamma_pdf(t, alpha, tau1)
    pdf2 = gamma_pdf(t, alpha, tau2)
    
    return A * pdf1 - B * pdf2


def hierarchical_kernel_model(event_times_list, track_indices, n_tracks):
    """
    Hierarchical Bayesian model for kernel parameters.
    
    Population-level priors:
      τ₁_pop ~ LogNormal(log(0.3), 0.5)  # Fast timescale ~0.3s
      τ₂_pop ~ LogNormal(log(4.0), 0.5)  # Slow timescale ~4s
      A_pop  ~ LogNormal(log(1.0), 0.5)  # Excitation amplitude
      B_pop  ~ LogNormal(log(15.0), 0.5) # Suppression amplitude
    
    Individual-level (partial pooling):
      τ₁ᵢ ~ LogNormal(log(τ₁_pop), σ_τ1)
      etc.
    """
    # ==========================================================================
    # Population-level hyperpriors
    # ==========================================================================
    
    # Population means (log-scale)
    mu_tau1 = numpyro.sample('mu_tau1', dist.Normal(-1.2, 0.5))  # exp(-1.2) ≈ 0.3
    mu_tau2 = numpyro.sample('mu_tau2', dist.Normal(1.4, 0.5))   # exp(1.4) ≈ 4.0
    mu_A = numpyro.sample('mu_A', dist.Normal(0.0, 0.5))         # exp(0) = 1.0
    mu_B = numpyro.sample('mu_B', dist.Normal(2.7, 0.5))         # exp(2.7) ≈ 15
    
    # Population standard deviations (how much individuals vary)
    sigma_tau1 = numpyro.sample('sigma_tau1', dist.HalfNormal(0.5))
    sigma_tau2 = numpyro.sample('sigma_tau2', dist.HalfNormal(0.5))
    sigma_A = numpyro.sample('sigma_A', dist.HalfNormal(0.5))
    sigma_B = numpyro.sample('sigma_B', dist.HalfNormal(0.5))
    
    # Baseline hazard
    beta0 = numpyro.sample('beta0', dist.Normal(-3.5, 0.5))
    
    # ==========================================================================
    # Individual-level parameters (partial pooling)
    # ==========================================================================
    
    with numpyro.plate('tracks', n_tracks):
        # Individual deviations (non-centered parameterization for better sampling)
        z_tau1 = numpyro.sample('z_tau1', dist.Normal(0, 1))
        z_tau2 = numpyro.sample('z_tau2', dist.Normal(0, 1))
        z_A = numpyro.sample('z_A', dist.Normal(0, 1))
        z_B = numpyro.sample('z_B', dist.Normal(0, 1))
        
        # Transform to actual parameters
        log_tau1 = numpyro.deterministic('log_tau1', mu_tau1 + sigma_tau1 * z_tau1)
        log_tau2 = numpyro.deterministic('log_tau2', mu_tau2 + sigma_tau2 * z_tau2)
        log_A = numpyro.deterministic('log_A', mu_A + sigma_A * z_A)
        log_B = numpyro.deterministic('log_B', mu_B + sigma_B * z_B)
        
        tau1 = numpyro.deterministic('tau1', jnp.exp(log_tau1))
        tau2 = numpyro.deterministic('tau2', jnp.exp(log_tau2))
        A = numpyro.deterministic('A', jnp.exp(log_A))
        B = numpyro.deterministic('B', jnp.exp(log_B))
    
    # ==========================================================================
    # Likelihood (simplified: use PSTH bins instead of raw events for speed)
    # ==========================================================================
    
    # For each event, compute log-likelihood
    # This is a simplified version - full version would iterate over all frames
    
    # We'll use the event times directly
    for i, (events, track_idx) in enumerate(zip(event_times_list, track_indices)):
        if len(events) == 0:
            continue
            
        # Get this track's parameters
        t1 = tau1[track_idx]
        t2 = tau2[track_idx]
        a = A[track_idx]
        b = B[track_idx]
        
        # Compute kernel at event times
        K_events = gamma_kernel_jax(events, t1, t2, a, b)
        
        # Log-hazard at event times
        log_lambda = beta0 + K_events
        
        # Simplified likelihood: sum of log-hazards at event times
        # (This is an approximation to the full point process likelihood)
        numpyro.factor(f'events_{i}', jnp.sum(log_lambda))


def simplified_hierarchical_model(psth_data, n_tracks):
    """
    Simplified hierarchical model using PSTH as sufficient statistic.
    Much faster than full event-level model.
    
    Model: PSTH_i(t) ~ Normal(exp(β₀ + K(t; θᵢ)), σ_obs)
    """
    n_bins = psth_data.shape[1]
    t_bins = jnp.linspace(0.25, 9.75, n_bins)  # Bin centers
    
    # ==========================================================================
    # Population-level hyperpriors
    # ==========================================================================
    
    mu_tau1 = numpyro.sample('mu_tau1', dist.Normal(-1.2, 0.5))
    mu_tau2 = numpyro.sample('mu_tau2', dist.Normal(1.4, 0.5))
    mu_A = numpyro.sample('mu_A', dist.Normal(0.0, 1.0))
    mu_B = numpyro.sample('mu_B', dist.Normal(2.7, 1.0))
    
    sigma_tau1 = numpyro.sample('sigma_tau1', dist.HalfNormal(0.3))
    sigma_tau2 = numpyro.sample('sigma_tau2', dist.HalfNormal(0.3))
    sigma_A = numpyro.sample('sigma_A', dist.HalfNormal(0.5))
    sigma_B = numpyro.sample('sigma_B', dist.HalfNormal(0.5))
    
    beta0 = numpyro.sample('beta0', dist.Normal(-3.5, 1.0))
    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(0.5))
    
    # ==========================================================================
    # Individual-level parameters
    # ==========================================================================
    
    with numpyro.plate('tracks', n_tracks):
        z_tau1 = numpyro.sample('z_tau1', dist.Normal(0, 1))
        z_tau2 = numpyro.sample('z_tau2', dist.Normal(0, 1))
        z_A = numpyro.sample('z_A', dist.Normal(0, 1))
        z_B = numpyro.sample('z_B', dist.Normal(0, 1))
        
        tau1 = numpyro.deterministic('tau1', jnp.exp(mu_tau1 + sigma_tau1 * z_tau1))
        tau2 = numpyro.deterministic('tau2', jnp.exp(mu_tau2 + sigma_tau2 * z_tau2))
        A = numpyro.deterministic('A', jnp.exp(mu_A + sigma_A * z_A))
        B = numpyro.deterministic('B', jnp.exp(mu_B + sigma_B * z_B))
    
    # ==========================================================================
    # Likelihood
    # ==========================================================================
    
    # Compute expected PSTH for each track
    # Shape: (n_tracks, n_bins)
    expected_rate = jnp.zeros((n_tracks, n_bins))
    
    for j in range(n_bins):
        t = t_bins[j]
        K_t = gamma_kernel_jax(t, tau1, tau2, A, B)
        expected_rate = expected_rate.at[:, j].set(jnp.exp(beta0 + K_t))
    
    # Observation model
    with numpyro.plate('bins', n_bins):
        with numpyro.plate('tracks_obs', n_tracks):
            numpyro.sample('psth_obs', 
                          dist.Normal(expected_rate, sigma_obs),
                          obs=psth_data)


def load_psth_data():
    """Load PSTH data from deep EDA results."""
    psth_path = RESULTS_DIR / 'deep_eda' / 'psth_matrix.npy'
    if psth_path.exists():
        return np.load(psth_path)
    else:
        raise FileNotFoundError(f"PSTH matrix not found at {psth_path}")


def main():
    print("=" * 70)
    print("HIERARCHICAL BAYESIAN KERNEL ESTIMATION")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # =========================================================================
    # 1. LOAD DATA
    # =========================================================================
    print(f"\n{'='*70}")
    print("1. LOADING DATA")
    print(f"{'='*70}")
    
    psth_data = load_psth_data()
    n_tracks = len(psth_data)
    print(f"Loaded PSTH data: {psth_data.shape}")
    
    # Normalize PSTH for numerical stability
    psth_mean = psth_data.mean()
    psth_std = psth_data.std()
    psth_normalized = (psth_data - psth_mean) / psth_std
    
    # Convert to JAX array
    psth_jax = jnp.array(psth_normalized)
    
    # =========================================================================
    # 2. RUN MCMC
    # =========================================================================
    print(f"\n{'='*70}")
    print("2. RUNNING MCMC SAMPLING")
    print(f"{'='*70}")
    
    print(f"\nMCMC Configuration:")
    print(f"  Warmup: {N_WARMUP}")
    print(f"  Samples: {N_SAMPLES}")
    print(f"  Chains: {N_CHAINS}")
    print(f"  Total iterations: {(N_WARMUP + N_SAMPLES) * N_CHAINS}")
    
    # Initialize NUTS sampler
    kernel = NUTS(simplified_hierarchical_model)
    mcmc = MCMC(kernel, num_warmup=N_WARMUP, num_samples=N_SAMPLES, num_chains=N_CHAINS)
    
    # Run MCMC
    rng_key = random.PRNGKey(42)
    print(f"\nRunning MCMC (this may take 15-30 minutes)...")
    mcmc.run(rng_key, psth_data=psth_jax, n_tracks=n_tracks)
    
    # =========================================================================
    # 3. EXTRACT POSTERIORS
    # =========================================================================
    print(f"\n{'='*70}")
    print("3. EXTRACTING POSTERIORS")
    print(f"{'='*70}")
    
    samples = mcmc.get_samples()
    
    # Population-level summaries
    print(f"\nPopulation-level parameter estimates:")
    for param in ['mu_tau1', 'mu_tau2', 'mu_A', 'mu_B', 'sigma_tau1', 'sigma_tau2', 'sigma_A', 'sigma_B']:
        vals = samples[param]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        if 'mu' in param:
            # Transform from log-scale
            print(f"  {param}: {mean_val:.3f} ± {std_val:.3f} (exp: {np.exp(mean_val):.3f})")
        else:
            print(f"  {param}: {mean_val:.3f} ± {std_val:.3f}")
    
    # Individual-level posteriors
    tau1_samples = samples['tau1']  # Shape: (n_samples, n_tracks)
    tau2_samples = samples['tau2']
    A_samples = samples['A']
    B_samples = samples['B']
    
    # Posterior means and credible intervals
    individual_params = pd.DataFrame({
        'tau1_mean': np.mean(tau1_samples, axis=0),
        'tau1_std': np.std(tau1_samples, axis=0),
        'tau1_ci_low': np.percentile(tau1_samples, 2.5, axis=0),
        'tau1_ci_high': np.percentile(tau1_samples, 97.5, axis=0),
        'tau2_mean': np.mean(tau2_samples, axis=0),
        'tau2_std': np.std(tau2_samples, axis=0),
        'tau2_ci_low': np.percentile(tau2_samples, 2.5, axis=0),
        'tau2_ci_high': np.percentile(tau2_samples, 97.5, axis=0),
        'A_mean': np.mean(A_samples, axis=0),
        'A_std': np.std(A_samples, axis=0),
        'B_mean': np.mean(B_samples, axis=0),
        'B_std': np.std(B_samples, axis=0),
    })
    
    print(f"\nIndividual-level posterior summary:")
    print(f"  τ₁: {individual_params['tau1_mean'].mean():.3f} ± {individual_params['tau1_mean'].std():.3f}")
    print(f"  τ₂: {individual_params['tau2_mean'].mean():.3f} ± {individual_params['tau2_mean'].std():.3f}")
    print(f"  A:  {individual_params['A_mean'].mean():.3f} ± {individual_params['A_mean'].std():.3f}")
    print(f"  B:  {individual_params['B_mean'].mean():.3f} ± {individual_params['B_mean'].std():.3f}")
    
    # =========================================================================
    # 4. SHRINKAGE ANALYSIS
    # =========================================================================
    print(f"\n{'='*70}")
    print("4. SHRINKAGE ANALYSIS")
    print(f"{'='*70}")
    
    # Compare posterior uncertainty to prior uncertainty
    # Tracks with more data should have narrower posteriors
    
    # Load original kernel fits for comparison
    fits_path = RESULTS_DIR / 'empirical_10min_kernel_fits_v2.csv'
    if fits_path.exists():
        original_fits = pd.read_csv(fits_path)
        n_compare = min(len(original_fits), len(individual_params))
        
        # Correlation between MLE and Bayesian estimates
        corr_tau1 = np.corrcoef(original_fits['tau1'][:n_compare], 
                                individual_params['tau1_mean'][:n_compare])[0, 1]
        corr_tau2 = np.corrcoef(original_fits['tau2'][:n_compare], 
                                individual_params['tau2_mean'][:n_compare])[0, 1]
        
        print(f"\nCorrelation with MLE estimates:")
        print(f"  τ₁: r = {corr_tau1:.3f}")
        print(f"  τ₂: r = {corr_tau2:.3f}")
        
        # Shrinkage toward population mean
        pop_tau1 = np.exp(float(np.mean(samples['mu_tau1'])))
        pop_tau2 = np.exp(float(np.mean(samples['mu_tau2'])))
        
        mle_deviation = np.std(original_fits['tau1'][:n_compare])
        bayes_deviation = np.std(individual_params['tau1_mean'][:n_compare])
        shrinkage_tau1 = 1 - (bayes_deviation / mle_deviation)
        
        print(f"\nShrinkage analysis:")
        print(f"  Population τ₁: {pop_tau1:.3f}")
        print(f"  MLE std(τ₁): {mle_deviation:.3f}")
        print(f"  Bayes std(τ₁): {bayes_deviation:.3f}")
        print(f"  Shrinkage: {shrinkage_tau1*100:.1f}%")
    
    # =========================================================================
    # 5. CLUSTERING ON POSTERIOR MEANS
    # =========================================================================
    print(f"\n{'='*70}")
    print("5. CLUSTERING ON POSTERIOR MEANS")
    print(f"{'='*70}")
    
    # Cluster using posterior means
    features = individual_params[['tau1_mean', 'tau2_mean', 'A_mean', 'B_mean']].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Load PSTH clusters for comparison
    psth_pcs = np.load(RESULTS_DIR / 'deep_eda' / 'psth_pcs.npy')
    
    results = {'clustering': {}}
    
    print(f"\n{'k':<5} {'Silhouette':<12} {'ARI vs PSTH':<15} {'ARI vs MLE':<15}")
    print("-" * 50)
    
    for k in [3, 4, 5]:
        bayes_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(features_scaled)
        sil = silhouette_score(features_scaled, bayes_labels)
        
        # Compare to PSTH
        psth_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(psth_pcs[:len(features), :3])
        ari_psth = adjusted_rand_score(bayes_labels, psth_labels)
        
        # Compare to MLE
        if fits_path.exists():
            mle_features = original_fits[['tau1', 'tau2', 'A', 'B']][:len(features)].values
            mle_scaled = scaler.fit_transform(mle_features)
            mle_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(mle_scaled)
            ari_mle = adjusted_rand_score(bayes_labels, mle_labels)
        else:
            ari_mle = float('nan')
        
        print(f"{k:<5} {sil:<12.3f} {ari_psth:<15.3f} {ari_mle:<15.3f}")
        
        results['clustering'][k] = {
            'silhouette': round(float(sil), 4),
            'ari_vs_psth': round(float(ari_psth), 4),
            'ari_vs_mle': round(float(ari_mle), 4) if not np.isnan(ari_mle) else None
        }
    
    # =========================================================================
    # 6. IDENTIFY OUTLIERS
    # =========================================================================
    print(f"\n{'='*70}")
    print("6. IDENTIFYING GENUINE OUTLIERS")
    print(f"{'='*70}")
    
    # Tracks where 95% CI doesn't overlap with population mean
    pop_tau1 = np.exp(float(np.mean(samples['mu_tau1'])))
    pop_tau2 = np.exp(float(np.mean(samples['mu_tau2'])))
    
    outliers_tau1 = ((individual_params['tau1_ci_high'] < pop_tau1) | 
                     (individual_params['tau1_ci_low'] > pop_tau1))
    outliers_tau2 = ((individual_params['tau2_ci_high'] < pop_tau2) | 
                     (individual_params['tau2_ci_low'] > pop_tau2))
    
    print(f"\nTracks with 95% CI not overlapping population mean:")
    print(f"  τ₁ outliers: {outliers_tau1.sum()} / {n_tracks} ({100*outliers_tau1.mean():.1f}%)")
    print(f"  τ₂ outliers: {outliers_tau2.sum()} / {n_tracks} ({100*outliers_tau2.mean():.1f}%)")
    
    results['outliers'] = {
        'tau1_n_outliers': int(outliers_tau1.sum()),
        'tau2_n_outliers': int(outliers_tau2.sum()),
        'tau1_pct_outliers': round(float(100*outliers_tau1.mean()), 1),
        'tau2_pct_outliers': round(float(100*outliers_tau2.mean()), 1)
    }
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("HIERARCHICAL BAYESIAN SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n✓ Fitted hierarchical model to {n_tracks} tracks")
    print(f"✓ Population τ₁ = {pop_tau1:.3f}s (fast response)")
    print(f"✓ Population τ₂ = {pop_tau2:.3f}s (slow suppression)")
    
    if outliers_tau1.sum() < n_tracks * 0.1:
        print(f"\n⚠ Few genuine outliers ({outliers_tau1.sum()}/{n_tracks})")
        print(f"  → Individual variation is within population distribution")
        print(f"  → Discrete phenotypes NOT supported")
    else:
        print(f"\n✓ Significant outliers detected ({outliers_tau1.sum()}/{n_tracks})")
        print(f"  → Some individuals genuinely differ from population")
    
    # Save results
    results['population'] = {
        'tau1_mean': float(pop_tau1),
        'tau2_mean': float(pop_tau2),
        'sigma_tau1': float(np.mean(samples['sigma_tau1'])),
        'sigma_tau2': float(np.mean(samples['sigma_tau2']))
    }
    results['mcmc'] = {
        'n_warmup': N_WARMUP,
        'n_samples': N_SAMPLES,
        'n_chains': N_CHAINS
    }
    
    # Save
    with open(OUTPUT_DIR / 'hierarchical_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    individual_params.to_csv(OUTPUT_DIR / 'individual_posteriors.csv', index=False)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == '__main__':
    main()


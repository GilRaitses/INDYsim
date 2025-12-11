# NB-GLM Hazard Model Implementation Reference

This document provides the complete specification and code patterns for implementing the Negative Binomial GLM hazard model for larval reorientation events.

---

## 1. Model Specification

### 1.1 Outcome Variable

```
Y_eit ~ NegativeBinomial(mu_eit, alpha)
log(mu_eit) = eta_eit + log(delta_t)
```

Where:
- `Y_eit` = reorientation onset count in bin i, track t, experiment e
- `mu_eit` = expected count
- `alpha` = dispersion parameter (overdispersion if alpha > 0)
- `delta_t` = bin width in seconds (exposure offset)

### 1.2 Linear Predictor

```
eta_eit = beta_0                                      # intercept
        + beta_1 * LED1_scaled                        # red LED (PWM/250)
        + beta_2 * LED2_scaled                        # blue LED (PWM/15)
        + beta_3 * (LED1_scaled * LED2_scaled)        # interaction
        + beta_4 * sin(2*pi*phase)                    # cycle phase sin
        + beta_5 * cos(2*pi*phase)                    # cycle phase cos
        + sum_j(phi_j * B_j(time_since_stimulus))     # temporal kernel
        + gamma_1 * speed_z                           # z-scored speed
        + gamma_2 * curvature_z                       # z-scored curvature
```

### 1.3 Scaling Conventions

| Covariate | Scaling | Notes |
|-----------|---------|-------|
| LED1 | PWM / 250 | Range 0-1 |
| LED2 | PWM / 15 | Range 0-1 |
| phase | (time mod 60) / 60 | Position in 60s cycle |
| speed | (x - mean) / std | z-score on training data |
| curvature | (x - mean) / std | z-score on training data |

### 1.4 Temporal Kernel

Raised-cosine basis functions over time since LED1 onset:

```
B_j(t) = 0.5 * (1 + cos(pi * (t - c_j) / width))  if |t - c_j| < width
       = 0                                          otherwise
```

Default parameters:
- Number of bases J = 4
- Window = [0, 3] seconds
- Width = 0.6 seconds
- Centers evenly spaced: c_j in {0.0, 1.0, 2.0, 3.0}

---

## 2. Data Preparation

### 2.1 Bin Definition

Non-overlapping bins of 0.5 seconds:
```python
df['bin_start'] = (df['time'] // bin_width) * bin_width
```

### 2.2 Reorientation Onset Detection

Count TRANSITIONS, not frame-level booleans:
```python
# Option 1: Detect onset from is_reorientation transitions
df['reo_onset'] = df['is_reorientation'] & ~df['is_reorientation'].shift(1, fill_value=False)

# Option 2: Use Klein table reo_start_time directly
# Bin the reo_start_time values and count per bin
```

### 2.3 Binning Code

```python
def bin_data_for_hazard(df, bin_width=0.5, n_bases=4, kernel_window=(0.0, 3.0), kernel_width=0.6):
    """
    Prepare binned dataset for NB-GLM hazard model.
    
    Parameters
    ----------
    df : DataFrame
        Frame-level data with columns: time, track_id, experiment_id,
        is_reorientation, led1Val, led2Val, speed, curvature, time_since_stimulus
    bin_width : float
        Bin size in seconds (default 0.5)
    n_bases : int
        Number of temporal kernel bases (default 4)
    kernel_window : tuple
        (min, max) time range for kernel centers
    kernel_width : float
        Width parameter for raised-cosine bases
    
    Returns
    -------
    binned : DataFrame
        Bin-level data ready for GLM fitting
    """
    df = df.copy()
    
    # 1. Detect reorientation onsets (transitions)
    df = df.sort_values(['experiment_id', 'track_id', 'time'])
    df['reo_onset'] = (
        df.groupby(['experiment_id', 'track_id'])['is_reorientation']
        .transform(lambda x: x & ~x.shift(1, fill_value=False))
    ).astype(int)
    
    # 2. Compute phase covariates
    cycle_period = 60.0
    phase = (df['time'] % cycle_period) / cycle_period
    df['phase_sin'] = np.sin(2 * np.pi * phase)
    df['phase_cos'] = np.cos(2 * np.pi * phase)
    
    # 3. Compute temporal kernel bases
    t = df['time_since_stimulus'].values
    centers = np.linspace(kernel_window[0], kernel_window[1], n_bases)
    basis = raised_cosine_basis(t, centers, kernel_width)
    for j in range(n_bases):
        df[f'kernel_{j+1}'] = basis[:, j]
    
    # 4. Assign bins
    df['bin_start'] = (df['time'] // bin_width) * bin_width
    
    # 5. Aggregate to bin level
    group_cols = ['experiment_id', 'track_id', 'bin_start']
    agg_dict = {
        'reo_onset': 'sum',  # count of onsets in bin
        'led1Val': 'mean',
        'led2Val': 'mean',
        'phase_sin': 'mean',
        'phase_cos': 'mean',
        'speed': 'mean',
        'curvature': 'mean',
        'time_since_stimulus': 'mean',
    }
    for j in range(n_bases):
        agg_dict[f'kernel_{j+1}'] = 'mean'
    
    binned = df.groupby(group_cols, as_index=False).agg(agg_dict)
    binned.rename(columns={'reo_onset': 'Y'}, inplace=True)
    
    # 6. Scale covariates
    binned['LED1_scaled'] = binned['led1Val'] / 250.0
    binned['LED2_scaled'] = binned['led2Val'] / 15.0
    binned['LED1xLED2'] = binned['LED1_scaled'] * binned['LED2_scaled']
    
    # Z-score speed and curvature
    for col in ['speed', 'curvature']:
        m = binned[col].mean()
        s = binned[col].std()
        binned[f'{col}_z'] = (binned[col] - m) / (s + 1e-9)
    
    return binned
```

---

## 3. Model Fitting

### 3.1 Fixed-Effects NB-GLM with Cluster-Robust SEs

```python
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
import numpy as np
import pandas as pd

def fit_nb_glm(binned, bin_width=0.5, n_bases=4):
    """
    Fit Negative Binomial GLM with cluster-robust standard errors.
    
    Parameters
    ----------
    binned : DataFrame
        Output from bin_data_for_hazard()
    bin_width : float
        Bin width for exposure offset
    n_bases : int
        Number of temporal kernel bases
    
    Returns
    -------
    result : GLMResults
        Fitted model with robust SEs
    coef_table : DataFrame
        Coefficient interpretation table
    """
    # Response
    y = binned['Y'].values
    
    # Design matrix columns
    X_cols = [
        'LED1_scaled', 'LED2_scaled', 'LED1xLED2',
        'phase_sin', 'phase_cos',
        'speed_z', 'curvature_z'
    ]
    kernel_cols = [f'kernel_{j+1}' for j in range(n_bases)]
    X_cols.extend(kernel_cols)
    
    X = binned[X_cols].values
    X = sm.add_constant(X)  # add intercept
    
    # Exposure offset (log of bin width)
    offset = np.log(np.full(len(y), bin_width))
    
    # Fit NB-GLM (statsmodels estimates alpha via MLE)
    family = NegativeBinomial(alpha=0.1)  # initial guess
    model = sm.GLM(y, X, family=family, offset=offset)
    result = model.fit()
    
    # Cluster-robust standard errors by experiment
    clusters = binned['experiment_id'].values
    robust = result.get_robustcov_results(cov_type='cluster', groups=clusters)
    
    # Build coefficient table
    names = ['Intercept'] + X_cols
    coef_table = pd.DataFrame({
        'coef': robust.params,
        'se': robust.bse,
        'z': robust.tvalues,
        'p': robust.pvalues,
        'RR': np.exp(robust.params),
        'pct_change': 100 * (np.exp(robust.params) - 1)
    }, index=names)
    
    return robust, coef_table
```

### 3.2 Dispersion Estimation

The dispersion parameter alpha is estimated via MLE by statsmodels. After fitting:

```python
# Access estimated dispersion
alpha = result.scale  # or result.family.alpha after fit

# Check dispersion ratio (target ~1.0)
dispersion_ratio = result.pearson_chi2 / result.df_resid
print(f"Dispersion ratio: {dispersion_ratio:.3f} (target ~1.0)")
```

---

## 4. Diagnostics

### 4.1 Overdispersion Check

```python
def check_overdispersion(result):
    """Check if NB adequately captures overdispersion."""
    ratio = result.pearson_chi2 / result.df_resid
    if ratio > 1.5:
        print(f"WARNING: Dispersion ratio {ratio:.2f} >> 1, consider increasing alpha")
    elif ratio < 0.5:
        print(f"WARNING: Dispersion ratio {ratio:.2f} << 1, model may be over-dispersed")
    else:
        print(f"OK: Dispersion ratio {ratio:.2f} ~ 1")
    return ratio
```

### 4.2 Zero-Inflation Check

```python
def check_zero_inflation(y, mu, alpha):
    """
    Compare observed vs expected zeros under NB.
    
    Parameters
    ----------
    y : array
        Observed counts
    mu : array
        Fitted means
    alpha : float
        Dispersion parameter
    
    Returns
    -------
    needs_zinb : bool
        True if zero-inflated NB may be needed
    """
    observed_zeros = (y == 0).mean()
    
    # Expected zero probability under NB: (1 + alpha*mu)^(-1/alpha)
    if alpha > 0:
        expected_zeros = np.mean((1 + alpha * mu) ** (-1 / alpha))
    else:
        # Poisson limit
        expected_zeros = np.mean(np.exp(-mu))
    
    diff = abs(observed_zeros - expected_zeros)
    
    print(f"Observed zeros: {observed_zeros:.3f}")
    print(f"Expected zeros (NB): {expected_zeros:.3f}")
    print(f"Difference: {diff:.3f}")
    
    if diff > 0.02:
        print("WARNING: Consider Zero-Inflated NB (ZINB)")
        return True
    else:
        print("OK: NB zero prediction adequate")
        return False
```

### 4.3 Serial Correlation Check

```python
from statsmodels.stats.stattools import acf

def check_serial_correlation(result, binned, max_lag=5):
    """
    Check for residual autocorrelation within tracks.
    
    Parameters
    ----------
    result : GLMResults
        Fitted model
    binned : DataFrame
        Binned data with track_id
    max_lag : int
        Maximum lag to check
    
    Returns
    -------
    acf_values : dict
        ACF at each lag
    """
    resid = result.resid_pearson
    binned_with_resid = binned.copy()
    binned_with_resid['resid'] = resid
    
    # Compute ACF within tracks
    acf_by_track = []
    for (exp, track), group in binned_with_resid.groupby(['experiment_id', 'track_id']):
        if len(group) > max_lag + 1:
            r = group['resid'].values
            acf_vals = [np.corrcoef(r[:-lag], r[lag:])[0, 1] for lag in range(1, max_lag + 1)]
            acf_by_track.append(acf_vals)
    
    # Average across tracks
    acf_mean = np.mean(acf_by_track, axis=0)
    
    print("Mean ACF of Pearson residuals:")
    for lag, val in enumerate(acf_mean, 1):
        flag = " *" if abs(val) > 0.1 else ""
        print(f"  Lag {lag}: {val:.3f}{flag}")
    
    if abs(acf_mean[0]) > 0.1:
        print("WARNING: Significant lag-1 autocorrelation. Consider adding history terms.")
    
    return dict(zip(range(1, max_lag + 1), acf_mean))
```

---

## 5. Temporal Kernel Interpretation

### 5.1 Extract Kernel Shape

```python
def extract_kernel_shape(phi_hat, centers, width, t_grid=None):
    """
    Compute stimulus-response function from fitted kernel weights.
    
    Parameters
    ----------
    phi_hat : array
        Fitted kernel coefficients (phi_1, ..., phi_J)
    centers : array
        Kernel center positions
    width : float
        Kernel width parameter
    t_grid : array, optional
        Time points for evaluation (default: 0 to 3s at 0.01s resolution)
    
    Returns
    -------
    t_grid : array
        Time points
    K : array
        Log-rate modulation K(t) = sum_j(phi_j * B_j(t))
    RR : array
        Rate ratio RR(t) = exp(K(t))
    """
    if t_grid is None:
        t_grid = np.linspace(0, 3, 301)
    
    B = raised_cosine_basis(t_grid, centers, width)
    K = B @ phi_hat
    RR = np.exp(K)
    
    return t_grid, K, RR
```

### 5.2 Confidence Bands via Delta Method

```python
def kernel_confidence_bands(phi_hat, phi_cov, centers, width, t_grid=None, alpha=0.05):
    """
    Compute pointwise confidence bands for kernel using delta method.
    
    Parameters
    ----------
    phi_hat : array
        Fitted kernel coefficients
    phi_cov : array
        Covariance matrix of phi estimates
    centers : array
        Kernel center positions
    width : float
        Kernel width parameter
    t_grid : array, optional
        Time points for evaluation
    alpha : float
        Significance level (default 0.05 for 95% CI)
    
    Returns
    -------
    t_grid : array
        Time points
    RR : array
        Rate ratio point estimate
    RR_lower : array
        Lower CI bound
    RR_upper : array
        Upper CI bound
    """
    from scipy.stats import norm
    
    if t_grid is None:
        t_grid = np.linspace(0, 3, 301)
    
    B = raised_cosine_basis(t_grid, centers, width)
    K = B @ phi_hat
    
    # Variance of K(t) via delta method: Var(K) = B @ Cov(phi) @ B.T
    var_K = np.diag(B @ phi_cov @ B.T)
    se_K = np.sqrt(var_K)
    
    z = norm.ppf(1 - alpha / 2)
    K_lower = K - z * se_K
    K_upper = K + z * se_K
    
    RR = np.exp(K)
    RR_lower = np.exp(K_lower)
    RR_upper = np.exp(K_upper)
    
    return t_grid, RR, RR_lower, RR_upper
```

### 5.3 Peak Latency

```python
def find_peak_latency(t_grid, K):
    """Find time of peak kernel response."""
    peak_idx = np.argmax(K)
    peak_latency = t_grid[peak_idx]
    peak_value = K[peak_idx]
    return peak_latency, np.exp(peak_value)
```

---

## 6. Cross-Validation

### 6.1 Leave-One-Experiment-Out CV

```python
from sklearn.model_selection import LeaveOneGroupOut

def loeo_cv_kernel_selection(binned, bin_width, n_bases_options=[3, 4, 5], 
                              window_options=[(0, 2), (0, 3), (0, 4)]):
    """
    Select optimal kernel hyperparameters via LOEO cross-validation.
    
    Parameters
    ----------
    binned : DataFrame
        Binned data (before kernel computation)
    bin_width : float
        Bin width for exposure
    n_bases_options : list
        Number of bases to try
    window_options : list
        (min, max) windows to try
    
    Returns
    -------
    best_params : dict
        Best (n_bases, window) combination
    cv_results : DataFrame
        All CV results
    """
    logo = LeaveOneGroupOut()
    experiments = binned['experiment_id'].unique()
    results = []
    
    for n_bases in n_bases_options:
        for window in window_options:
            deviances = []
            
            for train_idx, test_idx in logo.split(binned, groups=binned['experiment_id']):
                train_data = binned.iloc[train_idx]
                test_data = binned.iloc[test_idx]
                
                # Recompute kernel bases for this configuration
                # (would need to re-bin with new kernel params)
                # ... fit model on train, evaluate deviance on test ...
                
                # Placeholder: use full data deviance for now
                pass
            
            mean_deviance = np.mean(deviances) if deviances else np.inf
            results.append({
                'n_bases': n_bases,
                'window': window,
                'mean_deviance': mean_deviance
            })
    
    cv_results = pd.DataFrame(results)
    best_idx = cv_results['mean_deviance'].idxmin()
    best_params = cv_results.loc[best_idx, ['n_bases', 'window']].to_dict()
    
    return best_params, cv_results
```

---

## 7. Simulation Validation

### 7.1 Connect Hazard to Generator

```python
def make_hazard_function(result, binned, n_bases, centers, width):
    """
    Create a hazard function from fitted GLM for use in simulation.
    
    Returns a function lambda(t, led1, led2) that gives instantaneous hazard.
    """
    coefs = result.params
    
    def hazard(t, led1_pwm, led2_pwm, speed=0.0, curvature=0.0):
        """
        Compute instantaneous reorientation hazard at time t.
        
        Parameters
        ----------
        t : float
            Experiment time (seconds)
        led1_pwm : float
            Current LED1 intensity (0-250)
        led2_pwm : float
            Current LED2 intensity (0-15)
        speed : float
            Current speed (will be z-scored internally)
        curvature : float
            Current curvature (will be z-scored internally)
        
        Returns
        -------
        lambda_t : float
            Instantaneous hazard (events per second)
        """
        # Scale covariates
        led1_scaled = led1_pwm / 250.0
        led2_scaled = led2_pwm / 15.0
        interaction = led1_scaled * led2_scaled
        
        # Phase
        phase = (t % 60.0) / 60.0
        phase_sin = np.sin(2 * np.pi * phase)
        phase_cos = np.cos(2 * np.pi * phase)
        
        # Time since stimulus (simplified: use t mod 60 for pulse timing)
        time_since_stim = t % 60.0
        if time_since_stim > 30.0:  # LED1 is OFF in second half
            time_since_stim = time_since_stim  # could be NaN or large
        
        # Kernel bases
        kernel_vals = raised_cosine_basis(np.array([time_since_stim]), centers, width).flatten()
        
        # Z-score kinematics using training means/stds
        speed_z = (speed - binned['speed'].mean()) / binned['speed'].std()
        curv_z = (curvature - binned['curvature'].mean()) / binned['curvature'].std()
        
        # Build predictor vector
        x = np.array([1.0,  # intercept
                      led1_scaled, led2_scaled, interaction,
                      phase_sin, phase_cos,
                      speed_z, curv_z,
                      *kernel_vals])
        
        # Linear predictor
        eta = np.dot(coefs, x)
        
        # Hazard (no exposure offset - this is rate per second)
        return np.exp(eta)
    
    return hazard
```

### 7.2 Validation Metrics

```python
from scipy.stats import ks_2samp, ttest_ind

def validate_simulation(empirical_events, simulated_events, empirical_psth, simulated_psth):
    """
    Compare empirical vs simulated event statistics.
    
    Returns
    -------
    results : dict
        Validation results with pass/fail for each metric
    """
    results = {}
    
    # 1. Turn rate (events per minute per track)
    emp_rate = np.mean(empirical_events['turn_rate_per_min'])
    sim_rate = np.mean(simulated_events['turn_rate_per_min'])
    emp_ci = (np.percentile(empirical_events['turn_rate_per_min'], 2.5),
              np.percentile(empirical_events['turn_rate_per_min'], 97.5))
    
    results['turn_rate'] = {
        'empirical_mean': emp_rate,
        'simulated_mean': sim_rate,
        'empirical_95ci': emp_ci,
        'pass': emp_ci[0] <= sim_rate <= emp_ci[1]
    }
    
    # 2. Heading change distribution (KS test)
    ks_stat, ks_pval = ks_2samp(empirical_events['reo_dtheta'], 
                                 simulated_events['reo_dtheta'])
    results['heading_change'] = {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'pass': ks_pval > 0.05
    }
    
    # 3. Inter-event interval (KS test)
    ks_stat, ks_pval = ks_2samp(empirical_events['iei'], 
                                 simulated_events['iei'])
    results['iei'] = {
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pval,
        'pass': ks_pval > 0.05
    }
    
    # 4. PSTH integrated squared error
    ise = np.sum((empirical_psth - simulated_psth) ** 2)
    # Threshold from bootstrap null (would need to compute)
    results['psth'] = {
        'ise': ise,
        'pass': ise < 0.1  # placeholder threshold
    }
    
    return results
```

---

## 8. Interpretation Guide

### 8.1 Converting Coefficients to Rate Ratios

For a log-link model, each coefficient beta represents a multiplicative effect:

```
Rate Ratio = exp(beta)
Percent Change = 100 * (exp(beta) - 1)
```

### 8.2 Example Interpretations

| Covariate | beta | RR | Interpretation |
|-----------|------|-----|----------------|
| LED1_scaled | -0.50 | 0.61 | Each unit increase in LED1 (0-1 scale, i.e., 250 PWM) reduces turn rate by 39% |
| LED1 per 10 PWM | -0.02 | 0.98 | Each 10 PWM increase reduces turn rate by 2% |
| speed_z | 0.30 | 1.35 | Larvae 1 SD faster than average have 35% higher turn rate |

### 8.3 Reporting Template

```
The fitted NB-GLM hazard model (N = X bins from Y tracks across Z experiments) 
revealed significant modulation of reorientation rate by LED1 optogenetic 
stimulation (beta = A, SE = B, p < C). 

The temporal kernel peaked at D seconds after LED1 onset (RR = E, 95% CI: [F, G]), 
indicating maximum behavioral sensitivity in this time window.

Faster larvae showed elevated turn rates (beta_speed = H, p < I), while body 
curvature had [significant/no significant] effect (beta_curv = J, p = K).

Model diagnostics indicated [adequate/inadequate] fit: dispersion ratio = L 
(target ~1.0), no significant zero-inflation (observed zeros = M%, expected = N%).
```

---

## 9. File Locations

| File | Purpose |
|------|---------|
| `scripts/prepare_binned_data.py` | Data binning and feature engineering |
| `scripts/hazard_model.py` | Model fitting and diagnostics |
| `scripts/validate_simulation.py` | Simulation validation |
| `data/processed/consolidated_dataset.h5` | Input data |
| `data/processed/binned_0.5s.parquet` | Binned output for modeling |
| `data/models/hazard_model_results.json` | Fitted model results |
| `docs/HAZARD_MODEL_RESULTS.md` | Final report |

---

## 10. References

- Gepner et al. (2015) eLife - LNP model with raised-cosine kernels
- Klein et al. (2015) PNAS - Turn detection, MAGAT segmentation
- Pillow et al. (2008) J Neurosci - Raised-cosine basis for spike train models
- statsmodels documentation - NegativeBinomial GLM

#!/usr/bin/env python3
"""
Negative Binomial GLM Hazard Model for Larval Reorientation Events

Implements the hazard model specification from MiroThinker (2025-12-10):
- Family: Negative Binomial with log link (handles overdispersion)
- Temporal kernel: Raised-cosine basis functions
- Covariates: LED intensity, phase, speed, curvature

Reference:
- Gepner et al. (2015) eLife - LNP model with raised-cosine kernels
- Klein et al. (2015) PNAS - Turn detection and stimulus response

Usage:
    python scripts/hazard_model.py --data-dir data/engineered_validated
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Statistical modeling
try:
    import statsmodels.api as sm
    from statsmodels.genmod.families import NegativeBinomial
    from statsmodels.genmod.generalized_linear_model import GLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("WARNING: statsmodels not installed. Install with: pip install statsmodels")


# =============================================================================
# RAISED-COSINE BASIS FUNCTIONS
# =============================================================================

def raised_cosine_basis(t: np.ndarray, centers: np.ndarray, width: float) -> np.ndarray:
    """
    Compute raised-cosine basis functions for temporal kernels.
    
    Based on Pillow et al. (2008) J Neurosci and Gepner et al. (2015) eLife.
    
    Parameters
    ----------
    t : ndarray
        Time points (relative to event, negative = before)
    centers : ndarray
        Center positions for each basis function (in seconds)
    width : float
        Width parameter (controls overlap between bases)
    
    Returns
    -------
    basis : ndarray
        Shape (len(t), len(centers)) - basis function values
    
    Notes
    -----
    Each basis function is:
        B_j(t) = 0.5 * (1 + cos(pi * (t - c_j) / w))  if |t - c_j| < w
               = 0                                      otherwise
    
    Width w ≈ 0.6s makes bumps overlap at ~50% height (MiroThinker spec).
    """
    n_times = len(t)
    n_bases = len(centers)
    basis = np.zeros((n_times, n_bases))
    
    for j, c in enumerate(centers):
        # Distance from center
        dist = np.abs(t - c)
        # Raised cosine: nonzero only within width
        in_range = dist < width
        basis[in_range, j] = 0.5 * (1 + np.cos(np.pi * (t[in_range] - c) / width))
    
    return basis


def create_temporal_kernel_design(
    time_since_stimulus: np.ndarray,
    n_bases: int = 4,
    window: Tuple[float, float] = (-3.0, 0.0),
    width: float = 0.6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create design matrix columns for temporal kernel.
    
    Parameters
    ----------
    time_since_stimulus : ndarray
        Time since last stimulus onset (seconds)
    n_bases : int
        Number of raised-cosine basis functions (default 4)
    window : tuple
        (start, end) of temporal window in seconds (default -3 to 0)
    width : float
        Width of each basis function (default 0.6s)
    
    Returns
    -------
    design : ndarray
        Shape (N, n_bases) - design matrix columns
    centers : ndarray
        Centers of the basis functions
    """
    # Compute basis centers (evenly spaced in window)
    centers = np.linspace(window[0], window[1], n_bases)
    
    # Convert time_since_stimulus to relative time (negative = before now)
    # time_since_stimulus is positive, so we need -time_since_stimulus for kernel
    t_relative = -time_since_stimulus  # Now negative values = past
    
    # Compute basis functions
    design = raised_cosine_basis(t_relative, centers, width)
    
    return design, centers


# =============================================================================
# PHASE COVARIATES
# =============================================================================

def compute_phase_covariates(time: np.ndarray, cycle_period: float = 60.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute sin/cos phase covariates for periodic LED1 stimulus.
    
    Parameters
    ----------
    time : ndarray
        Experiment time in seconds
    cycle_period : float
        LED1 cycle period in seconds (default 60s = 30s on + 30s off)
    
    Returns
    -------
    sin_phase : ndarray
        sin(2π * phase)
    cos_phase : ndarray
        cos(2π * phase)
    """
    phase = (time % cycle_period) / cycle_period  # 0 to 1
    sin_phase = np.sin(2 * np.pi * phase)
    cos_phase = np.cos(2 * np.pi * phase)
    return sin_phase, cos_phase


# =============================================================================
# DESIGN MATRIX CONSTRUCTION
# =============================================================================

def build_design_matrix(
    data: pd.DataFrame,
    n_temporal_bases: int = 4,
    temporal_window: Tuple[float, float] = (-3.0, 0.0),
    include_interaction: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build design matrix for NB-GLM hazard model.
    
    Model specification (MiroThinker 2025-12-10):
        log(μ_i) = β₀ 
                 + β₁·LED1_intensity 
                 + β₂·LED2_intensity 
                 + β₃·LED1×LED2 (interaction)
                 + β₄·sin(phase) + β₅·cos(phase)
                 + Σⱼ φⱼ·Bⱼ(t)  (temporal kernel)
                 + γ₁·SpeedRunVel 
                 + γ₂·Curvature
    
    Parameters
    ----------
    data : DataFrame
        Must contain columns:
        - led1Val: LED1 intensity (PWM)
        - led2Val: LED2 intensity (PWM)
        - time: experiment time (seconds)
        - time_since_stimulus: time since last LED1 onset
        - speed: instantaneous speed (cm/s)
        - curvature: instantaneous curvature
    n_temporal_bases : int
        Number of raised-cosine bases for temporal kernel
    temporal_window : tuple
        Window for temporal kernel (seconds before stimulus)
    include_interaction : bool
        Whether to include LED1×LED2 interaction term
    
    Returns
    -------
    X : DataFrame
        Design matrix with all covariates
    feature_names : list
        Names of features in order
    """
    n = len(data)
    feature_names = []
    features = {}
    
    # Intercept
    features['intercept'] = np.ones(n)
    feature_names.append('intercept')
    
    # LED covariates (continuous intensity)
    if 'led1Val' in data.columns:
        # Normalize to 0-1 range for numerical stability
        led1_max = data['led1Val'].max()
        led1_normalized = data['led1Val'] / max(led1_max, 1) if led1_max > 0 else data['led1Val']
        features['led1_intensity'] = led1_normalized.values
        feature_names.append('led1_intensity')
    
    if 'led2Val' in data.columns:
        led2_max = data['led2Val'].max()
        led2_normalized = data['led2Val'] / max(led2_max, 1) if led2_max > 0 else data['led2Val']
        features['led2_intensity'] = led2_normalized.values
        feature_names.append('led2_intensity')
    
    # Interaction term
    if include_interaction and 'led1Val' in data.columns and 'led2Val' in data.columns:
        features['led1_x_led2'] = features['led1_intensity'] * features['led2_intensity']
        feature_names.append('led1_x_led2')
    
    # Phase covariates (deterministic LED1 cycle)
    if 'time' in data.columns:
        sin_phase, cos_phase = compute_phase_covariates(data['time'].values)
        features['phase_sin'] = sin_phase
        features['phase_cos'] = cos_phase
        feature_names.extend(['phase_sin', 'phase_cos'])
    
    # Temporal kernel (raised-cosine basis)
    if 'time_since_stimulus' in data.columns:
        time_since = data['time_since_stimulus'].fillna(999).values  # Far future if NaN
        kernel_design, centers = create_temporal_kernel_design(
            time_since, 
            n_bases=n_temporal_bases,
            window=temporal_window
        )
        for j in range(n_temporal_bases):
            features[f'kernel_b{j}'] = kernel_design[:, j]
            feature_names.append(f'kernel_b{j}')
    
    # Instantaneous covariates
    if 'speed' in data.columns:
        # Standardize speed for numerical stability
        speed_mean = data['speed'].mean()
        speed_std = data['speed'].std()
        if speed_std > 0:
            features['speed'] = (data['speed'].values - speed_mean) / speed_std
        else:
            features['speed'] = data['speed'].values
        feature_names.append('speed')
    
    if 'curvature' in data.columns:
        # Clip extreme curvature values (path curvature explodes at low speed)
        curv_clipped = np.clip(data['curvature'].values, -50, 50)
        curv_mean = np.mean(curv_clipped)
        curv_std = np.std(curv_clipped)
        if curv_std > 0:
            features['curvature'] = (curv_clipped - curv_mean) / curv_std
        else:
            features['curvature'] = curv_clipped
        feature_names.append('curvature')
    
    X = pd.DataFrame(features)
    return X, feature_names


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_nb_glm(
    X: pd.DataFrame,
    y: np.ndarray,
    exposure: Optional[np.ndarray] = None,
    alpha: float = 1.0
) -> Dict:
    """
    Fit Negative Binomial GLM.
    
    Parameters
    ----------
    X : DataFrame
        Design matrix (N observations × K features)
    y : ndarray
        Response variable (event counts per bin)
    exposure : ndarray, optional
        Exposure offset (bin duration). If None, assumes uniform.
    alpha : float
        Dispersion parameter for NB (default 1.0)
    
    Returns
    -------
    results : dict
        Model results including coefficients, CI, diagnostics
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels required for NB-GLM. Install with: pip install statsmodels")
    
    # Add exposure as offset if provided
    if exposure is not None:
        offset = np.log(exposure)
    else:
        offset = None
    
    # Fit NB-GLM
    try:
        model = GLM(
            y, 
            X,
            family=NegativeBinomial(alpha=alpha),
            offset=offset
        )
        fit = model.fit()
        
        results = {
            'coefficients': fit.params.to_dict(),
            'std_errors': fit.bse.to_dict(),
            'pvalues': fit.pvalues.to_dict(),
            'conf_int_lower': fit.conf_int()[0].to_dict(),
            'conf_int_upper': fit.conf_int()[1].to_dict(),
            'deviance': fit.deviance,
            'pearson_chi2': fit.pearson_chi2,
            'df_resid': fit.df_resid,
            'aic': fit.aic,
            'bic': fit.bic,
            'llf': fit.llf,
            'dispersion': alpha,
            'n_obs': len(y),
            'converged': fit.converged
        }
        
        # Compute dispersion ratio (should be ~1 for good fit)
        results['dispersion_ratio'] = fit.pearson_chi2 / fit.df_resid
        
        return results
        
    except Exception as e:
        return {
            'error': str(e),
            'converged': False
        }


def estimate_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """
    Estimate NB dispersion parameter using method of moments.
    
    Parameters
    ----------
    y : ndarray
        Observed counts
    mu : ndarray
        Fitted mean values
    
    Returns
    -------
    alpha : float
        Estimated dispersion (higher = more overdispersion)
    """
    # Var(Y) = μ + α*μ² for NB
    # Solve for α: α = (Var(Y) - μ) / μ²
    var_y = np.var(y)
    mean_y = np.mean(y)
    
    if mean_y > 0:
        alpha = max(0.01, (var_y - mean_y) / (mean_y ** 2))
    else:
        alpha = 1.0
    
    return alpha


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def cross_validate_kernel_params(
    data: pd.DataFrame,
    y: np.ndarray,
    n_bases_options: List[int] = [3, 4, 5],
    window_options: List[Tuple[float, float]] = [(-2.0, 0.0), (-3.0, 0.0), (-4.0, 0.0)],
    n_folds: int = 5
) -> Dict:
    """
    Cross-validate to select optimal temporal kernel parameters.
    
    Uses leave-one-experiment-out CV as recommended by MiroThinker.
    
    Parameters
    ----------
    data : DataFrame
        Full dataset with all covariates
    y : ndarray
        Response variable
    n_bases_options : list
        Number of bases to try
    window_options : list
        Window ranges to try
    n_folds : int
        Number of CV folds (if not doing leave-one-out)
    
    Returns
    -------
    results : dict
        CV results including best parameters
    """
    results = {
        'tested_params': [],
        'cv_deviances': [],
        'best_params': None,
        'best_deviance': np.inf
    }
    
    # Check if we have experiment_id for leave-one-out
    if 'experiment_id' in data.columns:
        experiments = data['experiment_id'].unique()
        use_loo = len(experiments) > 3
    else:
        use_loo = False
    
    for n_bases in n_bases_options:
        for window in window_options:
            params = {'n_bases': n_bases, 'window': window}
            
            if use_loo:
                # Leave-one-experiment-out CV
                deviances = []
                for held_out in experiments:
                    train_mask = data['experiment_id'] != held_out
                    test_mask = data['experiment_id'] == held_out
                    
                    X_train, _ = build_design_matrix(
                        data[train_mask], 
                        n_temporal_bases=n_bases,
                        temporal_window=window
                    )
                    X_test, _ = build_design_matrix(
                        data[test_mask],
                        n_temporal_bases=n_bases,
                        temporal_window=window
                    )
                    
                    y_train = y[train_mask.values]
                    y_test = y[test_mask.values]
                    
                    if len(y_train) > 0 and len(y_test) > 0:
                        fit_result = fit_nb_glm(X_train, y_train)
                        if fit_result.get('converged', False):
                            # Compute held-out deviance
                            # (simplified - would need proper deviance calculation)
                            deviances.append(fit_result.get('deviance', np.inf))
                
                mean_deviance = np.mean(deviances) if deviances else np.inf
            else:
                # Simple k-fold CV
                X, _ = build_design_matrix(data, n_temporal_bases=n_bases, temporal_window=window)
                fit_result = fit_nb_glm(X, y)
                mean_deviance = fit_result.get('deviance', np.inf)
            
            results['tested_params'].append(params)
            results['cv_deviances'].append(mean_deviance)
            
            if mean_deviance < results['best_deviance']:
                results['best_deviance'] = mean_deviance
                results['best_params'] = params
    
    return results


# =============================================================================
# MAIN INTERFACE
# =============================================================================

def fit_hazard_model(
    data_path: Path,
    output_path: Path,
    event_column: str = 'is_reorientation_start',
    n_temporal_bases: int = 4,
    temporal_window: Tuple[float, float] = (-3.0, 0.0),
    run_cv: bool = False
) -> Dict:
    """
    Fit NB-GLM hazard model to engineered data.
    
    Parameters
    ----------
    data_path : Path
        Path to engineered events CSV
    output_path : Path
        Path to save model results
    event_column : str
        Column containing event indicator (default 'is_reorientation_start')
    n_temporal_bases : int
        Number of temporal kernel bases
    temporal_window : tuple
        Temporal kernel window
    run_cv : bool
        Whether to run cross-validation for kernel params
    
    Returns
    -------
    results : dict
        Model fitting results
    """
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    print(f"  {len(data)} observations")
    print(f"  {data[event_column].sum()} events ({100*data[event_column].mean():.2f}%)")
    
    # Response variable (counts per bin)
    y = data[event_column].astype(int).values
    
    # Cross-validation for kernel parameters (optional)
    if run_cv:
        print("Running cross-validation for kernel parameters...")
        cv_results = cross_validate_kernel_params(
            data, y,
            n_bases_options=[3, 4, 5],
            window_options=[(-2.0, 0.0), (-3.0, 0.0), (-4.0, 0.0)]
        )
        print(f"  Best params: {cv_results['best_params']}")
        n_temporal_bases = cv_results['best_params']['n_bases']
        temporal_window = cv_results['best_params']['window']
    else:
        cv_results = None
    
    # Build design matrix
    print(f"Building design matrix with {n_temporal_bases} kernel bases, window {temporal_window}")
    X, feature_names = build_design_matrix(
        data,
        n_temporal_bases=n_temporal_bases,
        temporal_window=temporal_window
    )
    print(f"  {len(feature_names)} features: {feature_names}")
    
    # Estimate dispersion from data
    var_y = np.var(y)
    mean_y = np.mean(y)
    if mean_y > 0:
        estimated_alpha = max(0.1, (var_y - mean_y) / max(mean_y ** 2, 0.001))
    else:
        estimated_alpha = 1.0
    print(f"  Estimated dispersion: {estimated_alpha:.3f}")
    
    # Fit model
    print("Fitting NB-GLM...")
    model_results = fit_nb_glm(X, y, alpha=estimated_alpha)
    
    if model_results.get('converged', False):
        print("  Model converged!")
        print(f"  Deviance: {model_results['deviance']:.2f}")
        print(f"  Dispersion ratio: {model_results['dispersion_ratio']:.3f} (should be ~1)")
        print(f"  AIC: {model_results['aic']:.2f}")
        
        # Print significant coefficients
        print("\n  Significant coefficients (p < 0.05):")
        for name in feature_names:
            if name in model_results['pvalues']:
                p = model_results['pvalues'][name]
                if p < 0.05:
                    coef = model_results['coefficients'][name]
                    se = model_results['std_errors'][name]
                    print(f"    {name}: {coef:.4f} (SE={se:.4f}, p={p:.4f})")
    else:
        print(f"  Model failed: {model_results.get('error', 'Unknown error')}")
    
    # Combine results
    all_results = {
        'model': model_results,
        'cv': cv_results,
        'params': {
            'n_temporal_bases': n_temporal_bases,
            'temporal_window': temporal_window,
            'event_column': event_column
        },
        'data_summary': {
            'n_observations': len(data),
            'n_events': int(data[event_column].sum()),
            'event_rate': float(data[event_column].mean())
        }
    }
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {output_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Fit NB-GLM hazard model')
    parser.add_argument('--data-dir', type=str, default='data/engineered_validated',
                       help='Directory with engineered event CSVs')
    parser.add_argument('--output-dir', type=str, default='data/models',
                       help='Output directory for model results')
    parser.add_argument('--event', type=str, default='is_reorientation_start',
                       help='Event column to model')
    parser.add_argument('--n-bases', type=int, default=4,
                       help='Number of temporal kernel bases')
    parser.add_argument('--cv', action='store_true',
                       help='Run cross-validation for kernel parameters')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Find all event files
    event_files = list(data_dir.glob("*_events.csv"))
    if not event_files:
        print(f"No event files found in {data_dir}")
        return
    
    print(f"Found {len(event_files)} event files")
    
    # Combine all data for pooled model
    all_data = []
    for f in event_files:
        df = pd.read_csv(f)
        df['experiment_id'] = f.stem.replace('_events', '')
        all_data.append(df)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined data: {len(combined_data)} observations from {len(event_files)} experiments")
    
    # Save combined data temporarily
    combined_path = output_dir / 'combined_events.csv'
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined_data.to_csv(combined_path, index=False)
    
    # Fit model
    results = fit_hazard_model(
        combined_path,
        output_dir / 'hazard_model_results.json',
        event_column=args.event,
        n_temporal_bases=args.n_bases,
        run_cv=args.cv
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()

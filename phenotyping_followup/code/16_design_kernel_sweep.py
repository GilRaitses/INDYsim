#!/usr/bin/env python3
"""
Systematic Design × Kernel Regime Sweep
Rigorous analysis of experimental design choices across kernel types.

This produces the key figure showing when each design works.
"""

import numpy as np
from scipy import stats, special, optimize
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import time
from datetime import datetime
from itertools import product

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)

log("="*70)
log("DESIGN x KERNEL REGIME SYSTEMATIC SWEEP")
log("="*70)

OUTPUT_DIR = Path('results/design_kernel_sweep')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fixed parameters
ALPHA = 2.0
TAU1_TRUE = 0.63  # True value we're trying to estimate
TAU2 = 2.48
BASELINE = -3.5
N_CYCLES = 20
N_LARVAE = 200
N_BOOTSTRAP = 100
DT = 0.02

# Sweep dimensions
AB_RATIOS = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]  # Current kernel is 0.125
DESIGNS = {
    'Continuous 10s': [(0, 10)],  # Single 10s pulse
    'Burst 10x0.5s': [(i, i+0.5) for i in np.arange(0, 10, 1)],  # 10 pulses, 0.5s gap
    'Medium 4x1s': [(i*2.5, i*2.5+1) for i in range(4)],  # 4 pulses, 1.5s gap
    'Long 2x2s': [(0, 2), (5, 7)],  # 2 pulses, 3s gap
}

def gamma_pdf(t, alpha, tau):
    beta = tau / alpha
    result = np.zeros_like(t)
    valid = t > 0
    if np.any(valid):
        log_pdf = ((alpha - 1) * np.log(t[valid]) - t[valid] / beta 
                   - alpha * np.log(beta) - special.gammaln(alpha))
        result[valid] = np.exp(log_pdf)
    return result

def kernel(t, tau1, A, B):
    """Gamma-difference kernel with given A, B."""
    beta1, beta2 = tau1 / ALPHA, TAU2 / ALPHA
    return A * gamma_pdf(t, ALPHA, beta1) - B * gamma_pdf(t, ALPHA, beta2)

def compute_fisher_info(tau1, A, B, design, n_points=1000):
    """Compute Fisher information for tau1 under given design."""
    # Numerical derivative of kernel w.r.t. tau1
    eps = 0.001
    
    fisher = 0.0
    for on_start, on_end in design:
        t = np.linspace(on_start + 0.01, on_end, n_points)
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        
        K = kernel(t - on_start, tau1, A, B)
        K_plus = kernel(t - on_start, tau1 + eps, A, B)
        K_minus = kernel(t - on_start, tau1 - eps, A, B)
        
        dK_dtau1 = (K_plus - K_minus) / (2 * eps)
        
        # Lambda(t) = exp(baseline + K(t))
        lam = np.exp(BASELINE + K)
        
        # Fisher info: integral of (dK/dtau1)^2 * lambda(t) dt
        fisher += np.sum(dK_dtau1**2 * lam) * dt
    
    return fisher

def simulate_events(tau1, A, B, design, n_cycles, seed):
    """Simulate events under given kernel and design."""
    np.random.seed(seed)
    
    cycle_duration = 30.0  # 10s ON + 20s OFF
    total_duration = n_cycles * cycle_duration
    times = np.arange(0, total_duration, DT)
    
    event_times = []
    
    for t in times:
        cycle_time = t % cycle_duration
        
        # Check if in any ON window
        in_on = False
        time_since_onset = 0
        for on_start, on_end in design:
            if on_start <= cycle_time < on_end:
                in_on = True
                time_since_onset = cycle_time - on_start
                break
        
        if in_on:
            K = kernel(np.array([time_since_onset]), tau1, A, B)[0]
        else:
            K = 0.0
        
        log_hazard = BASELINE + K
        hazard = np.exp(log_hazard)
        p_event = 1 - np.exp(-hazard * DT)
        
        if np.random.random() < p_event:
            event_times.append(t)
    
    return np.array(event_times)

def fit_tau1(event_times, A, B, design, n_cycles):
    """Fit tau1 given fixed A, B and design."""
    cycle_duration = 30.0
    total_duration = n_cycles * cycle_duration
    
    def neg_log_likelihood(tau1_val):
        if tau1_val <= 0.1 or tau1_val > 2.0:
            return 1e10
        
        # Compute log-likelihood
        ll = 0.0
        
        # Event term
        for et in event_times:
            cycle_time = et % cycle_duration
            for on_start, on_end in design:
                if on_start <= cycle_time < on_end:
                    time_since_onset = cycle_time - on_start
                    K = kernel(np.array([time_since_onset]), tau1_val, A, B)[0]
                    ll += BASELINE + K
                    break
            else:
                ll += BASELINE
        
        # Integral term (approximate)
        times = np.arange(0, total_duration, DT * 5)  # Coarser for speed
        for t in times:
            cycle_time = t % cycle_duration
            in_on = False
            for on_start, on_end in design:
                if on_start <= cycle_time < on_end:
                    time_since_onset = cycle_time - on_start
                    K = kernel(np.array([time_since_onset]), tau1_val, A, B)[0]
                    in_on = True
                    break
            if not in_on:
                K = 0.0
            ll -= np.exp(BASELINE + K) * DT * 5
        
        return -ll
    
    # Grid search
    tau1_grid = np.linspace(0.2, 1.5, 30)
    best_tau1 = tau1_grid[0]
    best_nll = neg_log_likelihood(best_tau1)
    
    for tau1_val in tau1_grid:
        nll = neg_log_likelihood(tau1_val)
        if nll < best_nll:
            best_nll = nll
            best_tau1 = tau1_val
    
    return best_tau1

def analyze_condition(ab_ratio, design_name, design):
    """Analyze one design × kernel combination."""
    # Set A and B based on ratio (keeping B=12 as reference)
    B = 12.0
    A = B * ab_ratio
    
    # Compute Fisher information
    fisher = compute_fisher_info(TAU1_TRUE, A, B, design)
    
    # Simulate and fit
    fitted_tau1s = []
    event_counts = []
    
    for i in range(N_LARVAE):
        events = simulate_events(TAU1_TRUE, A, B, design, N_CYCLES, seed=i*1000)
        event_counts.append(len(events))
        
        if len(events) >= 3:
            fitted = fit_tau1(events, A, B, design, N_CYCLES)
            fitted_tau1s.append(fitted)
    
    fitted_tau1s = np.array(fitted_tau1s)
    event_counts = np.array(event_counts)
    
    if len(fitted_tau1s) < 10:
        return {
            'ab_ratio': ab_ratio, 'design': design_name,
            'fisher': fisher, 'mean_events': np.mean(event_counts),
            'bias': np.nan, 'rmse': np.nan, 'power': 0.0
        }
    
    # Compute metrics
    bias = np.mean(fitted_tau1s) - TAU1_TRUE
    rmse = np.sqrt(np.mean((fitted_tau1s - TAU1_TRUE)**2))
    
    # Power: can we detect tau1 < 0.5 (fast responder)?
    # Bootstrap to get SE, then compute power to detect 0.3 effect
    se = np.std(fitted_tau1s) / np.sqrt(len(fitted_tau1s))
    effect_size = 0.3 / (np.std(fitted_tau1s) + 1e-6)
    
    # Power via t-distribution
    from scipy.stats import nct
    df = len(fitted_tau1s) - 1
    nc = effect_size * np.sqrt(len(fitted_tau1s))
    crit = stats.t.ppf(0.975, df)
    power = 1 - nct.cdf(crit, df, nc) + nct.cdf(-crit, df, nc)
    power = min(max(power, 0), 1) * 100
    
    return {
        'ab_ratio': ab_ratio, 'design': design_name,
        'A': A, 'B': B,
        'fisher': fisher, 'mean_events': np.mean(event_counts),
        'bias': bias, 'rmse': rmse, 'power': power,
        'fitted_mean': np.mean(fitted_tau1s), 'fitted_std': np.std(fitted_tau1s)
    }

def main():
    log(f"Sweeping {len(AB_RATIOS)} kernel regimes x {len(DESIGNS)} designs = {len(AB_RATIOS)*len(DESIGNS)} conditions")
    log(f"N_LARVAE={N_LARVAE}, N_CYCLES={N_CYCLES}")
    
    results = []
    total = len(AB_RATIOS) * len(DESIGNS)
    idx = 0
    start = time.time()
    
    for ab_ratio in AB_RATIOS:
        for design_name, design in DESIGNS.items():
            idx += 1
            log(f"\n[{idx}/{total}] A/B={ab_ratio:.3f}, Design={design_name}")
            
            t0 = time.time()
            result = analyze_condition(ab_ratio, design_name, design)
            results.append(result)
            
            log(f"  Fisher={result['fisher']:.4f}, Events={result['mean_events']:.1f}")
            log(f"  Bias={result.get('bias', np.nan):.3f}, RMSE={result.get('rmse', np.nan):.3f}, Power={result.get('power', 0):.1f}%")
            log(f"  Time: {time.time()-t0:.1f}s")
            
            elapsed = time.time() - start
            remain = elapsed / idx * (total - idx)
            log(f"  Elapsed: {elapsed:.0f}s, Remaining: {remain:.0f}s")
    
    log(f"\n{'='*70}")
    log(f"COMPLETE in {time.time()-start:.0f}s")
    log("="*70)
    
    # Create summary tables
    design_names = list(DESIGNS.keys())
    
    log("\n" + "="*80)
    log("FISHER INFORMATION TABLE")
    log("="*80)
    header = f"{'A/B Ratio':<12}" + "".join(f"{d:<18}" for d in design_names)
    log(header)
    for ab in AB_RATIOS:
        row = f"{ab:<12.3f}"
        for dn in design_names:
            r = next((x for x in results if x['ab_ratio']==ab and x['design']==dn), None)
            row += f"{r['fisher']:<18.4f}" if r else "N/A"
        log(row)
    
    log("\n" + "="*80)
    log("POWER TABLE (%)")
    log("="*80)
    log(header)
    for ab in AB_RATIOS:
        row = f"{ab:<12.3f}"
        for dn in design_names:
            r = next((x for x in results if x['ab_ratio']==ab and x['design']==dn), None)
            row += f"{r.get('power', 0):<18.1f}" if r else "N/A"
        log(row)
    
    log("\n" + "="*80)
    log("BIAS TABLE")
    log("="*80)
    log(header)
    for ab in AB_RATIOS:
        row = f"{ab:<12.3f}"
        for dn in design_names:
            r = next((x for x in results if x['ab_ratio']==ab and x['design']==dn), None)
            bias = r.get('bias', np.nan) if r else np.nan
            row += f"{bias:<18.3f}" if not np.isnan(bias) else "N/A".ljust(18)
        log(row)
    
    log("\n" + "="*80)
    log("MEAN EVENTS TABLE")
    log("="*80)
    log(header)
    for ab in AB_RATIOS:
        row = f"{ab:<12.3f}"
        for dn in design_names:
            r = next((x for x in results if x['ab_ratio']==ab and x['design']==dn), None)
            row += f"{r['mean_events']:<18.1f}" if r else "N/A"
        log(row)
    
    # Key findings
    log("\n" + "="*80)
    log("KEY FINDINGS")
    log("="*80)
    
    # Best design for each kernel regime
    for ab in AB_RATIOS:
        sub = [r for r in results if r['ab_ratio'] == ab and not np.isnan(r.get('power', np.nan))]
        if sub:
            best = max(sub, key=lambda x: x.get('power', 0))
            log(f"A/B = {ab:.3f}: Best design = {best['design']} (Power={best['power']:.1f}%, Fisher={best['fisher']:.4f})")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Prepare data for heatmaps
    fisher_matrix = np.zeros((len(AB_RATIOS), len(design_names)))
    power_matrix = np.zeros((len(AB_RATIOS), len(design_names)))
    bias_matrix = np.zeros((len(AB_RATIOS), len(design_names)))
    events_matrix = np.zeros((len(AB_RATIOS), len(design_names)))
    
    for i, ab in enumerate(AB_RATIOS):
        for j, dn in enumerate(design_names):
            r = next((x for x in results if x['ab_ratio']==ab and x['design']==dn), None)
            if r:
                fisher_matrix[i, j] = r['fisher']
                power_matrix[i, j] = r.get('power', 0)
                bias_matrix[i, j] = r.get('bias', 0) if not np.isnan(r.get('bias', np.nan)) else 0
                events_matrix[i, j] = r['mean_events']
    
    # Fisher Information heatmap
    ax = axes[0, 0]
    im = ax.imshow(fisher_matrix, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(design_names)))
    ax.set_xticklabels(design_names, rotation=45, ha='right')
    ax.set_yticks(range(len(AB_RATIOS)))
    ax.set_yticklabels([f'{ab:.3f}' for ab in AB_RATIOS])
    ax.set_ylabel('A/B Ratio (Kernel Type)')
    ax.set_title('Fisher Information for τ₁')
    plt.colorbar(im, ax=ax, label='Fisher Info')
    # Add text annotations
    for i in range(len(AB_RATIOS)):
        for j in range(len(design_names)):
            ax.text(j, i, f'{fisher_matrix[i,j]:.3f}', ha='center', va='center', fontsize=8)
    
    # Power heatmap
    ax = axes[0, 1]
    im = ax.imshow(power_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax.set_xticks(range(len(design_names)))
    ax.set_xticklabels(design_names, rotation=45, ha='right')
    ax.set_yticks(range(len(AB_RATIOS)))
    ax.set_yticklabels([f'{ab:.3f}' for ab in AB_RATIOS])
    ax.set_ylabel('A/B Ratio (Kernel Type)')
    ax.set_title('Power to Detect τ₁ Difference (%)')
    plt.colorbar(im, ax=ax, label='Power %')
    for i in range(len(AB_RATIOS)):
        for j in range(len(design_names)):
            ax.text(j, i, f'{power_matrix[i,j]:.0f}', ha='center', va='center', fontsize=8)
    
    # Bias heatmap
    ax = axes[1, 0]
    max_bias = max(abs(bias_matrix.min()), abs(bias_matrix.max()), 0.1)
    im = ax.imshow(bias_matrix, aspect='auto', cmap='RdBu_r', vmin=-max_bias, vmax=max_bias)
    ax.set_xticks(range(len(design_names)))
    ax.set_xticklabels(design_names, rotation=45, ha='right')
    ax.set_yticks(range(len(AB_RATIOS)))
    ax.set_yticklabels([f'{ab:.3f}' for ab in AB_RATIOS])
    ax.set_ylabel('A/B Ratio (Kernel Type)')
    ax.set_title('Bias in τ₁ Estimation (s)')
    plt.colorbar(im, ax=ax, label='Bias (s)')
    for i in range(len(AB_RATIOS)):
        for j in range(len(design_names)):
            ax.text(j, i, f'{bias_matrix[i,j]:.2f}', ha='center', va='center', fontsize=8)
    
    # Mean events heatmap
    ax = axes[1, 1]
    im = ax.imshow(events_matrix, aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(design_names)))
    ax.set_xticklabels(design_names, rotation=45, ha='right')
    ax.set_yticks(range(len(AB_RATIOS)))
    ax.set_yticklabels([f'{ab:.3f}' for ab in AB_RATIOS])
    ax.set_ylabel('A/B Ratio (Kernel Type)')
    ax.set_title('Mean Events per Track')
    plt.colorbar(im, ax=ax, label='Events')
    for i in range(len(AB_RATIOS)):
        for j in range(len(design_names)):
            ax.text(j, i, f'{events_matrix[i,j]:.0f}', ha='center', va='center', fontsize=8)
    
    # Mark current kernel regime
    current_idx = AB_RATIOS.index(0.125)
    for ax in axes.flat:
        ax.axhline(current_idx - 0.5, color='red', lw=2, ls='--', alpha=0.7)
        ax.axhline(current_idx + 0.5, color='red', lw=2, ls='--', alpha=0.7)
    
    plt.suptitle('Design × Kernel Regime Sweep\n(Red box = current GMR61 kernel, A/B = 0.125)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'design_kernel_sweep.png', dpi=150, bbox_inches='tight')
    log(f"\nFigure saved: {OUTPUT_DIR / 'design_kernel_sweep.png'}")
    
    # Save results
    import json
    with open(OUTPUT_DIR / 'sweep_results.json', 'w') as f:
        # Convert to serializable format
        serializable = []
        for r in results:
            sr = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in r.items()}
            serializable.append(sr)
        json.dump(serializable, f, indent=2)
    log(f"Results saved: {OUTPUT_DIR / 'sweep_results.json'}")
    
    log("\nDONE")

if __name__ == '__main__':
    main()


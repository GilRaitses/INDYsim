#!/usr/bin/env python3
"""
Composite Phenotype Validation - FIXED BURSTINESS
Precision: modulates ON/OFF hazard ratio
Burstiness: modulates event clustering via self-excitation
"""

import numpy as np
from scipy import stats, special
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)

log("="*70)
log("COMPOSITE PHENOTYPE VALIDATION - FIXED BURSTINESS")
log("="*70)

OUTPUT_DIR = Path('results/composite_validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
A, B, ALPHA, TAU1, TAU2 = 1.5, 12.0, 2.0, 0.63, 2.48
LED_ON, LED_OFF = 10.0, 20.0
CYCLE_DURATION = LED_ON + LED_OFF
DT = 0.05  # Finer timestep for burstiness
BETA_P = 0.5   # Precision modulation strength
BETA_B = 2.0   # Burstiness modulation (self-excitation decay)

BASELINES = [-3.5, -2.5, -1.5]
TARGET_EVENTS = [10, 20, 40, 80]
N_LARVAE = 500
N_REPLICATES = 50

log(f"N_LARVAE={N_LARVAE}, DT={DT}, Conditions={len(BASELINES)*len(TARGET_EVENTS)}")

def gamma_pdf(x, alpha, beta):
    result = np.zeros_like(x)
    valid = x > 0
    if np.any(valid):
        log_pdf = ((alpha - 1) * np.log(x[valid]) - x[valid] / beta 
                   - alpha * np.log(beta) - special.gammaln(alpha))
        result[valid] = np.exp(log_pdf)
    return result

def simulate_larva(precision, burstiness, baseline, n_cycles, seed):
    """
    Simulate one larva with:
    - Precision: modulates ON vs OFF hazard
    - Burstiness: self-exciting process (events trigger more events)
    """
    np.random.seed(seed)
    
    duration = n_cycles * CYCLE_DURATION
    times = np.arange(0, duration, DT)
    n_steps = len(times)
    
    # Precompute LED state and kernel
    cycle_time = times % CYCLE_DURATION
    led_on = cycle_time < LED_ON
    time_since_onset = np.where(led_on, cycle_time, 0.0)
    
    beta1, beta2 = TAU1 / ALPHA, TAU2 / ALPHA
    K = A * gamma_pdf(time_since_onset, ALPHA, beta1) - B * gamma_pdf(time_since_onset, ALPHA, beta2)
    
    # Precision modulates ON/OFF differently
    b_mod = np.where(led_on, baseline + BETA_P * precision, baseline - BETA_P * precision * 0.3)
    
    # Self-excitation kernel for burstiness
    # Higher burstiness = events cause more nearby events
    self_excite_tau = 0.5  # 0.5s decay
    self_excite_amp = 0.5 + burstiness * 0.3  # Amplitude scales with burstiness
    self_excite_amp = np.clip(self_excite_amp, 0.1, 1.5)
    
    # Simulate with self-excitation
    event_times = []
    excitation = 0.0
    
    for i, t in enumerate(times):
        # Decay excitation
        excitation *= np.exp(-DT / self_excite_tau)
        
        # Total log-hazard
        log_h = b_mod[i] + K[i] + excitation
        hazard = np.exp(log_h)
        p_event = 1 - np.exp(-hazard * DT)
        
        if np.random.random() < p_event:
            event_times.append(t)
            # Self-excitation boost
            excitation += self_excite_amp
    
    return np.array(event_times), n_cycles

def compute_measures(event_times, n_cycles):
    """Compute 7 behavioral measures."""
    X = np.zeros(7)
    
    # 1. ON/OFF ratio
    on_events = sum(1 for t in event_times if (t % CYCLE_DURATION) < LED_ON)
    off_events = len(event_times) - on_events
    X[0] = (on_events + 0.5) / (off_events + 0.5)
    
    # 2. Median first-event latency
    latencies = []
    for c in range(n_cycles):
        cycle_start = c * CYCLE_DURATION
        cycle_events = event_times[(event_times >= cycle_start) & (event_times < cycle_start + LED_ON)]
        if len(cycle_events) > 0:
            latencies.append(cycle_events[0] - cycle_start)
    X[1] = np.median(latencies) if latencies else LED_ON
    
    # 3. IEI-CV (key burstiness measure)
    if len(event_times) > 1:
        ieis = np.diff(event_times)
        X[2] = np.std(ieis) / (np.mean(ieis) + 1e-6)
    else:
        X[2] = 0
    
    # 4. Fano factor (key burstiness measure)
    counts = []
    for c in range(n_cycles):
        cycle_start = c * CYCLE_DURATION
        counts.append(np.sum((event_times >= cycle_start) & (event_times < cycle_start + LED_ON)))
    counts = np.array(counts)
    X[3] = np.var(counts) / (np.mean(counts) + 1e-6) if np.mean(counts) > 0 else 1
    
    # 5. Reliability
    X[4] = np.mean(counts > 0)
    
    # 6. Habituation slope
    if n_cycles > 2 and np.std(counts) > 0:
        X[5] = stats.linregress(np.arange(n_cycles), counts)[0]
    else:
        X[5] = 0
    
    # 7. Phase coherence
    on_phases = []
    for et in event_times:
        phase = et % CYCLE_DURATION
        if phase < LED_ON:
            on_phases.append(2 * np.pi * phase / LED_ON)
    if len(on_phases) > 1:
        X[6] = np.abs(np.mean(np.exp(1j * np.array(on_phases))))
    else:
        X[6] = 0
    
    return X

def run_condition(baseline, target_events, condition_idx, total_conditions):
    """Run one experimental condition."""
    log(f"\nCondition {condition_idx}/{total_conditions}: baseline={baseline}, target={target_events}")
    
    # Estimate cycles needed
    base_rate = np.exp(baseline)
    n_cycles = max(5, min(50, int(target_events / (base_rate * LED_ON * 0.3 + 0.5))))
    log(f"  Using {n_cycles} cycles")
    
    # Generate latent factors
    np.random.seed(42)
    true_precision = np.random.randn(N_LARVAE)
    true_burstiness = np.random.randn(N_LARVAE)
    
    # Simulate all larvae
    t0 = time.time()
    X = np.zeros((N_LARVAE, 7))
    event_counts = np.zeros(N_LARVAE)
    
    for i in range(N_LARVAE):
        events, nc = simulate_larva(true_precision[i], true_burstiness[i], baseline, n_cycles, seed=i*1000)
        X[i] = compute_measures(events, nc)
        event_counts[i] = len(events)
        
        if (i + 1) % 100 == 0:
            log(f"  Simulated {i+1}/{N_LARVAE} larvae...")
    
    log(f"  Simulation complete: {time.time()-t0:.1f}s, mean events={np.mean(event_counts):.1f}")
    
    # Factor analysis
    X_clean = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    try:
        fa = FactorAnalysis(n_components=2, random_state=42)
        scores = fa.fit_transform(X_scaled)
        loadings = fa.components_
        log(f"  Factor loadings:")
        log(f"    F1: {loadings[0].round(2)}")
        log(f"    F2: {loadings[1].round(2)}")
    except:
        pca = PCA(n_components=2)
        scores = pca.fit_transform(X_scaled)
    
    # Match factors to true latents (check both orderings)
    c00 = np.abs(np.corrcoef(true_precision, scores[:,0])[0,1])
    c01 = np.abs(np.corrcoef(true_precision, scores[:,1])[0,1])
    c10 = np.abs(np.corrcoef(true_burstiness, scores[:,0])[0,1])
    c11 = np.abs(np.corrcoef(true_burstiness, scores[:,1])[0,1])
    
    # Best assignment
    if c00 + c11 > c01 + c10:
        corr_P, corr_B = c00, c11
    else:
        corr_P, corr_B = c01, c10
    
    # Also check direct correlations with key measures
    corr_P_direct = np.abs(np.corrcoef(true_precision, X[:,0])[0,1])  # ON/OFF ratio
    corr_B_direct = np.abs(np.corrcoef(true_burstiness, X[:,2])[0,1])  # IEI-CV
    
    log(f"  Factor correlations: Precision={corr_P:.3f}, Burstiness={corr_B:.3f}")
    log(f"  Direct correlations: Prec-ON/OFF={corr_P_direct:.3f}, Burst-IEI_CV={corr_B_direct:.3f}")
    
    # Power estimation
    def compute_power(true_f, recovered_f, n_reps=50):
        n = len(true_f)
        sig = 0
        for r in range(n_reps):
            np.random.seed(r * 7777)
            # Split by true factor (high vs low)
            median_f = np.median(true_f)
            high_idx = true_f > median_f
            low_idx = ~high_idx
            _, p = stats.ttest_ind(recovered_f[high_idx], recovered_f[low_idx])
            if p < 0.05:
                sig += 1
        return sig / n_reps * 100
    
    pow_P = compute_power(true_precision, scores[:,0] if c00 > c01 else scores[:,1])
    pow_B = compute_power(true_burstiness, scores[:,1] if c00 > c01 else scores[:,0])
    
    log(f"  Power: Precision={pow_P:.0f}%, Burstiness={pow_B:.0f}%")
    
    return {
        'baseline': baseline, 'target': target_events,
        'events': np.mean(event_counts),
        'corr_P': corr_P, 'corr_B': corr_B,
        'corr_P_direct': corr_P_direct, 'corr_B_direct': corr_B_direct,
        'pow_P': pow_P, 'pow_B': pow_B
    }

def main():
    log("Starting sweep...")
    results = []
    total = len(BASELINES) * len(TARGET_EVENTS)
    start = time.time()
    
    idx = 0
    for baseline in BASELINES:
        for target_events in TARGET_EVENTS:
            idx += 1
            result = run_condition(baseline, target_events, idx, total)
            results.append(result)
            
            elapsed = time.time() - start
            remain = elapsed / idx * (total - idx)
            log(f"  Elapsed: {elapsed:.0f}s, Remaining: {remain:.0f}s")
    
    log(f"\n{'='*70}")
    log(f"COMPLETE in {time.time()-start:.0f}s")
    log("="*70)
    
    # Summary tables
    log("\nCORRELATION TABLE (Factor Analysis):")
    log(f"{'Target':<10} " + " ".join(f"b={b:<8}" for b in BASELINES))
    log("PRECISION:")
    for t in TARGET_EVENTS:
        row = f"{t:<10} "
        for b in BASELINES:
            r = next((x for x in results if x['baseline']==b and x['target']==t), None)
            row += f"{r['corr_P']:<10.3f}" if r else ""
        log(row)
    log("BURSTINESS:")
    for t in TARGET_EVENTS:
        row = f"{t:<10} "
        for b in BASELINES:
            r = next((x for x in results if x['baseline']==b and x['target']==t), None)
            row += f"{r['corr_B']:<10.3f}" if r else ""
        log(row)
    
    log("\nDIRECT CORRELATIONS (True -> Measure):")
    log("PRECISION -> ON/OFF ratio:")
    for t in TARGET_EVENTS:
        row = f"{t:<10} "
        for b in BASELINES:
            r = next((x for x in results if x['baseline']==b and x['target']==t), None)
            row += f"{r['corr_P_direct']:<10.3f}" if r else ""
        log(row)
    log("BURSTINESS -> IEI-CV:")
    for t in TARGET_EVENTS:
        row = f"{t:<10} "
        for b in BASELINES:
            r = next((x for x in results if x['baseline']==b and x['target']==t), None)
            row += f"{r['corr_B_direct']:<10.3f}" if r else ""
        log(row)
    
    log("\nPOWER TABLE (%):")
    log(f"{'Target':<10} " + " ".join(f"b={b:<8}" for b in BASELINES))
    log("PRECISION:")
    for t in TARGET_EVENTS:
        row = f"{t:<10} "
        for b in BASELINES:
            r = next((x for x in results if x['baseline']==b and x['target']==t), None)
            row += f"{r['pow_P']:<10.0f}" if r else ""
        log(row)
    log("BURSTINESS:")
    for t in TARGET_EVENTS:
        row = f"{t:<10} "
        for b in BASELINES:
            r = next((x for x in results if x['baseline']==b and x['target']==t), None)
            row += f"{r['pow_B']:<10.0f}" if r else ""
        log(row)
    
    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = {-3.5: 'coral', -2.5: 'gold', -1.5: 'forestgreen'}
    
    plots = [
        ('corr_P', 'Precision Correlation (FA)', 0, 0),
        ('corr_B', 'Burstiness Correlation (FA)', 0, 1),
        ('corr_P_direct', 'Precision -> ON/OFF', 0, 2),
        ('corr_B_direct', 'Burstiness -> IEI-CV', 1, 0),
        ('pow_P', 'Precision Power (%)', 1, 1),
        ('pow_B', 'Burstiness Power (%)', 1, 2),
    ]
    
    for key, title, row, col in plots:
        ax = axes[row, col]
        for b in BASELINES:
            sub = [r for r in results if r['baseline'] == b]
            yvals = [r[key] for r in sub]
            ax.plot([r['target'] for r in sub], yvals, 'o-', color=colors[b], label=f'b={b}', lw=2, ms=8)
        
        if 'pow' in key:
            ax.axhline(80, color='red', ls='--', alpha=0.5, label='80% power')
            ax.set_ylim(0, 105)
            ax.set_ylabel('Power %')
        else:
            ax.axhline(0.7, color='red', ls='--', alpha=0.5, label='r=0.7')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Correlation')
        
        ax.set_xlabel('Target Events')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'composite_validation_results.png', dpi=150)
    log(f"\nFigure saved: {OUTPUT_DIR / 'composite_validation_results.png'}")
    log("DONE")

if __name__ == '__main__':
    main()

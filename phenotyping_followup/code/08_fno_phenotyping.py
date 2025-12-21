#!/usr/bin/env python3
"""
FNO-Based Phenotyping: Learn Event → Kernel Mapping

Uses Fourier Neural Operator to learn the mapping from sparse event patterns
to kernel parameters, bypassing traditional parametric fitting.

Architecture:
  Input:  PSTH (20 bins, 0-10s post-LED)
  Hidden: FNO layers with spectral convolution
  Output: Kernel parameters (τ₁, τ₂, A, B) or K(t) curve

Training:
  - Generate 2000 synthetic tracks with known kernels
  - Train FNO to recover kernel from events
  - Validate on held-out synthetic data
  - Apply to empirical tracks

Runtime: ~30 min (CPU), ~10 min (GPU)
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

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print(f"PyTorch version: {torch.__version__}")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")
    sys.exit(1)

# Paths
RESULTS_DIR = Path('/Users/gilraitses/INDYsim_project/scripts/2025-12-17/phenotyping_experiments/results')
EDA_DIR = RESULTS_DIR / 'deep_eda'
OUTPUT_DIR = RESULTS_DIR / 'fno_phenotyping'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint paths
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DATA_CHECKPOINT = CHECKPOINT_DIR / 'training_data.npz'
FNO_MODEL_CHECKPOINT = CHECKPOINT_DIR / 'fno_model_trained.pt'
MLP_MODEL_CHECKPOINT = CHECKPOINT_DIR / 'mlp_model_trained.pt'
SCALER_CHECKPOINT = CHECKPOINT_DIR / 'psth_scaler.npz'

# Model parameters
PSTH_BINS = 20  # Number of PSTH bins (input dimension)
KERNEL_POINTS = 60  # Points to evaluate K(t) on (output dimension)
KERNEL_T = np.linspace(0, 10, KERNEL_POINTS)  # Time grid for kernel

# Training parameters
N_TRAIN = 2000
N_VAL = 500
BATCH_SIZE = 64
N_EPOCHS = 100
LEARNING_RATE = 1e-3

# Simulation parameters
LED_CYCLE = 30.0
LED_ON_DURATION = 10.0
FIRST_LED_ONSET = 21.3
TRACK_DURATION = 1200.0
DT = 0.05  # 50ms frames


class SpectralConv1d(nn.Module):
    """1D Spectral Convolution layer for FNO."""
    
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to keep
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
    
    def forward(self, x):
        # x: (batch, channels, spatial)
        batchsize = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights
        )
        
        # Inverse FFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator for PSTH → Kernel mapping.
    
    Architecture:
      - Lift input to higher dimension
      - 4 FNO layers (spectral conv + pointwise MLP)
      - Project to output dimension
    """
    
    def __init__(self, in_dim, out_dim, hidden_dim=64, modes=12, n_layers=4):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.modes = modes
        self.n_layers = n_layers
        
        # Lifting layer
        self.lift = nn.Linear(1, hidden_dim)
        
        # FNO layers
        self.spectral_convs = nn.ModuleList([
            SpectralConv1d(hidden_dim, hidden_dim, modes) for _ in range(n_layers)
        ])
        self.pointwise_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, 1) for _ in range(n_layers)
        ])
        
        # Projection layers
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Final mapping to output grid
        self.output_map = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        # x: (batch, in_dim) - PSTH values
        batch_size = x.shape[0]
        
        # Reshape for convolution: (batch, 1, in_dim)
        x = x.unsqueeze(1).unsqueeze(-1)  # (batch, 1, in_dim, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (batch, in_dim, 1)
        
        # Lift to hidden dimension
        x = self.lift(x)  # (batch, in_dim, hidden_dim)
        x = x.permute(0, 2, 1)  # (batch, hidden_dim, in_dim)
        
        # FNO layers
        for i in range(self.n_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.pointwise_convs[i](x)
            x = x1 + x2
            if i < self.n_layers - 1:
                x = torch.nn.functional.gelu(x)
        
        # Project back
        x = x.permute(0, 2, 1)  # (batch, in_dim, hidden_dim)
        x = self.project(x)  # (batch, in_dim, 1)
        x = x.squeeze(-1)  # (batch, in_dim)
        
        # Map to output grid
        x = self.output_map(x)  # (batch, out_dim)
        
        return x


class SimpleEncoder(nn.Module):
    """
    Simpler MLP encoder as baseline comparison.
    """
    
    def __init__(self, in_dim, out_dim, hidden_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


def gamma_diff_kernel(t, tau1, tau2, A, B, alpha1=2.0, alpha2=2.0):
    """Compute gamma-difference kernel K(t)."""
    beta1 = tau1
    beta2 = tau2
    pdf1 = gamma_dist.pdf(t, alpha1, scale=beta1)
    pdf2 = gamma_dist.pdf(t, alpha2, scale=beta2)
    return A * np.nan_to_num(pdf1) - B * np.nan_to_num(pdf2)


def generate_events_discrete(kernel_params, baseline=-3.5, rng=None):
    """Generate events using discrete-time Bernoulli simulation."""
    if rng is None:
        rng = np.random.default_rng()
    
    tau1, tau2, A, B = kernel_params
    
    # LED onsets
    led_onsets = np.arange(FIRST_LED_ONSET, TRACK_DURATION, LED_CYCLE)
    
    events = []
    t = 0
    while t < TRACK_DURATION:
        # Find most recent LED onset
        onsets_before = led_onsets[led_onsets <= t]
        if len(onsets_before) > 0:
            t_since_onset = t - onsets_before[-1]
            if t_since_onset < LED_ON_DURATION:
                # LED is on
                K_t = gamma_diff_kernel(t_since_onset, tau1, tau2, A, B)
                log_hazard = baseline + K_t
            else:
                log_hazard = baseline
        else:
            log_hazard = baseline
        
        # Bernoulli draw
        p = np.exp(log_hazard)
        p = np.clip(p, 0, 1)
        
        if rng.random() < p:
            events.append(t)
        
        t += DT
    
    return np.array(events)


def compute_psth(events, n_bins=PSTH_BINS):
    """Compute PSTH from events."""
    bins = np.linspace(0, 10, n_bins + 1)
    led_onsets = np.arange(FIRST_LED_ONSET, TRACK_DURATION, LED_CYCLE)
    
    aligned_times = []
    for event_time in events:
        onsets_before = led_onsets[led_onsets <= event_time]
        if len(onsets_before) > 0:
            t_since = event_time - onsets_before[-1]
            if t_since < 10:
                aligned_times.append(t_since)
    
    psth, _ = np.histogram(aligned_times, bins=bins)
    
    # Normalize
    n_cycles = len(led_onsets)
    if n_cycles > 0:
        psth = psth / n_cycles
    
    return psth.astype(np.float32)


def generate_training_data(n_samples, param_ranges=None, rng=None):
    """Generate synthetic training data with varied kernel parameters."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    if param_ranges is None:
        param_ranges = {
            'tau1': (0.1, 1.0),
            'tau2': (2.0, 8.0),
            'A': (0.5, 3.0),
            'B': (5.0, 25.0)
        }
    
    psth_data = []
    kernel_data = []
    param_data = []
    
    for i in tqdm(range(n_samples), desc="Generating training data"):
        # Sample parameters
        tau1 = rng.uniform(*param_ranges['tau1'])
        tau2 = rng.uniform(*param_ranges['tau2'])
        A = rng.uniform(*param_ranges['A'])
        B = rng.uniform(*param_ranges['B'])
        params = (tau1, tau2, A, B)
        
        # Generate events
        events = generate_events_discrete(params, rng=rng)
        
        # Compute PSTH
        psth = compute_psth(events)
        
        # Compute kernel on grid
        kernel = gamma_diff_kernel(KERNEL_T, tau1, tau2, A, B)
        
        psth_data.append(psth)
        kernel_data.append(kernel)
        param_data.append(params)
    
    return (np.array(psth_data), 
            np.array(kernel_data, dtype=np.float32), 
            np.array(param_data, dtype=np.float32))


def train_model(model, train_loader, val_loader, n_epochs=N_EPOCHS, lr=LEARNING_RATE):
    """Train the FNO model."""
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses


def main():
    print("=" * 70)
    print("FNO-BASED PHENOTYPING (with checkpointing)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    
    rng = np.random.default_rng(42)
    
    # =========================================================================
    # 1. GENERATE OR LOAD TRAINING DATA
    # =========================================================================
    print(f"\n{'='*70}")
    print("1. TRAINING DATA")
    print(f"{'='*70}")
    
    if TRAIN_DATA_CHECKPOINT.exists():
        print(f"\n✓ Loading cached training data from {TRAIN_DATA_CHECKPOINT}")
        data = np.load(TRAIN_DATA_CHECKPOINT)
        train_psth = data['train_psth']
        train_kernel = data['train_kernel']
        train_params = data['train_params']
        val_psth = data['val_psth']
        val_kernel = data['val_kernel']
        val_params = data['val_params']
        print(f"  Loaded {len(train_psth)} training + {len(val_psth)} validation samples")
    else:
        print(f"\nGenerating {N_TRAIN} training + {N_VAL} validation samples...")
        
        # Training data
        train_psth, train_kernel, train_params = generate_training_data(N_TRAIN, rng=rng)
        
        # Validation data  
        val_psth, val_kernel, val_params = generate_training_data(N_VAL, rng=rng)
        
        # Save checkpoint
        np.savez(TRAIN_DATA_CHECKPOINT,
                 train_psth=train_psth, train_kernel=train_kernel, train_params=train_params,
                 val_psth=val_psth, val_kernel=val_kernel, val_params=val_params)
        print(f"✓ Saved training data checkpoint to {TRAIN_DATA_CHECKPOINT}")
    
    print(f"Training PSTH shape: {train_psth.shape}")
    print(f"Training kernel shape: {train_kernel.shape}")
    
    # Normalize PSTH
    psth_scaler = StandardScaler()
    train_psth_scaled = psth_scaler.fit_transform(train_psth)
    val_psth_scaled = psth_scaler.transform(val_psth)
    
    # Save scaler parameters
    np.savez(SCALER_CHECKPOINT, mean=psth_scaler.mean_, scale=psth_scaler.scale_)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_psth_scaled),
        torch.FloatTensor(train_kernel)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_psth_scaled),
        torch.FloatTensor(val_kernel)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # =========================================================================
    # 2. TRAIN OR LOAD FNO MODEL
    # =========================================================================
    print(f"\n{'='*70}")
    print("2. FNO MODEL")
    print(f"{'='*70}")
    
    # FNO model
    fno_model = FNO1d(
        in_dim=PSTH_BINS,
        out_dim=KERNEL_POINTS,
        hidden_dim=64,
        modes=8,
        n_layers=4
    )
    
    if FNO_MODEL_CHECKPOINT.exists():
        print(f"\n✓ Loading trained FNO model from {FNO_MODEL_CHECKPOINT}")
        fno_model.load_state_dict(torch.load(FNO_MODEL_CHECKPOINT, map_location=DEVICE))
        fno_model = fno_model.to(DEVICE)
        fno_train_losses, fno_val_losses = [], []
    else:
        print(f"\nFNO Architecture:")
        print(f"  Input: {PSTH_BINS} (PSTH bins)")
        print(f"  Output: {KERNEL_POINTS} (kernel grid points)")
        print(f"  Hidden dim: 64")
        print(f"  Fourier modes: 8")
        print(f"  Layers: 4")
        
        n_params = sum(p.numel() for p in fno_model.parameters())
        print(f"  Total parameters: {n_params:,}")
        
        print(f"\nTraining for {N_EPOCHS} epochs...")
        fno_model, fno_train_losses, fno_val_losses = train_model(
            fno_model, train_loader, val_loader, n_epochs=N_EPOCHS
        )
        
        # Save checkpoint
        torch.save(fno_model.state_dict(), FNO_MODEL_CHECKPOINT)
        print(f"✓ Saved FNO model checkpoint to {FNO_MODEL_CHECKPOINT}")
    
    # =========================================================================
    # 3. TRAIN OR LOAD BASELINE MLP
    # =========================================================================
    print(f"\n{'='*70}")
    print("3. BASELINE MLP")
    print(f"{'='*70}")
    
    mlp_model = SimpleEncoder(PSTH_BINS, KERNEL_POINTS, hidden_dim=128)
    
    if MLP_MODEL_CHECKPOINT.exists():
        print(f"\n✓ Loading trained MLP model from {MLP_MODEL_CHECKPOINT}")
        mlp_model.load_state_dict(torch.load(MLP_MODEL_CHECKPOINT, map_location=DEVICE))
        mlp_model = mlp_model.to(DEVICE)
        mlp_train_losses, mlp_val_losses = [], []
    else:
        print(f"\nMLP Architecture:")
        n_params_mlp = sum(p.numel() for p in mlp_model.parameters())
        print(f"  Total parameters: {n_params_mlp:,}")
        
        print(f"\nTraining for {N_EPOCHS} epochs...")
        mlp_model, mlp_train_losses, mlp_val_losses = train_model(
            mlp_model, train_loader, val_loader, n_epochs=N_EPOCHS
        )
        
        # Save checkpoint
        torch.save(mlp_model.state_dict(), MLP_MODEL_CHECKPOINT)
        print(f"✓ Saved MLP model checkpoint to {MLP_MODEL_CHECKPOINT}")
    
    # =========================================================================
    # 4. EVALUATE ON VALIDATION SET
    # =========================================================================
    print(f"\n{'='*70}")
    print("4. VALIDATION RESULTS")
    print(f"{'='*70}")
    
    fno_model.eval()
    mlp_model.eval()
    
    with torch.no_grad():
        val_x = torch.FloatTensor(val_psth_scaled).to(DEVICE)
        
        fno_pred = fno_model(val_x).cpu().numpy()
        mlp_pred = mlp_model(val_x).cpu().numpy()
    
    # Compute correlations
    fno_corrs = []
    mlp_corrs = []
    for i in range(len(val_kernel)):
        fno_r = np.corrcoef(fno_pred[i], val_kernel[i])[0, 1]
        mlp_r = np.corrcoef(mlp_pred[i], val_kernel[i])[0, 1]
        fno_corrs.append(fno_r)
        mlp_corrs.append(mlp_r)
    
    print(f"\nKernel recovery correlation:")
    print(f"  FNO: r = {np.mean(fno_corrs):.3f} ± {np.std(fno_corrs):.3f}")
    print(f"  MLP: r = {np.mean(mlp_corrs):.3f} ± {np.std(mlp_corrs):.3f}")
    
    # MSE
    fno_mse = np.mean((fno_pred - val_kernel) ** 2)
    mlp_mse = np.mean((mlp_pred - val_kernel) ** 2)
    print(f"\nMSE:")
    print(f"  FNO: {fno_mse:.6f}")
    print(f"  MLP: {mlp_mse:.6f}")
    
    # =========================================================================
    # 5. APPLY TO EMPIRICAL DATA
    # =========================================================================
    print(f"\n{'='*70}")
    print("5. APPLYING TO EMPIRICAL DATA")
    print(f"{'='*70}")
    
    # Load empirical PSTH
    psth_matrix = np.load(EDA_DIR / 'psth_matrix.npy')
    print(f"Loaded {len(psth_matrix)} empirical PSTH vectors")
    
    # Normalize using training scaler
    psth_scaled = psth_scaler.transform(psth_matrix)
    
    # Predict kernels
    with torch.no_grad():
        emp_x = torch.FloatTensor(psth_scaled).to(DEVICE)
        emp_kernels_fno = fno_model(emp_x).cpu().numpy()
        emp_kernels_mlp = mlp_model(emp_x).cpu().numpy()
    
    print(f"Predicted kernel shapes: {emp_kernels_fno.shape}")
    
    # =========================================================================
    # 6. CLUSTER LEARNED KERNELS
    # =========================================================================
    print(f"\n{'='*70}")
    print("6. CLUSTERING LEARNED KERNELS")
    print(f"{'='*70}")
    
    # Standardize for clustering
    kernel_scaler = StandardScaler()
    emp_kernels_scaled = kernel_scaler.fit_transform(emp_kernels_fno)
    
    print(f"\n{'Method':<25} {'k':<5} {'Silhouette':<12}")
    print("-" * 45)
    
    results = {
        'fno_clustering': {},
        'comparison_to_psth': {},
        'validation': {
            'fno_kernel_corr': float(np.mean(fno_corrs)),
            'mlp_kernel_corr': float(np.mean(mlp_corrs)),
            'fno_mse': float(fno_mse),
            'mlp_mse': float(mlp_mse)
        }
    }
    
    # Load PSTH clusters for comparison
    psth_pcs = np.load(EDA_DIR / 'psth_pcs.npy')
    
    for k in [3, 4, 5]:
        # Cluster FNO kernels
        fno_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(emp_kernels_scaled)
        fno_sil = silhouette_score(emp_kernels_scaled, fno_labels)
        
        # Cluster PSTH PCs for comparison
        psth_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(psth_pcs[:, :3])
        
        # Agreement
        ari_vs_psth = adjusted_rand_score(fno_labels, psth_labels)
        
        print(f"FNO kernels              k={k}   {fno_sil:.3f}        (ARI vs PSTH: {ari_vs_psth:.3f})")
        
        results['fno_clustering'][k] = {
            'silhouette': round(fno_sil, 4),
            'ari_vs_psth': round(ari_vs_psth, 4)
        }
    
    # =========================================================================
    # 7. COMPARE FNO KERNELS TO FITTED KERNELS
    # =========================================================================
    print(f"\n{'='*70}")
    print("7. FNO vs PARAMETRIC FITTING")
    print(f"{'='*70}")
    
    # Load parametric fits
    fits_path = RESULTS_DIR / 'empirical_10min_kernel_fits_v2.csv'
    if fits_path.exists():
        fits_df = pd.read_csv(fits_path)
        
        # Use only the first N tracks that match (PSTH may have fewer due to filtering)
        n_common = min(len(emp_kernels_fno), len(fits_df))
        print(f"\nComparing {n_common} tracks (FNO: {len(emp_kernels_fno)}, Parametric: {len(fits_df)})")
        
        # Compute kernel curves from parametric fits (only first n_common)
        param_kernels = []
        for i, (_, row) in enumerate(fits_df.iterrows()):
            if i >= n_common:
                break
            k = gamma_diff_kernel(KERNEL_T, row['tau1'], row['tau2'], row['A'], row['B'])
            param_kernels.append(k)
        param_kernels = np.array(param_kernels)
        
        # Use only first n_common FNO kernels
        fno_kernels_subset = emp_kernels_fno[:n_common]
        fno_scaled_subset = emp_kernels_scaled[:n_common]
        
        # Correlation between FNO and parametric kernels
        kernel_corrs = []
        for i in range(n_common):
            r = np.corrcoef(fno_kernels_subset[i], param_kernels[i])[0, 1]
            if not np.isnan(r):
                kernel_corrs.append(r)
        
        print(f"\nCorrelation between FNO and parametric kernels:")
        print(f"  Mean r = {np.mean(kernel_corrs):.3f} ± {np.std(kernel_corrs):.3f}")
        
        results['fno_vs_parametric'] = {
            'mean_correlation': round(float(np.mean(kernel_corrs)), 4),
            'std_correlation': round(float(np.std(kernel_corrs)), 4),
            'n_compared': n_common
        }
        
        # Cluster parametric kernels for comparison
        param_kernels_scaled = kernel_scaler.transform(param_kernels)
        
        for k in [4]:
            fno_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(fno_scaled_subset)
            param_labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(param_kernels_scaled)
            ari = adjusted_rand_score(fno_labels, param_labels)
            print(f"\nCluster agreement (k={k}): ARI = {ari:.3f}")
            results['fno_vs_parametric']['cluster_ari_k4'] = round(ari, 4)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("FNO PHENOTYPING SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n✓ Trained FNO on {N_TRAIN} synthetic tracks")
    print(f"✓ Validation kernel recovery: r = {np.mean(fno_corrs):.3f}")
    print(f"✓ Applied to {len(psth_matrix)} empirical tracks")
    
    best_k = max(results['fno_clustering'].keys(), 
                 key=lambda k: results['fno_clustering'][k]['silhouette'])
    best_sil = results['fno_clustering'][best_k]['silhouette']
    best_ari = results['fno_clustering'][best_k]['ari_vs_psth']
    
    print(f"\nBest clustering: k={best_k}")
    print(f"  Silhouette: {best_sil:.3f}")
    print(f"  ARI vs PSTH: {best_ari:.3f}")
    
    if best_ari > 0.3:
        print(f"\n✓ FNO clusters AGREE with PSTH clusters")
        print(f"  → Learned kernels capture behavioral patterns")
    else:
        print(f"\n⚠ FNO clusters still DISAGREE with PSTH")
        print(f"  → The phenotypic signal may be weak or absent")
    
    # Save results
    results['training'] = {
        'n_train': N_TRAIN,
        'n_val': N_VAL,
        'n_epochs': N_EPOCHS,
        'final_train_loss': float(fno_train_losses[-1]),
        'final_val_loss': float(fno_val_losses[-1])
    }
    
    output_path = OUTPUT_DIR / 'fno_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save learned kernels
    np.save(OUTPUT_DIR / 'empirical_fno_kernels.npy', emp_kernels_fno)
    
    # Save model
    torch.save(fno_model.state_dict(), OUTPUT_DIR / 'fno_model.pt')
    
    print(f"\nResults saved to: {output_path}")
    print(f"FNO model saved to: {OUTPUT_DIR / 'fno_model.pt'}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return results


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
"""
Event Generator for Larval Behavior Simulation

Implements event generation algorithms for simulating reorientation events:
1. Inversion-based sampling (recommended for low-rate events)
2. Thinning algorithm (fallback for highly time-varying rates)

Based on MiroThinker specification (2025-12-10):
- Low event frequency (λ ≈ 0.01-0.1 events/s) makes thinning inefficient
- Inversion-based sampling via cumulative hazard is more efficient
- Direct draw from fitted hazard by integrating λ(t)

References:
- Lewis & Shedler (1979) - Thinning for inhomogeneous Poisson
- Kelton, Law & Kelton (2024) - Simulation Modeling & Analysis
"""

import numpy as np
from typing import Callable, List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SimulatedEvent:
    """Container for a simulated behavioral event."""
    time: float
    event_type: str  # 'reorientation', 'reversal', 'head_swing'
    heading_change: float  # reo_dtheta (radians)
    n_head_swings: int  # numHS
    duration: float  # event duration (seconds)


# =============================================================================
# INVERSION-BASED EVENT GENERATOR (Recommended)
# =============================================================================

class InversionEventGenerator:
    """
    Generate events by inverting the cumulative hazard function.
    
    More efficient than thinning for low-rate events (MiroThinker recommendation).
    
    Algorithm:
    1. Build dense vector of λ(t) at fine time steps
    2. Compute cumulative hazard H(t) = ∫₀ᵗ λ(s) ds
    3. Draw uniform U ~ [0,1] and solve H(τ) = -log(1-U) for event time τ
    4. Use binary search to find τ
    """
    
    def __init__(
        self,
        hazard_func: Callable[[np.ndarray], np.ndarray],
        t_start: float = 0.0,
        t_end: float = 1200.0,  # 20-minute experiment
        dt: float = 0.05  # 50ms resolution
    ):
        """
        Initialize generator with hazard function.
        
        Parameters
        ----------
        hazard_func : callable
            Function that takes time array and returns hazard rates λ(t)
        t_start : float
            Simulation start time (seconds)
        t_end : float
            Simulation end time (seconds)
        dt : float
            Time resolution for hazard computation (seconds)
        """
        self.hazard_func = hazard_func
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        
        # Precompute hazard and cumulative hazard on grid
        self._precompute_hazard()
    
    def _precompute_hazard(self):
        """Precompute hazard and cumulative hazard on time grid."""
        # Dense time grid
        self.t_grid = np.arange(self.t_start, self.t_end + self.dt, self.dt)
        n_points = len(self.t_grid)
        
        # Compute hazard at each point
        self.lambda_grid = self.hazard_func(self.t_grid)
        
        # Ensure non-negative
        self.lambda_grid = np.maximum(self.lambda_grid, 0)
        
        # Compute cumulative hazard using trapezoidal integration
        # H(t) = ∫₀ᵗ λ(s) ds
        self.H_grid = np.zeros(n_points)
        self.H_grid[1:] = np.cumsum(
            0.5 * (self.lambda_grid[:-1] + self.lambda_grid[1:]) * self.dt
        )
    
    def sample_next_event_time(self, current_time: float, rng: np.random.Generator = None) -> Optional[float]:
        """
        Sample the next event time given current time.
        
        Uses inversion method: solve H(τ) - H(t₀) = -log(1-U) for τ.
        
        Parameters
        ----------
        current_time : float
            Current simulation time
        rng : np.random.Generator
            Random number generator (uses numpy default if None)
        
        Returns
        -------
        event_time : float or None
            Next event time, or None if no event before t_end
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Get cumulative hazard at current time
        if current_time >= self.t_end:
            return None
        
        # Find H(current_time) by interpolation
        idx = np.searchsorted(self.t_grid, current_time)
        if idx == 0:
            H_current = 0.0
        elif idx >= len(self.t_grid):
            return None
        else:
            # Linear interpolation
            t0, t1 = self.t_grid[idx-1], self.t_grid[idx]
            H0, H1 = self.H_grid[idx-1], self.H_grid[idx]
            frac = (current_time - t0) / (t1 - t0)
            H_current = H0 + frac * (H1 - H0)
        
        # Draw target cumulative hazard: H_target = H_current - log(1-U)
        U = rng.uniform()
        H_target = H_current - np.log(1 - U)
        
        # Check if event occurs before end
        if H_target > self.H_grid[-1]:
            return None
        
        # Binary search for event time
        idx = np.searchsorted(self.H_grid, H_target)
        if idx >= len(self.t_grid):
            return None
        
        # Linear interpolation for precise time
        if idx > 0:
            H0, H1 = self.H_grid[idx-1], self.H_grid[idx]
            t0, t1 = self.t_grid[idx-1], self.t_grid[idx]
            if H1 > H0:
                frac = (H_target - H0) / (H1 - H0)
                event_time = t0 + frac * (t1 - t0)
            else:
                event_time = t0
        else:
            event_time = self.t_grid[0]
        
        return event_time if event_time <= self.t_end else None
    
    def generate_events(self, rng: np.random.Generator = None) -> List[float]:
        """
        Generate all event times for the simulation period.
        
        Parameters
        ----------
        rng : np.random.Generator
            Random number generator
        
        Returns
        -------
        event_times : list
            List of event times (sorted)
        """
        if rng is None:
            rng = np.random.default_rng()
        
        events = []
        current_time = self.t_start
        
        while current_time < self.t_end:
            next_time = self.sample_next_event_time(current_time, rng)
            if next_time is None:
                break
            events.append(next_time)
            current_time = next_time
        
        return events


# =============================================================================
# THINNING ALGORITHM (Fallback)
# =============================================================================

class ThinningEventGenerator:
    """
    Generate events using Lewis-Shedler thinning algorithm.
    
    Less efficient for low-rate events but handles highly time-varying rates.
    
    Algorithm:
    1. Find λ_max = max(λ(t)) over simulation period
    2. Generate candidate events from homogeneous Poisson(λ_max)
    3. Accept each candidate with probability λ(t)/λ_max
    """
    
    def __init__(
        self,
        hazard_func: Callable[[np.ndarray], np.ndarray],
        t_start: float = 0.0,
        t_end: float = 1200.0,
        lambda_max: Optional[float] = None
    ):
        """
        Initialize thinning generator.
        
        Parameters
        ----------
        hazard_func : callable
            Function that takes time and returns hazard rate
        t_start : float
            Simulation start time
        t_end : float
            Simulation end time
        lambda_max : float, optional
            Upper bound on hazard rate. If None, estimated from grid.
        """
        self.hazard_func = hazard_func
        self.t_start = t_start
        self.t_end = t_end
        
        # Estimate lambda_max if not provided
        if lambda_max is None:
            t_test = np.linspace(t_start, t_end, 10000)
            self.lambda_max = np.max(self.hazard_func(t_test)) * 1.1  # 10% buffer
        else:
            self.lambda_max = lambda_max
    
    def generate_events(self, rng: np.random.Generator = None) -> List[float]:
        """
        Generate events using thinning.
        
        Parameters
        ----------
        rng : np.random.Generator
            Random number generator
        
        Returns
        -------
        event_times : list
            List of accepted event times
        """
        if rng is None:
            rng = np.random.default_rng()
        
        events = []
        current_time = self.t_start
        
        while current_time < self.t_end:
            # Generate candidate from homogeneous Poisson
            wait = rng.exponential(1.0 / self.lambda_max)
            candidate_time = current_time + wait
            
            if candidate_time >= self.t_end:
                break
            
            # Accept/reject
            lambda_t = self.hazard_func(np.array([candidate_time]))[0]
            accept_prob = lambda_t / self.lambda_max
            
            if rng.uniform() < accept_prob:
                events.append(candidate_time)
            
            current_time = candidate_time
        
        return events


# =============================================================================
# EVENT ATTRIBUTE SAMPLERS
# =============================================================================

def sample_heading_change(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """
    Sample heading changes (reo_dtheta) from empirical distribution.
    
    Based on typical larval reorientation angle distribution:
    - Approximately Gaussian with mean ~0 and std ~60 degrees
    - Wrapped to [-π, π]
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    n : int
        Number of samples
    
    Returns
    -------
    dtheta : ndarray
        Heading changes in radians
    """
    # Empirical parameters (from MAGAT data)
    mean_dtheta = 0.0  # Unbiased left/right
    std_dtheta = np.deg2rad(60)  # ~60 degree standard deviation
    
    dtheta = rng.normal(mean_dtheta, std_dtheta, n)
    
    # Wrap to [-π, π]
    dtheta = np.mod(dtheta + np.pi, 2 * np.pi) - np.pi
    
    return dtheta


def sample_num_head_swings(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """
    Sample number of head swings per reorientation.
    
    Based on empirical distribution (MiroThinker spec):
    - Most reorientations have 0-5 head swings
    - Modal value is 1-2
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    n : int
        Number of samples
    
    Returns
    -------
    num_hs : ndarray
        Number of head swings (integers)
    """
    # Empirical probabilities for numHS = 0, 1, 2, 3, 4, 5+
    probs = np.array([0.15, 0.35, 0.25, 0.15, 0.07, 0.03])
    probs = probs / probs.sum()  # Normalize
    
    num_hs = rng.choice(len(probs), size=n, p=probs)
    return num_hs


def sample_run_speed(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """
    Sample forward run speeds from empirical distribution.
    
    Based on MAGAT data:
    - Log-normal distribution
    - Mean ~0.024 cm/s (0.24 mm/s)
    - Only positive values (forward movement)
    
    Parameters
    ----------
    rng : np.random.Generator
        Random number generator
    n : int
        Number of samples
    
    Returns
    -------
    speed : ndarray
        Forward speeds in cm/s
    """
    # Log-normal parameters (fit to empirical data)
    mu_log = np.log(0.024)  # Log of mean speed
    sigma_log = 0.5  # Standard deviation in log space
    
    speed = rng.lognormal(mu_log, sigma_log, n)
    
    # Clip to reasonable range
    speed = np.clip(speed, 0.001, 0.5)  # 0.01 to 5 mm/s
    
    return speed


# =============================================================================
# FULL SIMULATION
# =============================================================================

def simulate_larva_trajectory(
    hazard_func: Callable[[np.ndarray], np.ndarray],
    t_end: float = 1200.0,
    dt: float = 0.1,
    initial_pos: Tuple[float, float] = (32.0, 32.0),  # Center of 64mm arena
    initial_heading: float = 0.0,
    arena_size: float = 64.0,  # mm
    seed: Optional[int] = None,
    use_inversion: bool = True
) -> Dict:
    """
    Simulate a complete larva trajectory.
    
    Parameters
    ----------
    hazard_func : callable
        Hazard function for reorientation events
    t_end : float
        Simulation duration (seconds)
    dt : float
        Time step for trajectory integration
    initial_pos : tuple
        Starting (x, y) position in mm
    initial_heading : float
        Starting heading in radians
    arena_size : float
        Arena size (mm) for boundary reflection
    seed : int, optional
        Random seed
    use_inversion : bool
        Use inversion-based generator (recommended)
    
    Returns
    -------
    trajectory : dict
        - time: time points
        - x, y: positions
        - heading: headings
        - speed: speeds
        - events: list of SimulatedEvent
    """
    rng = np.random.default_rng(seed)
    
    # Generate event times
    if use_inversion:
        generator = InversionEventGenerator(hazard_func, t_start=0.0, t_end=t_end)
    else:
        generator = ThinningEventGenerator(hazard_func, t_start=0.0, t_end=t_end)
    
    event_times = generator.generate_events(rng)
    
    # Generate event attributes
    n_events = len(event_times)
    heading_changes = sample_heading_change(rng, n_events)
    num_head_swings = sample_num_head_swings(rng, n_events)
    
    # Create event objects
    events = []
    for i, t in enumerate(event_times):
        events.append(SimulatedEvent(
            time=t,
            event_type='reorientation',
            heading_change=heading_changes[i],
            n_head_swings=int(num_head_swings[i]),
            duration=0.5 + rng.exponential(0.5)  # 0.5-2s typical duration
        ))
    
    # Integrate trajectory
    n_steps = int(t_end / dt) + 1
    time = np.linspace(0, t_end, n_steps)
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)
    heading = np.zeros(n_steps)
    speed = np.zeros(n_steps)
    
    x[0], y[0] = initial_pos
    heading[0] = initial_heading
    
    event_idx = 0
    
    for i in range(1, n_steps):
        t = time[i]
        
        # Check for events
        while event_idx < n_events and events[event_idx].time <= t:
            # Apply heading change
            heading[i-1] += events[event_idx].heading_change
            event_idx += 1
        
        # Sample speed for this step
        current_speed = sample_run_speed(rng, 1)[0]
        speed[i] = current_speed
        
        # Update position
        heading[i] = heading[i-1]
        dx = current_speed * np.cos(heading[i]) * dt * 10  # Convert cm/s to mm
        dy = current_speed * np.sin(heading[i]) * dt * 10
        
        x[i] = x[i-1] + dx
        y[i] = y[i-1] + dy
        
        # Boundary reflection
        if x[i] < 0:
            x[i] = -x[i]
            heading[i] = np.pi - heading[i]
        elif x[i] > arena_size:
            x[i] = 2 * arena_size - x[i]
            heading[i] = np.pi - heading[i]
        
        if y[i] < 0:
            y[i] = -y[i]
            heading[i] = -heading[i]
        elif y[i] > arena_size:
            y[i] = 2 * arena_size - y[i]
            heading[i] = -heading[i]
        
        # Wrap heading
        heading[i] = np.mod(heading[i] + np.pi, 2 * np.pi) - np.pi
    
    return {
        'time': time,
        'x': x,
        'y': y,
        'heading': heading,
        'speed': speed,
        'events': events,
        'n_events': n_events
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Example hazard function (constant + stimulus response)
    def example_hazard(t):
        """Example hazard: baseline + stimulus-locked increase."""
        baseline = 0.02  # 0.02 events/s baseline
        
        # Simulate 30s on / 30s off stimulus cycle
        cycle_period = 60.0
        phase = (t % cycle_period) / cycle_period
        
        # Increase rate during first 10s of each cycle (stimulus on)
        stimulus_effect = np.where(phase < 10/60, 0.05, 0.0)
        
        return baseline + stimulus_effect
    
    # Generate events
    print("Generating events with inversion method...")
    generator = InversionEventGenerator(example_hazard, t_start=0, t_end=1200, dt=0.05)
    events = generator.generate_events(np.random.default_rng(42))
    print(f"  Generated {len(events)} events")
    print(f"  Rate: {len(events) / 20:.2f} events/min")
    
    # Simulate full trajectory
    print("\nSimulating full trajectory...")
    trajectory = simulate_larva_trajectory(
        example_hazard,
        t_end=1200,
        dt=0.1,
        seed=42
    )
    print(f"  {trajectory['n_events']} reorientations")
    print(f"  Final position: ({trajectory['x'][-1]:.1f}, {trajectory['y'][-1]:.1f}) mm")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trajectory
    ax = axes[0]
    ax.plot(trajectory['x'], trajectory['y'], 'b-', alpha=0.5, linewidth=0.5)
    ax.plot(trajectory['x'][0], trajectory['y'][0], 'go', markersize=10, label='Start')
    ax.plot(trajectory['x'][-1], trajectory['y'][-1], 'ro', markersize=10, label='End')
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 64)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Simulated Trajectory')
    ax.legend()
    ax.set_aspect('equal')
    
    # Event rate histogram
    ax = axes[1]
    event_times = [e.time for e in trajectory['events']]
    ax.hist(event_times, bins=60, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Event count')
    ax.set_title('Event Times (should cluster in stimulus windows)')
    
    plt.tight_layout()
    plt.savefig('simulated_trajectory.png', dpi=150)
    print("\nSaved plot to simulated_trajectory.png")

#!/usr/bin/env python3
"""
MATLAB method functions for period-relative timing.

Recreated from deleted adapt_matlab_turnrate_method.py
"""

import numpy as np
from typing import Tuple


def compute_led12Val(led1_values: np.ndarray, led2_values: np.ndarray) -> np.ndarray:
    """
    Compute led12Val from led1Val and led2Val (MATLAB line 78).
    
    MATLAB: ydata = GQled1Val.yData + GQled2Val.yData
    Also: ydata(1:index) = ydata(1:index) + 60 where index is last zero of led1Val
    
    Parameters
    ----------
    led1_values : ndarray
        LED1 intensity values
    led2_values : ndarray
        LED2 intensity values
    
    Returns
    -------
    ndarray
        Combined LED values (led12Val)
    """
    led12Val = led1_values + led2_values
    
    # Find last zero of led1Val (MATLAB line 79)
    zero_indices = np.where(led1_values == 0)[0]
    if len(zero_indices) > 0:
        index = zero_indices[-1]
        # Add 60 to values before index (MATLAB line 80)
        led12Val[:index+1] = led12Val[:index+1] + 60
    
    return led12Val


def add_ton_toff_matlab(
    led12Val: np.ndarray,
    time_axis: np.ndarray,
    tperiod: float = 10.0,
    method: str = 'square'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    MATLAB addTonToff equivalent: Create time values relative to period start.
    
    MATLAB: addTonToff('led12Val', 'square', 'period', tperiod)
    Creates led12Val_ton and led12Val_toff with time values in [0, tperiod] range.
    
    For each time point, calculates:
    - led12Val_ton: Time relative to LED ON (period start)
    - led12Val_toff: Time relative to LED OFF (period end/next ON)
    
    Parameters
    ----------
    led12Val : ndarray
        Combined LED values
    time_axis : ndarray
        Time axis for LED values
    tperiod : float
        Period length (default 10 seconds, but should be detected from data)
    method : str
        Method: 'square' for square wave detection
    
    Returns
    -------
    tuple
        (led12Val_ton, led12Val_toff)
        - led12Val_ton: Time relative to LED ON for each point (modulo tperiod)
        - led12Val_toff: Time relative to LED OFF for each point (modulo tperiod)
    """
    # Detect LED ON/OFF transitions
    threshold = np.max(led12Val) * 0.1  # 10% threshold
    is_on = led12Val > threshold
    
    # Find ON/OFF transitions
    on_transitions = np.where(np.diff(is_on.astype(int)) > 0)[0] + 1  # OFF->ON
    off_transitions = np.where(np.diff(is_on.astype(int)) < 0)[0] + 1  # ON->OFF
    
    # Initialize arrays
    led12Val_ton = np.full(len(time_axis), np.nan)
    led12Val_toff = np.full(len(time_axis), np.nan)
    
    # For each time point, calculate time relative to period start
    for i, t in enumerate(time_axis):
        # Find most recent period start (LED ON)
        period_start = np.mod(t, tperiod)
        led12Val_ton[i] = period_start
        
        # Find most recent LED OFF (for toff)
        # Look for OFF transition before this time point
        off_before = off_transitions[off_transitions <= i]
        if len(off_before) > 0:
            last_off_idx = off_before[-1]
            last_off_time = time_axis[last_off_idx]
            # Time relative to LED OFF
            time_since_off = t - last_off_time
            led12Val_toff[i] = np.mod(time_since_off, tperiod)
        else:
            # No OFF found, use period start
            led12Val_toff[i] = period_start
    
    return led12Val_ton, led12Val_toff


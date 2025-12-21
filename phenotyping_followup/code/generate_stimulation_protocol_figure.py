#!/usr/bin/env python3
"""
Generate stimulation protocol figure showing LED timing for each design.
Shows full 30-second cycle (10s ON window + 20s OFF) with:
- Red LED (stimulus) signal
- Blue LED (constant or cycling)
- Annotations for statistical power and limitations
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
from pathlib import Path

# Cinnamoroll color palette
sys.path.insert(0, str(Path(__file__).parent))
from cinnamoroll_palette import COLORS

# Typography
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 10

# Design specifications (within 10s ON window)
DESIGNS = {
    'Current: Continuous 10s': {
        'red_pulses': [(0, 10)],  # Single continuous pulse
        'duty_cycle': '100%',
        'total_on': '10s',
        'power': '42%',
        'events': '1.9/track',
        'rmse': '0.108s',
        'note': 'Current protocol. Low power, high bias.'
    },
    'Recommended: Burst 10x0.5s': {
        'red_pulses': [(i, i+0.5) for i in np.arange(0, 10, 1)],  # 0.5s ON, 0.5s OFF
        'duty_cycle': '50%',
        'total_on': '5s',
        'power': '100%',
        'events': '14.9/track',
        'rmse': '0.036s',
        'note': '8x more events, 3x lower error.'
    },
    'Alternative: Medium 4x1s': {
        'red_pulses': [(i*2.5, i*2.5+1) for i in range(4)],  # 1s ON, 1.5s OFF
        'duty_cycle': '40%',
        'total_on': '4s',
        'power': '100%',
        'events': '6.0/track',
        'rmse': '0.048s',
        'note': 'Moderate improvement over continuous.'
    },
    'Alternative: Long 2x2s': {
        'red_pulses': [(0, 2), (5, 7)],  # 2s ON, 3s OFF
        'duty_cycle': '40%',
        'total_on': '4s',
        'power': '100%',
        'events': '3.1/track',
        'rmse': '0.077s',
        'note': 'Fewer but longer pulses.'
    },
}

# Blue LED options
BLUE_OPTIONS = {
    'Constant 7 PWM': lambda t: np.ones_like(t) * 7,
    'Cycling 5-15 PWM': lambda t: 10 + 5 * np.sin(2 * np.pi * t / 30)  # Sync with 30s cycle
}

def create_figure():
    fig = plt.figure(figsize=(14, 12))
    
    # Create grid for 4 designs
    n_designs = len(DESIGNS)
    
    for idx, (design_name, design) in enumerate(DESIGNS.items()):
        ax = fig.add_subplot(n_designs, 1, idx + 1)
        
        # Time axis: full 30-second cycle
        t = np.linspace(0, 30, 3000)
        
        # Create red LED signal
        red_signal = np.zeros_like(t)
        for start, end in design['red_pulses']:
            red_signal[(t >= start) & (t < end)] = 1
        
        # Plot red LED (stimulus)
        ax.fill_between(t, red_signal * 0.9, 0, color=COLORS['failure'], alpha=0.8, step='post', label='Red LED (Stimulus)')
        ax.step(t, red_signal * 0.9, where='post', color=COLORS['failure_dark'], linewidth=1.5)
        
        # Plot blue LED (constant 7 PWM, shown as line below)
        blue_constant = BLUE_OPTIONS['Constant 7 PWM'](t) / 15  # Normalize to 0-1 range
        ax.plot(t, -blue_constant * 0.3 - 0.1, color=COLORS['primary'], linewidth=2, label='Blue LED (7 PWM)')
        
        # Mark ON window (0-10s) and OFF window (10-30s)
        ax.axvspan(0, 10, alpha=0.1, color=COLORS['primary_light'], zorder=0)
        ax.axvspan(10, 30, alpha=0.1, color=COLORS['grid'], zorder=0)
        
        # Add annotations
        ax.text(5, 1.15, 'LED ON Window', ha='center', fontsize=10, fontweight='bold', color=COLORS['failure_dark'])
        ax.text(20, 1.15, 'LED OFF Window', ha='center', fontsize=10, fontweight='bold', color=COLORS['text_light'])
        
        # Design name and stats
        ax.set_title(f'{design_name}', fontsize=12, fontweight='bold', loc='left', pad=10)
        
        # Stats box on right
        stats_text = (f"Events: {design['events']}\n"
                     f"RMSE: {design['rmse']}\n"
                     f"Power: {design['power']}\n"
                     f"Duty: {design['duty_cycle']}")
        
        ax.text(31, 0.4, stats_text, fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8FAFC', edgecolor='#CBD5E1'),
                verticalalignment='center')
        
        # Note below
        note_color = COLORS['recommended'] if 'Recommended' in design_name else COLORS['text_light']
        ax.text(0, -0.7, design['note'], fontsize=9, color=note_color, style='italic')
        
        # Formatting
        ax.set_xlim(-1, 38)
        ax.set_ylim(-0.8, 1.4)
        ax.set_ylabel('Signal', fontsize=10)
        if idx == n_designs - 1:
            ax.set_xlabel('Time (seconds)', fontsize=11)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
        
        # Add legend only on first plot
        if idx == 0:
            ax.legend(loc='upper right', frameon=True, fontsize=9)
    
    fig.suptitle('Stimulation Protocol Comparison: Red LED Designs with Constant Blue LED', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    
    # Add limitation note at bottom
    fig.text(0.02, 0.01, 
             'Limitation: This analysis focused on red LED timing. Blue LED was held constant at 7 PWM. '
             'Future work should explore blue LED cycling (5-15 PWM) interactions with burst designs.',
             fontsize=9, color='#6B7280', style='italic', wrap=True)
    
    return fig


if __name__ == '__main__':
    fig = create_figure()
    
    output_path = '/Users/gilraitses/INDYsim_project/phenotyping_followup/figures/fig_stimulation_protocols.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    plt.close()


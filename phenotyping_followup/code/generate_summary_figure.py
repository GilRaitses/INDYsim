#!/usr/bin/env python3
"""
Generate summary figure showing design comparison and bottleneck analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set Avenir font
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.weight'] = 'normal'

# Data from simulation results (A/B = 0.125, GMR61 kernel)
designs = ['Continuous\n10s', 'Burst\n10×0.5s', 'Medium\n4×1s', 'Long\n2×2s']
events = [1.9, 14.9, 6.0, 3.1]
rmse = [0.108, 0.036, 0.048, 0.077]
fisher = [0.27, 2.70, 1.08, 0.54]

# Create 2x2 figure
fig = plt.figure(figsize=(14, 10))

<<<<<<< HEAD:phenotyping_followup/code/generate_summary_figure.py
# Cinnamoroll color palette
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from cinnamoroll_palette import COLORS

def get_colors(highlight_idx=0):
    """Return colors with highlight on current (idx 0) or best (idx 1)"""
    cols = [COLORS['current'], COLORS['recommended'], COLORS['alternative'], COLORS['accent']]
=======
# Colors
current_color = '#EF4444'  # red for current design
best_color = '#10B981'     # green for best
other_colors = ['#3B82F6', '#A855F7']  # blue, purple for others

def get_colors(highlight_idx=0):
    """Return colors with highlight on current (idx 0) or best (idx 1)"""
    cols = [current_color, best_color, other_colors[0], other_colors[1]]
>>>>>>> 300694426984e06c404a7f98f9992b80ce815cc0:phenotyping_followup/code_for_sharing/generate_summary_figure.py
    return cols

colors = get_colors()

# ============ Panel A: Current Design Bottleneck ============
ax1 = fig.add_subplot(2, 2, 1)
bars1 = ax1.bar(designs, events, color=colors, edgecolor='white', linewidth=2)
ax1.set_ylabel('Events per Track', fontsize=13)
ax1.set_title('A. Event Yield: Current Design is Bottleneck', fontsize=14, fontweight='bold', pad=10)

# Highlight current design
<<<<<<< HEAD:phenotyping_followup/code/generate_summary_figure.py
bars1[0].set_edgecolor(COLORS['failure_dark'])
bars1[0].set_linewidth(3)
ax1.annotate('CURRENT\nDESIGN', xy=(0, events[0]), xytext=(0, events[0] + 3),
             fontsize=10, fontweight='bold', color=COLORS['failure_dark'], ha='center',
             arrowprops=dict(arrowstyle='->', color=COLORS['failure_dark'], lw=2))

# Show improvement
ax1.annotate('8× more\nevents', xy=(1, events[1]), xytext=(2.2, events[1]),
             fontsize=11, fontweight='bold', color=COLORS['success_dark'], ha='center',
             arrowprops=dict(arrowstyle='->', color=COLORS['success_dark'], lw=2))

ax1.axhline(y=events[0], color=COLORS['current'], linestyle='--', alpha=0.5, linewidth=2)
=======
bars1[0].set_edgecolor('#991B1B')
bars1[0].set_linewidth(3)
ax1.annotate('CURRENT\nDESIGN', xy=(0, events[0]), xytext=(0, events[0] + 3),
             fontsize=10, fontweight='bold', color='#991B1B', ha='center',
             arrowprops=dict(arrowstyle='->', color='#991B1B', lw=2))

# Show improvement
ax1.annotate('8× more\nevents', xy=(1, events[1]), xytext=(2.2, events[1]),
             fontsize=11, fontweight='bold', color='#059669', ha='center',
             arrowprops=dict(arrowstyle='->', color='#059669', lw=2))

ax1.axhline(y=events[0], color='#EF4444', linestyle='--', alpha=0.5, linewidth=2)
>>>>>>> 300694426984e06c404a7f98f9992b80ce815cc0:phenotyping_followup/code_for_sharing/generate_summary_figure.py
ax1.set_ylim(0, max(events) * 1.4)

# ============ Panel B: Estimation Error ============
ax2 = fig.add_subplot(2, 2, 2)
bars2 = ax2.bar(designs, rmse, color=colors, edgecolor='white', linewidth=2)
ax2.set_ylabel('RMSE (seconds)', fontsize=13)
ax2.set_title('B. Estimation Error: 3× Reduction Possible', fontsize=14, fontweight='bold', pad=10)

<<<<<<< HEAD:phenotyping_followup/code/generate_summary_figure.py
bars2[0].set_edgecolor(COLORS['failure_dark'])
bars2[0].set_linewidth(3)

ax2.annotate('3× lower\nerror', xy=(1, rmse[1]), xytext=(2.2, rmse[1] + 0.02),
             fontsize=11, fontweight='bold', color=COLORS['success_dark'], ha='center',
             arrowprops=dict(arrowstyle='->', color=COLORS['success_dark'], lw=2))

ax2.axhline(y=rmse[0], color=COLORS['current'], linestyle='--', alpha=0.5, linewidth=2)
=======
bars2[0].set_edgecolor('#991B1B')
bars2[0].set_linewidth(3)

ax2.annotate('3× lower\nerror', xy=(1, rmse[1]), xytext=(2.2, rmse[1] + 0.02),
             fontsize=11, fontweight='bold', color='#059669', ha='center',
             arrowprops=dict(arrowstyle='->', color='#059669', lw=2))

ax2.axhline(y=rmse[0], color='#EF4444', linestyle='--', alpha=0.5, linewidth=2)
>>>>>>> 300694426984e06c404a7f98f9992b80ce815cc0:phenotyping_followup/code_for_sharing/generate_summary_figure.py
ax2.set_ylim(0, max(rmse) * 1.5)

# ============ Panel C: Tradeoff - Events vs Information ============
ax3 = fig.add_subplot(2, 2, 3)

# Scatter plot showing events vs fisher info per event
fisher_per_event = [f/e for f, e in zip(fisher, events)]
scatter_colors = colors
design_labels = ['Continuous 10s', 'Burst 10x0.5s', 'Medium 4x1s', 'Long 2x2s']

for i, (e, fpe, d) in enumerate(zip(events, fisher_per_event, design_labels)):
    ax3.scatter(e, fpe, s=350, c=scatter_colors[i], edgecolor='white', linewidth=2, zorder=5)

# Add labels without overlap
ax3.annotate('Continuous 10s\n(current)', xy=(events[0], fisher_per_event[0]), 
             xytext=(events[0] + 1, fisher_per_event[0] - 0.015),
<<<<<<< HEAD:phenotyping_followup/code/generate_summary_figure.py
             fontsize=10, fontweight='medium', color=COLORS['failure_dark'])
ax3.annotate('Burst 10x0.5s\n(recommended)', xy=(events[1], fisher_per_event[1]), 
             xytext=(events[1] - 5, fisher_per_event[1] + 0.005),
             fontsize=10, fontweight='medium', color=COLORS['success_dark'])
=======
             fontsize=10, fontweight='medium', color='#991B1B')
ax3.annotate('Burst 10x0.5s\n(recommended)', xy=(events[1], fisher_per_event[1]), 
             xytext=(events[1] - 5, fisher_per_event[1] + 0.005),
             fontsize=10, fontweight='medium', color='#059669')
>>>>>>> 300694426984e06c404a7f98f9992b80ce815cc0:phenotyping_followup/code_for_sharing/generate_summary_figure.py

ax3.set_xlabel('Events per Track', fontsize=13)
ax3.set_ylabel('Fisher Info per Event', fontsize=13)
ax3.set_title('C. Tradeoff: More Events = More Information', fontsize=14, fontweight='bold', pad=10)

# Draw arrow showing optimal direction
ax3.annotate('', xy=(16, 0.19), xytext=(4, 0.14),
<<<<<<< HEAD:phenotyping_followup/code/generate_summary_figure.py
             arrowprops=dict(arrowstyle='->', color=COLORS['success_dark'], lw=3, alpha=0.4))
ax3.text(10, 0.175, 'Better', fontsize=12, color=COLORS['success_dark'], fontweight='bold', ha='center', alpha=0.7)
=======
             arrowprops=dict(arrowstyle='->', color='#059669', lw=3, alpha=0.4))
ax3.text(10, 0.175, 'Better', fontsize=12, color='#059669', fontweight='bold', ha='center', alpha=0.7)
>>>>>>> 300694426984e06c404a7f98f9992b80ce815cc0:phenotyping_followup/code_for_sharing/generate_summary_figure.py

ax3.set_xlim(0, max(events) * 1.4)
ax3.set_ylim(0.12, 0.2)

# ============ Panel D: Summary Recommendation ============
ax4 = fig.add_subplot(2, 2, 4)
ax4.axis('off')

# Text summary - cleaner format
summary_lines = [
<<<<<<< HEAD:phenotyping_followup/code/generate_summary_figure.py
    ("CURRENT PROTOCOL", COLORS['failure_dark'], True),
    ("Continuous 10s ON / 20s OFF", COLORS['text'], False),
    ("1.9 events per track | RMSE = 0.108s", COLORS['text_light'], False),
    ("", COLORS['text'], False),
    ("RECOMMENDED", COLORS['success_dark'], True),
    ("Burst 10x0.5s pulses", COLORS['text'], False),
    ("14.9 events (8x more) | RMSE = 0.036s (3x lower)", COLORS['text_light'], False),
    ("", COLORS['text'], False),
    ("BOTTLENECK", COLORS['accent_dark'], True),
    ("Sparse events limit tau1 identifiability", COLORS['text'], False),
    ("regardless of experimental design", COLORS['text_light'], False),
    ("", COLORS['text'], False),
    ("ALTERNATIVE STRATEGY", COLORS['primary_dark'], True),
    ("Phenotype on composite scores", COLORS['text'], False),
    ("(Precision, Burstiness) instead of tau1", COLORS['text_light'], False),
=======
    ("CURRENT PROTOCOL", "#991B1B", True),
    ("Continuous 10s ON / 20s OFF", "#374151", False),
    ("1.9 events per track | RMSE = 0.108s", "#6B7280", False),
    ("", "#000000", False),
    ("RECOMMENDED", "#059669", True),
    ("Burst 10x0.5s pulses", "#374151", False),
    ("14.9 events (8x more) | RMSE = 0.036s (3x lower)", "#6B7280", False),
    ("", "#000000", False),
    ("BOTTLENECK", "#7C3AED", True),
    ("Sparse events limit tau1 identifiability", "#374151", False),
    ("regardless of experimental design", "#6B7280", False),
    ("", "#000000", False),
    ("ALTERNATIVE STRATEGY", "#3B82F6", True),
    ("Phenotype on composite scores", "#374151", False),
    ("(Precision, Burstiness) instead of tau1", "#6B7280", False),
>>>>>>> 300694426984e06c404a7f98f9992b80ce815cc0:phenotyping_followup/code_for_sharing/generate_summary_figure.py
]

y_pos = 0.92
for text, color, is_header in summary_lines:
    weight = 'bold' if is_header else 'normal'
    size = 13 if is_header else 11
    ax4.text(0.05, y_pos, text, transform=ax4.transAxes, fontsize=size,
             color=color, fontweight=weight, fontfamily='Avenir')
    y_pos -= 0.055 if text else 0.03

ax4.set_title('D. Summary & Recommendations', fontsize=14, fontweight='bold', pad=10, loc='left')

# Style all axes
for ax in [ax1, ax2, ax3]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', labelsize=11, width=1.5)
    ax.set_facecolor('#FAFAFA')

fig.patch.set_facecolor('white')

plt.tight_layout()

# Save
output_path = '/Users/gilraitses/INDYsim_project/phenotyping_followup/figures/fig_design_comparison_summary.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

plt.close()


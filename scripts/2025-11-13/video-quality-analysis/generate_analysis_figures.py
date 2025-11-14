#!/usr/bin/env python3
"""
Generate figures for video quality analysis report
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749']

# Create output directory
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

# Load data from JSON report
json_path = Path(r"C:\Users\l-skanatalab\Desktop\IMG_3788.ml_quality_report.json")
if json_path.exists():
    with open(json_path) as f:
        data = json.load(f)
else:
    # Fallback data
    data = {
        'visual_quality': {
            'sharpness_mean': 33.65,
            'brightness_mean': 72.8,
            'contrast_mean': 44.53,
            'noise_mean': 5.32,
            'exposure_mean': 98.0,
            'stability_mean': 5.43,
            'smoothness_mean': 0.937,
            'color_compression_mean': 0.876
        },
        'usable_analysis': {
            'total_seconds': 15.83,
            'usable_seconds': 15.83,
            'high_quality_seconds': 6.93
        },
        'valuation': {
            'per_second_breakdown': {
                'base_rate': 50,
                'quality_multiplier': 0.5,
                'resolution_multiplier': 1.0,
                'significance_multiplier': 1.5,
                'narrative_multiplier': 2.5,
                'multiple_use_multiplier': 1.5,
                'final_rate_per_second': 140.62
            },
            'conservative_estimate': 2226,
            'optimistic_estimate': 2713
        },
        'narrative_significance': {
            'narrative_score': 100
        }
    }

# Figure 1: Quality Metrics Radar Chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

metrics = ['Sharpness', 'Brightness', 'Contrast', 'Exposure', 'Stability', 'Smoothness']
values = [
    data['visual_quality']['sharpness_mean'] / 500 * 100,  # Normalized
    data['visual_quality']['brightness_mean'] / 255 * 100,  # Normalized
    data['visual_quality']['contrast_mean'] / 50 * 100,  # Normalized
    data['visual_quality']['exposure_mean'],  # Already 0-100
    100 - (data['visual_quality']['stability_mean'] / 10 * 100),  # Inverted (lower is better)
    data['visual_quality']['smoothness_mean'] * 100  # Normalized
]

# Normalize to 0-100 scale
values = [min(100, max(0, v)) for v in values]

angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
values += values[:1]  # Complete the circle
angles += angles[:1]

ax.plot(angles, values, 'o-', linewidth=2, color=colors[0], label='Video Quality')
ax.fill(angles, values, alpha=0.25, color=colors[0])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10)
ax.set_title('Video Quality Metrics (Normalized 0-100)', size=16, fontweight='bold', pad=20)
ax.grid(True)

plt.tight_layout()
plt.savefig(output_dir / 'quality_metrics_radar.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'quality_metrics_radar.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Usability Breakdown (Stacked Bar)
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Total Duration', 'Usable Seconds', 'High Quality Seconds']
seconds = [
    data['usable_analysis']['total_seconds'],
    data['usable_analysis']['usable_seconds'],
    data['usable_analysis']['high_quality_seconds']
]

bars = ax.bar(categories, seconds, color=[colors[2], colors[0], colors[4]], alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, seconds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}s\n({val/data["usable_analysis"]["total_seconds"]*100:.1f}%)',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Duration (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Usability Breakdown', fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(0, max(seconds) * 1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'usability_breakdown.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'usability_breakdown.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Valuation Multipliers (Waterfall-style)
fig, ax = plt.subplots(figsize=(12, 8))

breakdown = data['valuation']['per_second_breakdown']
factors = ['Base Rate', 'Quality\n(0.5×)', 'Resolution\n(1.0×)', 'Content\n(1.5×)', 
           'Narrative\n(2.5×)', 'Multiple-Use\n(1.5×)', 'Final Rate']
values_waterfall = [
    breakdown['base_rate'],
    breakdown['base_rate'] * breakdown['quality_multiplier'] - breakdown['base_rate'],
    breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'] - breakdown['base_rate'] * breakdown['quality_multiplier'],
    breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'] * breakdown['significance_multiplier'] - breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'],
    breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'] * breakdown['significance_multiplier'] * breakdown['narrative_multiplier'] - breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'] * breakdown['significance_multiplier'],
    breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'] * breakdown['significance_multiplier'] * breakdown['narrative_multiplier'] * breakdown['multiple_use_multiplier'] - breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'] * breakdown['significance_multiplier'] * breakdown['narrative_multiplier'],
    breakdown['final_rate_per_second']
]

# Calculate cumulative positions
cumulative = [0]
for i, val in enumerate(values_waterfall[:-1]):
    cumulative.append(cumulative[-1] + val)

# Colors for each bar
bar_colors = [colors[2], colors[3], colors[1], colors[4], colors[0], colors[5], colors[2]]

# Create waterfall
for i, (val, cum) in enumerate(zip(values_waterfall, cumulative)):
    if i == 0:
        ax.bar(i, val, bottom=0, color=bar_colors[i], alpha=0.8, edgecolor='black', linewidth=1.5)
    elif i < len(values_waterfall) - 1:
        ax.bar(i, val, bottom=cum, color=bar_colors[i], alpha=0.8, edgecolor='black', linewidth=1.5)
    else:
        ax.bar(i, val, bottom=0, color=bar_colors[i], alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels
for i, (val, cum) in enumerate(zip(values_waterfall, cumulative)):
    if i == 0 or i == len(values_waterfall) - 1:
        y_pos = val / 2
    else:
        y_pos = cum + val / 2
    ax.text(i, y_pos, f'${val:.2f}', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

ax.set_xticks(range(len(factors)))
ax.set_xticklabels(factors, fontsize=11, rotation=0)
ax.set_ylabel('Rate per Second ($)', fontsize=12, fontweight='bold')
ax.set_title('Valuation Rate Calculation Breakdown', fontsize=16, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, max(values_waterfall) * 1.15)

plt.tight_layout()
plt.savefig(output_dir / 'valuation_breakdown.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'valuation_breakdown.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Value Range Comparison
fig, ax = plt.subplots(figsize=(10, 6))

estimates = ['Conservative\nEstimate', 'Recommended\nEstimate', 'Optimistic\nEstimate']
values_est = [
    data['valuation']['conservative_estimate'],
    2500,  # Recommended mid-range estimate
    data['valuation']['optimistic_estimate']
]

bars = ax.bar(estimates, values_est, color=[colors[1], colors[2], colors[4]], alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, val in zip(bars, values_est):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'${val:,.0f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add range line
ax.plot([0, 2], [data['valuation']['conservative_estimate'], data['valuation']['optimistic_estimate']], 
        'k--', linewidth=2, alpha=0.5, label='Estimated Range')
ax.fill_between([0, 2], data['valuation']['conservative_estimate'], data['valuation']['optimistic_estimate'],
                alpha=0.1, color='gray')

ax.set_ylabel('Value ($)', fontsize=12, fontweight='bold')
ax.set_title('License Fee Value Range', fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(0, max(values_est) * 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'value_range.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'value_range.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Narrative Significance Factors (Pie Chart)
fig, ax = plt.subplots(figsize=(10, 8))

factors_narrative = [
    'Rare/Unique Event\n(40 pts)',
    'Historic Moment\n(30 pts)',
    'Subject Visibility\n(15 pts)',
    'Multiple-Use Potential\n(20 pts)'
]
sizes = [40, 30, 15, 20]
explode = (0.1, 0.05, 0, 0.05)

wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=factors_narrative, colors=colors[:4],
                                   autopct='%1.0f%%', shadow=True, startangle=90,
                                   textprops={'fontsize': 11, 'fontweight': 'bold'})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')

    ax.set_title('Narrative Significance Score Components\n(Total: 100/100 - High Importance)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(output_dir / 'narrative_significance.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'narrative_significance.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 6: Quality Score Components (Stacked Bar)
fig, ax = plt.subplots(figsize=(10, 6))

components = ['Sharpness', 'Exposure', 'Contrast', 'Noise\n(Low)', 'Stability']
weights = [0.25, 0.20, 0.15, 0.15, 0.25]
scores = [
    min(100, (data['visual_quality']['sharpness_mean'] / 500) * 100),
    data['visual_quality']['exposure_mean'],
    min(100, (data['visual_quality']['contrast_mean'] / 40) * 100),
    max(0, 100 - (data['visual_quality']['noise_mean'] / 50) * 100),
    max(0, 100 - (data['visual_quality']['stability_mean'] / 5) * 100)
]
weighted_scores = [s * w for s, w in zip(scores, weights)]

# Create stacked bar
bottom = 0
colors_comp = [colors[0], colors[4], colors[2], colors[1], colors[3]]
for i, (comp, score, weight, color) in enumerate(zip(components, scores, weights, colors_comp)):
    ax.barh(0, weighted_scores[i], left=bottom, label=f'{comp} ({weight*100:.0f}%)', 
            color=color, alpha=0.8, edgecolor='black', linewidth=1)
    # Add label
    ax.text(bottom + weighted_scores[i]/2, 0, f'{comp}\n{score:.1f}', 
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    bottom += weighted_scores[i]

ax.set_xlim(0, 100)
ax.set_xlabel('Weighted Contribution to Quality Score', fontsize=12, fontweight='bold')
ax.set_title(f'Overall Quality Score Components\n(Total Score: {sum(weighted_scores):.1f}/100)', 
             fontsize=16, fontweight='bold', pad=15)
ax.set_yticks([])
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'quality_score_components.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'quality_score_components.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 7: Content Detection Timeline
fig, ax = plt.subplots(figsize=(14, 4))

total_frames = 32
frames_with_bird = 14
frames_with_dog = 0
frames_with_interaction = 0

# Create timeline
timeline = np.linspace(0, 15.83, total_frames)
bird_detected = [1 if i < frames_with_bird else 0 for i in range(total_frames)]
dog_detected = [0] * total_frames
interaction = [0] * total_frames

ax.fill_between(timeline, 0, bird_detected, alpha=0.6, color=colors[0], label=f'Subject (Bird/Owl) - {frames_with_bird} frames ({frames_with_bird/total_frames*100:.1f}%)')
ax.fill_between(timeline, 0, dog_detected, alpha=0.6, color=colors[1], label=f'Dog - {frames_with_dog} frames')
ax.fill_between(timeline, 0, interaction, alpha=0.6, color=colors[2], label=f'Interaction - {frames_with_interaction} frames')

ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax.set_ylabel('Detection', fontsize=12, fontweight='bold')
ax.set_title('Content Detection Timeline', fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(-0.1, 1.1)
ax.set_yticks([0, 1])
ax.set_yticklabels(['Not Detected', 'Detected'])
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(output_dir / 'content_timeline.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'content_timeline.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"Generated {len(list(output_dir.glob('*.pdf')))} figures in {output_dir}/")


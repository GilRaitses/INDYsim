#!/usr/bin/env python3
"""
Generate figures for video quality analysis report
Takes JSON report path and output directory as arguments
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import sys

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749']

def generate_figures(json_path, output_dir):
    """Generate all figures for a video analysis report"""
    
    # Load data from JSON report
    json_path = Path(json_path)
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
    else:
        print(f"Error: JSON file not found: {json_path}")
        return False
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
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
    percentages = [
        100.0,
        (data['usable_analysis']['usable_seconds'] / data['usable_analysis']['total_seconds'] * 100) if data['usable_analysis']['total_seconds'] > 0 else 0,
        (data['usable_analysis']['high_quality_seconds'] / data['usable_analysis']['total_seconds'] * 100) if data['usable_analysis']['total_seconds'] > 0 else 0
    ]
    
    bars = ax.bar(categories, seconds, color=[colors[0], colors[1], colors[2]], alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, sec, pct) in enumerate(zip(bars, seconds, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{sec:.2f}s ({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Seconds', fontsize=12, fontweight='bold')
    ax.set_title('Usability Breakdown', size=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'usability_breakdown.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'usability_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Value Range Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    estimates = ['Conservative', 'Recommended', 'Optimistic']
    values = [
        data['valuation']['conservative_estimate'],
        (data['valuation']['conservative_estimate'] + data['valuation']['optimistic_estimate']) / 2,
        data['valuation']['optimistic_estimate']
    ]
    
    bars = ax.bar(estimates, values, color=[colors[0], colors[2], colors[1]], alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'${val:,.0f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Estimated Value ($)', fontsize=12, fontweight='bold')
    ax.set_title('Estimated Value Range', size=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'value_range.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'value_range.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 4: Valuation Breakdown (Waterfall)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    breakdown = data['valuation']['per_second_breakdown']
    stages = ['Base Rate', 'Quality', 'Resolution', 'Content', 'Narrative', 'Multiple-Use', 'Final Rate']
    values = [
        breakdown['base_rate'],
        breakdown['base_rate'] * breakdown['quality_multiplier'] - breakdown['base_rate'],
        breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'] - breakdown['base_rate'] * breakdown['quality_multiplier'],
        breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'] * breakdown['significance_multiplier'] - breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'],
        breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'] * breakdown['significance_multiplier'] * breakdown['narrative_multiplier'] - breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'] * breakdown['significance_multiplier'],
        breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'] * breakdown['significance_multiplier'] * breakdown['narrative_multiplier'] * breakdown['multiple_use_multiplier'] - breakdown['base_rate'] * breakdown['quality_multiplier'] * breakdown['resolution_multiplier'] * breakdown['significance_multiplier'] * breakdown['narrative_multiplier'],
        breakdown['final_rate_per_second']
    ]
    
    # Calculate cumulative for waterfall
    cumulative = [values[0]]
    for i in range(1, len(values)-1):
        cumulative.append(cumulative[-1] + values[i])
    cumulative.append(values[-1])
    
    # Create waterfall bars
    colors_waterfall = [colors[0], colors[3], colors[4], colors[1], colors[2], colors[5], colors[0]]
    for i, (stage, val, cum, col) in enumerate(zip(stages, values, cumulative, colors_waterfall)):
        if i == 0 or i == len(values) - 1:
            ax.bar(i, val, bottom=0 if i == 0 else cumulative[i-1], color=col, alpha=0.8)
        else:
            ax.bar(i, val, bottom=cumulative[i-1], color=col, alpha=0.8)
    
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(stages, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Rate per Second ($)', fontsize=12, fontweight='bold')
    ax.set_title('Valuation Breakdown', size=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (val, cum) in enumerate(zip(values, cumulative)):
        if i == 0:
            y_pos = val / 2
        elif i == len(values) - 1:
            y_pos = cum
        else:
            y_pos = cumulative[i-1] + val / 2
        ax.text(i, y_pos, f'${val:.2f}', ha='center', va='center', fontsize=9, fontweight='bold', color='white' if val < 0 else 'black')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'valuation_breakdown.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'valuation_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Narrative Significance (if available)
    if 'narrative_significance' in data and 'narrative_score' in data['narrative_significance']:
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Simple pie chart showing narrative score
        score = data['narrative_significance']['narrative_score']
        sizes = [score, 100 - score]
        labels = [f'Narrative Value\n{score}/100', f'Remaining\n{100-score}/100']
        colors_pie = [colors[2], '#E0E0E0']
        
        ax.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax.set_title('Narrative Significance Score', size=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'narrative_significance.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'narrative_significance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Figures generated successfully in {output_dir}")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python generate_figures_for_video.py <json_report_path> <output_dir>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    output_dir = sys.argv[2]
    generate_figures(json_path, output_dir)



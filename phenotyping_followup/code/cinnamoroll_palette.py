#!/usr/bin/env python3
"""
Cinnamoroll Color Palette
==========================
Standardized soft pastel color palette for all figures.
Based on the cinnamoroll theme (#B8E6E6 soft cyan-blue).
"""

# Cinnamoroll-inspired soft pastel palette
CINNAMOROLL_PALETTE = {
    # Primary colors (soft pastels)
    'primary': '#B8E6E6',        # Soft cyan-blue (cinnamoroll)
    'primary_dark': '#7FCACA',   # Darker cyan-blue
    'primary_light': '#E6F5F5',  # Very light cyan-blue
    
    # Secondary colors (complementary pastels)
    'secondary': '#FFD6E8',      # Soft pink
    'secondary_dark': '#FFB3D9', # Darker pink
    'accent': '#FFE6CC',         # Soft peach
    'accent_dark': '#FFCC99',    # Darker peach
    
    # Semantic colors (soft versions)
    'success': '#B8E6B8',         # Soft green
    'success_dark': '#7FCA7F',   # Darker green
    'warning': '#FFE6B8',         # Soft yellow
    'warning_dark': '#FFCC7F',   # Darker yellow
    'failure': '#FFB3B3',         # Soft coral/red
    'failure_dark': '#FF7F7F',   # Darker coral/red
    
    # Neutral colors
    'text': '#4A5568',           # Dark gray-blue
    'text_light': '#718096',     # Medium gray-blue
    'background': '#FAFAFA',      # Off-white
    'grid': '#E2E8F0',           # Light gray-blue
    'border': '#CBD5E0',          # Medium gray-blue
    
    # Data visualization
    'data': '#B8E6E6',           # Primary cinnamoroll
    'data_light': '#E6F5F5',     # Very light
    'data_dark': '#7FCACA',      # Darker
    'population': '#FFB3D9',     # Soft pink for population
    'outlier': '#FFCC99',        # Soft peach for outliers
    'ci': '#E2E8F0',             # Light gray for confidence intervals
    
    # Design comparison
    'current': '#FFB3B3',        # Soft coral for current design
    'recommended': '#B8E6B8',   # Soft green for recommended
    'alternative': '#FFE6CC',   # Soft peach for alternatives
}

# Export as COLORS for backward compatibility
COLORS = CINNAMOROLL_PALETTE


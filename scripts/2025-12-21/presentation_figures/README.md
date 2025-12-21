# Presentation Figure Scripts

These scripts are copies of the original figure-generating code, intended for modification to create presentation-specific versions with individual panels.

## Purpose

The manuscript figures use multi-panel layouts (triptychs, 2x2 grids) that are too dense for presentation slides. These scripts should be modified to output individual panels as separate PDF files.

## Scripts Included

| Script | Original Source | Panels to Split |
|--------|-----------------|-----------------|
| `generate_validation_figures.py` | phenotyping_followup/code | 4 panels (A, B, C, D) |
| `generate_identifiability_figure.py` | phenotyping_followup/code | 4 panels (A, B, C, D) |
| `generate_stimulation_schematic.py` | phenotyping_followup/code | 4 panels (A, B, C, D) |
| `generate_data_sparsity_v2.py` | phenotyping_followup/code | 3 panels |
| `generate_summary_figure.py` | phenotyping_followup/code | 4 panels |
| `16_design_kernel_sweep.py` | phenotyping_followup/code | 4 panels (A, B, C, D) |
| `11_power_analysis.py` | phenotyping_followup/code | 2 panels (A, B) |
| `generate_figure5.py` | code | 2 panels (A, B) |
| `generate_figure6_combined.py` | code | Combined figure |
| `fit_gamma_per_condition.py` | code | Kernel fitting |
| `generate_kernel_comparison_figure.py` | phenotyping_followup/code | Single panel |

## Modification Strategy

For each script, add a `--presentation` flag or create a `_presentation` variant that:

1. Outputs each panel to a separate PDF file
2. Increases font sizes by 1.5x for readability
3. Uses larger markers and line widths
4. Removes panel labels (A, B, C, D) from within the figure since slide titles will identify them

## Example Modification

```python
# Original
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# ... plotting code ...
fig.savefig('fig_combined.pdf')

# Presentation version
for i, ax in enumerate(axes):
    fig_single, ax_single = plt.subplots(figsize=(8, 6))
    # ... copy plotting code for panel i ...
    fig_single.savefig(f'fig_panel_{chr(65+i)}.pdf')
```

## Output Location

Save presentation figures to:
`phenotyping_followup/presentation/figures/`

## Date Created

2025-12-21


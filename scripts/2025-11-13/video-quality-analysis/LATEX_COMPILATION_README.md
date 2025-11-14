# LaTeX Document Compilation Instructions

## Files Created

1. **video_quality_analysis_report.tex** - Comprehensive technical analysis report with appendix
2. **licensing_agreement.tex** - Standard video licensing agreement

## Compilation

### Option 1: Using pdflatex (Recommended)

If you have LaTeX installed (MiKTeX, TeX Live, etc.):

```bash
pdflatex video_quality_analysis_report.tex
pdflatex licensing_agreement.tex
```

Run twice to resolve all references.

### Option 2: Online LaTeX Compilers

Use online services:
- **Overleaf**: https://www.overleaf.com
- **ShareLaTeX**: https://www.sharelatex.com
- **LaTeX Base**: https://latexbase.com

Simply upload the `.tex` files and compile.

### Option 3: Using Docker

```bash
docker run --rm -v "%cd%":/data pdflatex video_quality_analysis_report.tex
docker run --rm -v "%cd%":/data pdflatex licensing_agreement.tex
```

## Required LaTeX Packages

The documents use standard packages that should be included in most LaTeX distributions:
- `geometry`
- `graphicx`
- `booktabs`
- `longtable`
- `hyperref`
- `amsmath`
- `xcolor`
- `fancyhdr`
- `enumitem`

## Output

Compiling will generate:
- `video_quality_analysis_report.pdf`
- `licensing_agreement.pdf`

## Customization

Before compiling, you may want to fill in:
- **Licensing Agreement**: Addresses, contact information, payment terms, dates
- **Report**: Any additional context or notes

## Notes

- The licensing agreement uses a license fee of **$2,500.00** (within the estimated range of $2,226 - $2,713)
- All values are pulled from the ML analysis report
- The agreement includes standard video licensing terms
- Both documents are ready for professional use


#!/usr/bin/env python3
"""
Verify PDF rendering and convert to PNGs for visual inspection
"""

import subprocess
import sys
from pathlib import Path

def check_pdf_tools():
    """Check if pdftoppm or other PDF tools are available"""
    tools = {
        'pdftoppm': 'poppler-utils',
        'magick': 'ImageMagick',
        'gs': 'Ghostscript'
    }
    
    available = {}
    for tool, name in tools.items():
        try:
            result = subprocess.run([tool, '--version'], capture_output=True, timeout=5)
            available[tool] = True
            print(f"✓ {name} ({tool}) is available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            available[tool] = False
            print(f"✗ {name} ({tool}) not found")
    
    return available

def convert_pdf_to_png_pdftoppm(pdf_path, output_dir):
    """Convert PDF to PNG using pdftoppm"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    pdf_name = Path(pdf_path).stem
    output_pattern = output_dir / f"{pdf_name}_page_%02d.png"
    
    cmd = ['pdftoppm', '-png', '-r', '150', str(pdf_path), str(output_dir / pdf_name)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Rename files to have page numbers
        png_files = sorted(output_dir.glob(f"{pdf_name}-*.png"))
        for i, png_file in enumerate(png_files, 1):
            new_name = output_dir / f"{pdf_name}_page_{i:02d}.png"
            png_file.rename(new_name)
        print(f"✓ Converted {pdf_path} to {len(png_files)} PNG pages")
        return True
    except Exception as e:
        print(f"✗ Error converting {pdf_path}: {e}")
        return False

def convert_pdf_to_png_magick(pdf_path, output_dir):
    """Convert PDF to PNG using ImageMagick"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    pdf_name = Path(pdf_path).stem
    output_pattern = str(output_dir / f"{pdf_name}_page_%02d.png")
    
    cmd = ['magick', '-density', '150', str(pdf_path), output_pattern]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        png_files = list(output_dir.glob(f"{pdf_name}_page_*.png"))
        print(f"✓ Converted {pdf_path} to {len(png_files)} PNG pages")
        return True
    except Exception as e:
        print(f"✗ Error converting {pdf_path}: {e}")
        return False

def check_latex_errors(log_file):
    """Check for LaTeX errors in log file"""
    if not Path(log_file).exists():
        return []
    
    errors = []
    warnings = []
    
    with open(log_file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if 'Error' in line or '! ' in line:
                errors.append((i+1, line.strip()))
            elif 'Warning' in line and 'Overfull' not in line and 'Underfull' not in line:
                warnings.append((i+1, line.strip()))
    
    return errors, warnings

def check_pdf_exists(pdf_path):
    """Check if PDF exists and get page count"""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return False, 0
    
    try:
        # Try to get page count using pdftk or pdfinfo
        result = subprocess.run(['pdfinfo', str(pdf_path)], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'Pages:' in line:
                    pages = int(line.split(':')[1].strip())
                    return True, pages
    except:
        pass
    
    # Fallback: just check if file exists
    return True, -1

def main():
    print("="*60)
    print("PDF Verification and PNG Conversion")
    print("="*60)
    
    # Check available tools
    print("\n1. Checking PDF conversion tools...")
    tools = check_pdf_tools()
    
    # Find PDFs
    pdfs = list(Path('.').glob('*.pdf'))
    print(f"\n2. Found {len(pdfs)} PDF files:")
    for pdf in pdfs:
        exists, pages = check_pdf_exists(pdf)
        if exists:
            print(f"   - {pdf.name} ({pages if pages > 0 else 'unknown'} pages)")
    
    # Convert PDFs to PNGs
    print("\n3. Converting PDFs to PNGs...")
    png_output = Path('pdf_pages_png')
    png_output.mkdir(exist_ok=True)
    
    for pdf in pdfs:
        print(f"\n   Processing {pdf.name}...")
        if tools.get('pdftoppm'):
            success = convert_pdf_to_png_pdftoppm(pdf, png_output)
        elif tools.get('magick'):
            success = convert_pdf_to_png_magick(pdf, png_output)
        else:
            print(f"   ✗ No PDF conversion tool available")
            success = False
        
        if success:
            # Check for LaTeX log files
            log_file = pdf.with_suffix('.log')
            if log_file.exists():
                errors, warnings = check_latex_errors(log_file)
                if errors:
                    print(f"   ⚠ Found {len(errors)} LaTeX errors:")
                    for line_num, error in errors[:5]:  # Show first 5
                        print(f"      Line {line_num}: {error}")
                if warnings:
                    print(f"   ⚠ Found {len(warnings)} LaTeX warnings (showing first 5):")
                    for line_num, warning in warnings[:5]:
                        print(f"      Line {line_num}: {warning}")
    
    print("\n" + "="*60)
    print("Verification complete!")
    print(f"PNG pages saved to: {png_output}/")
    print("="*60)

if __name__ == '__main__':
    main()


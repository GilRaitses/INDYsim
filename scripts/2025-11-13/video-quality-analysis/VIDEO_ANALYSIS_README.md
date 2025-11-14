# Video Quality Analyzer

A comprehensive tool to analyze video quality for documentary production and licensing assessment.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements_video_analysis.txt
```

Or install manually:
```bash
pip install opencv-python numpy
```

## Usage

### Basic Usage
```bash
python analyze_video_quality.py <video_path> [sample_frames]
```

### Example - Analyze Flaco Video
```bash
python analyze_video_quality.py "C:\Users\l-skanatalab\Desktop\IMG_3788.mov"
```

### Quick Run (Windows)
Double-click `analyze_flaco_video.bat` or run:
```bash
analyze_flaco_video.bat
```

## What It Analyzes

### Technical Specifications
- Resolution and resolution category (4K, 1080p, 720p, etc.)
- Frame rate (fps)
- Duration and total frames
- Codec information
- Bitrate
- Pixel format

### Visual Quality Metrics
- **Sharpness**: Laplacian variance to detect blur
- **Brightness**: Average brightness levels
- **Contrast**: RMS contrast measurement
- **Noise Level**: Standard deviation of Laplacian
- **Exposure**: Over/under exposure detection
- **Color Balance**: RGB balance assessment
- **Stability**: Camera shake/motion analysis using optical flow

### Overall Assessment
- Quality score (0-100)
- Letter grade (A+ to F)
- Broadcast readiness
- Strengths and issues
- Recommendations for documentary use

## Output

The script prints a detailed report to the console and saves a JSON file (`<video_name>.quality_report.json`) with all metrics for further analysis.

## Notes

- The script samples frames evenly throughout the video (default: 30 frames)
- More sample frames = more accurate but slower analysis
- For best results, ensure ffprobe (from FFmpeg) is installed for detailed metadata
- The script works with OpenCV alone, but ffprobe provides additional codec/bitrate info


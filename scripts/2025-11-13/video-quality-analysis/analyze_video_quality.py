#!/usr/bin/env python3
"""
Video Quality Analyzer for Documentary Footage Assessment

Analyzes video files for technical and visual quality metrics relevant
to documentary production and licensing decisions.
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import os


def get_video_metadata(video_path: str) -> Dict:
    """Extract video metadata using ffprobe if available."""
    metadata = {}
    
    try:
        # Try to use ffprobe for detailed metadata
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        # Extract video stream info
        video_stream = next((s for s in data['streams'] if s['codec_type'] == 'video'), None)
        if video_stream:
            metadata.update({
                'codec': video_stream.get('codec_name', 'unknown'),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '0/1')),
                'bitrate': int(video_stream.get('bit_rate', 0)) if video_stream.get('bit_rate') else None,
                'duration': float(video_stream.get('duration', 0)),
                'pixel_format': video_stream.get('pix_fmt', 'unknown'),
            })
        
        # Format info
        if 'format' in data:
            format_info = data['format']
            metadata.update({
                'file_size': int(format_info.get('size', 0)),
                'bitrate_overall': int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None,
                'duration_overall': float(format_info.get('duration', 0)),
            })
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        # Fallback to OpenCV if ffprobe not available
        pass
    
    return metadata


def calculate_sharpness(frame: np.ndarray) -> float:
    """Calculate Laplacian variance as sharpness metric."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def calculate_brightness(frame: np.ndarray) -> float:
    """Calculate average brightness."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return np.mean(gray)


def calculate_contrast(frame: np.ndarray) -> float:
    """Calculate RMS contrast."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    mean = np.mean(gray)
    return np.sqrt(np.mean((gray - mean) ** 2))


def calculate_noise_level(frame: np.ndarray) -> float:
    """Estimate noise level using standard deviation of Laplacian."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return np.std(laplacian)


def calculate_exposure_quality(frame: np.ndarray) -> Dict[str, float]:
    """Assess exposure quality (over/under exposure)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Percentage of pixels at extremes
    total_pixels = gray.size
    overexposed = np.sum(hist[240:]) / total_pixels * 100  # >240 is overexposed
    underexposed = np.sum(hist[:16]) / total_pixels * 100  # <16 is underexposed
    
    return {
        'overexposed_pct': overexposed,
        'underexposed_pct': underexposed,
        'exposure_score': 100 - (overexposed + underexposed)  # Higher is better
    }


def calculate_color_balance(frame: np.ndarray) -> Dict[str, float]:
    """Assess color balance."""
    if len(frame.shape) != 3:
        return {'color_balance_score': 0}
    
    b, g, r = cv2.split(frame)
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    
    # Calculate color balance deviation from neutral
    total = b_mean + g_mean + r_mean
    if total > 0:
        b_ratio = b_mean / total
        g_ratio = g_mean / total
        r_ratio = r_mean / total
        
        # Ideal ratios for neutral: ~0.33 each
        ideal = 1/3
        deviation = np.sqrt((b_ratio - ideal)**2 + (g_ratio - ideal)**2 + (r_ratio - ideal)**2)
        balance_score = max(0, 100 - deviation * 300)  # Scale to 0-100
    else:
        balance_score = 0
    
    return {
        'b_ratio': b_mean / total if total > 0 else 0,
        'g_ratio': g_mean / total if total > 0 else 0,
        'r_ratio': r_mean / total if total > 0 else 0,
        'color_balance_score': balance_score
    }


def calculate_stability(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate frame-to-frame stability using optical flow."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
    
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Calculate magnitude of motion vectors
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    avg_motion = np.mean(magnitude)
    
    return avg_motion


def analyze_video_quality(video_path: str, sample_frames: int = 30) -> Dict:
    """
    Analyze video quality metrics.
    
    Args:
        video_path: Path to video file
        sample_frames: Number of frames to sample for analysis
    
    Returns:
        Dictionary with quality metrics
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"Analyzing video: {video_path}")
    print(f"File size: {video_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Get metadata
    metadata = get_video_metadata(str(video_path))
    
    # Open video with OpenCV
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get basic properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    # Update metadata with OpenCV values if ffprobe didn't work
    if not metadata.get('width'):
        metadata.update({
            'width': width,
            'height': height,
            'fps': fps,
            'duration': duration,
            'total_frames': total_frames
        })
    
    print(f"Resolution: {width}x{height}")
    print(f"Frame rate: {fps:.2f} fps")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total frames: {total_frames}")
    
    # Sample frames evenly throughout the video
    frame_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
    
    sharpness_scores = []
    brightness_scores = []
    contrast_scores = []
    noise_scores = []
    exposure_scores = []
    color_balance_scores = []
    stability_scores = []
    
    prev_frame = None
    
    print(f"\nSampling {len(frame_indices)} frames for analysis...")
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Calculate metrics
        sharpness = calculate_sharpness(frame)
        brightness = calculate_brightness(frame)
        contrast = calculate_contrast(frame)
        noise = calculate_noise_level(frame)
        exposure = calculate_exposure_quality(frame)
        color_balance = calculate_color_balance(frame)
        
        sharpness_scores.append(sharpness)
        brightness_scores.append(brightness)
        contrast_scores.append(contrast)
        noise_scores.append(noise)
        exposure_scores.append(exposure['exposure_score'])
        color_balance_scores.append(color_balance['color_balance_score'])
        
        # Calculate stability if we have previous frame
        if prev_frame is not None:
            stability = calculate_stability(prev_frame, frame)
            stability_scores.append(stability)
        
        prev_frame = frame
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(frame_indices)} frames...")
    
    cap.release()
    
    # Calculate statistics
    results = {
        'file_path': str(video_path),
        'file_name': video_path.name,
        'metadata': metadata,
        'technical_specs': {
            'resolution': f"{width}x{height}",
            'resolution_category': get_resolution_category(width, height),
            'fps': fps,
            'duration_seconds': duration,
            'total_frames': total_frames,
            'codec': metadata.get('codec', 'unknown'),
            'bitrate_kbps': metadata.get('bitrate', 0) / 1000 if metadata.get('bitrate') else None,
            'pixel_format': metadata.get('pixel_format', 'unknown'),
        },
        'visual_quality': {
            'sharpness': {
                'mean': float(np.mean(sharpness_scores)),
                'std': float(np.std(sharpness_scores)),
                'min': float(np.min(sharpness_scores)),
                'max': float(np.max(sharpness_scores)),
                'assessment': assess_sharpness(np.mean(sharpness_scores))
            },
            'brightness': {
                'mean': float(np.mean(brightness_scores)),
                'std': float(np.std(brightness_scores)),
                'assessment': assess_brightness(np.mean(brightness_scores))
            },
            'contrast': {
                'mean': float(np.mean(contrast_scores)),
                'std': float(np.std(contrast_scores)),
                'assessment': assess_contrast(np.mean(contrast_scores))
            },
            'noise_level': {
                'mean': float(np.mean(noise_scores)),
                'std': float(np.std(noise_scores)),
                'assessment': assess_noise(np.mean(noise_scores))
            },
            'exposure': {
                'mean_score': float(np.mean(exposure_scores)),
                'assessment': assess_exposure(np.mean(exposure_scores))
            },
            'color_balance': {
                'mean_score': float(np.mean(color_balance_scores)),
                'assessment': assess_color_balance(np.mean(color_balance_scores))
            },
            'stability': {
                'mean_motion': float(np.mean(stability_scores)) if stability_scores else None,
                'std_motion': float(np.std(stability_scores)) if stability_scores else None,
                'assessment': assess_stability(np.mean(stability_scores)) if stability_scores else 'N/A'
            }
        },
        'overall_assessment': {}
    }
    
    # Calculate overall quality score
    quality_score = calculate_overall_quality_score(results)
    results['overall_assessment'] = {
        'quality_score': quality_score,
        'grade': get_quality_grade(quality_score),
        'documentary_suitability': assess_documentary_suitability(results)
    }
    
    return results


def get_resolution_category(width: int, height: int) -> str:
    """Categorize resolution."""
    if width >= 3840 or height >= 2160:
        return "4K UHD"
    elif width >= 1920 or height >= 1080:
        return "Full HD (1080p)"
    elif width >= 1280 or height >= 720:
        return "HD (720p)"
    elif width >= 640 or height >= 480:
        return "SD (480p)"
    else:
        return "Low Resolution"


def assess_sharpness(score: float) -> str:
    """Assess sharpness level."""
    if score > 500:
        return "Excellent"
    elif score > 300:
        return "Good"
    elif score > 150:
        return "Acceptable"
    elif score > 50:
        return "Soft"
    else:
        return "Very Soft/Blurry"


def assess_brightness(score: float) -> str:
    """Assess brightness level."""
    if 100 <= score <= 180:
        return "Well-exposed"
    elif 50 <= score < 100:
        return "Underexposed"
    elif score > 200:
        return "Overexposed"
    else:
        return "Very Dark"


def assess_contrast(score: float) -> str:
    """Assess contrast level."""
    if score > 40:
        return "High Contrast"
    elif score > 25:
        return "Good Contrast"
    elif score > 15:
        return "Moderate Contrast"
    else:
        return "Low Contrast"


def assess_noise(score: float) -> str:
    """Assess noise level."""
    if score < 10:
        return "Very Clean"
    elif score < 20:
        return "Low Noise"
    elif score < 35:
        return "Moderate Noise"
    elif score < 50:
        return "High Noise"
    else:
        return "Very Noisy"


def assess_exposure(score: float) -> str:
    """Assess exposure quality."""
    if score > 90:
        return "Excellent Exposure"
    elif score > 75:
        return "Good Exposure"
    elif score > 60:
        return "Acceptable Exposure"
    else:
        return "Poor Exposure"


def assess_color_balance(score: float) -> str:
    """Assess color balance."""
    if score > 90:
        return "Neutral/Well-Balanced"
    elif score > 75:
        return "Slight Color Cast"
    elif score > 60:
        return "Moderate Color Cast"
    else:
        return "Strong Color Cast"


def assess_stability(score: float) -> str:
    """Assess camera stability."""
    if score < 1.0:
        return "Very Stable"
    elif score < 2.5:
        return "Stable"
    elif score < 5.0:
        return "Moderate Shake"
    else:
        return "Unstable/Shaky"


def calculate_overall_quality_score(results: Dict) -> float:
    """Calculate overall quality score (0-100)."""
    visual = results['visual_quality']
    
    # Weighted components
    weights = {
        'sharpness': 0.25,
        'exposure': 0.20,
        'contrast': 0.15,
        'noise': 0.15,
        'color_balance': 0.10,
        'stability': 0.15
    }
    
    # Normalize scores to 0-100 scale
    sharpness_norm = min(100, (visual['sharpness']['mean'] / 500) * 100)
    exposure_norm = visual['exposure']['mean_score']
    contrast_norm = min(100, (visual['contrast']['mean'] / 40) * 100)
    noise_norm = max(0, 100 - (visual['noise_level']['mean'] / 50) * 100)
    color_norm = visual['color_balance']['mean_score']
    
    if visual['stability']['mean_motion']:
        stability_norm = max(0, 100 - (visual['stability']['mean_motion'] / 5) * 100)
    else:
        stability_norm = 50  # Neutral if can't calculate
    
    score = (
        sharpness_norm * weights['sharpness'] +
        exposure_norm * weights['exposure'] +
        contrast_norm * weights['contrast'] +
        noise_norm * weights['noise'] +
        color_norm * weights['color_balance'] +
        stability_norm * weights['stability']
    )
    
    return round(score, 1)


def get_quality_grade(score: float) -> str:
    """Get letter grade for quality."""
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"


def assess_documentary_suitability(results: Dict) -> Dict:
    """Assess suitability for documentary production."""
    tech = results['technical_specs']
    visual = results['visual_quality']
    
    suitability = {
        'broadcast_ready': False,
        'issues': [],
        'strengths': [],
        'recommendations': []
    }
    
    # Check resolution
    if tech['resolution_category'] in ['4K UHD', 'Full HD (1080p)']:
        suitability['strengths'].append(f"High resolution ({tech['resolution_category']})")
    elif tech['resolution_category'] == 'HD (720p)':
        suitability['recommendations'].append("Resolution is acceptable but not ideal for broadcast")
    else:
        suitability['issues'].append(f"Low resolution ({tech['resolution_category']}) may limit broadcast use")
    
    # Check sharpness
    if visual['sharpness']['assessment'] in ['Excellent', 'Good']:
        suitability['strengths'].append("Sharp, clear footage")
    elif visual['sharpness']['assessment'] in ['Soft', 'Very Soft/Blurry']:
        suitability['issues'].append("Footage is soft/blurry - may need sharpening in post")
    
    # Check exposure
    if visual['exposure']['assessment'] in ['Excellent Exposure', 'Good Exposure']:
        suitability['strengths'].append("Well-exposed footage")
    elif visual['exposure']['assessment'] == 'Poor Exposure':
        suitability['issues'].append("Exposure issues may require color correction")
    
    # Check noise
    if visual['noise_level']['assessment'] in ['Very Clean', 'Low Noise']:
        suitability['strengths'].append("Clean footage with minimal noise")
    elif visual['noise_level']['assessment'] in ['High Noise', 'Very Noisy']:
        suitability['issues'].append("High noise levels may require denoising")
    
    # Check stability
    if visual['stability']['assessment'] in ['Very Stable', 'Stable']:
        suitability['strengths'].append("Stable camera work")
    elif visual['stability']['assessment'] == 'Unstable/Shaky':
        suitability['issues'].append("Camera shake may require stabilization")
    
    # Overall assessment
    if len(suitability['issues']) == 0 and tech['resolution_category'] in ['4K UHD', 'Full HD (1080p)']:
        suitability['broadcast_ready'] = True
    
    return suitability


def print_report(results: Dict):
    """Print formatted analysis report."""
    print("\n" + "="*70)
    print("VIDEO QUALITY ANALYSIS REPORT")
    print("="*70)
    
    print(f"\nFile: {results['file_name']}")
    print(f"Path: {results['file_path']}")
    
    print("\n" + "-"*70)
    print("TECHNICAL SPECIFICATIONS")
    print("-"*70)
    tech = results['technical_specs']
    print(f"Resolution: {tech['resolution']} ({tech['resolution_category']})")
    print(f"Frame Rate: {tech['fps']:.2f} fps")
    print(f"Duration: {tech['duration_seconds']:.2f} seconds")
    print(f"Total Frames: {tech['total_frames']}")
    print(f"Codec: {tech['codec']}")
    if tech.get('bitrate_kbps'):
        print(f"Bitrate: {tech['bitrate_kbps']:.0f} kbps")
    print(f"Pixel Format: {tech['pixel_format']}")
    
    print("\n" + "-"*70)
    print("VISUAL QUALITY METRICS")
    print("-"*70)
    visual = results['visual_quality']
    
    print(f"\nSharpness:")
    print(f"  Score: {visual['sharpness']['mean']:.2f} (std: {visual['sharpness']['std']:.2f})")
    print(f"  Assessment: {visual['sharpness']['assessment']}")
    
    print(f"\nBrightness:")
    print(f"  Mean: {visual['brightness']['mean']:.1f} (0-255 scale)")
    print(f"  Assessment: {visual['brightness']['assessment']}")
    
    print(f"\nContrast:")
    print(f"  Score: {visual['contrast']['mean']:.2f} (std: {visual['contrast']['std']:.2f})")
    print(f"  Assessment: {visual['contrast']['assessment']}")
    
    print(f"\nNoise Level:")
    print(f"  Score: {visual['noise_level']['mean']:.2f} (std: {visual['noise_level']['std']:.2f})")
    print(f"  Assessment: {visual['noise_level']['assessment']}")
    
    print(f"\nExposure:")
    print(f"  Score: {visual['exposure']['mean_score']:.1f}/100")
    print(f"  Assessment: {visual['exposure']['assessment']}")
    
    print(f"\nColor Balance:")
    print(f"  Score: {visual['color_balance']['mean_score']:.1f}/100")
    print(f"  Assessment: {visual['color_balance']['assessment']}")
    
    print(f"\nStability:")
    if visual['stability']['mean_motion']:
        print(f"  Motion Score: {visual['stability']['mean_motion']:.2f}")
        print(f"  Assessment: {visual['stability']['assessment']}")
    else:
        print(f"  Assessment: {visual['stability']['assessment']}")
    
    print("\n" + "-"*70)
    print("OVERALL ASSESSMENT")
    print("-"*70)
    overall = results['overall_assessment']
    print(f"\nQuality Score: {overall['quality_score']}/100")
    print(f"Grade: {overall['grade']}")
    
    print(f"\nDocumentary Suitability:")
    suit = overall['documentary_suitability']
    print(f"  Broadcast Ready: {'Yes' if suit['broadcast_ready'] else 'No'}")
    
    if suit['strengths']:
        print(f"\n  Strengths:")
        for strength in suit['strengths']:
            print(f"    + {strength}")
    
    if suit['issues']:
        print(f"\n  Issues:")
        for issue in suit['issues']:
            print(f"    - {issue}")
    
    if suit['recommendations']:
        print(f"\n  Recommendations:")
        for rec in suit['recommendations']:
            print(f"    • {rec}")
    
    print("\n" + "="*70)


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_video_quality.py <video_path> [sample_frames]")
        print("\nExample:")
        print("  python analyze_video_quality.py C:\\Users\\l-skanatalab\\Desktop\\IMG_3788.mov")
        sys.exit(1)
    
    video_path = sys.argv[1]
    sample_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    try:
        results = analyze_video_quality(video_path, sample_frames)
        print_report(results)
        
        # Save JSON report
        output_path = Path(video_path).with_suffix('.quality_report.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed report saved to: {output_path}")
        
    except Exception as e:
        print(f"Error analyzing video: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


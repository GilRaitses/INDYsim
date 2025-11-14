#!/usr/bin/env python3
"""
Enhanced Video Quality Analyzer with ML Vision Models

Analyzes video files for technical quality, content significance, 
usable seconds, and provides valuation guidance for documentary licensing.
"""

import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import os

# Try to import ML libraries
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: YOLO not available. Install with: pip install ultralytics")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Install with: pip install pillow")


def get_video_metadata(video_path: str) -> Dict:
    """Extract video metadata using ffprobe if available."""
    metadata = {}
    
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
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
        
        if 'format' in data:
            format_info = data['format']
            metadata.update({
                'file_size': int(format_info.get('size', 0)),
                'bitrate_overall': int(format_info.get('bit_rate', 0)) if format_info.get('bit_rate') else None,
                'duration_overall': float(format_info.get('duration', 0)),
            })
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
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
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    total_pixels = gray.size
    overexposed = np.sum(hist[240:]) / total_pixels * 100
    underexposed = np.sum(hist[:16]) / total_pixels * 100
    
    return {
        'overexposed_pct': overexposed,
        'underexposed_pct': underexposed,
        'exposure_score': 100 - (overexposed + underexposed)
    }


def calculate_stability(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Calculate frame-to-frame stability using optical flow."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    return np.mean(magnitude)


def calculate_smoothness(frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, float]:
    """Calculate temporal smoothness and consistency."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
    
    # Frame difference (lower = smoother)
    frame_diff = cv2.absdiff(gray1, gray2)
    mean_diff = np.mean(frame_diff)
    
    # Histogram similarity (higher = smoother)
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    hist_correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Color consistency (if color frames)
    color_consistency = 1.0
    if len(frame1.shape) == 3:
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        hue_diff = np.mean(np.abs(hsv1[:,:,0].astype(float) - hsv2[:,:,0].astype(float)))
        color_consistency = 1.0 - (hue_diff / 180.0)  # Normalize to 0-1
    
    return {
        'mean_frame_diff': float(mean_diff),
        'histogram_similarity': float(hist_correlation),
        'color_consistency': float(color_consistency),
        'smoothness_score': float((hist_correlation + color_consistency) / 2)  # Combined score
    }


def analyze_color_compression(frame: np.ndarray, metadata: Dict) -> Dict[str, float]:
    """Analyze color depth and compression artifacts."""
    if len(frame.shape) != 3:
        return {'color_depth': 'grayscale', 'compression_score': 0}
    
    # Analyze color distribution
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Check for color banding (compression artifact)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    
    # Banding detection: look for quantization in histograms
    def detect_banding(hist):
        """Detect quantization/banding in histogram."""
        non_zero_bins = np.count_nonzero(hist)
        total_bins = len(hist)
        if total_bins > 0:
            utilization = non_zero_bins / total_bins
            # Low utilization suggests compression/banding
            return 1.0 - utilization
        return 0
    
    hue_banding = detect_banding(hist_h)
    saturation_banding = detect_banding(hist_s)
    value_banding = detect_banding(hist_v)
    
    # Overall compression score (lower = less compression artifacts)
    compression_score = 1.0 - np.mean([hue_banding, saturation_banding, value_banding])
    
    # Estimate color depth from pixel format
    pixel_format = metadata.get('pixel_format', 'unknown')
    if '10' in pixel_format or '12' in pixel_format:
        color_depth = '10-12 bit'
    elif '8' in pixel_format or pixel_format == 'unknown':
        color_depth = '8 bit'
    else:
        color_depth = 'unknown'
    
    return {
        'color_depth': color_depth,
        'compression_score': float(compression_score),
        'hue_banding': float(hue_banding),
        'saturation_banding': float(saturation_banding),
        'value_banding': float(value_banding)
    }


def identify_clear_focused_segments(usability_data: List[Dict], fps: float, 
                                    min_duration: float = 1.0) -> List[Dict]:
    """Identify continuous segments that are clear, focused, and usable."""
    clear_segments = []
    current_segment = None
    
    for frame_data in usability_data:
        is_clear = (
            frame_data['is_high_quality'] and
            'blurry' not in frame_data.get('issues', []) and
            'very_blurry' not in frame_data.get('issues', [])
        )
        
        if is_clear:
            if current_segment is None:
                current_segment = {
                    'start': frame_data['timestamp'],
                    'end': frame_data['timestamp'],
                    'start_frame': frame_data['frame_idx'],
                    'end_frame': frame_data['frame_idx'],
                    'quality_score': frame_data['usability_score']
                }
            else:
                current_segment['end'] = frame_data['timestamp']
                current_segment['end_frame'] = frame_data['frame_idx']
                current_segment['quality_score'] = max(
                    current_segment['quality_score'],
                    frame_data['usability_score']
                )
        else:
            if current_segment is not None:
                duration = current_segment['end'] - current_segment['start']
                if duration >= min_duration:
                    current_segment['duration'] = duration
                    current_segment['frame_count'] = (
                        current_segment['end_frame'] - current_segment['start_frame'] + 1
                    )
                    clear_segments.append(current_segment)
                current_segment = None
    
    # Close last segment if exists
    if current_segment is not None:
        duration = current_segment['end'] - current_segment['start']
        if duration >= min_duration:
            current_segment['duration'] = duration
            current_segment['frame_count'] = (
                current_segment['end_frame'] - current_segment['start_frame'] + 1
            )
            clear_segments.append(current_segment)
    
    return clear_segments


def detect_objects_ml(frame: np.ndarray, model=None) -> Dict:
    """Detect objects in frame using YOLO."""
    if not YOLO_AVAILABLE or model is None:
        return {'objects': [], 'has_bird': False, 'has_dog': False, 'confidence': 0}
    
    try:
        results = model(frame, verbose=False)
        detections = []
        has_bird = False
        has_dog = False
        max_confidence = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = model.names[cls]
                
                detections.append({
                    'class': class_name,
                    'confidence': conf,
                    'bbox': box.xyxy[0].cpu().numpy().tolist()
                })
                
                if 'bird' in class_name.lower() or 'owl' in class_name.lower():
                    has_bird = True
                if 'dog' in class_name.lower():
                    has_dog = True
                
                max_confidence = max(max_confidence, conf)
        
        return {
            'objects': detections,
            'has_bird': has_bird,
            'has_dog': has_dog,
            'confidence': max_confidence,
            'object_count': len(detections)
        }
    except Exception as e:
        return {'objects': [], 'has_bird': False, 'has_dog': False, 'confidence': 0, 'error': str(e)}


def assess_frame_usability(frame: np.ndarray, frame_idx: int, fps: float, 
                          sharpness: float, brightness: float, stability: float,
                          object_detection: Dict) -> Dict:
    """Assess if a frame is usable for documentary production."""
    usability_score = 0
    issues = []
    strengths = []
    
    # Sharpness check (0-20 points) - More lenient for archival/doc footage
    if sharpness > 300:
        usability_score += 20
        strengths.append("excellent_sharpness")
    elif sharpness > 150:
        usability_score += 15
    elif sharpness > 50:
        usability_score += 10
    elif sharpness > 20:
        usability_score += 5  # Even soft footage can be usable with post-processing
    else:
        issues.append("very_blurry")
    
    # Brightness check (0-20 points) - More lenient
    if 100 <= brightness <= 180:
        usability_score += 20
    elif 50 <= brightness < 100:
        usability_score += 15  # Underexposed but recoverable
        issues.append("underexposed")
    elif 30 <= brightness < 50:
        usability_score += 8  # Dark but potentially usable
        issues.append("dark")
    elif brightness > 200:
        usability_score += 10
        issues.append("overexposed")
    else:
        issues.append("very_dark")
    
    # Stability check (0-15 points) - Shake can be stabilized in post
    if stability < 2.5:
        usability_score += 15
    elif stability < 5.0:
        usability_score += 10
    elif stability < 8.0:
        usability_score += 5  # Shaky but stabilizable
    else:
        issues.append("very_shaky")
    
    # Content check (0-30 points) - Most important for documentary value
    # For archival/rare footage, assume content value even without ML detection
    content_bonus = 15  # Base value for having footage of the event
    if object_detection.get('has_bird', False):
        content_bonus += 10
        strengths.append("subject_visible")
        if object_detection.get('confidence', 0) > 0.7:
            content_bonus += 5
    
    if object_detection.get('has_dog', False):
        content_bonus += 5
        strengths.append("interaction_moment")
    
    usability_score += content_bonus
    
    # Composition check (0-15 points) - Less critical for archival footage
    h, w = frame.shape[:2]
    center_region = frame[h//4:3*h//4, w//4:3*w//4]
    center_brightness = np.mean(cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY) if len(center_region.shape) == 3 else center_region)
    if abs(center_brightness - brightness) < brightness * 0.3:
        usability_score += 10
    
    # Determine usability threshold - Lowered for archival/doc footage
    # Archival footage has value even if it needs post-processing
    is_usable = usability_score >= 35  # Lowered from 50
    is_high_quality = usability_score >= 60  # Lowered from 70
    
    return {
        'frame_idx': frame_idx,
        'timestamp': frame_idx / fps,
        'usability_score': usability_score,
        'is_usable': is_usable,
        'is_high_quality': is_high_quality,
        'issues': issues,
        'strengths': strengths
    }


def calculate_usable_seconds(usability_data: List[Dict], fps: float, duration: float) -> Dict:
    """Calculate usable seconds and segments."""
    usable_frames = [f for f in usability_data if f['is_usable']]
    high_quality_frames = [f for f in usability_data if f['is_high_quality']]
    
    # Calculate percentage of sampled frames that are usable, then scale to total duration
    total_sampled = len(usability_data)
    if total_sampled > 0:
        usable_percentage = len(usable_frames) / total_sampled
        high_quality_percentage = len(high_quality_frames) / total_sampled
        usable_seconds = duration * usable_percentage
        high_quality_seconds = duration * high_quality_percentage
    else:
        usable_seconds = 0
        high_quality_seconds = 0
    
    # Find continuous usable segments
    segments = []
    current_segment_start = None
    
    for i, frame_data in enumerate(usability_data):
        if frame_data['is_usable']:
            if current_segment_start is None:
                current_segment_start = frame_data['timestamp']
        else:
            if current_segment_start is not None:
                segments.append({
                    'start': current_segment_start,
                    'end': usability_data[i-1]['timestamp'],
                    'duration': usability_data[i-1]['timestamp'] - current_segment_start
                })
                current_segment_start = None
    
    # Close last segment if video ends on usable frame
    if current_segment_start is not None:
        segments.append({
            'start': current_segment_start,
            'end': duration,
            'duration': duration - current_segment_start
        })
    
    return {
        'total_seconds': duration,
        'usable_seconds': usable_seconds,
        'high_quality_seconds': high_quality_seconds,
        'usable_percentage': (usable_seconds / duration * 100) if duration > 0 else 0,
        'high_quality_percentage': (high_quality_seconds / duration * 100) if duration > 0 else 0,
        'usable_segments': segments,
        'segment_count': len(segments)
    }


def calculate_narrative_significance(content_significance: Dict, clear_segments: List[Dict],
                                    video_context: Dict = None) -> Dict:
    """Calculate narrative significance score for documentary storytelling."""
    score = 0
    factors = []
    
    # Rare/unique event (critical moment in story)
    if content_significance.get('is_rare_event', False):
        score += 40
        factors.append("Rare/unique event: First-night landing on Fifth Ave")
    
    # Historical significance (moment Flaco was freed from zoo)
    if video_context and video_context.get('is_historic_moment', True):
        score += 30
        factors.append("Historic moment: First appearance after zoo escape")
    
    # Subject visibility
    if content_significance.get('bird_percentage', 0) > 30:
        score += 15
        factors.append(f"Subject visible in {content_significance['bird_percentage']:.0f}% of footage")
    
    # Clear, focused segments (usable for key moments)
    if clear_segments:
        total_clear_duration = sum(s['duration'] for s in clear_segments)
        if total_clear_duration >= 3.0:
            score += 10
            factors.append(f"{total_clear_duration:.1f}s of clear, focused footage")
        if len(clear_segments) >= 2:
            score += 5
            factors.append(f"Multiple clear segments ({len(clear_segments)}) for editing")
    
    # Interaction moments
    if content_significance.get('has_interaction_moments', False):
        score += 10
        factors.append("Contains interaction moments (owl-dog)")
    
    # Multiple-use potential (key story moment can be reused)
    # This is THE moment - can be used in opening, transitions, key scenes
    score += 20
    factors.append("High reuse potential: Can be used multiple times in doc")
    
    # Normalize to 0-100 scale
    narrative_score = min(100, score)
    
    return {
        'narrative_score': narrative_score,
        'factors': factors,
        'multiple_use_potential': True,  # This moment is reusable
        'story_importance': 'critical' if narrative_score >= 70 else 'high' if narrative_score >= 50 else 'moderate'
    }


def calculate_valuation(usable_data: Dict, quality_score: float, 
                       technical_specs: Dict, content_significance: Dict,
                       narrative_significance: Dict, clear_segments: List[Dict]) -> Dict:
    """Calculate valuation guidance based on quality, usability, and narrative significance."""
    
    # Base rate per usable second (documentary footage)
    base_rate_per_second = 50  # $50 per usable second baseline
    
    # Quality multiplier (0.5x to 2.0x)
    if quality_score >= 80:
        quality_multiplier = 2.0
    elif quality_score >= 70:
        quality_multiplier = 1.5
    elif quality_score >= 60:
        quality_multiplier = 1.0
    elif quality_score >= 50:
        quality_multiplier = 0.75
    else:
        quality_multiplier = 0.5
    
    # Resolution multiplier
    resolution_multiplier = {
        '4K UHD': 1.5,
        'Full HD (1080p)': 1.0,
        'HD (720p)': 0.75,
        'SD (480p)': 0.5
    }.get(technical_specs.get('resolution_category', 'HD (720p)'), 0.75)
    
    # Content significance multiplier
    significance_multiplier = 1.0
    if content_significance.get('has_subject_throughout', False):
        significance_multiplier += 0.3
    if content_significance.get('has_interaction_moments', False):
        significance_multiplier += 0.2
    if content_significance.get('is_rare_event', True):  # First-night Flaco
        significance_multiplier += 0.5
    
    # Narrative significance multiplier (story importance)
    narrative_multiplier = 1.0 + (narrative_significance.get('narrative_score', 0) / 100) * 1.5
    # Critical story moments get 2.5x multiplier
    if narrative_significance.get('story_importance') == 'critical':
        narrative_multiplier = 2.5
    
    # Multiple-use factor (if footage can be reused multiple times)
    multiple_use_multiplier = 1.0
    if narrative_significance.get('multiple_use_potential', False):
        # Can be used in opening, transitions, key scenes - add 50% value
        multiple_use_multiplier = 1.5
    
    # Calculate per-second value
    rate_per_usable_second = (
        base_rate_per_second * 
        quality_multiplier * 
        resolution_multiplier * 
        significance_multiplier *
        narrative_multiplier *
        multiple_use_multiplier
    )
    
    # Calculate total value ranges
    usable_seconds = usable_data['usable_seconds']
    high_quality_seconds = usable_data['high_quality_seconds']
    
    # Conservative estimate (all usable seconds)
    conservative_value = usable_seconds * rate_per_usable_second
    
    # Optimistic estimate (high quality seconds at premium)
    premium_rate = rate_per_usable_second * 1.5
    optimistic_value = high_quality_seconds * premium_rate + (usable_seconds - high_quality_seconds) * rate_per_usable_second
    
    # Per-second breakdown
    per_second_breakdown = {
        'base_rate': base_rate_per_second,
        'quality_multiplier': quality_multiplier,
        'resolution_multiplier': resolution_multiplier,
        'significance_multiplier': significance_multiplier,
        'narrative_multiplier': narrative_multiplier,
        'multiple_use_multiplier': multiple_use_multiplier,
        'final_rate_per_second': rate_per_usable_second
    }
    
    return {
        'rate_per_usable_second': rate_per_usable_second,
        'conservative_estimate': conservative_value,
        'optimistic_estimate': optimistic_value,
        'recommended_range': f"${conservative_value:.0f} - ${optimistic_value:.0f}",
        'per_second_breakdown': per_second_breakdown,
        'valuation_notes': generate_valuation_notes(usable_data, quality_score, content_significance)
    }


def generate_valuation_notes(usable_data: Dict, quality_score: float, content_significance: Dict) -> List[str]:
    """Generate valuation guidance notes."""
    notes = []
    
    usable_pct = usable_data['usable_percentage']
    
    if usable_pct >= 80:
        notes.append(f"High usability ({usable_pct:.1f}% usable) - most of the clip is production-ready")
    elif usable_pct >= 60:
        notes.append(f"Moderate usability ({usable_pct:.1f}% usable) - substantial usable content")
    elif usable_pct >= 40:
        notes.append(f"Limited usability ({usable_pct:.1f}% usable) - requires selective editing")
    else:
        notes.append(f"Low usability ({usable_pct:.1f}% usable) - primarily archival value")
    
    if content_significance.get('is_rare_event', False):
        notes.append("Rare/unique event adds significant archival and narrative value")
    
    if usable_data['segment_count'] > 1:
        notes.append(f"Multiple usable segments ({usable_data['segment_count']}) provide editing flexibility")
    
    if quality_score < 60:
        notes.append("Lower technical quality may require post-processing, factor into negotiations")
    
    return notes


def analyze_video_quality_ml(video_path: str, sample_frames: int = None, use_ml: bool = True) -> Dict:
    """
    Analyze video quality with ML vision models for content analysis.
    
    Args:
        video_path: Path to video file
        sample_frames: Number of frames to sample (None = analyze all frames)
        use_ml: Whether to use ML models for object detection
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    print(f"Analyzing video: {video_path}")
    print(f"File size: {video_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Load ML model if available
    model = None
    if use_ml and YOLO_AVAILABLE:
        try:
            print("Loading YOLO model for object detection...")
            model = YOLO('yolov8n.pt')  # Nano model for speed
            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load YOLO model: {e}")
            model = None
    
    # Get metadata
    metadata = get_video_metadata(str(video_path))
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
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
    
    # Determine frame sampling
    if sample_frames is None:
        # Analyze every Nth frame for full analysis
        frame_step = max(1, int(fps / 2))  # Sample 2 frames per second
        frame_indices = list(range(0, total_frames, frame_step))
    else:
        frame_indices = np.linspace(0, total_frames - 1, min(sample_frames, total_frames), dtype=int)
    
    print(f"\nAnalyzing {len(frame_indices)} frames...")
    
    # Analysis arrays
    sharpness_scores = []
    brightness_scores = []
    contrast_scores = []
    noise_scores = []
    exposure_scores = []
    stability_scores = []
    smoothness_scores = []
    color_compression_scores = []
    usability_data = []
    
    # Content significance tracking
    frames_with_bird = 0
    frames_with_dog = 0
    frames_with_interaction = 0
    
    prev_frame = None
    first_frame = True
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Calculate technical metrics
        sharpness = calculate_sharpness(frame)
        brightness = calculate_brightness(frame)
        contrast = calculate_contrast(frame)
        noise = calculate_noise_level(frame)
        exposure = calculate_exposure_quality(frame)
        
        sharpness_scores.append(sharpness)
        brightness_scores.append(brightness)
        contrast_scores.append(contrast)
        noise_scores.append(noise)
        exposure_scores.append(exposure['exposure_score'])
        
        # Calculate stability and smoothness
        stability = 0
        smoothness = None
        if prev_frame is not None:
            stability = calculate_stability(prev_frame, frame)
            stability_scores.append(stability)
            smoothness = calculate_smoothness(prev_frame, frame)
            smoothness_scores.append(smoothness['smoothness_score'])
        else:
            # First frame - assume moderate stability for scoring
            stability = 3.0
        
        # Analyze color compression (store first one for color_depth)
        color_compression = analyze_color_compression(frame, metadata)
        color_compression_scores.append(color_compression['compression_score'])
        if i == 0:  # Store first frame's color depth
            first_color_depth = color_compression.get('color_depth', 'unknown')
        
        # ML object detection
        object_detection = detect_objects_ml(frame, model) if use_ml else {}
        
        if object_detection.get('has_bird', False):
            frames_with_bird += 1
        if object_detection.get('has_dog', False):
            frames_with_dog += 1
        if object_detection.get('has_bird', False) and object_detection.get('has_dog', False):
            frames_with_interaction += 1
        
        # Assess frame usability
        frame_usability = assess_frame_usability(
            frame, frame_idx, fps, sharpness, brightness, stability, object_detection
        )
        usability_data.append(frame_usability)
        
        prev_frame = frame
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(frame_indices)} frames...")
    
    cap.release()
    
    # Calculate usable seconds
    usable_analysis = calculate_usable_seconds(usability_data, fps, duration)
    
    # Identify clear, focused segments
    clear_segments = identify_clear_focused_segments(usability_data, fps, min_duration=1.0)
    
    # Content significance
    content_significance = {
        'frames_analyzed': len(frame_indices),
        'frames_with_bird': frames_with_bird,
        'frames_with_dog': frames_with_dog,
        'frames_with_interaction': frames_with_interaction,
        'bird_percentage': (frames_with_bird / len(frame_indices) * 100) if frame_indices else 0,
        'has_subject_throughout': frames_with_bird / len(frame_indices) > 0.5 if frame_indices else False,
        'has_interaction_moments': frames_with_interaction > 0,
        'is_rare_event': True  # First-night Flaco sighting
    }
    
    # Calculate narrative significance
    video_context = {
        'is_historic_moment': True,  # First landing on Fifth Ave after zoo escape
        'event_description': 'First-night landing on Fifth Ave after being freed from zoo'
    }
    narrative_significance = calculate_narrative_significance(
        content_significance, clear_segments, video_context
    )
    
    # Calculate overall quality score (simplified version)
    quality_score = calculate_overall_quality_score_simple(
        sharpness_scores, exposure_scores, contrast_scores, noise_scores, stability_scores
    )
    
    # Calculate valuation
    technical_specs = {
        'resolution': f"{width}x{height}",
        'resolution_category': get_resolution_category(width, height),
        'fps': fps,
        'duration_seconds': duration
    }
    
    valuation = calculate_valuation(
        usable_analysis, quality_score, technical_specs, 
        content_significance, narrative_significance, clear_segments
    )
    
    # Build results
    results = {
        'file_path': str(video_path),
        'file_name': video_path.name,
        'metadata': metadata,
        'technical_specs': technical_specs,
        'visual_quality': {
            'sharpness_mean': float(np.mean(sharpness_scores)),
            'brightness_mean': float(np.mean(brightness_scores)),
            'contrast_mean': float(np.mean(contrast_scores)),
            'noise_mean': float(np.mean(noise_scores)),
            'exposure_mean': float(np.mean(exposure_scores)),
            'stability_mean': float(np.mean(stability_scores)) if stability_scores else None,
            'smoothness_mean': float(np.mean(smoothness_scores)) if smoothness_scores else None,
            'color_compression_mean': float(np.mean(color_compression_scores)) if color_compression_scores else None,
            'color_depth': first_color_depth if 'first_color_depth' in locals() else 'unknown'
        },
        'content_significance': content_significance,
        'narrative_significance': narrative_significance,
        'clear_focused_segments': clear_segments,
        'usable_analysis': usable_analysis,
        'quality_score': quality_score,
        'valuation': valuation
    }
    
    return results


def calculate_overall_quality_score_simple(sharpness, exposure, contrast, noise, stability):
    """Calculate simplified quality score."""
    sharpness_norm = min(100, (np.mean(sharpness) / 500) * 100)
    exposure_norm = np.mean(exposure)
    contrast_norm = min(100, (np.mean(contrast) / 40) * 100)
    noise_norm = max(0, 100 - (np.mean(noise) / 50) * 100)
    stability_norm = max(0, 100 - (np.mean(stability) / 5) * 100) if stability else 50
    
    weights = [0.25, 0.20, 0.15, 0.15, 0.25]
    score = (
        sharpness_norm * weights[0] +
        exposure_norm * weights[1] +
        contrast_norm * weights[2] +
        noise_norm * weights[3] +
        stability_norm * weights[4]
    )
    return round(score, 1)


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


def print_ml_report(results: Dict):
    """Print comprehensive ML-enhanced analysis report."""
    print("\n" + "="*80)
    print("ENHANCED VIDEO QUALITY ANALYSIS WITH ML CONTENT ASSESSMENT")
    print("="*80)
    
    print(f"\nFile: {results['file_name']}")
    print(f"Path: {results['file_path']}")
    
    # Technical specs
    print("\n" + "-"*80)
    print("TECHNICAL SPECIFICATIONS")
    print("-"*80)
    tech = results['technical_specs']
    print(f"Resolution: {tech['resolution']} ({tech['resolution_category']})")
    print(f"Frame Rate: {tech['fps']:.2f} fps")
    print(f"Duration: {tech['duration_seconds']:.2f} seconds")
    
    # Visual quality metrics
    print("\n" + "-"*80)
    print("VISUAL QUALITY METRICS")
    print("-"*80)
    visual = results['visual_quality']
    print(f"Sharpness: {visual['sharpness_mean']:.2f}")
    print(f"Brightness: {visual['brightness_mean']:.1f}/255")
    print(f"Contrast: {visual['contrast_mean']:.2f}")
    print(f"Noise Level: {visual['noise_mean']:.2f}")
    print(f"Exposure: {visual['exposure_mean']:.1f}/100")
    print(f"Stability: {visual['stability_mean']:.2f}" if visual['stability_mean'] else "Stability: N/A")
    if visual.get('smoothness_mean'):
        print(f"Smoothness: {visual['smoothness_mean']:.3f} (higher = smoother)")
    if visual.get('color_compression_mean'):
        print(f"Color Compression Score: {visual['color_compression_mean']:.3f} (higher = less compression)")
    if visual.get('color_depth'):
        print(f"Color Depth: {visual['color_depth']}")
    
    # Content significance
    print("\n" + "-"*80)
    print("CONTENT SIGNIFICANCE ANALYSIS")
    print("-"*80)
    content = results['content_significance']
    print(f"Frames analyzed: {content['frames_analyzed']}")
    print(f"Frames with subject (bird/owl): {content['frames_with_bird']} ({content['bird_percentage']:.1f}%)")
    print(f"Frames with dog: {content['frames_with_dog']}")
    print(f"Frames with interaction: {content['frames_with_interaction']}")
    print(f"Subject visible throughout: {'Yes' if content['has_subject_throughout'] else 'No'}")
    print(f"Interaction moments detected: {'Yes' if content['has_interaction_moments'] else 'No'}")
    print(f"Rare/unique event: {'Yes' if content['is_rare_event'] else 'No'}")
    
    # Narrative significance
    print("\n" + "-"*80)
    print("NARRATIVE SIGNIFICANCE (STORY VALUE)")
    print("-"*80)
    narrative = results.get('narrative_significance', {})
    print(f"Narrative Score: {narrative.get('narrative_score', 0)}/100")
    print(f"Story Importance: {narrative.get('story_importance', 'unknown').upper()}")
    print(f"Multiple-Use Potential: {'Yes' if narrative.get('multiple_use_potential', False) else 'No'}")
    if narrative.get('factors'):
        print("\nSignificance Factors:")
        for factor in narrative['factors']:
            print(f"  • {factor}")
    
    # Clear focused segments
    print("\n" + "-"*80)
    print("CLEAR, FOCUSED SEGMENTS")
    print("-"*80)
    clear_segments = results.get('clear_focused_segments', [])
    if clear_segments:
        total_clear_duration = sum(s['duration'] for s in clear_segments)
        print(f"Total clear/focused duration: {total_clear_duration:.2f} seconds")
        print(f"Number of clear segments: {len(clear_segments)}")
        print("\nClear segments:")
        for i, seg in enumerate(clear_segments[:10], 1):
            print(f"  Segment {i}: {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s)")
            print(f"    Quality score: {seg.get('quality_score', 0):.1f}")
        if len(clear_segments) > 10:
            print(f"  ... and {len(clear_segments) - 10} more segments")
    else:
        print("No continuous clear/focused segments found (minimum 1 second)")
    
    # Usable analysis
    print("\n" + "-"*80)
    print("USABILITY ANALYSIS (PER SECOND)")
    print("-"*80)
    usable = results['usable_analysis']
    print(f"Total duration: {usable['total_seconds']:.2f} seconds")
    print(f"Usable seconds: {usable['usable_seconds']:.2f} seconds ({usable['usable_percentage']:.1f}%)")
    print(f"High quality seconds: {usable['high_quality_seconds']:.2f} seconds ({usable['high_quality_percentage']:.1f}%)")
    print(f"Usable segments: {usable['segment_count']}")
    
    if usable['usable_segments']:
        print("\nUsable segments:")
        for i, seg in enumerate(usable['usable_segments'][:10], 1):  # Show first 10
            print(f"  Segment {i}: {seg['start']:.2f}s - {seg['end']:.2f}s ({seg['duration']:.2f}s)")
        if len(usable['usable_segments']) > 10:
            print(f"  ... and {len(usable['usable_segments']) - 10} more segments")
    
    # Quality score
    print("\n" + "-"*80)
    print("OVERALL QUALITY SCORE")
    print("-"*80)
    print(f"Quality Score: {results['quality_score']}/100")
    
    # Valuation
    print("\n" + "-"*80)
    print("VALUATION GUIDANCE")
    print("-"*80)
    val = results['valuation']
    breakdown = val['per_second_breakdown']
    print(f"\nRate per usable second: ${breakdown['final_rate_per_second']:.2f}")
    print(f"  Base rate: ${breakdown['base_rate']:.0f}")
    print(f"  Quality multiplier: {breakdown['quality_multiplier']:.2f}x")
    print(f"  Resolution multiplier: {breakdown['resolution_multiplier']:.2f}x")
    print(f"  Content significance multiplier: {breakdown['significance_multiplier']:.2f}x")
    print(f"  Narrative significance multiplier: {breakdown.get('narrative_multiplier', 1.0):.2f}x")
    print(f"  Multiple-use multiplier: {breakdown.get('multiple_use_multiplier', 1.0):.2f}x")
    
    print(f"\nEstimated Value Range: {val['recommended_range']}")
    print(f"  Conservative (all usable): ${val['conservative_estimate']:.0f}")
    print(f"  Optimistic (premium segments): ${val['optimistic_estimate']:.0f}")
    
    print("\nValuation Notes:")
    for note in val['valuation_notes']:
        print(f"  • {note}")
    
    print("\n" + "="*80)
    print("\nINTERPRETATION GUIDE:")
    print("-"*80)
    print("""
USABLE SECONDS: The portion of footage that meets minimum quality thresholds
  for documentary use. Higher percentage = more value.

PER-SECOND VALUE: Based on:
  • Base market rate for documentary footage
  • Technical quality (sharpness, exposure, stability)
  • Resolution (4K > 1080p > 720p)
  • Content significance (rare events, subject visibility)

VALUATION RANGE:
  • Conservative: Assumes all usable seconds at standard rate
  • Optimistic: Premium rate for high-quality segments
  
For licensing negotiations:
  • Start with conservative estimate as floor
  • Use optimistic estimate for exclusive/premium use
  • Factor in post-production needs (stabilization, sharpening)
  • Consider archival/narrative value of rare events
    """)


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_video_quality_ml.py <video_path> [sample_frames] [--no-ml]")
        print("\nExample:")
        print("  python analyze_video_quality_ml.py C:\\Users\\l-skanatalab\\Desktop\\IMG_3788.mov")
        print("  python analyze_video_quality_ml.py video.mov 60  # Sample 60 frames")
        print("  python analyze_video_quality_ml.py video.mov --no-ml  # Skip ML analysis")
        sys.exit(1)
    
    video_path = sys.argv[1]
    sample_frames = None
    use_ml = True
    
    if len(sys.argv) > 2:
        if sys.argv[2] == '--no-ml':
            use_ml = False
        else:
            try:
                sample_frames = int(sys.argv[2])
            except ValueError:
                if sys.argv[2] == '--no-ml':
                    use_ml = False
    
    if '--no-ml' in sys.argv:
        use_ml = False
    
    try:
        results = analyze_video_quality_ml(video_path, sample_frames, use_ml)
        print_ml_report(results)
        
        # Save JSON report
        output_path = Path(video_path).with_suffix('.ml_quality_report.json')
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


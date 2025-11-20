#!/usr/bin/env python3
"""Extract full technical specifications from video files"""

import subprocess
import json
import sys
from pathlib import Path

def get_full_specs(video_path):
    """Get full technical specifications using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        format_info = data.get('format', {})
        video_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), {})
        audio_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'audio'), None)
        
        # File size
        file_size_bytes = int(format_info.get('size', 0))
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Bitrate
        bitrate = format_info.get('bit_rate', '0')
        bitrate_mbps = int(bitrate) / 1000000 if bitrate != '0' else None
        
        specs = {
            'file_size_mb': round(file_size_mb, 2),
            'bitrate_mbps': round(bitrate_mbps, 2) if bitrate_mbps else None,
            'codec': video_stream.get('codec_name', 'Unknown'),
            'codec_long': video_stream.get('codec_long_name', 'Unknown'),
            'pixel_format': video_stream.get('pix_fmt', 'Unknown'),
            'color_space': video_stream.get('color_space', 'Unknown'),
            'color_range': video_stream.get('color_range', 'Unknown'),
            'profile': video_stream.get('profile', 'Unknown'),
            'level': video_stream.get('level', 'Unknown'),
            'has_audio': audio_stream is not None,
            'audio_codec': audio_stream.get('codec_name', 'None') if audio_stream else 'None',
            'audio_sample_rate': audio_stream.get('sample_rate', 'None') if audio_stream else 'None',
            'audio_channels': audio_stream.get('channels', 'None') if audio_stream else 'None',
        }
        
        return specs
    except Exception as e:
        print(f"Error extracting specs: {e}", file=sys.stderr)
        return None

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python get_video_specs.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    specs = get_full_specs(video_path)
    
    if specs:
        print(json.dumps(specs, indent=2))
    else:
        sys.exit(1)



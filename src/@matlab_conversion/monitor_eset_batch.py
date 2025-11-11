#!/usr/bin/env python3
"""
ESET Batch Processing Monitor with Aurora Animation

Features:
- Each bubble/sphere represents an ESET folder
- Currently processing ESET is in center, largest, revolving, changing colors
- Blinking fraction proportional to tracks processed/total tracks
- As ESETs complete, they dissolve into aurora-colored wind/dust clouds
- Aurora curtain grows as ESETs are processed

Author: mechanobro
Date: 2025-11-11
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ANSI color codes for aurora theme
class Colors:
    # Aurora colors - greens, blues, purples, pinks
    AURORA_GREEN = '\033[38;5;46m'      # Bright green
    AURORA_LIGHT_GREEN = '\033[38;5;82m'  # Light green
    AURORA_BLUE = '\033[38;5;33m'       # Sky blue
    AURORA_BRIGHT_BLUE = '\033[38;5;39m'  # Bright blue
    AURORA_CYAN = '\033[38;5;51m'       # Bright cyan
    AURORA_PURPLE = '\033[38;5;93m'     # Bright purple
    AURORA_MAGENTA = '\033[38;5;165m'   # Pink-purple
    AURORA_PINK = '\033[38;5;177m'      # Pink
    
    # Dimmed versions
    AURORA_DIM_GREEN = '\033[2;38;5;46m'
    AURORA_DIM_BLUE = '\033[2;38;5;33m'
    AURORA_DIM_PURPLE = '\033[2;38;5;93m'
    
    # Status colors
    PROCESSING = '\033[1;38;5;51m'      # Bright cyan (processing)
    COMPLETE = '\033[1;38;5;46m'        # Bright green (complete)
    PENDING = '\033[38;5;93m'           # Purple (pending)
    
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CLEAR = '\033[2J\033[H'  # Clear screen and move to top


@dataclass
class Speck:
    """A small speck for the aurora curtain."""
    x: float
    y: float
    vx: float
    vy: float
    size: float
    color: str
    char: str
    lifetime: float
    
    def update(self, dt: float, terminal_width: int, terminal_height: int, wind_x: float, wind_y: float):
        """Update speck physics with wind."""
        wind_strength = 1.2
        self.vx += wind_x * wind_strength * dt
        self.vy += wind_y * wind_strength * dt
        
        damping = 0.92
        self.vx *= damping
        self.vy *= damping
        
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Wrap around edges
        if self.x < 0:
            self.x = terminal_width
        elif self.x > terminal_width:
            self.x = 0
        
        if self.y < 0:
            self.y = terminal_height
        elif self.y > terminal_height:
            self.y = 0
        
        self.lifetime += dt


@dataclass
class ESETBubble:
    """A bubble/sphere representing an ESET."""
    eset_name: str
    x: float
    y: float
    size: float
    current_size: float
    vx: float
    vy: float
    rotation: float
    rotation_speed: float
    orbit_center_x: float
    orbit_center_y: float
    orbit_radius: float
    orbit_angle: float
    orbit_speed: float
    color: str
    is_processing: bool
    is_complete: bool
    tracks_processed: int
    total_tracks: int
    blink_phase: float
    dissolve_phase: float  # 0.0 = solid, 1.0 = fully dissolved
    
    def update(self, dt: float, terminal_width: int, terminal_height: int, wind_x: float = 0.0, wind_y: float = 0.0):
        """Update bubble physics."""
        # If processing, orbit around center
        if self.is_processing and self.orbit_radius > 0:
            self.orbit_angle += self.orbit_speed * dt
            self.x = self.orbit_center_x + math.cos(self.orbit_angle) * self.orbit_radius
            self.y = self.orbit_center_y + math.sin(self.orbit_angle) * self.orbit_radius
        
        # Apply wind
        wind_strength = 0.8 if not self.is_processing else 0.3  # Processing bubbles less affected
        self.vx += wind_x * wind_strength * dt
        self.vy += wind_y * wind_strength * dt
        
        damping = 0.95
        self.vx *= damping
        self.vy *= damping
        
        if not self.is_processing:  # Only free float if not processing
            self.x += self.vx * dt
            self.y += self.vy * dt
        
        # Bounce off edges
        if self.x < self.current_size:
            self.x = self.current_size
            self.vx = abs(self.vx) * 0.7
        elif self.x > terminal_width - self.current_size:
            self.x = terminal_width - self.current_size
            self.vx = -abs(self.vx) * 0.7
        
        if self.y < self.current_size:
            self.y = self.current_size
            self.vy = abs(self.vy) * 0.7
        elif self.y > terminal_height - self.current_size:
            self.y = terminal_height - self.current_size
            self.vy = -abs(self.vy) * 0.7
        
        # Rotation
        rotation_from_wind = (wind_x + wind_y) * 0.3
        self.rotation_speed += rotation_from_wind * dt * 0.1
        self.rotation += self.rotation_speed * dt
        
        # Blink phase (for processing bubbles)
        if self.is_processing:
            self.blink_phase += dt * 3.0  # Blink speed
        
        # Dissolve phase (for complete bubbles)
        if self.is_complete:
            self.dissolve_phase = min(1.0, self.dissolve_phase + dt * 0.5)  # Dissolve speed
    
    def get_char(self, dx: int, dy: int) -> str:
        """Get character at offset from bubble center."""
        dist = math.sqrt(dx*dx + dy*dy)
        
        # If dissolving, make it more transparent
        opacity = 1.0 - self.dissolve_phase
        
        if dist > self.current_size + 0.5:
            return ' '
        
        edge_dist = abs(dist - self.current_size)
        
        # Blinking effect for processing bubbles
        blink_opacity = 1.0
        if self.is_processing:
            # Blink based on progress fraction
            progress_fraction = self.tracks_processed / max(self.total_tracks, 1)
            blink_frequency = 2.0 + progress_fraction * 3.0  # Faster blink as progress increases
            blink_opacity = 0.5 + 0.5 * math.sin(self.blink_phase * blink_frequency)
        
        final_opacity = opacity * blink_opacity
        
        if final_opacity < 0.3:
            return ' '
        
        if edge_dist < 0.3:
            return '○'
        elif edge_dist < 0.6:
            return '◉'
        elif dist < self.current_size * 0.5:
            return '◯'
        elif dist < self.current_size * 0.8:
            return '·'
        else:
            return ' '


class SpeckSystem:
    """Manages aurora curtain specks."""
    
    def __init__(self, terminal_width: int, terminal_height: int):
        self.terminal_width = terminal_width
        self.terminal_height = terminal_height
        self.specks: list[Speck] = []
        self.wind_time = 0.0
        self.wind_gust_phase = random.uniform(0, 2 * math.pi)
        
        self.speck_chars = ['·', '•', '▪', '▫', '○', '◯', '✦', '✧', '✩', '✪']
        self.speck_colors = [
            Colors.AURORA_GREEN,
            Colors.AURORA_LIGHT_GREEN,
            Colors.AURORA_BLUE,
            Colors.AURORA_BRIGHT_BLUE,
            Colors.AURORA_CYAN,
            Colors.AURORA_PURPLE,
            Colors.AURORA_MAGENTA,
            Colors.AURORA_PINK,
            Colors.AURORA_DIM_GREEN,
            Colors.AURORA_DIM_BLUE,
            Colors.AURORA_DIM_PURPLE,
        ]
        
        # Dense curtain
        self.target_speck_count = terminal_width * terminal_height // 2
        for _ in range(self.target_speck_count):
            self.spawn_speck()
    
    def spawn_speck(self, x: float | None = None, y: float | None = None):
        """Spawn a new speck with aurora colors."""
        if x is None:
            x = random.uniform(0, self.terminal_width)
        if y is None:
            y = random.uniform(0, self.terminal_height)
        
        vx = random.uniform(-0.3, 0.3)
        vy = random.uniform(-0.2, 0.2)
        size = random.uniform(0.3, 0.6)
        
        char_idx = random.randint(0, len(self.speck_chars) - 1)
        color_idx = random.randint(0, len(self.speck_colors) - 1)
        
        speck = Speck(
            x=x, y=y,
            vx=vx, vy=vy,
            size=size,
            color=self.speck_colors[color_idx],
            char=self.speck_chars[char_idx],
            lifetime=0.0,
        )
        self.specks.append(speck)
    
    def get_wind_at_position(self, x: float, y: float, t: float) -> tuple[float, float]:
        """Calculate wind vector."""
        base_angle = math.sin(t * 0.1) * 2 * math.pi
        base_strength = 0.5 + 0.3 * math.sin(t * 0.15)
        wind_x = math.cos(base_angle) * base_strength
        wind_y = math.sin(base_angle) * base_strength
        
        gust_x = math.sin(x * 0.1 + t * 0.3) * 0.4
        gust_y = math.cos(y * 0.1 + t * 0.25) * 0.4
        
        gust_phase = math.sin(t * 0.05 + self.wind_gust_phase)
        if abs(gust_phase) > 0.8:
            gust_strength = abs(gust_phase) * 1.5
            gust_x += math.cos(t * 0.2) * gust_strength
            gust_y += math.sin(t * 0.18) * gust_strength
        
        return wind_x + gust_x, wind_y + gust_y
    
    def update(self, dt: float, progress: float):
        """Update all specks."""
        self.wind_time += dt
        
        for speck in self.specks:
            wind_x, wind_y = self.get_wind_at_position(
                speck.x, speck.y, self.wind_time
            )
            speck.update(dt, self.terminal_width, self.terminal_height, wind_x, wind_y)
        
        # Maintain speck count
        while len(self.specks) < self.target_speck_count:
            self.spawn_speck()
        
        # Add specks to progress area
        progress_width = int((progress / 100.0) * self.terminal_width)
        if progress_width > 0:
            progress_area_ratio = progress_width / self.terminal_width
            target_progress_specks = int(self.target_speck_count * progress_area_ratio)
            specks_in_progress = sum(1 for s in self.specks if int(s.x) < progress_width)
            
            if specks_in_progress < target_progress_specks:
                needed = target_progress_specks - specks_in_progress
                for _ in range(min(needed, 5)):
                    x = random.uniform(0, progress_width)
                    y = random.uniform(0, self.terminal_height)
                    self.spawn_speck(x, y)
    
    def render(self, progress: float) -> tuple[list[str], list[str]]:
        """Render specks as background and progress curtain."""
        bg_canvas = [[' ' for _ in range(self.terminal_width)] 
                    for _ in range(self.terminal_height)]
        progress_canvas = [[' ' for _ in range(self.terminal_width)] 
                          for _ in range(self.terminal_height)]
        
        progress_width = int((progress / 100.0) * self.terminal_width)
        
        for speck in self.specks:
            x = int(speck.x)
            y = int(speck.y)
            
            if 0 <= x < self.terminal_width and 0 <= y < self.terminal_height:
                bg_canvas[y][x] = speck.color + speck.char + Colors.RESET
                
                if x < progress_width:
                    progress_canvas[y][x] = speck.color + speck.char + Colors.RESET
        
        return [''.join(row) for row in bg_canvas], [''.join(row) for row in progress_canvas]


class ESETBubbleSystem:
    """Manages ESET bubbles."""
    
    def __init__(self, terminal_width: int, terminal_height: int):
        self.terminal_width = terminal_width
        self.terminal_height = terminal_height
        self.bubbles: list[ESETBubble] = []
        self.center_x = terminal_width // 2
        self.center_y = terminal_height // 2
        self.wind_time = 0.0
        self.wind_gust_phase = random.uniform(0, 2 * math.pi)
    
    def add_eset(self, eset_name: str, total_tracks: int, is_processing: bool = False):
        """Add an ESET bubble."""
        # Check if already exists
        for bubble in self.bubbles:
            if bubble.eset_name == eset_name:
                bubble.is_processing = is_processing
                bubble.total_tracks = total_tracks
                return
        
        # Processing bubbles go to center, others float around
        if is_processing:
            x = self.center_x
            y = self.center_y
            size = 8.0  # Large
            orbit_radius = 5.0
            orbit_speed = 0.8
            color = Colors.PROCESSING
        else:
            x = random.uniform(10, self.terminal_width - 10)
            y = random.uniform(5, self.terminal_height - 5)
            size = random.uniform(3.0, 5.0)
            orbit_radius = 0.0
            orbit_speed = 0.0
            color = Colors.PENDING
        
        bubble = ESETBubble(
            eset_name=eset_name,
            x=x, y=y,
            size=size,
            current_size=size,
            vx=random.uniform(-0.3, 0.3),
            vy=random.uniform(-0.2, 0.2),
            rotation=random.uniform(0, 2 * math.pi),
            rotation_speed=random.uniform(-0.3, 0.3),
            orbit_center_x=self.center_x,
            orbit_center_y=self.center_y,
            orbit_radius=orbit_radius,
            orbit_angle=random.uniform(0, 2 * math.pi),
            orbit_speed=orbit_speed,
            color=color,
            is_processing=is_processing,
            is_complete=False,
            tracks_processed=0,
            total_tracks=total_tracks,
            blink_phase=0.0,
            dissolve_phase=0.0,
        )
        self.bubbles.append(bubble)
    
    def update_eset_progress(self, eset_name: str, tracks_processed: int):
        """Update progress for an ESET."""
        for bubble in self.bubbles:
            if bubble.eset_name == eset_name:
                bubble.tracks_processed = tracks_processed
                break
    
    def mark_complete(self, eset_name: str):
        """Mark an ESET as complete (starts dissolving)."""
        for bubble in self.bubbles:
            if bubble.eset_name == eset_name:
                bubble.is_complete = True
                bubble.is_processing = False
                bubble.color = Colors.COMPLETE
                # Start floating away
                bubble.orbit_radius = 0.0
                bubble.orbit_speed = 0.0
                break
    
    def get_wind_at_position(self, x: float, y: float, t: float) -> tuple[float, float]:
        """Calculate wind vector."""
        base_angle = math.sin(t * 0.1) * 2 * math.pi
        base_strength = 0.5 + 0.3 * math.sin(t * 0.15)
        wind_x = math.cos(base_angle) * base_strength
        wind_y = math.sin(base_angle) * base_strength
        
        gust_x = math.sin(x * 0.1 + t * 0.3) * 0.4
        gust_y = math.cos(y * 0.1 + t * 0.25) * 0.4
        
        return wind_x + gust_x, wind_y + gust_y
    
    def update(self, dt: float):
        """Update all bubbles."""
        self.wind_time += dt
        
        for bubble in self.bubbles:
            wind_x, wind_y = self.get_wind_at_position(
                bubble.x, bubble.y, self.wind_time
            )
            bubble.update(dt, self.terminal_width, self.terminal_height, wind_x, wind_y)
        
        # Remove fully dissolved bubbles
        self.bubbles = [b for b in self.bubbles if b.dissolve_phase < 1.0]
    
    def render(self) -> list[str]:
        """Render all bubbles."""
        canvas = [[' ' for _ in range(self.terminal_width)] 
                 for _ in range(self.terminal_height)]
        
        # Render processing bubble last (on top)
        processing_bubbles = [b for b in self.bubbles if b.is_processing]
        other_bubbles = [b for b in self.bubbles if not b.is_processing]
        
        for bubble in other_bubbles + processing_bubbles:
            radius = int(bubble.current_size) + 1
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x = int(bubble.x) + dx
                    y = int(bubble.y) + dy
                    
                    if 0 <= x < self.terminal_width and 0 <= y < self.terminal_height:
                        char = bubble.get_char(dx, dy)
                        if char != ' ':
                            canvas[y][x] = bubble.color + char + Colors.RESET
        
        return [''.join(row) for row in canvas]


def count_tracks_in_eset(eset_dir: Path) -> int:
    """Count total tracks in an ESET by looking in matfiles/tracks directories."""
    matfiles_dir = eset_dir / "matfiles"
    if not matfiles_dir.exists():
        return 0
    
    total_tracks = 0
    
    # Look for track directories: {GENOTYPE}_{TIMESTAMP} - tracks
    for track_dir in matfiles_dir.glob("* - tracks"):
        if track_dir.is_dir():
            # Count .mat files in track directory (each represents a track)
            track_files = list(track_dir.glob("*.mat"))
            total_tracks += len(track_files)
    
    return total_tracks


def get_eset_status(genotype_dir: Path, output_dir: Path) -> Dict[str, Dict]:
    """Get status of all ESETs in a genotype directory."""
    status = {}
    
    # Find all ESET folders
    for eset_dir in genotype_dir.iterdir():
        if not eset_dir.is_dir():
            continue
        
        if not (eset_dir / "matfiles").exists():
            continue
        
        eset_name = eset_dir.name
        
        # Count tracks
        total_tracks = count_tracks_in_eset(eset_dir)
        
        # Check if H5 files exist (completed)
        h5_files = list(output_dir.glob(f"{eset_name.split('_')[0]}*{eset_name.split('_')[-1]}*.h5"))
        # More accurate: check for H5 files matching experiment base names
        matfiles_dir = eset_dir / "matfiles"
        completed_experiments = set()
        for mat_file in matfiles_dir.glob("*.mat"):
            base_name = mat_file.stem
            h5_file = output_dir / f"{base_name}.h5"
            if h5_file.exists():
                completed_experiments.add(base_name)
        
        # Count tracks processed (rough estimate based on completed experiments)
        # This is approximate - we'd need to check H5 files for exact track counts
        tracks_processed = len(completed_experiments) * (total_tracks // max(len(list(matfiles_dir.glob("*.mat"))), 1))
        
        status[eset_name] = {
            'total_tracks': total_tracks,
            'tracks_processed': min(tracks_processed, total_tracks),
            'is_complete': len(completed_experiments) == len(list(matfiles_dir.glob("*.mat"))),
            'completed_experiments': len(completed_experiments),
            'total_experiments': len(list(matfiles_dir.glob("*.mat"))),
        }
    
    return status


def monitor_batch_process(
    genotype_dir: Path,
    output_dir: Path,
    refresh_interval: float = 0.2,
):
    """Monitor ESET batch processing."""
    terminal_width = 80
    terminal_height = 24
    
    bubble_system = ESETBubbleSystem(terminal_width, terminal_height)
    speck_system = SpeckSystem(terminal_width, terminal_height)
    
    start_time = time.time()
    last_update = start_time
    
    # Initialize ESETs
    initial_status = get_eset_status(genotype_dir, output_dir)
    for eset_name, status in initial_status.items():
        bubble_system.add_eset(eset_name, status['total_tracks'], is_processing=False)
    
    try:
        while True:
            current_time = time.time()
            dt = current_time - last_update
            last_update = current_time
            elapsed = current_time - start_time
            
            # Get current status
            status = get_eset_status(genotype_dir, output_dir)
            
            # Update bubbles
            processing_eset = None
            total_esets = len(status)
            completed_esets = sum(1 for s in status.values() if s['is_complete'])
            progress = (completed_esets / max(total_esets, 1)) * 100.0
            
            for eset_name, eset_status in status.items():
                # Check if this ESET is currently being processed
                # (heuristic: has some tracks processed but not complete)
                is_processing = (
                    eset_status['tracks_processed'] > 0 and 
                    not eset_status['is_complete'] and
                    eset_status['tracks_processed'] < eset_status['total_tracks']
                )
                
                if is_processing and processing_eset is None:
                    processing_eset = eset_name
                
                # Update or add bubble
                bubble_system.add_eset(eset_name, eset_status['total_tracks'], is_processing)
                bubble_system.update_eset_progress(eset_name, eset_status['tracks_processed'])
                
                if eset_status['is_complete']:
                    bubble_system.mark_complete(eset_name)
            
            # Update systems
            bubble_system.update(dt)
            speck_system.update(dt, progress)
            
            # Clear screen
            print(Colors.CLEAR, end='')
            
            # Header
            print(Colors.BOLD + Colors.AURORA_CYAN + "=" * terminal_width + Colors.RESET)
            print(Colors.BOLD + Colors.AURORA_CYAN + f"ESET Batch Processing Monitor - {genotype_dir.name}" + Colors.RESET)
            print(Colors.BOLD + Colors.AURORA_CYAN + "=" * terminal_width + Colors.RESET)
            print()
            
            # Stats
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(Colors.AURORA_PINK + f"Time: {time_str}" + Colors.RESET)
            
            elapsed_hours = int(elapsed // 3600)
            elapsed_min = int((elapsed % 3600) // 60)
            elapsed_sec = int(elapsed % 60)
            if elapsed_hours > 0:
                print(Colors.AURORA_BLUE + f"Elapsed: {elapsed_hours}:{elapsed_min:02d}:{elapsed_sec:02d}" + Colors.RESET)
            else:
                print(Colors.AURORA_BLUE + f"Elapsed: {elapsed_min:02d}:{elapsed_sec:02d}" + Colors.RESET)
            
            print(Colors.AURORA_GREEN + f"ESETs: {completed_esets}/{total_esets} complete ({progress:.1f}%)" + Colors.RESET)
            
            if processing_eset:
                proc_status = status[processing_eset]
                proc_progress = (proc_status['tracks_processed'] / max(proc_status['total_tracks'], 1)) * 100.0
                print(Colors.PROCESSING + f"Processing: {processing_eset}" + Colors.RESET)
                print(Colors.PROCESSING + f"  Tracks: {proc_status['tracks_processed']}/{proc_status['total_tracks']} ({proc_progress:.1f}%)" + Colors.RESET)
            
            print()
            
            # Render
            bg_speck_canvas, progress_speck_canvas = speck_system.render(progress)
            bubble_canvas = bubble_system.render()
            
            # Combine canvases
            combined_canvas = [[' ' for _ in range(terminal_width)] 
                              for _ in range(terminal_height)]
            
            progress_width = int((progress / 100.0) * terminal_width)
            
            # Background specks (outside progress area)
            for y in range(min(len(bg_speck_canvas), terminal_height)):
                line = bg_speck_canvas[y]
                x_pos = 0
                i = 0
                current_color = ''
                while i < len(line) and x_pos < terminal_width:
                    if line[i] == '\033':
                        ansi_start = i
                        while i < len(line) and line[i] != 'm':
                            i += 1
                        if i < len(line):
                            current_color = line[ansi_start:i+1]
                            i += 1
                        continue
                    
                    char = line[i]
                    if char != ' ' and x_pos >= progress_width:
                        combined_canvas[y][x_pos] = current_color + char + Colors.RESET
                        x_pos += 1
                    elif char == ' ':
                        x_pos += 1
                    i += 1
            
            # Progress specks (aurora curtain)
            for y in range(min(len(progress_speck_canvas), terminal_height)):
                line = progress_speck_canvas[y]
                x_pos = 0
                i = 0
                current_color = ''
                while i < len(line) and x_pos < progress_width:
                    if line[i] == '\033':
                        ansi_start = i
                        while i < len(line) and line[i] != 'm':
                            i += 1
                        if i < len(line):
                            current_color = line[ansi_start:i+1]
                            i += 1
                        continue
                    
                    char = line[i]
                    if char != ' ' and x_pos < progress_width:
                        combined_canvas[y][x_pos] = current_color + char + Colors.RESET
                        x_pos += 1
                    elif char == ' ':
                        if x_pos < progress_width:
                            x_pos += 1
                    i += 1
            
            # Bubbles (on top)
            for y in range(min(len(bubble_canvas), terminal_height)):
                line = bubble_canvas[y]
                x_pos = 0
                i = 0
                current_color = ''
                bubble_chars = []
                while i < len(line):
                    if line[i] == '\033':
                        ansi_start = i
                        while i < len(line) and line[i] != 'm':
                            i += 1
                        if i < len(line):
                            current_color = line[ansi_start:i+1]
                            i += 1
                        continue
                    
                    char = line[i]
                    if char != ' ':
                        bubble_chars.append((x_pos, char, current_color))
                    x_pos += 1
                    i += 1
                
                for x_pos, char, color in bubble_chars:
                    if 0 <= x_pos < terminal_width:
                        combined_canvas[y][x_pos] = color + char + Colors.RESET
            
            # Print combined canvas
            for y in range(terminal_height):
                line = ''.join(combined_canvas[y])
                print(line)
            
            # Footer
            print()
            print(Colors.BOLD + Colors.AURORA_CYAN + "=" * terminal_width + Colors.RESET)
            print(Colors.AURORA_PINK + "Press Ctrl+C to stop" + Colors.RESET)
            print(Colors.BOLD + Colors.AURORA_CYAN + "=" * terminal_width + Colors.RESET)
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n" + Colors.BOLD + Colors.AURORA_PINK + "Monitoring stopped!" + Colors.RESET)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ESET batch processing monitor with aurora animation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--genotype-dir",
        type=Path,
        required=True,
        help="Genotype directory to monitor (e.g., data/matlab_data/GMR61@GMR61)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/h5_files"),
        help="Output directory for H5 files (default: data/h5_files)",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=0.2,
        help="Refresh interval in seconds (default: 0.2)",
    )
    
    args = parser.parse_args()
    
    monitor_batch_process(
        genotype_dir=args.genotype_dir,
        output_dir=args.output_dir,
        refresh_interval=args.refresh_interval,
    )


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Cinnamoroll-themed analysis progress monitor for INDYsim H5 file processing.

Features:
- Cinnamoroll head with animated floppy ears
- Floating clouds background
- Aurora progress curtain
- Real-time track processing progress
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# ANSI color codes for bubble theme
class Colors:
    BLUE = '\033[94m'      # Light blue
    PINK = '\033[95m'      # Pink
    MINT = '\033[96m'     # Mint
    CREAM = '\033[97m'    # Cream
    GREEN = '\033[92m'    # Bright green (ANSI 92)
    BRIGHT_GREEN = '\033[1;92m'  # Bold bright green
    DARK_GREEN = '\033[38;5;46m'  # 256-color green
    GREEN_ALT = '\033[38;5;82m'  # Alternative green
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CLEAR = '\033[2J\033[H'  # Clear screen and move to top


@dataclass
class Speck:
    """A small speck for the progress curtain."""
    x: float
    y: float
    vx: float  # Horizontal velocity
    vy: float  # Vertical velocity
    size: float  # Size (very small)
    color: str  # Speck color
    char: str  # Character to display
    lifetime: float  # How long speck has existed
    
    def update(self, dt: float, terminal_width: int, terminal_height: int, wind_x: float, wind_y: float):
        """Update speck physics with wind."""
        # Apply wind forces
        wind_strength = 1.2  # Specks are lighter than bubbles
        self.vx += wind_x * wind_strength * dt
        self.vy += wind_y * wind_strength * dt
        
        # Air resistance
        damping = 0.92
        self.vx *= damping
        self.vy *= damping
        
        # Update position
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
        
        # Update lifetime
        self.lifetime += dt


@dataclass
class MagicSwirl:
    """A swirling magic particle around Cinnamoroll."""
    angle: float  # Current angle around Cinnamoroll
    radius: float  # Distance from center
    speed: float  # Rotation speed
    color: str  # Magic color
    size: float  # Particle size
    phase: float  # Animation phase for pulsing
    
    def update(self, dt: float):
        """Update swirl animation."""
        self.angle += self.speed * dt
        self.phase += dt * 2.0
        # Pulsing radius
        self.radius = self.radius + 0.3 * math.sin(self.phase)
    
    def get_position(self, center_x: float, center_y: float) -> tuple[float, float]:
        """Get current position relative to center."""
        x = center_x + math.cos(self.angle) * self.radius
        y = center_y + math.sin(self.angle) * self.radius
        return x, y


@dataclass
class FloatingCloud:
    """A cloud that floats around the screen."""
    art: list[str]  # ASCII art lines
    x: float  # X position
    y: float  # Y position
    vx: float  # Horizontal velocity
    vy: float  # Vertical velocity
    phase: float  # Animation phase for floating motion
    speed: float  # Floating speed
    width: int  # Width of cloud art
    height: int  # Height of cloud art
    
    def update(self, dt: float, screen_width: int, screen_height: int):
        """Update cloud position with floating animation."""
        # Use sine/cosine for smooth floating motion
        self.phase += self.speed * dt
        
        # Floating motion (gentle sine wave)
        float_x = math.sin(self.phase * 0.5) * 0.5
        float_y = math.cos(self.phase * 0.7) * 0.3
        
        # Update position
        self.x += self.vx * dt + float_x
        self.y += self.vy * dt + float_y
        
        # Wrap around screen edges
        if self.x < -self.width:
            self.x = screen_width
        elif self.x > screen_width:
            self.x = -self.width
        
        if self.y < -self.height:
            self.y = screen_height
        elif self.y > screen_height:
            self.y = -self.height


def load_cloud_art(cloud_file: Path) -> list[str]:
    """Load cloud ASCII art from file."""
    if not cloud_file.exists():
        return []
    
    try:
        with open(cloud_file) as f:
            lines = [line.rstrip('\n') for line in f.readlines()]
        return [line for line in lines if line.strip()]  # Remove empty lines
    except Exception:
        return []


def create_clouds(assets_dir: Path) -> list[FloatingCloud]:
    """Create floating cloud objects from standardized assets."""
    clouds = []
    
    # Load cloud art from standardized assets
    cloud_small = load_cloud_art(assets_dir / "cloud_small.yaml")
    cloud_medium = load_cloud_art(assets_dir / "cloud_medium.yaml")
    cloud_large = load_cloud_art(assets_dir / "cloud_large.yaml")
    
    # Create small cloud
    if cloud_small:
        clouds.append(FloatingCloud(
            art=cloud_small,
            x=10.0,
            y=5.0,
            vx=5.0,
            vy=2.0,
            phase=0.0,
            speed=0.8,
            width=max(len(line) for line in cloud_small) if cloud_small else 0,
            height=len(cloud_small)
        ))
    
    # Create medium cloud
    if cloud_medium:
        clouds.append(FloatingCloud(
            art=cloud_medium,
            x=30.0,
            y=10.0,
            vx=-4.0,
            vy=1.5,
            phase=math.pi * 0.5,
            speed=0.6,
            width=max(len(line) for line in cloud_medium) if cloud_medium else 0,
            height=len(cloud_medium)
        ))
    
    # Create large cloud
    if cloud_large:
        clouds.append(FloatingCloud(
            art=cloud_large,
            x=50.0,
            y=15.0,
            vx=-3.0,
            vy=1.0,
            phase=math.pi,
            speed=0.5,
            width=max(len(line) for line in cloud_large) if cloud_large else 0,
            height=len(cloud_large)
        ))
    
    return clouds


def render_clouds_on_canvas(clouds: list[FloatingCloud], canvas: list[list[str]], 
                           terminal_width: int, terminal_height: int):
    """Render clouds onto a canvas."""
    for cloud in clouds:
        cloud_x = int(cloud.x)
        cloud_y = int(cloud.y)
        
        for i, line in enumerate(cloud.art):
            y = cloud_y + i
            if 0 <= y < terminal_height:
                for j, char in enumerate(line):
                    x = cloud_x + j
                    if 0 <= x < terminal_width and char != ' ':
                        # Only render non-space characters, use cream color for clouds
                        canvas[y][x] = Colors.CREAM + char + Colors.RESET


@dataclass
class CinnamorollHead:
    """Simple Cinnamoroll that bounces diagonally."""
    x: float  # Center X position
    y: float  # Center Y position
    bob_phase: float  # Bobbing animation phase
    bob_speed: float  # Bobbing speed
    color: str  # Cinnamoroll color
    
    def update(self, dt: float, progress: float, terminal_width: int, terminal_height: int):
        """Update Cinnamoroll position - simple diagonal bounce."""
        progress_factor = progress / 100.0
        bounce_intensity = 0.3 + progress_factor * 0.7
        
        # Update bobbing phase for smooth diagonal bouncing
        self.bob_phase += self.bob_speed * dt
        
        # Diagonal bounce: sine for X, cosine for Y (45 degree diagonal)
        bounce_x = math.sin(self.bob_phase) * bounce_intensity * 4.0
        bounce_y = math.cos(self.bob_phase) * bounce_intensity * 3.0
        
        # Calculate target position (center + diagonal bounce)
        center_x = terminal_width // 2
        center_y = terminal_height // 2
        
        target_x = center_x + bounce_x
        target_y = center_y + bounce_y
        
        # Smooth interpolation
        self.x = self.x * 0.7 + target_x * 0.3
        self.y = self.y * 0.7 + target_y * 0.3
        
        # Clamp to safe bounds (simple circle, so smaller bounds)
        margin = 3
        self.x = max(margin, min(terminal_width - margin, self.x))
        self.y = max(margin, min(terminal_height - margin, self.y))
    
    def get_head_art(self) -> list[str]:
        """Get Cinnamoroll from YAML file."""
        # Load from YAML file
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        cinnamoroll_file = project_root / "src" / "indysim" / "monitors" / "assets" / "cinnamoroll.yaml"
        
        if cinnamoroll_file.exists():
            try:
                with open(cinnamoroll_file) as f:
                    lines = [line.rstrip('\n') for line in f.readlines()]
                # Apply cream color to the art
                colored_lines = []
                for line in lines:
                    if line.strip():  # Only non-empty lines
                        colored_lines.append(Colors.CREAM + line + Colors.RESET)
                if colored_lines:
                    return colored_lines
            except Exception:
                pass
        
        # Fallback: simple circle if YAML not found
        return [
            Colors.CREAM + "  ╭─────╮" + Colors.RESET,
            Colors.CREAM + " ╱  ●  ●  ╲" + Colors.RESET,
            Colors.CREAM + "│   ●   │" + Colors.RESET,
            Colors.CREAM + " ╲  ╰─╯  ╱" + Colors.RESET,
            Colors.CREAM + "  ╰─────╯" + Colors.RESET
        ]


class SpeckSystem:
    """Manages progress curtain specks."""
    
    def __init__(self, terminal_width: int, terminal_height: int):
        self.terminal_width = terminal_width
        self.terminal_height = terminal_height
        self.specks: list[Speck] = []
        self.wind_time = 0.0
        self.wind_gust_phase = random.uniform(0, 2 * math.pi)
        
        # Speck characters and colors (aurora palette)
        self.speck_chars = ['·', '•', '▪', '▫', '▪', '·', '○', '◯', '✦', '✧']
        # Aurora colors - greens, blues, purples, pinks
        self.speck_colors = [
            '\033[38;5;46m',   # Bright green
            '\033[38;5;82m',   # Light green
            '\033[38;5;118m',  # Lime green
            '\033[38;5;154m',  # Yellow-green
            '\033[38;5;27m',   # Deep blue
            '\033[38;5;33m',   # Sky blue
            '\033[38;5;39m',   # Bright blue
            '\033[38;5;45m',   # Cyan blue
            '\033[38;5;51m',   # Bright cyan
            '\033[38;5;55m',   # Deep purple
            '\033[38;5;93m',   # Bright purple
            '\033[38;5;129m',  # Magenta-purple
            '\033[38;5;165m',  # Pink-purple
            '\033[38;5;171m',  # Light magenta
            '\033[38;5;177m',  # Pink
            '\033[2;38;5;46m', # Dim green
            '\033[2;38;5;33m', # Dim blue
            '\033[2;38;5;93m', # Dim purple
        ]
        
        # Spawn initial specks
        self.target_speck_count = terminal_width * terminal_height // 3
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
        
        # Aurora color selection
        if random.random() < 0.3:
            color_idx = random.choice([0, 1, 2, 4, 5, 6, 9, 10, 11, 12])
        elif random.random() < 0.5:
            color_idx = random.choice([3, 7, 8, 13, 14])
        else:
            color_idx = random.choice([15, 16, 17])
        
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
        
        # Update specks
        for speck in self.specks:
            wind_x, wind_y = self.get_wind_at_position(
                speck.x, speck.y, self.wind_time
            )
            speck.update(dt, self.terminal_width, self.terminal_height, wind_x, wind_y)
        
        # Maintain speck count
        while len(self.specks) < self.target_speck_count:
            self.spawn_speck()
        
        # Ensure progress area has proportional coverage
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
        
        # Render specks
        for speck in self.specks:
            x = int(speck.x)
            y = int(speck.y)
            
            if 0 <= x < self.terminal_width and 0 <= y < self.terminal_height:
                bg_canvas[y][x] = speck.color + speck.char + Colors.RESET
                
                if x < progress_width:
                    progress_canvas[y][x] = speck.color + speck.char + Colors.RESET
        
        return [''.join(row) for row in bg_canvas], [''.join(row) for row in progress_canvas]


class CinnamorollSystem:
    """Manages Cinnamoroll head animation."""
    
    def __init__(self, terminal_width: int, terminal_height: int):
        self.terminal_width = terminal_width
        self.terminal_height = terminal_height
        
        center_x = terminal_width // 2
        center_y = terminal_height // 2
        
        self.head = CinnamorollHead(
            x=float(center_x),
            y=float(center_y),
            bob_phase=random.uniform(0, 2 * math.pi),
            bob_speed=1.5,
            color=Colors.CREAM,
        )
        
        self.current_progress = 0.0
    
    def update(self, dt: float, progress: float):
        """Update Cinnamoroll head animation."""
        self.current_progress = progress
        self.head.update(dt, progress, self.terminal_width, self.terminal_height)
    
    def render(self) -> list[str]:
        """Render Cinnamoroll head to a canvas."""
        canvas = [[' ' for _ in range(self.terminal_width)] 
                 for _ in range(self.terminal_height)]
        
        head_lines = self.head.get_head_art()
        art_width = len(head_lines[0]) if head_lines else 10
        art_height = len(head_lines)
        
        head_x = int(self.head.x) - art_width // 2
        head_y = int(self.head.y) - art_height // 2
        
        # Render head
        for i, line in enumerate(head_lines):
            y = head_y + i
            if 0 <= y < self.terminal_height:
                for j, char in enumerate(line):
                    x = head_x + j
                    if 0 <= x < self.terminal_width:
                        canvas[y][x] = char
        
        return [''.join(row) for row in canvas]


def load_progress(progress_file: Path) -> dict:
    """Load progress from JSON file."""
    if not progress_file.exists():
        return {}
    
    try:
        with open(progress_file) as f:
            return json.load(f)
    except Exception:
        return {}


def monitor_analysis(progress_file: Path, refresh_interval: float = 0.1):
    """Monitor analysis progress with Cinnamoroll animation."""
    # Terminal dimensions
    terminal_width = 80
    terminal_height = 24
    
    # Load clouds from assets
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    assets_dir = project_root / "src" / "indysim" / "monitors" / "assets"
    clouds = create_clouds(assets_dir)
    last_cloud_update = time.time()
    
    # Cinnamoroll system
    cinnamoroll_system = CinnamorollSystem(terminal_width, terminal_height)
    
    # Speck system for progress curtain
    speck_system = SpeckSystem(terminal_width, terminal_height)
    
    start_time = time.time()
    last_update = start_time
    
    try:
        while True:
            current_time = time.time()
            dt = current_time - last_update
            last_update = current_time
            
            # Load progress
            progress_data = load_progress(progress_file)
            
            # Extract progress info
            status = progress_data.get('status', 'unknown')
            stage = progress_data.get('stage', 'Starting...')
            progress_pct = progress_data.get('progress_pct', 0.0)
            current_track = progress_data.get('current_track', 0)
            total_tracks = progress_data.get('total_tracks', 0)
            elapsed_time = progress_data.get('elapsed_time', 0.0)
            eta_seconds = progress_data.get('eta_seconds', 0.0)
            h5_file = progress_data.get('h5_file', 'Unknown')
            experiment_id = progress_data.get('experiment_id', 'Unknown')
            messages = progress_data.get('messages', [])
            
            # Update systems
            cinnamoroll_system.update(dt, progress_pct)
            speck_system.update(dt, progress_pct)
            
            # Clear screen
            print(Colors.CLEAR, end='')
            
            # Header
            print(Colors.BOLD + Colors.BLUE + "=" * terminal_width + Colors.RESET)
            print()
            
            # Stats
            time_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
            print(Colors.CREAM + f"Time: {time_str}" + Colors.RESET)
            print()
            
            # Elapsed time
            elapsed_hours = int(elapsed_time // 3600)
            elapsed_min = int((elapsed_time % 3600) // 60)
            elapsed_sec = int(elapsed_time % 60)
            if elapsed_hours > 0:
                print(Colors.MINT + f"Elapsed: {elapsed_hours}:{elapsed_min:02d}:{elapsed_sec:02d}" + Colors.RESET)
            else:
                print(Colors.MINT + f"Elapsed: {elapsed_min:02d}:{elapsed_sec:02d}" + Colors.RESET)
            
            # ETA
            if eta_seconds > 0:
                eta_hours = int(eta_seconds // 3600)
                eta_min = int((eta_seconds % 3600) // 60)
                eta_sec = int(eta_seconds % 60)
                if eta_hours > 0:
                    print(Colors.MINT + f"ETA: {eta_hours}:{eta_min:02d}:{eta_sec:02d}" + Colors.RESET)
                else:
                    print(Colors.MINT + f"ETA: {eta_min:02d}:{eta_sec:02d}" + Colors.RESET)
            
            # Track progress
            if total_tracks > 0:
                print(Colors.PINK + f"Tracks: {current_track}/{total_tracks}" + Colors.RESET)
            
            print(Colors.BLUE + f"Progress: {progress_pct:.1f}%" + Colors.RESET)
            print(Colors.CREAM + f"Stage: {stage}" + Colors.RESET)
            print(Colors.CREAM + f"Status: {status}" + Colors.RESET)
            print(Colors.CREAM + f"H5 File: {Path(h5_file).name}" + Colors.RESET)
            print()
            
            # Update clouds
            dt_clouds = current_time - last_cloud_update
            last_cloud_update = current_time
            for cloud in clouds:
                cloud.update(dt_clouds, terminal_width, terminal_height)
            
            # Render specks
            bg_speck_canvas, progress_speck_canvas = speck_system.render(progress_pct)
            
            # Render clouds
            cloud_canvas = [[' ' for _ in range(terminal_width)] for _ in range(terminal_height)]
            render_clouds_on_canvas(clouds, cloud_canvas, terminal_width, terminal_height)
            
            # Render Cinnamoroll
            cinnamoroll_canvas = cinnamoroll_system.render()
            
            # Combine layers
            combined_canvas = [[' ' for _ in range(terminal_width)] 
                              for _ in range(terminal_height)]
            
            # Render clouds
            render_clouds_on_canvas(clouds, combined_canvas, terminal_width, terminal_height)
            
            # Render background specks (outside progress area)
            progress_width = int((progress_pct / 100.0) * terminal_width)
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
                    else:
                        x_pos += 1
                    i += 1
            
            # Render progress specks (in progress area)
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
            
            # Render Cinnamoroll (over everything)
            for y in range(min(len(cinnamoroll_canvas), terminal_height)):
                line = cinnamoroll_canvas[y]
                x_pos = 0
                i = 0
                current_color = ''
                
                cinnamoroll_chars = []
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
                        cinnamoroll_chars.append((x_pos, char, current_color))
                    x_pos += 1
                    i += 1
                
                for x_pos, char, color in cinnamoroll_chars:
                    if 0 <= x_pos < terminal_width:
                        combined_canvas[y][x_pos] = color + char + Colors.RESET
            
            # Print combined canvas
            for y in range(terminal_height):
                line = ''.join(combined_canvas[y])
                print(line)
            
            # Recent messages
            if messages:
                print()
                print(Colors.BOLD + Colors.BLUE + "=" * terminal_width + Colors.RESET)
                print(Colors.CREAM + "Recent Messages:" + Colors.RESET)
                for msg in messages[-5:]:  # Show last 5 messages
                    print(Colors.MINT + f"  {msg}" + Colors.RESET)
            
            # Footer
            print(Colors.BOLD + Colors.BLUE + "=" * terminal_width + Colors.RESET)
            print(Colors.CREAM + "Press Ctrl+C to stop" + Colors.RESET)
            print(Colors.BOLD + Colors.BLUE + "=" * terminal_width + Colors.RESET)
            
            # Check if complete
            if status == 'complete':
                print()
                print(Colors.BOLD + Colors.GREEN + "[OK] Analysis Complete!" + Colors.RESET)
                time.sleep(5)
                break
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n" + Colors.BOLD + Colors.PINK + "Monitoring stopped!" + Colors.RESET)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cinnamoroll-themed analysis progress monitor for INDYsim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--progress-file",
        type=Path,
        required=True,
        help="Path to progress JSON file",
    )
    parser.add_argument(
        "--refresh-interval",
        type=float,
        default=0.1,
        help="Refresh interval in seconds (default: 0.1)",
    )
    
    args = parser.parse_args()
    
    monitor_analysis(
        progress_file=args.progress_file,
        refresh_interval=args.refresh_interval,
    )


if __name__ == "__main__":
    main()



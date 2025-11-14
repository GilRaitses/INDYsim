#!/usr/bin/env python3
"""
Save analysis results back to H5 files.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional

def save_results_to_h5(h5_file: Path, 
                       events_df: pd.DataFrame,
                       trajectories_df: pd.DataFrame,
                       klein_runs_df: Optional[pd.DataFrame] = None):
    """
    Save analysis results back to H5 file.
    
    Structure:
    /analysis/
      /events/          - Event records (binned data)
      /trajectories/    - Full trajectory data
      /klein_runs/      - Klein run table (if available)
    
    Parameters
    ----------
    h5_file : Path
        Path to H5 file (will be opened in append mode)
    events_df : pd.DataFrame
        Event records DataFrame
    trajectories_df : pd.DataFrame
        Trajectories DataFrame
    klein_runs_df : pd.DataFrame, optional
        Klein run table DataFrame
    """
    if not h5_file.exists():
        raise FileNotFoundError(f"H5 file not found: {h5_file}")
    
    print(f"  Saving results to H5: {h5_file.name}")
    
    with h5py.File(h5_file, 'a') as f:  # 'a' mode = append/read-write
        # Create analysis group (delete if exists to avoid conflicts)
        if 'analysis' in f:
            del f['analysis']
        analysis_group = f.create_group('analysis')
        
        # Save events
        events_group = analysis_group.create_group('events')
        for col in events_df.columns:
            data = events_df[col].values
            # Handle different data types
            if data.dtype == object:
                # Convert to string array
                data = data.astype(str)
                dt = h5py.special_dtype(vlen=str)
                events_group.create_dataset(col, data=data, dtype=dt)
            elif pd.api.types.is_bool_dtype(events_df[col]):
                # Boolean arrays
                events_group.create_dataset(col, data=data.astype(int), dtype='i1')
            elif pd.api.types.is_integer_dtype(events_df[col]):
                events_group.create_dataset(col, data=data, dtype=events_df[col].dtype)
            elif pd.api.types.is_float_dtype(events_df[col]):
                events_group.create_dataset(col, data=data, dtype='f8')
            else:
                # Fallback: convert to string
                data = data.astype(str)
                dt = h5py.special_dtype(vlen=str)
                events_group.create_dataset(col, data=data, dtype=dt)
        
        # Add metadata
        events_group.attrs['num_records'] = len(events_df)
        events_group.attrs['columns'] = list(events_df.columns)
        
        # Save trajectories
        trajectories_group = analysis_group.create_group('trajectories')
        for col in trajectories_df.columns:
            data = trajectories_df[col].values
            # Handle different data types
            if data.dtype == object:
                # Convert to string array
                data = data.astype(str)
                dt = h5py.special_dtype(vlen=str)
                trajectories_group.create_dataset(col, data=data, dtype=dt)
            elif pd.api.types.is_bool_dtype(trajectories_df[col]):
                # Boolean arrays
                trajectories_group.create_dataset(col, data=data.astype(int), dtype='i1')
            elif pd.api.types.is_integer_dtype(trajectories_df[col]):
                trajectories_group.create_dataset(col, data=data, dtype=trajectories_df[col].dtype)
            elif pd.api.types.is_float_dtype(trajectories_df[col]):
                trajectories_group.create_dataset(col, data=data, dtype='f8')
            else:
                # Fallback: convert to string
                data = data.astype(str)
                dt = h5py.special_dtype(vlen=str)
                trajectories_group.create_dataset(col, data=data, dtype=dt)
        
        # Add metadata
        trajectories_group.attrs['num_records'] = len(trajectories_df)
        trajectories_group.attrs['columns'] = list(trajectories_df.columns)
        
        # Save Klein run table if provided
        if klein_runs_df is not None and len(klein_runs_df) > 0:
            klein_group = analysis_group.create_group('klein_runs')
            for col in klein_runs_df.columns:
                data = klein_runs_df[col].values
                # Handle different data types
                if data.dtype == object:
                    data = data.astype(str)
                    dt = h5py.special_dtype(vlen=str)
                    klein_group.create_dataset(col, data=data, dtype=dt)
                elif pd.api.types.is_bool_dtype(klein_runs_df[col]):
                    klein_group.create_dataset(col, data=data.astype(int), dtype='i1')
                elif pd.api.types.is_integer_dtype(klein_runs_df[col]):
                    klein_group.create_dataset(col, data=data, dtype=klein_runs_df[col].dtype)
                elif pd.api.types.is_float_dtype(klein_runs_df[col]):
                    klein_group.create_dataset(col, data=data, dtype='f8')
                else:
                    data = data.astype(str)
                    dt = h5py.special_dtype(vlen=str)
                    klein_group.create_dataset(col, data=data, dtype=dt)
            
            # Add metadata
            klein_group.attrs['num_records'] = len(klein_runs_df)
            klein_group.attrs['columns'] = list(klein_runs_df.columns)
        
        # Add analysis metadata
        analysis_group.attrs['analysis_date'] = pd.Timestamp.now().isoformat()
        analysis_group.attrs['num_tracks'] = trajectories_df['track_id'].nunique() if 'track_id' in trajectories_df.columns else 0
        analysis_group.attrs['num_events'] = len(events_df)
        analysis_group.attrs['num_trajectories'] = len(trajectories_df)
        if klein_runs_df is not None:
            analysis_group.attrs['num_klein_runs'] = len(klein_runs_df)
        
        print(f"    ✅ Saved {len(events_df):,} events, {len(trajectories_df):,} trajectory records")
        if klein_runs_df is not None:
            print(f"    ✅ Saved {len(klein_runs_df):,} Klein run table records")


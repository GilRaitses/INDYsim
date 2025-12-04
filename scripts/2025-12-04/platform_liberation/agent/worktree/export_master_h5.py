"""
export_master_h5.py
-------------------

Pack combined_analysis.json into a master H5 for simulator intake.

Inputs:
  --combined : path to combined_analysis.json
  --output   : output H5 path (e.g., master_sim_input.h5)

The combined_analysis.json is produced by running:
  python3 engineer_dataset_from_h5.py <validated_h5_dir> -o <analysis_dir>
"""

import argparse
import json
from pathlib import Path
import h5py
import numpy as np


def write_group_json(g: h5py.Group, name: str, obj):
    """Write JSON-serializable object as a UTF-8 string dataset."""
    data = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    g.create_dataset(name, data=np.string_(data))


def export_master(combined_path: Path, output_path: Path):
    data = json.loads(combined_path.read_text())
    files = data.get("files", [])

    with h5py.File(output_path, "w") as h5:
        h5.attrs["source_combined_json"] = str(combined_path)
        h5.attrs["generated_by"] = "export_master_h5.py"

        # Per-file groups
        for f in files:
            fname = Path(f.get("file", "unknown")).stem
            grp = h5.create_group(f"files/{fname}")
            write_group_json(grp, "tracks", f.get("tracks", []))
            write_group_json(grp, "summary", f.get("summary", {}))
            write_group_json(grp, "windows", f.get("windows", []))
            write_group_json(grp, "track_windows", f.get("track_windows", []))
            write_group_json(grp, "population_windows", f.get("population_windows", {}))
            write_group_json(grp, "concurrency", f.get("concurrency", []))

        # Combined summary
        write_group_json(h5, "processed_at", data.get("processed_at", ""))
        write_group_json(h5, "combined_files", [f.get("file", "") for f in files])

    # Manifest sidecar
    manifest = {
        "combined_source": str(combined_path),
        "output_h5": str(output_path),
        "files": [f.get("file", "") for f in files],
    }
    (output_path.parent / "master_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Master H5 written: {output_path}")
    print(f"Manifest written: {output_path.parent/'master_manifest.json'}")


def main():
    ap = argparse.ArgumentParser(description="Export master H5 from combined_analysis.json")
    ap.add_argument("--combined", required=True, help="Path to combined_analysis.json")
    ap.add_argument("--output", required=True, help="Output H5 path")
    args = ap.parse_args()

    combined = Path(args.combined)
    output = Path(args.output)
    if not combined.exists():
        raise FileNotFoundError(f"combined_analysis.json not found: {combined}")
    output.parent.mkdir(parents=True, exist_ok=True)
    export_master(combined, output)


if __name__ == "__main__":
    main()


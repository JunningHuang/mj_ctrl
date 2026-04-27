"""
Print keys, shapes, and dtypes of every .npz file found under real_robot_data/.

Usage:
    python real_robot_data/inspect_data.py
    python real_robot_data/inspect_data.py real_robot_data/run_baseline_franka/
    python real_robot_data/inspect_data.py path/to/data.npz
"""

import sys
from pathlib import Path

import numpy as np


def inspect(npz_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"  {npz_path}")
    print(f"{'='*60}")
    d = np.load(npz_path)
    for key in d.files:
        arr = d[key]
        if arr.ndim == 0:
            print(f"  {key:<30} scalar  = {arr.item():.4g}")
        else:
            print(f"  {key:<30} shape={str(arr.shape):<18} dtype={arr.dtype}")


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("real_robot_data")

    if root.is_file() and root.suffix == ".npz":
        paths = [root]
    else:
        paths = sorted(root.rglob("*.npz"))

    if not paths:
        print(f"No .npz files found under {root}")
        return

    for p in paths:
        inspect(p)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Calculate background intensity statistics across different methods.

Requires input images to be named as <FOV_Name>_<Method>.tif (e.g., FOV01_Raw.tif,
FOV01_PBC.tif). Background is estimated from the Raw image and applied to all
methods. Outputs mean and standard deviation in either DN or photons/pixel.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.stats import wilcoxon


def get_bg_mask(raw_img: np.ndarray, offset: float, readout: float) -> np.ndarray:
    """Generate a background mask using a 3-sigma threshold above the offset."""
    return raw_img < (offset + 3 * readout)


def parse_filename(path: Path):
    """Parse <FOV>_<Method>.tif"""
    stem = path.stem
    if "_" not in stem:
        return None, None
    parts = stem.rsplit("_", 1)
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(description="Background statistics across methods.")
    parser.add_argument("--in-dir", required=True, type=Path, help="Input image directory.")
    parser.add_argument("--out-csv", required=True, type=Path, help="Output CSV path.")
    parser.add_argument("--offset", type=float, default=97.4, help="Camera offset in DN.")
    parser.add_argument("--readout", type=float, default=3.03, help="Readout noise in DN.")
    parser.add_argument("--gain", type=float, default=2.2302, help="Camera gain (DN / photon). Set to 1.0 to keep DN.")
    parser.add_argument("--min-bg-pixels", type=int, default=100, help="Minimum background pixels.")
    args = parser.parse_args()

    files = list(args.in_dir.glob("*.tif*"))
    if not files:
        raise ValueError(f"No tif files found in {args.in_dir}")

    fov_dict = {}
    for f in files:
        fov, method = parse_filename(f)
        if fov and method:
            if fov not in fov_dict:
                fov_dict[fov] = {}
            fov_dict[fov][method] = f

    rows = []
    for fov, methods in fov_dict.items():
        if "Raw" not in methods:
            print(f"Skipping {fov}: no 'Raw' image found for mask generation.")
            continue

        raw_img = tifffile.imread(methods["Raw"]).astype(np.float64)
        base_mask = get_bg_mask(raw_img, args.offset, args.readout)

        if base_mask.sum() < args.min_bg_pixels:
            print(f"Skipping {fov}: not enough background pixels ({base_mask.sum()}).")
            continue

        for method, path in methods.items():
            img = tifffile.imread(path).astype(np.float64)
            if img.shape != raw_img.shape:
                print(f"Skipping {fov}/{method}: shape mismatch with Raw.")
                continue

            # Convert to photons if gain != 1.0
            if args.gain != 1.0:
                img_processed = (img - args.offset) / args.gain
            else:
                img_processed = img

            bg_vals = img_processed[base_mask]
            rows.append({
                "FOV": fov,
                "Method": method,
                "Mean_BG": bg_vals.mean(),
                "STD_BG": bg_vals.std(),
                "N_Pixels": int(base_mask.sum())
            })

    if not rows:
        raise SystemExit("No valid data extracted.")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    # Summary by method
    summ = df.groupby("Method").agg(
        Mean_BG_avg=("Mean_BG", "mean"),
        Mean_BG_std=("Mean_BG", "std"),
        STD_BG_avg=("STD_BG", "mean"),
        STD_BG_std=("STD_BG", "std"),
        N_FOV=("FOV", "count")
    ).reset_index()

    print("\n=== Background Statistics (Mean +- SD) ===")
    print(summ.round(3).to_string(index=False))

    # Pairwise Wilcoxon test (Paired by FOV)
    print("\n=== Pairwise Wilcoxon Test (Paired by FOV) ===")
    methods_list = summ["Method"].tolist()
    for i in range(len(methods_list)):
        for j in range(i + 1, len(methods_list)):
            m1, m2 = methods_list[i], methods_list[j]
            df1 = df[df.Method == m1].set_index("FOV")["Mean_BG"]
            df2 = df[df.Method == m2].set_index("FOV")["Mean_BG"]
            common = df1.index.intersection(df2.index)
            if len(common) >= 3:
                try:
                    _, p_val = wilcoxon(df1.loc[common], df2.loc[common])
                    print(f"  {m1:10s} vs {m2:10s}: p = {p_val:.4g} (n={len(common)})")
                except ValueError:
                    pass


if __name__ == "__main__":
    main()
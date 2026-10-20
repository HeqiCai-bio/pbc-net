#!/usr/bin/env python3
"""Calculate PSNR, SSIM, and NRMSE between GT and test images.

Assumes images are properly aligned and normalized to the same scale before
calculation. Files are matched by name (or sorted order if directories are used).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.metrics import structural_similarity as ssim


def calc_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred - gt) ** 2)
    if mse == 0:
        return float("inf")
    peak = np.max(gt)
    return 10 * np.log10((peak ** 2) / mse)


def calc_nrmse(pred: np.ndarray, gt: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((pred - gt) ** 2))
    gt_range = np.max(gt) - np.min(gt)
    return rmse / gt_range if gt_range > 0 else 0.0


def calc_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    dr = np.max(gt) - np.min(gt)
    return ssim(pred, gt, data_range=dr) if dr > 0 else 1.0


def get_pairs(gt_dir: Path, test_dir: Path):
    if gt_dir.is_file() and test_dir.is_dir():
        gt_files = [gt_dir] * len(list(test_dir.glob("*.tif*")))
        test_files = sorted(test_dir.glob("*.tif*"))
        return list(zip(gt_files, test_files))
    
    if gt_dir.is_dir() and test_dir.is_dir():
        gts = {f.name: f for f in gt_dir.glob("*.tif*")}
        tests = sorted(test_dir.glob("*.tif*"))
        pairs = []
        for t in tests:
            if t.name in gts:
                pairs.append((gts[t.name], t))
            else:
                print(f"Warning: No matching GT found for {t.name}")
        return pairs
        
    raise ValueError("Unsupported directory/file combination.")


def main():
    parser = argparse.ArgumentParser(description="Calculate image quality metrics.")
    parser.add_argument("--gt", required=True, type=Path, help="GT file or directory.")
    parser.add_argument("--test", required=True, type=Path, help="Test file or directory.")
    parser.add_argument("--out-csv", required=True, type=Path, help="Output CSV path.")
    args = parser.parse_args()

    pairs = get_pairs(args.gt, args.test)
    if not pairs:
        raise ValueError("No image pairs found.")

    results = []
    for gt_path, test_path in pairs:
        gt = tifffile.imread(gt_path).astype(np.float64)
        test = tifffile.imread(test_path).astype(np.float64)

        if gt.shape != test.shape:
            print(f"Skipping {test_path.name}: shape mismatch (GT {gt.shape} vs Test {test.shape})")
            continue

        results.append({
            "GT": gt_path.name,
            "Test": test_path.name,
            "PSNR": calc_psnr(test, gt),
            "SSIM": calc_ssim(test, gt),
            "NRMSE": calc_nrmse(test, gt)
        })

    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)

    print(f"\n=== Metrics Summary (n={len(df)}) ===")
    print(df[["PSNR", "SSIM", "NRMSE"]].mean().round(4).to_string())
    print(f"\nResults saved to {args.out_csv}")


if __name__ == "__main__":
    main()
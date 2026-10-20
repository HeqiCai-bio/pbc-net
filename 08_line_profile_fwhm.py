#!/usr/bin/env python3
"""Calculate Full Width at Half Maximum (FWHM) from line profiles.

Reads CSV files exported from Fiji/ImageJ (e.g., Distance vs. Gray Value).
Uses multi-Gaussian fitting to resolve overlapping peaks.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def read_xy(csv_path: Path):
    df = pd.read_csv(csv_path, header=0)
    x_col = next((c for c in df.columns if "dist" in c.lower() or c.lower() == "x"), df.columns[0])
    y_col = next((c for c in df.columns if "gray" in c.lower() or "intensity" in c.lower() or c.lower() == "y"), df.columns[1])
    
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    idx = np.argsort(x)
    return x[idx], y[idx]


def multi_gaussian(x, *params):
    baseline = params[-1]
    y = np.zeros_like(x) + baseline
    num_peaks = (len(params) - 1) // 3
    for i in range(num_peaks):
        amp, mu, sigma = params[i*3: i*3+3]
        y += amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y


def fit_profile(x, y, max_peaks: int = 2):
    n = len(y)
    k = max(1, int(round(n * 0.15)))
    baseline = float(np.mean(np.r_[y[:k], y[-k:]]))
    
    y_corr = y - baseline
    max_corr = np.max(y_corr)
    if max_corr <= 0:
        return None, baseline

    x_range = x[-1] - x[0]
    min_dist = max(1, int(len(x) * 0.05))
    
    peaks, _ = find_peaks(y_corr, height=max_corr * 0.1, distance=min_dist)
    
    if len(peaks) == 0 or len(peaks) > max_peaks:
        # Fallback to single peak
        peaks = [int(np.argmax(y_corr))]

    p0 = []
    bounds_lower = []
    bounds_upper = []
    
    for i in peaks:
        amp_init = y_corr[i]
        mu_init = x[i]
        sigma_init = x_range / (len(peaks) * 4)
        
        p0.extend([amp_init, mu_init, sigma_init])
        bounds_lower.extend([0, x[0], x_range * 0.001])
        bounds_upper.extend([max_corr * 2, x[-1], x_range * 0.5])
        
    p0.append(baseline)
    bounds_lower.append(0)
    bounds_upper.append(max_corr)

    try:
        popt, _ = curve_fit(
            multi_gaussian, x, y, p0=p0, 
            bounds=(bounds_lower, bounds_upper), maxfev=10000
        )
        return popt, baseline
    except Exception as e:
        print(f"  Fit failed: {e}")
        return None, baseline


def main():
    parser = argparse.ArgumentParser(description="Calculate FWHM from line profile CSVs.")
    parser.add_argument("--in-dir", required=True, type=Path, help="Directory with CSV files.")
    parser.add_argument("--out-csv", required=True, type=Path, help="Output CSV path.")
    parser.add_argument("--max-peaks", type=int, default=2, help="Max peaks to fit.")
    args = parser.parse_args()

    files = sorted(args.in_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {args.in_dir}")

    rows = []
    for f in files:
        x, y = read_xy(f)
        popt, baseline = fit_profile(x, y, args.max_peaks)
        
        if popt is None:
            rows.append({"File": f.stem, "Status": "Fit Failed"})
            continue

        num_peaks = (len(popt) - 1) // 3
        for i in range(num_peaks):
            amp = popt[i*3]
            mu = popt[i*3 + 1]
            sigma = abs(popt[i*3 + 2])
            fwhm = 2.35482 * sigma
            
            rows.append({
                "File": f.stem,
                "Peak_Index": i + 1,
                "Peak_Center_nm": round(mu, 2),
                "FWHM_nm": round(fwhm, 2),
                "Baseline": round(baseline, 2),
                "Status": "OK"
            })
            print(f"{f.name}: Peak {i+1} @ {mu:.1f} nm, FWHM = {fwhm:.1f} nm")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nFWHM results saved to {args.out_csv}")


if __name__ == "__main__":
    main()
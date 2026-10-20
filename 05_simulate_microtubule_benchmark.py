#!/usr/bin/env python3
"""Synthetic microtubule benchmark generation for PBC and PBC+RL.

Stages:
  --stage gt    : Generate 256x256 GT, crop to 235x235.
  --stage noise : Read GT_235.tif, generate PSF, simulate 16 pre/16 post
                  frames, and output PBC (float32 and 16-bit).
  --stage all   : Run both stages sequentially.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.signal import fftconvolve
from scipy.special import j1
from skimage.draw import line_aa


def generate_gt(
    seed: int,
    width: int = 256,
    height: int = 256,
    n_tubes: int = 12,
    n_steps: int = 200,
    step_length: int = 3,
    max_angle_change: float = 5.0,
    scale_factor: int = 4,
    line_width_nm: float = 25.0,
    pixel_size_nm: float = 65.0,
) -> np.ndarray:
    """Generate a synthetic microtubule network using a random walk."""
    rng = np.random.RandomState(seed)
    hr_height = height * scale_factor
    hr_width = width * scale_factor
    gt_hr = np.zeros((hr_height, hr_width), dtype=np.float64)

    for _ in range(n_tubes):
        x = 30 + rng.random() * (width - 60)
        y = 30 + rng.random() * (height - 60)
        angle = rng.random() * 360

        for _ in range(n_steps):
            angle_change = (rng.random() - 0.5) * 2 * max_angle_change
            angle += angle_change
            new_x = x + step_length * np.cos(angle * np.pi / 180)
            new_y = y + step_length * np.sin(angle * np.pi / 180)

            if not (0 <= new_x < width and 0 <= new_y < height):
                break

            y1_hr = int(y * scale_factor)
            x1_hr = int(x * scale_factor)
            y2_hr = int(new_y * scale_factor)
            x2_hr = int(new_x * scale_factor)

            rr, cc, val = line_aa(y1_hr, x1_hr, y2_hr, x2_hr)
            valid = (rr >= 0) & (rr < hr_height) & (cc >= 0) & (cc < hr_width)
            gt_hr[rr[valid], cc[valid]] = np.maximum(
                gt_hr[rr[valid], cc[valid]], val[valid]
            )
            x, y = new_x, new_y

    # Gaussian profile corresponding to 25 nm FWHM
    sigma_nm = line_width_nm / 2.355
    sigma_hr = sigma_nm / (pixel_size_nm / scale_factor)
    gt_hr = gaussian_filter(gt_hr, sigma=sigma_hr)
    
    # Downsample by uniform filtering
    gt_hr = uniform_filter(gt_hr, size=scale_factor)
    gt = gt_hr[::scale_factor, ::scale_factor].copy()

    if gt.max() > 0:
        gt = gt / gt.max()
        
    # Crop 256x256 to 235x235 (center crop)
    start_y = (height - 235) // 2
    start_x = (width - 235) // 2
    gt_235 = gt[start_y:start_y+235, start_x:start_x+235]
    
    return gt_235.astype(np.float32)


def generate_airy_psf(
    size: int = 21,
    pixel_size_nm: float = 65.0,
    lambda_nm: float = 488.0,
    na: float = 1.5,
) -> np.ndarray:
    """Generate a 2D Airy PSF."""
    center = size // 2
    y, x = np.ogrid[-center:center+1, -center:center+1]
    r = np.sqrt(x**2 + y**2) * pixel_size_nm

    k = 2 * np.pi / lambda_nm
    arg = k * na * r

    psf = np.zeros_like(r, dtype=np.float64)
    mask = r != 0
    psf[mask] = (2 * j1(arg[mask]) / arg[mask]) ** 2
    psf[~mask] = 1.0
    
    return (psf / psf.sum()).astype(np.float64)


def convolve_with_psf(gt: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Convolve ground truth with PSF and normalize to [0, 1]."""
    blurred = fftconvolve(gt, psf, mode="same")
    if blurred.max() > 0:
        blurred = blurred / blurred.max()
    return blurred


def add_physical_noise(
    gt_blur: np.ndarray,
    photon_scale: float,
    readout_sigma: float,
    background: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Add Poisson shot noise and Gaussian readout noise."""
    ideal_signal = gt_blur * photon_scale + background
    ideal_signal = np.clip(ideal_signal, 0, None)

    photon_counts = rng.poisson(ideal_signal).astype(np.float64)
    readout_noise = rng.normal(0, readout_sigma, ideal_signal.shape)

    return photon_counts + readout_noise


def run_gt_stage(args: argparse.Namespace, out_dir: Path) -> Path:
    print("Stage: Generate Ground Truth")
    gt = generate_gt(seed=args.seed)
    gt_path = out_dir / "GT_235.tif"
    
    # Save as 16-bit for standard visualization, though it's in [0,1] range
    tifffile.imwrite(gt_path, (gt * 65535).astype(np.uint16))
    print(f"Saved: {gt_path} | Shape: {gt.shape}")
    return gt_path


def run_noise_stage(args: argparse.Namespace, out_dir: Path, gt_path: Path) -> None:
    print("Stage: Simulate Noise and PBC")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
        
    gt_clear = tifffile.imread(gt_path).astype(np.float64)
    if gt_clear.max() > 1:
        gt_clear = gt_clear / 65535.0  # Normalize back to [0,1] if loaded from 16-bit
        
    psf = generate_airy_psf()
    gt_blur = convolve_with_psf(gt_clear, psf)
    
    tifffile.imwrite(out_dir / "GT_Blur.tif", (gt_blur * 65535).astype(np.uint16))

    print(f"Generating {args.n_frames} pre-bleach and post-bleach frames ...")
    pre_stack = []
    post_stack = []
    rng = np.random.RandomState(args.seed + 1)
    
    for _ in range(args.n_frames):
        pre = add_physical_noise(gt_blur, args.photon_scale, args.readout_sigma, args.background, rng)
        post = add_physical_noise(gt_blur, args.photon_scale * args.bleach_factor, args.readout_sigma, args.background, rng)
        pre_stack.append(pre)
        post_stack.append(post)

    pre_stack = np.array(pre_stack)
    post_stack = np.array(post_stack)

    pre_avg = pre_stack.mean(axis=0)
    post_avg = post_stack.mean(axis=0)

    pbc_float = np.clip(pre_avg - post_avg, 0, None)
    
    # Convert to 16-bit using the experimental camera assumption
    max_12bit = 4095.0
    max_16bit = 65535.0
    pbc_16bit = (pbc_float / max_12bit) * max_16bit
    pbc_16bit = np.clip(pbc_16bit, 0, max_16bit).astype(np.uint16)

    # Save images
    tifffile.imwrite(out_dir / "02_RAW.tif", pre_stack[0].astype(np.float32))
    tifffile.imwrite(out_dir / "03_AVG-PRE.tif", pre_avg.astype(np.float32))
    tifffile.imwrite(out_dir / "04_POST.tif", post_stack[0].astype(np.float32))
    tifffile.imwrite(out_dir / "05_AVG-POST.tif", post_avg.astype(np.float32))
    tifffile.imwrite(out_dir / "06_PBC_float32.tif", pbc_float.astype(np.float32))
    tifffile.imwrite(out_dir / "06_PBC_16bit.tif", pbc_16bit)

    print(f"Saved simulated frames and PBC results to: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic microtubule benchmark generation."
    )
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    p.add_argument(
        "--stage", 
        choices=["gt", "noise", "all"], 
        default="all", 
        help="Run specific stage."
    )
    p.add_argument(
        "--gt", 
        type=Path, 
        default=None, 
        help="Path to existing GT_235.tif (required if --stage noise)."
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    
    # Noise parameters
    p.add_argument("--n-frames", type=int, default=16, help="Number of pre/post frames.")
    p.add_argument("--photon-scale", type=float, default=120.0, help="Peak photon count P.")
    p.add_argument("--readout-sigma", type=float, default=8.0, help="Gaussian readout noise.")
    p.add_argument("--background", type=float, default=0.0, help="Background photons B.")
    p.add_argument("--bleach-factor", type=float, default=0.05, help="Post-bleach signal amplitude.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ["gt", "all"]:
        gt_path = run_gt_stage(args, args.out_dir)
    else:
        if args.gt is None:
            raise ValueError("--gt is required when --stage is noise.")
        gt_path = args.gt

    if args.stage in ["noise", "all"]:
        run_noise_stage(args, args.out_dir, gt_path)


if __name__ == "__main__":
    main()
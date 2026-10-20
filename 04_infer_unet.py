#!/usr/bin/env python3
"""Single-frame inference with PBC-net.

Takes raw confocal frames (1x sampling), performs true 2x super-resolution,
and saves the output as 16-bit TIFF. Optionally filters input files based
on a split_info.json file and records per-image timing statistics.
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import tifffile
import torch

from utils_model import LightweightUNet2x


def load_norm(path: Path) -> np.ndarray:
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 3:
        img = img[0] if img.shape[0] <= 4 else img.mean(axis=0)
    elif img.ndim > 3:
        img = img[0, 0]
    mn, mx = img.min(), img.max()
    if mx - mn > 1e-6:
        return (img - mn) / (mx - mn)
    return np.zeros_like(img)


def infer_single(model: torch.nn.Module, lr_img: np.ndarray, device: torch.device) -> np.ndarray:
    h0, w0 = lr_img.shape
    pad_h = (4 - h0 % 4) % 4
    pad_w = (4 - w0 % 4) % 4
    if pad_h > 0 or pad_w > 0:
        lr_img = np.pad(lr_img, ((0, pad_h), (0, pad_w)), mode="reflect")

    inp = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp).squeeze().cpu().numpy()
    
    pred = np.clip(pred, 0.0, 1.0)
    
    # Crop the padding. Since the network outputs true 2x, the padding is also 2x.
    if pad_h > 0:
        pred = pred[: -(pad_h * 2), :]
    if pad_w > 0:
        pred = pred[:, : -(pad_w * 2)]
        
    return pred


def _extract_basenames(obj) -> set:
    names = set()
    if isinstance(obj, list):
        for item in obj:
            names.update(_extract_basenames(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            names.update(_extract_basenames(v))
    elif isinstance(obj, str):
        names.add(os.path.basename(obj))
    return names


def filter_by_split(files: list, split_info_path: str, split_key: str) -> list:
    if not os.path.exists(split_info_path):
        raise FileNotFoundError(f"Split info not found: {split_info_path}")

    with open(split_info_path, "r", encoding="utf-8") as f:
        split_info = json.load(f)

    if split_key not in split_info:
        raise KeyError(f"Key '{split_key}' not found in {split_info_path}. Available: {list(split_info.keys())}")

    allowed_names = _extract_basenames(split_info[split_key])
    filtered = [fp for fp in files if os.path.basename(fp) in allowed_names]

    print(f"[Split Filter] Parsed {len(allowed_names)} unique filenames from key '{split_key}'")
    print(f"[Split Filter] Matched: {len(filtered)} | Filtered out: {len(files) - len(filtered)}")
    return filtered


def setup_device(mode: str) -> torch.device:
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if mode == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="PBC-net inference.")
    parser.add_argument("--in-dir", required=True, type=Path, help="Input directory with raw LR images.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for SR images.")
    parser.add_argument("--model", required=True, type=Path, help="Path to trained .pth model checkpoint.")
    parser.add_argument("--base-ch", type=int, default=32, help="Model base channel width.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--pattern", default="*.tif*", help="File pattern to match.")
    
    parser.add_argument("--split-info", type=str, default=None, help="Path to split_info.json.")
    parser.add_argument("--split-key", type=str, default="test", choices=["train", "val", "test"], help="Split key to filter.")
    args = parser.parse_args()

    device = setup_device(args.device)
    print(f"[Device] Using {device}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = LightweightUNet2x(base_ch=args.base_ch).to(device)
    print(f"[Model] Loading weights from {args.model} ...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Warmup
    dummy = torch.randn(1, 1, 512, 512).to(device)
    for _ in range(5):
        with torch.no_grad():
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    files = sorted(args.in_dir.glob(args.pattern))
    print(f"[Input] Found {len(files)} file(s) in {args.in_dir}")

    if args.split_info:
        files = filter_by_split(files, args.split_info, args.split_key)

    if not files:
        print("Warning: No files to process.")
        return

    print(f"[Process] Final number of files: {len(files)}\n")

    records = []
    for fpath in files:
        fname = fpath.name
        t_start = time.perf_counter()

        lr_raw = load_norm(fpath)
        t_after_read = time.perf_counter()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_infer_start = time.perf_counter()
        
        pred = infer_single(model, lr_raw, device)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_after_infer = time.perf_counter()

        pred_16bit = (pred * 65535).astype(np.uint16)
        out_path = args.out_dir / f"{fpath.stem}_PBCnet.tif"
        tifffile.imwrite(out_path, pred_16bit)
        t_after_save = time.perf_counter()

        t_read = (t_after_read - t_start) * 1000
        t_infer = (t_after_infer - t_infer_start) * 1000
        t_save = (t_after_save - t_after_infer) * 1000
        t_total = (t_after_save - t_start) * 1000

        records.append({
            "file": fname,
            "lr_size": f"{lr_raw.shape[0]}x{lr_raw.shape[1]}",
            "sr_size": f"{pred.shape[0]}x{pred.shape[1]}",
            "t_read_ms": round(t_read, 2),
            "t_infer_ms": round(t_infer, 2),
            "t_save_ms": round(t_save, 2),
            "t_total_ms": round(t_total, 2),
        })
        print(f"{fname}: LR={lr_raw.shape} -> SR={pred.shape} | Infer={t_infer:.1f}ms | Total={t_total:.1f}ms")

    csv_path = args.out_dir / "timing_per_image.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"\n{'=' * 60}")
    print(f"Processed {len(records)} images | Device: {device}")
    
    infer_times = [r["t_infer_ms"] for r in records]
    print(f"Inference time (ms): mean={np.mean(infer_times):.2f}, median={np.median(infer_times):.2f}")
    
    print(f"CSV saved to: {csv_path}")
    print(f"SR images saved to: {args.out_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
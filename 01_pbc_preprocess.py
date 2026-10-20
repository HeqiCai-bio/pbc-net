#!/usr/bin/env python3
"""PBC preprocessing for paired pre-bleach and post-bleach stacks.

The script averages pre-bleach and post-bleach stacks separately,
subtracts the post-bleach average from the pre-bleach average,
clips negative values to zero, and saves the PBC image.
"""

import argparse
from pathlib import Path

import numpy as np
from tifffile import imread, imwrite


def load_stack(filepath: Path, n_frames: int) -> np.ndarray:
    img = imread(filepath)
    if img.ndim == 2:
        if n_frames == 1:
            return img.astype(np.float32)
        raise ValueError(
            f"{filepath.name} is a single frame, but --n-frames={n_frames}"
        )
    if img.ndim == 3:
        if img.shape[0] == n_frames:
            return img.astype(np.float32)
        if img.shape[-1] == n_frames:
            return np.transpose(img, (2, 0, 1)).astype(np.float32)
        raise ValueError(
            f"{filepath.name}: expected {n_frames} frames, got shape={img.shape}"
        )
    raise ValueError(f"Unsupported image dimension: {img.ndim}")


def split_name_ext(filename: str):
    for ext in [".ome.tif", ".ome.tiff", ".tif", ".tiff", ".ome"]:
        if filename.lower().endswith(ext):
            return filename[: -len(ext)], ext
    return filename, ""


def strip_tag(base: str, tag: str) -> str:
    if tag and base.endswith(tag):
        return base[: -len(tag)]
    return base


def to_uint16_stretch(arr: np.ndarray, input_max: float) -> np.ndarray:
    scaled = (arr.astype(np.float64) / input_max) * 65535.0
    return np.clip(scaled, 0, 65535).astype(np.uint16)


def process_pair(
    pre_path: Path,
    post_path: Path,
    n_frames: int,
    mode: str,
    input_max: float,
):
    pre_stack = load_stack(pre_path, n_frames)
    post_stack = load_stack(post_path, n_frames)

    if pre_stack.shape[0] != post_stack.shape[0]:
        print(
            f"Warning: frame count mismatch for {pre_path.name} and "
            f"{post_path.name}: {pre_stack.shape[0]} vs {post_stack.shape[0]}"
        )

    pre_avg = np.mean(pre_stack, axis=0).astype(np.float64)
    post_avg = np.mean(post_stack, axis=0).astype(np.float64)

    corrected = pre_avg - post_avg
    corrected = np.clip(corrected, 0, None)

    if mode == "float32":
        return (
            corrected.astype(np.float32),
            pre_avg.astype(np.float32),
            post_avg.astype(np.float32),
        )
    if mode == "uint16_stretch":
        return (
            to_uint16_stretch(corrected, input_max),
            to_uint16_stretch(pre_avg, input_max),
            to_uint16_stretch(post_avg, input_max),
        )
    if mode == "uint16":
        return (
            np.clip(corrected, 0, 65535).astype(np.uint16),
            np.clip(pre_avg, 0, 65535).astype(np.uint16),
            np.clip(post_avg, 0, 65535).astype(np.uint16),
        )
    raise ValueError(f"Unknown output mode: {mode}")


def batch_process(
    pre_dir: Path,
    post_dir: Path,
    output_dir: Path,
    n_frames: int,
    mode: str,
    input_bit_depth: int,
    pre_tag: str,
    post_tag: str,
    save_averages: bool,
):
    if not pre_dir.exists():
        raise FileNotFoundError(f"PRE directory does not exist: {pre_dir}")
    if not post_dir.exists():
        raise FileNotFoundError(f"POST directory does not exist: {post_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_max = float(2**input_bit_depth - 1)

    pre_files = []
    for f in sorted(list(pre_dir.glob("*.tif*")) + list(pre_dir.glob("*.ome*"))):
        base, _ = split_name_ext(f.name)
        if base.endswith(pre_tag):
            pre_files.append(f)

    post_map = {}
    for f in sorted(list(post_dir.glob("*.tif*")) + list(post_dir.glob("*.ome*"))):
        base, _ = split_name_ext(f.name)
        if base.endswith(post_tag):
            fov = strip_tag(base, post_tag)
            post_map[fov] = f

    print(f"PRE directory: {pre_dir}")
    print(f"POST directory: {post_dir}")
    print(f"Output directory: {output_dir}")
    print(f"N_FRAMES = {n_frames}")
    print(f"Output mode: {mode}")
    if mode == "uint16_stretch":
        print(f"Input bit depth for stretch: {input_bit_depth}")
    print(f"Found {len(pre_files)} PRE stacks and {len(post_map)} POST stacks")
    print(f"Save pre/post averages: {save_averages}")

    n_ok = 0
    for pre_file in pre_files:
        pre_base, pre_ext = split_name_ext(pre_file.name)
        fov = strip_tag(pre_base, pre_tag)

        post_file = post_map.get(fov)
        if post_file is None:
            print(f"Skip {pre_file.name}: no matching POST file for FOV {fov}")
            continue

        print(f"Processing: {pre_file.name} <-> {post_file.name} ...", end=" ")
        try:
            pbc_img, pre_img, post_img = process_pair(
                pre_file, post_file, n_frames, mode, input_max
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        output_name = f"{fov}-PBC{pre_ext}"
        imwrite(output_dir / output_name, pbc_img)

        if save_averages:
            imwrite(output_dir / f"{fov}-pre_avg{pre_ext}", pre_img)
            imwrite(output_dir / f"{fov}-post_avg{pre_ext}", post_img)

        print(
            f"done (dtype={pbc_img.dtype}, "
            f"range={pbc_img.min()}-{pbc_img.max()})"
        )
        n_ok += 1

    print(f"Finished: {n_ok}/{len(pre_files)} pairs written to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="PBC preprocessing from paired pre/post-bleach stacks."
    )
    parser.add_argument("--pre-dir", required=True, type=Path)
    parser.add_argument("--post-dir", required=True, type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <pre-dir>/../PBC_results",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=20,
        help="Frames per stack. Experimental PBC default: 20.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["float32", "uint16_stretch", "uint16"],
        default="float32",
    )
    parser.add_argument(
        "--input-bit-depth",
        type=int,
        default=12,
        help="Used only by uint16_stretch.",
    )
    parser.add_argument("--pre-tag", default="-PRE")
    parser.add_argument("--post-tag", default="-POST")
    parser.add_argument(
        "--save-averages",
        action="store_true",
        help="Also save averaged pre-bleach and post-bleach images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.out_dir
    if output_dir is None:
        output_dir = args.pre_dir.parent / "PBC_results"

    batch_process(
        pre_dir=args.pre_dir,
        post_dir=args.post_dir,
        output_dir=output_dir,
        n_frames=args.n_frames,
        mode=args.output_mode,
        input_bit_depth=args.input_bit_depth,
        pre_tag=args.pre_tag,
        post_tag=args.post_tag,
        save_averages=args.save_averages,
    )


if __name__ == "__main__":
    main()
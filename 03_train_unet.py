#!/usr/bin/env python3
"""Train PBC-net: single-frame confocal to 2x PBC+RL super-resolution."""

import argparse
import glob
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tifffile
from skimage.metrics import structural_similarity as ssim_skimage
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.model import LightweightUNet2x


def get_organizer_type(fname: str) -> str:
    f = fname.upper()
    if "LYSO" in f or "LAMP" in f:
        return "LYSO"
    if "OMM" in f or "TOM" in f or "MITO" in f:
        return "OMM"
    if "ACTIN" in f or "FACTIN" in f or "F_ACTIN" in f or "FILA" in f:
        return "F-actin"
    return "Unknown"


def load_tif_norm(path: str) -> np.ndarray:
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 3:
        img = img[0] if img.shape[0] <= 4 else img.mean(axis=0)
    elif img.ndim > 3:
        img = img[0, 0]
    mn, mx = img.min(), img.max()
    if mx - mn > 1e-6:
        return (img - mn) / (mx - mn)
    return np.zeros_like(img)


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.channel = 1
        self.register_buffer(
            "window", self._create_window(window_size, self.channel)
        )

    def _create_window(self, window_size: int, channel: int) -> torch.Tensor:
        import math
        sigma = 1.5
        gauss = torch.tensor(
            [
                math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
                for x in range(window_size)
            ]
        )
        gauss = gauss / gauss.sum()
        window_1d = gauss.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
        return window_2d.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        c1, c2 = 0.01**2, 0.03**2
        pad = self.window_size // 2
        mu1 = F.conv2d(img1, self.window, padding=pad, groups=self.channel)
        mu2 = F.conv2d(img2, self.window, padding=pad, groups=self.channel)
        mu1_sq, mu2_sq = mu1**2, mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = (
            F.conv2d(img1 * img1, self.window, padding=pad, groups=self.channel)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(img2 * img2, self.window, padding=pad, groups=self.channel)
            - mu2_sq
        )
        sigma12 = (
            F.conv2d(img1 * img2, self.window, padding=pad, groups=self.channel)
            - mu1_mu2
        )
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return 1 - ssim_map.mean()


class ConfocalSRPatchDataset(Dataset):
    def __init__(
        self,
        file_pairs,
        patch_size: int = 256,
        patches_per_img: int = 16,
        seed: int = 42,
    ):
        self.file_pairs = file_pairs
        self.patch_size = patch_size
        self.patches_per_img = patches_per_img
        self.rng = np.random.RandomState(seed)

        print("Preloading training images and checking 2x relation ...")
        self.lr_imgs, self.hr_imgs = [], []
        valid_pairs = []
        skipped = 0
        total_pixels = 0

        for lr_path, hr_path in file_pairs:
            lr = load_tif_norm(lr_path)
            hr = load_tif_norm(hr_path)
            rh = hr.shape[0] / lr.shape[0]
            rw = hr.shape[1] / lr.shape[1]
            if not (1.98 < rh < 2.02 and 1.98 < rw < 2.02):
                print(
                    f"  [WARN] skip {os.path.basename(lr_path)}: "
                    f"LR={lr.shape}, HR={hr.shape} is not 2x"
                )
                skipped += 1
                continue
            h_target, w_target = lr.shape[0] * 2, lr.shape[1] * 2
            if hr.shape[0] != h_target or hr.shape[1] != w_target:
                hr = hr[:h_target, :w_target]
            self.lr_imgs.append(lr)
            self.hr_imgs.append(hr)
            valid_pairs.append((lr_path, hr_path))
            total_pixels += lr.size + hr.size

        self.file_pairs = valid_pairs
        if skipped:
            print(f"  [WARN] skipped {skipped} pairs that are not 2x")

        org_counts = defaultdict(int)
        for lr_path, _ in self.file_pairs:
            org_counts[get_organizer_type(os.path.basename(lr_path))] += 1
        print(
            f"  Train: {len(self.file_pairs)} FOVs | {dict(org_counts)} | "
            f"~{total_pixels * 4 / 1e6:.1f} MB (float32)"
        )

    def __len__(self) -> int:
        return len(self.file_pairs) * self.patches_per_img

    def __getitem__(self, idx: int):
        pair_idx = idx // self.patches_per_img
        lr = self.lr_imgs[pair_idx]
        hr = self.hr_imgs[pair_idx]

        ps_lr = self.patch_size
        ps_hr = self.patch_size * 2

        h_lr, w_lr = lr.shape
        if h_lr < ps_lr or w_lr < ps_lr:
            pad_h = max(0, ps_lr - h_lr)
            pad_w = max(0, ps_lr - w_lr)
            lr = np.pad(lr, ((0, pad_h), (0, pad_w)), mode="reflect")
            hr = np.pad(hr, ((0, pad_h * 2), (0, pad_w * 2)), mode="reflect")
            h_lr, w_lr = lr.shape

        top_lr = self.rng.randint(0, h_lr - ps_lr + 1)
        left_lr = self.rng.randint(0, w_lr - ps_lr + 1)

        lr_patch = lr[top_lr : top_lr + ps_lr, left_lr : left_lr + ps_lr].copy()
        hr_patch = hr[
            top_lr * 2 : top_lr * 2 + ps_hr,
            left_lr * 2 : left_lr * 2 + ps_hr,
        ].copy()

        if self.rng.rand() > 0.5:
            lr_patch = np.fliplr(lr_patch)
            hr_patch = np.fliplr(hr_patch)
        if self.rng.rand() > 0.5:
            lr_patch = np.flipud(lr_patch)
            hr_patch = np.flipud(hr_patch)
        if self.rng.rand() > 0.5:
            k = self.rng.choice([1, 2, 3])
            lr_patch = np.rot90(lr_patch, k)
            hr_patch = np.rot90(hr_patch, k)

        lr_patch = np.ascontiguousarray(lr_patch)
        hr_patch = np.ascontiguousarray(hr_patch)

        return (
            torch.from_numpy(lr_patch).unsqueeze(0).float(),
            torch.from_numpy(hr_patch).unsqueeze(0).float(),
        )


class ConfocalSRFullImageDataset(Dataset):
    def __init__(self, file_pairs, name: str = "Val"):
        self.file_pairs = []
        self.lr_padded, self.hr_imgs = [], []
        self.shapes = []

        print(f"Preloading {name} full images ...")
        skipped = 0
        for lr_path, hr_path in file_pairs:
            lr = load_tif_norm(lr_path)
            hr = load_tif_norm(hr_path)
            rh = hr.shape[0] / lr.shape[0]
            rw = hr.shape[1] / lr.shape[1]
            if not (1.98 < rh < 2.02 and 1.98 < rw < 2.02):
                print(
                    f"  [WARN] {name} skip {os.path.basename(lr_path)}: "
                    f"shape is not 2x"
                )
                skipped += 1
                continue
            h_target, w_target = lr.shape[0] * 2, lr.shape[1] * 2
            if hr.shape[0] != h_target or hr.shape[1] != w_target:
                hr = hr[:h_target, :w_target]

            h_lr, w_lr = lr.shape
            h_hr, w_hr = hr.shape

            pad_h = (4 - h_lr % 4) % 4
            pad_w = (4 - w_lr % 4) % 4
            if pad_h > 0 or pad_w > 0:
                lr_p = np.pad(lr, ((0, pad_h), (0, pad_w)), mode="reflect")
            else:
                lr_p = lr

            self.lr_padded.append(lr_p)
            self.hr_imgs.append(hr)
            self.shapes.append((h_lr, w_lr, h_hr, w_hr))
            self.file_pairs.append((lr_path, hr_path))

        if skipped:
            print(f"  [WARN] {name} skipped {skipped} pairs")

        org_counts = defaultdict(int)
        for lr_path, _ in self.file_pairs:
            org_counts[get_organizer_type(os.path.basename(lr_path))] += 1
        print(f"  {name}: {len(self.file_pairs)} FOVs | {dict(org_counts)}")

    def __len__(self) -> int:
        return len(self.file_pairs)

    def __getitem__(self, idx: int):
        lr_p = self.lr_padded[idx]
        hr = self.hr_imgs[idx]
        fname = os.path.basename(self.file_pairs[idx][0])
        h_lr, w_lr, h_hr, w_hr = self.shapes[idx]
        shape_tensor = torch.tensor([h_lr, w_lr, h_hr, w_hr], dtype=torch.long)
        return (
            torch.from_numpy(lr_p).unsqueeze(0).float(),
            torch.from_numpy(hr).unsqueeze(0).float(),
            fname,
            shape_tensor,
        )


def stratified_split(
    lr_dir: str,
    hr_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    split_seed: int = 42,
):
    lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.tif*")))
    org_groups = {"LYSO": [], "OMM": [], "F-actin": [], "Unknown": []}
    for lr_path in lr_files:
        fname = os.path.basename(lr_path)
        hr_path = os.path.join(hr_dir, fname)
        if not os.path.exists(hr_path):
            continue
        org_groups[get_organizer_type(fname)].append((lr_path, hr_path))

    train_pairs, val_pairs, test_pairs = [], [], []
    split_info = {
        "meta": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": 1 - train_ratio - val_ratio,
            "split_seed": split_seed,
        },
        "train": {},
        "val": {},
        "test": {},
    }

    rng = np.random.RandomState(split_seed)

    for org, pairs in org_groups.items():
        n = len(pairs)
        if n == 0:
            continue
        indices = rng.permutation(n)

        n_test = max(1, int(n * (1 - train_ratio - val_ratio))) if n >= 3 else 0
        n_val = max(1, int(n * val_ratio)) if n >= 2 else 0
        n_train = n - n_val - n_test

        if n_train <= 0:
            n_train = max(1, n - 1)
            n_val = n - n_train
            n_test = 0
        if n == 1:
            n_train, n_val, n_test = 1, 0, 0

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        for i in train_idx:
            train_pairs.append(pairs[i])
        for i in val_idx:
            val_pairs.append(pairs[i])
        for i in test_idx:
            test_pairs.append(pairs[i])

        split_info["train"][org] = [os.path.basename(pairs[i][0]) for i in train_idx]
        split_info["val"][org] = [os.path.basename(pairs[i][0]) for i in val_idx]
        split_info["test"][org] = [os.path.basename(pairs[i][0]) for i in test_idx]
        print(
            f"  {org}: total={n}, train={len(train_idx)}, "
            f"val={len(val_idx)}, test={len(test_idx)}"
        )

    return train_pairs, val_pairs, test_pairs, split_info


def evaluate_full_fov(model, dataset, device):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    per_fov = []

    with torch.no_grad():
        for lr, hr, fname, shape_tensor in loader:
            h_lr, w_lr, h_hr, w_hr = shape_tensor.squeeze(0).tolist()
            fname_str = fname[0] if isinstance(fname, (list, tuple)) else fname

            lr = lr.to(device)
            pred = model(lr)
            pred_np = np.clip(pred.squeeze(0).squeeze(0).cpu().numpy(), 0.0, 1.0)
            hr_np = hr.squeeze(0).squeeze(0).numpy()

            pred_np = pred_np[:h_hr, :w_hr]
            hr_np = hr_np[:h_hr, :w_hr]

            s = ssim_skimage(pred_np, hr_np, data_range=1.0)
            mse = float(np.mean((pred_np - hr_np) ** 2))
            p = 10 * np.log10(1.0 / (mse + 1e-10))

            per_fov.append(
                {
                    "fname": fname_str,
                    "organiser": get_organizer_type(fname_str),
                    "ssim": float(s),
                    "psnr": float(p),
                }
            )

    ssims = np.array([r["ssim"] for r in per_fov]) if per_fov else np.array([0.0])
    psnrs = np.array([r["psnr"] for r in per_fov]) if per_fov else np.array([0.0])

    by_org = {}
    for org in sorted(set(r["organiser"] for r in per_fov)):
        vals = [r for r in per_fov if r["organiser"] == org]
        ss = np.array([v["ssim"] for v in vals])
        pp = np.array([v["psnr"] for v in vals])
        by_org[org] = {
            "n_fov": len(vals),
            "ssim_mean": float(ss.mean()),
            "ssim_std": float(ss.std()),
            "psnr_mean": float(pp.mean()),
            "psnr_std": float(pp.std()),
            "fovs": [v["fname"] for v in vals],
        }

    summary = {
        "overall_ssim_mean": float(ssims.mean()),
        "overall_ssim_std": float(ssims.std()),
        "overall_psnr_mean": float(psnrs.mean()),
        "overall_psnr_std": float(psnrs.std()),
        "n_fov": len(per_fov),
        "by_organiser": by_org,
    }
    return per_fov, summary


def save_training_curves(
    history,
    output_path,
    best_ssim,
    best_epoch,
    test_ssim=None,
    test_psnr=None,
    dpi=600,
):
    rc = {
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.dpi": dpi,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        epochs = history["epoch"]

        axes[0].plot(epochs, history["train_loss"], "b-", lw=2.0, label="Training Loss")
        axes[0].set_title("Training Loss", fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (L1 + 0.5 SSIM)")
        axes[0].grid(True, alpha=0.3, linestyle="--")
        axes[0].legend(frameon=True)

        axes[1].plot(epochs, history["val_ssim"], "g-", lw=2.0, label="Validation SSIM")
        axes[1].axhline(
            best_ssim,
            color="r",
            linestyle="--",
            lw=1.5,
            alpha=0.7,
            label=f"Best = {best_ssim:.4f} (Epoch {best_epoch})",
        )
        axes[1].set_title("Validation SSIM", fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("SSIM (image-level)")
        axes[1].grid(True, alpha=0.3, linestyle="--")
        axes[1].legend(frameon=True)
        vs = np.asarray(history["val_ssim"], dtype=float)
        if np.isfinite(vs).any():
            axes[1].set_ylim([float(np.nanmin(vs)) - 0.02, 1.0])

        if history.get("val_psnr") and len(history["val_psnr"]) == len(epochs):
            axes[2].plot(
                epochs, history["val_psnr"], "m-", lw=2.0, label="Validation PSNR"
            )
            axes[2].set_title("Validation PSNR", fontweight="bold")
            axes[2].set_xlabel("Epoch")
            axes[2].set_ylabel("PSNR (dB)")
            axes[2].grid(True, alpha=0.3, linestyle="--")
            axes[2].legend(frameon=True)
        else:
            axes[2].text(
                0.5,
                0.5,
                "PSNR data unavailable",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
                fontsize=14,
                color="gray",
            )
            axes[2].set_title("Validation PSNR", fontweight="bold")

        if test_ssim is not None and test_psnr is not None:
            fig.text(
                0.5,
                -0.02,
                f"Independent Test Set (image-level): "
                f"SSIM = {test_ssim:.4f} | PSNR = {test_psnr:.2f} dB "
                f"(Best Val SSIM = {best_ssim:.4f} @ Epoch {best_epoch})",
                ha="center",
                fontsize=12,
                style="italic",
                color="navy",
            )

        output_path = str(output_path)
        plt.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.15,
            facecolor="white",
            edgecolor="none",
        )
        pdf_path = os.path.splitext(output_path)[0] + ".pdf"
        plt.savefig(
            pdf_path,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.15,
            facecolor="white",
            edgecolor="none",
        )
        plt.close()


def save_preview(
    model,
    val_set,
    epoch,
    output_dir,
    device,
    n_samples=3,
    lr_size=512,
):
    model.eval()
    rng = np.random.RandomState(epoch)
    n = min(n_samples, len(val_set))
    if n == 0:
        model.train()
        return
    sample_indices = rng.choice(len(val_set), size=n, replace=False)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    try:
        with torch.no_grad():
            for row, idx in enumerate(sample_indices):
                fname = os.path.basename(val_set.file_pairs[idx][0])
                lr_full = val_set.lr_padded[idx]
                hr_full = val_set.hr_imgs[idx]
                h_lr, w_lr, h_hr, w_hr = val_set.shapes[idx]

                lr_crop_size = min(lr_size, h_lr, w_lr)
                lr_crop_size = (lr_crop_size // 4) * 4
                if lr_crop_size < 4:
                    continue

                top = (h_lr - lr_crop_size) // 2
                left = (w_lr - lr_crop_size) // 2
                lr_crop = lr_full[
                    top : top + lr_crop_size, left : left + lr_crop_size
                ]

                hr_top = top * 2
                hr_left = left * 2
                hr_h = min(lr_crop_size * 2, h_hr - hr_top)
                hr_w = min(lr_crop_size * 2, w_hr - hr_left)
                hr_crop = hr_full[hr_top : hr_top + hr_h, hr_left : hr_left + hr_w]

                h_pad = (4 - lr_crop.shape[0] % 4) % 4
                w_pad = (4 - lr_crop.shape[1] % 4) % 4
                if h_pad > 0 or w_pad > 0:
                    lr_crop = np.pad(
                        lr_crop, ((0, h_pad), (0, w_pad)), mode="reflect"
                    )

                inp = torch.from_numpy(lr_crop).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(inp).squeeze().cpu().numpy()
                pred = np.clip(pred, 0.0, 1.0)

                if h_pad > 0:
                    pred = pred[: -(h_pad * 2), :]
                if w_pad > 0:
                    pred = pred[:, : -(w_pad * 2)]

                hh = min(pred.shape[0], hr_crop.shape[0])
                ww = min(pred.shape[1], hr_crop.shape[1])
                hh = (hh // 2) * 2
                ww = (ww // 2) * 2
                pred = pred[:hh, :ww]
                hr_crop = hr_crop[:hh, :ww]
                lr_crop = lr_crop[: hh // 2, : ww // 2]

                axes[row, 0].imshow(lr_crop, cmap="gray")
                axes[row, 0].set_title(f"Input (1x): {fname[:20]}...", fontsize=8)
                axes[row, 1].imshow(pred, cmap="gray")
                axes[row, 1].set_title("PBC-net prediction (2x)", fontsize=8)
                axes[row, 2].imshow(hr_crop, cmap="gray")
                axes[row, 2].set_title("PBC+RL target (2x)", fontsize=8)
                for ax in axes[row]:
                    ax.axis("off")

        plt.tight_layout()
        out_path = output_dir / f"preview_epoch{epoch:03d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close()
        model.train()


def setup_device(mode: str) -> torch.device:
    if mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if mode == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(args) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = setup_device(args.device)
    output_dir = Path(args.out_base) / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = output_dir / f"pbcnet_seed{args.seed}_best.pth"
    hist_path = output_dir / "training_history_live.json"

    if args.force_fresh_start:
        for p in [ckpt_path, hist_path]:
            if p.exists():
                p.unlink()
                print(f"[Reset] removed {p}")

    print("=" * 60)
    print(
        f"SEED = {args.seed} | SPLIT_SEED = {args.split_seed} | "
        f"RESUME = {args.resume}"
    )
    print("Stratified train/val/test split ...")
    train_pairs, val_pairs, test_pairs, split_info = stratified_split(
        args.lr_dir,
        args.hr_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
    )
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    train_set = ConfocalSRPatchDataset(
        train_pairs,
        patch_size=args.patch_size,
        patches_per_img=args.patches_per_image,
        seed=args.seed,
    )
    val_set = ConfocalSRFullImageDataset(val_pairs, name="Val")
    test_set = ConfocalSRFullImageDataset(test_pairs, name="Test")

    g_train = torch.Generator()
    g_train.manual_seed(args.seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=g_train,
        pin_memory=pin,
    )

    model = LightweightUNet2x(base_ch=args.base_ch).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n{'=' * 60}")
    print(
        f"Model: LightweightUNet2x | BaseCH={args.base_ch} | "
        f"Params={n_params:.3f}M"
    )
    print(
        f"Super-resolution factor: 2x "
        f"(LR {args.patch_size} -> HR {args.patch_size * 2})"
    )
    print(
        f"Loss: L1 + {args.ssim_weight} * SSIM | Device: {device} | "
        f"Batch={args.batch_size}"
    )
    print(
        f"Train/Val/Test: {len(train_pairs)}/{len(val_pairs)}/"
        f"{len(test_pairs)} FOVs"
    )
    print(f"{'=' * 60}\n")

    l1_loss = nn.L1Loss()
    ssim_loss = SSIMLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=10, factor=0.5
    )

    history = {"epoch": [], "train_loss": [], "val_ssim": [], "val_psnr": []}
    best_ssim = -float("inf")
    best_epoch = 0
    start_epoch = 1

    if args.resume and hist_path.exists():
        try:
            with open(hist_path) as f:
                history = json.load(f)
            if history["epoch"]:
                start_epoch = max(history["epoch"]) + 1
                print(f"[Resume] history restored to epoch {start_epoch - 1}")
        except Exception as exc:
            print(f"[Resume] history load failed, starting fresh: {exc}")
            history = {"epoch": [], "train_loss": [], "val_ssim": [], "val_psnr": []}
            start_epoch = 1

    if args.resume and ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as exc:
                print(f"[Resume] optimizer state not loaded: {exc}")
            best_ssim = ckpt["best_ssim"]
            best_epoch = ckpt["epoch"]
            print(
                f"[Resume] model restored from epoch {best_epoch} | "
                f"best_ssim={best_ssim:.4f}"
            )
        except Exception as exc:
            print(f"[Resume] checkpoint load failed, starting fresh: {exc}")
            best_ssim = -float("inf")
            best_epoch = 0
            start_epoch = 1
            history = {"epoch": [], "train_loss": [], "val_ssim": [], "val_psnr": []}

    if start_epoch > args.epochs:
        print(
            f"[Resume] already reached epoch {start_epoch - 1} "
            f">= {args.epochs}, running final evaluation."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        model.train()
        epoch_loss = 0.0
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs}",
            disable=args.no_progress,
        )
        for lr, hr in pbar:
            lr = lr.to(device, non_blocking=pin)
            hr = hr.to(device, non_blocking=pin)
            pred = model(lr)
            loss = l1_loss(pred, hr) + args.ssim_weight * ssim_loss(pred, hr)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / len(train_loader)

        _, val_summary = evaluate_full_fov(model, val_set, device)
        avg_ssim = val_summary["overall_ssim_mean"]
        avg_psnr = val_summary["overall_psnr_mean"]

        history["epoch"].append(epoch)
        history["train_loss"].append(float(avg_train_loss))
        history["val_ssim"].append(float(avg_ssim))
        history["val_psnr"].append(float(avg_psnr))

        scheduler.step(avg_ssim)
        epoch_time = time.time() - t0

        if epoch % args.save_curve_every == 0 or epoch == 1:
            curve_name = f"training_curves_epoch{epoch:03d}.png"
            save_training_curves(
                history,
                output_dir / curve_name,
                best_ssim if np.isfinite(best_ssim) else 0.0,
                best_epoch,
                dpi=150,
            )
            with open(hist_path, "w") as f:
                json.dump(history, f, indent=2)

        if epoch % args.save_preview_every == 0 or epoch == args.epochs:
            save_preview(
                model,
                val_set,
                epoch,
                output_dir,
                device,
                args.preview_samples,
                args.preview_lr_size,
            )

        if avg_ssim > best_ssim:
            best_ssim = avg_ssim
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_ssim": best_ssim,
                    "config": {
                        "base_ch": args.base_ch,
                        "patch_size": args.patch_size,
                        "seed": args.seed,
                        "split_seed": args.split_seed,
                    },
                },
                ckpt_path,
            )
            print(f"  >>> New best at epoch {epoch}: Val SSIM={avg_ssim:.4f}")

        print(
            f"Epoch {epoch} ({epoch_time:.1f}s): Loss={avg_train_loss:.4f} | "
            f"Val SSIM={avg_ssim:.4f} | Val PSNR={avg_psnr:.2f} dB | "
            f"Best={best_ssim:.4f}@{best_epoch}"
        )

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Saved] training_history.json ({len(history['epoch'])} epochs)")

    print(f"\n{'=' * 60}")
    print("Final evaluation on independent test set ...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_per_fov, test_summary = evaluate_full_fov(model, test_set, device)
    final_test_ssim = test_summary["overall_ssim_mean"]
    final_test_psnr = test_summary["overall_psnr_mean"]

    print(f"\n{'=' * 60}")
    print("FINAL INDEPENDENT TEST SET RESULTS (2x SR, image-level):")
    print(
        f"  Overall SSIM = {final_test_ssim:.4f} "
        f"+/- {test_summary['overall_ssim_std']:.4f}"
    )
    print(
        f"  Overall PSNR = {final_test_psnr:.2f} "
        f"+/- {test_summary['overall_psnr_std']:.2f} dB"
    )
    print(f"  Test set size: {len(test_pairs)} FOVs")
    print("  --- by organelle ---")
    for org, st in test_summary["by_organiser"].items():
        print(
            f"    {org:10s}: n={st['n_fov']:2d} | "
            f"SSIM={st['ssim_mean']:.4f}+/-{st['ssim_std']:.4f} | "
            f"PSNR={st['psnr_mean']:.2f}+/-{st['psnr_std']:.2f} dB"
        )
    print(f"{'=' * 60}")

    test_eval = {
        "seed": args.seed,
        "split_seed": args.split_seed,
        "protocol": "image-level full FOV, 2x SR",
        "best_val_ssim": float(best_ssim),
        "best_val_epoch": int(best_epoch),
        "test_summary": test_summary,
        "test_per_fov": test_per_fov,
    }
    with open(output_dir / "test_evaluation_summary.json", "w") as f:
        json.dump(test_eval, f, indent=2)
    print("[Saved] test_evaluation_summary.json")

    save_training_curves(
        history,
        output_dir / "training_curves_final.png",
        best_ssim,
        best_epoch,
        test_ssim=final_test_ssim,
        test_psnr=final_test_psnr,
        dpi=args.dpi,
    )
    print(f"[Saved] training_curves_final.png / .pdf at {args.dpi} dpi")
    print(f"\nTraining finished. Best Val SSIM: {best_ssim:.4f} (Epoch {best_epoch})")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train PBC-net (single-frame 2x super-resolution)."
    )
    p.add_argument("--lr-dir", required=True, help="Directory with LR .tif files.")
    p.add_argument("--hr-dir", required=True, help="Directory with HR .tif files.")
    p.add_argument("--out-base", required=True, help="Base output directory.")
    p.add_argument("--base-ch", type=int, default=32, help="Base channel width.")
    p.add_argument("--seed", type=int, default=42, help="Training seed.")
    p.add_argument("--split-seed", type=int, default=42, help="Data split seed.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=256, help="LR patch size.")
    p.add_argument("--patches-per-image", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4, help="Adam learning rate.")
    p.add_argument("--ssim-weight", type=float, default=0.5)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force-fresh-start", action="store_true")
    p.add_argument("--save-curve-every", type=int, default=5)
    p.add_argument("--save-preview-every", type=int, default=10)
    p.add_argument("--preview-samples", type=int, default=3)
    p.add_argument("--preview-lr-size", type=int, default=512)
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
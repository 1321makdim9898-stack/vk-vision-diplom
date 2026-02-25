"""
train_age_gender_utk_dldl_resnet50.py

DLDL (Label Distribution Learning) for AGE + gender classification (UTKFace).
- Predicts age distribution over 0..120 (121 classes)
- Soft target: Gaussian around true age (sigma)
- Age estimate: expected value sum(p * age)

Metrics for VKR:
- MAE years (main)
- Bin Accuracy (0-12,13-25,26-40,41-60,61+)
- Acc@5 / Acc@10
- Gender accuracy

Recommended: train on MediaPipe-cropped dataset to reduce domain shift:
C:\\vk-vision-demo\\data\\UTKFace\\utkface_mediapipe_cropped
"""

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm


# -----------------------------
# Settings / bins
# -----------------------------
MAX_AGE = 120
NUM_AGES = MAX_AGE + 1

AGE_BINS = [(0, 12), (13, 25), (26, 40), (41, 60), (61, 120)]

def age_to_bin_idx(age_years: float) -> int:
    a = max(0.0, min(float(age_years), 120.0))
    for i, (lo, hi) in enumerate(AGE_BINS):
        if lo <= a <= hi:
            return i
    return len(AGE_BINS) - 1


# UTKFace filename: [age]_[gender]_[race]_[date&time].jpg ; gender: 0=male, 1=female
def parse_utk_filename(path: str) -> Tuple[int, int]:
    name = os.path.basename(path)
    parts = name.split("_")
    age = int(parts[0])
    gender = int(parts[1])
    return age, gender


@dataclass
class Sample:
    path: str
    age: int
    gender: int


def load_samples(data_root: Path) -> List[Sample]:
    paths = glob.glob(str(data_root / "**/*.jpg"), recursive=True)
    samples: List[Sample] = []
    for p in paths:
        try:
            age, gender = parse_utk_filename(p)
        except Exception:
            continue
        if gender not in (0, 1):
            continue
        if age < 0 or age > 120:
            continue
        samples.append(Sample(path=p, age=age, gender=gender))
    return samples


# -----------------------------
# DLDL soft label
# -----------------------------
def gaussian_soft_label(age: int, sigma: float, device=None) -> torch.Tensor:
    """
    Returns distribution over ages 0..120 as float tensor [121]
    """
    ages = torch.arange(0, NUM_AGES, dtype=torch.float32, device=device)
    mu = float(age)
    dist = torch.exp(-0.5 * ((ages - mu) / sigma) ** 2)
    dist = dist / (dist.sum() + 1e-12)
    return dist


class UtkDLDLDataset(Dataset):
    def __init__(self, samples: List[Sample], transform=None, sigma: float = 3.0):
        self.samples = samples
        self.transform = transform
        self.sigma = sigma

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            img = Image.open(s.path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        # soft label distribution on CPU here; move later to device in training loop
        y_age = gaussian_soft_label(s.age, self.sigma, device=None)  # [121] on CPU
        y_gender = torch.tensor(s.gender, dtype=torch.long)

        return img, y_age, torch.tensor(s.age, dtype=torch.long), y_gender


# -----------------------------
# Model
# -----------------------------
class AgeGenderDLDLResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head_age = nn.Linear(in_features, NUM_AGES)  # 121
        self.head_gender = nn.Linear(in_features, 2)

    def forward(self, x):
        feats = self.backbone(x)
        age_logits = self.head_age(feats)      # [B,121]
        gender_logits = self.head_gender(feats) # [B,2]
        return age_logits, gender_logits


# -----------------------------
# Loss (soft cross-entropy)
# -----------------------------
def soft_ce_loss(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    """
    logits: [B,121]
    soft_targets: [B,121] sums to 1
    returns scalar
    """
    logp = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * logp).sum(dim=1).mean()
    return loss


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def metrics_from_age_probs(age_probs: torch.Tensor, true_ages: torch.Tensor):
    """
    age_probs: [B,121] (softmax)
    true_ages: [B] long
    returns: pred_age_years [B] float
    """
    device = age_probs.device
    ages = torch.arange(0, NUM_AGES, device=device, dtype=torch.float32).view(1, -1)  # [1,121]
    pred_age = (age_probs * ages).sum(dim=1)  # expected value
    true_age = true_ages.float()
    abs_err = torch.abs(pred_age - true_age)

    mae = abs_err.mean().item()
    acc5 = (abs_err <= 5.0).float().mean().item()
    acc10 = (abs_err <= 10.0).float().mean().item()

    # bin acc
    pred_bins = torch.tensor([age_to_bin_idx(v) for v in pred_age.detach().cpu().tolist()], dtype=torch.long)
    true_bins = torch.tensor([age_to_bin_idx(v) for v in true_age.detach().cpu().tolist()], dtype=torch.long)
    bin_acc = (pred_bins == true_bins).float().mean().item()

    return pred_age, mae, bin_acc, acc5, acc10


# -----------------------------
# Transforms
# -----------------------------
def make_transforms():
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, val_tf


def make_balanced_sampler(samples: List[Sample]) -> WeightedRandomSampler:
    # balance by age bins (helps reduce regression-to-the-mean)
    bins = [age_to_bin_idx(s.age) for s in samples]
    counts = np.bincount(np.array(bins, dtype=np.int64), minlength=len(AGE_BINS)).astype(np.float32)
    weights_by_bin = 1.0 / np.clip(counts, 1.0, None)
    weights = np.array([weights_by_bin[b] for b in bins], dtype=np.float32)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    return sampler


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    total = float(counts.sum())
    weights = total / (num_classes * np.clip(counts, 1.0, None))
    return torch.tensor(weights, dtype=torch.float32)


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def eval_epoch(model, loader, device, gender_loss_fn):
    model.eval()

    total = 0
    loss_sum = 0.0

    mae_sum = 0.0
    bin_acc_sum = 0.0
    acc5_sum = 0.0
    acc10_sum = 0.0

    gender_correct = 0

    for x, y_soft, y_age_int, y_gender in tqdm(loader, desc="Val", leave=False):
        x = x.to(device)
        y_soft = y_soft.to(device)
        y_age_int = y_age_int.to(device)
        y_gender = y_gender.to(device)

        age_logits, gender_logits = model(x)

        loss_age = soft_ce_loss(age_logits, y_soft)
        loss_gender = gender_loss_fn(gender_logits, y_gender)
        loss = loss_age + loss_gender

        bs = x.size(0)
        total += bs
        loss_sum += loss.item() * bs

        age_probs = F.softmax(age_logits, dim=1)
        _, mae, bin_acc, acc5, acc10 = metrics_from_age_probs(age_probs, y_age_int)

        mae_sum += mae * bs
        bin_acc_sum += bin_acc * bs
        acc5_sum += acc5 * bs
        acc10_sum += acc10 * bs

        g_pred = gender_logits.argmax(dim=1)
        gender_correct += (g_pred == y_gender).sum().item()

    return (
        loss_sum / total,
        mae_sum / total,
        bin_acc_sum / total,
        acc5_sum / total,
        acc10_sum / total,
        gender_correct / total
    )


def train():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=r"data/utkface_mediapipe_cropped")
    ap.add_argument("--save_path", type=str, default=r"models/age_gender_dldl_resnet50.pth")

    # "serious" defaults for GPU
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=128)          # 128 is a good start for RTX 3060 12GB
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)

    # DLDL params
    ap.add_argument("--sigma", type=float, default=3.0)        # try 2.5..4.0
    ap.add_argument("--age_loss_w", type=float, default=1.0)
    ap.add_argument("--gender_loss_w", type=float, default=1.0)

    # training tricks
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_sampler", action="store_true", help="Balance sampling by age bins")
    ap.add_argument("--accum", type=int, default=1, help="Gradient accumulation steps (use if OOM)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)
    print("[INFO] data_root:", args.data_root)
    print("[INFO] sigma:", args.sigma)

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    samples = load_samples(data_root)
    if not samples:
        raise RuntimeError(f"No samples found in {data_root}")

    rng = np.random.default_rng(args.seed)
    rng.shuffle(samples)

    val_size = int(0.2 * len(samples))
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    train_tf, val_tf = make_transforms()
    train_ds = UtkDLDLDataset(train_samples, transform=train_tf, sigma=args.sigma)
    val_ds = UtkDLDLDataset(val_samples, transform=val_tf, sigma=args.sigma)

    pin_memory = (device == "cuda")

    if args.use_sampler:
        sampler = make_balanced_sampler(train_samples)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, sampler=sampler,
            num_workers=args.num_workers, pin_memory=pin_memory
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True,
            num_workers=args.num_workers, pin_memory=pin_memory
        )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )

    # gender class weights
    gender_weights = compute_class_weights([s.gender for s in train_samples], num_classes=2).to(device)
    gender_loss_fn = nn.CrossEntropyLoss(weight=gender_weights)

    model = AgeGenderDLDLResNet50().to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        loss_sum = 0.0

        opt.zero_grad(set_to_none=True)

        for step, (x, y_soft, y_age_int, y_gender) in enumerate(tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", leave=False), start=1):
            x = x.to(device)
            y_soft = y_soft.to(device)
            y_gender = y_gender.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                age_logits, gender_logits = model(x)

                loss_age = soft_ce_loss(age_logits, y_soft) * args.age_loss_w
                loss_gender = gender_loss_fn(gender_logits, y_gender) * args.gender_loss_w
                loss = loss_age + loss_gender
                loss = loss / max(1, args.accum)

            scaler.scale(loss).backward()

            if step % max(1, args.accum) == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            bs = x.size(0)
            total += bs
            loss_sum += (loss.item() * max(1, args.accum)) * bs

        sch.step()
        train_loss = loss_sum / total

        val_loss, val_mae, val_bin_acc, val_acc5, val_acc10, val_gender_acc = eval_epoch(
            model, val_loader, device, gender_loss_fn
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_mae_years={val_mae:.2f} | val_age_bin_acc={val_bin_acc:.4f} | "
            f"val_acc@5={val_acc5:.4f} | val_acc@10={val_acc10:.4f} | "
            f"val_gender_acc={val_gender_acc:.4f}"
        )

        # Best selection for VKR: primarily MAE, then bin_acc, then gender_acc
        score = (val_mae, -val_bin_acc, -val_gender_acc)
        if best is None or score < best[0]:
            best = (score, epoch, val_mae, val_bin_acc, val_gender_acc, val_acc5, val_acc10)
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] saved best -> {save_path}")

    print("\nBest:", best)


if __name__ == "__main__":
    train()

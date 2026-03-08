# train_age_gender_utk_regression_resnet_v2.py
import os
import re
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from tqdm import tqdm

# ✅ ВАЖНО: headless backend matplotlib (исправляет Tkinter/Tcl_AsyncDelete на Windows)
import matplotlib
matplotlib.use("Agg")  # must be BEFORE importing pyplot
import matplotlib.pyplot as plt


AGE_MIN = 0
AGE_MAX = 120

BIN_LABELS = ["0-12", "13-25", "26-40", "41-60", "61+"]
BIN_TO_IDX = {b: i for i, b in enumerate(BIN_LABELS)}


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def parse_utk_filename(fn: str) -> Optional[Tuple[int, int]]:
    base = os.path.basename(fn)
    m = re.match(r"^(\d{1,3})_(\d)_(\d)_(\d+).*?\.(jpg|jpeg|png|webp)$", base, flags=re.IGNORECASE)
    if not m:
        return None
    age = int(m.group(1))
    gender = int(m.group(2))  # 0 female, 1 male
    if age < AGE_MIN or age > AGE_MAX:
        return None
    if gender not in (0, 1):
        return None
    return age, gender


def list_images(root: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp")
    out = []
    for dp, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                out.append(os.path.join(dp, f))
    return out


def age_to_bin(age: float) -> str:
    a = float(age)
    if a <= 12:
        return "0-12"
    if a <= 25:
        return "13-25"
    if a <= 40:
        return "26-40"
    if a <= 60:
        return "41-60"
    return "61+"


def bin_idx_from_age(age: float) -> int:
    return BIN_TO_IDX[age_to_bin(age)]


def compute_confusion_bins(y_true_age: np.ndarray, y_pred_age: np.ndarray) -> np.ndarray:
    cm = np.zeros((len(BIN_LABELS), len(BIN_LABELS)), dtype=np.int64)
    for t, p in zip(y_true_age, y_pred_age):
        ti = bin_idx_from_age(float(t))
        pi = bin_idx_from_age(float(p))
        cm[ti, pi] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, out_path: str, title: str):
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(BIN_LABELS))
    plt.xticks(tick_marks, BIN_LABELS, rotation=45, ha="right")
    plt.yticks(tick_marks, BIN_LABELS)
    plt.ylabel("True bin")
    plt.xlabel("Pred bin")

    thresh = cm.max() * 0.6 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            plt.text(j, i, str(v), ha="center", va="center",
                     color="white" if v > thresh else "black")

    plt.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_curves(history: List[Dict], out_dir: str):
    def series(key: str):
        return [h.get(key) for h in history]
    epochs = [h["epoch"] for h in history]

    fig = plt.figure()
    plt.plot(epochs, series("train_loss"), label="train_loss")
    plt.plot(epochs, series("val_loss"), label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.25)
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(epochs, series("train_mae"), label="train_mae")
    plt.plot(epochs, series("val_mae"), label="val_mae")
    plt.xlabel("epoch")
    plt.ylabel("MAE (years)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    fig.savefig(os.path.join(out_dir, "mae.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(epochs, series("train_bin_acc"), label="train_bin_acc")
    plt.plot(epochs, series("val_bin_acc"), label="val_bin_acc")
    plt.xlabel("epoch")
    plt.ylabel("bin_acc")
    plt.legend()
    plt.grid(True, alpha=0.25)
    fig.savefig(os.path.join(out_dir, "bin_acc.png"), dpi=160)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(epochs, series("train_gender_acc"), label="train_gender_acc")
    plt.plot(epochs, series("val_gender_acc"), label="val_gender_acc")
    plt.xlabel("epoch")
    plt.ylabel("gender_acc")
    plt.legend()
    plt.grid(True, alpha=0.25)
    fig.savefig(os.path.join(out_dir, "gender_acc.png"), dpi=160)
    plt.close(fig)


def save_topk(history: List[Dict], out_csv: str, k: int = 10):
    sorted_hist = sorted(
        history,
        key=lambda h: (h.get("val_mae", 1e9), -h.get("val_bin_acc", 0.0), h.get("val_loss", 1e9))
    )[:k]

    header = [
        "epoch",
        "val_mae", "val_bin_acc", "val_gender_acc", "val_loss",
        "train_mae", "train_bin_acc", "train_gender_acc", "train_loss"
    ]
    lines = [",".join(header)]
    for h in sorted_hist:
        lines.append(",".join(str(h.get(c, "")) for c in header))

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class UTKFaceCropsDataset(Dataset):
    def __init__(self, root: str, split: str, img_size: int, seed: int):
        paths = list_images(root)
        items = []
        for p in paths:
            parsed = parse_utk_filename(p)
            if parsed is None:
                continue
            age, gender = parsed
            items.append((p, age, gender))

        if len(items) == 0:
            raise RuntimeError(f"No valid UTKFace images found in: {root}")

        rng = np.random.RandomState(seed)
        rng.shuffle(items)

        n = len(items)
        n_train = int(n * 0.85)
        n_val = int(n * 0.10)
        if split == "train":
            self.items = items[:n_train]
        elif split == "val":
            self.items = items[n_train:n_train + n_val]
        elif split == "test":
            self.items = items[n_train + n_val:]
        else:
            raise ValueError("split must be train|val|test")

        if split == "train":
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
                transforms.RandomGrayscale(p=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, age, gender = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.tf(img)

        # ✅ возвращаем тензоры, чтобы не делать torch.tensor(...) в train/val
        age_t = torch.tensor(float(age), dtype=torch.float32)
        gender_t = torch.tensor(float(gender), dtype=torch.float32)
        return x, age_t, gender_t, os.path.basename(path)


class AgeGenderRegressionResNet18(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        in_f = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.drop = nn.Dropout(p=dropout)
        self.age_head = nn.Linear(in_f, 1)
        self.gender_head = nn.Linear(in_f, 1)

    def forward(self, x):
        f = self.backbone(x)
        f = self.drop(f)
        age = self.age_head(f).squeeze(1)
        gender_logit = self.gender_head(f).squeeze(1)
        return age, gender_logit


@dataclass
class TrainConfig:
    data_root: str
    out_root: str
    run_name: str
    img_size: int = 224
    batch_size: int = 64
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    lambda_gender: float = 0.35
    seed: int = 42
    num_workers: int = 4
    pretrained: bool = True
    dropout: float = 0.2
    amp: bool = True
    topk: int = 10
    age_clip: bool = True


def train_one_epoch(model, loader, optimizer, scaler, device, cfg: TrainConfig):
    model.train()
    total_loss = 0.0
    n = 0

    total_abs_err = 0.0
    bin_ok = 0
    gender_ok = 0

    for x, age, gender, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True)
        gender = gender.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # ✅ новый AMP API (убирает FutureWarning)
        with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
            pred_age, gender_logit = model(x)
            if cfg.age_clip:
                pred_age = torch.clamp(pred_age, AGE_MIN, AGE_MAX)

            loss_age = F.l1_loss(pred_age, age)
            loss_gender = F.binary_cross_entropy_with_logits(gender_logit, gender)
            loss = loss_age + cfg.lambda_gender * loss_gender

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            abs_err = torch.abs(pred_age - age).sum().item()
            total_abs_err += abs_err

            pred_bins = torch.tensor([bin_idx_from_age(a.item()) for a in pred_age], device=device)
            true_bins = torch.tensor([bin_idx_from_age(a.item()) for a in age], device=device)
            bin_ok += (pred_bins == true_bins).sum().item()

            pred_gender = (torch.sigmoid(gender_logit) > 0.5).long()
            gender_ok += (pred_gender == gender.long()).sum().item()

            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs

    return {
        "loss": total_loss / max(n, 1),
        "mae": total_abs_err / max(n, 1),
        "bin_acc": bin_ok / max(n, 1),
        "gender_acc": gender_ok / max(n, 1),
    }


@torch.no_grad()
def eval_one_epoch(model, loader, device, cfg: TrainConfig):
    model.eval()
    total_loss = 0.0
    n = 0

    total_abs_err = 0.0
    bin_ok = 0
    gender_ok = 0

    all_true_age = []
    all_pred_age = []

    for x, age, gender, _ in tqdm(loader, desc="val", leave=False):
        x = x.to(device, non_blocking=True)
        age = age.to(device, non_blocking=True)
        gender = gender.to(device, non_blocking=True)

        # eval без AMP
        with torch.amp.autocast("cuda", enabled=False):
            pred_age, gender_logit = model(x)
            if cfg.age_clip:
                pred_age = torch.clamp(pred_age, AGE_MIN, AGE_MAX)

            loss_age = F.l1_loss(pred_age, age)
            loss_gender = F.binary_cross_entropy_with_logits(gender_logit, gender)
            loss = loss_age + cfg.lambda_gender * loss_gender

        all_true_age.append(age.cpu().numpy())
        all_pred_age.append(pred_age.cpu().numpy())

        abs_err = torch.abs(pred_age - age).sum().item()
        total_abs_err += abs_err

        pred_bins = torch.tensor([bin_idx_from_age(a.item()) for a in pred_age], device=device)
        true_bins = torch.tensor([bin_idx_from_age(a.item()) for a in age], device=device)
        bin_ok += (pred_bins == true_bins).sum().item()

        pred_gender = (torch.sigmoid(gender_logit) > 0.5).long()
        gender_ok += (pred_gender == gender.long()).sum().item()

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

    y_true = np.concatenate(all_true_age) if all_true_age else np.array([])
    y_pred = np.concatenate(all_pred_age) if all_pred_age else np.array([])

    return (
        {
            "loss": total_loss / max(n, 1),
            "mae": total_abs_err / max(n, 1),
            "bin_acc": bin_ok / max(n, 1),
            "gender_acc": gender_ok / max(n, 1),
        },
        y_true,
        y_pred,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_root", default="runs_age_gender")
    ap.add_argument("--run_name", default=f"reg_resnet18_v2_{now_tag()}")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lambda_gender", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--no_age_clip", action="store_true")
    args = ap.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        out_root=args.out_root,
        run_name=args.run_name,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_gender=args.lambda_gender,
        seed=args.seed,
        num_workers=args.num_workers,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        amp=not args.no_amp,
        topk=args.topk,
        age_clip=not args.no_age_clip,
    )

    seed_everything(cfg.seed)

    run_dir = os.path.join(cfg.out_root, cfg.run_name)
    plots_dir = os.path.join(run_dir, "plots")
    ensure_dir(run_dir)
    ensure_dir(plots_dir)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    ds_train = UTKFaceCropsDataset(cfg.data_root, "train", cfg.img_size, cfg.seed)
    ds_val = UTKFaceCropsDataset(cfg.data_root, "val", cfg.img_size, cfg.seed)

    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"), drop_last=False
    )
    dl_val = DataLoader(
        ds_val, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"), drop_last=False
    )

    model = AgeGenderRegressionResNet18(pretrained=cfg.pretrained, dropout=cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ✅ новый GradScaler API
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    history: List[Dict] = []
    best_mae = 1e9
    best_path = None

    csv_path = os.path.join(run_dir, "metrics.csv")
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,train_mae,val_mae,train_bin_acc,val_bin_acc,train_gender_acc,val_gender_acc\n")

    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, dl_train, optimizer, scaler, device, cfg)
        va, y_true, y_pred = eval_one_epoch(model, dl_val, device, cfg)

        row = {
            "epoch": epoch,
            "train_loss": float(tr["loss"]),
            "val_loss": float(va["loss"]),
            "train_mae": float(tr["mae"]),
            "val_mae": float(va["mae"]),
            "train_bin_acc": float(tr["bin_acc"]),
            "val_bin_acc": float(va["bin_acc"]),
            "train_gender_acc": float(tr["gender_acc"]),
            "val_gender_acc": float(va["gender_acc"]),
        }
        history.append(row)

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f'{epoch},{row["train_loss"]:.6f},{row["val_loss"]:.6f},{row["train_mae"]:.4f},{row["val_mae"]:.4f},'
                f'{row["train_bin_acc"]:.4f},{row["val_bin_acc"]:.4f},{row["train_gender_acc"]:.4f},{row["val_gender_acc"]:.4f}\n'
            )
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"[epoch {epoch:02d}] "
            f"val_mae={row['val_mae']:.3f}  val_bin_acc={row['val_bin_acc']:.3f}  val_gender_acc={row['val_gender_acc']:.3f}  val_loss={row['val_loss']:.4f}"
        )

        if y_true.size and y_pred.size:
            cm = compute_confusion_bins(y_true, y_pred)
            plot_confusion_matrix(cm, os.path.join(plots_dir, f"confusion_bins_e{epoch:02d}.png"),
                                  title=f"Age bins confusion (epoch {epoch})")

        if row["val_mae"] < best_mae:
            best_mae = row["val_mae"]
            best_path = os.path.join(run_dir, "age_gender_regression_resnet18_v2_best.pth")
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "best_mae": best_mae, "config": asdict(cfg)},
                best_path,
            )

        if epoch % 10 == 0 or epoch == cfg.epochs:
            ckpt_path = os.path.join(run_dir, f"age_gender_regression_resnet18_v2_e{epoch:02d}.pth")
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": asdict(cfg)}, ckpt_path)

        plot_curves(history, plots_dir)

    save_topk(history, os.path.join(run_dir, "topk_epochs.csv"), k=cfg.topk)

    summary = {"best_mae": best_mae, "best_checkpoint": best_path, "run_dir": run_dir}
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Run dir: {run_dir}")
    if best_path:
        print(f"Best: {best_path} (val_mae={best_mae:.3f})")


if __name__ == "__main__":
    main()
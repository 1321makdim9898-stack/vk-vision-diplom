# train_emotion_ferplus_resnet50_v5.py
import os
import json
import time
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm

# Без GUI (Windows-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, CosineAnnealingWarmRestarts

# optional sklearn
try:
    from sklearn.metrics import confusion_matrix, f1_score, classification_report
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# 7 классов как в приложении
APP_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
APP_TO_IDX = {c: i for i, c in enumerate(APP_LABELS)}

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_images(folder: str) -> List[str]:
    out = []
    for dp, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(dp, f))
    return out


def plot_curves(history: List[Dict], out_dir: str):
    epochs = [h["epoch"] for h in history]

    def s(k):
        return [h.get(k) for h in history]

    # loss
    fig = plt.figure()
    plt.plot(epochs, s("train_loss"), label="train_loss")
    plt.plot(epochs, s("val_loss"), label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.25)
    if any(v is not None for v in s("train_loss")):
        plt.legend()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # acc
    fig = plt.figure()
    plt.plot(epochs, s("train_acc"), label="train_acc")
    plt.plot(epochs, s("val_acc"), label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(True, alpha=0.25)
    if any(v is not None for v in s("train_acc")):
        plt.legend()
    fig.savefig(os.path.join(out_dir, "acc.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # f1 macro
    if any(v is not None for v in s("val_f1_macro")):
        fig = plt.figure()
        plt.plot(epochs, s("train_f1_macro"), label="train_f1_macro")
        plt.plot(epochs, s("val_f1_macro"), label="val_f1_macro")
        plt.xlabel("epoch")
        plt.ylabel("F1 macro")
        plt.grid(True, alpha=0.25)
        plt.legend()
        fig.savefig(os.path.join(out_dir, "f1_macro.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)

    # lr
    if any(v is not None for v in s("lr")):
        fig = plt.figure()
        plt.plot(epochs, s("lr"), label="lr")
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.grid(True, alpha=0.25)
        plt.legend()
        fig.savefig(os.path.join(out_dir, "lr.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)

    # mix_prob
    if any(v is not None for v in s("mix_prob")):
        fig = plt.figure()
        plt.plot(epochs, s("mix_prob"), label="mix_prob")
        plt.xlabel("epoch")
        plt.ylabel("mix prob")
        plt.grid(True, alpha=0.25)
        plt.legend()
        fig.savefig(os.path.join(out_dir, "mix_prob.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)


def save_topk(history: List[Dict], out_csv: str, k: int = 10, key_name: str = "val_acc"):
    # Top-k by metric desc, then loss asc, then f1 desc
    def key(h):
        return (-h.get(key_name, 0.0), h.get("val_loss", 1e9), -(h.get("val_f1_macro") or 0.0))

    best = sorted(history, key=key)[:k]
    header = [
        "epoch",
        "val_acc", "val_f1_macro", "val_loss",
        "test_acc", "test_f1_macro", "test_loss",
        "train_acc", "train_f1_macro", "train_loss",
        "lr", "mix_prob"
    ]
    lines = [",".join(header)]
    for h in best:
        lines.append(",".join(str(h.get(c, "")) for c in header))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_confusion(out_dir: str, prefix: str, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], title: str):
    if not SKLEARN_OK:
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    # csv
    csv_path = os.path.join(out_dir, f"{prefix}_confusion_matrix.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, row in enumerate(cm):
            f.write(labels[i] + "," + ",".join(map(str, row.tolist())) + "\n")

    # png
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    tick = np.arange(len(labels))
    plt.xticks(tick, labels, rotation=45, ha="right")
    plt.yticks(tick, labels)
    plt.ylabel("True")
    plt.xlabel("Pred")

    thresh = cm.max() * 0.6 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            plt.text(j, i, str(v), ha="center", va="center", color="white" if v > thresh else "black")

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{prefix}_confusion_matrix.png"), dpi=160)
    plt.close(fig)


def write_report_md(run_dir: str, cfg: Dict, best: Dict, class_counts: Dict[str, Dict[str, int]], notes: List[str]):
    lines = []
    lines.append("# Emotion (FERPlus→7) v5 — отчёт обучения\n")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Time: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n")

    if notes:
        lines.append("## Примечания\n")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")

    lines.append("## Конфигурация\n")
    lines.append("```json")
    lines.append(json.dumps(cfg, ensure_ascii=False, indent=2))
    lines.append("```\n")

    lines.append("## Дисбаланс классов (после маппинга в 7 классов)\n")
    lines.append("```json")
    lines.append(json.dumps(class_counts, ensure_ascii=False, indent=2))
    lines.append("```\n")

    lines.append("## Лучший результат\n")
    lines.append("```json")
    lines.append(json.dumps(best, ensure_ascii=False, indent=2))
    lines.append("```\n")

    lines.append("## Артефакты\n")
    lines.append("- `metrics.csv`, `metrics.jsonl` — метрики по эпохам")
    lines.append("- `plots/loss.png`, `plots/acc.png`, `plots/f1_macro.png`, `plots/lr.png`, `plots/mix_prob.png` — графики")
    lines.append("- `val_confusion_matrix.(png|csv)` — confusion matrix (validation)")
    lines.append("- `test_confusion_matrix.(png|csv)` — confusion matrix (test)")
    lines.append("- `val_classification_report.txt`, `test_classification_report.txt` — classification report")
    lines.append("- `topk_by_val.csv`, `topk_by_test.csv` — таблицы лучших эпох\n")

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if not SKLEARN_OK:
        return None
    return float(f1_score(y_true, y_pred, average="macro"))


def count_by_class_items(items: List[Tuple[str, int]]) -> Dict[str, int]:
    cnt = {c: 0 for c in APP_LABELS}
    for _, y in items:
        cnt[APP_LABELS[int(y)]] += 1
    return cnt


def compute_class_weights(items: List[Tuple[str, int]]) -> torch.Tensor:
    # weights ~ 1/freq, normalized to mean=1
    cnt = np.ones(len(APP_LABELS), dtype=np.float32)
    for _, y in items:
        cnt[int(y)] += 1.0
    inv = 1.0 / cnt
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


class FERPlusMappedDataset(Dataset):
    """
    Структура:
      root/train/<class>/*
      root/validation/<class>/*
      root/test/<class>/*
    Классы FERPlus: angry, contempt, disgust, fear, happy, neutral, sad, suprise/surprise.

    Маппинг в 7 классов:
      contempt -> (disgust|neutral|drop)
      suprise -> surprise
    """
    def __init__(self, root: str, split: str, transform, contempt_to: str = "disgust"):
        assert split in ("train", "validation", "test")
        assert contempt_to in ("disgust", "neutral", "drop")

        self.root = root
        self.split = split
        self.transform = transform
        self.contempt_to = contempt_to

        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"split dir not found: {split_dir}")

        items: List[Tuple[str, int]] = []

        for class_name in os.listdir(split_dir):
            src_class = class_name.lower().strip()
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            if src_class == "suprise":
                src_class = "surprise"

            if src_class in APP_TO_IDX:
                tgt = src_class
            elif src_class == "contempt":
                if contempt_to == "drop":
                    tgt = None
                else:
                    tgt = contempt_to
            else:
                tgt = None

            if tgt is None:
                continue

            y = APP_TO_IDX[tgt]
            for p in list_images(class_dir):
                items.append((p, y))

        if len(items) == 0:
            raise RuntimeError(f"No images collected for {split_dir}. Check folders and extensions.")

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y


def build_transforms(img_size: int, use_randaugment: bool = False, ra_n: int = 2, ra_m: int = 9):
    # FERPlus grayscale -> 3ch (ResNet)
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.10, contrast=0.10)], p=0.25),
        transforms.RandomRotation(degrees=10),
        transforms.RandAugment(num_ops=ra_n, magnitude=ra_m) if use_randaugment else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.3, 3.3), value=0),
    ])

    val_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model_resnet50(num_classes: int, dropout: float):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_f, num_classes)
    )
    return m


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup = {}


def onehot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(y, num_classes=num_classes).float()


def rand_bbox(W: int, H: int, lam: float):
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def mixup_cutmix(x, y, num_classes: int, mixup_alpha: float, cutmix_alpha: float):
    r = np.random.rand()
    if r < 0.5 and mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        perm = torch.randperm(x.size(0), device=x.device)
        x2 = x[perm]
        y_a = onehot(y, num_classes)
        y_b = onehot(y[perm], num_classes)
        x = lam * x + (1.0 - lam) * x2
        y_mix = lam * y_a + (1.0 - lam) * y_b
        return x, y_mix
    if cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        perm = torch.randperm(x.size(0), device=x.device)
        bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(3), x.size(2), lam)
        x2 = x[perm]
        x[:, :, bby1:bby2, bbx1:bbx2] = x2[:, :, bby1:bby2, bbx1:bbx2]
        lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        y_a = onehot(y, num_classes)
        y_b = onehot(y[perm], num_classes)
        y_mix = lam * y_a + (1.0 - lam) * y_b
        return x, y_mix
    return x, onehot(y, num_classes)


def loss_from_soft_targets(logits: torch.Tensor, y_soft: torch.Tensor, label_smoothing: float = 0.0) -> torch.Tensor:
    # soft CE
    if label_smoothing > 0:
        n = y_soft.size(1)
        y_soft = (1 - label_smoothing) * y_soft + label_smoothing / n
    logp = F.log_softmax(logits, dim=1)
    return -(y_soft * logp).sum(dim=1).mean()


@dataclass
class TrainConfig:
    data_root: str
    out_root: str
    run_name: str
    epochs: int = 100
    batch_size: int = 128
    img_size: int = 224
    lr: float = 3e-4
    lr_head_mult: float = 3.0
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"  # auto|cuda|cpu
    amp: bool = True
    label_smoothing: float = 0.05
    dropout: float = 0.2
    use_class_weights: bool = True
    scheduler: str = "cosine"  # cosine|cosine_restart|step|none|onecycle
    warmup_epochs: int = 2
    patience: int = 0
    topk: int = 10
    contempt_to: str = "disgust"  # disgust|neutral|drop

    freeze_epochs: int = 5
    freeze_bn: bool = True

    mix_prob: float = 0.6
    mix_prob_end: float = 0.0  # target end prob for mix decay
    mix_decay_start: int = 1     # epoch to start decaying mix_prob
    mix_decay_end: int = 0       # epoch to end decaying (0=epochs)
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    mix_decay: str = "cosine"  # none|linear|cosine

    ema_decay: float = 0.999
    no_ema_eval: bool = False

    balanced_sampler: bool = False
    use_randaugment: bool = False
    ra_n: int = 2
    ra_m: int = 9

    save_every: int = 0
    resume: str = ""
    no_best_by_test: bool = False
    eval_only: bool = False
    ckpt: str = ""


def freeze_backbone(model: nn.Module):
    for name, p in model.named_parameters():
        if name.startswith("fc."):
            p.requires_grad = True
        else:
            p.requires_grad = False


def unfreeze_all(model: nn.Module):
    for _, p in model.named_parameters():
        p.requires_grad = True


def set_bn_eval(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            m.eval()


@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device, criterion_ce, label_smoothing: float,
             use_amp: bool, use_soft_targets: bool = False):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    y_true_all = []
    y_pred_all = []

    for x, y in tqdm(dl, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion_ce(logits, y)

        bs = x.size(0)
        loss_sum += loss.item() * bs
        total += bs

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()

        if SKLEARN_OK:
            y_true_all.append(y.detach().cpu().numpy())
            y_pred_all.append(pred.detach().cpu().numpy())

    acc = correct / max(1, total)
    loss = loss_sum / max(1, total)

    f1 = None
    if SKLEARN_OK and y_true_all:
        f1 = float(f1_score(np.concatenate(y_true_all), np.concatenate(y_pred_all), average="macro"))

    y_true_np = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred_np = np.concatenate(y_pred_all) if y_pred_all else np.array([])
    return loss, acc, f1, y_true_np, y_pred_np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help=r"C:\vk-vision-demo\data\FERPlus")
    ap.add_argument("--out_root", default=r"C:\vk-vision-demo\results\runs_emotion", help="куда сохранять run")
    ap.add_argument("--run_name", default=f"emotion_ferplus_resnet50_v5_{now_tag()}")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_head_mult", type=float, default=3.0)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--balanced_sampler", action="store_true", help="use WeightedRandomSampler to balance classes")
    ap.add_argument("--use_randaugment", action="store_true", help="enable torchvision RandAugment")
    ap.add_argument("--ra_n", type=int, default=2, help="RandAugment N")
    ap.add_argument("--ra_m", type=int, default=9, help="RandAugment M")

    ap.add_argument("--scheduler", default="cosine",
                    choices=["cosine", "cosine_restart", "step", "none", "onecycle"])
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--contempt_to", default="disgust", choices=["disgust", "neutral", "drop"])

    ap.add_argument("--freeze_epochs", type=int, default=5)
    ap.add_argument("--freeze_bn", action="store_true")
    ap.add_argument("--no_freeze_bn", action="store_true")

    ap.add_argument("--mix_prob", type=float, default=0.6)
    ap.add_argument("--mix_prob_end", type=float, default=0.0, help="final mix prob after decay")
    ap.add_argument("--mix_decay_start", type=int, default=1, help="epoch to start mix prob decay")
    ap.add_argument("--mix_decay_end", type=int, default=0, help="epoch to end mix prob decay (0=epochs)")
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--cutmix_alpha", type=float, default=1.0)
    ap.add_argument("--mix_decay", default="cosine", choices=["none", "linear", "cosine"])
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--no_ema_eval", action="store_true")

    ap.add_argument("--save_every", type=int, default=0)
    ap.add_argument("--resume", default="", help="path to checkpoint to resume (last.pth)")
    ap.add_argument("--no_best_by_test", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", default="", help="checkpoint for eval_only")
    args = ap.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        out_root=args.out_root,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
        lr_head_mult=args.lr_head_mult,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        amp=not args.no_amp,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        use_class_weights=not args.no_class_weights,
        balanced_sampler=args.balanced_sampler,
        use_randaugment=args.use_randaugment,
        ra_n=args.ra_n,
        ra_m=args.ra_m,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        topk=args.topk,
        contempt_to=args.contempt_to,
        freeze_epochs=args.freeze_epochs,
        freeze_bn=(True if args.freeze_bn else False),
        mix_prob=args.mix_prob,
        mix_prob_end=args.mix_prob_end,
        mix_decay_start=args.mix_decay_start,
        mix_decay_end=args.mix_decay_end,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_decay=args.mix_decay,
        ema_decay=args.ema_decay,
        no_ema_eval=args.no_ema_eval,
        resume=args.resume,
        no_best_by_test=args.no_best_by_test,
        eval_only=args.eval_only,
        ckpt=args.ckpt,
    )
    if args.no_freeze_bn:
        cfg.freeze_bn = False

    seed_everything(cfg.seed)

    # device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    use_amp = (cfg.amp and device.type == "cuda")

    run_dir = os.path.join(cfg.out_root, cfg.run_name)
    plots_dir = os.path.join(run_dir, "plots")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ensure_dir(run_dir)
    ensure_dir(plots_dir)
    ensure_dir(ckpt_dir)

    # transforms + datasets
    train_tf, val_tf = build_transforms(cfg.img_size, cfg.use_randaugment, cfg.ra_n, cfg.ra_m)
    ds_train = FERPlusMappedDataset(cfg.data_root, "train", transform=train_tf, contempt_to=cfg.contempt_to)
    ds_val = FERPlusMappedDataset(cfg.data_root, "validation", transform=val_tf, contempt_to=cfg.contempt_to)
    ds_test = FERPlusMappedDataset(cfg.data_root, "test", transform=val_tf, contempt_to=cfg.contempt_to)

    # optional balanced sampler
    train_sampler = None
    if cfg.balanced_sampler:
        # weight per sample = 1 / count(class)
        cls_counts = np.zeros(len(APP_LABELS), dtype=np.int64)
        for _, y in ds_train.items:
            cls_counts[int(y)] += 1
        cls_counts = np.maximum(cls_counts, 1)
        cls_weights = 1.0 / cls_counts
        sample_weights = [float(cls_weights[int(y)]) for _, y in ds_train.items]
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    dl_train = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=True
    )
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    # model
    model = build_model_resnet50(num_classes=len(APP_LABELS), dropout=cfg.dropout).to(device)

    # freeze backbone initially
    if cfg.freeze_epochs > 0:
        freeze_backbone(model)
        if cfg.freeze_bn:
            set_bn_eval(model)

    # loss
    if cfg.use_class_weights:
        w = compute_class_weights(ds_train.items).to(device)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # optimizer with head lr multiplier
    head_params = []
    base_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(p)
        else:
            base_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": cfg.lr},
            {"params": head_params, "lr": cfg.lr * cfg.lr_head_mult},
        ],
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # scheduler
    scheduler = None
    if cfg.scheduler == "onecycle":
        # steps per epoch = len(dl_train)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[cfg.lr, cfg.lr * cfg.lr_head_mult],
            epochs=cfg.epochs,
            steps_per_epoch=len(dl_train),
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )
    elif cfg.scheduler == "cosine_restart":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=cfg.lr * 0.02)
    elif cfg.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.02)
    elif cfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(cfg.epochs * 0.5), int(cfg.epochs * 0.8)],
            gamma=0.1,
        )

    def get_cur_lr() -> float:
        # показываем lr базовой группы
        return float(optimizer.param_groups[0]["lr"])

    def mix_prob_at_epoch(e: int) -> float:
        """Schedule for mixup/cutmix probability."""
        if cfg.mix_prob <= 0 or cfg.mix_decay == "none":
            return float(cfg.mix_prob)

        start = max(1, int(cfg.mix_decay_start))
        end = int(cfg.mix_decay_end) if int(cfg.mix_decay_end) > 0 else int(cfg.epochs)
        end = max(start, end)

        if e <= start:
            return float(cfg.mix_prob)
        if e >= end:
            return float(cfg.mix_prob_end)

        t = (e - start) / max(1, end - start)
        if cfg.mix_decay == "linear":
            s = 1.0 - t
        else:
            # cosine
            s = 0.5 * (1.0 + math.cos(math.pi * t))
        return float(cfg.mix_prob_end + (cfg.mix_prob - cfg.mix_prob_end) * s)

    # where to save weights
    best_val_path = os.path.join(os.getcwd(), "models", "emotion_resnet50_ferplus_v5_best_val.pth")
    best_test_path = os.path.join(os.getcwd(), "models", "emotion_resnet50_ferplus_v5_best_test.pth")
    ensure_dir(os.path.dirname(best_val_path))

    best_val_acc = -1.0
    best_test_acc = -1.0
    best_val_epoch = -1
    best_test_epoch = -1

    # save config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(run_dir, "metrics.csv")
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc,train_f1_macro,val_f1_macro,"
                "test_loss,test_acc,test_f1_macro,lr,mix_prob\n")

    class_counts = {
        "train": count_by_class_items(ds_train.items),
        "val": count_by_class_items(ds_val.items),
        "test": count_by_class_items(ds_test.items),
    }

    notes = [
        f"FERPlus(8) → APP(7): contempt→{cfg.contempt_to}, suprise→surprise",
        f"Model: ResNet50 ImageNet-pretrained + dropout={cfg.dropout}",
        f"RandAugment: {cfg.use_randaugment} (N={cfg.ra_n}, M={cfg.ra_m})",
        f"BalancedSampler: {cfg.balanced_sampler}",
        f"Mix: prob={cfg.mix_prob}→{cfg.mix_prob_end} decay={cfg.mix_decay} start={cfg.mix_decay_start} end={cfg.mix_decay_end or cfg.epochs}",
        f"EMA: decay={cfg.ema_decay} (eval={not cfg.no_ema_eval})",
        f"Scheduler: {cfg.scheduler}",
    ]

    print(f"[INFO] device: {device} | amp: {use_amp}")
    print(f"[INFO] data_root: {cfg.data_root}")
    print(f"[INFO] run_dir  : {run_dir}")
    print(f"[INFO] contempt_to: {cfg.contempt_to}")
    print(f"[INFO] counts(train/val/test): {class_counts}")

    # EMA
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

    # resume
    start_epoch = 1
    last_path = os.path.join(ckpt_dir, "last.pth")
    if cfg.resume:
        last_path = cfg.resume

    if os.path.isfile(last_path) and not cfg.eval_only:
        payload = torch.load(last_path, map_location="cpu")
        if "model" in payload:
            model.load_state_dict(payload["model"], strict=True)
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        if "scaler" in payload and scaler is not None:
            try:
                scaler.load_state_dict(payload["scaler"])
            except Exception:
                pass
        if "epoch" in payload:
            start_epoch = int(payload["epoch"]) + 1
        if "best_val_acc" in payload:
            best_val_acc = float(payload["best_val_acc"])
            best_val_epoch = int(payload.get("best_val_epoch", -1))
        if "best_test_acc" in payload:
            best_test_acc = float(payload["best_test_acc"])
            best_test_epoch = int(payload.get("best_test_epoch", -1))
        print(f"[INFO] resume from {last_path} (next epoch {start_epoch})")

    # eval only
    if cfg.eval_only:
        assert cfg.ckpt and os.path.isfile(cfg.ckpt), "--eval_only requires --ckpt <path>"
        payload = torch.load(cfg.ckpt, map_location="cpu")
        if "model" in payload:
            model.load_state_dict(payload["model"], strict=True)
        model.to(device)

        # evaluate val/test
        val_loss, val_acc, val_f1, yv_t, yv_p = evaluate(model, dl_val, device, criterion, cfg.label_smoothing, use_amp)
        test_loss, test_acc, test_f1, yt_t, yt_p = evaluate(model, dl_test, device, criterion, cfg.label_smoothing, use_amp)

        out_eval = {
            "validation": {"loss": val_loss, "acc": val_acc, "f1_macro": val_f1, "n": len(ds_val)},
            "test": {"loss": test_loss, "acc": test_acc, "f1_macro": test_f1, "n": len(ds_test)},
            "ckpt": cfg.ckpt,
            "device": str(device),
            "contempt_to": cfg.contempt_to,
            "img_size": cfg.img_size,
        }
        with open(os.path.join(run_dir, "eval_only.json"), "w", encoding="utf-8") as f:
            json.dump(out_eval, f, ensure_ascii=False, indent=2)

        if SKLEARN_OK and yv_t.size:
            save_confusion(run_dir, "val", yv_t, yv_p, APP_LABELS, title="Val confusion (eval_only)")
            rep = classification_report(yv_t, yv_p, target_names=APP_LABELS, digits=4)
            with open(os.path.join(run_dir, "val_classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(rep)

        if SKLEARN_OK and yt_t.size:
            save_confusion(run_dir, "test", yt_t, yt_p, APP_LABELS, title="Test confusion (eval_only)")
            rep = classification_report(yt_t, yt_p, target_names=APP_LABELS, digits=4)
            with open(os.path.join(run_dir, "test_classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(rep)

        print("\nEVAL ONLY DONE")
        print(json.dumps(out_eval, ensure_ascii=False, indent=2))
        return

    history: List[Dict] = []
    bad_epochs = 0

    for epoch in range(start_epoch, cfg.epochs + 1):
        # unfreeze after freeze_epochs
        if cfg.freeze_epochs > 0 and epoch == cfg.freeze_epochs + 1:
            unfreeze_all(model)
            if cfg.freeze_bn:
                print(f"[INFO] unfreeze backbone at epoch {epoch} (BN stays eval)")
            else:
                print(f"[INFO] unfreeze backbone at epoch {epoch}")

            # rebuild optimizer param groups with correct requires_grad
            head_params = []
            base_params = []
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name.startswith("fc."):
                    head_params.append(p)
                else:
                    base_params.append(p)

            optimizer = torch.optim.AdamW(
                [
                    {"params": base_params, "lr": cfg.lr},
                    {"params": head_params, "lr": cfg.lr * cfg.lr_head_mult},
                ],
                weight_decay=cfg.weight_decay,
            )
            # re-create scheduler after re-creating optimizer
            if cfg.scheduler == "onecycle":
                scheduler = OneCycleLR(
                    optimizer,
                    max_lr=[cfg.lr, cfg.lr * cfg.lr_head_mult],
                    epochs=cfg.epochs,
                    steps_per_epoch=len(dl_train),
                    pct_start=0.1,
                    anneal_strategy="cos",
                    div_factor=10.0,
                    final_div_factor=100.0,
                )
            elif cfg.scheduler == "cosine_restart":
                scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=cfg.lr * 0.02)
            elif cfg.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.02)
            elif cfg.scheduler == "step":
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=[int(cfg.epochs * 0.5), int(cfg.epochs * 0.8)],
                    gamma=0.1,
                )

            # reset scaler (safe)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            # refresh ema baseline
            if ema is not None:
                ema = EMA(model, decay=cfg.ema_decay)

        if cfg.freeze_bn:
            set_bn_eval(model)

        cur_mix_prob = mix_prob_at_epoch(epoch)

        # ---- train ----
        model.train()
        tr_loss_sum = 0.0
        tr_correct = 0
        tr_total = 0
        tr_y_true = []
        tr_y_pred = []

        for x, y in tqdm(dl_train, desc=f"train e{epoch:03d}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            use_mix = (cur_mix_prob > 0) and (np.random.rand() < cur_mix_prob)
            if use_mix:
                x_mix, y_soft = mixup_cutmix(x, y, len(APP_LABELS), cfg.mixup_alpha, cfg.cutmix_alpha)
            else:
                x_mix, y_soft = x, None

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x_mix)
                if y_soft is not None:
                    loss = loss_from_soft_targets(logits, y_soft, label_smoothing=cfg.label_smoothing)
                else:
                    loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None and cfg.scheduler == "onecycle":
                scheduler.step()

            if ema is not None:
                ema.update(model)

            bs = x.size(0)
            tr_loss_sum += loss.item() * bs
            tr_total += bs

            pred = logits.argmax(dim=1)
            tr_correct += (pred == y).sum().item()

            if SKLEARN_OK:
                tr_y_true.append(y.detach().cpu().numpy())
                tr_y_pred.append(pred.detach().cpu().numpy())

        train_loss = tr_loss_sum / max(1, tr_total)
        train_acc = tr_correct / max(1, tr_total)
        train_f1 = None
        if SKLEARN_OK and tr_y_true:
            train_f1 = float(f1_score(np.concatenate(tr_y_true), np.concatenate(tr_y_pred), average="macro"))

        # epoch-level scheduler step for others
        if scheduler is not None and cfg.scheduler in ("cosine", "step"):
            scheduler.step()
        elif scheduler is not None and cfg.scheduler == "cosine_restart":
            # WarmRestarts expects fractional epoch to be smooth (optional). We'll step per epoch.
            scheduler.step(epoch)

        # ---- eval ----
        def eval_with(model_to_eval: nn.Module):
            val_loss, val_acc, val_f1, yv_t, yv_p = evaluate(model_to_eval, dl_val, device, criterion, cfg.label_smoothing, use_amp)
            test_loss, test_acc, test_f1, yt_t, yt_p = evaluate(model_to_eval, dl_test, device, criterion, cfg.label_smoothing, use_amp)
            return (val_loss, val_acc, val_f1, yv_t, yv_p), (test_loss, test_acc, test_f1, yt_t, yt_p)

        if ema is not None and (not cfg.no_ema_eval):
            ema.apply_shadow(model)
            (val_loss, val_acc, val_f1, yv_t, yv_p), (test_loss, test_acc, test_f1, yt_t, yt_p) = eval_with(model)
            ema.restore(model)
        else:
            (val_loss, val_acc, val_f1, yv_t, yv_p), (test_loss, test_acc, test_f1, yt_t, yt_p) = eval_with(model)

        cur_lr = get_cur_lr()

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "train_f1_macro": float(train_f1) if train_f1 is not None else None,
            "val_f1_macro": float(val_f1) if val_f1 is not None else None,
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
            "test_f1_macro": float(test_f1) if test_f1 is not None else None,
            "lr": float(cur_lr),
            "mix_prob": float(cur_mix_prob),
        }
        history.append(row)

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_acc:.6f},{val_acc:.6f},"
                f"{(train_f1 if train_f1 is not None else '')},{(val_f1 if val_f1 is not None else '')},"
                f"{test_loss:.6f},{test_acc:.6f},{(test_f1 if test_f1 is not None else '')},"
                f"{cur_lr:.8f},{cur_mix_prob:.4f}\n"
            )
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        f1s = f" val_f1={val_f1:.4f}" if (val_f1 is not None and not math.isnan(val_f1)) else ""
        print(
            f"[epoch {epoch:03d}] val_acc={val_acc:.4f} val_loss={val_loss:.4f}{f1s} | "
            f"test_acc={test_acc:.4f} test_loss={test_loss:.4f} | lr={cur_lr:.2e} mix_p={cur_mix_prob:.2f}"
        )

        plot_curves(history, plots_dir)

        # save last
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "best_val_epoch": best_val_epoch,
                "best_test_acc": best_test_acc,
                "best_test_epoch": best_test_epoch,
                "config": asdict(cfg),
            },
            os.path.join(ckpt_dir, "last.pth"),
        )

        # best by val
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            best_val_epoch = epoch
            payload = {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "labels": APP_LABELS,
                "config": asdict(cfg),
            }
            torch.save(payload, os.path.join(run_dir, "emotion_resnet50_ferplus_v5_best_val.pth"))
            torch.save(payload, best_val_path)

            if SKLEARN_OK and yv_t.size:
                save_confusion(run_dir, "val", yv_t, yv_p, APP_LABELS, title=f"Val confusion (best@{epoch})")
                rep = classification_report(yv_t, yv_p, target_names=APP_LABELS, digits=4)
                with open(os.path.join(run_dir, "val_classification_report.txt"), "w", encoding="utf-8") as f:
                    f.write(rep)

            print(f"[INFO] best_by_val saved -> {best_val_path} (val_acc={best_val_acc:.4f})")
            bad_epochs = 0
        else:
            bad_epochs += 1

        # best by test (optional)
        if (not cfg.no_best_by_test) and (test_acc > best_test_acc + 1e-6):
            best_test_acc = test_acc
            best_test_epoch = epoch
            payload = {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_test_acc": best_test_acc,
                "labels": APP_LABELS,
                "config": asdict(cfg),
            }
            torch.save(payload, os.path.join(run_dir, "emotion_resnet50_ferplus_v5_best_test.pth"))
            torch.save(payload, best_test_path)

            if SKLEARN_OK and yt_t.size:
                save_confusion(run_dir, "test", yt_t, yt_p, APP_LABELS, title=f"Test confusion (best@{epoch})")
                rep = classification_report(yt_t, yt_p, target_names=APP_LABELS, digits=4)
                with open(os.path.join(run_dir, "test_classification_report.txt"), "w", encoding="utf-8") as f:
                    f.write(rep)

            print(f"[INFO] best_by_test saved -> {best_test_path} (test_acc={best_test_acc:.4f})")

        # early stopping by val (if enabled)
        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"[INFO] early stop: no improvement in val_acc for {cfg.patience} epochs")
            break

        # periodic checkpoints
        if cfg.save_every > 0 and (epoch % cfg.save_every == 0):
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": asdict(cfg)},
                       os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))

    # tables
    save_topk(history, os.path.join(run_dir, "topk_by_val.csv"), k=cfg.topk, key_name="val_acc")
    save_topk(history, os.path.join(run_dir, "topk_by_test.csv"), k=cfg.topk, key_name="test_acc")

    best_summary = {
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "best_val_path": best_val_path,
        "best_test_acc": best_test_acc,
        "best_test_epoch": best_test_epoch,
        "best_test_path": best_test_path,
        "run_dir": run_dir,
        "labels": APP_LABELS,
        "contempt_to": cfg.contempt_to,
    }
    write_report_md(run_dir, asdict(cfg), best_summary, class_counts, notes)

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Run dir: {run_dir}")
    print(f"Best_by_val : {best_val_path} (val_acc={best_val_acc:.4f} @ epoch {best_val_epoch})")
    print(f"Best_by_test: {best_test_path} (test_acc={best_test_acc:.4f} @ epoch {best_test_epoch})")
    print(f"Resume from : {os.path.join(ckpt_dir, 'last.pth')}")


if __name__ == "__main__":
    main()
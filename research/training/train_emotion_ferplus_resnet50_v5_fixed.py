# train_emotion_ferplus_resnet50_v5_fixed.py
import os
import json
import time
import math
import argparse
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm

# Без GUI (Windows-safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def count_by_class_items(items: List[Tuple[str, int]]) -> Dict[str, int]:
    cnt = {c: 0 for c in APP_LABELS}
    for _, y in items:
        cnt[APP_LABELS[int(y)]] += 1
    return cnt


def compute_class_weights(items: List[Tuple[str, int]]) -> torch.Tensor:
    # weights ~ 1/freq, normalized to mean=1
    cnt = np.ones((len(APP_LABELS),), dtype=np.float32)
    for _, y in items:
        cnt[int(y)] += 1.0
    inv = 1.0 / cnt
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


def plot_curves(history: List[Dict], out_dir: str):
    epochs = [h["epoch"] for h in history]

    def s(k):
        return [h.get(k) for h in history]

    # loss
    fig = plt.figure()
    plt.plot(epochs, s("train_loss"), label="train_loss")
    plt.plot(epochs, s("val_loss"), label="val_loss")
    if any(v is not None for v in s("test_loss")):
        plt.plot(epochs, s("test_loss"), label="test_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    fig.savefig(os.path.join(out_dir, "loss.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # acc
    fig = plt.figure()
    plt.plot(epochs, s("train_acc"), label="train_acc")
    plt.plot(epochs, s("val_acc"), label="val_acc")
    if any(v is not None for v in s("test_acc")):
        plt.plot(epochs, s("test_acc"), label="test_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    fig.savefig(os.path.join(out_dir, "acc.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # f1 macro
    if any(v is not None for v in s("val_f1_macro")):
        fig = plt.figure()
        plt.plot(epochs, s("train_f1_macro"), label="train_f1_macro")
        plt.plot(epochs, s("val_f1_macro"), label="val_f1_macro")
        if any(v is not None for v in s("test_f1_macro")):
            plt.plot(epochs, s("test_f1_macro"), label="test_f1_macro")
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

    # mix_p
    if any(v is not None for v in s("mix_p")):
        fig = plt.figure()
        plt.plot(epochs, s("mix_p"), label="mix_p")
        plt.xlabel("epoch")
        plt.ylabel("mix probability")
        plt.grid(True, alpha=0.25)
        plt.legend()
        fig.savefig(os.path.join(out_dir, "mix_p.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)


def save_topk(history: List[Dict], out_csv: str, k: int = 10, key_name: str = "val_acc"):
    # Top-k by metric desc, then loss asc
    def key(h):
        acc = h.get(key_name, 0.0) or 0.0
        loss = h.get("val_loss", 1e9) or 1e9
        return (-acc, loss)

    best = sorted(history, key=key)[:k]
    header = [
        "epoch",
        "train_loss", "train_acc", "train_f1_macro",
        "val_loss", "val_acc", "val_f1_macro",
        "test_loss", "test_acc", "test_f1_macro",
        "lr", "mix_p"
    ]
    lines = [",".join(header)]
    for h in best:
        lines.append(",".join(str(h.get(c, "")) for c in header))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_confusion(out_dir: str, prefix: str, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]):
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
    fig = plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{prefix} confusion matrix")
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
    lines.append("- `plots/loss.png`, `plots/acc.png`, `plots/f1_macro.png`, `plots/lr.png`, `plots/mix_p.png` — графики")
    lines.append("- `val_confusion_matrix.*`, `test_confusion_matrix.*` — confusion matrices")
    lines.append("- `val_classification_report.txt`, `test_classification_report.txt` — classification report")
    lines.append("- `topk_by_val.csv`, `topk_by_test.csv` — таблицы лучших эпох\n")

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


class FERPlusMappedDataset(Dataset):
    """
    root/train/<class>/*
    root/validation/<class>/*
    root/test/<class>/*
    Классы FERPlus: angry, contempt, disgust, fear, happy, neutral, sad, suprise/surprise.

    Маппинг в 7 классов:
      contempt -> (disgust|neutral|drop)  (по умолчанию: disgust)
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


def build_transforms(img_size: int, use_randaugment: bool, ra_n: int, ra_m: int):
    # FERPlus — grayscale face crops. ResNet ждёт 3 канала.
    ra = []
    if use_randaugment:
        try:
            ra = [transforms.RandAugment(num_ops=int(ra_n), magnitude=int(ra_m))]
        except Exception:
            ra = []

    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        *ra,
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.10, contrast=0.10)], p=0.25),
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


def set_requires_grad_backbone(model: nn.Module, req: bool):
    # resnet50: всё кроме fc
    for name, p in model.named_parameters():
        if name.startswith("fc."):
            p.requires_grad = True
        else:
            p.requires_grad = req


def freeze_bn(model: nn.Module):
    # BN в eval (заморозить running stats)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def one_hot(y: torch.Tensor, num_classes: int):
    return F.one_hot(y, num_classes=num_classes).float()


def mixup_cutmix_batch(x, y, num_classes: int, mixup_alpha: float, cutmix_alpha: float):
    # returns x_mixed, y_soft
    bs = x.size(0)
    device = x.device

    use_cutmix = (cutmix_alpha > 0) and (mixup_alpha <= 0 or random.random() < 0.5)

    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        rand_idx = torch.randperm(bs, device=device)
        x2 = x[rand_idx]
        y2 = y[rand_idx]

        _, _, H, W = x.shape
        cut_rat = math.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = random.randint(0, W - 1)
        cy = random.randint(0, H - 1)

        x1 = max(0, cx - cut_w // 2)
        x2b = min(W, cx + cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        y2b = min(H, cy + cut_h // 2)

        x[:, :, y1:y2b, x1:x2b] = x2[:, :, y1:y2b, x1:x2b]
        lam_adj = 1.0 - ((x2b - x1) * (y2b - y1) / (W * H))

        y1h = one_hot(y, num_classes)
        y2h = one_hot(y2, num_classes)
        y_soft = lam_adj * y1h + (1.0 - lam_adj) * y2h
        return x, y_soft

    # mixup
    lam = np.random.beta(mixup_alpha, mixup_alpha) if mixup_alpha > 0 else 1.0
    rand_idx = torch.randperm(bs, device=device)
    x2 = x[rand_idx]
    y2 = y[rand_idx]

    x_mix = lam * x + (1.0 - lam) * x2
    y1h = one_hot(y, num_classes)
    y2h = one_hot(y2, num_classes)
    y_soft = lam * y1h + (1.0 - lam) * y2h
    return x_mix, y_soft


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        sd = model.state_dict()
        for k, v in sd.items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=(1.0 - self.decay))

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        sd = model.state_dict()
        for k in self.shadow.keys():
            self.backup[k] = sd[k].detach().clone()
        model.load_state_dict({**sd, **self.shadow}, strict=False)

    def restore(self, model: nn.Module):
        if not self.backup:
            return
        sd = model.state_dict()
        model.load_state_dict({**sd, **self.backup}, strict=False)
        self.backup = {}


def mix_probability(epoch: int, cfg) -> float:
    # epoch starts at 1
    p0 = float(cfg.mix_prob)
    p1 = float(cfg.mix_prob_end)

    if cfg.mix_decay == "none":
        return p0

    # decay window
    s = int(cfg.mix_decay_start)
    e = int(cfg.mix_decay_end)
    if e <= s:
        return p1

    if epoch <= s:
        return p0
    if epoch >= e:
        return p1

    t = (epoch - s) / float(e - s)

    if cfg.mix_decay == "linear":
        return p0 + (p1 - p0) * t

    # cosine
    ct = 0.5 * (1.0 - math.cos(math.pi * t))  # 0..1
    return p0 + (p1 - p0) * ct


@dataclass
class TrainConfig:
    data_root: str
    out_root: str
    run_name: str
    epochs: int = 100
    batch_size: int = 128
    img_size: int = 224
    lr: float = 2e-4
    lr_head_mult: float = 3.0
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    device: str = "auto"  # auto|cuda|cpu
    amp: bool = True
    label_smoothing: float = 0.05
    dropout: float = 0.2
    use_class_weights: bool = True

    # schedulers
    scheduler: str = "cosine"  # cosine|step|none|onecycle|cosine_restart
    warmup_epochs: int = 0
    patience: int = 0
    topk: int = 10

    # mapping
    contempt_to: str = "disgust"  # disgust|neutral|drop

    # freeze
    freeze_epochs: int = 5
    freeze_bn: bool = True

    # mixup/cutmix
    mix_prob: float = 0.6
    mix_prob_end: float = 0.15
    mix_decay_start: int = 10
    mix_decay_end: int = 60
    mix_decay: str = "cosine"  # none|linear|cosine
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0

    # EMA
    ema_decay: float = 0.999
    ema_eval: bool = True

    # balanced sampler
    balanced_sampler: bool = False

    # RandAugment
    use_randaugment: bool = False
    ra_n: int = 2
    ra_m: int = 9

    # save
    save_every: int = 10
    resume: str = ""
    no_best_by_test: bool = False

    # eval only
    eval_only: bool = False
    ckpt: str = ""


def build_optimizer(model: nn.Module, cfg: TrainConfig):
    # разный LR: head = lr * lr_head_mult
    head_params = []
    base_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(p)
        else:
            base_params.append(p)

    params = [
        {"params": base_params, "lr": cfg.lr},
        {"params": head_params, "lr": cfg.lr * cfg.lr_head_mult},
    ]
    return torch.optim.AdamW(params, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    # scheduler step per-epoch (кроме onecycle)
    if cfg.scheduler == "none":
        return None

    if cfg.scheduler == "step":
        # грубая ступень: *0.1 на 60% и 80%
        def lr_lambda(epoch0):
            e = epoch0 + 1
            p = e / max(1, cfg.epochs)
            if p < 0.6:
                return 1.0
            if p < 0.8:
                return 0.1
            return 0.01
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))

    if cfg.scheduler == "cosine_restart":
        # Warm restarts: T_0=10, T_mult=2 — хороший дефолт для fine-tune
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    if cfg.scheduler == "onecycle":
        # onecycle — step per batch
        max_lr = [pg["lr"] for pg in optimizer.param_groups]
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=cfg.epochs,
            steps_per_epoch=max(1, steps_per_epoch),
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=1000.0,
            anneal_strategy="cos",
        )

    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")


@torch.no_grad()
def eval_split(model: nn.Module, loader: DataLoader, device, criterion, num_classes: int) -> Dict:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    y_true = []
    y_pred = []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total += bs
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())

    y_true_np = np.concatenate(y_true) if y_true else np.array([])
    y_pred_np = np.concatenate(y_pred) if y_pred else np.array([])

    out = {
        "loss": float(total_loss / max(1, total)),
        "acc": float(correct / max(1, total)),
        "n": int(total),
        "y_true": y_true_np,
        "y_pred": y_pred_np
    }
    if SKLEARN_OK and y_true_np.size:
        out["f1_macro"] = float(f1_score(y_true_np, y_pred_np, average="macro"))
    else:
        out["f1_macro"] = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help=r"C:\vk-vision-demo\data\FERPlus")
    ap.add_argument("--out_root", default=r"C:\vk-vision-demo\results\runs_emotion")
    ap.add_argument("--run_name", default=f"emotion_ferplus_resnet50_v5_{now_tag()}")

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_head_mult", type=float, default=3.0)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no_class_weights", action="store_true")

    ap.add_argument("--scheduler", default="cosine_restart",
                    choices=["cosine", "step", "none", "onecycle", "cosine_restart"])
    ap.add_argument("--warmup_epochs", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)

    ap.add_argument("--contempt_to", default="disgust", choices=["disgust", "neutral", "drop"])

    ap.add_argument("--freeze_epochs", type=int, default=5)
    ap.add_argument("--freeze_bn", action="store_true")
    ap.add_argument("--no_freeze_bn", action="store_true")

    ap.add_argument("--mix_prob", type=float, default=0.6)
    ap.add_argument("--mix_prob_end", type=float, default=0.15)
    ap.add_argument("--mix_decay_start", type=int, default=10)
    ap.add_argument("--mix_decay_end", type=int, default=60)
    ap.add_argument("--mix_decay", default="cosine", choices=["none", "linear", "cosine"])
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--cutmix_alpha", type=float, default=1.0)

    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--no_ema_eval", action="store_true")

    ap.add_argument("--balanced_sampler", action="store_true")

    ap.add_argument("--use_randaugment", action="store_true")
    ap.add_argument("--ra_n", type=int, default=2)
    ap.add_argument("--ra_m", type=int, default=9)

    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--resume", default="")
    ap.add_argument("--no_best_by_test", action="store_true")

    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--ckpt", default="")

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
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        topk=args.topk,
        contempt_to=args.contempt_to,
        freeze_epochs=args.freeze_epochs,
        freeze_bn=(True if args.freeze_bn else False) and (not args.no_freeze_bn),
        mix_prob=args.mix_prob,
        mix_prob_end=args.mix_prob_end,
        mix_decay_start=args.mix_decay_start,
        mix_decay_end=args.mix_decay_end,
        mix_decay=args.mix_decay,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        ema_decay=args.ema_decay,
        ema_eval=not args.no_ema_eval,
        balanced_sampler=args.balanced_sampler,
        use_randaugment=args.use_randaugment,
        ra_n=args.ra_n,
        ra_m=args.ra_m,
        save_every=args.save_every,
        resume=args.resume,
        no_best_by_test=args.no_best_by_test,
        eval_only=args.eval_only,
        ckpt=args.ckpt,
    )

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

    # counts
    class_counts = {
        "train": count_by_class_items(ds_train.items),
        "val": count_by_class_items(ds_val.items),
        "test": count_by_class_items(ds_test.items),
    }

    # sampler
    sampler = None
    shuffle = True
    if cfg.balanced_sampler:
        # weight per sample = 1/freq(class)
        cnt = np.zeros((len(APP_LABELS),), dtype=np.float32)
        for _, y in ds_train.items:
            cnt[int(y)] += 1.0
        cnt = np.maximum(cnt, 1.0)
        class_w = 1.0 / cnt
        sample_w = np.array([class_w[int(y)] for _, y in ds_train.items], dtype=np.float64)
        sampler = WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)
        shuffle = False

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=shuffle, sampler=sampler,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    # model
    model = build_model_resnet50(num_classes=len(APP_LABELS), dropout=cfg.dropout).to(device)

    # loss
    if cfg.use_class_weights:
        w = compute_class_weights(ds_train.items).to(device)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # freeze backbone first
    set_requires_grad_backbone(model, req=False)

    # optimizer & scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(dl_train))

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # EMA
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

    # save config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    # metrics files
    csv_path = os.path.join(run_dir, "metrics.csv")
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,test_loss,train_acc,val_acc,test_acc,train_f1_macro,val_f1_macro,test_f1_macro,lr,mix_p\n")

    # model paths (не трогаем старые)
    models_dir = os.path.abspath(os.path.join(os.getcwd(), "models"))
    ensure_dir(models_dir)

    best_by_val_run = os.path.join(run_dir, "emotion_resnet50_ferplus_v5_best_val.pth")
    best_by_test_run = os.path.join(run_dir, "emotion_resnet50_ferplus_v5_best_test.pth")

    best_by_val_models = os.path.join(models_dir, "emotion_resnet50_ferplus_v5_best_val.pth")
    best_by_test_models = os.path.join(models_dir, "emotion_resnet50_ferplus_v5_best_test.pth")

    last_path = os.path.join(ckpt_dir, "last.pth")

    # resume
    start_epoch = 1
    best_val_acc = -1.0
    best_test_acc = -1.0
    bad_epochs = 0
    history: List[Dict] = []

    if cfg.resume:
        payload = torch.load(cfg.resume, map_location="cpu")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload.get("optimizer", optimizer.state_dict()))
        if "scaler" in payload and use_amp:
            scaler.load_state_dict(payload["scaler"])
        if "scheduler" in payload and scheduler is not None:
            try:
                scheduler.load_state_dict(payload["scheduler"])
            except Exception:
                pass
        if ema is not None and "ema" in payload and isinstance(payload["ema"], dict):
            ema.shadow = payload["ema"]
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_val_acc = float(payload.get("best_val_acc", best_val_acc))
        best_test_acc = float(payload.get("best_test_acc", best_test_acc))
        history = payload.get("history", [])
        print(f"[INFO] resumed from: {cfg.resume} | start_epoch={start_epoch}")

    # eval_only
    if cfg.eval_only:
        ckpt = cfg.ckpt or best_by_val_run
        payload = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(payload["model"], strict=True)
        model.to(device)
        if cfg.freeze_bn:
            freeze_bn(model)

        # EMA eval if present in checkpoint
        if ema is not None and cfg.ema_eval and "ema" in payload and isinstance(payload["ema"], dict):
            ema.shadow = payload["ema"]

        def _eval_with_optional_ema():
            if ema is not None and cfg.ema_eval:
                ema.apply_shadow(model)
            val_res = eval_split(model, dl_val, device, criterion, len(APP_LABELS))
            test_res = eval_split(model, dl_test, device, criterion, len(APP_LABELS))
            if ema is not None and cfg.ema_eval:
                ema.restore(model)
            return val_res, test_res

        val_res, test_res = _eval_with_optional_ema()

        out_dir = os.path.join(run_dir, "eval_only")
        ensure_dir(out_dir)

        if SKLEARN_OK:
            save_confusion(out_dir, "val", val_res["y_true"], val_res["y_pred"], APP_LABELS)
            save_confusion(out_dir, "test", test_res["y_true"], test_res["y_pred"], APP_LABELS)

            vr = classification_report(val_res["y_true"], val_res["y_pred"], target_names=APP_LABELS, digits=4, zero_division=0)
            tr = classification_report(test_res["y_true"], test_res["y_pred"], target_names=APP_LABELS, digits=4, zero_division=0)
            with open(os.path.join(out_dir, "val_classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(vr)
            with open(os.path.join(out_dir, "test_classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(tr)

        summary = {
            "ckpt": ckpt,
            "val": {k: val_res[k] for k in ["loss", "acc", "f1_macro", "n"]},
            "test": {k: test_res[k] for k in ["loss", "acc", "f1_macro", "n"]},
            "device": str(device),
        }
        with open(os.path.join(out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("\nEVAL ONLY DONE")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    notes = [
        f"FERPlus(8) → APP(7): contempt→{cfg.contempt_to}, suprise→surprise",
        "Model: ResNet50 ImageNet-pretrained + dropout",
        f"RandAugment: {cfg.use_randaugment} (N={cfg.ra_n}, M={cfg.ra_m})",
        f"Balanced sampler: {cfg.balanced_sampler}",
        f"Mix: p={cfg.mix_prob}->{cfg.mix_prob_end} decay={cfg.mix_decay} [{cfg.mix_decay_start}..{cfg.mix_decay_end}], mixup={cfg.mixup_alpha}, cutmix={cfg.cutmix_alpha}",
        f"EMA: decay={cfg.ema_decay}, ema_eval={cfg.ema_eval}",
        f"Scheduler: {cfg.scheduler}",
    ]

    print(f"[INFO] device: {device} | amp: {use_amp}")
    print(f"[INFO] data_root: {cfg.data_root}")
    print(f"[INFO] run_dir  : {run_dir}")
    print(f"[INFO] contempt_to: {cfg.contempt_to}")
    print(f"[INFO] counts(train/val/test): {class_counts}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        # unfreeze backbone
        if epoch == cfg.freeze_epochs + 1:
            set_requires_grad_backbone(model, req=True)
            # пересоберём optimizer, чтобы включить параметры backbone (иначе они были frozen)
            optimizer = build_optimizer(model, cfg)
            scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(dl_train))
            print(f"[INFO] unfreeze backbone at epoch {epoch}")

        # freeze BN if asked
        if cfg.freeze_bn:
            freeze_bn(model)

        # mix probability schedule
        cur_mix_p = mix_probability(epoch, cfg)

        # ---- train ----
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        tr_y_true = []
        tr_y_pred = []

        pbar = tqdm(dl_train, desc=f"train e{epoch:03d}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # mixup/cutmix
            if cur_mix_p > 0 and random.random() < cur_mix_p:
                x_m, y_soft = mixup_cutmix_batch(x, y, len(APP_LABELS), cfg.mixup_alpha, cfg.cutmix_alpha)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x_m)
                    logp = F.log_softmax(logits, dim=1)
                    loss = -(y_soft * logp).sum(dim=1).mean()
            else:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None and cfg.scheduler == "onecycle":
                scheduler.step()

            if ema is not None:
                ema.update(model)

            bs = x.size(0)
            tr_loss += loss.item() * bs
            tr_total += bs

            pred = logits.argmax(dim=1)
            tr_correct += (pred == y).sum().item()

            if SKLEARN_OK:
                tr_y_true.append(y.detach().cpu().numpy())
                tr_y_pred.append(pred.detach().cpu().numpy())

        train_loss = tr_loss / max(1, tr_total)
        train_acc = tr_correct / max(1, tr_total)
        train_f1 = None
        if SKLEARN_OK and tr_y_true:
            train_f1 = float(f1_score(np.concatenate(tr_y_true), np.concatenate(tr_y_pred), average="macro"))

        # scheduler step per epoch
        if scheduler is not None and cfg.scheduler in ("cosine", "step"):
            scheduler.step()
        elif scheduler is not None and cfg.scheduler == "cosine_restart":
            # step with epoch as float
            scheduler.step(epoch - 1)

        # current lr (log first group)
        cur_lr = float(optimizer.param_groups[0]["lr"])

        # ---- eval (val/test) ----
        def _eval_with_optional_ema():
            if ema is not None and cfg.ema_eval:
                ema.apply_shadow(model)
            val_res = eval_split(model, dl_val, device, criterion, len(APP_LABELS))
            test_res = eval_split(model, dl_test, device, criterion, len(APP_LABELS))
            if ema is not None and cfg.ema_eval:
                ema.restore(model)
            return val_res, test_res

        val_res, test_res = _eval_with_optional_ema()

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_res["loss"]),
            "test_loss": float(test_res["loss"]),
            "train_acc": float(train_acc),
            "val_acc": float(val_res["acc"]),
            "test_acc": float(test_res["acc"]),
            "train_f1_macro": float(train_f1) if train_f1 is not None else None,
            "val_f1_macro": float(val_res["f1_macro"]) if val_res.get("f1_macro") is not None else None,
            "test_f1_macro": float(test_res["f1_macro"]) if test_res.get("f1_macro") is not None else None,
            "lr": float(cur_lr),
            "mix_p": float(cur_mix_p),
        }
        history.append(row)

        # write metrics
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{row['train_loss']:.6f},{row['val_loss']:.6f},{row['test_loss']:.6f},"
                f"{row['train_acc']:.6f},{row['val_acc']:.6f},{row['test_acc']:.6f},"
                f"{(row['train_f1_macro'] if row['train_f1_macro'] is not None else '')},"
                f"{(row['val_f1_macro'] if row['val_f1_macro'] is not None else '')},"
                f"{(row['test_f1_macro'] if row['test_f1_macro'] is not None else '')},"
                f"{row['lr']:.8f},{row['mix_p']:.4f}\n"
            )
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        vf1 = row["val_f1_macro"]
        tf1 = row["test_f1_macro"]
        vf1s = f" val_f1={vf1:.4f}" if isinstance(vf1, float) else ""
        print(
            f"[epoch {epoch:03d}] "
            f"val_acc={row['val_acc']:.4f} val_loss={row['val_loss']:.4f}{vf1s} | "
            f"test_acc={row['test_acc']:.4f} test_loss={row['test_loss']:.4f} | "
            f"lr={row['lr']:.2e} mix_p={row['mix_p']:.2f}"
        )

        # update plots each epoch
        plot_curves(history, plots_dir)

        # save bests
        improved_val = row["val_acc"] > best_val_acc + 1e-6
        improved_test = (not cfg.no_best_by_test) and (row["test_acc"] > best_test_acc + 1e-6)

        # when improved val -> dump confusion/reports
        if improved_val:
            best_val_acc = row["val_acc"]
            bad_epochs = 0

            payload = {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "best_test_acc": best_test_acc,
                "labels": APP_LABELS,
                "config": asdict(cfg),
                "history": history,
            }
            if ema is not None:
                payload["ema"] = ema.shadow

            torch.save(payload, best_by_val_run)
            torch.save(payload, best_by_val_models)

            if SKLEARN_OK and val_res["y_true"].size:
                save_confusion(run_dir, "val", val_res["y_true"], val_res["y_pred"], APP_LABELS)
                vr = classification_report(val_res["y_true"], val_res["y_pred"], target_names=APP_LABELS, digits=4, zero_division=0)
                with open(os.path.join(run_dir, "val_classification_report.txt"), "w", encoding="utf-8") as f:
                    f.write(vr)

            print(f"[INFO] best_by_val saved -> {best_by_val_models} (val_acc={best_val_acc:.4f})")
        else:
            bad_epochs += 1

        if improved_test:
            best_test_acc = row["test_acc"]
            payload = {
                "model": model.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "best_test_acc": best_test_acc,
                "labels": APP_LABELS,
                "config": asdict(cfg),
                "history": history,
            }
            if ema is not None:
                payload["ema"] = ema.shadow

            torch.save(payload, best_by_test_run)
            torch.save(payload, best_by_test_models)

            if SKLEARN_OK and test_res["y_true"].size:
                save_confusion(run_dir, "test", test_res["y_true"], test_res["y_pred"], APP_LABELS)
                tr = classification_report(test_res["y_true"], test_res["y_pred"], target_names=APP_LABELS, digits=4, zero_division=0)
                with open(os.path.join(run_dir, "test_classification_report.txt"), "w", encoding="utf-8") as f:
                    f.write(tr)

            print(f"[INFO] best_by_test saved -> {best_by_test_models} (test_acc={best_test_acc:.4f})")

        # periodic checkpoint (last + every N)
        payload_last = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if use_amp else None,
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
            "labels": APP_LABELS,
            "config": asdict(cfg),
            "history": history,
        }
        if scheduler is not None:
            try:
                payload_last["scheduler"] = scheduler.state_dict()
            except Exception:
                pass
        if ema is not None:
            payload_last["ema"] = ema.shadow

        torch.save(payload_last, last_path)
        if cfg.save_every > 0 and (epoch % cfg.save_every == 0):
            torch.save(payload_last, os.path.join(ckpt_dir, f"e{epoch:03d}.pth"))

        # early stop by val_acc
        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"[INFO] early stop: no improvement in val_acc for {cfg.patience} epochs")
            break

    # top-k tables
    save_topk(history, os.path.join(run_dir, "topk_by_val.csv"), k=cfg.topk, key_name="val_acc")
    save_topk(history, os.path.join(run_dir, "topk_by_test.csv"), k=cfg.topk, key_name="test_acc")

    best_summary = {
        "best_val_acc": best_val_acc,
        "best_test_acc": best_test_acc,
        "best_by_val": best_by_val_models,
        "best_by_test": best_by_test_models if not cfg.no_best_by_test else None,
        "run_dir": run_dir,
        "labels": APP_LABELS,
        "contempt_to": cfg.contempt_to,
    }
    write_report_md(run_dir, asdict(cfg), best_summary, class_counts, notes)

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Run dir: {run_dir}")
    print(f"Best_by_val : {best_by_val_models} (val_acc={best_val_acc:.4f})")
    if not cfg.no_best_by_test:
        print(f"Best_by_test: {best_by_test_models} (test_acc={best_test_acc:.4f})")
    print(f"Resume from : {last_path}")


if __name__ == "__main__":
    main()
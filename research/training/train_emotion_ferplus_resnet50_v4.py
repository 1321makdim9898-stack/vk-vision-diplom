# train_emotion_ferplus_resnet50_v4.py
# Улучшенная версия обучения эмоций на FERPlus (маппинг 8→7 классов).
# Добавлено: freeze/unfreeze, mixup/cutmix, EMA, resume, best_by_val & best_by_test,
# test-метрики и test_confusion_matrix, расширенные графики и отчёт.

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


# -----------------------------
# Utils: plots / reports
# -----------------------------
def plot_curves(history: List[Dict], out_dir: str):
    epochs = [h["epoch"] for h in history]

    def s(k):
        return [h.get(k) for h in history]

    def save_plot(keys: List[str], ylabel: str, fname: str):
        fig = plt.figure()
        for k in keys:
            if any(v is not None for v in s(k)):
                plt.plot(epochs, s(k), label=k)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend()
        fig.savefig(os.path.join(out_dir, fname), dpi=160, bbox_inches="tight")
        plt.close(fig)

    save_plot(["train_loss", "val_loss", "test_loss"], "loss", "loss.png")
    save_plot(["train_acc", "val_acc", "test_acc"], "accuracy", "acc.png")
    save_plot(["train_f1_macro", "val_f1_macro", "test_f1_macro"], "F1 macro", "f1_macro.png")
    save_plot(["lr"], "learning rate", "lr.png")


def save_topk(history: List[Dict], out_csv: str, k: int = 10, by: str = "val_acc"):
    # by: val_acc|test_acc
    assert by in ("val_acc", "test_acc")

    def key(h):
        return (
            -(h.get(by, 0.0) or 0.0),
            (h.get("val_loss", 1e9) or 1e9),
            (h.get("test_loss", 1e9) or 1e9),
        )

    best = sorted(history, key=key)[:k]
    header = [
        "epoch",
        "train_loss", "train_acc", "train_f1_macro",
        "val_loss", "val_acc", "val_f1_macro",
        "test_loss", "test_acc", "test_f1_macro",
        "lr"
    ]
    lines = [",".join(header)]
    for h in best:
        lines.append(",".join(str(h.get(c, "")) for c in header))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_confusion(prefix_path_no_ext: str, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], title: str):
    """
    prefix_path_no_ext: например .../val_confusion_matrix
    сохранит:
      - <prefix>.png
      - <prefix>.csv
    """
    if not SKLEARN_OK:
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    # csv
    csv_path = prefix_path_no_ext + ".csv"
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
    fig.savefig(prefix_path_no_ext + ".png", dpi=160)
    plt.close(fig)


def write_report_md(run_dir: str, cfg: Dict, summary: Dict, class_counts: Dict[str, Dict[str, int]], notes: List[str]):
    lines = []
    lines.append("# Emotion (FERPlus→7) v4 — отчёт обучения\n")
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

    lines.append("## Итоги / best checkpoints\n")
    lines.append("```json")
    lines.append(json.dumps(summary, ensure_ascii=False, indent=2))
    lines.append("```\n")

    lines.append("## Артефакты\n")
    lines.append("- `metrics.csv`, `metrics.jsonl` — метрики по эпохам (train/val/test)")
    lines.append("- `plots/loss.png`, `plots/acc.png`, `plots/f1_macro.png`, `plots/lr.png` — графики")
    lines.append("- `val_confusion_matrix.png/.csv` — confusion matrix (validation) для best_by_val")
    lines.append("- `test_confusion_matrix.png/.csv` — confusion matrix (test) для best_by_test")
    lines.append("- `classification_report_val.txt`, `classification_report_test.txt` — отчёты (если sklearn установлен)")
    lines.append("- `topk_by_val.csv`, `topk_by_test.csv` — таблицы лучших эпох")
    lines.append("- `test_metrics.json` — лучшая метрика на test\n")

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# Metrics helpers
# -----------------------------
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
    cnt = np.ones((len(APP_LABELS),), dtype=np.float32)
    for _, y in items:
        cnt[int(y)] += 1
    inv = 1.0 / cnt
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


# -----------------------------
# Dataset
# -----------------------------
class FERPlusMappedDataset(Dataset):
    """
    root/train/<class>/*
    root/validation/<class>/*
    root/test/<class>/*
    FERPlus classes: angry, contempt, disgust, fear, happy, neutral, sad, suprise/surprise.

    Mapping 8->7:
      contempt -> (disgust|neutral|drop)
      suprise  -> surprise
    """
    def __init__(self, root: str, split: str, transform, contempt_to: str = "disgust"):
        assert split in ("train", "validation", "test")
        assert contempt_to in ("disgust", "neutral", "drop")

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
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, torch.tensor(y, dtype=torch.long)


# -----------------------------
# MixUp / CutMix
# -----------------------------
def one_hot(y: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    y = y.view(-1)
    out = torch.zeros((y.size(0), num_classes), device=device, dtype=torch.float32)
    out.scatter_(1, y.unsqueeze(1), 1.0)
    return out


def rand_bbox(W: int, H: int, lam: float) -> Tuple[int, int, int, int]:
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = int(np.clip(cx - cut_w // 2, 0, W))
    y1 = int(np.clip(cy - cut_h // 2, 0, H))
    x2 = int(np.clip(cx + cut_w // 2, 0, W))
    y2 = int(np.clip(cy + cut_h // 2, 0, H))
    return x1, y1, x2, y2


def apply_mixup_cutmix(x: torch.Tensor, y: torch.Tensor, num_classes: int,
                       mixup_alpha: float, cutmix_alpha: float, prob: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if prob <= 0.0:
        return x, one_hot(y, num_classes, x.device)

    r = np.random.rand()
    if r > prob:
        return x, one_hot(y, num_classes, x.device)

    bs = x.size(0)
    idx = torch.randperm(bs, device=x.device)
    y1 = one_hot(y, num_classes, x.device)
    y2 = one_hot(y[idx], num_classes, x.device)

    use_cutmix = (np.random.rand() < 0.5) and (cutmix_alpha > 0)
    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        _, _, H, W = x.size()
        x1b, y1b, x2b, y2b = rand_bbox(W, H, lam)
        x_mix = x.clone()
        x_mix[:, :, y1b:y2b, x1b:x2b] = x[idx, :, y1b:y2b, x1b:x2b]
        area = (x2b - x1b) * (y2b - y1b)
        lam = 1.0 - area / float(W * H)
    else:
        if mixup_alpha <= 0:
            return x, y1
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        x_mix = x * lam + x[idx] * (1.0 - lam)

    y_soft = y1 * lam + y2 * (1.0 - lam)
    return x_mix, y_soft


def soft_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    return -(soft_targets * logp).sum(dim=1).mean()


# -----------------------------
# EMA
# -----------------------------
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
            if name in self.shadow:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def apply_to(self, model: nn.Module):
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


# -----------------------------
# Transforms / Model
# -----------------------------
def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(img_size, scale=(0.70, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.12, contrast=0.12)], p=0.25),
        transforms.RandomRotation(degrees=12),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
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


def build_model_resnet50(num_classes: int, dropout: float, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.resnet50(weights=weights)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_f, num_classes)
    )
    return m


def set_freeze_backbone(model: nn.Module, freeze: bool):
    for name, p in model.named_parameters():
        if name.startswith("fc."):
            p.requires_grad = True
        else:
            p.requires_grad = (not freeze)


def get_param_groups(model: nn.Module, lr: float, lr_head_mult: float) -> List[Dict]:
    head = []
    body = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc."):
            head.append(p)
        else:
            body.append(p)
    groups = []
    if body:
        groups.append({"params": body, "lr": lr})
    if head:
        groups.append({"params": head, "lr": lr * lr_head_mult})
    return groups


@dataclass
class TrainConfig:
    data_root: str
    out_root: str
    run_name: str
    epochs: int = 80
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
    dropout: float = 0.25
    use_class_weights: bool = True
    scheduler: str = "cosine"  # cosine|step|none
    warmup_epochs: int = 2
    topk: int = 10
    patience: int = 0
    contempt_to: str = "disgust"  # disgust|neutral|drop

    # v4 additions
    freeze_epochs: int = 3
    mix_prob: float = 0.6
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    ema_decay: float = 0.999
    eval_with_ema: bool = True
    save_every: int = 1
    resume: str = ""


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion_hard: nn.Module) -> Tuple[float, float, Optional[float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    ys = []
    ps = []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion_hard(logits, y)

            bs = x.size(0)
            total_loss += float(loss.item()) * bs
            total += bs

            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())

            ys.append(y.detach().cpu().numpy())
            ps.append(pred.detach().cpu().numpy())

    loss_avg = total_loss / max(1, total)
    acc = correct / max(1, total)
    y_true = np.concatenate(ys) if ys else np.array([])
    y_pred = np.concatenate(ps) if ps else np.array([])
    f1m = macro_f1(y_true, y_pred) if y_true.size else None
    return loss_avg, acc, f1m, y_true, y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help=r"C:\vk-vision-demo\data\FERPlus")
    ap.add_argument("--out_root", default=r"C:\vk-vision-demo\results\runs_emotion", help="куда сохранять run")
    ap.add_argument("--run_name", default=f"emotion_ferplus_resnet50_v4_{now_tag()}")
    ap.add_argument("--epochs", type=int, default=80)
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
    ap.add_argument("--dropout", type=float, default=0.25)
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "none"])
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--contempt_to", default="disgust", choices=["disgust", "neutral", "drop"])

    # v4
    ap.add_argument("--freeze_epochs", type=int, default=3)
    ap.add_argument("--mix_prob", type=float, default=0.6)
    ap.add_argument("--mixup_alpha", type=float, default=0.2)
    ap.add_argument("--cutmix_alpha", type=float, default=1.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)
    ap.add_argument("--no_ema_eval", action="store_true")
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--resume", default="", help="путь к checkpoint last.pth для продолжения")
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
        mix_prob=args.mix_prob,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        ema_decay=args.ema_decay,
        eval_with_ema=not args.no_ema_eval,
        save_every=args.save_every,
        resume=args.resume.strip(),
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

    # datasets
    train_tf, val_tf = build_transforms(cfg.img_size)
    ds_train = FERPlusMappedDataset(cfg.data_root, "train", transform=train_tf, contempt_to=cfg.contempt_to)
    ds_val = FERPlusMappedDataset(cfg.data_root, "validation", transform=val_tf, contempt_to=cfg.contempt_to)
    ds_test = FERPlusMappedDataset(cfg.data_root, "test", transform=val_tf, contempt_to=cfg.contempt_to)

    # loaders
    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
                          persistent_workers=(cfg.num_workers > 0))
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
                        persistent_workers=(cfg.num_workers > 0))
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True, drop_last=False,
                         persistent_workers=(cfg.num_workers > 0))

    # model
    model = build_model_resnet50(num_classes=len(APP_LABELS), dropout=cfg.dropout, pretrained=True).to(device)

    # freeze start
    if cfg.freeze_epochs > 0:
        set_freeze_backbone(model, freeze=True)

    # loss
    if cfg.use_class_weights:
        w = compute_class_weights(ds_train.items).to(device)
        criterion_hard = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    else:
        criterion_hard = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = torch.optim.AdamW(
        get_param_groups(model, lr=cfg.lr, lr_head_mult=cfg.lr_head_mult),
        weight_decay=cfg.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

    def lr_mult_at_epoch(e: int):
        if cfg.scheduler == "none":
            return 1.0
        if cfg.scheduler == "step":
            p = e / max(1, cfg.epochs)
            if p < 0.4:
                return 1.0
            if p < 0.7:
                return 0.1
            return 0.01
        return 0.5 * (1.0 + math.cos(math.pi * (e - 1) / max(1, cfg.epochs)))

    def set_lr_for_epoch(e: int):
        base_mult = lr_mult_at_epoch(e)
        if cfg.warmup_epochs > 0 and e <= cfg.warmup_epochs:
            warm = e / cfg.warmup_epochs
        else:
            warm = 1.0
        cur_lr = cfg.lr * base_mult * warm
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]["lr"] = cur_lr
        else:
            optimizer.param_groups[0]["lr"] = cur_lr
            optimizer.param_groups[1]["lr"] = cur_lr * cfg.lr_head_mult
        return cur_lr

    # save config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    # metrics files
    csv_path = os.path.join(run_dir, "metrics.csv")
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,test_loss,train_acc,val_acc,test_acc,train_f1_macro,val_f1_macro,test_f1_macro,lr\n")

    # counts
    counts = {
        "train": count_by_class_items(ds_train.items),
        "val": count_by_class_items(ds_val.items),
        "test": count_by_class_items(ds_test.items),
    }

    models_dir = os.path.abspath(os.path.join(os.getcwd(), "models"))
    ensure_dir(models_dir)

    best_val_path_run = os.path.join(run_dir, "emotion_resnet50_ferplus_v4_best_val.pth")
    best_test_path_run = os.path.join(run_dir, "emotion_resnet50_ferplus_v4_best_test.pth")
    best_val_path_models = os.path.join(models_dir, "emotion_resnet50_ferplus_v4_best_val.pth")
    best_test_path_models = os.path.join(models_dir, "emotion_resnet50_ferplus_v4_best_test.pth")

    last_ckpt_path = os.path.join(ckpt_dir, "last.pth")

    start_epoch = 1
    best_val_acc = -1.0
    best_test_acc = -1.0
    best_val_epoch = -1
    best_test_epoch = -1
    history: List[Dict] = []
    bad_epochs = 0

    def save_checkpoint(path: str, epoch: int, cur_lr: float):
        payload = {
            "model": model.state_dict(),
            "ema": ema.shadow if ema is not None else None,
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc,
            "best_val_epoch": best_val_epoch,
            "best_test_epoch": best_test_epoch,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "labels": APP_LABELS,
            "config": asdict(cfg),
            "history": history,
            "lr": cur_lr,
        }
        torch.save(payload, path)

    def try_resume(path: str):
        nonlocal start_epoch, best_val_acc, best_test_acc, best_val_epoch, best_test_epoch, history, bad_epochs
        if not path:
            return
        if not os.path.isfile(path):
            print(f"[WARN] resume file not found: {path}")
            return
        ck = torch.load(path, map_location="cpu")
        if isinstance(ck, dict) and "model" in ck:
            model.load_state_dict(ck["model"], strict=True)
            if ck.get("optimizer"):
                optimizer.load_state_dict(ck["optimizer"])
            if ck.get("scaler") and scaler is not None:
                try:
                    scaler.load_state_dict(ck["scaler"])
                except Exception:
                    pass
            if ema is not None and isinstance(ck.get("ema"), dict):
                ema.shadow = {k: v.clone() for k, v in ck["ema"].items()}
            best_val_acc = float(ck.get("best_val_acc", best_val_acc))
            best_test_acc = float(ck.get("best_test_acc", best_test_acc))
            best_val_epoch = int(ck.get("best_val_epoch", best_val_epoch))
            best_test_epoch = int(ck.get("best_test_epoch", best_test_epoch))
            history = list(ck.get("history", history))
            last_epoch = int(ck.get("epoch", 0))
            start_epoch = last_epoch + 1
            bad_epochs = 0
            print(f"[INFO] resumed from {path} at epoch {last_epoch}, next={start_epoch}")
        else:
            print(f"[WARN] resume file has unexpected format: {path}")

    try_resume(cfg.resume)

    notes = [
        f"FERPlus(8) → APP(7): contempt→{cfg.contempt_to}, suprise→surprise",
        "Model: ResNet50 ImageNet-pretrained + dropout",
        f"Freeze backbone первые {cfg.freeze_epochs} эпох",
        f"MixUp/CutMix: prob={cfg.mix_prob}, mixup_alpha={cfg.mixup_alpha}, cutmix_alpha={cfg.cutmix_alpha}",
        f"EMA: decay={cfg.ema_decay}, eval_with_ema={cfg.eval_with_ema}",
        "Eval: validation + test; test_confusion_matrix сохраняется отдельно",
    ]

    print(f"[INFO] device: {device} | amp: {use_amp}")
    print(f"[INFO] data_root: {cfg.data_root}")
    print(f"[INFO] run_dir  : {run_dir}")
    print(f"[INFO] contempt_to: {cfg.contempt_to}")
    print(f"[INFO] counts(train/val/test): {counts}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        if cfg.freeze_epochs > 0 and epoch == cfg.freeze_epochs + 1:
            set_freeze_backbone(model, freeze=False)
            optimizer = torch.optim.AdamW(
                get_param_groups(model, lr=cfg.lr, lr_head_mult=cfg.lr_head_mult),
                weight_decay=cfg.weight_decay
            )
            print(f"[INFO] unfreeze backbone at epoch {epoch}")

        cur_lr = set_lr_for_epoch(epoch)

        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        tr_y_true = []
        tr_y_pred = []

        for x, y in tqdm(dl_train, desc=f"train e{epoch:03d}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            x_mix, y_soft = apply_mixup_cutmix(
                x, y,
                num_classes=len(APP_LABELS),
                mixup_alpha=cfg.mixup_alpha,
                cutmix_alpha=cfg.cutmix_alpha,
                prob=cfg.mix_prob
            )

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x_mix)
                loss = soft_cross_entropy(logits, y_soft)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            bs = x.size(0)
            tr_loss += float(loss.item()) * bs
            tr_total += bs

            pred = logits.argmax(dim=1)
            tr_correct += int((pred == y).sum().item())

            if SKLEARN_OK:
                tr_y_true.append(y.detach().cpu().numpy())
                tr_y_pred.append(pred.detach().cpu().numpy())

        train_loss = tr_loss / max(1, tr_total)
        train_acc = tr_correct / max(1, tr_total)
        train_f1 = macro_f1(np.concatenate(tr_y_true), np.concatenate(tr_y_pred)) if (SKLEARN_OK and tr_y_true) else None

        # eval with EMA
        if ema is not None and cfg.eval_with_ema:
            ema.apply_to(model)

        val_loss, val_acc, val_f1, yv_true, yv_pred = evaluate(model, dl_val, device, criterion_hard)
        test_loss, test_acc, test_f1, yt_true, yt_pred = evaluate(model, dl_test, device, criterion_hard)

        if ema is not None and cfg.eval_with_ema:
            ema.restore(model)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "test_loss": float(test_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "test_acc": float(test_acc),
            "train_f1_macro": float(train_f1) if train_f1 is not None else None,
            "val_f1_macro": float(val_f1) if val_f1 is not None else None,
            "test_f1_macro": float(test_f1) if test_f1 is not None else None,
            "lr": float(cur_lr),
        }
        history.append(row)

        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},{test_loss:.6f},"
                f"{train_acc:.6f},{val_acc:.6f},{test_acc:.6f},"
                f"{(train_f1 if train_f1 is not None else '')},{(val_f1 if val_f1 is not None else '')},{(test_f1 if test_f1 is not None else '')},"
                f"{cur_lr:.8f}\n"
            )
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[epoch {epoch:03d}] val_acc={val_acc:.4f} val_loss={val_loss:.4f} | test_acc={test_acc:.4f} test_loss={test_loss:.4f} | lr={cur_lr:.2e}")

        plot_curves(history, plots_dir)

        if cfg.save_every > 0 and (epoch % cfg.save_every == 0):
            save_checkpoint(last_ckpt_path, epoch, cur_lr)

        improved_val = val_acc > best_val_acc + 1e-6
        if improved_val:
            best_val_acc = float(val_acc)
            best_val_epoch = int(epoch)
            payload = {
                "model": model.state_dict(),
                "ema": ema.shadow if ema is not None else None,
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "labels": APP_LABELS,
                "config": asdict(cfg),
            }
            torch.save(payload, best_val_path_run)
            torch.save(payload, best_val_path_models)

            if SKLEARN_OK and yv_true.size:
                save_confusion(os.path.join(run_dir, "val_confusion_matrix"), yv_true, yv_pred, APP_LABELS,
                               title=f"VAL confusion (best_by_val @ epoch {epoch})")
                rep = classification_report(yv_true, yv_pred, target_names=APP_LABELS, digits=4)
                with open(os.path.join(run_dir, "classification_report_val.txt"), "w", encoding="utf-8") as f:
                    f.write(rep)

            bad_epochs = 0
            print(f"[INFO] best_by_val saved -> {best_val_path_models} (val_acc={best_val_acc:.4f})")
        else:
            bad_epochs += 1

        improved_test = test_acc > best_test_acc + 1e-6
        if improved_test:
            best_test_acc = float(test_acc)
            best_test_epoch = int(epoch)
            payload = {
                "model": model.state_dict(),
                "ema": ema.shadow if ema is not None else None,
                "epoch": epoch,
                "best_test_acc": best_test_acc,
                "labels": APP_LABELS,
                "config": asdict(cfg),
            }
            torch.save(payload, best_test_path_run)
            torch.save(payload, best_test_path_models)

            if SKLEARN_OK and yt_true.size:
                save_confusion(os.path.join(run_dir, "test_confusion_matrix"), yt_true, yt_pred, APP_LABELS,
                               title=f"TEST confusion (best_by_test @ epoch {epoch})")
                rep = classification_report(yt_true, yt_pred, target_names=APP_LABELS, digits=4)
                with open(os.path.join(run_dir, "classification_report_test.txt"), "w", encoding="utf-8") as f:
                    f.write(rep)

            with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {"best_test_acc": best_test_acc, "best_test_epoch": best_test_epoch, "path": best_test_path_models},
                    f, ensure_ascii=False, indent=2
                )

            print(f"[INFO] best_by_test saved -> {best_test_path_models} (test_acc={best_test_acc:.4f})")

        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"[INFO] early stop: no improvement in val_acc for {cfg.patience} epochs")
            break

    save_topk(history, os.path.join(run_dir, "topk_by_val.csv"), k=cfg.topk, by="val_acc")
    save_topk(history, os.path.join(run_dir, "topk_by_test.csv"), k=cfg.topk, by="test_acc")

    summary = {
        "best_val_acc": best_val_acc,
        "best_val_epoch": best_val_epoch,
        "best_val_path_models": best_val_path_models,
        "best_test_acc": best_test_acc,
        "best_test_epoch": best_test_epoch,
        "best_test_path_models": best_test_path_models,
        "run_dir": run_dir,
        "labels": APP_LABELS,
        "contempt_to": cfg.contempt_to,
        "resume_hint": last_ckpt_path,
    }

    write_report_md(run_dir, asdict(cfg), summary, counts, notes)

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Run dir: {run_dir}")
    print(f"Best_by_val : {best_val_path_models} (val_acc={best_val_acc:.4f} @ epoch {best_val_epoch})")
    print(f"Best_by_test: {best_test_path_models} (test_acc={best_test_acc:.4f} @ epoch {best_test_epoch})")
    print(f"Resume from : {last_ckpt_path}")


if __name__ == "__main__":
    main()
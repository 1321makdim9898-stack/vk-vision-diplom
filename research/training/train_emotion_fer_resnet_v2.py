# train_emotion_fer_resnet_v2.py
import os
import json
import time
import math
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from tqdm import tqdm

# ВАЖНО: на Windows/без GUI, чтобы не было tkinter ошибок
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# optional: для confusion matrix
try:
    from sklearn.metrics import confusion_matrix, f1_score, classification_report
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


APP_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


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


def compute_class_weights_from_folder(train_dir: str, class_names: List[str]) -> torch.Tensor:
    # weights ~ 1/freq (normalized)
    counts = []
    for c in class_names:
        p = os.path.join(train_dir, c)
        n = 0
        if os.path.isdir(p):
            for f in os.listdir(p):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    n += 1
        counts.append(max(1, n))
    counts = np.array(counts, dtype=np.float32)
    inv = 1.0 / counts
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


def save_topk(history: List[Dict], out_csv: str, k: int = 10):
    # Top-k by val_acc desc, then val_loss asc, then val_f1_macro desc
    def key(h):
        return (-h.get("val_acc", 0.0), h.get("val_loss", 1e9), -(h.get("val_f1_macro") or 0.0))

    best = sorted(history, key=key)[:k]
    header = [
        "epoch",
        "val_acc", "val_f1_macro", "val_loss",
        "train_acc", "train_f1_macro", "train_loss",
        "lr"
    ]
    lines = [",".join(header)]
    for h in best:
        lines.append(",".join(str(h.get(c, "")) for c in header))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_confusion(run_dir: str, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], title: str):
    if not SKLEARN_OK:
        return

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    # csv
    csv_path = os.path.join(run_dir, "confusion_matrix.csv")
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
    fig.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=160)
    plt.close(fig)


def write_report_md(run_dir: str, cfg: Dict, best: Dict, class_counts: Dict[str, Dict[str, int]]):
    lines = []
    lines.append("# Emotion v2 — отчёт обучения\n")
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Time: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n")
    lines.append("## Конфигурация\n")
    lines.append("```json")
    lines.append(json.dumps(cfg, ensure_ascii=False, indent=2))
    lines.append("```\n")

    lines.append("## Дисбаланс классов\n")
    lines.append("```json")
    lines.append(json.dumps(class_counts, ensure_ascii=False, indent=2))
    lines.append("```\n")

    lines.append("## Лучший результат\n")
    lines.append("```json")
    lines.append(json.dumps(best, ensure_ascii=False, indent=2))
    lines.append("```\n")

    lines.append("## Артефакты\n")
    lines.append("- `metrics.csv`, `metrics.jsonl` — метрики по эпохам")
    lines.append("- `plots/loss.png`, `plots/acc.png`, `plots/f1_macro.png`, `plots/lr.png` — графики")
    lines.append("- `confusion_matrix.png`, `confusion_matrix.csv` — confusion matrix (val/test)")
    lines.append("- `topk_epochs.csv` — таблица лучших эпох\n")

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


@dataclass
class TrainConfig:
    data_root: str
    out_root: str
    run_name: str
    epochs: int = 60
    batch_size: int = 128
    img_size: int = 224
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"  # cuda|cpu|auto
    amp: bool = True
    label_smoothing: float = 0.05
    dropout: float = 0.2
    use_class_weights: bool = True
    scheduler: str = "cosine"  # cosine|step|none
    warmup_epochs: int = 2
    topk: int = 10
    patience: int = 12  # early stop по val_acc без улучшения


def build_transforms(img_size: int):
    # FER серые — но ResNet ждёт 3 канала: делаем Grayscale(num_output_channels=3)
    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.10, contrast=0.10)], p=0.25),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
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


def build_model(num_classes: int, dropout: float):
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = m.fc.in_features
    # небольшой dropout перед головой
    m.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_f, num_classes)
    )
    return m


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def macro_f1_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if not SKLEARN_OK:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro"))


def count_images_by_class(root_dir: str, class_names: List[str]) -> Dict[str, int]:
    out = {}
    for c in class_names:
        p = os.path.join(root_dir, c)
        n = 0
        if os.path.isdir(p):
            for f in os.listdir(p):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    n += 1
        out[c] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help=r"C:\vk-vision-demo\data\fer2013")
    ap.add_argument("--out_root", default=r"C:\vk-vision-demo\results\runs_emotion", help="куда сохранять run")
    ap.add_argument("--run_name", default=f"emotion_resnet18_v2_{now_tag()}")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "none"])
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    cfg = TrainConfig(
        data_root=args.data_root,
        out_root=args.out_root,
        run_name=args.run_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        lr=args.lr,
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
    )

    seed_everything(cfg.seed)

    train_dir = os.path.join(cfg.data_root, "train")
    val_dir = os.path.join(cfg.data_root, "test")  # у тебя так устроено
    assert os.path.isdir(train_dir), f"train dir not found: {train_dir}"
    assert os.path.isdir(val_dir), f"test/val dir not found: {val_dir}"

    run_dir = os.path.join(cfg.out_root, cfg.run_name)
    plots_dir = os.path.join(run_dir, "plots")
    ensure_dir(run_dir)
    ensure_dir(plots_dir)

    # device
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    use_amp = (cfg.amp and device.type == "cuda")

    # transforms + datasets
    train_tf, val_tf = build_transforms(cfg.img_size)

    ds_train = datasets.ImageFolder(train_dir, transform=train_tf)
    ds_val = datasets.ImageFolder(val_dir, transform=val_tf)

    # фиксируем ожидаемый порядок классов (чтобы совпадал с сервисом)
    # ImageFolder сортирует по имени папки — у тебя как раз angry..surprise.
    class_names = ds_train.classes
    if class_names != APP_LABELS:
        # не ломаем, но предупреждаем
        print(f"[WARN] folder classes = {class_names}")
        print(f"[WARN] app labels     = {APP_LABELS}")
        # будем использовать фактические папки как truth
    num_classes = len(class_names)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)

    # model
    model = build_model(num_classes=num_classes, dropout=cfg.dropout).to(device)

    # loss
    if cfg.use_class_weights:
        w = compute_class_weights_from_folder(train_dir, class_names).to(device)
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # scheduler (cosine + warmup)
    def lr_at_epoch(e: int):
        # e starts at 1
        if cfg.scheduler == "none":
            return 1.0
        if cfg.scheduler == "step":
            # простая ступень: 0.1 на 40% и 70%
            p = e / max(1, cfg.epochs)
            if p < 0.4: return 1.0
            if p < 0.7: return 0.1
            return 0.01
        # cosine
        return 0.5 * (1.0 + math.cos(math.pi * (e - 1) / max(1, cfg.epochs)))

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # save config
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    # metrics files
    csv_path = os.path.join(run_dir, "metrics.csv")
    jsonl_path = os.path.join(run_dir, "metrics.jsonl")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,val_loss,train_acc,val_acc,train_f1_macro,val_f1_macro,lr\n")

    # class counts for report
    class_counts = {
        "train": count_images_by_class(train_dir, class_names),
        "val": count_images_by_class(val_dir, class_names),
    }

    best_val_acc = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(run_dir, "emotion_resnet18_v2_best.pth")  # в run
    # И отдельно копия в models (как ты хочешь)
    models_dir = os.path.abspath(os.path.join(os.getcwd(), "models"))
    ensure_dir(models_dir)
    best_models_path = os.path.join(models_dir, "emotion_resnet18_v2_best.pth")

    history: List[Dict] = []
    bad_epochs = 0

    print(f"[INFO] device: {device} | amp: {use_amp}")
    print(f"[INFO] train_dir: {train_dir}")
    print(f"[INFO] val_dir  : {val_dir}")
    print(f"[INFO] classes : {class_names}")
    print(f"[INFO] run_dir : {run_dir}")

    for epoch in range(1, cfg.epochs + 1):
        # set LR
        base_mult = lr_at_epoch(epoch)
        # warmup
        if cfg.warmup_epochs > 0 and epoch <= cfg.warmup_epochs:
            warm = epoch / cfg.warmup_epochs
        else:
            warm = 1.0
        cur_lr = cfg.lr * base_mult * warm
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

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

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
            train_f1 = macro_f1_from_preds(np.concatenate(tr_y_true), np.concatenate(tr_y_pred))

        # ---- val ----
        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0
        va_y_true = []
        va_y_pred = []

        with torch.no_grad():
            pbar = tqdm(dl_val, desc=f"val   e{epoch:03d}", leave=False)
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                logits = model(x)
                loss = criterion(logits, y)

                bs = x.size(0)
                va_loss += loss.item() * bs
                va_total += bs

                pred = logits.argmax(dim=1)
                va_correct += (pred == y).sum().item()

                va_y_true.append(y.detach().cpu().numpy())
                va_y_pred.append(pred.detach().cpu().numpy())

        val_loss = va_loss / max(1, va_total)
        val_acc = va_correct / max(1, va_total)

        val_f1 = None
        y_true_np = np.concatenate(va_y_true) if va_y_true else np.array([])
        y_pred_np = np.concatenate(va_y_pred) if va_y_pred else np.array([])
        if SKLEARN_OK and y_true_np.size:
            val_f1 = macro_f1_from_preds(y_true_np, y_pred_np)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "train_f1_macro": float(train_f1) if train_f1 is not None else None,
            "val_f1_macro": float(val_f1) if val_f1 is not None else None,
            "lr": float(cur_lr),
        }
        history.append(row)

        # write metrics
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},{train_acc:.6f},{val_acc:.6f},"
                f"{(train_f1 if train_f1 is not None else '')},{(val_f1 if val_f1 is not None else '')},{cur_lr:.8f}\n"
            )
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # print short
        f1s = f" val_f1={val_f1:.4f}" if (val_f1 is not None and not math.isnan(val_f1)) else ""
        print(f"[epoch {epoch:03d}] val_acc={val_acc:.4f} val_loss={val_loss:.4f}{f1s} lr={cur_lr:.2e}")

        # update plots each epoch
        plot_curves(history, plots_dir)

        # save confusion each epoch (optional heavy). Сделаем только когда улучшили best.
        improved = val_acc > best_val_acc + 1e-6
        if improved:
            best_val_acc = val_acc
            best_epoch = epoch
            bad_epochs = 0

            # save checkpoint in run
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "classes": class_names,
                    "config": asdict(cfg),
                },
                best_ckpt_path,
            )
            # copy to models\emotion_resnet18_v2_best.pth
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "best_val_acc": best_val_acc,
                    "classes": class_names,
                    "config": asdict(cfg),
                },
                best_models_path,
            )

            if SKLEARN_OK and y_true_np.size:
                save_confusion(run_dir, y_true_np, y_pred_np, class_names, title=f"Confusion (best @ epoch {epoch})")

                # classification report text
                rep = classification_report(y_true_np, y_pred_np, target_names=class_names, digits=4)
                with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
                    f.write(rep)

            print(f"[INFO] new best saved -> {best_models_path} (val_acc={best_val_acc:.4f})")
        else:
            bad_epochs += 1

        # early stopping
        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            print(f"[INFO] early stop: no improvement in val_acc for {cfg.patience} epochs")
            break

    # top-k table
    save_topk(history, os.path.join(run_dir, "topk_epochs.csv"), k=cfg.topk)

    # final report.md
    best_summary = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_models_path": best_models_path,
        "run_dir": run_dir,
        "classes": class_names,
    }
    write_report_md(run_dir, asdict(cfg), best_summary, class_counts)

    # summary.json
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(best_summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Run dir: {run_dir}")
    print(f"Best: {best_models_path} (val_acc={best_val_acc:.4f} @ epoch {best_epoch})")


if __name__ == "__main__":
    main()
# eval_emotion_ferplus_v4_only.py
import os
import json
import argparse
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report, f1_score


APP_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
APP_TO_IDX = {c: i for i, c in enumerate(APP_LABELS)}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def list_images(folder: str) -> List[str]:
    out = []
    for dp, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(dp, f))
    return out


class FERPlusMappedDataset(Dataset):
    """
    root/train|validation|test/<class>/*
    FERPlus classes include: angry, contempt, disgust, fear, happy, neutral, sad, suprise/surprise.
    Mapped to APP_LABELS(7):
      contempt -> (disgust|neutral|drop)
      suprise -> surprise
    """
    def __init__(self, root: str, split: str, transform, contempt_to: str = "disgust"):
        assert split in ("train", "validation", "test")
        assert contempt_to in ("disgust", "neutral", "drop")

        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"split dir not found: {split_dir}")

        items: List[Tuple[str, int]] = []

        for class_name in os.listdir(split_dir):
            src = class_name.lower().strip()
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            if src == "suprise":
                src = "surprise"

            tgt: Optional[str]
            if src in APP_TO_IDX:
                tgt = src
            elif src == "contempt":
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

        if not items:
            raise RuntimeError(f"No images collected for {split_dir}. Check folders and extensions.")

        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, y


def build_val_tf(img_size: int):
    # FERPlus grayscale -> 3ch, как в train script
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_model_resnet50(num_classes: int = 7, dropout: float = 0.2):
    m = models.resnet50(weights=None)  # weights не нужны при загрузке чекпоинта
    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_f, num_classes))
    return m


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    ys, ps = [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total += bs

        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())

        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(ps) if ps else np.array([], dtype=np.int64)

    acc = correct / max(1, total)
    loss = total_loss / max(1, total)
    f1m = float(f1_score(y_true, y_pred, average="macro")) if y_true.size else float("nan")
    return loss, acc, f1m, y_true, y_pred


def save_confusion(out_dir: str, y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], prefix: str):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help=r"C:\vk-vision-demo\data\FERPlus")
    ap.add_argument("--ckpt", required=True, help="path to .pth (best_val or best_test)")
    ap.add_argument("--out_dir", required=True, help="куда сохранить результаты оценки (png/csv/txt/json)")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--contempt_to", default="disgust", choices=["disgust", "neutral", "drop"])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--eval_split", default="both", choices=["validation", "test", "both"])
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    payload = torch.load(args.ckpt, map_location="cpu")
    sd = payload["model"] if isinstance(payload, dict) and "model" in payload else payload

    model = build_model_resnet50(num_classes=len(APP_LABELS), dropout=args.dropout)
    model.load_state_dict(sd, strict=True)
    model.to(device)

    tf = build_val_tf(args.img_size)

    criterion = nn.CrossEntropyLoss()

    out: Dict[str, Dict] = {}

    if args.eval_split in ("validation", "both"):
        ds_val = FERPlusMappedDataset(args.data_root, "validation", transform=tf, contempt_to=args.contempt_to)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        v_loss, v_acc, v_f1, y_true, y_pred = evaluate(model, dl_val, device, criterion)
        out["validation"] = {"loss": v_loss, "acc": v_acc, "f1_macro": v_f1, "n": len(ds_val)}
        save_confusion(args.out_dir, y_true, y_pred, APP_LABELS, prefix="val")
        rep = classification_report(y_true, y_pred, target_names=APP_LABELS, digits=4)
        with open(os.path.join(args.out_dir, "val_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(rep)

    if args.eval_split in ("test", "both"):
        ds_test = FERPlusMappedDataset(args.data_root, "test", transform=tf, contempt_to=args.contempt_to)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
        t_loss, t_acc, t_f1, y_true, y_pred = evaluate(model, dl_test, device, criterion)
        out["test"] = {"loss": t_loss, "acc": t_acc, "f1_macro": t_f1, "n": len(ds_test)}
        save_confusion(args.out_dir, y_true, y_pred, APP_LABELS, prefix="test")
        rep = classification_report(y_true, y_pred, target_names=APP_LABELS, digits=4)
        with open(os.path.join(args.out_dir, "test_classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(rep)

    out["ckpt"] = os.path.abspath(args.ckpt)
    out["device"] = str(device)
    out["contempt_to"] = args.contempt_to
    out["img_size"] = args.img_size

    with open(os.path.join(args.out_dir, "eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\nEVAL DONE")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
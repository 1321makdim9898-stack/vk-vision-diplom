import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# ===== Настройки =====

DATA_ROOT = Path("data/UTKFace/utkface_aligned_cropped")  # при необходимости поменяй
PATTERN = "**/*.jpg"  # рекурсивно по всем подпапкам

AGE_BINS = [(0, 12), (13, 25), (26, 40), (41, 60), (61, 120)]
AGE_BIN_LABELS = ["0-12", "13-25", "26-40", "41-60", "61+"]

BATCH_SIZE = 64
NUM_EPOCHS = 25
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = Path("models")
SAVE_DIR.mkdir(exist_ok=True)
MODEL_PATH = SAVE_DIR / "age_gender_resnet18.pth"


@dataclass
class UtkSample:
    path: str
    age: int
    gender: int  # 0 or 1
    age_bin: int


def age_to_bin(age: int) -> int:
    for i, (a1, a2) in enumerate(AGE_BINS):
        if a1 <= age <= a2:
            return i
    return len(AGE_BINS) - 1


def parse_utk_filename(fname: str) -> Tuple[int, int]:
    """
    Имя вида: age_gender_race_date.jpg.chip.jpg
    Нас интересуют только age и gender.
    """
    base = os.path.basename(fname)
    parts = base.split("_")
    age = int(parts[0])
    gender = int(parts[1])
    return age, gender


class UtkFaceDataset(Dataset):
    def __init__(self, samples: List[UtkSample], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample.path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, sample.age_bin, sample.gender


# ===== Аугментации =====

train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ===== Модель (ResNet18 + две головы) =====

class AgeGenderResNet(nn.Module):
    def __init__(self, num_age_bins: int, num_genders: int = 2):
        super().__init__()
        backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()  # будем использовать как feature extractor
        self.backbone = backbone
        self.head_age = nn.Linear(in_features, num_age_bins)
        self.head_gender = nn.Linear(in_features, num_genders)

    def forward(self, x):
        feats = self.backbone(x)
        age_logits = self.head_age(feats)
        gender_logits = self.head_gender(feats)
        return age_logits, gender_logits


# ===== Подготовка данных =====

def load_all_samples() -> List[UtkSample]:
    pattern = str(DATA_ROOT / PATTERN)
    paths = glob.glob(pattern, recursive=True)
    samples: List[UtkSample] = []

    for p in paths:
        try:
            age, gender = parse_utk_filename(p)
        except Exception:
            continue
        if gender not in (0, 1):
            continue
        bin_idx = age_to_bin(age)
        samples.append(UtkSample(path=p, age=age, gender=gender, age_bin=bin_idx))

    print(f"[INFO] Найдено образцов UTKFace: {len(samples)}")
    return samples


def make_loaders():
    samples = load_all_samples()
    rng = np.random.default_rng(42)
    rng.shuffle(samples)

    val_size = int(0.2 * len(samples))
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]

    train_ds = UtkFaceDataset(train_samples, transform=train_transform)
    val_ds = UtkFaceDataset(val_samples, transform=val_transform)

    print(f"[INFO] train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    return train_loader, val_loader


# ===== Обучение =====

def train_one_epoch(model, loader, criterion_age, criterion_gender, optimizer):
    model.train()
    total = 0
    running_loss = 0.0
    correct_age = 0
    correct_gender = 0

    for images, age_bins, genders in tqdm(loader, desc="Train", leave=False):
        images = images.to(DEVICE)
        age_bins = age_bins.to(DEVICE)
        genders = genders.to(DEVICE)

        optimizer.zero_grad()
        age_logits, gender_logits = model(images)

        loss_age = criterion_age(age_logits, age_bins)
        loss_gender = criterion_gender(gender_logits, genders)

        loss = loss_age + loss_gender  # можно добавить веса, если нужно
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        _, age_pred = age_logits.max(1)
        _, gender_pred = gender_logits.max(1)

        correct_age += age_pred.eq(age_bins).sum().item()
        correct_gender += gender_pred.eq(genders).sum().item()

    return (
        running_loss / total,
        correct_age / total,
        correct_gender / total,
    )


def eval_one_epoch(model, loader, criterion_age, criterion_gender):
    model.eval()
    total = 0
    running_loss = 0.0
    correct_age = 0
    correct_gender = 0

    with torch.inference_mode():
        for images, age_bins, genders in tqdm(loader, desc="Val", leave=False):
            images = images.to(DEVICE)
            age_bins = age_bins.to(DEVICE)
            genders = genders.to(DEVICE)

            age_logits, gender_logits = model(images)

            loss_age = criterion_age(age_logits, age_bins)
            loss_gender = criterion_gender(gender_logits, genders)
            loss = loss_age + loss_gender

            running_loss += loss.item() * images.size(0)
            total += images.size(0)

            _, age_pred = age_logits.max(1)
            _, gender_pred = gender_logits.max(1)
            correct_age += age_pred.eq(age_bins).sum().item()
            correct_gender += gender_pred.eq(genders).sum().item()

    return (
        running_loss / total,
        correct_age / total,
        correct_gender / total,
    )


def main():
    print("[INFO] Используем устройство:", DEVICE)
    train_loader, val_loader = make_loaders()

    model = AgeGenderResNet(num_age_bins=len(AGE_BINS)).to(DEVICE)

    criterion_age = nn.CrossEntropyLoss()
    criterion_gender = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    best_age_acc = 0.0
    best_gender_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nЭпоха {epoch}/{NUM_EPOCHS}")

        train_loss, train_age_acc, train_gender_acc = train_one_epoch(
            model, train_loader, criterion_age, criterion_gender, optimizer
        )
        val_loss, val_age_acc, val_gender_acc = eval_one_epoch(
            model, val_loader, criterion_age, criterion_gender
        )
        scheduler.step()

        print(
            f"Train: loss={train_loss:.4f}, "
            f"age_acc={train_age_acc:.4f}, gender_acc={train_gender_acc:.4f} | "
            f"Val: loss={val_loss:.4f}, "
            f"age_acc={val_age_acc:.4f}, gender_acc={val_gender_acc:.4f}"
        )

        # сохраняем по gender_acc и age_acc одновременно
        score = (val_age_acc + val_gender_acc) / 2
        best_score = (best_age_acc + best_gender_acc) / 2
        if score > best_score:
            best_age_acc = val_age_acc
            best_gender_acc = val_gender_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[INFO] Новая лучшая модель сохранена в {MODEL_PATH}")

    print(
        f"\nЛучшие метрики: age_acc={best_age_acc:.4f}, "
        f"gender_acc={best_gender_acc:.4f}"
    )


if __name__ == "__main__":
    main()

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ===== Настройки ПОД ТВОЙ ПУТЬ =====

# Важно: у тебя data/fer2013 (с маленькими буквами)
DATA_ROOT = Path("data/fer2013")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "test"   # используем test как validation

NUM_CLASSES = 7  # angry, disgust, fear, happy, neutral, sad, surprise
BATCH_SIZE = 64
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = Path("models")
SAVE_DIR.mkdir(exist_ok=True)
MODEL_PATH = SAVE_DIR / "emotion_resnet18.pth"

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ===== Датасеты и аугментация =====

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet
        std=[0.229, 0.224, 0.225],
    ),
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def make_loaders():
    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transform)
    val_ds = datasets.ImageFolder(str(VAL_DIR), transform=val_transform)

    print("[INFO] Классы train:", train_ds.classes)
    if train_ds.classes != EMOTION_CLASSES:
        print("[WARN] Порядок классов в папках отличается от EMOTION_CLASSES.")
        print("       Текущие классы:", train_ds.classes)

    # num_workers=0 — безопасно для Windows; если всё ок, можно поднять до 2–4
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader


# ===== Модель =====

def create_model():
    # предобученный ResNet18
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, NUM_CLASSES)
    return backbone


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def eval_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Val", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    print("[INFO] Используем устройство:", DEVICE)
    print("[INFO] TRAIN_DIR:", TRAIN_DIR)
    print("[INFO] VAL_DIR:", VAL_DIR)

    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(f"Не найдена папка {TRAIN_DIR}")
    if not VAL_DIR.is_dir():
        raise FileNotFoundError(f"Не найдена папка {VAL_DIR}")

    train_loader, val_loader = make_loaders()

    model = create_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    best_val_acc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nЭпоха {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion
        )
        scheduler.step()

        print(
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"[INFO] Новая лучшая модель сохранена в {MODEL_PATH}")

    print(f"\nЛучшее val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

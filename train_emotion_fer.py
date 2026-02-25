# train_emotion_fer.py
"""
Обучение модели эмоций на FER-2013 (папки train/val или train/test).
Ожидается структура:
data/fer2013/
    train/
        angry/
        disgust/
        ...
    val/   (или test/)
"""

import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# поменяй путь при необходимости
DATA_ROOT = os.path.join("data", "fer2013")
BATCH_SIZE = 128
NUM_EPOCHS = 30
LR = 3e-4
VAL_SPLIT = 0.15  # если нет отдельной val-папки, мы отделим от train


class EmotionCNN(nn.Module):
    """
    Более глубокая CNN для 48x48 grayscale:
    Conv-BN-ReLU x3 + GlobalAvgPool + Dropout.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # блок 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 24x24
            nn.Dropout(0.25),

            # блок 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12x12
            nn.Dropout(0.25),

            # блок 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 6x6
            nn.Dropout(0.25),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.fc(x)
        return x


def get_dataloaders() -> Tuple[DataLoader, DataLoader, List[str]]:
    # аугментации только для train
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.Pad(4),
            transforms.RandomCrop((48, 48)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ]
    )

    # если внутри DATA_ROOT уже есть train/val
    if os.path.isdir(os.path.join(DATA_ROOT, "train")):
        full_train = datasets.ImageFolder(
            os.path.join(DATA_ROOT, "train"), transform=train_transform
        )
        classes = full_train.classes

        # пытаемся взять отдельную val-папку, если есть
        val_dir = os.path.join(DATA_ROOT, "val")
        test_dir = os.path.join(DATA_ROOT, "test")

        if os.path.isdir(val_dir):
            val_ds = datasets.ImageFolder(val_dir, transform=val_transform)
        elif os.path.isdir(test_dir):
            val_ds = datasets.ImageFolder(test_dir, transform=val_transform)
        else:
            # нет отдельной валидации -> делим train
            val_size = int(len(full_train) * VAL_SPLIT)
            train_size = len(full_train) - val_size
            full_train, val_ds = random_split(full_train, [train_size, val_size])

        train_loader = DataLoader(
            full_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

    else:
        # если структура другая, кидаем понятную ошибку
        raise FileNotFoundError(
            f"Не найдена папка train внутри {DATA_ROOT}. "
            f"Ожидается data/fer2013/train/... "
        )

    print("Найдены классы:", classes)
    return train_loader, val_loader, classes


def main():
    train_loader, val_loader, classes = get_dataloaders()
    num_classes = len(classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)

    model = EmotionCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        scheduler.step()
        train_loss = total_loss / max(1, total_samples)

        # eval
        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += imgs.size(0)

        val_acc = correct / max(1, total)
        print(
            f"Эпоха {epoch+1}/{NUM_EPOCHS}: "
            f"train_loss={train_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            save_path = os.path.join("models", "emotion_cnn.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Новая лучшая модель сохранена в {save_path}")

    print("Лучшее val_acc:", best_val_acc)


if __name__ == "__main__":
    main()

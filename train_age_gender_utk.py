# train_age_gender_utk.py

import os
from glob import glob
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

DATA_ROOT = os.path.join("data", "UTKFace", "utkface_aligned_cropped", "UTKFace")

AGE_BINS: List[Tuple[int, int]] = [
    (0, 12),
    (13, 25),
    (26, 40),
    (41, 60),
    (61, 116),
]
AGE_BIN_LABELS: List[str] = ["0-12", "13-25", "26-40", "41-60", "61+"]


def age_to_bin(age: int) -> int:
    for i, (lo, hi) in enumerate(AGE_BINS):
        if lo <= age <= hi:
            return i
    return len(AGE_BINS) - 1


class UtkFaceAgeGenderDataset(Dataset):
    def __init__(self, root: str, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

        exts = ["*.jpg", "*.jpeg", "*.png"]
        files = []
        for ext in exts:
            pattern = os.path.join(root, "**", ext)
            files.extend(glob(pattern, recursive=True))

        self.files = sorted(files)

        if not self.files:
            raise FileNotFoundError(
                "Не найдено ни одного изображения в "
                f"{root}. Проверь путь и структуру датасета UTKFace."
            )

        self.samples = []
        for path in self.files:
            fname = os.path.basename(path)
            parts = fname.split("_")
            if len(parts) < 4:
                continue
            try:
                age = int(parts[0])
                gender = int(parts[1])  # 0/1
            except ValueError:
                continue

            age_bin = age_to_bin(age)
            self.samples.append((path, age_bin, gender))

        if not self.samples:
            raise RuntimeError("Не удалось распарсить ни одного файла UTKFace.")

        print(f"Загружено образцов UTKFace: {len(self.samples)}")
        print("Пример файла:", os.path.basename(self.samples[0][0]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, age_bin, gender = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        age_tensor = torch.tensor(age_bin, dtype=torch.long)
        gender_tensor = torch.tensor(gender, dtype=torch.long)
        return img, age_tensor, gender_tensor


class AgeGenderCNN(nn.Module):
    def __init__(self, num_age_bins: int = 5, num_genders: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Dropout(0.25),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.fc_age = nn.Linear(128, num_age_bins)
        self.fc_gender = nn.Linear(128, num_genders)

    def forward(self, x):
        x = self.features(x)
        x = self.fc_shared(x)
        age_logits = self.fc_age(x)
        gender_logits = self.fc_gender(x)
        return age_logits, gender_logits


def main():
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]
    )

    dataset = UtkFaceAgeGenderDataset(DATA_ROOT, transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)

    model = AgeGenderCNN(num_age_bins=len(AGE_BIN_LABELS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    num_epochs = 20  # можно ещё увеличить

    best_val_gender = 0.0
    best_val_age = 0.0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for imgs, age_labels, gender_labels in train_loader:
            imgs = imgs.to(device)
            age_labels = age_labels.to(device)
            gender_labels = gender_labels.to(device)

            optimizer.zero_grad()
            age_logits, gender_logits = model(imgs)

            loss_age = criterion(age_logits, age_labels)
            loss_gender = criterion(gender_logits, gender_labels)

            # немного больше веса на возраст
            loss = 1.5 * loss_age + loss_gender

            loss.backward()
            optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

        scheduler.step()
        train_loss = total_loss / max(1, total_samples)

        model.eval()
        correct_age = 0
        correct_gender = 0
        total = 0

        with torch.inference_mode():
            for imgs, age_labels, gender_labels in val_loader:
                imgs = imgs.to(device)
                age_labels = age_labels.to(device)
                gender_labels = gender_labels.to(device)

                age_logits, gender_logits = model(imgs)
                age_pred = age_logits.argmax(dim=1)
                gender_pred = gender_logits.argmax(dim=1)

                correct_age += (age_pred == age_labels).sum().item()
                correct_gender += (gender_pred == gender_labels).sum().item()
                total += imgs.size(0)

        age_acc = correct_age / max(1, total)
        gender_acc = correct_gender / max(1, total)

        print(
            f"Эпоха {epoch + 1}/{num_epochs}: "
            f"train_loss={train_loss:.4f}, val_age_acc={age_acc:.4f}, "
            f"val_gender_acc={gender_acc:.4f}"
        )

        # сохраняем лучшую модель по средней метрике
        if (age_acc + gender_acc) / 2 > (best_val_age + best_val_gender) / 2:
            best_val_age = age_acc
            best_val_gender = gender_acc
            os.makedirs("models", exist_ok=True)
            save_path = os.path.join("models", "age_gender_cnn.pth")
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Новая лучшая модель сохранена в {save_path}")

    print("Лучшие метрики: age_acc=", best_val_age, "gender_acc=", best_val_gender)


if __name__ == "__main__":
    main()

# Отчёт по обучению моделей возраст/пол (UTKFace)

## 1. Данные и препроцессинг

**Датасет:** UTKFace  
**Формат:** crops лиц (MediaPipe)  
**Папка с данными:** `C:\vk-vision-demo\data\UTKFace\utkface_mediapipe_cropped` *(пример)*

**Сплит:** train/val/test = 85% / 10% / 5% (детерминированный shuffle по seed)

**Бины возраста (для метрик bin_acc и confusion matrix):**
- 0–12
- 13–25
- 26–40
- 41–60
- 61+

## 2. Модели

### 2.1 DLDL (основная)
- Backbone: ResNet50 (ImageNet pretrained)
- Возраст: распределение по 121 классам (0..120), целевая метка — Gaussian вокруг истинного возраста
- Пол: бинарная классификация (BCE)
- Итоговый loss: `KLDiv(age_dist) + λ * BCE(gender)` (λ = 0.35)

**Артефакты обучения:**
- `metrics.csv`, `metrics.jsonl`
- `plots/loss.png`, `plots/mae.png`, `plots/bin_acc.png`, `plots/gender_acc.png`
- `plots/confusion_bins_eXX.png` (confusion matrix по возрастным бинам на каждой эпохе)
- `topk_epochs.csv` (лучшие эпохи)
- `*_best.pth` (лучший чекпоинт по val_mae)

### 2.2 Regression (fallback)
- Backbone: ResNet18 (ImageNet pretrained)
- Возраст: регрессия (L1)
- Пол: бинарная классификация (BCE)
- Итоговый loss: `L1(age) + λ * BCE(gender)` (λ = 0.35)

Артефакты аналогичны DLDL.

## 3. Параметры запуска

### 3.1 DLDL (ResNet50)
```bash
python train_age_gender_utk_dldl_resnet50_v2.py ^
  --data_root "C:\vk-vision-demo\data\UTKFace\utkface_mediapipe_cropped" ^
  --out_root "C:\vk-vision-demo\runs_age_gender" ^
  --run_name "dldl_resnet50_v2_run01" ^
  --epochs 30 --batch_size 32 --img_size 224 --lr 0.0003 --weight_decay 0.0001 ^
  --sigma 2.0 --lambda_gender 0.35 --seed 42
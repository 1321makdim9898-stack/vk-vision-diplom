import base64
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Any, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import torchvision

# feat (Py-Feat) для AU
try:
    from feat import Detector  # type: ignore
except Exception:
    Detector = None

# MediaPipe
try:
    import mediapipe as mp  # type: ignore
except Exception:
    mp = None

# ===== Константы =====

EMOTION_CLASSES: List[str] = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

AGE_BIN_LABELS: List[str] = ["0-12", "13-25", "26-40", "41-60", "61+"]

AU_CLASSES = [
    "AU01",
    "AU02",
    "AU04",
    "AU05",
    "AU06",
    "AU07",
    "AU09",
    "AU10",
    "AU12",
    "AU14",
    "AU15",
    "AU17",
    "AU20",
    "AU23",
    "AU24",
    "AU25",
    "AU26",
]

AU_DESCRIPTIONS: Dict[str, str] = {
    "AU01": "поднятие внутренней части бровей",
    "AU02": "поднятие внешней части бровей",
    "AU04": "сведение бровей",
    "AU05": "поднятие верхнего века",
    "AU06": "напряжение щёк (улыбка глазами)",
    "AU07": "напряжение нижнего века",
    "AU09": "морщение носа",
    "AU10": "поднятие верхней губы",
    "AU12": "растягивание уголков губ (улыбка)",
    "AU14": "асимметричная улыбка/скепсис",
    "AU15": "опускание уголков губ",
    "AU17": "выдвижение подбородка",
    "AU20": "растягивание губ горизонтально",
    "AU23": "напряжение губ",
    "AU24": "сжатие губ",
    "AU25": "открытие рта",
    "AU26": "широкое открытие рта",
}

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# История: (base64-картинка, подпись)
history_entries: List[Tuple[str, str]] = []


# ===== МОДЕЛИ =====

# --- Emotion (FER2013) ---
# Новый вариант: ResNet18 (train_emotion_fer_resnet.py) -> models/emotion_resnet18.pth
# Фоллбек: простой CNN (старый) -> models/emotion_cnn.pth

FER_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def _strip_prefix_from_state_dict(state_dict: dict, prefix: str) -> dict:
    if not prefix:
        return state_dict
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out

def _infer_in_channels_from_resnet_state_dict(state_dict: dict) -> int:
    # resnet conv1 weight: [64, C, 7, 7]
    w = state_dict.get("conv1.weight")
    if w is None:
        # иногда сохраняют как backbone.conv1.weight
        w = state_dict.get("backbone.conv1.weight")
    if w is None:
        return 3
    return int(w.shape[1])

def build_resnet18_for_classification(num_classes: int, in_channels: int = 3) -> nn.Module:
    model = torchvision.models.resnet18(weights=None)
    if in_channels != 3:
        # заменить первый conv под число каналов
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class SimpleEmotionCNN(nn.Module):
    """Старый (упрощённый) вариант. Оставляем на всякий случай как fallback."""
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


_emotion_model: Optional[nn.Module] = None
_emotion_backend: str = "none"   # "resnet18" | "simple_cnn" | "none"

# --- Age/Gender (UTKFace) ---
# Новый вариант: ResNet18 multi-head (train_age_gender_utk_resnet.py) -> models/age_gender_resnet18.pth
# Фоллбек: старый CNN -> models/age_gender_cnn.pth

AGE_BIN_LABELS = ["0-12", "13-19", "20-29", "30-39", "40-49", "50-59", "60+"]

class AgeGenderCNN(nn.Module):
    def __init__(self, num_age_bins: int = 7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.age_head = nn.Linear(64, num_age_bins)
        self.gender_head = nn.Linear(64, 2)  # 0=female, 1=male

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        age_logits = self.age_head(x)
        gender_logits = self.gender_head(x)
        return age_logits, gender_logits


class AgeGenderResNet18(nn.Module):
    """ResNet18 + две головы: возраст (bin) и пол.
    Предполагаемый формат чекпоинта: state_dict этой модели (backbone.* / age_head.* / gender_head.*)
    или state_dict голого resnet (conv1/layer*/fc) — тогда загрузка не получится, но модель ещё не обучена,
    поэтому просто мягко отключим блок возраст/пол.
    """
    def __init__(self, num_age_bins: int = 7, in_channels: int = 3):
        super().__init__()
        self.backbone = torchvision.models.resnet18(weights=None)
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.age_head = nn.Linear(feat_dim, num_age_bins)
        self.gender_head = nn.Linear(feat_dim, 2)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        age_logits = self.age_head(feats)
        gender_logits = self.gender_head(feats)
        return age_logits, gender_logits


_age_gender_model: Optional[nn.Module] = None
_age_gender_backend: str = "none"   # "resnet18" | "simple_cnn" | "none"


def load_models():
    """Загружает модели с приоритетом на ResNet18 версии.
    Если resnet-веса ещё не готовы/отсутствуют — используем старые веса (если есть),
    иначе работаем без соответствующей подсистемы.
    """
    global _emotion_model, _emotion_backend, _age_gender_model, _age_gender_backend

    device = _device

    # --- EMOTION ---
    _emotion_model = None
    _emotion_backend = "none"

    # 1) emotion_resnet18.pth
    try:
        p = Path("models/emotion_resnet18.pth")
        if p.exists():
            sd = torch.load(p, map_location=device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # если вдруг сохранено как backbone.*
            sd = _strip_prefix_from_state_dict(sd, "backbone.")
            in_ch = _infer_in_channels_from_resnet_state_dict(sd)
            m = build_resnet18_for_classification(num_classes=len(FER_EMOTIONS), in_channels=in_ch)
            missing, unexpected = m.load_state_dict(sd, strict=False)
            m.to(device).eval()
            _emotion_model = m
            _emotion_backend = "resnet18"
            print(f"[INFO] Emotion model: ResNet18 ({p}) | in_channels={in_ch} | missing={len(missing)} unexpected={len(unexpected)}")
    except Exception as e:
        print(f"[WARN] Не удалось загрузить models/emotion_resnet18.pth: {e}")

    # 2) fallback emotion_cnn.pth
    if _emotion_model is None:
        try:
            p = Path("models/emotion_cnn.pth")
            if p.exists():
                m = SimpleEmotionCNN(num_classes=len(FER_EMOTIONS))
                sd = torch.load(p, map_location=device)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                m.load_state_dict(sd, strict=True)
                m.to(device).eval()
                _emotion_model = m
                _emotion_backend = "simple_cnn"
                print(f"[INFO] Emotion model: SimpleEmotionCNN ({p})")
        except Exception as e:
            print(f"[WARN] Не удалось загрузить models/emotion_cnn.pth: {e}")

    # --- AGE/GENDER ---
    _age_gender_model = None
    _age_gender_backend = "none"

    # 1) age_gender_resnet18.pth (может появиться позже)
    try:
        p = Path("models/age_gender_resnet18.pth")
        if p.exists():
            sd = torch.load(p, map_location=device)
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # пытаемся понять, это state_dict нашей multi-head модели или чего-то другого
            # если есть backbone.* -> наша, если нет — попробуем загрузить как есть (вдруг совпало)
            in_ch = 3
            if "backbone.conv1.weight" in sd:
                in_ch = int(sd["backbone.conv1.weight"].shape[1])
            m = AgeGenderResNet18(num_age_bins=len(AGE_BIN_LABELS), in_channels=in_ch)
            missing, unexpected = m.load_state_dict(sd, strict=False)
            m.to(device).eval()
            _age_gender_model = m
            _age_gender_backend = "resnet18"
            print(f"[INFO] Age/Gender model: ResNet18-multihead ({p}) | missing={len(missing)} unexpected={len(unexpected)}")
    except Exception as e:
        print(f"[WARN] Не удалось загрузить models/age_gender_resnet18.pth: {e}")

    # 2) fallback age_gender_cnn.pth
    if _age_gender_model is None:
        try:
            p = Path("models/age_gender_cnn.pth")
            if p.exists():
                m = AgeGenderCNN(num_age_bins=len(AGE_BIN_LABELS))
                sd = torch.load(p, map_location=device)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                m.load_state_dict(sd, strict=True)
                m.to(device).eval()
                _age_gender_model = m
                _age_gender_backend = "simple_cnn"
                print(f"[INFO] Age/Gender model: AgeGenderCNN ({p})")
        except Exception as e:
            print(f"[WARN] Не удалось загрузить models/age_gender_cnn.pth: {e}")

    _use_trained_emotion_model = _emotion_model is not None
    if _emotion_model is None:
        print("[WARN] Emotion model: НЕ загружена (будет отключено распознавание эмоций).")
    _use_age_gender_model = _age_gender_model is not None
    if _age_gender_model is None:
        print("[WARN] Age/Gender model: НЕ загружена (возраст/пол будут эвристикой или пустыми значениями).")


# Флаги совместимости со старым кодом
_use_trained_emotion_model = False
_use_age_gender_model = False

# ===== Детекторы лиц =====

class OpenCVHaarFaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def detect(self, bgr_img) -> List[FaceBox]:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        h, w = bgr_img.shape[:2]
        results: List[FaceBox] = []
        for (x, y, fw, fh) in faces:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w - 1, int(x + fw)), min(h - 1, int(y + fh))
            results.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, score=1.0))
        return results


class MediapipeFaceDetector:
    def __init__(self):
        if mp is None:
            raise RuntimeError("mediapipe не установлен")
        self.fd = mp.solutions.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        )

    def detect(self, bgr_img) -> List[FaceBox]:
        h, w = bgr_img.shape[:2]
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        res = self.fd.process(rgb)

        results: List[FaceBox] = []
        if not res.detections:
            return results

        img_area = h * w
        for det in res.detections:
            score = float(det.score[0]) if det.score else 0.0
            box = det.location_data.relative_bounding_box
            x1 = int(box.xmin * w)
            y1 = int(box.ymin * h)
            x2 = int((box.xmin + box.width) * w)
            y2 = int((box.ymin + box.height) * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            area = (x2 - x1) * (y2 - y1)
            if score < 0.5 or area < 0.01 * img_area:
                continue

            results.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, score=score))

        return results


# ===== Эмоции =====

def dummy_emotion_predict(face_crop) -> Dict[str, float]:
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    m = float(np.mean(gray)) / 255.0
    probs = np.array(
        [0.05, 0.05, 0.05, 0.1 + 0.2 * m, 0.1, 0.05, 0.6 + 0.2 * (1 - m)],
        dtype=np.float32,
    )
    probs = np.clip(probs, 1e-3, 1.0)
    probs = probs / probs.sum()
    return {EMOTION_CLASSES[i]: float(probs[i]) for i in range(len(EMOTION_CLASSES))}


def emotion_predict_face_crop(face_crop_bgr) -> Optional[dict]:
    """Возвращает словарь вероятностей эмоций для вырезанного лица (BGR).
    Поддерживает 2 бэкенда:
      - ResNet18: models/emotion_resnet18.pth
      - fallback SimpleEmotionCNN: models/emotion_cnn.pth
    """
    if not _use_trained_emotion_model or _emotion_model is None:
        return None

    with torch.inference_mode():
        if _emotion_backend == "resnet18":
            # определяем число каналов по conv1
            in_ch = 3
            try:
                in_ch = int(getattr(_emotion_model, "conv1").in_channels)
            except Exception:
                pass

            if in_ch == 1:
                gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(gray, (224, 224))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)  # (1, H, W)
            else:
                rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
                img = cv2.resize(rgb, (224, 224))
                img = img.astype("float32") / 255.0
                img = np.transpose(img, (2, 0, 1))  # (3, H, W)

            x = torch.tensor(img, dtype=torch.float32, device=_device).unsqueeze(0)  # (1,C,H,W)
            logits = _emotion_model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        else:
            # старый CNN (1 канал, 48x48)
            gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(gray, (48, 48))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)   # (1,H,W)
            img = np.expand_dims(img, axis=0)   # (1,1,H,W)
            tensor = torch.tensor(img, dtype=torch.float32, device=_device)
            logits = _emotion_model(tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    probs_dict = {FER_EMOTIONS[i]: float(probs[i]) for i in range(len(FER_EMOTIONS))}
    return probs_dict

def age_gender_predict_face_crop(face_crop_bgr) -> (Optional[str], Optional[str]):
    """Возраст/пол по лицу (BGR). Если модель не загружена — вернёт (None, None)."""
    if not _use_age_gender_model or _age_gender_model is None:
        return None, None

    with torch.inference_mode():
        if _age_gender_backend == "resnet18":
            # 224x224, 3 канала
            rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (224, 224)).astype("float32") / 255.0
            img = np.transpose(img, (2, 0, 1))  # (3,H,W)
            x = torch.tensor(img, dtype=torch.float32, device=_device).unsqueeze(0)
            age_logits, gender_logits = _age_gender_model(x)
        else:
            # старый CNN (64x64)
            rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
            img = cv2.resize(rgb, (64, 64)).astype("float32") / 255.0
            img = np.transpose(img, (2, 0, 1))
            x = torch.tensor(img, dtype=torch.float32, device=_device).unsqueeze(0)
            age_logits, gender_logits = _age_gender_model(x)

        age_probs = torch.softmax(age_logits, dim=1)[0].cpu().numpy()
        gender_probs = torch.softmax(gender_logits, dim=1)[0].cpu().numpy()

    age_idx = int(np.argmax(age_probs))
    gender_idx = int(np.argmax(gender_probs))

    age_group = AGE_BIN_LABELS[age_idx] if 0 <= age_idx < len(AGE_BIN_LABELS) else None

    # gender_head: 0=female, 1=male
    conf = float(gender_probs[gender_idx])
    base_gender = "female" if gender_idx == 0 else "male"
    gender = f"{base_gender} (?)" if conf < 0.6 else base_gender

    return age_group, gender

def dummy_au_predict(face_crop) -> Dict[str, float]:
    h, w = face_crop.shape[:2]
    rng = np.random.default_rng(h * 1000 + w)
    probs = rng.random(len(AU_CLASSES)) * 0.6
    return {AU_CLASSES[i]: float(probs[i]) for i in range(len(AU_CLASSES))}


def au_predict_face_crop(face_crop_bgr) -> Dict[str, float]:
    if not _use_au_model or _au_detector is None:
        return dummy_au_predict(face_crop_bgr)

    rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    try:
        res = _au_detector.detect_image(rgb)
        if res.aus is None or res.aus.shape[0] == 0:
            return {}
        aus_series = res.aus.iloc[0]
        return {k: float(v) for k, v in aus_series.to_dict().items()}
    except Exception as e:
        print("[WARN] Ошибка Py-Feat AU, fallback на заглушку:", e)
        return dummy_au_predict(face_crop_bgr)


# ===== Телосложение (MediaPipe Pose) =====

def analyze_body(bgr_img) -> Optional[BodyResult]:
    if not _use_body_model or _pose_detector is None or mp_pose is None:
        return None

    h, w, _ = bgr_img.shape
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    try:
        res = _pose_detector.process(rgb)
    except Exception as e:
        print("[WARN] Ошибка MediaPipe Pose:", e)
        return None

    if res.pose_landmarks is None:
        return None

    lms = res.pose_landmarks.landmark
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    body_width = max_x - min_x
    body_height = max_y - min_y

    if body_height <= 0 or body_width <= 0:
        return None

    thickness_ratio = body_width / body_height  # ширина/рост

    try:
        ls = lms[mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lms[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = lms[mp_pose.PoseLandmark.LEFT_HIP]
        rh = lms[mp_pose.PoseLandmark.RIGHT_HIP]

        sx1, sy1 = ls.x * w, ls.y * h
        sx2, sy2 = rs.x * w, rs.y * h
        hx1, hy1 = lh.x * w, lh.y * h
        hx2, hy2 = rh.x * w, rh.y * h

        shoulder_width = float(np.hypot(sx2 - sx1, sy2 - sy1))
        hip_width = float(np.hypot(hx2 - hx1, hy2 - hy1))
        shoulder_hip_ratio = shoulder_width / hip_width if hip_width > 1e-3 else 1.0
    except Exception:
        shoulder_hip_ratio = 1.0

    if thickness_ratio < 0.25:
        slim, avg, large = 0.8, 0.15, 0.05
        base_cat = "худощавое телосложение"
    elif thickness_ratio < 0.35:
        slim, avg, large = 0.2, 0.7, 0.1
        base_cat = "нормальное телосложение"
    else:
        slim, avg, large = 0.1, 0.2, 0.7
        base_cat = "крупное телосложение"

    add = ""
    if shoulder_hip_ratio > 1.1:
        add = " (широкие плечи относительно бёдер)"
    elif shoulder_hip_ratio < 0.9:
        add = " (широкие бёдра относительно плеч)"

    category = base_cat + add

    scores = {
        "slim": float(slim),
        "average": float(avg),
        "large": float(large),
        "thickness_ratio": float(thickness_ratio),
        "shoulder_hip_ratio": float(shoulder_hip_ratio),
    }

    return BodyResult(category=category, scores=scores)


def summarize_body(body: Optional[BodyResult]) -> str:
    if body is None:
        if _use_body_model:
            return "Телосложение не определено (не удалось надёжно найти позу человека)."
        else:
            return (
                "Модуль анализа телосложения недоступен. "
                "Установите `mediapipe`, чтобы активировать эту функцию."
            )

    parts = [f"**Тип телосложения:** {body.category}."]
    slim = body.scores.get("slim")
    avg = body.scores.get("average")
    large = body.scores.get("large")

    if all(v is not None for v in [slim, avg, large]):
        parts.append(
            f"\n- Вероятности (эвристика): худощавый — {slim:.2f}, "
            f"нормальный — {avg:.2f}, крупный — {large:.2f}."
        )

    thr = body.scores.get("thickness_ratio")
    shr = body.scores.get("shoulder_hip_ratio")
    if thr is not None and shr is not None:
        parts.append(
            f"\n- Отношение ширины к росту: {thr:.2f}; "
            f"отношение ширины плеч к бёдрам: {shr:.2f}."
        )

    return "".join(parts)


# ===== Пайплайн =====

class AnalyzerPipeline:
    def __init__(self):
        try:
            self.face_detector = MediapipeFaceDetector()
            self.face_detector_name = "mediapipe_face_detection"
            print("[INFO] Используется детектор лиц MediaPipe")
        except Exception as e:
            print("[WARN] MediaPipe FaceDetection недоступен, fallback на Haar:", e)
            self.face_detector = OpenCVHaarFaceDetector()
            self.face_detector_name = "opencv_haar"

    def analyze(self, bgr_img: np.ndarray) -> AnalysisResult:
        used_fallback = False

        faces = self.face_detector.detect(bgr_img)
        if len(faces) == 0 and self.face_detector_name == "mediapipe_face_detection":
            haar = OpenCVHaarFaceDetector()
            faces = haar.detect(bgr_img)
            used_fallback = True

        face_results: List[FaceResult] = []

        for box in faces:
            crop = bgr_img[box.y1: box.y2, box.x1: box.x2]
            if crop.size == 0:
                continue

            emotion = emotion_predict_face_crop(crop)
            aus = au_predict_face_crop(crop)
            age_group, gender = age_gender_predict_face_crop(crop)

            face_results.append(
                FaceResult(
                    box=box,
                    emotion=emotion,
                    aus=aus,
                    age_group=age_group,
                    gender=gender,
                )
            )

        body_res = analyze_body(bgr_img)

        meta = {
            "num_faces": len(face_results),
            "image_shape": [int(bgr_img.shape[0]), int(bgr_img.shape[1])],
            "face_detector": getattr(self, "face_detector_name", "unknown"),
            "face_detector_fallback_to_haar": used_fallback,
            "use_trained_emotion_model": _use_trained_emotion_model,
            "use_age_gender_model": _use_age_gender_model,
            "use_au_model": _use_au_model,
            "use_body_model": _use_body_model,
            "note": (
                "Эмоции: CNN (FER-2013) при наличии models/emotion_cnn.pth. "
                "Возраст/пол: CNN (UTKFace) при наличии models/age_gender_cnn.pth. "
                "Мимика: Py-Feat AU при наличии feat. "
                "Телосложение: эвристический анализ по позе (MediaPipe Pose)."
            ),
        }

        return AnalysisResult(
            faces=face_results,
            body=body_res,
            meta=meta,
        )


pipeline = AnalyzerPipeline()


# ===== Вспомогательные функции =====

def draw_faces(bgr_img, faces: List[FaceResult]):
    out = bgr_img.copy()
    for fr in faces:
        b = fr.box
        cv2.rectangle(out, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)

        top_em = (
            max(fr.emotion.items(), key=lambda x: x[1])[0]
            if fr.emotion
            else "face"
        )
        label = top_em
        if fr.gender is not None and fr.age_group is not None:
            label = f"{top_em} | {fr.gender}, {fr.age_group}"

        cv2.putText(
            out,
            label,
            (b.x1, max(0, b.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def encode_image_to_data_url(img_rgb: np.ndarray) -> str:
    thumb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = thumb.shape[:2]
    scale = 128 / max(h, w)
    if scale < 1.0:
        thumb = cv2.resize(
            thumb,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".png", thumb)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def render_history_html(entries: List[Tuple[str, str]]) -> str:
    if not entries:
        return "<div class='history-empty'>История пока пуста.</div>"

    lines = ["<div class='history-list'>"]
    for data_url, caption in reversed(entries):
        lines.append(
            f"""
            <div class="history-item">
               <img src="{data_url}" alt="preview"/>
               <div class="history-caption">{caption}</div>
            </div>
            """
        )
    lines.append("</div>")
    return "\n".join(lines)


# ===== Основная функция анализа =====

def analyze_image(img_rgb):
    global history_entries

    if img_rgb is None:
        return None, {}, [], "", render_history_html(history_entries)

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    result = pipeline.analyze(bgr)

    vis = draw_faces(bgr, result.faces)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    result_dict = {
        "faces": [
            {
                "box": asdict(fr.box),
                "emotion": fr.emotion,
                "aus": fr.aus,
                "age_group": fr.age_group,
                "gender": fr.gender,
            }
            for fr in result.faces
        ],
        "body": None
        if result.body is None
        else {
            "category": result.body.category,
            "scores": result.body.scores,
        },
        "meta": result.meta,
    }

    # краткая таблица без горизонтальной прокрутки:
    # face, emotion (топ-1), age_group, gender
    summary_rows: List[List[Any]] = []
    au_lines: List[str] = []

    for i, fr in enumerate(result.faces, start=1):
        if fr.emotion:
            top_em = max(fr.emotion.items(), key=lambda x: x[1])
            em_name, em_prob = top_em[0], top_em[1]
            em_str = f"{em_name} ({em_prob:.2f})"
        else:
            em_str = ""

        # топ-3 AU — в отдельном текстовом блоке
        top3_au = sorted(
            (fr.aus or {}).items(), key=lambda x: x[1], reverse=True
        )[:3]
        if top3_au:
            au_desc_parts = []
            for k, v in top3_au:
                desc = AU_DESCRIPTIONS.get(k, "")
                if desc:
                    au_desc_parts.append(f"{k} ({desc}): {v:.2f}")
                else:
                    au_desc_parts.append(f"{k}: {v:.2f}")
            au_line = f"- Лицо {i}: " + "; ".join(au_desc_parts)
            au_lines.append(au_line)

        summary_rows.append(
            [
                i,
                em_str,
                fr.age_group or "",
                fr.gender or "",
            ]
        )

    if not au_lines:
        au_text = "Мимика (AU): данные отсутствуют или модуль недоступен."
    else:
        au_text = "#### Мимика (Action Units)\n" + "\n".join(au_lines)

    body_text = summarize_body(result.body)
    combined_md = au_text + "\n\n---\n\n#### Телосложение\n" + body_text

    # История
    if result.faces:
        main_face = result.faces[0]
        if main_face.emotion:
            main_em = max(main_face.emotion.items(), key=lambda x: x[1])[0]
        else:
            main_em = "unknown"
        ag = main_face.age_group or "?"
        gd = main_face.gender or "?"
        history_caption = f"{main_em}, {gd}, {ag}"
    else:
        history_caption = "лицо не обнаружено"

    data_url = encode_image_to_data_url(img_rgb)
    if data_url:
        history_entries.append((data_url, history_caption))
        history_entries = history_entries[-5:]

    history_html = render_history_html(history_entries)

    return vis_rgb, result_dict, summary_rows, combined_md, history_html


def clear_form():
    """Очищаем только текущий ввод/вывод, но историю НЕ трогаем."""
    return None, None, {}, [], "", render_history_html(history_entries)


# ===== UI / оформление =====

custom_css = """
html, body {
    height: 100%;
    margin: 0;
    padding: 0;
    background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #020617 100%);
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    background: transparent !important;
    color: #e5e7eb !important;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.gradio-app, .gradio-interface, .gr-block {
    background: transparent !important;
}

/* Верхний бар */
#title-bar {
    background: linear-gradient(90deg, #1d4ed8, #22c55e);
    padding: 14px 20px;
    border-radius: 14px;
    color: white;
    margin: 16px 0 12px 0;
}
#title-bar h1 {
    font-size: 24px;
    margin: 0 0 4px 0;
}
#title-bar p {
    margin: 0;
    opacity: 0.9;
    font-size: 13px;
}

/* Карточки секций */
.section-card {
    background: rgba(15, 23, 42, 0.96) !important;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.5);
    padding: 14px 16px !important;
    backdrop-filter: blur(12px);
}

/* Текст */
.section-card .prose p,
.section-card .prose li,
.section-card .prose h1,
.section-card .prose h2,
.section-card .prose h3,
.section-card label,
.gradio-container .prose strong {
    color: #e5e7eb !important;
}

/* Кнопки */
button {
    border-radius: 999px !important;
}

/* Dataframe тёмная */
.gr-dataframe table {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    font-size: 13px !important;
}
.gr-dataframe th, .gr-dataframe td {
    background-color: #020617 !important;
    color: #e5e7eb !important;
    border-color: #1e293b !important;
}

/* История */
.history-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.history-item {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(15, 23, 42, 0.9);
    border-radius: 12px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    padding: 6px 8px;
}
.history-item img {
    width: 64px;
    height: 64px;
    border-radius: 10px;
    object-fit: cover;
}
.history-caption {
    font-size: 13px;
    color: #e5e7eb;
}
.history-empty {
    font-size: 13px;
    color: #9ca3af;
}
"""

with gr.Blocks(
    title="Visual Content Analyzer — Demo MVP",
    css=custom_css,
    theme=gr.themes.Soft()
) as demo:

    with gr.Row():
        with gr.Column():
            gr.HTML(
                """
                <div id="title-bar">
                  <h1>Visual Content Analyzer</h1>
                  <p>Эмоции • Возраст и пол • Мимика (Action Units) • Телосложение</p>
                </div>
                """
            )

    with gr.Row():
        with gr.Column(scale=5):
            with gr.Group(elem_classes="section-card"):
                gr.Markdown(
                    "### 1️⃣ Загрузите изображение\n"
                    "- Поддерживаются фотографии с одним или несколькими лицами\n"
                    "- Желательно, чтобы человек был виден по пояс или полностью "
                    "для анализа телосложения"
                )
                inp = gr.Image(type="numpy", label="Фото пользователя", height=420)

                with gr.Row():
                    analyze_btn = gr.Button("🔍 Анализировать", variant="primary")
                    clear_btn = gr.Button("♻ Очистить форму", variant="secondary")

            with gr.Group(elem_classes="section-card"):
                gr.Markdown(
                    "### Примеры\n"
                    "Если положить картинки в `assets/example_face1.jpg` и "
                    "`assets/example_face2.jpg`, их можно использовать для демо."
                )
                gr.Examples(
                    examples=[
                        ["assets/example_face1.jpg"],
                        ["assets/example_face2.jpg"],
                    ],
                    inputs=inp,
                    label="Демо-фотографии",
                )

        with gr.Column(scale=7):
            with gr.Group(elem_classes="section-card"):
                gr.Markdown("### 2️⃣ Детекция лиц и сводка по человеку")
                out_img = gr.Image(
                    type="numpy", label="Результат с боксами", height=420
                )

            with gr.Group(elem_classes="section-card"):
                with gr.Tabs():
                    with gr.Tab("Сводка по лицам"):
                        out_table = gr.Dataframe(
                            headers=["face", "emotion", "age_group", "gender"],
                            datatype=["number", "str", "str", "str"],
                            label="Краткое резюме по лицам",
                        )
                    with gr.Tab("JSON-структура"):
                        out_json = gr.JSON(label="Полный структурированный результат")

                body_md = gr.Markdown(label="Мимика и телосложение")

            with gr.Group(elem_classes="section-card"):
                gr.Markdown("### 3️⃣ История анализов (последние 5)")
                history_html = gr.HTML(render_history_html(history_entries))

    gr.Markdown(
        "#### ℹ️ Примечание\n"
        "- Эмоции: обученная CNN на FER-2013 при наличии `models/emotion_cnn.pth`.\n"
        "- Возраст и пол: CNN на UTKFace (aligned & cropped) при наличии `models/age_gender_cnn.pth`.\n"
        "- Мимика (AU): Py-Feat (AU-модель, обученная на BP4D/DISFA и др.) при наличии библиотеки `feat`.\n"
        "- Телосложение: анализ позы с использованием MediaPipe Pose при наличии `mediapipe`.\n"
        "При отсутствии соответствующих модулей используются заглушки — это видно в поле `meta` JSON."
    )

    analyze_btn.click(
        analyze_image,
        inputs=inp,
        outputs=[out_img, out_json, out_table, body_md, history_html],
    )

    clear_btn.click(
        clear_form,
        inputs=[],
        outputs=[inp, out_img, out_json, out_table, body_md, history_html],
    )


if __name__ == "__main__":
    demo.launch()
import io
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

APP_VERSION = "0.3.3"

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

AGE_BINS = [(0, 12), (13, 25), (26, 40), (41, 60), (61, 120)]
AGE_BIN_LABELS = ["0-12", "13-25", "26-40", "41-60", "61+"]

MAX_AGE_FOR_REG = 120.0
MAX_AGE_DLDL = 120
NUM_AGES_DLDL = MAX_AGE_DLDL + 1  # 121


def age_to_bin_label(age: float) -> str:
    a = max(0.0, min(float(age), 120.0))
    for (lo, hi), lab in zip(AGE_BINS, AGE_BIN_LABELS):
        if lo <= a <= hi:
            return lab
    return AGE_BIN_LABELS[-1]


def age_bin_prob_from_probs(age_probs: List[float], bin_lo: int, bin_hi: int) -> float:
    lo = max(0, int(bin_lo))
    hi = min(MAX_AGE_DLDL, int(bin_hi))
    return float(sum(age_probs[lo:hi + 1]))


def find_models_dir() -> str:
    env = os.getenv("MODELS_DIR")
    if env and os.path.isdir(env):
        return os.path.abspath(env)

    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.join(here, "models"),
        os.path.join(here, "..", "models"),
        os.path.join(here, "..", "..", "models"),
        os.path.join(os.path.abspath(os.getcwd()), "models"),
        r"C:\vk-vision-demo\models",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)

    return os.path.abspath(os.getcwd())


MODELS_DIR = find_models_dir()


def resolve_weights_path(env_key: str, candidates: List[str]) -> str:
    """
    1) Если задан env_key и файл существует — используем его.
    2) Иначе берём первый существующий из candidates в MODELS_DIR.
    3) Иначе возвращаем путь к первому кандидату (для /health).
    """
    env_val = os.getenv(env_key)
    if env_val and os.path.isfile(env_val):
        return os.path.abspath(env_val)

    for name in candidates:
        p = os.path.join(MODELS_DIR, name)
        if os.path.isfile(p):
            return os.path.abspath(p)

    return os.path.abspath(os.path.join(MODELS_DIR, candidates[0]))


# --- Weights paths (v2 -> legacy) ---
EMOTION_WEIGHTS_PATH = resolve_weights_path(
    "EMOTION_WEIGHTS",
    ["emotion_resnet18.pth"],
)

AGE_GENDER_DLDL_WEIGHTS_PATH = resolve_weights_path(
    "AGE_GENDER_DLDL_WEIGHTS",
    [
        "age_gender_dldl_resnet50_v2_best.pth",  # ✅ новый
        "age_gender_dldl_resnet50.pth",          # legacy
    ],
)

AGE_GENDER_REG_WEIGHTS_PATH = resolve_weights_path(
    "AGE_GENDER_REG_WEIGHTS",
    [
        "age_gender_regression_resnet18_v2_best.pth",  # ✅ новый
        "age_gender_regression_resnet18.pth",          # legacy
    ],
)

AGE_GENDER_BINS_WEIGHTS_PATH = resolve_weights_path(
    "AGE_GENDER_BINS_WEIGHTS",
    ["age_gender_resnet18.pth"],
)


# -------- optional deps --------
torch_ok = False
torch_error = None
torch = None
nn = None
F = None
torchvision_models = None

mp_ok = False
mp_error = None
mp = None

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from torchvision import models as _torchvision_models

    torch = _torch
    nn = _nn
    F = _F
    torchvision_models = _torchvision_models
    torch_ok = True
except Exception as e:
    torch_error = repr(e)

try:
    import mediapipe as _mp
    mp = _mp
    mp_ok = True
except Exception as e:
    mp_error = repr(e)

DEVICE = "cpu"
if torch_ok and torch is not None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class LoadStatus:
    ready: bool
    message: str
    path: str


emotion_model = None
emotion_status = LoadStatus(False, "not loaded", EMOTION_WEIGHTS_PATH)

age_gender_dldl_model = None
age_gender_dldl_status = LoadStatus(False, "not loaded", AGE_GENDER_DLDL_WEIGHTS_PATH)
age_gender_dldl_is_v2 = False  # чтобы инференс понимал формат gender выхода

age_gender_reg_model = None
age_gender_reg_status = LoadStatus(False, "not loaded", AGE_GENDER_REG_WEIGHTS_PATH)
age_gender_reg_is_v2 = False

age_gender_bins_model = None
age_gender_bins_status = LoadStatus(False, "not loaded", AGE_GENDER_BINS_WEIGHTS_PATH)


# -------- Face detector --------
_face_detector = None
face_detector_ok = False
face_detector_error = None


def init_face_detector():
    global _face_detector, face_detector_ok, face_detector_error
    if not mp_ok or mp is None:
        face_detector_ok = False
        face_detector_error = "mediapipe is not available"
        return
    try:
        _face_detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        face_detector_ok = True
        face_detector_error = None
    except Exception as e:
        face_detector_ok = False
        face_detector_error = repr(e)


def pil_to_rgb(pil_img: Image.Image) -> Image.Image:
    return pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img


def detect_faces_mediapipe(pil_img: Image.Image) -> List[Dict[str, Any]]:
    if not face_detector_ok or _face_detector is None:
        return []
    import numpy as np
    arr = np.array(pil_to_rgb(pil_img))
    res = _face_detector.process(arr)
    if not res.detections:
        return []
    H, W = arr.shape[:2]
    out = []
    for det in res.detections:
        bb = det.location_data.relative_bounding_box
        x = int(max(0.0, min(1.0, bb.xmin)) * W)
        y = int(max(0.0, min(1.0, bb.ymin)) * H)
        w = int(max(0.0, min(1.0, bb.width)) * W)
        h = int(max(0.0, min(1.0, bb.height)) * H)
        score = float(det.score[0]) if det.score else 0.0
        out.append({"bbox": [x, y, w, h], "score": score})
    return out


def crop_face(pil_img: Image.Image, bbox: List[int], pad: float = 0.15) -> Image.Image:
    x, y, w, h = bbox
    W, H = pil_img.size
    px = int(w * pad)
    py = int(h * pad)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(W, x + w + px)
    y2 = min(H, y + h + py)
    return pil_img.crop((x1, y1, x2, y2))


# -------- Preprocess --------
_IMAGENET_MEAN = None
_IMAGENET_STD = None


def get_imagenet_norm_tensors():
    global _IMAGENET_MEAN, _IMAGENET_STD
    if _IMAGENET_MEAN is None or _IMAGENET_STD is None:
        _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
        _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
    return _IMAGENET_MEAN, _IMAGENET_STD


def preprocess_for_resnet(pil_img: Image.Image):
    if not torch_ok or torch is None:
        raise RuntimeError("torch is not available")
    img = pil_to_rgb(pil_img).resize((224, 224))
    import numpy as np
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.transpose(2, 0, 1)
    x = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
    mean, std = get_imagenet_norm_tensors()
    x = (x - mean) / std
    return x


def preprocess_for_emotion(pil_img: Image.Image):
    gray3 = pil_img.convert("L").convert("RGB")
    return preprocess_for_resnet(gray3)


def safe_load_state_dict(model, sd):
    """
    Поддерживает:
    - чистый state_dict
    - checkpoint {"model": state_dict, ...}
    - checkpoint {"state_dict": state_dict, ...}
    - убирает "module."
    - ремапит ключи head_age/head_gender <-> age_head/gender_head
    """
    if sd is None:
        raise ValueError("state_dict is None")

    # unwrap checkpoint
    if isinstance(sd, dict):
        if "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        elif "state_dict" in sd and isinstance(sd["state_dict"], dict):
            sd = sd["state_dict"]

    if not isinstance(sd, dict):
        raise ValueError("state_dict is not a dict after unwrap")

    # remove module. prefix
    if any(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    keys = set(sd.keys())
    has_age_head = any(k.startswith("age_head.") for k in keys)
    has_head_age = any(k.startswith("head_age.") for k in keys)

    # if weights use age_head.* but model expects head_age.*
    if has_age_head and not has_head_age:
        sd = {
            (k.replace("age_head.", "head_age.", 1) if k.startswith("age_head.")
             else k.replace("gender_head.", "head_gender.", 1) if k.startswith("gender_head.")
             else k): v
            for k, v in sd.items()
        }

    keys = set(sd.keys())
    has_age_head = any(k.startswith("age_head.") for k in keys)
    has_head_age = any(k.startswith("head_age.") for k in keys)

    # if weights use head_age.* but model expects age_head.*
    if has_head_age and not has_age_head:
        sd = {
            (k.replace("head_age.", "age_head.", 1) if k.startswith("head_age.")
             else k.replace("head_gender.", "gender_head.", 1) if k.startswith("head_gender.")
             else k): v
            for k, v in sd.items()
        }

    model.load_state_dict(sd, strict=True)


def gender_label_from_softmax_idx(gender_idx: int) -> str:
    """
    UTKFace convention:
    0 = male, 1 = female
    """
    return "female" if int(gender_idx) == 1 else "male"


# -------- Models --------
def load_emotion_model():
    global emotion_model, emotion_status
    if not torch_ok:
        emotion_status = LoadStatus(False, f"torch not available: {torch_error}", EMOTION_WEIGHTS_PATH)
        return
    if not os.path.isfile(EMOTION_WEIGHTS_PATH):
        emotion_status = LoadStatus(False, "weights file not found", EMOTION_WEIGHTS_PATH)
        return
    try:
        m = torchvision_models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, len(EMOTION_LABELS))
        sd = torch.load(EMOTION_WEIGHTS_PATH, map_location="cpu")
        safe_load_state_dict(m, sd)
        m.eval()
        m.to(DEVICE)
        emotion_model = m
        emotion_status = LoadStatus(True, "loaded", EMOTION_WEIGHTS_PATH)
    except Exception as e:
        emotion_status = LoadStatus(False, f"load error: {repr(e)}", EMOTION_WEIGHTS_PATH)


class AgeGenderBinsResNet(nn.Module):
    def __init__(self, num_age_bins=5):
        super().__init__()
        b = torchvision_models.resnet18(weights=None)
        in_features = b.fc.in_features
        b.fc = nn.Identity()
        self.backbone = b
        self.age_head = nn.Linear(in_features, num_age_bins)
        self.gender_head = nn.Linear(in_features, 2)

    def forward(self, x):
        feats = self.backbone(x)
        return self.age_head(feats), self.gender_head(feats)


# --- LEGACY models (2-class gender) ---
class AgeGenderRegResNetLegacy(nn.Module):
    def __init__(self):
        super().__init__()
        b = torchvision_models.resnet18(weights=None)
        in_features = b.fc.in_features
        b.fc = nn.Identity()
        self.backbone = b
        self.age_head = nn.Linear(in_features, 1)
        self.gender_head = nn.Linear(in_features, 2)

    def forward(self, x):
        feats = self.backbone(x)
        age = self.age_head(feats).squeeze(1)
        gender_logits = self.gender_head(feats)
        return age, gender_logits


class AgeGenderDLDLResNet50Legacy(nn.Module):
    def __init__(self):
        super().__init__()
        b = torchvision_models.resnet50(weights=None)
        in_features = b.fc.in_features
        b.fc = nn.Identity()
        self.backbone = b
        self.age_head = nn.Linear(in_features, NUM_AGES_DLDL)
        self.gender_head = nn.Linear(in_features, 2)

    def forward(self, x):
        feats = self.backbone(x)
        return self.age_head(feats), self.gender_head(feats)


# --- V2 models (gender = 1 logit BCE) ---
class AgeGenderRegResNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        b = torchvision_models.resnet18(weights=None)
        in_features = b.fc.in_features
        b.fc = nn.Identity()
        self.backbone = b
        self.age_head = nn.Linear(in_features, 1)     # years
        self.gender_head = nn.Linear(in_features, 1)  # logit (class=1 means FEMALE in UTKFace)

    def forward(self, x):
        feats = self.backbone(x)
        age = self.age_head(feats).squeeze(1)
        gender_logit = self.gender_head(feats).squeeze(1)
        return age, gender_logit


class AgeGenderDLDLResNet50V2(nn.Module):
    def __init__(self):
        super().__init__()
        b = torchvision_models.resnet50(weights=None)
        in_features = b.fc.in_features
        b.fc = nn.Identity()
        self.backbone = b
        self.age_head = nn.Linear(in_features, NUM_AGES_DLDL)  # 121
        self.gender_head = nn.Linear(in_features, 1)           # logit (class=1 means FEMALE in UTKFace)

    def forward(self, x):
        feats = self.backbone(x)
        age_logits = self.age_head(feats)
        gender_logit = self.gender_head(feats).squeeze(1)
        return age_logits, gender_logit


def load_age_gender_dldl():
    global age_gender_dldl_model, age_gender_dldl_status, age_gender_dldl_is_v2
    age_gender_dldl_is_v2 = False

    if not torch_ok:
        age_gender_dldl_status = LoadStatus(False, f"torch not available: {torch_error}", AGE_GENDER_DLDL_WEIGHTS_PATH)
        return
    if not os.path.isfile(AGE_GENDER_DLDL_WEIGHTS_PATH):
        age_gender_dldl_status = LoadStatus(False, "weights file not found", AGE_GENDER_DLDL_WEIGHTS_PATH)
        return

    try:
        sd = torch.load(AGE_GENDER_DLDL_WEIGHTS_PATH, map_location="cpu")
    except Exception as e:
        age_gender_dldl_status = LoadStatus(False, f"load error: {repr(e)}", AGE_GENDER_DLDL_WEIGHTS_PATH)
        return

    # try v2 first
    try:
        m = AgeGenderDLDLResNet50V2().to("cpu")
        safe_load_state_dict(m, sd)
        m.eval()
        m.to(DEVICE)
        age_gender_dldl_model = m
        age_gender_dldl_is_v2 = True
        age_gender_dldl_status = LoadStatus(True, "loaded (v2)", AGE_GENDER_DLDL_WEIGHTS_PATH)
        return
    except Exception:
        pass

    # fallback legacy
    try:
        m = AgeGenderDLDLResNet50Legacy().to("cpu")
        safe_load_state_dict(m, sd)
        m.eval()
        m.to(DEVICE)
        age_gender_dldl_model = m
        age_gender_dldl_is_v2 = False
        age_gender_dldl_status = LoadStatus(True, "loaded (legacy)", AGE_GENDER_DLDL_WEIGHTS_PATH)
    except Exception as e:
        age_gender_dldl_status = LoadStatus(False, f"load error: {repr(e)}", AGE_GENDER_DLDL_WEIGHTS_PATH)


def load_age_gender_reg():
    global age_gender_reg_model, age_gender_reg_status, age_gender_reg_is_v2
    age_gender_reg_is_v2 = False

    if not torch_ok:
        age_gender_reg_status = LoadStatus(False, f"torch not available: {torch_error}", AGE_GENDER_REG_WEIGHTS_PATH)
        return
    if not os.path.isfile(AGE_GENDER_REG_WEIGHTS_PATH):
        age_gender_reg_status = LoadStatus(False, "weights file not found", AGE_GENDER_REG_WEIGHTS_PATH)
        return

    try:
        sd = torch.load(AGE_GENDER_REG_WEIGHTS_PATH, map_location="cpu")
    except Exception as e:
        age_gender_reg_status = LoadStatus(False, f"load error: {repr(e)}", AGE_GENDER_REG_WEIGHTS_PATH)
        return

    # try v2 first
    try:
        m = AgeGenderRegResNetV2().to("cpu")
        safe_load_state_dict(m, sd)
        m.eval()
        m.to(DEVICE)
        age_gender_reg_model = m
        age_gender_reg_is_v2 = True
        age_gender_reg_status = LoadStatus(True, "loaded (v2)", AGE_GENDER_REG_WEIGHTS_PATH)
        return
    except Exception:
        pass

    # fallback legacy
    try:
        m = AgeGenderRegResNetLegacy().to("cpu")
        safe_load_state_dict(m, sd)
        m.eval()
        m.to(DEVICE)
        age_gender_reg_model = m
        age_gender_reg_is_v2 = False
        age_gender_reg_status = LoadStatus(True, "loaded (legacy)", AGE_GENDER_REG_WEIGHTS_PATH)
    except Exception as e:
        age_gender_reg_status = LoadStatus(False, f"load error: {repr(e)}", AGE_GENDER_REG_WEIGHTS_PATH)


def load_age_gender_bins():
    global age_gender_bins_model, age_gender_bins_status
    if not torch_ok:
        age_gender_bins_status = LoadStatus(False, f"torch not available: {torch_error}", AGE_GENDER_BINS_WEIGHTS_PATH)
        return
    if not os.path.isfile(AGE_GENDER_BINS_WEIGHTS_PATH):
        age_gender_bins_status = LoadStatus(False, "weights file not found", AGE_GENDER_BINS_WEIGHTS_PATH)
        return
    try:
        m = AgeGenderBinsResNet(num_age_bins=len(AGE_BINS)).to("cpu")
        sd = torch.load(AGE_GENDER_BINS_WEIGHTS_PATH, map_location="cpu")
        safe_load_state_dict(m, sd)
        m.eval()
        m.to(DEVICE)
        age_gender_bins_model = m
        age_gender_bins_status = LoadStatus(True, "loaded", AGE_GENDER_BINS_WEIGHTS_PATH)
    except Exception as e:
        age_gender_bins_status = LoadStatus(False, f"load error: {repr(e)}", AGE_GENDER_BINS_WEIGHTS_PATH)


# -------- Inference --------
def infer_emotion(face_pil: Image.Image) -> Optional[Dict[str, Any]]:
    if not emotion_status.ready or emotion_model is None:
        return None
    x = preprocess_for_emotion(face_pil)
    with torch.no_grad():
        logits = emotion_model(x)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy().tolist()
    idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return {
        "label": EMOTION_LABELS[idx],
        "prob": float(probs[idx]),
        "probs": {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(probs))}
    }


def infer_age_gender_dldl(face_pil: Image.Image) -> Optional[Dict[str, Any]]:
    if not age_gender_dldl_status.ready or age_gender_dldl_model is None:
        return None

    x = preprocess_for_resnet(face_pil)
    with torch.no_grad():
        age_logits, gender_out = age_gender_dldl_model(x)
        age_probs = F.softmax(age_logits, dim=1)[0].cpu().numpy().tolist()

        # gender_out: либо (2 classes) либо (1 logit)
        if age_gender_dldl_is_v2:
            # ✅ FIX: sigmoid(logit) = P(class==1) = P(FEMALE) (UTKFace: 1=female)
            p_female = float(torch.sigmoid(gender_out)[0].item())
            gender_label = "female" if p_female >= 0.5 else "male"
            gender_prob = p_female if gender_label == "female" else 1.0 - p_female
        else:
            gender_probs = F.softmax(gender_out, dim=1)[0].cpu().numpy().tolist()
            gender_idx = int(max(range(len(gender_probs)), key=lambda i: gender_probs[i]))
            gender_label = gender_label_from_softmax_idx(gender_idx)
            gender_prob = float(gender_probs[gender_idx])

    age_est = float(sum(p * i for i, p in enumerate(age_probs)))
    age_est = max(0.0, min(120.0, age_est))

    bin_label = age_to_bin_label(age_est)
    bin_idx = AGE_BIN_LABELS.index(bin_label)
    lo, hi = AGE_BINS[bin_idx]
    bin_prob = age_bin_prob_from_probs(age_probs, lo, hi)

    return {
        "age_bin": bin_label,
        "age_bin_prob": float(bin_prob),
        "age_est": float(age_est),
        "gender": gender_label,
        "gender_prob": float(gender_prob),
        "mode": "dldl",
    }


def infer_age_gender_reg(face_pil: Image.Image) -> Optional[Dict[str, Any]]:
    if not age_gender_reg_status.ready or age_gender_reg_model is None:
        return None

    x = preprocess_for_resnet(face_pil)
    with torch.no_grad():
        age_out, gender_out = age_gender_reg_model(x)

        if age_gender_reg_is_v2:
            age_est = float(age_out[0].item())
            age_est = max(0.0, min(120.0, age_est))

            # ✅ FIX: sigmoid(logit) = P(class==1) = P(FEMALE) (UTKFace: 1=female)
            p_female = float(torch.sigmoid(gender_out)[0].item())
            gender_label = "female" if p_female >= 0.5 else "male"
            gender_prob = p_female if gender_label == "female" else 1.0 - p_female
        else:
            age_est = float(age_out[0].item() * MAX_AGE_FOR_REG)
            age_est = max(0.0, min(120.0, age_est))

            gender_probs = F.softmax(gender_out, dim=1)[0].cpu().numpy().tolist()
            gender_idx = int(max(range(len(gender_probs)), key=lambda i: gender_probs[i]))
            gender_label = gender_label_from_softmax_idx(gender_idx)
            gender_prob = float(gender_probs[gender_idx])

    return {
        "age_bin": age_to_bin_label(age_est),
        "age_bin_prob": 1.0,
        "age_est": float(age_est),
        "gender": gender_label,
        "gender_prob": float(gender_prob),
        "mode": "regression",
    }


def infer_age_gender_bins(face_pil: Image.Image) -> Optional[Dict[str, Any]]:
    if not age_gender_bins_status.ready or age_gender_bins_model is None:
        return None
    x = preprocess_for_resnet(face_pil)
    with torch.no_grad():
        age_logits, gender_logits = age_gender_bins_model(x)
        age_probs = F.softmax(age_logits, dim=1)[0].cpu().numpy().tolist()
        gender_probs = F.softmax(gender_logits, dim=1)[0].cpu().numpy().tolist()

    age_idx = int(max(range(len(age_probs)), key=lambda i: age_probs[i]))
    gender_idx = int(max(range(len(gender_probs)), key=lambda i: gender_probs[i]))
    gender_label = gender_label_from_softmax_idx(gender_idx)

    centers = [(a + b) / 2 for (a, b) in AGE_BINS]
    age_est = float(sum(p * c for p, c in zip(age_probs, centers)))

    return {
        "age_bin": AGE_BIN_LABELS[age_idx],
        "age_bin_prob": float(age_probs[age_idx]),
        "age_est": float(age_est),
        "gender": gender_label,
        "gender_prob": float(gender_probs[gender_idx]),
        "mode": "bins",
    }


def infer_age_gender(face_pil: Image.Image) -> Optional[Dict[str, Any]]:
    out = infer_age_gender_dldl(face_pil)
    if out is not None:
        return out
    out = infer_age_gender_reg(face_pil)
    if out is not None:
        return out
    return infer_age_gender_bins(face_pil)


# -------- App --------
app = FastAPI(title="VK Vision ML Service", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_models():
    init_face_detector()
    load_emotion_model()
    load_age_gender_dldl()
    load_age_gender_reg()
    load_age_gender_bins()


@app.on_event("startup")
def _startup():
    init_models()


@app.get("/health")
def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "time_ms": int(time.time() * 1000),
        "models_dir": MODELS_DIR,
        "device": DEVICE,
        "torch_ok": torch_ok,
        "torch_error": torch_error,
        "mediapipe_ok": mp_ok,
        "mediapipe_error": mp_error,
        "face_detector_ok": face_detector_ok,
        "face_detector_error": face_detector_error,
        "emotion": {"ready": emotion_status.ready, "message": emotion_status.message, "path": emotion_status.path},
        "age_gender_dldl": {"ready": age_gender_dldl_status.ready, "message": age_gender_dldl_status.message, "path": age_gender_dldl_status.path},
        "age_gender_regression": {"ready": age_gender_reg_status.ready, "message": age_gender_reg_status.message, "path": age_gender_reg_status.path},
        "age_gender_bins": {"ready": age_gender_bins_status.ready, "message": age_gender_bins_status.message, "path": age_gender_bins_status.path},
    }


@app.post("/infer")
async def infer(
    file: Optional[UploadFile] = File(default=None),
    image: Optional[UploadFile] = File(default=None),
):
    up = file or image
    if up is None:
        return {"ok": False, "error": "No file provided. Use form field 'file' or 'image'."}

    raw = await up.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")

    faces = detect_faces_mediapipe(pil_img)
    results = []

    for f in faces:
        bbox = f["bbox"]
        face_pil = crop_face(pil_img, bbox, pad=0.15)

        emo = infer_emotion(face_pil)
        ag = infer_age_gender(face_pil)

        results.append({"bbox": bbox, "score": f.get("score", 0.0), "emotion": emo, "age_gender": ag})

    return {
        "ok": True,
        "version": APP_VERSION,
        "time_ms": int(time.time() * 1000),
        "faces_count": len(results),
        "faces": results,
    }
from __future__ import annotations
import io, json, math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import cv2

# Optional torch models (if user copies models into backend/models/)
TORCH_AVAILABLE = True
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
except Exception:
    TORCH_AVAILABLE = False

EMOTION_CLASSES = ['angry','disgust','fear','happy','neutral','sad','surprise']

AU_LABELS = {
    "AU01":"inner brow raiser",
    "AU02":"outer brow raiser",
    "AU04":"brow lowerer",
    "AU06":"cheek raiser",
    "AU07":"lid tightener",
    "AU10":"upper lip raiser",
    "AU12":"lip corner puller (smile)",
    "AU14":"dimpler",
    "AU15":"lip corner depressor",
    "AU17":"chin raiser",
    "AU20":"lip stretcher",
    "AU23":"lip tightener",
    "AU24":"lip pressor",
    "AU25":"lips part",
    "AU26":"jaw drop",
    "AU45":"blink",
}

@dataclass
class LoadedModels:
    emotion_resnet: Optional[Any] = None
    age_gender_resnet: Optional[Any] = None
    device: str = "cpu"

def _safe_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

class Analyzer:
    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.models = self._load_models()

        if TORCH_AVAILABLE:
            self.tf_emotion = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
            self.tf_agegender = T.Compose([
                T.Resize((224,224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def _load_models(self) -> LoadedModels:
        lm = LoadedModels()
        if not TORCH_AVAILABLE:
            return lm

        device = "cuda" if torch.cuda.is_available() else "cpu"
        lm.device = device

        # Emotion ResNet18 (expects state_dict from torchvision resnet18 with replaced fc)
        try:
            m = models.resnet18(weights=None)
            m.fc = nn.Linear(m.fc.in_features, len(EMOTION_CLASSES))
            sd = torch.load("models/emotion_resnet18.pth", map_location=device)
            m.load_state_dict(sd)
            m.to(device).eval()
            lm.emotion_resnet = m
            print("[INFO] emotion_resnet18 loaded")
        except Exception as e:
            print("[WARN] emotion_resnet18 not loaded:", e)

        # Age/Gender ResNet18 (expected to be trained similarly; placeholder head sizes)
        # We assume: age_bins=10 (0-9,10-19,...90+) and gender=2
        try:
            class AG(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.backbone = models.resnet18(weights=None)
                    in_f = self.backbone.fc.in_features
                    self.backbone.fc = nn.Identity()
                    self.age_head = nn.Linear(in_f, 10)
                    self.gender_head = nn.Linear(in_f, 2)
                def forward(self,x):
                    feat = self.backbone(x)
                    return self.age_head(feat), self.gender_head(feat)
            m = AG()
            sd = torch.load("models/age_gender_resnet18.pth", map_location=device)
            m.load_state_dict(sd)
            m.to(device).eval()
            lm.age_gender_resnet = m
            print("[INFO] age_gender_resnet18 loaded")
        except Exception as e:
            print("[WARN] age_gender_resnet18 not loaded:", e)

        return lm

    def detect_faces(self, rgb: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6, minSize=(60,60))
        out = []
        for (x,y,w,h) in faces:
            # Haar doesn't give score; approximate by size
            score = float(min(1.0, (w*h)/(300*300)))
            out.append((int(x),int(y),int(w),int(h),score))
        # sort largest first
        out.sort(key=lambda t: t[2]*t[3], reverse=True)
        return out

    def _crop_face(self, rgb: np.ndarray, box: Tuple[int,int,int,int]) -> np.ndarray:
        x,y,w,h = box
        pad = int(0.15*max(w,h))
        x0 = max(0, x-pad); y0 = max(0,y-pad)
        x1 = min(rgb.shape[1], x+w+pad); y1 = min(rgb.shape[0], y+h+pad)
        return rgb[y0:y1, x0:x1].copy()

    def predict_emotion(self, face_rgb: np.ndarray) -> Dict[str, Any]:
        if TORCH_AVAILABLE and self.models.emotion_resnet is not None:
            import torch
            pil = Image.fromarray(face_rgb)
            x = self.tf_emotion(pil).unsqueeze(0).to(self.models.device)
            with torch.no_grad():
                logits = self.models.emotion_resnet(x).squeeze(0).detach().cpu().numpy()
            probs = _safe_softmax(logits)
            idxs = probs.argsort()[::-1][:3]
            top3 = [{"label": EMOTION_CLASSES[int(i)], "prob": float(probs[int(i)])} for i in idxs]
            return {"top3": top3, "label": top3[0]["label"], "confidence": top3[0]["prob"], "source":"emotion_resnet18"}
        # fallback heuristic
        return {"top3":[{"label":"neutral","prob":0.4},{"label":"happy","prob":0.2},{"label":"sad","prob":0.15}],
                "label":"neutral","confidence":0.4,"source":"fallback"}

    def predict_age_gender(self, face_rgb: np.ndarray) -> Dict[str, Any]:
        if TORCH_AVAILABLE and self.models.age_gender_resnet is not None:
            import torch
            pil = Image.fromarray(face_rgb)
            x = self.tf_agegender(pil).unsqueeze(0).to(self.models.device)
            with torch.no_grad():
                age_logits, gender_logits = self.models.age_gender_resnet(x)
                age_probs = torch.softmax(age_logits, dim=1).squeeze(0).cpu().numpy()
                gender_probs = torch.softmax(gender_logits, dim=1).squeeze(0).cpu().numpy()
            age_bin = int(np.argmax(age_probs))
            age_ranges = [(0,9),(10,19),(20,29),(30,39),(40,49),(50,59),(60,69),(70,79),(80,89),(90,120)]
            ar = age_ranges[age_bin]
            gender = "male" if int(np.argmax(gender_probs))==1 else "female"
            return {"age_range": f"{ar[0]}-{ar[1]}", "gender": gender,
                    "age_bin": age_bin, "gender_conf": float(np.max(gender_probs)),
                    "source":"age_gender_resnet18"}
        # fallback
        return {"age_range":"20-29","gender":"unknown","source":"fallback"}

    def estimate_aus(self, face_rgb: np.ndarray) -> Dict[str, Any]:
        # lightweight placeholder; real AU via py-feat can be plugged later
        # produce a stable top list based on simple smile detector (mouth width / brightness)
        gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
        mean = float(gray.mean()/255.0)
        # heuristic
        au12 = max(0.0, min(1.0, (mean-0.35)*1.8))
        au04 = max(0.0, min(1.0, (0.55-mean)*1.4))
        aus = [
            ("AU12", au12),
            ("AU06", max(0.0, min(1.0, au12*0.8))),
            ("AU04", au04),
            ("AU07", max(0.0, min(1.0, 0.2+au04*0.3))),
        ]
        aus.sort(key=lambda x: x[1], reverse=True)
        top = [{"au":k, "name": AU_LABELS.get(k,k), "value": float(v)} for k,v in aus]
        return {"top": top, "source":"heuristic"}

    def analyze(self, image_bytes: bytes) -> Dict[str, Any]:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        rgb = np.array(pil)
        faces = self.detect_faces(rgb)

        face_results = []
        for i,(x,y,w,h,score) in enumerate(faces, start=1):
            face_rgb = self._crop_face(rgb, (x,y,w,h))
            emo = self.predict_emotion(face_rgb)
            ag = self.predict_age_gender(face_rgb)
            aus = self.estimate_aus(face_rgb)
            face_results.append({
                "id": i,
                "bbox": {"x":x,"y":y,"w":w,"h":h,"score":score},
                "emotion": emo,
                "age_gender": ag,
                "aus": aus,
            })

        return {
            "faces_count": len(face_results),
            "faces": face_results,
        }

    def make_thumbnail(self, image_bytes: bytes, size: int = 220) -> bytes:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil.thumbnail((size,size))
        out = io.BytesIO()
        pil.save(out, format="JPEG", quality=85)
        return out.getvalue()

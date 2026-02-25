"""
prepare_utk_mediapipe_crops.py

Скрипт делает кропы лиц из UTKFace тем же способом, что и прод-сервис (MediaPipe FaceDetection + pad),
чтобы уменьшить domain shift (train ≈ prod).

- Вход: папка с UTKFace (обычно data/utkface_aligned_cropped)
- Выход: новая папка с кропами (по умолчанию data/utkface_mediapipe_cropped)
- Имена файлов сохраняются -> разметка age/gender из имени остаётся рабочей.

Пример:
python prepare_utk_mediapipe_crops.py --input data/utkface_aligned_cropped --output data/utkface_mediapipe_cropped --pad 0.15
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

try:
    import mediapipe as mp
except Exception as e:
    print("ERROR: mediapipe is not installed:", repr(e))
    sys.exit(1)

import numpy as np


def iter_images(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def crop_face(pil_img: Image.Image, bbox_xywh, pad: float):
    x, y, w, h = bbox_xywh
    W, H = pil_img.size
    px = int(w * pad)
    py = int(h * pad)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(W, x + w + px)
    y2 = min(H, y + h + py)
    return pil_img.crop((x1, y1, x2, y2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="data/utkface_aligned_cropped", help="Input UTKFace folder")
    ap.add_argument("--output", type=str, default="data/utkface_mediapipe_cropped", help="Output folder for cropped faces")
    ap.add_argument("--pad", type=float, default=0.15, help="Padding fraction around bbox (same as prod)")
    ap.add_argument("--min_conf", type=float, default=0.5, help="Min detection confidence")
    ap.add_argument("--model_selection", type=int, default=0, choices=[0, 1], help="0=short-range,1=full-range")
    ap.add_argument("--fallback", type=str, default="skip", choices=["skip", "copy"], help="What to do if no face detected")
    ap.add_argument("--jpeg_quality", type=int, default=95)
    ap.add_argument("--print_every", type=int, default=1000, help="Progress print interval")
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=args.model_selection, min_detection_confidence=args.min_conf
    )

    total = 0
    saved = 0
    missed = 0

    try:
        for p in iter_images(in_root):
            total += 1
            rel = p.relative_to(in_root)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                pil = Image.open(p).convert("RGB")
            except Exception:
                missed += 1
                continue

            arr = np.array(pil)
            res = detector.process(arr)

            if not res.detections:
                missed += 1
                if args.fallback == "copy":
                    pil.save(out_path.with_suffix(".jpg"), format="JPEG", quality=args.jpeg_quality)
                    saved += 1
                continue

            # take best score
            best = max(res.detections, key=lambda d: float(d.score[0]) if d.score else 0.0)
            bb = best.location_data.relative_bounding_box
            H, W = arr.shape[:2]
            x = int(clamp(bb.xmin, 0, 1) * W)
            y = int(clamp(bb.ymin, 0, 1) * H)
            w = int(clamp(bb.width, 0, 1) * W)
            h = int(clamp(bb.height, 0, 1) * H)

            face = crop_face(pil, (x, y, w, h), pad=args.pad)
            face.save(out_path.with_suffix(".jpg"), format="JPEG", quality=args.jpeg_quality)
            saved += 1

            if args.print_every > 0 and total % args.print_every == 0:
                print(f"[{total}] saved={saved}, missed={missed}")

    finally:
        # release native resources (nice-to-have)
        try:
            detector.close()
        except Exception:
            pass

    print("\nDone.")
    print(f"Total: {total}")
    print(f"Saved: {saved}")
    print(f"No face / errors: {missed}")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()

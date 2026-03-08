import os
import shutil
import argparse
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import mediapipe as mp

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def iter_images(root: str):
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                yield os.path.join(dp, fn)


def relpath_from(root: str, path: str) -> str:
    return os.path.relpath(path, root)


def clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, v)))


def resize_long_side(pil_img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return pil_img
    w, h = pil_img.size
    mx = max(w, h)
    if mx <= max_side:
        return pil_img
    scale = max_side / float(mx)
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return pil_img.resize((nw, nh), Image.BILINEAR)


def crop_with_pad(pil_img: Image.Image, bbox_xywh: Tuple[int, int, int, int], pad: float) -> Image.Image:
    x, y, w, h = bbox_xywh
    W, H = pil_img.size
    px = int(w * pad)
    py = int(h * pad)
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(W, x + w + px)
    y2 = min(H, y + h + py)
    return pil_img.crop((x1, y1, x2, y2))


def detect_best_face_bbox_solutions(
    pil_img: Image.Image,
    detector,
    min_conf: float,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns bbox as (x, y, w, h) in pixels for the "best" face.
    Best = max(score * area) among detections above min_conf.
    """
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)  # H,W,3
    H, W = arr.shape[:2]

    res = detector.process(arr)
    if not res.detections:
        return None

    best = None
    best_val = -1.0

    for det in res.detections:
        score = float(det.score[0]) if det.score else 0.0
        if score < min_conf:
            continue

        bb = det.location_data.relative_bounding_box
        x = clamp_int(bb.xmin * W, 0, W - 1)
        y = clamp_int(bb.ymin * H, 0, H - 1)
        w = clamp_int(bb.width * W, 1, W - x)
        h = clamp_int(bb.height * H, 1, H - y)

        area = float(w * h)
        val = score * area
        if val > best_val:
            best_val = val
            best = (x, y, w, h)

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help=r"например: C:\vk-vision-demo\data\fer2013")
    ap.add_argument("--dst_root", required=True, help=r"например: C:\vk-vision-demo\data\fer2013_mediapipe_cropped")
    ap.add_argument("--pad", type=float, default=0.18)
    ap.add_argument("--min_conf", type=float, default=0.5)
    ap.add_argument("--copy_if_no_face", action="store_true")
    ap.add_argument("--max_side", type=int, default=720, help="resize длинной стороны перед детекцией. 0=выкл")
    ap.add_argument("--model_selection", type=int, default=0, choices=[0, 1],
                    help="0=короткая дистанция (обычно лучше для лиц на FER), 1=дальняя")
    args = ap.parse_args()

    src_root = os.path.abspath(args.src_root)
    dst_root = os.path.abspath(args.dst_root)
    ensure_dir(dst_root)

    mp_fd = mp.solutions.face_detection

    total = cropped = noface = copied = failed = 0

    # FER у тебя: fer2013/train/<class>/..., fer2013/test/<class>/...
    splits = ["train", "test"]

    # создаём один детектор на весь прогон
    with mp_fd.FaceDetection(model_selection=args.model_selection, min_detection_confidence=args.min_conf) as detector:
        for split in splits:
            split_src = os.path.join(src_root, split)
            if not os.path.isdir(split_src):
                print(f"[WARN] split dir not found: {split_src}")
                continue

            files = list(iter_images(split_src))
            for path in tqdm(files, desc=f"{split}", unit="img"):
                total += 1

                rel = relpath_from(src_root, path)  # train\angry\xxx.png
                out_path = os.path.join(dst_root, rel)
                ensure_dir(os.path.dirname(out_path))

                try:
                    img = Image.open(path)
                    img = resize_long_side(img, args.max_side)

                    bbox = detect_best_face_bbox_solutions(img, detector, args.min_conf)
                    if bbox is None:
                        noface += 1
                        if args.copy_if_no_face:
                            shutil.copy2(path, out_path)
                            copied += 1
                        else:
                            # сохраняем как есть
                            img.convert("RGB").save(out_path, quality=95)
                        continue

                    face = crop_with_pad(img, bbox, args.pad)
                    face.convert("RGB").save(out_path, quality=95)
                    cropped += 1

                except Exception as e:
                    failed += 1
                    # на всякий случай стараемся сохранить исходник, чтобы структура не ломалась
                    try:
                        shutil.copy2(path, out_path)
                        copied += 1
                    except Exception:
                        pass
                    if failed <= 5:
                        print(f"[ERR] {path}: {repr(e)}")

    print("\nDone.")
    print(f"src_root: {src_root}")
    print(f"dst_root: {dst_root}")
    print(f"total: {total}")
    print(f"cropped: {cropped}")
    print(f"noface: {noface}")
    print(f"copied: {copied}")
    print(f"failed: {failed}")


if __name__ == "__main__":
    main()
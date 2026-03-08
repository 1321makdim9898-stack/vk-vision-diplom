import os
import shutil
import argparse
from typing import Optional, Tuple

from PIL import Image

import numpy as np
import mediapipe as mp


IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def iter_images(root: str):
    for dp, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                yield os.path.join(dp, fn)


def relpath_from(root: str, path: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")


def resize_long_side(pil_img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return pil_img
    w, h = pil_img.size
    mx = max(w, h)
    if mx <= max_side:
        return pil_img
    scale = max_side / float(mx)
    return pil_img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)


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


def detect_largest_face_bbox_tasks(
    pil_img: Image.Image,
    face_detector,
    min_conf: float
) -> Optional[Tuple[int, int, int, int]]:
    """
    Returns bbox as (x, y, w, h) in pixels for the largest detected face.
    Uses mediapipe.tasks.vision.FaceDetector
    """
    rgb = pil_img.convert("RGB")
    arr = np.array(rgb)
    H, W = arr.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    res = face_detector.detect(mp_image)

    if not res or not getattr(res, "detections", None):
        return None

    best = None
    best_area = -1.0

    for det in res.detections:
        # score
        score = 0.0
        if getattr(det, "categories", None) and len(det.categories) > 0:
            score = float(det.categories[0].score or 0.0)
        if score < min_conf:
            continue

        bb = det.bounding_box  # origin_x, origin_y, width, height (pixels!)
        x = int(clamp(bb.origin_x, 0, W))
        y = int(clamp(bb.origin_y, 0, H))
        w = int(clamp(bb.width, 0, W - x))
        h = int(clamp(bb.height, 0, H - y))

        area = float(w * h)
        if area > best_area and w > 0 and h > 0:
            best_area = area
            best = (x, y, w, h)

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help=r"e.g. C:\vk-vision-demo\data\fer2013")
    ap.add_argument("--dst_root", required=True, help=r"e.g. C:\vk-vision-demo\data\fer2013_mediapipe_cropped")
    ap.add_argument("--pad", type=float, default=0.18)
    ap.add_argument("--min_conf", type=float, default=0.5)
    ap.add_argument("--copy_if_no_face", action="store_true")
    ap.add_argument("--max_side", type=int, default=720, help="resize long side before detection. 0=off")
    args = ap.parse_args()

    src_root = os.path.abspath(args.src_root)
    dst_root = os.path.abspath(args.dst_root)
    ensure_dir(dst_root)

    # --- mediapipe tasks FaceDetector ---
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    base_options = mp_python.BaseOptions(model_asset_path=None)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    face_detector = mp_vision.FaceDetector.create_from_options(options)

    total = cropped = noface = failed = 0

    for split in ["train", "test"]:
        split_src = os.path.join(src_root, split)
        if not os.path.isdir(split_src):
            print(f"[WARN] split dir not found: {split_src}")
            continue

        for path in iter_images(split_src):
            total += 1
            rel = relpath_from(src_root, path)  # train/angry/xxx.jpg
            out_path = os.path.join(dst_root, rel.replace("/", os.sep))
            ensure_dir(os.path.dirname(out_path))

            try:
                img = Image.open(path)
                img = resize_long_side(img, args.max_side)

                bbox = detect_largest_face_bbox_tasks(img, face_detector, args.min_conf)
                if bbox is None:
                    noface += 1
                    if args.copy_if_no_face:
                        shutil.copy2(path, out_path)
                    else:
                        img.convert("RGB").save(out_path, quality=95)
                    continue

                face = crop_with_pad(img, bbox, args.pad)
                face.convert("RGB").save(out_path, quality=95)
                cropped += 1

            except Exception as e:
                failed += 1
                try:
                    shutil.copy2(path, out_path)
                except Exception:
                    pass
                if failed <= 5:
                    print(f"[ERR] {path}: {repr(e)}")

    face_detector.close()

    print("\nDone.")
    print(f"src_root: {src_root}")
    print(f"dst_root: {dst_root}")
    print(f"total: {total}")
    print(f"cropped: {cropped}")
    print(f"noface: {noface}")
    print(f"failed: {failed}")


if __name__ == "__main__":
    main()
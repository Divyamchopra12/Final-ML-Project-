"""
Phone / device-in-hand detection (distraction) – the detectors are trained on the COCO
dataset, which has "cell phone", "remote" (phones are often labelled remote) and "laptop".

Backends (picked automatically, best first):
  yolo        Ultralytics YOLOv8n            – `pip install ultralytics` (downloads yolov8n.pt once)
  lite2       MediaPipe EfficientDet-Lite2    – no extra install, ~7 MB, more accurate
  lite0       MediaPipe EfficientDet-Lite0    – smallest and fastest fallback
The .tflite files are downloaded once into models/.

Each detection = (x0, y0, x1, y1, score, kind) in pixels; kind is "phone" or "laptop".
"""
from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

MP_MODELS = {
    "lite2": ("EfficientDet-Lite2", "efficientdet_lite2.tflite",
              "https://storage.googleapis.com/mediapipe-models/object_detector/"
              "efficientdet_lite2/int8/latest/efficientdet_lite2.tflite"),
    "lite0": ("EfficientDet-Lite0", "efficientdet_lite0.tflite",
              "https://storage.googleapis.com/mediapipe-models/object_detector/"
              "efficientdet_lite0/int8/1/efficientdet_lite0.tflite"),
}
COCO_KIND = {"cell phone": "phone", "remote": "phone", "laptop": "laptop",
             # eating / drinking at the wheel
             "bottle": "drink", "cup": "drink", "wine glass": "alcohol",
             "banana": "food", "apple": "food", "sandwich": "food", "orange": "food", "hot dog": "food",
             "pizza": "food", "donut": "food", "cake": "food", "carrot": "food", "broccoli": "food",
             "bowl": "food", "spoon": "food", "fork": "food"}
YOLO_IDS = {67: "phone", 65: "phone", 63: "laptop",   # COCO ids: cell phone, remote, laptop
            39: "drink", 40: "alcohol", 41: "drink",  # bottle, wine glass, cup
            42: "food", 44: "food", 45: "food", 46: "food", 47: "food", 48: "food", 49: "food",
            50: "food", 51: "food", 52: "food", 53: "food", 54: "food", 55: "food"}
DEVICE_KINDS = ("phone", "laptop")
FOOD_KINDS = ("food", "drink", "alcohol")


class PhoneDetector:
    # a phone held at an angle often scores only 0.25-0.35; a laptop or cup needs a clearer view
    MIN_SCORE = {"phone": 0.3, "laptop": 0.45, "drink": 0.3, "alcohol": 0.3, "food": 0.35}

    def __init__(self, model_dir="models", backend="auto", min_score=0.25):
        self.backend, self.error, self._model = None, None, None
        self.min_score = min_score
        if backend in ("auto", "yolo"):
            try:
                from ultralytics import YOLO

                local = Path(model_dir) / "yolov8n.pt"
                self._model = YOLO(str(local) if local.exists() else "yolov8n.pt")
                self.backend = "YOLOv8n"
                return
            except Exception as exc:  # not installed or no download
                self.error = f"YOLO unavailable ({exc.__class__.__name__})"
                if backend == "yolo":
                    return
        order = ["lite2", "lite0"] if backend in ("auto", "lite2") else ["lite0"]
        for key in order:
            name, fname, url = MP_MODELS[key]
            try:
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                path = Path(model_dir) / fname
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    print(f"[phone] downloading {name} -> {path}")
                    tmp = path.with_suffix(".part")
                    urllib.request.urlretrieve(url, tmp)   # no half-written model on failure
                    tmp.replace(path)
                opts = vision.ObjectDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(path)),
                    running_mode=vision.RunningMode.IMAGE, score_threshold=min_score,
                    category_allowlist=list(COCO_KIND), max_results=8)
                self._model = vision.ObjectDetector.create_from_options(opts)
                self.backend, self.error = name, None
                return
            except Exception as exc:
                self.error = f"{name} unavailable: {exc}"
        self._model = None

    @property
    def ok(self) -> bool:
        return self._model is not None

    def detect(self, frame_bgr: np.ndarray):
        """-> list of (x0, y0, x1, y1, score, kind) in pixels, each kind above its own score."""
        return [b for b in self._detect(frame_bgr) if b[4] >= self.MIN_SCORE.get(b[5], self.min_score)]

    def _detect(self, frame_bgr: np.ndarray):
        if self._model is None:
            return []
        try:
            if self.backend == "YOLOv8n":
                r = self._model(frame_bgr, classes=list(YOLO_IDS), conf=self.min_score, verbose=False, imgsz=480)[0]
                return [(*map(float, b), float(c), YOLO_IDS[int(k)])
                        for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
            import mediapipe as mp

            rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            res = self._model.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            out = []
            for d in res.detections:
                bb, cat = d.bounding_box, d.categories[0]
                out.append((bb.origin_x, bb.origin_y, bb.origin_x + bb.width, bb.origin_y + bb.height,
                            cat.score, COCO_KIND.get(cat.category_name, "phone")))
            return out
        except Exception as exc:
            self.error = str(exc)
            return []


@lru_cache(maxsize=2)
def get_phone_detector(model_dir="models", backend="auto"):
    det = PhoneDetector(model_dir, backend)
    print(f"[phone] backend: {det.backend or 'none'}" + (f" ({det.error})" if det.error else ""))
    return det


def near_face(box, face_box) -> bool:
    """Device held at the ear or in front of the face/chest (face box enlarged)."""
    if face_box is None:
        return True   # face hidden by the phone counts too
    fx0, fy0, fx1, fy1 = face_box
    w, h = fx1 - fx0, fy1 - fy0
    ex0, ey0, ex1, ey1 = fx0 - 1.5 * w, fy0 - 0.5 * h, fx1 + 1.5 * w, fy1 + 2.5 * h
    x0, y0, x1, y1 = box[:4]
    return not (x1 < ex0 or x0 > ex1 or y1 < ey0 or y0 > ey1)


def near_mouth(box, face_box) -> bool:
    """Food / drink held at the mouth: overlaps the lower half of the face (a bottle in the cup
    holder or a snack on the seat does not count)."""
    if face_box is None:
        return False
    fx0, fy0, fx1, fy1 = face_box
    w, h = fx1 - fx0, fy1 - fy0
    mx0, my0, mx1, my1 = fx0 - 0.3 * w, fy0 + 0.45 * h, fx1 + 0.3 * w, fy1 + 0.5 * h
    x0, y0, x1, y1 = box[:4]
    return not (x1 < mx0 or x0 > mx1 or y1 < my0 or y0 > my1)


def counts_as_eating(box, face_box) -> bool:
    kind = box[5] if len(box) > 5 else "phone"
    return kind in FOOD_KINDS and near_mouth(box, face_box)


def counts_as_distraction(box, face_box, frame_w, frame_h) -> bool:
    """A phone anywhere in the driver camera is in the driver's hand; a laptop only near the face."""
    x0, y0, x1, y1 = box[:4]
    if (x1 - x0) * (y1 - y0) < 0.002 * frame_w * frame_h:   # tiny = noise / far background
        return False
    kind = box[5] if len(box) > 5 else "phone"
    if kind in FOOD_KINDS:
        return False                      # handled by counts_as_eating
    return True if kind == "phone" else near_face(box, face_box)

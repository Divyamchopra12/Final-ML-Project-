"""
Core drowsiness and distraction detection logic.

Signals per frame (MediaPipe Face Mesh landmarks):
  - EAR  (Eye Aspect Ratio)    -> eye closure, blinks, PERCLOS
  - MAR  (Mouth Aspect Ratio)  -> yawning
  - Head pose (solvePnP)       -> head nodding (pitch) and looking away (yaw)
  - Optional CNNs (train_model.py) -> eye-closed and yawn probabilities,
    fused with the geometric signals.

Extra features
  - Distraction warning (inspired by the EU ADDW requirement): head turned
    away from the road, or face lost, for longer than `distraction_seconds`.
  - Night mode: dark frames are enhanced with CLAHE before landmark detection.
  - Estimated KSS (Karolinska Sleepiness Scale 1-9) derived from the fatigue
    score. This is an approximation for display, not a validated measurement.
  - eye_mode = "fusion" | "ear" | "cnn" for the ablation study (evaluate.py).

A short calibration (default 3 s, eyes open, looking ahead) personalises the
EAR threshold and the neutral head pitch / yaw.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from phone_detector import counts_as_distraction, counts_as_eating, get_phone_detector
from temporal import FPS as SEQ_FPS, WINDOW as SEQ_WINDOW, frame_vector, load_temporal, predict_window

# MediaPipe Face Mesh landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]   # p1..p6 (corner, top, top, corner, bottom, bottom)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_CORNERS = (61, 291)
MOUTH_VERTICAL = [(81, 178), (13, 14), (311, 402)]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]

# Head pose: nose tip, chin, eye corners (image-left, image-right), mouth corners
POSE_IDX = [1, 152, 33, 263, 57, 287]
MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ],
    dtype=np.float64,
)

DETECTOR_VERSION = "16-Nov v9"   # shown at server start and on the Live page
print(f"[detector] DriveAlert detector {DETECTOR_VERSION}")

# lower = more important. A more important alarm may replace the one on screen at once;
# a less important one waits until the current alarm has been shown for alarm_hold_seconds.
# Only FOUR kinds of alert are given: drowsiness (eyes closed, head drop, getting sleepy), yawning
# (a warning), looking away / looking here and there, and phone use. Hands, cups and bottles are
# still detected, but only to PREVENT wrong alerts (a hand rubbing the eyes is not "eyes closed",
# a bottle at the mouth is not a yawn and not a phone) – they never give an alert of their own.
ALARM_PRIORITY = {"eyes_closed": 0, "head_down": 1, "phone_use": 2, "sleepy": 3, "distracted": 4}

KSS_LABELS = {
    1: "Extremely alert", 2: "Very alert", 3: "Alert", 4: "Rather alert",
    5: "Neither alert nor sleepy", 6: "Some signs of sleepiness",
    7: "Sleepy, no effort to stay awake", 8: "Sleepy, some effort to stay awake",
    9: "Very sleepy, fighting sleep",
}


@dataclass
class DetectorConfig:
    ear_threshold: float = 0.21        # used if calibration is off
    mar_threshold: float = 0.60
    eye_closed_seconds: float = 1.0    # continuous closure -> alarm
    yawn_seconds: float = 1.0          # mouth open this long = one yawn
    head_tilt_degrees: float = 18.0    # pitch change from neutral
    head_down_seconds: float = 1.0     # sustained tilt -> alarm
    distraction_degrees: float = 30.0  # yaw change from neutral = looking away
    distraction_seconds: float = 2.0   # looking away / face lost this long -> alarm
    perclos_window: float = 60.0       # seconds
    alarm_score: float = 70.0          # fatigue score that triggers the alarm
    low_light_enhance: bool = True     # CLAHE night mode
    low_light_threshold: float = 70.0  # mean brightness (0-255) below which frames are enhanced
    dark_threshold: float = 25.0       # below this and no face found -> camera is "blind" (too dark)
    dark_seconds: float = 2.0          # how long before "too dark" is reported
    night_eye_model: bool = True       # use the infrared-trained eye CNN (eye_night.keras) for IR / dark frames
    calibrate: bool = True
    calibration_seconds: float = 3.0
    use_cnn: bool = True
    eye_mode: str = "fusion"           # "fusion" (CNN+EAR) | "ear" | "cnn" | "classical" (best classical ML + EAR)
    fusion_weight: float = 0.6         # fusion rule: w*P(closed) + (1-w)*[EAR < thr] >= fusion_threshold
    fusion_threshold: float = 0.5      # (tuned on your own clips with tune_fusion.py)
    # --- real-world robustness ---
    circadian_risk: bool = True        # more sensitive at 1-5 AM, 2-4 PM and 6 PM-1 AM
    time_on_task_min: float = 90.0     # continuous driving after which fatigue is raised (0 = off)
    speed_gate_kmh: float = 0.0        # no alarms below this GPS speed, e.g. parked (0 = off)
    eye_occlusion_check: bool = True   # sunglasses / hidden eyes -> rely on head + mouth
    predict_window_min: float = 10.0   # window for the fatigue trend prediction
    use_temporal: bool = True          # LSTM/GRU sequence model (train_temporal.py), if trained
    phone_detection: bool = True       # phone-in-hand detection (phone_detector.py)
    glance_degrees: float = 22.0       # a head turn past this (left/right/up) counts as a glance away
    glance_count: int = 3              # this many glances away within glance_window -> warning
    glance_window: float = 10.0        # seconds
    glance_alarm_count: int = 5        # this many glances within glance_window -> full distraction alarm
    hand_detection: bool = True        # hand on face / rubbing eyes (MediaPipe Hands)
    touch_seconds: float = 1.5         # hand on the face this long -> alarm (at least 1 s: hands pass the face all the time)
    phone_seconds: float = 0.5         # phone / device visible this long -> alarm (0 = immediately)
    eat_detection: bool = True         # eating / drinking: food or a cup/bottle at the mouth, hand at a moving mouth
    eat_seconds: float = 1.0           # eating this long -> alarm (0 = immediately)
    drink_seconds: float = 0.0         # drinking (cup / bottle at the mouth) -> alarm at once
    smoke_detection: bool = False      # REMOVED (no cigarette data – a hand gesture gave false alarms); kept so old settings load
    smoke_seconds: float = 0.8         # (unused)
    alarm_hold_seconds: float = 2.0    # an alarm stays on screen at least this long, so it is read fully
    # removed alerts (off; the detection behind them only blocks false alerts now)
    eat_alert: bool = False            # eating / drinking alert
    touch_alert: bool = False          # hand-on-face alert
    fatigue_alert: bool = False        # fatigue-score alert and "early / rising fatigue" banners
    # --- feeling sleepy (before the eyes stay shut): yawns, slow blinks, heavy eyelids, nodding ---
    sleepy_alarm: bool = True
    sleepy_window: float = 180.0       # yawns / nods counted over this many seconds
    sleepy_yawns: int = 2              # this many yawns in the window -> SLEEPY alarm
    long_blink_seconds: float = 0.3    # eyes shut this long (but shorter than an eyes-closed alarm) = slow blink
    sleepy_long_blinks: int = 3        # this many slow blinks within 60 s -> SLEEPY alarm
    droopy_seconds: float = 3.0        # eyelids half closed (heavy eyes) this long -> SLEEPY alarm
    sleepy_show_seconds: float = 4.0   # the SLEEPY alarm is shown this long
    sleepy_gap_seconds: float = 8.0    # at most one SLEEPY alarm per this many seconds
    yawn_model: str = "auto"           # "auto" = best on the test set (Extra Trees, 97.2%) if trained, else CNN


# ------------------------------------------------------------ helpers ----
def eye_aspect_ratio(p: np.ndarray) -> float:
    a = np.linalg.norm(p[1] - p[5])
    b = np.linalg.norm(p[2] - p[4])
    c = np.linalg.norm(p[0] - p[3])
    return float((a + b) / (2.0 * c + 1e-6))


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    vertical = np.mean([np.linalg.norm(pts[a] - pts[b]) for a, b in MOUTH_VERTICAL])
    horizontal = np.linalg.norm(pts[MOUTH_CORNERS[0]] - pts[MOUTH_CORNERS[1]])
    return float(vertical / (horizontal + 1e-6))


def circadian_period(t: float):
    """Time-of-day risk factor. Alertness dips at 1-5 AM and 2-4 PM (IIT Bombay study),
    and most sleep-related crashes in the AIIMS Rishikesh study happened 6 PM - midnight."""
    h = time.localtime(t).tm_hour
    if 1 <= h < 5:
        return 1.25, "Night low (1–5 AM)"
    if 14 <= h < 16:
        return 1.10, "Afternoon dip (2–4 PM)"
    if h >= 18 or h < 1:
        return 1.10, "Evening (6 PM–1 AM)"
    return 1.0, "Daytime"


def eyes_occluded(gray: np.ndarray, pts: np.ndarray) -> bool:
    """Sunglasses check: the eye region is much darker and flatter than the rest of the face."""
    def region(idx, pad):
        p = pts[idx]
        (x0, y0), (x1, y1) = p.min(0), p.max(0)
        w, h = x1 - x0, y1 - y0
        a, b = int(max(x0 - pad * w, 0)), int(max(y0 - pad * h, 0))
        c, d = int(min(x1 + pad * w, gray.shape[1])), int(min(y1 + pad * h, gray.shape[0]))
        return gray[b:d, a:c]
    eyes = np.concatenate([region(LEFT_EYE, 0.3).ravel(), region(RIGHT_EYE, 0.3).ravel()])
    face = region(FACE_OVAL, 0.0)
    if eyes.size < 20 or face.size < 100:
        return False
    ratio = eyes.mean() / (face.mean() + 1e-6)
    return bool(ratio < 0.5 and eyes.std() < 25)


def estimate_kss(score: float) -> int:
    """Rough mapping of the 0-100 fatigue score onto the 1-9 KSS scale."""
    return int(np.clip(round(1 + score / 100 * 8), 1, 9))


def is_infrared(bgr: np.ndarray) -> bool:
    """IR / night-vision cameras give (almost) grey images: the colour channels are nearly equal."""
    small = bgr[::8, ::8].astype(np.int16)
    return float(np.abs(small[:, :, 0] - small[:, :, 1]).mean() + np.abs(small[:, :, 1] - small[:, :, 2]).mean()) < 6.0


def enhance_low_light(bgr: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB + mild gamma lift (night mode)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    lut = np.array([((i / 255.0) ** 0.7) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(out, lut)


def _square_crop(img, x0, y0, x1, y1, scale, size):
    h, w = img.shape[:2]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * scale / 2
    a, b = int(max(cx - half, 0)), int(max(cy - half, 0))
    c, d = int(min(cx + half, w)), int(min(cy + half, h))
    if c - a < 4 or d - b < 4:
        return None
    return cv2.resize(img[b:d, a:c], (size, size), interpolation=cv2.INTER_AREA)


class _Probs:
    """Small stand-in for a tensor: the result of FastModel(...) with .numpy()."""
    def __init__(self, arr):
        self._arr = arr

    def numpy(self):
        return self._arr


class FastModel:
    """A Keras model made ~50-100x faster for live use.

    Calling a Keras model eagerly on one small image costs 120-150 ms on a laptop CPU, which
    limited the live system to about 3 frames a second. At load time the model is converted to
    TensorFlow Lite (about 1-2 ms per call, same answers to 1e-6); if that fails, a compiled
    tf.function is used (about 10 ms); if that fails too, the plain model. Everything else
    (input_shape, count_params, ...) is passed through to the Keras model."""

    def __init__(self, model, path=""):
        import threading

        import tensorflow as tf

        self.model, self.kind = model, "keras"
        self._lock = threading.Lock()          # one interpreter, several sessions/threads
        self._tfl = self._fn = None
        self._shape = None
        try:
            conv = tf.lite.TFLiteConverter.from_keras_model(model)
            self._tfl = tf.lite.Interpreter(model_content=conv.convert(), num_threads=2)
            self._in = self._tfl.get_input_details()[0]["index"]
            self._out = self._tfl.get_output_details()[0]["index"]
            self.kind = "tflite"
        except Exception as exc:               # some TF versions cannot convert the augment layers
            print(f"[detector] TFLite conversion failed for {path}: {exc} – using tf.function")
            try:
                self._fn = tf.function(lambda x: model(x, training=False), reduce_retracing=True)
                self.kind = "tf.function"
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, batch, training=False):
        batch = np.ascontiguousarray(batch, dtype=np.float32)
        if self._tfl is not None:
            with self._lock:
                if self._shape != batch.shape:          # batch of 1 (yawn) or 2 (both eyes)
                    self._tfl.resize_tensor_input(self._in, list(batch.shape))
                    self._tfl.allocate_tensors()
                    self._shape = batch.shape
                self._tfl.set_tensor(self._in, batch)
                self._tfl.invoke()
                return _Probs(self._tfl.get_tensor(self._out).copy())
        if self._fn is not None:
            return _Probs(self._fn(batch).numpy())
        return _Probs(self.model(batch, training=False).numpy())


@lru_cache(maxsize=4)
def load_keras_model(path: str):
    """Load a model once and share it between detector instances (wrapped in FastModel)."""
    if not path or not Path(path).exists():
        return None
    try:
        import os

        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from tensorflow import keras

        model = FastModel(keras.models.load_model(path), path)
        print(f"[detector] loaded {path} ({model.kind})")
        return model
    except Exception as exc:  # TensorFlow missing or incompatible file
        print(f"[detector] could not load {path}: {exc}")
        return None


@lru_cache(maxsize=4)
def load_classical_model(path: str):
    """Best classical model saved by train_classical.py (Random Forest, SVM, ...)."""
    if not path or not Path(path).exists():
        return None
    try:
        import joblib

        bundle = joblib.load(path)
        print(f"[detector] loaded {bundle['name']} from {path}")
        return bundle
    except Exception as exc:
        print(f"[detector] could not load {path}: {exc}")
        return None


# ----------------------------------------------------------- detector ----
class DrowsinessDetector:
    def __init__(self, config: DetectorConfig | None = None,
                 eye_model_path: str | None = "models/eye_model.keras",
                 yawn_model_path: str | None = "models/yawn_model.keras"):
        self.cfg = config or DetectorConfig()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        )
        self.eye_model = load_keras_model(eye_model_path) if self.cfg.use_cnn else None
        self.yawn_model = load_keras_model(yawn_model_path) if self.cfg.use_cnn else None
        base = Path(eye_model_path).parent if eye_model_path else Path("models")
        night_path = base / "eye_night.keras"
        self.night_model = (load_keras_model(str(night_path))
                            if self.cfg.use_cnn and self.cfg.night_eye_model and night_path.exists() else None)
        self.ir_votes = deque(maxlen=30)
        self.ir_mode = False
        self.using_night = False
        self.dark_since = None
        self.classical_eye = load_classical_model(str(base / "eye_classical.joblib"))
        # yawn: the classical Extra Trees model scored 97.2% on the 218 test images against 79.4% for
        # the CNN (McNemar p < 0.0001), so it is used for the yawn probability when it is trained
        self.classical_yawn = (load_classical_model(str(base / "yawn_classical.joblib"))
                               if self.cfg.yawn_model in ("auto", "classical") else None)
        for bundle in (self.classical_yawn, self.classical_eye):
            if bundle is not None and hasattr(bundle["model"], "n_jobs"):
                bundle["model"].n_jobs = 1          # one image at a time: no thread start-up cost
        self.temporal = load_temporal(str(base))
        self.phone = get_phone_detector(str(base)) if self.cfg.phone_detection else None
        self.hands = None                    # MediaPipe Hands, created when hand_detection is on
        self.frame_i = 0
        self.last_pts = None
        self.speed_kmh = None      # set from the phone/laptop GPS by the server
        self.ear_scale = 1.0       # learned from "false alarm" feedback (per driver)
        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self):
        self.start = None
        self.calibrated = not self.cfg.calibrate
        self._calib_ear, self._calib_pitch, self._calib_yaw = [], [], []
        self.ear_thr = self.cfg.ear_threshold * self.ear_scale
        self.base_pitch = None
        self.base_yaw = None
        self.closed_since = None
        self.mouth_open_since = None
        self.yawn_counted = False
        self.tilt_since = None
        self.nod_counted = False
        self.away_since = None
        self.away_counted = False
        self.face_lost_since = None
        self.closed_hist = deque()   # (t, closed)
        self.blinks = deque()
        self.yawns = deque()
        self.nods = deque()
        self.distractions = deque()
        self.glances = deque()               # times the driver glanced away (looking here and there)
        self.hand_boxes = []                 # [(x0, y0, x1, y1, "hand")] in pixels
        self.touch_last = None               # last time a hand was seen on the face
        self.touch_since = None
        self.touch_kind = None               # "eyes" (rubbing eyes) or "face"
        self.touch_present = False
        self.mouth_touch = False             # fingertips at the mouth (eating / drinking with the hand)
        self.food_boxes = []
        self.food_last = None
        self.food_kind = None                # "food" or "drink"
        self.eat_since = None
        self.eat_present = False
        self.eat_kind = None                 # "eating" or "drinking"
        self.eat_events = deque()
        self.eat_last = None
        self.early_since = None              # score has stayed high since (early fatigue warning)
        self.busy_t = None                   # last time a phone / hand / food was in use
        self.smoke_present = False           # smoking detection removed (kept False for old callers)
        self.long_blinks = deque()           # slow blinks (sleepy eyes)
        self.droopy_since = None             # eyelids half closed since
        self.droopy_t = None
        self.sleepy_fire_t = None            # last SLEEPY alarm
        self.sleepy_why = None
        self._hold = None                    # (reason, start time, kinds) of the alarm on screen
        self.need_open = False
        self.mar_hist = deque()              # (t, mar) – chewing = mouth opening and closing
        self.prev_yaw = self.prev_yaw_t = None
        self.last_face_box = None
        self.glance_out = False
        self.sleepy_t = None                 # last time eyes were closed or the head dropped
        self.turned_t = None                 # last time the head was turned to the side
        self.sleepy_since = None
        self.slump_since = None              # face lost right after eyes closed / head drop = asleep
        self.score_samples = deque()        # (t, score) every 5 s for the trend prediction
        self.seq = deque(maxlen=SEQ_WINDOW)  # per-frame vectors for the temporal model
        self.seq_t = self.pred_t = None
        self.temporal_prob = None
        self.phone_boxes = []
        self.phone_since = None
        self.phone_last = None      # last time a device was seen (bridges single missed detections)
        self.phone_present = False
        self.phone_kind = None
        self.phone_hits = deque()   # detection times of a (possibly weak) phone box
        self.phone_counted = False
        self.phone_events = deque()
        self.occl_votes = deque(maxlen=15)  # sunglasses check, majority vote
        self.eyes_visible = True
        self.alarm_events = 0
        self.alarm_times = deque()
        self.alarm_since = None
        self._last_alarm = False

    def recalibrate(self):
        self.reset()

    # -- adaptive personalisation ----------------------------------------
    PROFILE_KEYS = ("eye_closed_seconds", "head_tilt_degrees", "distraction_seconds",
                    "distraction_degrees", "alarm_score")

    def apply_feedback(self, reason: str) -> dict:
        """Driver pressed "False alarm": make that alarm a little harder to trigger."""
        c = self.cfg
        if reason == "eyes_closed":
            # small steps with a floor: many presses must never make real eye closure undetectable
            self.ear_scale = max(0.95, self.ear_scale * 0.98)
            self.ear_thr = max(0.12, self.ear_thr * 0.98)
            c.eye_closed_seconds = min(c.eye_closed_seconds + 0.1, 1.5)
        elif reason == "head_down":
            c.head_tilt_degrees = min(c.head_tilt_degrees + 1, 24)
        elif reason == "distracted":
            c.distraction_seconds = min(c.distraction_seconds + 0.5, 6.0)
            c.distraction_degrees = min(c.distraction_degrees + 3, 50)
        elif reason == "high_fatigue":
            c.alarm_score = min(c.alarm_score + 3, 90)
        return self.profile()

    def profile(self) -> dict:
        return {"ear_scale": round(self.ear_scale, 3),
                **{k: getattr(self.cfg, k) for k in self.PROFILE_KEYS}}

    def load_profile(self, p: dict):
        self.ear_scale = float(np.clip(p.get("ear_scale", 1.0), 0.95, 1.05))   # old profiles: undo over-learning
        for k in self.PROFILE_KEYS:
            if k in p:
                setattr(self.cfg, k, p[k])

    def _update_hands(self, rgb, face_box, t, w, h):
        """Hand on the face: a fingertip inside the face box. Fingertips in the eye band = rubbing
        the eyes (a classic sign of tiredness). Runs every 2nd frame, every frame while touching."""
        if not self.cfg.hand_detection:
            self.touch_present, self.hand_boxes = False, []
            return
        if self.hands is None:
            self.hands = mp.solutions.hands.Hands(max_num_hands=2, model_complexity=0,
                                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
        if face_box is not None:
            self.last_face_box = face_box
        fb = face_box or self.last_face_box       # a hand over the face can hide the face itself
        recent = self.touch_last is not None and t - self.touch_last < 1.0
        if recent or self.frame_i % 2 == 0:
            res = self.hands.process(rgb)
            self.hand_boxes, kind, mouth = [], None, False
            for hl in res.multi_hand_landmarks or []:
                pts = np.array([(p.x * w, p.y * h) for p in hl.landmark])
                self.hand_boxes.append((*pts.min(0), *pts.max(0), "hand"))
                if fb is None:
                    continue
                fx0, fy0, fx1, fy1 = fb
                fw, fh = fx1 - fx0, fy1 - fy0
                tips = pts[[4, 8, 12, 16, 20]]
                # ON the face, not near it: the fingertips must be inside the face itself (a small
                # margin only), and at least 2 of them – one fingertip passing the face edge, a hand
                # raised in front of the camera or adjusting glasses is not "hand on face"
                inside = ((tips[:, 0] > fx0 + 0.05 * fw) & (tips[:, 0] < fx1 - 0.05 * fw) &
                          (tips[:, 1] > fy0) & (tips[:, 1] < fy1 + 0.05 * fh))
                if inside.sum() >= 2:
                    eye_band = (tips[:, 1] > fy0 + 0.2 * fh) & (tips[:, 1] < fy0 + 0.55 * fh) & inside
                    mouth_band = (tips[:, 1] > fy0 + 0.6 * fh) & (tips[:, 1] < fy1 + 0.15 * fh) & inside
                    mouth = mouth or bool(mouth_band.any() and not eye_band.any())
                    kind = "eyes" if eye_band.any() or kind == "eyes" else "face"
            self.mouth_touch = mouth
            if kind:
                self.touch_last, self.touch_kind = t, kind
        self.touch_present = (self.touch_last is not None and t - self.touch_last <= 0.6
                              and not getattr(self, "phone_present", False))   # holding a phone = phone event only
        if self.touch_present:
            self.touch_since = self.touch_since or t
        else:
            self.touch_since = None

    BUSY_SETTLE = 2.5   # seconds after a phone / hand / eating event before eyes and head count again

    def _mark_busy(self, t):
        if getattr(self, "phone_present", False) or self.touch_present or self.eat_present:
            self.busy_t = t
            self.glances.clear()          # head moves while using the phone are not "looking around"
            self.need_open = True         # after the phone: WAKE UP only once the eyes were seen open again

    def _busy_recent(self, t):
        b = getattr(self, "busy_t", None)
        return b is not None and t - b < self.BUSY_SETTLE

    def _update_eating(self, t, mar, pitch_delta):
        """Eating / drinking at the wheel:
          * food, a cup or a bottle (object detector) at the mouth, or
          * fingertips at the mouth while the mouth moves (chewing) or is open (taking a bite).
        Drinking = a cup / bottle, or the hand at the mouth with the head tipped back."""
        if not self.cfg.eat_detection:
            self.eat_present, self.eat_since = False, None
            return
        if mar is not None:
            self.mar_hist.append((t, mar))
        while self.mar_hist and t - self.mar_hist[0][0] > 2.0:
            self.mar_hist.popleft()
        mars = [m for _, m in self.mar_hist]
        chewing = len(mars) >= 6 and (max(mars) - min(mars)) > 0.08
        hand_mouth = self.mouth_touch and self.touch_last is not None and t - self.touch_last <= 0.6
        food_seen = self.food_last is not None and t - self.food_last <= 0.8
        now = food_seen or (hand_mouth and (chewing or (mar is not None and mar > 0.3)))
        if now:
            self.eat_last = t
            drinking = (food_seen and self.food_kind == "drink") or (
                hand_mouth and pitch_delta is not None and pitch_delta < -12)
            self.eat_kind = "drinking" if drinking else "eating"
        present = getattr(self, "eat_last", None) is not None and t - self.eat_last <= 0.6
        if present and not self.eat_present:
            self.eat_events.append(t)
        self.eat_present = present
        self.eat_since = (self.eat_since or t) if present else None
        if present:
            self.touch_present, self.touch_since = False, None   # one event: eating, not "hand on face"

    def close(self):
        if getattr(self, "_det_pool", None) is not None:
            self._det_pool.shutdown(wait=False, cancel_futures=True)
        if self.hands is not None:
            self.hands.close()
        self.face_mesh.close()

    # -- object detection (phone, bottle, cup, food) -----------------------------
    def _timed_detect(self, img):
        t_det = time.perf_counter()
        found = self.phone.detect(img)
        ms = (time.perf_counter() - t_det) * 1000
        self._det_ms = 0.8 * getattr(self, "_det_ms", ms) + 0.2 * ms
        return found

    def _detect_background(self, work):
        """Returns the newest finished detection (or None if none finished since the last frame) and
        starts the next one on the current frame. One worker, so the model is never run twice at once."""
        if getattr(self, "_det_pool", None) is None:
            from concurrent.futures import ThreadPoolExecutor
            self._det_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="objects")
            self._det_future = None
        res, fut = None, self._det_future
        if fut is not None and fut.done():
            try:
                res = fut.result()
            except Exception:
                res = []
            self._det_future = None
        if self._det_future is None:
            self._det_future = self._det_pool.submit(self._timed_detect, work.copy())
        return res

    @property
    def cnn_loaded(self) -> dict:
        return {"eye": self.eye_model is not None,
                "yawn": self.yawn_model is not None or self.classical_yawn is not None,
                "yawn_model": (self.classical_yawn["name"] if self.classical_yawn is not None
                               else "MobileNetV2 CNN" if self.yawn_model is not None else None),
                "night": self.night_model is not None,
                "classical": None if self.classical_eye is None else self.classical_eye["name"]}

    def _eye_model_name(self) -> str:
        mode = self.cfg.eye_mode
        if self.using_night and mode in ("fusion", "cnn"):
            return "Night IR CNN" + (" + EAR" if mode == "fusion" else " only")
        if mode == "classical" and self.classical_eye is not None:
            return f"{self.classical_eye['name']} + EAR"
        if mode == "fusion" and self.eye_model is not None:
            return "MobileNetV2 CNN + EAR"
        if mode == "cnn" and self.eye_model is not None:
            return "MobileNetV2 CNN only"
        return "EAR only"

    # -- models -----------------------------------------------------------
    def _model_size(self, model):
        return int(model.input_shape[1])

    def _eye_closed_prob(self, rgb, pts, model=None, gray=False):
        model = model or self.eye_model
        size = self._model_size(model)
        crops = []
        for idx in (LEFT_EYE, RIGHT_EYE):
            e = pts[idx]
            (x0, y0), (x1, y1) = e.min(0), e.max(0)
            crop = _square_crop(rgb, x0, y0, x1, y1, 1.8, size)
            if crop is not None:
                crops.append(crop)
        if not crops:
            return None
        batch = np.stack(crops).astype(np.float32)
        if gray:   # the night model was trained on grey infrared images
            batch = np.repeat(batch.mean(axis=3, keepdims=True), 3, axis=3)
        return float(model(batch, training=False).numpy().mean())

    def _eye_closed_prob_classical(self, rgb, pts):
        from features import batch_features

        crops = []
        for idx in (LEFT_EYE, RIGHT_EYE):
            e = pts[idx]
            (x0, y0), (x1, y1) = e.min(0), e.max(0)
            crop = _square_crop(rgb, x0, y0, x1, y1, 1.8, 64)
            if crop is not None:
                crops.append(crop)
        if not crops:
            return None
        return float(self.classical_eye["model"].predict_proba(batch_features(crops))[:, 1].mean())

    def _yawn_prob(self, rgb, pts):
        face = pts[FACE_OVAL]
        (x0, y0), (x1, y1) = face.min(0), face.max(0)
        if self.classical_yawn is not None:
            from features import batch_features
            crop = _square_crop(rgb, x0, y0, x1, y1, 1.15, 64)
            if crop is None:
                return None
            return float(self.classical_yawn["model"].predict_proba(batch_features([crop]))[0, 1])
        size = self._model_size(self.yawn_model)
        crop = _square_crop(rgb, x0, y0, x1, y1, 1.15, size)
        if crop is None:
            return None
        return float(self.yawn_model(crop[None].astype(np.float32), training=False).numpy()[0, 0])

    def _head_pose(self, pts, w, h):
        """Head angles in degrees. With only 6 points solvePnP can jump between two answers on a
        perfectly still face (e.g. yaw -17 deg <-> -30 deg); each jump looked like a fast head turn and
        made the detector ignore the eyes and the mouth. Fix: start every frame from the previous
        answer (useExtrinsicGuess), and drop one-frame spikes with a median of the last 3 frames."""
        cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
        img = pts[POSE_IDX].astype(np.float64)
        guess = getattr(self, "_pose_rt", None)
        if guess is not None:
            ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img, cam, np.zeros((4, 1)), guess[0].copy(),
                                          guess[1].copy(), True, cv2.SOLVEPNP_ITERATIVE)
        else:
            ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
            if ok:
                ok, rvec, tvec = cv2.solvePnP(MODEL_POINTS, img, cam, np.zeros((4, 1)), rvec, tvec, True,
                                              cv2.SOLVEPNP_ITERATIVE)
        if not ok or tvec[2, 0] <= 0:
            self._pose_rt = None
            return 0.0, 0.0, 0.0
        self._pose_rt = (rvec, tvec)
        p, y, r = self._angles(rvec)
        hist = getattr(self, "_pose_hist", None)
        if hist is None:
            hist = self._pose_hist = deque(maxlen=3)
        hist.append((p, y, r))
        if len(hist) == 3:
            p, y, r = (float(np.median([v[i] for v in hist])) for i in range(3))
        return p, y, r

    @staticmethod
    def _angles(rvec):
        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = cv2.RQDecomp3x3(rmat)[0]
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        return float(pitch), float(yaw), float(roll)

    def _eyes_closed(self, ear, eye_prob):
        ear_closed = ear < self.ear_thr
        mode = self.cfg.eye_mode
        if mode == "ear" or eye_prob is None:
            return ear_closed
        if mode == "cnn":
            return eye_prob >= 0.5
        # "fusion" (CNN + EAR) and "classical" (classical ML + EAR) use the same rule.
        # EAR veto: when the eye geometry is clearly open or clearly closed, trust it – the image
        # model only decides the unclear middle zone (it can be fooled by blur, light or glasses).
        if ear >= self.ear_thr * 1.15:
            return False
        if ear <= self.ear_thr * 0.85:
            return True
        w = self.cfg.fusion_weight
        return (w * eye_prob + (1 - w) * float(ear_closed)) >= self.cfg.fusion_threshold

    # -- main entry -------------------------------------------------------
    def process(self, frame_bgr: np.ndarray, timestamp: float | None = None) -> dict:
        t = time.time() if timestamp is None else timestamp
        if self.start is None:
            self.start = t
        h, w = frame_bgr.shape[:2]

        brightness = float(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).mean())
        enhanced = self.cfg.low_light_enhance and brightness < self.cfg.low_light_threshold
        work = enhance_low_light(frame_bgr) if enhanced else frame_bgr
        self.ir_votes.append(brightness > 8 and is_infrared(frame_bgr))   # black frames are grey too
        self.ir_mode = sum(self.ir_votes) > len(self.ir_votes) // 2
        light = dict(brightness=brightness, enhanced=enhanced)

        rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        self.frame_i += 1
        face_box = None
        if res.multi_face_landmarks:
            fo = np.array([(res.multi_face_landmarks[0].landmark[i].x * w, res.multi_face_landmarks[0].landmark[i].y * h)
                           for i in FACE_OVAL])
            face_box = (*fo.min(0), *fo.max(0))
        # device detection: every 2nd frame, every frame while a device was seen in the last second
        recent = ((self.phone_last is not None and t - self.phone_last < 1.0)
                  or (self.food_last is not None and t - self.food_last < 1.0))
        # every frame when the computer is fast enough (on-the-spot phone alarm), else every 2nd frame
        fast = getattr(self, "_det_ms", 0.0) < 80.0
        found = None
        if self.phone and self.phone.ok and self.cfg.phone_detection:
            if getattr(self, "_det_ms", 0.0) > 40.0:
                # slow object model (EfficientDet / YOLO on a laptop CPU, 70+ ms): run it in the
                # background on the newest frame, so face, eyes and yawns are checked on EVERY frame
                # without waiting for it; its answer is at most one frame old
                found = self._detect_background(work)
            elif recent or fast or self.frame_i % 2 == 0:
                found = self._timed_detect(work)
        if found is not None:
            self.food_boxes = ([b for b in found if counts_as_eating(b, face_box)]
                               if self.cfg.eat_detection else [])
            if self.food_boxes:
                self.food_last = t
                self.food_kind = max(self.food_boxes, key=lambda b: b[4])[5]
            drinks = [b for b in found if len(b) > 5 and b[5] in ("drink", "food")]
            self.phone_boxes = [b for b in found if counts_as_distraction(b, face_box, w, h)
                                and self._real_phone(b, face_box, drinks, t)]
            if self.phone_boxes:
                self.phone_hits.append(t)
                best = max(b[4] for b in self.phone_boxes)
                while self.phone_hits and t - self.phone_hits[0] > 0.8:
                    self.phone_hits.popleft()
                # a clear phone counts at once; a weak one must be seen twice (one-frame mistakes)
                if best >= 0.6 or len(self.phone_hits) >= 2:
                    self.phone_last = t
                    self.phone_kind = max(self.phone_boxes, key=lambda b: b[4])[5]
        # present = seen within the last 0.8 s, so one missed detection does not reset the timer
        self.phone_present = (self.cfg.phone_detection and self.phone_last is not None
                              and t - self.phone_last <= 0.8)
        if not self.phone_present:
            self.phone_boxes = []
        if self.phone_present:
            if self.phone_since is None:
                self.phone_since, self.phone_counted = t, False
            if t - self.phone_since >= self.cfg.phone_seconds and not self.phone_counted:
                self.phone_events.append(t)
                self.phone_counted = True
        else:
            self.phone_since = None
        self._phone_dur = t - self.phone_since if self.phone_since is not None else 0.0
        self._frame_wh = (w, h)
        self._update_hands(rgb, face_box, t, w, h)
        self._mark_busy(t)

        if not res.multi_face_landmarks:
            self.last_pts = None
            self._pose_rt = None                         # next face: fresh head-pose solve
            self._yaw_hist = None
            if getattr(self, "_pose_hist", None) is not None:
                self._pose_hist.clear()
            self._update_eating(t, None, None)       # a cup / bottle can hide the face while drinking
            self.closed_since = self.mouth_open_since = self.tilt_since = self.away_since = None
            lost = 0.0
            # Too dark to see anything: this is a camera problem, not the driver looking away.
            # Report it instead of raising a false "eyes on road" alarm.
            if brightness < self.cfg.dark_threshold:
                self.dark_since = self.dark_since or t
                self.face_lost_since = None
                return self._result(t, face=False, **light)
            self.dark_since = None
            # Face lost within 2 s of closed eyes or a dropping head: the driver has slumped
            # (asleep), NOT looked away – keep the eyes-closed timer running and escalate.
            # (a blink just before turning the head away is NOT sleep: the eyes must have been
            #  closed / head down for at least 0.4 s, and the head must not have been turning)
            was_sleepy = (self.sleepy_t is not None and t - self.sleepy_t <= 2.0
                          and self.sleepy_since is not None and self.sleepy_t - self.sleepy_since >= 0.4)
            if self.calibrated and not self.eat_present and not self._busy_recent(t) and not (
                    self.turned_t is not None and t - self.turned_t <= 2.0) and (
                    self.slump_since is not None or was_sleepy):
                self.slump_since = self.slump_since or self.sleepy_since or t
                self.face_lost_since = None
                return self._result(t, face=False, closed_dur=t - self.slump_since, slumped=True, **light)
            if self.calibrated and (self.eat_present or self._busy_recent(t)):
                # face behind the bottle / phone, or the driver is just putting it away:
                # that is the phone / eating event, not "looking away" and not sleep
                self.face_lost_since = None
                return self._result(t, face=False, **light)
            if self.calibrated:
                self.face_lost_since = self.face_lost_since or t
                lost = t - self.face_lost_since
                self._feed_temporal(t, frame_vector(face=False))
            return self._result(t, face=False, away_dur=lost, **light)
        self.face_lost_since = None
        self.dark_since = None
        self.slump_since = None

        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
        self.last_pts = pts

        ear = (eye_aspect_ratio(pts[LEFT_EYE]) + eye_aspect_ratio(pts[RIGHT_EYE])) / 2
        mar = mouth_aspect_ratio(pts)
        if self.cfg.eye_occlusion_check:   # sunglasses -> eye signals unreliable
            self.occl_votes.append(eyes_occluded(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), pts))
            self.eyes_visible = sum(self.occl_votes) <= len(self.occl_votes) // 2
        else:
            self.eyes_visible = True
        pitch, yaw, roll = self._head_pose(pts, w, h)
        mode = self.cfg.eye_mode
        if mode == "classical" and self.classical_eye is not None:
            eye_prob = self._eye_closed_prob_classical(rgb, pts)
        elif mode in ("fusion", "cnn") and self.night_model is not None and (self.ir_mode or enhanced):
            eye_prob = self._eye_closed_prob(rgb, pts, self.night_model, gray=True)
        elif mode in ("fusion", "cnn") and self.eye_model is not None:
            eye_prob = self._eye_closed_prob(rgb, pts)
        else:
            eye_prob = None
        self.using_night = eye_prob is not None and self.night_model is not None and (self.ir_mode or enhanced)
        # the yawn model is only needed when the mouth is at least partly open (it only confirms a
        # yawn there), so it is skipped on normal frames – this keeps the live frame rate high
        yawn_prob = (self._yawn_prob(rgb, pts)
                     if (self.yawn_model is not None or self.classical_yawn is not None)
                     and mar > 0.75 * self.cfg.mar_threshold else None)
        signals = dict(ear=ear, mar=mar, pitch=pitch, yaw=yaw, roll=roll,
                       eye_prob=eye_prob, yawn_prob=yawn_prob, **light)

        # Calibration: learn personal open-eye EAR and neutral head pose
        if not self.calibrated:
            self._calib_ear.append(ear)
            self._calib_pitch.append(pitch)
            self._calib_yaw.append(yaw)
            if t - self.start >= self.cfg.calibration_seconds and len(self._calib_ear) >= 10:
                self.ear_thr = float(np.clip(0.72 * np.median(self._calib_ear) * self.ear_scale, 0.12, 0.30))
                self.base_pitch = float(np.median(self._calib_pitch))
                self.base_yaw = float(np.median(self._calib_yaw))
                self.calibrated = True
                self.start = t  # session starts after calibration
            return self._result(t, face=True, calibrating=True, **signals)
        if self.base_pitch is None:
            self.base_pitch, self.base_yaw = pitch, yaw

        # Looking down at a phone makes the eyelids look closed and the head tip forward.
        # That is distraction, not tiredness: while a device is seen, eye closure and head
        # drops are not counted towards fatigue (the phone alarm handles it).
        self._update_eating(t, mar, pitch - self.base_pitch)
        self._mark_busy(t)
        # phone / hand / eating now, OR in the 2 s after it ended: the driver is still lowering the
        # phone and lifting the head – eyes look down and the head is tipped. That is NOT sleep and
        # NOT looking away, so no WAKE UP / EYES ON ROAD straight after the phone alarm.
        on_phone = (bool(getattr(self, "phone_present", False)) or self.touch_present   # hand over the eyes too
                    or self.eat_present or self._busy_recent(t))  # eating / drinking, settling
        # Head turned to the side (looking around): the eyes look narrow from the side and the
        # face may leave the picture – that is distraction, never sleep. Do not count it as fatigue.
        # Also a FAST head movement (looking here and there) counts, even before the angle is large,
        # and the eyes are ignored for 0.7 s after the head moved.
        # speed over the last ~0.3 s, not frame to frame: a few degrees of landmark jitter between two
        # frames is not a head turn (it used to switch the eye and yawn checks off on a still face)
        yh = getattr(self, "_yaw_hist", None)
        if yh is None:
            yh = self._yaw_hist = deque()
        yh.append((t, yaw))
        while len(yh) > 2 and t - yh[1][0] >= 0.3:
            yh.popleft()
        if t - yh[0][0] >= 0.2 and abs(yaw - yh[0][1]) >= 12.0:   # >= 12 degrees within 0.2-0.4 s
            self.turned_t = t
        self.prev_yaw, self.prev_yaw_t = yaw, t
        if abs(yaw - self.base_yaw) > min(self.cfg.glance_degrees, 14):
            self.turned_t = t
        turned = self.turned_t is not None and t - self.turned_t < 0.7

        # Eyes (ignored when hidden, e.g. sunglasses: head pose + mouth still work)
        usable = self.eyes_visible and not on_phone and not turned
        eyes_closed = usable and self._eyes_closed(ear, eye_prob)
        if eyes_closed and pitch - self.base_pitch > 8 and eye_prob is not None and eye_prob < 0.3:
            eyes_closed = False      # looking down (dashboard / gear) makes the eyes LOOK closed
        # hysteresis: once closed, stay closed until the eye clearly re-opens (stops flicker resets)
        # after a phone / hand / eating event the eyes must be seen OPEN once before closure counts:
        # lowering the phone with the eyes still looking down is not falling asleep
        if getattr(self, "need_open", False):
            if usable and ear >= self.ear_thr * 1.15:
                self.need_open = False
            else:
                eyes_closed = False
        if not eyes_closed and usable and self.closed_since is not None and ear < self.ear_thr * 1.04:
            eyes_closed = True
        if eyes_closed:
            self.closed_since = self.closed_since or t
            self.last_closed_t = t
            closed_dur = t - self.closed_since
        elif self.closed_since is not None and usable and t - getattr(self, "last_closed_t", t) < 0.15:
            # one noisy "open" frame does not restart the timer (the time is not counted either)
            closed_dur = self.last_closed_t - self.closed_since
        else:
            if self.closed_since is not None:
                d = t - self.closed_since
                if 0.05 <= d < self.cfg.eye_closed_seconds:
                    self.blinks.append(t)
                # how long the eyes were really shut (without the short grace period above)
                shut = getattr(self, "last_closed_t", self.closed_since) - self.closed_since
                if self.cfg.long_blink_seconds <= shut < self.cfg.eye_closed_seconds:
                    self.long_blinks.append(t)              # slow, heavy blink: a sign of sleepiness
            self.closed_since = None
            closed_dur = 0.0

        # Mouth: yawn when open wide long enough
        # the mouth must really be open (MAR); the yawn CNN alone is not trusted on webcam images
        # (a turned head does NOT block a yawn: seen from the side the mouth looks smaller, never
        # bigger, and people often tip or turn the head while yawning)
        mouth_open = (not on_phone) and (
            mar > self.cfg.mar_threshold or (yawn_prob is not None and yawn_prob > 0.9
                                             and mar > 0.75 * self.cfg.mar_threshold))
        if mouth_open:
            if self.mouth_open_since is None:
                self.mouth_open_since, self.yawn_counted = t, False
            yawn_dur = t - self.mouth_open_since
            if yawn_dur >= self.cfg.yawn_seconds and not self.yawn_counted:
                self.yawns.append(t)
                self.yawn_counted = True
        else:
            self.mouth_open_since = None
            yawn_dur = 0.0
        yawning = yawn_dur >= self.cfg.yawn_seconds

        # Head pitch: sustained change from neutral = nod / head drop
        # head DOWN only (pitch grows when the chin drops); looking up is a glance, not a nod
        drop = pitch - self.base_pitch
        limit = self.cfg.head_tilt_degrees - (4 if self.tilt_since is not None else 0)   # hysteresis
        tilt = drop > limit and not on_phone and not turned
        # a real nod-off comes with drooping eyes; head down with eyes wide open is looking at something
        self._eyes_low = eyes_closed or ear < self.ear_thr * 1.2
        if tilt:
            if self.tilt_since is None:
                self.tilt_since, self.nod_counted = t, False
            tilt_dur = t - self.tilt_since
            if tilt_dur >= 0.5 and not self.nod_counted:
                self.nods.append(t)
                self.nod_counted = True
        else:
            self.tilt_since = None
            tilt_dur = 0.0

        # remember the start of the latest sleepy moment (for the slumped-asleep check above)
        if eyes_closed or tilt:
            self.sleepy_since = self.closed_since or self.tilt_since or t
            self.sleepy_t = t

        # Glances: head turned left/right or up past glance_degrees, even briefly.
        # Many glances in a short time = looking here and there -> early warning.
        d_yaw, d_pitch = abs(yaw - self.base_yaw), pitch - self.base_pitch
        out = d_yaw > self.cfg.glance_degrees or abs(d_pitch) > self.cfg.glance_degrees + 7
        out = out and not on_phone                  # drinking / putting the phone away is not a glance
        if out and not self.glance_out:
            self.glances.append(t)
        self.glance_out = out

        # Head yaw: looking away from the road (distraction)
        away = abs(yaw - self.base_yaw) > self.cfg.distraction_degrees and not on_phone
        if away:
            if self.away_since is None:
                self.away_since, self.away_counted = t, False
            away_dur = t - self.away_since
            if away_dur >= self.cfg.distraction_seconds and not self.away_counted:
                self.distractions.append(t)
                self.away_counted = True
        else:
            self.away_since = None
            away_dur = 0.0

        self._update_sleepy(t, usable and not eyes_closed and not mouth_open and drop < 8
                            and ear < self.ear_thr * 1.15)

        if self.eyes_visible and not on_phone:
            # PERCLOS counts slow closures only – normal blinks (< 0.3 s) are not fatigue
            self.closed_hist.append((t, eyes_closed and closed_dur >= 0.3))
        self._feed_temporal(t, frame_vector(ear, self.ear_thr, mar, pitch - self.base_pitch, yaw - self.base_yaw,
                                            roll, eyes_closed, yawning, True))
        return self._result(t, face=True, eyes_closed=eyes_closed, closed_dur=closed_dur,
                            yawning=yawning, tilt=tilt, tilt_dur=tilt_dur, away=away,
                            away_dur=away_dur, **signals)

    # -- phone or not? ------------------------------------------------------
    def _real_phone(self, box, face_box, drinks, t):
        """The object model sometimes calls a bottle, a cup or a hand at the face a "cell phone"
        (drinking water, covering a yawn, rubbing tired eyes). Those must NOT give the phone alarm."""
        if len(box) > 5 and box[5] != "phone":
            return True                                   # laptop: decided by counts_as_distraction
        x0, y0, x1, y1, score = box[:5]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for d in drinks:                                  # the same object is also seen as a bottle / cup
            if d[0] - 10 <= cx <= d[2] + 10 and d[1] - 10 <= cy <= d[3] + 10:
                return False
        strong = score >= 0.6
        if not strong and self.food_last is not None and t - self.food_last < 1.5:
            return False                                  # drinking a moment ago: a weak "phone" is the bottle
        if face_box is not None and not strong:
            fx0, fy0, fx1, fy1 = face_box
            fw, fh = fx1 - fx0, fy1 - fy0
            in_face = fx0 + 0.15 * fw <= cx <= fx1 - 0.15 * fw and fy0 <= cy <= fy1 + 0.35 * fh
            if in_face:
                return False   # in front of the eyes / mouth / chin: a hand, cup or bottle, not a phone
        return True

    # -- feeling sleepy ------------------------------------------------------
    def _update_sleepy(self, t, droopy_now):
        """SLEEPY alarm: the driver is getting sleepy before the eyes stay shut.
        Signs: repeated yawns, slow (heavy) blinks, eyelids half closed, nodding."""
        c = self.cfg
        if droopy_now:
            if self.droopy_since is None or t - (self.droopy_t or t) > 0.4:   # short gaps are allowed
                self.droopy_since = t
            self.droopy_t = t
        elif self.droopy_since is not None and t - (self.droopy_t or t) > 0.4:
            self.droopy_since = None
        droopy_dur = t - self.droopy_since if self.droopy_since is not None else 0.0
        while self.long_blinks and t - self.long_blinks[0] > 60.0:
            self.long_blinks.popleft()
        if not c.sleepy_alarm:
            return
        yawns = [y for y in self.yawns if t - y <= c.sleepy_window]
        nods = [n for n in self.nods if t - n <= c.sleepy_window]
        lb = len(self.long_blinks)
        why = None
        if len(yawns) >= c.sleepy_yawns:
            why = f"{len(yawns)} yawns in {int(c.sleepy_window // 60)} min"
        elif droopy_dur >= c.droopy_seconds:
            why = "heavy eyelids"
        elif lb >= c.sleepy_long_blinks:
            why = f"{lb} slow blinks in 1 min"
        # (weaker mixes – one yawn plus one slow blink or one nod – were removed: they gave false alarms)
        if why is None:
            return
        signs = yawns[-1:] + nods[-1:] + list(self.long_blinks)[-1:]
        newest = max(signs) if signs else (self.droopy_since or t)
        last = self.sleepy_fire_t
        fresh = (last is None or newest > last
                 or (why == "heavy eyelids" and t - last >= c.sleepy_gap_seconds + c.sleepy_show_seconds))
        if fresh and (last is None or t - last >= c.sleepy_gap_seconds):
            self.sleepy_fire_t, self.sleepy_why = t, why

    def _sleepy_active(self, t):
        return (self.cfg.sleepy_alarm and self.sleepy_fire_t is not None
                and t - self.sleepy_fire_t < self.cfg.sleepy_show_seconds)

    # -- temporal model ----------------------------------------------------
    def _feed_temporal(self, t, vec):
        """Sample at SEQ_FPS; once the window is full, predict P(drowsy) once per second."""
        if not (self.cfg.use_temporal and self.temporal):
            self.temporal_prob = None
            return
        if self.seq_t is None or t - self.seq_t >= 0.9 / SEQ_FPS:
            self.seq.append(vec)
            self.seq_t = t
        if len(self.seq) == SEQ_WINDOW and (self.pred_t is None or t - self.pred_t >= 1.0):
            self.temporal_prob = predict_window(self.temporal, np.stack(self.seq))
            self.pred_t = t

    # -- aggregation ------------------------------------------------------
    def _prune(self, t):
        while self.closed_hist and t - self.closed_hist[0][0] > self.cfg.perclos_window:
            self.closed_hist.popleft()
        for dq, win in ((self.blinks, 60.0), (self.yawns, 300.0), (self.nods, 300.0),
                        (self.distractions, 300.0), (self.phone_events, 300.0), (self.eat_events, 300.0)):
            while dq and t - dq[0] > win:
                dq.popleft()

    def _result(self, t, face, calibrating=False, eyes_closed=False, closed_dur=0.0,
                yawning=False, tilt=False, tilt_dur=0.0, away=False, away_dur=0.0,
                ear=None, mar=None, pitch=None, yaw=None, roll=None, eye_prob=None,
                yawn_prob=None, brightness=None, enhanced=False, slumped=False):
        self._prune(t)
        while self.glances and t - self.glances[0] > self.cfg.glance_window:
            self.glances.popleft()
        # looking here and there: enough glances in the window AND the last one is recent, so the
        # warning clears 3 s after the driver looks ahead again (not 10 s later)
        glance_recent = bool(self.glances) and t - self.glances[-1] <= 3.0
        glancing = face and not calibrating and glance_recent and len(self.glances) >= self.cfg.glance_count
        elapsed = max(t - self.start, 1e-6)
        perclos = (float(np.mean([c for _, c in self.closed_hist]))
                   if self.closed_hist else 0.0)
        # PERCLOS needs a real window: over the first seconds one long closure would look like
        # "eyes closed 30% of the time" and raise a fatigue warning right after WAKE UP
        perclos_span = self.closed_hist[-1][0] - self.closed_hist[0][0] if self.closed_hist else 0.0
        p_ready = perclos_span >= 20.0
        blink_rate = len(self.blinks) * 60.0 / min(max(elapsed, 1.0), 60.0)
        yawns, nods = len(self.yawns), len(self.nods)

        # fatigue score = signals + time-on-task, scaled by the time-of-day risk
        p_perclos = min(perclos / 0.30, 1.0) * 50 if p_ready else 0.0
        p_yawn = min(yawns / 3, 1.0) * 25
        p_nod = min(nods / 3, 1.0) * 25
        drive_min = 0.0 if calibrating else elapsed / 60
        task = self.cfg.time_on_task_min
        p_task = min((drive_min - task) / 30 * 10, 20.0) if task and drive_min > task else 0.0
        circ, period = circadian_period(t) if self.cfg.circadian_risk else (1.0, "Off")
        base = p_perclos + p_yawn + p_nod + p_task
        score = min(base * circ, 100.0)
        # sequence model (LSTM/GRU) opinion: 30% of the score once it has 3 s of data
        p_temporal = None
        if self.temporal_prob is not None and face and not calibrating:
            p_temporal = self.temporal_prob * 100
            if self.temporal_prob >= 0.6:          # only a confident "drowsy" from the sequence model counts
                score = min(0.7 * score + 0.3 * p_temporal, 100.0)
        trend_score = score
        if closed_dur >= 0.3:                      # a blink does not raise the score
            score = max(score, min(closed_dur / self.cfg.eye_closed_seconds, 1.0) * 100)

        # fatigue trend: linear fit over the last N minutes -> minutes until score reaches 60
        if face and not calibrating and (not self.score_samples or t - self.score_samples[-1][0] >= 5):
            self.score_samples.append((t, trend_score))
        while self.score_samples and t - self.score_samples[0][0] > self.cfg.predict_window_min * 60:
            self.score_samples.popleft()
        slope = eta = None
        if len(self.score_samples) >= 6:
            ts = np.array([s[0] for s in self.score_samples])
            ys = np.array([s[1] for s in self.score_samples])
            if ts[-1] - ts[0] >= 30:
                slope = float(np.polyfit((ts - ts[0]) / 60, ys, 1)[0])  # points per minute
                if slope > 1.0 and 25 <= trend_score < 60:   # really rising, not noise around zero
                    eta = (60 - trend_score) / slope

        dark_dur = t - self.dark_since if (self.dark_since is not None and not face) else 0.0
        blind = dark_dur >= self.cfg.dark_seconds
        distracted = self.calibrated and not calibrating and away_dur >= self.cfg.distraction_seconds
        phone_dur = getattr(self, "_phone_dur", 0.0)
        reason = None
        if not calibrating:
            if (face or slumped) and closed_dur >= self.cfg.eye_closed_seconds:
                reason = "eyes_closed"
            elif face and tilt_dur >= self.cfg.head_down_seconds and getattr(self, "_eyes_low", True):
                reason = "head_down"          # chin down WITH drooping eyes; looking down with open eyes is not sleep
            elif getattr(self, "phone_present", False) and phone_dur >= self.cfg.phone_seconds:
                reason = "phone_use"
            elif face and self._sleepy_active(t):
                reason = "sleepy"                                   # getting sleepy: take a break
            elif self.cfg.eat_alert and self.eat_present and t - self.eat_since >= (
                    self.cfg.drink_seconds if self.eat_kind == "drinking" else self.cfg.eat_seconds):
                reason = "eating"
            elif self.cfg.touch_alert and self.touch_present and self.touch_last - self.touch_since >= max(self.cfg.touch_seconds, 1.0) + (
                    1.0 if self.mouth_touch else 0.0):          # hand at the mouth: wait to see if it is eating
                reason = "face_touch"
            elif distracted or (face and glance_recent and len(self.glances) >= self.cfg.glance_alarm_count):
                reason = "distracted"
            elif self.cfg.fatigue_alert and face and trend_score >= self.cfg.alarm_score:
                reason = "high_fatigue"
        # speed gating: no alarms while the vehicle is stopped (GPS speed below the gate)
        gated = bool(self.cfg.speed_gate_kmh and self.speed_kmh is not None
                     and self.speed_kmh < self.cfg.speed_gate_kmh)
        if gated:
            reason = None
        # ONE alarm, shown FULLY: an alarm stays on screen for at least alarm_hold_seconds, and a
        # less important alarm cannot replace it during that time (a more important one can).
        kinds = (self.eat_kind if self.eat_present else None, self.phone_kind, self.touch_kind)
        hold = self._hold
        if gated or calibrating:
            self._hold = None
        elif reason is not None:
            if (hold and hold[0] != reason and t - hold[1] < self.cfg.alarm_hold_seconds
                    and ALARM_PRIORITY.get(reason, 99) >= ALARM_PRIORITY.get(hold[0], 99)):
                reason, kinds = hold[0], hold[2]        # keep the alarm that is being shown
            elif not hold or hold[0] != reason:
                self._hold = (reason, t, kinds)
        elif hold and t - hold[1] < self.cfg.alarm_hold_seconds:
            reason, kinds = hold[0], hold[2]            # finish showing it
        else:
            self._hold = None
        self._kinds = kinds
        alarm = reason is not None
        if alarm and not self._last_alarm:
            self.alarm_events += 1
            self.alarm_times.append(t)
        self._last_alarm = alarm
        self.alarm_since = (self.alarm_since or t) if alarm else None
        while self.alarm_times and t - self.alarm_times[0] > 600:
            self.alarm_times.popleft()

        # Early fatigue only when the score STAYS high for 3 s (no flicker from one noisy moment);
        # it clears when the score falls below 30 (hysteresis).
        if face and not calibrating and trend_score >= 40:
            self.early_since = getattr(self, "early_since", None) or t
        elif trend_score < 30 or not face:
            self.early_since = None
        early = self.cfg.fatigue_alert and self.early_since is not None and t - self.early_since >= 3.0
        if not self.cfg.fatigue_alert:
            eta = None

        # Escalating alarm level (0-3), loosely following the EU DDAW idea of an
        # early warning (KSS 7) and a mandatory warning (KSS 8+):
        #   1 = early warning   : fatigue rising, yawning or looking away
        #   2 = alarm           : microsleep, head drop, distraction or high score
        #   3 = critical alarm  : alarm lasting >= 3 s, or 3+ alarms in 10 minutes
        alarm_seconds = t - self.alarm_since if alarm else 0.0
        sleeping = reason in ("eyes_closed", "head_down")
        high_risk = False
        if alarm and (alarm_seconds >= 3.0 or len(self.alarm_times) >= 3 or slumped or high_risk
                      or (sleeping and alarm_seconds >= 1.0)):
            level = 3                      # asleep: escalate to the siren fast
        elif alarm:
            level = 2
        elif not calibrating and not gated and (getattr(self, "phone_present", False) or blind
                                                or (self.cfg.eat_alert and self.eat_present)):
            level = 1
        elif face and not calibrating and not gated and (early or yawning or (away and away_dur >= 0.3) or glancing
                                                        or (self.cfg.touch_alert and self.touch_present)
                                                        or (eta is not None and eta <= 10)):
            level = 1
        else:
            level = 0

        # ONE event at a time – the most important thing happening now. The alarm, the banner,
        # the voice and the status all use this, so two messages never appear together.
        phone_now = bool(getattr(self, "phone_present", False))
        if calibrating or gated:
            event = None
        elif blind:
            event = "too_dark"
        elif slumped:
            event = "asleep"
        elif reason:
            event = reason                                  # see ALARM_PRIORITY
        elif phone_now:
            event = "phone_use"
        elif self.cfg.eat_alert and self.eat_present:
            event = self.eat_kind                           # "eating" or "drinking"
        elif self.cfg.touch_alert and self.touch_present:
            event = "face_touch"
        elif face and away and away_dur >= 0.3:            # a head tipping for a moment (yawn, mirror) is not looking away
            event = "looking_away"
        elif glancing:
            event = "looking_around"
        elif face and yawning:
            event = "yawning"
        elif face and eta is not None and eta <= 10:
            event = "rising_fatigue"
        elif face and early:
            event = "early_fatigue"
        else:
            event = None

        if calibrating:
            status = "calibrating"
        elif reason in ("distracted", "phone_use", "face_touch", "eating") or event in (
                "phone_use", "face_touch", "looking_away", "looking_around", "eating", "drinking",
                ):
            status = "distracted"
        elif alarm or (face and score >= 60):
            status = "drowsy"
        elif not face and blind:
            status = "too_dark"
        elif not face:
            status = "no_face"
        elif event in ("yawning", "rising_fatigue", "early_fatigue"):
            status = "warning"
        else:
            status = "alert"

        kss = estimate_kss(score)
        rnd = lambda v, n=3: None if v is None else round(float(v), n)  # noqa: E731
        return {
            "status": status,
            "event": event,
            "alarm": alarm,
            "alarm_reason": reason,
            "alarm_level": level,
            "alarm_seconds": round(alarm_seconds, 2),
            "alarms_last_10min": len(self.alarm_times),
            "face_detected": face,
            "calibrating": calibrating,
            "fatigue_score": round(score, 1),
            "est_kss": kss,
            "est_kss_label": KSS_LABELS[kss],
            "ear": rnd(ear),
            "ear_threshold": rnd(self.ear_thr),
            "mar": rnd(mar),
            "eye_closed_prob": rnd(eye_prob),
            "yawn_prob": rnd(yawn_prob),
            "eyes_closed": bool(eyes_closed),
            "eyes_closed_seconds": round(closed_dur, 2),
            "yawning": bool(yawning),
            "sleepy_why": self.sleepy_why if reason == "sleepy" else None,
            "slow_blinks": len(self.long_blinks),
            "yawn_count": yawns,
            "blink_rate": round(blink_rate, 1),
            "perclos": round(perclos, 3),
            "head_pitch": rnd(None if pitch is None or self.base_pitch is None
                              else pitch - self.base_pitch, 1),
            "head_yaw": rnd(None if yaw is None or self.base_yaw is None
                            else yaw - self.base_yaw, 1),
            "head_roll": rnd(roll, 1),
            "head_tilted": bool(tilt),
            "nod_count": nods,
            "looking_away": bool(away or (not face and away_dur > 0)),
            "glancing": bool(glancing),
            "hand_touch": bool(self.touch_present),
            "touch_kind": self.touch_kind if self.touch_present else None,
            "touch_seconds": round(t - self.touch_since, 2) if self.touch_present and self.touch_since else 0.0,
            "hand_boxes": [[round(b[0] / w_, 4), round(b[1] / h_, 4), round(b[2] / w_, 4), round(b[3] / h_, 4), 1.0, "hand"]
                           for b in (self.hand_boxes if self.touch_present else [])
                           for (w_, h_) in [getattr(self, "_frame_wh", (1, 1))]],
            "glances_recent": len(self.glances),
            "slumped": bool(slumped),
            "looking_away_seconds": round(away_dur, 2),
            "distraction_count": len(self.distractions),
            "brightness": rnd(brightness, 1),
            "low_light": bool(brightness is not None and brightness < self.cfg.low_light_threshold),
            "night_mode": bool(enhanced),
            "camera_blind": blind,
            "dark_seconds": round(dark_dur, 1),
            "ir_camera": bool(self.ir_mode),
            "night_model": self.night_model is not None,
            "night_model_active": bool(self.using_night),
            "eyes_visible": bool(self.eyes_visible) if face else None,
            "risk_period": period,
            "circadian_factor": circ,
            "drive_minutes": round(drive_min, 1),
            "score_parts": {"perclos": round(p_perclos, 1), "yawn": round(p_yawn, 1),
                            "nod": round(p_nod, 1), "time_on_task": round(p_task, 1),
                            "temporal": None if p_temporal is None else round(p_temporal, 1)},
            "temporal_prob": None if self.temporal_prob is None else round(self.temporal_prob, 3),
            "phone_detected": bool(getattr(self, "phone_present", False)),
            "phone_kind": self.phone_kind if getattr(self, "phone_present", False) else None,
            "phone_boxes": [[round(b[0] / self._frame_wh[0], 4), round(b[1] / self._frame_wh[1], 4),
                             round(b[2] / self._frame_wh[0], 4), round(b[3] / self._frame_wh[1], 4),
                             round(float(b[4]), 2), b[5]] for b in self.phone_boxes]
                           if getattr(self, "_frame_wh", None) else [],
            "phone_seconds": round(phone_dur, 2),
            "eating": bool(self.eat_present),
            "eat_kind": (getattr(self, "_kinds", (None,))[0] if reason == "eating"
                         else self.eat_kind if self.eat_present else None),
            "high_risk": False,
            "eat_seconds": round(t - self.eat_since, 2) if self.eat_present and self.eat_since else 0.0,
            "food_boxes": [[round(b[0] / self._frame_wh[0], 4), round(b[1] / self._frame_wh[1], 4),
                            round(b[2] / self._frame_wh[0], 4), round(b[3] / self._frame_wh[1], 4),
                            round(float(b[4]), 2), b[5]] for b in (self.food_boxes if self.eat_present else [])]
                          if getattr(self, "_frame_wh", None) else [],
            "phone_backend": self.phone.backend if (self.phone and self.phone.ok) else None,
            "temporal_model": self.temporal["meta"]["best_model"] if (self.temporal and self.cfg.use_temporal) else None,
            "fatigue_trend": None if slope is None else round(slope, 2),
            "predicted_drowsy_min": None if eta is None else round(eta, 1),
            "speed_kmh": None if self.speed_kmh is None else round(self.speed_kmh, 1),
            "speed_gated": gated,
            "alarm_events": self.alarm_events,
            "session_seconds": round(elapsed, 1) if not calibrating else 0.0,
            "cnn": self.cnn_loaded,
            "eye_mode": self.cfg.eye_mode,
            "eye_model_used": self._eye_model_name(),
            "timestamp": t,
        }

    # -- drawing ----------------------------------------------------------
    def annotate(self, frame, r: dict):
        colors = {"alert": (80, 200, 80), "warning": (0, 190, 255),
                  "drowsy": (40, 40, 230), "distracted": (220, 80, 200), "calibrating": (255, 180, 0),
                  "no_face": (160, 160, 160), "too_dark": (60, 60, 200)}
        col = colors[r["status"]]
        if self.last_pts is not None:
            for idx in (LEFT_EYE, RIGHT_EYE):
                cv2.polylines(frame, [self.last_pts[idx].astype(np.int32)], True, col, 1)
            mouth = [61, 81, 13, 311, 291, 402, 14, 178]
            cv2.polylines(frame, [self.last_pts[mouth].astype(np.int32)], True, col, 1)

        for b in self.phone_boxes:
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (200, 80, 220), 2)
            cv2.putText(frame, f"{b[5] if len(b) > 5 else 'phone'} {b[4]:.2f}", (int(b[0]), int(b[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (200, 80, 220), 1)
        cv2.rectangle(frame, (0, 0), (290, 197), (20, 20, 20), -1)
        lines = [
            f"STATUS: {r['status'].upper()}",
            f"Fatigue score: {r['fatigue_score']}  (est. KSS {r['est_kss']})",
            f"EAR: {r['ear']} (thr {r['ear_threshold']})",
            f"MAR: {r['mar']}",
            f"PERCLOS: {r['perclos'] * 100:.0f}%",
            f"Blinks/min: {r['blink_rate']}  Yawns: {r['yawn_count']}",
            f"Pitch: {r['head_pitch']}  Yaw: {r['head_yaw']}",
            f"Night: {'IR cam ' if r['ir_camera'] else ''}{'CLAHE ON' if r['night_mode'] else 'off'}"
            f"{' TOO DARK' if r['camera_blind'] else ''}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 if i else 0.65, col if i == 0 else (235, 235, 235), 1 + (i == 0))
        if r["alarm"]:
            h, w = frame.shape[:2]
            msg = ("PUT THE PHONE DOWN!" if r["alarm_reason"] == "phone_use" else
                   "HAND OFF YOUR FACE!" if r["alarm_reason"] == "face_touch" else
                   ("DRINK LATER - EYES ON ROAD!" if r.get("eat_kind") == "drinking" else "EAT LATER - EYES ON ROAD!")
                   if r["alarm_reason"] == "eating" else
                   "EYES ON ROAD!" if r["alarm_reason"] == "distracted" else "WAKE UP!")
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            cv2.putText(frame, msg, (w // 2 - 160, h - 40), cv2.FONT_HERSHEY_DUPLEX,
                        1.4, (0, 0, 255), 3)
        return frame


def config_dict(cfg: DetectorConfig) -> dict: 
    return asdict(cfg) 

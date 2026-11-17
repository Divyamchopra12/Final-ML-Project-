"""Detector behaviour on synthetic frames (no camera, no trained models needed)."""
import numpy as np

from detector import DetectorConfig, DrowsinessDetector, is_infrared


def make_detector(**kw):
    cfg = DetectorConfig(use_cnn=False, calibrate=False, phone_detection=False, use_temporal=False,
                         circadian_risk=False, **kw)
    return DrowsinessDetector(cfg, None, None)


def test_total_darkness_reports_too_dark_not_distraction():
    det = make_detector()
    black = np.full((240, 320, 3), 3, np.uint8)
    r = None
    for i in range(40):                      # 4 s at 10 fps
        r = det.process(black, timestamp=100 + i * 0.1)
    det.close()
    assert r["status"] == "too_dark"
    assert r["camera_blind"] is True
    assert r["alarm"] is False               # no false "eyes on road" alarm


def test_no_face_in_light_becomes_distraction_alarm():
    det = make_detector()
    grey = np.full((240, 320, 3), 128, np.uint8)
    r = None
    for i in range(40):
        r = det.process(grey, timestamp=100 + i * 0.1)
    det.close()
    assert r["camera_blind"] is False
    assert r["alarm_reason"] == "distracted"


def test_infrared_frame_detection():
    g = np.random.default_rng(0).integers(40, 200, (120, 160), dtype=np.uint8)
    assert is_infrared(np.dstack([g, g, g]))
    colour = np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    assert not is_infrared(colour)


def test_fusion_rule_uses_config_weights_and_ear_veto():
    det = make_detector(eye_mode="fusion", fusion_weight=0.6, fusion_threshold=0.5)
    det.ear_thr = 0.2
    assert det._eyes_closed(0.15, 0.9)          # both say closed
    assert not det._eyes_closed(0.30, 0.2)      # both say open
    assert not det._eyes_closed(0.30, 0.99)     # EAR clearly open -> CNN cannot overrule it
    assert det._eyes_closed(0.20, 0.9)          # unclear EAR: 0.6*0.9 = 0.54 >= 0.5 -> CNN decides
    det.cfg.fusion_weight = 0.4
    assert not det._eyes_closed(0.20, 0.9)      # 0.4*0.9 = 0.36 < 0.5
    det.close()


class FakeDevices:
    """Stands in for the object detector: reports a phone while `on` is True."""
    ok, backend = True, "fake"

    def __init__(self):
        self.on = False

    def detect(self, frame):
        return [(100, 100, 160, 200, 0.8, "phone")] if self.on else []


def test_phone_alarm_is_immediate_and_survives_missed_detections():
    det = make_detector(phone_seconds=0.0)
    det.phone = FakeDevices()
    det.cfg.phone_detection = True
    grey = np.full((240, 320, 3), 128, np.uint8)
    r = det.process(grey, timestamp=10.0)
    assert r["phone_detected"] is False and r["alarm_reason"] != "phone_use"
    det.phone.on = True
    r = det.process(grey, timestamp=10.1)
    r = det.process(grey, timestamp=10.2)
    assert r["phone_detected"] and r["alarm_reason"] == "phone_use" and r["phone_kind"] == "phone"
    assert r["phone_boxes"] and 0 <= r["phone_boxes"][0][0] < 1
    det.phone.on = False                       # one missed detection: still counted
    r = det.process(grey, timestamp=10.4)
    assert r["phone_detected"]
    r = det.process(grey, timestamp=11.5)      # gone for > 0.8 s: no longer detected ...
    assert not r["phone_detected"] and r["alarm_reason"] == "phone_use"   # ... but shown fully for 2 s
    r = det.process(grey, timestamp=12.5)
    assert r["alarm_reason"] != "phone_use"
    det.close()


def test_phone_delay_respected():
    det = make_detector(phone_seconds=1.0)
    det.phone = FakeDevices()
    det.cfg.phone_detection = True
    det.phone.on = True
    grey = np.full((240, 320, 3), 128, np.uint8)
    reasons = [det.process(grey, timestamp=20 + i * 0.1)["alarm_reason"] for i in range(16)]
    assert reasons[3] != "phone_use" and reasons[-1] == "phone_use"
    det.close()


def test_hand_on_face_alarm():
    det = make_detector(touch_seconds=0.5)
    grey = np.full((240, 320, 3), 128, np.uint8)
    det.process(grey, timestamp=30.0)
    # simulate the hand detector having seen fingertips on the eyes
    det.cfg.hand_detection = False            # skip the real MediaPipe Hands on a blank frame
    det.touch_present, det.touch_since, det.touch_last, det.touch_kind = True, 30.0, 31.1, "eyes"
    det._update_hands = lambda *a, **k: None
    r = det.process(grey, timestamp=31.1)        # hand held on the eyes for 1.1 s
    # the hand is still SEEN (it stops the eyes being read as closed) but hand-on-face is no longer an alert
    assert r["hand_touch"] and r["touch_kind"] == "eyes" and r["alarm_reason"] is None and r["event"] is None
    det.close()


def test_head_pose_sign_down_is_positive():
    """Synthetic face: chin towards the chest (looking down) must give a positive pitch."""
    import cv2
    from detector import MODEL_POINTS, POSE_IDX, DrowsinessDetector
    w, h = 640, 480
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], float)
    face = MODEL_POINTS * np.array([1, -1, -1])            # camera frame: y down, nose towards camera

    def pitch_for(deg):
        a = np.radians(deg)
        rot = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        img, _ = cv2.projectPoints(face @ rot.T + [0, 0, 1000.0], np.zeros(3), np.zeros(3), cam, np.zeros(4))
        pts = np.zeros((478, 2))
        for i, k in enumerate(POSE_IDX):
            pts[k] = img.reshape(-1, 2)[i]
        fresh = DrowsinessDetector.__new__(DrowsinessDetector)    # no models needed for the maths
        return fresh._head_pose(pts, w, h)[0]

    assert pitch_for(20) > 10 and pitch_for(-20) < -10     # +20 = forehead forward = looking down


def test_one_event_at_a_time_phone_beats_hand():
    """Holding a phone near the face must be reported as phone use only, not also as hand-on-face."""
    det = make_detector(phone_seconds=0.0)
    det.phone = FakeDevices()
    det.phone.on = True
    det.cfg.phone_detection = True
    det.cfg.hand_detection = False
    grey = np.full((240, 320, 3), 128, np.uint8)
    det.process(grey, timestamp=40.0)
    det.touch_last, det.touch_since, det.touch_kind = 40.1, 40.0, "face"   # the hand holding the phone
    r = det.process(grey, timestamp=40.1)
    assert r["phone_detected"] and r["event"] == "phone_use" and r["alarm_reason"] == "phone_use"
    assert r["hand_touch"] is False
    det.close()


def test_false_alarm_presses_never_make_eye_closure_undetectable():
    det = make_detector()
    for _ in range(50):                          # someone pressed "False alarm" 50 times
        det.apply_feedback("eyes_closed")
    assert det.ear_scale >= 0.95                 # threshold barely moves
    assert det.cfg.eye_closed_seconds <= 1.5     # alarm delay stays short
    det.load_profile({"ear_scale": 0.81})        # an old over-learned profile is repaired
    assert det.ear_scale == 0.95
    det.close()


def test_default_delays_are_short():
    c = DetectorConfig()
    assert c.eye_closed_seconds <= 1.0 and c.head_down_seconds <= 1.0


def test_eating_and_drinking_detection():
    from phone_detector import counts_as_distraction, counts_as_eating

    face = (100, 50, 200, 170)
    bottle_at_mouth = (130, 130, 170, 230, 0.6, "drink")
    bottle_far_away = (400, 300, 430, 380, 0.6, "drink")
    assert counts_as_eating(bottle_at_mouth, face)
    assert not counts_as_eating(bottle_far_away, face)                  # cup holder: ignored
    assert not counts_as_distraction(bottle_at_mouth, face, 640, 480)   # never reported as a phone
    det = make_detector(eat_seconds=0.5)
    det.food_last, det.food_kind = 10.0, "drink"
    det._update_eating(10.0, 0.2, 0.0)
    assert det.eat_present and det.eat_kind == "drinking"
    det.touch_present = True
    det._update_eating(10.3, 0.2, 0.0)
    assert det.touch_present is False                                   # one event: drinking, not hand on face
    # hand at a chewing mouth = eating
    det2 = make_detector()
    det2.mouth_touch, det2.touch_last = True, 20.0
    for i, m in enumerate([0.05, 0.25, 0.05, 0.25, 0.05, 0.25, 0.05]):
        det2._update_eating(20.0 + i * 0.05, m, 0.0)
    assert det2.eat_present and det2.eat_kind == "eating"
    det.close(); det2.close()


def test_early_fatigue_needs_a_steady_high_score():
    det = make_detector(fatigue_alert=True)      # fatigue banners are off by default now; logic kept
    det.calibrated, det.start = True, 0.0
    r = det._result(10.0, face=True, closed_dur=0.2)        # a blink: no score, no warning
    assert r["fatigue_score"] < 35 and r["event"] is None
    det.closed_hist.extend((10.0 + i * 0.75, True) for i in range(40))  # 30 s of slow closures
    r = det._result(40.0, face=True)
    assert r["event"] is None                                # high score, but not yet for 3 s
    r = det._result(43.5, face=True)
    assert r["event"] == "early_fatigue"
    det.close()


def test_no_wake_up_or_eyes_on_road_right_after_the_phone():
    """Putting the phone down: the face is hidden / head still down for a moment. That must not
    become WAKE UP (asleep) or EYES ON ROAD (looking away)."""
    det = make_detector(phone_seconds=0.0, distraction_seconds=1.0)
    det.phone = FakeDevices()
    det.cfg.phone_detection = True
    grey = np.full((240, 320, 3), 128, np.uint8)          # no face visible (phone in front of it)
    det.sleepy_t, det.sleepy_since = 39.5, 38.0           # eyes looked "closed" while looking at the phone
    det.phone.on = True
    for i in range(10):
        r = det.process(grey, timestamp=40 + i * 0.1)
    assert r["alarm_reason"] == "phone_use"
    det.phone.on = False
    reasons = {det.process(grey, timestamp=41.0 + i * 0.1)["alarm_reason"] for i in range(25)}  # next 2.5 s
    assert "eyes_closed" not in reasons and "head_down" not in reasons
    assert not det.process(grey, timestamp=43.0)["slumped"]
    det.close()


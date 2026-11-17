"""
End-to-end alert scenarios, frame by frame at 15 fps.

The camera, MediaPipe and the object detector are replaced by small fakes whose state each test
sets (eye openness, mouth, head angles, hand position, objects in view). Everything else – timers,
priorities, the one-alert-at-a-time rule, holding an alert on screen – is the real detector.

For every scenario we check three things:
  1. the right alert appears, and fast enough,
  2. no other alert or banner is mixed in while it is shown,
  3. it goes away again when the driver stops.
"""
import types

import numpy as np
import pytest

import detector as D

DT = 1 / 15
FACE = (256, 144, 384, 336)          # face box of the fake landmarks, in pixels (640 x 480 frame)
MOUTH = (320, 294)                   # centre of the lips in that box


class _P:
    def __init__(self, x, y):
        self.x, self.y = x, y


rng = np.random.default_rng(1)
LM = [_P(0.4 + 0.2 * rng.random(), 0.3 + 0.4 * rng.random()) for _ in range(478)]
for i in D.FACE_OVAL:                # make the face box exactly FACE
    LM[i] = _P(rng.choice([0.4, 0.6]), rng.choice([0.3, 0.7]))


def hand(tips: dict, base=(330, 420)):
    """21 hand landmarks: all at `base` except the fingertips given as {index: (x, y)} in px."""
    pts = [_P(base[0] / 640, base[1] / 480) for _ in range(21)]
    for i, (x, y) in tips.items():
        pts[i] = _P(x / 640, y / 480)
    return types.SimpleNamespace(landmark=pts)


CHEEK_HAND = hand({8: (270, 230), 12: (300, 200), 4: (262, 250)})


class Sim:
    current = None

    def __init__(self, **cfg):
        Sim.current = self
        self.s = dict(ear=0.27, mar=0.1, pitch=0.0, yaw=0.0, face=True, dark=False, hands=[], objects=[])
        config = D.DetectorConfig(use_cnn=False, use_temporal=False, circadian_risk=False,
                                  eye_occlusion_check=False, eye_closed_seconds=0.5, head_down_seconds=0.6,
                                  distraction_seconds=1.0, distraction_degrees=20, phone_seconds=0,
                                  touch_seconds=0.5, glance_alarm_count=4, **cfg)
        self.det = D.DrowsinessDetector(config, None, None)
        sim = self

        class Mesh:
            def process(self, rgb):
                ok = sim.s["face"] and not sim.s["dark"]
                return types.SimpleNamespace(multi_face_landmarks=[types.SimpleNamespace(landmark=LM)] if ok else None)

            def close(self):
                pass

        class Hands:
            def process(self, rgb):
                return types.SimpleNamespace(multi_hand_landmarks=list(sim.s["hands"]) or None)

            def close(self):
                pass

        class Objects:
            ok, backend = True, "fake"

            def detect(self, frame):
                return list(sim.s["objects"])

        self.det.face_mesh, self.det.hands, self.det.phone = Mesh(), Hands(), Objects()
        self.det._head_pose = lambda pts, w, h: (sim.s["pitch"], sim.s["yaw"], 0.0)
        self.t = 0.0
        self.run(3.5)                                     # calibration + a calm moment

    def run(self, sec, **state):
        self.s.update(state)
        out, end = [], self.t + sec - 1e-9
        while self.t < end:
            img = np.full((480, 640, 3), 5 if self.s["dark"] else 128, np.uint8)
            r = self.det.process(img, timestamp=self.t)
            r["t"] = self.t
            out.append(r)
            self.t = round(self.t + DT, 6)
        return out

    def close(self):
        self.det.close()


@pytest.fixture
def sim(monkeypatch):
    holder = {}
    monkeypatch.setattr(D, "eye_aspect_ratio", lambda p: Sim.current.s["ear"])
    monkeypatch.setattr(D, "mouth_aspect_ratio", lambda p: Sim.current.s["mar"])

    def make(**cfg):
        holder["sim"] = Sim(**cfg)
        return holder["sim"]
    yield make
    if "sim" in holder:
        holder["sim"].close()


def first(rs, key, value):
    return next((r["t"] - rs[0]["t"] for r in rs if r[key] == value), None)


def reasons(rs):
    return {r["alarm_reason"] for r in rs if r["alarm_reason"]}


def events(rs):
    return {r["event"] for r in rs if r["event"]}


# ------------------------------------------------------------------ scenarios
def test_calm_driver_gets_nothing(sim):
    s = sim()
    rs = s.run(5)
    assert not reasons(rs) and not events(rs)


def test_eyes_closed_wake_up_fast_and_alone(sim):
    s = sim()
    rs = s.run(1.5, ear=0.10)
    assert first(rs, "alarm_reason", "eyes_closed") <= 0.5 + 2 * DT
    assert reasons(rs) == {"eyes_closed"} and events(rs) == {"eyes_closed"}
    rs = s.run(4, ear=0.27)
    assert rs[-1]["alarm_reason"] is None


def test_blink_is_not_an_alarm(sim):
    s = sim()
    rs = s.run(0.2, ear=0.10) + s.run(2, ear=0.27)
    assert not reasons(rs)


def test_head_drop_with_drooping_eyes(sim):
    s = sim()
    rs = s.run(1.5, pitch=25, ear=0.21)
    assert first(rs, "alarm_reason", "head_down") <= 0.6 + 2 * DT
    assert reasons(rs) == {"head_down"}


def test_looking_down_with_open_eyes_is_not_sleep(sim):
    s = sim()
    rs = s.run(2.0, pitch=25, ear=0.27)
    assert "head_down" not in reasons(rs) and "eyes_closed" not in reasons(rs)


def test_turning_away_banner_then_eyes_on_road(sim):
    s = sim()
    rs = s.run(2.0, yaw=30, ear=0.15)          # eyes look narrow from the side
    assert first(rs, "event", "looking_away") <= 0.3 + 2 * DT
    assert first(rs, "alarm_reason", "distracted") <= 1.0 + 2 * DT
    assert reasons(rs) == {"distracted"}       # never WAKE UP while turned
    rs = s.run(4, yaw=0, ear=0.27)
    assert rs[-1]["alarm_reason"] is None and rs[-1]["event"] is None


def test_looking_here_and_there(sim):
    s = sim()
    rs = []
    for _ in range(5):
        rs += s.run(0.4, yaw=30) + s.run(0.6, yaw=0)
    assert "looking_around" in events(rs) or "distracted" in reasons(rs)
    assert "distracted" in reasons(rs)
    assert not reasons(rs) - {"distracted"}    # never WAKE UP / anything else
    rs = s.run(6, yaw=0)
    assert rs[-1]["alarm_reason"] is None and rs[-1]["event"] is None   # clears soon after looking ahead


def test_phone_at_once_and_no_wake_up_after(sim):
    s = sim()
    phone = [(100, 100, 160, 200, 0.8, "phone")]
    rs = s.run(1.5, objects=phone, ear=0.12, pitch=20)      # looking down at the phone
    assert first(rs, "alarm_reason", "phone_use") <= DT
    assert reasons(rs) == {"phone_use"}
    after = s.run(1.5, objects=[], ear=0.15, pitch=15) + s.run(2, ear=0.27, pitch=0)
    assert reasons(after) <= {"phone_use"}                  # only the phone alarm finishing
    assert after[-1]["alarm_reason"] is None


def test_drinking_from_a_bottle_gives_no_alert(sim):
    """Drinking is not an alert any more – and the open mouth is not a yawn, the bottle not a phone."""
    s = sim()
    bottle = [(300, 270, 340, 420, 0.7, "drink")]
    rs = s.run(2.0, objects=bottle, mar=0.8)
    assert not reasons(rs) and not events(rs)


def test_bottle_in_the_cup_holder_is_ignored(sim):
    s = sim()
    rs = s.run(2, objects=[(560, 400, 600, 470, 0.9, "drink")])
    assert not reasons(rs) and not events(rs)


def test_hand_on_cheek_gives_no_alert(sim):
    s = sim()
    rs = s.run(2.0, hands=[CHEEK_HAND])
    assert not reasons(rs) and not events(rs)


def test_alert_is_shown_fully_before_it_goes(sim):
    s = sim()
    rs = s.run(0.5, objects=[(100, 100, 160, 200, 0.8, "phone")]) + s.run(3, objects=[])
    on = [r["t"] for r in rs if r["alarm_reason"] == "phone_use"]
    assert on and on[-1] - on[0] >= 2.0 - 2 * DT          # stays at least 2 s
    assert reasons(rs) == {"phone_use"}


def test_more_important_alert_replaces_at_once(sim):
    s = sim()
    rs = s.run(2.0, yaw=30)                                 # EYES ON ROAD showing
    assert "distracted" in reasons(rs)
    rs = s.run(0.5, yaw=0, objects=[(100, 100, 160, 200, 0.8, "phone")])
    assert first(rs, "alarm_reason", "phone_use") <= DT     # the phone does not wait


def test_too_dark_is_not_looking_away(sim):
    s = sim()
    rs = s.run(3, dark=True)
    assert "too_dark" in events(rs) and not reasons(rs)


def test_one_event_per_frame_always(sim):
    """A busy mixture: every frame carries exactly one event and its alarm matches it."""
    s = sim()
    rs = (s.run(0.6, hands=[CHEEK_HAND]) + s.run(0.6, hands=[], objects=[(100, 100, 160, 200, 0.8, "phone")])
          + s.run(0.6, objects=[(300, 270, 340, 420, 0.7, "drink")], hands=[]) + s.run(1, yaw=30, objects=[])
          + s.run(1, yaw=0, ear=0.1))
    for r in rs:
        if r["alarm"]:
            assert r["event"] == r["alarm_reason"]


# ------------------------------------------------------------ feeling sleepy
def test_one_yawn_is_only_a_warning(sim):
    s = sim()
    rs = s.run(1.5, mar=0.8) + s.run(3, mar=0.1)
    assert "yawning" in events(rs) and not reasons(rs)


def test_second_yawn_gives_sleepy_alarm(sim):
    s = sim()
    s.run(1.5, mar=0.8)
    s.run(6, mar=0.1)
    rs = s.run(1.5, mar=0.8)
    t = first(rs, "alarm_reason", "sleepy")
    assert t is not None and t <= 1.0 + 2 * DT              # the moment the 2nd yawn counts
    assert reasons(rs) == {"sleepy"}
    hit = [r for r in rs if r["alarm_reason"] == "sleepy"]
    assert "yawns" in hit[0]["sleepy_why"]
    rs = s.run(6, mar=0.1)
    assert rs[-1]["alarm_reason"] is None                    # shown, then cleared


def test_slow_heavy_blinks_give_sleepy_alarm(sim):
    s = sim()
    rs = []
    for _ in range(3):
        rs += s.run(0.4, ear=0.10) + s.run(2.0, ear=0.27)
    assert "sleepy" in reasons(rs) and "eyes_closed" not in reasons(rs)


def test_normal_blinks_are_not_sleepy(sim):
    s = sim()
    rs = []
    for _ in range(8):
        rs += s.run(0.15, ear=0.10) + s.run(1.5, ear=0.27)
    assert not reasons(rs)


def test_heavy_eyelids_give_sleepy_alarm(sim):
    s = sim()
    rs = s.run(4.0, ear=0.21)                     # half-closed, not shut
    t = first(rs, "alarm_reason", "sleepy")
    assert t is not None and 3.0 - DT <= t <= 3.0 + 3 * DT
    assert reasons(rs) == {"sleepy"}


def test_eyes_shut_while_sleepy_is_wake_up_at_once(sim):
    s = sim()
    s.run(3.2, ear=0.21)                           # sleepy alarm showing
    rs = s.run(1.0, ear=0.10)
    assert first(rs, "alarm_reason", "eyes_closed") <= 0.5 + 2 * DT


# ------------------------------------------------------------ not a phone
def test_bottle_also_called_phone_is_no_alert(sim):
    s = sim()
    objs = [(300, 270, 340, 420, 0.7, "drink"), (302, 275, 338, 410, 0.45, "phone")]
    rs = s.run(1.5, objects=objs)
    assert not reasons(rs)


def test_weak_phone_after_drinking_is_ignored(sim):
    s = sim()
    s.run(0.6, objects=[(300, 270, 340, 420, 0.7, "drink")])
    rs = s.run(1.0, objects=[(300, 300, 345, 420, 0.4, "phone")])   # bottle lowered, misread
    assert "phone_use" not in reasons(rs)


def test_hand_over_a_yawn_is_not_a_phone(sim):
    s = sim()
    rs = s.run(2.0, mar=0.8, objects=[(295, 260, 345, 330, 0.45, "phone")])
    assert "phone_use" not in reasons(rs)


def test_weak_phone_in_the_lap_still_alarms_quickly(sim):
    s = sim()
    rs = s.run(1.0, objects=[(120, 380, 180, 460, 0.4, "phone")])
    assert first(rs, "alarm_reason", "phone_use") <= 2 * DT


# ------------------------------------------------------------ head-pose jitter
def _jitter(s, seed=3, sd_yaw=2.5):
    g = np.random.default_rng(seed)
    s.det._head_pose = lambda pts, w, h: (float(g.normal(0, 1.5)), float(g.normal(0, sd_yaw)), 0.0)


def test_head_jitter_does_not_hide_closed_eyes(sim):
    """A webcam's head angle wobbles a few degrees on a still face; that must not count as a turn."""
    s = sim()
    _jitter(s)
    s.run(2)
    rs = s.run(1.5, ear=0.10)
    assert first(rs, "alarm_reason", "eyes_closed") <= 0.5 + 2 * DT


def test_head_jitter_does_not_hide_a_yawn(sim):
    s = sim()
    _jitter(s)
    s.run(2)
    rs = s.run(1.6, mar=0.8)
    assert "yawning" in events(rs)


def test_real_quick_glance_is_still_seen(sim):
    s = sim()
    rs = []
    for _ in range(5):
        rs += s.run(0.4, yaw=30) + s.run(0.6, yaw=0)
    assert "distracted" in reasons(rs)


# ------------------------------------------------------------ smoking / alcohol removed, stricter hand
def test_glass_at_the_mouth_is_no_alert(sim):
    s = sim()
    rs = s.run(1.0, objects=[(300, 260, 345, 380, 0.6, "drink")])
    assert not reasons(rs) and not any(r["high_risk"] for r in rs)


def test_fingers_at_the_lips_is_not_smoking_any_more(sim):
    s = sim()
    lips = hand({8: (MOUTH[0] - 4, MOUTH[1]), 12: (MOUTH[0] + 6, MOUTH[1] + 3)})
    rs = s.run(2.0, hands=[lips])
    assert "smoking" not in reasons(rs) and "smoking" not in events(rs)


def test_one_fingertip_near_the_face_is_not_a_touch(sim):
    s = sim()
    edge = hand({8: (250, 240)})                      # just outside the face edge
    rs = s.run(2.0, hands=[edge])
    assert "face_touch" not in reasons(rs) and "face_touch" not in events(rs)


def test_short_touch_is_not_an_alarm(sim):
    s = sim()
    rs = s.run(0.8, hands=[CHEEK_HAND]) + s.run(2, hands=[])   # scratching the cheek
    assert "face_touch" not in reasons(rs)


def test_weak_remote_is_not_a_phone(sim):
    import phone_detector as P
    assert P.REMOTE_MIN >= 0.5




# ------------------------------------------------------------ only four kinds of alert
ALLOWED = {"eyes_closed", "head_down", "sleepy", "distracted", "phone_use"}
ALLOWED_EVENTS = ALLOWED | {"yawning", "looking_away", "looking_around", "too_dark", "asleep"}


def test_only_the_four_alert_kinds_ever_appear(sim):
    s = sim()
    rs = (s.run(1.0, hands=[CHEEK_HAND]) + s.run(1.0, hands=[], objects=[(300, 270, 340, 420, 0.7, "drink")], mar=0.8)
          + s.run(1.0, objects=[(300, 260, 345, 380, 0.6, "food")], mar=0.1) + s.run(1.0, objects=[], ear=0.10)
          + s.run(2.0, ear=0.27, yaw=30) + s.run(2.0, yaw=0, objects=[(100, 100, 160, 200, 0.8, "phone")])
          + s.run(2.0, objects=[], mar=0.8) + s.run(4, mar=0.1))
    assert reasons(rs) <= ALLOWED and events(rs) <= ALLOWED_EVENTS


def test_looking_down_with_open_eyes_for_long_is_not_head_drop(sim):
    s = sim()
    rs = s.run(4.0, pitch=25, ear=0.27)            # reading the dashboard: eyes open
    assert "head_down" not in reasons(rs) and "eyes_closed" not in reasons(rs)


def test_one_yawn_and_one_slow_blink_is_not_sleepy(sim):
    s = sim()
    rs = s.run(1.5, mar=0.8) + s.run(2, mar=0.1) + s.run(0.4, ear=0.10) + s.run(3, ear=0.27)
    assert "sleepy" not in reasons(rs)

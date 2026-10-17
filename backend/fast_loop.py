"""EYES — detect(frame) -> list[Detection].  Owner: B.

FAKE: a 'person' box that drifts toward the camera so the hazard path can be tested.
REAL: Ultralytics YOLO + ByteTrack; side from box centre, distance from box height,
closing from box-area growth over the last SMOOTH_FRAMES frames per track.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

import numpy as np

from backend import config
from backend.contracts import Detection

# --------------------------------------------------------------------------- shared geometry
def side_of(box, W) -> str:
    cx = (box[0] + box[2]) / 2 / W
    return "left" if cx < 0.38 else "right" if cx > 0.62 else "centre"


def distance_of(box, H) -> str:
    h = (box[3] - box[1]) / H
    return "near" if h > config.NEAR_FRAC else "mid" if h > config.MID_FRAC else "far"


_hist: dict[int, deque] = defaultdict(lambda: deque(maxlen=config.SMOOTH_FRAMES))


def closing_of(track_id: int, box) -> bool:
    """Box area now vs mean area over the window. Smoothed → no alarm when standing still."""
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    h = _hist[track_id]
    oldest = h[0] if len(h) >= 3 else None
    h.append(area)
    return oldest is not None and area / oldest > config.CLOSING_GROWTH


# --------------------------------------------------------------------------- FAKE
_t0 = time.time()


def detect_fake(frame: np.ndarray) -> list[Detection]:
    H, W = frame.shape[:2]
    # a person walks toward the camera on the left over ~5 s, then resets
    phase = ((time.time() - _t0) % 6) / 6
    h = H * (0.15 + 0.6 * min(1.0, phase * 1.2))
    cx, cy = W * 0.25, H * 0.55
    box = [cx - h / 4, cy - h / 2, cx + h / 4, cy + h / 2]
    out = [{"label": "person", "side": side_of(box, W), "distance": distance_of(box, H),
            "closing": closing_of(1, box), "box": box, "track_id": 1}]
    # a static cup on the right so last_seen has something to find
    cup = [W * 0.72, H * 0.6, W * 0.8, H * 0.72]
    out.append({"label": "cup", "side": "right", "distance": "mid", "closing": False, "box": cup, "track_id": 2})
    return out


# --------------------------------------------------------------------------- REAL
_model = None


def detect_real(frame: np.ndarray) -> list[Detection]:
    global _model
    if _model is None:
        from ultralytics import YOLO
        _model = YOLO(config.YOLO_WEIGHTS)
    H, W = frame.shape[:2]
    r = _model.track(frame, persist=True, verbose=False, device=config.YOLO_DEVICE,
                     tracker="bytetrack.yaml", conf=0.35)[0]
    out: list[Detection] = []
    if r.boxes is None:
        return out
    ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [-1] * len(r.boxes)
    for box, cls, tid in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), ids):
        box = [float(v) for v in box]
        out.append({"label": _model.names[int(cls)], "side": side_of(box, W), "distance": distance_of(box, H),
                    "closing": closing_of(int(tid), box) if tid >= 0 else False, "box": box, "track_id": int(tid)})
    return out


detect = detect_real if config.USE_REAL_YOLO else detect_fake

#!/usr/bin/env python3
import os
import time
import json
import signal
import argparse
import logging
from dataclasses import dataclass, asdict
from collections import deque
from typing import Optional, Dict, Tuple, Any

import cv2
import numpy as np
from picamera2 import Picamera2


TRUST_LEVELS = ("UNKNOWN", "Guest", "Friend", "OWNER")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))

DEFAULT_DETECT_MODEL = os.path.join(_THIS_DIR, "models", "face_detection_yunet.onnx")
DEFAULT_RECOG_MODEL = os.path.join(_THIS_DIR, "models", "face_recognition_sface.onnx")
DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "people", "face_db.json")
DEFAULT_TRUST_MAP_PATH = os.path.join(_PROJECT_ROOT, "data", "people", "trust_map.json")


def now_ts() -> float:
    # Wall clock timestamp (useful for logs & cross-thread correlation)
    return time.time()


def mono() -> float:
    # Monotonic time for scheduling (robust against clock changes)
    return time.monotonic()


def parse_size(size_str: str) -> Tuple[int, int]:
    try:
        w_str, h_str = size_str.lower().split("x")
        w = int(w_str)
        h = int(h_str)
    except ValueError as e:
        raise ValueError(f"Invalid --size format '{size_str}'. Expected WxH (e.g. 640x480).") from e
    if w <= 0 or h <= 0:
        raise ValueError("Camera size must be positive.")
    return w, h


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _load_json_dict(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in '{path}', got {type(data).__name__}")
    return data


@dataclass
class VisionIdentityEvent:
    type: str
    ts: float
    name: str
    trust_level: str
    trust_score: int
    similarity: float
    stable: bool
    face_detected: bool
    faces: int
    bbox: Optional[Tuple[int, int, int, int]]
    last_seen_ts: Optional[float]
    owner_last_seen_ts: Optional[float]
    seconds_since_last_seen: Optional[float]
    seconds_since_owner_seen: Optional[float]


class FaceEngine:
    def __init__(
        self,
        detect_model_path: str,
        recog_model_path: str,
        db_path: str,
        trust_map_path: str,
        match_threshold: float,
        stable_k: int,
        stable_window: int,
        default_known_trust: str = "Guest",
        detector_score_threshold: float = 0.80,
        detector_nms_threshold: float = 0.30,
        reload_every_s: float = 2.0,
        camera_size: Tuple[int, int] = (640, 480),
    ):
        self.detect_model_path = detect_model_path
        self.recog_model_path = recog_model_path
        self.db_path = db_path
        self.trust_map_path = trust_map_path

        self.match_threshold = match_threshold
        self.stable_k = stable_k
        self.stable_window = stable_window
        self.default_known_trust = default_known_trust

        self.detector_score_threshold = detector_score_threshold
        self.detector_nms_threshold = detector_nms_threshold
        self.reload_every_s = max(0.25, reload_every_s)

        self.camera_size = camera_size

        self.db: Dict[str, np.ndarray] = {}
        self.trust_map: Dict[str, str] = {}

        self.recent_names = deque(maxlen=self.stable_window)
        self.last_seen: Dict[str, float] = {}  # wall clock timestamps for each recognized identity
        self._next_reload_mono = 0.0

        self._load_db()
        self._load_trust_map()

        # Models will be initialized once we know frame size
        self.detector = None
        self.recognizer = None

    def _load_db(self) -> None:
        try:
            raw = _load_json_dict(self.db_path)
        except Exception as e:
            logging.warning("Failed loading face DB '%s': %s", self.db_path, e)
            self.db = {}
            return

        out: Dict[str, np.ndarray] = {}
        for name, emb_list in raw.items():
            try:
                out[str(name)] = np.array(emb_list, dtype=np.float32)
            except Exception as e:
                logging.warning("Skipping invalid embedding for '%s': %s", name, e)
        self.db = out

    def _load_trust_map(self) -> None:
        try:
            raw = _load_json_dict(self.trust_map_path)
        except Exception as e:
            logging.warning("Failed loading trust map '%s': %s", self.trust_map_path, e)
            self.trust_map = {}
            return

        out: Dict[str, str] = {}
        for name, level in raw.items():
            lvl = str(level)
            if lvl not in TRUST_LEVELS:
                lvl = "Guest"
            out[str(name)] = lvl
        self.trust_map = out

    def _maybe_reload_data(self) -> None:
        now_m = mono()
        if now_m < self._next_reload_mono:
            return
        self._next_reload_mono = now_m + self.reload_every_s
        self._load_db()
        self._load_trust_map()

    def init_models_for_frame(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        if self.detector is None:
            self.detector = cv2.FaceDetectorYN.create(
                self.detect_model_path,
                "",
                (w, h),
                score_threshold=self.detector_score_threshold,
                nms_threshold=self.detector_nms_threshold,
                top_k=5000,
            )
        if self.recognizer is None:
            self.recognizer = cv2.FaceRecognizerSF.create(self.recog_model_path, "")

    def detect_faces(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        _, faces = self.detector.detect(frame_bgr)
        return faces

    @staticmethod
    def _pick_primary_face(faces: np.ndarray) -> np.ndarray:
        # Pick the largest face (closest to camera)
        return max(faces, key=lambda f: float(f[2] * f[3]))

    def _embedding_for_face(self, frame_bgr: np.ndarray, face: np.ndarray) -> np.ndarray:
        aligned = self.recognizer.alignCrop(frame_bgr, face)
        feat = self.recognizer.feature(aligned)
        return feat.flatten().astype(np.float32)

    def recognize(
        self,
        frame_bgr: np.ndarray,
        faces: Optional[np.ndarray],
    ) -> Tuple[str, float, Optional[Tuple[int, int, int, int]], int]:
        if faces is None or len(faces) == 0:
            return "UNKNOWN", 0.0, None, 0

        primary = self._pick_primary_face(faces)
        x, y, w, h = map(int, primary[:4])
        face_count = int(len(faces))

        # We still report face count/bbox if DB is empty.
        if not self.db:
            return "UNKNOWN", 0.0, (x, y, w, h), face_count

        emb = self._embedding_for_face(frame_bgr, primary)

        best_name = "UNKNOWN"
        best_sim = -1.0
        for name, ref_emb in self.db.items():
            sim = cosine_sim(emb, ref_emb)
            if sim > best_sim:
                best_sim = sim
                best_name = name

        # Apply threshold
        if best_sim < self.match_threshold:
            return "UNKNOWN", float(best_sim), (x, y, w, h), face_count

        return best_name, float(best_sim), (x, y, w, h), face_count

    def _stable_identity(self, name: str) -> bool:
        # Stability gate on recognized names (not UNKNOWN)
        self.recent_names.append(name)
        if name == "UNKNOWN":
            return False
        return sum(1 for n in self.recent_names if n == name) >= self.stable_k

    def _trust_level_for(self, name: str) -> str:
        if name == "UNKNOWN":
            return "UNKNOWN"
        return self.trust_map.get(name, self.default_known_trust)

    def _trust_score(self, name: str, similarity: float, stable: bool) -> int:
        if name == "UNKNOWN":
            return 0

        # Similarity contribution: map [threshold..1] -> [60..100]
        thr = self.match_threshold
        sim_clamped = max(thr, min(1.0, similarity))
        sim_score = 60 + int(40 * (sim_clamped - thr) / max(1e-6, (1.0 - thr)))

        # Stability contribution: if not stable, reduce
        if not stable:
            sim_score = int(sim_score * 0.5)

        # Trust-level cap tweak
        lvl = self._trust_level_for(name)
        if lvl == "Guest":
            sim_score = min(sim_score, 80)
        elif lvl == "Friend":
            sim_score = min(sim_score, 90)
        elif lvl == "OWNER":
            sim_score = min(sim_score, 100)

        return max(1, min(100, sim_score))

    def step(self, frame_bgr: np.ndarray) -> VisionIdentityEvent:
        self._maybe_reload_data()
        self.init_models_for_frame(frame_bgr)
        faces = self.detect_faces(frame_bgr)

        name, sim, bbox, face_count = self.recognize(frame_bgr, faces)
        stable = self._stable_identity(name)

        ts = now_ts()

        # Update last_seen only when stable (prevents brief mis-hits from granting access)
        if stable and name != "UNKNOWN":
            self.last_seen[name] = ts

        # If not stable, report UNKNOWN (as requested)
        report_name = name if stable else "UNKNOWN"
        report_level = self._trust_level_for(name) if stable else "UNKNOWN"
        report_score = self._trust_score(name, sim, stable) if stable else 0

        # Last-seen delta for reported identity
        last_seen_ts = None
        seconds_since_last_seen = None
        if report_name != "UNKNOWN" and report_name in self.last_seen:
            last_seen_ts = self.last_seen[report_name]
            seconds_since_last_seen = max(0.0, ts - last_seen_ts)

        # Owner seen delta (for downstream access-control logic)
        owner_names = [n for n, lvl in self.trust_map.items() if lvl == "OWNER"]
        owner_last_seen_ts = None
        seconds_since_owner_seen = None
        if owner_names:
            owner_last_seen_ts = max((self.last_seen.get(n, 0.0) for n in owner_names), default=0.0)
            if owner_last_seen_ts > 0.0:
                seconds_since_owner_seen = max(0.0, ts - owner_last_seen_ts)
            else:
                owner_last_seen_ts = None

        return VisionIdentityEvent(
            type="VISION_IDENTITY",
            ts=ts,
            name=report_name,
            trust_level=report_level,
            trust_score=int(report_score),
            similarity=float(sim),
            stable=bool(stable),
            face_detected=face_count > 0,
            faces=int(face_count),
            bbox=bbox,
            last_seen_ts=last_seen_ts,
            owner_last_seen_ts=owner_last_seen_ts,
            seconds_since_last_seen=seconds_since_last_seen,
            seconds_since_owner_seen=seconds_since_owner_seen,
        )


class JsonLinePublisher:
    def publish(self, event: VisionIdentityEvent) -> None:
        print(json.dumps(asdict(event), separators=(",", ":"), ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Continuous face recognition service (JSON line emitter)")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--size", type=str, default="640x480", help="camera size WxH")
    ap.add_argument("--match-threshold", type=float, default=0.60)
    ap.add_argument("--stable-k", type=int, default=3)
    ap.add_argument("--stable-window", type=int, default=5)
    ap.add_argument("--default-known-trust", type=str, default="Guest")
    ap.add_argument("--reload-every-s", type=float, default=2.0)
    ap.add_argument("--detector-score-threshold", type=float, default=0.80)
    ap.add_argument("--detector-nms-threshold", type=float, default=0.30)
    ap.add_argument("--detect-model", type=str, default=DEFAULT_DETECT_MODEL)
    ap.add_argument("--recog-model", type=str, default=DEFAULT_RECOG_MODEL)
    ap.add_argument("--db", type=str, default=DEFAULT_DB_PATH)
    ap.add_argument("--trust-map", type=str, default=DEFAULT_TRUST_MAP_PATH)
    args = ap.parse_args()

    if args.default_known_trust not in TRUST_LEVELS:
        raise SystemExit(f"Invalid --default-known-trust '{args.default_known_trust}'. Use one of {TRUST_LEVELS}.")

    cam_size = parse_size(args.size)

    engine = FaceEngine(
        detect_model_path=args.detect_model,
        recog_model_path=args.recog_model,
        db_path=args.db,
        trust_map_path=args.trust_map,
        match_threshold=args.match_threshold,
        stable_k=args.stable_k,
        stable_window=args.stable_window,
        default_known_trust=args.default_known_trust,
        detector_score_threshold=args.detector_score_threshold,
        detector_nms_threshold=args.detector_nms_threshold,
        reload_every_s=args.reload_every_s,
        camera_size=cam_size,
    )

    publisher = JsonLinePublisher()

    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"format": "RGB888", "size": cam_size})
    picam2.configure(cfg)
    picam2.start()
    time.sleep(0.4)  # warmup

    running = True

    def _stop(*_) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    period = 1.0 / max(0.1, args.fps)
    next_tick = mono()

    try:
        while running:
            # Schedule at fixed cadence
            now_m = mono()
            sleep_for = next_tick - now_m
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # If we're late, recover without spiraling.
                next_tick = now_m
            next_tick += period

            try:
                frame_rgb = picam2.capture_array()
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                ev = engine.step(frame_bgr)
                publisher.publish(ev)
            except Exception as e:
                # Fail-safe: never crash the robot main loop
                err_ev = {
                    "type": "VISION_ERROR",
                    "ts": now_ts(),
                    "error": str(e),
                }
                print(json.dumps(err_ev, separators=(",", ":"), ensure_ascii=False), flush=True)
    finally:
        picam2.stop()


if __name__ == "__main__":
    main()

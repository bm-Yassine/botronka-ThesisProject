from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import logging
import os
import queue
import time
from pathlib import Path

import numpy as np

import cv2
from picamera2 import Picamera2

from src.core.message import Message
from src.threads.baseThread import BaseThread
from src.vision.face_service import (
    FaceEngine,
    TRUST_LEVELS,
    DEFAULT_DETECT_MODEL,
    DEFAULT_RECOG_MODEL,
    DEFAULT_DB_PATH,
    DEFAULT_TRUST_MAP_PATH,
)


@dataclass
class VisionThreadConfig:
    width: int = 640
    height: int = 480
    recognition_fps: float = 2.0
    presence_hold_s: float = 5.0
    match_threshold: float = 0.60
    stable_k: int = 3
    stable_window: int = 5
    default_known_trust: str = "Guest"
    detector_score_threshold: float = 0.80
    detector_nms_threshold: float = 0.30
    reload_every_s: float = 2.0
    detect_model_path: str = DEFAULT_DETECT_MODEL
    recog_model_path: str = DEFAULT_RECOG_MODEL
    db_path: str = DEFAULT_DB_PATH
    trust_map_path: str = DEFAULT_TRUST_MAP_PATH
    enroll_samples: int = 3
    enroll_max_attempts: int = 12
    enroll_interval_s: float = 0.25
    captures_dir: str = os.path.join("data", "captures")


class VisionThread(BaseThread):
    def __init__(self, config: VisionThreadConfig, message_queue: queue.Queue[Message]):
        super().__init__(name="VisionThread", queue=message_queue)
        self.config = config
        self.picam2: Picamera2 | None = None
        self._control_inbox: queue.Queue[Message] = queue.Queue()
        self._hold_until_monotonic: float = 0.0
        self._last_face_event_payload: dict | None = None

        if self.config.default_known_trust not in TRUST_LEVELS:
            raise ValueError(
                f"Invalid default_known_trust '{self.config.default_known_trust}'. Allowed: {TRUST_LEVELS}"
            )

        self.engine = FaceEngine(
            detect_model_path=self.config.detect_model_path,
            recog_model_path=self.config.recog_model_path,
            db_path=self.config.db_path,
            trust_map_path=self.config.trust_map_path,
            match_threshold=self.config.match_threshold,
            stable_k=self.config.stable_k,
            stable_window=self.config.stable_window,
            default_known_trust=self.config.default_known_trust,
            detector_score_threshold=self.config.detector_score_threshold,
            detector_nms_threshold=self.config.detector_nms_threshold,
            reload_every_s=self.config.reload_every_s,
            camera_size=(self.config.width, self.config.height),
        )

    @staticmethod
    def _load_json_object(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object at {path}")
        return data

    @staticmethod
    def _save_json_object(path: str, data: dict):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp.replace(p)

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        return v / (n + 1e-9)

    def _register_identity(self, name: str, trust_level: str) -> dict:
        if self.picam2 is None:
            raise RuntimeError("camera is not initialized")

        embeddings: list[np.ndarray] = []
        last_frame_bgr = None

        for _attempt in range(1, max(1, self.config.enroll_max_attempts) + 1):
            frame_rgb = self.picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            last_frame_bgr = frame_bgr

            self.engine.init_models_for_frame(frame_bgr)
            faces = self.engine.detect_faces(frame_bgr)
            if faces is None or len(faces) == 0:
                time.sleep(max(0.0, self.config.enroll_interval_s))
                continue

            primary = self.engine._pick_primary_face(faces)
            emb = self.engine._embedding_for_face(frame_bgr, primary)
            embeddings.append(self._l2_normalize(emb))

            if len(embeddings) >= max(1, self.config.enroll_samples):
                break

            time.sleep(max(0.0, self.config.enroll_interval_s))

        if len(embeddings) < max(1, self.config.enroll_samples):
            raise RuntimeError(
                f"captured only {len(embeddings)}/{self.config.enroll_samples} face samples"
            )

        avg = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        avg = self._l2_normalize(avg)

        db = self._load_json_object(self.config.db_path)
        trust_map = self._load_json_object(self.config.trust_map_path)
        db[name] = avg.tolist()
        trust_map[name] = trust_level

        self._save_json_object(self.config.db_path, db)
        self._save_json_object(self.config.trust_map_path, trust_map)

        capture_path = None
        if last_frame_bgr is not None:
            capture_dir = Path(self.config.captures_dir)
            if not capture_dir.is_absolute():
                capture_dir = Path.cwd() / capture_dir
            capture_dir.mkdir(parents=True, exist_ok=True)
            capture_path = capture_dir / f"enroll_{name}_{int(time.time())}.jpg"
            cv2.imwrite(str(capture_path), last_frame_bgr)

        # Refresh in-memory engine state right away
        self.engine._load_db()
        self.engine._load_trust_map()

        return {
            "ok": True,
            "name": name,
            "trust_level": trust_level,
            "samples": len(embeddings),
            "capture_path": str(capture_path) if capture_path is not None else None,
            "ts": time.time(),
        }

    def _handle_register_request(self, message: Message):
        try:
            payload = json.loads(message.content)
            name = str(payload.get("name", "")).strip()
            trust_level = str(payload.get("trust_level", "Guest")).strip()
            if not name:
                raise ValueError("name is required")
            if trust_level not in TRUST_LEVELS:
                trust_level = "Guest"
            result = self._register_identity(name=name, trust_level=trust_level)
        except Exception as e:
            result = {
                "ok": False,
                "name": None,
                "trust_level": None,
                "error": str(e),
                "ts": time.time(),
            }

        self.broadcast_message(
            "vision_register_result",
            json.dumps(result, ensure_ascii=False),
        )

    def _drain_control_messages(self):
        while True:
            try:
                msg = self._control_inbox.get_nowait()
            except queue.Empty:
                break

            if msg.type == "vision_register_request":
                self._handle_register_request(msg)

    def handle_message(self, message: Message):
        if message.type == "vision_register_request":
            self._control_inbox.put(message)

    def _start_camera(self) -> None:
        self.picam2 = Picamera2()
        cam_size = (self.config.width, self.config.height)
        cfg = self.picam2.create_preview_configuration(main={"format": "RGB888", "size": cam_size})
        self.picam2.configure(cfg)
        self.picam2.start()
        time.sleep(0.4)

    def _stop_camera(self) -> None:
        if self.picam2 is not None:
            self.picam2.stop()
            self.picam2 = None

    def run(self) -> None:
        try:
            self._start_camera()
            period = 1.0 / max(0.1, self.config.recognition_fps)
            next_tick = time.monotonic()

            while self.running:
                now_m = time.monotonic()
                sleep_for = next_tick - now_m
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = now_m
                next_tick += period

                try:
                    self._drain_control_messages()

                    frame_rgb = self.picam2.capture_array()
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    event = self.engine.step(frame_bgr)
                    payload_obj = asdict(event)

                    now_m = time.monotonic()
                    if event.face_detected:
                        self._hold_until_monotonic = now_m + max(0.0, self.config.presence_hold_s)
                        self._last_face_event_payload = dict(payload_obj)
                    elif (
                        self._last_face_event_payload is not None
                        and now_m < self._hold_until_monotonic
                    ):
                        payload_obj = dict(self._last_face_event_payload)
                        payload_obj["ts"] = time.time()
                        payload_obj["face_detected"] = True

                    payload = json.dumps(payload_obj, separators=(",", ":"), ensure_ascii=False)

                    self.broadcast_message("vision_identity", payload)
                    logging.debug(
                        "Vision identity: face=%s name=%s trust=%s sim=%.3f",
                        event.face_detected,
                        event.name,
                        event.trust_level,
                        event.similarity,
                    )
                except Exception as e:
                    err_event = {
                        "type": "VISION_ERROR",
                        "ts": time.time(),
                        "error": str(e),
                    }
                    self.broadcast_message(
                        "vision_error",
                        json.dumps(err_event, separators=(",", ":"), ensure_ascii=False),
                    )
                    logging.exception("VisionThread step failed")
        finally:
            self._stop_camera()

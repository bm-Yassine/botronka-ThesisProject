from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import logging
import queue
import time

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
    recognition_fps: float = 5.0
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


class VisionThread(BaseThread):
    def __init__(self, config: VisionThreadConfig, queue: queue.Queue[Message]):
        super().__init__(name="VisionThread", queue=queue)
        self.config = config
        self.picam2: Picamera2 | None = None

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
                    frame_rgb = self.picam2.capture_array()
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    event = self.engine.step(frame_bgr)
                    payload = json.dumps(asdict(event), separators=(",", ":"), ensure_ascii=False)

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

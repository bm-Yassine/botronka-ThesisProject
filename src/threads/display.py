from __future__ import annotations
from dataclasses import dataclass
import queue
import logging
import json
import time
from typing import Optional

from src.threads.baseThread import BaseThread
from src.core.state import Emotion, TrustLevel
from src.core.message import Message
from src.hardware.oledDisplay import OledDisplay, OledConfig


@dataclass
class BehaviorConfig:
    lonely_after_s: float
    stuck_distance_cm: float
    stuck_after_s: float
    face_check_hz: float


class DisplayThread(BaseThread):
    def __init__(
        self,
        config: OledConfig,
        behaviorConfig: BehaviorConfig,
        queue: queue.Queue[Message],
    ):
        super().__init__(name="OledDisplay", queue=queue)

        self.display = OledDisplay(config)

        self.lonely_after: float = behaviorConfig.lonely_after_s
        self.stuck_cm: float = behaviorConfig.stuck_distance_cm
        self.stuck_after: float = behaviorConfig.stuck_after_s
        self.face_check_hz: float = behaviorConfig.face_check_hz

        self.recognition_error = False
        self.distance_cm = None
        self.last_face_ts = 0.0
        self.face_detected = False
        self.face_present = False
        self.trust: TrustLevel = TrustLevel.UNKNOWN
        self.last_seen_name: str = "UNKNOWN"
        self.last_similarity: float = 0.0
        self.last_seen_ts: Optional[float] = None
        self.owner_last_seen_ts: Optional[float] = None
        self.seconds_since_last_seen: Optional[float] = None
        self.seconds_since_owner_seen: Optional[float] = None
        self.last_seen_by_name: dict[str, float] = {}
        self.stuck_since = None

    def run(self):
        while self.running:
            now = time.monotonic()
            emotion = self.decide_emotion(now)

            subtitle_parts: list[str] = []
            if self.distance_cm is not None:
                subtitle_parts.append(f"{self.distance_cm:.0f}cm")
            if self.face_detected:
                subtitle_parts.append(self.last_seen_name)
                subtitle_parts.append(self.trust.name)
            subtitle = " ".join(subtitle_parts)

            logging.debug(
                "Display emotion=%s face=%s name=%s trust=%s dist=%s",
                emotion.name,
                self.face_detected,
                self.last_seen_name,
                self.trust.name,
                self.distance_cm,
            )
            self.display.draw(
                emotion,
                subtitle,
            )

            # Track edge for GREETING decision
            self.face_present = self.face_detected
            time.sleep(0.2)

    def handle_message(self, message: Message):
        if message.type == "distance_cm":
            try:
                data = json.loads(message.content)
                self.distance_cm = data.get("value")
            except json.JSONDecodeError:
                logging.error(
                    "Failed to decode distance_cm message: %s", message.content
                )
        elif message.type == "vision_identity":
            try:
                data = json.loads(message.content)
                self.face_detected = bool(data.get("face_detected", False))
                self.last_seen_name = str(data.get("name", "UNKNOWN"))
                self.last_similarity = float(data.get("similarity", 0.0))
                self.last_seen_ts = data.get("last_seen_ts")
                self.owner_last_seen_ts = data.get("owner_last_seen_ts")
                self.seconds_since_last_seen = data.get("seconds_since_last_seen")
                self.seconds_since_owner_seen = data.get("seconds_since_owner_seen")

                if (
                    self.last_seen_name != "UNKNOWN"
                    and isinstance(self.last_seen_ts, (int, float))
                ):
                    self.last_seen_by_name[self.last_seen_name] = float(self.last_seen_ts)

                trust_level = str(data.get("trust_level", "UNKNOWN"))
                self.trust = {
                    "OWNER": TrustLevel.OWNER,
                    "Friend": TrustLevel.FRIEND,
                    "Guest": TrustLevel.GUEST,
                    "UNKNOWN": TrustLevel.UNKNOWN,
                }.get(trust_level, TrustLevel.UNKNOWN)

                if self.face_detected:
                    self.last_face_ts = time.monotonic()

                logging.info(
                    "Vision->Display identity: name=%s trust=%s similarity=%.3f "
                    "last_seen_ts=%s since_last_seen=%s owner_last_seen_ts=%s owner_seen_delta=%s",
                    self.last_seen_name,
                    trust_level,
                    self.last_similarity,
                    self.last_seen_ts,
                    self.seconds_since_last_seen,
                    self.owner_last_seen_ts,
                    self.seconds_since_owner_seen,
                )
            except Exception:
                logging.exception("Failed to process vision_identity message: %s", message.content)
        elif message.type == "vision_error":
            self.recognition_error = True
            logging.error("Vision error message: %s", message.content)

    def decide_emotion(self, now: float) -> Emotion:
        if self.recognition_error:
            return Emotion.ANGRY

        # LONELY
        if (now - self.last_face_ts) > self.lonely_after:
            return Emotion.LONELY

        # STUCK based on distance
        d = self.distance_cm

        if d is not None and d < self.stuck_cm:
            if self.stuck_since is None:
                self.stuck_since = now
            elif (now - self.stuck_since) > self.stuck_after:
                return Emotion.STUCK
        else:
            self.stuck_since = None

        # SUSPICIOUS vs HAPPY (v0: any face can be suspicious until recognition exists)
        if self.face_detected and self.trust == TrustLevel.UNKNOWN:
            return Emotion.SUSPICIOUS

        # GREETING if a face just appeared
        if self.face_detected and not self.face_present:
            return Emotion.GREETING

        return Emotion.HAPPY

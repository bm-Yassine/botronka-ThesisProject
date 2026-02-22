from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class Emotion(Enum):
    GREETING = auto()
    HAPPY = auto()
    SUSPICIOUS = auto()
    LONELY = auto()
    STUCK = auto()
    ANGRY = auto()
    CURIOUS = auto()
    SLEEPY = auto()
    ALERT = auto()


class TrustLevel(Enum):
    UNKNOWN = 0
    GUEST = 1
    FRIEND = 2
    OWNER = 3


@dataclass
class Perception:
    face_detected: bool = False
    trust: TrustLevel = TrustLevel.UNKNOWN
    distance_cm: Optional[float] = None
    error: Optional[str] = None


@dataclass
class BotState:
    emotion: Emotion = Emotion.HAPPY
    last_emotion: Emotion = Emotion.HAPPY

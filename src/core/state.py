from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import threading
import time
from copy import deepcopy
from typing import Optional

from src.core.message import Message


class Emotion(Enum):
    GREETING = auto()
    HAPPY = auto()
    SUSPICIOUS = auto()
    LONELY = auto()
    STUCK = auto()
    ANGRY = auto()
    SLEEPY = auto()
    #no practical implementation yet
    #CURIOUS = auto()
    #ALERT = auto()
    


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


class AudioMode(Enum):
    IDLE = auto()
    ENGAGED = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


@dataclass
class AudioRuntimeState:
    mode: AudioMode = AudioMode.IDLE
    mic_muted: bool = False
    tts_playing: bool = False
    llm_thinking: bool = False
    robot_moving: bool = False
    buzzer_active: bool = False
    last_utterance_wav: Optional[str] = None
    last_user_text: str = ""
    last_reply_text: str = ""
    last_command: Optional[str] = None
    wake_override_until_monotonic: float = 0.0


@dataclass
class SharedRuntimeState:
    face_present: bool = False
    trust: TrustLevel = TrustLevel.UNKNOWN
    distance_cm: Optional[float] = None
    audio: AudioRuntimeState = field(default_factory=AudioRuntimeState)


class RuntimeStateStore:
    """Thread-safe shared runtime state that can be consumed by workers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._state = SharedRuntimeState()

    def snapshot(self) -> SharedRuntimeState:
        with self._lock:
            return deepcopy(self._state)

    def _wake_active_unlocked(self) -> bool:
        return time.monotonic() < float(self._state.audio.wake_override_until_monotonic)

    def can_open_mic(self) -> bool:
        with self._lock:
            audio = self._state.audio
            wake_active = self._wake_active_unlocked()

            if (
                not wake_active
                and not self._state.face_present
                and audio.mode == AudioMode.ENGAGED
                and not audio.tts_playing
                and not audio.llm_thinking
            ):
                audio.mode = AudioMode.IDLE

            return (
                (self._state.face_present or wake_active)
                and not audio.mic_muted
                and not audio.tts_playing
                and not audio.robot_moving
                and not audio.buzzer_active
                and audio.mode in (AudioMode.ENGAGED, AudioMode.IDLE)
            )

    def set_audio_mode(self, mode: AudioMode):
        with self._lock:
            self._state.audio.mode = mode

    def apply_message(self, msg: Message):
        with self._lock:
            if msg.type == "distance_cm":
                try:
                    payload = json.loads(msg.content)
                    value = payload.get("value")
                    if isinstance(value, (float, int)):
                        self._state.distance_cm = float(value)
                except Exception:
                    return

            elif msg.type == "vision_identity":
                try:
                    payload = json.loads(msg.content)
                    face = bool(payload.get("face_detected", False))
                    self._state.face_present = face
                    trust_text = str(payload.get("trust_level", "UNKNOWN")).upper()
                    self._state.trust = {
                        "OWNER": TrustLevel.OWNER,
                        "FRIEND": TrustLevel.FRIEND,
                        "GUEST": TrustLevel.GUEST,
                        "UNKNOWN": TrustLevel.UNKNOWN,
                    }.get(trust_text, TrustLevel.UNKNOWN)

                    if face:
                        if self._state.audio.mode == AudioMode.IDLE:
                            self._state.audio.mode = AudioMode.ENGAGED
                    elif self._state.audio.mode != AudioMode.SPEAKING:
                        self._state.audio.mode = (
                            AudioMode.ENGAGED
                            if self._wake_active_unlocked()
                            else AudioMode.IDLE
                        )
                except Exception:
                    return

            elif msg.type == "audio_listening_started":
                self._state.audio.mode = AudioMode.LISTENING

            elif msg.type == "audio_listening_finished":
                self._state.audio.mode = (
                    AudioMode.ENGAGED
                    if (self._state.face_present or self._wake_active_unlocked())
                    else AudioMode.IDLE
                )

            elif msg.type == "audio_utterance":
                try:
                    payload = json.loads(msg.content)
                    self._state.audio.last_utterance_wav = payload.get("wav_path")
                except Exception:
                    return

            elif msg.type == "stt_text":
                try:
                    payload = json.loads(msg.content)
                    self._state.audio.last_user_text = str(payload.get("text", "")).strip()
                    self._state.audio.mode = AudioMode.THINKING
                except Exception:
                    return

            elif msg.type == "llm_thinking":
                try:
                    payload = json.loads(msg.content)
                    self._state.audio.llm_thinking = bool(payload.get("value", False))
                except Exception:
                    self._state.audio.llm_thinking = False

            elif msg.type == "agent_reply":
                try:
                    payload = json.loads(msg.content)
                    self._state.audio.last_reply_text = str(payload.get("speak", "")).strip()
                    cmd = payload.get("command")
                    self._state.audio.last_command = str(cmd) if cmd else None
                except Exception:
                    return

            elif msg.type == "tts_started":
                self._state.audio.tts_playing = True
                self._state.audio.mic_muted = True
                self._state.audio.mode = AudioMode.SPEAKING

            elif msg.type == "tts_finished":
                self._state.audio.tts_playing = False
                self._state.audio.mic_muted = (
                    self._state.audio.robot_moving or self._state.audio.buzzer_active
                )
                self._state.audio.mode = (
                    AudioMode.ENGAGED
                    if (self._state.face_present or self._wake_active_unlocked())
                    else AudioMode.IDLE
                )

            elif msg.type == "audio_wake_detected":
                try:
                    payload = json.loads(msg.content)
                    duration_s = float(payload.get("duration_s", 10.0))
                except Exception:
                    duration_s = 10.0

                self._state.audio.wake_override_until_monotonic = (
                    time.monotonic() + max(0.5, duration_s)
                )
                if self._state.audio.mode == AudioMode.IDLE:
                    self._state.audio.mode = AudioMode.ENGAGED

            elif msg.type == "motion_state":
                try:
                    payload = json.loads(msg.content)
                    moving = bool(payload.get("moving", False))
                    self._state.audio.robot_moving = moving
                    self._state.audio.mic_muted = (
                        moving
                        or self._state.audio.tts_playing
                        or self._state.audio.buzzer_active
                    )
                except Exception:
                    return

            elif msg.type == "buzzer_state":
                try:
                    payload = json.loads(msg.content)
                    active = bool(payload.get("active", False))
                    self._state.audio.buzzer_active = active
                    self._state.audio.mic_muted = (
                        active
                        or self._state.audio.tts_playing
                        or self._state.audio.robot_moving
                    )
                except Exception:
                    return

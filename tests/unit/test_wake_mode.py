from __future__ import annotations

import json
import time

from src.core.message import Message
from src.core.state import RuntimeStateStore, AudioMode
from src.threads.STTworker import is_wake_phrase


def _msg(msg_type: str, content: dict | str = "{}") -> Message:
    payload = content if isinstance(content, str) else json.dumps(content)
    return Message(
        sender="test",
        type=msg_type,
        content=payload,
        sent_at=time.time(),
    )


def test_wake_phrase_matcher_requires_greeting_and_name():
    assert is_wake_phrase("hello botronka") is True
    assert is_wake_phrase("hi botronka") is True
    assert is_wake_phrase("hey Botronka can you listen") is True

    assert is_wake_phrase("hello") is True
    assert is_wake_phrase("botronka") is False
    assert is_wake_phrase("start listening") is False


def test_wake_phrase_matcher_accepts_extended_wake_commands():
    assert is_wake_phrase("wake up botronka") is True
    assert is_wake_phrase("botronka listen") is True
    assert is_wake_phrase("start listening botronka") is True
    assert is_wake_phrase("can you hear me botronka") is True


def test_wake_phrase_rejects_noise_and_random_nonwake_text():
    assert is_wake_phrase("(blows raspberry)") is False
    assert is_wake_phrase("(sniffing)") is False
    assert is_wake_phrase("be of the known kind") is False


def test_runtime_state_keeps_mic_closed_without_face_and_without_wake():
    state = RuntimeStateStore()

    # Default: no face, no wake override.
    assert state.can_open_mic() is False

    # Even if mode is ENGAGED manually, can_open_mic should reject and normalize.
    state.set_audio_mode(AudioMode.ENGAGED)
    assert state.can_open_mic() is False


def test_runtime_state_allows_open_after_audio_wake_detected_then_expires():
    state = RuntimeStateStore()

    state.apply_message(_msg("audio_wake_detected", {"duration_s": 0.6}))
    snap = state.snapshot()
    assert snap.audio.mode == AudioMode.ENGAGED
    assert state.can_open_mic() is True

    time.sleep(0.75)
    assert state.can_open_mic() is False
    assert state.snapshot().audio.mode == AudioMode.IDLE


def test_face_presence_still_opens_mic_without_wake_phrase():
    state = RuntimeStateStore()

    state.apply_message(
        _msg(
            "vision_identity",
            {
                "face_detected": True,
                "trust_level": "Guest",
            },
        )
    )
    assert state.can_open_mic() is True

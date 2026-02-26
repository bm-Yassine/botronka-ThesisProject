from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import queue
import time
import threading
from pathlib import Path

from src.core.message import Message
from src.core.state import RuntimeStateStore
from src.audio.vad import VADConfig, record_utterance
from src.threads.baseThread import BaseThread


@dataclass
class AudioIOConfig:
    utterance_dir: str = "/tmp/botfriend_audio"
    mic_device: str | None = None
    poll_interval_s: float = 0.15
    greeting_idle_s: float = 20.0
    greeting_delay_s: float = 0.8
    greeting_min_open_s: float = 4.0
    greeting_known_template: str = "Greetings {name}."
    greeting_unknown_text: str = "Hi, who are you?"
    listen_cooldown_s: float = 0.2
    wake_listen_enabled: bool = True
    wake_poll_interval_s: float = 0.25
    wake_min_open_s: float = 1.1
    wake_max_record_s: float = 2.2


class AudioIOThread(BaseThread):
    """Continuously opens mic when allowed and emits utterance WAV events."""

    def __init__(
        self,
        queue: queue.Queue[Message],
        config: AudioIOConfig,
        vad_config: VADConfig,
        state_store: RuntimeStateStore,
    ):
        super().__init__(name="AudioIOThread", queue=queue)
        self.config = config
        self.vad_config = vad_config
        self.state_store = state_store
        self._counter = 0
        self._utterance_dir = Path(config.utterance_dir)
        self._utterance_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._face_present = False
        self._last_face_lost_ts: float | None = None
        self._greeted_once = False
        self._pending_greeting_name: str | None = None
        self._pending_greeting_due_ts: float | None = None
        self._force_open_until_ts: float = 0.0

    def _make_wake_vad_cfg(self) -> VADConfig:
        # Shorter windows for wake phrase probing while face is absent.
        return VADConfig(
            sample_rate=self.vad_config.sample_rate,
            frame_ms=self.vad_config.frame_ms,
            aggressiveness=self.vad_config.aggressiveness,
            silence_ms=self.vad_config.silence_ms,
            min_speech_ms=self.vad_config.min_speech_ms,
            min_open_s=max(0.4, self.config.wake_min_open_s),
            pre_roll_ms=self.vad_config.pre_roll_ms,
            max_record_s=max(0.6, self.config.wake_max_record_s),
        )

    def _next_wav_path(self) -> Path:
        self._counter += 1
        ts = int(time.time() * 1000)
        return self._utterance_dir / f"utt_{ts}_{self._counter}.wav"

    def handle_message(self, message: Message):
        if message.type != "vision_identity":
            return

        try:
            payload = json.loads(message.content)
        except Exception:
            return

        now = time.monotonic()
        face_detected = bool(payload.get("face_detected", False))
        raw_name = str(payload.get("name", "")).strip()
        name = "" if raw_name.upper() == "UNKNOWN" else raw_name

        with self._lock:
            if face_detected and not self._face_present:
                idle_s = (
                    float("inf")
                    if self._last_face_lost_ts is None
                    else max(0.0, now - self._last_face_lost_ts)
                )
                should_greet = (not self._greeted_once) or (idle_s >= self.config.greeting_idle_s)
                if should_greet:
                    self._pending_greeting_name = name
                    self._pending_greeting_due_ts = now + max(0.0, self.config.greeting_delay_s)

            if face_detected:
                # If we are waiting to greet and now know the name, upgrade greeting text.
                if self._pending_greeting_due_ts is not None and name:
                    self._pending_greeting_name = name
            else:
                if self._face_present:
                    self._last_face_lost_ts = now

            self._face_present = face_detected

    def _next_greeting_text(self) -> str | None:
        with self._lock:
            if self._pending_greeting_due_ts is None:
                return None

            now = time.monotonic()
            if now < self._pending_greeting_due_ts:
                return None

            name = self._pending_greeting_name
            self._pending_greeting_due_ts = None
            self._pending_greeting_name = None
            self._greeted_once = True
            self._force_open_until_ts = now + max(0.0, self.config.greeting_min_open_s)

        if name:
            return self.config.greeting_known_template.format(name=name)
        return self.config.greeting_unknown_text

    def _should_force_open_mic(self) -> bool:
        snapshot = self.state_store.snapshot()
        now = time.monotonic()
        return (
            snapshot.face_present
            and now < self._force_open_until_ts
            and not snapshot.audio.mic_muted
            and not snapshot.audio.tts_playing
            and not snapshot.audio.robot_moving
            and not snapshot.audio.llm_thinking
        )

    def _should_probe_wake_phrase(self) -> bool:
        if not self.config.wake_listen_enabled:
            return False

        snapshot = self.state_store.snapshot()
        now = time.monotonic()
        wake_active = now < float(snapshot.audio.wake_override_until_monotonic)

        # Wake probing should only happen when no normal mic session is allowed,
        # no face is present, and robot audio/motion channels are quiet.
        return (
            not snapshot.face_present
            and not wake_active
            and not snapshot.audio.mic_muted
            and not snapshot.audio.tts_playing
            and not snapshot.audio.robot_moving
            and not snapshot.audio.buzzer_active
            and not snapshot.audio.llm_thinking
        )

    def run(self):
        wake_vad_cfg = self._make_wake_vad_cfg()

        while self.running:
            greeting = self._next_greeting_text()
            if greeting:
                self.broadcast_message(
                    "tts_request",
                    json.dumps(
                        {
                            "text": greeting,
                            "is_filler": False,
                            "is_greeting": True,
                            "created_at": time.time(),
                        },
                        ensure_ascii=False,
                    ),
                )

            should_listen = self.state_store.can_open_mic() or self._should_force_open_mic()
            if not should_listen:
                if self._should_probe_wake_phrase():
                    wav_path = self._next_wav_path()
                    has_speech = False
                    capture_started = time.perf_counter()
                    try:
                        has_speech = record_utterance(
                            out_wav=wav_path,
                            cfg=wake_vad_cfg,
                            mic_device=self.config.mic_device,
                        )
                        capture_ms = (time.perf_counter() - capture_started) * 1000.0
                        logging.info(
                            "Audio timing: stage=vad mode=wake duration_ms=%.1f has_speech=%s path=%s",
                            capture_ms,
                            has_speech,
                            wav_path,
                        )
                    except Exception as e:
                        logging.exception("AudioIO wake probe failed")
                        self.broadcast_message(
                            "audio_error",
                            json.dumps({"error": str(e), "ts": time.time()}),
                        )

                    if has_speech and wav_path.exists():
                        self.broadcast_message(
                            "audio_wake_candidate",
                            json.dumps({"wav_path": str(wav_path), "ts": time.time()}),
                        )
                    else:
                        wav_path.unlink(missing_ok=True)

                    time.sleep(max(0.01, self.config.wake_poll_interval_s))
                    continue

                time.sleep(max(0.05, self.config.poll_interval_s))
                continue

            wav_path = self._next_wav_path()
            self.broadcast_message("audio_listening_started", json.dumps({"ts": time.time()}))
            has_speech = False
            capture_started = time.perf_counter()

            try:
                has_speech = record_utterance(
                    out_wav=wav_path,
                    cfg=self.vad_config,
                    mic_device=self.config.mic_device,
                )
                capture_ms = (time.perf_counter() - capture_started) * 1000.0
                logging.info(
                    "Audio timing: stage=vad mode=normal duration_ms=%.1f has_speech=%s path=%s",
                    capture_ms,
                    has_speech,
                    wav_path,
                )
            except Exception as e:
                logging.exception("AudioIO capture failed")
                self.broadcast_message(
                    "audio_error",
                    json.dumps({"error": str(e), "ts": time.time()}),
                )
            finally:
                self.broadcast_message("audio_listening_finished", json.dumps({"ts": time.time()}))

            if has_speech and wav_path.exists():
                self.broadcast_message(
                    "audio_utterance",
                    json.dumps({"wav_path": str(wav_path), "ts": time.time()}),
                )
            else:
                wav_path.unlink(missing_ok=True)

            time.sleep(max(0.01, self.config.listen_cooldown_s))
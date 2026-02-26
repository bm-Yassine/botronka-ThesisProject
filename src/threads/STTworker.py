from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import queue as queue_mod
from pathlib import Path
import re
import time
from difflib import SequenceMatcher
import threading

from src.audio.STT import STTConfig, transcribe
from src.core.message import Message
from src.threads.baseThread import BaseThread


@dataclass
class STTWorkerConfig:
    min_text_chars: int = 2
    delete_wav_after_stt: bool = True
    wake_open_s: float = 12.0
    wake_candidate_max_age_s: float = 4.0


_HI_RE = re.compile(r"\b(hi|hello|hey)\b", re.IGNORECASE)
_BOTRONKA_RE = re.compile(r"\bbotronka\b", re.IGNORECASE)
_WAKE_UP_RE = re.compile(r"\bwake(?:\s+up)?\s+botronka\b", re.IGNORECASE)
_LISTEN_RE_1 = re.compile(r"\bbotronka\b.*\b(listen|start listening)\b", re.IGNORECASE)
_LISTEN_RE_2 = re.compile(r"\b(listen|start listening)\b.*\bbotronka\b", re.IGNORECASE)
_NAME_VARIANTS = ("botronka", "biedronka")
_SHORT_GREET_RE = re.compile(r"^(hi|hello|hey)\b", re.IGNORECASE)
_WAKE_INTENT_RE = re.compile(r"\b(wake|listen|start listening|can you hear me)\b", re.IGNORECASE)
_NOISE_ONLY_RE = re.compile(r"^\([^)]*\)$")


def _contains_name_variant(text: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    for tok in tokens:
        if tok in _NAME_VARIANTS:
            return True
        for name in _NAME_VARIANTS:
            if SequenceMatcher(None, tok, name).ratio() >= 0.74:
                return True
    return False


def is_wake_phrase(text: str) -> bool:
    """Wake phrase matcher for phrases like 'hello botronka' or 'wake up botronka'."""
    normalized = " ".join(str(text).strip().split())
    if not normalized:
        return False

    if _NOISE_ONLY_RE.match(normalized):
        return False

    lower = normalized.lower()
    words = re.findall(r"[a-zA-Z']+", lower)
    word_count = len(words)

    has_name = _contains_name_variant(normalized)
    has_greeting = bool(_HI_RE.search(normalized) or _SHORT_GREET_RE.search(normalized))
    has_listen_intent = bool(
        _WAKE_UP_RE.search(normalized)
        or _LISTEN_RE_1.search(normalized)
        or _LISTEN_RE_2.search(normalized)
        or _WAKE_INTENT_RE.search(normalized)
    )

    # Preferred: wake phrase includes bot name (exact/fuzzy variants).
    if has_name and (has_greeting or has_listen_intent):
        return True

    # Fallback for clipped STT: allow very short greeting-only wake.
    if has_greeting and word_count <= 2:
        return True

    # Fallback for clipped wake-check utterance.
    if "can you hear me" in lower:
        return True

    return False


class STTWorker(BaseThread):
    def __init__(
        self,
        message_queue: queue_mod.Queue[Message],
        stt_config: STTConfig,
        config: STTWorkerConfig | None = None,
    ):
        super().__init__(name="STTWorker", queue=message_queue)
        self.stt_config = stt_config
        self.config = config or STTWorkerConfig()
        self._normal_inbox: queue_mod.Queue[Message] = queue_mod.Queue()
        self._wake_lock = threading.Lock()
        self._latest_wake: Message | None = None

    @staticmethod
    def _wav_path_from_message(message: Message) -> str | None:
        try:
            payload = json.loads(message.content)
            wav_path = str(payload.get("wav_path", "")).strip()
            return wav_path or None
        except Exception:
            return None

    @staticmethod
    def _delete_wav_safe(path: str | None):
        if not path:
            return
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            logging.debug("Could not delete wav file: %s", path)

    def _pop_latest_wake(self) -> Message | None:
        with self._wake_lock:
            msg = self._latest_wake
            self._latest_wake = None
            return msg

    def handle_message(self, message: Message):
        if message.type == "audio_utterance":
            self._normal_inbox.put(message)
            return

        if message.type == "audio_wake_candidate":
            # Keep only newest wake sample to avoid long STT backlog.
            stale: Message | None = None
            with self._wake_lock:
                stale = self._latest_wake
                self._latest_wake = message

            if stale is not None:
                stale_path = self._wav_path_from_message(stale)
                self._delete_wav_safe(stale_path)
                logging.debug("Dropped stale wake candidate to keep newest sample")

    def run(self):
        while self.running:
            message = self._pop_latest_wake()
            if message is None:
                try:
                    message = self._normal_inbox.get(timeout=0.1)
                except queue_mod.Empty:
                    continue

            wav_path: str | None = None
            try:
                payload = json.loads(message.content)
                wav_path = str(payload.get("wav_path", "")).strip() or None
                if not wav_path:
                    continue

                if message.type == "audio_wake_candidate":
                    age_s = max(0.0, time.time() - float(message.sent_at))
                    if age_s > max(0.1, float(self.config.wake_candidate_max_age_s)):
                        logging.info(
                            "Dropping stale wake candidate age_s=%.2f path=%s",
                            age_s,
                            wav_path,
                        )
                        continue

                stt_started = time.perf_counter()
                text = transcribe(wav_path, self.stt_config).strip()
                stt_ms = (time.perf_counter() - stt_started) * 1000.0
                if len(text) < self.config.min_text_chars:
                    logging.info(
                        "Audio timing: stage=stt mode=%s duration_ms=%.1f text_chars=%d accepted=%s",
                        message.type,
                        stt_ms,
                        len(text),
                        False,
                    )
                    continue

                logging.info(
                    "Audio timing: stage=stt mode=%s duration_ms=%.1f text_chars=%d accepted=%s",
                    message.type,
                    stt_ms,
                    len(text),
                    True,
                )

                if message.type == "audio_wake_candidate":
                    if is_wake_phrase(text):
                        # STT can be slow on-device; extend wake window by STT latency
                        # so the mic is still open after detection is confirmed.
                        wake_duration_s = max(
                            0.5,
                            float(self.config.wake_open_s) + max(1.0, stt_ms / 1000.0),
                        )
                        self.broadcast_message(
                            "audio_wake_detected",
                            json.dumps(
                                {
                                    "text": text,
                                    "duration_s": wake_duration_s,
                                    "stt_ms": stt_ms,
                                    "ts": message.sent_at,
                                },
                                ensure_ascii=False,
                            ),
                        )
                        self.broadcast_message(
                            "buzzer_countdown",
                            json.dumps(
                                {
                                    "steps": 1,
                                    "interval_s": 0.05,
                                    "created_at": time.time(),
                                },
                                ensure_ascii=False,
                            ),
                        )
                        logging.info(
                            "Wake phrase detected: text=%r wake_duration_s=%.2f stt_ms=%.1f",
                            text,
                            wake_duration_s,
                            stt_ms,
                        )
                    else:
                        logging.info("Wake phrase rejected: %s", text)
                    continue

                self.broadcast_message(
                    "stt_text",
                    json.dumps(
                        {
                            "text": text,
                            "wav_path": wav_path,
                            "ts": message.sent_at,
                        },
                        ensure_ascii=False,
                    ),
                )
            except Exception as e:
                logging.exception("STT transcription failed")
                self.broadcast_message(
                    "stt_error",
                    json.dumps({"error": str(e), "ts": message.sent_at}),
                )
            finally:
                if self.config.delete_wav_after_stt and wav_path:
                    self._delete_wav_safe(wav_path)
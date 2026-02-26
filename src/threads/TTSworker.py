from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import queue as queue_mod
import time

from src.audio.TTS import PiperTTS, TTSConfig
from src.core.message import Message
from src.threads.baseThread import BaseThread


@dataclass
class TTSWorkerConfig:
    pre_generate_phrases: list[str] = field(
        default_factory=lambda: [
            "Hi, who are you?",
            "Working on it.",
            "Let me think.",
            "Hmm, gotcha.",
        ]
    )


class TTSWorker(BaseThread):
    def __init__(
        self,
        message_queue: queue_mod.Queue[Message],
        tts_config: TTSConfig,
        config: TTSWorkerConfig | None = None,
    ):
        super().__init__(name="TTSWorker", queue=message_queue)
        self.config = config or TTSWorkerConfig()
        self.tts = PiperTTS(tts_config)
        self._inbox: queue_mod.Queue[Message] = queue_mod.Queue()

        # Warm cache for common short phrases to reduce perceived latency.
        try:
            self.tts.pre_generate(self.config.pre_generate_phrases)
        except Exception:
            logging.exception("TTS pre-generate failed during startup")

    def handle_message(self, message: Message):
        if message.type == "tts_request":
            self._inbox.put(message)

    def run(self):
        while self.running:
            try:
                message = self._inbox.get(timeout=0.1)
            except queue_mod.Empty:
                continue

            text = ""
            payload: dict = {}
            try:
                payload = json.loads(message.content)
                text = str(payload.get("text", "")).strip()
                if not text:
                    continue

                self.broadcast_message(
                    "tts_started",
                    json.dumps(
                        {
                            "ts": time.time(),
                            "text": text,
                            "is_filler": bool(payload.get("is_filler", False)),
                        },
                        ensure_ascii=False,
                    ),
                )

                self.tts.speak(text)

                self.broadcast_message(
                    "tts_finished",
                    json.dumps(
                        {
                            "ts": time.time(),
                            "text": text,
                            "is_filler": bool(payload.get("is_filler", False)),
                            "command": payload.get("command"),
                        },
                        ensure_ascii=False,
                    ),
                )
            except Exception as e:
                logging.exception("TTS worker failed")
                self.broadcast_message(
                    "tts_error",
                    json.dumps(
                        {
                            "error": str(e),
                            "text": text,
                            "payload": payload,
                            "ts": time.time(),
                        },
                        ensure_ascii=False,
                    ),
                )
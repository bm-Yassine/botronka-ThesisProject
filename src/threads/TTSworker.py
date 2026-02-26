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
    startup_announcement_enabled: bool = True
    startup_announcement_text: str = "Botronka is waking up"
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
            warm_phrases = list(self.config.pre_generate_phrases)
            startup_text = self.config.startup_announcement_text.strip()
            if (
                self.config.startup_announcement_enabled
                and startup_text
                and startup_text not in warm_phrases
            ):
                warm_phrases.append(startup_text)
            self.tts.pre_generate(warm_phrases)
        except Exception:
            logging.exception("TTS pre-generate failed during startup")

    def handle_message(self, message: Message):
        if message.type == "tts_request":
            self._inbox.put(message)

    def _speak_payload(self, payload: dict):
        text = str(payload.get("text", "")).strip()
        if not text:
            return

        self.broadcast_message(
            "tts_started",
            json.dumps(
                {
                    "ts": time.time(),
                    "text": text,
                    "is_filler": bool(payload.get("is_filler", False)),
                    "is_startup": bool(payload.get("is_startup", False)),
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
                    "is_startup": bool(payload.get("is_startup", False)),
                    "command": payload.get("command"),
                },
                ensure_ascii=False,
            ),
        )

    def run(self):
        if self.config.startup_announcement_enabled:
            startup_text = self.config.startup_announcement_text.strip()
            if startup_text:
                try:
                    self._speak_payload(
                        {
                            "text": startup_text,
                            "is_filler": False,
                            "is_startup": True,
                        }
                    )
                except Exception as e:
                    logging.exception("TTS startup announcement failed")
                    self.broadcast_message(
                        "tts_error",
                        json.dumps(
                            {
                                "error": str(e),
                                "text": startup_text,
                                "payload": {"is_startup": True},
                                "ts": time.time(),
                            },
                            ensure_ascii=False,
                        ),
                    )

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
                self._speak_payload(payload)
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
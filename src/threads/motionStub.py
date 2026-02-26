from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import queue as queue_mod
import time

from src.core.message import Message
from src.threads.baseThread import BaseThread


@dataclass
class MotionStubConfig:
    output_dir: str = "tests/out/wheels"
    latest_file: str = "latest_command.json"
    log_file: str = "commands.log"
    simulate_move_s: float = 1.0


class MotionStubThread(BaseThread):
    """Temporary motion executor: writes motor commands to disk for inspection."""

    def __init__(
        self,
        message_queue: queue_mod.Queue[Message],
        config: MotionStubConfig,
    ):
        super().__init__(name="MotionStubThread", queue=message_queue)
        self.config = config
        self._inbox: queue_mod.Queue[Message] = queue_mod.Queue()

        self._output_dir = Path(self.config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._latest_path = self._output_dir / self.config.latest_file
        self._log_path = self._output_dir / self.config.log_file

    def handle_message(self, message: Message):
        if message.type == "motion_command":
            self._inbox.put(message)

    def _persist_command(self, payload: dict):
        payload = dict(payload)
        payload.setdefault("created_at", time.time())

        self._latest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def run(self):
        while self.running:
            try:
                message = self._inbox.get(timeout=0.1)
            except queue_mod.Empty:
                continue

            try:
                payload = json.loads(message.content)
                command = str(payload.get("command", "")).strip()
                if not command:
                    continue

                self.broadcast_message("motion_state", json.dumps({"moving": True}))
                self._persist_command(payload)
                logging.info("Motion stub command persisted: %s", command)
                time.sleep(max(0.0, self.config.simulate_move_s))
            except Exception:
                logging.exception("MotionStubThread failed to process motion command")
            finally:
                self.broadcast_message("motion_state", json.dumps({"moving": False}))

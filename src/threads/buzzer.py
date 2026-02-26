from collections.abc import Callable
from dataclasses import dataclass
import json
import queue
import logging

from src.threads.baseThread import BaseThread
from src.core.message import Message

from src.hardware.buzzer import Buzzer, BuzzerConfig


@dataclass
class BuzzerThreadConfig:
    pin: int


class BuzzerThread(BaseThread):
    def __init__(self, queue: queue.Queue[Message], config: BuzzerThreadConfig):
        super().__init__(name="BuzzerThread", queue=queue)
        self.buzzer = Buzzer(BuzzerConfig(pin=config.pin))

    def handle_message(self, message: Message):
        logging.debug(f"BuzzerThread received message: {message}")

        if message.type == "distance_cm":
            distance = json.loads(message.content).get("value")
            if isinstance(distance, (int, float)) and distance < 15:
                logging.info(f"Distance {distance} cm is too close, activating buzzer.")
                self.broadcast_message("buzzer_state", json.dumps({"active": True}))
                try:
                    self.buzzer.pattern_too_close()
                finally:
                    self.broadcast_message("buzzer_state", json.dumps({"active": False}))

        elif message.type == "buzzer_countdown":
            payload = json.loads(message.content)
            steps = int(payload.get("steps", 3))
            interval_s = float(payload.get("interval_s", 0.6))
            logging.info("Buzzer countdown: steps=%s interval=%s", steps, interval_s)
            self.broadcast_message("buzzer_state", json.dumps({"active": True}))
            try:
                self.buzzer.pattern_countdown(steps=steps, interval_s=interval_s)
            finally:
                self.broadcast_message("buzzer_state", json.dumps({"active": False}))

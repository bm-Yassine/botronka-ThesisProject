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
            if isinstance(distance, (int, float)) and distance < 20:
                logging.info(f"Distance {distance} cm is too close, activating buzzer.")
                self.buzzer.pattern_too_close()

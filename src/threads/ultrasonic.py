from __future__ import annotations
import queue
import time
import json

from src.threads.baseThread import BaseThread
from src.core.message import Message
from src.hardware.distanceSensor import UltrasonicSensor, UltrasonicConfig


class UltrasonicFront(BaseThread):
    def __init__(self, cfg: UltrasonicConfig, queue: queue.Queue[Message]):
        super().__init__(name="UltrasonicFront", queue=queue)
        self.cfg = cfg
        self.sensor = UltrasonicSensor(
            UltrasonicConfig(
                trigger_pin=cfg.trigger_pin,
                echo_pin=cfg.echo_pin,
            )
        )

    def run(self):
        while self.running:
            dist_cm = self.sensor.read_cm()
            self.broadcast_message(
                "distance_cm",
                json.dumps({"value": dist_cm}),
            )
            time.sleep(0.1)

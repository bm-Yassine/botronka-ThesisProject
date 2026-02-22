from __future__ import annotations
import logging
import queue

from dataclasses import dataclass

from src.core.state import BotState, Perception
from src.hardware.oledDisplay import OledConfig
from src.threads.ultrasonic import UltrasonicFront, UltrasonicConfig
from src.threads.display import BehaviorConfig, DisplayThread
from src.threads.buzzer import BuzzerThread, BuzzerThreadConfig
from src.threads.vision import VisionThread, VisionThreadConfig
from src.threads.threadManager import ThreadManager
from src.core.message import Message


@dataclass
class AppConfig:
    ultrasonic: UltrasonicConfig
    buzzer: BuzzerThreadConfig
    oled: OledConfig
    behavior: BehaviorConfig
    vision: VisionThreadConfig

    @staticmethod
    def from_dict(d: dict) -> AppConfig:
        # Backward compatibility: if "vision" is absent, infer width/height/fps from legacy "camera".
        vision_dict = dict(d.get("vision", {}))
        camera_dict = d.get("camera", {})
        if "width" not in vision_dict and "width" in camera_dict:
            vision_dict["width"] = camera_dict["width"]
        if "height" not in vision_dict and "height" in camera_dict:
            vision_dict["height"] = camera_dict["height"]
        if "recognition_fps" not in vision_dict and "fps" in camera_dict:
            vision_dict["recognition_fps"] = camera_dict["fps"]

        return AppConfig(
            ultrasonic=UltrasonicConfig(**d["ultrasonic"]),
            buzzer=BuzzerThreadConfig(**d["buzzer"]),
            oled=OledConfig(**d["oled"]),
            behavior=BehaviorConfig(**d["behavior"]),
            vision=VisionThreadConfig(**vision_dict),
        )


class BotApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        print(f"AppConfig. {cfg}")
        self.queue: queue.Queue[Message] = queue.Queue()
        self.state = BotState()
        self.perc = Perception()

        self.threadManager = ThreadManager(self.queue)

        uc = cfg.ultrasonic
        self.ultra = UltrasonicFront(
            UltrasonicConfig(
                trigger_pin=uc.trigger_pin,
                echo_pin=uc.echo_pin,
            ),
            self.queue,
        )

        self.buzzer = BuzzerThread(self.queue, BuzzerThreadConfig(pin=cfg.buzzer.pin))

        self.display = DisplayThread(
            config=cfg.oled,
            behaviorConfig=cfg.behavior,
            queue=self.queue,
        )
        self.vision = VisionThread(config=cfg.vision, queue=self.queue)

        self.threadManager.add_thread(self.ultra)
        self.threadManager.add_thread(self.buzzer)
        self.threadManager.add_thread(self.display)
        self.threadManager.add_thread(self.vision)

    ### TODO : Audio and movement Threads 
    
    def run_forever(self):
        logging.info("Starting threads...")
        self.threadManager.start_all()
        self.threadManager.run()  # blocks forever

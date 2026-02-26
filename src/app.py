from __future__ import annotations
import logging
import queue

from dataclasses import asdict, dataclass

from src.core.state import BotState, Perception, RuntimeStateStore
from src.hardware.oledDisplay import OledConfig
from src.audio.STT import STTConfig
from src.audio.TTS import TTSConfig
from src.audio.vad import VADConfig
from src.agent.llm_client import LLMConfig
from src.threads.audioIO import AudioIOConfig, AudioIOThread
from src.threads.STTworker import STTWorker, STTWorkerConfig
from src.threads.AgentWorker import AgentWorker, AgentWorkerConfig
from src.threads.TTSworker import TTSWorker, TTSWorkerConfig
from src.threads.motion import MotionControlConfig, MotionControlThread
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
    audio_io: AudioIOConfig
    vad: VADConfig
    stt: STTConfig
    stt_worker: STTWorkerConfig
    llm: LLMConfig
    agent: AgentWorkerConfig
    tts: TTSConfig
    tts_worker: TTSWorkerConfig
    motion: MotionControlConfig

    @staticmethod
    def from_dict(d: dict) -> AppConfig:
        # Backward compatibility: if "vision" is absent, infer width/height/fps from legacy "camera".
        vision_dict = dict(d.get("vision", {}))
        motion_dict = dict(d.get("motion", {}))
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
            audio_io=AudioIOConfig(
                **{
                    **asdict(AudioIOConfig()),
                    **dict(d.get("audio_io", {})),
                }
            ),
            vad=VADConfig(
                **{
                    **asdict(VADConfig()),
                    **dict(d.get("vad", {})),
                }
            ),
            stt=STTConfig(
                **{
                    **asdict(STTConfig()),
                    **dict(d.get("stt", {})),
                }
            ),
            stt_worker=STTWorkerConfig(
                **{
                    **asdict(STTWorkerConfig()),
                    **dict(d.get("stt_worker", {})),
                }
            ),
            llm=LLMConfig(
                **{
                    **asdict(LLMConfig()),
                    **dict(d.get("llm", {})),
                }
            ),
            agent=AgentWorkerConfig(
                **{
                    **asdict(AgentWorkerConfig()),
                    **dict(d.get("agent", {})),
                }
            ),
            tts=TTSConfig(
                **{
                    **asdict(TTSConfig()),
                    **dict(d.get("tts", {})),
                }
            ),
            tts_worker=TTSWorkerConfig(
                **{
                    **asdict(TTSWorkerConfig()),
                    **dict(d.get("tts_worker", {})),
                }
            ),
            motion=MotionControlConfig(
                **{
                    **asdict(MotionControlConfig()),
                    **motion_dict,
                }
            ),
        )


class BotApp:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        print(f"AppConfig. {cfg}")
        self.queue: queue.Queue[Message] = queue.Queue()
        self.state = BotState()
        self.perc = Perception()
        self.shared_state = RuntimeStateStore()

        self.threadManager = ThreadManager(self.queue, state_store=self.shared_state)

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
        self.vision = VisionThread(config=cfg.vision, message_queue=self.queue)
        self.audio_io = AudioIOThread(
            queue=self.queue,
            config=cfg.audio_io,
            vad_config=cfg.vad,
            state_store=self.shared_state,
        )
        self.stt_worker = STTWorker(
            message_queue=self.queue,
            stt_config=cfg.stt,
            config=cfg.stt_worker,
        )
        self.agent_worker = AgentWorker(
            message_queue=self.queue,
            llm_config=cfg.llm,
            state_store=self.shared_state,
            config=cfg.agent,
        )
        self.tts_worker = TTSWorker(
            message_queue=self.queue,
            tts_config=cfg.tts,
            config=cfg.tts_worker,
        )
        self.motion = MotionControlThread(
            message_queue=self.queue,
            config=cfg.motion,
        )

        self.threadManager.add_thread(self.ultra)
        self.threadManager.add_thread(self.buzzer)
        self.threadManager.add_thread(self.display)
        self.threadManager.add_thread(self.vision)
        self.threadManager.add_thread(self.audio_io)
        self.threadManager.add_thread(self.stt_worker)
        self.threadManager.add_thread(self.agent_worker)
        self.threadManager.add_thread(self.tts_worker)
        self.threadManager.add_thread(self.motion)

        # Keep potential relative data paths resolved from project root
        if hasattr(self.cfg.agent, "face_db_path"):
            logging.info("Agent face_db_path=%s", self.cfg.agent.face_db_path)
        if hasattr(self.cfg.agent, "trust_map_path"):
            logging.info("Agent trust_map_path=%s", self.cfg.agent.trust_map_path)

    def run_forever(self):
        logging.info("Starting threads...")
        self.threadManager.start_all()
        self.threadManager.run()  # blocks forever

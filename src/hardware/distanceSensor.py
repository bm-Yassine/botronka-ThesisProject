from dataclasses import dataclass
from gpiozero import DistanceSensor  # type: ignore


@dataclass
class UltrasonicConfig:
    trigger_pin: int
    echo_pin: int


class UltrasonicSensor:
    def __init__(self, cfg: UltrasonicConfig):
        self.cfg = cfg
        self.sensor = DistanceSensor(
            echo=cfg.echo_pin, trigger=cfg.trigger_pin, queue_len=3, max_distance=3
        )

    def read_cm(self) -> float:
        if not isinstance(
            self.sensor.distance, float  # pyright: ignore[reportUnknownMemberType]
        ):
            return 0.0

        return self.sensor.distance * 100

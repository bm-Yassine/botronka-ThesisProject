from dataclasses import dataclass
import time

from gpiozero import Buzzer as gpioBuzzer  # pyright: ignore[reportMissingTypeStubs]


@dataclass
class BuzzerConfig:
    pin: int


class Buzzer:
    def __init__(self, config: BuzzerConfig):
        self.b = gpioBuzzer(pin=config.pin)

    def chirp(self, t: float = 0.08):
        self.b.on()
        time.sleep(t)
        self.b.off()

    def pattern_stuck(self):
        for _ in range(3):
            self.chirp(0.06)
            time.sleep(0.06)

    def pattern_error(self):
        self.b.on()
        time.sleep(0.35)
        self.b.off()
        time.sleep(0.15)
        self.b.on()
        time.sleep(0.35)
        self.b.off()

    def pattern_too_close(self):
        for _ in range(6):
            self.chirp(0.03)
            time.sleep(0.03)

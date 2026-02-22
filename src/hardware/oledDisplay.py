from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass

import board  # pyright: ignore[reportMissingTypeStubs]
import adafruit_ssd1306  # pyright: ignore[reportMissingTypeStubs]

from src.core.state import Emotion


@dataclass
class OledConfig:
    i2c_bus: int
    address: int
    width: int
    height: int


class OledDisplay:
    def __init__(
        self,
        config: OledConfig,
    ):
        # On Pi, i2c_bus is usually 1; board.I2C() uses default bus.
        i2c = board.I2C()
        self.width = config.width
        self.height = config.height
        self.disp = adafruit_ssd1306.SSD1306_I2C(
            self.width, self.height, i2c, addr=config.address
        )
        self.disp.fill(0)
        self.disp.show()

        self.font = ImageFont.load_default()

    def draw(self, emotion: Emotion, subtitle: str = ""):
        img = Image.new("1", (self.width, self.height))
        draw = ImageDraw.Draw(img)

        big = {
            Emotion.GREETING: "(^_^)/",
            Emotion.HAPPY: "^_^",
            Emotion.SUSPICIOUS: "(o_O)",
            Emotion.LONELY: "(._.)",
            Emotion.STUCK: "(>_<)",
            Emotion.ANGRY: "(ಠ_ಠ)",
            Emotion.CURIOUS: "(?_?)",
            Emotion.SLEEPY: "(-_-) zZ",
            Emotion.ALERT: "(!)",
        }.get(emotion, ":-)")

        draw.text((0, 0), f"{emotion.name}", font=self.font, fill=255)
        draw.text((0, 18), big, font=self.font, fill=255)
        if subtitle:
            draw.text((0, 45), subtitle[:20], font=self.font, fill=255)

        self.disp.image(img)
        self.disp.show()

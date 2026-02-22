from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    data: dict

    @staticmethod
    def load(path: str | Path) -> "Config":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return Config(yaml.safe_load(f))

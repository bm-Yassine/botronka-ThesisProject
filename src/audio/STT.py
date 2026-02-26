from __future__ import annotations

from dataclasses import dataclass
import os
import re
import subprocess
from pathlib import Path


@dataclass
class STTConfig:
    whisper_bin: str = "/home/botronka/whisper.cpp/build/bin/whisper-cli"
    model_path: str = "/home/botronka/botfriend/models/stt/ggml-small.en.bin"
    language: str = "en"
    threads: int = max(1, (os.cpu_count() or 2) - 1)
    max_context: int = 1024
    timeout_s: int = 120
    wake_grammar_path: str = "/tmp/botfriend_wake.gbnf"


def _normalize_whisper_output(raw: str) -> str:
    # Remove common timestamp prefixes if present and compact whitespace
    text = re.sub(r"\[[^\]]+\]", " ", raw)
    text = " ".join(text.split())
    return text.strip()


def _ensure_wake_grammar(path: str) -> str:
    """Create/update a compact grammar used to bias wake-phrase decoding."""
    grammar = (
        'root ::= wake\n'
        'wake ::= greet ws? name | wakeup ws? name | name ws? listen | hearme\n'
        'greet ::= "hi" | "hello" | "hey"\n'
        'wakeup ::= "wake" ws "up"\n'
        'listen ::= "listen" | "start" ws "listening"\n'
        'hearme ::= "can" ws "you" ws "hear" ws "me"\n'
        'name ::= "botronka" | "biedronka"\n'
        'ws ::= " "\n'
    )

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists() or p.read_text(encoding="utf-8") != grammar:
        p.write_text(grammar, encoding="utf-8")
    return str(p)


def transcribe(wav_path: str, cfg: STTConfig, *, wake_mode: bool = False) -> str:
    wav = Path(wav_path)
    if not wav.exists():
        raise FileNotFoundError(f"Audio file not found: {wav}")

    cmd = [
        cfg.whisper_bin,
        "-m",
        cfg.model_path,
        "-f",
        str(wav),
        "-l",
        cfg.language,
        "-t",
        str(max(1, cfg.threads)),
        "-mc",
        str(max(0, cfg.max_context)),
        "-nt",
        "-np",
    ]

    if wake_mode:
        # Speed/bias for wake words on noisy Raspberry Pi conditions.
        grammar_path = _ensure_wake_grammar(cfg.wake_grammar_path)
        cmd.extend(
            [
                "--grammar",
                grammar_path,
                "--grammar-rule",
                "root",
                "-bo",
                "1",
                "-bs",
                "1",
                "-ml",
                "48",
            ]
        )

    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(10, cfg.timeout_s),
        check=True,
    )
    return _normalize_whisper_output(p.stdout)
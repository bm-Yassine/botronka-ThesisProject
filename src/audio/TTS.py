from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import subprocess
import time
from pathlib import Path


@dataclass
class TTSConfig:
    piper_bin: str = "piper"
    model_path: str = "/home/botronka/botfriend/models/tts/en_US-lessac-medium.onnx"
    model_config_path: str = "/home/botronka/botfriend/models/tts/en_US-lessac-medium.onnx.json"
    sample_rate: int = 22050
    aplay_bin: str = "aplay"
    cache_dir: str = "/tmp/botfriend_tts"


class PiperTTS:
    def __init__(self, cfg: TTSConfig):
        self.cfg = cfg
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, text: str) -> Path:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"phrase_{digest}.wav"

    def synthesize_to_wav(self, text: str, out_wav: str | Path):
        if not text.strip():
            raise ValueError("TTS text cannot be empty")

        out = Path(out_wav)
        out.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.cfg.piper_bin,
            "-m",
            self.cfg.model_path,
            "-c",
            self.cfg.model_config_path,
            "-f",
            str(out),
        ]
        subprocess.run(cmd, input=(text.strip() + "\n"), text=True, check=True)

    def speak_wav(self, wav_path: str | Path):
        subprocess.run([self.cfg.aplay_bin, str(wav_path)], check=True)

    def speak(self, text: str):
        out = self._cache_path(text)
        synth_ms = 0.0
        cache_hit = out.exists()
        if not out.exists():
            synth_started = time.perf_counter()
            self.synthesize_to_wav(text, out)
            synth_ms = (time.perf_counter() - synth_started) * 1000.0

        play_started = time.perf_counter()
        self.speak_wav(out)
        play_ms = (time.perf_counter() - play_started) * 1000.0
        logging.info(
            "Audio timing: stage=tts cache_hit=%s synth_ms=%.1f play_ms=%.1f text_chars=%d",
            cache_hit,
            synth_ms,
            play_ms,
            len(text),
        )

    def pre_generate(self, phrases: list[str]):
        for phrase in phrases:
            clean = phrase.strip()
            if not clean:
                continue
            wav = self._cache_path(clean)
            if not wav.exists():
                self.synthesize_to_wav(clean, wav)
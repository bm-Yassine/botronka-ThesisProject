from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import subprocess
import time
import wave
from pathlib import Path

try:
    import webrtcvad  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    webrtcvad = None


@dataclass
class VADConfig:
    sample_rate: int = 16000
    frame_ms: int = 30
    aggressiveness: int = 2
    silence_ms: int = 700
    min_speech_ms: int = 250
    min_open_s: float = 2.0
    pre_roll_ms: int = 180
    max_record_s: float = 6.0


def _frame_bytes(cfg: VADConfig) -> int:
    return int(cfg.sample_rate * (cfg.frame_ms / 1000.0) * 2)  # s16 mono


def _is_speech_frame(frame: bytes, cfg: VADConfig, vad) -> bool:
    if vad is not None:
        try:
            return bool(vad.is_speech(frame, cfg.sample_rate))
        except Exception:
            pass

    # Lightweight fallback: short-time energy (RMS) without audioop.
    # 250 is intentionally low to capture quiet speech on small mics.
    if not frame:
        return False
    samples = array("h")
    samples.frombytes(frame)
    if len(samples) == 0:
        return False
    energy = sum(int(s) * int(s) for s in samples) / len(samples)
    rms = math.sqrt(energy)
    return rms >= 250


def record_utterance(out_wav: str | Path, cfg: VADConfig, mic_device: str | None = None) -> bool:
    """Record from microphone and write a VAD-trimmed utterance.

    Returns True if speech-like audio was detected.
    """
    out = Path(out_wav)
    out.parent.mkdir(parents=True, exist_ok=True)

    fbytes = _frame_bytes(cfg)
    if fbytes <= 0:
        return False

    cmd = [
        "arecord",
        "-q",
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        str(cfg.sample_rate),
        "-t",
        "raw",
    ]
    if mic_device:
        cmd[1:1] = ["-D", mic_device]

    vad = None
    if webrtcvad is not None:
        try:
            vad = webrtcvad.Vad(max(0, min(3, cfg.aggressiveness)))
        except Exception:
            vad = None

    pre_roll_frames = max(0, cfg.pre_roll_ms // max(10, cfg.frame_ms))
    hangover_frames = max(1, cfg.silence_ms // max(10, cfg.frame_ms))
    min_speech_frames = max(1, cfg.min_speech_ms // max(10, cfg.frame_ms))

    frames: list[bytes] = []
    first_speech_idx: int | None = None
    last_speech_idx: int | None = None
    speech_frames = 0
    trailing_silence_ms = 0
    speech_started = False

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=fbytes,
    )

    try:
        started = time.monotonic()

        while True:
            if proc.stdout is None:
                break

            frame = proc.stdout.read(fbytes)
            if len(frame) < fbytes:
                break

            idx = len(frames)
            frames.append(frame)

            is_speech = _is_speech_frame(frame, cfg, vad)
            if is_speech:
                speech_frames += 1
                trailing_silence_ms = 0
                speech_started = True
                if first_speech_idx is None:
                    first_speech_idx = idx
                last_speech_idx = idx
            elif speech_started:
                trailing_silence_ms += cfg.frame_ms

            elapsed = time.monotonic() - started
            if elapsed >= max(0.2, cfg.max_record_s):
                break

            if elapsed >= max(0.2, cfg.min_open_s):
                if speech_started and trailing_silence_ms >= cfg.silence_ms:
                    break
                if not speech_started:
                    break

        if speech_frames < min_speech_frames or first_speech_idx is None or last_speech_idx is None:
            return False

        start_idx = max(0, first_speech_idx - pre_roll_frames)
        end_idx = min(len(frames), last_speech_idx + hangover_frames + 1)
        clipped = b"".join(frames[start_idx:end_idx])
        if len(clipped) < fbytes:
            return False

        with wave.open(str(out), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(cfg.sample_rate)
            wf.writeframes(clipped)

        return True
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
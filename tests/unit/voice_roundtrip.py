#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

WHISPER_CLI   = Path.home() / "whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = Path.home() / "whisper.cpp/models/ggml-small.en.bin"

PIPER_MODEL  = Path.home() / "botfriend/models/tts/en_US-lessac-medium.onnx"
PIPER_CONFIG = Path.home() / "botfriend/models/tts/en_US-lessac-medium.onnx.json"

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

IN_WAV  = Path("/tmp/in.wav")
OUT_WAV = Path("/tmp/out.wav")

def sh(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

def check_files():
    missing = []
    for p in (WHISPER_CLI, WHISPER_MODEL, PIPER_MODEL, PIPER_CONFIG):
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise SystemExit("Missing required file(s):\n" + "\n".join(missing))

def record(seconds=5):
    # If your mic needs a specific ALSA device, add: ["-D","plughw:X,Y"]
    cmd = ["arecord", "-f", "S16_LE", "-c1", "-r", "16000", "-t", "wav", "-d", str(seconds), str(IN_WAV)]
    print("[REC]", " ".join(cmd))
    subprocess.check_call(cmd)

def stt() -> str:
    cmd = [str(WHISPER_CLI), "-m", str(WHISPER_MODEL), "-f", str(IN_WAV), "-nt", "-np"]
    print("[STT]", " ".join(cmd))
    out = sh(cmd).strip()
    # whisper-cli often outputs a leading newline; normalize
    return " ".join(out.split())

def llm(user_text: str) -> str:
    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": "You are a helpful robot assistant. Answer in 1 short sentence."},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.2,
    }
    print("[LLM] POST", LLM_URL)
    raw = sh(["curl", "-s", LLM_URL, "-H", "Content-Type: application/json", "-d", json.dumps(payload)])
    data = json.loads(raw)
    if "choices" not in data:
        raise SystemExit("LLM returned unexpected JSON:\n" + raw[:800])
    return data["choices"][0]["message"]["content"].strip()

def tts(text: str):
    # Make sure output file is fresh
    if OUT_WAV.exists():
        OUT_WAV.unlink()
    cmd = [
        "piper",
        "-m", str(PIPER_MODEL),
        "-c", str(PIPER_CONFIG),
        "-f", str(OUT_WAV),
    ]
    print("[TTS] generating", OUT_WAV)
    p = subprocess.run(cmd, input=(text + "\n"), text=True)
    if p.returncode != 0:
        raise SystemExit("Piper failed.")
    if not OUT_WAV.exists() or OUT_WAV.stat().st_size < 1000:
        raise SystemExit("TTS output WAV not created or too small.")

def play():
    print("[PLAY] aplay", OUT_WAV)
    subprocess.check_call(["aplay", str(OUT_WAV)])

def main():
    check_files()
    print("=== MIC -> STT -> LLM -> TTS -> SPEAKER ===")
    record(5)
    text = stt()
    print("You said:", text)
    reply = llm(text)
    print("Assistant:", reply)
    tts(reply)
    play()

if __name__ == "__main__":
    main()
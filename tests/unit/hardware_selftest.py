#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

# Run tests :
# cd ~/botfriend
# source .venv/bin/activate
# sudo -E .venv/bin/python tests/hardware_selftest.py

### Ultrasound and audio tests outdated.

# =========================
# CONFIG — 
# =========================

# Ultrasonic (HC-SR04 style)
US_TRIG_GPIO = 24
US_ECHO_GPIO = 23
# IMPORTANT: Echo is 5V on many modules -> use a voltage divider / level shifter to 3.3V!

# Buzzer 
BUZZER_GPIO = 17  # change if you wired another GPIO

# OLED (common SSD1306 I2C)
I2C_BUS = 1
OLED_I2C_ADDR = 0x3C  

# L298N (optional). These are example pins — change to your plan.
# If you don't have the motor driver connected yet, leaving these as-is is fine (dry-run).
MOTOR_ENA = 12  # PWM capable
MOTOR_IN1 = 5
MOTOR_IN2 = 6
MOTOR_ENB = 13  # PWM capable
MOTOR_IN3 = 20
MOTOR_IN4 = 21

# Audio devices: leave None to use system default ALSA device.
# If playback/recording goes to HDMI instead of USB, set these after running:
#   arecord -L
#   aplay -L
ALSA_RECORD_DEV = None   # e.g. "plughw:1,0"
ALSA_PLAY_DEV = None     # e.g. "plughw:1,0"

# Output folder
OUT_DIR = os.path.join(os.path.dirname(__file__), "out")


def run_cmd(cmd, timeout=30, check=True):
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, timeout=timeout, check=check)


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def test_camera():
    """
    Uses rpicam-still (recommended on modern Pi OS/Debian).
    """
    ensure_out_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"camera_{ts}.jpg")

    if shutil.which("rpicam-still") is None:
        print("❌ rpicam-still not found. Install: sudo apt install rpicam-apps")
        return False

    try:
        run_cmd([
            "rpicam-still",
            "-o", out_path,
            "--timeout", "800",
            "--width", "1280",
            "--height", "720",
            "--nopreview",
        ], timeout=20)
        print(f"✅ Camera OK. Saved: {out_path}")
        return True
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False


def test_i2c_scan():
    """
    Quick scan to confirm OLED address appears on I2C bus.
    """
    if shutil.which("i2cdetect") is None:
        print("❌ i2cdetect not found. Install: sudo apt install i2c-tools")
        return False

    try:
        run_cmd(["i2cdetect", "-y", str(I2C_BUS)], timeout=10, check=True)
        print("✅ I2C scan done. Confirm your OLED address is visible (often 3c/3d).")
        return True
    except Exception as e:
        print(f"❌ I2C scan failed: {e}")
        return False


def oled_init():
    """
    Tries to init an SSD1306 OLED via luma.oled.
    If libs missing, returns None.
    """
    try:
        from luma.core.interface.serial import i2c
        from luma.oled.device import ssd1306
        serial = i2c(port=I2C_BUS, address=OLED_I2C_ADDR)
        device = ssd1306(serial)
        return device
    except Exception as e:
        print(f"⚠️ OLED init skipped (missing lib or wrong addr). Details: {e}")
        print("   If you want OLED: pip install luma.oled pillow + set OLED_I2C_ADDR correctly.")
        return None


def oled_show(device, lines):
    try:
        from luma.core.render import canvas
        with canvas(device) as draw:
            y = 0
            for line in lines[:6]:
                draw.text((0, y), line, fill=255)
                y += 10
    except Exception as e:
        print(f"⚠️ OLED draw failed: {e}")


def test_buzzer():
    try:
        from gpiozero import Buzzer
        buz = Buzzer(BUZZER_GPIO)

        print("Beeping buzzer...")
        for _ in range(3):
            buz.on()
            time.sleep(0.15)
            buz.off()
            time.sleep(0.15)

        buz.on()
        time.sleep(0.4)
        buz.off()

        print("✅ Buzzer OK (you should have heard beeps).")
        return True
    except Exception as e:
        print(f"❌ Buzzer test failed: {e}")
        return False


def test_ultrasonic(oled_device=None):
    """
    Timeout-safe ultrasonic test.
    Works even if sensor is NOT wired: it will fail fast and return.
    """
    try:
        from gpiozero import DigitalOutputDevice, DigitalInputDevice

        trig = DigitalOutputDevice(US_TRIG_GPIO, initial_value=False)
        echo = DigitalInputDevice(US_ECHO_GPIO, pull_up=False)

        def ping_once(timeout_s=0.03):
            # Send 10us trigger pulse
            trig.off()
            time.sleep(0.0002)
            trig.on()
            time.sleep(0.00001)
            trig.off()

            # Wait for echo to go high
            t0 = time.monotonic()
            while echo.value == 0:
                if time.monotonic() - t0 > timeout_s:
                    return None

            pulse_start = time.monotonic()

            # Wait for echo to go low
            while echo.value == 1:
                if time.monotonic() - pulse_start > timeout_s:
                    return None

            pulse_end = time.monotonic()

            # Distance: speed of sound ~34300 cm/s, round trip -> /2
            duration = pulse_end - pulse_start
            distance_cm = (duration * 34300.0) / 2.0
            return distance_cm

        print("Reading ultrasonic distance for 5 seconds (timeout-safe)...")
        t_end = time.time() + 5.0
        consecutive_none = 0

        while time.time() < t_end:
            cm = ping_once(timeout_s=0.03)

            if cm is None:
                consecutive_none += 1
                msg = "No echo (not wired?)"
                print(msg)

                if oled_device:
                    oled_show(oled_device, [
                        "SELF TEST",
                        "Ultrasonic:",
                        msg,
                        "",
                        f"TRIG={US_TRIG_GPIO} ECHO={US_ECHO_GPIO}",
                    ])

                # Fail fast if clearly not connected
                if consecutive_none >= 3:
                    print("❌ Ultrasonic not responding (expected if not wired). Skipping.")
                    return False

            else:
                consecutive_none = 0
                msg = f"Distance: {cm:6.1f} cm"
                print(msg)

                if oled_device:
                    oled_show(oled_device, [
                        "SELF TEST",
                        "Ultrasonic:",
                        msg,
                        "",
                        f"TRIG={US_TRIG_GPIO} ECHO={US_ECHO_GPIO}",
                    ])

            time.sleep(0.3)

        print("✅ Ultrasonic OK (values printed).")
        return True

    except Exception as e:
        print(f"❌ Ultrasonic test failed: {e}")
        return False



def test_audio():
    """
    Records 3 seconds and plays it back using ALSA tools (arecord/aplay).
    """
    ensure_out_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(OUT_DIR, f"mic_{ts}.wav")

    if shutil.which("arecord") is None or shutil.which("aplay") is None:
        print("❌ arecord/aplay not found. Install: sudo apt install alsa-utils")
        return False

    rec_cmd = ["arecord", "-f", "cd", "-d", "3", wav_path]
    play_cmd = ["aplay", wav_path]

    if ALSA_RECORD_DEV:
        rec_cmd = ["arecord", "-D", ALSA_RECORD_DEV] + rec_cmd[1:]
    if ALSA_PLAY_DEV:
        play_cmd = ["aplay", "-D", ALSA_PLAY_DEV, wav_path]

    try:
        print("Recording 3 seconds from microphone...")
        run_cmd(rec_cmd, timeout=10)
        print(f"Recorded: {wav_path}")

        print("Playing it back now (you should hear your recording)...")
        run_cmd(play_cmd, timeout=10)

        print("✅ Audio OK (record + playback completed).")
        return True
    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        print("Tip: run `arecord -l` and `aplay -l` then set ALSA_RECORD_DEV / ALSA_PLAY_DEV in the script.")
        return False


def test_motors_dry_run():
    """
    Toggles GPIO pins as if an L298N is connected.
    This won't confirm motor movement, but confirms GPIO + logic path.
    """
    try:
        from gpiozero import DigitalOutputDevice, PWMOutputDevice

        ena = PWMOutputDevice(MOTOR_ENA, frequency=1000)
        enb = PWMOutputDevice(MOTOR_ENB, frequency=1000)
        in1 = DigitalOutputDevice(MOTOR_IN1)
        in2 = DigitalOutputDevice(MOTOR_IN2)
        in3 = DigitalOutputDevice(MOTOR_IN3)
        in4 = DigitalOutputDevice(MOTOR_IN4)

        print("Motor DRY-RUN: forward 1s, backward 1s, stop.")

        # forward
        ena.value = 0.6
        enb.value = 0.6
        in1.on(); in2.off()
        in3.on(); in4.off()
        time.sleep(1.0)

        # backward
        in1.off(); in2.on()
        in3.off(); in4.on()
        time.sleep(1.0)

        # stop
        ena.value = 0.0
        enb.value = 0.0
        in1.off(); in2.off(); in3.off(); in4.off()

        print("✅ Motor GPIO dry-run OK (no errors).")
        return True
    except Exception as e:
        print(f"❌ Motor dry-run failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Bot hardware self-test (headless)")
    parser.add_argument("--skip-camera", action="store_true")
    parser.add_argument("--skip-oled", action="store_true")
    parser.add_argument("--skip-buzzer", action="store_true")
    parser.add_argument("--skip-ultrasonic", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-motors", action="store_true")
    args = parser.parse_args()

    print("=== BOT HARDWARE SELF TEST ===")
    print(f"Output dir: {OUT_DIR}")
    ensure_out_dir()

    results = {}

    # OLED init early so other tests can use it
    oled_device = None
    if not args.skip_oled:
        test_i2c_scan()
        oled_device = oled_init()
        if oled_device:
            oled_show(oled_device, ["SELF TEST", "OLED OK", "", "Starting tests..."])
            results["oled"] = True
        else:
            results["oled"] = False

    if not args.skip_camera:
        results["camera"] = test_camera()

    if not args.skip_buzzer:
        results["buzzer"] = test_buzzer()

    if not args.skip_ultrasonic:
        results["ultrasonic"] = test_ultrasonic(oled_device=oled_device)

    if not args.skip_audio:
        results["audio"] = test_audio()

    if not args.skip_motors:
        results["motors_dry_run"] = test_motors_dry_run()

    print("\n=== SUMMARY ===")
    for k, v in results.items():
        print(f"{k:15s}: {'OK' if v else 'FAIL/SKIP'}")

    if oled_device:
        oled_show(oled_device, [
            "SELF TEST DONE",
            f"Cam: {'OK' if results.get('camera') else 'NO'}",
            f"Aud: {'OK' if results.get('audio') else 'NO'}",
            f"US : {'OK' if results.get('ultrasonic') else 'NO'}",
            f"Buz: {'OK' if results.get('buzzer') else 'NO'}",
        ])

    # Exit code = 0 if all non-skipped tests passed
    failed = [k for k, v in results.items() if not v]
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()

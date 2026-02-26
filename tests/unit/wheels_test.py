#!/usr/bin/env python3
"""
motor_test.py — updated for your new wiring

WHEELS (L298N):
- ENA/ENB are NOT connected to the Pi (enabled/tied), so we only control direction:
  Left side pair  = IN1/IN2
  Right side pair = IN3/IN4

STEPPERS (L9110S boards #1 and #2):
- Both boards are driven together from the same 4 GPIO pins, so they rotate together.

Run:
  sudo python3 motor_test.py --test wheels
  sudo python3 motor_test.py --test steppers
  sudo python3 motor_test.py --test all

Optional:
  --dwell 2.0
  --steps 50
  --step-delay 0.01
  --invert-steppers   (reverse direction)
"""

import time
import argparse

try:
    from gpiozero import DigitalOutputDevice
except ImportError:
    raise SystemExit(
        "Missing gpiozero. Install with:\n"
        "  sudo apt update && sudo apt install -y python3-gpiozero\n"
    )

# ----------------------------
# Pin mapping (BCM)
# ----------------------------

# L298N (wheels) – left side on channel A, right side on channel B
# ENA/ENB are enabled/tied; NOT connected to Pi.
L298_IN1 = 5
L298_IN2 = 6
L298_IN3 = 13
L298_IN4 = 19

# L9110S #1 and #2 driven together (same inputs wired to both boards)
S_IA1 = 12
S_IA2 = 20
S_IB1 = 16
S_IB2 = 21


# ----------------------------
# L298N wheels helper
# ----------------------------

class L298NWheels:
    def __init__(self, in1, in2, in3, in4):
        self.in1 = DigitalOutputDevice(in1, initial_value=False)
        self.in2 = DigitalOutputDevice(in2, initial_value=False)
        self.in3 = DigitalOutputDevice(in3, initial_value=False)
        self.in4 = DigitalOutputDevice(in4, initial_value=False)

    def stop(self):
        self.in1.off(); self.in2.off(); self.in3.off(); self.in4.off()

    # Left side
    def left_forward(self):
        self.in1.on(); self.in2.off()

    def left_backward(self):
        self.in1.off(); self.in2.on()

    # Right side
    def right_forward(self):
        self.in3.on(); self.in4.off()

    def right_backward(self):
        self.in3.off(); self.in4.on()

    # Both
    def forward(self):
        self.left_forward()
        self.right_forward()

    def backward(self):
        self.left_backward()
        self.right_backward()

    def spin_left(self):
        # left backward, right forward
        self.left_backward()
        self.right_forward()

    def spin_right(self):
        # left forward, right backward
        self.left_forward()
        self.right_backward()

    def close(self):
        self.stop()
        for dev in (self.in1, self.in2, self.in3, self.in4):
            dev.close()


# ----------------------------
# L9110S bipolar stepper helper
# ----------------------------

class L9110StepperTogether:
    """
    Drives one stepper via L9110S inputs:
      IA1/IA2 = coil A direction
      IB1/IB2 = coil B direction

    In your setup, these same inputs are connected to BOTH L9110S boards,
    so both stepper groups move together.
    """
    def __init__(self, ia1, ia2, ib1, ib2, invert=False):
        self.ia1 = DigitalOutputDevice(ia1, initial_value=False)
        self.ia2 = DigitalOutputDevice(ia2, initial_value=False)
        self.ib1 = DigitalOutputDevice(ib1, initial_value=False)
        self.ib2 = DigitalOutputDevice(ib2, initial_value=False)

        seq = [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]  # full-step 2-phase on
        self.seq = list(reversed(seq)) if invert else seq

    def _set_a(self, d):
        if d > 0:
            self.ia1.on(); self.ia2.off()
        else:
            self.ia1.off(); self.ia2.on()

    def _set_b(self, d):
        if d > 0:
            self.ib1.on(); self.ib2.off()
        else:
            self.ib1.off(); self.ib2.on()

    def step(self, steps: int, delay_s: float):
        if steps == 0:
            return

        direction = 1 if steps > 0 else -1
        n = abs(steps)

        seq = self.seq if direction > 0 else list(reversed(self.seq))

        for i in range(n):
            a_dir, b_dir = seq[i % 4]
            self._set_a(a_dir)
            self._set_b(b_dir)
            time.sleep(delay_s)

    def release(self):
        self.ia1.off(); self.ia2.off(); self.ib1.off(); self.ib2.off()

    def close(self):
        self.release()
        for dev in (self.ia1, self.ia2, self.ib1, self.ib2):
            dev.close()


# ----------------------------
# Tests
# ----------------------------

def test_wheels(dwell: float):
    print("\n=== WHEELS TEST (L298N, direction only / full speed) ===")
    print("Lift robot off ground for safety.\n")

    w = L298NWheels(L298_IN1, L298_IN2, L298_IN3, L298_IN4)
    try:
        print("[1] LEFT forward")
        w.left_forward(); time.sleep(dwell)
        print("[2] STOP")
        w.stop(); time.sleep(0.5)

        print("[3] LEFT backward")
        w.left_backward(); time.sleep(dwell)
        print("[4] STOP")
        w.stop(); time.sleep(0.5)

        print("[5] RIGHT forward")
        w.right_forward(); time.sleep(dwell)
        print("[6] STOP")
        w.stop(); time.sleep(0.5)

        print("[7] RIGHT backward")
        w.right_backward(); time.sleep(dwell)
        print("[8] STOP")
        w.stop(); time.sleep(0.5)

        print("[9] BOTH forward")
        w.forward(); time.sleep(dwell)
        print("[10] BOTH backward")
        w.backward(); time.sleep(dwell)

        print("[11] SPIN left")
        w.spin_left(); time.sleep(dwell)
        print("[12] SPIN right")
        w.spin_right(); time.sleep(dwell)

        print("[13] STOP")
        w.stop()
        print("Wheels test complete.\n")
    finally:
        w.close()


def test_steppers(steps: int, step_delay: float, invert: bool):
    print("\n=== STEPPERS TEST (L9110S boards driven together) ===")
    print("If both L9110S boards share these inputs, both sides should move together.\n")

    s = L9110StepperTogether(S_IA1, S_IA2, S_IB1, S_IB2, invert=invert)
    try:
        print(f"[1] Forward {steps} steps (delay={step_delay}s, invert={invert})")
        s.step(+steps, delay_s=step_delay)
        time.sleep(0.5)

        print(f"[2] Backward {steps} steps")
        s.step(-steps, delay_s=step_delay)
        time.sleep(0.5)

        print("[3] Release coils")
        s.release()
        print("Steppers test complete.\n")
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", choices=["wheels", "steppers", "all"], default="all")
    ap.add_argument("--dwell", type=float, default=2.0, help="Seconds per wheel move")
    ap.add_argument("--steps", type=int, default=50, help="Stepper steps per move")
    ap.add_argument("--step-delay", type=float, default=0.01, help="Delay per step state (seconds)")
    ap.add_argument("--invert-steppers", action="store_true", help="Reverse stepper direction")
    args = ap.parse_args()

    print("Motor test starting. Ctrl+C to stop.\n")
    try:
        if args.test in ("wheels", "all"):
            test_wheels(dwell=args.dwell)

        if args.test in ("steppers", "all"):
            test_steppers(steps=args.steps, step_delay=args.step_delay, invert=args.invert_steppers)

        print("Done.")
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping...")
        time.sleep(0.2)


if __name__ == "__main__":
    main()
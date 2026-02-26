from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import queue as queue_mod
import re
import time
from typing import Protocol

from src.core.message import Message
from src.threads.baseThread import BaseThread

try:
    from gpiozero import DigitalOutputDevice
except Exception:  # pragma: no cover - runtime/hardware dependent
    DigitalOutputDevice = None  # type: ignore[assignment]


@dataclass
class MotionControlConfig:
    # L298N wheels (direction-only control, ENA/ENB tied/enabled)
    wheel_in1: int = 5
    wheel_in2: int = 6
    wheel_in3: int = 13
    wheel_in4: int = 19

    # L9110S stepper control pins
    stepper_ia1: int = 12
    stepper_ia2: int = 20
    stepper_ib1: int = 16
    stepper_ib2: int = 21
    stepper_invert: bool = False
    stepper_steps_per_90deg: int = 50
    stepper_step_delay_s: float = 0.01

    # motion behavior
    move_duration_s: float = 0.8
    turn_duration_s: float = 0.6
    loop_sleep_s: float = 0.05

    # follow behavior
    follow_tolerance_cm: float = 3.0
    follow_pulse_s: float = 0.25
    follow_replan_interval_s: float = 0.25

    # useful for dev without hardware
    dry_run: bool = False


@dataclass
class ParsedMotionCommand:
    action: str
    duration_s: float | None = None
    follow_target_cm: float | None = None


class WheelsDriver(Protocol):
    def stop(self): ...
    def forward(self): ...
    def backward(self): ...
    def spin_left(self): ...
    def spin_right(self): ...
    def close(self): ...


class StepperDriver(Protocol):
    def step(self, steps: int, delay_s: float): ...
    def release(self): ...
    def close(self): ...


class L298NWheels:
    def __init__(self, in1: int, in2: int, in3: int, in4: int):
        if DigitalOutputDevice is None:  # pragma: no cover
            raise RuntimeError("gpiozero DigitalOutputDevice is unavailable")

        self.in1 = DigitalOutputDevice(in1, initial_value=False)
        self.in2 = DigitalOutputDevice(in2, initial_value=False)
        self.in3 = DigitalOutputDevice(in3, initial_value=False)
        self.in4 = DigitalOutputDevice(in4, initial_value=False)

    def stop(self):
        self.in1.off()
        self.in2.off()
        self.in3.off()
        self.in4.off()

    def left_forward(self):
        self.in1.on()
        self.in2.off()

    def left_backward(self):
        self.in1.off()
        self.in2.on()

    def right_forward(self):
        self.in3.on()
        self.in4.off()

    def right_backward(self):
        self.in3.off()
        self.in4.on()

    def forward(self):
        self.left_forward()
        self.right_forward()

    def backward(self):
        self.left_backward()
        self.right_backward()

    def spin_left(self):
        self.left_backward()
        self.right_forward()

    def spin_right(self):
        self.left_forward()
        self.right_backward()

    def close(self):
        self.stop()
        for dev in (self.in1, self.in2, self.in3, self.in4):
            dev.close()


class L9110StepperTogether:
    """Bipolar stepper full-step drive for two mirrored L9110S boards in parallel."""

    def __init__(self, ia1: int, ia2: int, ib1: int, ib2: int, invert: bool = False):
        if DigitalOutputDevice is None:  # pragma: no cover
            raise RuntimeError("gpiozero DigitalOutputDevice is unavailable")

        self.ia1 = DigitalOutputDevice(ia1, initial_value=False)
        self.ia2 = DigitalOutputDevice(ia2, initial_value=False)
        self.ib1 = DigitalOutputDevice(ib1, initial_value=False)
        self.ib2 = DigitalOutputDevice(ib2, initial_value=False)

        seq = [(+1, +1), (-1, +1), (-1, -1), (+1, -1)]
        self.seq = list(reversed(seq)) if invert else seq

    def _set_a(self, d: int):
        if d > 0:
            self.ia1.on()
            self.ia2.off()
        else:
            self.ia1.off()
            self.ia2.on()

    def _set_b(self, d: int):
        if d > 0:
            self.ib1.on()
            self.ib2.off()
        else:
            self.ib1.off()
            self.ib2.on()

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
            time.sleep(max(0.0, delay_s))

    def release(self):
        self.ia1.off()
        self.ia2.off()
        self.ib1.off()
        self.ib2.off()

    def close(self):
        self.release()
        for dev in (self.ia1, self.ia2, self.ib1, self.ib2):
            dev.close()


class NoopWheels:
    def stop(self):
        return

    def forward(self):
        return

    def backward(self):
        return

    def spin_left(self):
        return

    def spin_right(self):
        return

    def close(self):
        return


class NoopStepper:
    def step(self, steps: int, delay_s: float):
        return

    def release(self):
        return

    def close(self):
        return


_SECONDS_RE = re.compile(r"(?P<v>\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b")
_CM_RE = re.compile(r"(?P<v>\d+(?:\.\d+)?)\s*cm\b")
_M_RE = re.compile(r"(?P<v>\d+(?:\.\d+)?)\s*m\b")


def _extract_seconds(command: str) -> float | None:
    m = _SECONDS_RE.search(command)
    if not m:
        return None
    try:
        return max(0.0, float(m.group("v")))
    except Exception:
        return None


def _extract_distance_cm(command: str) -> float | None:
    m = _CM_RE.search(command)
    if m:
        try:
            return max(0.0, float(m.group("v")))
        except Exception:
            return None

    m = _M_RE.search(command)
    if m:
        try:
            return max(0.0, float(m.group("v")) * 100.0)
        except Exception:
            return None

    return None


def parse_motion_command(command_text: str) -> ParsedMotionCommand:
    cmd = command_text.strip().lower()
    if not cmd:
        return ParsedMotionCommand(action="unknown")

    duration_s = _extract_seconds(cmd)

    if any(tok in cmd for tok in ("stop", "halt", "cancel", "freeze")):
        return ParsedMotionCommand(action="stop")

    if "follow" in cmd:
        return ParsedMotionCommand(
            action="follow",
            duration_s=duration_s,
            follow_target_cm=_extract_distance_cm(cmd),
        )

    stepper_hint = any(tok in cmd for tok in ("stepper", "steer", "steering", "head", "pan"))

    if "center" in cmd and stepper_hint:
        return ParsedMotionCommand(action="stepper_center")

    if "left" in cmd:
        if stepper_hint and not any(tok in cmd for tok in ("turn", "move", "go", "rotate")):
            return ParsedMotionCommand(action="stepper_left")
        return ParsedMotionCommand(action="left", duration_s=duration_s)

    if "right" in cmd:
        if stepper_hint and not any(tok in cmd for tok in ("turn", "move", "go", "rotate")):
            return ParsedMotionCommand(action="stepper_right")
        return ParsedMotionCommand(action="right", duration_s=duration_s)

    if any(tok in cmd for tok in ("backward", "back", "reverse")):
        return ParsedMotionCommand(action="backward", duration_s=duration_s)

    if any(tok in cmd for tok in ("forward", "ahead", "front", "straight")):
        return ParsedMotionCommand(action="forward", duration_s=duration_s)

    return ParsedMotionCommand(action="unknown")


class MotionControlThread(BaseThread):
    """Real motor-control thread consuming `motion_command` and `distance_cm` messages."""

    def __init__(
        self,
        message_queue: queue_mod.Queue[Message],
        config: MotionControlConfig,
        wheels: WheelsDriver | None = None,
        stepper: StepperDriver | None = None,
    ):
        super().__init__(name="MotionControlThread", queue=message_queue)
        self.config = config
        self._inbox: queue_mod.Queue[Message] = queue_mod.Queue()

        self.wheels: WheelsDriver
        self.stepper: StepperDriver
        if wheels is not None and stepper is not None:
            self.wheels = wheels
            self.stepper = stepper
        else:
            self.wheels, self.stepper = self._build_hardware_drivers()

        self._latest_distance_cm: float | None = None

        self._follow_enabled = False
        self._follow_target_cm: float | None = None
        self._last_follow_action_ts = 0.0

        self._drive_direction: str | None = None
        self._drive_until = 0.0
        self._is_moving = False

        # -1 = 90° left, 0 = center, +1 = 90° right
        self._stepper_side = 0

    def _build_hardware_drivers(self) -> tuple[WheelsDriver, StepperDriver]:
        if self.config.dry_run:
            logging.info("Motion thread running in dry_run mode (no GPIO control).")
            return NoopWheels(), NoopStepper()

        try:
            wheels = L298NWheels(
                self.config.wheel_in1,
                self.config.wheel_in2,
                self.config.wheel_in3,
                self.config.wheel_in4,
            )
            stepper = L9110StepperTogether(
                self.config.stepper_ia1,
                self.config.stepper_ia2,
                self.config.stepper_ib1,
                self.config.stepper_ib2,
                invert=self.config.stepper_invert,
            )
            return wheels, stepper
        except Exception:
            logging.exception("Failed to initialize motor GPIO drivers; falling back to dry-run")
            return NoopWheels(), NoopStepper()

    def handle_message(self, message: Message):
        if message.type in {"motion_command", "distance_cm"}:
            self._inbox.put(message)

    def _set_motion_state(self, moving: bool):
        if moving == self._is_moving:
            return
        self._is_moving = moving
        self.broadcast_message("motion_state", json.dumps({"moving": moving}))

    def _start_drive(self, direction: str, duration_s: float, now: float | None = None):
        if direction == "forward":
            self.wheels.forward()
        elif direction == "backward":
            self.wheels.backward()
        elif direction == "left":
            self.wheels.spin_left()
        elif direction == "right":
            self.wheels.spin_right()
        else:
            self._stop_drive()
            return

        self._drive_direction = direction
        anchor = now if now is not None else time.monotonic()
        self._drive_until = anchor + max(0.0, duration_s)
        self._set_motion_state(True)

    def _stop_drive(self):
        self.wheels.stop()
        self._drive_direction = None
        self._drive_until = 0.0
        self._set_motion_state(False)

    def _set_stepper_side(self, side: int):
        clamped = max(-1, min(1, side))
        if clamped == self._stepper_side:
            return

        steps_per_side = max(1, int(self.config.stepper_steps_per_90deg))
        delta_steps = (clamped - self._stepper_side) * steps_per_side

        self.stepper.step(delta_steps, delay_s=max(0.0, self.config.stepper_step_delay_s))
        self._stepper_side = clamped

    def execute_command_text(self, command_text: str):
        parsed = parse_motion_command(command_text)
        if parsed.action == "unknown":
            logging.warning("Motion thread ignored unknown command: %r", command_text)
            return

        logging.info("Motion command: raw=%r parsed=%s", command_text, parsed)

        if parsed.action == "stop":
            self._follow_enabled = False
            self._follow_target_cm = None
            self._stop_drive()
            return

        if parsed.action == "follow":
            self._stop_drive()
            # Follow mode should keep longitudinal distance, so steering
            # must be centered before follow pulses begin.
            self._set_stepper_side(0)
            self._follow_enabled = True
            self._follow_target_cm = (
                parsed.follow_target_cm
                if parsed.follow_target_cm is not None
                else self._latest_distance_cm
            )
            self._last_follow_action_ts = 0.0
            logging.info("Follow mode enabled with target=%.2fcm", self._follow_target_cm or -1.0)
            return

        # Any manual command exits follow mode.
        self._follow_enabled = False
        self._follow_target_cm = None

        if parsed.action == "stepper_left":
            self._set_stepper_side(-1)
            return

        if parsed.action == "stepper_right":
            self._set_stepper_side(+1)
            return

        if parsed.action == "stepper_center":
            self._set_stepper_side(0)
            return

        if parsed.action in {"left", "right"}:
            # Keep steering in 180° total range: 90° each side.
            self._set_stepper_side(-1 if parsed.action == "left" else +1)
            self._start_drive(
                parsed.action,
                parsed.duration_s
                if parsed.duration_s is not None
                else max(0.0, self.config.turn_duration_s),
            )
            return

        if parsed.action in {"forward", "backward"}:
            # Respect previous steering state: explicitly return to center
            # before straight drive commands.
            self._set_stepper_side(0)
            self._start_drive(
                parsed.action,
                parsed.duration_s
                if parsed.duration_s is not None
                else max(0.0, self.config.move_duration_s),
            )

    def _handle_distance_message(self, content: str):
        try:
            payload = json.loads(content)
            value = payload.get("value")
            if isinstance(value, (int, float)):
                self._latest_distance_cm = float(value)
        except Exception:
            logging.debug("Invalid distance message payload: %s", content)

    def _handle_motion_message(self, content: str):
        try:
            payload = json.loads(content)
            command = str(payload.get("command", "")).strip()
            if command:
                self.execute_command_text(command)
        except Exception:
            logging.exception("Failed to decode motion_command payload: %s", content)

    def _tick_follow(self, now: float):
        if not self._follow_enabled or self._drive_direction is not None:
            return

        if self._follow_target_cm is None:
            if self._latest_distance_cm is not None:
                self._follow_target_cm = self._latest_distance_cm
            return

        current_cm = self._latest_distance_cm
        if current_cm is None:
            return

        error_cm = current_cm - self._follow_target_cm
        tolerance_cm = max(0.5, float(self.config.follow_tolerance_cm))

        if abs(error_cm) <= tolerance_cm:
            return

        if (now - self._last_follow_action_ts) < max(0.05, self.config.follow_replan_interval_s):
            return

        direction = "forward" if error_cm > 0 else "backward"
        self._start_drive(direction, max(0.05, self.config.follow_pulse_s), now=now)
        self._last_follow_action_ts = now

    def _tick(self, now: float):
        if self._drive_direction is not None and now >= self._drive_until:
            self._stop_drive()

        self._tick_follow(now)

    def _shutdown_hardware(self):
        try:
            self._stop_drive()
        except Exception:
            logging.debug("Motion shutdown: wheels stop failed", exc_info=True)
        try:
            self.stepper.release()
        except Exception:
            logging.debug("Motion shutdown: stepper release failed", exc_info=True)
        try:
            self.wheels.close()
        except Exception:
            logging.debug("Motion shutdown: wheels close failed", exc_info=True)
        try:
            self.stepper.close()
        except Exception:
            logging.debug("Motion shutdown: stepper close failed", exc_info=True)

    def run(self):
        try:
            while self.running:
                now = time.monotonic()

                while True:
                    try:
                        message = self._inbox.get_nowait()
                    except queue_mod.Empty:
                        break

                    if message.type == "distance_cm":
                        self._handle_distance_message(message.content)
                    elif message.type == "motion_command":
                        self._handle_motion_message(message.content)

                self._tick(now)
                time.sleep(max(0.01, self.config.loop_sleep_s))
        finally:
            self._shutdown_hardware()

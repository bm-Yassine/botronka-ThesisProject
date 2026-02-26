from __future__ import annotations

import queue

from src.threads.motion import (
    MotionControlConfig,
    MotionControlThread,
    parse_motion_command,
)


class FakeWheels:
    def __init__(self):
        self.actions: list[str] = []

    def stop(self):
        self.actions.append("stop")

    def forward(self):
        self.actions.append("forward")

    def backward(self):
        self.actions.append("backward")

    def spin_left(self):
        self.actions.append("spin_left")

    def spin_right(self):
        self.actions.append("spin_right")

    def close(self):
        self.actions.append("close")


class FakeStepper:
    def __init__(self):
        self.steps: list[int] = []

    def step(self, steps: int, delay_s: float):
        _ = delay_s
        self.steps.append(steps)

    def release(self):
        return

    def close(self):
        return


def _make_thread() -> tuple[MotionControlThread, FakeWheels, FakeStepper]:
    wheels = FakeWheels()
    stepper = FakeStepper()
    cfg = MotionControlConfig(
        stepper_steps_per_90deg=50,
        stepper_step_delay_s=0.0,
        follow_tolerance_cm=2.0,
        follow_pulse_s=0.2,
        follow_replan_interval_s=0.0,
    )
    t = MotionControlThread(
        message_queue=queue.Queue(),
        config=cfg,
        wheels=wheels,
        stepper=stepper,
    )
    return t, wheels, stepper


def test_parse_motion_command_basic():
    assert parse_motion_command("move forward").action == "forward"
    assert parse_motion_command("go straight now").action == "forward"
    assert parse_motion_command("go back").action == "backward"
    assert parse_motion_command("turn left").action == "left"
    assert parse_motion_command("right").action == "right"
    assert parse_motion_command("follow").action == "follow"
    assert parse_motion_command("stop now").action == "stop"


def test_stepper_turn_range_is_clamped_to_90_each_side():
    thread, wheels, stepper = _make_thread()

    thread.execute_command_text("left")
    thread.execute_command_text("left")  # should not move beyond -90°
    thread.execute_command_text("right")
    thread.execute_command_text("right")  # should not move beyond +90°

    # 0 -> -1: -50 steps, repeated left: 0 steps
    # -1 -> +1: +100 steps, repeated right: 0 steps
    assert stepper.steps == [-50, 100]
    assert "spin_left" in wheels.actions
    assert "spin_right" in wheels.actions


def test_follow_mode_moves_back_and_forth_around_initial_distance():
    thread, wheels, _ = _make_thread()

    # Initial distance at follow command time becomes target.
    thread._latest_distance_cm = 100.0
    thread.execute_command_text("follow")

    # Too far -> move forward pulse.
    thread._latest_distance_cm = 120.0
    now = 10.0
    thread._tick(now)
    assert wheels.actions[-1] == "forward"

    # After pulse duration and near target -> stop.
    thread._latest_distance_cm = 101.0
    thread._tick(now + 1.0)
    assert wheels.actions[-1] == "stop"

    # Too close -> move backward pulse.
    thread._latest_distance_cm = 80.0
    thread._tick(now + 2.0)
    assert wheels.actions[-1] == "backward"


def test_straight_command_recenters_after_turn_before_driving_forward():
    thread, wheels, stepper = _make_thread()

    thread.execute_command_text("right")
    thread.execute_command_text("go straight now")

    # right turn: 0 -> +1 (50 steps), straight: +1 -> 0 (-50 steps)
    assert stepper.steps == [50, -50]
    assert wheels.actions[-1] == "forward"
    assert thread._stepper_side == 0


def test_follow_recenters_steering_before_following_distance():
    thread, wheels, stepper = _make_thread()

    thread.execute_command_text("left")
    thread._latest_distance_cm = 85.0
    thread.execute_command_text("follow")

    # left turn: 0 -> -1, follow enters center first: -1 -> 0
    assert stepper.steps == [-50, 50]
    assert wheels.actions[-1] == "stop"
    assert thread._stepper_side == 0
    assert thread._follow_enabled is True
    assert thread._follow_target_cm == 85.0

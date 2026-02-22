"""
Integration tests for src/ui/buzzer.py

These tests require a physical buzzer wired to the GPIO pin defined in
tests/integration/config.py (default: pin 17).

Run on the Pi with:
    pytest tests/integration/ -m hardware -v

Or skip hardware tests in CI:
    pytest tests/ -m "not hardware"
"""

import json
import queue
import unittest

import pytest

from tests.integration.config import BUZZER_PIN
from src.ui.buzzer import BuzzerThread


@pytest.mark.hardware
class TestBuzzerIntegration(unittest.TestCase):
    """Integration tests that exercise BotBuzzer against real GPIO hardware."""

    def setUp(self):
        self.queue: queue.Queue[str] = queue.Queue()
        self.buzzer = BuzzerThread(self.queue, pin=BUZZER_PIN)

    def tearDown(self):
        # Ensure the buzzer is left in the off state after every test.
        try:
            self.buzzer.buzzer.off()
        except Exception:
            pass

    # ── helper ────────────────────────────────────────────────────────────────

    def _send(self, pattern: str) -> None:
        """Send a buzzer_pattern message through handle_message."""
        msg = json.dumps(
            {
                "emitter": "integration_test",
                "type": "buzzer_pattern",
                "pattern": pattern,
            }
        )
        self.buzzer.handle_message(msg)

    # ── individual pattern tests ───────────────────────────────────────────────

    def test_chirp(self):
        """A single short beep should be audible (0.08 s)."""
        print("\n[integration] Playing 'chirp' — you should hear one short beep.")
        self._send("chirp")

    def test_pattern_stuck(self):
        """Three rapid beeps indicating the robot is stuck."""
        print("\n[integration] Playing 'stuck' — you should hear 3 rapid beeps.")
        self._send("stuck")

    def test_pattern_error(self):
        """Two longer beeps indicating an error condition."""
        print("\n[integration] Playing 'error' — you should hear 2 long beeps.")
        self._send("error")

    def test_pattern_too_close(self):
        """Six very short beeps indicating an obstacle is too close."""
        print(
            "\n[integration] Playing 'too_close' — you should hear 6 very fast beeps."
        )
        self._send("too_close")

    # ── direct method tests ────────────────────────────────────────────────────

    def test_chirp_direct(self):
        """Calling chirp() directly with a custom duration works."""
        print("\n[integration] Direct chirp(0.1) call.")
        self.buzzer.chirp(0.1)

    def test_buzzer_turns_off_after_each_pattern(self):
        """After every pattern the buzzer must be in the off state."""
        for pattern in ("chirp", "stuck", "error", "too_close"):
            with self.subTest(pattern=pattern):
                self._send(pattern)
                # gpiozero Buzzer exposes .is_active (True when on)
                self.assertFalse(
                    self.buzzer.buzzer.is_active,
                    msg=f"Buzzer still active after pattern '{pattern}'",
                )


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for src/ui/buzzer.py

These tests run without any physical hardware. Both `gpiozero.Buzzer` and
`time.sleep` are mocked so the suite is fast and CI-friendly.
"""

import json
import queue
import unittest
from unittest.mock import MagicMock, call, patch


# Patch gpiozero.Buzzer before the module under test is imported so that the
# import itself does not fail on machines without GPIO hardware.
with patch("gpiozero.Buzzer", MagicMock()):
    from src.ui.buzzer import BuzzerThread  # noqa: E402  (import inside patch context)


class TestBotBuzzerChirp(unittest.TestCase):
    """Tests for the BotBuzzer.chirp() method."""

    def setUp(self):
        with patch("gpiozero.Buzzer"):
            self.buzzer = BuzzerThread(queue.Queue(), pin=17)
        # Replace the internal Buzzer instance with a fresh mock
        self.mock_buzzer = MagicMock()
        self.buzzer.buzzer = self.mock_buzzer

    @patch("src.ui.buzzer.time.sleep")
    def test_chirp_default_duration(self, mock_sleep):
        """chirp() with no argument uses 0.08 s."""
        self.buzzer.chirp()

        self.mock_buzzer.on.assert_called_once()
        mock_sleep.assert_called_once_with(0.08)
        self.mock_buzzer.off.assert_called_once()

    @patch("src.ui.buzzer.time.sleep")
    def test_chirp_custom_duration(self, mock_sleep):
        """chirp(t) passes the correct duration to time.sleep."""
        self.buzzer.chirp(0.2)

        mock_sleep.assert_called_once_with(0.2)

    @patch("src.ui.buzzer.time.sleep")
    def test_chirp_call_order(self, mock_sleep):
        """chirp() must call on → sleep → off in that order."""
        call_log = []
        self.mock_buzzer.on.side_effect = lambda: call_log.append("on")
        self.mock_buzzer.off.side_effect = lambda: call_log.append("off")
        mock_sleep.side_effect = lambda _: call_log.append("sleep")

        self.buzzer.chirp()

        self.assertEqual(call_log, ["on", "sleep", "off"])


class TestBotBuzzerPatternStuck(unittest.TestCase):
    """Tests for BotBuzzer.pattern_stuck()."""

    def setUp(self):
        with patch("gpiozero.Buzzer"):
            self.buzzer = BuzzerThread(queue.Queue(), pin=17)
        self.mock_buzzer = MagicMock()
        self.buzzer.buzzer = self.mock_buzzer

    @patch("src.ui.buzzer.time.sleep")
    def test_pattern_stuck_three_chirps(self, mock_sleep):
        """pattern_stuck() should produce exactly 3 on/off cycles."""
        self.buzzer.pattern_stuck()

        self.assertEqual(self.mock_buzzer.on.call_count, 3)
        self.assertEqual(self.mock_buzzer.off.call_count, 3)

    @patch("src.ui.buzzer.time.sleep")
    def test_pattern_stuck_timings(self, mock_sleep):
        """Each chirp uses 0.06 s on and 0.06 s gap between chirps."""
        self.buzzer.pattern_stuck()

        # 3 chirp-sleeps (0.06) + 3 inter-chirp sleeps (0.06) = 6 calls of 0.06
        self.assertEqual(mock_sleep.call_count, 6)
        for c in mock_sleep.call_args_list:
            self.assertAlmostEqual(c.args[0], 0.06)


class TestBotBuzzerPatternError(unittest.TestCase):
    """Tests for BotBuzzer.pattern_error()."""

    def setUp(self):
        with patch("gpiozero.Buzzer"):
            self.buzzer = BuzzerThread(queue.Queue(), pin=17)
        self.mock_buzzer = MagicMock()
        self.buzzer.buzzer = self.mock_buzzer

    @patch("src.ui.buzzer.time.sleep")
    def test_pattern_error_two_beeps(self, mock_sleep):
        """pattern_error() emits exactly 2 on/off cycles."""
        self.buzzer.pattern_error()

        self.assertEqual(self.mock_buzzer.on.call_count, 2)
        self.assertEqual(self.mock_buzzer.off.call_count, 2)

    @patch("src.ui.buzzer.time.sleep")
    def test_pattern_error_timings(self, mock_sleep):
        """pattern_error() uses the correct sleep durations in order."""
        self.buzzer.pattern_error()

        expected_sleeps = [call(0.35), call(0.15), call(0.35)]
        self.assertEqual(mock_sleep.call_args_list, expected_sleeps)

    @patch("src.ui.buzzer.time.sleep")
    def test_pattern_error_call_order(self, mock_sleep):
        """pattern_error() must follow on→sleep(0.35)→off→sleep(0.15)→on→sleep(0.35)→off."""
        call_log = []
        self.mock_buzzer.on.side_effect = lambda: call_log.append("on")
        self.mock_buzzer.off.side_effect = lambda: call_log.append("off")
        mock_sleep.side_effect = lambda t: call_log.append(f"sleep({t})")

        self.buzzer.pattern_error()

        self.assertEqual(
            call_log,
            ["on", "sleep(0.35)", "off", "sleep(0.15)", "on", "sleep(0.35)", "off"],
        )


class TestBotBuzzerPatternTooClose(unittest.TestCase):
    """Tests for BotBuzzer.pattern_too_close()."""

    def setUp(self):
        with patch("gpiozero.Buzzer"):
            self.buzzer = BuzzerThread(queue.Queue(), pin=17)
        self.mock_buzzer = MagicMock()
        self.buzzer.buzzer = self.mock_buzzer

    @patch("src.ui.buzzer.time.sleep")
    def test_pattern_too_close_six_chirps(self, mock_sleep):
        """pattern_too_close() should produce exactly 6 on/off cycles."""
        self.buzzer.pattern_too_close()

        self.assertEqual(self.mock_buzzer.on.call_count, 6)
        self.assertEqual(self.mock_buzzer.off.call_count, 6)

    @patch("src.ui.buzzer.time.sleep")
    def test_pattern_too_close_timings(self, mock_sleep):
        """Each chirp and gap uses 0.03 s."""
        self.buzzer.pattern_too_close()

        self.assertEqual(mock_sleep.call_count, 12)
        for c in mock_sleep.call_args_list:
            self.assertAlmostEqual(c.args[0], 0.03)


class TestBotBuzzerHandleMessage(unittest.TestCase):
    """Tests for BotBuzzer.handle_message() dispatch logic."""

    def setUp(self):
        with patch("gpiozero.Buzzer"):
            self.buzzer = BuzzerThread(queue.Queue(), pin=17)
        self.mock_buzzer = MagicMock()
        self.buzzer.buzzer = self.mock_buzzer

    def _make_msg(self, pattern: str) -> str:
        return json.dumps(
            {"emitter": "test", "type": "buzzer_pattern", "pattern": pattern}
        )

    @patch.object(BuzzerThread, "chirp")
    def test_dispatches_chirp(self, mock_chirp):
        self.buzzer.handle_message(self._make_msg("chirp"))
        mock_chirp.assert_called_once()

    @patch.object(BuzzerThread, "pattern_stuck")
    def test_dispatches_stuck(self, mock_stuck):
        self.buzzer.handle_message(self._make_msg("stuck"))
        mock_stuck.assert_called_once()

    @patch.object(BuzzerThread, "pattern_error")
    def test_dispatches_error(self, mock_error):
        self.buzzer.handle_message(self._make_msg("error"))
        mock_error.assert_called_once()

    @patch.object(BuzzerThread, "pattern_too_close")
    def test_dispatches_too_close(self, mock_too_close):
        self.buzzer.handle_message(self._make_msg("too_close"))
        mock_too_close.assert_called_once()

    def test_unknown_pattern_does_nothing(self):
        """An unrecognised pattern should not call any buzzer method."""
        msg = json.dumps(
            {"emitter": "test", "type": "buzzer_pattern", "pattern": "unknown"}
        )
        # Should not raise and should not touch the hardware mock
        self.buzzer.handle_message(msg)
        self.mock_buzzer.on.assert_not_called()
        self.mock_buzzer.off.assert_not_called()

    def test_wrong_message_type_ignored(self):
        """Messages with a type other than 'buzzer_pattern' are silently ignored."""
        msg = json.dumps({"emitter": "test", "type": "oled_update", "data": "hello"})
        self.buzzer.handle_message(msg)
        self.mock_buzzer.on.assert_not_called()
        self.mock_buzzer.off.assert_not_called()

    def test_missing_pattern_key_ignored(self):
        """A buzzer_pattern message with no 'pattern' key is silently ignored."""
        msg = json.dumps({"emitter": "test", "type": "buzzer_pattern"})
        self.buzzer.handle_message(msg)
        self.mock_buzzer.on.assert_not_called()
        self.mock_buzzer.off.assert_not_called()


if __name__ == "__main__":
    unittest.main()

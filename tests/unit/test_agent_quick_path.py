from __future__ import annotations

from src.agent.nlu import quick_rule_decision


def test_quick_rule_greeting_and_identity():
    greet = quick_rule_decision("hello botronka")
    assert greet is not None
    assert greet.type == "chat"
    assert greet.command is None

    identity = quick_rule_decision("who are you")
    assert identity is not None
    assert "biedronka" in identity.speak.lower()


def test_quick_rule_motion_shortcuts():
    stop = quick_rule_decision("stop")
    assert stop is not None
    assert stop.type == "command"
    assert stop.command == "stop"

    right = quick_rule_decision("turn right")
    assert right is not None
    assert right.type == "command"
    assert "right" in (right.command or "")

    forward = quick_rule_decision("go straight now")
    assert forward is not None
    assert forward.type == "command"
    assert "straight" in (forward.command or "")

    beep = quick_rule_decision("beep")
    assert beep is not None
    assert beep.type == "command"
    assert beep.command == "beep"


def test_quick_rule_time_is_answered_locally():
    ans = quick_rule_decision("what time is it")
    assert ans is not None
    assert ans.type == "chat"
    assert ans.command is None
    assert ans.speak.lower().startswith("it is ")


def test_quick_rule_returns_none_for_nontrivial_question():
    assert quick_rule_decision("what is the capital of poland") is None

from __future__ import annotations

from dataclasses import dataclass


MOVEMENT_COMMANDS = {"move", "turn", "rotate", "go", "back", "forward", "follow"}
FORBIDDEN_TOKENS = {"shutdown", "reboot", "format", "delete", "rm", "poweroff"}


def can_execute(command_name: str, trust_level: int, min_move: int) -> bool:
    if command_name in MOVEMENT_COMMANDS:
        return trust_level >= min_move
    return True


def normalize_command_name(command: str | None) -> str:
    if not command:
        return ""
    return command.strip().split()[0].lower() if command.strip() else ""


@dataclass
class PolicyResult:
    allowed: bool
    reason: str = ""


def evaluate_command(command: str | None, trust_level: int, min_move: int) -> PolicyResult:
    if not command:
        return PolicyResult(allowed=True)

    low = command.lower()
    if any(tok in low for tok in FORBIDDEN_TOKENS):
        return PolicyResult(allowed=False, reason="unsafe command")

    command_name = normalize_command_name(command)
    if not can_execute(command_name, trust_level, min_move):
        return PolicyResult(
            allowed=False,
            reason=f"trust level {trust_level} is too low for {command_name}",
        )

    return PolicyResult(allowed=True)
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import re


@dataclass
class AgentDecision:
    type: str = "chat"
    speak: str = ""
    command: str | None = None
    requires_trust: int = 0
    raw: dict = field(default_factory=dict)


@dataclass
class AdminIntent:
    type: str = "none"  # none | promote | register_guest
    name: str = ""
    trust_level: str = ""


DEFAULT_SYSTEM_PROMPT = (
    "You are Botronka, a tiny helpful ladybug robot (biedronka) companion. "
    "Personality: warm, brave, practical, and concise. "
    "For normal chat, sound like a friendly little ladybug helper. "
    "Answer accurately from what you know; if unsure, say so briefly. "
    "Reply with JSON only and no markdown. "
    "Schema: {\"type\":\"chat|command\",\"speak\":\"string\","
    "\"command\":\"string|null\",\"requires_trust\":0-3}. "
    "If user asks robot movement, set type=command and command to concise instruction. "
    "If no movement/action needed, command must be null and type=chat. "
    "The speak field must be short and natural."
)


DEFAULT_FEW_SHOT_MESSAGES: list[dict[str, str]] = [
    {"role": "user", "content": "move forward 10 cm"},
    {
        "role": "assistant",
        "content": '{"type":"command","speak":"On it! Little ladybug moving forward ten centimeters.","command":"move forward 10cm","requires_trust":2}',
    },
    {"role": "user", "content": "what is your name?"},
    {
        "role": "assistant",
        "content": '{"type":"chat","speak":"I am Botronka, your tiny biedronka helper.","command":null,"requires_trust":0}',
    },
]


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def quick_rule_decision(user_text: str) -> AgentDecision | None:
    """Fast local intent path to avoid LLM latency for common requests."""
    t = _normalize_text(user_text)
    if not t:
        return None

    # Lightweight conversational shortcuts.
    if t in {"hi", "hello", "hey", "hello botronka", "hi botronka", "hey botronka"}:
        return AgentDecision(
            type="chat",
            speak="Hi! Botronka the little ladybug is here to help.",
            command=None,
            requires_trust=0,
            raw={"fast_path": "greeting"},
        )

    if "your name" in t or "who are you" in t:
        return AgentDecision(
            type="chat",
            speak="I am Botronka, your tiny biedronka helper.",
            command=None,
            requires_trust=0,
            raw={"fast_path": "identity"},
        )

    if any(tok in t for tok in ("thank you", "thanks", "thx")):
        return AgentDecision(
            type="chat",
            speak="You are welcome. Happy to help.",
            command=None,
            requires_trust=0,
            raw={"fast_path": "thanks"},
        )

    if any(
        tok in t
        for tok in (
            "what time is it",
            "what's the time",
            "tell me the time",
            "current time",
            "time now",
        )
    ):
        now_str = datetime.now().strftime("%H:%M")
        return AgentDecision(
            type="chat",
            speak=f"It is {now_str}.",
            command=None,
            requires_trust=0,
            raw={"fast_path": "time"},
        )

    if any(tok in t for tok in ("status", "show status", "how are you now")):
        return AgentDecision(
            type="chat",
            speak="Checking my status.",
            command=None,
            requires_trust=0,
            raw={"fast_path": "status"},
        )

    if any(tok in t for tok in ("beep", "chirp", "buzz")):
        return AgentDecision(
            type="command",
            speak="Beep beep.",
            command="beep",
            requires_trust=0,
            raw={"fast_path": "beep"},
        )

    # Low-latency motion intent shortcuts.
    if any(tok in t for tok in ("stop", "halt", "cancel")):
        return AgentDecision(
            type="command",
            speak="Stopping now.",
            command="stop",
            requires_trust=2,
            raw={"fast_path": "motion_stop"},
        )

    if "follow" in t:
        return AgentDecision(
            type="command",
            speak="Got it. I will follow your distance.",
            command=t,
            requires_trust=2,
            raw={"fast_path": "motion_follow"},
        )

    if any(tok in t for tok in ("forward", "ahead", "straight")):
        return AgentDecision(
            type="command",
            speak="Moving forward.",
            command=t,
            requires_trust=2,
            raw={"fast_path": "motion_forward"},
        )

    if any(tok in t for tok in ("backward", "go back", "reverse", "move back")):
        return AgentDecision(
            type="command",
            speak="Moving backward.",
            command=t,
            requires_trust=2,
            raw={"fast_path": "motion_backward"},
        )

    if "left" in t and any(tok in t for tok in ("turn", "go", "move", "left")):
        return AgentDecision(
            type="command",
            speak="Turning left.",
            command=t,
            requires_trust=2,
            raw={"fast_path": "motion_left"},
        )

    if "right" in t and any(tok in t for tok in ("turn", "go", "move", "right")):
        return AgentDecision(
            type="command",
            speak="Turning right.",
            command=t,
            requires_trust=2,
            raw={"fast_path": "motion_right"},
        )

    return None


def _extract_first_json_blob(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def parse_agent_reply(raw_text: str) -> AgentDecision:
    blob = _extract_first_json_blob(raw_text.strip())
    if not blob:
        return AgentDecision(type="chat", speak=raw_text.strip(), command=None, requires_trust=0)

    try:
        data = json.loads(blob)
    except Exception:
        return AgentDecision(type="chat", speak=raw_text.strip(), command=None, requires_trust=0)

    decision_type = str(data.get("type", "chat")).strip().lower()
    if decision_type not in {"chat", "command"}:
        decision_type = "chat"

    speak = str(data.get("speak", "")).strip()
    if not speak:
        speak = "I heard you."

    command_raw = data.get("command")
    command = None if command_raw is None else str(command_raw).strip()
    if command == "":
        command = None

    try:
        requires_trust = int(data.get("requires_trust", 0))
    except Exception:
        requires_trust = 0
    requires_trust = max(0, min(3, requires_trust))

    if decision_type == "chat":
        command = None

    return AgentDecision(
        type=decision_type,
        speak=speak,
        command=command,
        requires_trust=requires_trust,
        raw=data if isinstance(data, dict) else {},
    )


_PROMOTE_RE = re.compile(
    r"\bpromote\s+[\"']?(?P<name>[a-zA-Z][\w\- ]{0,30})[\"']?\s+to\s+(?P<level>friend|owner)\b",
    re.IGNORECASE,
)
_REGISTER_RE_PATTERNS = [
    re.compile(r"\bregister\s+me\s+as\s+[\"']?(?P<name>[a-zA-Z][\w\- ]{0,30})[\"']?\b", re.IGNORECASE),
    re.compile(r"\bmy\s+name\s+is\s+[\"']?(?P<name>[a-zA-Z][\w\- ]{0,30})[\"']?\b", re.IGNORECASE),
]


def parse_admin_intent(user_text: str) -> AdminIntent:
    text = user_text.strip()
    if not text:
        return AdminIntent()

    m = _PROMOTE_RE.search(text)
    if m:
        name = m.group("name").strip(" \"'")
        level_raw = m.group("level").strip().lower()
        level = "Friend" if level_raw == "friend" else "OWNER"
        return AdminIntent(type="promote", name=name, trust_level=level)

    for rgx in _REGISTER_RE_PATTERNS:
        r = rgx.search(text)
        if r:
            name = r.group("name").strip(" \"'")
            return AdminIntent(type="register_guest", name=name, trust_level="Guest")

    return AdminIntent()
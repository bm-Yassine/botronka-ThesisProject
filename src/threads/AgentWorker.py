from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import queue as queue_mod
import time
from pathlib import Path

from src.agent.admin import promote_person
from src.agent.llm_client import LLMConfig, ask_llm
from src.agent.nlu import (
    AgentDecision,
    AdminIntent,
    DEFAULT_FEW_SHOT_MESSAGES,
    DEFAULT_SYSTEM_PROMPT,
    parse_admin_intent,
    parse_agent_reply,
    quick_rule_decision,
)
from src.agent.policy import evaluate_command
from src.core.message import Message
from src.core.state import RuntimeStateStore
from src.threads.baseThread import BaseThread


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACE_DB_PATH = str(_PROJECT_ROOT / "data" / "people" / "face_db.json")
DEFAULT_TRUST_MAP_PATH = str(_PROJECT_ROOT / "data" / "people" / "trust_map.json")


@dataclass
class AgentWorkerConfig:
    min_move_trust: int = 2
    owner_trust_level: int = 3
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    face_db_path: str = DEFAULT_FACE_DB_PATH
    trust_map_path: str = DEFAULT_TRUST_MAP_PATH
    register_timeout_s: float = 18.0
    register_countdown_steps: int = 3
    register_countdown_interval_s: float = 0.6
    use_few_shot: bool = True
    enable_filler_speech: bool = True
    filler_phrases: list[str] = field(
        default_factory=lambda: [
            "Working on it.",
            "Let me think.",
            "Hmm, gotcha.",
        ]
    )


class AgentWorker(BaseThread):
    def __init__(
        self,
        message_queue: queue_mod.Queue[Message],
        llm_config: LLMConfig,
        state_store: RuntimeStateStore,
        config: AgentWorkerConfig | None = None,
        few_shot_messages: list[dict[str, str]] | None = None,
    ):
        super().__init__(name="AgentWorker", queue=message_queue)
        self.llm_config = llm_config
        self.state_store = state_store
        self.config = config or AgentWorkerConfig()
        self.few_shot_messages = few_shot_messages or DEFAULT_FEW_SHOT_MESSAGES
        self._inbox: queue_mod.Queue[Message] = queue_mod.Queue()
        self._register_results: queue_mod.Queue[Message] = queue_mod.Queue()
        self._filler_idx = 0

    def handle_message(self, message: Message):
        if message.type == "stt_text":
            self._inbox.put(message)
        elif message.type == "vision_register_result":
            self._register_results.put(message)

    def _next_filler(self) -> str | None:
        if not self.config.enable_filler_speech or not self.config.filler_phrases:
            return None
        phrase = self.config.filler_phrases[self._filler_idx % len(self.config.filler_phrases)]
        self._filler_idx += 1
        return phrase

    def _build_denied_decision(self, reason: str) -> AgentDecision:
        return AgentDecision(
            type="chat",
            speak=f"I cannot do that right now: {reason}.",
            command=None,
            requires_trust=0,
            raw={"policy_denied": reason},
        )

    def _emit_reply(self, user_text: str, decision: AgentDecision):
        reply_payload = {
            "type": decision.type,
            "speak": decision.speak,
            "command": decision.command,
            "requires_trust": decision.requires_trust,
            "user_text": user_text,
            "raw": decision.raw,
            "created_at": time.time(),
        }
        self.broadcast_message(
            "agent_reply",
            json.dumps(reply_payload, ensure_ascii=False),
        )

        self.broadcast_message(
            "tts_request",
            json.dumps(
                {
                    "text": decision.speak,
                    "is_filler": False,
                    "command": decision.command,
                    "created_at": time.time(),
                },
                ensure_ascii=False,
            ),
        )

        if decision.command:
            command_name = decision.command.strip().split()[0].lower()
            if command_name in {"beep", "buzz", "chirp"}:
                # Fast local utility command routed directly to buzzer thread.
                self.broadcast_message(
                    "buzzer_countdown",
                    json.dumps(
                        {
                            "steps": 1,
                            "interval_s": 0.05,
                            "created_at": time.time(),
                        },
                        ensure_ascii=False,
                    ),
                )
            else:
                self.broadcast_message(
                    "motion_command",
                    json.dumps(
                        {
                            "command": decision.command,
                            "created_at": time.time(),
                        },
                        ensure_ascii=False,
                    ),
                )

    def _drain_register_results(self):
        while True:
            try:
                self._register_results.get_nowait()
            except queue_mod.Empty:
                return

    def _wait_for_register_result(self) -> dict:
        deadline = time.monotonic() + max(1.0, self.config.register_timeout_s)
        while time.monotonic() < deadline and self.running:
            timeout = min(0.3, deadline - time.monotonic())
            if timeout <= 0:
                break
            try:
                msg = self._register_results.get(timeout=timeout)
            except queue_mod.Empty:
                continue

            try:
                data = json.loads(msg.content)
                if isinstance(data, dict):
                    return data
            except Exception:
                continue

        return {"ok": False, "error": "registration timed out"}

    def _handle_admin_intent(
        self,
        user_text: str,
        admin_intent: AdminIntent,
        trust_level: int,
    ) -> bool:
        if admin_intent.type == "none":
            return False

        if admin_intent.type == "promote":
            if trust_level < self.config.owner_trust_level:
                self._emit_reply(
                    user_text,
                    self._build_denied_decision(
                        "only the owner can promote trust levels"
                    ),
                )
                return True

            res = promote_person(
                name=admin_intent.name,
                target_level=admin_intent.trust_level,
                trust_map_path=self.config.trust_map_path,
                face_db_path=self.config.face_db_path,
            )
            speak = (
                res.message
                if not res.ok
                else f"Done. {admin_intent.name} is now {admin_intent.trust_level}."
            )
            self._emit_reply(
                user_text,
                AgentDecision(
                    type="chat",
                    speak=speak,
                    command=None,
                    requires_trust=0,
                    raw={
                        "admin_intent": "promote",
                        "ok": res.ok,
                        "message": res.message,
                    },
                ),
            )
            return True

        if admin_intent.type == "register_guest":
            # Unknown person can self-register as Guest by saying their name.
            # Owner can also trigger registration for supervised onboarding.
            allowed = trust_level in (0, self.config.owner_trust_level)
            if not allowed:
                self._emit_reply(
                    user_text,
                    self._build_denied_decision(
                        "registration is only for unknown users or the owner"
                    ),
                )
                return True

            self._drain_register_results()
            self.broadcast_message(
                "buzzer_countdown",
                json.dumps(
                    {
                        "steps": self.config.register_countdown_steps,
                        "interval_s": self.config.register_countdown_interval_s,
                        "created_at": time.time(),
                    }
                ),
            )
            # Keep ordering deterministic even if buzzer is made async later.
            time.sleep(
                max(0.0, self.config.register_countdown_steps)
                * max(0.05, self.config.register_countdown_interval_s)
                + 0.1
            )
            self.broadcast_message(
                "vision_register_request",
                json.dumps(
                    {
                        "name": admin_intent.name,
                        "trust_level": "Guest",
                        "created_at": time.time(),
                    },
                    ensure_ascii=False,
                ),
            )

            result = self._wait_for_register_result()
            ok = bool(result.get("ok", False))
            if ok:
                speak = f"{admin_intent.name} has been registered as your guest."
            else:
                speak = f"I could not register {admin_intent.name}. {result.get('error', 'Please try again.')}"

            self._emit_reply(
                user_text,
                AgentDecision(
                    type="chat",
                    speak=speak,
                    command=None,
                    requires_trust=0,
                    raw={
                        "admin_intent": "register_guest",
                        "result": result,
                    },
                ),
            )
            return True

        return False

    def run(self):
        while self.running:
            try:
                message = self._inbox.get(timeout=0.1)
            except queue_mod.Empty:
                continue

            try:
                payload = json.loads(message.content)
                user_text = str(payload.get("text", "")).strip()
                if not user_text:
                    continue

                trust_level = self.state_store.snapshot().trust.value
                admin_intent = parse_admin_intent(user_text)
                if self._handle_admin_intent(user_text, admin_intent, trust_level):
                    continue

                # Fast local path for common intents to reduce latency.
                fast_decision = quick_rule_decision(user_text)
                if fast_decision is not None:
                    logging.info(
                        "Agent fast path hit: kind=%s text=%r",
                        fast_decision.raw.get("fast_path"),
                        user_text,
                    )
                    policy = evaluate_command(
                        command=fast_decision.command,
                        trust_level=trust_level,
                        min_move=self.config.min_move_trust,
                    )
                    if not policy.allowed:
                        fast_decision = self._build_denied_decision(policy.reason)

                    self._emit_reply(user_text=user_text, decision=fast_decision)
                    continue

                filler = self._next_filler()
                if filler:
                    self.broadcast_message(
                        "tts_request",
                        json.dumps(
                            {
                                "text": filler,
                                "is_filler": True,
                                "created_at": time.time(),
                            },
                            ensure_ascii=False,
                        ),
                    )

                self.broadcast_message("llm_thinking", json.dumps({"value": True}))
                llm_started = time.perf_counter()
                raw_reply = ask_llm(
                    cfg=self.llm_config,
                    system_prompt=self.config.system_prompt,
                    user_text=user_text,
                    few_shot_messages=(
                        self.few_shot_messages if self.config.use_few_shot else None
                    ),
                )
                llm_ms = (time.perf_counter() - llm_started) * 1000.0
                logging.info(
                    "Audio timing: stage=llm duration_ms=%.1f chars_in=%d chars_out=%d",
                    llm_ms,
                    len(user_text),
                    len(raw_reply),
                )
                decision = parse_agent_reply(raw_reply)

                policy = evaluate_command(
                    command=decision.command,
                    trust_level=trust_level,
                    min_move=self.config.min_move_trust,
                )
                if not policy.allowed:
                    decision = self._build_denied_decision(policy.reason)

                self._emit_reply(user_text=user_text, decision=decision)
            except Exception as e:
                logging.exception("AgentWorker failed to process STT text")
                self.broadcast_message(
                    "agent_error",
                    json.dumps({"error": str(e), "created_at": time.time()}),
                )
            finally:
                self.broadcast_message("llm_thinking", json.dumps({"value": False}))
from dataclasses import dataclass


@dataclass
class Message:
    sender: str
    type: str
    content: str
    sent_at: float

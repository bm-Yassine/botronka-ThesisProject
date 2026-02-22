import threading
import logging
import queue
import time

from src.core.message import Message


class BaseThread:
    def __init__(self, name: str, queue: queue.Queue[Message]):
        self.name = name
        self.queue = queue
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self):
        if self.thread is not None:
            raise RuntimeError(f"Thread {self.name} is already running.")
        self.thread = threading.Thread(target=self.run, name=self.name, daemon=True)
        self.running = True
        self.thread.start()

    def run(self) -> None:
        while self.running:
            time.sleep(1.0)  # Keep thread alive
        if self.thread is not None:
            self.thread.join()

    def handle_message(self, message: Message):
        pass

    def broadcast_message(self, type: str, content: str):
        logging.info("Thread %s broadcasting message: %s", self.name, content)
        self.queue.put(
            Message(sender=self.name, type=type, content=content, sent_at=time.time())
        )

    def stop(self):
        logging.info("Stopping thread: %s", self.name)
        self.running = False

import queue
import logging

from src.threads.baseThread import BaseThread
from src.core.message import Message


class ThreadManager:
    def __init__(self, queue: queue.Queue[Message]):
        self.threads: dict[str, BaseThread] = {}
        self.queue = queue
        self.running = False

    def add_thread(self, thread: BaseThread):
        self.threads[thread.name] = thread

    def broadcast_message(self, msg: Message):
        logging.debug("Broadcasting message to all threads: %s", msg)
        for thread in self.threads.values():
            thread.handle_message(msg)

    def stop_all(self):
        for thread in self.threads.values():
            thread.stop()
        self.running = False

    def start_all(self):
        self.running = True
        logging.info("Starting all threads...")
        for thread in self.threads.values():
            logging.debug("Starting thread: %s", thread.name)
            thread.start()

    def run(self):
        try:
            while self.running:
                # If the queue is not empty, broadcast the message to all threads
                try:
                    msg = self.queue.get(timeout=0.05)
                    self.broadcast_message(msg)
                except queue.Empty:
                    pass

        except KeyboardInterrupt:
            logging.info("Stopping all threads...")
            self.stop_all()

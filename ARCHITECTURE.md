# Botronka Software Architecture diagram

The Botronka architecture is designed to be modular and easily extensible.

The core of the system is the `App` class, that contains the `ThreadManager` and the message queue system.
The individual threads are responsible for interacting with the hardware, and all events are sent to a message queue, and then redistributed to every thread. Each thread can choose to process or ignore the messages depending on their type and content.

```mermaid
graph TD
    %% Node Definitions & Styling
    classDef core fill:#2d3436,color:#fff,stroke:#636e72,stroke-width:2px;
    classDef thread fill:#0984e3,color:#fff,stroke:#74b9ff,stroke-width:2px;
    classDef hardware fill:#e17055,color:#fff,stroke:#fab1a0,stroke-width:2px;
    classDef queue fill:#6c5ce7,color:#fff,stroke:#a29bfe,stroke-width:2px;

    subgraph Botronka [Botronka System Architecture]
        direction TB

        %% Orchestration
        App([Start App]):::core --> ThreadMgr[Thread Manager]:::core
        
        %% The Hub
        MessageQueue[[Message Queue / Bus]]:::queue

        %% Thread Logic
        subgraph Workers [Worker Threads]
            direction LR
            CamT[Camera Thread]:::thread
            DispT[OLED Thread]:::thread
            DistT[Distance Thread]:::thread
        end

        %% Hardware Logic
        subgraph Physical [Physical Hardware]
            direction LR
            PiCam[/Pi Camera/]:::hardware
            OLED[/OLED Display/]:::hardware
            USensor[/Ultrasonic Sensor/]:::hardware
        end

        %% Connections
        ThreadMgr --> CamT
        ThreadMgr --> DispT
        ThreadMgr --> DistT

        %% Message Flow (Simplified)
        CamT <--> MessageQueue
        DispT <--> MessageQueue
        DistT <--> MessageQueue

        %% Hardware Interaction
        CamT --- PiCam
        DispT --- OLED
        DistT --- USensor
    end
```

The base components of a thread are illustrated by the `BaseThread` class:

```py
import threading
import queue


class BaseThread:
    def __init__(self, name: str, queue: queue.Queue[str]):
        #...

    def start(self):
        self.thread = threading.Thread(...)
        self.thread.start()

    def run(self) -> None:
        ...

    def handle_message(self, msg: str):
        ...

    def broadcast_message(self, msg: str):
        self.queue.put(msg)

    def stop(self):
        self.running = False
        ...
```

Whenever a message gets put into the queue, the `ThreadManager` is responsible for distributing it to all threads, by calling their `handle_message` method. Each thread can then choose to process or ignore the message depending on its type and content.

import uuid
from langgraph.checkpoint.memory import MemorySaver
from logging_config.logger import LOG
from typing import Optional
# singleton_state_tracker.py
class StateTracker:
    _instance = None
    memory: Optional[MemorySaver]
    thread: Optional[dict]
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(StateTracker, cls).__new__(cls)
            cls._instance.thread = kwargs.get("thread", None)
            cls._instance.memory = kwargs.get("memory", MemorySaver())
        else:
            if "memory" in kwargs or "thread" in kwargs:
                LOG.warning("StateTracker already initialized. Ignoring new memory/thread values.")
        return cls._instance

    def create_new_state(self):
        if self.thread is None:
            self.thread = {"configurable": {"thread_id": str(uuid.uuid4())}}
        return self.thread


import uuid
from langgraph.checkpoint.memory import MemorySaver
from logging_config.logger import LOG
# singleton_state_tracker.py
class StateTracker:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(StateTracker, cls).__new__(cls)
            cls._instance.memory = kwargs.get("memory", None)
            cls._instance.thread = kwargs.get("thread", None)
        return cls._instance

    def create_new_state(self):
        if self.thread is None:
            self.thread = {"configurable": {"thread_id": uuid.uuid4()}}
        return self.thread


import uuid
from langgraph.checkpoint.memory import MemorySaver
from logging_config.logger import LOG
from typing import Optional
from langchain_core.runnables import RunnableConfig

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

    def get_runnable_config(self) -> RunnableConfig:
        """
        Returns a RunnableConfig with the current thread.
        """
        if self.thread is None:
            raise ValueError("Thread is not initialized.")
        return RunnableConfig(configurable=self.thread['configurable'])
    
    def get_current_state(self) -> dict:
        """
        Returns the current state of the tracker.
        """
        if self.memory is None or self.thread is None:
            raise ValueError("Memory or thread is not initialized.")
        checkpoint_tuple =  self.memory.get_tuple(self.get_runnable_config())

        if checkpoint_tuple is None:
            LOG.warning("No checkpoint found for the current runnable config.")
            return {}
        LOG.info(f'checkpoint tuple is LEN {len(checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", []))} for config {self.get_runnable_config()} and memory id {id(self.memory)}')
        return checkpoint_tuple.checkpoint.get('channel_values', {})
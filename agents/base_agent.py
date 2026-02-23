from abc import ABC, abstractmethod
from datetime import datetime, timezone
from loguru import logger


class BaseAgent(ABC):
    """Abstract base class for all agents. Defines the execute/run contract."""

    def __init__(self, name: str):
        self.name = name
        self.last_run = None
        self.logger = logger.bind(agent=name)

    @abstractmethod
    def run(self, input_data: dict) -> dict:
        """Core processing method. Receives input dict, returns output dict."""
        pass

    def pre_run(self):
        self.logger.info(f"Agent '{self.name}' starting...")

    def post_run(self, output: dict):
        self.last_run = datetime.now(timezone.utc)
        self.logger.info(f"Agent '{self.name}' completed.")

    def execute(self, input_data: dict) -> dict:
        """Template method: pre_run -> run -> post_run."""
        self.pre_run()
        output = self.run(input_data)
        self.post_run(output)
        return output

from client import create_llm_client
from prompt_builder import create_prompt_builder
from .memory_agent import MemoryAgent

class AgentFactory:
    def __init__(self, config):
        self.config = config

    def create_agent(self):
        client_factory = create_llm_client(self.config.client)
        prompt_builder = create_prompt_builder(self.config.agent)
        if self.config.agent.type == "memory":
            return MemoryAgent(client_factory, prompt_builder, config=self.config)
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent}")

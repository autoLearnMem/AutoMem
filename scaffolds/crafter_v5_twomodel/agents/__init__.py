from client import create_llm_client
from prompt_builder import create_prompt_builder
from .memory_agent import MemoryAgent

from omegaconf import OmegaConf

class AgentFactory:
    def __init__(self, config):
        self.config = config

    def create_agent(self):
        client_factory = create_llm_client(self.config.client)
        prompt_builder = create_prompt_builder(self.config.agent)

        gameplay_client_factory = None
        gameplay_cfg = OmegaConf.select(self.config, "client_gameplay", default=None)
        if gameplay_cfg is not None:
            gameplay_client_factory = create_llm_client(gameplay_cfg)

        if self.config.agent.type == "memory":
            return MemoryAgent(
                client_factory, prompt_builder, config=self.config,
                gameplay_client_factory=gameplay_client_factory,
            )
        else:
            raise ValueError(f"Unknown agent type: {self.config.agent}")
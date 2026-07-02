import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from agents import AgentFactory
from evaluator import EvaluatorManager
from utils import collect_and_summarize_results, print_summary_table, setup_environment

@contextmanager
def redirect_to_file(filepath):
    original = sys.stdout
    with open(filepath, "w") as file:
        sys.stdout = file
        try:
            yield
        finally:
            sys.stdout = original

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(config: DictConfig):
    original_cwd = get_original_cwd()
    setup_environment(original_cwd=original_cwd)

    if config.eval.resume_from is not None:
        output_dir = config.eval.resume_from
    else:
        run_name = OmegaConf.select(config, "eval.run_name", default=None)
        if run_name:
            run_name = str(run_name)
        else:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
            run_name = f"{timestamp}_{config.agent.type}_{config.client.model_id.replace('/', '_')}"
        output_dir = os.path.join(config.eval.output_dir, run_name)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_filename = os.path.join(output_dir, "eval.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename)],
        force=True,
    )

    evaluator_manager = EvaluatorManager(config, original_cwd=original_cwd, output_dir=output_dir)
    agent_factory = AgentFactory(config)
    with redirect_to_file(log_filename):
        evaluator_manager.run(agent_factory)

    summary = collect_and_summarize_results(output_dir)
    print_summary_table(summary)

if __name__ == "__main__":
    main()
import csv
import json
import logging
import multiprocessing
import os
import random
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from environments import make_env
from utils import get_unique_seed

logger = logging.getLogger(__name__)

class EvaluatorManager:
    def __init__(self, config, original_cwd="", output_dir="."):
        self.config = config
        self.original_cwd = original_cwd
        self.output_dir = output_dir

        self.env_names = config.envs.names.split("-")
        self.env_evaluators = {}
        self.tasks = []
        for env_name in self.env_names:
            evaluator = Evaluator(env_name, config, original_cwd=original_cwd, output_dir=self.output_dir)
            self.env_evaluators[env_name] = evaluator
            for task in evaluator.tasks:
                for episode_idx in range(evaluator.num_episodes):
                    json_filename = os.path.join(
                        self.output_dir,
                        env_name,
                        task,
                        f"{task}_run_{episode_idx:02d}.json",
                    )
                    if os.path.exists(json_filename):
                        logging.info(f"Skipping completed task: {env_name}, {task}, episode {episode_idx}")
                    else:
                        self.tasks.append((env_name, task, episode_idx))
        self.num_workers = config.eval.num_workers

    def run(self, agent_factory):
        if self.num_workers > 1:
            results = self._run_parallel(agent_factory)
        else:
            results = self._run_sequential(agent_factory)
        return results

    def _run_sequential(self, agent_factory):
        results = defaultdict(list)
        total_episodes = len(self.tasks)
        max_episode_retries = 3
        with tqdm(total=total_episodes, desc="Evaluating Episodes", position=0) as pbar:
            for env_name, task, episode_idx in self.tasks:
                evaluator = self.env_evaluators[env_name]
                agent = agent_factory.create_agent()
                for attempt in range(max_episode_retries):
                    try:
                        episode_log = evaluator.run_episode(task, agent, position=1, episode_idx=episode_idx)
                        results[env_name].append(episode_log)
                        break
                    except Exception as e:
                        tb = traceback.format_exc()
                        logging.error(
                            f"Episode attempt {attempt + 1}/{max_episode_retries} failed "
                            f"for {env_name}/{task}/ep{episode_idx}: {e}\n{tb}"
                        )
                        if attempt == max_episode_retries - 1:
                            logging.error(
                                f"All {max_episode_retries} attempts failed for "
                                f"{env_name}/{task}/ep{episode_idx}. Skipping."
                            )
                pbar.update(1)
        return results

    def _run_parallel(self, agent_factory):
        task_queue = multiprocessing.Queue()
        results_queue = multiprocessing.Queue()

        ctx = multiprocessing.get_context("fork")

        for item in self.tasks[: self.num_workers]:
            task_queue.put(item)

        pbar = tqdm(total=len(self.tasks), position=0, leave=True)

        positions = list(range(self.num_workers))

        processes = []
        for idx in range(self.num_workers):
            position = positions[idx]
            p = ctx.Process(
                target=self._worker,
                args=(task_queue, results_queue, agent_factory, position),
            )
            processes.append(p)
            p.start()

        results = defaultdict(list)
        tasks_completed = 0
        tasks_queued = self.num_workers

        total_tasks = len(self.tasks)

        while tasks_completed < total_tasks:
            result = results_queue.get()
            if "error" in result:
                logging.error(f"Error in task {result['task']} processed by {result['process_num']}: {result['error']}")
                logging.error(f"Traceback:\n{result['traceback']}")
            else:
                results[result["env_name"]].append(result)
            tasks_completed += 1

            pbar.update(1)
            pbar.set_description(f"Last task: {result['task']}, Process: {result.get('process_num', 'N/A')}")

            if tasks_queued < total_tasks:
                task_queue.put(self.tasks[tasks_queued])
                tasks_queued += 1

        for _ in range(self.num_workers):
            task_queue.put(None)

        for p in processes:
            p.join()

        pbar.close()

        return results

    def _worker(self, task_queue, results_queue, agent_factory, position):
        seed = get_unique_seed(process_num=position)
        random.seed(seed)
        np.random.seed(seed)

        agent = agent_factory.create_agent()
        process_num = multiprocessing.current_process().name
        while True:
            item = task_queue.get()
            if item is None:
                break
            env_name, task, episode_idx = item
            max_episode_retries = 3
            for attempt in range(max_episode_retries):
                try:
                    evaluator = self.env_evaluators[env_name]
                    result = evaluator.run_episode(
                        task,
                        agent,
                        process_num=process_num,
                        position=position + 1,
                        episode_idx=episode_idx,
                    )
                    result["process_num"] = process_num
                    result["env_name"] = env_name
                    results_queue.put(result)
                    break
                except Exception as e:
                    tb = traceback.format_exc()
                    logging.error(
                        f"Episode attempt {attempt + 1}/{max_episode_retries} failed "
                        f"for {env_name}/{task}/ep{episode_idx}: {e}\n{tb}"
                    )
                    if attempt == max_episode_retries - 1:
                        results_queue.put(
                            {
                                "env_name": env_name,
                                "task": task,
                                "error": str(e),
                                "traceback": tb,
                                "process_num": process_num,
                            }
                        )

class Evaluator:
    def __init__(self, env_name, config, original_cwd="", output_dir="."):
        self.env_name = env_name.strip()
        self.config = config
        self.output_dir = output_dir
        self.tasks = config.tasks[f"{self.env_name}_tasks"]

        self.num_episodes = config.eval.num_episodes[self.env_name]
        self.num_workers = config.eval.num_workers
        self.max_steps_per_episode = config.eval.max_steps_per_episode

        self.seeds = OmegaConf.select(config, "eval.seeds", default=None)
        if self.seeds is not None:
            self.seeds = list(self.seeds)
            
            self.num_episodes = min(self.num_episodes, len(self.seeds))

    def _get_episode_seed(self, episode_idx, process_num=None):
        if self.seeds is not None and episode_idx < len(self.seeds):
            return int(self.seeds[episode_idx])
        seed = self.config.envs.env_kwargs.seed
        if seed is not None:
            return int(seed)
        return get_unique_seed(process_num=process_num, episode_idx=episode_idx)

    def run_episode(self, task, agent, process_num=None, position=0, episode_idx=0):
        seed = self._get_episode_seed(episode_idx, process_num)

        
        
        with open_dict(self.config):
            self.config.envs.env_kwargs.seed = seed
            if self.env_name == "crafter":
                self.config.envs.crafter_kwargs.seed = seed

        env = make_env(self.env_name, task, self.config)
        agent.reset()

        is_memory_agent = hasattr(agent, "configure_memory")

        if is_memory_agent:
            agent.configure_memory(self.env_name, task, episode_idx)

        random.seed(seed)
        np.random.seed(seed)
        obs, info = env.reset(seed=seed)
        episode_log = {
            "task": task,
            "action_frequency": defaultdict(int),
            "input_tokens": 0,
            "output_tokens": 0,
        }

        instructions = None
        instruction_prompt = env.get_instruction_prompt(instructions=instructions)
        agent.prompt_builder.update_instruction_prompt(instruction_prompt)

        if is_memory_agent and hasattr(agent, '_get_instruction_prompt_override'):
            prompt_override = agent._get_instruction_prompt_override()
            if prompt_override is not None:
                instruction_prompt = prompt_override
                agent.prompt_builder.update_instruction_prompt(prompt_override)

        episode_return = 0.0

        max_steps_per_episode = env.max_steps if self.max_steps_per_episode is None else self.max_steps_per_episode

        csv_filename = os.path.join(self.output_dir, self.env_name, task, f"{task}_run_{episode_idx:02d}.csv")
        Path(csv_filename).parent.mkdir(exist_ok=True, parents=True)

        debug_filename = os.path.join(
            self.output_dir, self.env_name, task, f"{task}_run_{episode_idx:02d}_debug.jsonl"
        )

        images_dir = os.path.join(
            self.output_dir, self.env_name, task, f"episode_{episode_idx:02d}"
        )

        with open(csv_filename, mode="w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file, escapechar="˘", quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["Step", "Action", "Validated_Action", "Reasoning", "Observation", "Reward", "Done"])

            debug_file = open(debug_filename, "w", encoding="utf-8")

            try:

                pbar_desc = f"Task: {task}, Proc: {process_num}"
                pbar = tqdm(
                    total=max_steps_per_episode,
                    desc=pbar_desc,
                    position=position,
                    leave=False,
                    dynamic_ncols=True,
                )

                prev_raw_action = None
                prev_validated_action = None

                for step in range(max_steps_per_episode):
                    obs_long_before = obs.get("text", {}).get("long_term_context", "")
                    obs_short_before = obs.get("text", {}).get("short_term_context", "")
                    obs_image_before = obs.get("image")

                    if is_memory_agent:
                        response = agent.act(
                            obs,
                            prev_action=prev_raw_action,
                            prev_validated_action=prev_validated_action,
                        )
                    else:
                        response = agent.act(obs, prev_action=prev_raw_action)

                    validated_action = env.check_action_validity(response.completion)
                    reasoning = response.reasoning if hasattr(response, "reasoning") else ""

                    episode_log["action_frequency"][validated_action] += 1
                    episode_log["input_tokens"] += response.input_tokens
                    episode_log["output_tokens"] += response.output_tokens

                    obs, reward, terminated, truncated, info = env.step(validated_action)
                    done = terminated or truncated

                    episode_return += reward

                    obs["text"]["long_term_context"] = (
                        f"\n\nYour previous output did not contain a valid action. Defaulted to action: {validated_action}\n\nObservation:\n"
                        + obs["text"]["long_term_context"]
                        if (validated_action != response.completion) and (self.config.eval.feedback_on_invalid_action)
                        else obs["text"]["long_term_context"]
                    )

                    csv_writer.writerow(
                        [
                            step,
                            response.completion,
                            validated_action,
                            reasoning,
                            obs["text"]["long_term_context"],
                            reward,
                            done,
                        ]
                    )

                    step_debug_entry = {
                        "step": step,
                        "obs_long_term_context": obs_long_before,
                        "obs_short_term_context": obs_short_before,
                        "raw_action": response.completion,
                        "validated_action": validated_action,
                        "reward": reward,
                        "done": done,
                        "reasoning": reasoning if reasoning else None,
                        "instruction_prompt": instruction_prompt if step == 0 else None,
                        "input_tokens": response.input_tokens,
                        "output_tokens": response.output_tokens,
                    }

                    if is_memory_agent and hasattr(agent, "last_step_debug") and agent.last_step_debug:
                        step_debug_entry["memory_debug"] = agent.last_step_debug
                    else:
                        step_debug_entry["memory_debug"] = None

                    image_saved = False
                    if self.config.eval.save_images and obs_image_before is not None:
                        Path(images_dir).mkdir(exist_ok=True, parents=True)
                        image_filename = os.path.join(images_dir, f"step_{step:04d}.png")
                        obs_image_before.save(image_filename)
                        image_saved = True

                    step_debug_entry["image_saved"] = image_saved

                    debug_file.write(json.dumps(step_debug_entry, default=str) + "\n")

                    prev_raw_action = response.completion
                    prev_validated_action = validated_action

                    pbar.update(1)

                    if done:
                        logging.info(f"Episode done with reward: {episode_return}")
                        episode_log["done"] = True
                        if pbar.n < pbar.total:
                            pbar.update(pbar.total - pbar.n)
                        pbar.set_postfix_str("DONE")
                        break

                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)
                if "done" not in episode_log:
                    pbar.set_postfix_str("DONE")
                pbar.close()

            finally:
                debug_file.close()

            episode_log["episode_return"] = episode_return
            episode_log["num_steps"] = step + 1
            episode_log["failed_candidates"] = env.failed_candidates
            episode_log.update(env.get_stats())
            episode_log["process_num"] = process_num
            episode_log["seed"] = seed
            episode_log["episode_idx"] = episode_idx
            episode_log["agent"] = OmegaConf.to_container(self.config.agent, resolve=True)
            episode_log["client"] = OmegaConf.to_container(self.config.client, resolve=True)

            json_filename = os.path.join(
                self.output_dir,
                self.env_name,
                task,
                f"{task}_run_{episode_idx:02d}.json",
            )
            Path(json_filename).parent.mkdir(exist_ok=True, parents=True)
            with open(json_filename, "w") as f:
                json.dump(episode_log, f, indent=4)

        return episode_log
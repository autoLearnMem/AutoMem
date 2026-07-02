import re
import gymnasium as gym

_NLE_DISTANCE_REPLACEMENTS = [
    
    (re.compile(r'\bvery far\b'),  '9+ steps away'),
    (re.compile(r'\bvery near\b'), '2 steps away'),
    (re.compile(r'\bfar\b'),       '5-8 steps away'),
    (re.compile(r'\bnear\b'),      '3-4 steps away'),
    (re.compile(r'\badjacent\b'),  '1 step away (directly reachable)'),
]

_NLE_DIRECTION_EXPANSIONS = [

    (re.compile(r'\bnorthnortheast\b'), 'between north and northeast'),
    (re.compile(r'\beastnortheast\b'),  'between east and northeast'),
    (re.compile(r'\beastsoutheast\b'),  'between east and southeast'),
    (re.compile(r'\bsouthsoutheast\b'), 'between south and southeast'),
    (re.compile(r'\bsouthsouthwest\b'), 'between south and southwest'),
    (re.compile(r'\bwestsouthwest\b'),  'between west and southwest'),
    (re.compile(r'\bwestnorthwest\b'),  'between west and northwest'),
    (re.compile(r'\bnorthnorthwest\b'), 'between north and northwest'),
]

def _parse_nle_distances(text):
    for pattern, replacement in _NLE_DISTANCE_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    for pattern, replacement in _NLE_DIRECTION_EXPANSIONS:
        text = pattern.sub(replacement, text)
    return text

class EnvWrapper(gym.Wrapper):
    def __init__(self, env, env_name, task_name):
        super().__init__(env)
        self.env_name = env_name
        self.task_name = task_name
        self.failed_candidates = []

    @property
    def max_steps(self):
        return self.env.max_steps

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_observation(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        processed_obs = self._process_observation(obs)
        return processed_obs, reward, terminated, truncated, info

    def _process_observation(self, obs):
        if self.env_name in ["nle", "minihack"]:
            
            if "text" in obs:
                if "long_term_context" in obs["text"]:
                    obs["text"]["long_term_context"] = _parse_nle_distances(
                        obs["text"]["long_term_context"]
                    )
                if "short_term_context" in obs["text"]:
                    obs["text"]["short_term_context"] = _parse_nle_distances(
                        obs["text"]["short_term_context"]
                    )
        elif self.env_name == "babyai":
            obs = obs
        elif self.env_name == "textworld":
            obs = obs
        elif self.env_name == "babaisai":
            obs = obs
        elif self.env_name == "crafter":
            obs = obs
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")

        return obs

    @property
    def actions(self):
        return self.env.actions if hasattr(self.env, "actions") else list(range(len(self.env.action_space)))

    def get_text_action(self, action):
        return self.env.get_text_action(action)

    def get_instruction_prompt(self, instructions=None):
        if self.env_name == "nle":
            from environments.nle import get_instruction_prompt

            return get_instruction_prompt()
        elif self.env_name == "minihack":
            from environments.minihack import get_instruction_prompt

            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "babyai":
            from environments.babyai_text import get_instruction_prompt

            return get_instruction_prompt(self.env, mission=instructions)
        elif self.env_name == "textworld":
            from environments.textworld import get_instruction_prompt

            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "babaisai":
            from environments.babaisai import get_instruction_prompt

            return get_instruction_prompt(self.env, self.task_name)
        elif self.env_name == "crafter":
            from environments.crafter import get_instruction_prompt

            return get_instruction_prompt(self.task_name)
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")

    def check_action_validity(self, candidate_action):
        valid_action = None
        if candidate_action in self.env.language_action_space:
            valid_action = candidate_action
        else:
            valid_action = self.env.default_action
            self.failed_candidates.append(candidate_action)
        return valid_action

    def get_stats(self):
        return self.env.get_stats()
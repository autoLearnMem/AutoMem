import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime

import importlib.util as _ilu
import os as _os
_pp_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "postprocess.py")
_spec = _ilu.spec_from_file_location("postprocess", _pp_path)
_pp = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pp)
validate_conversation = _pp.validate_conversation
apply_postprocess = _pp.apply_postprocess

DEFAULT_MODEL = "opus"
DEFAULT_MAX_TURNS = 300
SESSION_TIMEOUT = 3600
DEFAULT_EPISODE_FILTER_TOP_PCT = 100.0
DEFAULT_MIN_TARGET_EXAMPLES = 300

ENV_CONTEXT = {
    "nle": """\
ENVIRONMENT: NetHack (NLE) A procedurally generated roguelike dungeon crawler. Episodes can last tens of thousands of steps.

OBSERVATION FORMAT (in debug JSONL files):
- obs_long_term_context: game messages, language description of nearby entities with distances, player coordinates, ASCII dungeon map
- obs_short_term_context: inventory listing

REWARD: In-game score delta per step. Sparse — most steps have reward=0.
PROGRESSION: Based on dungeon depth and experience level, mapped to [0%, 100%].""",

    "crafter": """\
ENVIRONMENT: Crafter A 2D survival and crafting game.

OBSERVATION FORMAT (in debug JSONL files):
- obs_long_term_context: terrain description and what the player faces
- obs_short_term_context: player status (health/food/drink/energy) and inventory

REWARD: +1 for each first-time achievement unlock (22 total).
PROGRESSION: Percentage of 22 achievements unlocked.""",

    "minihack": """\
ENVIRONMENT: MiniHack Focused tasks on the NetHack engine. Episodes are short (max ~100 steps).

OBSERVATION FORMAT (in debug JSONL files):
- obs_long_term_context and obs_short_term_context (same structure as NLE)

REWARD: Small negative per step. Positive for task completion.
PROGRESSION: Binary task completion (0% or 100%).""",
}

EXCLUDE_DIRS = {
    "__pycache__", ".git", ".vscode", ".idea", ".pytest_cache", ".mypy_cache",
    "venv", "env", "node_modules",
    "memory", "balrog_memory", "results", "logs", "logs_vllm_serve",
}
EXCLUDE_EXTS = {
    ".pyc", ".pyo", ".pyd",
    ".log", ".out", ".err",
    ".jsonl", ".json.gz",
    ".ttf", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".wav", ".pkl",
    ".lock", ".DS_Store",
}
EXCLUDE_FILES = {"eval.log", ".DS_Store", "SECRETS", ".env"}

def build_codebase_tree(codebase_dir, max_depth=6):
    lines = []
    codebase_dir = os.path.abspath(codebase_dir)
    base_name = os.path.basename(codebase_dir)
    for root, dirs, files in os.walk(codebase_dir):
        dirs[:] = [
            d for d in sorted(dirs)
            if d not in EXCLUDE_DIRS and not d.startswith(".")
        ]
        rel = os.path.relpath(root, codebase_dir)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth > max_depth:
            dirs.clear()
            continue
        indent = "  " * depth
        name = base_name if rel == "." else os.path.basename(root)
        lines.append("{}{}/".format(indent, name))
        file_indent = "  " * (depth + 1)
        for f in sorted(files):
            if any(f.endswith(ext) for ext in EXCLUDE_EXTS):
                continue
            if f in EXCLUDE_FILES:
                continue
            lines.append("{}{}".format(file_indent, f))
    return "\n".join(lines)

def build_codebase_context_section(codebase_dir, memory_dir):
    tree = build_codebase_tree(codebase_dir)

    memory_section = ""
    if memory_dir:
        memory_section = """\
### Memory Directory
`{memory_dir}` contains, per episode, the ACTUAL memory files the agent wrote / the scaffold auto-maintained during that episode. Subdirectory layout follows `{{env}}/{{task}}/ep_{{NN}}/*.txt`. This is the final on-disk state after the episode concluded.
""".format(memory_dir=memory_dir)

    return """\
## How to build your selection — read deeply, judge each item
The codebase at `{codebase_dir}` defines the agent you are producing training data for; your selection is only as good as your understanding of it. Build that understanding by reading the actual code and traces in full — not summaries, not samples, not grep-and-dump:
1. **Read the ENTIRE codebase in full** to wholistically understand the scaffold — what it does programmatically, what it leaves to the LLM, how the LLM fails within it, and which multi-step chains it composes reliably vs. unreliably. This includes the environment subtree that defines the action space (which action strings are valid per task, what happens on an invalid one). Do not restrict yourself to the prompts or an intuitive "core" subset.
2. **Walk whole episodes step by step** in the debug JSONLs to see how the base model actually uses memory — what it logs, what it consults, when it skips consultation, how it reacts to what it finds.
3. **Judge every candidate semantically, in context.** grep/classifiers may narrow the search, but each kept item gets one non-templated entry in `manual_review_notes.md` explaining why it is the training signal you want.

### Codebase File Tree
```
{tree}
```

{memory_section}""".format(
        codebase_dir=codebase_dir,
        tree=tree,
        memory_section=memory_section,
    )

SFT_SELECTION_PRINCIPLES = """\
## Selection Principles
In scope to select: the base model's own trace turns — LOG-phase turns, and the memory-consultation portion of PLAN turns — that you judge to be good training signal for a memory-ops specialist.
- **Derive your own criteria.** Decide for yourself what makes a turn good vs. bad training signal. Form your criteria from what you observe and document them in `selection_logic.md` with quoted accepted/rejected datapoints.
- **Verbatim only.** Every assistant message must be the exact text the base model produced in a trace — no editing, trimming, rewriting, or fabricating. (Fixed deterministic post-processing is applied afterward — see Deployment Context — but that is mechanical; YOU do not modify text.)
"""

DEPLOYMENT_CONTEXT = """\
## Deployment Context: Memory-Ops Specialist
The trained model is deployed inside a **two-model architecture**:
1. **Memory model** (the one you are producing training data for): handles all LOG-phase turns AND the memory-consultation portion of PLAN turns (SEARCH/READ/TAIL/COUNT/APPEND/WRITE/CREATE/UPSERT_MAP).
2. **Gameplay model** (an untrained copy of the same base): handles only the final action commitment (`<|ACTION|>...`).

What the memory model sees at inference is the same conversation form as the one-model traces. The handoff to the gameplay model happens AFTER the memory model's last generation in the PLAN phase. The memory model does not see the gameplay model's turns. So focus on memory-op quality for your selection, not the commited gameplay action.

### Post-processing pipeline (fixed, deterministic, applied to every item)
Your `selected_data.json` is transformed before training. The training target is the OUTPUT of this transformation:
1. Markdown code-block fences (` ``` `) wrapping op tags are unwrapped.
2. LOG items: last assistant message truncated at `<|DONE|>`.
3. PLAN items: `<|ACTION|>...<|END|>` is stripped from the last assistant message, because the memory model is never asked to commit actions; the gameplay model does that.
4. **Filter:** items whose last assistant message has no memory op AFTER step 3 are dropped, as these would train the model to produce post-result-validation prose with no op.
5. **Post-op tail trim:** anything after the LAST memory-op terminator (`<|END|>` of any op block, or `<|DONE|>`) in the last assistant message is removed, which removes hallucinated speculation and stray action commitments.
6. **Fence-survivor drop:** any item whose last assistant message still contains a markdown code-fence (` ``` `) after the steps above is dropped entirely.

So when you evaluate a candidate, mentally apply these steps to the last assistant message, and the result is the training target. For verbatim verification: the training target is verbatim-equal to the trace's `raw_response` MINUS the deterministic transformations above.
"""

REQUIRED_OUTPUTS_TEMPLATE = """\
## Required Output Files (write ONLY to `{output_rel}/`, and all files must be present)
1. **`codebase_understanding.md`** — How the scaffold works, in your words from reading the code and traces. Include the **prompt-shape inventory**: every distinct prompt shape the scaffold produces at inference (LOG and PLAN, with all code branches that inject different context). Include failure clusters observed in the traces with episode/step refs.
2. **`selection_logic.md`** — Your selection criteria, derived from the scaffold's prompts and base traces. Per pattern: the behavior shift expected, lift type, accept/reject criteria, one accepted and one rejected example quoted inline. Coverage check against the prompt-shape inventory. Concentration check.
3. **`selected_data.json`** — The selection itself. See "Training Data Format" below. Verbatim from traces.
4. **`review_decisions.jsonl`** — One line per reviewed step-phase: `{{"episode": "...", "step": N, "phase": "log"|"plan", "decision": "accept"|"reject", "reason": "..."}}`.
5. **`manual_review_notes.md`** — One prose entry per selected item, quoting prompt excerpt and response, explaining in **NON-TEMPLATED** language why this is the training signal you want. 
6. **`training_expectation.md`** — Why this dataset will improve deployed performance: predicted behavior shift, mechanism, risk analysis, etc.
7. **`contribution_map.md`** — Per-pattern: which items, base behavior, expected trained behavior, cross-pattern blend risks.
8. **`analysis.md`** — Narrative of dominant patterns, edge cases, data-collection artifacts.
9. **`selection_stats.json`** — Counts by phase, by episode, rejection reasons.
10. **`selection_revision_log.md`** — Every revision triggered by later-stage analysis: which file triggered it, what changed, why. If no revisions, explain why initial selection held.
"""

TRAINING_DATA_FORMAT = """\
## Training Data Format
Output `selected_data.json`: a JSON list of conversation objects.
```json
[
  {{
    "conversations": [
      {{"role": "user", "content": "...the prompt..."}},
      {{"role": "assistant", "content": "...the response..."}},
      {{"role": "user", "content": "...tool results (if multi-turn)..."}},
      {{"role": "assistant", "content": "...continuation response..."}}
    ]
  }}
]
```

**Format rules:**
- Each example is ONE phase (LOG or PLAN), not both combined.
- Roles alternate user/assistant starting with user.
- LOG examples: last assistant message ends at (or contains) `<|DONE|>`.
- PLAN examples: last assistant message ends with `<|ACTION|>...<|END|>` (this keeps the verbatim-from-trace rule satisfied; post-processing strips it).
- For both phases, the last assistant message should be verbatim through the terminator that's present in the trace.
- See Deployment Context's "Post-processing pipeline" for what gets stripped/trimmed before training.
"""

SCALING_REVIEW_TEMPLATE = """\
## On Scaling Review
There is a minimum of **{min_target_examples} datapoints**. If your selection falls below this, explain in `training_expectation.md` why more signal cannot be selected. Above the minimum there is no ceiling.
"""

_debug_logger = None

def setup_debug_logger(output_dir):
    global _debug_logger
    _debug_logger = logging.getLogger("data_engine_sft_debug")
    _debug_logger.setLevel(logging.DEBUG)
    _debug_logger.handlers.clear()
    handler = logging.FileHandler(
        os.path.join(output_dir, "data_engine_debug.log"), encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    _debug_logger.addHandler(handler)
    return _debug_logger

def debug_log(msg, level="info"):
    if _debug_logger is None:
        return
    getattr(_debug_logger, level, _debug_logger.info)(msg)

def scan_jsonl_files(results_dir, env_name):
    env_dir = os.path.join(results_dir, env_name)
    if not os.path.isdir(env_dir):
        env_dir = results_dir
    files = []
    for root, dirs, filenames in os.walk(env_dir):
        for fname in sorted(filenames):
            if fname.endswith("_debug.jsonl"):
                files.append(os.path.join(root, fname))
    return files

def scan_episode_json_files(results_dir, env_name):
    env_dir = os.path.join(results_dir, env_name)
    if not os.path.isdir(env_dir):
        env_dir = results_dir
    files = []
    for root, dirs, filenames in os.walk(env_dir):
        for fname in sorted(filenames):
            if (fname.endswith(".json")
                    and not fname.endswith("_summary.json")
                    and fname != "summary.json"):
                files.append(os.path.join(root, fname))
    return files

def extract_episode_metadata(jsonl_path):
    file_size_mb = os.path.getsize(jsonl_path) / (1024 * 1024)
    metadata = {
        "file": jsonl_path,
        "filename": os.path.basename(jsonl_path),
        "file_size_mb": round(file_size_mb, 2),
        "total_steps": 0,
        "total_reward": 0.0,
        "reward_steps": [],
        "done": False,
        "invalid_action_count": 0,
    }
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                step = data.get("step", 0)
                reward = data.get("reward", 0.0)
                raw = data.get("raw_action", "")
                validated = data.get("validated_action", "")
                done = data.get("done", False)

                metadata["total_steps"] = step + 1
                try:
                    metadata["total_reward"] += reward
                except TypeError:
                    pass

                if reward != 0 and reward != 0.0:
                    metadata["reward_steps"].append({
                        "step": step, "reward": reward,
                        "action": validated, "done": done,
                    })
                if raw != validated:
                    metadata["invalid_action_count"] += 1
                if done:
                    metadata["done"] = True
    except Exception as e:
        metadata["error"] = str(e)
        debug_log("Error extracting metadata from {}: {}".format(
            jsonl_path, e), "error")
    return metadata

def load_log_summary(log_file):
    if not log_file or not os.path.isfile(log_file):
        return "(no log file provided)"
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
        idx = content.rfind("Summary of Results:")
        if idx != -1:
            return content[idx:].strip()
        lines = content.strip().split("\n")
        return "\n".join(lines[-50:])
    except Exception as e:
        return "(error reading log: {})".format(e)

def build_episode_filter_steer(episode_filter_top_pct, n_episodes):
    if episode_filter_top_pct >= 100.0:
        return ""
    n_target = max(1, int(n_episodes * episode_filter_top_pct / 100.0))
    return (
        "\n\n### Episode Prioritization\n"
        "**Prioritize the top ~{pct:.0f}% of episodes** (~{n_target} out of {n_total}) when choosing where to dig in. Rank episodes by the per-episode `.json` summary metrics. Stronger episodes tend to carry more of the useful training signal; weaker episodes more often contain stuck loops, format failures, and other patterns you'd be training the model to AVOID rather than adopt. This is a soft steer, not a hard filter — if you spot an episode that contains a specifically interesting recovery or edge case, include it. But the default allocation of your review budget should be on the top ~{pct:.0f}%.\n".format(
            pct=episode_filter_top_pct,
            n_target=n_target,
            n_total=n_episodes,
        )
    )

def build_claude_md(env_name, codebase_dir, results_dir, memory_dir,
                    output_dir,
                    min_target_examples,
                    jsonl_files=None, json_summary_files=None):
    output_rel = os.path.join(output_dir, "output")

    jsonl_listing = ""
    if jsonl_files:
        jsonl_listing = (
            "### Debug JSONL Files (full step traces)\n"
            "**{n_jsonl}** files matching `{results_dir}/**/*_debug.jsonl` — one per episode. Each line is one game step (`step`, `reward`, `raw_action`, `validated_action`, and `memory_debug.{{log_phase,plan_phase}}.turns`).\n"
        ).format(n_jsonl=len(jsonl_files), results_dir=results_dir)

    json_listing = ""
    if json_summary_files:
        json_listing = (
            "### Episode Summary JSON Files (quick performance lookup)\n"
            "**{n_json}** files matching `{results_dir}/**/*.json` (excluding `summary.json` and `*_summary.json`) — one per episode, holding aggregate reward / progression / step counts. Read these FIRST when deciding where to focus.\n"
        ).format(n_json=len(json_summary_files), results_dir=results_dir)

    codebase_section = build_codebase_context_section(
        codebase_dir, memory_dir)

    required_outputs = REQUIRED_OUTPUTS_TEMPLATE.format(output_rel=output_rel)
    scaling_review = SCALING_REVIEW_TEMPLATE.format(
        min_target_examples=min_target_examples)

    return """\
# Data Engine: SFT Selection (Codebase-Grounded)
Your goal: produce training data `selected_data.json` — a verbatim selection of the base model's own memory-op turns (the SFT training data) — plus the supporting docs that document and justify it. The sections below are HOW.

{codebase_section}

{selection_principles}

{deployment_context}

## Process
The output files below have a natural dependency order but are NOT a strict waterfall — later-stage analysis may surface problems that require revising the selection. Cross-reference all output files against each other throughout, and revise the selection whenever a later file exposes an issue.

## Result Files Location
- CODEBASE: `{codebase_dir}`
- RESULTS: `{results_dir}`

{json_listing}

{jsonl_listing}

## Debug JSONL Format
Each line in a debug JSONL file is one game step. Key fields:
```
step_data["step"]                                   — step number
step_data["reward"]                                 — reward at this step
step_data["raw_action"]                             — what the model output
step_data["validated_action"]                       — what was actually executed
step_data["memory_debug"]["log_phase"]["turns"]     — list of LOG turns
step_data["memory_debug"]["plan_phase"]["turns"]    — list of PLAN turns
step_data["memory_debug"]["memory_snapshot"]        — memory file contents at this step
```
Each turn typically has `"prompt"` (user message), `"raw_response"` (assistant message), `"tool_results"`, and `"committed_action"` (PLAN only). Check memory_agent.py to confirm what is stored vs. reconstructed.

{training_data_format}

{required_outputs}

{scaling_review}
""".format(
        codebase_section=codebase_section,
        selection_principles=SFT_SELECTION_PRINCIPLES,
        deployment_context=DEPLOYMENT_CONTEXT,
        codebase_dir=codebase_dir,
        results_dir=results_dir,
        json_listing=json_listing,
        jsonl_listing=jsonl_listing,
        training_data_format=TRAINING_DATA_FORMAT,
        required_outputs=required_outputs,
        scaling_review=scaling_review,
    )

def build_traces_section(metadata_list, jsonl_files, json_summary_files,
                         results_dir, episode_filter_top_pct):
    n = len(metadata_list)
    if n == 0:
        return "## Episode Traces\n\nNo episodes found in `{}`.".format(
            results_dir)

    by_name = sorted(metadata_list, key=lambda m: m["filename"])
    first = by_name[0]["filename"]
    last = by_name[-1]["filename"]

    rewards = [m["total_reward"] for m in metadata_list]
    r_min = min(rewards)
    r_max = max(rewards)
    r_mean = sum(rewards) / len(rewards)
    n_pos = sum(1 for r in rewards if r > 0)
    total_steps = sum(m["total_steps"] for m in metadata_list)
    total_invalid = sum(m["invalid_action_count"] for m in metadata_list)
    n_json = len(json_summary_files) if json_summary_files else 0

    filter_steer = build_episode_filter_steer(episode_filter_top_pct, n)

    return (
        "## Episode Traces\n"
        "There are **{n}** episodes in `{results_dir}`.\n\n"
        "File naming range (alphabetic):\n"
        "  first: `{first}`\n"
        "  last:  `{last}`\n\n"
        "Enumerate:\n"
        "  - Debug JSONL (one line per step, full LOG/PLAN phase traces):\n"
        "      `{results_dir}/**/*_debug.jsonl`  ({n} files)\n"
        "  - Per-episode summary JSON (quick reward / progression lookup):\n"
        "      `{results_dir}/**/*.json` (excluding `summary.json` and `*_summary.json`)  ({n_json} files)\n\n"
        "Aggregate stats across all {n} episodes:\n"
        "  - total steps:              {total_steps}\n"
        "  - episodes with reward > 0: {n_pos}\n"
        "  - reward distribution:      min={r_min:.2f}, max={r_max:.2f}, "
        "mean={r_mean:.2f}\n"
        "  - total invalid actions:    {total_invalid}"
        "{filter_steer}"
    ).format(
        n=n, results_dir=results_dir, first=first, last=last,
        total_steps=total_steps, n_pos=n_pos,
        r_min=r_min, r_max=r_max, r_mean=r_mean,
        total_invalid=total_invalid, n_json=n_json,
        filter_steer=filter_steer,
    )

def build_prompt(env_name, codebase_dir, results_dir, memory_dir,
                 output_dir, summary_text, episode_metadata_list,
                 jsonl_files, episode_filter_top_pct,
                 min_target_examples):
    parts = []

    env_ctx = ENV_CONTEXT.get(env_name, "No specific context available.")
    parts.append("## Environment Context\n{}".format(env_ctx))

    if summary_text and summary_text != "(no log file provided)":
        parts.append(
            "## Training Data Collection: Evaluation Metrics\n{}".format(
                summary_text)
        )

    mem_line = ""
    if memory_dir:
        mem_line = "\n- Memory directory (per-episode memory files): `{}`".format(
            memory_dir)

    parts.append(
        "## Key Paths\n"
        "- Codebase (must read before reviewing traces): `{}`\n"
        "- Results dir: `{}`{}\n"
        "- Output dir (write here): `{}/output/`".format(
            codebase_dir, codebase_dir, results_dir, mem_line,
            output_dir,
        )
    )

    json_summary_files = scan_episode_json_files(results_dir, env_name)
    parts.append(build_traces_section(
        episode_metadata_list, jsonl_files, json_summary_files,
        results_dir, episode_filter_top_pct))

    parts.append(
        "## Your Task\n"
        "Produce the training data and all required output files described in CLAUDE.md.\n\n"
        "All outputs go to: `{output}/output/`\n\n"
        "Reminders (full context in CLAUDE.md):\n"
        "- The trained model is a memory-ops specialist in a two-model architecture. Focus on memory-op quality, not action choice.\n"
        "- Every assistant response is VERBATIM from the traces. Post-processing applies fixed transformations afterward.\n"
        "- Cover the scaffold's prompt-shape inventory; derive coverage from the code.\n"
        "- Derive your concrete accept/reject criteria from the scaffold's prompts and the actual base traces, not from generic rules.\n"
        "- Minimum {min_target} selected datapoints. No upper bound.".format(
            output=output_dir,
            min_target=min_target_examples,
        )
    )

    return "\n\n".join(parts)

def run_claude_code(
    prompt, workspace_dir, output_dir,
    model=DEFAULT_MODEL, max_turns=DEFAULT_MAX_TURNS,
    effort="max", timeout=SESSION_TIMEOUT,
):
    output_rel = os.path.join(output_dir, "output")
    os.makedirs(output_rel, exist_ok=True)

    allowed_tools = "Read,Write({o}/*),Write({o}/**),Bash(*),Grep,Glob".format(
        o=output_rel,
    )

    cmd = [
        "claude",
        "-p", prompt,
        "--model", model,
        "--max-turns", str(max_turns),
        "--permission-mode", "acceptEdits",
        "--allowedTools", allowed_tools,
        "--output-format", "stream-json",
        "--verbose",
    ]
    if effort:
        cmd.extend(["--effort", effort])

    transcript_path = os.path.join(output_dir, "claude_code_transcript.jsonl")

    debug_log("Running Claude Code:\n  workspace: {}\n  max_turns: {}".format(
        workspace_dir, max_turns))
    debug_log("Prompt length: {} chars".format(len(prompt)))

    print("  Running Claude Code (max {} turns, timeout {}s)...".format(
        max_turns, timeout))

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workspace_dir,
            stdin=subprocess.DEVNULL,
        )
        elapsed = time.time() - start_time

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        if result.stderr:
            with open(os.path.join(output_dir, "cc_stderr.txt"), "w") as f:
                f.write(result.stderr)

        session_id = None
        cost = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "result":
                    session_id = event.get("session_id")
                    cost = event.get("total_cost_usd")
                    with open(os.path.join(output_dir, "cc_result.json"),
                              "w") as f:
                        json.dump(event, f, indent=2)
            except json.JSONDecodeError:
                continue

        print("  Session: {}, Cost: ${}, Elapsed: {:.0f}s".format(
            session_id, cost, elapsed))

        sel_path = os.path.join(output_rel, "selected_data.json")
        success = os.path.isfile(sel_path)
        if success:
            try:
                with open(sel_path) as f:
                    data = json.load(f)
                print("  => {} examples selected".format(len(data)))
            except Exception:
                success = False

        return success, session_id, transcript_path, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print("  TIMEOUT after {:.0f}s".format(elapsed))
        return False, None, None, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print("  ERROR: {}".format(e))
        return False, None, None, elapsed

def validate_required_outputs(output_rel):
    required = [
        "codebase_understanding.md",
        "selection_logic.md",
        "manual_review_notes.md",
        "training_expectation.md",
        "contribution_map.md",
        "selected_data.json",
        "review_decisions.jsonl",
        "analysis.md",
        "selection_stats.json",
        "selection_revision_log.md",
    ]
    missing = [
        name for name in required
        if not os.path.isfile(os.path.join(output_rel, name))
    ]
    return (len(missing) == 0, missing)

def main():
    parser = argparse.ArgumentParser(
        description="SFT data engine (codebase-grounded, unified).",
    )
    parser.add_argument("--env", required=True,
                        help="Environment name: nle, minihack, crafter")
    parser.add_argument("--codebase-dir", required=True,
                        help="Codebase directory to read")
    parser.add_argument("--results-dir", required=True,
                        help="Results directory with debug JSONLs")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory")
    parser.add_argument("--memory-dir", default="",
                        help="Per-episode memory directory on disk (optional but recommended)")
    parser.add_argument("--log-file", default="",
                        help="Optional eval log file for summary")
    parser.add_argument("--episode-filter-top-pct", type=float,
                        default=DEFAULT_EPISODE_FILTER_TOP_PCT,
                        help="Prioritize the top X%% of episodes by reward when reviewing (100 = no steer)")
    parser.add_argument("--min-target-examples", type=int,
                        default=DEFAULT_MIN_TARGET_EXAMPLES,
                        help="Minimum reviewed dataset size below which training is expected to be under-signaled. No ceiling.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="max",
                        choices=["low", "medium", "high", "max"])
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--timeout", type=int, default=SESSION_TIMEOUT)

    args = parser.parse_args()
    args.codebase_dir = os.path.abspath(args.codebase_dir)
    args.results_dir = os.path.abspath(args.results_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.memory_dir:
        args.memory_dir = os.path.abspath(args.memory_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "output"), exist_ok=True)

    setup_debug_logger(args.output_dir)

    print("=" * 70)
    print("Data Engine — SFT (Codebase-Grounded, Unified)")
    print("=" * 70)
    print("Environment:           {}".format(args.env))
    print("Codebase:              {}".format(args.codebase_dir))
    print("Results:               {}".format(args.results_dir))
    print("Memory dir:            {}".format(
        args.memory_dir if args.memory_dir else "(none)"))
    print("Output:                {}".format(args.output_dir))
    print("Episode filter:        top {:.0f}%".format(
        args.episode_filter_top_pct))
    print("Min target examples:   {}".format(args.min_target_examples))
    print("CC model:              {}, effort={}".format(
        args.model, args.effort))
    print("Max turns / timeout:   {} / {}s".format(
        args.max_turns, args.timeout))
    print("=" * 70)

    jsonl_files = scan_jsonl_files(args.results_dir, args.env)
    if not jsonl_files:
        jsonl_files = scan_jsonl_files(args.results_dir, "")
    print("Found {} debug JSONL files".format(len(jsonl_files)))
    if not jsonl_files:
        print("ERROR: no debug JSONL files under {}".format(args.results_dir))
        sys.exit(1)

    json_summary_files = scan_episode_json_files(args.results_dir, args.env)
    if not json_summary_files:
        json_summary_files = scan_episode_json_files(args.results_dir, "")
    print("Found {} per-episode JSON summary files".format(
        len(json_summary_files)))

    print("Extracting episode metadata...")
    metas = [extract_episode_metadata(fp) for fp in jsonl_files]
    metas.sort(key=lambda m: m["total_reward"], reverse=True)

    summary_text = load_log_summary(args.log_file) if args.log_file else "(no log)"

    claude_md_path = os.path.join(args.output_dir, "CLAUDE.md")
    claude_md = build_claude_md(
        args.env, args.codebase_dir, args.results_dir, args.memory_dir,
        args.output_dir,
        args.min_target_examples,
        jsonl_files=jsonl_files, json_summary_files=json_summary_files,
    )
    claude_md = re.sub(r"\n{3,}", "\n\n", claude_md)
    with open(claude_md_path, "w", encoding="utf-8") as f:
        f.write(claude_md)
    print("Wrote CLAUDE.md ({} chars)".format(len(claude_md)))

    prompt = build_prompt(
        args.env, args.codebase_dir, args.results_dir, args.memory_dir,
        args.output_dir, summary_text, metas, jsonl_files,
        args.episode_filter_top_pct,
        args.min_target_examples,
    )
    prompt_path = os.path.join(args.output_dir, "initial_prompt.md")
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print("Wrote initial prompt ({} chars)".format(len(prompt)))

    config_snapshot = {
        "strategy": "sft",
        "env": args.env,
        "codebase_dir": args.codebase_dir,
        "results_dir": args.results_dir,
        "memory_dir": args.memory_dir,
        "output_dir": args.output_dir,
        "episode_filter_top_pct": args.episode_filter_top_pct,
        "min_target_examples": args.min_target_examples,
        "model": args.model,
        "effort": args.effort,
        "max_turns": args.max_turns,
        "num_episodes_found": len(jsonl_files),
        "num_json_summary_found": len(json_summary_files),
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "data_engine_config.json"),
              "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print("\n" + "─" * 70)
    print("Data-engine selection session")
    print("─" * 70)

    success, session_id, transcript_path, elapsed = run_claude_code(
        prompt, args.output_dir, args.output_dir,
        model=args.model, max_turns=args.max_turns,
        effort=args.effort, timeout=args.timeout,
    )

    output_rel = os.path.join(args.output_dir, "output")
    all_present, missing = validate_required_outputs(output_rel)
    if not all_present:
        print("\n❌ Missing required output files: {}".format(missing))
        print("   Training pipeline should NOT proceed.")
        sys.exit(1)
    print("\n✓ All required output files present.")

    sel_path = os.path.join(output_rel, "selected_data.json")
    try:
        with open(sel_path) as f:
            data = json.load(f)
    except Exception as e:
        print("\n❌ selected_data.json not valid JSON: {}".format(e))
        sys.exit(1)

    if not isinstance(data, list):
        print("\n❌ selected_data.json is not a list.")
        sys.exit(1)

    valid = [x for x in data if validate_conversation(x.get("conversations", []))]
    print("Format-validated: {}/{} examples".format(len(valid), len(data)))

    valid, pp_info = apply_postprocess(valid)
    pp_stats = pp_info["postprocess"]
    print("Post-processing: cleaned={}, action_only_filtered={}, action_trimmed={}, prose_only_tail_filtered={}, post_op_prose_trimmed={}, final={}".format(
              pp_stats["format_cleaned"],
              pp_stats["action_only_filtered"],
              pp_stats["action_trimmed"],
              pp_stats["prose_only_tail_filtered"],
              pp_stats["post_op_prose_trimmed"],
              pp_stats["output_count"]))

    with open(sel_path, "w") as f:
        json.dump(valid, f, indent=2)
    with open(os.path.join(os.path.dirname(sel_path), "postprocess_report.json"), "w") as f:
        json.dump(pp_info, f, indent=2)

    print("Post-processing: format_valid={} output={} violations={}".format(
        pp_info["format_valid_count"], pp_info["postprocess"]["output_count"],
        pp_info["violations"]))
    if pp_info["violations"] > 0:
        print("FATAL: {} validity violation(s) after post-processing.".format(
            pp_info["violations"]))
        sys.exit(1)
    if pp_info["postprocess"]["output_count"] == 0:
        print("FATAL: zero examples survived post-processing.")
        sys.exit(4)

    if len(valid) == 0:
        print("\n⚠️  WARNING: zero format-valid examples. "
              "Training should NOT proceed.")
        sys.exit(2)

    if len(valid) < args.min_target_examples:
        print("\n⚠️  ADVISORY: {} examples is below the "
              "min-target of {}.".format(
                  len(valid), args.min_target_examples))

    print("\n✅ Data engine done. {} examples in {}".format(
        len(valid), sel_path))
    sys.exit(0)

if __name__ == "__main__":
    main()
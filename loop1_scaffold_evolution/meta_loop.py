import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

DEFAULT_MODEL = "opus"
DEFAULT_MAX_TURNS = 300
SESSION_TIMEOUT = 3600

ENV_CONTEXT = {
    "nle": """\
ENVIRONMENT: NetHack (NLE)
A procedurally generated roguelike dungeon crawler. Episodes can last tens of thousands of steps.

OBSERVATION FORMAT (in debug JSONL files):
- obs_long_term_context: game messages, language description of nearby entities with distances, player coordinates, ASCII dungeon map
- obs_short_term_context: inventory listing

REWARD: In-game score delta per step. Sparse — most steps have reward=0.
PROGRESSION: Based on dungeon depth and experience level, mapped to [0%, 100%].""",

    "crafter": """\
ENVIRONMENT: Crafter
A 2D survival and crafting game.

OBSERVATION FORMAT (in debug JSONL files):
- obs_long_term_context: terrain description and what the player faces
- obs_short_term_context: player status (health/food/drink/energy) and inventory

REWARD: +1 for each first-time achievement unlock (22 total).
PROGRESSION: Percentage of 22 achievements unlocked.""",

    "minihack": """\
ENVIRONMENT: MiniHack
Focused tasks on the NetHack engine. Episodes are short (max ~100 steps).

OBSERVATION FORMAT (in debug JSONL files):
- obs_long_term_context and obs_short_term_context (same structure as NLE)

REWARD: Small negative per step. Positive for task completion.
PROGRESSION: Binary task completion (0% or 100%).""",
}

ENV_OBJECTIVES = {
    "nle": "Maximize progression percentage and average return.",
    "crafter": "Maximize progression percentage (achievement unlock rate).",
    "minihack": "Maximize task completion rate (progression percentage).",
}

_debug_logger = None

def setup_debug_logger(output_dir):
    global _debug_logger
    _debug_logger = logging.getLogger("meta_loop_debug")
    _debug_logger.setLevel(logging.DEBUG)
    _debug_logger.handlers.clear()
    handler = logging.FileHandler(
        os.path.join(output_dir, "meta_loop_debug.log"), encoding="utf-8"
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

def scan_memory_folders(codebase_dir, episode_json_files):
    memory_listings = {}
    seen_bases = set()
    for jf in episode_json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        mem_base = data.get("agent", {}).get("memory_dir", "")
        if not mem_base:
            continue
        full_base = os.path.join(codebase_dir, mem_base)
        if full_base in seen_bases or not os.path.isdir(full_base):
            continue
        seen_bases.add(full_base)
        for dirpath, dirnames, filenames in os.walk(full_base):
            if os.path.basename(dirpath).startswith("ep_"):
                files = sorted(f for f in filenames if not f.startswith("."))
                if files:
                    rel = os.path.relpath(dirpath, codebase_dir)
                    memory_listings[rel] = files
    return memory_listings

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
        "unchanged_obs_count": 0,
        "zero_reward_streaks": [],
    }
    try:
        current_streak = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                step = data.get("step", 0)
                reward = data.get("reward", 0.0)
                raw = data.get("raw_action", "")
                validated = data.get("validated_action", "")
                done = data.get("done", False)

                metadata["total_steps"] = step + 1
                metadata["total_reward"] += reward

                if reward != 0 and reward != 0.0:
                    metadata["reward_steps"].append({
                        "step": step, "reward": reward,
                        "action": validated, "done": done,
                    })
                    if current_streak > 0:
                        metadata["zero_reward_streaks"].append(current_streak)
                    current_streak = 0
                else:
                    current_streak += 1

                if raw != validated:
                    metadata["invalid_action_count"] += 1
                md = data.get("memory_debug")
                if md and md.get("diff_notice"):
                    metadata["unchanged_obs_count"] += 1
                if done:
                    metadata["done"] = True

        if current_streak > 0:
            metadata["zero_reward_streaks"].append(current_streak)
        streaks = metadata["zero_reward_streaks"]
        if streaks:
            metadata["max_zero_streak"] = max(streaks)
            metadata["avg_zero_streak"] = round(sum(streaks) / len(streaks), 1)
        else:
            metadata["max_zero_streak"] = 0
            metadata["avg_zero_streak"] = 0
    except Exception as e:
        metadata["error"] = str(e)
        debug_log("Error extracting metadata: {}".format(e), "error")
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

def build_claude_md(env_name, codebase_dir, results_dir, output_dir):
    objective = ENV_OBJECTIVES.get(env_name, "Maximize overall performance.")

    return """\
# Meta-Loop Analysis & Revision Task

You are a senior AI researcher improving a memory-augmented LLM agent system.
The agent uses a "memory-as-filesystem" approach where the LLM can read, write, and create files to maintain long-horizon context across thousands of game steps.
You are evaluating and improving this system on the {env} environment.

## Performance Objective
{objective}

## The Memory-as-Filesystem Idea
The agent has file tools (read, write, search, append, create) that operate on a persistent virtual filesystem across steps. These files give the agent long-term memory that survives the sliding context window.

The CURRENT implementation uses a two-phase loop each step (LOG then PLAN).
In the LOG phase, the agent records what happened into memory files.
In the PLAN phase, the agent consults memory before choosing an action.

## CORE METHOD CONSTRAINT — LLM-DRIVEN MEMORY OPERATIONS MUST BE PRESERVED
The fundamental novelty is that the inner-loop LLM has memory file operations (read, write, search, append, create) as part of its action space. The LLM DECIDES what to store, what to retrieve, and how to organize its memory files.

You MUST preserve:
1. The inner-loop LLM must have explicit memory tool calls available during gameplay
2. The LLM must actively decide what to store and retrieve — not fully automated
3. Memory files must persist across steps and be accessible via the LLM's tool calls

You MUST NOT:
- Replace LLM-driven memory with a purely programmatic history buffer
- Remove memory file operations from the LLM's action space
- Make all storage/retrieval decisions automatic (no LLM involvement)
- Reduce the system to a sliding context window with automatic summarization

The goal is to improve HOW the LLM uses its memory, not eliminate LLM involvement.

## CRITICAL ARCHITECTURAL CONSTRAINT — TWO-PHASE LOGGING
The environment's reaction to action at step t is NOT available until step t+1.
Therefore:
- The LOG phase at step t records the OUTCOME of step t-1's action (now observable)
- The PLAN phase at step t chooses the action for step t

You MAY restructure the phases, but MUST preserve this invariant: nothing is written to memory about an action's outcome until the environment's actual response has been observed at the NEXT step.

## File Access Rules
- CODEBASE: `{codebase_dir}` — READ ONLY. Do not modify any files here.
- RESULTS: `{results_dir}` — READ ONLY. Do not modify any files here.
- OUTPUT: `{output_dir}/output/` — Write revised code and analysis here.
  - Write revised code files to: `{output_dir}/output/revised_code/`
  - Write analysis summary to: `{output_dir}/output/analysis.md`
  - Write method description to: `{output_dir}/output/method_description.md`

CRITICAL — Revised code files MUST preserve the codebase directory structure.
For example, if you revise `agents/memory_agent.py`, write it to:
  `{output_dir}/output/revised_code/agents/memory_agent.py`
NOT to:
  `{output_dir}/output/revised_code/memory_agent.py`
The revised files will be copied directly into the codebase, so paths must match.

## Your Task — Two Phases

### Phase 1: ANALYSIS
Investigate the evaluation results progressively. Start with a broad overview (e.g., episode summaries, metrics, memory operation stats, etc.), then drill into specific episodes, steps, or memory operations based on what you find.

You may run multiple analysis scripts per step, but do not try to cover all questions in a single step — leave room to adjust based on what you find.

Key questions to investigate:
1. MEMORY UTILIZATION: How does the agent interact with its memory files?
2. MEMORY ACCURACY: Are memory contents accurate or hallucinated?
3. MEMORY IMPACT: Does memory actually influence action decisions?
4. MEMORY GAPS: Where did lack of memory cause mistakes?
5. STRUCTURAL ISSUES: What is wrong with the memory SYSTEM itself?

### Phase 2: REVISION
After analysis, write your revised code to `{output_dir}/output/revised_code/`.
Write COMPLETE file contents, not patches. Only write files you changed.
Preserve the codebase directory structure (e.g., `agents/memory_agent.py`, `config/config.yaml`, NOT bare filenames in the root).
Also write `{output_dir}/output/analysis.md` with your analysis findings and `{output_dir}/output/method_description.md` describing what changed and why.

Your revision MUST include at least one STRUCTURAL change to the memory system.
Prompt wording changes alone do NOT count. Do NOT optimize for token efficiency.

## Important
- Do NOT modify any files in the codebase or results directories
- Do NOT run eval.py or execute the inner-loop system
- All output goes to `{output_dir}/output/` only
""".format(
        env=env_name,
        objective=objective,
        codebase_dir=codebase_dir,
        results_dir=results_dir,
        output_dir=output_dir,
    )

def build_prompt(
    env_name, codebase_dir, results_dir, output_dir,
    summary_text, prev_summary_text, method_description,
    episode_metadata_list,
    jsonl_files, summary_json_files, memory_listings,
):
    parts = []

    env_ctx = ENV_CONTEXT.get(env_name, "No specific context available.")
    parts.append("## Environment Context\n{}".format(env_ctx))

    parts.append(
        "## Evaluation Note\n"
        "Fixed seeds are used across all iterations for fair comparison. "
        "The seed list is defined in config.yaml (eval.seeds). Each episode_idx "
        "always uses the same seed, so you can compare per-episode results "
        "between the current and previous iteration."
    )

    if method_description:
        parts.append(
            "## Previous Iteration: Method Description\n"
            "(What was changed in the last iteration and why.)\n\n"
            "{}".format(method_description)
        )
    if prev_summary_text and prev_summary_text != "(no log file provided)":
        parts.append(
            "## Previous Iteration: Evaluation Metrics\n"
            "{}".format(prev_summary_text)
        )

    parts.append("## Current Evaluation Metrics\n{}".format(summary_text))

    parts.append("## Key File Paths\n"
        "- Codebase: `{}`\n"
        "- Results: `{}`\n"
        "- Output (write here): `{}/output/`\n"
        "- Main agent code: `{}/agents/memory_agent.py`\n"
        "- Config: `{}/config/config.yaml`".format(
            codebase_dir, results_dir, output_dir,
            codebase_dir, codebase_dir,
        )
    )

    parts.append("## Episode Results ({} episodes)".format(len(episode_metadata_list)))
    for i, meta in enumerate(episode_metadata_list):
        reward_summary = ""
        if meta["reward_steps"]:
            events = ", ".join(
                "step {}(r={})".format(r["step"], r["reward"])
                for r in meta["reward_steps"][:20]
            )
            reward_summary = ", rewards: [{}]".format(events)

        streak_info = ""
        if meta.get("max_zero_streak", 0) > 0:
            streak_info = ", max_zero_streak={}".format(meta["max_zero_streak"])

        parts.append(
            "  {i}. `{fname}` — {steps} steps, {size}MB, reward={reward:.2f}, done={done}, invalid={invalid}, unchanged={unchanged}{streaks}{rewards}".format(
                i=i + 1,
                fname=meta["filename"],
                steps=meta["total_steps"],
                size=meta["file_size_mb"],
                reward=meta["total_reward"],
                done=meta["done"],
                invalid=meta["invalid_action_count"],
                unchanged=meta["unchanged_obs_count"],
                streaks=streak_info,
                rewards=reward_summary,
            )
        )

    parts.append("\n## Debug JSONL Files (for analysis)")
    for fp in jsonl_files:
        parts.append("  - `{}`".format(fp))

    if summary_json_files:
        parts.append("\n## Per-Episode Summary Files")
        for fp in summary_json_files:
            parts.append("  - `{}`".format(fp))

    if memory_listings:
        parts.append("\n## Memory File Listings")
        for rel_dir, files in sorted(memory_listings.items()):
            parts.append("  - `{}/`: {}".format(rel_dir, ", ".join(files)))

    parts.append(
        "\n## Your Task\n"
        "1. Analyze the evaluation results for the {env} environment.\n"
        "   Start with a broad overview, then drill deeper.\n"
        "2. After analysis, write revised code to `{output}/output/revised_code/`.\n"
        "3. Write analysis summary to `{output}/output/analysis.md`.\n"
        "4. Write method description to `{output}/output/method_description.md`.\n\n"
        "Start by reading the codebase (especially `agents/memory_agent.py`) "
        "and examining the episode results.".format(
            env=env_name, output=output_dir,
        )
    )

    return "\n\n".join(parts)

def build_continuation_prompt(
    env_name, output_dir, failed_summary, original_summary,
    failed_metadata_list, failed_jsonl_files,
):
    parts = []

    parts.append(
        "## Revision Feedback — Changes Did NOT Improve Performance\n\n"
        "Your previous revised code was evaluated but did NOT improve over baseline.\n\n"
        "BASELINE METRICS:\n{}\n\n"
        "YOUR REVISION'S METRICS:\n{}".format(original_summary, failed_summary)
    )

    parts.append("\n## Failed Revision: Episode Results")
    for i, meta in enumerate(failed_metadata_list):
        reward_summary = ""
        if meta["reward_steps"]:
            events = ", ".join(
                "step {}(r={})".format(r["step"], r["reward"])
                for r in meta["reward_steps"][:15]
            )
            reward_summary = ", rewards: [{}]".format(events)
        parts.append(
            "  {i}. `{fname}` — {steps} steps, reward={reward:.2f}, invalid={invalid}{rewards}".format(
                i=i + 1, fname=meta["filename"],
                steps=meta["total_steps"],
                reward=meta["total_reward"],
                invalid=meta["invalid_action_count"],
                rewards=reward_summary,
            )
        )

    parts.append("\n## Failed Revision Debug Files")
    for fp in failed_jsonl_files:
        parts.append("  - `{}`".format(fp))

    parts.append(
        "\n## Your Task\n"
        "Analyze what went wrong with your revision. Then produce a better "
        "revision.\n"
        "Write revised code to `{}/output/revised_code/`.\n"
        "Write updated analysis to `{}/output/analysis.md`.\n"
        "Write method description to `{}/output/method_description.md`.".format(
            output_dir, output_dir, output_dir,
        )
    )

    return "\n\n".join(parts)

def run_claude_code(
    prompt, workspace_dir, output_dir,
    model=DEFAULT_MODEL, max_turns=DEFAULT_MAX_TURNS,
    effort="max", timeout=SESSION_TIMEOUT,
):
    revised_code_dir = os.path.join(output_dir, "output", "revised_code")
    os.makedirs(revised_code_dir, exist_ok=True)

    output_rel = os.path.join(output_dir, "output")
    allowed_tools = "Read,Write({output}/*),Write({output}/**),Bash(*),Grep,Glob".format(
        output=output_rel,
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

    debug_log("Running Claude Code:\n  cmd: {}\n  workspace: {}\n  max_turns: {}".format(
        " ".join(cmd[:6]) + " ...", workspace_dir, max_turns
    ))
    debug_log("Prompt length: {} chars".format(len(prompt)))

    print("  Running Claude Code (max {} turns, timeout {}s)...".format(max_turns, timeout))

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

        debug_log("Claude Code finished in {:.1f}s, return code {}".format(
            elapsed, result.returncode
        ))

        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        if result.stderr:
            stderr_path = os.path.join(output_dir, "claude_code_stderr.txt")
            with open(stderr_path, "w", encoding="utf-8") as f:
                f.write(result.stderr)
            debug_log("Stderr ({} chars) saved to {}".format(
                len(result.stderr), stderr_path
            ))

        session_id = None
        cost_usd = None
        num_turns = None
        result_event = None
        n_events = 0

        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                n_events += 1
                if event.get("type") == "result":
                    result_event = event
                    session_id = event.get("session_id")
                    cost_usd = event.get("total_cost_usd")
                    num_turns = event.get("num_turns")
            except json.JSONDecodeError:
                continue

        debug_log("Parsed {} events from stream, result_event={}".format(
            n_events, result_event is not None
        ))

        if result_event:
            is_error = result_event.get("is_error", False)
            debug_log("Session: {}, Cost: ${}, Turns: {}, Error: {}".format(
                session_id, cost_usd, num_turns, is_error
            ))
            print("  Session: {}, Cost: ${}, Turns: {}, Events: {}, Elapsed: {:.0f}s".format(
                session_id, cost_usd, num_turns, n_events, elapsed
            ))
            if is_error:
                debug_log("Claude Code reported error: {}".format(
                    result_event.get("result", "")[:500]
                ), "error")
        else:
            debug_log("No result event found in {} events".format(n_events), "warning")
            print("  (No result event found, {} events captured, elapsed {:.0f}s)".format(
                n_events, elapsed
            ))

        if result_event:
            result_path = os.path.join(output_dir, "claude_code_result.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result_event, f, indent=2, ensure_ascii=False)

        revised_files = []
        for root, dirs, filenames in os.walk(revised_code_dir):
            for fname in filenames:
                rel = os.path.relpath(os.path.join(root, fname), revised_code_dir)
                revised_files.append(rel)

        success = len(revised_files) > 0
        if success:
            print("  => {} revised file(s):".format(len(revised_files)))
            for f in revised_files:
                print("     - {}".format(f))
        else:
            print("  => No revised files produced")
            debug_log("No revised files found in {}".format(revised_code_dir), "warning")

        return success, session_id, transcript_path, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        debug_log("Claude Code TIMEOUT after {:.0f}s".format(elapsed), "error")
        print("  TIMEOUT after {:.0f}s".format(elapsed))
        return False, None, None, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        debug_log("Claude Code ERROR: {}".format(e), "error")
        print("  ERROR: {}".format(e))
        return False, None, None, elapsed

def main():
    parser = argparse.ArgumentParser(
        description="Meta-Loop (Claude Code): Automated review and improvement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env", required=True, help="Environment to optimize for")
    parser.add_argument("--codebase-dir", required=True, help="Path to the codebase directory")
    parser.add_argument("--results-dir", required=True, help="Path to evaluation results")
    parser.add_argument("--log-file", default=None, help="Path to evaluation log file")
    parser.add_argument("--prev-log-file", default=None, help="Path to previous iteration's log")
    parser.add_argument("--output-dir", required=True, help="Path to save outputs")
    parser.add_argument("--method-description", default=None, help="Path to previous method description")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude Code model (default: opus)")
    parser.add_argument("--effort", default="max", choices=["low", "medium", "high", "max"], help="Thinking effort level (default: max)")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Max Claude Code turns")
    parser.add_argument("--timeout", type=int, default=SESSION_TIMEOUT, help="Session timeout in seconds")
    parser.add_argument("--continue-results-dir", default=None, help="Results dir from failed revision")
    parser.add_argument("--continue-log-file", default=None, help="Log file from failed revision")

    args = parser.parse_args()

    args.codebase_dir = os.path.abspath(args.codebase_dir)
    args.results_dir = os.path.abspath(args.results_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    if args.log_file:
        args.log_file = os.path.abspath(args.log_file)
    if args.prev_log_file:
        args.prev_log_file = os.path.abspath(args.prev_log_file)
    if args.method_description and os.path.isfile(args.method_description):
        args.method_description = os.path.abspath(args.method_description)
    if args.continue_results_dir:
        args.continue_results_dir = os.path.abspath(args.continue_results_dir)
    if args.continue_log_file:
        args.continue_log_file = os.path.abspath(args.continue_log_file)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "output", "revised_code"), exist_ok=True)

    setup_debug_logger(args.output_dir)

    is_continuation = args.continue_results_dir is not None

    print("=" * 70)
    print("Meta-Loop (Claude Code)" + (" [CONTINUATION]" if is_continuation else ""))
    print("=" * 70)
    print("Environment:  {}".format(args.env))
    print("Codebase:     {}".format(args.codebase_dir))
    print("Results:      {}".format(args.results_dir))
    print("Output:       {}".format(args.output_dir))
    print("Model:        {}".format(args.model))
    print("Effort:       {}".format(args.effort))
    print("Max turns:    {}".format(args.max_turns))
    if is_continuation:
        print("Failed results: {}".format(args.continue_results_dir))
    print("=" * 70)

    claude_md_content = build_claude_md(
        args.env, args.codebase_dir, args.results_dir, args.output_dir,
    )
    claude_md_path = os.path.join(args.output_dir, "CLAUDE.md")
    with open(claude_md_path, "w", encoding="utf-8") as f:
        f.write(claude_md_content)
    debug_log("Wrote CLAUDE.md ({} chars)".format(len(claude_md_content)))

    print("\nScanning result files...")
    jsonl_files = scan_jsonl_files(args.results_dir, args.env)
    if not jsonl_files:
        jsonl_files = scan_jsonl_files(args.results_dir, "")
    print("  Found {} JSONL files".format(len(jsonl_files)))

    summary_json_files = scan_episode_json_files(args.results_dir, args.env)
    if not summary_json_files:
        summary_json_files = scan_episode_json_files(args.results_dir, "")
    print("  Found {} episode summary files".format(len(summary_json_files)))

    memory_listings = scan_memory_folders(args.codebase_dir, summary_json_files)
    print("  Found {} memory episode folders".format(len(memory_listings)))

    print("Extracting episode metadata...")
    episode_metadata = [extract_episode_metadata(fp) for fp in jsonl_files]

    if is_continuation:
        failed_results_dir = args.continue_results_dir or args.results_dir
        failed_jsonl = scan_jsonl_files(failed_results_dir, args.env)
        if not failed_jsonl:
            failed_jsonl = scan_jsonl_files(failed_results_dir, "")
        failed_metadata = [extract_episode_metadata(fp) for fp in failed_jsonl]
        failed_summary = load_log_summary(args.continue_log_file or args.log_file)
        original_summary = load_log_summary(args.log_file)

        prompt = build_continuation_prompt(
            args.env, args.output_dir,
            failed_summary, original_summary,
            failed_metadata, failed_jsonl,
        )
    else:
        summary_text = load_log_summary(args.log_file)
        prev_summary_text = None
        if args.prev_log_file:
            prev_summary_text = load_log_summary(args.prev_log_file)

        method_desc = None
        if args.method_description:
            if os.path.isfile(args.method_description):
                with open(args.method_description, "r", encoding="utf-8") as f:
                    method_desc = f.read()
            else:
                method_desc = args.method_description

        prompt = build_prompt(
            args.env, args.codebase_dir, args.results_dir, args.output_dir,
            summary_text, prev_summary_text, method_desc,
            episode_metadata, jsonl_files, summary_json_files, memory_listings,
        )

    prompt_path = os.path.join(args.output_dir, "prompt.md")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(prompt)
    print("  Prompt: {} chars, saved to {}".format(len(prompt), prompt_path))

    config_path = os.path.join(args.output_dir, "meta_loop_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "env": args.env,
            "codebase_dir": args.codebase_dir,
            "results_dir": args.results_dir,
            "log_file": args.log_file,
            "model": args.model,
            "max_turns": args.max_turns,
            "effort": args.effort,
            "is_continuation": is_continuation,
            "continue_results_dir": args.continue_results_dir,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    for retry in range(3):
        print("\n" + "=" * 70)
        print("Starting Claude Code session...")
        print("=" * 70)
        success, session_id, transcript_path, elapsed = run_claude_code(
            prompt=prompt,
            workspace_dir=args.output_dir,
            output_dir=args.output_dir,
            model=args.model,
            max_turns=args.max_turns,
            effort=args.effort,
            timeout=args.timeout,
        )
        if success:
            break
        else:
            continue

    print("\n" + "=" * 70)
    revised_code_dir = os.path.join(args.output_dir, "output", "revised_code")
    analysis_path = os.path.join(args.output_dir, "output", "analysis.md")
    description_path = os.path.join(args.output_dir, "output", "method_description.md")

    revised_files = []
    if os.path.isdir(revised_code_dir):
        for root, dirs, filenames in os.walk(revised_code_dir):
            for fname in filenames:
                revised_files.append(os.path.relpath(
                    os.path.join(root, fname), revised_code_dir
                ))

    if revised_files:
        print("✅ Meta-loop complete. {} file(s) revised in {:.0f}s.".format(
            len(revised_files), elapsed
        ))
        for f in revised_files:
            print("   - {}".format(f))
    else:
        print("⚠️  Meta-loop complete but no revised files produced.")

    if os.path.isfile(analysis_path):
        print("  Analysis:    {}".format(analysis_path))
    if os.path.isfile(description_path):
        print("  Description: {}".format(description_path))
    if transcript_path and os.path.isfile(transcript_path):
        print("  Transcript:  {}".format(transcript_path))
    if session_id:
        print("  Session ID:  {}".format(session_id))

    print("  Debug log:   {}".format(os.path.join(args.output_dir, "meta_loop_debug.log")))

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
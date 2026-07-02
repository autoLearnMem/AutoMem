import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
import importlib.util as _ilu

_HELPER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.py")
_spec = _ilu.spec_from_file_location("_ledger_helper", _HELPER_PATH)
_LH = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_LH)
format_ledger_md = _LH.format_ledger_md_decider   
ledger_row_desc = _LH.ledger_row_desc             

DEFAULT_MODEL = "opus"
DEFAULT_MAX_TURNS = 300
SESSION_TIMEOUT = 3600
DEFAULT_MAX_CONFIGS = 2

CONFIG_FIELDS = [
    "RANK", "ALPHA", "DROPOUT", "LR", "EPOCHS", "BATCH", "GRAD_ACCUM",
    "CUTOFF", "LR_SCHED", "WARMUP", "SAVE", "NUM_GPUS", "TARGET",
]
_VALID_TARGET_ALIASES = {"ALL", "ATTN", "QV", "MLP"}

ENV_CONTEXT = {
    "nle": """\
ENVIRONMENT: NetHack (NLE). Long roguelike; episodes can run tens of thousands of steps. PROGRESSION = dungeon depth + experience, mapped to [0,100]%.""",
    "crafter": """\
ENVIRONMENT: Crafter. 2D survival/crafting, 22 achievements. PROGRESSION = % of 22 achievements unlocked.""",
    "minihack": """\
ENVIRONMENT: MiniHack. Short focused NetHack tasks. PROGRESSION = task completion (0/100%).""",
}

_debug_logger = None

def setup_debug_logger(output_dir):
    global _debug_logger
    _debug_logger = logging.getLogger("train_engine_cc_debug")
    _debug_logger.setLevel(logging.DEBUG)
    _debug_logger.handlers.clear()
    handler = logging.FileHandler(
        os.path.join(output_dir, "train_engine_debug.log"), encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    _debug_logger.addHandler(handler)
    return _debug_logger

def debug_log(msg, level="info"):
    if _debug_logger is None:
        return
    getattr(_debug_logger, level, _debug_logger.info)(msg)

def load_ledger(ledger_path):
    rows = []
    if not ledger_path or not os.path.isfile(ledger_path):
        return rows
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return rows

def build_context_section(env_name, has_prior_runs):
    if not has_prior_runs:
        return ("## No prior runs yet\n"
                "This is the first iteration: the ledger is empty, so there are no prior-run artifacts to read. Pick a sensible starting recipe (any training-config recommendation in your prompt is a PRIOR you may use or ignore).")
    return (
        "## Where each prior run's artifacts live\n"
        "The ledger has one row per training run. To reason about a row, read its artifacts: each row's `artifacts_dir` is that run's per-config archive, and `<iter_dir>` (its parent) holds that iteration's data-engine selection and recipe alongside the per-config archives.\n"
        + ledger_row_desc(env_name) + "\n"
        "  <artifacts_dir>/\n"
        "    train_lora.yaml, merge_lora.yaml      # the LoRA + merge hyperparameters used (effective batch = batch x grad_accum x num_gpus)\n"
        "    training.log                          # the training loss curve\n"
        "    selected_data.json                    # the post-processed selected data actually trained on\n"
        "    results/<run>/                        # eval outputs: <env>/<task>/*_run_*.{json,_debug.jsonl,csv} (per-episode result + full per-step trace), plus <env>/<env>_summary.json and a top-level summary.json\n"
        "    memory/<run>/                         # the final memory files the trained agent maintained during eval (<env>/<task>/ep_NN/*.txt)\n"
        "  <iter_dir>/data_engine_shared/output/   # the data engine's selection + its writeup\n"
        "  <iter_dir>/recipe/                      # that iteration's training recipe + rationale, and any revised pipeline code for that iteration\n\n"
        "Any training-config recommendation included in your prompt is a PRIOR you may use or ignore.")

def build_claude_md(env_name, baseline_metric, max_configs,
                    data_engine_code_path, postprocess_path, has_prior_runs):
    return """\
# Training Engine: Iterative SFT-LoRA Optimization ({env})
You are optimizing the SFT-LoRA training process for the **{env}** environment. Each iteration you review the actual results of past training runs and decide what to change next to push eval **progression** up. The scaffold-only (no training) baseline for this env is **{baseline}**.

## The pipeline you are steering (one inner run = one training config)
1. Base-model traces are collected ONCE (shared across all iterations).
2. **Data engine** (an inner Claude Code session) reads the traces and selects SFT training examples -> `selected_data.json`. Its selection logic lives in `{de_code}` — you MAY edit it (see latitude below).
3. The deterministic **postprocess** step (`{postprocess}`) turns `selected_data.json` into the final training target (6 fixed transforms + 4 validity checks). This is applied no matter what — you cannot and need not defeat it.
4. LoRA SFT trains a **memory model** on the target.
5. Two-model eval: the trained **memory model** handles LOG + PLAN consultation; **untrained base gameplay model** commits the action. Progression is measured here over fixed eval seeds.

## HARD CONSTRAINTS (never change these)
- SFT-LoRA stays the training method.
- The data-selection paradigm stays (we always train on selected example data).
- The two-model split stays (trained memory model + untrained base gameplay model).
- The postprocess step (`{postprocess}`) is applied externally to whatever data is selected. Do not try to remove/bypass it.
- The eval of each run is fixed over a seed set — you cannot change the episodes or seeds. Do not try to fix any small-sample noise.

## What the postprocess step does (and why) — be data-engine aware
The postprocess step (`{postprocess}`) runs on the data engine's output regardless of how that output was produced; rewriting the data-engine logic does NOT change it. It applies six fixed transforms to the LAST assistant message of each selected item, then 4 validity checks (and aborts the iteration if any check fails):
1. Strip markdown code-fences around op tags.
2. LOG items: truncate at `<|DONE|>`.
3. PLAN items: strip the trailing `<|ACTION|>...<|END|>`. THIS IS THE MECHANICAL ENFORCEMENT OF THE TWO-MODEL SPLIT — the trained memory model must never emit a game action (the untrained base gameplay model does that), so the action is removed from every PLAN training target.
4. Drop items whose last turn has NO memory op after the action strip (a prose-only tail would train hallucinated narration).
5. Trim anything after the last memory-op terminator (hallucinated tool-results, conditional plans, stray action commitments). So the training target is always a memory-op chain WITHOUT the trailing action and WITHOUT post-op prose. Design any selection edits to work WITH this: selecting examples meant to train action-emission or post-op narration is futile (it gets trimmed); PLAN items whose last turn ENDS in an op survive intact, while items that merely rationalize-then-act become op-less and are dropped.
6. Drop any item whose last turn still contains a markdown code-fence (```). 
You cannot change these transforms.

## YOUR LATITUDE — the levers you control (NOT just training configs)
Each iteration you choose a RECIPE. Training configs are the cheapest and narrowest lever; the point of this engine is that you ALSO control the data. Do NOT default to a config-only iteration, and if you ever do a config-only iteration, justify it in your rationale. The levers:
1. **Training configs** (at most **{max_configs}**/iter) — rank, alpha, dropout, LR, epochs, batch, grad-accum, cutoff, LR schedule, warmup, target modules.
2. **Data composition / amount** — via `data_engine_overrides`: `min_target_examples` (minimum training dataset size) and `episode_filter_top_pct` (bias toward higher-reward episodes). Changes HOW MUCH / from WHICH episodes, without touching selection logic.
3. **Data-engine selection logic** (`{de_code}`) — REWRITE it (COMPLETE files under `revised_code/`, paths mirrored, e.g. `revised_code/data_engine.py`) to change WHAT gets selected: the selection principles/prompt, the LOG-vs-PLAN balance, which response shapes are kept, step/episode bucketing, etc.
4. **Anything else** you conclude from the review that is not a hard constraint.

Mechanics: you do NOT select examples yourself — you steer the inner data engine. `reuse` copies a prior iteration's certified selection verbatim (sweep configs cheaply on already-built data); `regenerate` re-runs the data engine (applying any `revised_code/` edits + `data_engine_overrides` first).

**Data-engine versioning — `data.data_engine_base`.** Before overlaying this iter's `revised_code/`, the engine resets to a chosen base, so edits stay revertable instead of silently accumulating:
- `"pristine"` (DEFAULT) — reset to the original starting engine, then overlay. Purpose: REVERT a prior edit (just omit the data-engine file from prior `revised_code/`), or RE-BASELINE cleanly (include a COMPLETE rewritten engine).
- `"current"` — keep the prior iter's evolved engine, then overlay. Purpose: BUILD ON a prior edit you judged good (add further changes while keeping last iter's logic).
You always receive the current (possibly-evolved) engine source ({de_code}), so you can see what the prior iter changed. `reuse` ignores this data engine versioning choice (it copies a prior certified selection, independent of engine code).

{context_section}

## Required outputs (write to your output dir)
1. **`recipe.json`** — the next iteration's decision. Schema:
```json
{{
  "iteration": <int>,
  "data": {{ "action": "reuse"|"regenerate",
             "reuse_from": "iter<K>",
             "data_engine_base": "pristine"|"current",
             "data_engine_overrides": {{ "min_target_examples": <int>, "episode_filter_top_pct": <num> }} }},
  "train_configs": [ "RANK:ALPHA:DROPOUT:LR:EPOCHS:BATCH:GRAD_ACCUM:CUTOFF:LR_SCHED:WARMUP:SAVE:NUM_GPUS:TARGET", ... ],
  "rationale": "<grounded in the ledger>"
}}
```
   - `train_configs`: 1 to {max_configs} tuples (colon-delimited, the meta_orch format).
   - `data.action="reuse"` copies a prior iteration's certified selection verbatim; `"regenerate"` re-runs the data engine (applies any `revised_code/` edits + overrides first).
2. **`revised_code/`** — OPTIONAL. Only if you are editing pipeline code this iteration. Complete files, paths mirrored.
3. **`method_description.md`** — what you are changing this iteration and why (this is fed to you next iteration as history).
4. **`analysis.md`** — your diagnosis of the ledger + artifacts that justifies the recipe.

## Operating scope
Operate ONLY within this engine directory. Your inputs are: this iteration's ledger, any training-config recommendation included in this prompt, and the pipeline codebase + base traces (in the pipeline working dir). Do NOT read anything outside this directory.
""".format(
        env=env_name,
        baseline=baseline_metric or "(unknown)",
        de_code=data_engine_code_path,
        postprocess=postprocess_path,
        max_configs=max_configs,
        context_section=build_context_section(env_name, has_prior_runs),
    )

def build_prompt(env_name, ledger_rows, ledger_md, config_recommendations,
                 artifacts_index, data_engine_code_path, output_dir,
                 max_configs, baseline_metric):
    has_prior_runs = bool(ledger_rows)
    parts = []
    parts.append("## Environment\n{}".format(
        ENV_CONTEXT.get(env_name, "(no context)")))

    if has_prior_runs:
        parts.append("## Experiment ledger (every prior run)\n{}".format(ledger_md))
    else:
        parts.append("## Experiment ledger (every prior run)\n"
                     "(ledger is empty — this is the FIRST iteration; no prior runs to read yet)")

    if artifacts_index:
        parts.append("## Per-run artifact locations\n{}".format(artifacts_index))

    if config_recommendations:
        parts.append("## Training-config recommendation (PRIOR — use or ignore)\n{}".format(
            config_recommendations.strip()))

    parts.append("## Current data-engine selection code\n"
                 "Read `{}` if you intend to change WHAT gets selected. To edit it, write the complete revised file to `revised_code/` (mirror the path).".format(
                     data_engine_code_path))

    if has_prior_runs:
        step1 = ("1. Read the prior runs' artifacts (CLAUDE.md says where each lives). Diagnose what to change.\n")
    else:
        step1 = ("1. This is the first iteration — the ledger is empty, so there are no prior runs to read. Pick a sensible starting recipe (the data action must be `regenerate`: there is no prior certified selection to reuse yet).\n")

    parts.append(
        "## Your task\n"
        + step1 +
        "2. Decide the recipe: which data (reuse a prior certified selection, or regenerate — possibly with edits to the data engine), and 1 to {maxc} train configs. You control the data<->config decomposition (e.g. keep the data and only sweep configs is fine).\n"
        "3. Write `recipe.json`, optional `revised_code/`, `method_description.md`, `analysis.md` to `{out}/`. Ground every claim in `analysis.md` in what you actually read.\n"
        "Compare configs by their eval avg_progression over the fixed seed set. The postprocess step is always applied, and the hard constraints hold.".format(maxc=max_configs, out=output_dir))

    return "\n\n".join(parts)

def run_claude_code(prompt, workspace_dir, output_dir, model, max_turns,
                    effort, timeout):
    os.makedirs(output_dir, exist_ok=True)
    allowed_tools = "Read,Write({o}/*),Write({o}/**),Bash(*),Grep,Glob".format(
        o=output_dir)
    cmd = [
        "claude", "-p", "--model", model,
        "--max-turns", str(max_turns), "--permission-mode", "acceptEdits",
        "--allowedTools", allowed_tools,
        "--output-format", "stream-json", "--verbose",
    ]
    if effort:
        cmd.extend(["--effort", effort])
    transcript_path = os.path.join(workspace_dir, "cc_transcript.jsonl")
    print("  Running Claude Code decider (max {} turns, timeout {}s)...".format(
        max_turns, timeout))
    debug_log("Prompt length: {} chars".format(len(prompt)))
    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=workspace_dir, input=prompt)
        elapsed = time.time() - start
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        if result.stderr:
            with open(os.path.join(workspace_dir, "cc_stderr.txt"), "w") as f:
                f.write(result.stderr)
        session_id = cost = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "result":
                session_id = ev.get("session_id")
                cost = ev.get("total_cost_usd")
                with open(os.path.join(workspace_dir, "cc_result.json"), "w") as f:
                    json.dump(ev, f, indent=2)
        print("  Session: {}, Cost: ${}, Elapsed: {:.0f}s".format(
            session_id, cost, elapsed))
        return True, session_id, elapsed
    except subprocess.TimeoutExpired:
        print("  TIMEOUT after {:.0f}s".format(time.time() - start))
        return False, None, time.time() - start
    except Exception as e:
        print("  ERROR: {}".format(e))
        return False, None, time.time() - start

def validate_config_tuple(tup):
    parts = tup.split(":")
    if len(parts) != len(CONFIG_FIELDS):
        return False, "expected {} colon-fields, got {}: {!r}".format(
            len(CONFIG_FIELDS), len(parts), tup)
    d = dict(zip(CONFIG_FIELDS, parts))
    for f in ("RANK", "ALPHA", "EPOCHS", "BATCH", "GRAD_ACCUM", "CUTOFF", "NUM_GPUS"):
        if not d[f].lstrip("-").isdigit():
            return False, "{} must be int, got {!r}".format(f, d[f])
    try:
        float(d["LR"])
        float(d["WARMUP"])
        float(d["DROPOUT"])
    except ValueError:
        return False, "LR/WARMUP/DROPOUT must be numeric in {!r}".format(tup)
    tgt = d["TARGET"]
    if tgt not in _VALID_TARGET_ALIASES and "," not in tgt and "_proj" not in tgt:
        return False, "TARGET {!r} not a known alias or comma-list".format(tgt)
    return True, "ok"

def validate_recipe(output_dir, max_configs):
    path = os.path.join(output_dir, "recipe.json")
    if not os.path.isfile(path):
        return False, "recipe.json not produced"
    try:
        with open(path) as f:
            recipe = json.load(f)
    except Exception as e:
        return False, "recipe.json not valid JSON: {}".format(e)

    data = recipe.get("data", {})
    action = data.get("action")
    if action not in ("reuse", "regenerate"):
        return False, "data.action must be 'reuse' or 'regenerate', got {!r}".format(action)
    if action == "reuse" and not data.get("reuse_from"):
        return False, "data.action=reuse requires data.reuse_from"
    de_base = data.get("data_engine_base", "pristine")
    if de_base not in ("pristine", "current"):
        return False, "data.data_engine_base must be 'pristine' or 'current', got {!r}".format(de_base)

    configs = recipe.get("train_configs", [])
    if not isinstance(configs, list) or not (1 <= len(configs) <= max_configs):
        return False, "train_configs must be a list of 1..{} tuples (got {})".format(
            max_configs, len(configs) if isinstance(configs, list) else "non-list")
    for c in configs:
        ok, msg = validate_config_tuple(c)
        if not ok:
            return False, "bad config tuple: {}".format(msg)
    return True, "ok ({} config(s), data.action={})".format(len(configs), action)

def read_text(path, max_chars=None):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read()
        if max_chars and len(s) > max_chars:
            s = s[:max_chars] + "\n...[truncated]..."
        return s
    except Exception:
        return None

def build_artifacts_index(ledger_rows):
    if not ledger_rows:
        return ""
    lines = ["Each ledger row's `artifacts_dir` is a per-config archive (layout in CLAUDE.md):"]
    for r in ledger_rows:
        art = r.get("artifacts_dir")
        if not art:
            continue
        lines.append("- iter {} `{}` data={} -> `{}`".format(
            r.get("iteration", "?"), r.get("config", "?"),
            r.get("data_source", "?"), art))
    return "\n".join(lines)

def main():
    p = argparse.ArgumentParser(description="Training-engine decider (Claude Code).")
    p.add_argument("--env", required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--output-dir", required=True, help="Where CC writes recipe.json + revised_code/ + md files")
    p.add_argument("--ledger", required=True, help="Path to ledger.jsonl")
    p.add_argument("--data-engine-code", required=True, help="Path to the CURRENT data-engine.py CC may edit")
    p.add_argument("--config-recommendations", default="")
    p.add_argument("--baseline-metric", default="", help="scaffold-only (no-training) baseline progression for context; leave empty for unknown")
    p.add_argument("--max-configs", type=int, default=DEFAULT_MAX_CONFIGS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--effort", default="max", choices=["low", "medium", "high", "max"])
    p.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    p.add_argument("--timeout", type=int, default=SESSION_TIMEOUT)
    args = p.parse_args()

    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    setup_debug_logger(args.output_dir)

    postprocess_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprocess.py")

    print("=" * 70)
    print("Training-Engine Decider — {} — iteration {}".format(
        args.env, args.iteration))
    print("=" * 70)

    ledger_rows = load_ledger(args.ledger)
    has_prior_runs = bool(ledger_rows)
    ledger_md = format_ledger_md(ledger_rows, args.env)
    artifacts_index = build_artifacts_index(ledger_rows)

    config_recs = read_text(args.config_recommendations) if args.iteration == 1 else None

    print("Ledger rows: {}".format(len(ledger_rows)))

    claude_md = build_claude_md(
        args.env, args.baseline_metric, args.max_configs,
        args.data_engine_code, postprocess_path, has_prior_runs)
    with open(os.path.join(args.output_dir, "CLAUDE.md"), "w", encoding="utf-8") as f:
        f.write(claude_md)

    prompt = build_prompt(
        args.env, ledger_rows, ledger_md,
        config_recs, artifacts_index, args.data_engine_code,
        args.output_dir, args.max_configs, args.baseline_metric)
    with open(os.path.join(args.output_dir, "prompt.md"), "w", encoding="utf-8") as f:
        f.write(prompt)
    print("Wrote CLAUDE.md ({}c) + prompt.md ({}c)".format(len(claude_md), len(prompt)))

    ok_run, session_id, elapsed = run_claude_code(
        prompt, args.output_dir, args.output_dir,
        model=args.model, max_turns=args.max_turns,
        effort=args.effort, timeout=args.timeout)
    if not ok_run:
        print("\n❌ Decider CC session failed/timed out.")
        sys.exit(1)

    ok, msg = validate_recipe(args.output_dir, args.max_configs)
    if not ok:
        print("\n❌ Invalid recipe: {}".format(msg))
        sys.exit(2)

    revised = os.path.join(args.output_dir, "revised_code")
    n_revised = 0
    if os.path.isdir(revised):
        for root, _dirs, files in os.walk(revised):
            n_revised += len(files)
    print("\n✅ Decider done. Recipe OK: {}. revised_code files: {}".format(
        msg, n_revised))
    sys.exit(0)

if __name__ == "__main__":
    main()

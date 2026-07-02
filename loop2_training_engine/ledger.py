import argparse
import glob
import json
import os
import re

NLE_STEP_STATUS = {-1: "aborted", 0: "running", 1: "death", 2: "task_success"}

def _clean_end_reason(v):
    if v is None or v == "?" or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return NLE_STEP_STATUS.get(v, "status_{}".format(v))
    return v

def per_episode_completed(json_path):
    try:
        with open(json_path) as f:
            d = json.load(f)
    except Exception:
        return None
    return {
        "progression": "{:.2f}%".format(100 * float(d.get("progression", 0.0))),
        "_prog_float": float(d.get("progression", 0.0)),
        "end_reason": d.get("end_reason", "?"),
        "steps": d.get("num_steps", d.get("total_steps")),
        "episode_return": d.get("episode_return"),
    }

def _task_of(path):
    parts = path.replace(os.sep, "/").split("/")
    return parts[-2] if len(parts) >= 2 else "?"

def _avg_pct(prog_floats):
    if not prog_floats:
        return None
    return round(100 * sum(prog_floats) / len(prog_floats), 2)

def build_run_row(run_dir, env, iteration, data_source,
                  config, num_examples, train_loss, eval_seeds):
    runs_glob = os.path.join(run_dir, "results", "*", env, "*", "*_run_*.json")
    jsons = sorted(p for p in glob.glob(runs_glob) if not p.endswith("_summary.json"))

    def run_idx(p):
        m = re.search(r"_run_(\d+)", p)
        return int(m.group(1)) if m else -1

    per_episode = []
    for jp in jsons:
        idx = run_idx(jp)
        seed = eval_seeds[idx] if 0 <= idx < len(eval_seeds) else None
        ep = per_episode_completed(jp)
        if ep is None:
            continue
        ep["seed"] = seed
        ep["task"] = _task_of(jp)
        per_episode.append(ep)

    tasks = sorted({e["task"] for e in per_episode})
    multitask = len(tasks) > 1

    def fmt_eps():
        items = []
        for e in sorted(per_episode, key=lambda x: (x.get("task", ""), x.get("seed", 0) or 0)):
            tag = (str(e.get("task")) + "/") if multitask else ""
            items.append("{}s{}={}".format(tag, e.get("seed"), e.get("progression", "?")))
        return " ".join(items) if items else "(none)"

    cleaned = []
    for e in per_episode:
        entry = {"seed": e.get("seed")}
        if multitask:
            entry["task"] = e.get("task")
        entry["progression"] = e.get("progression")
        for k in ("steps", "episode_return"):
            if e.get(k) is not None:
                entry[k] = e[k]
        er = _clean_end_reason(e.get("end_reason"))
        if er is not None:
            entry["end_reason"] = er
        cleaned.append(entry)

    prog = {"n_episodes": len(per_episode),
            "avg_progression": _avg_pct([e["_prog_float"] for e in per_episode])}
    if multitask:
        prog["n_tasks"] = len(tasks)
        prog["per_task_avg_progression"] = {
            tk: _avg_pct([e["_prog_float"] for e in per_episode if e["task"] == tk])
            for tk in tasks}
    prog["episodes"] = fmt_eps()
    prog["per_episode"] = cleaned

    return {
        "iteration": int(iteration), "data_source": data_source, "config": config,
        "num_examples": num_examples, "train_loss": train_loss,
        "progression": prog, "artifacts_dir": run_dir,
    }

def _is_multitask_row(prog):
    return "per_task_avg_progression" in prog

def _headline(prog):
    a = prog.get("avg_progression")
    n = prog.get("n_episodes", 0)
    return "{:.2f}% (avg of {} eps)".format(a, n) if a is not None else "(no eps)"

def _per_task_str(prog, short=False):
    parts = []
    for tk, v in sorted(prog.get("per_task_avg_progression", {}).items()):
        name = tk.replace("MiniHack-", "").replace("-v0", "") if short else tk
        parts.append("{}={}".format(name, "{:.0f}%".format(v) if short and v is not None
                                    else ("{:.2f}%".format(v) if v is not None else "?")))
    return "; ".join(parts)

def format_ledger_md_decider(rows, env):
    if not rows:
        return "(ledger is empty -- this is the FIRST iteration)"
    multitask = any(_is_multitask_row(r.get("progression", {})) for r in rows)
    detail_hdr = "per-task avg" if multitask else "per-episode"

    legend = ("`avg_progression` = this run's average progression over its fixed eval episodes (the eval metric the scaffold baseline is quoted in).")

    cols = ["iter", "data_src", "config", "n_ex", "loss",
            "avg_progression", detail_hdr, "artifacts_dir"]

    lines = [legend, "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        prog = r.get("progression", {})
        if not isinstance(prog, dict) or "n_episodes" not in prog:
            continue
        detail = _per_task_str(prog, short=True) if multitask else prog.get("episodes", "?")
        cells = [str(r.get("iteration", "?")), str(r.get("data_source", "?")),
                 "`{}`".format(r.get("config", "?")), str(r.get("num_examples", "?")),
                 str(r.get("train_loss", "?")), _headline(prog), detail,
                 r.get("artifacts_dir", "?")]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)

_LEDGER_ROW_DESC = {
    "default": ("- **Row in `ledger.jsonl`.** `progression` carries this run's `avg_progression` (its average progression over its fixed eval episodes = the eval metric the scaffold baseline is quoted in), `n_episodes`, `episodes` (each seed listed individually as `sSEED=prog`), and `per_episode` (an array: seed, progression, steps, episode_return)."),
    "minihack": ("- **Row in `ledger.jsonl`.** `progression` carries this run's `avg_progression` (macro-average over all 40 eval episodes = 8 tasks x 5 seeds, the eval metric), `per_task_avg_progression` (the average per task), `n_episodes`, `n_tasks`, `episodes` (each `task/sSEED=prog`), and `per_episode` (seed, task, progression, steps, episode_return, end_reason as task_success/death)."),
}

def ledger_row_desc(env):
    if env == "minihack":
        return _LEDGER_ROW_DESC["minihack"]
    return _LEDGER_ROW_DESC["default"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="")
    p.add_argument("--env", default="nle")
    p.add_argument("--eval-seeds", default="42,43,44,45,46")
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--data-source", default="")
    p.add_argument("--config", default="")
    p.add_argument("--num-examples", default="")
    p.add_argument("--train-loss", default="")
    args = p.parse_args()

    seeds = [int(x) for x in args.eval_seeds.split(",") if x.strip()]
    row = build_run_row(
        args.run_dir, args.env, args.iteration, args.data_source, args.config,
        args.num_examples, args.train_loss, seeds)
    print(json.dumps(row))

if __name__ == "__main__":
    main()

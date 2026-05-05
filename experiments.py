"""
experiments.py -- Automated experiment runner for LBMPC vs MPC comparison.

Runs batches of experiments with different configurations, collects metrics
from CSV logs, and produces summary tables.

Usage:
    python experiments.py                        # run all experiments
    python experiments.py --experiment baseline  # run one experiment
    python experiments.py --list                 # list available experiments
"""

import os
import sys
import subprocess
import csv
import json
import time
import argparse
import numpy as np
from pathlib import Path


# --------------------------------------------------------------------------
# Result parsing
# --------------------------------------------------------------------------

def parse_log_csv(csv_path: str) -> dict:
    """Parse a trajectory log CSV and compute summary metrics."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"rms_error": float("nan"), "max_error": float("nan"),
                "mean_error": float("nan"), "final_error": float("nan"),
                "steps": 0}

    errors = [float(r["pos_err"]) for r in rows]
    errors = np.array(errors)

    return {
        "steps": len(rows),
        "rms_error": float(np.sqrt(np.mean(errors ** 2))),
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "final_error": float(errors[-1]),
        "std_error": float(np.std(errors)),
    }


def find_latest_log(log_dir: str = "logs", prefix: str = "") -> str:
    """Find the most recently created CSV log file."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return None
    csvs = sorted(log_path.glob(f"{prefix}*.csv"), key=os.path.getmtime)
    return str(csvs[-1]) if csvs else None


# --------------------------------------------------------------------------
# Single experiment runner
# --------------------------------------------------------------------------

def run_single(config: dict, python: str = sys.executable) -> dict:
    """
    Run a single main.py experiment with given config and return metrics.

    Args:
        config: dict of CLI arguments for main.py
            e.g. {"traj": "circle", "steps": 100, "residual_type": "nn"}
        python: path to python executable

    Returns:
        dict with config + metrics
    """
    # Build command
    cmd = [python, "main.py", "--no-gui"]
    for key, val in config.items():
        if key.startswith("_"):
            continue  # skip metadata keys
        flag = f"--{key.replace('_', '-')}"
        if isinstance(val, bool):
            if val:
                cmd.append(flag)
            continue
        if isinstance(val, (list, tuple)):
            cmd.append(flag)
            cmd.extend([str(v) for v in val])
        elif val is not None:
            cmd.extend([flag, str(val)])

    label = config.get("_label", " ".join(cmd[2:]))
    print(f"\n{'='*60}")
    print(f"  Experiment: {label}")
    print(f"  Command:    {' '.join(cmd)}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0

    # Print output
    if result.stdout:
        # Only print last few lines (summary)
        lines = result.stdout.strip().split("\n")
        for line in lines[-10:]:
            print(f"  {line}")

    if result.returncode != 0:
        print(f"  [FAILED] exit code {result.returncode}")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-5:]:
                print(f"  ERR: {line}")
        return {**config, "status": "FAILED", "elapsed_s": elapsed}

    # Parse the latest log
    prefix = config.get("traj", "circle")
    if prefix.startswith("wobbly_"):
        pass  # prefix already correct
    log_file = find_latest_log("logs", prefix)
    metrics = parse_log_csv(log_file) if log_file else {}

    return {**config, **metrics, "status": "OK",
            "elapsed_s": round(elapsed, 1), "log_file": log_file}


# --------------------------------------------------------------------------
# Experiment definitions
# --------------------------------------------------------------------------

def exp_baseline() -> list:
    """Experiment A: Baseline comparison -- NN vs GP on circle and figure-8."""
    configs = []
    for traj in ["circle", "figure8"]:
        for rtype in ["nn", "gp"]:
            configs.append({
                "_label": f"Baseline | {traj} | {rtype}",
                "traj": traj,
                "steps": 300,
                "residual_type": rtype,
            })
    return configs


def exp_speed_sweep() -> list:
    """Experiment B: Speed sweep -- vary lap_time."""
    configs = []
    for lap in [20.0, 15.0, 10.0, 7.0]:
        configs.append({
            "_label": f"Speed sweep | lap_time={lap}s",
            "traj": "circle",
            "steps": 300,
            "lap_time": lap,
        })
    return configs


def exp_horizon_sweep() -> list:
    """Experiment C: Horizon sweep -- vary prediction horizon."""
    configs = []
    for h in [5, 10, 15, 20]:
        configs.append({
            "_label": f"Horizon sweep | N={h}",
            "traj": "circle",
            "steps": 200,
            "horizon": h,
        })
    return configs


def exp_constraint_sweep() -> list:
    """Experiment D: Constraint sweep -- vary max velocity."""
    configs = []
    for mv in [1.0, 2.0, 3.0, 5.0]:
        configs.append({
            "_label": f"Constraint sweep | max_vel={mv}m/s",
            "traj": "circle",
            "steps": 200,
            "max_vel": mv,
        })
    # Unconstrained baseline
    configs.append({
        "_label": "Constraint sweep | unconstrained",
        "traj": "circle",
        "steps": 200,
    })
    return configs


def exp_wobbly() -> list:
    """Experiment E: Wobbly trajectories -- NN vs GP."""
    configs = []
    for traj in ["wobbly_circle", "wobbly_figure8"]:
        for rtype in ["nn", "gp"]:
            configs.append({
                "_label": f"Wobbly | {traj} | {rtype}",
                "traj": traj,
                "steps": 300,
                "residual_type": rtype,
            })
    return configs


def exp_wind() -> list:
    """Experiment F: Wind disturbance -- NN vs GP under wind."""
    configs = []
    for rtype in ["nn", "gp"]:
        # No wind baseline
        configs.append({
            "_label": f"Wind | no_wind | {rtype}",
            "traj": "circle",
            "steps": 200,
            "residual_type": rtype,
        })
        # Moderate wind
        configs.append({
            "_label": f"Wind | moderate | {rtype}",
            "traj": "circle",
            "steps": 200,
            "residual_type": rtype,
            "wind": [0.5, 0.3, 0.0],
        })
        # Strong wind
        configs.append({
            "_label": f"Wind | strong | {rtype}",
            "traj": "circle",
            "steps": 200,
            "residual_type": rtype,
            "wind": [1.0, 0.5, 0.0],
        })
    return configs


EXPERIMENTS = {
    "baseline":   exp_baseline,
    "speed":      exp_speed_sweep,
    "horizon":    exp_horizon_sweep,
    "constraint": exp_constraint_sweep,
    "wobbly":     exp_wobbly,
    "wind":       exp_wind,
}


# --------------------------------------------------------------------------
# Results table
# --------------------------------------------------------------------------

def print_results_table(results: list):
    """Print a formatted comparison table of experiment results."""
    if not results:
        print("No results to display.")
        return

    print(f"\n{'='*90}")
    print(f"  {'Label':<40} {'RMS Err':>8} {'Max Err':>8} {'Final':>8} {'Time':>7} {'Status':>6}")
    print(f"{'='*90}")

    for r in results:
        label = r.get("_label", "?")[:40]
        rms   = r.get("rms_error", float("nan"))
        mx    = r.get("max_error", float("nan"))
        final = r.get("final_error", float("nan"))
        t     = r.get("elapsed_s", 0)
        status = r.get("status", "?")

        rms_s   = f"{rms:.4f}" if not np.isnan(rms) else "N/A"
        mx_s    = f"{mx:.4f}" if not np.isnan(mx) else "N/A"
        final_s = f"{final:.4f}" if not np.isnan(final) else "N/A"

        print(f"  {label:<40} {rms_s:>8} {mx_s:>8} {final_s:>8} {t:>6.1f}s {status:>6}")

    print(f"{'='*90}\n")


def save_results_json(results: list, path: str = "results/experiments.json"):
    """Save results to JSON for later analysis/plotting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Clean non-serializable items
    clean = []
    for r in results:
        cr = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                cr[k] = float(v)
            else:
                cr[k] = v
        clean.append(cr)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print(f"[experiments] Results saved -> {path}")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LBMPC experiments")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=list(EXPERIMENTS.keys()) + ["all"],
                        help="Which experiment to run")
    parser.add_argument("--list", action="store_true",
                        help="List available experiments and exit")
    parser.add_argument("--output", type=str, default="results/experiments.json",
                        help="Output JSON path for results")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable experiments:")
        for name, fn in EXPERIMENTS.items():
            configs = fn()
            print(f"  {name:>12} -- {fn.__doc__.strip()} ({len(configs)} runs)")
        sys.exit(0)

    # Collect configs
    if args.experiment == "all":
        all_configs = []
        for name, fn in EXPERIMENTS.items():
            all_configs.extend(fn())
    else:
        all_configs = EXPERIMENTS[args.experiment]()

    print(f"\n[experiments] Running {len(all_configs)} experiments...\n")

    results = []
    for i, config in enumerate(all_configs):
        print(f"\n--- [{i+1}/{len(all_configs)}] ---")
        result = run_single(config)
        results.append(result)

    print_results_table(results)
    save_results_json(results, args.output)

"""
plotting.py -- Plotting utilities for LBMPC experiment analysis.

Generates comparative plots from experiment JSON results and individual
trajectory log CSVs.

Usage:
    python plotting.py                                         # plot from latest results JSON
    python plotting.py --results results/experiments.json      # specific results file
    python plotting.py --log logs/circle_20260505_191545.csv   # plot single trajectory
"""

import os
import sys
import json
import csv
import argparse
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for headless runs
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[plotting] WARNING: matplotlib not installed. Install with: pip install matplotlib")


# --------------------------------------------------------------------------
# Color palette (premium, distinguishable)
# --------------------------------------------------------------------------

COLORS = {
    "nn":       "#3498db",   # blue
    "gp":       "#e74c3c",   # red
    "circle":   "#2ecc71",   # green
    "figure8":  "#9b59b6",   # purple
    "wobbly_circle":  "#f39c12",  # orange
    "wobbly_figure8": "#1abc9c",  # teal
}

STYLE_MAP = {
    "nn": {"marker": "o", "ls": "-"},
    "gp": {"marker": "s", "ls": "--"},
}


def _setup_style():
    """Apply publication-quality plot styling."""
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# --------------------------------------------------------------------------
# Plot: Single trajectory log
# --------------------------------------------------------------------------

def plot_trajectory_log(csv_path: str, out_dir: str = "plots"):
    """
    Generate plots from a single trajectory log CSV.

    Creates:
        - XY path plot (actual vs reference)
        - Tracking error over time
        - Control inputs over time
    """
    if not HAS_MPL:
        print("[plotting] matplotlib required. Skipping.")
        return

    _setup_style()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]

    # Parse CSV
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})

    if not rows:
        print(f"[plotting] Empty log: {csv_path}")
        return

    steps   = [r["step"] for r in rows]
    x_true  = [r["x_true"] for r in rows]
    y_true  = [r["y_true"] for r in rows]
    x_ref   = [r["x_ref"] for r in rows]
    y_ref   = [r["y_ref"] for r in rows]
    pos_err = [r["pos_err"] for r in rows]
    u_T     = [r["u_T"] for r in rows]
    u_roll  = [r["u_roll"] for r in rows]
    u_pitch = [r["u_pitch"] for r in rows]

    # --- Plot 1: XY trajectory ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_ref, y_ref, "r--", lw=2, alpha=0.7, label="Reference")
    ax.plot(x_true, y_true, "b-", lw=1.5, alpha=0.9, label="Actual")
    ax.plot(x_true[0], y_true[0], "go", ms=10, label="Start")
    ax.plot(x_true[-1], y_true[-1], "r^", ms=10, label="End")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Trajectory: {base}")
    ax.set_aspect("equal")
    ax.legend()
    path1 = os.path.join(out_dir, f"{base}_xy.png")
    fig.savefig(path1, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path1}")

    # --- Plot 2: Tracking error ---
    fig, ax = plt.subplots()
    ax.plot(steps, pos_err, "b-", lw=1.5)
    ax.axhline(np.mean(pos_err), color="r", ls="--", alpha=0.5,
               label=f"Mean = {np.mean(pos_err):.3f} m")
    ax.set_xlabel("Step")
    ax.set_ylabel("Position Error (m)")
    ax.set_title(f"Tracking Error: {base}")
    ax.legend()
    path2 = os.path.join(out_dir, f"{base}_error.png")
    fig.savefig(path2, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path2}")

    # --- Plot 3: Control inputs ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, u_T, "b-", lw=1)
    axes[0].set_ylabel("Thrust (N)")
    axes[0].set_title(f"Control Inputs: {base}")
    axes[1].plot(steps, u_roll, "r-", lw=1, label="Roll")
    axes[1].plot(steps, u_pitch, "g-", lw=1, label="Pitch")
    axes[1].set_ylabel("Torque (N*m)")
    axes[1].legend()
    axes[2].plot(steps, pos_err, "purple", lw=1)
    axes[2].set_ylabel("Position Error (m)")
    axes[2].set_xlabel("Step")
    fig.tight_layout()
    path3 = os.path.join(out_dir, f"{base}_controls.png")
    fig.savefig(path3, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path3}")


# --------------------------------------------------------------------------
# Plot: Experiment comparison bar charts
# --------------------------------------------------------------------------

def plot_experiment_comparison(results: list, out_dir: str = "plots"):
    """
    Create comparison bar charts from experiment results JSON.

    Groups by experiment label prefix and plots RMS error bars.
    """
    if not HAS_MPL:
        print("[plotting] matplotlib required. Skipping.")
        return

    _setup_style()
    os.makedirs(out_dir, exist_ok=True)

    # Filter to successful runs
    ok = [r for r in results if r.get("status") == "OK"]
    if not ok:
        print("[plotting] No successful results to plot.")
        return

    # Group by experiment type (first word of label)
    groups = {}
    for r in ok:
        label = r.get("_label", "unknown")
        group = label.split("|")[0].strip()
        groups.setdefault(group, []).append(r)

    for group_name, group_results in groups.items():
        labels = [r.get("_label", "?").split("|", 1)[-1].strip()
                  for r in group_results]
        rms = [r.get("rms_error", 0) for r in group_results]
        max_e = [r.get("max_error", 0) for r in group_results]

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
        x = np.arange(len(labels))
        w = 0.35

        bars1 = ax.bar(x - w/2, rms, w, label="RMS Error", color="#3498db", alpha=0.85)
        bars2 = ax.bar(x + w/2, max_e, w, label="Max Error", color="#e74c3c", alpha=0.85)

        ax.set_xlabel("Configuration")
        ax.set_ylabel("Error (m)")
        ax.set_title(f"{group_name} -- Tracking Error Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.legend()

        # Value labels on bars
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8)

        fig.tight_layout()
        safe_name = group_name.lower().replace(" ", "_")
        path = os.path.join(out_dir, f"compare_{safe_name}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [plot] {path}")


# --------------------------------------------------------------------------
# Plot: Multi-trajectory overlay
# --------------------------------------------------------------------------

def plot_multi_trajectory(csv_paths: list, labels: list = None,
                          out_dir: str = "plots"):
    """
    Overlay multiple trajectory logs on the same XY plot for comparison.

    Args:
        csv_paths: list of CSV file paths
        labels:    list of labels (one per path)
        out_dir:   output directory
    """
    if not HAS_MPL:
        print("[plotting] matplotlib required. Skipping.")
        return

    _setup_style()
    os.makedirs(out_dir, exist_ok=True)
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]

    fig_xy, ax_xy = plt.subplots(figsize=(8, 8))
    fig_err, ax_err = plt.subplots(figsize=(10, 6))

    for i, csv_path in enumerate(csv_paths):
        if not os.path.exists(csv_path):
            continue

        rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({k: float(v) for k, v in row.items()})

        if not rows:
            continue

        label = labels[i] if labels and i < len(labels) else os.path.basename(csv_path)
        c = colors[i % len(colors)]

        x_true = [r["x_true"] for r in rows]
        y_true = [r["y_true"] for r in rows]
        steps  = [r["step"] for r in rows]
        errors = [r["pos_err"] for r in rows]

        ax_xy.plot(x_true, y_true, "-", color=c, lw=1.5, alpha=0.8, label=label)
        ax_err.plot(steps, errors, "-", color=c, lw=1.5, alpha=0.8, label=label)

    # Reference from first log
    if csv_paths and os.path.exists(csv_paths[0]):
        rows0 = []
        with open(csv_paths[0], "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows0.append({k: float(v) for k, v in row.items()})
        if rows0:
            x_ref = [r["x_ref"] for r in rows0]
            y_ref = [r["y_ref"] for r in rows0]
            ax_xy.plot(x_ref, y_ref, "k--", lw=2, alpha=0.4, label="Reference")

    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Trajectory Comparison")
    ax_xy.set_aspect("equal")
    ax_xy.legend()
    fig_xy.savefig(os.path.join(out_dir, "multi_trajectory_xy.png"), bbox_inches="tight")
    plt.close(fig_xy)

    ax_err.set_xlabel("Step")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Error Comparison")
    ax_err.legend()
    fig_err.savefig(os.path.join(out_dir, "multi_trajectory_error.png"), bbox_inches="tight")
    plt.close(fig_err)

    print(f"  [plot] plots/multi_trajectory_xy.png")
    print(f"  [plot] plots/multi_trajectory_error.png")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LBMPC plotting utilities")
    parser.add_argument("--results", type=str, default="results/experiments.json",
                        help="Experiment results JSON to plot")
    parser.add_argument("--log", type=str, default=None,
                        help="Single trajectory log CSV to plot")
    parser.add_argument("--logs", type=str, nargs="+", default=None,
                        help="Multiple log CSVs for overlay comparison")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Labels for multi-log overlay")
    parser.add_argument("--out", type=str, default="plots",
                        help="Output directory for plots")
    args = parser.parse_args()

    if not HAS_MPL:
        print("[plotting] ERROR: matplotlib is required. Install: pip install matplotlib")
        sys.exit(1)

    if args.log:
        # Single trajectory plot
        print(f"\n[plotting] Plotting single trajectory: {args.log}")
        plot_trajectory_log(args.log, out_dir=args.out)

    elif args.logs:
        # Multi-trajectory overlay
        print(f"\n[plotting] Overlay {len(args.logs)} trajectories...")
        plot_multi_trajectory(args.logs, labels=args.labels, out_dir=args.out)

    elif os.path.exists(args.results):
        # Experiment comparison
        print(f"\n[plotting] Loading results: {args.results}")
        with open(args.results, "r") as f:
            results = json.load(f)
        print(f"[plotting] {len(results)} results loaded.")
        plot_experiment_comparison(results, out_dir=args.out)

    else:
        print(f"[plotting] No input found. Provide --log, --logs, or --results.")
        print(f"  --results {args.results}  (file not found)")
        sys.exit(1)

    print("\n[plotting] Done.")

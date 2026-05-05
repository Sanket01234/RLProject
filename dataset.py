# dataset.py
"""
Data collection script for the LBMPC pipeline.

Runs the PyBullet sim with a mixed exploration policy and records
(x_t, u_t, x_{t+1}) transition tuples.

Usage:
    python dataset.py
    python dataset.py --episodes 100 --steps 300 --gui
"""

import argparse
import os
import numpy as np

from env import DroneEnvironment, DRONE_PARAMS, U_MIN, U_MAX


# ------------------------------------------------------------------------------
# Mixed exploration policy
# ------------------------------------------------------------------------------

def sample_control(policy: str, params: dict) -> np.ndarray:
    """
    Sample a control input according to the chosen exploration policy.

    Policy mix (see plan):
        hover            -> 20% of episodes  - pure equilibrium
        small_noise      -> 50% of episodes  - normal operating regime
        large_disturbance-> 30% of episodes  - edge cases & aggressive motion

    Args:
        policy: one of {"hover", "small_noise", "large_disturbance"}
        params: drone parameter dict (needs "mass", "g")

    Returns:
        u: (4,) control vector [T, tau_roll, tau_pitch, tau_yaw]
    """
    T_hover = params["mass"] * params["g"]

    if policy == "hover":
        return np.array([T_hover, 0.0, 0.0, 0.0])

    elif policy == "small_noise":
        return np.array([
            T_hover + np.random.uniform(-0.5, 0.5),
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.05, 0.05),
            np.random.uniform(-0.02, 0.02),
        ])

    elif policy == "large_disturbance":
        return np.array([
            T_hover + np.random.uniform(-2.0, 2.0),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(-0.1, 0.1),
        ])

    else:
        raise ValueError(f"Unknown policy: {policy!r}")


def _pick_policy() -> str:
    """Randomly choose a policy according to the 20/50/30 split."""
    r = np.random.random()
    if r < 0.20:
        return "hover"
    elif r < 0.70:           # 0.20 + 0.50
        return "small_noise"
    else:
        return "large_disturbance"


# ------------------------------------------------------------------------------
# Diverse start positions
# ------------------------------------------------------------------------------

def _sample_start_pos() -> list:
    """
    Sample a random starting position to improve state-space coverage.
    Stays within a safe region above the ground plane.
    """
    x = np.random.uniform(-1.0,  1.0)
    y = np.random.uniform(-1.0,  1.0)
    z = np.random.uniform( 0.8,  2.0)
    return [float(x), float(y), float(z)]


# ------------------------------------------------------------------------------
# Collection loop
# ------------------------------------------------------------------------------

def collect_data(env: DroneEnvironment,
                 n_episodes: int = 50,
                 steps_per_episode: int = 200) -> dict:
    """
    Run the simulation and collect (x_t, u_t, x_{t+1}) tuples.

    Args:
        env:               DroneEnvironment instance (already initialised)
        n_episodes:        number of episodes to run
        steps_per_episode: steps per episode (total N ≈ n_episodes × steps)

    Returns:
        dataset dict with keys:
            "X"      -> (N, 12)   current states
            "U"      -> (N,  4)   applied controls
            "X_next" -> (N, 12)   resulting next states
    """
    X_list, U_list, X_next_list = [], [], []

    for ep in range(n_episodes):
        policy   = _pick_policy()
        start_pos = _sample_start_pos()
        x = env.reset(pos=start_pos)

        ep_x, ep_u, ep_xn = [], [], []

        for _ in range(steps_per_episode):
            u = sample_control(policy, env.params)
            u = np.clip(u, U_MIN, U_MAX)   # safety: enforce bounds before logging

            x_next = env.step(u)

            # Prevent learning ground reaction forces as "aerodynamics"
            if x_next[2] < 0.15:
                break

            ep_x.append(x.copy())
            ep_u.append(u.copy())
            ep_xn.append(x_next.copy())

            x = x_next

        X_list.append(np.array(ep_x))
        U_list.append(np.array(ep_u))
        X_next_list.append(np.array(ep_xn))

        if (ep + 1) % 10 == 0 or ep == 0:
            total = sum(len(a) for a in X_list)
            print(f"  Episode {ep+1:>3}/{n_episodes} | policy={policy:<18} "
                  f"| steps={len(ep_x):>3} | total transitions={total}")

    dataset = {
        "X":      np.concatenate(X_list,     axis=0),
        "U":      np.concatenate(U_list,     axis=0),
        "X_next": np.concatenate(X_next_list, axis=0),
    }
    return dataset


# ------------------------------------------------------------------------------
# MPC-controlled data collection (for GP training — per project slides)
# ------------------------------------------------------------------------------

def collect_mpc_data(env, mpc_controller, trajectory_fn,
                     traj_kwargs: dict = None,
                     dt: float = 0.05) -> dict:
    """
    Collect transition data while an MPC controller tracks a trajectory.

    Per the project slides (Step 1):
        "Run the existing MPC on circle and figure-eight trajectories.
         Log (x_k, y_k), (v_{x,k}, v_{y,k}), (a_{x,k}, a_{y,k}),
         and (x_{k+1}, y_{k+1})."

    This produces higher-quality training data for the GP model compared
    to random exploration, because it captures the velocity-dependent
    residuals that appear during actual trajectory tracking.

    Args:
        env:             DroneEnvironment instance
        mpc_controller:  LBMPC instance (or any controller with .solve())
        trajectory_fn:   function returning (n, 12) reference trajectory
        traj_kwargs:     keyword arguments for trajectory_fn
        dt:              timestep

    Returns:
        dataset dict with keys: "X", "U", "X_next"
    """
    if traj_kwargs is None:
        traj_kwargs = {}

    x_ref_full = trajectory_fn(**traj_kwargs)
    n_steps    = len(x_ref_full)

    # Draw trajectory and reset drone to start position
    env.draw_trajectory(x_ref_full[:, 0], x_ref_full[:, 1],
                        z_height=x_ref_full[0, 2])
    x = env.reset(pos=[x_ref_full[0, 0], x_ref_full[0, 1], x_ref_full[0, 2]])

    X_list, U_list, X_next_list = [], [], []

    print(f"[dataset] Collecting MPC-controlled data | {n_steps} steps")

    for step in range(n_steps):
        x_ref_window = x_ref_full[step : step + mpc_controller.horizon]

        # Solve MPC for optimal control
        u = mpc_controller.solve(x, x_ref_window)

        # Step environment
        x_next = env.step(u)

        # Log transition
        X_list.append(x.copy())
        U_list.append(u.copy())
        X_next_list.append(x_next.copy())

        if (step + 1) % 100 == 0 or step == 0:
            pos_err = np.linalg.norm(x_next[:3] - x_ref_full[step, :3])
            print(f"  Step {step+1:>4}/{n_steps} | "
                  f"pos_err={pos_err:.3f}m | "
                  f"v=[{x_next[3]:.2f}, {x_next[4]:.2f}]")

        x = x_next

    dataset = {
        "X":      np.array(X_list, dtype=np.float32),
        "U":      np.array(U_list, dtype=np.float32),
        "X_next": np.array(X_next_list, dtype=np.float32),
    }
    print(f"[dataset] Collected {len(X_list)} MPC-controlled transitions")
    return dataset


# ------------------------------------------------------------------------------
# Save / load helpers
# ------------------------------------------------------------------------------

def save_dataset(dataset: dict, path: str = "data/transitions.npz") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **dataset)
    size = os.path.getsize(path) / 1024
    print(f"[dataset] Saved {dataset['X'].shape[0]} transitions -> {path}  ({size:.1f} KB)")


def load_dataset(path: str = "data/transitions.npz") -> dict:
    data = np.load(path)
    dataset = {k: data[k] for k in data.files}
    print(f"[dataset] Loaded {dataset['X'].shape[0]} transitions from {path}")
    return dataset


# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------

def validate_dataset(dataset: dict) -> None:
    """Print coverage statistics and run shape assertions."""
    X, U, Xn = dataset["X"], dataset["U"], dataset["X_next"]

    print("\n-- Phase 2 Validation ----------------------------------")
    print(f"  Transitions : {X.shape[0]}")
    print(f"  X  shape    : {X.shape}     (expected (N, 12))")
    print(f"  U  shape    : {U.shape}      (expected (N,  4))")
    print(f"  X_next shape: {Xn.shape}    (expected (N, 12))")
    print(f"  State mean  : {X.mean(axis=0).round(3)}")
    print(f"  State std   : {X.std(axis=0).round(3)}")
    print(f"  Pos std (x,y,z): {X[:, :3].std(axis=0).round(3)}  (target > 0.1)")
    print(f"  Control mean: {U.mean(axis=0).round(3)}")
    print(f"  Control std : {U.std(axis=0).round(3)}")

    assert X.shape[1]  == 12, "X must be (N, 12)"
    assert U.shape[1]  ==  4, "U must be (N, 4)"
    assert Xn.shape[1] == 12, "X_next must be (N, 12)"
    assert X.shape[0]  == U.shape[0] == Xn.shape[0], "Row counts must match"
    assert X[:, :3].std(axis=0).min() > 0.05, (
        "Position coverage too low - increase n_episodes or use large_disturbance policy"
    )
    print("  PASS Phase 2 PASSED")


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LBMPC data collection")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of episodes (default: 50)")
    parser.add_argument("--steps",    type=int, default=200,
                        help="Steps per episode (default: 200)")
    parser.add_argument("--out",      type=str, default="data/transitions.npz",
                        help="Output path (default: data/transitions.npz)")
    parser.add_argument("--gui",      action="store_true",
                        help="Show PyBullet GUI (slow but visual)")
    args = parser.parse_args()

    print(f"[dataset] Collecting data | episodes={args.episodes} "
          f"steps/ep={args.steps} | gui={args.gui}")

    env = DroneEnvironment("quodcopter.urdf.xml", gui=args.gui)
    dataset = collect_data(env, n_episodes=args.episodes,
                           steps_per_episode=args.steps)
    env.disconnect()

    save_dataset(dataset, args.out)
    validate_dataset(dataset)

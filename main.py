# main.py
"""
Full LBMPC control loop with trajectory tracking and optional online learning.

Trajectories:
    circle    — standard circular path at 1 m altitude
    figure8   — figure-of-eight path

Usage:
    # First run data collection + training:
    python dataset.py --episodes 50
    python train.py

    # Then run the controller:
    python main.py
    python main.py --traj figure8 --horizon 20 --online
"""

import argparse
import os
import time
import numpy as np

from logger import TrajectoryLogger
from env import DroneEnvironment, DRONE_PARAMS
from nominal import f_nominal
from model import ResidualModel, Normalizer, load_model, save_model
from LBMPC import LBMPC
from dataset import collect_data, save_dataset, load_dataset
from train import train, compute_residuals
from gp_model import GPResidualModel


# ──────────────────────────────────────────────────────────────────────────────
# Reference trajectories
# ──────────────────────────────────────────────────────────────────────────────

def make_circle(radius: float = 3.0, z: float = 1.0, n: int = 800,
                lap_time: float = 15.0,
                max_accel: float = None) -> np.ndarray:
    """
    Circular trajectory at constant altitude.
    Aligned with MPC/main.py: same radius, lap_time, and starting position.

    Circle is offset so the drone starts at the origin (0, 0),
    matching MPC:  x = radius*cos(ωt) − radius

    The peak centripetal acceleration is:  a_max = radius * omega^2
    If max_accel is set, lap_time is derived automatically so that
    a_max <= max_accel (overrides the lap_time argument).

    Args:
        radius:    orbit radius (m)  [MPC default: 3.0]
        z:         flight altitude (m)
        n:         number of waypoints
        lap_time:  time for one full circle (s)  [MPC default: 15.0]
        max_accel: optional acceleration cap (m/s^2)

    Returns:
        x_ref: (n, 12)  [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
    """
    dt = 0.05
    if max_accel is not None and max_accel > 0:
        # a = radius * omega^2  =>  omega = sqrt(max_accel / radius)
        omega    = np.sqrt(max_accel / radius)
        T_total  = 2 * np.pi / omega
        n        = max(30, int(T_total / dt) + 1)
    else:
        omega = (2 * np.pi) / lap_time
        n     = max(30, int(lap_time / dt) + 1) if n is None else n

    t = np.linspace(0, 2 * np.pi, n)
    x_ref = np.zeros((n, 12))
    x_ref[:, 0] = radius * np.cos(t) - radius   # x  (matching MPC: starts at origin)
    x_ref[:, 1] = radius * np.sin(t)             # y
    x_ref[:, 2] = z                               # z (constant altitude)
    x_ref[:, 3] = -radius * omega * np.sin(t)     # vx
    x_ref[:, 4] =  radius * omega * np.cos(t)     # vy
    # angles, rates remain zero — drone should stay level
    print(f"[make_circle] radius={radius}m | omega={omega:.3f} rad/s | "
          f"peak_accel={radius*omega**2:.3f} m/s^2 | n={n} steps")
    return x_ref


def make_figure8(scale: float = 3.0, z: float = 1.0, n: int = 800,
                 lap_time: float = 15.0,
                 max_accel: float = None) -> np.ndarray:
    """
    Lissajous Curve (Figure-8) trajectory.
    Aligned with MPC/main.py:
        x(t) = scale * sin(ωt)
        y(t) = (scale / 2) * sin(2ωt)

    The y-axis has 2x the frequency, so its peak acceleration is:
        a_max_y = 4 * (scale/2) * omega^2 = 2 * scale * omega^2
    If max_accel is set, lap_time is derived so that a_max_y <= max_accel
    (overrides the lap_time argument).

    Args:
        scale:     spatial half-width of the figure-8 (m)  [MPC default: 3.0]
        z:         flight altitude (m)
        n:         number of waypoints
        lap_time:  time for one full figure-8 (s)  [MPC default: 15.0]
        max_accel: optional acceleration cap (m/s^2)

    Returns:
        x_ref: (n, 12)
    """
    dt = 0.05
    if max_accel is not None and max_accel > 0:
        # Dominant constraint: 2 * scale * omega^2 <= max_accel
        omega    = np.sqrt(max_accel / (2.0 * scale))
        T_total  = 2 * np.pi / omega
        n        = max(30, int(T_total / dt) + 1)
    else:
        omega = (2 * np.pi) / lap_time
        n     = max(30, int(lap_time / dt) + 1) if n is None else n

    t = np.linspace(0, 2 * np.pi, n)
    x_ref = np.zeros((n, 12))
    x_ref[:, 0] = scale * np.sin(t)                         # x  (matches MPC)
    x_ref[:, 1] = (scale / 2.0) * np.sin(2 * t)             # y  (matches MPC: radius/2)
    x_ref[:, 2] = z
    x_ref[:, 3] = scale * omega * np.cos(t)                  # vx
    x_ref[:, 4] = (scale / 2.0) * 2 * omega * np.cos(2 * t) # vy = scale * omega * cos(2t)
    print(f"[make_figure8] scale={scale}m | omega={omega:.3f} rad/s | "
          f"peak_accel_y={2*scale*omega**2:.3f} m/s^2 | n={n} steps")
    return x_ref


def make_wobbly_circle(radius: float = 3.0, z: float = 1.0, n: int = 800,
                       lap_time: float = 15.0,
                       wobble_amp: float = 0.3, wobble_freq: float = 5.0,
                       max_accel: float = None) -> np.ndarray:
    """
    Circle with radial sinusoidal perturbation.

    The radius oscillates:  r(t) = radius + wobble_amp * sin(wobble_freq * omega * t)
    This tests the controller's ability to handle non-smooth references.

    Velocities are computed analytically (derivatives of the perturbed path)
    to avoid noisy reference velocities.

    Args:
        radius:      base orbit radius (m)
        z:           flight altitude (m)
        n:           number of waypoints
        lap_time:    time for one full circle (s)
        wobble_amp:  amplitude of radial perturbation (m)
        wobble_freq: frequency multiplier for the wobble
        max_accel:   optional acceleration cap (m/s^2)

    Returns:
        x_ref: (n, 12)
    """
    dt = 0.05
    if max_accel is not None and max_accel > 0:
        omega   = np.sqrt(max_accel / radius)
        T_total = 2 * np.pi / omega
        n       = max(30, int(T_total / dt) + 1)
    else:
        omega = (2 * np.pi) / lap_time
        n     = max(30, int(lap_time / dt) + 1) if n is None else n

    t  = np.linspace(0, 2 * np.pi, n)
    wf = wobble_freq

    # r(t) = R + A*sin(wf*t)
    r_t  = radius + wobble_amp * np.sin(wf * t)
    dr_t = wobble_amp * wf * np.cos(wf * t)   # dr/dt (wrt parametric t)

    x_ref = np.zeros((n, 12))
    x_ref[:, 0] = r_t * np.cos(t) - radius       # x (offset to start at origin)
    x_ref[:, 1] = r_t * np.sin(t)                 # y
    x_ref[:, 2] = z

    # Analytical velocities: dx/d(time) = dx/dt * dt/d(time) = dx/dt * omega
    # dx/dt = dr_t*cos(t) - r_t*sin(t)
    # dy/dt = dr_t*sin(t) + r_t*cos(t)
    x_ref[:, 3] = omega * (dr_t * np.cos(t) - r_t * np.sin(t))   # vx
    x_ref[:, 4] = omega * (dr_t * np.sin(t) + r_t * np.cos(t))   # vy

    print(f"[make_wobbly_circle] radius={radius}m | wobble_amp={wobble_amp}m | "
          f"wobble_freq={wobble_freq} | n={n} steps")
    return x_ref


def make_wobbly_figure8(scale: float = 3.0, z: float = 1.0, n: int = 800,
                        lap_time: float = 15.0,
                        wobble_amp: float = 0.3, wobble_freq: float = 5.0,
                        max_accel: float = None) -> np.ndarray:
    """
    Figure-8 with sinusoidal perturbation on both axes.

    x(t) = scale * sin(t)      + wobble_amp * sin(wobble_freq * t)
    y(t) = (scale/2) * sin(2t) + wobble_amp * cos(wobble_freq * t)

    Args:
        scale:       spatial half-width (m)
        z:           flight altitude (m)
        n:           number of waypoints
        lap_time:    time for one full figure-8 (s)
        wobble_amp:  amplitude of perturbation (m)
        wobble_freq: frequency multiplier for the wobble
        max_accel:   optional acceleration cap (m/s^2)

    Returns:
        x_ref: (n, 12)
    """
    dt = 0.05
    if max_accel is not None and max_accel > 0:
        omega   = np.sqrt(max_accel / (2.0 * scale))
        T_total = 2 * np.pi / omega
        n       = max(30, int(T_total / dt) + 1)
    else:
        omega = (2 * np.pi) / lap_time
        n     = max(30, int(lap_time / dt) + 1) if n is None else n

    t  = np.linspace(0, 2 * np.pi, n)
    wf = wobble_freq

    x_ref = np.zeros((n, 12))
    x_ref[:, 0] = scale * np.sin(t) + wobble_amp * np.sin(wf * t)
    x_ref[:, 1] = (scale / 2.0) * np.sin(2 * t) + wobble_amp * np.cos(wf * t)
    x_ref[:, 2] = z

    # Analytical velocities
    x_ref[:, 3] = omega * (scale * np.cos(t) + wobble_amp * wf * np.cos(wf * t))
    x_ref[:, 4] = omega * (scale * np.cos(2 * t) + wobble_amp * (-wf) * np.sin(wf * t))

    print(f"[make_wobbly_figure8] scale={scale}m | wobble_amp={wobble_amp}m | "
          f"wobble_freq={wobble_freq} | n={n} steps")
    return x_ref


TRAJECTORIES = {
    "circle":          make_circle,
    "figure8":         make_figure8,
    "wobbly_circle":   make_wobbly_circle,
    "wobbly_figure8":  make_wobbly_figure8,
}


# ──────────────────────────────────────────────────────────────────────────────
# Simple replay buffer for online learning
# ──────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular buffer storing (x, u, x_next) transitions."""

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self.X      = []
        self.U      = []
        self.X_next = []

    def add(self, x: np.ndarray, u: np.ndarray, x_next: np.ndarray) -> None:
        self.X.append(x.copy())
        self.U.append(u.copy())
        self.X_next.append(x_next.copy())
        if len(self.X) > self.max_size:
            self.X.pop(0); self.U.pop(0); self.X_next.pop(0)

    def as_arrays(self) -> tuple:
        return (np.array(self.X, dtype=np.float32),
                np.array(self.U, dtype=np.float32),
                np.array(self.X_next, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.X)


# ──────────────────────────────────────────────────────────────────────────────
# Online retraining
# ──────────────────────────────────────────────────────────────────────────────

def retrain_online(buffer: ReplayBuffer,
                   current_model: ResidualModel,
                   norm: Normalizer,
                   epochs: int = 10,
                   lr: float = 1e-4,
                   dt: float = 0.05) -> ResidualModel:
    """
    Lightweight retraining on the replay buffer.

    Samples the full buffer each call — keep epochs small (5–15).
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    X, U, X_next = buffer.as_arrays()
    params = DRONE_PARAMS

    X_nom = np.zeros_like(X)
    for i in range(len(X)):
        X_nom[i] = f_nominal(X[i], U[i], dt, params)
    R_true = (X_next - X_nom).astype(np.float32)

    X_norm = torch.from_numpy(norm.normalize_x(X))
    U_norm = torch.from_numpy(norm.normalize_u(U))
    R_t    = torch.from_numpy(R_true)

    optimizer = optim.Adam(current_model.parameters(), lr=lr)
    loss_fn   = nn.MSELoss()

    current_model.train()
    for _ in range(epochs):
        pred = current_model(X_norm, U_norm)
        loss = loss_fn(pred, R_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    current_model.eval()
    print(f"  [online] Retrained on {len(buffer)} samples | "
          f"last loss = {loss.item():.6f}")
    return current_model


# ──────────────────────────────────────────────────────────────────────────────
# Main control loop
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    params = DRONE_PARAMS

    # ── Validate model files exist ────────────────────────────────────────────
    model_path = os.path.join(args.models, "residual_model.pt")
    norm_path  = os.path.join(args.models, "normalizer.npz")
    gp_path    = os.path.join(args.models, "gp_residual.pkl")

    # ── Load NN model (always needed as fallback) ─────────────────────────────
    if not os.path.exists(model_path) or not os.path.exists(norm_path):
        print("[main] [!] No trained NN model found.")
        print("[main]    Run:  python dataset.py && python train.py  first.")
        print("[main]    Falling back to nominal-only mode (no residual).\n")
        # Create a dummy zero-output model + normalizer
        X_fake = np.zeros((100, 12), dtype=np.float32)
        U_fake = np.zeros((100,  4), dtype=np.float32)
        U_fake[:, 0] = params["mass"] * params["g"]
        norm      = Normalizer(X_fake, U_fake)
        res_model = ResidualModel()
        res_model.eval()
    else:
        res_model = load_model(model_path)
        norm      = Normalizer.load(norm_path)

    # ── Load GP model (if requested) ──────────────────────────────────────────
    gp_model = None
    if args.residual_type == "gp":
        if os.path.exists(gp_path):
            gp_model = GPResidualModel.load(gp_path)
        else:
            print("[main] [!] No trained GP model found.")
            print("[main]    Run:  python train.py --type gp  first.")
            print("[main]    Falling back to NN residual.\n")
            args.residual_type = "nn"

    # ── Build MPC ─────────────────────────────────────────────────────────────
    mpc = LBMPC(
        nominal_fn      = f_nominal,
        residual_model  = res_model,
        norm            = norm,
        params          = params,
        horizon         = args.horizon,
        dt              = 0.05,
        cem_samples     = args.cem_samples,
        cem_elites      = max(5, args.cem_samples // 10),
        cem_iters       = args.cem_iters,
        residual_type   = args.residual_type,
        gp_model        = gp_model,
        max_velocity    = args.max_vel,
        max_accel       = getattr(args, 'max_accel_constraint', None),
    )

    # ── Build reference trajectory ────────────────────────────────────────────
    traj_fn = TRAJECTORIES.get(args.traj, make_circle)
    # Common kwargs: radius/scale and lap_time (aligned with MPC defaults)
    traj_kwargs = {}
    if args.traj in ("circle", "wobbly_circle"):
        traj_kwargs["radius"] = args.radius
    else:
        traj_kwargs["scale"] = args.radius   # figure8 uses 'scale' param
    traj_kwargs["lap_time"] = args.lap_time
    # Wobbly params (ignored by non-wobbly functions via **kwargs)
    if args.traj.startswith("wobbly_"):
        traj_kwargs["wobble_amp"]  = args.wobble_amp
        traj_kwargs["wobble_freq"] = args.wobble_freq

    if args.max_accel is not None:
        traj_kwargs["max_accel"] = args.max_accel
        x_ref_full = traj_fn(**traj_kwargs)
        args.steps = len(x_ref_full)      # align step count to trajectory length
    else:
        traj_kwargs["n"] = args.steps
        x_ref_full = traj_fn(**traj_kwargs)

    # ── Initialise env ────────────────────────────────────────────────────────
    wind = args.wind if args.wind is not None else None
    env = DroneEnvironment("quodcopter.urdf.xml", gui=args.gui, wind_force=wind)
    env.draw_trajectory(x_ref_full[:, 0], x_ref_full[:, 1], z_height=x_ref_full[0, 2])

    x = env.reset(pos=[x_ref_full[0, 0], x_ref_full[0, 1], x_ref_full[0, 2]])

    # ── Replay buffer (online learning) ───────────────────────────────────────
    buffer     = ReplayBuffer(max_size=10_000)
    ONLINE_MIN = 500    # minimum transitions before first retraining
    RETRAIN_EVERY = 100 # retrain every N steps

    # ── Logging ───────────────────────────────────────────────────────────────
    tracking_errors = []
    logger = TrajectoryLogger(log_dir="logs")

    print(f"\n[main] Starting control loop | traj={args.traj} "
          f"steps={args.steps} horizon={args.horizon} online={args.online}\n")

    # ── Control loop ──────────────────────────────────────────────────────────
    try:
        for step in range(args.steps):
            x_ref_window = x_ref_full[step : step + mpc.horizon]

            # Solve MPC
            t_solve = time.time()
            u = mpc.solve(x, x_ref_window)
            dt_solve = time.time() - t_solve

            # Step simulation
            x_next = env.step(u)

            # Logging
            pos_err = np.linalg.norm(x_next[:3] - x_ref_full[step, :3])
            tracking_errors.append(pos_err)
            logger.log_step(step, x_next, x_ref_full[step], u)

            if step % 20 == 0:
                print(f"  Step {step:>4}/{args.steps} | "
                      f"pos_err={pos_err:.3f}m | "
                      f"z={x_next[2]:.2f}m | "
                      f"u={u.round(3)} | "
                      f"solve={dt_solve*1000:.1f}ms")

            # Online learning
            if args.online:
                buffer.add(x, u, x_next)
                if (len(buffer) >= ONLINE_MIN and
                        step > 0 and step % RETRAIN_EVERY == 0):
                    print(f"\n  [online] Retraining at step {step}...")
                    res_model = retrain_online(buffer, res_model, norm)
                    mpc.update_model(res_model)
                    print()

            x = x_next

    except KeyboardInterrupt:
        print("\n\n  [main] Interrupted by user (Ctrl+C).")
    except Exception as e:
        if "Not connected" in str(e):
            print("\n\n  [main] PyBullet window closed. Saving partial results...")
        else:
            print(f"\n\n  [main] Error: {e}")

    try:
        env.disconnect()
    except Exception:
        pass
    logger.save(prefix=args.traj)

    # ── Summary ───────────────────────────────────────────────────────────────
    errors = np.array(tracking_errors)
    print("\n-- Run Summary ------------------------------------------")
    print(f"  Steps run          : {args.steps}")
    print(f"  Mean tracking error: {errors.mean():.4f} m")
    print(f"  Max  tracking error: {errors.max():.4f} m")
    print(f"  Final error        : {errors[-1]:.4f} m")
    print(f"  Online learning    : {'ON' if args.online else 'OFF'}")
    if args.online:
        print(f"  Buffer size        : {len(buffer)}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LBMPC drone controller")
    parser.add_argument("--traj",        type=str,   default="circle",
                        choices=["circle", "figure8",
                                 "wobbly_circle", "wobbly_figure8"],
                        help="Reference trajectory type")
    parser.add_argument("--steps",       type=int,   default=800,
                        help="Total simulation steps (MPC default: 800 = 40s)")
    parser.add_argument("--horizon",     type=int,   default=20,
                        help="MPC prediction horizon (MPC default: 20)")
    parser.add_argument("--lap-time",    type=float, default=15.0,
                        dest="lap_time",
                        help="Time for one full lap (s, MPC default: 15.0)")
    parser.add_argument("--radius",      type=float, default=3.0,
                        help="Trajectory radius/scale (m, MPC default: 3.0)")
    parser.add_argument("--cem-samples", type=int,   default=300,
                        dest="cem_samples",
                        help="CEM sample count (K)")
    parser.add_argument("--cem-iters",   type=int,   default=4,
                        dest="cem_iters",
                        help="CEM iterations per solve")
    parser.add_argument("--models",      type=str,   default="models",
                        help="Directory containing trained model")
    parser.add_argument("--max-accel",   type=float, default=None,
                        dest="max_accel",
                        help="Max lateral acceleration for trajectory shaping (m/s^2). Overrides --steps.")
    parser.add_argument("--max-vel",     type=float, default=None,
                        dest="max_vel",
                        help="Max velocity constraint for MPC (m/s). Soft penalty in CEM.")
    parser.add_argument("--max-accel-constraint", type=float, default=None,
                        dest="max_accel_constraint",
                        help="Max acceleration constraint for MPC (m/s^2). Soft penalty in CEM.")
    parser.add_argument("--wobble-amp",  type=float, default=0.3,
                        dest="wobble_amp",
                        help="Wobble amplitude for wobbly trajectories (m)")
    parser.add_argument("--wobble-freq", type=float, default=5.0,
                        dest="wobble_freq",
                        help="Wobble frequency multiplier for wobbly trajectories")
    parser.add_argument("--online",      action="store_true",
                        help="Enable online retraining during rollout")
    parser.add_argument("--residual-type", type=str, default="nn",
                        choices=["nn", "gp"],
                        dest="residual_type",
                        help="Residual model type: 'nn' (neural net) or 'gp' (Gaussian Process)")
    parser.add_argument("--wind",        type=float, nargs=3, default=None,
                        metavar=("FX", "FY", "FZ"),
                        help="Wind force [fx fy fz] in Newtons (world frame)")
    parser.add_argument("--gui",         action="store_true", default=True,
                        help="Show PyBullet GUI")
    parser.add_argument("--no-gui",      action="store_false", dest="gui",
                        help="Disable PyBullet GUI")

    args = parser.parse_args()
    main(args)

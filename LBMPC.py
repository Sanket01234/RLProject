# LBMPC.py
"""
Learnable Belief-space Model Predictive Controller (LBMPC).

Core equation:
    x_next = f_nominal(x, u) + f_learned(x, u)

Solver: Cross-Entropy Method (CEM) — gradient-free, parallelisable.

Usage (standalone test):
    python LBMPC.py
"""

import numpy as np
import torch

from nominal import f_nominal
from model import ResidualModel, Normalizer, load_model
from env import DRONE_PARAMS, U_MIN, U_MAX
from gp_model import GPResidualModel


# ──────────────────────────────────────────────────────────────────────────────
# Cost function (weighted Q / R)
# ──────────────────────────────────────────────────────────────────────────────

# State: [x, y, z,  vx, vy, vz,  phi, theta, psi,  p, q, r]
Q_DEFAULT = np.diag([
    10.0, 10.0, 100.0,  # x, y, z   -- z is CRITICAL (altitude loss = crash)
     1.0,  1.0,  20.0,  # vx, vy, vz -- penalise downward velocity heavily
    15.0, 15.0,   1.0,  # phi, theta (limit tilt to preserve vertical thrust), psi
     0.5,  0.5,   0.1,  # p, q, r   -- moderate angular rate damping
]).astype(np.float32)

# Control: [T, tau_roll, tau_pitch, tau_yaw]
R_DEFAULT = np.diag([
    0.01,   # total thrust  -- allow large thrust variance (needed for altitude recovery)
    0.1,    # roll torque   -- moderate (altitude protection via Q_z=100)
    0.1,    # pitch torque  -- moderate
    0.05,   # yaw torque    -- cheapest, penalise less
]).astype(np.float32)


def compute_cost_batch(X_seq: np.ndarray, U_seq: np.ndarray,
                       X_ref: np.ndarray,
                       Q: np.ndarray, R: np.ndarray,
                       u_hover: np.ndarray = None,
                       max_velocity: float = None,
                       max_accel: float = None,
                       dt: float = 0.05) -> np.ndarray:
    """
    Vectorised cost for K candidate trajectories.

    J_k = sum_{t=0}^{N-1} [ (x_t - x_ref_t)^T Q (x_t - x_ref_t)
                           + (u_t - u_ref)^T R (u_t - u_ref)
                           + penalty_vel + penalty_accel ]

    Velocity and acceleration constraints are enforced as soft penalties
    (large weight on squared violation) since CEM is sampling-based
    and cannot handle hard constraints directly.

    Args:
        X_seq: (K, N, 12)  -- K sample trajectories, N horizon steps
        U_seq: (K, N,  4)  -- corresponding control sequences
        X_ref: (N, 12)     -- reference trajectory
        Q:     (12, 12) diagonal cost matrix
        R:     ( 4,  4) diagonal control cost matrix
        u_hover: (4,) hover control vector [T_hover, 0, 0, 0]
        max_velocity: optional max speed constraint (m/s), None = unconstrained
        max_accel:    optional max accel constraint (m/s^2), None = unconstrained
        dt:           timestep (used for acceleration estimation)

    Returns:
        costs: (K,) total cost per sample
    """
    q_diag = np.diag(Q)   # (12,)
    r_diag = np.diag(R)   # (4,)

    err    = X_seq - X_ref[np.newaxis, :, :]           # (K, N, 12)
    s_cost = (err ** 2 * q_diag).sum(axis=(1, 2))      # (K,)

    if u_hover is not None:
        u_err = U_seq - u_hover[np.newaxis, np.newaxis, :]
    else:
        u_err = U_seq

    c_cost = (u_err ** 2 * r_diag).sum(axis=(1, 2))    # (K,)

    total = s_cost + c_cost                             # (K,)

    # -- Soft velocity constraint (barrier penalty) ----------------------------
    if max_velocity is not None:
        # Speed = ||[vx, vy, vz]||  (indices 3, 4, 5)
        vel_mag = np.sqrt(
            X_seq[:, :, 3]**2 + X_seq[:, :, 4]**2 + X_seq[:, :, 5]**2
        )  # (K, N)
        vel_violation = np.maximum(0.0, vel_mag - max_velocity)  # (K, N)
        total += 1000.0 * (vel_violation ** 2).sum(axis=1)       # (K,)

    # -- Soft acceleration constraint (barrier penalty) ------------------------
    if max_accel is not None and X_seq.shape[1] >= 2:
        # Estimate acceleration from velocity differences
        dv = X_seq[:, 1:, 3:6] - X_seq[:, :-1, 3:6]   # (K, N-1, 3)
        accel = dv / dt                                  # (K, N-1, 3)
        accel_mag = np.sqrt((accel ** 2).sum(axis=2))    # (K, N-1)
        accel_violation = np.maximum(0.0, accel_mag - max_accel)  # (K, N-1)
        total += 1000.0 * (accel_violation ** 2).sum(axis=1)     # (K,)

    # -- Altitude floor penalty (ALWAYS active) --------------------------------
    # Prevents the cascading failure: tilt → lift loss → crash.
    # Any predicted z below 0.3m gets a massive penalty.
    z_vals = X_seq[:, :, 2]                               # (K, N)
    z_violation = np.maximum(0.0, 0.3 - z_vals)           # (K, N)
    total += 5000.0 * (z_violation ** 2).sum(axis=1)      # (K,)

    # -- Excessive tilt penalty (ALWAYS active) --------------------------------
    # Tilt > ~25 deg causes significant lift loss (cos(25°) = 0.91).
    # Penalise phi/theta beyond 0.4 rad (23 deg) to keep vertical thrust.
    TILT_LIMIT = 0.4   # radians (~23 degrees)
    phi   = X_seq[:, :, 6]                                # (K, N)
    theta = X_seq[:, :, 7]                                # (K, N)
    phi_violation   = np.maximum(0.0, np.abs(phi) - TILT_LIMIT)    # (K, N)
    theta_violation = np.maximum(0.0, np.abs(theta) - TILT_LIMIT)  # (K, N)
    total += 2000.0 * (phi_violation ** 2 + theta_violation ** 2).sum(axis=1)

    return total


# ──────────────────────────────────────────────────────────────────────────────
# Combined dynamics
# ──────────────────────────────────────────────────────────────────────────────

def predict_next(x: np.ndarray, u: np.ndarray,
                 residual_model: ResidualModel,
                 norm: Normalizer,
                 dt: float, params: dict,
                 residual_type: str = "nn",
                 gp_model: GPResidualModel = None) -> np.ndarray:
    """
    One-step prediction using nominal + learned residual.

    x_next = f_nominal(x, u)  +  f_learned(x, u)

    Supports two residual modes:
        "nn" — full 12-D neural network residual (original)
        "gp" — axis-wise GP correction on velocity only

    Args:
        x:              (12,) current state
        u:              ( 4,) control input (raw, un-clipped)
        residual_model: trained ResidualModel (used when residual_type="nn")
        norm:           fitted Normalizer     (used when residual_type="nn")
        dt:             timestep
        params:         drone params dict
        residual_type:  "nn" or "gp"
        gp_model:       fitted GPResidualModel (used when residual_type="gp")

    Returns:
        x_next: (12,)
    """
    u_clipped = np.clip(u, U_MIN, U_MAX)
    x_nom     = f_nominal(x, u_clipped, dt, params)

    if residual_type == "gp" and gp_model is not None:
        # GP correction: predict residual acceleration from velocity
        v = x[3:5]  # [v_x, v_y]
        delta_ax, delta_ay = gp_model.predict(v)
        x_corrected = x_nom.copy()
        x_corrected[3] += delta_ax * dt   # correct v_x
        x_corrected[4] += delta_ay * dt   # correct v_y
        return x_corrected
    else:
        # Neural network residual (original behaviour)
        x_res = residual_model.predict_numpy(x, u_clipped, norm)
        return x_nom + x_res


def rollout_batch(x0: np.ndarray, U_seq: np.ndarray,
                  residual_model: ResidualModel,
                  norm: Normalizer,
                  dt: float, params: dict,
                  residual_type: str = "nn",
                  gp_model: GPResidualModel = None) -> np.ndarray:
    """
    Roll out K control sequences from state x0 over a horizon N.

    Args:
        x0:    (12,)      initial state
        U_seq: (K, N, 4)  K candidate control sequences
        residual_type: "nn" or "gp"
        gp_model:      fitted GPResidualModel (when residual_type="gp")

    Returns:
        X_seq: (K, N, 12) resulting state trajectories
    """
    K, N, _ = U_seq.shape
    X_seq    = np.zeros((K, N, 12), dtype=np.float64)

    x_batch = np.tile(x0, (K, 1))   # (K, 12)

    for t in range(N):
        u_t = U_seq[:, t, :]        # (K, 4)

        # Batched predict using torch model
        u_clipped = np.clip(u_t, U_MIN, U_MAX)

        # Nominal model (vectorised via loop — fast enough for K≤500, N≤20)
        x_nom_batch = np.zeros((K, 12), dtype=np.float64)
        for k in range(K):
            x_nom_batch[k] = f_nominal(x_batch[k], u_clipped[k], dt, params)

        if residual_type == "gp" and gp_model is not None:
            # GP correction: velocity-based residual acceleration
            V_batch = x_batch[:, 3:5].copy()  # (K, 2)
            delta_a = gp_model.predict_batch(V_batch)  # (K, 2)
            x_batch = x_nom_batch.copy()
            x_batch[:, 3] += delta_a[:, 0] * dt   # correct v_x
            x_batch[:, 4] += delta_a[:, 1] * dt   # correct v_y
        else:
            # Neural network residual (batched torch)
            x_res_batch = residual_model.predict_batch_numpy(
                x_batch.astype(np.float32),
                u_clipped.astype(np.float32),
                norm
            )
            x_batch = x_nom_batch + x_res_batch

        X_seq[:, t, :] = x_batch

    return X_seq


# ──────────────────────────────────────────────────────────────────────────────
# CEM solver
# ──────────────────────────────────────────────────────────────────────────────

class LBMPC:
    """
    Learnable Belief-space MPC using Cross-Entropy Method (CEM).

    Supports two residual model types:
        "nn" — neural network (ResidualModel from model.py)
        "gp" — Gaussian Process (GPResidualModel from gp_model.py)

    Args:
        nominal_fn:      f_nominal function from nominal.py
        residual_model:  trained ResidualModel (used when residual_type="nn")
        norm:            fitted Normalizer (used when residual_type="nn")
        params:          drone parameters dict
        horizon:         prediction horizon N (steps)
        dt:              timestep (seconds)
        cem_samples:     K — number of candidate sequences per iteration
        cem_elites:      M — top-M sequences kept as elite set
        cem_iters:       number of CEM iterations
        Q:               state cost matrix (12×12)
        R:               control cost matrix (4×4)
        residual_type:   "nn" or "gp"
        gp_model:        fitted GPResidualModel (used when residual_type="gp")
        max_velocity:    optional max speed constraint (m/s)
        max_accel:       optional max acceleration constraint (m/s^2)
    """

    def __init__(self,
                 nominal_fn,
                 residual_model: ResidualModel,
                 norm: Normalizer,
                 params: dict,
                 horizon: int    = 20,
                 dt: float       = 0.05,
                 cem_samples: int = 300,
                 cem_elites: int  = 30,
                 cem_iters: int   = 4,
                 Q: np.ndarray   = None,
                 R: np.ndarray   = None,
                 residual_type: str = "nn",
                 gp_model: GPResidualModel = None,
                 max_velocity: float = None,
                 max_accel: float = None):

        self.nominal_fn      = nominal_fn
        self.residual_model  = residual_model
        self.norm            = norm
        self.params          = params
        self.horizon         = horizon
        self.dt              = dt
        self.cem_samples     = cem_samples
        self.cem_elites      = cem_elites
        self.cem_iters       = cem_iters
        self.Q               = Q if Q is not None else Q_DEFAULT
        self.R               = R if R is not None else R_DEFAULT
        self.residual_type   = residual_type
        self.gp_model        = gp_model
        self.max_velocity    = max_velocity
        self.max_accel       = max_accel

        # CEM distribution state -- warm-started across calls
        T_hover = params["mass"] * params["g"]
        self._mu  = np.tile(
            np.array([T_hover, 0.0, 0.0, 0.0]),
            (horizon, 1)
        ).astype(np.float64)                           # (N, 4)
        self._sigma = np.array([
            [2.0, 0.2, 0.2, 0.1]
        ] * horizon, dtype=np.float64)

        constraints = []
        if max_velocity is not None:
            constraints.append(f"max_vel={max_velocity}m/s")
        if max_accel is not None:
            constraints.append(f"max_accel={max_accel}m/s^2")
        c_str = " | ".join(constraints) if constraints else "none"
        print(f"[LBMPC] Initialized | residual_type={residual_type} | "
              f"horizon={horizon} | cem_samples={cem_samples} | "
              f"constraints={c_str}")

    def solve(self, x_current: np.ndarray,
              x_ref: np.ndarray) -> np.ndarray:
        """
        Find the optimal first control action via CEM.

        Args:
            x_current: (12,)  current state
            x_ref:     (N, 12) reference trajectory over the horizon

        Returns:
            u_opt: (4,) optimal control (clipped to U_MIN/U_MAX)
        """
        mu    = self._mu.copy()       # (N, 4)
        sigma = self._sigma.copy()    # (N, 4)

        # Pad reference if shorter than horizon
        if x_ref.shape[0] < self.horizon:
            pad   = np.tile(x_ref[-1], (self.horizon - x_ref.shape[0], 1))
            x_ref = np.vstack([x_ref, pad])
        x_ref = x_ref[:self.horizon]  # (N, 12)

        for _ in range(self.cem_iters):
            # ── Sample K control sequences ────────────────────────────────────
            # Shape: (K, N, 4)
            noise  = np.random.randn(self.cem_samples, self.horizon, 4)
            U_samp = mu[np.newaxis] + sigma[np.newaxis] * noise
            U_samp = np.clip(U_samp, U_MIN, U_MAX)

            # ── Roll out all sequences ────────────────────────────────────────
            X_seq = rollout_batch(
                x_current, U_samp,
                self.residual_model, self.norm,
                self.dt, self.params,
                residual_type=self.residual_type,
                gp_model=self.gp_model,
            )  # (K, N, 12)

            # ── Evaluate costs ────────────────────────────────────────────────
            u_hover = np.array([self.params["mass"] * self.params["g"], 0.0, 0.0, 0.0])
            costs = compute_cost_batch(
                X_seq, U_samp, x_ref, self.Q, self.R, u_hover,
                max_velocity=self.max_velocity,
                max_accel=self.max_accel,
                dt=self.dt,
            )  # (K,)

            # ── Elite set ────────────────────────────────────────────────────
            elite_idx = np.argsort(costs)[:self.cem_elites]
            U_elite   = U_samp[elite_idx]   # (M, N, 4)

            # ── Update distribution ───────────────────────────────────────────
            mu    = U_elite.mean(axis=0)    # (N, 4)
            sigma = U_elite.std(axis=0) + 1e-6

        # Store warm-start for next call (shift by one, repeat last step)
        self._mu    = np.vstack([mu[1:],    mu[[-1]]])
        # Re-inflate sigma to prevent variance collapse
        self._sigma = np.array([[2.0, 0.2, 0.2, 0.1]] * self.horizon, dtype=np.float64)

        u_opt = np.clip(mu[0], U_MIN, U_MAX)
        return u_opt

    def update_model(self, new_model: ResidualModel) -> None:
        """
        Hot-swap the residual model (used during online learning).

        Args:
            new_model: freshly retrained ResidualModel
        """
        self.residual_model = new_model
        print("[LBMPC] Residual model updated.")


# ──────────────────────────────────────────────────────────────────────────────
# Standalone validation  (run: python LBMPC.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from model import ResidualModel, Normalizer

    print("-- Phase 4 Validation ------------------------------------------")

    # ── Check if trained models exist ────────────────────────────────────────
    model_path = "models/residual_model.pt"
    norm_path  = "models/normalizer.npz"

    if os.path.exists(model_path) and os.path.exists(norm_path):
        res_model = load_model(model_path)
        norm      = Normalizer.load(norm_path)
        print("  [Using trained model + normalizer]")
    else:
        print("  [No trained model found — using untrained model for shape test]")
        # Dummy dataset to fit normalizer
        X_fake = np.random.randn(200, 12).astype(np.float32)
        U_fake = np.random.randn(200,  4).astype(np.float32)
        norm      = Normalizer(X_fake, U_fake)
        res_model = ResidualModel()
        res_model.eval()

    params = DRONE_PARAMS
    mpc    = LBMPC(f_nominal, res_model, norm, params, horizon=15)

    # Hover at z=1, target z=1.5
    x0    = np.zeros(12); x0[2] = 1.0
    x_ref = np.tile([0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0], (15, 1)).astype(np.float64)

    print("  Running MPC solve (may take a few seconds)...")
    u_opt = mpc.solve(x0, x_ref)

    print(f"  u_opt        : {u_opt.round(4)}")
    print(f"  Within bounds: {np.all(u_opt >= U_MIN) and np.all(u_opt <= U_MAX)}")

    assert u_opt.shape == (4,),           "Output must be (4,)"
    assert np.all(u_opt >= U_MIN),        "u_opt violates U_MIN"
    assert np.all(u_opt <= U_MAX),        "u_opt violates U_MAX"
    assert u_opt[0] > 0,                  "Thrust must be positive"
    print("  [PASS] Phase 4 PASSED")

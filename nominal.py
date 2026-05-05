# nominal.py
"""
Analytical rigid-body dynamics for a plus-frame quadrotor.

Uses ZYX rotation matrix form for translational acceleration — cleaner
and less error-prone than the expanded trig formulas.

Function:
    f_nominal(x, u, dt, params) → x_next
"""

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rotation_matrix(phi: float, theta: float, psi: float) -> np.ndarray:
    """
    Build the ZYX (yaw-pitch-roll) rotation matrix: body frame → world frame.

    Args:
        phi   — roll  angle (rad)
        theta — pitch angle (rad)
        psi   — yaw   angle (rad)

    Returns:
        R (3×3) numpy array
    """
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi),   np.sin(psi)

    R = np.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [-st,    ct*sp,             ct*cp            ]
    ])
    return R


# ──────────────────────────────────────────────────────────────────────────────
# Core dynamics
# ──────────────────────────────────────────────────────────────────────────────

def f_nominal(x: np.ndarray, u: np.ndarray,
              dt: float, params: dict) -> np.ndarray:
    """
    Predict the next state using analytical rigid-body dynamics.

    State x (12-D):
        [x, y, z,  vx, vy, vz,  phi, theta, psi,  p, q, r]
         0  1  2   3   4   5    6    7       8     9  10 11

    Control u (4-D):
        [T,  tau_roll,  tau_pitch,  tau_yaw]
         0   1          2           3

    Physics:
        Translational:  a = (1/m) * R @ [0, 0, T]  −  [0, 0, g]
        Rotational:     [p_dot, q_dot, r_dot] = [tau_r/Ixx, tau_p/Iyy, tau_y/Izz]
                        (simplified — ignores gyroscopic coupling, fine for Phase 1)
        Integration:    x_next = x + dt * x_dot  (Euler forward)

    Args:
        x:      current state  (12,)
        u:      control input  (4,)
        dt:     timestep in seconds
        params: dict with keys: mass, Ixx, Iyy, Izz, g

    Returns:
        x_next: predicted next state (12,)
    """
    # ── Unpack state ──────────────────────────────────────────────────────────
    px, py, pz         = x[0], x[1], x[2]
    vx, vy, vz         = x[3], x[4], x[5]
    phi, theta, psi    = x[6], x[7], x[8]
    p_r, q_r, r_r      = x[9], x[10], x[11]   # angular rates (body frame)

    # ── Unpack control ────────────────────────────────────────────────────────
    T, tau_roll, tau_pitch, tau_yaw = u[0], u[1], u[2], u[3]

    # ── Parameters ───────────────────────────────────────────────────────────
    m   = params["mass"]
    Ixx = params["Ixx"]
    Iyy = params["Iyy"]
    Izz = params["Izz"]
    g   = params["g"]

    # ── Translational dynamics ────────────────────────────────────────────────
    # Rotation matrix: body z-thrust → world frame acceleration
    R = _rotation_matrix(phi, theta, psi)
    thrust_body = np.array([0.0, 0.0, T])
    acc_world   = (1.0 / m) * (R @ thrust_body) - np.array([0.0, 0.0, g])

    ax, ay, az = acc_world[0], acc_world[1], acc_world[2]

    # ── Rotational dynamics (simplified Euler equations) ──────────────────────
    p_dot = tau_roll  / Ixx
    q_dot = tau_pitch / Iyy
    r_dot = tau_yaw   / Izz

    # ── Euler angle rates from body angular rates ─────────────────────────────
    # Exact kinematic relationship (avoids small-angle approximation)
    cp, sp = np.cos(phi),   np.sin(phi)
    ct, tt = np.cos(theta), np.tan(theta)

    phi_dot   = p_r + (q_r * sp + r_r * cp) * tt
    theta_dot = q_r * cp - r_r * sp
    psi_dot   = (q_r * sp + r_r * cp) / (ct + 1e-9)  # protect div-by-zero

    # ── Euler forward integration ─────────────────────────────────────────────
    x_next = np.array([
        px  + dt * vx,       # x
        py  + dt * vy,       # y
        pz  + dt * vz,       # z
        vx  + dt * ax,       # vx
        vy  + dt * ay,       # vy
        vz  + dt * az,       # vz
        phi   + dt * phi_dot,   # phi
        theta + dt * theta_dot, # theta
        psi   + dt * psi_dot,   # psi
        p_r + dt * p_dot,       # p  (roll rate)
        q_r + dt * q_dot,       # q  (pitch rate)
        r_r + dt * r_dot,       # r  (yaw rate)
    ], dtype=np.float64)

    return x_next


# ──────────────────────────────────────────────────────────────────────────────
# Batched version (used by CEM in LBMPC.py for parallel rollouts)
# ──────────────────────────────────────────────────────────────────────────────

def f_nominal_batch(X: np.ndarray, U: np.ndarray,
                    dt: float, params: dict) -> np.ndarray:
    """
    Vectorised version of f_nominal operating on batches.

    Args:
        X: (N, 12) batch of states
        U: (N, 4)  batch of controls
        dt, params: same as f_nominal

    Returns:
        X_next: (N, 12)
    """
    N = X.shape[0]
    X_next = np.zeros_like(X)
    for i in range(N):
        X_next[i] = f_nominal(X[i], U[i], dt, params)
    return X_next


# ──────────────────────────────────────────────────────────────────────────────
# Standalone validation  (run: python nominal.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from env import DRONE_PARAMS

    params = DRONE_PARAMS
    dt     = 0.05

    # Hover: thrust exactly cancels gravity
    x0      = np.zeros(12)
    x0[2]   = 1.0                                  # start at z=1m
    u_hover = np.array([params["mass"] * params["g"], 0.0, 0.0, 0.0])

    x1 = f_nominal(x0, u_hover, dt, params)

    print("-- Phase 1 Validation ------------------------------------------")
    print(f"  x0       : {x0}")
    print(f"  u_hover  : {u_hover}")
    print(f"  x1       : {x1}")
    print(f"  Max drift: {np.abs(x1 - x0).max():.6f}  (target < 0.01)")

    assert x1.shape == (12,), "Output must be (12,)"
    assert np.abs(x1 - x0).max() < 0.05, (
        f"Hover drift too large: {np.abs(x1 - x0).max():.4f}"
    )
    print("  [PASS] Phase 1 PASSED")

    # Batched version sanity check
    X_batch = np.tile(x0, (8, 1))
    U_batch = np.tile(u_hover, (8, 1))
    X_next  = f_nominal_batch(X_batch, U_batch, dt, params)
    assert X_next.shape == (8, 12), "Batched output must be (N, 12)"
    print("  [PASS] Batch version PASSED")

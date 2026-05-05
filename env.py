# env.py
import numpy as np
import pybullet as p
import pybullet_data
import time

# ──────────────────────────────────────────────────────────────────────────────
# Physical parameters extracted directly from quodcopter.urdf.xml
# ──────────────────────────────────────────────────────────────────────────────
DRONE_PARAMS = {
    "mass":     1.2,    # kg  (base 1.0 + 4 rotors × 0.05)
    "Ixx":      0.02,   # kg·m²  (from URDF inertia)
    "Iyy":      0.02,
    "Izz":      0.04,
    "g":        9.81,
    "arm":      0.15,   # m  (rotor offset from CoM — same for all 4)
    "k_torque": 0.01,   # N·m / N  (reaction torque-to-thrust ratio, tunable)
}

# Control bounds  [T,  tau_roll,  tau_pitch,  tau_yaw]
# Tight torque bounds prevent CEM from sampling extreme inputs that flip the drone.
# Attitude tracking relies on the Q matrix cost weights, not torque authority.
U_MIN = np.array([0.0,  -0.5,  -0.5,  -0.2])
U_MAX = np.array([2.0 * DRONE_PARAMS["mass"] * DRONE_PARAMS["g"],
                   0.5,   0.5,   0.2])


class DroneEnvironment:
    """
    PyBullet drone environment with a clean gym-style interface.

    Control abstraction:
        u = [T, tau_roll, tau_pitch, tau_yaw]
            T         — total thrust   (N)
            tau_roll  — roll  torque   (N·m)
            tau_pitch — pitch torque   (N·m)
            tau_yaw   — yaw   torque   (N·m)

    State vector (12-D numpy array):
        [x, y, z,  vx, vy, vz,  phi, theta, psi,  p, q, r]
         0  1  2   3   4   5    6    7       8     9 10 11

    Frame layout (plus / + frame from URDF):
        Link 0 = motor_front  (+x)
        Link 1 = motor_right  (-y)
        Link 2 = motor_back   (-x)
        Link 3 = motor_left   (+y)
    """

    def __init__(self, urdf_path: str, dt: float = 0.05, gui: bool = True,
                 wind_force: list = None):
        self.dt       = dt
        self.params   = DRONE_PARAMS
        self.mass     = DRONE_PARAMS["mass"]
        self.urdf_path = urdf_path
        self.wind_force = wind_force   # [fx, fy, fz] in N (world frame)

        mode = p.GUI if gui else p.DIRECT
        self.client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")

        self._spawn_drone()
        p.setTimeStep(self.dt)
        wind_str = f" | wind={wind_force}N" if wind_force else ""
        print(f"[DroneEnvironment] Initialized | dt={dt}s | mass={self.mass}kg | gui={gui}{wind_str}")

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, pos: list = None, euler: list = None) -> np.ndarray:
        """
        Reset drone to a given position and orientation.

        Args:
            pos:   [x, y, z]  (default [0, 0, 1.0])
            euler: [roll, pitch, yaw] in radians  (default [0, 0, 0])

        Returns:
            Initial state vector (12,)
        """
        if pos is None:
            pos = [0.0, 0.0, 1.0]
        if euler is None:
            euler = [0.0, 0.0, 0.0]

        quat = p.getQuaternionFromEuler(euler)
        p.resetBasePositionAndOrientation(self.drone_id, pos, quat)
        p.resetBaseVelocity(self.drone_id,
                            linearVelocity=[0, 0, 0],
                            angularVelocity=[0, 0, 0])
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Returns the current drone state as a 12-D numpy vector.

        Order: [x, y, z,  vx, vy, vz,  phi, theta, psi,  p, q, r]
        """
        pos, quat       = p.getBasePositionAndOrientation(self.drone_id)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone_id)
        euler            = p.getEulerFromQuaternion(quat)

        return np.array([
            pos[0], pos[1], pos[2],           # x, y, z
            lin_vel[0], lin_vel[1], lin_vel[2],# vx, vy, vz
            euler[0], euler[1], euler[2],      # phi, theta, psi
            ang_vel[0], ang_vel[1], ang_vel[2] # p, q, r
        ], dtype=np.float64)

    def step(self, u: np.ndarray) -> np.ndarray:
        """
        Apply control u, advance simulation by one timestep, return next state.

        Full internal chain:
            u  →  clip to U_MIN/U_MAX
               →  allocation matrix  →  motor forces
               →  apply_motor_forces()
               →  p.stepSimulation()
               →  get_state()

        Args:
            u: control vector (4,) = [T, tau_roll, tau_pitch, tau_yaw]

        Returns:
            x_next: (12,) next state vector
        """
        u = np.asarray(u, dtype=np.float64)
        u = np.clip(u, U_MIN, U_MAX)

        T, tau_r, tau_p, tau_y = u

        # ── Motor allocation matrix ──────────────────────────────────────
        # Matches MPC/main.py motor mixer EXACTLY (lines 190-193):
        #   F_FRONT = (T_d / 4) - tau_theta + tau_psi
        #   F_RIGHT = (T_d / 4) - tau_phi   - tau_psi
        #   F_BACK  = (T_d / 4) + tau_theta + tau_psi
        #   F_LEFT  = (T_d / 4) + tau_phi   - tau_psi
        #
        # The MPC treats tau as direct force offsets (NOT physical torques
        # that need dividing by arm length). The U_MIN/U_MAX bounds on
        # tau_roll, tau_pitch, tau_yaw are calibrated for this convention.
        #
        # Plus-frame layout (from URDF):
        #   Link 0 = front (+x)    Link 1 = right (-y)
        #   Link 2 = back  (-x)    Link 3 = left  (+y)
        # ─────────────────────────────────────────────────────────────────
        F_front = T / 4.0 - tau_p + tau_y
        F_right = T / 4.0 - tau_r - tau_y
        F_back  = T / 4.0 + tau_p + tau_y
        F_left  = T / 4.0 + tau_r - tau_y

        # Apply per-motor forces (matching MPC/env.py)
        motor_forces = [F_front, F_right, F_back, F_left]
        for i in range(4):
            f_up = max(0.0, motor_forces[i])  # propellers can only push up
            p.applyExternalForce(
                objectUniqueId=self.drone_id,
                linkIndex=i,
                forceObj=[0.0, 0.0, f_up],
                posObj=[0.0, 0.0, 0.0],
                flags=p.LINK_FRAME
            )

        # Apply wind disturbance (world frame, on CoM)
        if self.wind_force is not None:
            p.applyExternalForce(
                objectUniqueId=self.drone_id,
                linkIndex=-1,
                forceObj=self.wind_force,
                posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME
            )

        p.stepSimulation()

        return self.get_state()

    def draw_trajectory(self, x_ref: np.ndarray, y_ref: np.ndarray,
                        z_height: float = 1.0) -> None:
        """
        Draw the target trajectory in the PyBullet GUI as red lines.

        Args:
            x_ref:    array of x-coordinates
            y_ref:    array of y-coordinates
            z_height: constant z level for visualisation
        """
        for i in range(len(x_ref) - 1):
            p.addUserDebugLine(
                lineFromXYZ=[x_ref[i],   y_ref[i],   z_height],
                lineToXYZ  =[x_ref[i+1], y_ref[i+1], z_height],
                lineColorRGB=[1, 0, 0],
                lineWidth=2.0
            )
        print("[DroneEnvironment] Trajectory drawn.")

    def disconnect(self) -> None:
        """Cleanly disconnect from the PyBullet server."""
        p.disconnect(self.client)

    def set_wind(self, force: list) -> None:
        """
        Set or update the wind disturbance force.

        Args:
            force: [fx, fy, fz] in Newtons (world frame), or None to disable
        """
        self.wind_force = force
        if force is not None:
            print(f"[DroneEnvironment] Wind set to {force} N")
        else:
            print("[DroneEnvironment] Wind disabled")

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers (do NOT call these from outside env.py)
    # ──────────────────────────────────────────────────────────────────────────

    def _spawn_drone(self) -> None:
        """Load the drone URDF at the default start pose."""
        start_pos  = [0.0, 0.0, 1.0]
        start_quat = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
        # No URDF_USE_INERTIA_FROM_FILE flag — matches MPC/env.py loading
        self.drone_id = p.loadURDF(
            self.urdf_path, start_pos, start_quat
        )

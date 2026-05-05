import os
import numpy as np
from env import DroneEnvironment

from scipy.optimize import minimize
current_directory = os.path.dirname(os.path.abspath(__file__))
urdf_file_path = os.path.join(current_directory, "quodcopter.urdf.xml")

DT = 0.05
env = DroneEnvironment(urdf_path=urdf_file_path, dt=DT)

# Drone Physical Constants
MASS = 1.2  
G = 9.81

# ==========================================
# PHASE 1: PRE-COMPUTE TARGET TRAJECTORY
# ==========================================
print("Generating Target Trajectory...")

def generate_trajectory(shape, timestamps, radius=3.0, lap_time=15.0):
    """
    Generates X, Y positions and Vx, Vy velocities for a given shape.
    Supported shapes: 'circle', 'figure8'
    """
    omega = (2 * np.pi) / lap_time
    
    if shape == "circle":
        # Position
        X_ref = radius * np.cos(omega * timestamps) - radius 
        Y_ref = radius * np.sin(omega * timestamps)
        # Velocity (Derivatives)
        Vx_ref = -radius * omega * np.sin(omega * timestamps)
        Vy_ref =  radius * omega * np.cos(omega * timestamps)
        
    elif shape == "figure8":
        # Position (Lissajous Curve)
        X_ref = radius * np.sin(omega * timestamps)
        Y_ref = (radius / 2) * np.sin(2 * omega * timestamps)
        # Velocity (Derivatives)
        Vx_ref = radius * omega * np.cos(omega * timestamps)
        Vy_ref = radius * omega * np.cos(2 * omega * timestamps) 
        
    else:
        raise ValueError(f"Unknown shape '{shape}'. Please choose 'circle' or 'figure8'.")
        
    return X_ref, Y_ref, Vx_ref, Vy_ref

# --- Setup Time Array ---
SIMULATION_TIME = 40.0    
HORIZON_TIME = 2.0        
TOTAL_TIME = SIMULATION_TIME + HORIZON_TIME
timestamps = np.arange(0, TOTAL_TIME, DT)

# ----------------------------------------------------
# CHOOSE YOUR TRAJECTORY HERE
# ----------------------------------------------------
TARGET_SHAPE = "figure8"   # Change this to "circle" to instantly swap!

X_ref, Y_ref, Vx_ref, Vy_ref = generate_trajectory(
    shape=TARGET_SHAPE, 
    timestamps=timestamps, 
    radius=3.0, 
    lap_time=15.0
)

# Draw the red path in the PyBullet world
env.draw_trajectory(X_ref, Y_ref, z_height=1.0)

# ==========================================
# PHASE 2: MAIN SIMULATION LOOP
# ==========================================

def calculate_cost(u_guess, current_state, target_X, target_Y, Q, Qv, R, dt,target_Vx=None, target_Vy=None):
    total_cost = 0.0
    
    x = current_state['x']
    y = current_state['y']
    vx = current_state['vx']
    vy = current_state['vy']
    
    N = len(target_X) 
    
    for i in range(N):
        ax = u_guess[i * 2]
        ay = u_guess[(i * 2) + 1]
        
        # 1. Simulate Physics
        x_next = x + (vx * dt)
        y_next = y + (vy * dt)
        vx_next = vx + (ax * dt)
        vy_next = vy + (ay * dt)
        
        # 2. Position Error Penalty
        error_x = target_X[i] - x_next
        error_y = target_Y[i] - y_next
        total_cost += Q * ((error_x ** 2) + (error_y ** 2))
        
        # 3. NEW: Velocity Error Penalty (The "Brakes")
        # error_vx = target_Vx[i] - vx_next
        # error_vy = target_Vy[i] - vy_next
        # total_cost += Qv * ((error_vx ** 2) + (error_vy ** 2))
        
        # 4. Control Effort Penalty
        total_cost += R * ((ax ** 2) + (ay ** 2))
        
        x = x_next
        y = y_next
        vx = vx_next
        vy = vy_next
        
    return total_cost
# --- MPC TUNING PARAMETERS ---
N = 20                  # Prediction Horizon (Look 10 steps ahead)
Q_WEIGHT = 500.0
         # Penalty for drifting off the red line
QV_WEIGHT = 50.0         # Penalty for velocity error (braking)
R_WEIGHT = 20.0          # Penalty for aggressive tilt
MAX_ACCEL = 7.0         # Physical limit: Don't ask for impossible accelerations

# We need an initial guess to kickstart the solver (zeros are fine to start)
# It's size 2*N because we need an ax and ay for every step
u_initial_guess = np.zeros(2 * N) 

# Tell the solver it cannot exceed our MAX_ACCEL limit
# This creates a list of limits: [(-5, 5), (-5, 5), (-5, 5)...]
bounds = [(-MAX_ACCEL, MAX_ACCEL)] * (2 * N)

try:
    print("Taking off! Press Ctrl+C to stop.")
    total_steps = int(SIMULATION_TIME / DT)
    
    for k in range(total_steps):
        state = env.get_state()
        
        # ----------------------------------------------------
        # 1. THE MPC BRAIN
        # ----------------------------------------------------
        # Grab the next N points of the red circle
        target_X_horizon = X_ref[k : k + N]
        target_Y_horizon = Y_ref[k : k + N]
        # target_Vx_horizon = Vx_ref[k : k + N]
        # target_Vy_horizon = Vy_ref[k : k + N]
        
        # Optimization: Run the SciPy Solver!
        result = minimize(
            calculate_cost, 
            u_initial_guess, 
            args=(state, target_X_horizon, target_Y_horizon, Q_WEIGHT, QV_WEIGHT, R_WEIGHT, DT),
            bounds=bounds,
            method='SLSQP' 
        )
        
        # The solver is done! Extract ONLY the very first action (ax0, ay0)
        optimal_u = result.x
        a_xd = optimal_u[0]
        a_yd = optimal_u[1]
        
        u_initial_guess = np.roll(optimal_u, -2) 
        u_initial_guess[-2:] = [0, 0] 
        
        # ----------------------------------------------------
        # 2. THE PROFESSOR's EQUATIONS (Fixed for PyBullet Axes)
        # ----------------------------------------------------
        z_error = 1.0 - state['z']
        vz_error = 0.0 - state['vz'] 
        T_d = (MASS * G) + (10.0 * z_error) + (5.0 * vz_error)
        
        # INVERTED SIGNS: PyBullet's Z-axis is UP, so we flip the textbook math
        theta_d =  (a_xd * np.cos(state['psi']) + a_yd * np.sin(state['psi'])) / G
        phi_d   = -(a_yd * np.cos(state['psi']) - a_xd * np.sin(state['psi'])) / G
        psi_d   = 0.0 
        
        # ----------------------------------------------------
        # 3. ATTITUDE PID (Tuned for 20Hz stability)
        # ----------------------------------------------------
        # Lowered the gains from 5.0 to 1.5 so it doesn't violently jerk
        tau_phi = 1.5 * (phi_d - state['phi']) + 0.5 * (0.0 - state['p'])    
        tau_theta = 1.5 * (theta_d - state['theta']) + 0.5 * (0.0 - state['q']) 
        tau_psi = 0.5 * (psi_d - state['psi']) + 0.1 * (0.0 - state['r'])     
        
        # SAFETY CLAMP: Never let torque overpower the base thrust!
        # Base thrust per motor is roughly 3.0N. We cap torque at 1.5N.
        MAX_TAU = 1.5
        tau_phi = np.clip(tau_phi, -MAX_TAU, MAX_TAU)
        tau_theta = np.clip(tau_theta, -MAX_TAU, MAX_TAU)
        tau_psi = np.clip(tau_psi, -MAX_TAU, MAX_TAU)
        
        # 4. MOTOR MIXER
        F_FRONT = (T_d / 4) - tau_theta + tau_psi
        F_RIGHT = (T_d / 4) - tau_phi   - tau_psi
        F_BACK  = (T_d / 4) + tau_theta + tau_psi
        F_LEFT  = (T_d / 4) + tau_phi   - tau_psi
        
        env.apply_motor_forces(f_fl=F_FRONT, f_fr=F_RIGHT, f_bl=F_BACK, f_br=F_LEFT)
        env.step()

except KeyboardInterrupt:
    print("Simulation stopped by user.")
finally:
    env.disconnect()
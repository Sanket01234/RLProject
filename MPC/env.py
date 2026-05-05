# env.py
import pybullet as p
import pybullet_data
import time

class DroneEnvironment:
    def __init__(self, urdf_path, dt=0.05):
        self.dt = dt
        self.client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        
        start_position = [0, 0, 1.0] 
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.drone_id = p.loadURDF(urdf_path, start_position, start_orientation)
        p.setTimeStep(self.dt)
        print("Environment Initialized!")

    def get_state(self):
        pos, quat = p.getBasePositionAndOrientation(self.drone_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.drone_id)
        euler = p.getEulerFromQuaternion(quat)
        
        state = {
            "x": pos[0], "y": pos[1], "z": pos[2],
            "vx": linear_vel[0], "vy": linear_vel[1], "vz": linear_vel[2],
            "phi": euler[0], "theta": euler[1], "psi": euler[2],
            "p": angular_vel[0], "q": angular_vel[1], "r": angular_vel[2]
        }
        return state

    def step(self):
        p.stepSimulation()
        time.sleep(self.dt)
    def apply_motor_forces(self, f_fl, f_fr, f_bl, f_br):
        """
        Applies calculated upward thrust to each specific motor link.
        Link Indices based on our URDF: 0=FL, 1=FR, 2=BL, 3=BR
        """
        forces = [f_fl, f_fr, f_bl, f_br]
        
        for i in range(4):
            # We must clip negative forces because propellers can only push up!
            f_up = max(0.0, forces[i]) 
            
            p.applyExternalForce(
                objectUniqueId=self.drone_id,
                linkIndex=i,
                forceObj=[0, 0, f_up], # Force pointing up in the Z direction
                posObj=[0, 0, 0],      # Apply right at the center of the rotor
                flags=p.LINK_FRAME     # Apply relative to the drone's current tilt!
            )
    def draw_trajectory(self, x_ref, y_ref, z_height=1.0):
        """
        Draws the target trajectory in the PyBullet environment using red lines.
        """
        # We draw a line from point i to point i+1
        for i in range(len(x_ref) - 1):
            p.addUserDebugLine(
                lineFromXYZ=[x_ref[i], y_ref[i], z_height],
                lineToXYZ=[x_ref[i+1], y_ref[i+1], z_height],
                lineColorRGB=[1, 0, 0], # Red
                lineWidth=2.0
            )
        print("Trajectory drawn in environment!")
    def disconnect(self):
        p.disconnect()
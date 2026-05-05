import os
import csv
import time
import numpy as np

class TrajectoryLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.data = []
        
    def log_step(self, step: int, x_true: np.ndarray, x_ref: np.ndarray, u: np.ndarray):
        pos_err = np.linalg.norm(x_true[:3] - x_ref[:3])
        row = {
            "step": step,
            "x_true": x_true[0], "y_true": x_true[1], "z_true": x_true[2],
            "x_ref": x_ref[0], "y_ref": x_ref[1], "z_ref": x_ref[2],
            "pos_err": pos_err,
            "u_T": u[0], "u_roll": u[1], "u_pitch": u[2], "u_yaw": u[3]
        }
        self.data.append(row)
        
    def save(self, prefix="run"):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.csv"
        filepath = os.path.join(self.log_dir, filename)
        
        if not self.data:
            print("[Logger] No data to save.")
            return filepath
            
        fieldnames = self.data[0].keys()
        with open(filepath, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.data)
            
        print(f"  [Logger] Saved trajectory log to {filepath}")
        return filepath

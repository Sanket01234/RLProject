# LBMPC Project -- Complete Command Reference

> All commands run from `c:\Users\sanke\Desktop\dataset\`

---

## Step 1: Data Collection & Model Training

These must be run **once** before any experiments. Re-run if you change `env.py` physics.

```bash
# Collect training data (random exploration, 20 episodes)
python dataset.py --episodes 20

# Train Neural Network residual model
python train.py

# Train Gaussian Process residual model
python train.py --type gp
```

After this you'll have:
- `data/transitions.npz` -- training data
- `models/residual_model.pt` -- NN model
- `models/normalizer.npz` -- input normalizer
- `models/gp_residual.pkl` -- GP model

---

## Step 2: Run All Experiments (Automated)

```bash
# Run ALL experiments (baseline + speed + horizon + constraint + wobbly + wind)
# WARNING: This takes ~2-3 hours due to CEM solve per step
python experiments.py --experiment all

# Or run specific experiment groups:
python experiments.py --experiment baseline      # NN vs GP on circle & figure-8 (4 runs)
python experiments.py --experiment speed         # vary lap_time (4 runs)
python experiments.py --experiment horizon       # vary prediction horizon (4 runs)
python experiments.py --experiment constraint    # vary velocity constraints (5 runs)
python experiments.py --experiment wobbly        # wobbly trajectories (4 runs)
python experiments.py --experiment wind          # wind disturbance (6 runs)

# List available experiments
python experiments.py --list
```

Results are saved to `results/experiments.json`.

---

## Step 3: Generate Plots

```bash
# Plot comparison bar charts from experiment results
python plotting.py --results results/experiments.json

# Plot a single trajectory run (XY path + error + controls)
python plotting.py --log logs/circle_20260505_194100.csv

# Overlay multiple trajectories for visual comparison
python plotting.py --logs logs/circle_*.csv --labels "Run1" "Run2"
```

Plots are saved to `plots/`.

---

## Individual Run Commands (Manual Testing)

### Baseline Runs

```bash
# MPC baseline (classical -- from MPC/ folder, separate code)
cd MPC && python main.py

# LBMPC with Neural Network on circle (matches MPC defaults: radius=3, lap=15s, 800 steps)
python main.py --no-gui --traj circle

# LBMPC with Gaussian Process on circle
python main.py --no-gui --traj circle --residual-type gp

# LBMPC (NN) on figure-8
python main.py --no-gui --traj figure8

# LBMPC (GP) on figure-8
python main.py --no-gui --traj figure8 --residual-type gp
```

### Velocity & Acceleration Constraints

```bash
# Max velocity = 2 m/s (soft penalty in CEM)
python main.py --no-gui --max-vel 2.0

# Max velocity = 5 m/s
python main.py --no-gui --max-vel 5.0

# Max acceleration = 3 m/s^2 (soft penalty in CEM)
python main.py --no-gui --max-accel-constraint 3.0

# Both constraints at once
python main.py --no-gui --max-vel 3.0 --max-accel-constraint 5.0
```

### Trajectory Shaping (via --max-accel)

```bash
# Slow the trajectory itself so peak accel doesn't exceed 0.3 m/s^2
# (this changes the lap time / step count, NOT the MPC constraint)
python main.py --no-gui --max-accel 0.3

# Faster trajectory (peak accel up to 1.0 m/s^2)
python main.py --no-gui --max-accel 1.0
```

### Wobbly Trajectories

```bash
# Wobbly circle (default: amp=0.3m, freq=5.0)
python main.py --no-gui --traj wobbly_circle

# Custom wobble parameters
python main.py --no-gui --traj wobbly_circle --wobble-amp 0.5 --wobble-freq 3.0

# Wobbly figure-8
python main.py --no-gui --traj wobbly_figure8

# Wobbly + GP
python main.py --no-gui --traj wobbly_circle --residual-type gp
```

### Wind Disturbance

```bash
# Moderate wind (0.5N in x, 0.3N in y)
python main.py --no-gui --wind 0.5 0.3 0.0

# Strong wind (1.0N in x)
python main.py --no-gui --wind 1.0 0.0 0.0

# Wind + GP
python main.py --no-gui --wind 0.5 0.3 0.0 --residual-type gp
```

### Combined Scenarios

```bash
# GP + wobbly + velocity constraint + wind
python main.py --no-gui --residual-type gp --traj wobbly_circle --max-vel 3.0 --wind 0.2 0.0 0.0

# NN + figure-8 + acceleration constraint + wind
python main.py --no-gui --traj figure8 --max-accel-constraint 5.0 --wind 0.3 0.2 0.0

# Online learning (NN retrained during tracking)
python main.py --no-gui --online --steps 1000
```

### Other Parameters

```bash
# Change CEM parameters (more samples = better but slower)
python main.py --no-gui --cem-samples 500 --cem-iters 6

# Custom radius and lap time
python main.py --no-gui --radius 5.0 --lap-time 20.0

# Short test run (fewer steps)
python main.py --no-gui --steps 50

# With GUI visualization
python main.py --gui
```

---

## Recommended Full Pipeline (Copy-Paste)

```bash
# 1. Collect data and train both models
python dataset.py --episodes 20
python train.py
python train.py --type gp

# 2. Run key experiments
python experiments.py --experiment baseline --output results/baseline.json
python experiments.py --experiment wobbly --output results/wobbly.json
python experiments.py --experiment wind --output results/wind.json
python experiments.py --experiment constraint --output results/constraint.json

# 3. Generate all plots
python plotting.py --results results/baseline.json
python plotting.py --results results/wobbly.json
python plotting.py --results results/wind.json
python plotting.py --results results/constraint.json
```

---

## File Overview

| File | Purpose |
|------|---------|
| `dataset.py` | Collect training data |
| `train.py` | Train NN or GP residual model |
| `main.py` | Run LBMPC controller (single run) |
| `experiments.py` | Automated experiment runner |
| `plotting.py` | Generate analysis plots |
| `env.py` | PyBullet drone environment |
| `LBMPC.py` | CEM-based MPC with learned residuals |
| `model.py` | Neural network residual model |
| `gp_model.py` | Gaussian Process residual model |
| `nominal.py` | Analytical drone dynamics |
| `logger.py` | CSV trajectory logging |
| `MPC/` | Classical MPC baseline (read-only) |

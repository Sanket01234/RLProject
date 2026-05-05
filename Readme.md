# Project README

This repository contains code and assets for experimenting with a quadcopter-like system: training dynamics models and running model-based controllers. Below is a beginner-friendly explanation of what each file does and how they fit together.

## Quick project overview
- Purpose: collect, train, and use dynamics models of a vehicle (quadcopter) and test controllers (MPC/Learning-based MPC).
- Typical workflow: collect or load data (`dataset.py`) → train a model (`train.py`, `model.py`) → evaluate with a controller or simulation (`main.py`, `LBMPC.py`, `env.py`).

## Files (detailed)
- **[dataset.py](dataset.py)**: Data loading and preprocessing utilities.
  - Purpose: centralizes code to read recorded trajectories (states, actions, next-states), format them into training batches, and perform simple preprocessing (normalization, train/validation split).
  - Typical contents: dataset class or functions that return PyTorch/Numpy arrays or iterators used by `train.py`.
  - Beginner tip: If you want to build a dataset from raw logs, add a function here that converts log files into the expected arrays (state, action, next_state).

- **[env.py](env.py)**: Simulation environment wrapper.
  - Purpose: defines the environment where the agent/controller runs. It likely provides `reset()`, `step(action)`, and optionally `render()` functions similar to OpenAI Gym environments.
  - How it's used: `main.py` or controller scripts create an instance of this environment to run rollouts and evaluate performance.
  - Beginner tip: Inspect `env.py` to find the observation and action shapes; controllers and models must agree with these shapes.

- **[implementation_plan.md](implementation_plan.md)**: Project plan and notes.
  - Purpose: contains the high-level plan, tasks, and ideas for implementing features or experiments in this repository.
  - Beginner tip: Read this file to understand the author's intended roadmap and any assumptions about data or experiments.

- **[LBMPC.py](LBMPC.py)**: Learning-Based Model Predictive Control implementation.
  - Purpose: implements a controller class or functions for MPC that incorporate learned model predictions (the "learning-based" part). It probably predicts future states using a learned model and solves an optimization for control inputs.
  - Typical contents: code to construct cost terms, call a solver (CVX/OSQP/CasADi or custom gradient-based planning), and return the next control action.
  - Beginner tip: This file is the place to tune horizons, cost weights, and constraints. If you want a simple baseline, compare this to a nominal controller from `nominal.py`.

- **[logger.py](logger.py)**: Logging utilities.
  - Purpose: sets up logging for experiments, saving scalar metrics, text logs, or files. It centralizes how results and debug messages are written to disk.
  - Typical contents: helper functions or a small `Logger` class that configures Python `logging`, writes CSV/JSON records, or saves models and checkpoints.
  - Beginner tip: Use these helpers in `train.py` and `main.py` so experiment outputs are consistent and reproducible.

- **[main.py](main.py)**: Top-level experiment / evaluation script.
  - Purpose: orchestrates running simulations or experiments. It typically wires together the environment (`env.py`), controller (`LBMPC.py` or `nominal.py`), the model (`model.py`), and logging (`logger.py`).
  - How to use: run this script to test controllers in simulation. It may accept CLI args (check file header) to choose runs.
  - Beginner tip: If `main.py` accepts command-line options, run `python main.py --help` to see available flags.

- **[model.py](model.py)**: Neural-network or regression model definitions used to predict dynamics.
  - Purpose: contains model architectures and helper functions to make predictions (e.g., next-state given current-state & action), and save/load weights.
  - How it's used: instantianted by `train.py` to learn dynamics, then used by `LBMPC.py` or `main.py` to predict transitions.
  - Beginner tip: Check for a `forward()` function signature and expected input shapes (batch, state_dim + action_dim).

- **[nominal.py](nominal.py)**: Nominal (analytic) dynamics or baseline controller.
  - Purpose: provides a hand-derived or physics-based baseline model and/or controller. This is useful as a comparison to the learned model.
  - Typical contents: deterministic dynamics equations, linearized models, or a simple PID controller.
  - Beginner tip: Use this to sanity-check whether your learned model/controller outperforms a simple analytic baseline.

- **[quodcopter.urdf.xml](quodcopter.urdf.xml)**: URDF robot description.
  - Purpose: a robot description file (Unified Robot Description Format) describing links, joints, and inertial properties of the quadcopter.
  - How it's used: if `env.py` or any simulation utilities use a physics engine (e.g., PyBullet, ROS tools), they may load this URDF to create the simulated robot.
  - Beginner tip: URDF files are XML and describe the geometry and physical parameters—edit with care.

- **[train.py](train.py)**: Model training script.
  - Purpose: uses `dataset.py` and `model.py` to train the dynamics model. It handles batching, optimization loop, checkpoints, and basic evaluation on a validation split.
  - How to use: typically run `python train.py` (or with config/args). After training, the script should save a model file that `main.py` or `LBMPC.py` can load.
  - Beginner tip: Look for a saved weights filename or `checkpoints/` directory to find or resume training from saved models.

## Typical workflows (beginner-friendly)
- Train a model:

```bash
# (prefer to use your existing environment; avoid reinstalling torch unless needed)
python train.py
```

- Run a controller / simulation:

```bash
python main.py
```

If `main.py` or `train.py` accept flags, inspect them with `python train.py --help` or open the files to see the parameter names.

## Dependencies & environment
- The codebase likely uses Python (the environment you mentioned is Python 3.12.10). Common packages needed:
  - `torch` (PyTorch) — for models and training
  - `numpy`, `scipy` — numeric utilities
  - `matplotlib` — plotting results
  - `gym` or custom env dependencies — if `env.py` follows Gym style
  - an optimization solver (optional) if `LBMPC.py` calls one (e.g., `osqp`, `cvxpy`, `casadi`).

Install suggestion (adjust to your environment):

```bash
pip install numpy scipy matplotlib pyyaml pandas
# install torch separately according to your CUDA/Python combination
# e.g., follow https://pytorch.org/get-started/locally/
```

Note: you indicated a preference to use an existing Python 3.12.10 environment with CUDA-enabled PyTorch — that is recommended to avoid re-installing heavy packages.

## Where to look first (recommended for newcomers)
1. Open `dataset.py` to see how data is loaded.
2. Open `model.py` to see the model input/output shapes.
3. Run `python train.py` to train a model on the dataset (or inspect it first for required CLI args).
4. Run `python main.py` to run experiments and visualize results.

## If something is missing
- If there is no `requirements.txt` or saved models, create them or inspect the top of `train.py`/`main.py` for imports to deduce dependencies.
- If `env.py` depends on external simulators (PyBullet/ROS), ensure those are installed and compatible with Windows.

---

If you'd like, I can:
- generate a `requirements.txt` inferred from the imports used in the repo,
- open each file and add short docstrings or expand this README with exact function names and example calls.

Tell me which follow-up you'd prefer.

## New: `MPC` folder (added)
- **Location:** `MPC/`
- **Purpose:** A lightweight, classical MPC implementation that runs a 2D position MPC in PyBullet using SciPy's `minimize` solver. It plans accelerations in X/Y over a short horizon, converts desired accelerations into desired attitude setpoints, then uses simple PID-style attitude controllers and a motor mixer to apply forces to the simulated quadcopter.
- **Key files:**
  - `MPC/main.py`: top-level script that generates a reference trajectory (circle/figure-8), formulates an MPC optimization (optimize ax, ay over horizon), calls SciPy `minimize` (SLSQP), extracts the first action, converts accel to thrust/attitude targets, and applies motor commands to the `DroneEnvironment`.
  - `MPC/env.py`: environment wrapper used by this MPC (separate from the root `env.py` in the repo). It contains the PyBullet setup and helper methods such as `get_state()`, `apply_motor_forces()`, `step()`, and `draw_trajectory()`.
  - `MPC/quodcopter.urdf.xml`: robot description copy used by the MPC environment.

## Comparison: `LBMPC.py` (learning-based CEM) vs `MPC/main.py` (SciPy-based classical MPC)
This section summarizes differences, trade-offs, and when to use each approach in beginner terms.

- **Type of controller**:
  - `LBMPC.py`: Learning-enhanced MPC — it combines an analytical nominal dynamics model (`nominal.py`) with a learned residual (`model.py`), and solves for control using Cross-Entropy Method (CEM), a sampling-based optimizer.
  - `MPC/main.py` (in `MPC/`): Classical MPC formulated as a constrained optimization over acceleration inputs and solved with SciPy's `minimize` (SLSQP). Uses analytical/integrator approximations in the cost simulation.

- **Dynamics modeling**:
  - `LBMPC.py`: Uses a physics-based nominal model plus a trainable residual model (neural network) to better capture unmodelled dynamics and biases. This improves long-term prediction accuracy if the residual is trained on representative data.
  - `MPC/main.py`: Uses a simple kinematic/accelerations rollout inside the cost function (x_next = x + vx*dt; vx_next = vx + ax*dt). Simpler, faster to evaluate, but less accurate for full 12-DOF quad dynamics.

- **Optimizer / Solver**:
  - `LBMPC.py`: CEM — samples many candidate control sequences and keeps elites. Pros: robust to non-convex problems, easy to implement, parallelisable. Cons: can be sample-inefficient and slower per decision if not vectorised.
  - `MPC/main.py`: SciPy `minimize` (SLSQP) — gradient-based constrained optimization. Pros: fewer function evaluations for smooth convex problems; deterministic convergence properties. Cons: needs a good initial guess, may get stuck in local minima, and requires gradients (or approximations) to be meaningful.

- **Control dimensionality & fidelity**:
  - `LBMPC.py`: Operates on full 4-D control (thrust + roll/pitch/yaw torques) and full 12-D state — higher fidelity, suitable for aggressive or attitude-sensitive maneuvers.
  - `MPC/main.py`: Optimises 2-D accelerations (ax, ay) at a translational level, then maps them to attitude commands; simpler and typically faster but may not capture fast attitude dynamics accurately.

- **Use cases**:
  - `LBMPC.py`: When you have (or can train) a residual model and need high-fidelity control and better prediction over longer horizons; valuable if you want to close the sim-to-real gap with learned corrections.
  - `MPC/main.py`: When you want a simple, easy-to-understand MPC baseline that runs quickly in simulation (good for prototyping trajectories and testing high-level planning logic).

- **Implementation complexity**:
  - `LBMPC.py`: Higher complexity — requires `model.py` training, normalization (`Normalizer`), and managing a sample-based optimizer. More code but more flexible.
  - `MPC/main.py`: Lower complexity — a single script that formulates and solves a numeric optimization per step and uses straightforward PID mixing.

- **Performance considerations**:
  - `LBMPC.py`: Can be slower each control step because of sampling and neural net inference, but batched rollouts mitigate this. Requires GPU/CPU trade-offs for the learned model.
  - `MPC/main.py`: Likely faster per-step in simple scenarios (low-dim controls) and runs entirely on CPU with SciPy; good for real-time-friendly prototyping.

## Practical suggestions / Beginner guidance
- If you want to **compare quantitatively**, try these steps:
  1. Pick a fixed reference trajectory (e.g., the `figure8` in `MPC/main.py`).
  2. Run `MPC/main.py` to collect a baseline log: position error, control effort, and any crash/failure flags.
  3. Run `main.py` at the repo root wired to use `LBMPC` (or run `LBMPC.py` standalone) on the same trajectory and log the same metrics.
  4. Compare mean/median tracking error, max error, and total control effort. Also inspect qualitative behaviour (oscillations, overshoot).

- If you'd like, I can:
  - add a short wrapper script that runs both controllers on the same trajectory and produces a CSV of comparison metrics, or
  - generate a `requirements.txt` inferred from imports and add small docstrings to `MPC/env.py` and `LBMPC.py` for clarity.

---

If you want me to implement the automated comparison runner now, tell me to proceed and I will add the script and run/read logs where possible.
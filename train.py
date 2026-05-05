# train.py
"""
Train the residual dynamics model on collected transition data.

Usage:
    python train.py                          # train NN residual (default)
    python train.py --type gp                # train GP residual
    python train.py --data data/transitions.npz --epochs 200 --lr 3e-4
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from nominal import f_nominal_batch
from model import ResidualModel, Normalizer, save_model
from env import DRONE_PARAMS
from gp_model import GPResidualModel


# ------------------------------------------------------------------------------
# Residual computation
# ------------------------------------------------------------------------------

def compute_residuals(X: np.ndarray, U: np.ndarray, X_next: np.ndarray,
                      dt: float, params: dict) -> np.ndarray:
    """
    Compute true residuals: residual = x_true_next − f_nominal(x, u)

    Args:
        X:      (N, 12) current states
        U:      (N,  4) controls
        X_next: (N, 12) true next states (from sim)
        dt:     timestep
        params: drone params dict

    Returns:
        R: (N, 12) residuals
    """
    X_nom = f_nominal_batch(X, U, dt, params)   # (N, 12)
    return X_next - X_nom                        # (N, 12) - what nominal missed


# ------------------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------------------

def train(data_path: str  = "data/transitions.npz",
          model_dir: str  = "models",
          epochs: int     = 150,
          batch_size: int = 256,
          lr: float       = 3e-4,
          val_split: float = 0.1,
          dt: float       = 0.05) -> ResidualModel:
    """
    Full training pipeline:
        1. Load dataset
        2. Compute residuals against nominal model
        3. Fit Normalizer
        4. Train ResidualModel
        5. Save model + normalizer

    Returns:
        Trained ResidualModel (eval mode)
    """
    os.makedirs(model_dir, exist_ok=True)
    params = DRONE_PARAMS

    # -- 1. Load data ----------------------------------------------------------
    print("[train] Loading dataset...")
    data   = np.load(data_path)
    X      = data["X"].astype(np.float32)        # (N, 12)
    U      = data["U"].astype(np.float32)        # (N,  4)
    X_next = data["X_next"].astype(np.float32)   # (N, 12)
    N      = X.shape[0]
    print(f"[train] Dataset: {N} transitions")

    # -- 2. Compute residuals --------------------------------------------------
    print("[train] Computing residuals against nominal model...")
    R = compute_residuals(X, U, X_next, dt, params).astype(np.float32)
    print(f"[train] Residual stats - mean: {R.mean(axis=0).round(4)}")
    print(f"[train]               - std : {R.std(axis=0).round(4)}")

    # -- 3. Fit Normalizer on training data ----------------------------------─
    norm = Normalizer(X, U)
    norm.save(os.path.join(model_dir, "normalizer.npz"))

    # -- 4. Build tensors + DataLoader ----------------------------------------
    X_norm = torch.from_numpy(norm.normalize_x(X))   # (N, 12)
    U_norm = torch.from_numpy(norm.normalize_u(U))   # (N,  4)
    R_t    = torch.from_numpy(R)                      # (N, 12)

    full_ds   = TensorDataset(X_norm, U_norm, R_t)
    val_size  = max(1, int(N * val_split))
    train_size = N - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # -- 5. Model + optimiser --------------------------------------------------
    model     = ResidualModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, verbose=True
    )
    loss_fn   = nn.MSELoss()

    best_val_loss  = float("inf")
    best_ckpt_path = os.path.join(model_dir, "residual_model.pt")

    print(f"\n[train] Starting training | epochs={epochs} "
          f"batch={batch_size} lr={lr}")
    print(f"[train] Train samples: {train_size} | Val samples: {val_size}\n")

    # -- 6. Training loop ------------------------------------------------------
    for epoch in range(1, epochs + 1):
        # ─ train ─
        model.train()
        train_loss = 0.0
        for x_b, u_b, r_b in train_loader:
            pred = model(x_b, u_b)
            loss = loss_fn(pred, r_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x_b)
        train_loss /= train_size

        # ─ validate ─
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_b, u_b, r_b in val_loader:
                pred = model(x_b, u_b)
                val_loss += loss_fn(pred, r_b).item() * len(x_b)
        val_loss /= val_size

        scheduler.step(val_loss)

        # ─ log & checkpoint ─
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{epochs}  "
                  f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, best_ckpt_path)

    # -- 7. Final report ------------------------------------------------------─
    print(f"\n[train] Training complete | best val_loss = {best_val_loss:.6f}")
    print(f"[train] Model saved -> {best_ckpt_path}")

    # Reload best checkpoint
    model.load_state_dict(torch.load(best_ckpt_path, weights_only=True))
    model.eval()
    return model


# ------------------------------------------------------------------------------
# Post-training validation
# ------------------------------------------------------------------------------

def validate_model(model: ResidualModel, norm: Normalizer,
                   data_path: str = "data/transitions.npz",
                   dt: float = 0.05) -> None:
    """
    Check that the trained model meaningfully reduces prediction error
    compared to using the nominal model alone.
    """
    from model import load_model
    params = DRONE_PARAMS

    data   = np.load(data_path)
    X      = data["X"].astype(np.float32)
    U      = data["U"].astype(np.float32)
    X_next = data["X_next"].astype(np.float32)

    # Nominal-only prediction error
    X_nom = f_nominal_batch(X, U, dt, params).astype(np.float32)
    nom_mse = float(np.mean((X_nom - X_next) ** 2))

    # Learned model residual prediction
    R_true = (X_next - X_nom)
    R_pred = model.predict_batch_numpy(X, U, norm)
    X_pred = X_nom + R_pred

    full_mse = float(np.mean((X_pred - X_next) ** 2))

    print("\n-- Phase 3 Validation ----------------------------------")
    print(f"  Nominal-only MSE       : {nom_mse:.6f}")
    print(f"  Nominal + residual MSE : {full_mse:.6f}")
    improvement = (1.0 - full_mse / nom_mse) * 100.0
    print(f"  Improvement            : {improvement:.1f}%")
    assert full_mse < nom_mse, (
        "Residual model made things WORSE - check normalizer or data quality"
    )
    print("  PASS Phase 3 PASSED")


# ------------------------------------------------------------------------------
# GP training pipeline
# ------------------------------------------------------------------------------

def train_gp(data_path: str = "data/transitions.npz",
             model_dir: str = "models",
             dt: float = 0.05) -> GPResidualModel:
    """
    Train axis-wise GP residual model from collected transition data.

    Pipeline (per project slides):
        1. Load transitions (X, U, X_next)
        2. Compute nominal predictions
        3. Extract residual accelerations:
              residual_a = (v_next_true - v_next_nominal) / dt
        4. Build training pairs: inputs=[v_x, v_y], outputs=residual_a
        5. Fit GP_x and GP_y
        6. Save model

    Args:
        data_path: path to transitions.npz
        model_dir: output directory for saved GP model
        dt:        simulation timestep

    Returns:
        Fitted GPResidualModel
    """
    os.makedirs(model_dir, exist_ok=True)
    params = DRONE_PARAMS

    # -- 1. Load data ----------------------------------------------------------
    print("[train_gp] Loading dataset...")
    data   = np.load(data_path)
    X      = data["X"].astype(np.float64)        # (N, 12)
    U      = data["U"].astype(np.float64)        # (N,  4)
    X_next = data["X_next"].astype(np.float64)   # (N, 12)
    N      = X.shape[0]
    print(f"[train_gp] Dataset: {N} transitions")

    # -- 2. Compute nominal predictions ----------------------------------------
    print("[train_gp] Computing nominal predictions...")
    X_nom = f_nominal_batch(X, U, dt, params)    # (N, 12)

    # -- 3. Extract residual accelerations -------------------------------------
    # residual_accel = (v_next_true - v_next_nominal) / dt
    # v_x is state index 3, v_y is state index 4
    residual_ax = (X_next[:, 3] - X_nom[:, 3]) / dt   # (N,)
    residual_ay = (X_next[:, 4] - X_nom[:, 4]) / dt   # (N,)

    print(f"[train_gp] Residual accel stats:")
    print(f"  d_a_x: mean={residual_ax.mean():.4f}, std={residual_ax.std():.4f}")
    print(f"  d_a_y: mean={residual_ay.mean():.4f}, std={residual_ay.std():.4f}")

    # -- 4. Build training pairs: inputs = [v_x, v_y] -------------------------
    V = X[:, 3:5].copy()   # (N, 2)  velocities
    print(f"[train_gp] Velocity range: "
          f"v_x=[{V[:,0].min():.2f}, {V[:,0].max():.2f}], "
          f"v_y=[{V[:,1].min():.2f}, {V[:,1].max():.2f}]")

    # -- 5. Fit GPs ------------------------------------------------------------
    gp = GPResidualModel()
    gp.fit(V, residual_ax, residual_ay)

    # -- 6. Save ---------------------------------------------------------------
    gp_path = os.path.join(model_dir, "gp_residual.pkl")
    gp.save(gp_path)

    # -- Quick validation ------------------------------------------------------
    pred_ax, pred_ay = gp.predict(V)
    mse_x = float(np.mean((pred_ax - residual_ax) ** 2))
    mse_y = float(np.mean((pred_ay - residual_ay) ** 2))
    print(f"\n[train_gp] Training fit MSE:")
    print(f"  GP_x MSE: {mse_x:.6f}")
    print(f"  GP_y MSE: {mse_y:.6f}")
    print(f"[train_gp] Model saved -> {gp_path}")

    return gp


# ------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LBMPC residual model")
    parser.add_argument("--data",    type=str,   default="data/transitions.npz")
    parser.add_argument("--models",  type=str,   default="models")
    parser.add_argument("--epochs",  type=int,   default=150)
    parser.add_argument("--batch",   type=int,   default=256)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--type",    type=str,   default="nn",
                        choices=["nn", "gp"],
                        dest="model_type",
                        help="Model type: 'nn' (neural net) or 'gp' (Gaussian Process)")
    args = parser.parse_args()

    if args.model_type == "gp":
        train_gp(
            data_path = args.data,
            model_dir = args.models,
        )
    else:
        model = train(
            data_path  = args.data,
            model_dir  = args.models,
            epochs     = args.epochs,
            batch_size = args.batch,
            lr         = args.lr,
        )

        norm = Normalizer.load(os.path.join(args.models, "normalizer.npz"))
        validate_model(model, norm, data_path=args.data)

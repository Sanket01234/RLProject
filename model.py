# model.py
"""
Residual dynamics model for LBMPC.

Architecture:
    Input  : (x, u) normalized and concatenated -> 16-D
    Hidden : 2 × Linear(64) + Tanh
    Output : residual -> 12-D

Also contains the Normalizer class (fit once on dataset, used everywhere).
"""

import numpy as np
import torch
import torch.nn as nn


# ------------------------------------------------------------------------------
# Normalizer
# ------------------------------------------------------------------------------

class Normalizer:
    """
    Zero-mean, unit-variance normalizer fitted on the collected dataset.

    IMPORTANT: Fit this ONCE on the training data, then reuse the SAME
    instance in both train.py and LBMPC.py. If you fit a new normalizer
    at inference time the scales will differ and the model will break.
    """

    def __init__(self, X: np.ndarray = None, U: np.ndarray = None):
        """
        Args:
            X: (N, 12) state array  - if None, call load() before using
            U: (N,  4) control array
        """
        if X is not None and U is not None:
            self.x_mean = X.mean(axis=0).astype(np.float32)   # (12,)
            self.x_std  = X.std(axis=0).astype(np.float32) + 1e-8
            self.u_mean = U.mean(axis=0).astype(np.float32)   # (4,)
            self.u_std  = U.std(axis=0).astype(np.float32) + 1e-8
        else:
            # Will be populated by load()
            self.x_mean = self.x_std = self.u_mean = self.u_std = None

    # -- numpy interface ------------------------------------------------------─

    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.x_mean) / self.x_std).astype(np.float32)

    def normalize_u(self, u: np.ndarray) -> np.ndarray:
        return ((u - self.u_mean) / self.u_std).astype(np.float32)

    # -- torch interface (used inside training loop) --------------------------─

    def normalize_x_torch(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.from_numpy(self.x_mean)
        x_std  = torch.from_numpy(self.x_std)
        return (x - x_mean) / x_std

    def normalize_u_torch(self, u: torch.Tensor) -> torch.Tensor:
        u_mean = torch.from_numpy(self.u_mean)
        u_std  = torch.from_numpy(self.u_std)
        return (u - u_mean) / u_std

    # -- persistence ----------------------------------------------------------─

    def save(self, path: str = "models/normalizer.npz") -> None:
        import os; os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 x_mean=self.x_mean, x_std=self.x_std,
                 u_mean=self.u_mean, u_std=self.u_std)
        print(f"[Normalizer] Saved -> {path}")

    @classmethod
    def load(cls, path: str = "models/normalizer.npz") -> "Normalizer":
        norm = cls()                           # empty instance
        data = np.load(path)
        norm.x_mean = data["x_mean"].astype(np.float32)
        norm.x_std  = data["x_std"].astype(np.float32)
        norm.u_mean = data["u_mean"].astype(np.float32)
        norm.u_std  = data["u_std"].astype(np.float32)
        print(f"[Normalizer] Loaded from {path}")
        return norm


# ------------------------------------------------------------------------------
# Neural network
# ------------------------------------------------------------------------------

class ResidualModel(nn.Module):
    """
    Learns the residual between true dynamics and the nominal model:

        residual(x, u) = x_true_next − f_nominal(x, u)

    The model operates on NORMALISED inputs.

    Input:  concat([x_norm, u_norm]) -> 16-D float tensor
    Output: residual prediction      -> 12-D float tensor
    """

    INPUT_DIM  = 12 + 4   # state + control
    OUTPUT_DIM = 12        # residual has the same dimension as the state

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.OUTPUT_DIM),
        )
        # Initialise output layer near zero - residuals should start small
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_norm: torch.Tensor,
                u_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_norm: (batch, 12) normalised state
            u_norm: (batch,  4) normalised control

        Returns:
            residual: (batch, 12)
        """
        xu = torch.cat([x_norm, u_norm], dim=-1)   # (batch, 16)
        return self.net(xu)                         # (batch, 12)

    # -- numpy inference wrapper (used by LBMPC.py) --------------------------─

    @torch.no_grad()
    def predict_numpy(self, x: np.ndarray, u: np.ndarray,
                      norm: Normalizer) -> np.ndarray:
        """
        Convenience wrapper for single-step numpy inference.

        Args:
            x:    (12,) state vector (raw, un-normalised)
            u:    (4,)  control vector (raw, un-normalised)
            norm: fitted Normalizer instance

        Returns:
            residual: (12,) numpy array
        """
        self.eval()
        x_t = torch.from_numpy(norm.normalize_x(x)).unsqueeze(0)  # (1, 12)
        u_t = torch.from_numpy(norm.normalize_u(u)).unsqueeze(0)  # (1,  4)
        res = self.forward(x_t, u_t)                               # (1, 12)
        return res.squeeze(0).numpy()

    @torch.no_grad()
    def predict_batch_numpy(self, X: np.ndarray, U: np.ndarray,
                             norm: Normalizer) -> np.ndarray:
        """
        Batched numpy inference.

        Args:
            X: (N, 12)
            U: (N,  4)

        Returns:
            residuals: (N, 12)
        """
        self.eval()
        X_t = torch.from_numpy(norm.normalize_x(X))   # (N, 12)
        U_t = torch.from_numpy(norm.normalize_u(U))   # (N,  4)
        res = self.forward(X_t, U_t)                  # (N, 12)
        return res.numpy()


# ------------------------------------------------------------------------------
# Persistence helpers
# ------------------------------------------------------------------------------

def save_model(model: ResidualModel,
               path: str = "models/residual_model.pt") -> None:
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[ResidualModel] Saved -> {path}")


def load_model(path: str = "models/residual_model.pt") -> ResidualModel:
    model = ResidualModel()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    print(f"[ResidualModel] Loaded from {path}")
    return model


# ------------------------------------------------------------------------------
# Quick sanity check  (run: python model.py)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("-- model.py sanity check ----------------------------------------")

    # Fake dataset
    N = 500
    X  = np.random.randn(N, 12).astype(np.float32)
    U  = np.random.randn(N,  4).astype(np.float32)

    norm  = Normalizer(X, U)
    model = ResidualModel()

    # Forward pass
    X_n = torch.from_numpy(norm.normalize_x(X))
    U_n = torch.from_numpy(norm.normalize_u(U))
    out = model(X_n, U_n)

    assert out.shape == (N, 12), f"Expected (N,12), got {out.shape}"
    print(f"  Forward pass  : output shape {out.shape}  [PASS]")

    # numpy wrapper
    res = model.predict_numpy(X[0], U[0], norm)
    assert res.shape == (12,), f"Expected (12,), got {res.shape}"
    print(f"  Numpy wrapper : output shape {res.shape}  [PASS]")

    # Output layer near zero check
    assert out.abs().mean().item() < 1.0, "Outputs suspiciously large at init"
    print(f"  Init scale    : mean |output| = {out.abs().mean().item():.4f}  [PASS]")

    print("  [PASS] model.py PASSED")

# gp_model.py
"""
Gaussian Process residual model for LBMPC.

Architecture (per project slides):
    Two independent GPs — one per planar axis:
        Δa_x = μ_x(v_x, v_y)
        Δa_y = μ_y(v_x, v_y)

    Input : velocity vector [v_x, v_y]  (2-D)
    Output: residual acceleration        (scalar, per axis)

During MPC rollout the corrected dynamics are:
    v_{k+1} = v_k + (a_k + Δa_k) · dt
    p_{k+1} = p_k + v_k · dt

Only the GP mean μ(z*) is plugged into MPC.
The variance σ²(z*) is available for analysis and plotting.

Usage:
    # Training
    gp = GPResidualModel()
    gp.fit(V_train, residual_ax, residual_ay)
    gp.save("models/gp_residual.pkl")

    # Inference
    gp = GPResidualModel.load("models/gp_residual.pkl")
    delta_ax, delta_ay = gp.predict(v)
"""

import os
import numpy as np
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


# ──────────────────────────────────────────────────────────────────────────────
# Default kernel
# ──────────────────────────────────────────────────────────────────────────────

def default_kernel():
    """
    Squared-exponential (RBF) kernel with automatic relevance determination.

    ConstantKernel * RBF  +  WhiteKernel (noise)

    The length-scale and noise level are optimized during fit().
    Starting values are deliberately broad so the optimizer can explore.
    """
    return (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
    )


# ──────────────────────────────────────────────────────────────────────────────
# GP Residual Model
# ──────────────────────────────────────────────────────────────────────────────

class GPResidualModel:
    """
    Axis-wise Gaussian Process regression for residual acceleration.

    Training pairs:
        inputs  z_i = [v_{x,i}, v_{y,i}]^T   (2-D velocity)
        outputs y_i = residual acceleration     (scalar, per axis)

    Two independent GPs are fitted:
        gp_x: predicts Δa_x from velocity
        gp_y: predicts Δa_y from velocity

    For a new velocity z*, the GP predicts:
        f(z*) | D ~ N(μ(z*), σ²(z*))

    Only the mean μ(z*) is used in MPC; σ² is for analysis.
    """

    def __init__(self, kernel=None, alpha: float = 1e-6,
                 n_restarts: int = 3):
        """
        Args:
            kernel:      scikit-learn kernel (default: ConstantKernel * RBF + WhiteKernel)
            alpha:       regularization (noise added to diagonal of K)
            n_restarts:  number of optimizer restarts for kernel hyperparameters
        """
        if kernel is None:
            kernel = default_kernel()

        self.gp_x = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
        )
        self.gp_y = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha,
            n_restarts_optimizer=n_restarts,
            normalize_y=True,
        )
        self._fitted = False

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, V: np.ndarray,
            residual_ax: np.ndarray,
            residual_ay: np.ndarray) -> None:
        """
        Fit the two GPs on collected data.

        Args:
            V:           (N, 2)  velocity pairs [v_x, v_y]
            residual_ax: (N,)    x-axis residual accelerations
            residual_ay: (N,)    y-axis residual accelerations
        """
        assert V.ndim == 2 and V.shape[1] == 2, \
            f"V must be (N, 2), got {V.shape}"
        assert residual_ax.shape[0] == V.shape[0], \
            "residual_ax length must match V"
        assert residual_ay.shape[0] == V.shape[0], \
            "residual_ay length must match V"

        print(f"[GPResidualModel] Fitting GP_x on {V.shape[0]} samples...")
        self.gp_x.fit(V, residual_ax)
        print(f"  GP_x kernel: {self.gp_x.kernel_}")

        print(f"[GPResidualModel] Fitting GP_y on {V.shape[0]} samples...")
        self.gp_y.fit(V, residual_ay)
        print(f"  GP_y kernel: {self.gp_y.kernel_}")

        self._fitted = True

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, v: np.ndarray) -> tuple:
        """
        Predict residual acceleration (mean only).

        Args:
            v: (2,) or (N, 2) velocity vector(s)

        Returns:
            delta_ax: scalar or (N,) x-axis residual acceleration
            delta_ay: scalar or (N,) y-axis residual acceleration
        """
        assert self._fitted, "Call fit() before predict()"
        v = np.atleast_2d(v)  # (N, 2)

        delta_ax = self.gp_x.predict(v)   # (N,)
        delta_ay = self.gp_y.predict(v)   # (N,)

        # Return scalar for single input
        if v.shape[0] == 1:
            return float(delta_ax[0]), float(delta_ay[0])
        return delta_ax, delta_ay

    def predict_with_variance(self, v: np.ndarray) -> tuple:
        """
        Predict residual acceleration with uncertainty estimate.

        Args:
            v: (2,) or (N, 2) velocity vector(s)

        Returns:
            (mean_x, std_x, mean_y, std_y)
            Each is scalar or (N,) depending on input shape.
        """
        assert self._fitted, "Call fit() before predict_with_variance()"
        v = np.atleast_2d(v)

        mean_x, std_x = self.gp_x.predict(v, return_std=True)
        mean_y, std_y = self.gp_y.predict(v, return_std=True)

        if v.shape[0] == 1:
            return float(mean_x[0]), float(std_x[0]), \
                   float(mean_y[0]), float(std_y[0])
        return mean_x, std_x, mean_y, std_y

    # ── Batch inference for LBMPC rollout ────────────────────────────────────

    def predict_batch(self, V: np.ndarray) -> np.ndarray:
        """
        Batched prediction returning a (N, 2) array of [Δa_x, Δa_y].

        Designed for use inside LBMPC rollout_batch().

        Args:
            V: (N, 2) velocity vectors

        Returns:
            delta_a: (N, 2) residual accelerations
        """
        assert self._fitted, "Call fit() before predict_batch()"
        V = np.atleast_2d(V)
        delta_ax = self.gp_x.predict(V)
        delta_ay = self.gp_y.predict(V)
        return np.column_stack([delta_ax, delta_ay])  # (N, 2)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "models/gp_residual.pkl") -> None:
        """Save the fitted GP model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "gp_x": self.gp_x,
                "gp_y": self.gp_y,
                "fitted": self._fitted,
            }, f)
        print(f"[GPResidualModel] Saved -> {path}")

    @classmethod
    def load(cls, path: str = "models/gp_residual.pkl") -> "GPResidualModel":
        """Load a previously saved GP model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.gp_x = data["gp_x"]
        obj.gp_y = data["gp_y"]
        obj._fitted = data["fitted"]
        print(f"[GPResidualModel] Loaded from {path}")
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Standalone sanity check  (run: python gp_model.py)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("-- gp_model.py sanity check ------------------------------------")

    # Generate synthetic data: Δa = f(v_x, v_y) with a known pattern
    np.random.seed(42)
    N = 100
    V = np.random.randn(N, 2).astype(np.float64) * 2.0   # velocities
    # True residual: quadratic drag-like function
    true_ax = -0.05 * V[:, 0] * np.abs(V[:, 0])
    true_ay = -0.05 * V[:, 1] * np.abs(V[:, 1])
    # Add noise
    residual_ax = true_ax + np.random.randn(N) * 0.01
    residual_ay = true_ay + np.random.randn(N) * 0.01

    # Fit
    gp = GPResidualModel()
    gp.fit(V, residual_ax, residual_ay)

    # Single prediction
    v_test = np.array([1.0, -0.5])
    dax, day = gp.predict(v_test)
    print(f"  predict([1.0, -0.5]) -> d_ax={dax:.4f}, d_ay={day:.4f}")

    # With variance
    mx, sx, my, sy = gp.predict_with_variance(v_test)
    print(f"  with variance -> mx={mx:.4f}+/-{sx:.4f}, my={my:.4f}+/-{sy:.4f}")

    # Batch prediction
    V_batch = np.random.randn(10, 2)
    delta_a = gp.predict_batch(V_batch)
    assert delta_a.shape == (10, 2), f"Expected (10, 2), got {delta_a.shape}"
    print(f"  batch predict -> shape {delta_a.shape}  [PASS]")

    # Save / load roundtrip
    gp.save("models/gp_test.pkl")
    gp2 = GPResidualModel.load("models/gp_test.pkl")
    dax2, day2 = gp2.predict(v_test)
    assert abs(dax - dax2) < 1e-10, "Save/load changed predictions!"
    print(f"  save/load roundtrip  [PASS]")

    # Cleanup
    os.remove("models/gp_test.pkl")
    print("  [PASS] gp_model.py PASSED")

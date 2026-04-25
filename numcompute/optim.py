from typing import Callable, Literal, Optional

import numpy as np


def _as_1d_array(x: np.ndarray, name: str = "x") -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _validate_h(h: float) -> float:
    h_val = float(h)
    if not np.isfinite(h_val) or h_val <= 0.0:
        raise ValueError("h must be a positive finite float.")
    return h_val


def _as_1d_output(y: np.ndarray, name: str = "output") -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be scalar or a 1D array.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    return arr


def _as_scalar(y: np.ndarray, name: str = "f(x)") -> float:
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1 and arr.size == 1:
        return float(arr[0])
    raise ValueError(f"{name} must be scalar-valued.")


def grad(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    h: float = 1e-5,
    method: Literal["central", "forward"] = "central",
) -> np.ndarray:
    """
    Estimate gradient of a scalar-valued function using finite differences.
    """
    x_arr = _as_1d_array(x)
    h_val = _validate_h(h)

    if method not in {"central", "forward"}:
        raise ValueError("method must be one of: 'central', 'forward'.")

    g = np.empty_like(x_arr, dtype=float)
    if method == "forward":
        fx = _as_scalar(f(x_arr), name="f(x)")

    for i in range(x_arr.size):
        x_plus = x_arr.copy()
        x_plus[i] += h_val

        if method == "central":
            x_minus = x_arr.copy()
            x_minus[i] -= h_val
            f_plus = _as_scalar(f(x_plus), name="f(x + h e_i)")
            f_minus = _as_scalar(f(x_minus), name="f(x - h e_i)")
            g[i] = (f_plus - f_minus) / (2.0 * h_val)
        else:
            f_plus = _as_scalar(f(x_plus), name="f(x + h e_i)")
            g[i] = (f_plus - fx) / h_val

    return g


def jacobian(
    F: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    h: float = 1e-5,
    method: Literal["central", "forward"] = "central",
) -> np.ndarray:
    """
    Estimate Jacobian of a vector-valued function using finite differences.
    """
    x_arr = _as_1d_array(x)
    h_val = _validate_h(h)

    if method not in {"central", "forward"}:
        raise ValueError("method must be one of: 'central', 'forward'.")

    Fx = _as_1d_output(F(x_arr), name="F(x)")
    m = Fx.size
    n = x_arr.size
    J = np.empty((m, n), dtype=float)

    for i in range(n):
        x_plus = x_arr.copy()
        x_plus[i] += h_val

        if method == "central":
            x_minus = x_arr.copy()
            x_minus[i] -= h_val
            f_plus = _as_1d_output(F(x_plus), name="F(x + h e_i)")
            f_minus = _as_1d_output(F(x_minus), name="F(x - h e_i)")
            if f_plus.size != m or f_minus.size != m:
                raise ValueError("F must return outputs with consistent dimension.")
            J[:, i] = (f_plus - f_minus) / (2.0 * h_val)
        else:
            f_plus = _as_1d_output(F(x_plus), name="F(x + h e_i)")
            if f_plus.size != m:
                raise ValueError("F must return outputs with consistent dimension.")
            J[:, i] = (f_plus - Fx) / h_val

    return J


def line_search(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    direction: np.ndarray,
    grad_x: Optional[np.ndarray] = None,
    alpha0: float = 1.0,
    c: float = 1e-4,
    tau: float = 0.5,
    max_iter: int = 50,
) -> float:
    """
    Armijo backtracking line search.

    Returns the accepted step size and falls back to 0.0 if no acceptable
    step is found within max_iter reductions.
    """
    x_arr = _as_1d_array(x)
    p = _as_1d_array(direction, name="direction")
    if p.shape != x_arr.shape:
        raise ValueError("direction must have the same shape as x.")

    alpha = float(alpha0)
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha0 must be a positive finite float.")
    if not (0.0 < c < 1.0):
        raise ValueError("c must be in (0, 1).")
    if not (0.0 < tau < 1.0):
        raise ValueError("tau must be in (0, 1).")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")

    fx = _as_scalar(f(x_arr), name="f(x)")

    if grad_x is None:
        g = grad(f, x_arr)
    else:
        g = _as_1d_array(grad_x, name="grad_x")
        if g.shape != x_arr.shape:
            raise ValueError("grad_x must have the same shape as x.")

    directional_derivative = float(np.dot(g, p))
    if directional_derivative >= 0.0:
        raise ValueError(
            "direction must be a descent direction (grad_x dot direction < 0)."
        )

    for _ in range(max_iter):
        candidate = x_arr + alpha * p
        f_candidate = _as_scalar(f(candidate), name="f(x + alpha * direction)")
        if f_candidate <= fx + c * alpha * directional_derivative:
            return alpha
        alpha *= tau

    return 0.0

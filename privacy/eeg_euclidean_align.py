"""
[IJCAI 2025] ID-RemovalNet 中的 EEG Data Align (EA)：
R = (1/N) Σ X_n X_n^T，再 X'_n = R^{-1/2} X_n，降低跨被试/会话分布差异。
"""
from __future__ import annotations

import numpy as np


def reference_covariance_spatial(X: np.ndarray) -> np.ndarray:
    """X: (N, C, T) -> R: (C, C) 平均空间协方差。"""
    n, c, _ = X.shape
    R = np.zeros((c, c), dtype=np.float64)
    for i in range(n):
        xn = X[i].astype(np.float64)
        R += xn @ xn.T
    R /= max(n, 1)
    R += np.eye(c, dtype=np.float64) * 1e-6
    return R


def matrix_inv_sqrt_psd(R: np.ndarray) -> np.ndarray:
    """对称正定矩阵的 R^{-1/2}。"""
    w, v = np.linalg.eigh(R)
    w = np.clip(w, 1e-10, None)
    return (v @ np.diag(w ** (-0.5)) @ v.T).astype(np.float32)


def apply_euclidean_alignment(X: np.ndarray, R_inv_sqrt: np.ndarray) -> np.ndarray:
    """X: (N,C,T)，左乘 R^{-1/2} 到每个试次。"""
    n, c, t = X.shape
    out = np.empty_like(X, dtype=np.float32)
    for i in range(n):
        out[i] = R_inv_sqrt @ X[i]
    return out


def fit_ea_from_trials(X: np.ndarray):
    """返回 (R, R^{-1/2})，用于训练/评估一致保存。"""
    R = reference_covariance_spatial(X)
    R_inv_sqrt = matrix_inv_sqrt_psd(R)
    return R.astype(np.float32), R_inv_sqrt

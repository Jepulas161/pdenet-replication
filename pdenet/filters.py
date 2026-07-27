"""
Constrained filters for PDE-Net based on moment matrix constraints.

Each differential operator D_ij (approximating ∂^{i+j}/∂x^i∂y^j) is
represented by a convolution filter whose moment matrix satisfies:

  (M(q_ij))_{k1,k2} = 0  for k1+k2 ≤ i+j+2, (k1,k2) ≠ (i+1,j+1)
  (M(q_ij))_{i+1,j+1} = 1

M(q) entry (i,j):  m_{i,j} = 1/((i-1)!(j-1)!) * Σ_{n,m} n^{i-1} m^{j-1} q[n,m]
"""

import math
import torch
import numpy as np


def _coord_range(N):
    """Return coordinate grid for an N×N filter (N odd)."""
    half = (N - 1) // 2
    return np.arange(-half, half + 1)


def _build_moment_weight_matrix(N, max_order):
    """
    Build weight matrix W such that W @ q.ravel() gives the vector of moments
    up to max_order (i.e. all (k1,k2) with k1+k2 ≤ max_order, where
    k1, k2 are the moment orders in x and y).

    Returns:
        W: (n_moments, N²) numpy array
        moment_keys: list of (k1, k2) tuples
    """
    coords = _coord_range(N)
    n_filters = N * N

    moment_keys = []
    rows = []
    for k1 in range(max_order + 1):
        for k2 in range(max_order + 1 - k1):
            moment_keys.append((k1, k2))
            row = np.zeros(n_filters)
            for idx_n, n in enumerate(coords):
                for idx_m, m in enumerate(coords):
                    flat_idx = idx_n * N + idx_m
                    row[flat_idx] = (_safe_pow(n, k1) * _safe_pow(m, k2)
                                     / (math.factorial(k1) * math.factorial(k2)))
            rows.append(row)

    return np.stack(rows), moment_keys


def _build_constrained_filter_basis(N, i, j):
    """
    Build a parametrized basis for an N×N filter that approximates
    ∂^{i+j}/∂x^i∂y^j.

    For i+j > 0:
      - All moments with order ≤ i+j+2 are constrained to 0
        except the (i,j) moment which is constrained to 1
      - Higher-order moments are free

    For i=j=0 (D0/D00 — averaging):
      - (M(q))_{1,1} = 1  (the 0th moment = 1, meaning sum of filter = 1)
      - No further constraints

    Returns:
        init_filter: numpy array of shape (N, N) — the particular solution
        nullspace_basis: numpy array of shape (N², n_free) — nullspace basis
        n_constraints: int — number of equality constraints
    """
    max_order = i + j + 2 if i + j > 0 else 0
    W, moment_keys = _build_moment_weight_matrix(N, max_order)

    n_total = N * N
    n_constr = len(moment_keys)

    # Build right-hand side
    rhs = np.zeros(n_constr)
    if i + j == 0:
        # Averaging filter: 0th moment = 1
        rhs[0] = 1.0
    else:
        # Find (i,j) in moment_keys
        target_idx = moment_keys.index((i, j))
        rhs[target_idx] = 1.0

    # Solve: find a particular solution and nullspace
    # Use SVD to find nullspace of W
    U, S, Vt = np.linalg.svd(W, full_matrices=True)

    # Rank determination
    tol = max(W.shape) * np.finfo(float).eps * S[0]
    r = np.sum(S > tol)

    # Particular solution: x_p = V @ S^{-1} @ U^T @ rhs
    S_inv = np.zeros((W.shape[1], W.shape[0]))
    S_inv[:r, :r] = np.diag(1.0 / S[:r])
    x_p = Vt.T @ S_inv @ U.T @ rhs

    # Nullspace: columns of V corresponding to zero singular values
    nullspace = Vt[r:, :].T  # (n_total, n_free)

    n_free = N * N - r
    return x_p.reshape(N, N), nullspace, n_free


def create_constrained_filter(N, i, j, requires_grad=True):
    """
    Create a parametrized constrained filter as a PyTorch module.

    The filter is represented as:
        q = q0 + N @ theta
    where q0 is the particular solution (frozen base), N is the nullspace
    basis, and theta are learnable parameters. The constraints are
    automatically satisfied for any theta.

    Args:
        N: filter size (odd number)
        i, j: order of differentiation (∂^{i+j}/∂x^i∂y^j)
        requires_grad: whether to make parameters trainable

    Returns:
        q0: (N, N) numpy array — the base filter
        theta: torch Parameter or None
        N_basis: (N², n_params) numpy array
        n_free: number of free parameters
    """
    q0, nullspace, _ = _build_constrained_filter_basis(N, i, j)
    n_free = nullspace.shape[1]

    return q0, nullspace, n_free


def _safe_pow(base, exp):
    """Compute base**exp, returning 1 when exp==0 even if base==0."""
    if exp == 0:
        return 1.0
    return float(base) ** exp


def moment_matrix(q, N):
    """
    Compute the moment matrix of a filter q (N×N).
    Entry (i,j) in 0-indexed return = (i,j)-moment (where i=k1, j=k2).

    m_{k1,k2} = 1/(k1! k2!) * Σ_{n,m} n^{k1} m^{k2} q[n,m]
    """
    coords = _coord_range(N)
    M = np.zeros((N, N))
    for k1 in range(N):
        for k2 in range(N):
            moment = 0.0
            for idx_n, n in enumerate(coords):
                for idx_m, m in enumerate(coords):
                    term = (_safe_pow(float(n), k1) * _safe_pow(float(m), k2)
                            * q[idx_n, idx_m]
                            / (math.factorial(k1) * math.factorial(k2)))
                    moment += term
            M[k1, k2] = moment
    return M


def verify_constraints(q, N, i, j):
    """Verify that filter q satisfies the moment constraints for D_{ij}."""
    M = moment_matrix(q, N)
    max_order = i + j + 2 if i + j > 0 else 0

    print(f"Filter D_{{{i},{j}}} (N={N}) moment matrix [k1+k2 ≤ 4]:")
    # Print sub-matrix for low orders
    sub = M[:min(5, N), :min(5, N)]
    print(np.array2string(sub, precision=4, suppress_small=True))

    violations = []
    for k1 in range(max_order + 1):
        for k2 in range(max_order + 1 - k1):
            val = M[k1, k2]
            if k1 == i and k2 == j:
                if abs(val - 1.0) > 1e-8:
                    violations.append(f"  ({k1},{k2}) = {val:.6f} (expected 1)")
            else:
                if abs(val) > 1e-8:
                    violations.append(f"  ({k1},{k2}) = {val:.6f} (expected 0)")

    if violations:
        print("  VIOLATIONS:")
        for v in violations:
            print(v)
    else:
        print("  All constraints satisfied.")

    return M


def initialize_filter_parameters(N, i, j):
    """
    Initialize both frozen and learnable parts for a constrained filter.

    Returns:
        init_filter_np: (N, N) numpy array — initial filter values
        n_free: number of free parameters
    """
    q0, nullspace, n_free = create_constrained_filter(N, i, j)
    return q0, nullspace, n_free


if __name__ == "__main__":
    # Test moment constraints for various differential operators
    for N in [5]:
        for (i, j) in [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]:
            q0, nullspace, n_free = create_constrained_filter(N, i, j)
            print(f"\n=== D_{{{i},{j}}} filter, N={N}, free params={n_free} ===")
            verify_constraints(q0, N, i, j)
            print(f"Filter sum: {q0.sum():.4f}")

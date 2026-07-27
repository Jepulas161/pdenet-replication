"""
PDE-Net: Learning PDEs from Data

Core implementation of the PDE-Net architecture as described in:
Long, Lu, Ma, Dong (ICML 2018).

Architecture:
- δt-block: ũ(t_{i+1}) = D0 u(t_i) + Δt · F(D00 u, D10 u, D01 u, ..., D04 u, ...)
- Multiple δt-blocks stacked with shared parameters for long-term prediction
- Moment-constrained convolution filters for differential operators
- Learnable variable coefficients for linear PDEs
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .filters import _build_constrained_filter_basis


def _torch_safe_pow(base, exp):
    """Compute base**exp safely, handling 0**0."""
    if exp == 0:
        return torch.ones_like(base)
    return base ** exp


class ConstrainedConv2d(nn.Module):
    """
    2D convolution with moment-matrix-constrained filter.

    The filter is parameterized as: q = q0 + N·θ
    where q0 is a particular solution satisfying the constraints,
    N is the nullspace basis, and θ are learnable parameters.
    """

    def __init__(self, N, i, j, stride=1, padding='same'):
        super().__init__()
        self.N = N
        self.i = i
        self.j = j
        self.stride = stride
        self.padding = padding

        # Build constrained filter basis (numpy)
        q0_np, nullspace_np, n_free = _build_constrained_filter_basis(N, i, j)

        # Register frozen base filter
        self.register_buffer('q0', torch.from_numpy(q0_np).float().unsqueeze(0).unsqueeze(0))

        # Register nullspace basis
        self.register_buffer('nullspace', torch.from_numpy(nullspace_np).float())

        # Learnable parameters (coefficients in the nullspace)
        if n_free > 0:
            self.theta = nn.Parameter(self._init_theta(q0_np, nullspace_np, i, j, N))
        else:
            self.theta = None

        self.n_free = n_free

    @staticmethod
    def _init_theta(q0_np, nullspace_np, i, j, N):
        """Initialize theta so the filter starts from a good guess."""
        if i == 0 and j == 0:
            # For D0 (averaging): target = identity (delta at center)
            q_target = np.zeros((N, N))
            q_target[N // 2, N // 2] = 1.0
            # Find theta s.t. q0 + N@theta ≈ q_target
            delta = q_target.ravel() - q0_np.ravel()
            theta = nullspace_np.T @ delta  # project onto nullspace
        else:
            # For derivatives: q0 is already a good FD approximation
            # But we add a tiny random perturbation to break symmetry
            rng = np.random.RandomState(42 + i * 10 + j)
            theta = rng.randn(nullspace_np.shape[1]) * 0.01

        return torch.from_numpy(theta).float().unsqueeze(0)  # (1, n_free)

    def get_filter(self):
        """Get the current constrained filter."""
        if self.theta is not None:
            # q = q0 + N @ theta
            delta = self.nullspace @ self.theta.t()  # (N², 1)
            q = self.q0.view(-1, 1) + delta  # (N², 1)
            q = q.view(1, 1, self.N, self.N)
        else:
            q = self.q0
        return q

    def forward(self, x):
        """
        Args:
            x: (B, 1, H, W) input tensor

        Returns:
            y: (B, 1, H, W) output tensor
        """
        q = self.get_filter()

        if self.padding == 'same':
            pad = self.N // 2
            x_pad = F.pad(x, (pad, pad, pad, pad), mode='replicate')
        else:
            x_pad = x

        return F.conv2d(x_pad, q, stride=self.stride)


class PDEBlock(nn.Module):
    """
    A single δt-block of the PDE-Net.

    For linear PDEs (order ≤ K):
      ũ(t_{n+1}) = D0 u(t_n) + Δt · Σ_{0≤i+j≤K} c_ij ⊙ (D_ij u)

    where c_ij are learnable coefficient fields (variable coefficients),
    D_ij are convolution operators with constrained filters.

    For nonlinear PDEs:
      ũ(t_{n+1}) = D0 u(t_n) + Δt · F(D00 u, D10 u, D01 u, ...)

    where F is a pointwise neural network.
    """

    def __init__(self, filter_size, max_order=4, use_nn=False, hidden_dims=None,
                 grid_shape=None, dx=1.0, dy=None, piecewise_order=2):
        """
        Args:
            filter_size: size of convolution filters (odd number, e.g. 5 or 7)
            max_order: maximum order of derivatives (i+j ≤ max_order)
            use_nn: if True, use a pointwise NN for nonlinear F;
                    if False, use linear combination with variable coefficients
            hidden_dims: list of hidden dims for pointwise NN (if use_nn=True)
            grid_shape: (H, W) of the grid, needed for variable coefficient fields
            dx: grid spacing in x direction (for scaling physical derivatives)
            dy: grid spacing in y direction (defaults to dx)
            piecewise_order: order of piecewise polynomial for coefficient fields (1 or 2)
        """
        super().__init__()
        self.filter_size = filter_size
        self.max_order = max_order
        self.use_nn = use_nn
        self.grid_shape = grid_shape
        self.dx = dx
        self.dy = dy if dy is not None else dx

        # Build list of (i, j) pairs for all derivative orders
        # EXCLUDE (0,0): D0 handles the identity part; coefficients should
        # only apply to derivative terms. Including (0,0) lets the model
        # cheat by learning an unphysical c_00 * u reaction term.
        self.deriv_pairs = []
        for order in range(max_order + 1):
            if order == 0:
                continue  # skip (0,0) — D0 handles it
            for i in range(order + 1):
                j = order - i
                self.deriv_pairs.append((i, j))

        # Averaging operator D0
        self.D0 = ConstrainedConv2d(filter_size, 0, 0)

        # Derivative operators D_ij
        self.deriv_convs = nn.ModuleDict()
        for i, j in self.deriv_pairs:
            self.deriv_convs[f'D_{i}{j}'] = ConstrainedConv2d(filter_size, i, j)

        if use_nn:
            # Pointwise neural network for nonlinear F
            n_derivs = len(self.deriv_pairs)
            if hidden_dims is None:
                hidden_dims = [100, 100]
            layers = []
            prev_dim = n_derivs
            for h in hidden_dims:
                layers.extend([
                    nn.Conv2d(prev_dim, h, 1),
                    nn.Tanh(),
                ])
                prev_dim = h
            # Output layer: single channel
            layers.append(nn.Conv2d(prev_dim, 1, 1))
            self.F_nn = nn.Sequential(*layers)
            self.coeff_fields = None
        else:
            # Learnable variable coefficient fields c_ij(x,y)
            # Represented as bilinearly-interpolated coarse fields
            self.coeff_fields = nn.ParameterDict()
            H, W = grid_shape
            # Coarse grid size (5×5 patches for a 50×50 grid)
            self.coarse_h = max(4, H // 10)
            self.coarse_w = max(4, W // 10)

            for i, j in self.deriv_pairs:
                self.coeff_fields[f'c_{i}{j}'] = nn.Parameter(
                    torch.randn(1, 1, self.coarse_h, self.coarse_w) * 0.01
                )

            self.piecewise_order = piecewise_order
            self.F_nn = None

    def evaluate_coeff_field(self, i, j, device):
        """
        Evaluate the variable coefficient field c_ij(x,y) on the full grid.
        Uses bilinear interpolation from a coarse representation.
        """
        H, W = self.grid_shape
        params = self.coeff_fields[f'c_{i}{j}']  # (1, 1, Ph, Pw)

        # Upsample coarse coefficients to full grid
        field = F.interpolate(params, size=(H, W), mode='bilinear',
                              align_corners=False)

        return field  # (1, 1, H, W)

    def forward(self, u, dt=1.0):
        """
        Args:
            u: (B, 1, H, W) input
            dt: time step size

        Returns:
            u_next: (B, 1, H, W) prediction at next time step
        """
        # Apply averaging operator
        u_avg = self.D0(u)

        # Apply derivative operators, scaling to physical derivatives
        # Constrained filters output Δx^i·Δy^j · ∂^{i+j}u/∂x^i∂y^j
        # So divide by dx^i * dy^j to get the actual partial derivative
        derivs = []
        for i, j in self.deriv_pairs:
            d_ij = self.deriv_convs[f'D_{i}{j}'](u)
            d_ij = d_ij / (self.dx ** i * self.dy ** j)
            derivs.append(d_ij)

        if self.use_nn:
            # Pointwise NN
            deriv_stack = torch.cat(derivs, dim=1)  # (B, n_derivs, H, W)
            F_out = self.F_nn(deriv_stack)
        else:
            # Linear combination with variable coefficients
            F_out = torch.zeros_like(u)
            device = u.device
            for idx, (i, j) in enumerate(self.deriv_pairs):
                coeff = self.evaluate_coeff_field(i, j, device)
                F_out = F_out + coeff * derivs[idx]

        u_next = u_avg + F_out
        return u_next


class PDENet(nn.Module):
    """
    PDE-Net: stack of multiple δt-blocks with shared parameters.

    Supports layer-wise training.
    """

    def __init__(self, filter_size, max_order=4, n_blocks=6, use_nn=False,
                 hidden_dims=None, grid_shape=(50, 50), dx=1.0, dy=None,
                 piecewise_order=2):
        super().__init__()
        self.n_blocks = n_blocks

        # Single block (all blocks share parameters)
        self.block = PDEBlock(
            filter_size=filter_size,
            max_order=max_order,
            use_nn=use_nn,
            hidden_dims=hidden_dims,
            grid_shape=grid_shape,
            dx=dx,
            dy=dy,
            piecewise_order=piecewise_order,
        )

    def forward(self, u, dt=1.0, n_steps=None):
        """
        Args:
            u: (B, 1, H, W) initial state
            dt: time step
            n_steps: number of δt-blocks to apply (default: all n_blocks)

        Returns:
            u_out: (B, 1, H, W) state after n_steps steps
        """
        if n_steps is None:
            n_steps = self.n_blocks

        u_out = u
        for _ in range(n_steps):
            u_out = self.block(u_out, dt=dt)
        return u_out

    def get_coefficient_fields(self):
        """Extract learned coefficient fields for PDE identification."""
        if self.block.coeff_fields is None:
            return None

        device = next(self.block.parameters()).device
        fields = {}
        for i, j in self.block.deriv_pairs:
            field = self.block.evaluate_coeff_field(i, j, device)
            fields[(i, j)] = field.squeeze().cpu().detach().numpy()
        return fields


def get_frozen_filters(filter_size, max_order=4):
    """
    Build frozen (non-learnable) constrained filters for comparison.
    Returns a dict of (i,j) -> tensor filter.
    """
    filters = {}
    deriv_pairs = []
    for order in range(max_order + 1):
        for i in range(order + 1):
            j = order - i
            deriv_pairs.append((i, j))

    for i, j in deriv_pairs:
        q0_np, _, _ = _build_constrained_filter_basis(filter_size, i, j)
        filters[(i, j)] = torch.from_numpy(q0_np).float()

    return filters

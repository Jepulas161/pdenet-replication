"""
Data generation for PDE-Net experiments.

Example 1: Linear variable-coefficient convection-diffusion equation
  u_t = a(x,y)u_x + b(x,y)u_y + 0.2 u_xx + 0.3 u_yy

  a(x,y) = 0.5(cos(y) + x(2π - x)sin(x)) + 0.6
  b(x,y) = 2(cos(y) + sin(x)) + 0.8

  Domain: [0, 2π]², periodic BC
  Generated with spectral method (30×30 modes) + RK4, δt=0.015

Example 2: Diffusion equation with nonlinear source
  u_t = c Δu + f_s(u),  c = 0.3, f_s(u) = 15 sin(u)
"""

import numpy as np
from numpy.fft import fft2, ifft2, fftfreq


def _truncate_modes(u_hat, n_modes=30):
    """Keep only the lowest n_modes Fourier modes (Galerkin truncation)."""
    Nx, Ny = u_hat.shape
    # FFT bin layout: [0, 1, ..., N/2-1, -N/2, ..., -1]
    # We keep |k| < n_modes/2 in each direction
    half = n_modes // 2
    out = np.zeros_like(u_hat)
    out[:half, :half] = u_hat[:half, :half]
    out[-half:, :half] = u_hat[-half:, :half]
    out[:half, -half:] = u_hat[:half, -half:]
    out[-half:, -half:] = u_hat[-half:, -half:]
    return out


def convection_diffusion_rhs(u_hat, kx, ky, a_coeff, b_coeff, nu1=0.2, nu2=0.3):
    """
    Compute RHS of convection-diffusion equation using pseudo-spectral method
    with Galerkin truncation to 30×30 modes.
    """
    Nx, Ny = u_hat.shape

    # Compute u in physical space
    u = np.real(ifft2(u_hat))

    # Compute derivatives in spectral space
    ux_hat = 1j * kx[:, None] * u_hat
    uy_hat = 1j * ky[None, :] * u_hat
    uxx_hat = -kx[:, None] ** 2 * u_hat
    uyy_hat = -ky[None, :] ** 2 * u_hat

    # Transform derivatives to physical space
    ux = np.real(ifft2(ux_hat))
    uy = np.real(ifft2(uy_hat))
    uxx = np.real(ifft2(uxx_hat))
    uyy = np.real(ifft2(uyy_hat))

    # Compute RHS in physical space
    rhs = a_coeff * ux + b_coeff * uy + nu1 * uxx + nu2 * uyy

    # Transform back to spectral space and apply Galerkin truncation
    rhs_hat = fft2(rhs)
    rhs_hat = _truncate_modes(rhs_hat, n_modes=30)

    return rhs_hat


def generate_convection_diffusion_data(
    n_samples=1000,
    grid_size=200,
    downsampled_size=50,
    dt=0.015,
    n_steps=20,
    max_freq=9,
    noise_level=0.015,
    random_shift=True,
):
    """
    Generate training data for convection-diffusion equation.

    Args:
        n_samples: number of initial conditions
        grid_size: resolution of the fine grid (200×200)
        downsampled_size: resolution after downsampling (50×50)
        dt: time step size
        n_steps: number of time steps to generate
        max_freq: maximum frequency in initial condition
        noise_level: standard deviation of Gaussian noise (relative to max)
        random_shift: whether to apply random shifts

    Returns:
        data: (n_samples, n_steps+1, downsampled_size, downsampled_size) array
        t: time points
    """
    N = grid_size
    L = 2 * np.pi
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Spectral wavenumbers
    kx = fftfreq(N, dx/(2*np.pi))
    ky = fftfreq(N, dx/(2*np.pi))

    # Coefficient fields
    a_coeff = 0.5 * (np.cos(Y) + X * (L - X) * np.sin(X)) + 0.6
    b_coeff = 2 * (np.cos(Y) + np.sin(X)) + 0.8

    # Storage
    data = np.zeros((n_samples, n_steps + 1, downsampled_size, downsampled_size))

    for s in range(n_samples):
        # Generate random initial condition
        u0 = np.zeros((N, N))
        for k in range(1, max_freq + 1):
            for l in range(1, max_freq + 1):
                if np.random.random() < 0.5:
                    continue  # sparsity
                amp = np.random.randn() * (1.0 / 50)
                phase = np.random.rand() * 2 * np.pi
                u0 += amp * np.cos(k * X + l * Y + phase)
                amp = np.random.randn() * (1.0 / 50)
                phase = np.random.rand() * 2 * np.pi
                u0 += amp * np.sin(k * X + l * Y + phase)

        # Solve with spectral method + RK4
        u_hat = fft2(u0)
        for step in range(n_steps + 1):
            # Downsample and store
            if downsampled_size < N:
                stride = N // downsampled_size
                u_phys = np.real(ifft2(u_hat))
                u_down = u_phys[::stride, ::stride]

                # Random shift (0-3 pixels)
                if random_shift and step > 0:
                    shift_x = np.random.randint(0, 4)
                    shift_y = np.random.randint(0, 4)
                    u_down = np.roll(u_down, shift_x, axis=0)
                    u_down = np.roll(u_down, shift_y, axis=1)

                data[s, step] = u_down
            else:
                data[s, step] = np.real(ifft2(u_hat))

            if step == n_steps:
                break

            # RK4 step
            k1 = convection_diffusion_rhs(u_hat, kx, ky, a_coeff, b_coeff)
            k2 = convection_diffusion_rhs(u_hat + 0.5 * dt * k1, kx, ky, a_coeff, b_coeff)
            k3 = convection_diffusion_rhs(u_hat + 0.5 * dt * k2, kx, ky, a_coeff, b_coeff)
            k4 = convection_diffusion_rhs(u_hat + dt * k3, kx, ky, a_coeff, b_coeff)
            u_hat = u_hat + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            u_hat = _truncate_modes(u_hat, n_modes=30)

    # Add noise
    if noise_level > 0:
        M = np.max(np.abs(data))
        noise = np.random.randn(*data.shape) * noise_level * M
        data = data + noise

    t_points = np.arange(n_steps + 1) * dt

    return data, t_points


def generate_nonlineardiffusion_data(
    n_samples=1000,
    grid_size=100,
    downsampled_size=50,
    dt=0.0009,
    n_steps=20,
    max_freq=6,
    noise_level=0.015,
):
    """
    Generate training data for diffusion with nonlinear source.

    u_t = cΔu + 15 sin(u),  c=0.3
    Domain: [0, 2π]², zero Dirichlet BC
    """
    N = grid_size
    L = 2 * np.pi
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)

    c = 0.3

    # Central difference Laplacian
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / dx ** 2

    data = np.zeros((n_samples, n_steps + 1, downsampled_size, downsampled_size))

    for s in range(n_samples):
        # Initial condition on fine grid
        u00 = np.zeros((N, N))
        X, Y = np.meshgrid(x, y, indexing='ij')
        for k in range(1, max_freq + 1):
            for l in range(1, max_freq + 1):
                if np.random.random() < 0.5:
                    continue
                amp = np.random.randn() * (1.0 / 50)
                u00 += amp * np.cos(k * X + l * Y + np.random.rand() * 2 * np.pi)
                amp = np.random.randn() * (1.0 / 50)
                u00 += amp * np.sin(k * X + l * Y + np.random.rand() * 2 * np.pi)

        # Apply boundary condition: u = 0 on boundary via envelope
        envelope = X * (L - X) * Y * (L - Y) / (L ** 4)
        u = u00 * envelope

        for step in range(n_steps + 1):
            if downsampled_size < N:
                stride = N // downsampled_size
                data[s, step] = u[::stride, ::stride]
            else:
                data[s, step] = u

            if step == n_steps:
                break

            # Forward Euler with central difference Laplacian
            # Using scipy's convolution for simplicity
            from scipy.ndimage import convolve
            laplacian_u = convolve(u, laplacian_kernel, mode='constant', cval=0.0)
            u = u + dt * (c * laplacian_u + 15 * np.sin(u))

    if noise_level > 0:
        M = np.max(np.abs(data))
        noise = np.random.randn(*data.shape) * noise_level * M
        data = data + noise

    t_points = np.arange(n_steps + 1) * dt

    return data, t_points


def create_training_batches(data, n_blocks, batch_size=28):
    """
    Create training pairs for layer-wise training.

    For n blocks, we need pairs (u(t_i), u(t_{i+n})) for various i.

    Returns:
        batches: list of lists, where batches[k] contains
                 [(input, target), ...] for training with k+1 blocks
    """
    n_samples, n_time, H, W = data.shape
    batches = []

    for nb in range(1, n_blocks + 1):
        batch = []
        for s in range(n_samples):
            for t_start in range(n_time - nb):
                inp = data[s, t_start]
                target = data[s, t_start + nb]
                batch.append((inp, target))

        # Shuffle
        np.random.shuffle(batch)

        # Group into mini-batches
        mini_batches = []
        for i in range(0, len(batch), batch_size):
            inps = np.stack([b[0] for b in batch[i:i+batch_size]])
            tgts = np.stack([b[1] for b in batch[i:i+batch_size]])
            mini_batches.append((inps, tgts))

        batches.append(mini_batches)

    return batches

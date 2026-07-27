"""
Direct least-squares coefficient identification.
For each grid point (x,y), solve:
  min_c || Σ c_ij * Δt * D_ij u - (D0 u - u_next) ||²
using spectral derivatives and all sample pairs.

This bypasses the degeneracy in gradient-based training.
"""

import numpy as np
import torch
from scipy.signal import correlate2d
from pdenet.data import generate_convection_diffusion_data
from pdenet.filters import create_constrained_filter

np.random.seed(42)
L = 2 * np.pi
N_grid = 50
dx = L / N_grid
dt_data = 0.015

# FFT setup
kx = np.fft.fftfreq(N_grid, d=dx) * 2 * np.pi
ky = np.fft.fftfreq(N_grid, d=dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')

def spectral_deriv(u, i, j):
    """Compute derivative ∂^{i+j}u/∂x^i∂y^j via FFT."""
    op = (1j * KX) ** i * (1j * KY) ** j
    return np.fft.ifft2(op * np.fft.fft2(u)).real

# Build constrained filters
N_filt = 5
filters = {}
for (i,j) in [(1,0),(0,1),(2,0),(0,2),(1,1)]:
    q0, _, _ = create_constrained_filter(N_filt, i, j)
    filters[(i,j)] = q0

def filter_deriv(u, q, i, j):
    """Compute derivative using convolution filter."""
    d = correlate2d(u, q, mode='same', boundary='wrap') / (dx**i) / (dx**j)
    return d

# Generate data (using the fixed anti-aliased data generation)
print("Generating data...")
data, t = generate_convection_diffusion_data(
    n_samples=300, n_steps=3, dt=dt_data,
    grid_size=200, downsampled_size=50, max_freq=9,
    noise_level=0.0, random_shift=True)

print(f"Data shape: {data.shape}")

# Collect all (u(t), u(t+Δt)) pairs
X_all = []
Y_all = []
for s in range(data.shape[0]):
    for t_start in range(data.shape[1] - 1):
        X_all.append(data[s, t_start])
        Y_all.append(data[s, t_start + 1])

print(f"Total pairs: {len(X_all)}")

# For each pair, compute derivatives and target
n_pairs = len(X_all)
n_grid = N_grid * N_grid

# Method 1: Spectral derivatives (ground truth) for coefficient fitting
print("\n=== Direct spectral least-squares ===")
# Build: for each grid point, Y = X @ coeff where
# Y = u_next - D0*u (the "residual" after identity part)
# X columns = dt * [u_x, u_y, u_xx, u_xy, u_yy]

# D0 = identity (since D0 filter is delta)
D0_kernel = np.zeros((N_filt, N_filt))
D0_kernel[N_filt//2, N_filt//2] = 1.0

# Compute using spectral derivatives
deriv_names = [(1,0), (0,1), (2,0), (1,1), (0,2)]
n_derivs = len(deriv_names)

# For each grid point (ix, iy), stack all samples
# System: For each sample s, Y_s = Σ c_ij * dt * D_ij u_s
# Rearranged by grid point: Y_g = [dt*u_x_g, dt*u_y_g, dt*u_xx_g, dt*u_xy_g, dt*u_yy_g] @ [c_10, c_01, c_20, c_11, c_02]_g

coeff_spectral = {}  # (i,j) -> (50,50) array
for (i,j) in deriv_names:
    coeff_spectral[(i,j)] = np.zeros((N_grid, N_grid))

for ix in range(N_grid):
    for iy in range(N_grid):
        # Build design matrix for this grid point
        A = np.zeros((n_pairs, n_derivs))
        b = np.zeros(n_pairs)

        for s_idx in range(n_pairs):
            u_in = X_all[s_idx]
            u_out = Y_all[s_idx]

            b[s_idx] = u_out[ix, iy] - u_in[ix, iy]  # u_next - u (since D0=identity)

            # Spectral derivatives
            for d_idx, (i,j) in enumerate(deriv_names):
                d = spectral_deriv(u_in, i, j)
                A[s_idx, d_idx] = dt_data * d[ix, iy]

        # Least squares: (A^T A) x = A^T b
        AtA = A.T @ A
        Atb = A.T @ b
        try:
            x = np.linalg.solve(AtA + 1e-10 * np.eye(n_derivs), Atb)
        except:
            x = np.linalg.lstsq(A, b, rcond=None)[0]

        for d_idx, (i,j) in enumerate(deriv_names):
            coeff_spectral[(i,j)][ix, iy] = x[d_idx]

    if ix % 10 == 0:
        print(f"  Processed row {ix}/{N_grid}")

# Evaluate
x = np.linspace(0, L, N_grid, False)
y = np.linspace(0, L, N_grid, False)
Xg, Yg = np.meshgrid(x, y, indexing='ij')
true = {
    (1,0): 0.5*(np.cos(Yg)+Xg*(L-Xg)*np.sin(Xg))+0.6,
    (0,1): 2*(np.cos(Yg)+np.sin(Xg))+0.8,
    (2,0): 0.2*np.ones((N_grid,N_grid)),
    (0,2): 0.3*np.ones((N_grid,N_grid)),
    (1,1): np.zeros((N_grid,N_grid)),
}

print(f"\nDirect spectral LS coefficients:")
for (i,j) in deriv_names:
    c = coeff_spectral[(i,j)]
    t = true[(i,j)]
    mae = np.abs(c - t).mean()
    print(f"  c_{{{i},{j}}}: true={t.mean():.6f}, learned={c.mean():.6f}, MAE={mae:.6f}")

# Now test: use constrained filters instead of spectral
print(f"\n=== Constrained filter least-squares ===")
coeff_filter = {}
for (i,j) in deriv_names:
    coeff_filter[(i,j)] = np.zeros((N_grid, N_grid))

for ix in range(N_grid):
    for iy in range(N_grid):
        A = np.zeros((n_pairs, n_derivs))
        b = np.zeros(n_pairs)

        for s_idx in range(n_pairs):
            u_in = X_all[s_idx]
            u_out = Y_all[s_idx]
            b[s_idx] = u_out[ix, iy] - u_in[ix, iy]

            for d_idx, (i,j) in enumerate(deriv_names):
                d = filter_deriv(u_in, filters[(i,j)], i, j)
                A[s_idx, d_idx] = dt_data * d[ix, iy]

        AtA = A.T @ A
        Atb = A.T @ b
        try:
            x = np.linalg.solve(AtA + 1e-10 * np.eye(n_derivs), Atb)
        except:
            x = np.linalg.lstsq(A, b, rcond=None)[0]

        for d_idx, (i,j) in enumerate(deriv_names):
            coeff_filter[(i,j)][ix, iy] = x[d_idx]

    if ix % 10 == 0:
        print(f"  Processed row {ix}/{N_grid}")

print(f"\nConstrained filter LS coefficients:")
for (i,j) in deriv_names:
    c = coeff_filter[(i,j)]
    t = true[(i,j)]
    mae = np.abs(c - t).mean()
    print(f"  c_{{{i},{j}}}: true={t.mean():.6f}, learned={c.mean():.6f}, MAE={mae:.6f}")

# Save
OUT = '/home/user/.clawsgo/projects/19856bd5-b20a-4d4e-8a9e-658022f9347a/workspace/output'
for prefix, coeff_dict in [('spectral', coeff_spectral), ('filter', coeff_filter)]:
    np.savez(f'{OUT}/{prefix}_coeffs.npz', **{f'c_{i}{j}': v for (i,j), v in coeff_dict.items()})

print(f"\nDone. Saved to {OUT}/")

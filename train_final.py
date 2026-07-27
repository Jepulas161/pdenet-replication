"""
PDE-Net: focus on prediction accuracy with trainable filters.
Layer-wise training on direct 50×50 spectral data.
"""

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
import os, sys, time

OUT = '/home/user/.clawsgo/projects/19856bd5-b20a-4d4e-8a9e-658022f9347a/workspace/output'
os.makedirs(OUT, exist_ok=True)

np.random.seed(42); torch.manual_seed(42)
L = 2 * np.pi; dx = L / 50; dt = 0.015

# ============ Data (direct 50×50) ============
def make_data(n_samples, n_steps):
    from numpy.fft import fft2, ifft2, fftfreq
    N = 50
    x = np.linspace(0, L, N, False)
    y = np.linspace(0, L, N, False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    kx = fftfreq(N, d=dx) * 2 * np.pi; ky = fftfreq(N, d=dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    a = 0.5 * (np.cos(Y) + X * (L - X) * np.sin(X)) + 0.6
    b = 2 * (np.cos(Y) + np.sin(X)) + 0.8
    nu1, nu2 = 0.2, 0.3

    def trunc(uh, m=20):
        h = m // 2; o = np.zeros_like(uh)
        o[:h,:h]=uh[:h,:h]; o[-h:,:h]=uh[-h:,:h]; o[:h,-h:]=uh[:h,-h:]; o[-h:,-h:]=uh[-h:,-h:]
        return o

    def rhs(uh):
        u = np.real(ifft2(uh)); ux=np.real(ifft2(1j*KX*uh)); uy=np.real(ifft2(1j*KY*uh))
        uxx=np.real(ifft2(-KX**2*uh)); uyy=np.real(ifft2(-KY**2*uh))
        return trunc(fft2(a*ux+b*uy+nu1*uxx+nu2*uyy), 20)

    data = np.zeros((n_samples, n_steps+1, N, N))
    for s in range(n_samples):
        u0 = np.zeros((N, N))
        for k in range(1, 6):
            for l in range(1, 6):
                if np.random.random() < 0.4: continue
                u0 += np.random.randn()/50 * np.cos(k*X+l*Y+np.random.rand()*2*np.pi)
                u0 += np.random.randn()/50 * np.sin(k*X+l*Y+np.random.rand()*2*np.pi)
        uh = trunc(fft2(u0), 20)
        for step in range(n_steps+1):
            data[s,step] = np.real(ifft2(uh))
            if step == n_steps: break
            k1=rhs(uh); k2=rhs(uh+0.5*dt*k1); k3=rhs(uh+0.5*dt*k2); k4=rhs(uh+dt*k3)
            uh = uh + (dt/6)*(k1+2*k2+2*k3+k4); uh = trunc(uh, 20)
    return data

print("Generating data...")
t0 = time.time()
train_data = make_data(300, 20)
test_data = make_data(14, 40)
print(f"Train: {train_data.shape}, Test: {test_data.shape}, time={time.time()-t0:.0f}s")

# ============ Model ============
from pdenet.model import PDENet

model = PDENet(5, max_order=2, n_blocks=20, use_nn=False, grid_shape=(50,50), dx=dx)
print(f"Deriv pairs: {model.block.deriv_pairs}")

# D0 is identity — freeze it
model.block.D0.theta.requires_grad_(False)

# Unfreeze D_ij thetas for training
for n, p in model.named_parameters():
    if 'theta' in n and 'D0' not in n:
        p.requires_grad_(True)

tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {tr}")

# ============ Layer-wise training ============
def train_step(model, data, n_steps, n_iter, bs=64, lr=0.0005):
    """Train for n_steps-ahead prediction."""
    X, Y = [], []
    for s in range(data.shape[0]):
        for t in range(data.shape[1] - n_steps):
            X.append(data[s,t]); Y.append(data[s,t+n_steps])
    X_t = torch.from_numpy(np.stack(X)).float().unsqueeze(1)
    Y_t = torch.from_numpy(np.stack(Y)).float().unsqueeze(1)
    N = len(X_t)
    print(f"  Pairs: {N}, batch={bs}, lr={lr}")

    opt = optim.Adam(model.parameters(), lr=lr)
    best = float('inf'); t0 = time.time()
    for i in range(n_iter):
        idx = np.random.choice(N, min(bs, N), replace=False)
        opt.zero_grad()
        pred = model(X_t[idx], dt=1.0, n_steps=n_steps)
        loss = nn.MSELoss()(pred, Y_t[idx])
        loss.backward()
        opt.step()
        if loss.item() < best: best = loss.item()
        if i % 500 == 0:
            print(f"    iter {i}: loss={loss.item():.6e}, best={best:.6e}, time={time.time()-t0:.0f}s")
    return best

schedule = [(1, 1000, 0.001), (2, 1000, 0.001), (4, 1500, 0.0005), (6, 2000, 0.0005), (8, 2000, 0.0003), (10, 2000, 0.0003)]
for n_steps, n_iter, lr in schedule:
    print(f"\n--- n_steps={n_steps}, {n_iter} iters ---")
    loss = train_step(model, train_data, n_steps, n_iter, lr=lr)
    print(f"  Best: {loss:.6e}")

# ============ Evaluate ============
print(f"\n{'='*60}")
print("Evaluation")
print(f"{'='*60}")

# Coefficients
coeff = model.get_coefficient_fields()
x = np.linspace(0,L,50,False); y = np.linspace(0,L,50,False)
Xg,Yg = np.meshgrid(x,y,indexing='ij')
true = {
    (1,0): 0.5*(np.cos(Yg)+Xg*(L-Xg)*np.sin(Xg))+0.6,
    (0,1): 2*(np.cos(Yg)+np.sin(Xg))+0.8,
    (2,0): 0.2*np.ones((50,50)),
    (0,2): 0.3*np.ones((50,50)),
    (1,1): np.zeros((50,50)),
}
print("Coefficients:")
for (i,j) in [(1,0),(0,1),(2,0),(1,1),(0,2)]:
    l = coeff.get((i,j),np.zeros((50,50)))
    t = true.get((i,j),np.zeros((50,50)))
    mae = np.abs(l-t).mean()
    print(f"  c_{{{i},{j}}}: true={t.mean():.6f}, learned={l.mean():.6f}, MAE={mae:.6f}")

# Long-term prediction
print(f"\nLong-term prediction:")
model.eval()
errors = []
u = torch.from_numpy(test_data[:,0]).float().unsqueeze(1)
for step in range(min(40, test_data.shape[1]-1)):
    with torch.no_grad():
        u = model.block(u, dt=1.0)
    target = torch.from_numpy(test_data[:,step+1]).float().unsqueeze(1)
    err = ((u-target)**2).mean().sqrt().item()
    errors.append(err)
    if step <= 4 or step % 10 == 9:
        print(f"  Step {step+1}: RMSE={err:.4f}")
print(f"  Mean RMSE (all): {np.mean(errors):.4f}")
print(f"  First 10 mean: {np.mean(errors[:10]):.4f}")
print(f"  Final step: {errors[-1]:.4f}")

torch.save(model.state_dict(), os.path.join(OUT, 'pdenet_final.pt'))
print(f"\nSaved to {OUT}/pdenet_final.pt")

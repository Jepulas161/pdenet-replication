"""
PDE-Net 复现结果可视化
"""
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pdenet.model import PDENet
from numpy.fft import fft2, ifft2, fftfreq

L=2*np.pi; dx=L/50; dt=0.015; N=50

model=PDENet(5,2,20,use_nn=False,grid_shape=(50,50),dx=dx)
model.load_state_dict(torch.load('/home/user/.clawsgo/projects/19856bd5-b20a-4d4e-8a9e-658022f9347a/workspace/output/pdenet_extended.pt'))

x=np.linspace(0,L,N,False); y=np.linspace(0,L,N,False)
X,Y=np.meshgrid(x,y,indexing='ij')
kx=fftfreq(N,d=dx)*2*np.pi; ky=fftfreq(N,d=dx)*2*np.pi
KX,KY=np.meshgrid(kx,ky,indexing='ij')
a=0.5*(np.cos(Y)+X*(L-X)*np.sin(X))+0.6; b=2*(np.cos(Y)+np.sin(X))+0.8; nu1,nu2=0.2,0.3

def trunc(uh,m=20):
    h=m//2; o=np.zeros_like(uh); o[:h,:h]=uh[:h,:h]; o[-h:,:h]=uh[-h:,:h]; o[:h,-h:]=uh[:h,-h:]; o[-h:,-h:]=uh[-h:,-h:]; return o
def rhs(uh):
    u=np.real(ifft2(uh)); ux=np.real(ifft2(1j*KX*uh)); uy=np.real(ifft2(1j*KY*uh))
    uxx=np.real(ifft2(-KX**2*uh)); uyy=np.real(ifft2(-KY**2*uh))
    return trunc(fft2(a*ux+b*uy+nu1*uxx+nu2*uyy),20)
def make_test_data(n_samples,n_steps):
    data=np.zeros((n_samples,n_steps+1,N,N))
    for s in range(n_samples):
        u0=np.zeros((N,N))
        np.random.seed(200+s)
        for k in range(1,6):
            for l in range(1,6):
                if np.random.random()<0.4: continue
                u0+=np.random.randn()/50*np.cos(k*X+l*Y+np.random.rand()*2*np.pi)
                u0+=np.random.randn()/50*np.sin(k*X+l*Y+np.random.rand()*2*np.pi)
        uh=trunc(fft2(u0),20)
        for step in range(n_steps+1):
            data[s,step]=np.real(ifft2(uh))
            if step==n_steps: break
            k1=rhs(uh); k2=rhs(uh+0.5*dt*k1); k3=rhs(uh+0.5*dt*k2); k4=rhs(uh+dt*k3)
            uh=uh+(dt/6)*(k1+2*k2+2*k3+k4); uh=trunc(uh,20)
    return data

np.random.seed(42)
test_data=make_test_data(7,30)

# Generate predictions
model.eval()
preds=[]; u=torch.from_numpy(test_data[:,0]).float().unsqueeze(1)
for step in range(30):
    with torch.no_grad(): u=model.block(u,dt=1.0)
    preds.append(u.numpy())
preds=np.stack(preds,axis=1).squeeze()  # (7, 30, 50, 50)

OUT = '/home/user/.clawsgo/projects/19856bd5-b20a-4d4e-8a9e-658022f9347a/workspace/output'

# Figure 1: Prediction vs Ground Truth (sample 0)
fig,axes=plt.subplots(2,5,figsize=(16,6))
steps_plot=[0,4,9,14,19,5,10,15,20,25]
for idx,st in enumerate(steps_plot):
    ax=axes[idx//5,idx%5]
    vmin=min(test_data[0,st].min(),preds[0,st].min())
    vmax=max(test_data[0,st].max(),preds[0,st].max())
    if idx<5:
        im=ax.imshow(test_data[0,st],cmap='RdBu_r',vmin=vmin,vmax=vmax)
        ax.set_title(f'True t={st*dt:.3f}')
    else:
        im=ax.imshow(preds[0,st-5],cmap='RdBu_r',vmin=vmin,vmax=vmax)
        ax.set_title(f'Pred t={(st-5)*dt:.3f}')
    plt.colorbar(im,ax=ax,fraction=0.046)
    ax.set_xlabel('x'); ax.set_ylabel('y')
plt.suptitle('PDE-Net: True vs Predicted (sample 0)',fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/pdenet_prediction.png',dpi=150)
plt.close()

# Figure 2: RMSE over time
model2=PDENet(5,2,20,use_nn=False,grid_shape=(50,50),dx=dx)
model2.load_state_dict(torch.load('/home/user/.clawsgo/projects/19856bd5-b20a-4d4e-8a9e-658022f9347a/workspace/output/pdenet_final.pt'))
model2.eval()

fig,ax=plt.subplots(figsize=(10,5))
for mdL,name in [(model2,'10-block'),(model,'18-block extended')]:
    mdL.eval()
    u=torch.from_numpy(test_data[:,0]).float().unsqueeze(1)
    errs=[]
    for step in range(30):
        with torch.no_grad(): u=mdL.block(u,dt=1.0)
        target=torch.from_numpy(test_data[:,step+1]).float().unsqueeze(1)
        err=((u-target)**2).mean().sqrt().item(); errs.append(err)
    t_vals=np.arange(1,len(errs)+1)*dt
    ax.semilogy(t_vals,errs,'o-',label=name)
ax.set_xlabel('Time'); ax.set_ylabel('RMSE'); ax.legend(); ax.grid(True,alpha=0.3)
ax.set_title('PDE-Net Prediction Error vs Time')
plt.tight_layout()
plt.savefig(f'{OUT}/pdenet_error.png',dpi=150)
plt.close()

# Figure 3: Coefficients comparison
coeff=model.get_coefficient_fields()
fig,axes=plt.subplots(3,5,figsize=(18,10))
Xg,Yg=np.meshgrid(x,y,indexing='ij')
true_dt={(1,0):dt*(0.5*(np.cos(Yg)+Xg*(L-Xg)*np.sin(Xg))+0.6),(0,1):dt*(2*(np.cos(Yg)+np.sin(Xg))+0.8),(2,0):dt*0.2*np.ones((50,50)),(0,2):dt*0.3*np.ones((50,50)),(1,1):np.zeros((50,50))}
pairs=[(1,0),(0,1),(2,0),(0,2),(1,1)]
names=['c_{10} (advection x)','c_{01} (advection y)','c_{20} (diffusion x)','c_{02} (diffusion y)','c_{11} (mixed)']
for idx,(i,j) in enumerate(pairs):
    t=true_dt[(i,j)]; l=coeff.get((i,j),np.zeros((50,50)))
    vmax=max(np.abs(t).max(),np.abs(l).max())
    ax=axes[0,idx]; im=ax.imshow(t,cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    ax.set_title(f'True {names[idx]}'); plt.colorbar(im,ax=ax,fraction=0.046)
    ax=axes[1,idx]; im=ax.imshow(l,cmap='RdBu_r',vmin=-vmax,vmax=vmax)
    ax.set_title(f'Learned {names[idx]}'); plt.colorbar(im,ax=ax,fraction=0.046)
    ax=axes[2,idx]; im=ax.imshow(np.abs(l-t),cmap='hot',vmin=0)
    ax.set_title(f'Error'); plt.colorbar(im,ax=ax,fraction=0.046)
plt.suptitle('Coefficient Identification (PDE-Net)',fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUT}/pdenet_coefficients.png',dpi=150)
plt.close()

print(f'Figures saved to {OUT}/')
print(f'  1. {OUT}/pdenet_prediction.png')
print(f'  2. {OUT}/pdenet_error.png')
print(f'  3. {OUT}/pdenet_coefficients.png')

"""
 Straining flow and plane wave.
"""
import timeit
start = timeit.default_timer()

import matplotlib.pyplot as plt
plt.rcParams['contour.negative_linestyle'] = 'dashed'
import numpy as np
import h5py

from niwqg import CoupledModel as Model
from niwqg import InitialConditions as ic

import cmocean

plt.close('all')

patho = "outputs/straining"

# parameters
nx = 512
f0 = 1.e-4
N = 0.005
L = 2*np.pi*200e3
λz = 325
m = 2*np.pi/λz
nu4, nu4w = 5e7, 1.e7   # hyperviscosity

# initial conditions
Ue = 5.e-2
Uw = 10*Ue
ke = 15*(2*np.pi/L)
Le = 2*np.pi/ke

# relevant parameters
Te = (Ue*ke)**-1 # eddy turn-over time scale

lam2 = (N/f0/m)**2
h = f0*lam2
hslash = h/(Ue/ke)
Ro = Ue*ke/f0
alpha = Ro*( (Uw/Ue)**2 )

# simulation parameters
dt = .0025*Te
tmax = 0.75*Te

## setup model class
model = Model.Model(L=L,nx=nx, tmax = tmax,dt = dt,
                m=m,N=N,f=f0, twrite=int(0.1*Te/dt),
                nu4=nu4*0,nu4w=nu4w*0,nu=0,nuw=0, mu=0,muw=0,use_filter=True,
                U =-Ue, tdiags=10,
                save_to_disk=True,tsave_snapshots=25, path=patho)

# a quasi-straining flow
psi = -np.sin(2*np.pi*model.x/model.L)*np.sin(2*np.pi*model.y/model.L)
psi_hat = np.fft.fft2(psi)
q = np.fft.ifft2(-model.wv2*psi_hat).real/(Ue*ke)

# a wavepacket
phi = Uw*ic.WavePacket(model, k=ke, l=ke, R=1.5*Le,
                              x0=model.x.mean(), y0=model.x.mean())

model.set_q(q)
model.set_phi(phi)

## run the model
model.run()

stop = timeit.default_timer()
print("Time elapsed: %3.2f seconds" %(stop - start))

## save parameter
fno = model.path+"/parameters.h5"
h5file = h5py.File(fno, 'w')

h5file.create_dataset("dimensional/Ue", data=(Ue))
h5file.create_dataset("dimensional/Uw", data=(Uw))
h5file.create_dataset("dimensional/ke", data=(ke))
h5file.create_dataset("dimensional/m", data=(m))
h5file.create_dataset("dimensional/N", data=(N))
h5file.create_dataset("dimensional/f0", data=(f0))
h5file.create_dataset("dimensional/Te", data=(Te))
h5file.create_dataset("dimensional/L", data=(L))
h5file.create_dataset("dimensional/nu4", data=(nu4))
h5file.create_dataset("dimensional/nu4w", data=(nu4w))
h5file.create_dataset("nondimensional/nx", data=(nx))
h5file.create_dataset("nondimensional/alpha", data=(alpha))
h5file.create_dataset("nondimensional/hslash", data=(hslash))
h5file.create_dataset("nondimensional/Ro", data=(Ro))

h5file.close()

# plot Gamma_a
psi = np.fft.ifft2(model.ph).real
phix, phiy = np.fft.ifft2(-model.ik*model.phih), np.fft.ifft2(-model.il*model.phih)
lapphix,lapphiy = np.fft.ifft2(-model.wv2*model.ik*model.phih), np.fft.ifft2(-model.wv2*model.il*model.phih)

J_phi_lapphi = phix*lapphiy - phiy*lapphix
Gamma_a = lam2*((J_phi_lapphi)).real

x, y = model.x-model.x.mean(),model.y-model.y.mean()
x, y = x/Le, y/Le

psih = np.fft.fft2(psi)
u, v = np.fft.ifft2(-model.il*psih).real, np.fft.ifft2(model.ik*psih).real

psi = psi/psi.max()
u, v = u/u.max(),v/v.max()


phi_2 = model.phi

cp = np.hstack([np.array([-.45,-.3,-.15,-.05]),np.array([0.05,0.15,0.3,.45])])
cq = np.linspace(-2.,2.,10)

phi = np.ma.masked_array(phi,np.abs(phi.real)<1e-3)
phi_2 = np.ma.masked_array(phi_2,np.abs(phi_2.real)<1e-3)

fig = plt.figure(figsize=(5,3))

ax1 = fig.add_subplot(121,aspect=1)

plt.contour(x,y,psi,cp,colors='k')
#plt.streamplot(x,y, u, v, density=2.,color='k', linewidth=2)

plt.contourf(x,y,phi.real/phi.real.max(),cq,cmap=cmocean.cm.curl,extend='both')

plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xticks([])
plt.yticks([])
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

plt.text(-2.05,2.01,r"$t \times \alpha/2\pi = 0$")

ax2 = fig.add_subplot(122,aspect=1)
plt.contour(x,y,psi,cp,colors='k')
plt.contourf(x,y,phi_2.real/phi.real.max(),cq,cmap=cmocean.cm.curl,
                extend='both')

#plt.streamplot(x,y, u, v, density=2.,color='k', linewidth=2)


plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xticks([])
plt.yticks([])
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

plt.text(-2.05,2.01,r"$t \times \alpha/2\pi = 0.75$")

plt.savefig("../meetings/munk100/poster/figs/Gamma_a.pdf",
            bbox_inches='tight',
            transparent=True,
            pad_inches=0)

#
# # figure for presentation
# plt.style.use('dark_background')
#
#
# fig = plt.figure(figsize=(5,3))
#
# ax1 = fig.add_subplot(121,aspect=1)
#
# plt.contour(x,y,psi,cp,colors='w')
# #plt.streamplot(x,y, u, v, density=2.,color='k', linewidth=2)
# phi = np.ma.masked_array(phi,np.abs(phi.real)<1e-2)
# plt.contourf(x,y,phi.real/phi.real.max(),cq,cmap=cmocean.cm.curl,extend='both')
#
# plt.xlim(-2,2)
# plt.ylim(-2,2)
# plt.xticks([])
# plt.yticks([])
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
# ax1.spines['left'].set_visible(False)
#
# plt.text(-2.05,2.01,r"$t \times \alpha/2\pi = 0$")
#
# ax2 = fig.add_subplot(122,aspect=1)
# plt.contour(x,y,psi,cp,colors='k')
# plt.contourf(x,y,phi_2.real/phi.real.max(),cq,cmap=cmocean.cm.curl,
#                 extend='both')
#
# #plt.streamplot(x,y, u, v, density=2.,color='k', linewidth=2)
#
#
# plt.xlim(-2,2)
# plt.ylim(-2,2)
# plt.xticks([])
# plt.yticks([])
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)
# ax2.spines['left'].set_visible(False)
#
# plt.text(-2.05,2.01,r"$t \times \alpha/2\pi = 0.75$")
#
# plt.savefig("../meetings/munk100/poster/figs/Gamma_a_black.pdf",
#             bbox_inches='tight',
#             pad_inches=0)

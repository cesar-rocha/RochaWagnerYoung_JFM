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
tmax = 1*Te

## setup model class
model = Model.Model(L=L,nx=nx, tmax = tmax,dt = dt,
                m=m,N=N,f=f0, twrite=int(0.1*Te/dt),
                nu4=nu4*0,nu4w=nu4w*0,nu=0,nuw=0, mu=0,muw=0,use_filter=True,
                U =-Ue, tdiags=10,
                save_to_disk=True,tsave_snapshots=25, path=patho)

# initial conditions
#q = ic.LambDipole(model, U=Ue,R = 2*np.pi/ke)

#psi = -alpha*(model.x-model.x.mean())*(model.y-model.y.mean())
#psi *= np.hanning(nx)[np.newaxis,:] * np.hanning(nx)[:,np.newaxis]

# a quasi-straining flow
psi = np.sin(2*np.pi*model.x/model.L)*np.sin(2*np.pi*model.y/model.L)
psi_hat = np.fft.fft2(psi)
q = np.fft.ifft2(-model.wv2*psi_hat).real/(Ue*ke)

# a wavepacket
phi = Uw*ic.WavePacket(model, k=2*ke, l=2*ke, R=2*Le, x0=model.x.mean(), y0=model.x.mean())


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

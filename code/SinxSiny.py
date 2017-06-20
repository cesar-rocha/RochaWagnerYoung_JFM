"""
 Initially laterally coherent near-inertial oscillation
    coupled with SinxSiny flow

 Notes:

"""
import timeit
start = timeit.default_timer()

import matplotlib.pyplot as plt
plt.rcParams['contour.negative_linestyle'] = 'dashed'
import numpy as np
import h5py

from niwqg import CoupledModel as Model
#from niwqg import QGModel as Model
from niwqg import InitialConditions as ic

plt.close('all')

patho = "outputs/sinxsiny_wavepacket"
# parameters
nx = 256*4
f0 = 1.e-4
N = 0.005
L = 2*np.pi*200e3
#λz = 2000
#λz = 4000
λz = 225

m = 2*np.pi/λz
#nu4, nu4w = 1e10, 3.5e9   # hyperviscosity
nu4, nu4w = 1e7, 1e7 # hyperviscosity

# initial conditions
Ue = 0.05
Uw = 6.3*Ue
ke = 1*(2*np.pi/L)
Le = 2*np.pi/ke

# relevant parameters
Te = (Ue*ke)**-1 # eddy turn-over time scale

lam2 = (N/f0/m)**2
h = f0*lam2
hslash = h/(Ue/ke)
Ro = Ue*ke/f0
alpha = Ro*( (Uw/Ue)**2 )

# simulation parameters
#dt = .0025*Te
dt = .0025*Te/4
tmax = 10*Te

## setup model class
model = Model.Model(L=L,nx=nx, tmax = tmax,dt = dt,
                 m=m,N=N,f=f0, twrite=int(0.1*Te/dt),
                 nu4=nu4*0,nu4w=nu4w*0,nu=0,nuw=0, mu=0,muw=0,use_filter=True,
                 U =0, tdiags=10,
                 save_to_disk=True,tsave_snapshots=100, path=patho)

#model = Model.Model(L=L,nx=nx, tmax = tmax,dt = dt,
#                twrite=int(0.1*Te/dt),
#                nu4=nu4, use_filter=True,
#                U =0, tdiags=10,
#                save_to_disk=True,tsave_snapshots=10, path=patho,passive_scalar=True)


# initial conditions
#q = ic.LambDipole(model, U=Ue,R = 2*np.pi/ke)
p = (Ue/ke)*( np.sin(ke*model.x) + np.sin(ke*model.y) )
q = -(ke**2)*p
phi = (np.ones_like(q) + 1j)*Uw/np.sqrt(2)
phi1 = Uw*ic.WavePacket(model, k=10*ke, l=0*ke, R=L/8,
                              x0=model.x.mean()+L/4, y0=model.x.mean()-L/4)
phi = phi1

model.set_q(q)
model.set_phi(phi)
#model.set_c(phi.real)

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

"""
 Initially laterally coherent near-inertial oscillation
    decaying macroturbulence emergent from random initial
    conditions with prescribed spectrum (McWilliams, 1984)

 Notes:
    - It took ...

"""
import timeit
start = timeit.default_timer()

import matplotlib.pyplot as plt
plt.rcParams['contour.negative_linestyle'] = 'dashed'
import numpy as np
import h5py

from niwqg import CoupledModel as CoupledModel
from niwqg import QGModel as QGModel
from niwqg import InitialConditions as ic

plt.close('all')

patho_qg = "outputs/decaying_turbulence/qg_initial_condition_for_uncoupled"
patho = "outputs/decaying_turbulence/reference/"

# parameters
nx = 512
f0 = 1.e-4
N = 0.005
L = 2*np.pi*200e3

#λz = 198.75  # hslash = 0.25
λz = 400  # hslash = 1
#λz = 794.8  # hslash = 4

m = 2*np.pi/λz
nu4, nu4w = 3.5e7, 4.25e6 # hyperviscosity

# initial conditions
Ue = 5.e-2
Uw = 2*Ue
ke = 10*(2*np.pi/L)
Le = np.sqrt(2)*np.pi/ke

# relevant parameters
Te = (Ue*ke)**-1 # eddy turn-over time scale
lam2 = (N/f0/m)**2
h = f0*lam2
hslash = h/(Ue/ke)
Ro = Ue*ke/f0
alpha = Ro*( (Uw/Ue)**2 )


#
# First run the QG model to 20 eddy-turnover time units
#

# qg simulation parameters
#dt = .01*Te
#tmax = 20*Te
#
#qgmodel = QGModel.Model(L=L,nx=nx, tmax = tmax,dt = dt,
#                twrite=int(0.1*Te/dt),
#                nu4=nu4, use_filter=False,
#                U =-Ue, tdiags=1,
#                save_to_disk=True,tsave_snapshots=25, path=patho_qg,)#use_fftw=True, fftw_nthreads=3)
#
## initial conditions
#q = ic.McWilliams1984(qgmodel, E=(Ue**2)/2,k0=ke)
#qgmodel.set_q(q)
#
#qgmodel.run()

dt = .0025*Te
tmax = 100*Te


#
# setup model classes
#

coupledmodel = CoupledModel.Model(L=L,nx=nx, tmax = tmax,dt = dt,
                m=m,N=N,f=f0, twrite=int(0.1*Te/dt),
                nu4=nu4,nu4w=nu4w,nu=0, nuw=0, mu=0, muw=0, use_filter=False,
                U =-Ue, tdiags=10, save_to_disk=True,tsave_snapshots=25, path=patho+"coupled/",)

coupledmodel.logger.info(" ")

#
# initial conditions
#
qinit = np.load("q_init_512.npz")
q = qinit['q']
phi = (np.ones_like(q) + 1j)*Uw/np.sqrt(2)

#model.set_q(qgmodel.q)
coupledmodel.set_q(q)
coupledmodel.set_phi(phi)

#
# run the model
#

coupledmodel.logger.info(" Running coupled model")
coupledmodel.run()


stop = timeit.default_timer()
print("Time elapsed: %3.2f seconds" %(stop - start))


#
# save parameters
#
fno = patho+"parameters.h5"
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


"""
 Initially laterally coherent near-inertial oscillation
    coupled with Lamb dipole.
"""
import timeit
start = timeit.default_timer()

import matplotlib.pyplot as plt
plt.rcParams['contour.negative_linestyle'] = 'dashed'
import numpy as np

from niwqg import CoupledModel as Model
from niwqg import InitialConditions as ic

plt.close('all')

# parameters
nx = 512
f0 = 1.e-4
N = 0.01
L = 2*np.pi*200e3
λz = 280
m = 2*np.pi/λz
nu4, nu4w = 5e7, 5e7 # hyperviscosity

# initial conditions
Ue = 1.e-1
Uw = 3*Ue
ke = 10*(2*np.pi/L)
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
tmax = 10*Te

model = Model.Model(L=L,nx=nx, tmax = tmax,dt = dt,
                m=m,N=N,f=f0, twrite=int(0.1*Te/dt),
                nu4=nu4,nu4w=nu4w,use_filter=False,
                U =-Ue, tdiags=10,save_to_disk=True)

# initial conditions
q = ic.LambDipole(model, U=Ue,R = 2*np.pi/ke)
phi = (np.ones_like(q) + 1j)*Uw/np.sqrt(2)

model.set_q(q)
model.set_phi(phi)

# run the model
model.run()

stop = timeit.default_timer()
print("Time elapsed: %3.2f seconds" %(stop - start))



# plot stuff

# get diagnostics
time = model.diagnostics['time']['value']
KE_qg = model.diagnostics['ke_qg']['value']
PE_niw = model.diagnostics['pe_niw']['value']
KE_niw = model.diagnostics['ke_niw']['value']
ENS_qg = model.diagnostics['ens']['value']

g1 = model.diagnostics['gamma_r']['value']
g2 = model.diagnostics['gamma_a']['value']
pi = model.diagnostics['pi']['value']
cKE_niw = model.diagnostics['cke_niw']['value']
iKE_niw = model.diagnostics['ike_niw']['value']

ep_phi = model.diagnostics['ep_phi']['value']
ep_psi = model.diagnostics['ep_psi']['value']
chi_q =  model.diagnostics['chi_q']['value']
chi_phi =  model.diagnostics['chi_phi']['value']

dt = time[1]-time[0]
dPE = np.gradient(PE_niw,dt)
dKE = np.gradient(KE_qg,dt)
diKE_niw = np.gradient(iKE_niw,dt)

res_ke = dKE-(-g1-g2+ep_psi)
res_pe = dPE-g1-g2-chi_phi

fig = plt.figure(figsize=(16,9))
lw, alp = 3.,.5
KE0 = KE_qg[0]

ax = fig.add_subplot(221)
plt.plot(time/Te,KE_qg/KE0,label='KE QG',linewidth=lw,alpha=alp)
plt.plot(time/Te,KE_niw/KE_niw[0],label='KE NIW',linewidth=lw,alpha=alp)
plt.plot(time/Te,ENS_qg/ENS_qg[0],label='ENS QG',linewidth=lw,alpha=alp)
plt.xticks([])
plt.ylabel(r'Energy/Enstrophy $[E/E_0, Z/Z_0]$')
plt.legend(loc=3)

ax = fig.add_subplot(222)
plt.plot(time/Te,(KE_qg-KE_qg[0])/KE0,label='KE QG',linewidth=lw,alpha=alp)
plt.plot(time/Te,(PE_niw-PE_niw[0])/KE0,label='PE NIW',linewidth=lw,alpha=alp)
plt.plot(time/Te,(KE_niw-KE_niw[0])/KE0,label='KE NIW',linewidth=lw,alpha=alp)
plt.xticks([])
plt.ylabel(r'Energy  change $[(E-E_0) \times {2}/{U_0^2} ]$')
plt.legend(loc=3)

ax = fig.add_subplot(223)
plt.plot(time/Te,Te*g1/KE0,label=r'Refrac. conversion $\Gamma_r$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*g2/KE0,label=r'Adv. conversion $\Gamma_a$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*chi_phi/KE0,label=r'PE NIW diss. $\chi_\phi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*(g1+g2+chi_phi)/KE0,label=r'$(\Gamma_r+\Gamma_a+\chi_\phi)$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*dPE/KE0,'k--',label=r'PE NIW tendency $\dot K_e$',linewidth=lw,alpha=alp)
plt.legend(loc=3,ncol=2)
plt.xlabel(r"Time [$t \times U_0 k_0$]")
plt.ylabel(r'Power $[\dot E \times {2 k_0}/{U_0} ]$')

ax = fig.add_subplot(224)
plt.plot(time/Te,Te*pi/KE0,label=r'Inc. KE NIW conversion $\Pi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*ep_psi/KE0,label=r'KE NIW disspation $\epsilon_\phi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*(pi+ep_phi)/KE0,label=r'$\pi+\epsilon_\phi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*diKE_niw/KE0,'k--',label=r'Inc. NIW KE tendency',linewidth=lw,alpha=alp)
plt.xlabel(r"Time [$t \times U_0 k_0$]")
plt.ylabel(r'Power $[\dot E \times {2 k_0}/{U_0} ]$')
plt.legend(loc=1)

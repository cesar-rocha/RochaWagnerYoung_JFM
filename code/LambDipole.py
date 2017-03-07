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
N = 0.005
L = 2*np.pi*200e3
λz = 325
m = 2*np.pi/λz
nu4, nu4w = 5e7, 5e7 # hyperviscosity

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
tmax = 25*Te

model = Model.Model(L=L,nx=nx, tmax = tmax,dt = dt,
                m=m,N=N,f=f0, twrite=int(0.1*Te/dt),
                nu4=nu4,nu4w=nu4w,use_filter=False,
                U =-Ue, tdiags=10,
                save_to_disk=True,tsave_snapshots=25, path="output/LambdDipole")

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

conc_niw =  model.diagnostics['conc_niw']['value']
skew =  model.diagnostics['skew']['value'] # take a look at this in the source


dt = time[1]-time[0]
dPE = np.gradient(PE_niw,dt)
dKE = np.gradient(KE_qg,dt)
diKE_niw = np.gradient(iKE_niw,dt)

res_ke = dKE-(-g1-g2+ep_psi)
res_pe = dPE-g1-g2-chi_phi


#
# Plots
#

def plot_fig_label(xc=.95, yc=0.075 ,label="a"):
    plt.text(0.95, 0.1,label,
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes,bbox=dict(boxstyle='circle',facecolor='white'))

fig = plt.figure(figsize=(8.5,6.))
lw, alp = 3.,.5
KE0 = KE_qg[0]

ax = fig.add_subplot(221)
plt.plot(time/Te,(KE_qg-KE_qg[0])/KE0,label=r"$K_e$",linewidth=lw,alpha=alp)
plt.plot(time/Te,(PE_niw-PE_niw[0])/KE0,label=r'$P_w$',linewidth=lw,alpha=alp)
#plt.plot(time/Te,(KE_niw-KE_niw[0])/KE0,label='KE NIW',linewidth=lw,alpha=alp)
plt.xticks([])
#plt.text(0.0,0.425,'a',bbox=dict(boxstyle='circle',facecolor='white'))
plot_fig_label(label="a")
plt.ylabel(r'Energy  change $[(E-E_0) \times {2}/{U_e^2} ]$')
plt.legend(loc=3)
plt.plot([0,tmax/Te],[0]*2,'--',color="0.5")
fig.subplots_adjust(wspace=.3)

ax = fig.add_subplot(222)
plt.plot(time/Te,Te*g1/KE0,label=r'$\Gamma_r$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*g2/KE0,label=r'$\Gamma_a$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*chi_phi/KE0,label=r'$\chi_\phi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*(g1+g2+chi_phi)/KE0,label=r'$\Gamma_r+\Gamma_a+\chi_\phi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*dPE/KE0,'k--',label=r'$\dot P_w$',linewidth=lw,alpha=alp)
plt.legend(loc=1,ncol=2)
plt.xticks([])
#plt.text(0.0,0.0695,'b',bbox=dict(boxstyle='circle',facecolor='white'))
plot_fig_label(label="b")
plt.ylabel(r'Power $[\dot E \times {2 k_e}/{U_e} ]$')
#plt.plot([0,tmax/Te],[0]*2,'--',color="0.5")

ax = fig.add_subplot(223)
plt.plot(time/Te,Te*pi/KE0,label=r'$\Pi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*ep_psi/KE0,label=r'$\epsilon_\phi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*(pi+ep_phi)/KE0,label=r'$\Pi+\epsilon_\phi$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*diKE_niw/KE0,'k--',label=r'$\dot K_w^i$'
                ,linewidth=lw,alpha=alp)
plt.xlabel(r"Time [$t \times U_e k_e$]")
plt.ylabel(r'Power $[\dot E \times {2 k_e}/{U_e} ]$')
plt.legend(loc=1,ncol=2)
#plt.text(0.0,11.35,'c',bbox=dict(boxstyle='circle',facecolor='white'))
plot_fig_label(label="c")

#plt.plot([0,tmax/Te],[0]*2,'--',color="0.5")

fig.subplots_adjust(hspace=.125)

ax = fig.add_subplot(224)
p1 = ax.plot(time/Te,conc_niw,linewidth=lw,alpha=alp,label='NIW concentration, $C$')
plt.ylabel(r"NIW concentration, C")
#ax2 = ax.twinx()
#p2 = ax2.plot(time/Te,skew,linewidth=lw,alpha=alp,label='Rel. vorticity skewness, $S$')
#plt.xlabel(r"Time [$t \times U_e k_e$]")
#plt.ylabel(r"Vorticity skewness")
#plt.legend(loc=4)
plt.plot([0,tmax/Te],[0]*2,'--',color="0.5")
#plt.text(0.0,0.38,'d',bbox=dict(boxstyle='circle',facecolor='white'))
plot_fig_label(label="d")

#plt.savefig("figs/fig1.pdf", bbox_inches='tight')

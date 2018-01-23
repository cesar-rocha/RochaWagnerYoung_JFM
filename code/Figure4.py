""""
    Plots figure 4: energy time series, budgets,
                    wave-vorticity correlation
                    of the decaying macroturbulence solution.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

from Utils import *

plt.close('all')

pathi = "outputs/decaying_turbulence/reference/coupled/"
pathi = "outputs/high_res/decaying_turbulence/parameter_exploration/Uw0.1/lambdaz397.5/"

patho = "../writeup/figs/"

params = h5py.File(pathi+"parameters.h5","r")
diags = h5py.File(pathi+"diagnostics.h5")

## get params
Ue, ke = params['dimensional/Ue'][()], params['dimensional/ke'][()]
Te = params['dimensional/Te'][()]
Uw = params['dimensional/Uw'][()]

## get diagnostics
time = diags['time'][:]
KE_qg = diags['ke_qg'][:]
PE_niw = diags['pe_niw'][:]
KE_niw = diags['ke_niw'][:]
ENS_qg = diags['ens'][:]
g1 = diags['gamma_r'][:]
g2 = diags['gamma_a'][:]
pi = diags['pi'][:]
cKE_niw = diags['cke_niw'][:]
iKE_niw = diags['ike_niw'][:]
ep_phi = diags['ep_phi'][:]
ep_psi = diags['ep_psi'][:]
chi_q =  diags['chi_q'][:]
chi_phi =  diags['chi_phi'][:]
conc_niw =  diags['conc_niw'][:]
skew =  diags['skew'][:]

## calculate tendency from energy time series
dt = time[1]-time[0]
dPE = np.gradient(PE_niw,dt)
dKE = np.gradient(KE_qg,dt)
diKE_niw = np.gradient(iKE_niw,dt)

res_ke = dKE-(-g1-g2+ep_psi)
res_pe = dPE-g1-g2-chi_phi

## plotting
fig = plt.figure(figsize=(8.5,4))
lw, alp = 2.,1.
KE0 = KE_qg[0]
tmax = time[-1]

ax = fig.add_subplot(121)
fig.subplots_adjust(wspace=.55)
plt.plot([-5,35],[0,0],'k-',linewidth=0.85)

#plt.plot(time/Te,(KE_qg-KE_qg[0])/KE0,label=r"$K_e$",linewidth=lw,alpha=alp)
#plt.plot(time/Te,(PE_niw-PE_niw[0])/KE0,label=r'$P_w$',linewidth=lw,alpha=alp)
plt.plot(time/Te,(KE_niw-KE_niw[0])/KE_niw[0],label=r"$\Delta\langle\mathcal{A}\rangle/\langle\mathcal{A}\rangle(0)$",linewidth=lw,alpha=alp)
plt.plot(time/Te,(KE_qg-KE_qg[0])/KE0,label=r"$\Delta\langle\mathcal{K}\rangle/\langle\mathcal{K}\rangle(0)$",linewidth=lw,alpha=alp)
plt.plot(time/Te,(PE_niw-PE_niw[0])/KE0,label=r'$\Delta\langle\mathcal{P}\rangle/\langle\mathcal{K}\rangle(0)$',linewidth=lw,alpha=alp)

plt.plot(time/Te,(PE_niw-PE_niw[0]+KE_qg-KE_qg[0])/KE0,'--',
        label=r'$(\Delta\langle\mathcal{P}\rangle+\Delta\langle\mathcal{K}\rangle)/\langle\mathcal{K}\rangle(0)$',
        linewidth=lw,alpha=alp)
plt.ylim(-0.15,0.15)
plt.ylabel(r'Energy change about $t=0$')
#plt.legend(loc=(0.05,0.9))
#plt.plot([0,tmax/Te],[0]*2,'-',linewidth=1,color="0.5")
plt.xlim(-1,25)

plot_fig_label(ax, label="a",xc=0.05,yc = 0.05)
plt.xlabel(r"Time [$t \times U_e k_e$]")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


plt.text(10,.09,r"$\Delta\langle \mathcal{P} \rangle$")
plt.text(10,-.105,r"$\Delta\langle \mathcal{K} \rangle$")
plt.text(10,.0025,r"$\Delta\langle \mathcal{A} \rangle$")
plt.text(20,-.025,r"$\Delta\langle \mathcal{P} + \mathcal{K} \rangle$")

ax = fig.add_subplot(122)
fig.subplots_adjust(wspace=.45)
plt.xticks([0,10,20,30])
plt.plot([-5,35],[0,0],'k-',linewidth=0.85)

plt.plot(time/Te,Te*g1/KE0,label=r'$\Gamma_r$',linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*g2/KE0,label=r'$\Gamma_a$',linewidth=lw,alpha=alp)
#plt.plot(time/Te,Te*chi_phi/KE0,label=r'$\varepsilon_\mathcal{P}$',linewidth=lw,alpha=alp)
#plt.plot(time/Te,Te*(g1+g2+chi_phi)/KE0,label=r'$\Gamma_r+\Gamma_a+ \varepsilon_\mathcal{P}$',
#                        linewidth=lw,alpha=alp)

plt.xlim(-1,25)

plt.plot(time/Te,Te*dPE/KE0,label=r'$\mathrm{d}\langle\mathcal{P}\rangle/\mathrm{d}t$',linewidth=lw,alpha=alp)

#plt.legend(loc=1,ncol=1)
plt.xlabel(r"Time [$t \times U_e k_e$]")
plt.ylim(-0.01/2,0.0225)
plt.ylabel(r'Power $[\langle \dot \mathcal{P} \rangle \times {2 k_e}/{U_e}^2 ]$')
plot_fig_label(ax, label="b",xc=0.05,yc = 0.05)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.text(10,-.0038,r"$\Gamma_r$")
plt.text(5,.008,r"$\Gamma_a$")
plt.text(2.1,.013,r"$\partial_t \langle \mathcal{P} \rangle $")

plt.savefig(patho+"fig4.pdf", bbox_inches='tight')

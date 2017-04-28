"""
    Plots figure xx: energy time series, budgets,
                    wave-vorticity correlation
                    of the Lamb-Chapygin dipole solution.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

from Utils import *

plt.close('all')

pathi = "outputs/decaying_turbulence/reference/nowaves/"
patho = "../writeup/figs/"

params = h5py.File(pathi+"parameters.h5","r")
diags = h5py.File(pathi+"diagnostics.h5")

## get params
Ue, ke = params['dimensional/Ue'][()], params['dimensional/ke'][()]
Te = params['dimensional/Te'][()]

## get diagnostics
time = diags['time'][:]
KE_qg = diags['ke_qg'][:]
ep_psi = diags['ep_psi'][:]

## calculate tendency from energy time series
dt = time[1]-time[0]
dKE = np.gradient(KE_qg,dt)

res_ke = dKE-ep_psi

## plotting
fig = plt.figure(figsize=(8.5,3.))
lw, alp = 3.,.5
KE0 = KE_qg[0]
tmax = time[-1]

ax = fig.add_subplot(121)
plt.plot(time/Te,(KE_qg-KE_qg[0])/KE0,label=r"$K_e$",linewidth=lw,alpha=alp)
plt.ylim(-0.025,0.)
plot_fig_label(ax, label="a")
plt.ylabel(r'Energy  change $[(K_e-K_{e_0}) \times {2}/{U_e^2} ]$')
plt.xlabel(r"Time [$t \times U_e k_e$]")

fig.subplots_adjust(wspace=.4)
ax = fig.add_subplot(122)
plt.plot(time/Te,Te*(ep_psi)/KE0,label=r'$\varepsilon_\psi$',
                        linewidth=lw,alpha=alp)
plt.plot(time/Te,Te*dKE/KE0,'k--',label=r'$\dot K_e$',linewidth=lw,alpha=alp)
plot_fig_label(ax, label="b")
plt.xlabel(r"Time [$t \times U_e k_e$]")
plt.ylabel(r'Power $[\dot E \times {2 k_e}/{U_e} ]$')
plt.legend(loc=(.725,.65),ncol=1)

plt.savefig(patho+"figc2.pdf", bbox_inches='tight')

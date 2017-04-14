""""
    Plots figure 6: energy budgets for decaying turbulence
                    with alpha=0.1 and various dispersivities.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

from Utils import *

plt.close('all')

path = "outputs/decaying_turbulence/Uw0.1/lambdaz"
patho = "../writeup/figs/"

for lambdaz in [300,400,500,600,700,800]:

    pathi = path+str(lambdaz)+"/"
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
    x1 = diags['xi_r'][:]
    x2 = diags['xi_a'][:]
    chi_q =  diags['chi_q'][:]
    chi_phi =  diags['chi_phi'][:]
    conc_niw =  diags['conc_niw'][:]
    skew =  diags['skew'][:]

    ## calculate tendency from energy time series
    dt = time[1]-time[0]
    dPE = np.gradient(PE_niw,dt)
    dKE = np.gradient(KE_qg,dt)
    diKE_niw = np.gradient(iKE_niw,dt)

    #res_ke = dKE-(-g1-g2+ep_psi)
    #res_pe = dPE-g1-g2-chi_phi

    ## calculate relative contribution
    KE, PE = KE_qg[-1]-KE_qg[0], PE_niw[-1]-PE_niw[0]

    G1, G2 = integrate.simps(y=g1[:],x=time[:]),  integrate.simps(y=g2[:],x=time[:])
    X1 = -integrate.simps(y=x1[:],x=time[:])
    X2 = -integrate.simps(y=x2[:],x=time[:])
    G1_Pw, G2_Pw = G1/PE, G2/PE
    G1_Ke, G2_Ke, X1_Ke, X2_Ke = G1/KE, G2/KE, X1/KE, X2/KE
    G_Ke = G1_Ke+G2_Ke
    CHI_Pw = integrate.simps(y=chi_phi[:],x=time[:])/PE
    EP_Ke = -integrate.simps(y=ep_psi[:],x=time[:])/KE

    RES_PE = 1-(G1_Pw+G2_Pw+CHI_Pw)
    RES_KE = 1+(G1_Ke+G2_Ke+X1_Ke+X2_Ke+EP_Ke)


    # arrays
    try:
        g1_ke, g2_ke = np.hstack([g1_ke,G1_Ke]), np.hstack([g2_ke,G2_Ke])
        x1_ke, x2_ke = np.hstack([x1_ke,X1_Ke]), np.hstack([x2_ke,X2_Ke])
        ep_ke = np.hstack([ep_ke,EP_Ke])
        res_ke = np.hstack([res_ke,RES_KE])
        hslash = np.hstack([hslash, params['nondimensional/hslash'][()]])

        g1_pw, g2_pw = np.hstack([g1_pw,G1_Pw]), np.hstack([g2_pw,G2_Pw])
        chi_pw = np.hstack([chi_pw,CHI_Pw])
        res_pw = np.hstack([res_pw,RES_PE])
    except:
        hslash = np.array(params['nondimensional/hslash'][()])
        g1_ke, g2_ke = G1_Ke, G2_Ke
        x1_ke, x2_ke = X1_Ke, X2_Ke
        ep_ke = EP_Ke
        res_ke = RES_KE

        g1_pw, g2_pw = G1_Pw, G2_Pw
        chi_pw = CHI_Pw
        res_pw = RES_PE

# ## plotting
fig = plt.figure(figsize=(8.5,3.))
lw, alp = 3.,.5
KE0 = KE_qg[0]
tmax = time[-1]


ax = fig.add_subplot(121)

plt.plot(hslash,g1_pw,'o-',label=r"$\Gamma_r$")
plt.plot(hslash,g2_pw,'o-',label=r"$\Gamma_a$")
plt.plot(hslash,chi_pw,'o-',label=r"$\chi_\phi$")
plt.plot(hslash,g1_pw+g2_pw+chi_pw,'o-',label=r"Sum")

plot_fig_label(ax, label="a")
plt.xlabel(r"Dispersivity $[\hslash = f_0 \lambda^2 \times k_e/U_e]$")
plt.ylabel(r"Energy change $\left[\int \dot P_w \mathrm{d}t\,\,/\,\,\Delta P_w\right]$")
plt.legend(loc=(-.075,1.05),ncol=4)

fig.subplots_adjust(wspace=.4)

ax = fig.add_subplot(122)

plt.plot(hslash,g1_ke,'o-',label=r"$\Gamma_r$")
plt.plot(hslash,g2_ke,'o-',label=r"$\Gamma_a$")
plt.plot(hslash,x1_ke,'o-',label=r"$\Xi_r$")
plt.plot(hslash,x2_ke,'o-',label=r"$\Xi_a$")
plt.plot(hslash,ep_ke,'o-',label=r"$\epsilon_\psi$")
plt.plot(hslash,g1_ke+g2_ke+x1_ke+x2_ke+ep_ke,'o-',label=r"Sum")
plt.ylabel(r"Energy change $\left[\int \dot K_e \mathrm{d}t\,\,/\,\,\Delta K_e\right]$")

plt.legend(loc=(0.05,1.05),ncol=3)
plot_fig_label(ax, label="b")
plt.xlabel(r"Dispersivity $[\hslash = f_0 \lambda^2 \times k_e/U_e]$")
plt.savefig(patho+"fig7.pdf", bbox_inches='tight')

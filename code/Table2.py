"""
    Table 2: Relative contribution of each term in
                the energy budget of the decaying
                turbulence solution.
"""

import h5py
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

from astropy.io import ascii

from Utils import *

plt.close('all')

pathi = "outputs/decaying_turbulence/coupled/"
patho = "../writeup/figs/"

params = h5py.File(pathi[:-8]+"parameters.h5","r")
diags = h5py.File(pathi+"diagnostics.h5")

## get params
Ue, ke = params['dimensional/Ue'][()], params['dimensional/ke'][()]
Te = params['dimensional/Te'][()]
Uw = params['dimensional/Uw'][()]

## get diagnostics
time = diags['time'][:]
KE_qg = diags['ke_qg'][:]
PE_niw = diags['pe_niw'][:]
ENS_qg = diags['ens'][:]
g1 = diags['gamma_r'][:]
g2 = diags['gamma_a'][:]
x1 = diags['xi_r'][:]
x2 = diags['xi_a'][:]
ep_psi = diags['ep_psi'][:]
chi_phi =  diags['chi_phi'][:]

## calculate relative contribution
i = g1.size

KE, PE = KE_qg[i-1]-KE_qg[0], PE_niw[i-1]-PE_niw[0]

G1, G2 = integrate.simps(y=g1[:i],x=time[:i]),  integrate.simps(y=g2[:i],x=time[:i])
X1 = -integrate.simps(y=x1[:i],x=time[:i])
X2 = -integrate.simps(y=x2[:i],x=time[:i])
G1_Pw, G2_Pw = G1/PE, G2/PE
G1_Ke, G2_Ke, X1_Ke, X2_Ke = G1/KE, G2/KE, X1/KE, X2/KE
G_Ke = G1_Ke+G2_Ke
CHI_Pw = integrate.simps(y=chi_phi[:i],x=time[:i])/PE
EP_Ke = -integrate.simps(y=ep_psi[:i],x=time[:i])/KE

RES_PE = 1-(G1_Pw+G2_Pw+CHI_Pw)
RES_KE = 1+(G1_Ke+G2_Ke+X1_Ke+X2_Ke+EP_Ke)

## export LaTeX table
Labels_Pw = np.array(['$\Gamma_r$','$\Gamma_a$','$-$','$-$','$\chi_\phi$','Res.'])
Labels_Ke = np.array(['-$\Gamma_r$','-$\Gamma_a$','$\Xi_r$','$\Xi_a$','$\epsilon_\psi$','Res.'])
fmt = 1e3
Pw_budget = np.round(np.array([G1_Pw, G2_Pw,np.nan, np.nan, CHI_Pw, RES_PE])*fmt)/fmt
Ke_budget = np.round(np.array([G1_Ke, G2_Ke, X1_Ke, X2_Ke, EP_Ke, RES_KE])*fmt)/fmt

data = np.concatenate([Labels_Pw[:,np.newaxis],Pw_budget[:,np.newaxis],
                        Labels_Ke[:,np.newaxis], Ke_budget[:,np.newaxis]], axis=1)

data[2,1] = "$-$"
data[3,1] = "$-$"

fno = "../writeup/table2.tex"
caption = "The time-integrated budget of wave potential energy and quasigeostrophic\
                kinetic energy of the decaying turbulence dipole expirement. \label{table2}"
ascii.write(data, output=fno, Writer=ascii.Latex, names=['$\dot{P}_w$ budget',
                    'Rel. contribution ($\int!\dot{P}_w \dd t/\Delta P_w dt$)','$\dot{K}_e$ budget',
                    'Rel. contribution ($\int!\dot{K}_e \dd t/\Delta K_e$)'],
                    overwrite=True, caption=caption,
                    latexdict={'preamble': r'\begin{center}',
                       'tablefoot': r'\end{center}',
                       'tabletype': 'table'})

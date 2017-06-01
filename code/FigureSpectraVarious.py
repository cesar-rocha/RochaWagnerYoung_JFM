
""""
    Plots figure spectra: spectra of equilibrated solutions
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pyspec import spectrum 
from Utils import *

plt.close('all')

pathi = "outputs/high_res/decaying_turbulence/parameter_exploration/Uw0.1/lambdaz397.5/"
#pathi = "outputs/high_res/decaying_turbulence/parameter_exploration/Uw0.1/lambdaz562.149891043/"
patho = "../writeup/figs/"

params = h5py.File(pathi+"parameters.h5","r")
diags = h5py.File(pathi+"diagnostics.h5")

## get params
Ue, ke = params['dimensional/Ue'][()], params['dimensional/ke'][()]
Te = params['dimensional/Te'][()]
Uw = params['dimensional/Uw'][()]
f0 = params['dimensional/f0'][()]
m = params['dimensional/m'][()]
N0 = params['dimensional/N'][()]
nu4w = params['dimensional/nu4'][()]
lam2  = (N0/f0/m)**2
lam  = np.sqrt(lam2)

## get grid
setup = h5py.File(pathi+"setup.h5","r")
x, y = setup['grid/x'][:]*ke/(2*np.pi), setup['grid/y'][:]*ke/(2*np.pi)
k,l, = setup['grid/k'][:], setup['grid/l'][:]
ki,li = np.meshgrid(k,l)
wv2 = ki**2+li**2
wv2i = np.zeros_like(wv2)
fnz = wv2 != 0
wv2i[fnz] = 1./wv2[fnz]
wv = np.sqrt(wv2)

files = ['000000000400000.h5', '000000004220000.h5', '000000010000000.h5']

def calc_spec(snap):

    k,l, = setup['grid/k'][:], setup['grid/l'][:]
    t = snap['t'][()]
    q = snap['q'][:]
    phi = snap['phi'][:]
    qh, phih = np.fft.fft2(q), np.fft.fft2(phi)
    ph = -wv2i*qh
    ug, vg = np.fft.ifft2(-1j*l*ph).real, np.fft.ifft2(1j*k*ph).real
    
    Kes = np.abs(wv*ph)**2
    Pws = (lam2/4)*np.abs(wv*phih)**2
    Kws = np.abs(phih)**2
    
    ki, Kesi = spectrum.calc_ispec(k,l,Kes)
    _, Pwsi = spectrum.calc_ispec(k,l,Pws)
    _, Kwsi = spectrum.calc_ispec(k,l,Kws)

    return ki, Kesi, Kwsi, Pwsi, t 


# calculate the batchelor-like scale
kdisp = np.sqrt((Ue*ke)/f0)/lam
kdisp2 = Ue/(f0*lam2)
kbatch = ((Ue*ke)/nu4w)**0.25

fig = plt.figure(figsize=(8.5,4.25))
ax1 = fig.add_subplot(131)
fig.subplots_adjust(wspace=.075)
ax2 = fig.add_subplot(132)
fig.subplots_adjust(wspace=.075)
ax3 = fig.add_subplot(133)
#ax3.plot([kbatch/ke]*2,[0,0.2],linewidth=1.5,color='0.75')
#ax3.plot([kdisp2/ke]*2,[0,0.2],linewidth=1.5,color='0.75')
#ax3.text(kbatch/ke,0.1435,r'$k_{diss}$',rotation=90,color='0.75')
#ax3.text(kdisp2/ke,0.1435,r'$k_{disp}$',rotation=90,color='0.75')

dir = pathi[:-6]

for lambdaz in ['281.074945522', '397.5','562.149891043']:

    fni = dir+lambdaz+"/snapshots/"+files[-1]
    ki, Kesi, Kwsi, Pwsi,t = calc_spec(h5py.File(fni))


    if lambdaz == '281.074945522':
        label = '0.5'
    elif lambdaz == '397.5':
        label = '1.0'
    elif lambdaz == '562.149891043':
        label = '2.0'

    ax1.semilogx(ki/ke,ki*Kesi/Kesi.sum()/ke,label=r"$\hslash = $"+ label)
    ax2.semilogx(ki/ke,ki*Kwsi/Kwsi.sum()/ke ,label=r"$\hslash = $"+ label)
    ax3.semilogx(ki/ke,ki*Pwsi/Pwsi.sum()/ke ,label=r"$\hslash = $"+ label)

# calculate the eddy spectrum at t = 0
Ke_McW = wv/( ( 1 + (wv/ke)**4 ))
KE = Ke_McW.sum()/(Ke_McW.size**2)
Ke_McW *= ((Ue**2)/2)/KE

fni = dir+lambdaz+"/snapshots/000000000000000.h5"
ki, Kesi0, Kwsi0, Pwsi0,t0 = calc_spec(h5py.File(fni))


_, Kei_McW = spectrum.calc_ispec(k,l,Ke_McW)

ax1.semilogx(ki/ke,ki*Kesi0/Kesi0.sum()/ke,'--',color='0.65')
ax1.set_xlabel(r'Wavenumber [$|\mathbf{k}|/k_e$]')
ax1.set_ylabel(r'Energy-preserving spectrum [$(|\mathbf{k}|/k_e)\,\mathcal{S}/$ sum$(\mathcal{S})$]')
plot_fig_label(ax1, label="$\mathcal{K}_e$" ,xc=.95,yc=0.05)
ax1.set_ylim(0.,0.15)
ax2.set_xlabel(r'Wavenumber [$|\mathbf{k}|/k_e$]')
#ax2.set_ylabel(r'Energy-preserving spectrum [$|\mathbf{k}|\mathcal{K}_w \times U_w^{-2}$]')
plot_fig_label(ax2, label="$\mathcal{K}_w$",xc=.95,yc=0.05)
ax2.set_yticks([])
ax2.set_ylim(0.,0.15)
ax3.set_xlabel(r'Wavenumber [$|\mathbf{k}|/k_e$]')
ax2.legend(loc=1,ncol=3)
#ax3.set_ylabel(r'Energy-preserving spectrum [$|\mathbf{k}|\mathcal{P}_w \times U_e^{-2}$]')
plot_fig_label(ax3, label="$\mathcal{P}_w$" ,xc=.95,yc=0.05)
ax3.set_yticks([])
ax3.set_ylim(0.,0.15)


ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
#ax1.spines['left'].set_smart_bounds(True)
ax1.spines['left'].set_position(('axes', -0.1))
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['left'].set_visible(False)

plt.savefig(patho+"FigSpectraVarious.pdf",pad_inches=0,
            bbox_inches='tight')




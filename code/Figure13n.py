""""
    Plots figure sinxsiny: energy time series, budgets,
                    wave-vorticity correlation
                    of the decaying macroturbulence solution.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from Utils import *

plt.close('all')

pathi = "outputs/high_res/decaying_turbulence/parameter_exploration/Uw0.1/lambdaz397.5/"
#pathi = "outputs/sinxsiny/"
patho = "../writeup/figs/"

params = h5py.File(pathi+"parameters.h5","r")
diags = h5py.File(pathi+"diagnostics.h5")
setup = h5py.File(pathi+"setup.h5")
## get params
Ue, ke = params['dimensional/Ue'][()], params['dimensional/ke'][()]
Te = params['dimensional/Te'][()]
Uw = params['dimensional/Uw'][()]
#lam = params['dimensional/lamda'][()]
alpha = params['nondimensional/alpha'][()]
hslash = params['nondimensional/hslash'][()]

params = h5py.File(pathi+"parameters.h5","r")
Ue, ke = params['dimensional/Ue'][()], params['dimensional/ke'][()]
Te = params['dimensional/Te'][()]
Uw = params['dimensional/Uw'][()]
f0 = params['dimensional/f0'][()]
m = params['dimensional/m'][()]
N0 = params['dimensional/N'][()]
#nu4w = params['dimensional/nu4'][()]
lam2  = (N0/f0/m)**2
lam  = np.sqrt(lam2)


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
lw, alp = 2.,1.
KE0 = KE_qg[0]
tmax = time[-1]

k,l, = setup['grid/k'][:], setup['grid/l'][:]
k, l = np.meshgrid(k,l)
wv2 = k**2 + l**2
wv = np.sqrt(wv2)
fnz = wv2 != 0
wv2i = np.zeros_like(wv2)
wv2i[fnz] = 1./wv2[fnz]


files = glob.glob(pathi[:-1]+"/snapshots/*.h5")
KE = []
KL = []
KS = []
KC = []
AQ = []
GR = []
GA = []
SP = []
QDIVF = []
QWDIVF = []
T = []

for fni in files[0::1]:

#fni = files[-1]

    snap = h5py.File(fni)

    t = snap['t'][()]
    q = snap['q'][:]
    phi = snap['phi'][:]
    qh, phih = np.fft.fft2(q), np.fft.fft2(phi)
    phi2h = np.fft.fft2(np.abs(phi)**2)
    phix, phiy = np.fft.ifft2(1j*k*phih), np.fft.ifft2(1j*l*phih)
    J_phic_phi = np.conj(phix)*phiy - np.conj(phiy)*phix

    qw1 = np.fft.ifft2(-wv2*phi2h).real/(4*f0)
    qw2 = (1j*J_phic_phi).real/(2*f0)
    qw = qw1+qw2
    qpsi = q-qw
    qpsih = np.fft.fft2(qpsi)
    ph = -wv2i*qpsih
    A = (np.abs(phi)**2)/(2*f0)
    psis = -A
    psh = np.fft.fft2(psis)
    peh =  ph-psh

    psis_x, psis_y = np.fft.ifft2(1j*k*psh), np.fft.ifft2(1j*l*psh)
    psie_x, psie_y = np.fft.ifft2(1j*k * peh), np.fft.ifft2(1j*l*peh)
    psi_x, psi_y = np.fft.ifft2(1j*k * ph), np.fft.ifft2(1j*l*ph)

    phi_x, phi_y = np.fft.ifft2(1j*k * phih), np.fft.ifft2(1j*l*phih)
    lapphi = np.fft.ifft2(-wv2*phih)   

    Kc = (psie_x*psis_x + psie_y*psis_y).mean()
    Ke = 0.5*(psie_x**2 + psie_y**2).mean()
    Ks = 0.5*(psis_x**2 + psis_y**2).mean()
    Kl = 0.5*(psi_x**2 + psi_y**2).mean()


    ga = 0.5*lam2*(np.conj(lapphi)*(psi_x*phi_y-psi_y*phi_x)).real.mean()
    divF = 0.5*lam2*(np.conj(phi)*lapphi).imag
    gr = 0.5*(qpsi*divF).mean()
    sp = gr-ga + (qw*divF).mean() 


    KE.append(Ke)
    KL.append(Kl)
    KS.append(Ks)
    KC.append(Kc)
    AQ.append((A*q).mean())
    GR.append(gr)
    GA.append(ga)
    SP.append(sp)
    QDIVF.append((q*divF).mean())
    QWDIVF.append((qw*divF).mean())
    T.append(t)



KC = np.array(KC)
KE = np.array(KE)
KS = np.array(KS)
KL = np.array(KL)
AQ = np.array(AQ)
GR = np.array(GR)
GA = np.array(GA)
SP = np.array(SP)
QDIVF = np.array(QDIVF)
QWDIVF = np.array(QWDIVF)
T = np.array(T)

dt = T[1]-T[0]
KEt = np.gradient(KE,dt)
KLt = np.gradient(KL,dt)
KSt = np.gradient(KS,dt)
KCt = np.gradient(KC,dt)

K = (Ue**2)
Pw = K/Te

fig = plt.figure(figsize=(8.5,8.5))
ax = fig.add_subplot(211)
plt.plot([0,25],[0,0],'k--')
plt.plot(T/Te,(KE-KE[0])/K)
plt.plot(T/Te,(KS-KS[0])/K)
plt.plot(T/Te,(KC-KC[0])/K)
plt.plot(T/Te,(KL-KL[0])/K)
plt.ylabel(r"Kinetic energy diff. about $t=0$ [$\Delta\langle\mathcal{K}\rangle/U_e^2$]")
#plt.text(20.7,1.6/2,r'$\mathcal{K}^E \equiv \langle |\nabla\psi^E|^2\rangle/2$',rotation=68)
plt.text(16,-0.0175,r'$\langle \nabla\psi^E\cdot\nabla\psi^S\rangle$',rotation=10)
plt.text(12,0.007,r'$\langle |\nabla\psi^S|^2\rangle/2$',rotation=0)
plt.text(12,-0.05,r'$ \langle |\nabla\psi^L|^2\rangle/2$',rotation=-17)
plt.text(18,-0.046,r'$ \langle |\nabla\psi^E|^2\rangle/2$',rotation=-17)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.xticks([])
ax = fig.add_subplot(212)
plt.ylabel(r"Power [$\langle\mathcal{K}\rangle_t/(U_e^3/k_e)$]")
plt.xlabel(r"Time [$t/(U_e k_e)^{-1}$]")
plt.plot([0,25],[0,0],'k--')
plt.plot(T/Te,-(GR)/(K/Te))
plt.plot(T/Te,-(GA)/(K/Te))
#plt.plot(T/Te,-(GR+GA)/(K/Te))
plt.plot(T/Te, SP/(K/Te))
#plt.plot(T/Te, QDIVF/(K/Te))
#plt.plot(time/Te,-g2/(K/Te))
#plt.plot(T/Te, KEt/(K/Te))
#plt.plot(T/Te, (KSt+KCt)/(K/Te))

plt.text(.95,.0065,r'$\langle SP \rangle$',rotation=-65)
plt.text(.95,-.0075,r'$-\Gamma_r$',rotation=-25)
plt.text(7.,-.00185,r'$-\Gamma_a$',rotation=-0)

#plt.text(14,.29/2,r'$\mathcal{K}^L_t$',rotation=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.savefig("EulerianLagrangianKEdiff.pdf",pad_inches=0, bbox_inches='tight')


fig = plt.figure(figsize=(8.5,4.))
ax = fig.add_subplot(121)
plt.plot([-5,30],[0,0],'k-',linewidth=1.)
plt.plot(T/Te,(KE-KE[0])/K)
#plt.plot(T/Te,(KS-KS[0])/K)
plt.plot(T/Te,(KC-KC[0]+KS-KS[0])/K)
plt.plot(T/Te,(KL-KL[0])/K)
plt.ylabel(r"Energy diff. about $t=0$ [$\Delta\langle\mathcal{K}\rangle/U_e^2$]")
#plt.text(20.7,1.6/2,r'$\mathcal{K}^E \equiv \langle |\nabla\psi^E|^2\rangle/2$',rotation=68)
plt.text(16,-0.014,r'$\Delta\langle\mathcal{K}^S\rangle$',rotation=10)
plt.text(12,-0.052,r'$\Delta\langle\mathcal{K}\rangle$',rotation=-17)
plt.text(18,-0.048,r'$\Delta\langle\mathcal{K}^E\rangle$',rotation=-17)

plt.xlabel(r"Time [$t/(U_e k_e)^{-1}$]")
plot_fig_label(ax, label="a",xc=0.05,yc = 0.05)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.get_xaxis().tick_bottom()
#ax.get_yaxis().tick_left()

plt.xlim(-1,25)

fig.subplots_adjust(wspace=.425)
GS = SP+GA+GR

ax = fig.add_subplot(122)
plt.ylabel(r"Power [$\langle\mathcal{K}\rangle_t/(U_e^3/k_e)$]")
plt.xlabel(r"Time [$t/(U_e k_e)^{-1}$]")
plt.plot([-5,30],[0,0],'k-',linewidth=1.)
plt.plot(T/Te,-(GR)/(K/Te),label=r'$-\Gamma_r$')
plt.plot(T/Te,-(GA)/(K/Te) ,label=r'$-\Gamma_a$')
#plt.plot(T/Te,-(GR+GA)/(K/Te))
plt.plot(T/Te, SP/(K/Te) ,label=r'$RSP$')
plt.plot(T/Te, GS/(K/Te) ,label=r'$\Gamma_S$')
#plt.plot(T/Te, (GR+GA)/(K/Te))
#plt.plot(T/Te, QDIVF/(K/Te))
#plt.plot(time/Te,-g2/(K/Te))
#plt.plot(T/Te, KEt/(K/Te))
#plt.plot(T/Te, (KSt+KCt)/(K/Te))

#plt.text(1.1,.012,r'$\Gamma_S$',rotation=-80)
#plt.text(-.085,.0065,r'$RSP$',rotation=-80)
#plt.text(.95,-.0075,r'$-\Gamma_r$',rotation=-25)
#plt.text(.85,-.00185,r'$-\Gamma_a$',rotation=-0)
#plt.text(7.,-.00185,r'$BF$',rotation=-0)

plt.legend(loc=1)

plot_fig_label(ax, label="b",xc=0.05,yc = 0.05)

#plt.text(14,.29/2,r'$\mathcal{K}^L_t$',rotation=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

plt.xlim(-1,25)
plt.ylim(-0.0125,0.02)

plt.savefig("../writeup/figs/EulerianLagrangianKEdiff.pdf",pad_inches=0, bbox_inches='tight')
plt.savefig("../writeup/figs/fig13.tiff",pad_inches=0, bbox_inches='tight')

# Potential energy estimate

# nondimentional wave potential energy [P/Ue^2/2]
P = ((lam*ke)**2) * ((Uw/Ue)**2)




#plt.plot(T,SP,'r',label=r'$\leftangle\overline{SP}\rightangle$')
#plt.plot(T,GA+GR,label=r'$\Gamma_r+\Gamma_a$')
#plt.plot(T,GR-GA,'k--',label=r'$\Gamma_r-\Gamma_a$')
#plt.legend()
#plt.savefig("gammas_sp")

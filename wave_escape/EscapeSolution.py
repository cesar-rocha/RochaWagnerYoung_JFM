import numpy as np
from numpy import pi, exp
import matplotlib.pyplot as plt
import cmocean
plt.close("all")

#
# settings
#

# grid
N = 256
L = 2*pi
dx = L/N
x = np.arange(-L/2,L/2,dx)
y = x.copy()
x,y = np.meshgrid(x,y)

dk = 2*np.pi/L
kNy = 2*np.pi/(2*dx)
k = np.arange(-kNy,kNy,dk)
l = k.copy()
k,l = np.meshgrid(k,l)

# balanced flow
alpha = .5
psi = -alpha*x*y

# waves
eta = 0.05

#
# initial condition
#

C = 1.
R = .75
R2 = R**2
p,q = 5,5
phi0 = ( (2*pi)/(R2) ) * exp( - (x**2 + y**2)/(2*R2) + 1j*(p*x+q*y))
#A0 = exp( -(R2)*( (k-p)**2 + (l-q)**2 )/2 )

#
# non-dimensional numbers
#
PSI = R2*alpha
hslash = eta/PSI

#
# solution
#

# auxlliary functions
X = lambda t: exp(-alpha*t)*x
Y = lambda t: exp(+alpha*t)*y
f = lambda t: (1-exp(-2*alpha*t))/(2*alpha)
g = lambda t: (exp(2*alpha*t)-1)/(2*alpha)
C = lambda t: (2*pi)/np.sqrt( (R2 + 1j*eta*f(t))*(R2 + 1j*eta*g(t)) )
A = lambda t: (X(t)**2 - 2j*R2*p*X(t) + 1j*eta*f(t)*R2*(p**2) )/(R2 + 1j*eta*f(t))
B = lambda t: (Y(t)**2 - 2j*R2*q*Y(t) + 1j*eta*g(t)*R2*(q**2) )/(R2 + 1j*eta*g(t))

# the solution
phi = lambda t: C(t)*exp(-0.5*(A(t)+B(t)))

#
# plot the solution
#
cu = np.hstack([np.arange(-12,0,1), np.arange(1,13,1)])
cp = np.array([-8,-4,-1.-.5,-.25])
cp = np.hstack([cp,-np.flipud(cp)])

tmax = 6
fig = plt.figure(figsize=(6.5,6.5))
ax = fig.add_subplot(aspect=1)

for t in np.arange(0,6,.025):

    plt.clf()
    plt.contour(x/R,y/R,psi,cp,colors='k')
    plt.contourf(x/R,y/R,phi(t).real,cu,cmap=cmocean.cm.balance)
    plt.xlabel(r"$x/R$")
    plt.ylabel(r"$y/R$")
    plt.title(r"$t\,\alpha = %3.2f$" %(t*alpha))
    plt.draw()
    tit = "%3.2f" %(t*alpha)
    plt.savefig("figs/"+tit+".png")
    plt.pause(0.01)

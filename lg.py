import matplotlib.image as im
import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pylab import *

######################################################
### UNITS : mm
######################################################

pi = np.pi
sqrt = np.sqrt
exp = np.exp
arctan2 = np.arctan2
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift

######################################################
### FUNCTIONS
######################################################

            

def LG(ell,p,w0,r,phi,z,k):

    wavelength = (2*pi)/k
    zR = (pi*(w0**2))/(wavelength)
    
    Rc = z*(1.0+((zR/z)**2))
    wZ = w0*sqrt(1.0+((z/zR)**2))

    norm = sqrt((2.0*sp.gamma(1.0+p))/(pi*(sp.gamma(1.0+p+np.absolute(ell)))))
    
    term1 = 1.0/wZ
    term2 = ((r*(np.sqrt(2.0)))/(wZ))**(np.absolute(ell))
    term3 = np.exp((-1.0*(r**2))/(wZ**2))
    term4 = (sp.genlaguerre(p,np.absolute(ell)))((2.0*(r**2))/(wZ**2))
    term5 = np.exp((-1.0j*k*(r**2))/(2*Rc))
    term6 = np.exp(1.0j*ell*phi)
    term7 = np.exp(-1.0j*k*z)
    term8 = np.exp(1.0j*(np.absolute(ell)+(2*p)+1)*arctan2(z,zR))

    return norm*term1*term2*term3*term4*term5*term6*term7*term8

def intensity(field):
    return (np.abs(field))**2



## propagator for LG beam
def propagation(beam,zfinal,dz=1.0):
    M = int(round(zfinal/dz)) ## Total number of propagation steps
    print (M)

    for m in range(M) : ## Start propagation loop
        c = fft2(beam) ## Take 2D-Fourier transform
        c = fftshift(exp(0.5*(dz/wavevector)*1.0j*(x**2+y**2)))*c ## Advance in Fourier space
        beam = ifft2(c) ## Return to physical space
        print (m)

    return beam

######################################################
###PARAMETERS
######################################################


x = np.linspace(-5,5,200); ## Grid points along x
y = np.linspace(-5,5,200) ## Grid points along y
X,Y = np.meshgrid(x,y)
R = sqrt(X**2+Y**2)
PHI = np.mod(np.arctan2(Y,X),2*pi)

wavelength = 810E-6
beamWaist = 1.0
wavevector = (2.0*pi)/wavelength

 
beam = LG(2,0,beamWaist,R,PHI,0.000001,wavevector)#+5*exp(1.0j*pi)*LG(5,0,beamWaist,R,PHI,0.000001,wavevector)
##beam = LG(-2,0,beamWaist,R,PHI,0.000001,wavevector)+exp(1.0j*pi)*LG(3,0,beamWaist,R,PHI,0.000001,wavevector)

phase = np.mod(np.angle(beam),2*pi)

figInit = plt.figure()
ax1 = figInit.add_subplot(1,1,1)
plot1 = ax1.pcolormesh(X,Y,intensity(beam),cmap="inferno")
ax1.set_aspect('equal')
ax1.autoscale(tight=True)

figPhase = plt.figure()
ax2 = figPhase.add_subplot(1,1,1)
plot2 = ax2.pcolormesh(X,Y,phase,cmap="hsv",vmin=0,vmax=2*pi)
ax2.set_aspect('equal')
ax2.autoscale(tight=True)

plt.show()



beam = propagation(beam,10000,10000)


phase = np.mod(np.angle(beam),2*pi)

figInit = plt.figure()
ax1 = figInit.add_subplot(1,1,1)
plot1 = ax1.pcolormesh(X,Y,intensity(beam),cmap="inferno")
ax1.set_aspect('equal')
ax1.autoscale(tight=True)

figPhase = plt.figure()
ax2 = figPhase.add_subplot(1,1,1)
plot2 = ax2.pcolormesh(X,Y,phase,cmap="hsv",vmin=0,vmax=2*pi)
ax2.set_aspect('equal')
ax2.autoscale(tight=True)

plt.show()





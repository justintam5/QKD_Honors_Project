import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
from pylab import *
from scipy import special as sp
# from PIL import Image
# from scipy import interpolate
from scipy.fftpack import dst, idst
# from scipy.ndimage.filters import laplace
# import scipy.optimize as opt
# import os

pi = np.pi
exp = np.exp
sqrt = np.sqrt
cos = np.cos

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

## propagator for LG beam
def propagation(beam,zfinal,dz=1.0):
    M = int(round(zfinal/dz)) ## Total number of propagation steps
    ##print (M)

    for m in range(M) : ## Start propagation loop
        c = fft2(beam) ## Take 2D-Fourier transform
        c = fftshift(exp(0.5*(dz/wavevector)*1.0j*(Ex**2+Ey**2)))*c ## Advance in Fourier space
        beam = ifft2(c) ## Return to physical space

    return beam

beamWaist = 2      ##units all in mm
wavelength = 633.0E-6
wavevector = (2*pi)/wavelength

N=500   ##number of points
L=15
h = L/N ##transverse space step

n = np.arange(-N/2,N/2+1,1)
x = n*h;
y = n*h;
X,Y = np.meshgrid(x,y)
R = sqrt(X**2+Y**2)
PHI = np.mod(np.arctan2(Y,X),2*pi)


##Choose OAM beam with first index ell second index p 
beam = LG(3,1,beamWaist,R,PHI,1e-15,wavevector)

# distance = 500
# beamFar = propagation(beam,distance,distance)

intensity = (np.absolute(beam))**2
phase = np.angle(beam)

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(intensity,cmap="Greys_r",origin="lower")
ax2.imshow(phase,cmap="hsv",origin="lower",vmin=-pi,vmax=pi)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
from pylab import *
from scipy import special as sp
##from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from scipy import interpolate

from timeit import default_timer as timer

start = timer()
pi = np.pi
exp = np.exp
sqrt = np.sqrt
sin = np.sin
cos = np.cos


## Define Arrays

rads = np.linspace(-pi,0,200)
sinc = np.sinc(rads/pi)

arcsinc = interpolate.interp1d(sinc,rads, kind='linear',bounds_error=False, fill_value=-pi)

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

def Holo():
    
    wavelength = 780.0E-6
    wavevector = (2*pi)/wavelength
    beamWaist = 1.5


    x = 2*np.linspace(-5,5,1366)
    y = 2*np.linspace(-5,5,768)

    X,Y = np.meshgrid(x,y)

    R = sqrt(X**2+Y**2)
    PHI = np.mod(np.arctan2(Y,X),2*pi)

    lgstart = timer()
    beam = LG(2,0,beamWaist,R,PHI,1E-15,wavevector)
    lgend = timer()

    holo_start = timer()
    amplitude = np.absolute(beam)
    amplitudeNorm = amplitude/np.max(amplitude)
    print(amplitudeNorm.max())
    intensity = (np.absolute(beam))**2
    intensityNorm = intensity/np.max(intensity)
    print(intensityNorm.max())
    
    phase = np.angle(beam)
    print(phase.max(),phase.min())

    mTerm = 1.0 ##+ (1.0/pi)*arcsinc(amplitudeNorm)       ##comment out second term for no intensity masking
    fTerm = phase - pi*mTerm
    print(fTerm.min(),fTerm.max())
    
    Lambda = 0.4
    
    hologram = (255/(2*pi)*mTerm*np.mod(fTerm+(2*pi*X*(1.0/Lambda)),2*pi)).astype('uint8')
    holo_end = timer()
    end = timer()
    return hologram


im = plt.imshow(hologram,cmap="Greys_r")
plt.show(im)
##print(end-start)
##print('lg',lgend-lgstart)
##print('holo',holo_end-holo_start)

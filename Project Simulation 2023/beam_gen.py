import matplotlib.image as im
import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp

class BeamGen:
    def __init__(self, mode, ell, p, w0, r, phi, z, k):
        self.test = 2

        PI = np.pi
        sqrt = np.sqrt
        exp = np.exp
        arctan2 = np.arctan2


        if mode == "LG" or mode == "lg" : #Mode is chosen as LG 
            wavelength = (2*PI)/k
            zR = (PI*(w0**2))/(wavelength)
            
            Rc = z*(1.0+((zR/z)**2))
            wZ = w0*sqrt(1.0+((z/zR)**2))

            norm = sqrt((2.0*sp.gamma(1.0+p))/(PI*(sp.gamma(1.0+p+np.absolute(ell)))))
            term1 = 1.0/wZ
            term2 = ((r*(np.sqrt(2.0)))/(wZ))**(np.absolute(ell))
            term3 = np.exp((-1.0*(r**2))/(wZ**2))
            term4 = (sp.genlaguerre(p,np.absolute(ell)))((2.0*(r**2))/(wZ**2))
            term5 = np.exp((-1.0j*k*(r**2))/(2*Rc))
            term6 = np.exp(1.0j*ell*phi)
            term7 = np.exp(-1.0j*k*z)
            term8 = np.exp(1.0j*(np.absolute(ell)+(2*p)+1)*arctan2(z,zR))

            self.beam = norm*term1*term2*term3*term4*term5*term6*term7*term8
    
    def intensity(self):
        return (np.abs(self.beam))**2

    def phase(self):
        return (np.mod(np.angle(self.beam),2*PI))

###---------------------------------USE CASE EXAMPLE--------------------------------------

PI = np.pi
x = np.linspace(-5,5,200); ## Grid points along x
y = np.linspace(-5,5,200) ## Grid points along y
X,Y = np.meshgrid(x,y)
r = np.sqrt(X**2+Y**2)
phi = np.mod(np.arctan2(Y,X),2*PI)

wavelength = 810E-6
beamWaist = 1.0
wavevector = (2.0*PI)/wavelength

lg_beam = BeamGen("LG",2,0,beamWaist,r,phi,0.000001,wavevector)


figInit = plt.figure()
ax1 = figInit.add_subplot(1,1,1)
plot1 = ax1.pcolormesh(X,Y,lg_beam.intensity(),cmap="inferno")
ax1.set_aspect('equal')
ax1.autoscale(tight=True)

figPhase = plt.figure()
ax2 = figPhase.add_subplot(1,1,1)
plot2 = ax2.pcolormesh(X,Y,lg_beam.phase(),cmap="hsv",vmin=0,vmax=2*PI)
ax2.set_aspect('equal')
ax2.autoscale(tight=True)


plt.show()


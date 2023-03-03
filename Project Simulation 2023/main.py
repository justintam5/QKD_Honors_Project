import numpy as np
import matplotlib.pyplot as plt
from Channel import Channel
from beam_gen import BeamGen 


def __cart2pol(x, y):
        """
        This function is used for converting cartesian coordinates to polar coordinates. 
        Particularly useful when working with circular channels such as lenses or when 
        working with abberations (due to the definition of the Zernike polynomial in polar coordinates). 

        @type x: Number 
        @param x: Coordinate in x direction 
        @type y: Number 
        @param y: Coordinate in y direction 

        @rtype rho: Number 
        @return rho: Coordinate distance in rho direction (distance from center)
        @rtype phi: Number 
        @return phi: Coordinate angle in phi direction (angle from positive x axis)
        """
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)

L = 10
N = 1000

x = np.linspace(-L/2,L/2,N); ## Grid points along x
y = np.linspace(-L/2,L/2,N) ## Grid points along y
X,Y = np.meshgrid(x,y)
r, phi = __cart2pol(X, Y)
#r = np.sqrt(X**2+Y**2)
#phi = np.mod(np.arctan2(Y,X),2*PI)

wavelength = 810E-6
beamWaist = 1.0
wavevector = (2.0*np.pi)/wavelength

lg_beam = BeamGen("LG",4,2,beamWaist,r,phi,0.000001,wavevector)

figInit = plt.figure()
ax1 = figInit.add_subplot(2,2,1)
plot1 = ax1.pcolormesh(X,Y,lg_beam.intensity(),cmap="inferno")
ax1.set_aspect('equal')
ax1.autoscale(tight=True)

ax2 = figInit.add_subplot(2, 2, 2)
plot2 = ax2.pcolormesh(X,Y,lg_beam.phase(),cmap="hsv",vmin=0,vmax=2*np.pi)
ax2.set_aspect('equal')
ax2.autoscale(tight=True)

ch1 = Channel(type = Channel.ABBARATION, n = [3, 1, 4], m = [1, 1, 2], stre = np.array([0.8, 0.9, 0.7]), app = 5)
ch2 = Channel(type = Channel.FREE_SPACE, dist = 100000)
ch3 = Channel(type = Channel.LENS, diam = 10)
ch4 = Channel(type = Channel.FREE_SPACE, dist = 100000)

#beamGen = BeamGen("LG", 1, 3, 2, 3, 0.5, 5, 0.144)

beam_out_1 = ch1.output_beam(lg_beam.beam, wavevector, L=L)
beam_out_2 = ch2.output_beam(beam_out_1, wavevector, L=L)
beam_out_3 = ch3.output_beam(beam_out_2, wavevector, L=L)
beam_out_4 = ch4.output_beam(beam_out_3, wavevector, L=L)

#figInit = plt.figure()
ax1 = figInit.add_subplot(2, 2, 3)
plot1 = ax1.pcolormesh(X, Y, np.abs(beam_out_3)**2, cmap="inferno")
ax1.set_aspect('equal')
ax1.autoscale(tight=True)

ax2 = figInit.add_subplot(2, 2, 4)
plot2 = ax2.pcolormesh(X, Y, np.mod(np.angle(beam_out_3),2*np.pi),
                       cmap="hsv", vmin=0, vmax=2*np.pi)
ax2.set_aspect('equal')
ax2.autoscale(tight=True)


plt.show()



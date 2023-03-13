import matplotlib.image as im
import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp

class BeamGen:
    PI = np.pi
    def __init__(self, mode, ell, p, w0, r, phi, z, k):
        """
        Initialization Method:: Initializes both defined and callable attributes. Sets beam attribute. 
        Uses::      __create_lg_mode()
                    __create_pixel_mode()
        Used By::   n/a
        """
        self.mode = mode
        self.ell = ell
        self.p = p
        self.w0 = w0
        self.r = r
        self.phi = phi
        self.z = z
        self.k = k
        self.beam = None

        if self.mode == "LG" or self.mode == "lg" : #Mode is chosen as LG 
            self.beam = self.__create_lg_mode()
        
        if self.mode == "pixel" :
            self.beam = self.__create_pixel_mode()

    def intensity(self):
        """
        Public Method:: Returns the  intensity of the generated beam
        Uses::      
        Used By::   
        """
        return (np.abs(self.beam))**2

    def phase(self):
        """
        Public Method:: Returns the phase of the generated beam 
        Uses::      
        Used By::   
        """
        return (np.mod(np.angle(self.beam),2*PI))
    
    def __create_lg_mode(self):
        """
        Private Method::    Used to find the index of the nearest 
        Uses::      
        Used By::   __init__()
        """
        wavelength = (2*PI)/self.k
        zR = (PI*(self.w0**2))/(wavelength)
        
        Rc = self.z*(1.0+((zR/self.z)**2))
        wZ = self.w0*np.sqrt(1.0+((self.z/zR)**2))

        norm = np.sqrt((2.0*sp.gamma(1.0+self.p))/(PI*(sp.gamma(1.0+self.p+np.absolute(self.ell)))))
        term1 = 1.0/wZ
        term2 = ((self.r*(np.sqrt(2.0)))/(wZ))**(np.absolute(self.ell))
        term3 = np.exp((-1.0*(self.r**2))/(wZ**2))
        term4 = (sp.genlaguerre(self.p,np.absolute(self.ell)))((2.0*(self.r**2))/(wZ**2))
        term5 = np.exp((-1.0j*self.k*(self.r**2))/(2*Rc))
        term6 = np.exp(1.0j*self.ell*self.phi)
        term7 = np.exp(-1.0j*self.k*self.z)
        term8 = np.exp(1.0j*(np.absolute(self.ell)+(2*self.p)+1)*np.arctan2(self.z,zR))

        return norm*term1*term2*term3*term4*term5*term6*term7*term8

    def __create_pixel_mode(self):
        """
        Private Method:: Used to find the index of the nearest 
        Uses::      
        Used By::   __init__
        """
        self.w0 = np.amax(self.r)
        return 0
   
    def __create_pixels(self):
        """
        Private Method:: Used to find the index of the nearest 
        Uses::      __find_nearest()
        Used By::   __create_pixels()
        """
        i_phi = np.linspace(PI/6, 2*PI+PI/6, 7) # Each phi_i represents the angular position of each 'pixel' circle. Start at an angle of 30 Deg and rotate.
        i_phi = i_phi[:-1]
        i_r = 2/3*np.amax(self.r)
        i_rphi = np.vstack((np.full(6, i_r), i_phi)).T # Returns a 6x2 matrix containing 6 pairs of r/phi coordinates representing the 6 outer pixel origins
        i_rphi = np.append(i_rphi, [[0, 0]], axis=0) # Add the origin to the pixel coordinates. All 7 circle origins are now in this array

        #draw circle around each:
        return np.apply_along_axis(self.__draw_circle, 1, i_rphi)
    

    def __draw_circle(self, x): #callback function :: used to draw a circle around each pixel origin
        """
        Private Method:: Used to find the index of the nearest 
        Uses::      __find_nearest()
        Used By::   __create_pixels()
        """
        print(self.__find_nearest([5, 10])) 
        return x
    
    def __find_nearest(self, value):
        """
        Private Method:: Used to find the index of the nearest 
        Uses::      __dist_in_polar()
        Used By::   __draw_circle()
        """
        idx = np.unravel_index(np.argmin(self.__dist_in_polar(self.r, value[0], self.phi, value[1])), self.phi.shape)
        return [self.r[idx], self.phi[idx], idx]
    
    def __dist_in_polar(self, r2, r1, phi2, phi1):
        """
        Private Method:: Used to calculate the eulidian distance between 2 points given in polar coordinates
        Uses::
        Used By::   __find_nearest()
        """
        return np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(phi2 - phi1))


###---------------------------------USE CASE EXAMPLE--------------------------------------

if __name__ == "__main__":
    PI = np.pi
    wavelength = 810E-6
    beamWaist = 1.0
    wavevector = (2.0*PI)/wavelength

    app_radius = 5
    precision = 200

    x = np.linspace(-app_radius,app_radius,precision+1); ## Grid points along x
    y = np.linspace(-app_radius,app_radius,precision+1) ## Grid points along y
    X,Y = np.meshgrid(x,y)
    r = np.sqrt(X**2+Y**2)
    phi = np.mod(np.arctan2(Y,X),2*PI)
    



    # plt.plot(X, Y, marker='.', color='k', linestyle='none')
    # plt.show()

    lg_beam = BeamGen("LG",4,0,beamWaist,r,phi,0.000001,wavevector)
  
    lg_beam.__draw_circle(2)


    # figInit = plt.figure()
    # ax1 = figInit.add_subplot(1,1,1)
    # plot1 = ax1.pcolormesh(X,Y,lg_beam.circle_arrange_app(r, phi),cmap="inferno")
    # ax1.set_aspect('equal')
    # ax1.autoscale(tight=True)

    # figPhase = plt.figure()
    # ax2 = figPhase.add_subplot(1,1,1)
    # plot2 = ax2.pcolormesh(X,Y,lg_beam.phase(),cmap="hsv",vmin=0,vmax=2*PI)
    # ax2.set_aspect('equal')
    # ax2.autoscale(tight=True)
    # plt.show()
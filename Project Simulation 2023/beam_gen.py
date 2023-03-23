import matplotlib.image as im
import numpy as np
import matplotlib.pyplot as plt
from scipy import special as sp

class BeamGen:
    
    def __init__(self, mode, ell, p, w0, r, phi, z, k, E0 = 1):
        """
        Initialization Method:: Initializes both defined and callable attributes. Sets self.beam attribute uesd by intensity() and phase(). 
        Uses::      __create_lg_mode()
                    __create_pixel_mode()
        Used By::   None
        """
        self.error_marg = 1 / 100 #Units percent beam waist - Percentage error margin for each pixel to be within the apperature radius. 
        self.mode = mode
        self.ell = ell
        self.p = p
        self.w0 = w0
        self.r = r
        self.R = np.amax(self.r)/np.sqrt(2)
        self.phi = phi
        self.z = z
        self.k = k
        self.E0 = E0
        self.beam = None
        [self.x, self.y] = self.__polar_2_cart([self.r, self.phi])

        if self.mode == "LG" or self.mode == "lg" : #Mode is chosen as LG 
            self.beam = self.__create_lg_mode()
        
        if self.mode == "pixel" :
            self.beam = self.__create_pixel_mode()

    def intensity(self):
        """
        Public Method:: Returns the  intensity of the generated beam
        Uses::               
        Used By::        
        Ret Val::       Intensity of self.beam as an numpy array of dytype float64
        """
        return (np.abs(self.beam))**2

    def phase(self):
        """
        Public Method:: Returns the phase of the generated beam 
        Uses::               
        Used By::       
        Ret Val::       phase of self.beam as an numpy array of dytype float64
        """
        PI = np.pi
        return (np.mod(np.angle(self.beam),2*PI))
    
    def __create_lg_mode(self):
        """
        Private Method:: Returns a complex array of an lg beam, sized according to self.r and self.phi sizes. See https://en.wikipedia.org/wiki/Gaussian_beam for the equation, under the section 'Laguerre-Gaussian modes'.
        Uses:: 
        Used By::   __init__()
        """
        PI = np.pi
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
        Private Method:: Returns a complex array of a Gaussian beam, sized according to self.r and self.phi sizes. See https://en.wikipedia.org/wiki/Gaussian_beam for the equation, under the section 'Mathematical form' 
        Uses::          __cart_2_polar_arr(), 
                        __draw_apperature_radius(), 
        Used By::       __init__
        """
        PI = np.pi
        u1 = np.array(self.__polar_2_cart([2*self.w0, 0]))
        u2 = np.array(self.__polar_2_cart([2*self.w0, PI/3]))
        i = 100 #define the absolute limit of linear parameters to run through each circle
        ui = np.empty([(2*i+1)**2, 2]) #defines an empty array that will be filled with the coordinates for pixel origin points (all vectors that are a lin. comb. of basis vectors)
        
        for a in range (-i, i+1): # Iterate over both coeff. of the linear combination of basis. Covers a wide range as to calculate an uncessecarily large set of pixel centers. Will condition to be less than the app. radius later.
            for b in range (-i, i+1):
                ui[(a+i)*(2*i+1)+(b+i), :] = a*u1 + b*u2 # Note the index of ui. It maps both indexs a and b to a 1D range from 0 -> (2*i+1)^2 -1. It took me a while to figure out T_T

        ui = ui[np.sqrt(ui[:,0]**2 + ui[:,1]**2) <= (self.R-self.w0+self.w0*self.error_marg)]
        
        ui = self.__cart_2_polar_arr(ui)

        # For each row (axis 1) in the array (for each pixel center) call the private method __for_each_pixel. Pass the array ui as an argument:
        circle_i = np.apply_along_axis(self.__for_each_pixel, 1, ui) # For each pixel, call the __for_each_pixel funciton
        
        pixel_img = np.sum(circle_i, axis=0) # flatten our list of arrays into the same array (add each img on top of each other)
        max_gauss_val = np.max(pixel_img)
        pixel_img = self.__draw_apperature_radius(pixel_img) # draw the apperature radius onto the img

        #max_gauss_val = pixel_img[(np.shape(pixel_img)[0]-1)/2, (np.shape(pixel_img)[1]-1)/2]
        pixel_img[pixel_img >= max_gauss_val] = max_gauss_val #this value is the maximum value for a single 'cone end'. Prevents overlap when summing each pixel.

        # plt.figure(figsize=(7,7))
        # plt.pcolormesh(self.x, self.y, pixel_img, cmap='Blues')
        # plt.grid()
        # plt.xlim([-self.R, self.R])
        # plt.ylim([-self.R, self.R])
        # plt.show()

        return pixel_img
   
    def __for_each_pixel(self, pixel_center): #
        """
        Private Method:: Used as a callback function :: Computes all operations done on each pixel. This includes: handeling the difference b/w analytic and actual points in the meshgrid, creating a gaussian beam at each pixel center, 
        Parameters::    pixel_center::Nx2 numpy array of dytpe float64 containing the (x, y) coordinate of each pixel center
        Uses::          __find_nearest(), 
                        __polar_2_cart(), 
                        __gaussian()
        Used By::       __create_pixel_mode()
        Ret Val::       An LxL gaussian beam centered at the given coordinate in a numpy array of dtype complex128
        """
        PI = np.pi
        nearest_idx = self.__find_nearest(pixel_center) # Find the nearest grid point to the analytically determined pixel-center coordinate
        nearest_r = self.r[nearest_idx[0], nearest_idx[1]] # Find the r and phi values corresponding to the nearest index found above
        nearest_phi = self.phi[nearest_idx[0], nearest_idx[1]]
        [nearest_x, nearest_y] = self.__polar_2_cart([nearest_r, nearest_phi]) # convert polar to cartesian coordinates. These coordinates now have a 1:1 with the meshgrid.
        
        # Apply a circular function to the meshgrid, centered via the cartesian coordinates found dircetly above. The result is an upside down cone centered at the pixel center
        shifted_cone = np.sqrt((self.x-nearest_x)**2 + (self.y-nearest_y)**2) 
        shifted_cone[shifted_cone <= self.w0] = self.w0 # set bottom section of cone to the value of the radius 
        shifted_cone[shifted_cone > self.w0] = 0 # cut off upper section of cone
        shifted_cone = shifted_cone/self.w0 # normalize

        # Apply Gaussian function to each pixel:: 
        # To apply a shifted polar function here, I shift the meshgrid itself by subtracting the pixel coordinates, and apply the function. No 'un-shifting' is required as the original X and Y meshgrids will be used to plot our 3rd dimension regardless.
        shifted_x = self.x - nearest_x
        shifted_y = self.y - nearest_y
        shifted_r = np.sqrt(shifted_x**2+shifted_y**2)
        gaussian_pixel = self.__gaussian(shifted_r)

        gaussian_pixel = np.multiply(gaussian_pixel, shifted_cone)

        # ---------------------------------------------------------------
        # Uncomment to see the plot of each individual pixel before being summed::
        
        # plt.figure(figsize=(7,7))
        # plt.pcolormesh(self.x, self.y, shifted_cone, cmap='Blues')
        # plt.grid()
        # plt.xlim([-self.R, self.R])
        # plt.ylim([-self.R, self.R])
        # plt.show()

        return gaussian_pixel
    
    def __draw_apperature_radius(self, img):
        """
        Private Method:: Draws a circle around a given image
        Parameters::    img::An LxL numpy array containing the function of 2D LxL grid points.
        Uses::      
        Used By::       __create_pixel_mode()
        Ret Val::       An LxL numpy array of dtype complex128 containing the passed complex image with a superimposed circle of radius R.
        """
        #draw larger circle
        circle = np.sqrt((self.x)**2 + (self.y)**2) #technically an upside-down cone at this point
        circle[circle >= self.R] = 0 #slice the top off
        circle[circle <= (self.R-self.R*0.01)] = 0 #slice the bottom off, leaving a cicrle of width 1% of the radius
        circle[circle > 0] = 1 #Normalize nonzero points (the circle) to 1
        img = img + circle 
        return img

    def __gaussian(self, r):
        """
        Private Method:: Applys a Gaussian function to the passed meshgrid array, r
        Parameters::    r::LxL array containing grid points in polar coordinates, (r phi) 
        Uses::      
        Used By::       __for_each_pixel()
        Ret Val::       An LxL numpy array of dtype complex128 containing the Gaussian function.

        """
        PI = np.pi
        wavelength = (2*PI)/self.k
        zR = (PI*(self.w0**2))/(wavelength)
        
        Rc = self.z*(1.0+((zR/self.z)**2))
        wZ = self.w0*np.sqrt(1.0+((self.z/zR)**2))


        term1 = self.w0/wZ
        term2 = np.exp((-r**2)/(wZ**2))
        term3 = np.exp(-1.0j*(self.k*self.z + self.k*(r**2)/(2*Rc) + np.arctan2(self.z,zR)))
        gaussian_beam = self.E0*term1*term2*term3
        return gaussian_beam
    
    def __find_nearest(self, value):
        """
        Private Method:: Used to find the index of the nearest value
        Parameters::    value:: 1x2 numpy array of dtype float64 containing the (r, phi) value.
        Uses::          __dist_in_polar()
        Used By::       __for_each_pixel()
        """
        return np.unravel_index(np.argmin(self.__dist_in_polar(self.r, value[0], self.phi, value[1])), self.phi.shape)
    
    def __dist_in_polar(self, r2, r1, phi2, phi1):
        """
        Private Method:: Used to calculate the eulidian distance between 2 points given in polar coordinates
        Parameters::    r2:: 2nd r value, dtype float64
                        r1:: 1st r value, dtype float64
                        phi2:: 2nd phi value, dtype float64
                        phi1:: 1st phi value, dtype float64
        Uses::
        Used By::       __find_nearest()
        """
        return np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(phi2 - phi1))

    def __polar_2_cart(self, polar_cord):
        """
        Private Method:: Used to calculate the eulidian distance between 2 points given in polar coordinates
        Parameters::    polar_cord:: 1x2 array = [r, phi]
        Uses::
        Used By::       __for_each_pixel()
        Ret Val::       2x1 array of coordinates [x, y] 
        """
        r = polar_cord[0]
        phi = polar_cord[1]
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return [x, y]

    def __cart_2_polar_arr(self, cart_cord):
        """
        Private Method:: Converts an array of cartesian coordinates to polar
        Parameters::    cart_cord:: Nx2 numpy array of (x, y) coordinates
        Uses::
        Used By::       __create_pixel_mode()
        Ret Val::       Nx2 numpy array of (r, phi) coordinates
        """
        PI = np.pi
        x = cart_cord[:, 0]
        y = cart_cord[:, 1]
        r = np.sqrt((x)**2+(y)**2)
        phi = np.mod(np.arctan2(y,x),2*PI)
        return np.column_stack((r, phi))

###---------------------------------USE CASE EXAMPLE--------------------------------------

if __name__ == "__main__":
    PI = np.pi
    wavelength = 810E-6
    beamWaist = 1
    wavevector = (2.0*PI)/wavelength

    app_radius = 3
    precision = 1000

    x = np.linspace(-app_radius,app_radius,precision+1); ## Grid points along x
    y = np.linspace(-app_radius,app_radius,precision+1) ## Grid points along y
    X,Y = np.meshgrid(x,y)
    r = np.sqrt(X**2+Y**2)
    phi = np.mod(np.arctan2(Y,X),2*PI)
    
    # plt.plot(X, Y, marker='.', color='k', linestyle='none')
    # plt.show()

    lg_beam = BeamGen("pixel",4,0,beamWaist,r,phi,0.000001,wavevector)

    aa = lg_beam.intensity()
  
    plt.figure(figsize=(7,7))
    plt.pcolormesh(X, Y, lg_beam.intensity(), cmap='Blues')
    plt.grid()
    plt.xlim([-app_radius, app_radius])
    plt.ylim([-app_radius, app_radius])
    plt.show()

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
import numpy as np 
import math 

class Channel: 

    FREE_SPACE = 0 
    LENS = 1 
    ABBARATION = 2 

    def __init__(self, type, len = 0, diam = 0, n = [], m = []): 
        """
        Constructor of Channel class. 

        @type type: integer from set of class constants
        @param type: Number corresponds to type of channel 
        @type len: Number 
        @param len: Length of free space channel (Only save to class if type == FREE_SPACE) 
        @type diam: Number
        @param diam: Diameter of lens (Only save to class if type == LENS) 
        @type n: Array of integers (Same length as m)
        @param n: Each integer corresponds to the n parameter of the corresponding Zernike polynomial (Only save to class if type == ABBARATION)
        @type m: Array of integers (Same length as n)
        @param m: Each integer corresponds to the n parameter of the corresponding Zernike polynomial (Only save to class if type == ABBARATION)
        """
        self.type = type 

        if self.type == FREE_SPACE: # Set distance of propagation if channnel is LOS free space 
            self.len = len 

        elif self.type == LENS: # Set diameter of channel type is lens 
            self.diam = diam 

        elif self.type == ABBARATION: # Set Zernike polynomial types for the abbaration 
            if len(n) == len(m): 
                self.n = n 
                self.m = m 
            else:  # Arrays of Zernike indices must be the same
                raise Exception("Indices vectors of Zernike polynomials must be the same length (len(n) != len(m)).")
            
        else:  
            raise Exception("Class type is not defined; type must be an integer corresponding to one of the possible channels. ")

    def output_beam(self, beam, k, L = -1, LX = -1, LY = -1): 
        """
        Computes a 2x2 array representing the beam front of the beam leaving the channel 
        by using a 2x2 array representing the beam front of the beam entering the channel. 

        @type beam: 2x2 array 
        @param beam: Beam front going into the channel 
        @type k: Number 
        @param k: Wave vector of the beam in z direction (rad/m)
        @type L: Number 
        @param L: Length of beam if Lx == Ly == L (m) 
        @type LX: Number 
        @param LX: Length of x direction of beam (m) 
        @type LY: Number 
        @param LY: Length of y direction of beam (m) 

        @rtype: 2x2 array 
        @return: Beam front of beam leaving the channel 
        """

        # Define beam lengths LX and LY
        if ((LX == -1) or (LY == -1)) and (L != -1): # One of LX or LY not specified but L specified 
            LX = L
            LY = L 
        elif ((LX == -1) or (LY == -1)) and (L == -1): # One of LX or LY not specified but L not specified 
            raise Exception("Beam length must be specified. ")

        # Get beam output based on channel type and parameters 

        if self.type == FREE_SPACE: # Case of free space channel 
            output = __propTF(beam,L,k,self.dist) 

        elif self.type == LENS: # Case of lens channel 
            output = -1 

        elif self.type == ABBARATION: # Case of abbaration channel 
            output = -1

        # Return output beam 
        return output 

    def __propTF(u1,LX, LY, k,z):
        """
        This function computes the propagator of an input beam with the given parameters 
        after going through distance z. 

        @type u1: 2D array of numbers 
        @param u1: Represents beamfront of wave coming into channel 
        @type LX: Number 
        @param LX: Length of beam in the x direction 
        @type LY: Number 
        @param LY: Length of beam in the y direction 
        @type k: Number 
        @param k: Wavevector of beam in the z direction 
        @type z: Number 
        @param z: Distance of propagation in z direction 

        @rtype u2: 2D array of numbers  
        @return u2: Beam coming out of the channel
        """

        # Beam array size 
        M,nn=u1.shape

        # Steps in x and y directions
        dx=LX/M
        dy=LY/M

        # Array of wave vector range in x and y directions
        kx=np.arange(-1/(2*dx),1/(2*dx),1/LX)
        ky=np.arange(-1/(2*dy),1/(2*dy),1/LY)

        # Grid of wave vectors for evaluating values for each point in spacial frequency doman 
        Kx, Ky = np.meshgrid(kx, ky)

        # Compute propagator 
        H=np.exp(-1j*z*(Kx**2+Ky**2)/(2*k)) # Frequency domain filter 
        U2=H*np.fft.fftshift(np.fft.fft2(u1)) # Apply filter to FT of the beam at z0 (After centering the frequencies)
        u2=np.fft.ifft2(np.fft.ifftshift(U2)) # Apply inverse FT to get beam at z (After centering the frequencies of U2) 

        # Return output beam 
        return u2 

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

    def __RR(m,n):
        """
        Comptues array of coefficients and powers of the radial part of the Zernike polynomial 
        given indices m and n. 

        @type m: Integer 
        @param m: Index m of Zernike polynomial 
        @type n: Integer 
        @param n: Index n of Zernike polynomial 

        @rtype [coeff,powr]: Array of arrays (2D array) 
        @return [coeff,powr]: Array which contains array coeff of coefficients and array powr of corresponding powers. 
                              Ex. variable i will be coeff[i]*rho**powr[i]
        """
        nm=n-m
        coeff=[]
        powr=[]
        if (n-m)%2==0:
            for kk in range(0,int((n-m)/2+1)):
                aa=((-1)**kk*math.factorial(n-kk))/(math.factorial(kk)*math.factorial(int((n+m)/2)-kk)*math.factorial(int((n-m)/2-kk)))
                bb=n-2*kk
                coeff.append(aa)
                powr.append(bb)
        else: 
            coeff.append(0)
            powr.append(0)
        return([coeff,powr])

    def __Zernike(RHO,PHI,m,n):
        """
        Computes values of Zernike polynomial of indices m and n at the positions corresponding to the ranges 
        RHO and PHI in polar coordinates. 

        @rtype RHO: 1D Array 
        @return RHO: Coordinate distances in rho direction (distance from center)
        @rtype PHI: 1D Array 
        @return PHI: Coordinate angles in phi direction (angle from positive x axis)
        @type m: Integer 
        @param m: Index m of Zernike polynomial 
        @type n: Integer 
        @param n: Index n of Zernike polynomial 

        @rtype P: 2D Array 
        @return P: Zernike polynomial in polar coordinates 
        """
        ZR=np.zeros(RHO.shape); 
        rn=RR(np.abs(m),n)
        for ii in range(len(rn[0])):
            ZR=ZR+rn[0][ii]*RHO**rn[1][ii]

        if m>=0: 
            Z=ZR*np.cos(np.abs(m)*PHI)
        else:
            Z=ZR*np.sin(np.abs(m)*PHI)

        # M=(RHO<=1)
        # P=Z*M
        P = Z
        return(P)
    
    def __ApertureFilter(R, X, Y): pass 



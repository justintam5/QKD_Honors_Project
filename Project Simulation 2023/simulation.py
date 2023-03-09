from Channel import Channel
from beam_gen import BeamGen

import numpy as np 

class Simulation: 

    def __init__(self, name = "New Simulation", L = -1, LX = -1, LY = -1, N = -1, NX = -1, NY = -1, wavelength = -1, units = "m"):  
        """
        This is the contructor for the Simulation class. 

        @type name: string 
        @param name: Name associated with current simulation 
        @type L: Number 
        @param L: Physical Length of both x and y dimensions of simulation (-1 if indivudual x and y lengths are defined)
        @type LX: Number 
        @param LX: Physical Length of x dimension of simulation (-1 if same length is defined for both x and y)
        @type LY: Number 
        @param LY: Physical Length of y dimension of simulation (-1 if same length is defined for both x and y)
        @type N: Integer 
        @param N: Length of both x and y dimensions of array in simulation (-1 if indivudual x and y lengths are defined)
        @type NX: Integer 
        @param NX: Length of x dimension of array in simulation (-1 if same length is defined for both x and y)
        @type NY: Integer 
        @param NY: Length of y dimension of array in simulation (-1 if same length is defined for both x and y)
        @type wavelength: Number 
        @param wavelength: Wavelength of beams used in simulation (In defined unit; default is m)
        @type units: String 
        @param units: Length units to be used when communicating with simulation 
        """
        
        self.name = name 
        self.channels = []
        self.beam_generators = [] 
        self.num_chanels = 0 
        self.num_beam_generators = 0
        self.beams = []
        self.units = units 

        # Get unit multiplier 
        if units == "m": 
            multiplier = 1
        elif units == "cm": 
            multiplier = 1E-3
        elif units == "um": 
            multiplier = 1E-6
        elif units == "nm": 
            multiplier = 1E-9 
        elif units == "pm": 
            multiplier = 1E-12
        else: 
            raise Exception("Invalid unit entered.")
        
        self.multiplier = multiplier
        
        # Define beam lengths LX and LY
        if ((LX == -1) or (LY == -1)) and (L != -1): # One of LX or LY not specified but L specified 
            self.LX = L * multiplier
            self.LY = L * multiplier
        elif ((LX == -1) or (LY == -1)) and (L == -1): # One of LX or LY not specified but L not specified 
            raise Exception("Beam length must be specified. ")
        else: 
            self.LX = LX * multiplier
            self.LY = LY * multiplier

        # Define array lengths NX and NY
        if ((NX == -1) or (NY == -1)) and (N != -1): # One of NX or NY not specified but N specified 
            self.NX = N
            self.NY = N 
        elif ((LX == -1) or (LY == -1)) and (L == -1): # One of NX or NY not specified but N not specified 
            raise Exception("Beam array length must be specified. ")
        else: 
            self.NX = NX 
            self.NY = NY

        # Ensure valid wavelength was entered 
        if wavelength == -1: 
            raise Exception("Wavelength not defined for simulation") 
        else: 
            self.wavelength = wavelength * multiplier

    def add_beam_gen(self, mode = "LG", ell = 0, p = 0, w0 = 1.0, index = -1):         
        """
        
        """

        wavevector = (2*np.pi)/self.wavelength # Get wave vector 
        z = 0.00001 # Set z to very small value near origin to avoid divide by 0 error 
        x = np.linspace(-self.LX/2,self.LX/2,self.NX) ## Grid points along x
        y = np.linspace(-self.LY/2,self.LY/2,self.NY) ## Grid points along y
        X,Y = np.meshgrid(x,y)
        r, phi = Simulation.__cart2pol(X, Y) # Matrices of corresponding r and phi values of beam array 

        # If index was not entered, append new beam generator to end of the array
        if index == -1: 
            self.beam_generators.append(BeamGen(mode, ell, p, w0, r, phi, z, wavevector))

        # If index was entered, ensure that it is within a valid range
        elif index < self.num_beam_generators: 
            self.channels.insert(index, BeamGen(mode, ell, p, w0, r, phi, z, k)) 
        
        # Throw exception if invalid index was entered 
        else: 
            raise Exception("Index out of range (index == %d)", index) 
        
        # Increase count of channels 
        self.num_beam_generators = self.num_beam_generators + 1

    def add_channel(self, type, dist = 0, diam = 0, n = [], m = [], app = 0, stre = [], index = -1,):
        """
        
        """

        # If index was not entered, append new channel to end of the array
        if index == -1: 
            self.channels.append(Channel(type, dist = dist, diam = diam, n = n, m = m, app = app, stre = stre))

        # If index was entered, ensure that it is within a valid range 
        elif index < self.num_channels: 
            self.channels.insert(index, Channel(type, dist = dist, diam = diam, n = n, m = m, app = app, stre = stre)) 
        
        # Throw exception if invalid index was entered 
        else: 
            raise Exception("Index out of range (index == %d)", index) 
        
        # Increase count of channels 
        self.num_chanels = self.num_chanels + 1 

    def run(self, beam_gen_index = -1): pass

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

if __name__ == "__main__": 
    channels = []
    channels.append(Channel(type = Channel.ABBARATION, n = [3, 1, 4], m = [1, 1, 2], stre = np.array([0.8, 0.9, 0.7]), app = 5))
    channels.append(Channel(type = Channel.LENS, diam = 10))
    channels.insert(1, Channel(type = Channel.FREE_SPACE, dist = 100000))

    print(channels[1]) 
from Channel import Channel
from beam_gen import BeamGen

import numpy as np 
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix

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
        
        # Define object variables 
        self.name = name 
        self.channels = []
        self.beam_generators = [] 
        self.measurement_basis = []
        self.measurement_labels = []
        self.num_channels = 0 
        self.num_beam_generators = 0
        self.runs = []
        self.beam_labels = []
        self.num_runs = 0
        self.units = units 

        # Get unit multiplier 
        if units == "m": 
            multiplier = 1
        elif units == "cm": 
            multiplier = 1E-2
        elif units == "mm": 
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
        elif ((NX == -1) or (NY == -1)) and (N == -1): # One of NX or NY not specified but N not specified 
            raise Exception("Beam array length must be specified. ")
        else: 
            self.NX = NX 
            self.NY = NY

        # Ensure valid wavelength was entered 
        if wavelength == -1: 
            raise Exception("Wavelength not defined for simulation") 
        else: 
            self.wavelength = wavelength * multiplier

    def add_beam_gen(self, mode = "LG", p = 0, ell = 0,  beam_waist = 1.0, index = -1):         
        """
        This function addes a beam generator of a specified mode and given parameter to the simulation. 
        This beam generator can then be selected when running a simulation, or compared with the rest of the 
        beam generators as possible bases. 

        @type mode: String 
        @param mode: Mode of beam to be generated by this beam generator (Defaults to LG modes) 
        @type ell: Integer 
        @param ell: Index ell of LG mode 
        @type p: Integer 
        @param p: Index p of LG mode 
        @type beam_waist: Number 
        @param beam_waist: Waist of generated beam at origin 
        @type index: Integer 
        @param index: Index of location in beam_generators array where the new beam_gen is placed (default -1 will append to end of array) 
        """

        wavevector = (2*np.pi)/self.wavelength # Get wave vector 
        z = 0.00001 # Set z to very small value near origin to avoid divide by 0 error 
        x = np.linspace(-self.LX/2,self.LX/2,self.NX) ## Grid points along x
        y = np.linspace(-self.LY/2,self.LY/2,self.NY) ## Grid points along y
        X,Y = np.meshgrid(x,y)
        r, phi = Simulation.__cart2pol(X, Y) # Matrices of corresponding r and phi values of beam array 

        # If index was not entered, append new beam generator to end of the array
        if index == -1: 
            self.beam_generators.append(BeamGen(mode, ell, p, beam_waist * self.multiplier, r, phi, z, wavevector))
            if mode == "LG":
                self.beam_labels.append(("(p = " + str(p) + ", l = " + str(ell) + ")"))

        # If index was entered, ensure that it is within a valid range
        elif index < self.num_beam_generators: 
            self.beam_generators.insert(index, BeamGen(mode, ell, p, beam_waist * self.multiplier, r, phi, z, wavevector)) 
            if mode == "LG":
                self.beam_labels.insert(index, ("(p = " + str(p) + ", l = " + str(ell) + ")"))
        
        # Throw exception if invalid index was entered 
        else: 
            raise Exception("Index out of range (index == %d)", index) 
        
        # Increase count of channels 
        self.num_beam_generators = self.num_beam_generators + 1

    def add_measurement_basis(self, mode="LG", p=0, ell=0,  beam_waist=1.0, index=-1):
        """
        This function addes a beam generator of a specified mode and given parameter to the simulation. 
        This beam generator can then be selected when running a simulation, or compared with the rest of the 
        beam generators as possible bases. 

        @type mode: String 
        @param mode: Mode of beam to be generated by this beam generator (Defaults to LG modes) 
        @type ell: Integer 
        @param ell: Index ell of LG mode 
        @type p: Integer 
        @param p: Index p of LG mode 
        @type beam_waist: Number 
        @param beam_waist: Waist of generated beam at origin 
        @type index: Integer 
        @param index: Index of location in beam_generators array where the new beam_gen is placed (default -1 will append to end of array) 
        """

        wavevector = (2*np.pi)/self.wavelength  # Get wave vector
        z = 0.00001  # Set z to very small value near origin to avoid divide by 0 error
        x = np.linspace(-self.LX/2, self.LX/2, self.NX)  # Grid points along x
        y = np.linspace(-self.LY/2, self.LY/2, self.NY)  # Grid points along y
        X, Y = np.meshgrid(x, y)
        # Matrices of corresponding r and phi values of beam array
        r, phi = Simulation.__cart2pol(X, Y)

        # If index was not entered, append new beam generator to end of the array
        if index == -1:
            self.measurement_basis.append(BeamGen(mode, ell, p, beam_waist * self.multiplier, r, phi, z, wavevector).beam)
            if mode == "LG":
                self.measurement_labels.append(
                    ("(p = " + str(p) + ", l = " + str(ell) + ")"))

        # If index was entered, ensure that it is within a valid range
        elif index < self.num_beam_generators:
            self.measurement_basis.insert(index, BeamGen(mode, ell, p, beam_waist * self.multiplier, r, phi, z, wavevector).beam)
            if mode == "LG":
                self.measurement_labels.insert(index, ("(p = " + str(p) + ", l = " + str(ell) + ")"))

        # Throw exception if invalid index was entered
        else:
            raise Exception("Index out of range (index == %d)", index)

    def add_channel(self, type, dist = 0, diam = 0, n = [], m = [], app = 0, stre = [], index = -1,):
        """
        This function adds a channel through which the generated beams travel. 

        @type type: Integer 
        @param type: Integer corresponding to type of channel being added (Values of possible integers stored in Channel class) 
        @type dist: Number 
        @param dist: Distance of travel for free space channel 
        @type diam: Number 
        @param diam: Diameter of apperture filter for lens channel
        @type n: Array of intergers 
        @param n: Indices n of Zernike polynomials to be added in abberation channel 
        @type m: Array of intergers 
        @param m: Indices m of Zernike polynomials to be added in abberation channel 
        @type app: Number 
        @param app: Apperture of aberations for aberation channel 
        @type stre: Array of numbers  
        @param stre: Strengths of corresponding Zerinke polynomials applied in abberation channel 
        @type index: Integer 
        @param index: Index of position in channels array in which the new channel is added (default -1 will append to end of array) 
        """

        # If index was not entered, append new channel to end of the array
        if index == -1: 
            self.channels.append(Channel(type, dist = dist * self.multiplier, diam = diam * self.multiplier, n = n, m = m, app = app * self.multiplier, stre = stre))

        # If index was entered, ensure that it is within a valid range 
        elif index < self.num_channels: 
            self.channels.insert(index, Channel(type, dist=dist * self.multiplier,
                                 diam=diam * self.multiplier, n=n, m=m, app=app * self.multiplier, stre=stre))
        
        # Throw exception if invalid index was entered 
        else: 
            raise Exception("Index out of range (index == %d)", index) 
        
        # Increase count of channels 
        self.num_channels = self.num_channels + 1 

    def run(self, beam_gen_indices = []): 
        """
        This function is used to run the simulation and generate beam arrays for different runs and different channel input/outputs.

        @type beam_gen_indices: Array of Integers 
        @param beam_gen_indices: Indices of beam generators over which the current run should be performed 
        """

        # If no particular beam generator indices selected, run for all beams 
        if beam_gen_indices == []: 
            for beam_gen_index in range(self.num_beam_generators):
                beams = self.__run_per_beam_gen(beam_gen_index)
                self.runs.append(beams)
                self.num_runs = self.num_runs + 1

        # Runs for subset of beam generators 
        else:
            for beam_gen_index in beam_gen_indices: 
                beams = self.__run_per_beam_gen(beam_gen_index) 
                self.runs.append(beams)
                self.num_runs = self.num_runs + 1
        
    def __run_per_beam_gen(self, beam_gen_index): 
        """
        This function is used to run the simulation for a single beam generator and return the resultant array of beams for each channel input/ouput

        @type beam_gen_index: Integer 
        @param beam_gen_index: Index of beam generator to be used for current run 

        @rtype beams: An Array of 2D arrays of numbers 
        @return beams: An array, with each element being the beam at a corresponding index within its path, where index 0 represents the initially 
        generated beam, and each index after represents the beam coming out of the corresponding channel with index lower than beam index by 1. 
        """

        # Ensure the beam generator index is valid 
        if (beam_gen_index < 0) or (beam_gen_index >= self.num_beam_generators):
            raise Exception("Index of beam generator out of bounds. ")

        # Define array to store the beams across the different channels 
        beams = []

        # Store current beam as the initial beam and append to beams array 
        current_beam = self.beam_generators[beam_gen_index].beam
        beams.append(current_beam)   

        # Loop through channels getting the output of each one and appending to the beams array 
        for ch in self.channels: 
            current_beam = ch.output_beam(current_beam, (2*np.pi)/self.wavelength, LX = self.LX, LY = self.LY) 
            beams.append(current_beam) 

        # Return the array of beams computed 
        return beams 
    
    def plot_beams(self, run_indices = [], channel_index = -1, channel_indices = [], run_index = -1, plot_measurement_basis = False): 
        """
        This function is used to plot the magnitude and phase of the beamfronts for different runs (i.e. different beam generators) 
        and for different points within the channels path (i.e. in between different channels).

        Note that this function can be called in one of two ways: 
        1) For a particular channel index, and a set of run indices (Including all runs if not specified). 
        2) For a particular run_index and a set of channel indices (Including all channels if not specified). 

        For option 1, a particular channel output will be plotted for a set of runs (beam generators), and 
        for option 2 a particular run will be plotted for a set of channel outputs.   

        @type run_indices: Array of integers 
        @param run_indices: Indices of runs to be plotted (leave unspecified to plot all runs)
        @type channel_index: Integer
        @param channel_index: Index of channel input/output to be plotted with 0 being the initially generated beam
        @type channel_indices: Array of integers 
        @param channel_indices: Indices of channel input/outputs to be plotted (leave unspecified to plot all channels)
        @type run_index: Integer 
        @param run_index: Index of run to be plotted
        """

        # Define variables for axes and plots 
        ax_mag = []
        ax_phase = []
        plot_mag = []
        plot_phase = []

        # Define X and Y variables 
        x = np.linspace(-self.LX/2,self.LX/2,self.NX) ## Grid points along x
        y = np.linspace(-self.LY/2,self.LY/2,self.NY) ## Grid points along y
        X,Y = np.meshgrid(x,y)

        # Define plot index 
        plot_idx = 1

        # Plot for all runs at a particular channel index 
        if (run_indices == []) and (channel_index >= 0) and (channel_index <= self.num_channels): 

            # Initialize figure 
            figInit = plt.figure()

            for run_idx in range(self.num_runs): 
                # Get current beam 
                beam = self.runs[run_idx][channel_index]

                # Plot magnitude of current beam 
                ax_mag.append(figInit.add_subplot(self.num_runs, 2, plot_idx))
                plot_mag.append(ax_mag[run_idx].pcolormesh(X/self.multiplier, Y/self.multiplier, np.abs(beam)**2, cmap="inferno"))
                ax_mag[run_idx].set_aspect('equal')
                ax_mag[run_idx].autoscale(tight=True)

                # Add labels to subplot 
                ax_mag[run_idx].set(xlabel = ("x [" + self.units + "]"), ylabel = ("y [" + self.units + "]"))
                ax_mag[run_idx].set_title(self.beam_labels[run_idx] + " (Magnitude)")

                # Increment plot index 
                plot_idx = plot_idx + 1

                # Plot phase of current beam 
                ax_phase.append(figInit.add_subplot(self.num_runs, 2, plot_idx))
                plot_phase.append(ax_phase[run_idx].pcolormesh(X/self.multiplier, Y/self.multiplier, np.mod(np.angle(beam), 2*np.pi),cmap="hsv", vmin=0, vmax=2*np.pi))
                ax_phase[run_idx].set_aspect('equal')
                ax_phase[run_idx].autoscale(tight=True)

                # Add labels to subplot
                ax_phase[run_idx].set(xlabel=("x [" + self.units + "]"), ylabel=("y [" + self.units + "]"))
                ax_phase[run_idx].set_title(self.beam_labels[run_idx] + " (Phase)")

                # Increment plot index
                plot_idx = plot_idx + 1
            
            # Display plot 
            plt.show()

        # Plot for specified runs at a particular channel index 
        elif (run_indices != []) and (channel_index >= 0) and (channel_index <= self.num_channels): 

            # Initialize figure 
            figInit = plt.figure()

            for run_idx in range(len(run_indices)): 
                # Get current beam 
                beam = self.runs[run_indices[run_idx]][channel_index]

                # Plot magnitude of current beam 
                ax_mag.append(figInit.add_subplot(len(run_indices), 2, plot_idx))
                plot_mag.append(ax_mag[run_idx].pcolormesh(X/self.multiplier, Y/self.multiplier, np.abs(beam)**2, cmap="inferno"))
                ax_mag[run_idx].set_aspect('equal')
                ax_mag[run_idx].autoscale(tight=True)

                # Add labels to subplot 
                ax_mag[run_idx].set(xlabel = ("x [" + self.units + "]"), ylabel = ("y [" + self.units + "]"))
                ax_mag[run_idx].set_title(self.beam_labels[run_idx] + " (Magnitude)")

                # Increment plot index 
                plot_idx = plot_idx + 1

                # Plot phase of current beam 
                ax_phase.append(figInit.add_subplot(len(run_indices), 2, plot_idx))
                plot_phase.append(ax_phase[run_idx].pcolormesh(X/self.multiplier, Y/self.multiplier, np.mod(np.angle(beam), 2*np.pi),cmap="hsv", vmin=0, vmax=2*np.pi))
                ax_phase[run_idx].set_aspect('equal')
                ax_phase[run_idx].autoscale(tight=True)

                # Add labels to subplot
                ax_phase[run_idx].set(xlabel=("x [" + self.units + "]"), ylabel=("y [" + self.units + "]"))
                ax_phase[run_idx].set_title(self.beam_labels[run_idx] + " (Phase)")

                # Increment plot index
                plot_idx = plot_idx + 1
            
            # Display plot 
            plt.show()

        # Plot for all channels at a particular run 
        elif (channel_indices == []) and (run_index >= 0) and (run_index < self.num_runs):
            # Initialize figure 
            figInit = plt.figure()

            for channel_idx in range(self.num_channels + 1): 
                # Get current beam 
                beam = self.runs[run_index][channel_idx]

                # Plot magnitude of current beam 
                ax_mag.append(figInit.add_subplot((self.num_channels + 1), 2, plot_idx))
                plot_mag.append(ax_mag[channel_idx].pcolormesh(X/self.multiplier, Y/self.multiplier, np.abs(beam)**2, cmap="inferno"))
                ax_mag[channel_idx].set_aspect('equal')
                ax_mag[channel_idx].autoscale(tight=True)
                
                # Add labels to subplot 
                ax_mag[channel_idx].set(xlabel = ("x [" + self.units + "]"), ylabel = ("y [" + self.units + "]"))
                ax_mag[channel_idx].set_title("Channel " + str(channel_idx) + " (Magnitude)")

                # Increment plot index 
                plot_idx = plot_idx + 1

                # Plot phase of current beam 
                ax_phase.append(figInit.add_subplot((self.num_channels + 1), 2, plot_idx))
                plot_phase.append(ax_phase[channel_idx].pcolormesh(X/self.multiplier, Y/self.multiplier, np.mod(np.angle(beam), 2*np.pi),cmap="hsv", vmin=0, vmax=2*np.pi))
                ax_phase[channel_idx].set_aspect('equal')
                ax_phase[channel_idx].autoscale(tight=True)

                # Add labels to subplot
                ax_phase[channel_idx].set(xlabel=("channel_idx [" + self.units + "]"), ylabel=("y [" + self.units + "]"))
                ax_phase[channel_idx].set_title("Channel " + str(channel_idx) + " (Phase)")

                # Increment plot index
                plot_idx = plot_idx + 1
            
            # Display plot 
            plt.show()

        # Plot for specified channels at a particular run
        elif (channel_indices != []) and (run_index >= 0) and (run_index < self.num_runs):
            # Initialize figure 
            figInit = plt.figure()

            for channel_idx in range(len(channel_indices)): 
                # Get current beam 
                beam = self.runs[run_index][channel_indices[channel_idx]]

                # Plot magnitude of current beam 
                ax_mag.append(figInit.add_subplot(len(channel_indices), 2, plot_idx))
                plot_mag.append(ax_mag[channel_idx].pcolormesh(X/self.multiplier, Y/self.multiplier, np.abs(beam)**2, cmap="inferno"))
                ax_mag[channel_idx].set_aspect('equal')
                ax_mag[channel_idx].autoscale(tight=True)

                # Add labels to subplot 
                ax_mag[channel_idx].set(xlabel = ("x [" + self.units + "]"), ylabel = ("y [" + self.units + "]"))
                ax_mag[channel_idx].set_title("Channel " + str(channel_indices[channel_idx]) + " (Magnitude)")

                # Increment plot index 
                plot_idx = plot_idx + 1

                # Plot phase of current beam 
                ax_phase.append(figInit.add_subplot(len(channel_indices), 2, plot_idx))
                plot_phase.append(ax_phase[channel_idx].pcolormesh(X/self.multiplier, Y/self.multiplier, np.mod(np.angle(beam), 2*np.pi),cmap="hsv", vmin=0, vmax=2*np.pi))
                ax_phase[channel_idx].set_aspect('equal')
                ax_phase[channel_idx].autoscale(tight=True)

                # Add labels to subplot
                ax_phase[channel_idx].set(xlabel=("channel_idx [" + self.units + "]"), ylabel=("y [" + self.units + "]"))
                ax_phase[channel_idx].set_title("Channel " + str(channel_indices[channel_idx]) + " (Phase)")

                # Increment plot index
                plot_idx = plot_idx + 1
            
            # Display plot 
            plt.show()

        elif plot_measurement_basis: 
            # Initialize figure
            figInit = plt.figure()

            for beam_idx in range(len(self.measurement_basis)):
                # Get current beam
                beam = self.measurement_basis[beam_idx]

                # Plot magnitude of current beam
                ax_mag.append(figInit.add_subplot(len(self.measurement_basis), 2, plot_idx))
                plot_mag.append(ax_mag[beam_idx].pcolormesh(X/self.multiplier, Y/self.multiplier, np.abs(beam)**2, cmap="inferno"))
                ax_mag[beam_idx].set_aspect('equal')
                ax_mag[beam_idx].autoscale(tight=True)

                # Add labels to subplot 
                ax_mag[beam_idx].set(xlabel = ("x [" + self.units + "]"), ylabel = ("y [" + self.units + "]"))
                ax_mag[beam_idx].set_title(self.measurement_labels[beam_idx] + " (Magnitude)")

                # Increment plot index
                plot_idx = plot_idx + 1

                # Plot phase of current beam
                ax_phase.append(figInit.add_subplot(len(self.measurement_basis), 2, plot_idx))
                plot_phase.append(ax_phase[beam_idx].pcolormesh(X/self.multiplier, Y/self.multiplier, np.mod(np.angle(beam), 2*np.pi),cmap="hsv", vmin=0, vmax=2*np.pi))
                ax_phase[beam_idx].set_aspect('equal')
                ax_phase[beam_idx].autoscale(tight=True)

                # Add labels to subplot
                ax_phase[beam_idx].set(xlabel=("x [" + self.units + "]"), ylabel=("y [" + self.units + "]"))
                ax_phase[beam_idx].set_title(self.measurement_labels[beam_idx] + " (Phase)")

                # Increment plot index
                plot_idx = plot_idx + 1

            # Display plot
            plt.show()
            

        # Exception if none of the above combinations were entered
        else: 
            raise Exception("Invalid parameters for plot_beams function.")
    
    def delete_runs(self): 
        """
        This function is used to delete all runs data to avoid appending to old runs (Also rests number of runs). 
        """
        self.runs = [] 
        self.num_runs = 0

    def delete_measurement_basis(self):
        """
        This function is used to delete all measurement basis data to avoid appending to previously added bases if necessary.
        """
        self.measurement_basis = []

    def compute_detection_matrix(self, channel_index, run_indices=[], use_measurement_basis = False):
        """
        This function computes the detection matrix between the beams at a particular channel index and either the originally 
        generated beams, or a seperately defined measurement basis. Note that the computed detection matrix is normalized for 
        each row (i.e. the values are normalized so that the probabilities in each row add up to 1). 

        @type channel_index: Integer 
        @param channel_index: Index of channel for which the matrix is computed
        @type run_indices: Array of integers
        @param run_indices: Indices of runs (beams) for which matrix will be computed 
        @type use_measurement_basis: Boolean 
        @param use_measurement_basis: Indicator for using measurement basis to detect beams rather than the originally generated beams 

        @rtype normalized_detection_matrix: Matrix of numbers values between 0 and 1
        @return normalized_detection_matrix: Normalized detection matrix between specified measurement basis or originally generated beams, 
        and beams at specified channel index 
        """

        # If no run indices defined, compute for all runs 
        if run_indices == []: 
            run_indices = range(self.num_runs)

        # Define detection matrix
        detection_matrix = np.zeros([len(run_indices), len(run_indices)])

        # Populate detection matrix 
        if use_measurement_basis: # Use measurement basis different than original beams 
            # Ensure that length of measurement basis matches number of beam generators 
            if len(self.measurement_basis) != self.num_runs:  
                raise Exception("Number of beams in measurement basis does not match number of generated beams. ")

            for row_idx in run_indices:  # Sift through rows
                for col_idx in run_indices:  # Sift through columns
                    # Use corresponding beam in measurement basis list for columns to indicate computation with respect to initial beam
                    detection_matrix[row_idx, col_idx] = Simulation.__normalized_inner_product(self.measurement_basis[row_idx], self.runs[col_idx][channel_index])
        else: # Use original generated beam for comparison 
            for row_idx in run_indices: # Sift through rows 
                for col_idx in run_indices: # Sift through columns
                    # Use channel index of 0 for clumns to indicate computation with respect to initial beam 
                    detection_matrix[row_idx, col_idx] = Simulation.__normalized_inner_product(self.runs[row_idx][0],self.runs[col_idx][channel_index])

        # Normalize rows in detection matrix 
        normalized_detection_matrix = np.zeros([len(run_indices), len(run_indices)])

        for row_idx in run_indices: # Sift through rows 
            for col_idx in run_indices: # Sift through columns
                normalized_detection_matrix[row_idx, col_idx] = detection_matrix[row_idx, col_idx] / (np.sum(detection_matrix[row_idx,:]))

        # Return normalized detection matrix 
        return normalized_detection_matrix
    
    def plot_detection_matrix(self, channel_index, run_indices = [], use_measurement_basis = False): 
        """
        This function plots the detection matrix between the beams at a particular channel index and either the originally 
        generated beams, or a seperately defined measurement basis. Note that the displayed detection matrix is normalized for 
        each row (i.e. the values are normalized so that the probabilities in each row add up to 1). 

        @type channel_index: Integer 
        @param channel_index: Index of channel for which the matrix is plotted
        @type run_indices: Array of integers
        @param run_indices: Indices of runs (beams) for which matrix will be plotted 
        @type use_measurement_basis: Boolean 
        @param use_measurement_basis: Indicator for using measurement basis to detect beams rather than the originally generated beams 
        """

        if run_indices == []: 
            run_indices = range(self.num_runs)

        # Get detection matrix 
        detection_matrix = self.compute_detection_matrix(channel_index, run_indices, use_measurement_basis = use_measurement_basis)

        # Define classes 
        classes = [self.beam_labels[idx] for idx in run_indices] 

        # Plot matrix
        figure, ax = plot_confusion_matrix(conf_mat = detection_matrix,
                                   class_names = classes,
                                   show_absolute = False,
                                   show_normed = True,
                                   colorbar = True)

        plt.show()

    def compute_inner_product(self, index1, index2, use_measurement_basis_for_1=False, use_measurement_basis_for_2 = False, channel_index_1 = -1, channel_index_2 = -1):
        """
        This function computes the normalized inner product between two beams specified from the possible beams stored in the simulation class, 
        including the beams generated, the beams transmitted through the different channels, and the measurement basis used for beam detection. 

        @type index1: Integer greater than or equal to 0
        @param index1: Index of first beam to be used for computation 
        @type index2: Integer greater than or equal to 0
        @param index2: Index of second beam to be used for computation 
        @type use_measurement_basis_for_1: Boolean
        @param use_measurement_basis_for_1: Indicator to use beam measurement basis for first index instead of generated or transmitted beams 
        @type use_measurement_basis_for_2: Boolean
        @param use_measurement_basis_for_2: Indicator to use beam measurement basis for second index instead of generated or transmitted beams 
        @type channel_index_1: Integer greater than or equal to 0
        @param channel_index_1: If measurement basis is not used for index1, this indicated the channel at which the first beam is taken 
        @type channel_index_2: Integer greater than or equal to 0
        @param channel_index_2: If measurement basis is not used for index2, this indicated the channel at which the second beam is taken 

        @rtype: Number between 0 and 1
        @return: Normalized inner product between the two specified beams 
        """

        if use_measurement_basis_for_1:
            beam1 = self.measurement_basis[index1] 
        else: 
            if channel_index_1 == -1:
                beam1 = self.beam_generators[index1].beam 
            else: 
                beam1 = self.runs[index1][channel_index_1]
        
        if use_measurement_basis_for_2:
            beam2 = self.measurement_basis[index2]
        else:
            if channel_index_2 == -1:
                beam2 = self.beam_generators[index2].beam
            else: 
                beam2 = self.runs[index2][channel_index_2]

        return Simulation.__normalized_inner_product(beam1, beam2)


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
    
    def __normalized_inner_product(beam1, beam2):
        """
        This function computes the normalized inner product between two beams that are given to it as an input. 
        
        @type beam1: Matrix of numbers
        @param beam1: Values of the first beam in the xy plane 
        @type beam2: Matrix of numbers
        @param beam2: Values of the second beam in the xy plane 

        @rtype normalized_inner_product: Number between 0 and 1 
        @return normalized_inner_product: Inner product between the two given beams 
        """

        # Get beam product 
        product = beam1 * np.conjugate(beam2)

        # Integrate product using the sum function 
        integral = np.sum(product) 

        # Take inner product to be the square of the integral 
        inner_product = np.abs(integral)**2 

        # Get square magnitude of both beams 
        product1 = np.abs(beam1)**2
        product2 = np.abs(beam2)**2

        # Integrate both products 
        integral1 = np.sum(product1)
        integral2 = np.sum(product2) 

        # Get normalized inner product 
        normalized_inner_product = inner_product / (integral1 * integral2) 

        # Return normalized inner prduct 
        return normalized_inner_product


# Use case example 
if __name__ == "__main__": 

    beamWaist = 2 # Define beam waist of 2 mm 

    sim = Simulation(L = 10 * beamWaist, N = 1000, wavelength = 810E-6, units = "mm")
    sim.add_beam_gen(ell = 0, p = 1, beam_waist = beamWaist)
    sim.add_beam_gen(ell = 0, p = 3, beam_waist = beamWaist)
    sim.add_beam_gen(ell = 0, p = 5, beam_waist = beamWaist)

    sim.add_measurement_basis(ell = 0, p = 1, beam_waist = 1.7*beamWaist)
    sim.add_measurement_basis(ell=0, p=3, beam_waist=1.7*beamWaist)
    sim.add_measurement_basis(ell=0, p=5, beam_waist=2*beamWaist)

    sim.add_channel(type=Channel.FREE_SPACE, dist=100E3)
    sim.add_channel(type = Channel.ABBARATION, n = [3, 1, 4], m = [1, 1, 2], stre = np.array([0.9, 0.9, 0.9]), app = 3*beamWaist)
    sim.add_channel(type = Channel.FREE_SPACE, dist = 10E3) 
    sim.add_channel(type = Channel.LENS, diam = 2 * beamWaist)
    sim.add_channel(type = Channel.FREE_SPACE, dist = 5)

    channel_idx = 1

    sim.run()

    #sim.plot_detection_matrix(channel_idx, use_measurement_basis = True)
    sim.plot_beams(run_index = 0, channel_indices=[0,1,3])
    sim.plot_beams(plot_measurement_basis = True)

    # Observe affect of changing waist param of measurement basis
    waist_factor = np.arange(1, 3, 0.01)

    inner_products = np.zeros(len(waist_factor))

    sim.delete_measurement_basis()

    for i in range(len(waist_factor)):
        sim.add_measurement_basis(ell=0, p=1, beam_waist=waist_factor[i]*beamWaist)
        inner_products[i] = sim.compute_inner_product(0, 0, channel_index_1=channel_idx, use_measurement_basis_for_2=True)
        sim.delete_measurement_basis()

    plt.plot(waist_factor, inner_products)
    plt.xlabel("Waist Factor")
    plt.ylabel("Inner Product")
    plt.show()

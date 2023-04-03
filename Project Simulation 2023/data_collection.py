"# Imports \n",
import numpy as np
import matplotlib.pyplot as plt
from Channel import Channel
from beam_gen import BeamGen
from simulation import Simulation 

# Constant Definitions
units = "mm"
beamWaist = 2  # Define beam waist of 2 mm
apperture = 20 # Define apperature of 2 cm
wavelength = 810E-6 # Define wavelenegth of 10 nm
observed_length = 20 # Define observed length for both the x and y directions of 2 cm
n = 1000 # Number of points used for computation in both the x and y directions
propagation_distance = 10E3 # Propagation distance of 10 m 

# Define simulation
sim = Simulation(L=observed_length, N=n, wavelength=wavelength, units=units)

# Define maximum used ell parameter for LG beams in OAM
max_ell = 5

# Define OAM Basis
ell_values = np.arange(-max_ell,max_ell+1)

for l in ell_values:
    sim.add_beam_gen(ell=l, p=0, beam_waist=beamWaist)

# Add initial apperture
sim.add_channel(type=Channel.LENS, diam=apperture)

# Define Zernike polynomial parameters
n_max = 4
aberation_strength = "strong"

if aberation_strength == "weak": 
    min_strength = 0.0
    max_strength = 0.2
elif aberation_strength == "medium":
    min_strength = 0.4
    max_strength = 0.6
elif aberation_strength == "strong":
    min_strength = 0.8
    max_strength = 1

# Define Zernike polynomial indices
n = []
for i in range(n_max):
    # Repeat each n index a number of times equal to its location in the pyramid
    for j in range(i + 1):
        n.append(i)

m = []
for n_idx in range(n_max):
    for m_val in range(-n_idx, n_idx + 1, 2):
        m.append(m_val)

# Generate random strengths of Zernike polynomials
np.random.seed(0)  # Define seed

strengths = []  # Initialized strength list
for i in range(len(n)): # Iterate over all Zerinke indices
    strength = np.random.rand() * (max_strength - min_strength) + min_strength
    strengths.append(strength)

sim.add_channel(type = Channel.ABBARATION, n = n, m = m, stre = np.array(strengths), app = apperture)
sim.add_channel(type = Channel.FREE_SPACE, dist = propagation_distance)
sim.add_channel(type=Channel.LENS, diam=apperture)

sim.run(use_mub=True)


#sim.plot_beams(channel_index=4)
sim.plot_detection_matrix(separate_mub=True, channel_index=4)

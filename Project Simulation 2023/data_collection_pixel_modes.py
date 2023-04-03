# Imports 
import numpy as np
import matplotlib.pyplot as plt
from Channel import Channel
from beam_gen import BeamGen
from simulation import Simulation 

wavelength = 810E-6
PI = np.pi
wavevector = (2.0*PI)/wavelength

graph_radius = 30
precision = 500

x = np.linspace(-graph_radius,graph_radius,precision+1); ## Grid points along x
y = np.linspace(-graph_radius,graph_radius,precision+1) ## Grid points along y
X,Y = np.meshgrid(x,y)
r = np.sqrt(X**2+Y**2)
phi = np.mod(np.arctan2(Y,X),2*PI)

beamWaist = 2  # Define beam waist of 2 mm
R = 20 #define radius of aperture to be 20 mm
pixel_spacing = 0.5

sim = Simulation(L=30*2, N=400, wavelength=810E-6, units="mm")
beamWaist_arr = np.arange(2, R/3, 0.2)
pixel_spacing_arr = np.arange(0, 3, 0.2)
qber_arr = np.zeros((np.size(pixel_spacing_arr), np.size(beamWaist_arr)))


for pixel_spacing in pixel_spacing_arr:
    for beamWaist in beamWaist_arr:
        dimension = BeamGen("pixel",0,0,beamWaist,r,phi,0.000001,wavevector, pixel_spacing=pixel_spacing, R=R).dimension
        for i in range (dimension):
            sim.add_beam_gen(mode="pixel", s=i, R=R, beam_waist=beamWaist, pixel_spacing=pixel_spacing)
        sim.add_channel(type=Channel.FREE_SPACE, dist=100E3)
        sim.add_channel(type = Channel.ABBARATION, n = [3, 1, 4], m = [1, 1, 2], stre = np.array([0.9, 0.9, 0.9]), app = 3*beamWaist)
        sim.run(use_mub=True)
        channel_idx = 1
        qber_arr[int(pixel_spacing*2)][int(beamWaist-2)] = sim.compute_qber(channel_index=channel_idx)
    with open('QBER_100m.csv','a') as csvfile:
        np.savetxt(csvfile, np.array([qber_arr[int(pixel_spacing*2)]]), delimiter=',', fmt='%s', comments='')

# for pixel_spacing in pixel_spacing_arr:
#     for beamWaist in beamWaist_arr:
#         dimension = BeamGen("pixel",0,0,beamWaist,r,phi,0.000001,wavevector, pixel_spacing=pixel_spacing, R=R).dimension
#         for i in range (dimension):
#             sim.add_beam_gen(mode="pixel", s=i, R=R, beam_waist=beamWaist, pixel_spacing=pixel_spacing)
#         sim.add_channel(type=Channel.FREE_SPACE, dist=100E3)
#         #sim.add_channel(type = Channel.ABBARATION, n = [3, 1, 4], m = [1, 1, 2], stre = np.array([0.9, 0.9, 0.9]), app = 3*beamWaist)
#         #sim.add_channel(type = Channel.FREE_SPACE, dist = 10E3) 
#         sim.run(use_mub=True)
#         channel_idx = 1
#         qber_arr[int(pixel_spacing*2)][int(beamWaist-2)] = sim.compute_qber(channel_index=channel_idx)
#     with open('QBER_1km.csv','a') as csvfile:
#         np.savetxt(csvfile, np.array([qber_arr[int(pixel_spacing*2)]]), delimiter=',', fmt='%s', comments='')

# for pixel_spacing in pixel_spacing_arr:
#     for beamWaist in beamWaist_arr:
#         dimension = BeamGen("pixel",0,0,beamWaist,r,phi,0.000001,wavevector, pixel_spacing=pixel_spacing, R=R).dimension
#         for i in range (dimension):
#             sim.add_beam_gen(mode="pixel", s=i, R=R, beam_waist=beamWaist, pixel_spacing=pixel_spacing)
#         sim.add_channel(type=Channel.FREE_SPACE, dist=100E3)
#         #sim.add_channel(type = Channel.ABBARATION, n = [3, 1, 4], m = [1, 1, 2], stre = np.array([0.9, 0.9, 0.9]), app = 3*beamWaist)
#         #sim.add_channel(type = Channel.FREE_SPACE, dist = 10E3) 
#         sim.run(use_mub=True)
#         channel_idx = 1
#         qber_arr[int(pixel_spacing*2)][int(beamWaist-2)] = sim.compute_qber(channel_index=channel_idx)
#     with open('QBER_dist_50km.csv','a') as csvfile:
#         np.savetxt(csvfile, np.array([qber_arr[int(pixel_spacing*2)]]), delimiter=',', fmt='%s', comments='')

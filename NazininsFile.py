import numpy as np
import matplotlib.pyplot as plt
import math


def propTF(u1,L,la,z):
    M,nn=u1.shape
    dx=L/M
    fx=np.arange(-1/(2*dx),1/(2*dx),1/L)
    Fx, Fy = np.meshgrid(fx, fx)

    H=np.exp(-1j*np.pi*0.25*la*z*(Fx**2+Fy**2))
    U2=H*np.fft.fftshift(np.fft.fft2(u1))
    u2=np.fft.ifft2(np.fft.ifftshift(U2))
    return u2 

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def RR(m,n):
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

def Zernike(RHO,PHI,m,n):
    ZR=np.zeros(RHO.shape); 
    rn=RR(np.abs(m),n)
    for ii in range(len(rn[0])):
        ZR=ZR+rn[0][ii]*RHO**rn[1][ii]
    if m>=0: 
        Z=ZR*np.cos(np.abs(m)*PHI)
    else:
        Z=ZR*np.sin(np.abs(m)*PHI)
    M=(RHO<=1)
    P=Z*M
    return(P)


################Section 2######################
#Coordinate systme
k=2*np.pi  # [m^-1] wavenumber         # [mm] beam waist


maxx = 24
# [units of w0] Half length of the numerical window If your propagation is noisy consider choosing a bigger window 
# But remember by changing the window size you will change the size of the beam.


N=2048    
# [Number of points per dimension] This number determines the resolution of your beam and propagator
# If you have a messy noisy propagation consider increasing the resolution as an option of improving the situation!


# Space definition 
X=np.linspace(-maxx,maxx,N);
Y=np.linspace(-maxx,maxx,N);


h=np.abs(X[1]-X[2]);
xx,yy=np.meshgrid(X,Y);
r, phi= cart2pol(xx,yy)

plt.figure(figsize=(5,5),dpi=250)
plt.subplot(121)
plt.imshow(r)
plt.title('r')
plt.axis('off')
plt.subplot(122)
plt.imshow(phi, cmap='hsv', interpolation='nearest')
plt.title('phi')
plt.axis('off')
plt.show()

#----------------------Section 3 ---------------------
# Beam Waist
w = 1

# Indices of the Zernike Polynomial (The Aberration)
m = 0
n = 2

# Strength of the Aberration (The coefficient of the beam)
# When the factor in the exponential is pi, the value of strength goes from 0 to 1 because Zernike Polynomials are bounded between
# -1 and 1 therefore, the phase between -pi and pi
stre = 1

#Aperture of the Aberration
A = w * 10

# Zernike Function
P=Zernike(r/A,np.transpose(phi),m,n)


rs = r/w

#HyGG Modes - DEMO FOR QKD BANDWIDTH CAPACTIY PROJECT
l = 3
AK = np.exp(1j*l*phi) * np.exp(-1*rs**2) * (1-np.heaviside(r/A-1, 0.05))

# Normalization
AK = AK/np.sqrt(np.sum(np.conj(AK)*AK * h**2))

# Applying the Aberration
AK1 = np.exp(1j*np.pi*stre*P) * AK

#Printing the New Beam
plt.figure(figsize=(20,20),dpi=250)
plt.subplot(121)
plt.imshow((np.abs(AK1)**2 * (1-np.heaviside(r/A-1, 0.05)))+np.max((np.abs(AK1)**2))*(np.heaviside(r/A-1, 0.05)))
plt.title('Amplitude')
plt.axis('off')
plt.subplot(122)
plt.imshow(np.angle(AK1),cmap="hsv")
plt.title('Phase')
plt.axis('off')
plt.show()

#---------------Section 4-----------
#DEMO FOR QKD BANDWIDTH CAPACTIY PROJECT
Z0 = 10
NZ = 20

# Propagation space
Z=np.linspace(0,Z0,NZ)

# Propagation steps
dz=np.abs(Z[0]-Z[1])

# Saving the field at every plane
F=[AK1] 

# Initialization the propagation 
u=AK1

# Forward Propagation 
for ii in range(0,NZ):
        U=propTF(u,2*maxx,1,Z[ii]) # Field at plane z->z_0
        F.append(U)

#---------------Section 5---------------

import ipywidgets as widgets
import threading

# View the intensity and phase plots
def f(value):

  plt.figure(figsize=(15,20))

  plt.subplot(1,2,1),plt.title('Phase')
  plt.imshow(np.angle(F[value][750:-750,750:-750]), cmap = "hsv", interpolation='nearest')

  plt.subplot(1,2,2),plt.title('Amplitude')
  plt.imshow(np.abs(F[value][750:-750,750:-750]),  interpolation='nearest')

w = widgets.interactive(f, value=widgets.IntSlider(min=0, max=len(F)-1, step=1))

def work(w):
    for ii in range(10):
        time.sleep(0.5)
        print(w.kwargs, w.result)

thread = threading.Thread(target=work, args=(w,))
display(w)
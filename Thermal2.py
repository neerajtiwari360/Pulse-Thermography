"""
Created on Sun Dec 24 16:30:55 2017

@author: Neeraj Tiwari(SC16M031)

Code made for pulse thermography

"""

"""
Here the code is devided in some parts like:
    1- geting the themal data for pulse thermography
    2- select each pixel of frame and apply Savitzky filter
    3- Apply FFt on each pixel and get the Mag and Phase response(Dominant)
    4- 3D - Plotting of Mag and Phase and DeterMine Defect Points
    5- Take that point and plot the Temperature vs Time curve to show defect
    
"""


from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter as GF
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
#from skimage import io 
from scipy.fftpack import fft
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D

def Demo():
    print('its main')
    

#______________________________________________________________________________
'DATA EXTRACT'
Mat_Data = spio.loadmat('data_22_12_1.mat')
Mat_Time = spio.loadmat('time_22_12_1.mat')

Frame_Data = Mat_Data['data']
Frame_Time = Mat_Time['time']


#______________________________________________________________________________
'GET DATA FROM EVERY PIXEL'
x1 = [0 for x in range(600)]
x2 = [0 for x in range(600)]
x3 = [0 for x in range(600)]
x4 = [0 for x in range(600)]
x5 = [0 for x in range(600)]
Time = [0 for x in range(600)]

Y = [[[0 for d in range(600)]for c in range(320)] for r in range(256)]
for d in range(0,600):
   Frame = Frame_Data[d,0]
   Time[d] = Frame_Time[d,0]
   x1[d] = Frame[200,53]
   x2[d] = Frame[200,93]
   x3[d] = Frame[200,120]
   x4[d] = Frame[200,150]
   x5[d] = Frame[200,180]
   for r in range(0,256):
       for c in range(0,320):
           Y[r][c][d]= Frame[r][c]
           
#______________________________________________________________________________
'APPLYING SAVITZKY FILTER'
Y_SG_Filter = savgol_filter(Y, 21,18)  
#mag = SG(FFTM,9,4) 

  
#______________________________________________________________________________
'GET FFT FOR EACH DATA'
fft_Phase_Sum = [[0 for c in range(320)]for r in range(256)] 
fft_Mag_Sum = [[0 for c in range(320)]for r in range(256)]        
for r in range(0,256):
      for c in range(0,320):
           fft_Phase_Sum[r][c] = np.sum(np.angle(fft(Y_SG_Filter[r][c][:]))) 
           fft_Mag_Sum[r][c] = np.sum(np.abs(fft(Y_SG_Filter[r][c][:]))) 
        
#______________________________________________________________________________
'GAUSSIAN FILTER'
Gauss_Filter_Phase = GF(fft_Phase_Sum, sigma=5)
Gauss_Filter_Mag = GF(fft_Mag_Sum, sigma=5)

#______________________________________________________________________________
'FLOAT TO 8BIT INT CONVERSTION'
Phase_array = np.asarray(fft_Phase_Sum)
Mag_array = np.asarray(fft_Mag_Sum)

maxP = Phase_array.max()
minP = Phase_array.min()

maxM = Mag_array.max()
minM = Mag_array.min()

Pha_8bit = (Phase_array - minP)*(255/(maxP - minP))
Mag_8bit = (Mag_array - minM)*(255/(maxM - minM))

#Mag_8bit1 = Mag_8bit.astype(np.uint8) 'flot64 to uint8 bit conversion'

#Mag_8bit = Mag_8bit*(Mag_8bit > 150) 'Applying some thresholding value'

#______________________________________________________________________________
'3D PLOTING'
def Plot_3D():
    xx, yy = np.mgrid[0:Pha_8bit.shape[0], 0:Pha_8bit.shape[1]]
    
    '3D plot for PHASE RESPONSE'
    fig0 = plt.figure(0)
    ax0 = fig0.gca(projection='3d')
    surf0 = ax0.plot_surface(xx, yy, Pha_8bit ,cmap=cm.coolwarm ,linewidth=0,
                             antialiased=False)
    fig0.colorbar(surf0, shrink=0.5, aspect=5)
    
  
    '3D plot for MAGNITUDE RESPONSE'
    fig1 = plt.figure(1)
    ax1 = fig1.gca(projection='3d')
    surf1 =ax1.plot_surface(xx,yy,Mag_8bit,cmap=cm.coolwarm ,linewidth=0,
                            antialiased=False)
    fig1.colorbar(surf1, shrink=0.5, aspect=5)
   
         
#______________________________________________________________________________
'SHOW IN IMAGE FORM'
def plot_2D():    
    plt.figure(2)
    plt.imshow( Pha_8bit)        
    plt.figure(3)
    plt.imshow(Mag_8bit)   
           
#______________________________________________________________________________
'POINT CURVE'
def PointPlot():    
    plt.figure(4)
    plt.plot(Time,x1,'-',Time,x2,'-',Time,x3,'-',Time,x4,'-',Time,x5,'-')
    plt.legend(['x1','x2','x3','x4','x5'])
    plt.axis([5,20,25,45])
    plt.xlabel('Time')
    plt.ylabel('Temperature')
   #plt.grid(True)
    plt.title('Temperature V/s Time Graph')

    
#______________________________________________________________________________    
'MAIN START FROM HERE'
    
if __name__ == "__main__":
    Demo()
    
    Plot_3D()
    
    plot_2D()
    
    PointPlot()
  

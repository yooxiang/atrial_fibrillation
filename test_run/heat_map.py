import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import timeit
import copy
from scipy.optimize import curve_fit
import pandas as pd
import sys
from scipy.ndimage.filters import maximum_filter
plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
sys.path.append('/Users/yuxiang/Documents/master/python_code_git/atrial_fibrillation')
from functions_list import RunState, Resultant_Vectors, PixelatedVectors, MovieNodes, TimeLoc,focal_quality_indicator
#%%
smltn0302=RunState(200,10000,120,120,4,1,0.15,3,20)
MovieNodes(smltn0302,None).save('/Users/yuxiang/Documents/master/python_code_git/atrial_fibrillation/test_case_10_10.mp4', writer=writer,dpi = 300)
#%%
vectors0302=Resultant_Vectors(smltn0302,outv=True)
pxv0302=PixelatedVectors(smltn0302,vectors0302,15,15)


Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='James'), bitrate=1800)
#MovieNodes(smltn0302,None).save('/Users/yuxiang/Documents/master/python_code_git/atrial_fibrillation/test_case_10_10.mp4', writer=writer,dpi = 300)
#%%
xdim=15
tM=TimeLoc(pxv0302[0],pxv0302[1],smltn0302)
print(tM)

locall=[0]*xdim**2
if tM==0:
    locM=[0,0]
else:
    for t in range(tM,len(smltn0302[0])-5):
        print(t)
        for i in range(15**2):
            q=focal_quality_indicator(pxv0302[0],pxv0302[1],pxv0302[1][i],t,2,4)


            locall[i]+=q
#%%visualising total dot product in tissue
matrix=np.zeros([15,15])
for i in range(len(locall)):
    y=i//xdim
    x=i%xdim
    matrix[y][x]=locall[i]
fig=plt.figure()
s=plt.imshow(matrix,interpolation='none',cmap='jet',animated=True)
plt.gca().invert_yaxis()
fig.colorbar(s)
plt.savefig('/Users/yuxiang/Documents/master/python_code_git/atrial_fibrillation/1102alldot',dpi=1000)
#%% try remove the low values




#%% try another way
from skimage.feature import peak_local_max

coordinates = peak_local_max(matrix, min_distance = 1, exclude_border = False)

print(coordinates)
x = []
y = []
for i in range(len(coordinates)):
    x.append(coordinates[i][1])
    y.append(coordinates[i][0])
    
plt.imshow(matrix,interpolation='none',cmap='jet',animated=True)
plt.scatter(x,y, marker = 'x')
plt.gca().invert_yaxis()
plt.savefig('/Users/yuxiang/Documents/master/python_code_git/atrial_fibrillation/cross',dpi=1000)
plt.show()
    
#%%
im = np.array([[0,1,2,3,4],[0,1,1,1,0,],[0,1,2,1,0,],[0,1,1,1,0,],[0,0,0,0,0]])
#%%
plt.imshow(matrix,interpolation='none',cmap='jet',animated=True)
plt.gca().invert_yaxis()
plt.scatter(x,y)
plt.show()
#%%visualing top dot product values
locall=[]
if tM==0:
    locM=[0,0]
else:
    for t in range(tM,len(smltn0302[0])-5):
        print(t)
        q_all=[]
        for i in range(32**2):
            q=focal_quality_indicator(pxv0302[0],pxv0302[1],pxv0302[1][i],t,2,5)
            q_all.append(q)
        loci=q_all.index(max(q_all))
        locM=[loci%xdim,loci//xdim]
        locall.append(locM)


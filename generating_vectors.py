#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:25:45 2021

@author: yuxiang
"""

sys.path.append('/Users/yuxiang/Documents/master/python_code_git/atrial_fibrillation')
from functions_list import Pixelation,RunState, Resultant_Vectors, PixelatedVectors, MovieNodes, TimeLoc,focal_quality_indicator,VectorMovie
import numpy as np
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='James'), bitrate=1800)

x=RunState(100,500,5,5,3,1,0.1,0.5,2)
vectors0302=Resultant_Vectors(x,outv=True)
pvectors =PixelatedVectors(x,vectors0302,10,10)
pixels = Pixelation(x,10,10)
macro_vectors = generate_macro_vectors(pixels,20)

print(pixels[0])

MovieNodes(x,None).save('/Users/yuxiang/Documents/master/python_code_git/atrial_fibrillation/generating_vectors.mp4', writer=writer,dpi = 300)
VectorMovie(pvectors[0],pvectors[1],None).save('/Users/yuxiang/Documents/master/python_code_git/atrial_fibrillation/generating_vectors_micro.mp4', writer=writer,dpi = 300)
VectorMovie(macro_vectors,pixels[4],None).save('/Users/yuxiang/Documents/master/python_code_git/atrial_fibrillation/generating_vectors_macro.mp4', writer=writer,dpi = 300)
#%%
def generate_macro_vectors(data,threshold):
    #data output of pixelation
    charge_time = data[0]
    time_length = len(charge_time)
    x_dim = len(charge_time[0])
    y_dim = len(charge_time[0][0])
    print(time_length,x_dim,y_dim)
    
    def generate_vector(charge_time,time,threshold,i,j):
        neighbours = []
        
        if i == x_dim -1:
            neighbours.append(0)
        else:
            neighbours.append(charge_time[time][i+1][j])
        
        if i==0:
            neighbours.append(0)
        else:
            neighbours.append(charge_time[time][i-1][j])
        
        if j == y_dim-1:
            neighbours.append(charge_time[time][i][0])
        else:
            neighbours.append(charge_time[time][i][j+1])
        
        if j == 0:
            neighbours.append(charge_time[time][i][y_dim-1])
        else:
            neighbours.append(charge_time[time][i][j-1])
        
        
        index = neighbours.index(max(neighbours))
        if neighbours[index] != 0 and neighbours[index] > threshold:
            
            if index == 0:
                vector = [1,0]
            if index == 1:
                vector = [-1,0]
            if index == 2:
                vector = [0,1]
            if index == 3:
                vector = [0,-1]
        else:
            vector = [0,0]
        
        return vector
            
    vector_store = []
    for time in range(time_length):
        store_single = []
        for i in range(x_dim):
            for j in range(y_dim):
                store_single.append(generate_vector(charge_time,time,threshold,i,j))
        vector_store.append(store_single)
    
    return vector_store

#%%
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
def GenerateCoorArray(N,x_size,y_size):#prepares a list with the coordinates of all nodes
    pmcell=int(N/10)#number of pacemaker cells
    CoorStore=[[0,random.random()*y_size] if i<pmcell else 
    [random.random()*x_size,random.random()*x_size] for i in range(N)]
    return sorted(CoorStore , key=lambda k: [k[0], k[1]])

def distance_checker(R,i,j,CoorStore,y_size):#checks the distance between 2 nodes 
    #in a way to allow periodic BCs in y-direction and open in x-direction
    diff=CoorStore[i][1]-CoorStore[j][1]
    if abs(diff)<=R:# for rest cases
        d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1] - CoorStore[j][1]) **2 )**0.5
    elif abs(diff)>R:    
        if CoorStore[i][1]+R>y_size:#sets upper BC
            d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1]-y_size - CoorStore[j][1]) **2 )**0.5
        elif CoorStore[i][1]-R<0:#sets lower BC
            d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1]+y_size - CoorStore[j][1]) **2 )**0.5
        else:
            d = ((CoorStore[i][0]- CoorStore[j][0])**2 + (CoorStore[i][1] - CoorStore[j][1]) **2 )**0.5
    if d <= R:
        return True
    else:
        return False       

def ConnectionArray(N,x_size,y_size,CoorStore,R):
    #a list containing all the connections of the ith node in list
    ConnectionsStore = []
    
    for i in range(N):
        connections = []
        nodecoor=CoorStore[i]
        ll=nodecoor[0]-R
        rl=nodecoor[0]+R
        for j in range(N):
            if i !=j and ll<=CoorStore[j][0]<=rl:    
                if distance_checker(R,i,j,CoorStore,y_size) == True:
                    connections.append(j)
        ConnectionsStore.append(connections)
        
    return ConnectionsStore

def FunctionalArray(N,x_size,y_size,tau,delta,epsilon):
    #prepares two lists: one with refractory period and one with excitation
    #probability of each node
    taustore=[tau for i in range(N)]
    epsilonstore= [epsilon if random.random()<delta else 0 for i in range(N)]
    functionalstore=[taustore,epsilonstore]
    return functionalstore
 
def GenerateState(N):#prepares list with state of each node
    return [0]*N
#%%      
def PaceMaker(StateListP,FunctionListP,N):#excites pacemaker cells every T time steps
    pmcell=int(N/10)
    for i in range(pmcell):
        rand=random.random()
        if rand>FunctionListP[1][i]:
            StateListP[i]= FunctionListP[0][i] + 1                
    return StateListP

def UpdateStateLattice(N,StateListU,FunctionListU, Connection):
#updates the state of each node
  
    ChargeFlowIN=[]
    ChargeFlowOUT=[]
    
    Updatedlist = [0]* N
    for i in range(N):
        if StateListU[i] > 0: #reduce refractory period countdown by one
            Updatedlist[i] = StateListU[i]- 1                           
            #continue
        if StateListU[i]== 0: #here check if any neighbours are in firing state
            inv=[]#inputs starting location of incoming vectors
            c=0
            rand=random.random()
            for j in range(len(Connection[i])):
                NeighNode=Connection[i][j]                
                if StateListU[NeighNode] == FunctionListU[0][NeighNode]+1:
                    if rand>FunctionListU[1][NeighNode]:
                        c+=1#sets condition to store incoming vector
                        Updatedlist[i]=FunctionListU[0][i]+1                        
                        inv.append(NeighNode)
                        if NeighNode in ChargeFlowOUT:
                            pos=ChargeFlowOUT.index(NeighNode)
                            ChargeFlowOUT[pos+1].append(i)
                        else:
                            ChargeFlowOUT.append(NeighNode)
                            ChargeFlowOUT.append([i])
            if c>0:
                ChargeFlowIN.append(i)
                ChargeFlowIN.append(inv)
                    
    return Updatedlist,ChargeFlowIN, ChargeFlowOUT
#%%
def RunState(TimeSteps,N,x_size,y_size,tau,delta,epsilon,r,T):
    #runs the model for a specified number of timesteps
    #and returns a list with all timesteps
    SetupS=timeit.default_timer()
    Coordinates = GenerateCoorArray(N,x_size,y_size)
    Connections= ConnectionArray(N,x_size,y_size,Coordinates,r)
    FunctionListR = FunctionalArray(N,x_size,y_size,tau,delta,epsilon)
    StateListR = GenerateState(N)
    SetupE=timeit.default_timer()
    
    StateStore = []    
    ChargeFlowINAll=[]
    ChargeFlowOUTALL=[]
    
    RunS=timeit.default_timer()
    for i in range(TimeSteps):
        if i == 0 or i % T == 0:          
            StateListR = PaceMaker(StateListR,FunctionListR,N)
            StateListR, ChargeFlowIn, ChargeFlowOut = UpdateStateLattice(N,StateListR,FunctionListR,Connections)
            StateStore.append(StateListR)
            ChargeFlowINAll.append(ChargeFlowIn)
            ChargeFlowOUTALL.append(ChargeFlowOut)
 
        else:
            StateListR, ChargeFlowIn, ChargeFlowOut = UpdateStateLattice(N,StateListR,FunctionListR,Connections)
            StateStore.append(StateListR)
            ChargeFlowINAll.append(ChargeFlowIn)
            ChargeFlowOUTALL.append(ChargeFlowOut)
             
    RunE=timeit.default_timer()    
    TimeS=SetupE-SetupS
    TimeR=RunE-RunS
    data=[TimeSteps,N,x_size,y_size,tau,delta,epsilon,r,T]
    return StateStore, Coordinates, Connections,TimeS,TimeR, ChargeFlowINAll, ChargeFlowOUTALL, TimeSteps,data
 
def Resultant_Vectors(cc,outv=False,inv=False):
    #cc is output of RunState
    state,Coordinates,Connections,t1,t2,ChargeFlowINALL,ChargeFlowOUTALL,TimeSteps,param=cc
    R=param[7]
    y_size=param[3]
    ChargeFlowINsingle = copy.deepcopy(ChargeFlowINALL)
    ChargeFlowOUTsingle = copy.deepcopy(ChargeFlowOUTALL)    
    ChargeFlowBoth=[[] for i in range(TimeSteps)]
    
    for t in range(TimeSteps):
        if inv==True:
            InNode=ChargeFlowINALL[t][::2]#nodes to which excitation arrives
            InVNodes=ChargeFlowINALL[t][1::2]#nodes from which excitation arrives       
            for i in range(len(InNode)):
                node=Coordinates[InNode[i]]
                no_IN=len(InVNodes[i])#number of tails
                INvector=[0,0]
                for j in range(len(InVNodes[i])):
                    nodeTail=Coordinates[InVNodes[i][j]]
                    ucomIn=node[0]-nodeTail[0]
                    INvector[0]+=ucomIn/no_IN
                    
                    vcomIn=node[1]-nodeTail[1]
                    if abs(vcomIn)<=R:
                        INvector[1] +=vcomIn/no_IN
                    elif abs(vcomIn)>R:
                        if vcomIn+R>y_size:                            
                            INvector[1] += ((vcomIn-y_size)/no_IN)
                        elif vcomIn<0:                            
                            INvector[1] += ((vcomIn+y_size)/no_IN)  
                mag=np.sqrt((INvector[0])**2+(INvector[1])**2)
                INvector=[INvector[0]/mag,INvector[1]/mag]            
                ChargeFlowINsingle[t][2*i+1]=INvector
        if outv==True:        
            OutNode=ChargeFlowOUTALL[t][::2]
            OutVNodes=ChargeFlowOUTALL[t][1::2]#nodes form which excitation arrives       
            for i in range(len(OutNode)):
                node=Coordinates[OutNode[i]]
                no_IN=len(OutVNodes[i])#number of tails
                OUTvector=[0,0]
                for j in range(len(OutVNodes[i])):
                    nodeHead=Coordinates[OutVNodes[i][j]]
                    ucomOut=nodeHead[0]-node[0]
                    OUTvector[0]+=ucomOut/no_IN
                    
                    vcomOut=nodeHead[1]-node[1]
                    if abs(vcomOut)<=R:
                        OUTvector[1] +=vcomOut/no_IN
                    elif abs(vcomOut)>R:
                        if vcomOut+R>y_size:                            
                            OUTvector[1] += ((vcomOut-y_size)/no_IN)
                        elif vcomOut<0:                            
                            OUTvector[1] += ((vcomOut+y_size)/no_IN)
                mag=np.sqrt((OUTvector[0])**2+(OUTvector[1])**2)
                OUTvector=[OUTvector[0]/mag,OUTvector[1]/mag]
                ChargeFlowOUTsingle[t][2*i+1]=OUTvector
            
        if outv==True and inv==True:
            InNode=ChargeFlowINsingle[t-1][::2]
            InVectors=ChargeFlowINsingle[t-1][1::2]
            OutNode=ChargeFlowOUTsingle[t][::2]
            OutVectors=ChargeFlowOUTsingle[t][1::2]
        
            InNodeSet=set(InNode)
            OutNodeSet=set(OutNode)
            el=list(InNodeSet.intersection(OutNodeSet))
            
            for k in range(len(el)):
                if t!=0:
                    inpos=InNode.index(el[k])
                    INvector=InVectors[inpos]
                else:
                    INvector=[0,0]
                outpos=OutNode.index(el[k])
                OUTvector=OutVectors[outpos]
                
                FinalVector_i =[(INvector[0]+OUTvector[0])/2,(INvector[1]+OUTvector[1])/2]      
                mag= np.sqrt((FinalVector_i[0])**2+(FinalVector_i[1])**2)
                if mag!=0:
                    FinalVector=[FinalVector_i[0]/mag,FinalVector_i[1]/mag]
                else:
                    FinalVector=FinalVector_i
                ChargeFlowBoth[t].append(el[k])
                ChargeFlowBoth[t].append(FinalVector)
            
            inremain=list(InNodeSet-set(el))
            for k in range(len(inremain)):
                if t!=0:
                    inpos=InNode.index(inremain[k])
                    INvector=InVectors[inpos]
                else:
                    INvector=[0,0]
                OUTvector=[0,0]
                
                FinalVector_i =[(INvector[0]+OUTvector[0])/2,(INvector[1]+OUTvector[1])/2]      
                mag= np.sqrt((FinalVector_i[0])**2+(FinalVector_i[1])**2)
                if mag!=0:
                    FinalVector=[FinalVector_i[0]/mag,FinalVector_i[1]/mag]
                else:
                    FinalVector=FinalVector_i
                ChargeFlowBoth[t].append(inremain[k])
                ChargeFlowBoth[t].append(FinalVector)
    
            outremain=list(OutNodeSet-set(el))
            for k in range(len(outremain)):
                INvector=[0,0]
                outpos=OutNode.index(outremain[k])
                OUTvector=OutVectors[outpos]
                
                FinalVector_i =[(INvector[0]+OUTvector[0])/2,(INvector[1]+OUTvector[1])/2]      
                mag= np.sqrt((FinalVector_i[0])**2+(FinalVector_i[1])**2)
                if mag!=0:
                    FinalVector=[FinalVector_i[0]/mag,FinalVector_i[1]/mag]
                else:
                    FinalVector=FinalVector_i
                ChargeFlowBoth[t].append(outremain[k])
                ChargeFlowBoth[t].append(FinalVector)
            
    if outv==True and inv==True:          
        return ChargeFlowBoth #this is hopefully a list arranged first by time, and then each ndoe given a vector in the form of [x,y]       
    elif inv==True and outv==False:
        return ChargeFlowINsingle
    elif inv==False and outv==True:
        return ChargeFlowOUTsingle
        
def MovieNodes(a,frame):#input is list with all the lists describing the state of nodes(StateStore)
    #prepares movie for evolution of state of nodes
    x=[]
    y=[]
    for i in range(len(a[1])):
        x.append(a[1][i][0])
        y.append(a[1][i][1])
    fig=plt.figure()
    if frame==None:
        ims=[]
        for i in range(len(a[0])):
            im=plt.scatter(x,y,c=a[0][i],edgecolors='r',cmap=plt.cm.binary)
            ims.append([im])
        plt.colorbar()
        ani = animation.ArtistAnimation(fig, ims, interval=500, 
                                         repeat_delay=1000)
    else:
        ani=plt.scatter(x,y,c=a[0][frame],edgecolors='r',cmap=plt.cm.binary)
        plt.colorbar()
    return ani

def Pixelation(cc,x_grid_size,y_grid_size):
    #prepares pixelation of nodes based on resolution requested
    #cc is output of RunState
    x_size=cc[8][2]
    y_size=cc[8][3]
    tau=cc[8][4]
    grid_coor = []
    for j in range(int(y_grid_size)):
        for i in range(int(x_grid_size)):
            grid_coor.append([i,j])
    grid_container = []
    timeseries=[]#contains time-series for each cell
    for i in range(len(grid_coor)):
        grid_container.append([])
        timeseries.append([])
    for i in range(len(cc[1])):
        grid_coor_state = cc[1][i][0]//(x_size/x_grid_size),cc[1][i][1]//(y_size/y_grid_size) 
        grid_container[int(grid_coor_state[1]*(x_grid_size) + grid_coor_state[0] )].append(i)
    allgridvalues=[]
    for i in range(len(cc[0])):
        grid_sum = np.zeros([int(y_grid_size),int(x_grid_size)])
        for cell in range(len(grid_container)):
            sum_c = 0
            for node in range(len(grid_container[cell])):
                sum_c += cc[0][i][grid_container[cell][node]]
            grid_sum[y_grid_size-1-grid_coor[cell][1]][grid_coor[cell][0]]=sum_c
            timeseries[cell].append(sum_c)
        allgridvalues.append(grid_sum)                
    nodespc=[]#nodespercell(determining cell with max number of nodes)
    for i in range(len(grid_container)):
        nodespc.append(len(grid_container[i]))
    maxcellcolor=np.mean(nodespc)*(tau+1)#determining max value possible 
    #in grid_sum,required to set the color scale
    return allgridvalues,int(maxcellcolor) ,timeseries,grid_container,grid_coor

def MoviePixels(pixeldata):
    #input is output of Pixelation
    Allgridvalues=pixeldata[0]
    fig = plt.figure()
    ims=[]    
    for i in range(len(Allgridvalues)):
        im=plt.imshow(Allgridvalues[i],interpolation='none',cmap=plt.cm.binary,vmin=0,vmax=pixeldata[1],animated=True)
        if i==0:
            fig.colorbar(im)
        ims.append([im])
    plt.title('Pixelated Grid')
    ani = animation.ArtistAnimation(fig, ims, interval=500, 
                                    repeat_delay=1000)
    return ani

def PixelatedVectors(cc,cv,x_grid_size,y_grid_size):
    #prepares exact coarse graining
    #cc is output of Runstate
    #cv is output of resultant vector
    x_size=cc[8][2]
    y_size=cc[8][3]
    coordinates=cc[1]
    grid_coor = []#cotains coordinates for each pixel
    for j in range(int(y_grid_size)):
        for i in range(int(x_grid_size)):
            grid_coor.append([i,j])
    
    gridvectorTot=[]
    for t in range(len(cv)):
        nodes=cv[t][::2]
        vectors=cv[t][1::2]
        
        grid_container=[[] for i in range(len(grid_coor))]
        for i in range(len(nodes)):
            grid_coor_state = coordinates[nodes[i]][0]//(x_size/x_grid_size),coordinates[nodes[i]][1]//(y_size/y_grid_size) 
            grid_container[int(grid_coor_state[1]*(x_grid_size) + grid_coor_state[0] )].append(i)        
        
        gridvector_t=[]
        for cell in range(len(grid_container)):
            cellvector=[0,0]
            no_nodes=len(grid_container[cell])
            for i in range(no_nodes):
                cellvector[0]+=vectors[grid_container[cell][i]][0]/no_nodes
                cellvector[1]+=vectors[grid_container[cell][i]][1]/no_nodes
            cellvector_mag=np.sqrt((cellvector[0])**2+(cellvector[1])**2)
            if cellvector_mag!=0:
                cellvector_norm=[cellvector[0]/cellvector_mag,cellvector[1]/cellvector_mag]
            else:
                cellvector_norm=cellvector
            gridvector_t.append(cellvector_norm)
        gridvectorTot.append(gridvector_t)                       
    return gridvectorTot,grid_coor
#%%
def divergence(vector,Coordiantes,x_dim,y_dim): #x_dim and y_dim are nunber of cells, not the true length which is captured in the coordinates
    
    def internal_point(index): # this is for all the points apart from the edges on the left and right
        #vector = vector_field[index]
        y_coor =(index//x_dim)        
        dFx = (vector[index+1][0] - vector[index-1][0])/(Coordiantes[index+1][0] - Coordiantes[index-1][0])
                
        if y_coor == 0:
            dFy = (vector[index+x_dim][1] - vector[index+(x_dim*(y_dim -1))][1])/(2*(Coordiantes[index+x_dim][1] - Coordiantes[index][1]))
           
        if y_coor == y_dim - 1:   
            dFy = (vector[index-(x_dim*(y_dim -1))][1] - vector[index - x_dim][1])/(2*(Coordiantes[index][1] - Coordiantes[index-x_dim][1]))
        if y_coor > 0 and y_coor < y_dim -1:
            dFy = (vector[index+x_dim][1] - vector[index - x_dim][1])/(Coordiantes[index+x_dim][1] - Coordiantes[index-x_dim][1])         
     
        return dFx + dFy   
    
    def edge_point(index): #this is for all the points on the right and left ie x = 0 or x= y_di
        #here a central difference is not used
        #vector = vector_field[index]
        x_coor = index % x_dim 
        y_coor =index//x_dim
        
        if x_coor == 0:          
            dFx = (vector[index+1][0] - vector[index][0])/(Coordiantes[index+1][0] - Coordiantes[index][0])            
        if x_coor == y_dim - 1:
            dFx = (vector[index][0] - vector[index-1][0])/(Coordiantes[index][0] - Coordiantes[index-1][0])            
        if y_coor == 0:        
            dFy = (vector[index+x_dim][1] - vector[index+(x_dim*(y_dim -1))][1])/(2*(Coordiantes[index+x_dim][1] - Coordiantes[index][1]))
        if y_coor == y_dim - 1:
            dFy = (vector[index-(x_dim*(y_dim -1))][1] - vector[index - x_dim][1])/(2*(Coordiantes[index][1] - Coordiantes[index-x_dim][1]))            
        if y_coor > 0 and y_coor < y_dim - 1:
            dFy = (vector[index+x_dim][1] - vector[index - x_dim][1])/(Coordiantes[index+x_dim][1] - Coordiantes[index-x_dim][1])
        
        return dFx + dFy

    div_store = []
    for index in range(x_dim*y_dim):
        x_coor = index % y_dim 

        if x_coor > 0 and x_coor < y_dim -1:
            div = internal_point(index)
        else:
            div = edge_point(index)
        div_store.append(div)
    
    return div_store,Coordiantes

def VectorMovieNodes(vectordata,coordinates,frame):
    X=[i[0] for i in coordinates]
    Y=[i[1] for i in coordinates]
    U=[0]*len(coordinates)
    V=[0]*len(coordinates)
    
    fig, ax = plt.subplots(1,1)
    Q = ax.quiver(X,Y,U, V, pivot='tail', angles='xy', scale_units='xy',scale=1)
    
    def update_quiver(num,Q):            
        U=[0]*len(coordinates)
        V=[0]*len(coordinates)
        vectors=vectordata[num][1::2]
        nodes=vectordata[num][::2]
        for i in range(len(nodes)):
            U[nodes[i]]=vectors[i][0]
            V[nodes[i]]=vectors[i][1]
        Q.set_UVC(U,V)
        return Q,
    
    if frame==None:
        anim1 = animation.FuncAnimation(fig, update_quiver,frames=len(vectordata),fargs=(Q,), interval=500, blit=True)
    else:
        anim1=update_quiver(frame,Q)
    fig.tight_layout()
    
    return anim1

#following function creates vector movies
def VectorMovie(vectordata,points,frame):
    #vectordata must be the vector of either nodes or pixels
    #with their respective points
    #if frame==None then it returns the whole movie otherwise specify
    #the frame you need to visualise
    X=[i[0] for i in points]
    Y=[i[1] for i in points]
    U=[0]*len(points)
    V=[0]*len(points)
    
    fig, ax = plt.subplots(1,1)
    Q = ax.quiver(X,Y,U, V, pivot='tail', angles='xy', scale_units='xy',scale=1)
    
    def update_quiver(num,Q):
        U=[0]*len(points)
        V=[0]*len(points)
        Q.set_UVC(U,V)
        for i in range(len(points)):
            U[i]=vectordata[num][i][0]
            V[i]=vectordata[num][i][1]
            
        Q.set_UVC(U,V)
        return Q,
    
    if frame==None:
        anim1 = animation.FuncAnimation(fig, update_quiver,frames=len(vectordata),fargs=(Q,), interval=500, blit=True)
    else:
        anim1=update_quiver(frame,Q)
    fig.tight_layout()
    
    return anim1
#function that returns movie of divergence
def DivMovie(vectordata,coord,frame_no):
    #input vectordata is the vectors of each pixel
    #coord is the coordinates of the pixels
    #if frame_no==None then it returns a Movie
    x_s=max([i[0] for i in coord])+1#sets x length of grid
    y_s=max([i[1] for i in coord])+1#sets y length of grid
    div_all=[]#list with divergence data for all times
    div_max=[]
    div_min=[]
    for i in range(len(vectordata)):
        div_loop=divergence(vectordata[i],coord,y_s,x_s)
        div_all.append(div_loop[0])
        div_max.append(max(div_loop[0]))
        div_min.append(min(div_loop[0]))
    #frame_all contains the frames for all time steps
    frame_all=[]
    for time in range(len(div_all)):
        frame_i=np.zeros([y_s,x_s])
        for cell in range(len(coord)):
            frame_i[y_s-1-coord[cell][1]][coord[cell][0]]=div_all[time][cell]
        frame_all.append(frame_i)
        
    fig = plt.figure() 
    
    if frame_no==None:
        ims=[]    
        for i in range(len(frame_all)):
            im=plt.imshow(frame_all[i],interpolation='none',cmap='jet',vmin=min(div_min),vmax=max(div_max),animated=True)            
            if i==0:
                fig.colorbar(im)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=500,repeat_delay=1000)
    
    else:
        ani=plt.imshow(frame_all[frame_no],interpolation='none',cmap='jet',vmin=min(div_min),vmax=max(div_max),animated=True)
        fig.colorbar(ani)
    plt.title('Pixelated Grid')
    
    return ani
#%%
def ReentrantDot(vectordata,points,xdim,ydim):
    avgdot_all=[]
    for time in range(len(vectordata)):
        avgdot_time=[]
        for i in range(len(vectordata[time])):
            x_loc=i%xdim#find x coord. of respective pixel
            y_loc=i//xdim#finds y coord. of respective pixel
            surrvec=[]#surrounding vectors
            
            if x_loc==(xdim-1):#right BC
                right=0
            else:
                right=sum([vectordata[time][i][j]*vectordata[time][i+1][j] for j in range(len(vectordata[time][i]))])
                surrvec.append(right)
            if x_loc!=0:#left BC
                left=sum([vectordata[time][i][j]*vectordata[time][i-1][j] for j in range(len(vectordata[time][i]))])
                surrvec.append(left)
            else:
                left=0
            if y_loc==(ydim-1):# up BC
                up=sum([vectordata[time][i][j]*vectordata[time][i+xdim-(ydim*xdim)][j] for j in range(len(vectordata[time][i]))])
            else:
                up=sum([vectordata[time][i][j]*vectordata[time][i+xdim][j] for j in range(len(vectordata[time][i]))])
            surrvec.append(up)
            if y_loc==0:#down BC
                down=sum([vectordata[time][i][j]*vectordata[time][i-xdim+(ydim*xdim)][j] for j in range(len(vectordata[time][i]))])
            else:    
                down=sum([vectordata[time][i][j]*vectordata[time][i-xdim][j] for j in range(len(vectordata[time][i]))])
            surrvec.append(down)
            #below are diagonal terms
            if x_loc==xdim-1:#up right diag
                uprightdiag=0
            else:
                if y_loc==(ydim-1):
                    uprightdiag=sum([vectordata[time][i][j]*vectordata[time][i+xdim-(ydim*xdim)+1][j] for j in range(len(vectordata[time][i]))])
                else:
                    uprightdiag=sum([vectordata[time][i][j]*vectordata[time][i+xdim+1][j] for j in range(len(vectordata[time][i]))])
            surrvec.append(uprightdiag)
            if x_loc==(xdim-1):#down right diag
                downrightdiag=0
            else:
                if y_loc==0:
                    downrightdiag=sum([vectordata[time][i][j]*vectordata[time][i-xdim+(ydim*xdim)+1][j] for j in range(len(vectordata[time][i]))])
                else:
                    downrightdiag=sum([vectordata[time][i][j]*vectordata[time][i-xdim+1][j] for j in range(len(vectordata[time][i]))])                    
            surrvec.append(downrightdiag)
            if x_loc==0:#up left diag
                upleftdiag=0
            else:
                if y_loc==(ydim-1):
                    upleftdiag=sum([vectordata[time][i][j]*vectordata[time][i+xdim-(ydim*xdim)-1][j] for j in range(len(vectordata[time][i]))])
                else:
                    upleftdiag=sum([vectordata[time][i][j]*vectordata[time][i+xdim-1][j] for j in range(len(vectordata[time][i]))])            
            surrvec.append(upleftdiag)
            if x_loc==0:#down left diag
                downleftdiag=0
            else:
                if y_loc==0:
                    downleftdiag=sum([vectordata[time][i][j]*vectordata[time][i-xdim+(ydim*xdim)-1][j] for j in range(len(vectordata[time][i]))])
                else:
                    downleftdiag=sum([vectordata[time][i][j]*vectordata[time][i-xdim-1][j] for j in range(len(vectordata[time][i]))]) 
            surrvec.append(downleftdiag)
            
            surrvecN=len(surrvec)
            sc=0
            for i in range(len(surrvec)):
                if surrvec[i]==0:
                    sc+=1
            #print(sc,surrvecN)
            if sc==surrvecN:
                avgdot=None
            else:
                avgdot=sum(surrvec)/surrvecN
            avgdot_time.append(avgdot)
        
        avgdot_all.append(avgdot_time)
    
    return avgdot_all
#%%time locator that only works for pixelated data
def TimeLoc(pxvectors,coord,data):
    tau=data[8][4]
    res=int(np.sqrt(len(coord)))
    divstd=[np.std(divergence(t,coord,res,res)[0]) for t in pxvectors]
    m=2
    divstdmavg=[]
    for i in range(m,len(divstd)-m):
        divstdmavg.append(np.mean(divstd[i-m:i+m+1]))    
    topdiv=max(divstdmavg[:tau])
    diff=[]
    for i in range(len(divstdmavg)-1):
        diff.append(divstdmavg[i+1]-divstdmavg[i])
    topdiff=max(diff[:tau])
    #plt.plot(np.arange(0,len(divstdmavg),1),divstdmavg,'x')
    for i in range(len(divstdmavg)-2):
        d1=diff[i]
        d2=diff[i+1]
        if divstdmavg[i]>topdiv and divstdmavg[i]<divstdmavg[i+1] and divstdmavg[i]<divstdmavg[i+2] and d1>topdiff and d2>topdiff:
            timef=i-1+m
            break
        if i==len(divstdmavg)-3:
            timef=0
    return timef
#%%reviewing the focal_quality_indicator function
def focal_quality_indicator(vectors,coord,focalpoint,timestep,vcond,tau):
    x_dim=int(np.sqrt(len(coord)))
    radius=tau*vcond
    indexf=focalpoint[1]*x_dim+focalpoint[0]
    ll=focalpoint[0]-radius
    rl=focalpoint[0]+radius
    
    vfieldnew=[[0,0] for i in range(x_dim**2)]
    for t in range(timestep,timestep+tau+1):
        for cell in range(x_dim**2):
            if cell!=indexf and ll<=coord[cell][0]<=rl:    
                if distance_checker(radius,indexf,cell,coord,x_dim) == True:
                    vfieldnew[cell][0]+=vectors[t][cell][0]
                    vfieldnew[cell][1]+=vectors[t][cell][1]
    #print(len(vfieldnew))
    def vfieldp(point,focalpoint,radius,x_dim):
        ui=point[0]-focalpoint[0]
        vi=point[1]-focalpoint[1]
        if abs(vi)<=radius:
            vi=vi
        elif abs(vi)>radius:
            if vi+radius>x_dim:
                vi=vi-x_dim
            elif vi<0:
                vi=vi+x_dim        
        mag=np.sqrt(ui**2+vi**2)        
        return [ui/mag,vi/mag]
    
    dot_summation=[]
    
    for cell in range(len(vfieldnew)):
        if vfieldnew[cell]!=[0,0]:
            #print(cell)
            mag=np.sqrt(vfieldnew[cell][0]**2+vfieldnew[cell][1]**2)
            xn=vfieldnew[cell][0]/mag
            yn=vfieldnew[cell][1]/mag
            perfp=vfieldp(coord[cell],focalpoint,radius,x_dim)
            dot_pr=xn*perfp[0]+yn*perfp[1]
            dot_summation.append(dot_pr)
    #print(dot_summation)
    if len(dot_summation)==0:
        quality=0
    else:
        quality = sum(dot_summation)/len(dot_summation)
            
    #VectorMovie([vfieldnew],coord,0)
    return quality

def focal_quality_indicator3(vectors,coord,connections,timestep,vcond,tau):
    xdim=int(np.sqrt(len(coord)))
    radius=tau*vcond
    #preparing the connections
    
    def vfieldp(point,focalpoint,radius,x_dim):
        ui=point[0]-focalpoint[0]
        vi=point[1]-focalpoint[1]
        if abs(vi)<=radius:
            vi=vi
        elif abs(vi)>radius:
            if vi+radius>x_dim:
                vi=vi-x_dim
            elif vi<0:
                vi=vi+x_dim        
        mag=np.sqrt(ui**2+vi**2)        
        return [ui/mag,vi/mag]
    
    quality_all=[]
    for cell in range(xdim**2):
        focalpoint=[cell%xdim,cell//xdim]
        vfieldnew=[[0,0] for i in range(xdim**2)]    
        for t in range(timestep,timestep+tau+1):           
            for c in connections[t-timestep][cell]:
                if vfieldnew[c][0]==0:
                    vfieldnew[c][0]+=vectors[t][c][0]
                if vfieldnew[c][1]==0:
                    vfieldnew[c][1]+=vectors[t][c][1]
        dot_summation=[]        
        for cellv in range(len(vfieldnew)):
            if vfieldnew[cellv]!=[0,0]:
                mag=np.sqrt(vfieldnew[cellv][0]**2+vfieldnew[cellv][1]**2)
                xn=vfieldnew[cellv][0]/mag
                yn=vfieldnew[cellv][1]/mag
                perfp=vfieldp(coord[cellv],focalpoint,radius,xdim)
                dot_pr=xn*perfp[0]+yn*perfp[1]
                dot_summation.append(dot_pr)
        if len(dot_summation)==0:
            quality=0
        else:
            quality = sum(dot_summation)/len(dot_summation)
        quality_all.append(quality)
    
    #VectorMovie([vfieldnew],coord,0)
    return quality_all
#%%
def miss_fire2(x,T,Tau,variation):
    data = x[0]
    N = len(data[0])
    Tau_plus = Tau + 1
    #time_difference
    miss_fire_list = [0]*len(x[0])
    hist = []    
    data_fault=[]
    
    for node in range(N):
        time_fired = -1
        delta_time = 0
        for time in range(len(data)):
            if data[time][node] == Tau_plus:                
                if time_fired > -1:         
                    delta_time = time - time_fired                    
                    if np.abs(delta_time - T) > variation:
                        hist.append(delta_time)
                        miss_fire_list[time] = miss_fire_list[time] + 1
                        data_fault.append(time)
                        data_fault.append(node)
                    time_fired = time
            
                if time_fired == -1:
                    time_fired = time
        
    miss_fire_list_ratio = []
    for i in range(len(miss_fire_list)):
        miss_fire_list_ratio.append(miss_fire_list[i]/N)
    for i in range(len(miss_fire_list_ratio)):
        if miss_fire_list_ratio[i]>0 and miss_fire_list_ratio[i]>miss_fire_list_ratio[i-1] and miss_fire_list_ratio[i]<miss_fire_list_ratio[i+1]:
            t_ident=i
            break
        elif i==len(data)-1:
            t_ident=0
    time_fault=data_fault[::2]
    node_fault=data_fault[1::2]
    time_fault_index=[]
    for i in range(len(time_fault)):
        if time_fault[i]==t_ident:
            time_fault_index.append(i)
    node_fault_loc=[]
    for i in time_fault_index:
        node_fault_loc.append(node_fault[i])
    
    if t_ident!=0:
        locn=node_fault_loc[0]
    else:
        locn=None    
    return locn,t_ident

def condvelavg(smltn,it):
    #smtln is the result of the RunState function and it is the number of times
    #you want the function to be calculated
    xdim=smltn[8][2]
    vc_all=[]
    for i in range(it):
        minloc=min([i for i in range(len(smltn[1])) if smltn[1][i][0]>xdim-1])#finds 
        for t in range(len(smltn[0])):
            if set(smltn[0][t][minloc:])!={0}:
                vc=xdim/t
                vc_all.append(vc)
                break
            
    vc_m=np.mean(vc_all)
    vc_std=np.std(vc_all)/np.sqrt(it)
    return vc_m,vc_std
#%%
def distance_checkerD(R,i,j,y_size):#checks the distance between 2 nodes 
    #in a way to allow periodic BCs in y-direction and open in x-direction
    diff=i[1]-j[1]
    if abs(diff)<=R:# for rest cases
        d = ((i[0]- j[0])**2 + (i[1] - j[1]) **2 )**0.5
    elif abs(diff)>R:    
        if i[1]+R>y_size:#sets upper BC
            d = ((i[0]- j[0])**2 + (i[1]-y_size - j[1]) **2 )**0.5
        elif i[1]-R<0:#sets lower BC
            d = ((i[0]- j[0])**2 + (i[1]+y_size - j[1]) **2 )**0.5
        else:
            d = ((i[0]- j[0])**2 + (i[1] - j[1]) **2 )**0.5
    if d <= R:
        return d
    else:
        return ((i[0]- j[0])**2 + (i[1] - j[1]) **2 )**0.5
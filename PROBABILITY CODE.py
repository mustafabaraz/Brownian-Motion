# -*- coding: utf-8 -*-
"""
Created on Mon May 23 14:24:57 2022

@author: MUSTAFA
"""

######################
######################

# Below code plots the approximate probability distrubtion of a (non-interacting) statistical 
# ensemble that is put in the potential specified in Rolf Landauer's paper.

# In this code, Langevin dynamics equation is to be used to simulate the thermodynamical process.

######################
######################

# Import the necessary packages. 
# Note that functools--->reduce package collapses nested arrays/lists into a single array/list.
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce


# Define the parameters of the simulation.
# Note that the parameters are reduced around 1.
M=1 # Masses of the particles are taken to be identical to each other.
N=200 # Number of particles.
k_B=1 # Boltzmann constant.
gamma=0.75 # Damping constant for the Brownian motion of the particles.
t = np.linspace(0,50, 5000) # Time array for the simulation.
dt=t[1]-t[0] # Time step.

# Define the temperature profile as a piecewise function that takes discrete
# values in discrete intervals.
# Moreover, note that we assume (local) thermal equilibrium between the particles and the heat baths.
# Note also that A point is taken to be at x∼-1.53, B point at x∼-0.1, C point at x∼1.5, and
# D point at x∼1.63 (these are calculated from the graph of U(x)=x^4-5x^2-x).
def temp(x):
    temperature=np.piecewise(x, [(x<=-0.1), (x>-0.1) & (x<1.5) , (x >= 1.5)], [0.1, 500, 0.1])
    return temperature

# Define the force acting on each particle as -dU/dx since we are in 1-D.
def force(x):
    return -(4*x**3-10*x-1)

# !!!!! IMPORTANT !!!!! Prof. Landauer's potential is approximated here as U(x)=x^4-5x^2-x.
# Thus, -dU/dx=-4x^3-10x-1.

# Moreover, the low temperature regions are extended here to -inf and +inf to obtain
# better results.


# Particles are starting from random positions in a region that containts Landauer's A & D regions.
x0= np.random.uniform(low=-2.1, high=2.1, size=((N,)))
# Particles are given random kicks both to the right and to the left.
vx0= np.random.uniform(low=-0.6, high=0.6, size=((N,)))
# Below y-position array is only defined so that the motion of the particles can be simulated.
y=np.zeros(N,)

# Position list that stores all of the position values that the particles occupy in a given time.
pos_list=[]

for i in range (0,len(t)): # This loop is for iterating over time.
    R = np.random.randn(N,) # This is the Wiener noise factor. It takes random values chosen
                            # from a standard normal distribution with mean=0 and std.dev=1.
                            # Note that the factor is changing randomly at each time step.
    
    ### Below numerical method is called the "BAOAB" method. 
    ### This method is due to Leimkuhler and Matthews (JCP, 2013)
    
    ### Summary of the method: 
    # B---> Half a step in time for updating the velocities.
    # A---> Half a step in time for updating the positions.
    # O---> This step updates the velocities a full step in time as:
            # v(t+dt)=e^(-gamma*dt)v(t)+R(t)*sqrt(k_B*T/M)*sqrt(1-e^(-2*gamma*dt))
    # B---> Half a step in time for updating the velocities.
    # A---> Half a step in time for updating the positions.
    
    # Now we do the computation.
    
    # B
    vx0= vx0 + force(x0)*dt/2 
        
    # A    
    x0 = x0 + vx0*dt/2 
       
    # O  
    factor1 = np.exp(-gamma*dt)
    factor2 = np.sqrt(1- factor1**2)*np.sqrt(k_B*temp(x0))
    vx0 = factor1*vx0 + R*factor2
        
    # A   
    x0 = x0 + vx0*dt/2
        
    # B  
    vx0= vx0 + force(x0)*dt/2
    
    # At each time step, store the position array as a list (!).
    pos_list.append(x0.tolist()) 
    
    # If you delete the quotes, you can see the instantenous motion of the particles.
    """plt.plot(x0,y,'ko')
    plt.xlim([-5, 5])
    plt.show()"""

# Define a function to find the centers of a histogram.
def center(edges):
    return (edges[1:]+edges[:-1])/2.

# Seaborn package is utilized only for a good-looking plot.
import seaborn as sns
sns.set()

# Collapse the list of position arrays into a single list.
single_list = reduce(lambda x,y: x+y, pos_list)
# Use np.histogram to obtain the value of each position bin.
dist, bins=np.histogram(single_list,bins=200)
# Plot these values as a function of the positions of the centers.
p = plt.plot(center(bins), dist/len(single_list),marker='o',label="Total Data",linestyle='')
plt.xlim(-5,5)

#####

# Below codes are for the theoretical Boltzmann distribution plots in different regions of the potential profile.
# Note that the normalization constants are chosen by trial and error. 
# This is because we have (implicitly defined) discontinuities in the temperature profile, which forbids
# us to find the theoretical normalization factors by hand.

# Lastly, notice that Boltzmann distribution for the case is P(x)∼e^(-U/T)
# since k_B=1 is taken in the code. Temperature values can be updated by the temp(x) function.

#####

# REGION 1:
theoric_list=(np.linspace(-5,-0.1,num=3000)).tolist() # Position array between (5,5).
func_list1=[] # Function values are to be stored in this list.

for i in theoric_list:
    func_list1.append(np.exp((-10*(i**4-5*(i**2)-i)))/(1.2005*10**21)) # T=0.1 is taken.
    
# Plot the results.
plt.plot(theoric_list,func_list1,color='black',linestyle='dashed',label='Low Temperature')   


# REGION 2:
theoric_list=(np.linspace(-0.1,1.5,num=3000)).tolist() # Position array between (5,5).
func_list2=[] # Function values are to be stored in this list.
for i in theoric_list:
    func_list2.append(np.exp((-(1/500)*(i**4-5*(i**2)-i)))/1000) # T=500 is taken.
 
# Plot the results.
plt.plot(theoric_list,func_list2,color='#FAC205',linestyle='dashed',label='High Temperature')


# REGION 3:
theoric_list=(np.linspace(1.5,5,num=3000)).tolist() # Position array between (5,5).
func_list3=[] # Function values are to be stored in this list.
for i in theoric_list:
    func_list3.append(np.exp((-(10)*(i**4-5*(i**2)-i)))/(2*10**36)) # T=0.1 is taken.

# Plot the results.    
plt.plot(theoric_list,func_list3,color='#E50000',linestyle='dashed',label='Low Temperature')

# Below is for all of the plots together.
plt.xlabel('Position') 
plt.ylabel('Probability Density') 
plt.legend()
plt.show()

#########################
#########################

# !!! IMPORTANT NOTE CONCERNING THE CODE !!!
    
# Above "curve-fitting" is done by hand. If you change the parameters, you need to
# reset the scaling of the Boltzmann curves to obtain good results. 
# A built-in curve-fitting could have been done as well, but if we did this, we would not have
# shown that the fitting curve has an equation of P(x)∼e^(-U/T), which is basically
# what we are trying to prove here. We want show that a local Boltzmann distribution
# holds in different regions and the probability distribution favors region A in place of D
# if we increase the temperature in the BC region. These are achieved in this code.

##########################
##########################


## CHANGE THE LAST PART OF THIS CODE TO BELOW VALUES TO OBTAIN THE DIFFERENT CURVE FITTINGS 
## IN THE FINAL PROJECT.

## FOR INTERMEDIATE TEMPERATURE DIFFERENCE:
    
""" 
theoric_list=(np.linspace(-5,-0.1,num=3000)).tolist()
func_list1=[]
for i in theoric_list:
    func_list1.append(np.exp((-10*(i**4-5*(i**2)-i)))/(3.37*10**21))
plt.plot(theoric_list,func_list1,color='black',linestyle='dashed',label='Low Temperature')   

theoric_list=(np.linspace(-0.1,1.5,num=3000)).tolist()
func_list2=[]
for i in theoric_list:
    func_list2.append(np.exp((-(1/5)*(i**4-5*(i**2)-i)))/1300)
    
plt.plot(theoric_list,func_list2,color='#FAC205',linestyle='dashed',label='High Temperature')

theoric_list=(np.linspace(1.5,5,num=3000)).tolist()
func_list3=[]
for i in theoric_list:
    func_list3.append(np.exp((-(10)*(i**4-5*(i**2)-i)))/(3.91*10**35))
    
plt.plot(theoric_list,func_list3,color='#E50000',linestyle='dashed',label='Low Temperature')

plt.xlabel('Position') 
plt.ylabel('Probability Density') 
plt.legend()
"""

# FOR LOW TEMPERATURE DIFFERENCE:
    
""" 
theoric_list=(np.linspace(-5,-0.1,num=3000)).tolist()
func_list1=[]
for i in theoric_list:
    func_list1.append(np.exp((-10*(i**4-5*(i**2)-i)))/(4.73*10**21))
plt.plot(theoric_list,func_list1,color='black',linestyle='dashed',label='Low Temperature')   

theoric_list=(np.linspace(-0.1,1.5,num=3000)).tolist()
func_list2=[]
for i in theoric_list:
    func_list2.append(np.exp((-(5)*(i**4-5*(i**2)-i)))/(4*10**18))
    
plt.plot(theoric_list,func_list2,color='#FAC205',linestyle='dashed',label='High Temperature')

theoric_list=(np.linspace(1.5,5,num=3000)).tolist()
func_list3=[]
for i in theoric_list:
    func_list3.append(np.exp((-(10)*(i**4-5*(i**2)-i)))/(2.05*10**35))
    
plt.plot(theoric_list,func_list3,color='#E50000',linestyle='dashed',label='Low Temperature')

plt.xlabel('Position') 
plt.ylabel('Probability Density') 
plt.legend(loc=2, prop={'size':8})
"""  




# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:29:51 2022

@author: MUSTAFA
"""

##################
##################

# Below code simulates the Brownian motion of the particles that are put in the potential
# given in Rolf Landauer's paper.

##################
##################

# Import the necessary packages. 
# Note that the FuncAnimation package is used for simulating the instantenous motion.

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

# Define the discontinous temperature profile.
def temp(x):
    temperature=np.piecewise(x, [(x<=-0.1), (x>-0.1) & (x<1.5) , (x >= 1.5)], [0.1, 500, 0.1])
    return temperature

# Define the force acting on the particles.
# Note that the same potential profile is utilized as the previous code for the
# probability densities.
def force(x):
    return -(4*x**3-10*x-1)

# Define a generator function that yields the position (x,y) values.
# Notice that the y value should be always constant since we are in 1-D.
# Preferably, we set y=0.
def motion_gen(N,gamma,k_B,dt):
    # Random initial conditions for the positions and velocities as before.
    x= np.random.uniform(low=-2.1, high=2.1, size=((N,))) 
    y= np.zeros(N,) # y=0 is set.
    vx= np.random.uniform(low=-0.6, high=0.6, size=(N,))
    
    yield x,y # Yield the initial conditions. Note that the function will start to be executed
              # after this "yield" part.
    
    # In this loop, we take care of the simulation dynamics.
    while True: 
        R = np.random.randn(N,) # Wiener noise factor.
        
        #B
        vx= vx + force(x)*dt/2
        
        #A
        x = x + vx*dt/2
       
        #O
        factor1 = np.exp(-gamma*dt)
        factor2 = np.sqrt(1- factor1**2)*np.sqrt(k_B*temp(x))
        vx = factor1*vx + R*factor2
        
        #A
        x = x + vx*dt/2
        
        # B
        vx= vx + force(x)*dt/2
        
        yield x,y # yield x,y at each time step.
        

# Define a function to iterate the animation.
def update(t,dynamic,scatter):
    x,y = next(dynamic) # This "next" command allows us to pass the next position values to be calculated
                        # into the while loop of the motion_gen function.
    
    scatter.set_data([x,y]) # Data are stored.
    return scatter, # Comma is put to keep iteration going.

# Define the parameters of the simulation.
# Note that the parameters are normalized.
N=200
k_B=1
gamma=0.75
# Define the time interval.
t = np.linspace(0, 10, 1000)
# Define the "dynamic" input in terms of the generator function above.
dynamic = motion_gen(N,gamma,k_B,t[1]-t[0]) 

x,y = next(dynamic) # This "next" command will allow us to "plot" the instantenous positions below.

# Below part simulates the motion.
fig, ax = plt.subplots(1,1,figsize=(8,8)) # Create an empty figure and axes.
scatter, = ax.plot(x,y,'ob',ms=5) # Plot the "updated" positions on this empty figure.
ax.set_xlim(-6,6); ax.set_ylim(-2,2) # Set the x and y limits to see the process better.

# Animate the motion of the particles.
animation = FuncAnimation(fig, update, frames=t, fargs=(dynamic,scatter), blit=True, repeat=False)
# plt.show() is used to show the motion at each time step only once.
plt.show()



import numpy as np 
import matplotlib.pyplot as plt 
from itertools import islice # import this to slice time within the "for" loop

# parameters
Rm     = 1e6    # resistance (ohm)
Cm     = 2e-8   # capacitance (farad)
taum   = Rm*Cm  # time constant (seconds)
Vr     = -.060  # resting membrane potential (volt)
Vreset = -.070  # membrane potential after spike (volt)
Vth    = -.050  # spike threshold (volt)
Vs     = .020   # spiking potential (volt)

dt     = .001   # simulation time step (seconds)
T      = 1.0    # total time to simulate (seconds)
time   = np.linspace(dt,T,T/dt) # vector of timepoints we will simulate

#functions
def initialize_simulation():
    # zero-pad membrane potential vector 'V' and spike vector 'spikes'
    V      = np.zeros(time.size) # preallocate vector for simulated membrane potentials
    spikes = np.zeros(time.size) # vector to denote when spikes happen- spikes will be added after membrane simulation
    V[0]   = Vr # set first time point to resting potential
    return V,spikes

def logistic_map(a,x0,nsteps):
    # function to simulate logistic map function:
    #  x_{n+1} = a*x_n * (1-x_n)
    x    = np.zeros(nsteps)
    x[0] = x0
    for ii in range(1,nsteps):
        x[ii] = a*x[ii-1] * (1-x[ii-1])
    return x

def plot_potentials(time,V,timeSpikes):
    # plots membrane potential (V) against time (time), and marks spikes with red markers (timeSpikes)
    plt.show()
    plt.plot(time,V,'k',timeSpikes,np.ones(timeSpikes.size)*Vs,'ro')
    plt.ylabel('membrane potential (mV)')
    plt.xlabel('time (seconds)')

def check_solutions( result, solution_filename ):
    # check solutions against provided values
    solution = np.load( solution_filename )
    if ( np.linalg.norm( np.abs( result - solution ) ) < 0.1 ):
    	print( '\n\n ---- problem solved successfully ---- \n\n' )

def integrate_and_fire( V, spikes, ii, Ie ):
    # function to integrate changes in local membrane potential and fire if threshold reached
    # V - vector of membrane potential
    # spikes - spike marker vector
    # i - index (applied to V and spikes) for current time step
    # Ie - input current at this time step (scalar of unit amp)

    dV  =  ((Vr - V[ii-1]) + (Rm*Ie)) / taum # 1: calculate change in membrane potential (dV)
    V[ii] = V[ii-1] + (dV*dt) # 2: integrate over given time step (Euler method)
	
    if V[ii] > Vth :
        V[ii] = Vreset
        spikes[ii] = 1		# 3: does the membrane potential exceed threshold (V > Vth)?

    return V,spikes # output the membrane potential vector and the {0,1} vector of spikes

def problem_2():
    #//////////////////////////////////////////
    # problem 2 - oscillating current input //
    #////////////////////////////////////////
    # Use the LIF implementation from problem 1.
    # Create a current input which:
    #       - starts at 0 A
    #       - oscillates with amplitude 20 nA at stim_time[0]
    #       - stops oscillating and returns to 0 A at stim_time[1]
    #
    # output:
    # Plot the resulting simulated membrange potential of the LIF, and save the 
    # membrane potential in a vector named "V_prob2".
    
    # problem-specific parameters
    f     = 10 # Hz
    phase = np.pi
    A = 2.0e-8
    t = 0

    stim_time = [.2,.8]

    V,spikes = initialize_simulation() 	# initialize simulation

    for i in range(1,len(time)):
        if time[i] > stim_time[0] and time[i] < stim_time[1]:
            t += dt
            Ie = A * np.cos(2 * np.pi * f * t + phase)
        else :
            Ie = 0  
        integrate_and_fire( V, spikes, i, Ie ) 
    
    # add spikes
    V[spikes==1] = Vs
    
    # PLOT membrane potential
    plot_potentials(time,V,time[spikes==1])
    plt.title('P2 : oscillating current input')
    
    # output:
    V_prob2 = V
    return V_prob2
V_prob2 = problem_2()
plt.show()
check_solutions( V_prob2, 'problem2.npy' )
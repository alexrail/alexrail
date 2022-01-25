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

def problem_3():
    #////////////////////////////////////////////////////
    # problem 3 - scan across oscillation frequencies //
    #//////////////////////////////////////////////////
    # Using previous problem's simulation (i.e. oscillating current input),
    # run a simulation per frequency stored in the variable "freq".
    # 
    # output: plot the results, and then save the number of spikes generated in 
    # each run in a variable named "nspikes_prob3".
    
    # problem-specific parameters
    freq  = np.linspace(15,50,(50-15)+1) # Hz
    phase = np.pi
    oscillation_amplitude = 4e-8 # amps
    t = 0
    # initialize array
    nSpikes  = np.zeros(freq.size)
    stim_time = [.2,.8]

    # iterate each freq
    for j,f in enumerate(freq):
	    V,spikes = initialize_simulation()
        for i in range(1,len(time)):
            if time[i] > stim_time[0] and time[i] < stim_time[1]:
                t += dt
                Ie = A * np.cos(2 * np.pi * freq * t + phase)
            else :
                 Ie = 0  
            V, spikes = integrate_and_fire( V, spikes, i, Ie ) 

        nSpikes[j] += spikes[i]		# calculate sum over spikes

    
    # PLOT number of spikes per frequency
    plt.show()
    plt.plot( freq, nSpikes, 'ko' )
    plt.title('Problem 3: Scan across oscillating frequencies')
    plt.xlabel('frequency (Hz)')
    plt.ylabel('# of spikes')
    
    return nSpikes


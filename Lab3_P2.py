#Lab 3 - P2 - Alexander Railton - 250848086
from brian2 import *
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt
from brian2 import *
import matplotlib.pyplot as plt


#///////////////////////////////////////////////////////////////
# problem 2 - Output firing rate across balanced input rates //
#/////////////////////////////////////////////////////////////

start_scope()	 # initialize simulation
			
# parameters
taum   = 20*ms   # time constant
g_L    = 10*nS   # leak conductance
E_l    = -70*mV  # leak reversal potential
E_e    = 0*mV    # excitatory reversal potential
tau_e  = 5*ms    # excitatory synaptic time constant
E_i    = -80*mV  # inhibitory reversal potential
tau_i  = 10*ms   # inhibitory synaptic time constant
Nin    = 1000	 # number of synaptic inputs
Ne     = int(0.8*Nin)   # number of excitatory inputs
Ni     = int(0.2*Nin)   # number of inhibitory inputs
Vr     = E_l     # reset potential
Vth    = -50*mV  # spike threshold
refrac = 5*ms 	 # refractory period
	
we = 1*nS	 # excitatory synaptic weight
wi = 2*nS	 # inhibitory synaptic weight

# varying parameters
ve = np.linspace(1,50,50); vi = np.linspace(1,50,50)

outputspikes = np.zeros(len(ve)) #storing spikes

eqs = '''
dv/dt = ( E_l - v + g_e*(E_e-v) + g_i*(E_i-v) ) / taum  : volt (unless refractory)
dg_e/dt = -g_e/tau_e  : 1  # excitatory conductance (dimensionless units)
dg_i/dt = -g_i/tau_i  : 1  # inhibitory conductance (dimensionless units)
'''

# create neuron
N = NeuronGroup( 1, model=eqs, threshold='v>Vth', reset='v=Vr', refractory='5*ms', method='euler' )

# initialize neuron
N.v = E_l

# create inputs
Pe = PoissonGroup( 1, 1000*Hz ); Pi = PoissonGroup( 1, 800*Hz )

# create connections
synE = Synapses( Pe, N, 'w: 1', on_pre='g_e += w' ); synE.connect( i=0, j=0 ); synE.w[0] = ( we ) / g_L
synI = Synapses( Pi, N, 'w: 1', on_pre='g_i += w' ); synI.connect( i=0, j=0 ); synI.w[0] = ( wi ) / g_L

# record model state
M = StateMonitor( N, ('v','g_i'), record=True )
S = SpikeMonitor( N )


# use a single loop and increase the excitatory and inhibitory rate together
store()
for i in range(len(ve)):
    restore()
    Pe.rates = (ve[i]*Ne) * Hz
    Pi.rates = (vi[i]*Ni) * Hz
    run(1000*ms)
    outputspikes[i] = S.num_spikes

# ***PLOT OUTPUTS***
plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

plt.plot(ve, outputspikes, linewidth = 0.75)
plt.xlabel('Potential')
plt.ylabel('Spikes per Second')
plt.title('Spikes per Second for Simulations')
plt.xlim(xmin = ve.min())
plt.xlim(xmax = ve.max())
plt.show()
# after saving values, plot the output spike rate as a function of the input spike rates ve and vi
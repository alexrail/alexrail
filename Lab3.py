#Lab 3 - Alexander Railton
from brian2 import *
prefs.codegen.target = 'numpy'
import matplotlib.pyplot as plt
from brian2 import *
import matplotlib.pyplot as plt
tau = 20 * ms
E_l = -70 * mV
start_scope() # clears the workspace of previous Brian objects

# parameters and model equations
tau = 20*ms
#tau*x = -x
eqs = ''' 
dx/dt = -x/tau : 1 # : 1 is dimensionless
'''

# create model
N = NeuronGroup( 1, eqs )

# initialize model
N.x = 0

# record model state
M = StateMonitor( N, 'x', record=True )

run(100*ms)

N.x = 1

run(100*ms)

plt.plot( M.t/ms, M.x[0] )
# plt.show()

start_scope()
			
# parameters
taum   = 20*ms   # time constant
g_L    = 10*nS   # leak conductance
E_l    = -70*mV  # leak reversal potential
E_e    = 0*mV    # excitatory reversal potential
tau_e  = 5*ms    # excitatory synaptic time constant
Vr     = E_l     # reset potential
Vth    = 50*mV   # spike threshold
Vs     = 20*mV   # spiking potential
w_e    = 5*nS 	 # excitatory synaptic weight

# model equations
eqs = '''
dv/dt = ( E_l - v + g_e*(E_e-v) ) / taum  : volt (unless refractory)
dg_e/dt = -g_e/tau_e  : 1  # excitatory conductance (dimensionless units)
'''

# create neuron
N = NeuronGroup( 1, model=eqs, threshold='v>Vth', reset='v=Vr', refractory='5*ms', method='euler' )

# initialize neuron
N.v = E_l

# create inputs
indices = array([0, 0, 0]); times = array([25, 50, 75])*ms
input = SpikeGeneratorGroup( 1, indices, times )

# create connections
S = Synapses( input, N, 'w: 1', on_pre='g_e += w' )
S.connect( i=0, j=0 ); S.w[0] = ( w_e ) / g_L

# record model state
M = StateMonitor( N, ('v','g_e'), record=True )

# run simulation
run( 100*ms )

# plot output
fig, ax1 = plt.subplots(); ax2 = ax1.twinx()
ax1.plot( M.t/ms, M.v[0] ); ax2.plot( M.t/ms, M.g_e[0], 'g--' );
ax1.set_xlabel( 'time (ms)' ); ax1.set_ylabel( 'V_m (V)' ); ax2.set_ylabel( 'g_e (units of g_L)' )
#plt.show()

start_scope()
			
# parameters
taum   = 20*ms   # time constant
g_L    = 10*nS   # leak conductance
E_l    = -70*mV  # leak reversal potential
E_e    = 0*mV    # excitatory reversal potential
tau_e  = 5*ms    # excitatory synaptic time constant
Vr     = E_l     # reset potential
Vth    = -50*mV  # spike threshold
Vs     = 20*mV   # spiking potential
w_e    = 5*nS 	 # excitatory synaptic weight
v_e    = 5*Hz    # excitatory Poisson rate

# model equations
eqs = '''
dv/dt = ( E_l - v + g_e*(E_e-v) ) / taum  : volt (unless refractory)
dg_e/dt = -g_e/tau_e  : 1  # excitatory conductance (dimensionless units)
'''

# create neuron
N = NeuronGroup( 1, model=eqs, threshold='v>Vth', reset='v=Vr', refractory='5*ms', method='euler' )

# initialize neuron
N.v = E_l

# create inputs
P = PoissonGroup( 1, v_e )

# create connections
S = Synapses( P, N, 'w: 1', on_pre='g_e += w' )
S.connect( i=0, j=0 ); S.w[0] = ( w_e ) / g_L

# record model state
M = StateMonitor( N, ('v','g_e'), record=True )

# run simulation
run( 1000*ms )
#for ii in np.arange(10):
#   P.rates((np.arange(10,1000,10))[ii])*Hz
#   run( 1000*ms )

# plot output
fig, ax1 = plt.subplots(); ax2 = ax1.twinx()
ax1.plot( M.t/ms, M.v[0] );
ax1.set_xlabel( 'time (ms)' ); ax1.set_ylabel( 'V_m (V)' ); ax2.set_ylabel( 'g_e (units of g_L)' )

start_scope()
			
# parameters
taum   = 20*ms   # time constant
g_L    = 10*nS   # leak conductance
E_l    = -70*mV  # leak reversal potential
E_i    = -80*mV  # inhibitory reversal potential
tau_i  = 10*ms   # inhibitory synaptic time constant
Vr     = E_l     # reset potential
Vth    = 50*mV   # spike threshold
Vs     = 20*mV   # spiking potential
w_i    = 5*nS 	 # inhibitory synaptic weight
v_i    = 5*Hz    # inhibitory Poisson rate

# model equations
eqs = '''
dv/dt = ( E_l - v + g_i*(E_i-v) ) / taum  : volt (unless refractory)
dg_i/dt = -g_i/tau_i  : 1  # excitatory conductance (dimensionless units)
'''

# create neuron
N = NeuronGroup( 1, model=eqs, threshold='v>Vth', reset='v=Vr', refractory='5*ms', method='euler' )

# initialize neuron
N.v = E_l

# create inputs
P = PoissonGroup( 1, 10*Hz )

# create connections
S = Synapses( P, N, 'w: 1', on_pre='g_i += w' )
S.connect( i=0, j=0 ); S.w[0] = ( 5 * nS ) / g_L

# record model state
M = StateMonitor( N, ('v','g_i'), record=True )

# run simulation
run( 1000*ms )

# plot output
fig, ax1 = plt.subplots(); ax2 = ax1.twinx()
ax1.plot( M.t/ms, M.v[0] ); ax2.plot( M.t/ms, M.g_i[0], 'g--' );
ax1.set_xlabel( 'time (ms)' ); ax1.set_ylabel( 'V_m (V)' ); ax2.set_ylabel( 'g_i (units of g_L)' )
#plt.show()

start_scope()
			
# parameters
taum   = 20*ms   # time constant
g_L    = 10*nS   # leak conductance
E_l    = -70*mV  # leak reversal potential
E_e    = 0*mV    # excitatory reversal potential
tau_e  = 5*ms    # excitatory synaptic time constant
E_i    = -80*mV  # inhibitory reversal potential
tau_i  = 10*ms   # inhibitory synaptic time constant
K      = 1000	 # number of synaptic inputs
Ke     = int(0.8*K)   # number of excitatory inputs
Ki     = int(0.2*K)   # number of inhibitory inputs
Vr     = E_l     # reset potential
Vth    = -50*mV   # spike threshold
Vs     = 20*mV   # spiking potential

# model equations
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
synE = Synapses( Pe, N, 'w: 1', on_pre='g_e += w' ); synE.connect( i=0, j=0 ); synE.w[0] = ( 5 * nS ) / g_L
synI = Synapses( Pi, N, 'w: 1', on_pre='g_i += w' ); synI.connect( i=0, j=0 ); synI.w[0] = ( 25 * nS ) / g_L

# record model state
M = StateMonitor( N, ('v','g_i'), record=True )
S = SpikeMonitor( N )

# run simulation
run( 1000*ms )

# plot output
plt.figure(figsize=(15,5)); plt.plot( M.t/ms, M.v[0] );
plt.show()
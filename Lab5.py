# from brian2 import *

# # network parameters
# N = 1000; Ne = int( 0.8*N ); Ni = int( 0.2*N );

# # membrane parameters
# C = 200 * pF
# taum = 20 * msecond
# gL = C / taum
# El = -70 * mV
# Vth = -50 * mV

# # synapse parameters
# Ee = 0 * mvolt
# Ei = -80 * mvolt
# taue = 5 * msecond
# taui = 5 * msecond

# # excitatory and inhibitory weights
# g = 4 							# balance between excitation and inhibition
# we = 0.5 * nS 						# excitatory synaptic weight
# wi = ( 7 * g * (we/nS) ) * nS 				# inhibitory synaptic weight
# we = (we / gL) * nS; wi /= gL * nS

# # input parameters
# input_rate = 2*Hz					# Poisson input rate

# # membrane equation
# eqs = Equations('''
# dv/dt = ( gL*(El-v) + ge*(Ee-v) + gi*(Ei-v) ) * (1./C) : volt
# dge/dt = -ge*(1./taue) : siemens
# dgi/dt = -gi*(1./taui) : siemens''')

# # setup network
# P = NeuronGroup( N, model=eqs, threshold='v>Vth', reset='v=El', refractory=5*ms )
# Pe = P[0:Ne]; Pi = P[Ne:]

# # setup excitatory and inhibitory connections
# Ce = Synapses( Pe, P, on_pre='ge+=we' )
# Ce.connect( True, p=0.01 )
# Ci = Synapses( Pi, P, on_pre='gi+=wi' )
# Ci.connect( True, p=0.01 )

# # setup Poisson input
# Ie = PoissonInput( P, 'ge', Ne, input_rate, weight=we )

# # initialize network
# P.v = randn( len(P) ) * 5 * mV - 70 * mV

# # record spikes + rates (excitatory population)
# M = SpikeMonitor( Pe )
# R = PopulationRateMonitor( Pe )


# # run simulation
# run( 1 * second, report='text' )

# # Make plot
# figure( figsize=(15,5) )
# xlabel( 'time (ms)' ); ylabel( 'cell id' )
# plot( M.t/ms, M.i, '.k' )
# show()

# figure( figsize = (15,5))
# plot( R.rate )
# show()

from brian2 import *

def coefficient_of_variation( spike_monitor, N ):
	# calculate Cv of inter-spike intervals
	cv = np.zeros( N )
	for ii in range(N):
		tau = diff( spike_monitor.spike_trains()[ii] )
		cv[ii] = std( tau ) / mean( tau )
	return cv

# network parameters
N = 1000; Ne = int( 0.8*N ); Ni = int( 0.2*N );

# membrane parameters
C = 200 * pF
taum = 20 * msecond
gL = C / taum
El = -70 * mV
Vth = -50 * mV

# synapse parameters
Ee = 0 * mvolt
Ei = -80 * mvolt
taue = 5 * msecond
taui = 5 * msecond

# excitatory and inhibitory weights
g = 1							# balance between excitation and inhibition
we = 0.5 * nS 						# excitatory synaptic weight
wi = ( 7 * g * (we/nS) ) * nS 				# inhibitory synaptic weight

# input parameters
input_rate = 4*Hz					# Poisson input rate

# membrane equation
eqs = Equations('''
dv/dt = ( gL*(El-v) + ge*(Ee-v) + gi*(Ei-v) ) * (1./C) : volt
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens''')

# setup network
P = NeuronGroup( N, model=eqs, threshold='v>Vth', reset='v=El', refractory=5*ms )
Pe = P[0:Ne]; Pi = P[Ne:]

# setup excitatory and inhibitory connections
Ce = Synapses( Pe, P, on_pre='ge+=we' )
Ce.connect( True, p=0.01 )
Ci = Synapses( Pi, P, on_pre='gi+=wi' )
Ci.connect( True, p=0.01 )

# setup Poisson input
Ie = PoissonInput( P, 'ge', Ne, input_rate, weight=we )

# initialize network
P.v = randn( len(P) ) * 5 * mV - 70 * mV

# record spikes + rates (excitatory population)
M = SpikeMonitor( Pe )
R = PopulationRateMonitor( Pe )

# run simulation
run( 1 * second, report='text' )

# Make plot
figure( figsize=(15,5) )
xlabel( 'time (ms)' ); ylabel( 'cell id' )
plot( M.t/ms, M.i, '.k' )
show()

figure( figsize = (15,5))
plot( R.rate )
show()

m = mean(R.rate)
print(m)
cv = coefficient_of_variation( M , Ne)
print(cv)

#histogram 
figure( figsize = (15,5))
hist(cv)
show()


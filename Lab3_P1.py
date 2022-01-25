#Lab 3 - P1 - Alexander Railton - 250848086
from brian2 import *
prefs.codegen.target = 'numpy'
import matplotlib.pyplot as plt
from brian2 import *
import matplotlib.pyplot as plt


#/////////////////////////////////////////////////////////////
# problem 1 - V_m statistics in different input regimes    //
#///////////////////////////////////////////////////////////

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
ve = np.linspace(1,50,20); vi = np.linspace(1,50,20)

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

Mu = np.zeros((20,20))
Sig = np.zeros((20,20))

store()
for i in range(len(ve)):
    for j in range(len(vi)):
        restore()
        Pe.rates = (ve[i]*Ne) * Hz
        Pi.rates = (vi[j]*Ni) * Hz
        run(1000*ms)
        Mu[i,j] = np.mean(M.v[0])
        Sig[i,j] = np.std(M.v[0])



plt.rc('text', usetex = True)
plt.rc('font', family = 'serif')

fig = plt.figure(figsize=(12,12))
plot1 = fig.add_subplot(121)
image1 = plot1.imshow(Mu, interpolation = 'bicubic', cmap = 'gist_rainbow', extent = [ve.min() , ve.max(), vi.min(), vi.max()])
plt.xlabel('Ve')
plt.ylabel('Vi')
plt.title('Mean')
cb = fig.colorbar(image1 , orientation = 'vertical')
cb.set_label(r'$\mu_{V}$')
plot2 = fig.add_subplot(122)
image2 = plot2.imshow(Sig, interpolation = 'bicubic', cmap = 'gist_rainbow', extent = [ve.min() , ve.max(), vi.min(), vi.max()])
plt.xlabel('Ve')
plt.ylabel('Vi')
plt.title('Standard deviation')
cb = fig.colorbar(image2 , orientation = 'vertical')
cb.set_label(r'$\sigma_{V}^2$')

plt.show()
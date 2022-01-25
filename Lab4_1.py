from brian2 import *
prefs.codegen.target = "numpy"
import matplotlib.pyplot as plt
import numpy as np

N = 1000
taum = 10*ms
tau_i = 10*ms
taupre = 20*ms
taupost = taupre
Ee = 0*mV
E_i = -80*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
tau_e = 5*ms
tau_i = 10*ms
F = 15*Hz
wmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= wmax
dApre *= wmax 
g_L = 10 *nS

eqs_neurons = '''
dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
dge/dt = -ge / taue : 1
dg_i/dt = -g_i/tau_i  : 1  # inhibitory conductance (dimensionless units)
'''

M = StateMonitor( N, ('v','g_i'), record=True )
S2 = SpikeMonitor( N )
N = NeuronGroup( 1, model= eqs_neurons, threshold='v>Vth', reset='v=Vr', refractory='5*ms', method='euler' )

input = PoissonGroup(N, rates=F)
neurons = NeuronGroup(1, eqs_neurons, threshold='v>vt', reset='v = vr',
                      method='linear')
S = Synapses(input, neurons,
             '''w : 1
                dApre/dt = -Apre / taupre : 1 (event-driven)
                dApost/dt = -Apost / taupost : 1 (event-driven)''',
             on_pre='''ge += w
                    Apre += dApre
                    w = clip(w + Apost, 0, wmax)''',
             on_post='''Apost += dApost
                     w = clip(w + Apre, 0, wmax)''',
             )
S.connect()
S.w = 'rand() * wmax' # change to wmax

# Experiment #1
Iinput = PoissonGroup(1, rates=F)
Si = Synapses(Iinput, neurons, 'w:1', on_pre='g_i += w' )
Si.connect()
Si.w[0] = wmax/2

# store() # clear the sim
# for i in range(len(20)):
#     for j in range(len(20)):
#         restore() # start soring the sim
#         run(1000*ms)
#         Mu[i,j] = np.mean(M.v[0])
#         Sig[i,j] = np.std(M.v[0])
#         C = Sig[i,j]/Mu[i,j]

# experiement three the delayed factor 
# add a delay in to the on pre 
# run this sim so change the ratio 1.05 -> 1.2 dA post

# experiment 4 : 
# when theres a pre synaptic spike at Apre in DE of synapse above E_1
# amplify dApre

mon = StateMonitor(S, 'w', record=[0, 1])
s_mon = SpikeMonitor(input)
t = linespace(0,1,100)
# experiment 2
for ii in range(100):
        run(1*second, report='text')
        cv.append(std(mon.v[0]/taum)

subplot(411)
plot(S.w / wmax, '.k')
ylabel('Weight / wmax')
xlabel('Synapse index')
subplot(412)
hist(S.w / wmax, 20)
xlabel('Weight / wmax')
subplot(413)
plot(mon.t/second, mon.w.T/wmax)
xlabel('Time (s)')
ylabel('Weight / wmax')
subplot(414)
plot(t, cv)
xlabel('time(s)')
ylabel('Correlation Value')
plt.show()

# C_v is sigma / <tau>
# calculate the mean spike time 
# tau from the spike monitor ' s_mon.times' and ' s_mon.t' s_mon.num_spikes

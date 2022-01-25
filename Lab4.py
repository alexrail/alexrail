from brian2 import *

N = 1000
taum = 10*ms
taupre = 20*ms
taupost = taupre
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -74*mV
taue = 5*ms
F = 15*Hz
wmax = .01
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= wmax
dApre *= wmax

eqs_neurons = '''
dv/dt = (ge * (Ee-vr) + El - v) / taum : volt
dge/dt = -ge / taue : 1
'''

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
S.w = 'rand() * wmax'
mon = StateMonitor(S, 'w', record=[0, 1])
s_mon = SpikeMonitor(input)

run(100*second, report='text')

subplot(311)
plot(S.w / wmax, '.k')
ylabel('Weight / wmax')
xlabel('Synapse index')
subplot(312)
hist(S.w / wmax, 20)
xlabel('Weight / wmax')
subplot(313)
plot(mon.t/second, mon.w.T/wmax)
xlabel('Time (s)')
ylabel('Weight / wmax')
plt.show()
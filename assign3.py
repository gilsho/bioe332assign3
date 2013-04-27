import brian_no_units
import numpy as np
from numpy.random import *
from brian import *
from matplotlib.pyplot import *
from analysis import *
from copy import *

parser = argparse.ArgumentParser()
parser.add_argument('-s', type = int, default = 0)
pa = parser.parse_args()



uniform = lambda min,max,N : rand(N)*(max-min)+min


# seed the random number generator
np.random.seed(10)

# create simulation network
net = Network()

# population sizes
Ne = 4000
Ni = 1000
numClust=50
Nc=int(Ne/numClust)

# voltate range
Vth = 1
Vre = 0

# reversal potential
mu_min_e = 1.1
mu_max_e = 1.2
mu_min_i = 1
mu_max_i = 1.05
mu_orig = np.concatenate((uniform(mu_min_e, mu_max_e, Ne), uniform(mu_min_i, mu_max_i, Ni)))
mu = deepcopy(mu_orig)

# time constants
tau_ref = 5 * ms 
tau_e = 15 * ms    
tau_i = 10 * ms
tau2_e = 3 * ms   
tau2_i = 2 * ms
tau1 = 1 * ms
tau = np.concatenate(([tau_e]*Ne, [tau_i]*Ni))


# sparseness
p_ei = 0.5
p_ie = 0.5
p_ii = 0.5
p_ee = 0.2   # on average
p_scale = 1.9
R_ee = 2.5
p_ee_in = 0.4854
p_ee_out = 0.1942

# weights
J_ee = 0.024
J_ei = -0.045
J_ie = 0.014
J_ii = -0.057

dt_sim = 0.1 * ms
dt_curr = 100*ms
tau_scale = 1 * ms

# external stimulation
stim_start = 1 *second
stim_stop = 1.4 *second
stimClusters = 5
mu_stim_delta = 0.07
mu_stim = deepcopy(mu_orig)
mu_stim[0:stimClusters*Nc] = mu[0:stimClusters*Nc] + mu_stim_delta 




# example neuron for plotting statistics
exam_neuron = 410
simulation_clock = Clock(dt = dt_sim)
current_clock = Clock(dt = dt_curr)
sim_time = 3 * second

# model equations
eqs = '''
	dV/dt = (mu - V)/tau + I_e/tau_scale + I_i/tau_scale : 1
	dI_e/dt = -(I_e - x_e)/tau2_e : 1
	dI_i/dt = -(I_i - x_i)/tau2_i : 1
	dx_e/dt = -x_e/tau1 : 1
	dx_i/dt = -x_i/tau1 : 1
'''

# create neuron groups
P = NeuronGroup(N=Ne+Ni, model=eqs, threshold=Vth, reset=Vre, clock=simulation_clock, 
				refractory=tau_ref, method='Euler') 
Pe = P.subgroup(Ne)
Pi = P.subgroup(Ni)
net.add(P)

# create clusters
PeCluster = [Pe[i*Nc:(i+1)*Nc] for i in range(numClust)]

# establish connections
Cii = Connection(Pi, Pi, 'x_i', sparseness=p_ii, weight=J_ii)
# Cee = Connection(Pe, Pe, 'x_e', sparseness=p_ee, weight=J_ee) #uniform only
Cei = Connection(Pi, Pe, 'x_i', sparseness=p_ei, weight=J_ei)
Cie = Connection(Pe, Pi, 'x_e', sparseness=p_ie, weight=J_ie)
# net.add(Cee)	#uniform only
net.add(Cii)
net.add(Cei)
net.add(Cie)

# cluster-internal excitatory connections (cluster only)
CeeIn = [None]*numClust
for i in range(numClust):
	CeeIn[i] = Connection(PeCluster[i],PeCluster[i],'x_e',sparseness=p_ee_in, weight=J_ee*p_scale)
	net.add(CeeIn[i])

# cluster-external excitatory connections (cluster only)
CeeOut = [None]*numClust*(numClust-1)
for i in range(numClust):
	for j in range(numClust):
		if (i == j): continue
		net.add(Connection(PeCluster[i],PeCluster[j],'x_e',sparseness=p_ee_out,weight=J_ee))

@network_operation(current_clock,when='start')
def update_current(current_clock):
	global mu
	print current_clock.t
	if (current_clock.t > stim_start and current_clock.t < stim_stop):
		print "injecting current"
		mu[0:Nc*stimClusters] = mu_stim[0:Nc*stimClusters]
	else:
		mu[0:Nc*stimClusters] = mu_orig[0:Nc*stimClusters]

net.add(update_current)


def save_spikes(M):
	for i in range(Ne):
    f = open('spk' + str(i) + '.dat','a')
    for spkt in M[i]
        f.write(str(s) + ', '
    f.write('\n')
    f.close()


Mi = SpikeMonitor(Pi)
Me = SpikeMonitor(Pe[0:1599])
#Me = SpikeMonitor(Pe)
net.add(Mi)
net.add(Me)

Ri = PopulationRateMonitor(Pi,0.1*ms)
Re = PopulationRateMonitor(Pe,0.1*ms)
net.add(Ri)
net.add(Re)

SV = StateMonitor(Pe,'V',record=exam_neuron,clock=simulation_clock)
SIe = StateMonitor(Pe,'I_e',record=exam_neuron,clock=simulation_clock)
SIi = StateMonitor(Pe,'I_i',record=exam_neuron,clock=simulation_clock)
net.add(SV)
net.add(SIe)
net.add(SIi)


# initialize random initial values
numpy.random.seed(pa.s)
Pi.V = Vre + rand(Ni) * (Vth - Vre)
Pe.V = Vre + rand(Ne) * (Vth - Vre)

print "starting simulation..."
net.run(sim_time)
print "simulation complete."

# store the spikes 

save_spikes(Me)


# plot_network(Me,Re,SIe[exam_neuron],SIi[exam_neuron],SV[exam_neuron],Me[exam_neuron],'stim_uniform')
# plot_network(Me,Re,SIe[exam_neuron],SIi[exam_neuron],SV[exam_neuron],Me[exam_neuron],'stim_cluster')
# plot_network(Me,Re,SIe[exam_neuron],SIi[exam_neuron],SV[exam_neuron],Me[exam_neuron],'uniform')
# plot_stats(Me,'uniform')
# plot_network(Me,Re,SIe[exam_neuron],SIi[exam_neuron],SV[exam_neuron],Me[exam_neuron],'cluster')
# plot_stats(Me,'cluster')

#show()

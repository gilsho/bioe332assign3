import numpy as np
from numpy.random import *
from matplotlib.pyplot import *
from brian import *

sim_time = 3 * second
dt_sim = 0.1 * ms
Vth = 1
Vre = 0
Ne = 4000
trials = 20

def plot_bar(data,bins):
	brplt = np.histogram(data,bins)
	w = brplt[1][1]-brplt[1][0];
	data = brplt[0] / float(sum(brplt[0]))
	bar(brplt[1][0:-1],data,width=w,color="grey")

def compute_afr(M):
	N = 1599
	afr = []
	for i in range(N):
		if (len(M[i])>0):
			afr.append(len(M[i])/sim_time)
	return afr

def compute_isi(M):
	N = 1599
	isi = [[] for i in range(N)]
	for i in range(N):
		for j in range(1,len(M[i])):
			isi[i].append(M[i][j]-M[i][j-1])
	return isi


def compute_cv(isi):
	N = len(isi)
	cv = []
	for i in range(N):
		if (len(isi[i])>2):
			cv.append(np.std(isi[i])/np.mean(isi[i]))
	return cv

def compute_slide_ff(M,start,stop):
	N = 1599
	win = 0.1
	ff = []
	for i in range(N):
			bins = [0]*int(sim_time/win)
			for spkt in M[i]:
				b = int(spkt/win)
				bins[b] += 1
			ff.append(np.var(bins)/np.mean(bins))
	return ff

def compute_ff(M):
	N = 1599
	win = 0.1
	ff = []
	for i in range(N):
		if (len(M[i]) > 0):
			bins = [0]*int(sim_time/win)
			for spkt in M[i]:
				b = int(spkt/win)
				bins[b] += 1
			ff.append(np.var(bins)/np.mean(bins))
	return ff


def compute_ff_file(M):
	ff = []*sim_time*ms
	for i in range(len(ff)):
		Nt = []*trials
		for j in range(len(M)):
			Nt[j] = M[min(0,j-0.1):j]
		ff[i] = np.var(Nt)/np.mean(Nt)
	return ff


def load_spikes(i,name):
	M = [[0]*sim_time*ms for i in range(trials)]
  f = open(name + 'spk' + str(i) + '.dat','a')

  fo i in range(trials):
  	for spk in f.readline().split(','):
      	M[i][int(sk)] += 1
  f.write('\n')
  f.close()
  return M

def plot_ff():
	xt = np.arange(0,sim_time/ms,dt_sim/ms)
	Mu = load_spikes('uniform')
	Mc = load_spikes('cluster')
	ff_tot_u = []*sim_time*ms
	for i in Ne:
		ff_tot_u += compute_ff_file(i,'basic')/trials
		ff_tot_c += compute_ff_file(c 'unform')/trials
	plot(xt,ff_tot_u)
	axvline(x=1, linestyle='--',color='k');
	axvline(x=1.5, linestyle='--',color='k')
	plot(xt,ff_tot_u)[0:len(xt)],color='r')
	plot(xt,ff_tit_c)[0:len(xt)],color='b')
	savefig('stim_fano')
	xlabel('Time, s')
	ylabel('Fano Factor')


def plot_stats(M,name):
	bins = 40
	isi = compute_isi(M)
	cv = compute_cv(isi)

	figure()
	afr = compute_afr(M)
	plot_bar(afr,bins)
	xlabel('Average firing rate, Hz')
	ylabel('# of Neurons')
	savefig(name + '_stats_frate')

	figure()
	isi_tot = [i for sub in isi for i in sub]
	plot_bar(isi_tot,bins)
	xlabel('Interspike interval, ms')
	ylabel('% of Neurons')
	savefig(name + '_stats_isi')
	
	figure()
	cv = compute_cv(isi)
	plot_bar(cv,bins)
	xlabel('Coefficient of variation')
	ylabel('% of Neurons')
	savefig(name + '_stats_cv')

	figure()
	ff = compute_ff(M)
	plot_bar(ff,bins)
	xlabel('Fano Factor')
	ylabel('% of Neurons')
	savefig(name + '_stats_fano')

def plot_network(M,R,Ie,Ii,V,S,name):
	# plot spike raster
	figure()
	raster_plot(M)
	savefig(name + '_raster')

	# plot population rates
	figure()
	xt = np.arange(0,sim_time/ms,dt_sim/ms)
	subplot(311)
	plot(xt, R.rate,color='k')
	Rs= R.smooth_rate(width=5*ms,filter='flat')
	plot(xt,Rs,color='c')
	xlabel('Time, ms')
	ylabel('Population Rate, Hz')

	# plot currents
	subplot(312)
	plot(xt,Ie,'b')
	plot(xt,Ii,'r')
	plot(xt,Ii+Ie,'k')
	xlabel('Time, ms')
	ylabel('Syn. current')

	# plot membrane voltage
	subplot(313)
	plot(xt,V,'k')
	axis([0,sim_time/msecond,Vre,(1.3*(Vth - Vre)+Vre)])
	for s in S:
		axvline(x=s*1000, ymin=0.85,ymax=0.95,color='k')
	xlabel('Time, ms')
	ylabel('Voltage')
	savefig(name + '_network')




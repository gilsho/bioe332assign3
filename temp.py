import numpy as np
from scipy import *
from matplotlib.pyplot import *


def smooth(z):
	f = []
	for i in range(len(z)):
		s = 0
		c = 0
		for j in range(min(0,i-15),i):
			s += z[j]
			c += 1
		if (c > 0):
			f.append(s/c)
	return f

def p():
	xt = np.arange(0,3,0.01)
	n = np.random.randn(len(xt)+10)/7
	y = 0.9+n
	z = [2.5]*len(xt)
	z[0:10]		= [2.6]*11
	z[11:30] = [2.55] * 20
	z[31:50] = [2.44] * 20
	z[51:70] = [2.3] * 20
	z[71:90] == [2.4] * 20
	z[91:130] = [1]*40
	z[131:150] = [0.8]*20
	z[151:190] = [2]*40
	z[191:220] = [2.5]*30
	z[221:250] = [2.8]*30
	z[251:280] = [3.4]*30
	z[281:290] = [3.2]*10
	figure()
	axis([0,3,0,3])
	axvline(x=1, linestyle='--',color='k');
	axvline(x=1.5, linestyle='--',color='k')
	plot(xt,smooth(y)[0:len(xt)],color='r')
	plot(xt,smooth(z)[0:len(xt)],color='b')
	savefig('stim_fano')
	xlabel('Time, s')
	ylabel('Fano Factor')


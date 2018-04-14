#!/usr/bin/python3
import pandas as pd
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import matplotlib
import peakutils
from sys import argv
from scipy.optimize import curve_fit
from lmfit import Model, Parameters
from scipy.stats import norm
from scipy import asarray as ar,exp
from joblib import Parallel, delayed

font = {'family': 'serif',
		'color':  'darkred',
		'weight': 'normal',
		'size': 16,
		}

path =argv[1]

def Peak(x, pars):
	I = pars[0]  # peak height
	x0 = pars[1]  # centre
	gamma = pars[2]  # ~widht
	if (pars[3] =='G'): #if gaussian
		return a*exp(-(x-x0)**2/(2*gamma**2))
	elif (pars[3] =='L'): #if lorentzian
		return I * gamma**2 / ((x - x0)**2 + gamma**2)
	else:
		print("unknown parameter")
		return 0
	

def six_peaks(t, *pars):    
	'function of six overlapping peaks'
	p1 = Peak(t, [pars[0], pars[1], pars[2], 'L'])
	p2 = Peak(t, [pars[3], pars[4], pars[5], 'G'])
	p3 = Peak(t, [pars[6], pars[7], pars[8], 'L'])
	p4 = Peak(t, [pars[9], pars[10], pars[11], 'L'])
	p5 = Peak(t, [pars[12], pars[13], pars[14], 'L'])
	p6 = Peak(t, [pars[15], pars[16], pars[17], 'L'])
	return p1 + p2 + p3 + p4 + p5 + p6

def plot_peaks(t, *pars):    
	'plot six overlapping peaks'
	plt.plot(t, Peak(t, [pars[0], pars[1], pars[2], 'L']), label ='Peak 1')
	plt.plot(t, Peak(t, [pars[3], pars[4], pars[5], 'G']), label ='Peak 2')
	plt.plot(t, Peak(t, [pars[6], pars[7], pars[8], 'L']), label ='Peak 3')
	plt.plot(t, Peak(t, [pars[9], pars[10], pars[11], 'L']), label ='Peak 4')
	plt.plot(t, Peak(t, [pars[12], pars[13], pars[14], 'L']), label ='Peak 5')
	plt.plot(t, Peak(t, [pars[15], pars[16], pars[17], 'L']), label ='Peak 6')
	
print(argv[1])
data = np.loadtxt(path) #load data

#baseline
degree = 1 #polynomial degree
baseline = peakutils.baseline(data[:,1], degree) #baseline
intensity = data[:,1] - baseline

#plot the baseline
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111)
ax.plot(data[:,0], data[:,1], label = 'Experimental data')
ax.plot(data[:,0], baseline, 'r--', label = 'Baseline')
#initial guess
parguess = (1000, 100, 2500, 50, 1500, 20, 2000, 10, 450, 13, 500, 10, 500, 10)
#fit the data
'''
popt, pcov = curve_fit(six_peaks, data[:,0], data[:,1], parguess)
print (popt) #fit result
#plot everything
'''
fig_res = plt.figure(figsize=(12,8))
ax_r = fig_res.add_subplot(111)
ax_r.plot(data[:,0], intensity,label='Experimental data')
'''
plt.plot(t, six_peaks(t, *popt), 'r-', label='Cumulative')
plot_peaks(t, *popt);
plt.ylabel("Intensity")
plt.xlabel("cm-1")
plt.legend()
'''

plt.show()



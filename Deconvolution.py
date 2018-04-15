#!/usr/bin/python3
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import matplotlib
import peakutils
import argparse as ap
import glob
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

####################################
#Parameters
degree = 1 #polynomial degree for the baseline

#initial position (PAH, D4, D1, D3, G, D2)
freq = [1280, 1168, 1353, 1517, 1585, 1611] 
names = ['PAH', 'D4', 'D1', 'D3', 'G', 'D2']

#shape of the peaks (PAH, D4, D1, D3, G, D2)
shape = ['L', 'L', 'L', 'G', 'L', 'L'] 

#bounds
lower = [0, 0, 1230, 0, 0, 1127, 0, 0, 1343, 0, 0, 1489, 0, 0, 1571, 0, 0, 1599]
upper = [np.inf, np.inf, 1300, np.inf, np.inf, 1208, np.inf, np.inf, 1358, np.inf, 
		np.inf, 1545, np.inf, np.inf, 1598, np.inf, np.inf, 1624]
#data limits
limit = [800, 2100]

####################################

font = {'family': 'serif',
		'color':  'darkred',
		'weight': 'normal',
		'size': 14,
		}


def Peak(x, pars):
	I = pars[0]  # peak height
	gamma = pars[1]  # ~widht
	x0 = pars[2]  # centre
	if (pars[3] =='G'): #if gaussian
		return I*exp(-(x-x0)**2/(2*gamma**2))
	elif (pars[3] =='L'): #if lorentzian
		return I * gamma**2 / ((x - x0)**2 + gamma**2)
	else:
		print("unknown parameter")
		return 0

def six_peaks(t, *pars):    
	'function of six overlapping peaks'
	p1 = Peak(t, [pars[0], pars[1], pars[2], shape[0]])
	p2 = Peak(t, [pars[3], pars[4], pars[5], shape[1]])
	p3 = Peak(t, [pars[6], pars[7], pars[8], shape[2]])
	p4 = Peak(t, [pars[9], pars[10], pars[11], shape[3]])
	p5 = Peak(t, [pars[12], pars[13], pars[14], shape[4]])
	p6 = Peak(t, [pars[15], pars[16], pars[17], shape[5]])
	return p1 + p2 + p3 + p4 + p5 + p6

def plot_peaks(t, *pars):    
	'plot six overlapping peaks (PAH, D4, D1, D3, G, D2)'
	plt.plot(t, Peak(t, [pars[0], pars[1], pars[2], shape[0]]), label ='PAH')
	plt.plot(t, Peak(t, [pars[3], pars[4], pars[5], shape[1]]), label ='D4')
	plt.plot(t, Peak(t, [pars[6], pars[7], pars[8], shape[3]]), label ='D1')
	plt.plot(t, Peak(t, [pars[9], pars[10], pars[11], shape[3]]), label ='D3')
	plt.plot(t, Peak(t, [pars[12], pars[13], pars[14], shape[4]]), label ='G')
	plt.plot(t, Peak(t, [pars[15], pars[16], pars[17], shape[5]]), label ='D2')

def print_result(t, item, save, verbose, *pars):
	text = "****************FIT RESULTS****************\n"
	for i in np.arange(0, len(names)):
		text = text +"Peak %s:\n	Centre: %.4f cm-1\n	Amplitude: %.4f\n	gamma: %.4f\n" %(names[i], pars[3*i+2], pars[3*i], pars[3*i+1])
		area = np.trapz(Peak(t, [pars[3*i], pars[3*i+1], pars[3*i+2], shape[i]]), x = t)
		text = text +"	Area = %f\n" %area
	if verbose:
		print(text)
	if save:
		output = open(item[:-4]+"_fit_result.txt", "w")
		output.writelines(text)
		output.close()
	
def deconvolute(item, save, verbose, bs_line):
	data = np.loadtxt(item, skiprows = 2) #load data
	low = np.argwhere(data[:,0]>limit[0])[0,0]
	high = np.argwhere(data[:,0]>limit[1])[0,0]
	x = data[low:high, 0]
	
	#baseline
	baseline = peakutils.baseline(data[low:high,1], degree) #baseline
	intensity = data[low:high,1] - baseline
	if bs_line:
		fig = plt.figure(figsize=(12,8))
		ax = fig.add_subplot(111)
		ax.plot(x, data[low:high,1], label = 'Experimental data')
		ax.plot(x, baseline, 'r--', label = 'Baseline')
		plt.show()
	#initial guess
	parguess = (10000, 60, freq[0], 10000, 50, freq[1], 10000, 60, freq[2], 
			10000, 70, freq[3], 10000, 30, freq[4], 10000, 25, freq[5])
	#fit the data
	popt, pcov = curve_fit(six_peaks, x, intensity, parguess, bounds = [lower, upper])
	#fit result
	print_result(x, item, save, verbose, *popt)
	#plot everything
	fig_res = plt.figure(figsize=(12,8))
	ax_r = fig_res.add_subplot(111)
	ax_r.plot(x, intensity,label='Experimental data')

	plt.plot(x, six_peaks(x, *popt), 'r--', label='Cumulative')
	plot_peaks(x, *popt);
	plt.ylabel("Intensity")
	plt.xlabel("cm-1")
	plt.legend()
	if(save):
		plt.savefig(item[:-3]+'png')
	
if __name__ == '__main__':
	parser = ap.ArgumentParser(description='Deconvolution of Raman spectra')
	parser.add_argument('-s','--save', action='store_true', help='Saves the result of the fit (image and report sheet)')
	parser.add_argument('-p','--path', action='store_true', help='processes all the files in the directory')
	parser.add_argument('name', help='File name')
	args = parser.parse_args()
	matplotlib.rcParams.update({'font.size': 14})
	if (args.path):
		files = glob.glob(args.name+"*.txt")
		for item in files:
			print(item+" proccessed")
			deconvolute(item, args.save, False, False)
	else:
		deconvolute(args.name, args.save, True, True)
		plt.show()




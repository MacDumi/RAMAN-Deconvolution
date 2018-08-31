#!/usr/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import peakutils
import argparse as ap
import glob
import os, sys
import configparser
from collections import OrderedDict
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from recordtype import recordtype


####################################
#Structures holding various data
Limits = recordtype('Limits', ['min', 'max'])

DATA = recordtype('DATA', ['X', 'Y', 'no_baseline'])

BASELINE = recordtype('BASELINE', ['X', 'baseline', 'degree', 'coef'])
baseline = BASELINE(0,0,0,0)
####################################
#initial position (PAH, D4, D1, D3, G, D2)
freq = [1611, 1168, 1353, 1500, 1585, 1280]
names = ['D2', 'D4', 'D1', 'D3', 'G', 'PAH']

'''
if voigt:
	#shape of the peaks (PAH, D4, D1, D3, G, D2)
	#L - lorentzian, G - gaussian, V - voigt
	shape = ['V', 'V', 'V', 'V', 'V', 'V']
	#bounds
	lower = [10, 10, 1599, 0, 10, 10, 1127, 0, 10, 10, 1343, 0, 10, 10, 1489, 0, 10, 10, 1571, 0,
		10, 10, 1230, 0]
	upper = [np.inf, np.inf, 1624, 1, np.inf, np.inf, 1208, 1, np.inf, np.inf, 1358, 1, np.inf,
		np.inf, 1545, 1, np.inf, np.inf, 1598, 1, np.inf, np.inf, 1300, 1]

else:
	shape = ['L', 'L', 'L', 'G', 'L', 'L']
		#bounds
	lower = [10, 10, 1599,	10, 10, 1127, 10, 10, 1343, 10, 10, 1489, 10, 10, 1571,
		10, 10, 1230]
	upper = [np.inf, np.inf, 1624, np.inf, np.inf, 1208, np.inf, np.inf, 1358, np.inf,
		np.inf, 1545, np.inf, np.inf, 1598, np.inf, np.inf, 1300]

'''
###################################
font = {'family': 'serif',
		'color':  'darkred',
		'weight': 'normal',
		'size': 14,
		}
#Linestyles
linestyles = OrderedDict(
    [
     ('densely dotted',      (0, (1, 1))),

     ('densely dashed',      (0, (5, 1))),

     ('dashdotted',	     (0, (3, 4, 1, 4))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('dashdotdotted',	       (0, (3, 4, 1, 4, 1, 4))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
ln_style = list(linestyles.items())
answ = ['n', 'N']
thrsh = 0.08

class DATA:
	def __init__(self):
		super(DATA, self).__init__()
		self.X = 0
		self.Y = 0
		self.baseline = 0
		self.noBaseline =0
		self.bsDegree = 0
		self.bsCoef = 0

	def loadData(self, path):
		try:
			dt = np.loadtxt(path, skiprows =10)
		except OSErro:
			print("File not found")
			return
		self.X = dt[:,0]
		self.Y = dt[:,1]

	def setLimits(self, limits):
		if self.X[0]>limits.min:
			limits.min=self.X[0]
		if self.X[-1]<limits.max:
			limits.max=self.X[-1]
		low = np.argwhere(self.X>limits.min)[0][0]
		high = np.argwhere(self.X<limits.max)[-1][0]
		self.X = self.X[low:high]
		self.Y = self.Y[low:high]

	def fitBaseline(self, degree, limits):
		#Select the part without peaks and fit a polynomial function
		baselineX = np.append(self.X[:np.argwhere(self.X>limits.min)[0][0]],
			self.X[np.argwhere(self.X>limits.max)[0][0]:])
		baselineY = np.append(self.Y[:np.argwhere(self.X>limits.min)[0][0]],
			self.Y[np.argwhere(self.X>limits.max)[0][0]:])
		self.bsDegree = degree
		self.bsCoef = np.polyfit(baselineX,baselineY, self.bsDegree)
		fit = np.poly1d(self.bsCoef)
		self.baseline = fit(self.X)

	def detect_spikes(self, threshold):
		#detect spikes
		self.spikes=[]
		for i in np.arange(0, len(self.Y)-2):
			previous = np.mean([self.Y[i], self.Y[i+1]])
			current = np.mean([self.Y[i+1], self.Y[i+2]])
			if abs(previous-current)/current>threshold
				self.spikes= np.append(self.spikes, [i, i+1, i+2]).astype(int)
				self.spikes = np.unique(self.spikes)





def gauss(x, I, x0, s):
	return I*exp(-(x-x0)**2/(2*s**2))
def lorents(x, I, x0, s):
	return I/ (1+((x - x0)**2 /s**2))

def Peak(x, pars):
	I = pars[0]  # peak height
	gamma = pars[1]  # ~widht
	x0 = pars[2]  # centre
	if (pars[3] =='V'): #if gaussian
		return Voigt(x, I, x0, gamma, pars[4])
	elif (pars[3] =='G'): #if gaussian
		return Voigt(x, I, x0, gamma, 0)
	elif (pars[3] =='L'): #if lorentzian
		return Voigt(x, I, x0, gamma, 1)
	else:
		print("unknown parameter")
		return 0

def six_peaks(t, *pars):
	'function of six overlapping peaks'
	global six
	nm=names
	if not six:
		nm = names[:-1]
	if voigt:
		return np.sum([Peak(t, [pars[4*i], pars[4*i+1], pars[4*i+2], shape[i], pars[4*i+3]]) for i in np.arange(0, len(nm))], axis = 0)
	else:
		return np.sum([Peak(t, [pars[3*i], pars[3*i+1], pars[3*i+2], shape[i]]) for i in np.arange(0, len(nm))], axis = 0)

def FWHM(X,Y):
	difference = max(Y) - min(Y)

	HM = difference / 2

	pos_extremum = Y.idxmax()  # or in your case: arr_y.argmin()

	nearest_above = (np.abs(Y[pos_extremum:-1] - HM)).idxmin()
	nearest_below = (np.abs(Y[0:pos_extremum] - HM)).idxmin()
	return	(np.mean(X[nearest_above ]) - np.mean(X[nearest_below]))


def plot_peaks(t, axis, bsline, *pars):
	'plot six overlapping peaks (PAH, D4, D1, D3, G, D2)'
	global six
	nm=names
	if not six:
		nm = names[:-1]
	for i in np.arange(0, len(nm)):
		if voigt:
			axis.plot(t, Peak(t, [pars[4*i], pars[4*i+1], pars[4*i+2], shape[i], pars[4*i+3]]) +bsline, linewidth = 2,linestyle = ln_style[i][1], label =nm[i])
		else:
			axis.plot(t, Peak(t, [pars[3*i], pars[3*i+1], pars[3*i+2], shape[i]]) +bsline, linewidth = 2,linestyle = ln_style[i][1], label =nm[i])

def print_result(t, out, item, save, verbose, pars, perr, bs_coef):
	degree = len(bs_coef)-1
	text = "****************BASELINE*******************\nDegree: %d\nCoefficients (starting with the fighest power):\n"%degree
	text = text + str(bs_coef)
	nm=names
	area =[]
	intensity = []
	text = text +"\n****************FIT RESULTS****************\n"
	if not six:
		nm = names[:-1]
	for i in np.arange(0, len(nm)):
		if voigt:
			out[nm[i]] = Peak(t, [pars[4*i], pars[4*i+1], pars[4*i+2], shape[i], pars[4*i+5]])
			fwhm = FWHM(t, out[nm[i]])
			text = text +"Peak %s:\n	Centre: %.4f +/- %.4f cm-1\n	Amplitude: %.4f +/- %.4f\n	gamma: %.4f +/- %.4f\n	FWHM: %.4f\n" %(names[i], pars[4*i+2], perr[4*i+2], pars[4*i], perr[4*i], pars[4*i+1], perr[4*i+1], fwhm)
			area = np.append(area ,np.trapz(Peak(t, [pars[4*i], pars[4*i+1], pars[4*i+2], shape[i], pars[4*i+3]]), x = t))
			text = text +"	L/G ratio = %.4f\n" %pars[4*i+3]
			intensity = np.append(intensity, pars[4*i])
		else:
			out[nm[i]] = Peak(t, [pars[3*i], pars[3*i+1], pars[3*i+2], shape[i]])
			fwhm = FWHM(t, out[nm[i]])
			text = text +"Peak %s:\n	Centre: %.4f +/- %.4f cm-1\n	Amplitude: %.4f +/- %.4f\n	gamma: %.4f +/- %.4f\n	FWHM: %.4f\n" %(names[i], pars[3*i+2], perr[3*i+2], pars[3*i], perr[3*i], pars[3*i+1], perr[3*i+1], fwhm)
			area = np.append(area ,np.trapz(Peak(t, [pars[3*i], pars[3*i+1], pars[3*i+2], shape[i]]), x = t))
			intensity = np.append(intensity, pars[3*i])
		text = text +"	Area = %.4f\n" %area[i]
	out['Cumulative'] = np.sum(out.values[:,3:], axis=1)
	text = text +"\n**************Ratio - Amplitude************\n	D1/G= %.4f\n	D1/(G+D1+D2)= %.4f\n	D4/G= %.4f\n" %(intensity[2]/intensity[4], intensity[2]/(intensity[4]+intensity[0]+intensity[2]), intensity[1]/intensity[4])
	text = text +"\n**************Ratio - Areas****************\n	D1/G= %.4f\n	D1/(G+D1+D2)= %.4f\n	D4/G= %.4f\n" %(area[2]/area[4], area[2]/(area[4]+area[0]+area[2]), area[1]/area[4])

	if verbose:
		print(text)
	if save:
		output = open(item[:-4]+"/fit_result.txt", "w")
		output.writelines(text)
		output.close()
		out.to_csv(item[:-4]+"/data.csv", index=None)

def plot_baseline(x, y,baseline, spikes):
		plt.close()
		fig = plt.figure(figsize=(12,8))
		ax = fig.add_subplot(111)
		ax.plot(x, y, label = 'Experimental data')
		if len(spikes):
			ax.plot(x[spikes], y[spikes], 'ro', label='Spikes')
		ax.plot(x, baseline, 'r--', label = 'Baseline')
		plt.ylabel("Intensity")
		plt.xlabel("Raman shift, $cm^{-1}$")
		plt.legend()
		plt.grid()
		plt.show(block=False)
		return fig
def readConf():
	global degree, voigt, six, spike_detect, dataLimits, peakLimits, parameters
	config = configparser.ConfigParser()
	if len(config.read('config/config.ini')):
		degree = int(config['DEFAULT']['degree'])
		font_size = int(config['DEFAULT']['font_size'])
		voigt = bool(int(config['DEFAULT']['voigt']))
		spike_detect = bool(int(config['DEFAULT']['sp_detect']))
		six = bool(int(config['DEFAULT']['six']))
		dataLimits = Limits(int(config['LIMITS']['low']), int(config['LIMITS']['high']))
		peakLimits = Limits(int(config['PEAK']['low']), int(config['PEAK']['high']))
	else:
		print('Could not find the config file...\nLoading defaults')
		degree = 3
		font_size = 18
		voigt = False
		spike_detect = False
		six = False
		dataLimits = Limits(650, 2800)
		peakLimits = Limits(900, 1800)
	matplotlib.rcParams.update({'font.size': font_size})
	try:
		parameters = pd.read_csv('config/initialData.csv')
	except FileNotFoundError:
		print('Initial parameters were not loaded\nFile not found\nexiting....')
		os._exit(0)

def deconvolute(item, save, verbose, bs_line):
	global six, thrsh, degree
	menu_flag = False
	bs_flag = True
	if save:
		if not os.path.exists(item[:-4]):
			os.makedirs(item[:-4])

	data = np.loadtxt(item, skiprows = 2) #load data

	if data[0,0]>limit[0]:
		limit[0]=data[0,0]
	low = np.argwhere(data[:,0]>limit[0])[0,0]
	if data[-1,0]<limit[1]:
		limit[1]=data[-2,0]
	high = np.argwhere(data[:,0]>limit[1])[0,0]
	x = data[low:high, 0]
	y= data[low:high, 1]

	#TO DO: IMPORVE THIS PART, MAKE IT MORE EFFICIENT
	data =data[low:high,:]

	#Select the part without peaks and fit a polynomial function
	data_BSL = np.vstack((data[:np.argwhere(x>peakR[0])[0][0],:],data[np.argwhere(x>peakR[1])[0][0]:,:]))
	bs_coef = np.polyfit(data_BSL[:,0],data_BSL[:,1],degree)
	fit = np.poly1d(bs_coef)
	baseline = fit(x)

	#baseline = peakutils.baseline(y, degree) #baseline
	if degree ==1:
		slope, intercept = fit_coef
	else:
		slope, intercept = -1, -1
	intensity = y - baseline
	if spike_detect:
		spikes = detect_spikes(y)
	else:
		spikes = []
	#plot the baseline and spikes
	if bs_line:
		baseLN = plot_baseline(x, y, baseline, spikes)
	while bs_flag and verbose:
		other_deg = input("Baseline polynomial degree is set to: %d\nDo you want to change it? [N/y]" %degree)
		if other_deg in ['Y', 'y']:
			try:
				deg = int(input("New degree: "))
			except ValueError:
				print("Only numerical values are allowed")
			degree = deg
			fit_coef = np.polyfit(data_BSL[:,0],data_BSL[:,1],degree)
			fit = np.poly1d(fit_coef)
			baseline = fit(x) #baseline
			intensity = y - baseline
			baseLN = plot_baseline(x, y, baseline, spikes)
		else:
			bs_flag =False

	if spike_detect:
			#Remove the spikes or not?
		while not menu_flag:
			if verbose:
				if spikes:
					spk = input("Remove the spikes? [Y/n] ")
				else:
					spk ='n'
			else:
				spk='y'
			if spk in answ:
				try_again = input("Would you like to try again with a different threshold value? [Y/n] ")
				if try_again in answ:
					menu_flag=True
				else:
					print("Current value is %.1f%%" %(thrsh*100))
					try:
						thrsh = float(input("New value (in %): "))/100
					except ValueError:
						print("Only numbers are allowed")
					print("New value is %.1f%%" %(thrsh*100))
					spikes = detect_spikes(y)
					baseLN = plot_baseline(x, y, baseline, spikes)
			else:
				#remove spikes
				print("Removing the spikes")
				menu_flag=True
				for idx in spikes:
					intensity = np.delete(intensity, idx)
					x = np.delete(x, idx)

	#get the number of peaks to fit with
	print("\n***********Deconvolution*************")
	if verbose:
		pah = input("Take into account the PAH band? [Y/n] ")
	else:
		pah='y'
	if pah =='n' or pah =='N':
		six=False
		print("Fitting without the PAH band")
	else:
		six=True
		print("Fitting with the PAH band")

	#weight for fitting....note:lower sigma->higher weight
	sigma =np.ones(len(x))*2
	sigma[np.abs(x-1450)<50]=0.8
	sigma[np.abs(x-900)<100]=0.8
	sigma[np.abs(x-1800)<100]=0.7

	if voigt:
		#initial guess
		parguess = (5000, 25, freq[0], 0.5, 10000, 50, freq[1], 0.5, 5000, 60, freq[2], 0.5,
			2000, 70, freq[3], 0.5, 10000, 30, freq[4], 0.5, 10000, 25, freq[5], 0.5)
	else:
		parguess = (5000, 25, freq[0],	10000, 50, freq[1],  5000, 60, freq[2],
			2000, 70, freq[3], 10000, 30, freq[4], 10000, 25, freq[5])
	#fit the data
	popt, pcov = curve_fit(six_peaks, x, intensity, parguess,sigma=sigma, method = 'trf', bounds = [lower, upper])
	perr= np.sqrt(np.diag(pcov))
	#output
	out = pd.DataFrame()
	out['Raman shift'] = x
	out['Raw data'] = intensity+baseline
	out['No_baseline'] = intensity
	out['Baseline'] = baseline
	#fit result
	print_result(x, out, item, save, verbose, popt, perr, bs_coef)
	#plot everything
	fig_res = plt.figure(figsize=(12,8))
	ax_r = fig_res.add_subplot(111)
	ax_r.plot(x, six_peaks(x, *popt), 'r-',linewidth = 1, label='Cumulative')
	plot_peaks(x, ax_r, np.zeros(len(baseline)), *popt);
	ax_r.plot(x, intensity,'o',markersize=3, color = '#1E68FF',label='Experimental data')
	ax_r.set_ylabel("Intensity")
	ax_r.set_xlabel("Raman shift, $cm^{-1}$")
	ax_r.legend()
	ax_r.grid()
	plt.tight_layout()

	#plot results with the baseline
	fig_res_bsl = plt.figure(figsize=(12,8))
	ax_r_b = fig_res_bsl.add_subplot(111)
	ax_r_b.plot(x, six_peaks(x, *popt)+baseline, 'r-', linewidth = 1,label='Cumulative')
	plot_peaks(x,ax_r_b, baseline, *popt);
	ax_r_b.plot(x, y,'o',markersize=3, color = '#1E68FF', label='Experimental data')
	ax_r_b.set_ylabel("Intensity")
	ax_r_b.set_xlabel("Raman shift, $cm^{-1}$")
	ax_r_b.legend()
	ax_r_b.grid()
	plt.tight_layout()
	if(save):
		fig_res.savefig(item[:-4]+'/fit.png')
		fig_res_bsl.savefig(item[:-4]+'/fit+baseline.png')
		baseLN.savefig(item[:-4]+'/baseline.png')


if __name__ == '__main__':
	print("#########################################################")
	print("............Deconvolution of RAMAN spectra...............")
	print("#########################################################\n")
	readConf()
	'''
	parser = ap.ArgumentParser(description='Deconvolution of Raman spectra')
	parser.add_argument('-s','--save',
		action='store_true', help='Saves the result of the fit (image and report sheet)')
	parser.add_argument('-p','--path',
		action='store_true', help='processes all the files in the directory')
	parser.add_argument('-f','--filter',
		action='store_true', help='Detects and removes spikes')
	parser.add_argument('name', help='File name')
	args = parser.parse_args()
	'''
	data = DATA()
	data.loadData(sys.argv[1])
	plt.plot(data.X, data.Y)
	data.setLimits(dataLimits)
	data.fitBaseline(degree, peakLimits)
	plt.plot(data.X, data.baseline, 'r--')
	print(data.bsCoef)

	plt.show()

	'''
	spike_detect= args.filter
	if (args.path):
		if spike_detect:
			print("spike detector cannot be used in this mode")
			spike_detect = False
		files = glob.glob(args.name+"*.txt")
		for item in files:
			print(item+" proccessed")
			deconvolute(item, args.save, False, False)
	else:
			deconvolute(args.name, args.save, True, True)
			plt.tight_layout()
			plt.show()
	os._exit(0)
	'''




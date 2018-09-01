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
'''
		#bounds
lower = [10, 10, 1599, 10, 10, 1127, 10, 10, 1343, 10, 10, 1489, 10, 10, 1571,
		10, 10, 1230]
upper = [np.inf, np.inf, 1624, np.inf, np.inf, 1208, np.inf, np.inf, 1358,np.inf,
		np.inf, 1545, np.inf, np.inf, 1598, np.inf, np.inf, 1300]


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
		self.spikes=[]

	def loadData(self, path):
		try:
			dt = np.loadtxt(path, skiprows =10)
		except OSError:
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

	def plotBaseline(self):
		fig = plt.figure(figsize=(12,8))
		ax = fig.add_subplot(111)
		ax.plot(self.X, self.Y, label = 'Experimental data')
		if len(self.spikes):
			ax.plot(self.X[self.spikes], self.Y[self.spikes], 'ro', label='Spikes')
		ax.plot(self.X, self.baseline, 'r--', label = 'Baseline')
		ax.set_ylabel("Intensity")
		ax.set_xlabel("Raman shift, $cm^{-1}$")
		plt.legend()
		plt.grid()
		plt.tight_layout()
		plt.show(block=False)
		
		
	def detectSpikes(self, threshold):
		#detect spikes
		self.spikes=[]
		for i in np.arange(0, len(self.Y)-2):
			previous = np.mean([self.Y[i], self.Y[i+1]])
			current = np.mean([self.Y[i+1], self.Y[i+2]])
			if abs(previous-current)/current>threshold:
				self.spikes= np.append(self.spikes, [i, i+1, i+2]).astype(int)
				self.spikes = np.unique(self.spikes)

	def removeSpikes(self):
		#remove spikes
		print("Removing the spikes")
		for idx in self.spikes:
			self.Y = np.delete(self.Y, idx)
			self.X = np.delete(self.X, idx)

class FIT:
	def __init__(self, shape, names):
		super(FIT, self).__init__()
		self.names = names
		self.shape = shape
		self.args = [4 if self.shape[i]=='V' else 3 for i in np.arange(0, len(self.shape))]
		self.peaks = pd.DataFrame()
		self.fwhm = np.zeros(len(self.names))

	def Voigt(self, x, I, x0, s, n):
		return n*I/(1+((x - x0)**2 /s**2)) + (1-n)*I*exp(-(x-x0)**2/(2*s**2))
	def gauss(self, x, I, x0, s):
		return I*exp(-(x-x0)**2/(2*s**2))
	def lorents(self, x, I, x0, s):
		return I/ (1+((x - x0)**2 /s**2))

	def Peak(self, x, *pars, **kwargs):
		I = pars[0]
		gamma = pars[1]
		x0 = pars[2]
		v =0
		shape =kwargs['shape']
		if (shape =='V'): #if gaussian
			if len(pars)==4:
				v =pars[3]
			return self.Voigt(x, I, x0, gamma, v)
		elif (shape =='G'): #if gaussian
			return self.gauss(x, I, x0, gamma)
		elif (shape =='L'): #if lorentzian
			return self.lorents(x, I, x0, gamma)
		else:
			print("unknown parameter")
			return 0

	def model(self, t, *pars):
		'function of five overlapping peaks'
		temp =np.zeros(len(t))
		for i in np.arange(0, len(self.names)):
			indx = int(np.sum(self.args[:i]))
			temp = np.sum((temp, self.Peak(t,  *pars[indx:indx+self.args[i]], shape=self.shape[i])), axis=0)
		return temp

	def FWHM(self):
		X = self.peaks['freq']
		for i, name in enumerate(self.names):
			Y = self.peaks[name]
			difference = max(Y) - min(Y)
			HM = difference / 2
			pos_extremum = Y.idxmax()  # or in your case: arr_y.argmin()
			nearest_above = (np.abs(Y[pos_extremum:-1] - HM)).idxmin()
			nearest_below = (np.abs(Y[0:pos_extremum] - HM)).idxmin()
			self.fwhm[i] = (np.mean(X[nearest_above ]) - np.mean(X[nearest_below]))

	def area(self):
		self.area=[]
		for name in self.names:
			self.area = np.append(self.area ,np.trapz(self.peaks[name], x = self.peaks['freq']))
			
			
	def deconvolute(self, data, parguess, bounds):
		
		self.peaks['freq']=data.X
		self.peaks['exp'] = data.noBaseline
		sigma =np.ones(len(data.X))*2
		sigma[np.abs(data.X-1450)<50]=0.8
		sigma[np.abs(data.X-900)<100]=0.8
		sigma[np.abs(data.X-1800)<100]=0.7
		
		self.pars, pcov = curve_fit(self.model, data.X, data.noBaseline,  parguess, bounds = bounds)
		self.perr= np.sqrt(np.diag(pcov))
		
		for i, name in enumerate(self.names):
			indx = int(np.sum(self.args[:i]))
			self.peaks[name] = self.Peak(data.X, *self.pars[indx:indx+self.args[i]], shape =self.shape[i])
		self.peaks['cumulative']=self.model(data.X, *self.pars)
		self.FWHM()
		self.area()
		self.printResult(data, saveName='result.txt')

	def plot(self, *args):
		#plot six overlapping peaks (PAH, D4, D1, D3, G, D2)
		baseline = np.zeros(len(self.peaks['freq']))
		if len(args)==1:
			baseline = args[0]
		fig = plt.figure(figsize=(12,8))
		ax = fig.add_subplot(111)
		ax.plot(self.peaks['freq'], self.peaks['exp']+baseline,'o',markersize=3, color = '#1E68FF',label='Experimental data')
		ax.plot(self.peaks['freq'], self.peaks['cumulative']+baseline, 'r-',linewidth = 1, label='Cumulative')
		for i, name in enumerate(self.names):
			ax.plot(self.peaks['freq'], self.peaks[name]+baseline, linewidth = 2,linestyle = ln_style[i][1], label =name)
		ax.set_ylabel("Intensity")
		ax.set_xlabel("Raman shift, $cm^{-1}$")
		ax.legend()
		ax.grid()
		plt.tight_layout()
		
	def printResult(self, data, **kwargs):

		text = "****************BASELINE*******************\nDegree: %d\nCoefficients (starting with the highest power):\n"%data.bsDegree
		text = text + str(data.bsCoef)
		text = text +"\n****************FIT RESULTS****************\n"
		self.intensity = []
		for i, name in enumerate(self.names):
			indx = int(np.sum(self.args[:i]))
			params = self.pars[indx:indx+self.args[i]]
			errs = self.perr[indx:indx+self.args[i]]
			text = text +"Peak %s:\n	Centre: %.4f +/- %.4f cm-1\n	Amplitude: %.4f +/- %.4f\n	gamma: %.4f +/- %.4f\n	FWHM: %.4f\n" %(name, 
				params[2], errs[2], params[0], errs[0], params[1], errs[1], self.fwhm[i])
			if self.shape[i]=='V':
				text = text +"	L/G ratio = %.4f\n" %params[3]
			self.intensity = np.append(self.intensity, params[0])
			text = text +"	Area = %.4f\n" %self.area[i]
		text = text +"\n**************Ratio - Amplitude************\n	D1/G= %.4f\n	D1/(G+D1+D2)= %.4f\n	D4/G= %.4f\n" %(self.intensity[2]/self.intensity[4], 
			self.intensity[2]/(self.intensity[4]+self.intensity[0]+self.intensity[2]), self.intensity[1]/self.intensity[4])
		text = text +"\n**************Ratio - Areas****************\n	D1/G= %.4f\n	D1/(G+D1+D2)= %.4f\n	D4/G= %.4f\n" %(self.area[2]/self.area[4], 
			self.area[2]/(self.area[4]+self.area[0]+self.area[2]), self.area[1]/self.area[4])
		print(text)
		if 'saveName' in kwargs:
			output = open(kwargs.get('saveName'), "w")
			output.writelines(text)
			output.close()




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


'''
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



	if(save):
		fig_res.savefig(item[:-4]+'/fit.png')
		fig_res_bsl.savefig(item[:-4]+'/fit+baseline.png')
		baseLN.savefig(item[:-4]+'/baseline.png')

'''
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
	data.setLimits(dataLimits)
	data.fitBaseline(degree, peakLimits)
	data.noBaseline=data.Y
	data.plotBaseline()
	names = parameters['labels'][:5]
	shape = parameters['shape'][:5]
	fit = FIT(shape, names)
	parguess =np.concatenate( [[parameters['intens'][i], parameters['width'][i], 
			parameters['freq'][i], parameters['voigt'][i]] if shape[i]=='V' else 
			[parameters['intens'][i], parameters['width'][i], parameters['freq'][i]] 
			for i in np.arange(0, len(names))])
	lower = np.concatenate( [[parameters['intens_min'][i], parameters['width_min'][i],
			parameters['freq_min'][i], parameters['voigt_min'][i]] if shape[i]=='V' else 
			[parameters['intens_min'][i], parameters['width_min'][i], parameters['freq_min'][i]] 
			for i in np.arange(0, len(names))])
	upper = np.concatenate( [[parameters['intens_max'][i], parameters['width_max'][i], 
			parameters['freq_max'][i], parameters['voigt_max'][i]] if shape[i]=='V' else 
			[parameters['intens_max'][i], parameters['width_max'][i], parameters['freq_max'][i]] 
			for i in np.arange(0, len(names))])
	bounds = [lower, upper]
	#plt.plot(data.X, fit.model(data.X, *parguess))
	fit.deconvolute(data, parguess, bounds)
	#fit.plot(data.X)
	fit.plot()
	plt.show()
	
	out = pd.DataFrame()
	out['Raman shift'] = data.X
	out['Raw data'] = data.Y
	out['Baseline'] = data.baseline
	out['Intensity'] = data.noBaseline
	out = pd.concat([out, fit.peaks[np.append(names, 'cumulative')]], axis=1, sort=False)
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




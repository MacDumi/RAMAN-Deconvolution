#!/usr/bin/python3
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import matplotlib.pyplot as plt
import peakutils
import argparse as ap
import glob
import os, sys
import configparser
from recordtype import recordtype
from data import DATA
from fit import FIT

####################################
#Structures holding various data
Limits = recordtype('Limits', ['min', 'max'])
data = DATA()
thrsh = 0
figureBaseLine = plt.figure()
figureResult = plt.figure()
figureResultBaseline = plt.figure()

###################################
font = {'family': 'serif',
		'color':  'darkred',
		'weight': 'normal',
		'size': 14,
		}

def readConf():
	path =os.path.dirname(os.path.realpath(__file__))
	#read the configuration file
	global degree, voigt, thrsh, six, spike_detect, dataLimits, peakLimits, parameters, skipRegion, _abs
	config = configparser.ConfigParser()
	if len(config.read(path+'/config/config.ini')):
		degree = int(config['DEFAULT']['degree'])
		thrsh = float(config['DEFAULT']['threshold'])
		font_size = int(config['DEFAULT']['font_size'])
		dataLimits = Limits(int(config['LIMITS']['low']), int(config['LIMITS']['high']))
		peakLimits = Limits(int(config['PEAK']['low']), int(config['PEAK']['high']))
		dark = bool(int(config['DEFAULT']['dark']))
		if bool(int(config['SKIP_REGION']['skip'])):
			skipRegion = Limits(int(config['SKIP_REGION']['low']), int(config['SKIP_REGION']['high']))
			if skipRegion.max > dataLimits.max:
				skipRegion.max = dataLimits.max
			if skipRegion.min > dataLimits.max:
				skipRegion.min = dataLimits.max
			if skipRegion.max < dataLimits.min:
				skipRegion.min = dataLimits.min
			if skipRegion.min < dataLimits.min:
				skipRegion.min = dataLimits.min
		else:
			skipRegion = 0
		_abs = bool(int(config['DEFAULT']['abs']))
	else:
		#load the defaults
		print('Could not find the config file...\nLoading defaults')
		degree = 3
		font_size = 18
		voigt = False
		spike_detect = False
		six = False
		thrsh = 0.2
		dataLimits = Limits(650, 2800)
		peakLimits = Limits(900, 1800)
		dark = False
		_abs = False
	matplotlib.rcParams.update({'font.size': font_size})
	if dark:
		plt.style.use('dark_background')
	else:
		plt.style.use('default')
	#load fitting parameters
	try:
		parameters = pd.read_csv(path+'/config/initialData.csv')
	except FileNotFoundError:
		print('Initial parameters were not loaded\nFile not found\nexiting....')
		os._exit(0)


def firstMenu(**kwargs):
	#First menu
	os.system('clear')
	print("............Deconvolution of RAMAN spectra...............")
	print("*********************************************************\n")
	if 'error' in kwargs:
		print('Not a valid choice\n')
	print("Please choose an option:\n1 - Fit the baseline with different parameters")
	print("2 - Adjust the spike-detector\n3 - Proceed to the deconvolution step\nq - Exit")
	choice = input(" >> ")
	try:
		if choice.lower()=='3':
			return
		else:
			menuOneActions[choice.lower()]()
	except KeyError:
		firstMenu(error=True)

def bsLineMenu(**kwargs):
	#Baseline menu
	global data, figureBaseLine
	os.system('clear')
	if 'error' in kwargs:
		print('Only numerical values are allowed')
		choice = 'y'
	else:
		print('************Baseline menu****************\n')
		print('Baseline polynomial degree is set to: %d' %data.bsDegree)
		choice = input('Do you want to change it? [Y/n] >> ')
	if choice.lower()!='n':
		try:
			deg = int(input("New degree: "))
			if _abs:
				data.fitBaseline(deg, peakLimits, abs=_abs)
			else:
				data.fitBaseline(deg, peakLimits)
			figureBaseLine = data.plotBaseline()
			menuOneActions['0']()
		except ValueError:
			print("Only numerical values are allowed")
			bsLineMenu(error=True)

def spikeMenu(**kwargs):
	#Spike detector menu
	global data, thrsh, figureBaseLine
	os.system('clear')
	if 'error' in kwargs:
		print('Only numerical values are allowed')
		choice = 'y'
	else:
		print('**************Spike menu****************\n')
		print('Current threshold value is: %.2f' %thrsh)
		choice = input('Do you want to change it? [Y/n] >> ')
	if choice.lower()!='n':
		try:
			thrsh = float(input("New threshold: "))
			data.detectSpikes(thrsh)
			figureBaseLine = data.plotBaseline()
			menuOneActions['0']()
		except ValueError:
			print("Only numerical values are allowed")
			spikeMenu(error=True)

def secondMenu(**kwargs):
	#Second menu - deconvolution
	global data, parameters, figureResult, figureResultBaseline
	os.system('clear')
	if len(data.spikes):
		choice = input('Do you want to remove the spikes? [Y/n] >> ')
		if choice.lower()!='n':
			data.removeSpikes()
	if 'error' in kwargs:
		print('Not a valid choice\n')
	print("How many peaks do you want to use?\n1 - 4 (D4, D1, D3, G)")
	print("2 - 5 (D4, D1, D3, G, D2)\n3 - 6 (D4, D1, D3, G, D2, PAH)\nq - Exit")
	choice = input(" >> ")
	if choice =='1':
		nr = 4
	elif choice == '2':
		nr = 5
	elif choice =='3':
		nr=6
	elif choice =='q':
		menuOneActions[choice.lower()]()
	else:
		secondMenu(error=True)
	names = parameters['labels'][:nr]
	shape = parameters['shape'][:nr]
	fit = FIT(shape, names)
	#load the initial guess
	parguess =np.concatenate( [[parameters['intens'][i], parameters['width'][i],
			parameters['freq'][i], parameters['voigt'][i]] if shape[i]=='V' else
			[parameters['intens'][i], parameters['width'][i], parameters['freq'][i]]
			for i in np.arange(0, len(names))])
	#lower bound
	lower = np.concatenate( [[parameters['intens_min'][i], parameters['width_min'][i],
			parameters['freq_min'][i], parameters['voigt_min'][i]] if shape[i]=='V' else
			[parameters['intens_min'][i], parameters['width_min'][i], parameters['freq_min'][i]]
			for i in np.arange(0, len(names))])
	#higher bound
	upper = np.concatenate( [[parameters['intens_max'][i], parameters['width_max'][i],
			parameters['freq_max'][i], parameters['voigt_max'][i]] if shape[i]=='V' else
			[parameters['intens_max'][i], parameters['width_max'][i], parameters['freq_max'][i]]
			for i in np.arange(0, len(names))])
	bounds = [lower, upper]
	fit.deconvolute(data, parguess, bounds)
	figureResult = fit.plot()
	figureResultBaseline = fit.plot(data.baseline)
	plt.show()
	return fit

def ThirdMenu():
	#Third menu (Save menu)
	os.system('clear')
	print('**************Save menu****************\n')
	choice = input('Do you want to save all the results it? [y/N] >> ')
	if choice.lower()=='y':
		return True
	else:
		return False


def exit():
	#Exit action
	os._exit(0)


menuOneActions = {
	'1' : bsLineMenu,
	'2' : spikeMenu,
	'q' : exit,
	'0' : firstMenu
}


if __name__ == '__main__':
	#Parse the arguments && help
	parser = ap.ArgumentParser(description='Deconvolution of Raman spectra')
	parser.add_argument('-c', '--convert', help="convert *.wdf files to *.txt", action ="store_true")
	parser.add_argument('name', help='File/Directory name')
	args = parser.parse_args()

	readConf() #read the config file
	inputFile = args.name
	path = inputFile[:-4]
	data.loadData(inputFile) #load the data
	if not args.convert:
		data.setLimits(dataLimits) #set data limits
		if _abs:
			data.fitBaseline(degree, peakLimits, skipRegion, abs=_abs) #fit the baseline (first time)
		else:
			data.fitBaseline(degree, peakLimits, skipRegion) #fit the baseline (first time)
		data.detectSpikes(thrsh) #Look for the spikes
		figureBaseLine = data.plotBaseline() #plot the baseline
		firstMenu() #Show the first menu
		fit = secondMenu() #Show the second menu

		if ThirdMenu(): #to save or not to save?
			if not os.path.exists(path):
				os.makedirs(path) #create the directory for the report
			figureBaseLine.savefig(path+'/baseline.png')
			figureResult.savefig(path+'/result.png')
			figureResultBaseline.savefig(path+'/result+baseline.png')
			out = pd.DataFrame()
			out['Raman shift'] = data.X
			out['Raw data'] = data.Y
			out['Baseline'] = data.baseline
			out['Intensity'] = data.noBaseline
			out = pd.concat([out, fit.peaks[np.append(fit.names, 'cumulative')]], axis=1, sort=False)
			out.to_csv(path +'/data.csv', index=None)
			output = open(path+'/report.txt', "w")
			output.writelines(fit.report)
			output.close()




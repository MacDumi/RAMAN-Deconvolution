#!/usr/bin/python3
from functools import partial
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
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, Value
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

#default parameters
degree = 3
font_size = 18
voigt = False
spike_detect = False
six = False
thrsh = 0.2
dataLimits = Limits(600, 2300)
peakLimits = Limits(900, 1850)
dark = False
_abs = True
skipRegion = 0
nr_bands = 2
###################################
font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 14,
                }
def clear():
        #for Windows
        if os.name == 'nt':
            _ = os.system('cls')
        #for posix
        else:
            _ = os.system('clear')
def readConf():
        path =os.path.dirname(os.path.realpath(__file__))
        #read the configuration file
        global degree, voigt, thrsh, six, spike_detect, peakLimits, parameters, skipRegion, _abs
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
        clear()
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
        clear()
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
                                data.fitBaseline(deg, peakLimits, skipRegion, abs=_abs)
                        else:
                                data.fitBaseline(deg, peakLimits, skipRegion)
                        figureBaseLine = data.plotBaseline()
                        menuOneActions['0']()
                except ValueError:
                        print("Only numerical values are allowed")
                        bsLineMenu(error=True)

def spikeMenu(**kwargs):
        #Spike detector menu
        global data, thrsh, figureBaseLine
        clear()
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
        clear()
        if len(data.spikes):
                choice = input('Do you want to remove the spikes? [Y/n] >> ')
                if choice.lower()!='n':
                        data.removeSpikes()
        if 'error' in kwargs:
                print('Not a valid choice\n')
        peaks = list(parameters['labels'])
        print("How many peaks do you want to use?")
        for i in np.arange(2, len(peaks)+1):
            print(i, " - ", peaks[:i])
        choice = input(" >> ")
        if choice =='q':
            menuOneActions[choice.lower()]()
        else:
            try:
                nr = int(choice)
            except ValueError:
                secondMenu(error=True)
        if nr>len(peaks):
                secondMenu(error=True)
        fit, parguess, bounds = bound(nr, parameters)
        fit.deconvolute(data, parguess, bounds, False)
        figureResult = fit.plot()
        figureResultBaseline = fit.plot(data.baseline)
        plt.show()
        return fit

def bound(nr, parameters):
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
        return fit, parguess, [lower, upper]


def ThirdMenu():
        #Third menu (Save menu)
        clear()
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

def prepareInput(path):
        data.loadData(path) #load the data
        data.setLimits(dataLimits) #set data limits
        if _abs:
                data.fitBaseline(degree, peakLimits, skipRegion, abs=_abs) #fit the baseline (first time)
        else:
                data.fitBaseline(degree, peakLimits, skipRegion) #fit the baseline (first time)

def wrapper(parameters, nr, name):
        data = DATA()
        data.loadData(name) #load the data
        data.setLimits(dataLimits) #set data limits
        if _abs:
                data.fitBaseline(degree, peakLimits, skipRegion, abs=_abs) #fit the baseline (first time)
        else:
                data.fitBaseline(degree, peakLimits, skipRegion) #fit the baseline (first time)
        fit, parguess, bounds = bound(nr, parameters)
        fit.deconvolute(data, parguess, bounds, True)
        return name, fit.pars


if __name__ == '__main__':
        #Parse the arguments && help
        parser = ap.ArgumentParser(description='Deconvolution of Raman spectra')
        parser.add_argument('-b', '--batch', help="fit multiple files", action ="store_true")
        parser.add_argument('-n', type=int, help='number of bands (for batch fitting only)')
        parser.add_argument('name', help='File/Directory name')
        args = parser.parse_args()

        readConf() #read the config file
        inputFile = args.name
        if args.batch:
            clear()
            # mp.set_start_method('spawn')
            if args.n:
                nr_bands = args.n
            else:
                nr_bands = 2
            try:
                files = glob.glob(inputFile+'/*.txt')
                print(len(files), ' files found')
            except:
                print('Not a valid path')
                exit()
            if not len(files):
                print('Did not find text files...exiting')
                exit()

            nproc = cpu_count()
            pool  = Pool(nproc)
            names = parameters['labels'][:nr_bands]
            shape = parameters['shape'][:nr_bands]
            #list of parameter names
            labels = [[names[i]+'_amplitude', names[i]+'_gamma', names[i]+'_center', names[i]+'_voigt_ratio']
                            if shape[i]=='V' else [names[i]+'_amplitude', names[i]+'_gamma', names[i]+'_center']
                            for i in np.arange(0, len(names))]
            labels = [item for sublist in labels for item in sublist]
            function = partial(wrapper, parameters, nr_bands)
            out = pd.DataFrame()
            out['labels'] = np.asarray(labels).flatten()
            for n, r in tqdm(pool.imap_unordered(function, files), total=len(files)):
                out[os.path.basename(n)] =r
            out.to_csv(inputFile+'/fit.csv', index=None)
            print('Fitting completed...output saved')
        else:
            path = inputFile[:-4]
            prepareInput(inputFile)
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


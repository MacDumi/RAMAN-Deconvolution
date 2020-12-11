'''
Class handling the deconvolution process
'''
import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from collections import OrderedDict
from .mcmc import MCMC
from uncertainties import ufloat, unumpy
from uncertainties.umath import *
from PyQt5.QtCore import QObject, pyqtSignal

#Linestyles
linestyles = OrderedDict(
    [
     ('densely dotted',      (0, (1, 1))),

     ('densely dashed',      (0, (5, 1))),

     ('dashdotted',          (0, (3, 4, 1, 4))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('dashdotdotted',         (0, (3, 4, 1, 4, 1, 4))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
ln_style = list(linestyles.items())

class FIT(QObject):
    pg = pyqtSignal(int)
    def __init__(self, shape, names):
        # Constructor
        super(FIT, self).__init__()
        # Used as labels and as an indicator of the number of peaks
        self.names = names
        # Used to calculate the number of fitting parameters
        self.shape = shape
        self.args = [4 if self.shape[i] in ['V', 'B'] else 3 for i in
                                                 np.arange(0, len(self.shape))]
        self.peaks = pd.DataFrame()
        self.peaks_u = pd.DataFrame()
        self.fwhm = np.zeros(len(self.names))
        self.fitGoodness = 0
        self.error = False

    @staticmethod
    def BWF(x, I, x0, s, q):
        # Breit-Wigner-Fano line shape
        temp = (x-x0)/s
        return I*(1+temp*q)**2/(1+temp**2)

    @staticmethod
    def Voigt(x, I, x0, s, n, **kwargs):
        # Voigt peak
        return (n * FIT.lorents(x, I, x0, s) +
                                     (1 - n) * FIT.gauss(x, I, x0, s, **kwargs))

    @staticmethod
    def gauss(x, I, x0, s, **kwargs):
        # Gaussian peak
        if kwargs.get('uncert', None):
            return I * unumpy.exp(-(x - x0)**2 / (2 * s**2))
        else:
            return I * np.exp(-(x - x0)**2 / (2 * s**2))

    @staticmethod
    def lorents(x, I, x0, s):
         # Lorentzian peak
         return I / (1 + ((x - x0)**2 / s**2))

    @staticmethod
    def Peak(self, x, *pars, shape, **kwargs):
        # Returns the peak with the selected shape
        I     = pars[0]
        gamma = pars[1]
        x0    = pars[2]
        v     = 0
        if (shape =='V'):
            #if gaussian
            if len(pars) == 4:
                v = pars[3]
            return FIT.Voigt(x, I, x0, gamma, v, **kwargs)
        elif shape =='B':
            #if BWF
            if len(pars) == 4:
                v = pars[3]
            return FIT.BWF(x, I, x0, gamma, v)
        elif (shape =='G'):
            #if gaussian
            return FIT.gauss(x, I, x0, gamma, **kwargs)
        elif (shape =='L'):
            #if lorentzian
            return FIT.lorents(x, I, x0, gamma)
        else:
            print("unknown parameter")
            return 0

    def model(self, t, *pars, **kwargs):
        # Model used for fitting
        temp =np.zeros(len(t))
        for i in np.arange(0, len(self.names)):
            indx = int(np.sum(self.args[:i]))
            temp = np.sum((temp, self.Peak(self, t,
                              *pars[indx:indx+self.args[i]],
                    shape=self.shape[i], **kwargs)), axis=0)
        return temp

    def checkParams(self, params, bounds, corr, shape):
        # Take initial guess parameters, bounds and
        # add parameters for asymetry and L/G ratio
        new_params = []
        lower = []
        upper = []
        new_corr = []
        try:
            for i, s in enumerate(shape):
                new_params = np.append(new_params, params[3 * i : 3 * i + 3])
                lower = np.append(lower, bounds[0, 3 * i : 3 * i + 3])
                upper = np.append(upper, bounds[1, 3 * i : 3 * i + 3])
                new_corr = np.append(new_corr, corr[3 * i : 3 * i + 3])
                if s == 'V':
                    new_params = np.append(new_params, 0.5)
                    lower = np.append(lower, 0)
                    upper = np.append(upper, 1)
                    new_corr = np.append(new_corr, 1)
                elif s == 'B':
                    new_params = np.append(new_params, 0.02)
                    lower = np.append(lower, 0)
                    upper = np.append(upper, 0.5)
                    new_corr = np.append(new_corr, 1)
        except Exception as e:
            print(f'Exception (checkParams): {e}')
        return new_params, np.row_stack((lower, upper)), new_corr

    def FWHM(self):
        # Computes the full width at half maximum of all the peaks
        X = self.peaks['freq']
        self.fwhm = []
        for i, name in enumerate(self.names):
            Y = self.peaks_u[name].values
            Y_n = np.array([i.n for i in Y])
            Y_std = np.array([i.s for i in Y])
            if not len(Y_n):
                self.fwhm = np.zeros(len(self.names))
                return -1
            nominal = self.width(X, Y_n)
            max_val = self.width(X, Y_n+Y_std)
            err = np.abs(max_val - nominal)
            self.fwhm = np.append(self.fwhm, ufloat(nominal, err))


    def width(self, X, Y):
        HM = (max(Y) - min(Y))/2
        if not len(Y):
            return -1
        pos_max = Y.argmax()
        nearest_above = (np.abs(Y[pos_max:-1] - HM)).argmin()+pos_max
        nearest_below = (np.abs(Y[0:pos_max] - HM)).argmin()
        fwhm = (np.mean(X[nearest_above ]) - np.mean(X[nearest_below]))
        return fwhm

    def Area(self):
        # Integrates all the peaks
        self.area=[]
        for name in self.names:
            self.area = np.append(self.area, np.trapz(
                    self.peaks_u[name].values, x = self.peaks['freq']))

    def decMCMC(self, data, parguess, bounds, steps):
        # Deconvolution with MCMC
        self.deconvolute(data, parguess, bounds, False)

        parguess = self.pars
        Norm = np.max(data.current)/100
        Y = data.current/Norm

        corr = np.asarray([Norm if n%3 == 0 else 1 for n
                                         in np.arange(len(parguess))])

        _, bounds, corr = self.checkParams(parguess, np.asarray(bounds),
                                                        corr, self.shape)
        parguess = [par/cor for par, cor in zip(self.pars, corr)]

        try:
            mcmc = MCMC(self.model, parguess, bounds)
            mcmc.pg.connect(self.pg.emit)
            st = 0.5 * np.ones(len(parguess))
            arg_range = range(0, len(self.args))
            st[[int(np.sum(self.args[:i])+2) for i in arg_range]] = 0.1
            st[[int(np.sum(self.args[:i+1])-1) for i in
                                arg_range if self.args[i] == 4]] = 0.01
            mcmc.step = st
            self.pars, self.perr = mcmc(data.X, Y, steps, corr = corr )

            self.pars_u = [ufloat(p, abs(e)) for p, e in
                                             zip(self.pars, self.perr)]
            #Calculate each peak
            for i, name in enumerate(self.names):
                    indx = int(np.sum(self.args[:i]))
                    self.peaks[name] = self.Peak(self, data.X,
                            *self.pars[indx:indx+self.args[i]],
                                          shape =self.shape[i])
                    self.peaks_u[name] = self.Peak(self, data.X,
                           *self.pars_u[indx:indx+self.args[i]],
                            shape =self.shape[i], uncert = True)
            # Save the fit result
            self.peaks['cumulative']   = self.model(data.X, *self.pars)
            self.peaks_u['cumulative'] = self.model(data.X, *self.pars_u,
                                                             uncert=True)
            self.fitGoodness = self.goodness(data.current,
                                                self.peaks['cumulative'])
            # Calculate fwhm
            self.FWHM()
            # Calculate the areas
            self.Area()
            # Print the fit report
            self.printResult(data)
        except Exception as e:
            print('Exception\n', e)

    def deconvolute(self, data, parguess, bounds, batch):
        # Deconvolution routine
        self.peaks['freq']=data.X
        self.peaks['exp'] = data.current

        # Weighting function (lower value -> higher weight)
        sigma =np.ones(len(data.X))*2
        sigma[np.abs(data.X-1500)<100]=0.6
        sigma[np.abs(data.X-1360)<40]=0.7
        sigma[np.abs(data.X-1180)<100]=0.6
        sigma[np.abs(data.X-1600)<50]=0.7
        sigma[np.abs(data.X-900)<100]=0.9
        sigma[np.abs(data.X-1750)<100]=0.9

        # Normalize the spectrum and the initial guess
        Norm = np.max(data.current)/100
        Y = data.current/Norm
        parguess[::3] /= Norm

        corr = np.asarray([Norm if n%3 == 0 else 1 for n in
                                                np.arange(len(parguess))])

        parguess, bounds, corr = self.checkParams(parguess,
                                     np.asarray(bounds), corr, self.shape)

        try:
            # Fit
            self.pars, pcov = curve_fit(self.model, data.X, Y, parguess,
                                             sigma=sigma, bounds = bounds)
            self.perr= np.sqrt(np.diag(pcov))
            self.pars = [par*cor for par, cor in zip(self.pars, corr)]

            self.pars_u = [ufloat(p, abs(e)) for p, e in
                                                zip(self.pars, self.perr)]
            # Calculate each peak
            for i, name in enumerate(self.names):
                    indx = int(np.sum(self.args[:i]))
                    self.peaks[name] = self.Peak(self, data.X,
                                        *self.pars[indx:indx+self.args[i]],
                                                      shape =self.shape[i])
                    self.peaks_u[name] = self.Peak(self, data.X,
                                      *self.pars_u[indx:indx+self.args[i]],
                                       shape =self.shape[i], uncert = True)
            # Save the fit result
            self.peaks['cumulative']   = self.model(data.X, *self.pars)
            self.peaks_u['cumulative'] = self.model(data.X, *self.pars_u,
                                                             uncert=True)
            self.fitGoodness = self.goodness(data.current,
                                                self.peaks['cumulative'])
            # Calculate fwhm
            self.FWHM()
            # Calculate the areas
            self.Area()
            if not batch:
                # Print the fit report
                self.printResult(data)
        except RuntimeError as e:
            if not batch:
                print('Failed to deconvolute...\n')
                print(f'Try with a different initial guess:\n{e}')
                self.error = True
        except ValueError as e:
            if not batch:
                self.error = True
                print('Failed to deconvolute...\n')
                print(f'Try with different bounds:\n{e}')

    def goodness(self, data, fit):
        # Residual sum of squares
        ss_res = np.sum((data - fit) ** 2)

        # Total sum of squares
        ss_tot = np.sum((data - np.mean(data)) ** 2)

        # R-squared
        r2 = 1 - (ss_res / ss_tot)
        return r2

    def plot(self, *args, **kwargs):
        # Plot overlapping peaks with or without the baseline
        path = kwargs.get('path', None)
        fig = kwargs.get('figure', None)
        if not fig:
            return -1
        baseline = np.zeros(len(self.peaks['freq']))
        if kwargs.get('baseline'):
                baseline = args[0]
        ax = fig.add_subplot(111)
        ax.plot(self.peaks['freq'], self.peaks['exp']+baseline,
                                            label='Experimental data')
        ax.plot(self.peaks['freq'], self.peaks['cumulative']+baseline,
                               'r-',linewidth = 1, label='Cumulative')
        for i, name in enumerate(self.names):
                ax.plot(self.peaks['freq'], self.peaks[name]+baseline,
                            linewidth = 2, linestyle = ln_style[i][1],
                                                          label =name)
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Raman shift, $cm^{-1}$")
        ax.legend()
        ax.grid()
        plt.tight_layout()
        if path:
            plt.savefig(path)

    def printResult(self, data):
            #Print the fit report
            text  = '****************BASELINE*******************\n'
            text += f'Degree: {data.bsDegree}\n'
            text += 'Coefficients (starting with the highest power):\n'
            text += str(data.bsCoef)
            text += '\n****************FIT RESULTS****************\n'
            text += f'Goodness (R^2): {self.fitGoodness}\n\n'
            self.intensity = []
            for i, name in enumerate(self.names):
                    indx = int(np.sum(self.args[:i]))
                    params = self.pars[indx:indx+self.args[i]]
                    errs = self.perr[indx:indx+self.args[i]]
                    text += f'Peak {name}:\n'
                    text += f'Center: {params[2]:.4f} +/- {errs[2]:.4f} cm-1\n'
                    text += f'Amplitude: {params[0]:.4f} +/- {errs[0]:.4f}\n'
                    text += f'Gamma: {params[1]:.4f} +/- {errs[1]:.4f}\n'
                    text += f'\tFWHM: {self.fwhm[i]:10.4f}\n'
                    if self.shape[i] == 'V':
                        text = f'\tL/G ratio = {params[3]:.4f}\n'
                    elif self.shape[i] == 'B':
                        text += f'\tAsymmetry factor 1/q = {params[3]:.4f}\n'
                    self.intensity = np.append(self.intensity, params[0])
                    text += f'\tArea = {self.area[i]:10.4f}\n\n'
            print(text)
            self.report = text



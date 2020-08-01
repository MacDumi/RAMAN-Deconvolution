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
from mcmc import MCMC
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
                #constructor
                super(FIT, self).__init__()
                self.names = names #used as labels and as an indicator of the number of peaks
                self.shape = shape #used to calculate the number of fitting parameters
                self.args = [4 if self.shape[i] in ['V', 'B'] else 3 for i in np.arange(0, len(self.shape))]
                self.peaks = pd.DataFrame()
                self.peaks_u = pd.DataFrame()
                self.fwhm = np.zeros(len(self.names))
                self.fitGoodness = 0
                self.error = False

        @staticmethod
        def BWF(x, I, x0, s, q):
                #Breit-Wigner-Fano line shape
                temp = (x-x0)/s
                return I*(1+temp*q)**2/(1+temp**2)

        @staticmethod
        def Voigt(x, I, x0, s, n, **kwargs):
                #voigt peak
                return n*FIT.lorents(x, I, x0, s) + (1-n)*FIT.gauss(x,I, x0, s, **kwargs)

        @staticmethod
        def gauss(x, I, x0, s, **kwargs):
                #gaussian peak
                if kwargs.get('uncert', None):
                    return I*unumpy.exp(-(x-x0)**2/(2*s**2))
                else:
                    return I*np.exp(-(x-x0)**2/(2*s**2))

        @staticmethod
        def lorents(x, I, x0, s):
                #lorentzian peak
                return I/ (1+((x - x0)**2 /s**2))

        @staticmethod
        def Peak(self, x, *pars, shape, **kwargs):
                #returns the peak with the selected shape
                I = pars[0]
                gamma = pars[1]
                x0 = pars[2]
                v =0
                if (shape =='V'): #if gaussian
                        if len(pars)==4:
                                v =pars[3]
                        return FIT.Voigt(x, I, x0, gamma, v, **kwargs)
                elif shape =='B': #if BWF
                        if len(pars) == 4:
                            v = pars[3]
                        return FIT.BWF(x, I, x0, gamma, v)
                elif (shape =='G'): #if gaussian
                        return FIT.gauss(x, I, x0, gamma, **kwargs)
                elif (shape =='L'): #if lorentzian
                        return FIT.lorents(x, I, x0, gamma)
                else:
                        print("unknown parameter")
                        return 0

        def model(self, t, *pars, **kwargs):
                #model used for fitting
                temp =np.zeros(len(t))
                for i in np.arange(0, len(self.names)):
                        indx = int(np.sum(self.args[:i]))
                        temp = np.sum((temp, self.Peak(self, t,  *pars[indx:indx+self.args[i]], shape=self.shape[i], **kwargs)), axis=0)
                return temp

        def checkParams(self, params, bounds, corr, shape):
                #take initial guess parameters and bounds and add parameters for asymetry and L/G ratio
                new_params = []
                lower = []
                upper = []
                new_corr = []
                try:
                    for i, s in enumerate(shape):
                            new_params = np.append(new_params, params[3*i:3*i+3])
                            lower = np.append(lower, bounds[0, 3*i:3*i+3])
                            upper = np.append(upper, bounds[1, 3*i:3*i+3])
                            new_corr = np.append(new_corr, corr[3*i: 3*i+3])
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
                    print(e)
                return new_params, np.row_stack((lower, upper)), new_corr

        def FWHM(self):
                #returns the full width at half maximum of all the peaks
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

                #         difference = max(Y) - min(Y)
                #         HM = difference / 2
                #         pos_extremum = Y.idxmax()  # or in your case: arr_y.argmin()
                #         nearest_above = (np.abs(Y[pos_extremum:-1] - HM)).idxmin()
                #         nearest_below = (np.abs(Y[0:pos_extremum] - HM)).idxmin()
                #         self.fwhm[i] = (np.mean(X[nearest_above ]) - np.mean(X[nearest_below]))

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
                #integrates all the peaks
                self.area=[]
                print(self.peaks_u)
                for name in self.names:
                        self.area = np.append(self.area ,np.trapz(self.peaks_u[name].values, x = self.peaks['freq']))
                print(self.area)

        def decMCMC(self, data, parguess, bounds, steps):
                #deconvolution with MCMC
                self.deconvolute(data, parguess, bounds, False)

                parguess = self.pars
                Norm = np.max(data.current)/100
                Y = data.current/Norm

                corr = np.asarray([Norm if n%3 == 0 else 1 for n in np.arange(len(parguess))])

                _, bounds, corr = self.checkParams(parguess, np.asarray(bounds), corr, self.shape)
                parguess = [par/cor for par, cor in zip(self.pars, corr)]

                try:
                    mcmc = MCMC(self.model, parguess, bounds)
                    mcmc.pg.connect(self.pg.emit)
                    st = 0.5*np.ones(len(parguess))
                    st[[int(np.sum(self.args[:i])+2) for i in range(0, len(self.args))]] = 0.1
                    st[[int(np.sum(self.args[:i+1])-1) for i in range(0, len(self.args)) if self.args[i] ==4]] = 0.01
                    # mcmc.step = [0.1 if (n-1)%3 == 0 else 0.5 for n in np.arange(len(parguess))]
                    mcmc.step = st
                    self.pars, self.perr = mcmc(data.X, Y, steps, corr = corr )

                    self.pars_u = [ufloat(p, abs(e)) for p, e in zip(self.pars, self.perr)]
                    #Calculate each peak
                    for i, name in enumerate(self.names):
                            indx = int(np.sum(self.args[:i]))
                            self.peaks[name] = self.Peak(self, data.X, *self.pars[indx:indx+self.args[i]], shape =self.shape[i])
                            self.peaks_u[name] = self.Peak(self, data.X, *self.pars_u[indx:indx+self.args[i]], shape =self.shape[i], uncert = True)
                    self.peaks['cumulative']=self.model(data.X, *self.pars) #save the fit result
                    self.peaks_u['cumulative']=self.model(data.X, *self.pars_u, uncert=True) #save the fit result
                    self.fitGoodness = self.goodness(data.current, self.peaks['cumulative'])
                    self.FWHM() #calculate fwhm
                    self.Area() #calculate the areas
                    self.printResult(data) #print the fit report
                except Exception as e:
                    print('exception\n', e)

        def deconvolute(self, data, parguess, bounds, batch):
                #deconvolution routine
                self.peaks['freq']=data.X
                self.peaks['exp'] = data.current

                #weighting function (lower value -> higher weight)
                sigma =np.ones(len(data.X))*2
                sigma[np.abs(data.X-1500)<100]=0.6
                sigma[np.abs(data.X-1360)<40]=0.7
                sigma[np.abs(data.X-1180)<100]=0.6
                sigma[np.abs(data.X-1600)<50]=0.7
                sigma[np.abs(data.X-900)<100]=0.9
                sigma[np.abs(data.X-1750)<100]=0.9

                #Normalize the spectrum and the initial guess to reduce the fitting time
                Norm = np.max(data.current)/100
                Y = data.current/Norm
                parguess[::3] /= Norm

                corr = np.asarray([Norm if n%3 == 0 else 1 for n in np.arange(len(parguess))])

                parguess, bounds, corr = self.checkParams(parguess, np.asarray(bounds), corr, self.shape)

                try:
                    #Fit
                    self.pars, pcov = curve_fit(self.model, data.X, Y, parguess, sigma=sigma, bounds = bounds)
                    self.perr= np.sqrt(np.diag(pcov))
                    self.pars = [par*cor for par, cor in zip(self.pars, corr)]

                    self.pars_u = [ufloat(p, abs(e)) for p, e in zip(self.pars, self.perr)]
                    #Calculate each peak
                    for i, name in enumerate(self.names):
                            indx = int(np.sum(self.args[:i]))
                            self.peaks[name] = self.Peak(self, data.X, *self.pars[indx:indx+self.args[i]], shape =self.shape[i])
                            self.peaks_u[name] = self.Peak(self, data.X, *self.pars_u[indx:indx+self.args[i]], shape =self.shape[i], uncert = True)
                    self.peaks['cumulative']=self.model(data.X, *self.pars) #save the fit result
                    self.peaks_u['cumulative']=self.model(data.X, *self.pars_u, uncert=True) #save the fit result
                    self.fitGoodness = self.goodness(data.current, self.peaks['cumulative'])
                    self.FWHM() #calculate fwhm
                    self.Area() #calculate the areas
                    if not batch:
                        self.printResult(data) #print the fit report
                except RuntimeError as e:
                    if not batch:
                        print('Failed to deconvolute...\nTry with a different initial guess:\n', e)
                        self.error = True
                except ValueError as e:
                    if not batch:
                        self.error = True
                        print('Failed to deconvolute...\nTry with different bounds:\n', e)

        def goodness(self, data, fit):
            # residual sum of squares
            ss_res = np.sum((data - fit) ** 2)

            # total sum of squares
            ss_tot = np.sum((data - np.mean(data)) ** 2)

            # r-squared
            r2 = 1 - (ss_res / ss_tot)
            return r2


        def plot(self, *args, **kwargs):
                #plot overlapping peaks with or without the baseline
                path = kwargs.get('path', None)
                fig = kwargs.get('figure', None)
                if not fig:
                    return -1
                baseline = np.zeros(len(self.peaks['freq']))
                if kwargs.get('baseline'):
                        baseline = args[0]
                ax = fig.add_subplot(111)
                ax.plot(self.peaks['freq'], self.peaks['exp']+baseline,label='Experimental data')
                ax.plot(self.peaks['freq'], self.peaks['cumulative']+baseline, 'r-',linewidth = 1, label='Cumulative')
                for i, name in enumerate(self.names):
                        ax.plot(self.peaks['freq'], self.peaks[name]+baseline, linewidth = 2,linestyle = ln_style[i][1], label =name)
                ax.set_ylabel("Intensity")
                ax.set_xlabel("Raman shift, $cm^{-1}$")
                ax.legend()
                ax.grid()
                plt.tight_layout()
                if path:
                    plt.savefig(path)
                # plt.show()
                # return fig

        def printResult(self, data):
                #Print the fit report
                text = "****************BASELINE*******************\nDegree: %d\nCoefficients (starting with the highest power):\n"%data.bsDegree
                text = text + str(data.bsCoef)
                text = text +"\n****************FIT RESULTS****************\n"
                text += "Goodness (R^2): {}\n\n".format(self.fitGoodness)
                self.intensity = []
                for i, name in enumerate(self.names):
                        indx = int(np.sum(self.args[:i]))
                        params = self.pars[indx:indx+self.args[i]]
                        errs = self.perr[indx:indx+self.args[i]]
                        text = text +"Peak %s:\n    Center: %.4f +/- %.4f cm-1\n    Amplitude: %.4f +/- %.4f\n    gamma: %.4f +/- %.4f\n"%(name,
                                params[2], errs[2], params[0], errs[0], params[1], errs[1])
                        text += "    FWHM: {:10.4f}\n".format(self.fwhm[i])
                        if self.shape[i]=='V':
                                text = text +"    L/G ratio = %.4f\n" %params[3]
                        elif self.shape[i]=='B':
                                text = text +"    Asymmetry factor 1/q = %.4f\n" %params[3]
                        self.intensity = np.append(self.intensity, params[0])
                        text = text +"    Area = {:10.4f}\n\n".format(self.area[i])
                # text = text +"\n**************Ratio - Amplitude************\n   D1/G= %.4f\n    D4/G= %.4f\n" %(self.intensity[1]/self.intensity[3], self.intensity[0]/self.intensity[3])
                '''
                if len(self.intensity)==5:
                        text +="    D1/(G+D1+D2)= %.4f\n" %self.intensity[1]/(self.intensity[3]+self.intensity[4]+self.intensity[1])
                text = text +"\n**************Ratio - Areas****************\n   D1/G= %.4f\n    D4/G= %.4f\n" %(self.area[1]/self.area[3], self.area[0]/self.area[3])
                if len(self.area)==5:
                        text +="    D1/(G+D1+D2)= %.4f\n" %self.area[1]/(self.area[3]+self.area[4]+self.area[1])
                '''
                print(text)
                self.report = text



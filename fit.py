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

class FIT:

        def __init__(self, shape, names):
                #constructor
                super(FIT, self).__init__()
                self.names = names #used as labels and as an indicator of the number of peaks
                self.shape = shape #used to calculate the number of fitting parameters
                self.args = [4 if self.shape[i]=='V' else 3 for i in np.arange(0, len(self.shape))]
                self.peaks = pd.DataFrame()
                self.fwhm = np.zeros(len(self.names))

        def Voigt(self, x, I, x0, s, n):
                #voigt peak
                return n*I/(1+((x - x0)**2 /s**2)) + (1-n)*I*exp(-(x-x0)**2/(2*s**2))

        def gauss(self, x, I, x0, s):
                #gaussian peak
                return I*exp(-(x-x0)**2/(2*s**2))

        def lorents(self, x, I, x0, s):
                #lorentzian peak
                return I/ (1+((x - x0)**2 /s**2))

        def Peak(self, x, *pars, **kwargs):
                #returns the peak witht the selected shape
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
                #model used for fitting
                temp =np.zeros(len(t))
                for i in np.arange(0, len(self.names)):
                        indx = int(np.sum(self.args[:i]))
                        temp = np.sum((temp, self.Peak(t,  *pars[indx:indx+self.args[i]], shape=self.shape[i])), axis=0)
                return temp

        def FWHM(self):
                #returns the full width at half maximum of all the peaks
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
                #integrates all the peaks
                self.area=[]
                for name in self.names:
                        self.area = np.append(self.area ,np.trapz(self.peaks[name], x = self.peaks['freq']))


        def deconvolute(self, data, parguess, bounds, batch):
                #deconvolution routine
                self.peaks['freq']=data.X
                self.peaks['exp'] = data.noBaseline
                #weighting function (lower value - higher weight)
                sigma =np.ones(len(data.X))*2
                sigma[np.abs(data.X-1500)<100]=0.6
                sigma[np.abs(data.X-1360)<40]=0.7
                sigma[np.abs(data.X-1180)<100]=0.6
                sigma[np.abs(data.X-1600)<50]=0.7
                sigma[np.abs(data.X-900)<100]=0.9
                sigma[np.abs(data.X-1750)<100]=0.9
                try:
                    #Fit
                    self.pars, pcov = curve_fit(self.model, data.X, data.noBaseline, parguess, sigma=sigma, bounds = bounds)
                    self.perr= np.sqrt(np.diag(pcov))

                    if not batch:
                        #Calculate each peak
                        for i, name in enumerate(self.names):
                                indx = int(np.sum(self.args[:i]))
                                self.peaks[name] = self.Peak(data.X, *self.pars[indx:indx+self.args[i]], shape =self.shape[i])
                        self.peaks['cumulative']=self.model(data.X, *self.pars) #save the fit result
                        self.FWHM() #calculate fwhm
                        self.area() #calculate the areas
                        self.printResult(data) #print the fit report
                except RuntimeError:
                    if not batch:
                        print('Failed to deconvolute...\nTry with a different initial guess')
                        # os._exit(0)

        def plot(self, *args):
                #plot overlapping peaks with or without the baseline
                baseline = np.zeros(len(self.peaks['freq']))
                if len(args)==1:
                        baseline = args[0]
                fig = plt.figure(figsize=(12,8))
                ax = fig.add_subplot(111)
                ax.scatter(self.peaks['freq'], self.peaks['exp']+baseline, s=70,linewidth=1.5, facecolors = 'none', edgecolor = '#1E68FF',label='Experimental data')
                ax.plot(self.peaks['freq'], self.peaks['cumulative']+baseline, 'r-',linewidth = 1, label='Cumulative')
                for i, name in enumerate(self.names):
                        ax.plot(self.peaks['freq'], self.peaks[name]+baseline, linewidth = 2,linestyle = ln_style[i][1], label =name)
                ax.set_ylabel("Intensity")
                ax.set_xlabel("Raman shift, $cm^{-1}$")
                ax.legend()
                ax.grid()
                plt.tight_layout()
                plt.show(block=False)
                return fig

        def printResult(self, data):
                #Print the fit report
                text = "****************BASELINE*******************\nDegree: %d\nCoefficients (starting with the highest power):\n"%data.bsDegree
                text = text + str(data.bsCoef)
                text = text +"\n****************FIT RESULTS****************\n"
                self.intensity = []
                for i, name in enumerate(self.names):
                        indx = int(np.sum(self.args[:i]))
                        params = self.pars[indx:indx+self.args[i]]
                        errs = self.perr[indx:indx+self.args[i]]
                        text = text +"Peak %s:\n        Center: %.4f +/- %.4f cm-1\n    Amplitude: %.4f +/- %.4f\n      gamma: %.4f +/- %.4f\n  FWHM: %.4f\n" %(name,
                                params[2], errs[2], params[0], errs[0], params[1], errs[1], self.fwhm[i])
                        if self.shape[i]=='V':
                                text = text +"  L/G ratio = %.4f\n" %params[3]
                        self.intensity = np.append(self.intensity, params[0])
                        text = text +"  Area = %.4f\n" %self.area[i]
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



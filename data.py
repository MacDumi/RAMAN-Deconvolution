'''
Class dealing with the Raman data

'''
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from convertwdf import *

class DATA:
        def __init__(self):
                #constructor
                super(DATA, self).__init__()
                self.X          = 0
                self.Y          = 0
                self.prev       = 0
                self.current    = 0
                self.baseline   = 0
                self.bsDegree   = 0
                self.bsCoef     = 0
                self.spikes     =[]

        def loadData(self, path):
                #load data
                if path[-3:]=='wdf':
                        self.loadData(path[:-3]+'txt')
                else:
                        try:
                                dt = np.loadtxt(path, skiprows =5)
                        except OSError:
                                print("File not found")
                                return
                        self.setData(dt[:,0], dt[:,1])
                        self.Y = dt[:,1]

        def setData(self, X, Y):
            self.X = X
            self.current = Y.copy()
            self.baseline   = 0
            self.bsDegree   = 0
            self.bsCoef     = 0
            self.spikes     =[]

        def smooth(self, Y, window):
            return savgol_filter(Y, window, 3)

        def getData(self):
            data = pd.DataFrame()
            data['Raman shift'] = self.X
            comments = ''
            if np.shape(self.baseline):
                data['Baseline'] = self.baseline
                comments = '#Polynomial fit -- degree: {}\n'.format(self.bsDegree)
                comments += '#Fitting coefficients (high to low):\n#'
                coefficients = ['{:.4E}  'for coef in self.bsCoef]
                comments += ''.join(coefficients).format(*self.bsCoef)+'\n'
            if np.shape(self.current):
                data['Corrected'] = self.current
            data['Raw intensity'] = self.Y
            return data, comments

        def crop(self, _min, _max):
            if int(_min)> self.X[0]:
                _min = self.X[0]
            if int(_max)> self.X[-1]:
                _max = self.X[-1]
            min_ = np.argwhere(self.X>int(_min))[0][0]
            max_ = np.argwhere(self.X>int(_max))[0][0]
            self.prev = np.column_stack((self.X, self.current))
            self.X = self.X[min_ : max_]
            self.Y = self.Y[min_ : max_]
            self.current = self.current[min_ : max_]
            if np.shape(self.baseline):
                self.baseline = self.baseline[min_ : max_]

        def setLimits(self, limits):
                #set the limits for the loaded data and crop it
                if self.X[0]>limits.min:
                        limits.min=self.X[0]
                if self.X[-1]<limits.max:
                        limits.max=self.X[-1]
                self.crop(limits.min, limits.max)

        def fitBaseline(self, degree, limits, **kwargs):
                #Select the part without Raman peaks and fit a polynomial function
                self.limits = limits
                baselineX = np.append(self.X[:np.argwhere(self.X>self.limits.min)[0][0]],
                        self.X[np.argwhere(self.X>self.limits.max)[0][0]:])
                baselineY = np.append(self.current[:np.argwhere(self.X>self.limits.min)[0][0]],
                        self.current[np.argwhere(self.X>self.limits.max)[0][0]:])
                # if self.skip:
                #         if self.skip.min<self.X[0]:
                #                 self.skip.min =self.X[0]
                #         if self.skip.min>self.X[-2]:
                #                 self.skip.min =self.X[-2]
                #         if self.skip.max<self.X[0]:
                #                 self.skip.max =self.X[0]
                #         if self.skip.max>self.X[-2]:
                #                 self.skip.max =self.X[-2]
                #         baselineY = np.append(baselineY[:np.argwhere(baselineX>self.skip.min)[0][0]],
                #                 baselineY[np.argwhere(baselineX>self.skip.max)[0][0]:])
                #         baselineX = np.append(baselineX[:np.argwhere(baselineX>self.skip.min)[0][0]],
                #                 baselineX[np.argwhere(baselineX>self.skip.max)[0][0]:])
                #         self.Y = np.append(self.Y[:np.argwhere(self.X>self.skip.min)[0][0]],
                #                 self.Y[np.argwhere(self.X>self.skip.max)[0][0]:])
                #         self.X = np.append(self.X[:np.argwhere(self.X>self.skip.min)[0][0]],
                #                 self.X[np.argwhere(self.X>self.skip.max)[0][0]:])

                self.bsDegree = degree
                self.bsCoef = np.polyfit(baselineX, baselineY, self.bsDegree)
                fit = np.poly1d(self.bsCoef)
                self.baseline = fit(self.X)
                self.prev = np.column_stack((self.X, self.current))
                self.current -= self.baseline
                if 'abs' in kwargs:
                        self.current = abs(self.current)
                else:
                        Min = min(self.current)
                        if Min <0:
                                self.current -= Min


        def plotBaseline(self):
                #plot the baseline and spikes
                plt.close("all")
                fig = plt.figure(figsize=(12,8))
                ax = fig.add_subplot(111)
                ax.plot(self.X, self.Y, label = 'Experimental data')
                if len(self.spikes):
                        ax.plot(self.X[self.spikes], self.Y[self.spikes], 'ro', label='Spikes')
                ax.plot(self.X, self.baseline, 'r--', label = 'Baseline')
                ax.plot([self.limits.min, self.limits.min], [min(self.Y), max(self.Y)], 'r-')
                ax.plot([self.limits.max,self.limits.max],[min(self.Y), max(self.Y)], 'r-', label='Excluded region')
                # if self.skip:
                #         ax.plot([self.skip.min, self.skip.min], [min(self.Y), max(self.Y)], 'y-')
                #         ax.plot([self.skip.max, self.skip.max], [min(self.Y), max(self.Y)], 'y-', label = 'Second excluded region')
                ax.set_ylabel("Intensity")
                ax.set_xlabel("Raman shift, $cm^{-1}$")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.show()
                return fig

        def detectSpikes(self, threshold):
                #detect spikes
                self.spikes=[]
                for i in np.arange(0, len(self.current)-2):
                        previous = np.mean([self.current[i], self.current[i+1]])
                        current = np.mean([self.current[i+1], self.current[i+2]])
                        if abs(previous-current)/current>threshold:
                                self.spikes= np.append(self.spikes, [i, i+1, i+2]).astype(int)
                                self.spikes = np.unique(self.spikes)

        def removeSpikes(self):
                #remove spikes
                print("Removing the spikes")
                self.prev = np.column_stack((self.X, self.current))
                self.X = np.delete(self.X, self.spikes)
                self.current = np.delete(self.current, self.spikes )
                self.Y = np.delete(self.Y, self.spikes )
                if np.shape(self.baseline):
                    self.baseline= np.delete(self.baseline, self.spikes)



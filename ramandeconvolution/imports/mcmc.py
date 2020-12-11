#!/usr/bin/python3
#  mcmc.py
#
#  copyright 2019 dumitru duca <dumitru_duca@outlook.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the gnu general public license as published by
#  the free software foundation; either version 2 of the license, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but without any warranty; without even the implied warranty of
#  merchantability or fitness for a particular purpose. see the
#  gnu general public license for more details.
#
#  You should have received a copy of the gnu general public license
#  along with this program; if not, write to the free software
#  foundation, inc., 51 franklin street, fifth floor, boston,
#  ma 02110-1301, usa.
#
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from PyQt5.QtCore import pyqtSignal, QObject

"""
MCMC is an implementation of the Markov chain Monte Carlo method for fitting
various models.

General usage:
    mcmc = MCMC(model, guess, limits)
    result = mcmc(X, Y, nrOfSteps)

Optional:
    #changing the increment per itteration step
    mcmc.step(step)

    #if the data was normalized, a correction factor can be applied
    result = mcmc(X, Y, nrOfSteps, corr=factor)

"""
class MCMC(QObject):

    pg = pyqtSignal(int, name='pg')
    def __init__(self, model, guess, limits):
        #constructor
        super(MCMC, self).__init__()
        self.model = model #function for the model
        self.guess = guess #initial guess
        self.limits = limits #high/low limits for the fitting parameters
        self.__step = 0.1*np.ones(len(self.guess)) #maximum increment/step
        self.__chain = np.zeros(len(self.guess)) #array for the Markov chain
        self.__fit = np.zeros(len(self.guess)) #retrieved params
        self.__fit_err = np.zeros(len(self.guess)) #error of retrieved params
        self.__hpChain = 1000 #portion of the chain used to calculate params
        self.check_guess()

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, _model):
        self.__model = _model

    @property
    def guess(self):
        return self.__guess

    @guess.setter
    def guess(self, _guess):
        self.__guess = _guess

    @property
    def limits(self):
        return self.__limits

    @limits.setter
    def limits(self, _limits):
        self.__limits = _limits

    @property
    def step(self):
        return self.__step

    @step.setter
    def step(self, _step):
        if len(self.step) == len(_step):
            self.__step = _step
        else:
            raise ValueError('Shape mismatch')

    #check the initial guess
    def check_guess(self):
        for i, g in enumerate(self.guess):
            if g < self.limits[0, i] or g > self.limits[1, i]:
                raise ValueError('Bound error')

    #Probability function
    def __P (self, X, Y, theta):
        __y = self.__model(X, *theta)
        __chi2 = (__y - Y)**2
        return (-np.sum(__chi2))

    #check the new position
    def __checkBounds(self, th_curr):
        theta = np.copy(th_curr)
        for i, th in enumerate(theta):
            while th < self.limits[0, i] or th > self.limits[1, i]:
                #try a different number if out of bounds
                th = theta[i] + self.step[i]*np.random.randn()
            theta[i] = th
        return theta

    #Fitting the model
    def __call__(self, _X, _Y, steps, verbose = True, **kwargs):
        if len(_X) != len(_Y):
            raise ValueError("Shape mismatch")
        X = _X
        Y = _Y

        th_curr = self.guess
        #calculate the probability
        P_current = self.__P(X, Y, th_curr)

        #Do n steps
        for i in tqdm(range(steps)):
            #randomly draw  new proposed theta
            th_prop = th_curr + self.__step*np.random.randn(len(th_curr))

            #check that the new position is within the limits
            th_prop = self.__checkBounds(th_prop)

            #calculate the probability
            P_proposed = self.__P(X, Y, th_prop)

            try:
                #Calculate likelihood
                ratio = math.exp(P_proposed - P_current)
            except OverflowError:
                ratio = np.inf

            #Decide if to accept the new theta values
            if ratio > np.random.rand():
                th_curr, P_current = th_prop, P_proposed

            prog = i*100/steps
            if prog%1==0:
                self.pg.emit(prog)

            #save current theta value in the chain
            self.__chain = np.row_stack((self.__chain, th_curr))
        self.__chain = self.__chain[1:,:]

        #calculate the values from the chain
        self.calc_pars(verbose=verbose, **kwargs)
        return self.get_pars()

    #Plot the MCMC chains
    def plot(self, labels):
        #Plot the result
        if np.shape(self.__chain)[1] != len(labels):
            raise ValueError("Shape mismatch")

        f = plt.figure()
        ax = plt.subplot(111)
        for i, lb in enumerate(labels):
            ax.plot(self.__chain[:, i], label = lb)
        ax.legend()
        plt.show()

    #calculate the value of parameters and their errors
    def calc_pars(self, verbose=False, **kwargs):
        if "corr" in kwargs:
            corr = kwargs.get("corr").astype(float)
        else:
            corr = np.ones(len(self.__fit))

        for i in np.arange(0, np.shape(self.__chain)[1]):
            self.__fit[i] = corr[i] * np.mean(self.__chain[-self.__hpChain:, i])
            self.__fit_err[i] = corr[i] * (self.__fit[i] - np.percentile(self.__chain[-self.__hpChain:, i], 5))
        if verbose:
            for i, fit in enumerate(self.__fit):
                print("Parameter %d: " %i, fit, ' +/- ', self.__fit_err[i])

    #get the parameters
    def get_pars(self):
        return self.__fit, self.__fit_err


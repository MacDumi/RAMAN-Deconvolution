import os
import queue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
from PyQt5.QtCore import *
from recordtype import recordtype

from .fit import FIT
from .data import DATA

Limits = recordtype('Limits', ['min', 'max'])

def BatchWorker(file_queue, results_queue, error_queue, names, shape, bsParams,
                            crop, peakLimits, parguess, bounds, index, folder):
    result = pd.Series(index = index, dtype = object)
    while True:
        try:
            fname = file_queue.get_nowait()
            result.name = os.path.basename(fname)
            dt = DATA()
            try:
                dt.loadData(fname)
                dt.crop(crop.min, crop.max)
                dt.fitBaseline(bsParams, peakLimits, abs = True)

                text = ''
                for deg in np.arange(0, dt.bsDegree+1):
                    if dt.bsCoef[deg]>=0 and deg!=0:
                        text += '+'
                    text += f'{dt.bsCoef[deg]:.4E}*x^{dt.bsDegree-deg}'
                result.loc['baseline'] = text
            except Exception as e:
                print(f"Error {e}")
            try:
                fit = FIT(shape, names)
                fit.deconvolute(dt, parguess.copy(), bounds.copy(), batch=True)
                for i, pars in enumerate(zip(names, shape)):
                    result.loc[pars[0]+'_'+'Position'] = fit.pars[int(
                                                       np.sum(fit.args[:i])+2)]
                    result.loc[pars[0]+'_'+'Amplitude'] = fit.pars[int(
                                                       np.sum(fit.args[:i]))]
                    result.loc[pars[0]+'_'+'FWHM'] = fit.fwhm[i]
                    result.loc[pars[0]+'_'+'Area'] = fit.area[i]
                    if pars[1] == 'V':
                        result.loc[pars[0]+'_'+'L/G'] = fit.pars[int(
                                                       np.sum(fit.args[:i])+3)]
                    elif pars[1] == 'B':
                        result.loc[pars[0]+'_'+'1/q'] = fit.pars[int(
                                                       np.sum(fit.args[:i])+3)]
                path = folder+ '/' + os.path.basename(fname)+'.png'
                fig = plt.figure(figsize=(12,8))
                try:
                    fit.plot(figure=fig, path=path)
                except FileNotFoundError:
                    error_queue.put('MAXLENGTH')
                results_queue.put('+|+'.join(np.append(result.name,
                                                   result.values.astype(str))))
            except Exception as e:
                print(f'Could not deconvolute the {fname} file')
                result.iloc[1:len(index)] = -1*np.ones(len(index)-1)
                print(e)
        except queue.Empty:
            break

class Worker(QRunnable):
    '''
    Worker thread
    '''
    def __init__(self, fit, *args):
        super(Worker, self).__init__()

        self.fit = fit
        self.args = args

    @pyqtSlot()
    def run(self):
        '''
        Initialize the runner function with passed arguments
        '''
        try:
            self.fit.deconvolute(*self.args)
        except Exception as e:
            print('something went wrong\n', e)

class Deconvolute(QThread):

    error = pyqtSignal()
    progress = pyqtSignal(int, name='progress')
    def __init__(self, function, *args):
        QThread.__init__(self)
        self.function = function
        self.args = args

    def run(self):
        try:
            self.function(*self.args)
        except Exception as e:
            self.error.emit()
            print('Something went wrong: ', e)

class BatchDeconvolute(QThread):
    # Various sygnals
    error           = pyqtSignal(str, name='error')
    was_canceled    = pyqtSignal()
    finished        = pyqtSignal(str, name='finished')
    procs_ready     = pyqtSignal(name='ready')
    output_saved    = pyqtSignal(str, name='saved')
    progress        = pyqtSignal(int, name='progress')

    def __init__(self, _files, _parguess, _bounds, _fit, _params):
        QThread.__init__(self)
        self.files = _files
        self.parguess = _parguess
        self.bounds = _bounds
        self.cores = _params[0]
        self.bsParams = _params[1:]
        self.fit = _fit
        self.results          = Queue()
        self.files_to_process = Queue()
        self.error_queue      = Queue()
        self.cancel_job = False

    def run(self):
        # Create N processes that will read files and deconvolute them
        try:
            _min = int(self.bsParams[1])
            _max = int(self.bsParams[2])
            _crop_min = int(self.bsParams[3])
            _crop_max = int(self.bsParams[4])
        except ValueError:
            self.error.emit('Initial parameters: Value error')
            return

        self.cropLimits = Limits(_crop_min, _crop_max)
        self.peakLimits = Limits(_min, _max)

        index = ['baseline']
        for name, s in zip(self.fit.names, self.fit.shape):
            index = np.append(index, [name+'_'+nm for nm in
                                    ('Position', 'Amplitude', 'FWHM', 'Area')])
            if s == 'V':
                index = np.append(index, '_'.join((name, 'L/G')))
            elif s == 'B':
                index = np.append(index, '_'.join((name, '1/q')))
        out = pd.DataFrame(index = index)

        folder  = os.path.dirname(self.files[0])+'/BatchDeconvolution_'
        folder += f'{len(self.files)}_files'
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
            except OSError:
                err  = 'OS error|'
                err += 'Could not write to disk - Read-only file system'
                self.error.emit(err)
                self.was_canceled.emit()
                return

        arguments = [self.fit.names, self.fit.shape, self.bsParams[0],
                    self.cropLimits, self.peakLimits, self.parguess,
                    self.bounds, index, folder]

        # Fill the queue with files to process
        for file in self.files:
            self.files_to_process.put(file)

        # Create all the processes
        processes = []
        for w in range(self.cores):
            args = [self.files_to_process, self.results,
                                                  self.error_queue] + arguments
            p = Process(target=BatchWorker, args=args)
            processes.append(p)

        self.procs_ready.emit()

        # Start all the processes
        for p in processes:
            p.start()

        # Update the slider
        prog = 0
        while prog < len(self.files) and not self.cancel_job:
            if not self.results.empty():
                res = self.results.get().split('+|+')
                out[res[0]] = res[1:]
                prog += 1
                self.progress.emit(int(100*prog/len(self.files)))

        # Wait for the processes to finish
        for p in processes:
            p.join()

        if not self.cancel_job:
            # Process the results from the queue
            out.to_csv(folder+'/BatchDeconvolutionResults.csv')
            self.saved.emit(folder+'/BatchDeconvolutionResults.csv')
            if list(out.iloc[2]).count(-1) >0:
                pb = list(out.columns[out.iloc[0] == -1])
                self.finished.emit('<|>'.join(pb))
            else:
                self.finished.emit('OK')
        else:
            self.was_canceled.emit()
        if not self.error_queue.empty():
            err  = 'MAX_PATH error|'
            err += 'Please disable the 260 char limit or use a shorter name'
            self.error.emit(err)

    def cancel(self):
        # Cancel the batch deconvolution
        self.cancel_job = True
        while True:
            # Remove all the remaining files from the queue
            try:
                self.files_to_process.get_nowait()
            except queue.Empty:
                break

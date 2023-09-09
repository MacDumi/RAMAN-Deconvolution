#!/usr/bin/python
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, os
import ntpath
import configparser
import copy
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from recordtype import recordtype
from distutils.util import strtobool
import itertools
import multiprocessing as mp
from multiprocessing import cpu_count, Process, Queue
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from imports.gui import Ui_MainWindow as main_gui
from imports.dialogs import *
from imports.fit import FIT
from imports.data import DATA
from imports.workers import *
from imports.convertwdf import *

Limits = recordtype('Limits', ['min', 'max'])

class RD(QMainWindow, main_gui):

    header  = 50*'*'+'\n\n'+10*' '+'DECONVOLUTION OF RAMAN SPECTRA'
    header += 10*''+'\n\n'+50*'*'

    def __init__(self):
        super(RD, self).__init__()
        self.setupUi(self)
        #variables
        self.data = DATA()
        self.fit = 0
        self.changed = False
        self.peaksLimits = Limits(900, 1850)
        self.spike = 0 #scatter plot for spikes
        self.limit_low, self.limit_high = 0, 0 #exclude region
        self.baseline = 0
        self.path =os.path.dirname(os.path.realpath(__file__))

        self.actionQuit.triggered.connect(self.close)
        self.actionAbout.triggered.connect(self.about)
        self.actionGuess.triggered.connect(
                lambda: [self.dockGuess.setVisible(
                                              self.actionGuess.isChecked()),
                                                   self.dockGuess.raise_()])
        self.actionOutput.triggered.connect(
                lambda: [self.dockOut.setVisible(
                                              self.actionOutput.isChecked()),
                                                      self.dockOut.raise_()])
        self.dockGuess.visibilityChanged.connect(self.actionGuess.setChecked)
        self.dockOut.visibilityChanged.connect(self.actionOutput.setChecked)
        self.actionToolbar.triggered.connect(
                                     self.toolBar.toggleViewAction().trigger)
        self.actionNew.triggered.connect(self.New)
        self.actionSave.triggered.connect(self.Save)
        self.actionCrop.triggered.connect(self.Crop)
        self.actionRemove_Baseline.triggered.connect(self.Baseline)
        self.actionSpike_Removal.triggered.connect(self.removeSpikes)
        self.actionSmoothing.triggered.connect(self.Smoothing)
        self.actionDeconvolute.triggered.connect(self.Deconvolution)
        self.actionDeconvolute_MCMC.triggered.connect(self.DeconvMCMC)
        self.actionBatch_deconvolution.triggered.connect(self.BatchDeconv)
        self.actionLoadGuess.triggered.connect(self.LoadGuess)
        self.actionLoad_defaults.triggered.connect(lambda: self.LoadGuess(
                                fname=self.path+'/config/initialData.csv'))
        self.actionExportGuess.triggered.connect(self.ExportGuess)
        self.tabifyDockWidget(self.dockGuess, self.dockOut)
        self.tableWidget.horizontalHeader().setSectionResizeMode(
                                    QHeaderView.ResizeMode.ResizeToContents)
        self.tableWidget.itemChanged.connect(self.itemCheck)
        self.textOut.setReadOnly(True)
        self.textOut.setText(self.header)
        self.readConfig()
        self.dockGuess.raise_()


        self.spikeDialog = SpikeDialog(self.threshold)
        self.spikeDialog.slider.sliderReleased.connect(self.showSpikes)

        self.smoothDialog = SmoothDialog()
        self.smoothDialog.slider.sliderReleased.connect(self.previewSmoothed)

        self.figure = plt.figure()
        self.figure.set_tight_layout(True)
        self.subplot = self.figure.add_subplot(111) #add a subfigure
        #add widget
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.ClickFocus )
        self.canvas.setFocus()
        self.canvas.mpl_connect('button_press_event', self.onclick)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.gridLayout.addWidget(self.toolbar)
        self.gridLayout.addWidget(self.canvas)

        actionOpen     = QAction(QIcon("graphics/open.png"), "Open", self)
        actionSave     = QAction(QIcon("graphics/save.png"), "Save", self)
        actionCrop     = QAction(QIcon("graphics/crop.png"), "Crop", self)
        actionSpike    = QAction(QIcon("graphics/spike.svg"),
                                                         "Spike removal", self)
        actionSmooth   = QAction(QIcon("graphics/smooth.svg"),
                                                             "Smoothing", self)
        actionBaseline = QAction(QIcon("graphics/baseline.png"),
                                                        "Remove baseline",self)
        actionDeconv   = QAction(QIcon("graphics/deconv.svg"),
                                                         "Deconvolution", self)
        actionMCMC     = QAction(QIcon("graphics/mcmc.png"),
                                              "Deconvolution\nwith MCMC", self)
        actionBatch     = QAction(QIcon("graphics/batch.png"),
                                                   "Batch deconvolution", self)
        self.toolBar.addAction(actionOpen)
        self.toolBar.addAction(actionSave)
        self.toolBar.addSeparator()
        self.toolBar.addAction(actionCrop)
        self.toolBar.addAction(actionBaseline)
        self.toolBar.addAction(actionSpike)
        self.toolBar.addAction(actionSmooth)
        self.toolBar.addSeparator()
        self.toolBar.addAction(actionDeconv)
        self.toolBar.addAction(actionMCMC)
        self.toolBar.addSeparator()
        self.toolBar.addAction(actionBatch)
        self.toolBar.addSeparator()
        self.toolBar.toggleViewAction().setChecked(
                                                self.actionToolbar.isChecked())
        actionOpen.triggered.connect(self.New)
        actionCrop.triggered.connect(self.Crop)
        actionBaseline.triggered.connect(self.Baseline)
        actionSpike.triggered.connect(self.removeSpikes)
        actionSave.triggered.connect(self.Save)
        actionSmooth.triggered.connect(self.Smoothing)
        actionDeconv.triggered.connect(self.Deconvolution)
        actionMCMC.triggered.connect(self.DeconvMCMC)
        actionBatch.triggered.connect(self.BatchDeconv)

        self.startUp()

        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested.connect(
                                                    self.tableItemRightClicked)



    def startUp(self):
        # Generate dummy data to plot at start-up
        X = np.arange(800, 2000, 2)
        Y1 = FIT.lorents(X, 1800, 1358, 110)
        Y2 = FIT.lorents(X, 1700, 1590, 50)
        self.subplot.plot(X, (Y1+Y2)+20*np.random.randn(len(X)), 'o')
        self.subplot.plot(X, Y1+Y2, 'r--', label = 'Sample data')
        self.subplot.legend()
        self.subplot.plot(X, Y1)
        self.subplot.plot(X, Y2)
        self.subplot.set_xlim(np.min(X), np.max(X))
        self.plotAdjust()

    def removeSpikes(self):
        # Detect and remove spikes in the data (current data)
        if not np.shape(self.data.X):
            self.errorBox('There is no data', 'No data...')
            return
        self.Plot(self.data.X, self.data.current, 'Experimental data')
        self.showSpikes()
        result = self.spikeDialog.exec_()
        if not result:
            self.data.spikes = []
            if self.spike != 0:
                self.spike.remove()
                self.spike = 0
                self.subplot.legend()
                self.canvas.draw()
            return
        self.data.removeSpikes()
        self.statusbar.showMessage(
                           f"{len(self.data.spikes)} datapoints removed", 2000)
        self.Plot(self.data.X, self.data.current, "Experimental data",
                                                                    clear=True)
        if not self.changed:
            self.changed = True
            self.setWindowTitle(self.windowTitle()+'*')

    def Smoothing(self):
        # Smooth the current data according to the user defined parameters
        if not np.shape(self.data.X):
            self.errorBox('There is no data', 'No data...')
            return
        self.Plot(self.data.X, self.data.current, "Experimental data")
        self.previewSmoothed()
        result = self.smoothDialog.exec_()
        self.baseline.remove()
        self.baseline = 0
        if not result:
            self.subplot.legend()
            self.canvas.draw()
            return
        self.data.prev = np.column_stack((self.data.X, self.data.current))
        self.data.current = self.data.smooth(self.data.current,
                                        2*self.smoothDialog.slider.value()+1)

        self.Plot(self.data.X, self.data.current, "Smoothed data")
        self.plotAdjust()
        if not self.changed:
            # If the current data was changed add '*' to the title
            self.changed = True
            self.setWindowTitle(self.windowTitle()+'*')

    def previewSmoothed(self):
        # Plot a preview of the smoothed data
        y_ = self.data.smooth(self.data.current,
                                        2*self.smoothDialog.slider.value()+1)
        if self.baseline != 0:
            self.baseline.remove()
        self.baseline,  = self.subplot.plot(self.data.X, y_, 'r--',
                                                        label='Smoothed data')
        self.subplot.legend()
        self.canvas.draw()
        return y_

    def prepDataForDeconv(self, **kwargs):
        # Prepare the data for deconvolution
        # the function first checks if the data is present and that
        # the baseline was removed
        # fitting parameters are then taken from the table and put in a list
        batch = kwargs.get('batch', False)
        if not np.shape(self.data.X) and not batch:
            self.errorBox('There is no data', 'No data...')
            return -1, -1, -1
        if not np.shape(self.data.baseline) and not batch:
            result = QMessageBox.question(self,
                    "Baseline...",
                    "The baseline is still present!\nRemove it first?",
                    QMessageBox.Yes| QMessageBox.No)
            if result == QMessageBox.Yes:
                res = self.Baseline()
                if res == -1:
                    return -1, -1, -1
            else:
                return -1, -1, -1
        rows = self.tableWidget.rowCount()
        cols = self.tableWidget.columnCount()
        params = pd.DataFrame(columns=['lb', 'pos', 'pos_min', 'pos_max',
                       'int', 'int_min', 'int_max', 'width', 'width_min',
                                                        'width_max', 'shape'])
        for row in range(rows):
            if self.tableWidget.item(row, 0).checkState() & Qt.Checked:
                error = [self.itemCheck(self.tableWidget.item(row, col),
                                     silent=True) for col in range(1, cols-1)]
                if True in error:
                    self.errorBox("Bad parameters", "Bad values")
                    self.dockGuess.raise_()
                    return -1, -1, -1
                d =[self.tableWidget.item(row, col).text() for col in
                                                             range(1, cols-1)]
                p_shape = self.tableWidget.cellWidget(row,
                                              cols-1).currentText().strip()[0]
                params = params.append({'lb':d[0], 'pos':d[1], 'pos_min':d[2],
                   'pos_max':d[3], 'int':d[4], 'int_min':d[5], 'int_max':d[6],
                             'width':d[7], 'width_min':d[8], 'width_max':d[9],
                                         'shape':p_shape}, ignore_index =True)
        self.fit = FIT(params['shape'], params['lb'])
        parguess = np.concatenate([[params['int'][i], params['width'][i],
                                                             params['pos'][i]]
                    for i in range(len(params['lb'])) ]).astype(float)
        lower = np.concatenate([[params['int_min'][i],
                                 params['width_min'][i],
                                 params['pos_min'][i]] for i in
                                     range(len(params['lb'])) ]).astype(float)
        upper = np.concatenate([[params['int_max'][i],
                                 params['width_max'][i],
                                 params['pos_max'][i]] for i in
                                      range(len(params['lb']))]).astype(float)
        return parguess, lower, upper

    def Deconvolution(self):
        #deconvolute the data in a different thread
        parguess, lower, upper = self.prepDataForDeconv()
        if not np.shape(parguess):
            return
        bounds = [lower, upper]
        progress = QProgressDialog("Processing...", "Cancel", 0, 100)
        progress.setMinimumWidth(300)
        progress.setWindowTitle("Deconvolution")
        progress.setWindowIcon(QIcon('graphics/icon.svg'))
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(5)
        self.error = False
        self.deconvolutionThread = Deconvolute(self.fit.deconvolute, self.data,
                                                       parguess, bounds, False)
        progress.canceled.connect(self.deconvolutionThread.exit)
        self.deconvolutionThread.finished.connect(
                     lambda: [progress.setValue(100), self.plotDeconvResult()])
        self.deconvolutionThread.error.connect(
                                               lambda: [progress.setValue(100),
                                         self.errorBox("Wrong parameters...")])
        self.deconvolutionThread.start()

    def DeconvMCMC(self):
        parguess, lower, upper = self.prepDataForDeconv()
        if not np.shape(parguess):
            return
        bounds = [lower, upper]
        progress = QProgressDialog("Processing...", "Cancel", 0, 100)
        progress.setMinimumWidth(300)
        progress.setWindowTitle("MCMC deconvolution")
        progress.setWindowIcon(QIcon('graphics/icon.svg'))
        progress.setCancelButton(None)
        progress.show()
        progress.setValue(0)
        self.error = False
        self.deconvolutionThread = Deconvolute(self.fit.decMCMC, self.data,
                                                     parguess, bounds, 10000)
        self.fit.pg.connect(progress.setValue)
        progress.canceled.connect(self.deconvolutionThread.exit)
        self.deconvolutionThread.finished.connect(
                   lambda: [progress.setValue(100), self.plotDeconvResult()])
        self.deconvolutionThread.error.connect(
                                              lambda: [progress.setValue(100),
                                        self.errorBox("Wrong parameters...")])
        self.deconvolutionThread.start()

    def BatchDeconv(self):
        if self.changed:
            result = QMessageBox.question(self,
                    "Unsaved file...",
                    "Do you want to save the data before exiting?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if result == QMessageBox.Yes:
                self.Save()
            elif result == QMessageBox.Cancel:
                return
        self.changed = False
        path = self.initialDir
        filters = ["Text files (*.txt *.dat)",
                              "Wire data files (*.wdf)","All files (*.*)"]
        fnames, _filter = QFileDialog.getOpenFileNames(self, 'Open files',
                                                 path, ';;'.join(filters))
        if not fnames:
            return
        elif self.initialDir!=ntpath.dirname(fnames[0]):
            # Update the initial directory for the Open/Save dialog
            self.initialDir=ntpath.dirname(fnames[0])
            self.settings.setValue('directory', self.initialDir)
        self.setWindowTitle( 'Raman Deconvolution - Batch Deconvolution')
        if _filter == 'Wire data files (*.wdf)':
            for i, fname in enumerate(fnames):
                convert(fname)
                fnames[i] = fname[:-3]+'txt'
        fnames.sort()
        dialog = BatchDialog(len(fnames), self.degree, self.peakLimits.min,
                               self.peakLimits.max, 700, 2500, cpu_count())
        result = dialog.exec_()
        if not result:
            return
        params = dialog.getData()

        parguess, lower, upper = self.prepDataForDeconv(batch=True)
        if not np.shape(parguess):
            return
        bounds = [lower, upper]
        progress = QProgressDialog(f"Creating {params[0]} processes",
                                                        "Cancel", 0, 100)
        progress.setMinimumWidth(300)
        progress.setWindowIcon(QIcon('graphics/icon.svg'))
        progress.setWindowTitle("Batch deconvolution")
        progress.show()
        progress.setValue(0)
        self.batch = BatchDeconvolute(fnames, parguess, bounds,
                                                        self.fit, params)

        progress.canceled.connect(self.batch.cancel)
        self.textOut.clear()
        self.subplot.clear()

        self.batch.progress.connect(progress.setValue)
        self.batch.procs_ready.connect(
                            lambda: progress.setLabelText("Deconvoluting"))
        self.batch.finished.connect(
                                         lambda x: [progress.setValue(100),
                     self.errorBox('Some files were not deconvoluted...\n'+
                           '\n'.join(x.split('<|>')))
                                        if x!='OK' else print('all done')])
        self.batch.error.connect(
                 lambda x: self.errorBox(x.split('|')[1], x.split('|')[0]))
        self.batch.was_canceled.connect(
                   lambda: [progress.setValue(100), print("job canceled")])
        self.batch.saved.connect(
                lambda x:[self.textOut.append(f'Results are saved at {x}'),
                                                    self.dockOut.raise_()])
        self.batch.start()


    def plotDeconvResult(self):
        if self.error or self.fit.error:
            return
        self.statusbar.showMessage("Thread finished succesfully", 1000)
        self.Plot(self.data.X, self.data.current, "Experimental data")
        labels = self.fit.names
        [self.subplot.plot(self.data.X, self.fit.peaks[lb], label = lb) for
                                                              lb in labels]
        self.subplot.plot(self.data.X, self.fit.peaks['cumulative'], 'r--',
                                                      label = 'cumulative')
        self.plotAdjust()


        self.textOut.append(self.fit.report)
        self.dockOut.raise_()

    def showSpikes(self):
        self.data.detectSpikes(self.spikeDialog.slider.value())
        if self.spike != 0:
            self.spike.remove()
            self.spike = 0
            self.canvas.draw()
        sp = self.data.spikes
        if len(sp):
            self.spike, = self.subplot.plot(self.data.X[sp],
                            self.data.current[sp], 'ro', label = 'Spikes')
            self.subplot.legend()
            self.canvas.draw()

    def plotAdjust(self):
        self.subplot.set_xlabel(r'$\mathbf{Raman\ shift,\ cm^{-1}}$')
        self.subplot.set_ylabel(r'$\mathbf{Intensity}$')
        self.subplot.legend()
        self.canvas.draw()

    def onclick(self, event):
        # Right click on the plot
        if  np.shape(self.data.X):
            if event.button == 3:  #right click
                self.listMenu= QMenu()
                menu_item_0 = self.listMenu.addAction(
                                                    "Delete datapoint (spike)")
                idx = np.abs(self.data.X - event.xdata).argmin()
                self.statusbar.showMessage(
                        'Datapoint selected: X = %f, Y = %f'%(self.data.X[idx],
                                                 self.data.current[idx]), 1000)
                spike, = self.subplot.plot(self.data.X[idx],
                                                  self.data.current[idx], 'rs',
                                                  label = 'Selected datapoint')
                self.subplot.legend()
                self.canvas.draw()
                cursor = QCursor()
                menu_item_0.triggered.connect(lambda: self.deleteSpike(idx))
                self.listMenu.move(cursor.pos() )
                self.listMenu.show()
                self.listMenu.aboutToHide.connect(lambda : (spike.remove(),
                                                          self.canvas.draw()))

    def deleteSpike(self, x):
        self.statusbar.showMessage(
                'Spike deleted: X = %f, Y = %f'%(self.data.X[x],
                                                   self.data.current[x]), 1000)
        self.data.spikes = x
        self.data.removeSpikes()
        self.Plot(self.data.X, self.data.current, "Experimental data")
        if not self.changed:
            self.changed = True
            self.setWindowTitle(self.windowTitle()+'*')

    def Baseline(self):
        if not np.shape(self.data.X):
            self.errorBox('There is no data', 'No data...')
            return -1
        if np.shape(self.data.baseline):
            self.data.setData(self.data.X, self.data.Y.copy())
        self.Plot(self.data.X, self.data.current, 'Experimental data')

        if self.peakLimits.min <= self.data.X[0]:
            self.peakLimits.min = round(self.data.X[0] + 0.25 * (
                                               self.data.X[-1]-self.data.X[0]))
        if self.peakLimits.max >= self.data.X[-1]:
            self.peakLimits.max = round(self.data.X[-1] - 0.25 * (
                                               self.data.X[-1]-self.data.X[0]))
        dialog = BaselineDialog(self.degree, self.peakLimits.min,
                                                           self.peakLimits.max)
        dialog.btPreview.clicked.connect(lambda: self.previewBaseline(dialog))
        self.previewBaseline(dialog)
        result = dialog.exec_()
        self.limit_low.remove()
        self.limit_high.remove()
        self.baseline.remove()
        self.limit_low, self.limit_high = 0, 0
        self.baseline = 0
        if not result:
            self.subplot.legend()
            self.canvas.draw()
            return -1
        params = dialog.getData()
        try:
            _min = int(params[2])
        except ValueError:
            self.statusbar.showMessage(
                                      "Wrong value...setting to default", 3000)
        try:
            _max = int(params[3])
        except ValueError:
            self.statusbar.showMessage(
                                      "Wrong value...setting to default", 3000)
        self.peakLimits = Limits(_min, _max)
        self.data.fitBaseline(params[1], self.peakLimits, abs = True)
        self.Plot(self.data.X, self.data.current, "Baseline corrected data")
        self.plotAdjust()
        self.dockOut.setVisible(True)
        self.textOut.clear()
        self.textOut.append('                  BASELINE FIT                  ')
        self.textOut.append('************************************************')
        self.textOut.append(f'Polynomial fit -- degree: {self.data.bsDegree}')
        self.textOut.append('Fitting equation:')
        text = ''
        for deg in np.arange(0, self.data.bsDegree+1):
            if self.data.bsCoef[deg]>=0 and deg!=0:
                text += '+'
            text += '{:.4E}*x^{}'.format(self.data.bsCoef[deg],
                                                        self.data.bsDegree-deg)
        self.textOut.append(text + '\n')
        self.dockOut.raise_()
        if not self.changed:
            self.changed = True
            self.setWindowTitle(self.windowTitle()+'*')

    def previewBaseline(self, dialog):
        dt = DATA()
        dt.setData(self.data.X, self.data.current)
        try:
            _min = int(dialog.lineEdit_min.text())
            _max = int(dialog.lineEdit_max.text())
        except ValueError:
            self.statusbar.showMessage(
                                      "Wrong value...setting to default", 3000)
            _min = self.peakLimits.min
            _max = self.peakLimits.max
        if self.baseline != 0:
            self.limit_low.remove()
            self.limit_high.remove()
            self.baseline.remove()
        self.limit_low,  = self.subplot.plot([_min, _min],
                        [np.min(self.data.current), np.max(self.data.current)],
                                       color = 'red', label = 'Exclude region')
        self.limit_high, = self.subplot.plot([_max, _max],
                        [np.min(self.data.current), np.max(self.data.current)],
                                                                 color = 'red')

        peakLimits = Limits(_min, _max)
        dt.fitBaseline(dialog.spinBox.value(), peakLimits, abs = True)
        self.baseline, = self.subplot.plot(dt.X, dt.baseline, 'r--',
                                                            label = "Baseline")
        self.subplot.legend()
        self.canvas.draw()
        del dt


    def LoadGuess(self, **kwargs):
        # Load the initial guess
        if 'fname' in kwargs:
            fname = kwargs['fname']
        else:
            path = self.initialDir
            fname, _filter = QFileDialog.getOpenFileName(self, 'Open file',
                       path,"Comma separated values (*.csv);; All files (*.*)")
            if not fname:
                return
        try:
            parameters = pd.read_csv(fname)
            self.tableWidget.setRowCount(0)
            cols = ['labels', 'freq', 'freq_min', 'freq_max' ,'intens',
                    'intens_min', 'intens_max', 'width', 'width_min',
                                                                'width_max']
            shape = ['Lorentzian', 'Gaussian', 'Voigt', 'BWF']
            for i, l in enumerate(parameters['labels']):
                self.tableWidget.insertRow(i)
                self.tableWidget.setItem(i, 0, QTableWidgetItem(''))
                self.tableWidget.item(i, 0).setFlags(
                                     Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                self.tableWidget.item(i, 0).setCheckState(Qt.Checked)
                for j, col in enumerate(cols[::-1]):
                    self.tableWidget.setItem(i, len(cols)-j,
                                     QTableWidgetItem(str(parameters[col][i])))

                combo = QComboBox()
                [combo.addItem('  '+s) for s in shape]
                if parameters['shape'][i] in [s[0] for s in shape]:
                    combo.setCurrentIndex([s[0] for
                                     s in shape].index(parameters['shape'][i]))
                self.tableWidget.setCellWidget(i, len(cols)+1, combo)

            self.statusbar.showMessage("Initial parameters were loaded", 2000)
        except FileNotFoundError:
            self.errorBox('File not found', 'Parameters were not loaded')
        except KeyError:
            self.errorBox('Wrong file format', "Parameters were not loaded")
        except Exception as e:
            self.errorBox('Error\n{}'.format(e), 'Parameters were not loaded')

        error = []
        cols = self.tableWidget.columnCount()
        silent = True
        for row in range(self.tableWidget.rowCount()):
            error = np.append(error, [self.itemCheck(
                                  self.tableWidget.item(row, col), silent=True)
                                                  for col in range(1, cols-1)])

    def ExportGuess(self):
        # Save the initial guess
        path = self.initialDir
        fname, _filter = QFileDialog.getSaveFileName(self, 'Save file',
                                                  path,"Initial guess (*.csv)")

        if not fname:
            return
        cols = ['labels', 'freq', 'freq_min', 'freq_max' ,'intens', 'width']
        rows = self.tableWidget.rowCount()
        guess = pd.DataFrame()
        try:
            for i, c in enumerate(cols):
                guess[c] = [self.tableWidget.item(row, i+1).text()
                                                        for row in range(rows)]
            guess['shape'] = [
                  self.tableWidget.cellWidget(row, 7).currentText().strip()[0]
                                                        for row in range(rows)]
            guess.to_csv(fname, index=None)
            self.statusbar.showMessage(
                                 "Initial guess file saved succesfully", 2000)
        except Exception as e:
            self.errorBox('Error\n{}'.format(e), 'Parameters were not loaded')

    def readConfig(self):
        # Read the config file
        self.settings = QSettings("Raman Deconvolution")
        self.initialDir = self.settings.value('directory', './')
        self.restoreGeometry(self.settings.value(
                                   'MainWindow/geometry', self.saveGeometry()))
        self.restoreState(self.settings.value(
                                         'MainWindow/state', self.saveState()))
        self.actionToolbar.setChecked(strtobool(
                            self.settings.value('MainWindow/toolbar', 'true')))

        # Read the configuration file
        self.config = configparser.ConfigParser()
        if len(self.config.read(self.path+'/config/config.ini')):
               self.degree = int(self.config['DEFAULT']['degree'])
               self.threshold = int(self.config['DEFAULT']['threshold'])
               self.peakLimits = Limits(int(self.config['PEAK']['low']),
                                              int(self.config['PEAK']['high']))
        # Load fitting parameters
        self.LoadGuess(fname = self.path + '/config/initialData.csv')

    def New(self):
        # Load a new file
        if self.changed:
            result = QMessageBox.question(self,
                    "Unsaved file...",
                    "Do you want to save the data before exiting?",
                    QMessageBox.Yes| QMessageBox.No |QMessageBox.Cancel)
            if result == QMessageBox.Yes:
                self.Save()
            elif result == QMessageBox.Cancel:
                return
        path = self.initialDir
        filters = ["Text files (*.txt *.dat)",
                              "Wire data files (*.wdf)","All files (*.*)"]
        fname, _filter = QFileDialog.getOpenFileName(self, 'Open file', path,
                                                          ';;'.join(filters))
        if not fname:
            return
        elif self.initialDir!=ntpath.dirname(fname):
            # Update the initial directory for the Open/Save dialog
            self.initialDir=ntpath.dirname(fname)
            self.settings.setValue('directory', self.initialDir)
        if fname[-3:] == 'wdf':
            convert(fname)
            fname = fname[:-3]+'txt'
        try:
            self.data.loadData(fname)
            self.Plot(self.data.X, self.data.Y, 'Experimental data')
            self.setWindowTitle(
                    f'Raman Deconvolution - {ntpath.basename(fname)}')
            self.changed = False
            self.textOut.clear()
            self.textOut.setText(self.header)
        except Exception as e:
            self.errorBox(f'Could not load the file\n{e}', 'I/O error')
            self.statusbar.showMessage("Error loading the file", 2000)

    def Save(self):
        if not np.shape(self.data.X):
            self.statusbar.showMessage("No data...", 2000)
            return
        path = self.initialDir
        fname, _filter = QFileDialog.getSaveFileName(self, 'Save file', path,
                       "Text files (*.txt);; Comma separated values (*.csv)")
        delimiter = '\t'
        if not fname:
            return
        if _filter == "Text files (*.txt)" and fname[-3:]!='txt':
            fname += '.txt'
        if _filter == "Comma separated values (*.csv)" and fname[-3:]!='csv':
            fname += '.csv'
            delimiter = ','

        data, comments = self.data.getData()
        if hasattr(self.fit, 'report'):
            comments += self.fit.report
            data = pd.concat([data, self.fit.peaks],
                                                   ignore_index=True, axis = 1)
        f = open(fname, 'w')
        f.close()
        with open(fname, 'a') as f:
            [f.write(s) for s in comments]
            data.to_csv(f, index = None, sep = delimiter)

        self.statusbar.showMessage('File {} saved'.format(fname), 3000)


        if self.changed:
            self.changed =False
            self.setWindowTitle(self.windowTitle()[:-1])

    def Plot(self, X, Y, label, clear = True, limits=False):
        if clear:
            self.subplot.clear()
        if label == 'Baseline':
            line, = self.subplot.plot(X, Y, 'r--', label = label)
        else:
            line, = self.subplot.plot(X, Y, label = label)
        self.subplot.set_xlim(np.min(X), np.max(X))
        if limits:
            self.subplot.set_ylim(0.9 * np.min(Y), 1.1 * np.max(Y))
        self.plotAdjust()
        return line

    def errorBox(self, message, title="Error"):
        self.error = True
        self.statusbar.showMessage(message, 5000)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText(message)
        msg.setWindowTitle(title)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def Crop(self):
        if not np.shape(self.data.X):
            self.errorBox('There is no data', 'No data...')
            return
        _min = np.min(self.data.X)
        _max = np.max(self.data.X)
        dialog = CropDialog(_min, _max)
        result = dialog.exec_()
        params = dialog.getData()
        try:
            _min = int(params[0])
        except ValueError:
            self.statusbar.showMessage(
                                      "Wrong value...setting to default", 3000)
        try:
            _max = int(params[1])
        except ValueError:
            self.statusbar.showMessage(
                                      "Wrong value...setting to default", 3000)
        self.data.crop(_min, _max)
        self.statusbar.showMessage("Data cropped", 3000)
        self.Plot(self.data.X, self.data.current, "Experimental data")
        if self.peakLimits.min < self.data.X[0]:
            self.peakLimits.min = self.data.X[0]
        if self.peakLimits.max > self.data.X[-1]:
            self.peakLimits.max = self.data.X[-1]
        if not self.changed:
            self.changed = True
            self.setWindowTitle(self.windowTitle()+'*')

    def tableItemRightClicked(self, QPos):
        self.listMenu= QMenu()
        menu_item_2 = self.listMenu.addAction("Preview band")
        menu_item_3 = self.listMenu.addAction("Preview selected bands")
        self.listMenu.addSeparator()
        if self.tableWidget.item(
                   self.tableWidget.currentRow(), 0).checkState() & Qt.Checked:
            menu_item_0 = self.listMenu.addAction(
                                            "Don't use for deconvolution")
        else:
            menu_item_0 = self.listMenu.addAction("Use for deconvolution")
        menu_item_1 = self.listMenu.addAction("Use all")
        self.listMenu.addSeparator()
        menu_item_4 = self.listMenu.addAction("Add row")
        menu_item_5 = self.listMenu.addAction("Remove row")
        menu_item_0.triggered.connect(lambda:
                              self.tableRowUse(self.tableWidget.currentRow()))
        menu_item_1.triggered.connect(self.tableUseAll)
        menu_item_2.triggered.connect(lambda:
                              self.previewBand(self.tableWidget.currentRow()))
        menu_item_3.triggered.connect(self.previewAll)
        menu_item_4.triggered.connect(self.addRow)
        menu_item_5.triggered.connect(lambda:
                     self.tableWidget.removeRow(self.tableWidget.currentRow()))
        parentPosition = self.tableWidget.mapToGlobal(QPoint(0, 0))
        self.listMenu.move(parentPosition + QPos)
        self.listMenu.show()

    def addRow(self):
        rows = self.tableWidget.rowCount()
        self.tableWidget.insertRow(rows)
        self.tableWidget.setItem(rows, 0, QTableWidgetItem(''))
        self.tableWidget.item(rows, 0).setFlags(
                                    Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        self.tableWidget.item(rows, 0).setCheckState(Qt.Checked)
        [self.tableWidget.setItem(rows, j, QTableWidgetItem(str('--')))
                                                         for j in range(1, 11)]

        combo = QComboBox()
        [combo.addItem('  ' + s) for s
                                 in ['Lorentzian', 'Gaussian', 'Voigt', 'BWF']]
        self.tableWidget.setCellWidget(rows, 11, combo)

    def itemCheck(self, item, silent=False, **kwargs):
        col = item.column()
        value = item.text()
        error = False
        bg = self.tableWidget.item(0,0).background()
        if col in range(2, 11):
            try:
                value = float(value)
            except ValueError:
                error = True
                if not silent:
                    self.statusbar.showMessage('Not a valid number', 2000)
        elif col == 11:
            if value not in ['Gaussian', 'Lorentzian', 'Voigt', 'BWF']:
                error = True
                if not silent:
                    self.statusbar.showMessage('Unknown shape', 2000)

        idx = {'between' : {2: (3,4), 5:(6, 7), 8:(9, 10)},
               'lower'   : {3: 2, 6: 5, 9: 8},
               'higher'  : {4: 2, 7: 5, 10:8}}

        # Check if the value is between min and max
        if col in idx['between'].keys():
            min_val = float(self.tableWidget.item(item.row(),
                                              idx['between'][col][0]).text())
            max_val = float(self.tableWidget.item(item.row(),
                                              idx['between'][col][1]).text())
            if value < min_val or value > max_val:
                error = True

        # Check if the min is smaller than the value
        try:
            if col in idx['lower'].keys():
                if value > float(self.tableWidget.item(item.row(),
                                                      idx['lower'][col]).text()):
                    error = True
            # Check if the max is larger than the value
            if col in idx['higher'].keys():
                if value < float(self.tableWidget.item(item.row(),
                                                     idx['higher'][col]).text()):
                    error = True
        except AttributeError:
            # Should only happen when the table is being populated
            pass
        if error:
            bg = QBrush(Qt.red)
        item.setBackground(bg)
        item.setSelected(False)
        return error


    def previewBand(self, row):
        if not np.shape(self.data.X):
            self.errorBox('There is no data', 'No data...')
            return
        cols = self.tableWidget.columnCount()
        params =[self.tableWidget.item(row, col).text()
                                                  for col in range(1, cols-1)]
        shape = self.tableWidget.cellWidget(row,
                                              cols-1).currentText().strip()[0]
        label = params[0]
        params = np.asarray([params[4], params[5], params[1]]).astype(float)
        if shape == 'V':
            params = np.append(params, 0.5)
        elif shape == 'B':
            params = np.append(params, 0.02)
        self.Plot(self.data.X, self.data.current, "Experimental data")
        self.subplot.plot(self.data.X, FIT.Peak(FIT, self.data.X, *params,
                                                 shape = shape), label = label)
        self.subplot.legend()
        self.canvas.draw()

    def previewAll(self):
        if not np.shape(self.data.X):
            self.errorBox('There is no data', 'No data...')
            return
        self.subplot.clear()
        self.subplot.plot(self.data.X, self.data.current,
                                                   label = 'Experimental data')
        cols = self.tableWidget.columnCount()
        rows = self.tableWidget.rowCount()
        cumulative = np.zeros(len(self.data.X))
        intensity = [
             float(self.tableWidget.item(row, 5).text()) for row in range(rows)
                      if self.tableWidget.item(row, 0).checkState()&Qt.Checked]
        norm = max(self.data.current)/max(intensity)
        for row in range(rows):
            if self.tableWidget.item(row, 0).checkState() & Qt.Checked:
                error = [self.itemCheck(self.tableWidget.item(row, col),
                                      silent=True) for col in range(1, cols-1)]
                if True in error:
                    self.errorBox("Bad parameters", "Bad values")
                    return
                params =[float(self.tableWidget.item(row, col).text()) for col
                                                                  in [5, 6, 2]]
                params[0] *= norm
                label = self.tableWidget.item(row, 1).text()
                shape = self.tableWidget.cellWidget(
                                               row, 7).currentText().strip()[0]
                if shape == 'V':
                    params = np.append(params, 0.5)
                elif shape == 'B':
                    params = np.append(params, 0.02)
                p = FIT.Peak(FIT, self.data.X, *params, shape = shape)
                cumulative += p
                self.subplot.plot(self.data.X, p, label = label)
        self.subplot.plot(self.data.X, cumulative, 'r--', label = "Cumulative")
        self.plotAdjust()


    def tableUseAll(self):
        [self.tableWidget.item(row, 0).setCheckState(Qt.Checked) for row in
                                            range(self.tableWidget.rowCount())]

    def tableRowUse(self, row):
        cols = self.tableWidget.columnCount()
        if self.tableWidget.item(row, 0).checkState() & Qt.Checked:
            self.tableWidget.item(row, 0).setCheckState(Qt.Unchecked)
        else:
            self.tableWidget.item(row, 0).setCheckState(Qt.Checked)



    def about(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Version v1.02α \nMade by MacDumi \nLille, 2021")
        msg.setWindowTitle("About")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def closeEvent(self, event):

        self.settings.setValue('MainWindow/geometry', self.saveGeometry())
        self.settings.setValue('MainWindow/state', self.saveState())
        self.settings.setValue('MainWindow/toolbar',
                                           self.actionToolbar.isChecked())
        if self.changed:
            result = QMessageBox.question(self,
                    "Unsaved file...",
                    "Do you want to save the data before exiting?",
                    QMessageBox.Yes| QMessageBox.No |QMessageBox.Cancel)
            if result == QMessageBox.Yes:
                self.Save()
                event.ignore()
            elif result == QMessageBox.Cancel:
                event.ignore()
            else:
                event.accept()

if __name__ == '__main__':
    # Set the start process type to spawn (the only thing on Win)
    mp.set_start_method('spawn')
    app = QApplication([sys.argv])
    application = RD()
    application.show()
    app.exec()


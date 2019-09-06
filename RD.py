#!/usr/bin/python3
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, os
import ntpath
import configparser
import gui
import crop_dialog
import baseline_dialog
import spike_dialog
import smooth_dialog
from fit import FIT
from data import DATA
from convertwdf import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from recordtype import recordtype
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

Limits = recordtype('Limits', ['min', 'max'])

"Smooth dialog"
class SmoothDialog (QDialog, smooth_dialog.Ui_Dialog):
    def __init__(self):
        super(SmoothDialog, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.size())
        self.slider.valueChanged.connect(lambda : self.lb_value.setText("%d"%int(2*self.slider.value()+1)))

"Spike dialog"
class SpikeDialog (QDialog, spike_dialog.Ui_Dialog):
    def __init__(self, threshold):
        super(SpikeDialog, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.size())
        self.slider.setValue(threshold)
        self.lb_value.setText("%d" %int(threshold))
        self.slider.valueChanged.connect(lambda : self.lb_value.setText("%d"%self.slider.value()))

"Crop dialog"
class CropDialog (QDialog, crop_dialog.Ui_Dialog):
    def __init__(self, _min, _max):
        super(CropDialog, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.size())
        self.lbl_min.setText("%d"%_min)
        self.lbl_max.setText("%d"%_max)
        self.lineEdit_min.setFocus()

    def getData(self):
        _min = self.lineEdit_min.text()
        _max = self.lineEdit_max.text()
        return _min, _max

"Baseline dialog"
class BaselineDialog(QDialog, baseline_dialog.Ui_Dialog):
    def __init__(self, _deg, _min, _max):
        super(BaselineDialog, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.size())
        self.spinBox.setValue(_deg)
        self.lineEdit_min.setText(str(_min))
        self.lineEdit_max.setText(str(_max))

    def getData(self):
        _min = self.lineEdit_min.text()
        _max = self.lineEdit_max.text()
        return self.comboBox.currentIndex(), self.spinBox.value(), _min, _max

class RD(QMainWindow, gui.Ui_MainWindow):

    header  = 50*'*'+'\n\n'+10*' '+'DECONVOLUTION OF RAMAN SPECTRA'
    header += 10*''+'\n\n'+50*'*'

    def __init__(self):
        super(RD, self).__init__()
        self.setupUi(self)
        #variables
        self.settings = QSettings("Raman Deconvolution", "settings")
        self.initialDir = self.settings.value('directory', '/home/cat/')
        self.data = DATA()
        self.changed = False
        self.peaksLimits = Limits(900, 1850)
        self.spike = 0 #scatter plot for spikes
        self.limit_low, self.limit_high = 0, 0 #exclude region
        self.baseline = 0

        self.actionQuit.triggered.connect(self.close)
        self.actionAbout.triggered.connect(self.about)
        self.actionGuess.triggered.connect(lambda: [self.dockGuess.setVisible(self.actionGuess.isChecked()),
                                                    self.dockGuess.raise_()])
        self.actionOutput.triggered.connect(lambda: [self.dockOut.setVisible(self.actionOutput.isChecked()),
                                                    self.dockOut.raise_()])
        self.toolBar.setObjectName('toolbar')
        self.dockGuess.visibilityChanged.connect(self.actionGuess.setChecked)
        self.dockOut.visibilityChanged.connect(self.actionOutput.setChecked)
        self.actionToolbar.triggered.connect(self.toolBar.toggleViewAction().trigger)
        self.actionNew.triggered.connect(self.New)
        self.actionSave.triggered.connect(self.Save)
        self.actionCrop.triggered.connect(self.Crop)
        self.actionRemove_Baseline.triggered.connect(self.Baseline)
        self.actionSpike_Removal.triggered.connect(self.removeSpikes)
        self.actionSmoothing.triggered.connect(self.Smoothing)
        self.tabifyDockWidget(self.dockGuess, self.dockOut)
        self.dockGuess.raise_()
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.textOut.setReadOnly(True)
        self.textOut.setText(self.header)
        self.readConfig()

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

        actionOpen  = QAction(QIcon("graphics/open.png"),"Open",self)
        actionSave  = QAction(QIcon("graphics/save.png"),"Save",self)
        actionCrop  = QAction(QIcon("graphics/crop.png"),"Crop",self)
        actionSpike = QAction(QIcon("graphics/spike.png"),"Spike removal",self)
        actionSmooth = QAction(QIcon("graphics/smooth.png"),"Smoothing",self)
        actionBaseline = QAction(QIcon("graphics/baseline.png"),"Remove baseline",self)
        self.toolBar.addAction(actionOpen)
        self.toolBar.addAction(actionSave)
        self.toolBar.addSeparator()
        self.toolBar.addAction(actionCrop)
        self.toolBar.addAction(actionBaseline)
        self.toolBar.addAction(actionSpike)
        self.toolBar.addAction(actionSmooth)
        self.toolBar.addSeparator()
        self.toolBar.toggleViewAction().setChecked(True)
        actionOpen.triggered.connect(self.New)
        actionCrop.triggered.connect(self.Crop)
        actionBaseline.triggered.connect(self.Baseline)
        actionSpike.triggered.connect(self.removeSpikes)
        actionSave.triggered.connect(self.Save)
        actionSmooth.triggered.connect(self.Smoothing)

        self.startUp()


        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested.connect(self.tableItemRightClicked)



    def startUp(self):
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
        if not np.shape(self.data.X):
            self.errorBox('There is no data', 'No data...')
            return
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
        self.statusbar.showMessage("%d datapoints removed" %len(self.data.spikes), 2000)
        self.Plot(self.data.X, self.data.Y, "Experimental data", clear=True)
        if not self.changed:
            self.changed = True
            self.setWindowTitle(self.windowTitle()+'*')

    def Smoothing(self):
        if not np.shape(self.data.X):
            self.errorBox('There is no data', 'No data...')
            return
        self.previewSmoothed()
        result = self.smoothDialog.exec_()
        self.baseline.remove()
        self.baseline = 0
        if not result:
            self.subplot.legend()
            self.canvas.draw()
            return
        if np.shape(self.data.noBaseline):
            y_ = self.data.noBaseline
        else:
            y_ = self.data.Y
        y_ = self.data.smooth(y_, 2*self.smoothDialog.slider.value()+1)
        if np.shape(self.data.noBaseline):
            self.data.noBaseline = y_
        else:
            self.data.Y = y_
        self.Plot(self.data.X, y_, "Smoothed data")
        self.plotAdjust()
        if not self.changed:
            self.changed = True
            self.setWindowTitle(self.windowTitle()+'*')


    def previewSmoothed(self):
        if np.shape(self.data.noBaseline):
            y_ = self.data.noBaseline
        else:
            y_ = self.data.Y
        y_ = self.data.smooth(y_, 2*self.smoothDialog.slider.value()+1)
        if self.baseline != 0:
            self.baseline.remove()
        self.baseline,  = self.subplot.plot(self.data.X, y_, 'r--', label = 'Smoothed data')
        self.subplot.legend()
        self.canvas.draw()
        return y_



    def showSpikes(self):
        self.data.detectSpikes(self.spikeDialog.slider.value())
        if self.spike != 0:
            self.spike.remove()
            self.spike = 0
            self.canvas.draw()
        sp = self.data.spikes
        if len(sp):
            self.spike, = self.subplot.plot(self.data.X[sp], self.data.Y[sp], 'ro', label = 'Spikes')
            self.subplot.legend()
            self.canvas.draw()

    def plotAdjust(self):
        self.subplot.set_xlabel(r'$\mathbf{Raman\ shift,\ cm^{-1}}$')
        self.subplot.set_ylabel(r'$\mathbf{Intensty}$')
        self.canvas.draw()

    #right click on the plot
    def onclick(self, event):
        if  np.shape(self.data.X):
            if event.button == 3:  #right click
                self.listMenu= QMenu()
                menu_item_0 = self.listMenu.addAction("Delete datapoint (spike)")
                idx = np.abs(self.data.X - event.xdata).argmin()
                self.statusbar.showMessage('Datapoint selected: X = %f, Y = %f' %(self.data.X[idx], self.data.Y[idx]), 1000)
                self.spike, = self.subplot.plot(self.data.X[idx], self.data.Y[idx], 'rs', label = 'Selected datapoint')
                self.subplot.legend()
                self.canvas.draw()
                cursor = QCursor()
                menu_item_0.triggered.connect(lambda: self.deleteSpike(idx))
                self.listMenu.move(cursor.pos() )
                self.listMenu.show()
                self.listMenu.aboutToHide.connect(lambda : [self.spike.remove(), self.canvas.draw()])

    def deleteSpike(self, x):
        self.statusbar.showMessage('Spike deleted: X = %f, Y = %f' %(self.data.X[x], self.data.Y[x]), 1000)
        self.data.spikes = x
        self.data.removeSpikes()
        self.Plot(self.data.X, self.data.Y, "Experimental data")
        if not self.changed:
            self.changed = True
            self.setWindowTitle(self.windowTitle()+'*')

    def Baseline(self):
        if not np.shape(self.data.X):
            self.errorBox('There is no data', 'No data...')
            return
        if np.shape(self.data.noBaseline):
            self.Plot(self.data.X, self.data.Y, "Experimental data")
        dialog = BaselineDialog(self.degree, self.peakLimits.min, self.peakLimits.max)
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
            return
        params = dialog.getData()
        try:
            _min = int(params[2])
        except ValueError:
            self.statusbar.showMessage("Wrong value...setting to default", 3000)
        try:
            _max = int(params[3])
        except ValueError:
            self.statusbar.showMessage("Wrong value...setting to default", 3000)
        self.peakLimits = Limits(_min, _max)
        self.data.fitBaseline(params[1], self.peakLimits)
        self.Plot(self.data.X, self.data.noBaseline, "Baseline corrected data")
        self.plotAdjust()
        self.textOut.clear()
        self.textOut.append('                  BASELINE FIT                   ')
        self.textOut.append('*************************************************')
        self.textOut.append('Polynomial fit -- degree: {}'.format(self.data.bsDegree))
        self.textOut.append('Fitting equation:')
        text = ''
        for deg in np.arange(0, self.data.bsDegree+1):
            if self.data.bsCoef[deg]>=0 and deg!=0:
                text += '+'
            text += '{:.4E}*x^{}'.format(self.data.bsCoef[deg], self.data.bsDegree-deg)
        self.textOut.append(text + '\n')
        self.dockOut.raise_()
        if not self.changed:
            self.changed = True
            self.setWindowTitle(self.windowTitle()+'*')

    def previewBaseline(self, dialog):
        dt = DATA()
        dt.setData(self.data.X, self.data.Y)
        try:
            _min = int(dialog.lineEdit_min.text())
        except ValueError:
            self.statusbar.showMessage("Wrong value...setting to default", 3000)
        try:
            _max = int(dialog.lineEdit_max.text())
        except ValueError:
            self.statusbar.showMessage("Wrong value...setting to default", 3000)
        if self.baseline != 0:
            self.limit_low.remove()
            self.limit_high.remove()
            self.baseline.remove()
        self.limit_low,  = self.subplot.plot([_min, _min], [np.min(self.data.Y), np.max(self.data.Y)], color = 'red', label = 'Exclude region')
        self.limit_high, = self.subplot.plot([_max, _max], [np.min(self.data.Y), np.max(self.data.Y)], color = 'red')

        peakLimits = Limits(_min, _max)
        dt.fitBaseline(dialog.spinBox.value(), peakLimits)
        self.baseline, = self.subplot.plot(dt.X, dt.baseline, 'r--', label = "Baseline")
        self.subplot.legend()
        self.canvas.draw()

    def readConfig(self):
        path =os.path.dirname(os.path.realpath(__file__))
        #read the configuration file
        self.config = configparser.ConfigParser()
        if len(self.config.read(path+'/config/config.ini')):
               self.degree = int(self.config['DEFAULT']['degree'])
               # self.initialDir = self.config['USER_CONFIGS']['directory']
               self.threshold = float(self.config['DEFAULT']['threshold'])
        #        font_size = int(config['DEFAULT']['font_size'])
        #        dataLimits = Limits(int(config['LIMITS']['low']), int(config['LIMITS']['high']))
               self.peakLimits = Limits(int(self.config['PEAK']['low']), int(self.config['PEAK']['high']))
        #        dark = bool(int(config['DEFAULT']['dark']))
        #        if bool(int(config['SKIP_REGION']['skip'])):
        #                skipRegion = Limits(int(config['SKIP_REGION']['low']), int(config['SKIP_REGION']['high']))
        #                if skipRegion.max > dataLimits.max:
        #                        skipRegion.max = dataLimits.max
        #                if skipRegion.min > dataLimits.max:
        #                        skipRegion.min = dataLimits.max
        #                if skipRegion.max < dataLimits.min:
        #                        skipRegion.min = dataLimits.min
        #                if skipRegion.min < dataLimits.min:
        #                        skipRegion.min = dataLimits.min
        #        else:
        #                skipRegion = 0
        #        _abs = bool(int(config['DEFAULT']['abs']))
        #else:
        #        #load the defaults
        #        print('Could not find the config file...\nLoading defaults')
        #matplotlib.rcParams.update({'font.size': font_size})

        #load fitting parameters
        try:
            parameters = pd.read_csv(path+'/config/initialData.csv')
            cols = ['labels', 'freq', 'freq_min', 'freq_max' ,'intens', 'width', 'shape']
            for i, l in enumerate(parameters['labels']):
                self.tableWidget.insertRow(i)
                self.tableWidget.setItem(i, 0, QTableWidgetItem("Use"))
                self.tableWidget.item(i, 0).setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                for j, col in enumerate(cols):
                    self.tableWidget.setItem(i, j+1, QTableWidgetItem(str(parameters[col][i])))
            self.statusbar.showMessage("Initial parameters were loaded", 2000)
        except FileNotFoundError:
            elf.Error('Initial parameters were not loaded', 'File not found')

    def updateConfig (self, key, value, string):
        self.config.set(key, value, string)
        #Save the new config file
        path =os.path.dirname(os.path.realpath(__file__))
        with open(path + '/config/config.ini', 'w') as configfile:
            self.config.write(configfile)
        self.statusbar.showMessage("Configuration file updated", 2000)

    def New(self):
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
        fname, _filter = QFileDialog.getOpenFileName(self, 'Open file', path,"Text files (*.txt *.dat);; Wire data files (*.wdf);; All files (*.*)")
        if not fname:
            return
        elif self.initialDir!=ntpath.dirname(fname):
            self.initialDir=ntpath.dirname(fname) #update the initial directory for the Open/Save dialog
            self.settings.setValue('directory', self.initialDir)
        if fname[-3:] == 'wdf':
            convert(fname)
            fname = fname[:-3]+'txt'
        try:
            tmp = np.loadtxt(fname)
            self.data.setData(tmp[:,0], tmp[:,1])
            # self.statusbar.showMessage("Data loaded", 2000)
            self.Plot(self.data.X, self.data.Y, 'Experimental data')
            self.setWindowTitle( 'Raman Deconvolution - ' + ntpath.basename(fname))
            self.changed = False
            self.textOut.clear()
            self.textOut.setText(self.header)
        except:
            self.errorBox('Could not load the file', 'I/O error')
            self.statusbar.showMessage("Error loading the file", 2000)

    def Save(self):
        if not np.shape(self.data.X):
            self.statusbar.showMessage("No data...", 2000)
            return
        path = self.initialDir
        fname, _filter = QFileDialog.getSaveFileName(self, 'Save file', path,"Text files (*.txt);; Comma separated values (*.csv)")
        delimiter = '\t'
        if not fname:
            return
        if _filter == "Text files (*.txt)" and fname[-3:]!='txt':
            fname += '.txt'
        if _filter == "Comma separated values (*.csv)" and fname[-3:]!='csv':
            fname += '.csv'
            delimiter = ','

        data, comments = self.data.getData()
        f = open(fname, 'w')
        f.close()
        with open(fname, 'a') as f:
            [f.write(s) for s in comments]
            data.to_csv(f, index = None, sep = delimiter)

        self.statusbar.showMessage('File {} saved'.format(fname), 2000)


        if self.changed:
            self.changed =False
            self.setWindowTitle(self.windowTitle()[:-1])

    def Plot(self, X, Y, label, clear = True):
        if clear:
            self.subplot.clear()
        if label == 'Baseline':
            line, = self.subplot.plot(X, Y, 'r--', label = label)
        else:
            line, = self.subplot.plot(X, Y, label = label)
        self.subplot.set_xlim(np.min(X), np.max(X))
        if label in ['Experimental data', 'Baseline corrected data', 'Smoothed data']:
            self.subplot.set_ylim(0.9*np.min(Y), 1.1*np.max(Y))
        self.subplot.legend()
        self.plotAdjust()
        return line

    def errorBox(self, message, title):
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
            self.statusbar.showMessage("Wrong value...setting to default", 3000)
        try:
            _max = int(params[1])
        except ValueError:
            self.statusbar.showMessage("Wrong value...setting to default", 3000)
        self.data.crop(_min, _max)
        self.statusbar.showMessage("Data cropped", 3000)
        if np.shape(self.data.noBaseline):
            self.Plot(self.data.X, self.data.noBaseline, "Baseline corrected data")
        else:
            self.Plot(self.data.X, self.data.Y, "Experimental data")
        if self.peakLimits.min < self.data.X[0]:
            self.peakLimits.min = self.data.X[0]
        if self.peakLimits.max > self.data.X[-1]:
            self.peakLimits.max = self.data.X[-1]
        if not self.changed:
            self.changed = True
            self.setWindowTitle(self.windowTitle()+'*')

    def tableItemRightClicked(self, QPos):
        self.listMenu= QMenu()
        if self.tableWidget.item(self.tableWidget.currentRow(), 0).flags() & Qt.ItemIsEnabled:
            menu_item_0 = self.listMenu.addAction("Don't use for deconvolution")
        else:
            menu_item_0 = self.listMenu.addAction("Use for deconvolution")
        self.listMenu.addSeparator()
        menu_item_1 = self.listMenu.addAction("Use all")
        menu_item_0.triggered.connect(lambda: self.tableRowUse(self.tableWidget.currentRow()))
        menu_item_1.triggered.connect( self.tableUseAll)
        parentPosition = self.tableWidget.mapToGlobal(QPoint(0, 0))
        self.listMenu.move(parentPosition + QPos)
        self.listMenu.show()

    def tableUseAll(self):
        for row in np.arange(0, self.tableWidget.rowCount()):
            self.tableWidget.item(row,0).setFlags(Qt.ItemIsEnabled)

    def tableRowUse(self, row):
        if self.tableWidget.item(row, 0).flags() & Qt.ItemIsEnabled:
            self.tableWidget.item(row,0).setFlags(Qt.NoItemFlags)
        else:
            self.tableWidget.item(row,0).setFlags(Qt.ItemIsEnabled)



    def about(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Version v1.01α \nMade by CAT \nLille, 2019")
        msg.setWindowTitle("About")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def closeEvent(self, event):
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
    app = QApplication([sys.argv])
    application = RD()
    application.show()
    app.exec()


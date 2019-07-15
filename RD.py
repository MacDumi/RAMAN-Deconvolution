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

"Crop dialog"
class CropDialog (QDialog, crop_dialog.Ui_Dialog):
    def __init__(self, _min, _max):
        super(CropDialog, self).__init__()
        self.setupUi(self)
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
        self.spinBox.setValue(_deg)
        self.lineEdit_min.setText(str(_min))
        self.lineEdit_max.setText(str(_max))

    def getData(self):
        _min = self.lineEdit_min.text()
        _max = self.lineEdit_max.text()
        return self.comboBox.currentIndex(), self.spinBox.value(), _min, _max

class RD(QMainWindow, gui.Ui_MainWindow):


    def __init__(self):
        super(RD, self).__init__()
        self.setupUi(self)
        #variables
        self.initialDir = '/home/cat/RAMAN/'
        self.data = DATA()
        self.peaksLimits = Limits(900, 1850)

        # self.actionQuit.triggered.connect(self.close)
        self.actionAbout.triggered.connect(self.about)
        self.actionGuess.triggered.connect(lambda: [self.dockGuess.setVisible(self.actionGuess.isChecked()),
                                                    self.dockGuess.raise_()])
        self.actionOutput.triggered.connect(lambda: [self.dockOut.setVisible(self.actionOutput.isChecked()),
                                                    self.dockOut.raise_()])
        self.dockGuess.visibilityChanged.connect(self.actionGuess.setChecked)
        self.dockOut.visibilityChanged.connect(self.actionOutput.setChecked)
        self.actionToolbar.triggered.connect(self.toolBar.toggleViewAction().trigger)
        self.actionNew.triggered.connect(self.New)
        self.actionCrop.triggered.connect(self.Crop)
        self.actionRemove_Baseline.triggered.connect(self.Baseline)
        self.tabifyDockWidget(self.dockGuess, self.dockOut)
        self.dockGuess.raise_()
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.readConfig()

        self.figure = plt.figure()
        self.figure.set_tight_layout(True)
        self.subplot = self.figure.add_subplot(111) #add a subfigure
        #add widget
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setFocusPolicy(Qt.ClickFocus )
        self.canvas.setFocus()
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.gridLayout.addWidget(self.toolbar)
        self.gridLayout.addWidget(self.canvas)

        actionOpen = QAction(QIcon("graphics/open.png"),"Open",self)
        actionSave = QAction(QIcon("graphics/save.png"),"Save",self)
        actionCrop = QAction(QIcon("graphics/crop.png"),"Crop",self)
        actionBaseline = QAction(QIcon("graphics/baseline.png"),"Remove baseline",self)
        self.toolBar.addAction(actionOpen)
        self.toolBar.addAction(actionSave)
        self.toolBar.addSeparator()
        self.toolBar.addAction(actionCrop)
        self.toolBar.addAction(actionBaseline)
        self.toolBar.addSeparator()
        self.toolBar.toggleViewAction().setChecked(True)
        actionOpen.triggered.connect(self.New)
        actionCrop.triggered.connect(self.Crop)
        actionBaseline.triggered.connect(self.Baseline)

        self.startUp()


        self.tableWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tableWidget.customContextMenuRequested.connect(self.tableItemRightClicked)



    def startUp(self):
        X = np.arange(800, 2000, 2)
        Y1 = FIT.lorents(X, 1800, 1358, 110)
        Y2 = FIT.lorents(X, 1700, 1590, 50)
        self.subplot.plot(X, (Y1+Y2)+20*np.random.randn(len(X)), 'o')
        self.subplot.plot(X, Y1+Y2, 'r--')
        self.subplot.plot(X, Y1)
        self.subplot.plot(X, Y2)
        self.subplot.set_xlim(np.min(X), np.max(X))
        self.plotAdjust()

    def plotAdjust(self):
        self.subplot.set_xlabel(r'$\mathbf{Raman\ shift,\ cm^{-1}}$')
        self.subplot.set_ylabel(r'$\mathbf{Intensty}$')
        self.canvas.draw()

    def Baseline(self):
        if not np.shape(self.data.X):
            self.errorBox('There is no data', 'No data...')
            return
        _min, _max = self.peakLimits.min, self.peakLimits.max
        self.subplot.plot([_min, _min], [np.min(self.data.Y), np.max(self.data.Y)], color = 'red', label = 'Exclude region')
        self.subplot.plot([_max, _max], [np.min(self.data.Y), np.max(self.data.Y)], color = 'red')
        self.subplot.legend()
        self.canvas.draw()
        dialog = BaselineDialog(self.degree, self.peakLimits.min, self.peakLimits.max)
        result = dialog.exec_()
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
        self.Plot(self.data.X, self.data.Y, "Experimental data")
        self.Plot(self.data.X, self.data.baseline, "Baseline", clear = False)
        self.plotAdjust()




    def readConfig(self):
        path =os.path.dirname(os.path.realpath(__file__))
        #read the configuration file
        config = configparser.ConfigParser()
        if len(config.read(path+'/config/config.ini')):
               self.degree = int(config['DEFAULT']['degree'])
        #        thrsh = float(config['DEFAULT']['threshold'])
        #        font_size = int(config['DEFAULT']['font_size'])
        #        dataLimits = Limits(int(config['LIMITS']['low']), int(config['LIMITS']['high']))
               self.peakLimits = Limits(int(config['PEAK']['low']), int(config['PEAK']['high']))
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

    def New(self):
        path = self.initialDir
        fname, _filter = QFileDialog.getOpenFileName(self, 'Open file', path,"Text files (*.txt *.dat);; Wire data files (*.wdf);; All files (*.*)")
        if not fname:
            return
        else:
            self.initialDir=ntpath.dirname(fname) #update the initial directory for the Open/Save dialog
        if fname[-3:] == 'wdf':
            convert(fname)
            fname = fname[:-3]+'txt'
        try:
            tmp = np.loadtxt(fname)
            self.data.setData(tmp[:,0], tmp[:,1])
            self.statusbar.showMessage("Data loaded", 2000)
            self.Plot(self.data.X, self.data.Y, 'Experimental data')
            self.setWindowTitle( 'Raman Deconvolution - ' + ntpath.basename(fname))
        except:
            self.errorBox('Could not load the file', 'I/O error')
            self.statusbar.showMessage("Error loading the file", 2000)


    def Plot(self, X, Y, label, clear = True):
        if clear:
            self.subplot.clear()
        if label == 'Baseline':
            self.subplot.plot(X, Y, 'r--', label = label)
        else:
            self.subplot.plot(X, Y, label = label)
        self.subplot.set_xlim(np.min(X), np.max(X))
        if label == 'Experimental data':
            self.subplot.set_ylim(0.9*np.min(Y), 1.1*np.max(Y))
        self.subplot.legend()
        self.plotAdjust()

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
        self.Plot(self.data.X, self.data.Y, "Experimental data")

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

    def close(self):
        result = QMessageBox.question(self,
                "Confirm Exit...",
                "Are you sure you want to exit?",
                QMessageBox.Yes| QMessageBox.No)
        if result == QMessageBox.Yes:
            QApplication.exit()


    # def closeEvent(self, event):
    #     result = QMessageBox.question(self,
    #             "Confirm Exit...",
    #             "Are you sure you want to exit?",
    #             QMessageBox.Yes| QMessageBox.No)
    #     if result == QMessageBox.Yes:
    #         event.accept()
    #     else:
    #         event.ignore()

if __name__ == '__main__':
    app = QApplication([sys.argv])
    application = RD()
    application.show()
    app.exec()


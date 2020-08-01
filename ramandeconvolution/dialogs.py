#!/usr/bin/python3
import baseline_dialog
import batch_dialog
import crop_dialog
import spike_dialog
import smooth_dialog
import deconvolution
from fit import FIT
from data import DATA
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

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

"Batch dialog"
class BatchDialog(QDialog, batch_dialog.Ui_Dialog):
    def __init__(self, _files, _deg, _min, _max, _crop_min, _crop_max, _cores):
        super(BatchDialog, self).__init__()
        self.setupUi(self)
        self.setFixedSize(self.size())
        self.label_files.setText(str(_files))
        self.spinBox.setValue(_deg)
        self.lineEdit_min.setText(str(_min))
        self.lineEdit_max.setText(str(_max))
        self.lineEdit_cropMin.setText(str(_crop_min))
        self.lineEdit_cropMax.setText(str(_crop_max))
        self.comboBox_cores.addItems(list(map(str, range(1, _cores+1)))[::-1])

    def getData(self):
        _min = self.lineEdit_min.text()
        _max = self.lineEdit_max.text()
        _crop_min = self.lineEdit_cropMin.text()
        _crop_max = self.lineEdit_cropMax.text()
        return int(self.comboBox_cores.currentText()), self.spinBox.value(), _min, _max, _crop_min, _crop_max




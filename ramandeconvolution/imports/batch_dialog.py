# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'imports/batchDeconv.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(280, 455)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("imports/../graphics/batch.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        self.gridLayout_3 = QtWidgets.QGridLayout(Dialog)
        self.gridLayout_3.setContentsMargins(0, -1, -1, -1)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.frame = QtWidgets.QFrame(Dialog)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout = QtWidgets.QGridLayout(self.frame)
        self.gridLayout.setContentsMargins(0, 3, 3, 3)
        self.gridLayout.setObjectName("gridLayout")
        spacerItem = QtWidgets.QSpacerItem(74, 55, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 6, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(74, 17, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 8, 3, 1, 1)
        self.lineEdit_cropMin = QtWidgets.QLineEdit(self.frame)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.lineEdit_cropMin.setFont(font)
        self.lineEdit_cropMin.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly|QtCore.Qt.ImhPreferNumbers)
        self.lineEdit_cropMin.setObjectName("lineEdit_cropMin")
        self.gridLayout.addWidget(self.lineEdit_cropMin, 8, 2, 1, 1)
        self.lineEdit_max = QtWidgets.QLineEdit(self.frame)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.lineEdit_max.setFont(font)
        self.lineEdit_max.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly|QtCore.Qt.ImhPreferNumbers)
        self.lineEdit_max.setObjectName("lineEdit_max")
        self.gridLayout.addWidget(self.lineEdit_max, 6, 4, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setFamily("Noto Sans Adlam")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 0, 0, 1, 5)
        self.lineEdit_min = QtWidgets.QLineEdit(self.frame)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.lineEdit_min.setFont(font)
        self.lineEdit_min.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly|QtCore.Qt.ImhPreferNumbers)
        self.lineEdit_min.setObjectName("lineEdit_min")
        self.gridLayout.addWidget(self.lineEdit_min, 6, 2, 1, 1)
        self.lineEdit_cropMax = QtWidgets.QLineEdit(self.frame)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.lineEdit_cropMax.setFont(font)
        self.lineEdit_cropMax.setInputMethodHints(QtCore.Qt.ImhFormattedNumbersOnly|QtCore.Qt.ImhPreferNumbers)
        self.lineEdit_cropMax.setObjectName("lineEdit_cropMax")
        self.gridLayout.addWidget(self.lineEdit_cropMax, 8, 4, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(self.frame)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.spinBox.setFont(font)
        self.spinBox.setMaximum(10)
        self.spinBox.setProperty("value", 3)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout.addWidget(self.spinBox, 4, 4, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setFamily("Noto Sans Adlam")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 5, 2, 1, 3)
        self.label_7 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setFamily("Noto Sans Adlam")
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 7, 2, 1, 3)
        self.label_2 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 4, 2, 1, 1)
        self.gridLayout_3.addWidget(self.frame, 1, 0, 1, 1)
        self.frame_2 = QtWidgets.QFrame(Dialog)
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout_2.setContentsMargins(11, 3, 3, 3)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_3 = QtWidgets.QLabel(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)
        self.comboBox_cores = QtWidgets.QComboBox(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.comboBox_cores.setFont(font)
        self.comboBox_cores.setObjectName("comboBox_cores")
        self.gridLayout_2.addWidget(self.comboBox_cores, 0, 1, 1, 1)
        self.gridLayout_3.addWidget(self.frame_2, 2, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_3.addWidget(self.buttonBox, 3, 0, 1, 1)
        self.frame_3 = QtWidgets.QFrame(Dialog)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.frame_3)
        self.gridLayout_4.setContentsMargins(10, 3, 3, 3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_5 = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.gridLayout_4.addWidget(self.label_5, 0, 0, 1, 1)
        self.label_files = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(True)
        font.setWeight(75)
        self.label_files.setFont(font)
        self.label_files.setAlignment(QtCore.Qt.AlignCenter)
        self.label_files.setObjectName("label_files")
        self.gridLayout_4.addWidget(self.label_files, 0, 2, 1, 1)
        self.gridLayout_3.addWidget(self.frame_3, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.spinBox, self.lineEdit_min)
        Dialog.setTabOrder(self.lineEdit_min, self.lineEdit_max)
        Dialog.setTabOrder(self.lineEdit_max, self.lineEdit_cropMin)
        Dialog.setTabOrder(self.lineEdit_cropMin, self.lineEdit_cropMax)
        Dialog.setTabOrder(self.lineEdit_cropMax, self.comboBox_cores)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Batch Deconvolution"))
        self.label_6.setText(_translate("Dialog", "Baseline fitting"))
        self.label_4.setText(_translate("Dialog", "Exclude region"))
        self.label_7.setText(_translate("Dialog", "Crop"))
        self.label_2.setText(_translate("Dialog", "Degree"))
        self.label_3.setText(_translate("Dialog", "Nr. of cores"))
        self.label_5.setText(_translate("Dialog", "Files selected:"))
        self.label_files.setText(_translate("Dialog", "0"))

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'crop_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(284, 145)
        Dialog.setMaximumSize(QtCore.QSize(284, 145))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 110, 251, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.lbl_max = QtWidgets.QLabel(Dialog)
        self.lbl_max.setGeometry(QtCore.QRect(190, 70, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbl_max.setFont(font)
        self.lbl_max.setText("")
        self.lbl_max.setObjectName("lbl_max")
        self.lbl_min = QtWidgets.QLabel(Dialog)
        self.lbl_min.setGeometry(QtCore.QRect(30, 70, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbl_min.setFont(font)
        self.lbl_min.setText("")
        self.lbl_min.setObjectName("lbl_min")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 0, 231, 31))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lineEdit_min = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_min.setGeometry(QtCore.QRect(20, 40, 91, 32))
        self.lineEdit_min.setObjectName("lineEdit_min")
        self.lineEdit_max = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_max.setGeometry(QtCore.QRect(170, 40, 91, 32))
        self.lineEdit_max.setObjectName("lineEdit_max")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.lineEdit_min, self.lineEdit_max)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "Crop to"))

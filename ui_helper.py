# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'helper.ui'
#
# Created: Fri Aug 12 13:37:10 2011
#      by: PyQt4 UI code generator 4.8.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    _fromUtf8 = lambda s: s

class Ui_Helper(object):
    def setupUi(self, Helper):
        Helper.setObjectName(_fromUtf8("Helper"))
        Helper.resize(661, 701)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(Helper.sizePolicy().hasHeightForWidth())
        Helper.setSizePolicy(sizePolicy)
        self.verticalLayout = QtGui.QVBoxLayout(Helper)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.widget = QtGui.QWidget(Helper)
        self.widget.setMinimumSize(QtCore.QSize(0, 0))
        self.widget.setObjectName(_fromUtf8("widget"))
        self.horizontalLayout_2 = QtGui.QHBoxLayout(self.widget)
        self.horizontalLayout_2.setMargin(0)
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.groupBox = QtGui.QGroupBox(self.widget)
        self.groupBox.setObjectName(_fromUtf8("groupBox"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.groupBox)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.mi = QtGui.QDoubleSpinBox(self.groupBox)
        self.mi.setDecimals(2)
        self.mi.setSingleStep(0.05)
        self.mi.setProperty(_fromUtf8("value"), 1.0)
        self.mi.setObjectName(_fromUtf8("mi"))
        self.horizontalLayout.addWidget(self.mi)
        self.has_waitlimit = QtGui.QCheckBox(self.groupBox)
        self.has_waitlimit.setObjectName(_fromUtf8("has_waitlimit"))
        self.horizontalLayout.addWidget(self.has_waitlimit)
        self.waitlimit = QtGui.QSpinBox(self.groupBox)
        self.waitlimit.setEnabled(False)
        self.waitlimit.setMaximum(120)
        self.waitlimit.setSingleStep(30)
        self.waitlimit.setObjectName(_fromUtf8("waitlimit"))
        self.horizontalLayout.addWidget(self.waitlimit)
        self.update = QtGui.QPushButton(self.groupBox)
        self.update.setObjectName(_fromUtf8("update"))
        self.horizontalLayout.addWidget(self.update)
        self.horizontalLayout_2.addWidget(self.groupBox)
        self.groupBox_2 = QtGui.QGroupBox(self.widget)
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.groupBox_2)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.families = QtGui.QListWidget(self.groupBox_2)
        self.families.setMinimumSize(QtCore.QSize(30, 80))
        self.families.setObjectName(_fromUtf8("families"))
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        self.horizontalLayout_3.addWidget(self.families)
        self.best = QtGui.QPushButton(self.groupBox_2)
        self.best.setObjectName(_fromUtf8("best"))
        self.horizontalLayout_3.addWidget(self.best)
        self.horizontalLayout_2.addWidget(self.groupBox_2)
        self.verticalLayout.addWidget(self.widget)
        self.progressBar = QtGui.QProgressBar(Helper)
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty(_fromUtf8("value"), 0)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.verticalLayout.addWidget(self.progressBar)

        self.retranslateUi(Helper)
        self.families.setCurrentRow(0)
        QtCore.QObject.connect(self.has_waitlimit, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.waitlimit.setEnabled)
        QtCore.QMetaObject.connectSlotsByName(Helper)

    def retranslateUi(self, Helper):
        Helper.setWindowTitle(QtGui.QApplication.translate("Helper", "Route construction testing", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("Helper", "Savings heuristic", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Helper", "µ param", None, QtGui.QApplication.UnicodeUTF8))
        self.has_waitlimit.setText(QtGui.QApplication.translate("Helper", "use waitlimit", None, QtGui.QApplication.UnicodeUTF8))
        self.update.setText(QtGui.QApplication.translate("Helper", "Plot", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_2.setTitle(QtGui.QApplication.translate("Helper", "Test group selection", None, QtGui.QApplication.UnicodeUTF8))
        __sortingEnabled = self.families.isSortingEnabled()
        self.families.setSortingEnabled(False)
        self.families.item(0).setText(QtGui.QApplication.translate("Helper", "solomons/*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(1).setText(QtGui.QApplication.translate("Helper", "solomons/c*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(2).setText(QtGui.QApplication.translate("Helper", "solomons/r[12]*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(3).setText(QtGui.QApplication.translate("Helper", "solomons/rc*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.setSortingEnabled(__sortingEnabled)
        self.best.setText(QtGui.QApplication.translate("Helper", "Plot best", None, QtGui.QApplication.UnicodeUTF8))


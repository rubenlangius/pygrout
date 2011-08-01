# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'helper.ui'
#
# Created: Tue Aug  2 00:10:14 2011
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
        self.label_2 = QtGui.QLabel(self.groupBox_2)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_3.addWidget(self.label_2)
        self.families = QtGui.QListWidget(self.groupBox_2)
        self.families.setMinimumSize(QtCore.QSize(30, 80))
        self.families.setObjectName(_fromUtf8("families"))
        item = QtGui.QListWidgetItem(self.families)
        item.setCheckState(QtCore.Qt.Checked)
        item = QtGui.QListWidgetItem(self.families)
        item.setCheckState(QtCore.Qt.Checked)
        item = QtGui.QListWidgetItem(self.families)
        item.setCheckState(QtCore.Qt.Checked)
        self.horizontalLayout_3.addWidget(self.families)
        self.label_3 = QtGui.QLabel(self.groupBox_2)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.horizontalLayout_3.addWidget(self.label_3)
        self.horizontalLayout_2.addWidget(self.groupBox_2)
        self.verticalLayout.addWidget(self.widget)

        self.retranslateUi(Helper)
        QtCore.QObject.connect(self.has_waitlimit, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.waitlimit.setEnabled)
        QtCore.QMetaObject.connectSlotsByName(Helper)

    def retranslateUi(self, Helper):
        Helper.setWindowTitle(QtGui.QApplication.translate("Helper", "Route construction testing", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("Helper", "Savings heuristic", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Helper", "µ param", None, QtGui.QApplication.UnicodeUTF8))
        self.has_waitlimit.setText(QtGui.QApplication.translate("Helper", "use waitlimit", None, QtGui.QApplication.UnicodeUTF8))
        self.update.setText(QtGui.QApplication.translate("Helper", "Plot", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_2.setTitle(QtGui.QApplication.translate("Helper", "Test group selection", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Helper", "C / R / RC", None, QtGui.QApplication.UnicodeUTF8))
        __sortingEnabled = self.families.isSortingEnabled()
        self.families.setSortingEnabled(False)
        self.families.item(0).setText(QtGui.QApplication.translate("Helper", "C", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(1).setText(QtGui.QApplication.translate("Helper", "R", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(2).setText(QtGui.QApplication.translate("Helper", "RC", None, QtGui.QApplication.UnicodeUTF8))
        self.families.setSortingEnabled(__sortingEnabled)
        self.label_3.setText(QtGui.QApplication.translate("Helper", "to be done...", None, QtGui.QApplication.UnicodeUTF8))


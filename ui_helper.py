# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'helper.ui'
#
# Created: Wed Aug 24 13:29:41 2011
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
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        QtGui.QListWidgetItem(self.families)
        self.horizontalLayout_3.addWidget(self.families)
        self.best = QtGui.QPushButton(self.groupBox_2)
        self.best.setObjectName(_fromUtf8("best"))
        self.horizontalLayout_3.addWidget(self.best)
        self.horizontalLayout_2.addWidget(self.groupBox_2)
        self.groupBox_4 = QtGui.QGroupBox(self.widget)
        self.groupBox_4.setObjectName(_fromUtf8("groupBox_4"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.groupBox = QtGui.QGroupBox(self.groupBox_4)
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
        self.verticalLayout_2.addWidget(self.groupBox)
        self.groupBox_3 = QtGui.QGroupBox(self.groupBox_4)
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))
        self.horizontalLayout_4 = QtGui.QHBoxLayout(self.groupBox_3)
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.comboBox = QtGui.QComboBox(self.groupBox_3)
        self.comboBox.setObjectName(_fromUtf8("comboBox"))
        self.comboBox.addItem(_fromUtf8(""))
        self.horizontalLayout_4.addWidget(self.comboBox)
        self.pushButton = QtGui.QPushButton(self.groupBox_3)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setObjectName(_fromUtf8("pushButton"))
        self.horizontalLayout_4.addWidget(self.pushButton)
        self.verticalLayout_2.addWidget(self.groupBox_3)
        self.horizontalLayout_2.addWidget(self.groupBox_4)
        self.verticalLayout.addWidget(self.widget)
        self.textEdit = QtGui.QTextEdit(Helper)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.textEdit.sizePolicy().hasHeightForWidth())
        self.textEdit.setSizePolicy(sizePolicy)
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName(_fromUtf8("textEdit"))
        self.verticalLayout.addWidget(self.textEdit)
        self.progressBar = QtGui.QProgressBar(Helper)
        self.progressBar.setMaximum(100)
        self.progressBar.setProperty(_fromUtf8("value"), 0)
        self.progressBar.setObjectName(_fromUtf8("progressBar"))
        self.verticalLayout.addWidget(self.progressBar)
        self.clearPlot = QtGui.QPushButton(Helper)
        self.clearPlot.setObjectName(_fromUtf8("clearPlot"))
        self.verticalLayout.addWidget(self.clearPlot)

        self.retranslateUi(Helper)
        self.families.setCurrentRow(-1)
        QtCore.QObject.connect(self.has_waitlimit, QtCore.SIGNAL(_fromUtf8("toggled(bool)")), self.waitlimit.setEnabled)
        QtCore.QMetaObject.connectSlotsByName(Helper)

    def retranslateUi(self, Helper):
        Helper.setWindowTitle(QtGui.QApplication.translate("Helper", "Route construction testing", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_2.setTitle(QtGui.QApplication.translate("Helper", "Test group selection", None, QtGui.QApplication.UnicodeUTF8))
        __sortingEnabled = self.families.isSortingEnabled()
        self.families.setSortingEnabled(False)
        self.families.item(0).setText(QtGui.QApplication.translate("Helper", "solomons/*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(1).setText(QtGui.QApplication.translate("Helper", "solomons/c*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(2).setText(QtGui.QApplication.translate("Helper", "solomons/c1*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(3).setText(QtGui.QApplication.translate("Helper", "solomons/c2*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(4).setText(QtGui.QApplication.translate("Helper", "solomons/r[12]*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(5).setText(QtGui.QApplication.translate("Helper", "solomons/r1*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(6).setText(QtGui.QApplication.translate("Helper", "solomons/r2*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(7).setText(QtGui.QApplication.translate("Helper", "solomons/rc*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(8).setText(QtGui.QApplication.translate("Helper", "solomons/rc1*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(9).setText(QtGui.QApplication.translate("Helper", "solomons/rc2*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(10).setText(QtGui.QApplication.translate("Helper", "hombergers/*_2??.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(11).setText(QtGui.QApplication.translate("Helper", "hombergers/c?_2*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(12).setText(QtGui.QApplication.translate("Helper", "hombergers/c1_2*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(13).setText(QtGui.QApplication.translate("Helper", "hombergers/c2_2*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(14).setText(QtGui.QApplication.translate("Helper", "hombergers/r[12]_2*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(15).setText(QtGui.QApplication.translate("Helper", "hombergers/r1_2*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(16).setText(QtGui.QApplication.translate("Helper", "hombergers/r2_2*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(17).setText(QtGui.QApplication.translate("Helper", "hombergers/rc?_2*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(18).setText(QtGui.QApplication.translate("Helper", "hombergers/rc1_2*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.item(19).setText(QtGui.QApplication.translate("Helper", "hombergers/rc2_2*.txt", None, QtGui.QApplication.UnicodeUTF8))
        self.families.setSortingEnabled(__sortingEnabled)
        self.best.setText(QtGui.QApplication.translate("Helper", "Plot best", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_4.setTitle(QtGui.QApplication.translate("Helper", "Construction heuristic", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("Helper", "Savings heuristic", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Helper", "µ param", None, QtGui.QApplication.UnicodeUTF8))
        self.has_waitlimit.setText(QtGui.QApplication.translate("Helper", "use waitlimit", None, QtGui.QApplication.UnicodeUTF8))
        self.update.setText(QtGui.QApplication.translate("Helper", "Plot", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox_3.setTitle(QtGui.QApplication.translate("Helper", "Greedy build first", None, QtGui.QApplication.UnicodeUTF8))
        self.comboBox.setItemText(0, QtGui.QApplication.translate("Helper", "by_timewin", None, QtGui.QApplication.UnicodeUTF8))
        self.pushButton.setText(QtGui.QApplication.translate("Helper", "Plot", None, QtGui.QApplication.UnicodeUTF8))
        self.textEdit.setHtml(QtGui.QApplication.translate("Helper", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Ubuntu\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Welcome to route construction tester. This is a notification console.</p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.clearPlot.setText(QtGui.QApplication.translate("Helper", "Reset plot", None, QtGui.QApplication.UnicodeUTF8))


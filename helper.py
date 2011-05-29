#!/usr/bin/env python

from PyQt4 import QtCore, QtGui
import sys

from ui_helper import Ui_Dialog

class Helper(QtGui.QDialog):
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.fileSystemModel = QtGui.QFileSystemModel()
        self.fileSystemModel.setRootPath('.')
        self.ui.treeView.setModel(self.fileSystemModel)
        self.ui.treeView.setRootIndex(self.fileSystemModel.index('./output'))
        self.target = None

    def accept(self):
        for idx in self.ui.treeView.selectedIndexes():
            print idx
            print self.fileSystemModel.filePath(idx)
            print self.fileSystemModel.fileName(idx)
            print '-'*10
        QtGui.QDialog.accept(self)
        
    def reject(self):
        QtGui.QDialog.reject(self)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    helper = Helper()
    helper.show()
    sys.exit(app.exec_())

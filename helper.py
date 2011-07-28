#!/usr/bin/env python

import sys

import matplotlib
matplotlib.use('Qt4Agg')
import pylab

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from PyQt4 import QtCore, QtGui
from ui_helper import Ui_Helper

class Helper(QtGui.QDialog):
    def __init__(self, parent=None):
        QtGui.QDialog.__init__(self, parent)
        self.ui = Ui_Helper()
        self.ui.setupUi(self)
#        self.fileSystemModel = QtGui.QFileSystemModel()
#        self.fileSystemModel.setRootPath('.')
#        self.ui.treeView.setModel(self.fileSystemModel)
#        self.ui.treeView.setRootIndex(self.fileSystemModel.index('./output'))
#        self.target = None

    def accept(self):
#        for idx in self.ui.treeView.selectedIndexes():
#            print idx
#            print self.fileSystemModel.filePath(idx)
#            print self.fileSystemModel.fileName(idx)
#            print '-'*10
        QtGui.QDialog.accept(self)
        
    def reject(self):
        QtGui.QDialog.reject(self)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    
    fig = Figure(figsize=(600,600), dpi=72, facecolor=(1,1,1), edgecolor=(0,0,0))
    ax = fig.add_subplot(111)
    ax.plot([0,1])
    # generate the canvas to display the plot
    canvas = FigureCanvas(fig)
    
    helper = Helper()
    helper.ui.verticalLayout.addWidget(canvas)
    helper.show()
    
    sys.exit(app.exec_())

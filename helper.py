#!/usr/bin/env python

import sys

import matplotlib
matplotlib.use('Qt4Agg')
import pylab

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from PyQt4 import QtCore, QtGui
from ui_helper import Ui_Helper

def savings_val(task):
    """The mapping function for savings heuristic."""
    name, waitlimit, mi = task
    from pygrout import VrptwSolution, VrptwTask, build_by_savings    
    print "Should process", name
    sol = VrptwSolution(VrptwTask(name))
    build_by_savings(sol, waitlimit, mi)
    return sol.val()

def best_val(name):
    """The mapping function for best known value."""
    from pygrout import VrptwTask
    task = VrptwTask(name, False)
    return task.bestval()
 
class Worker(QtCore.QThread):
    def __init__(self, helper, parent = None):
        super(Worker, self).__init__(parent)
        self.helper = helper
        
    def run(self):
        self.helper.redraw()
        
class Helper(QtGui.QDialog):
    def __init__(self, parent=None):
        # boilerplate
        QtGui.QDialog.__init__(self, parent)
        self.ui = Ui_Helper()
        self.ui.setupUi(self)
        # add custom mpl canvas
        self.fig = Figure(figsize=(600,600), dpi=72, facecolor=(1,1,1), edgecolor=(0,0,0))
        self.ax_k = self.fig.add_subplot(211)
        self.ax_d = self.fig.add_subplot(212)
        # the canvas:
        self.canvas = FigureCanvas(self.fig)
        self.ui.verticalLayout.addWidget(self.canvas)
        # the worker thread (one, for now)
        self.worker = Worker(self)
        # custom slots
        QtCore.QObject.connect(self.ui.update, QtCore.SIGNAL("clicked()"), self.plot_savings)
        QtCore.QObject.connect(self.ui.best, QtCore.SIGNAL("clicked()"), self.plot_best)
        QtCore.QObject.connect(self.worker, QtCore.SIGNAL("finished()"), self.background_done)
        QtCore.QObject.connect(self.worker, QtCore.SIGNAL("terminated()"), self.background_done)
        QtCore.QObject.connect(self, QtCore.SIGNAL("progress(int)"), self.update_progress)
        # a single pool for processing
        from multiprocessing import Pool
        self.p = Pool()
        
    def background_done(self):
        self.ui.update.setEnabled(True)
        self.ui.progressBar.setEnabled(False)
        
    def plot_savings(self):
        self.ui.update.setEnabled(False)
        self.ui.progressBar.setEnabled(True)
        self.ui.progressBar.setMaximum(len(self.tests_chosen()))
        self.ui.progressBar.setValue(0)
        self.operation = 'savings'
        self.worker.start()
        
    def update_progress(self, progress):
        """The slot for updating progress in event thread."""
        print "--- one done ---"
        self.ui.progressBar.setValue(progress)
        
    def redraw(self):
        from stopwatch import StopWatch
        watch = StopWatch()
        mi = self.ui.mi.value()
        waitlimit = self.ui.waitlimit.value() if self.ui.has_waitlimit.checkState() else None
        from itertools import repeat
        tasks = zip(self.tests_chosen(), repeat(waitlimit), repeat(mi))
        numDone = 0
        data = []
        iterator = []
        if self.operation == 'savings':
            iterator = self.p.imap(savings_val, tasks)
        else:
            iterator = self.p.imap(best_val, self.tests_chosen())
        for result in iterator:
            data.append(result)
            numDone += 1
            self.emit(QtCore.SIGNAL('progress(int)'), numDone)
        print data
        self.ax_k.plot([x[0] for x in data], '.')
        self.ax_d.plot([x[1] for x in data], '.')
        print "What now?", watch
        self.fig.canvas.draw()

    def tests_chosen(self):
        """Return a sorted list of filenames by pattern in the families list."""
        from glob import glob
        pattern = str(self.ui.families.currentItem().text())
        return sorted(glob(pattern))
                
    def plot_best(self):
        print self.tests_chosen()
        self.ui.update.setEnabled(False)
        self.ui.progressBar.setEnabled(True)
        self.ui.progressBar.setMaximum(len(self.tests_chosen()))
        self.ui.progressBar.setValue(0)
        self.operation = 'best'
        self.worker.start()
            
    def accept(self):
        QtGui.QDialog.accept(self)
        
    def reject(self):
        QtGui.QDialog.reject(self)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    # almost standard:    
    helper = Helper()
    helper.show()
    sys.exit(app.exec_())

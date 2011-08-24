#!/usr/bin/env python

import sys
import glob

import matplotlib
matplotlib.use('Qt4Agg')
import pylab

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QTAgg as NavigationToolbar  
from matplotlib.figure import Figure


from PyQt4 import QtCore, QtGui
from ui_helper import Ui_Helper

class ArgMap(object):
    """A class for determining indexes of sets on the plot."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Empty the mapping and counters. Also initialize."""
        self.d = {}
        self.n = 0
        self.ticks = []
        self.ticklabels = []

    def checkTick(self, el):
        """Called by addOne - checks if element is first of a family."""
        norm = el.replace('_', '0')
        if norm.find('01.') <> -1 or norm.find('06.') <> -1:
            self.ticks.append(self.d[el])
            self.ticklabels.append(el[el.index('/')+1:el.index('.')])
        
    def addOne(self, el):
        """Single unckecked additon (use __call__ to add safely)."""
        self.n = self.d[el] = self.n+1
        self.checkTick(el)
        
    def add(self, els):
        """Adding multiple elements from an iterable."""
        map(self, els)
        
    def __call__(self, el):
        """Calling the object does safe mapping of element to index."""
        if el not in self.d:
            self.addOne(el)
        return self.d[el]

class Plot(object):
    """This encapsulates details connected with the plot."""
    def __init__(self, helper):
        self.helper = helper
        self.fig = Figure(figsize=(600,600), dpi=72, facecolor=(1,1,1), edgecolor=(0,0,0))
        self.ax_k = self.fig.add_subplot(211)
        self.ax_d = self.fig.add_subplot(212)
        # the canvas:
        self.canvas = FigureCanvas(self.fig)
        # and its toolbar
        self.toolbar = NavigationToolbar(self.canvas, helper)
        self.attachTo(helper.ui.verticalLayout)
        self.argmap = ArgMap()
        self._setup_plots()
        self._update_ticks()
    
    def _setup_plots(self):        
        self.ax_k.set_ylabel('route count')
        self.ax_d.set_ylabel('total distance')

    def _update_ticks(self):        
        self.ax_k.set_xlim((0, self.argmap.n+1))
        self.ax_k.set_xticks(self.argmap.ticks)
        self.ax_k.set_xticklabels(self.argmap.ticklabels)
        self.ax_d.set_xlim((0, self.argmap.n+1))
        self.ax_d.set_xticks(self.argmap.ticks)
        self.ax_d.set_xticklabels(self.argmap.ticklabels)

    def attachTo(self, layout):
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        
    def reset(self):
        """Remove plotted data from the drawing area."""
        self.argmap.reset()
        self.ax_k.cla()
        self.ax_d.cla()
        self._setup_plots()
        self._update_ticks()
        self.canvas.draw()
        
    def display(self, operation):
        xcoords = map(self.argmap, operation.args)
        lbl = operation.get_name()
        self.ax_k.plot(xcoords, operation.ks, 'o', label=lbl)
        self.ax_k.legend()
        ymin, ymax = self.ax_k.get_ylim()
        self.ax_k.set_ylim((ymin-1, ymax+1)) 
        
        self.ax_d.plot(xcoords, operation.ds, '.', label=lbl)
        self.ax_d.legend()
        ymin, ymax = self.ax_d.get_ylim()
        spread = (ymax - ymin)*.03
        self.ax_d.set_ylim((ymin-spread, ymax+spread)) 
        
        self._update_ticks()
        self.canvas.draw()
                
class Operation(object):
    """An abstract operation for the sets to perform"""
    def __init__(self, args):
        if type(args) == str:
            self.args = self.find_args(args)
        else:
            self.args = args
        self.ks = []
        self.ds = []
        
    def find_args(self, argstr):
        from glob import glob
        return sorted(glob(argstr))

    def get_name(self):
        """Description of operation, e.g. for plot label."""
        return 'abstract'

def best_val(name):
    """The mapping function for best known value."""
    from pygrout import VrptwTask
    task = VrptwTask(name, False)
    return task.bestval()

class BestOperation(Operation):
    def get_iterator(self, worker):
        return worker.p.imap(best_val, self.args)
    def get_name(self):
        return 'b.known'

def savings_val(task):
    """The mapping function for savings heuristic."""
    name, waitlimit, mi = task
    from pygrout import VrptwSolution, VrptwTask, build_by_savings    
    print "Should process", name
    sol = VrptwSolution(VrptwTask(name))
    build_by_savings(sol, waitlimit, mi)
    return sol.val()

class SavingsOperation(Operation):
    def __init__(self, args, mi, waitlimit):
        Operation.__init__(self, args)
        self.mi = mi
        self.waitlimit = waitlimit
    
    def get_iterator(self, worker):
        from itertools import repeat
        tasks = zip(self.args, repeat(self.waitlimit), repeat(self.mi))
        return worker.p.imap(savings_val, tasks)
    
    def get_name(self):
        desc ="sav(%.1f)" % self.mi
        if self.waitlimit:
            desc += "WL(%d)" % self.waitlimit
        return desc
         
class Worker(QtCore.QThread):
    """An active object for background computations."""
    def __init__(self, helper, parent = None):
        super(Worker, self).__init__(parent)
        self.helper = helper
        # custom signals for the GUI
        QtCore.QObject.connect(self, QtCore.SIGNAL("progress(int)"), helper.update_progress)
        QtCore.QObject.connect(self, QtCore.SIGNAL("newProgress(int)"), helper.init_progress)
        # terminating signals for the GUI
        QtCore.QObject.connect(self, QtCore.SIGNAL("finished()"), helper.background_done)
        QtCore.QObject.connect(self, QtCore.SIGNAL("terminated()"), helper.background_done)
        # the operation, passed before starting the thread
        self.currentOp = None
        # a single pool for processing
        from multiprocessing import Pool
        self.p = Pool()
                
    def run(self):
        if not self.currentOp:
            return
        self.emit(QtCore.SIGNAL('newProgress(int)'), len(self.currentOp.args))
        numDone = 0
        for k, d in self.currentOp.get_iterator(self):
            self.currentOp.ks.append(k)
            self.currentOp.ds.append(d)
            numDone += 1
            self.emit(QtCore.SIGNAL('progress(int)'), numDone)
        self.helper.plot.display(self.currentOp)
        
    def performOperation(self, operation):
        self.currentOp = operation
        self.helper.lock_ui()
        self.start()
        
class Helper(QtGui.QDialog):
    def __init__(self, parent=None):
        # boilerplate
        QtGui.QDialog.__init__(self, parent)
        self.ui = Ui_Helper()
        self.ui.setupUi(self)
        # add custom mpl canvas
        self.plot = Plot(self)
        # the worker thread (one, for now)
        self.worker = Worker(self)
        # the stopwatch placeholder
        self.watch = '(no watch set!)'

        QtCore.QObject.connect(self.ui.update, QtCore.SIGNAL("clicked()"), self.plot_savings)
        QtCore.QObject.connect(self.ui.best, QtCore.SIGNAL("clicked()"), self.plot_best)
        QtCore.QObject.connect(self.ui.clearPlot, QtCore.SIGNAL("clicked()"), self.clear_plot)
    
    def lock_ui(self):
        """Called before entering the background operation."""
        from stopwatch import StopWatch
        self.watch = StopWatch()
        self.ui.update.setEnabled(False)
        self.ui.best.setEnabled(False)
                
    def background_done(self):
        """Slot to unlock some UI elements after finished background operation."""
        self.ui.update.setEnabled(True)
        self.ui.best.setEnabled(True)
        self.ui.progressBar.setEnabled(False)
        self.ui.textEdit.append("Processing finished in %s seconds" % self.watch) 
        print "What now?", self.watch
        
    def plot_best(self):
        self.worker.performOperation(BestOperation(self.tests_chosen()))
            
    def plot_savings(self):
        mi = self.ui.mi.value()
        waitlimit = self.ui.waitlimit.value() if self.ui.has_waitlimit.checkState() else None
        self.worker.performOperation(SavingsOperation(self.tests_chosen(), mi, waitlimit))
                
    def clear_plot(self):
        """Slot for clearing the plot."""
        self.plot.reset()
        
    def init_progress(self, maxProgress):
        """Slot for resetting the progress bar's value to 0 with a new maximum."""
        self.ui.progressBar.setEnabled(True)
        self.ui.progressBar.setMaximum(maxProgress)
        self.ui.progressBar.setValue(0)

    def update_progress(self, progress):
        """Slot for updating the progress bar."""
        print "--- one done ---"
        self.ui.progressBar.setValue(progress)
        
    def tests_chosen(self):
        """Return the selected pattern in the families list."""
        return str(self.ui.families.currentItem().text())
        

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    # almost standard:    
    helper = Helper()
    helper.show()
    sys.exit(app.exec_())

from random import Random
from operator import itemgetter

import time
import cPickle
import os

import numpy as np

from undo import UndoStack
from consts import *

u = UndoStack()
"""Global undo - may be later made possible to override."""

r = Random()
"""The random number generator for the optimization."""
r_seed = int(time.time())
r.seed(r_seed)

# Possible customer ordering (when inserting into initial solution)
sort_keys = dict(
    by_opening      = lambda x: x[A], # by start of TW
    by_closing      = lambda x: x[B], # by end of TW
    by_midtime      = lambda x: x[A]+x[B], # by middle of TW
    by_weight       = lambda x: x[DEM], # by demand
    by_opening_desc = lambda x: -x[A], # by start of TW
    by_closing_desc = lambda x: -x[B], # by end of TW
    by_midtime_desc = lambda x: -x[A]-x[B], # by middle of TW
    by_weight_desc  = lambda x: -x[DEM], # by demand        
    by_timewin      = lambda x: x[B]-x[A], # ascending TW
    by_timewin_desc = lambda x: x[A]-x[B], # descending TW
    by_id           = lambda x: 0,         # unsorted
    by_random_ord   = lambda x: r.random() # random order
)

class VrptwTask(object):
    """Data loader - holds data of a VRPTW Solomon-formatted test."""
    
    sort_order = 'by_timewin'
    
    def __init__(self, stream):
        if type(stream)==str: stream = open(stream)
        lines = stream.readlines()
        self.filename = stream.name
        stream.close()
        self.name = lines[0].strip()
        self.Kmax, self.capa = map(int, lines[4].split())
        self.cust = [ map(int, x.split()) for x in lines[9:] ]
        import array
        self.cust = [ array.array('i', map(int, x.split())) for x in lines[9:] ]
        self.N = len(self.cust)-1
        self.precompute()
        self.load_best()
        
    def precompute(self):
        """Initialize or update computed members: distances and times."""
        # transpose customers, get Xs and Ys and SRVs
        x, y, srv, demands = itemgetter(X, Y, SRV, DEM)(zip(*self.cust))
        # make squares
        xx = np.tile(x, (len(x), 1))
        yy = np.tile(y, (len(y), 1))
        # compute hypots - distances
        self.dist = ((xx-xx.T)**2+(yy-yy.T)**2)**0.5
        # compute travel times (including service)
        self.time = self.dist + np.tile(srv, (len(srv),1)).T
        # calculating demand-related values
        self.demands = sorted(demands)
        self.sum_demand = sum(demands)
        self.kbound_min = -(-self.sum_demand//self.capa)
        print "Sum of q: %d (k_min >= %d), Q(0..4) = %d %d %d %d %d" % (
            self.sum_demand, self.kbound_min, self.demands[1],
            self.demands[self.N//4], self.demands[self.N//2],
            self.demands[self.N*3//4], self.demands[-1])

    def routeInfo(self, route):
        """Displays a route summary."""
        cap, dist = 0.0, 0.0
        print "Route:"
        for a, b, aa, lab in route[R_EDG]:
            print ("From %2d(%2d,%3d) to %2d(%4d,%4d): "
                   "start(%.2f)+svc(%d)+dist(%5.2f)=startb(%.2f);ltst(%.2f)"
                   % (a, self.cust[a][A], self.cust[a][B],
                      b, self.cust[b][A], self.cust[b][B],
                      aa, self.cust[a][SRV], self.dist[a][b],
                      aa + self.cust[a][SRV] + self.dist[a][b], lab) )
            if lab < aa + self.cust[a][SRV] + self.dist[a][b]:
                print "!"*70
            cap += self.cust[a][DEM]
            dist += self.dist[a][b]
            print "  Dist now %.2f, load now %.2f" % (dist, cap)
        print "Route stored dist %.2f, load %.2f" % (route[R_DIS], route[R_CAP])

    def getSortedCustomers(self):
        """Return customer tuples."""
        return sorted(self.cust[1:], key=sort_keys[VrptwTask.sort_order])

    def load_best(self):
        """Look for saved best solution values in the bestknown/ dir."""
        try:
            self.best_k, self.best_dist = map(eval,  open(
                os.path.join(os.path.dirname(__file__), 'bestknown',
                self.name+'.txt')).read().split())
            print("Best known solution for test %(name)s: %(best_k)d routes,"
                  " %(best_dist).2f total distance." % self.__dict__)
        except IOError as ioe:
            self.best_k, self.best_dist = None, None
            print >>sys.stderr, ("Best known solution not found for test: "
                                 +self.name)
            if os.path.exists(os.path.join('bestknown', self.name+'.txt')):
                raise
            
def error(msg):
    """A function to print or suppress errors."""
    print msg
    
class VrptwSolution(object):
    """A routes (lists of customer IDs) collection, basically."""
    
    # default output directory for saved solutions
    outdir = "output"
        
    def __init__(self, task):
        """The task could be used to keep track of it."""
        self.task = task
        self.r = []
        self.dist = 0.
        self.k = 0
        # additional field for any purpose
        self.mem = {}
        self.mem['r_seed'] = r_seed
        self.mem['t_start'] = time.time()
        self.history = []

    def loghist(self):
        """Put the current time and value into the history list."""
        self.history.append( [self.k, self.dist, time.time()-self.mem['t_start']] )
                
    def val(self):
        """Return a tuple to represent the solution value; less is better."""
        return (self.k, self.dist)
        
    def percentage(self):
        """Return a tuple of precentage of current solution vs best known."""
        if self.task.best_k:
            return (100.*self.k/self.task.best_k, 100.*self.dist/self.task.best_dist)
        return (100, 100)
        
    def flatten(self):
        """Make a string representation of the solution for grout program."""
        return "\n".join( 
            ["%d %f" % (self.k, self.dist)] +
            # E_TOW, i.e. edge targets
            [" ".join(str(e[1]) for e in rt[R_EDG]) for rt in self.r] + ['0\n'])
    
    def inflate(self, data):
        """Decode and recalculate routes from a string by flatten()."""
        # forget everything now:
        u.commit()
        # trusting the saved values
        lines = data.split("\n")
        k, dist = lines[0].split()
        self.k = int(k); self.dist = float(dist)
        # constructing routes
        self.r = []
        dist_glob = 0
        d = self.task.dist
        t = self.task.time
        cust = self.task.cust
        for l in xrange(1, len(lines)-2):
            # the last line should contain a newline, so -2
            customers = map(int, lines[l].split())
            edges = []
            load = 0
            dist = 0
            a = 0
            arr_a = 0
            for b in customers:
                edges.append([a, b, arr_a, 0])
                load += cust[b][DEM]
                dist += d[a][b]
                arr_a = max(arr_a+t[a][b], cust[b][A])
                a = b
            # set latest arrivat to depot, for propagating later
            edges[-1][3] = cust[0][B] 
            self.r.append([ len(customers), load, dist, edges ])
            propagate_deadline(self, -1, len(customers)-1)
            dist_glob += dist
        self.dist = dist_glob
                
    # Shorthands for access to task object.
    def d(self, a, b):
        return self.task.dist[a][b]
    def t(self, a, b):
        return self.task.time[a][b]
    def a(self, c):
        return self.task.cust[c][A]
    def b(self, c):
        return self.task.cust[c][B]
    def dem(self, c):
        return self.task.cust[c][DEM]
        
    def route(self, i):
        """Render a short representation of route i."""
        return "-".join(str(e[0]) for e in self.r[i][R_EDG])
        
    def check(self, complete=False):
        """Checks solution, possibly partial, for inconsistency."""
        unserviced = set(range(1, self.task.N+1))
        for i in xrange(len(self.r)):
            if not self.check_route(i, unserviced):
                return False
        if len(unserviced) and complete:
            error("Unserviced customers left: " + ", ".join(str(x) for x in sorted(unserviced)))
        return True
    
    def check_full(self):
        """Check full solution - shorthand method."""
        return self.check(True)

    def check_route(self, i, unserviced_ = None ):
        """Check route i for consistency. 
        Remove found customers from unserviced_"""
        now, dist, cap, l = 0, 0, 0, 0
        unserviced = unserviced_ or set(range(1, self.task.N+1))
        for fro, to, afro, lato in self.r[i][R_EDG]:
            actual = max(now, self.a(fro))
            if afro <> actual:
                error("Wrong time: %.2f (expected %.2f, err %.3f) on rt %d"
                      " edge %d from %d to %d, a(from) %d" 
                      % (afro, actual, actual-afro, i, l, fro, to, self.a(fro)))
                error(self.route(i))
                return False
            if fro:
                if not fro in unserviced:
                    error("Customer %d serviced again on route %d" % (fro, i))
                else:
                    unserviced.remove(fro)
            dist += self.d(fro, to)
            cap += self.dem(fro)
            if cap > self.task.capa:
                error("Vehicle capacity exceeded on route %d with customer %d" % (i, fro))
                return False
            l += 1
            now = actual + self.t(fro, to)
        if l != self.r[i][R_LEN]:
            error("Wrong length %d (actual %d) for route %d" % (self.r[i][R_LEN], l, i))
            return False
        return True
        
    def save(sol, extra=None):
        """Dump (pickle) the solution."""
        import uuid
        # handling unknown percentage (r207.50 and r208.50, actually)
        prec_k, prec_d = map(
            lambda x: "%05.1f" % x if sol.task.best_k else 'x'*5, 
            sol.percentage())  
        # time signature - minutes and seconds (too little?)
        time_sig = "%02d%02d" % divmod(int(time.time())%3600, 60)
        # additional markers
        if not extra is None: time_sig += str(extra)
        node_sig = hex(uuid.getnode())[-4:]
        save_name = "%s-%s-%s-%02d-%05.1f-%s-%s.p" % (
                sol.task.name, prec_k, prec_d, sol.k, sol.dist, 
                node_sig, time_sig)
        sol.mem['save_name'] = save_name
        sol.mem['save_time'] = time.time()
        sol.mem['t_elapsed'] = time.time() - sol.mem['t_start']
        sol.mem['host_sig'] = node_sig
        save_data = dict(
            routes = sol.r,
            mem = sol.mem,
            val = sol.val(),
            filename = sol.task.filename,
            name = sol.task.name,
            percentage = sol.percentage(),
            history = sol.history )
        if not os.path.exists(sol.outdir):
            os.makedirs(sol.outdir)
        cPickle.dump(save_data, open(os.path.join(sol.outdir, save_name), 'wb'))
        open(os.path.join(sol.outdir, save_name.replace('.p', '.vrp')), 'w').write(sol.flatten())
        return sol     
    
    def copy(self):
        """Return a copy the solution in a possibly cheap way."""
        clone = VrptwSolution(self.task)
        clone.assign(self)
        return clone
    
    def assign(self, rvalue):
        """Assignment operator - copy essential features from another solution."""
        self.k = rvalue.k
        self.dist = rvalue.dist
        self.r = cPickle.loads(cPickle.dumps(rvalue.r, 2))
        
    def get_essence(self):
        """Return the most interesting part of the solution - routes."""
        return (self.k, self.dist, self.r)
    
    def set_essence(self, essence):
        """Set new routes and value: use with result of get_essence."""
        self.k, self.dist, self.r = essence
        
    def infoline(self):
        return "(%d, %.2f) (%5.1f%%, %5.1f%%)" % (self.val()+self.percentage())

def propagate_arrival(sol, r, pos):
    """Update arrivals (actual service begin) on a route after pos."""
    edges = sol.r[r][R_EDG]
    time = sol.task.time
    cust = sol.task.cust
    a, b, arr_a, _ = edges[pos]
    for idx in xrange(pos+1, len(edges)):
        b, _, old_arrival, _ = edges[idx]
        new_arrival = max(arr_a + time[a][b], cust[b][A])
        # check, if there is a modification
        if new_arrival == old_arrival:
            break
        u.set(edges[idx], E_ARF, new_arrival)
        a = b
        arr_a = new_arrival

def propagate_deadline(sol, r, pos):
    """Update deadlines (latest legal service begin) on a route before pos."""
    edges = sol.r[r][R_EDG]
    _, b, _, larr_b = edges[pos]
    time = sol.task.time
    cust = sol.task.cust
    for idx in xrange(pos-1, -1, -1):
        _, a, _, old_deadline = edges[idx]
        new_deadline = min(larr_b-time[a][b], cust[a][B])
        # check, if there is a modification
        if new_deadline == old_deadline:
            break
        u.set(edges[idx], E_LAT, new_deadline)
        b = a
        larr_b = new_deadline

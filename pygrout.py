#!/usr/bin/env python

from random import Random
from itertools import count, izip, dropwhile
from collections import deque
from operator import itemgetter
from bisect import bisect_left

import sys
import os
import time
import cPickle

import numpy as np

# additional modules
from consts import *
from compat import *

from undo import UndoStack

u = UndoStack()
"""Global undo - may be later made possible to override."""

r = Random()
"""The random number generator for the optimization."""
r_seed = int(time.time())
r.seed(r_seed)

newer = False
"""A flag to portably switch to some newer code where possible."""

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
        if newer: time_sig += '_n'
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
        
# THE MODEL - basic operations on a solution (through UndoStack

def insert_new(sol, c):
    """Inserts customer C on a new route."""
    new_route = [
        2,                # number of edges
        sol.dem(c),       # demand on route 
        sol.d(0,c)+sol.d(c,0), # distance there and back
        [
            [0, c,                         0, sol.b(c)], # depot -> c
            [c, 0, max(sol.t(0,c), sol.a(c)), sol.b(0)]  # c -> depot
        ]
    ]
    u.ins(sol.r, sol.k, new_route)
    u.atr(sol, 'k', sol.k+1)                      # route no inc
    u.atr(sol, 'dist', sol.dist+new_route[R_DIS]) # total distance inc

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

def insert_at_pos(sol, c, r, pos):
    """Inserts c into route ad pos. Does no checks."""
    # update edges (with arival times)
    edges = sol.r[r][R_EDG]
    # old edge
    a, b, arr_a, larr_b = u.pop(edges, pos)
    # arrival and latest arrival time to middle
    arr_c = max(arr_a + sol.t(a, c), sol.a(c))
    larr_c = min(sol.b(c), larr_b-sol.t(c, b))
    assert arr_c <= larr_c, 'invalid insertion, time window violated'
    # new edges - second then first
    u.ins(edges, pos, [c, b, arr_c, larr_b])
    u.ins(edges, pos, [a, c, arr_a, larr_c])

    # propagate time window constraints - forward
    propagate_arrival(sol, r, pos+1)

    # propagate time window constraints - backward
    propagate_deadline(sol, r, pos)

    # update distances
    dinc = sol.d(a, c)+sol.d(c, b)-sol.d(a, b)
    u.add(sol.r[r], R_DIS, dinc)
    u.ada(sol, 'dist', dinc)
    # update capacity
    u.add(sol.r[r], R_CAP, sol.dem(c))
    # update count
    u.add(sol.r[r], R_LEN, 1)

def find_bestpos_on(sol, c, r):
    """Finds best position to insert customer on existing route."""
    # check capacity
    if sol.r[r][R_CAP] + sol.dem(c) > sol.task.capa:
        return None, None
    # pull out deep things locally
    time = sol.task.time
    cust = sol.task.cust
    dist = sol.task.dist
    c_a = cust[c][A]
    c_b = cust[c][B]

    def eval_edge(pack):
        pos, (a, b, arr_a, larr_b) = pack
        arr_c = max(arr_a + time[a][c], c_a) # earliest possible
        larr_c = min(c_b, larr_b-time[c][b]) # latest if c WAS here
        larr_a = min(sol.b(a), larr_c-time[a][c])
        if  arr_c <= larr_c and arr_a <= larr_a:
            return (-(dist[a][c] + dist[c][b] - dist[a][b]), pos)
        return None, None
        
    # find the best edge 
    return max(map(eval_edge, enumerate(sol.r[r][R_EDG])))

def find_bestpos(sol, c):
    """Find best positions on any route, return the route pos and distance.
    The exact format is a nested tuple: ((-dist increase, position), route)"""
    bdp = (None, None)
    br = None
    for i in xrange(sol.k):
        for m in find_allpos_on(sol, c, i):
            if m > bdp:
                bdp = m
                br = i
    return (bdp, br)

def insert_customer(sol, c):
    """Insert customer at best position or new route."""
    if sol.k == 0:
        insert_new(sol, c)
        return sol.k-1, 0
    else:
        # best distinc, best pos, best route
        (bd, bp), br = find_bestpos(sol, c)
        # found some route to insert
        if not bd is None:
            insert_at_pos(sol, c, br, bp)
            return br, bp
        else:
            insert_new(sol, c)
            return sol.k-1, 0

def remove_customer(sol, r, pos):
    """Remove customer at pos from a route and return his ID."""
    assert pos < sol.r[r][R_LEN], 'removal past route end'
    edges = sol.r[r][R_EDG]
    a, b, arr_a, larr_b = u.pop(edges, pos)
    d, c, arr_b, larr_c = u.pop(edges, pos)
    assert b == d, 'adjacent edges do not meet in one node'
    
    if sol.r[r][R_LEN] == 2: # last customer - remove route
        rt = u.pop(sol.r, r)
        # solution route count decrease
        u.ada(sol, 'k', -1)
        # solution distance decrease
        u.ada(sol, 'dist', -rt[R_DIS])
        return b

    assert arr_a + sol.t(a, c) < larr_c, 'time window error after removal'
    u.ins(edges, pos, [a, c, arr_a, larr_c])

    # propagating time window constraints
    propagate_arrival(sol, r, pos)
    propagate_deadline(sol, r, pos)

    # update distances (probably decrease)
    dinc = sol.d(a, c)-sol.d(a, b)-sol.d(b, c)
    u.add(sol.r[r], R_DIS, dinc)
    u.ada(sol, 'dist', dinc)
    # update capacity
    u.add(sol.r[r], R_CAP, -sol.dem(b))
    # update count
    u.add(sol.r[r], R_LEN, -1)
    return b

# NEIGBOURHOOD OPERATORS - single step trials
    
operations = dict()
def operation(func):
    """A decorator for single solution operations."""
    operations[func.__name__] = func
    return func

@operation
def op_greedy_single(sol, randint = r.randint):
    """Neighbourhood operator - remove random customer and insert back."""
    # pick a route
    r = randint(0, sol.k-1)
    pos = randint(0, sol.r[r][R_LEN]-2)
    c = remove_customer(sol, r, pos)  
    insert_customer(sol, c)

@operation
def op_greedy_multiple(sol, randint = r.randint):
    """Remove a few customers from a random route and insert them back."""
    r = randint(0, sol.k-1)
    num_removed = randint(1, min(9, sol.r[r][R_LEN]-1))
    removed = []
    for i in xrange(num_removed):
        removed.append(remove_customer(sol, r, randint(0, sol.r[r][R_LEN]-2)))
    for c in removed:
        insert_customer(sol, c)

def pick_short_route(sol, random=r.random):
    """Pick a route with preference for shortest."""
    r_lengths = np.array([1.0/(rt[R_LEN]-1) for rt in sol.r]).cumsum()
    return bisect_left(r_lengths, random()*r_lengths[-1])

def pick_long_route(sol, random=r.random):
    """Return a random route, with preference for the longer."""
    lengths = np.array([rt[R_LEN]-1 for rt in sol.r]).cumsum()
    return bisect_left(lengths, random()*lengths[-1])

def find_allpos_on(sol, c, r, startpos=0):
    """Find all positions where customer c can be inserted on route r
    and return them as tuples (distinc, position)."""
    # check capacity
    if sol.r[r][R_CAP] + sol.dem(c) > sol.task.capa:
        return
    # check route edges
    edges = sol.r[r][R_EDG]
    time = sol.task.time
    cust = sol.task.cust
    dist = sol.task.dist
    c_a = cust[c][A]
    c_b = cust[c][B]
    for pos in xrange(startpos, sol.r[r][R_LEN]):
        a, b, arr_a, larr_b = edges[pos]
        if c_a > larr_b:
            # too early
            continue
        if arr_a > c_b:
            # too late
            break
        arr_c  = max(arr_a + time[a][c], c_a) # earliest possible
        larr_c = min(c_b, larr_b-time[c][b]) # latest if c WAS here
        larr_a = min(cust[a][B], larr_c-time[a][c])
        if  arr_c <= larr_c and arr_a <= larr_a:
            # for some cases distinc in optional...
            distinc = -(dist[a][c] + dist[c][b] - dist[a][b])
            yield (distinc, pos)
                
@operation
def op_fight_shortest(sol, random=r.random, randint=r.randint):
    """Picks and tries to empty a random route with preference for shortest."""
    r = pick_short_route(sol)
    num_removed = min(randint(1, 10), sol.r[r][R_LEN]-1)
    removed = []
    for i in xrange(num_removed):
        removed.append(remove_customer(sol, r, randint(0, sol.r[r][R_LEN]-2)))
    for c in removed:
        insert_customer(sol, c)

    
@operation
def op_tabu_single(sol, randint = r.randint, choice=r.choice):
    """Pick one customer from a random route and move him to a different."""
    r = pick_short_route(sol)
    old_len = sol.r[r][R_LEN]
    old_k = sol.k
    pos = randint(0, old_len-2)
    c = remove_customer(sol, r, pos)
    # d("Route %d, from %d, removed customer %d"%(r,pos,c))  
    for tries in xrange(sol.k-1):
        # max k tries
        r2 = randint(0, sol.k-2)
        # picking all other with equal probability
        if r2 >= r and sol.k == old_k: r2 +=1
        # d("other route %d" % r2)
        if sol.r[r2][R_LEN] < old_len:
            continue
        candidates = sorted(find_allpos_on(sol, c, r2))
        if not candidates:
            continue
        dist, pos = candidates[-1] # choice(candidates)
        # d("found pos %d (%.2f inc)" % (pos, dist))
        insert_at_pos(sol, c, r2, pos)
        return
    # customer c from r failed to move - insert him back
    u.undo()

@operation
def op_tabu_shortest(sol, randint = r.randint):
    r = pick_short_route(sol)
    num_removed = randint(1, sol.r[r][R_LEN]-1)
    removed = []
    for i in xrange(num_removed):
        removed.append(remove_customer(sol, r, randint(0, sol.r[r][R_LEN]-2)))
    for c in removed:
        tried = set()
        found = False
        for tries in xrange(sol.k*2):
            # max k tries
            r2 = pick_long_route(sol)
            if r2 in tried:
                continue
            # print "Long route", r2
            # time.sleep(0.6)
            dist, pos = find_bestpos_on(sol, c, r2)
            if pos:
                insert_at_pos(sol, c, r2, pos)
                found = True
                break
            tried.add(r2)
        if not found:
            u.undo()
            return
    ##print "We displaced %d customers from %d:" % (num_removed, r), removed
    
# major solution functions (metaheuristics)

def build_first(sol):
    """Greedily construct the first solution."""
    sol.dist = 0
    sol.k = 0
    sol.r = []
    for c in sol.task.getSortedCustomers():
        insert_customer(sol, c[ID])
    sol.mem['init_order'] = VrptwTask.sort_order
    u.commit()
    sol.loghist()

def local_search(sol, oper, end=0, verb=False, speed=None):
    """Optimize solution by local search."""
    # local rebinds
    ci=u.commit; undo=u.undo; val=sol.val 
    oldval = val()
    from time import time
    # stats
    updates = 0
    steps = 0
    start = time()
    if end == 0:
        end = time()+3
    while time() < end:
        steps += 1
        oper(sol)
        newval = val()
        if  newval < oldval:
            oldval = newval
            updates += 1
            sol.loghist()
            ci()
        elif val()[0] == oldval[0]:
            # huh, not worse, when it comes to routes
            ci()
        else:
            undo()
    elapsed = time()-start
    if verb:
        print " ".join([ sol.infoline(), 
          "%.1f s, %.2f fps, %d acc (%.2f aps)" % (
          elapsed, steps/elapsed, updates, updates/elapsed) ])
    # fps measurement from outside
    if not speed is None:
        speed.append(steps/elapsed)
    sol.loghist()
    return sol
    
                
def simulated_annealing(sol, oper):
    pass

# MISC. SOLUTION FUNCTIONS - postprocessing

def save_solution(sol, extra=None):
    sol.save(extra)

def plot_history(sol):
    """Display a matplotlib graph of solution progress"""
    from matplotlib import pyplot as plt
    k, dist, t = zip(*sol.history)
    fig = plt.figure()
    
    fig.suptitle(sol.task.name+" "+sol.infoline())
    
    # subplot of routes vs. time
    kplt = fig.add_subplot(121)
    kline = kplt.plot(t, k, 'g')
    min_k = (sol.task.best_k or 2)-2
    # scaling
    kplt.axis([0, sol.history[-1][2], min_k, max(k)+1])
    # labels etc.
    plt.xlabel('time [s]')
    plt.ylabel('routes (k)')
    if sol.task.best_k:
        kplt.axhline(sol.task.best_k+0.03)
    
    # subplot of distance vs. time
    dplt = fig.add_subplot(122)
    dline = dplt.plot(t, dist, 'g')
    # scaling the plot
    min_d = min(dist+(sol.task.best_dist,))
    max_d = max(dist+(sol.task.best_dist,))
    span_d = max_d - min_d
    dplt.axis([0, sol.history[-1][2], min_d-span_d/20., max_d+span_d/20.])
    # decoration with labels, etc.
    plt.grid(True)
    dplt.set_xlabel('time [s]')
    dplt.set_ylabel('dist')
    dplt.yaxis.set_label_position("right")
    dplt.yaxis.set_ticks_position("right")
    if sol.task.best_dist:
        dplt.axhline(sol.task.best_dist)
    plt.show()

# aggressive route minimization

def find_replace_pos_on(sol, c, r):
    """Return a position (occupied), where the customer could be inserted."""
    # pull out deep things locally
    time = sol.task.time
    cust = sol.task.cust
    dist = sol.task.dist
    c_a = cust[c][A]
    c_b = cust[c][B]
    pos = 0
    for a, b, arr_a, larr_b in sol.r[r][R_EDG]:
        if c_a > larr_b:
            pos += 1
            continue
        if arr_a > c_b:
            break
        
    return None
    
def short_light_route(sol):
    """Return the index of the shortest of the three lightest routes."""
    from heapq import nsmallest
    if sol.k > 3:
        candidates = nsmallest(3, xrange(sol.k), key=lambda x: sol.r[x][R_CAP])
    else:
        candidates = xrange(sol.k)
    return min( (sol.r[i][R_LEN], i) for i in candidates )[1]
    
def remove_route(sol, r):
    """Remove a route and retur a list of its customers."""
    data = u.pop(sol.r, r)
    cust = map(itemgetter(0), data[R_EDG])[1:]
    u.ada(sol, 'k', -1)
    u.ada(sol, 'dist', -data[R_DIS])    
    return cust

#@operation
def op_route_min(sol, random=r.random, randint=r.randint, data=dict(die=0)):
    """Emulate the route minimization (RM) heuristic by Nagata et al."""
    from collections import deque
    
    r = short_light_route(sol)
    print "I'll try to eliminate route", r+1
    ep = deque(remove_route(sol, r))
    print "%d customers left to go: %s" % (len(ep), " ".join(map(str, ep)))
    def insert(c, r, pos, ep):
        print "Customer %d goes to %d at pos %d" % (c, r, pos)
        insert_at_pos(sol, c, r, pos)
        print_like_Czarnas(sol)
        print "Still left", ep
        
    while len(ep) > 0 and not data['die']:
        c = ep.pop()
        r = randint(0, sol.k-1)
        _, pos = find_bestpos_on(sol, c, r)
        if not pos is None:
            insert(c, r, pos, ep)
            continue
        
        (_, pos), r = find_bestpos(sol, c)
        if not pos is None:
            insert(c, r, pos, ep)
            continue
        
        pos = find_replace_pos_on(sol, c, r)
        if not pos is None:
            insert(c, r, pos, ep)
            continue
        
        ep.appendleft(c)
    raise RuntimeError, 'this is darnd!'
 
# MAIN COMMANDS
commands = set()
def command(func):
    """A command decorator - the decoratee should be a valid command."""
    commands.add(func.__name__)
    return func

@command
def resume(args):
    data = dict(die=0)
    def die():
        data['die'] = 1
    from threading import Timer
    Timer(args.wall, die).start()
    sol = load_solution(args.test)
    print_like_Czarnas(sol)
    op_route_min(sol, data=data)
    print_like_Czarnas(sol)    

@command
def grout(args):
    """Postprocess a solution using the proprietary grout program."""
    import grout
    sol = load_solution(args.test)
    grout.DataLoader_load(sol.task.filename)
    rk = grout.SolutionDistanceDecreaser()
    rk.inflate(sol.flatten())
    rk.setMaxEpochs(60)
    best = grout.Solution()
    rk.simulatedAnnealing(best)
    print best.flatten()
    
def _optimize(test, op, wall, intvl):
    """An optimization funtion, which does not use argparse namespace."""
    sol = VrptwSolution(VrptwTask(test))
    build_first(sol)
    print_like_Czarnas(sol)
    print "Starting optimization for %d s, update every %s s." % (wall, intvl)
    time_to_die = time.time() + wall
    next_feedback = time.time() + intvl
    while time.time() < time_to_die:
        local_search(sol, operations[op], next_feedback, True)
        print_like_Czarnas(sol)
        next_feedback = time.time()+intvl
    print "Wall time reached for %s." % test.name
    save_solution(sol)
    print(sol.mem)
    print_like_Czarnas(sol)
    return sol
    
@command
def optimize(args):
    """Perform optimization of a VRPTW instance according to the arguments."""
    sol = _optimize(args.test, args.op, args.wall, args.intvl)
    return sol

def _optimize_by_name(arg):
    # open the test filename (VrptwTask had problems with it)
    arg[0] = open(arg[0])
    return _optimize(*arg)
    
@command
def run_all(args):
    """As optimize, but runs all instances."""
    from glob import glob
    runs = args.runs or 1
    all_tasks = [[n, args.op, args.wall, args.intvl] 
                 for n in glob(args.glob) * args.runs]
    if args.multi:
        from multiprocessing import Pool
        p = Pool()    
        p.map(_optimize_by_name, all_tasks)
    else:
        map(_optimize_by_name, all_tasks)

def load_solution(f):
    """Unpickle solution from a stream."""
    solution_data = cPickle.load(f)
    filename = os.path.join(os.path.dirname(__file__),
                            solution_data['filename'])
    print "Loading solution from:", filename
    sol = VrptwSolution(VrptwTask(open(filename)))
    sol.k, sol.dist = solution_data['val']
    sol.r = solution_data['routes']
    sol.mem = solution_data['mem']
    try:
        sol.history = solution_data['history']
    except: pass    
    if not sol.check_full():
        return None
    print "Solution loaded:", sol.infoline()
    return sol
    
@command
def load(args):
    """Loads a previously saved solution for analysis."""
    sol = load_solution(args.test)
    print_like_Czarnas(sol)
    print sol.mem
    try:
        if len(sol.history):
            plot_history(sol)
        else:
            print "The solution has no history to plot"
    except ImportError:
        print "Plotting history impossible (missing GUI or matplotlib)"

# POOLCHAIN metaheuristic and friends

def worker(sol, pools, operators, config):
    """The actual working process in a poolchain."""
    import Queue as q
    from multiprocessing import Queue
    proc_id, size, intvl, deadline = config
    print "Worker launched, id:", proc_id
    # disperse workers' random nubmer generators
    r.jumpahead(20000*proc_id)
    # disperse workers' feedback a bit (actually: random)
    next_feedback = time.time() + (proc_id+1)*intvl
    num_produced = 0
    # the list for measurement of fps etc.
    myfps = []
    
    while time.time() < deadline:
        # choose solution to work on this round
        try:
            # fish in the the pool
            new_essence = pools[1].get_nowait()
            sol.set_essence(new_essence)
            print "Worker", proc_id, "got job:", sol.infoline()
        except q.Empty:
            # if nothing to take - produce new one or keep current
            if num_produced < 5 or r.random() < 4.0/num_produced:
                order = r.choice(sort_keys.keys())
                VrptwTask.sort_order = order
                build_first(sol)
                print("Worker %d produced new: %s by %s" % 
                      (proc_id, sol.infoline(), order))
            # else: go on with current
        
        # run optimization 
        local_search(sol, operators[1], next_feedback, speed=myfps)
        next_feedback = time.time() + intvl*(size+1)
        
        # throw the solution back to the pool
        pools[2].put(sol.get_essence())
    # endwhile:
    # declare not to do any more output, send 'fps'
    pools[2].put((proc_id, sum(myfps)/len(myfps), 0))
    # print "Worker", proc_id, "should now finish."
    
@command
def poolchain(args):
    """Parallel optimization using a pool of workers and a chain of queues."""
    import Queue as q
    from multiprocessing import cpu_count, Process, Queue
    
    time_to_die = time.time()+args.wall
    # create own solution object (for test data being inherited)
    began = time.time()
    sol = VrptwSolution(VrptwTask(args.test))
    # setup the queues
    poison_pills = Queue()
    input_ = Queue()
    output = Queue()
    queues = [ poison_pills, input_, output ]
    oplist = [ None, operations[args.op], None ]
    
    # create and launch the workers
    num_workers = args.runs or cpu_count()
    workers = [ Process(
        target=worker, args=(sol, queues, oplist, 
          (i, num_workers, args.intvl, time_to_die)))
        for i in xrange(num_workers) ]
    map(Process.start, workers)
    
    # get a solution from the fastest worker (we have to service them...)
    print "Master waits for first solution..."
    essence = output.get()
    input_.put(essence)
    sol.set_essence(essence)
    print "Got first solution:", sol.infoline(), "after", time.time()-began 
    sol.loghist()

    # the discriminators of the solution circulation
    best_seen_k = essence[0]
    best_essncs = [essence]
    
    if best_seen_k == sol.task.best_k:
        print "Best known route count immediately:", time.time()-began
        sol.mem['best_k_found'] = time.time()-began
        if args.strive:
            time_to_die = time.time() + args.wall / 5.0
            print "Wall time reduced to:", time_to_die - time.time()
    
    # manage the pool for a while (now - simply feed them back)
    # ---- START OF MAIN LOOP ----
    while time.time() < time_to_die:
        essence = output.get()
        # drop solutions worse than best_seen_k+1
        if essence[0] <= best_seen_k+1:
            # -- check for route count record
            if best_seen_k > essence[0]:
                best_seen_k = essence[0]
                if best_seen_k == sol.task.best_k:
                    print "Best known route count reached:", time.time()-began
                    sol.mem['best_k_found'] = time.time()-began

                    if args.strive and time_to_die > time.time()+args.wall/5.0:
                        time_to_die = time.time()+args.wall/5.0
                        print "Remaining time reduced to:", args.wall/5.0
            # -- check against pool
            pos = bisect_left(best_essncs, essence)            
            if ( len(best_essncs)<15 
                or (pos < 15 and best_essncs[pos][:2] <> essence[:2]) ):
                # this solution is ok - pay it forward
                input_.put(essence)
                best_essncs.insert(pos, essence)
                if len(best_essncs) > 15:
                    best_essncs.pop()
                if pos == 0:
                    # new global best - remembering as a historical event
                    sol.set_essence(essence)
                    sol.loghist()            
            else:
                # throw in one of the elite solutions
                input_.put(r.choice(best_essncs))
        elif r.random() < 0.5:
            # if solution was bad (route count), maybe throw in old 
            input_.put(r.choice(best_essncs))
    # ---- END OF MAIN LOOP ----
    print "Wall time passed, after:", time.time()-began

    fpss = []
    workers_left = num_workers
    while workers_left > 0:
        k, dist, routes = output.get()
        if routes == 0:
            workers_left -= 1
            print "Worker's",k,"pill-box received", time.time()-began
            fpss.append(dist)
        else:
            if (k, dist) < sol.val():
                sol.set_essence((k, dist, routes))
            print 'got out from output: ', k, dist

    print "Staff is to join: so much are alive:"
    print map(Process.is_alive, workers)

    print input_.qsize(), 'solutions still in queue 1'
    try:
        while True:
            # print "Waiting for a solution"
            k, dist, routes = input_.get(timeout=0.3)
            if (k, dist) < sol.val():
                sol.set_essence((k, dist, routes))
            print 'got out: ', k, dist
    except q.Empty:
        pass
    
    try:
        output.get(timeout=0.1)
    except q.Empty:
        pass
    else:
        print "Possible rubbish in output"
    
    print "Best solution chosen. Saving.", time.time()-began
    save_solution(sol, '_pc') # suffix for poolchain
    print_like_Czarnas(sol)
    print "summary:", sol.task.name, "%d %.1f"%sol.val(), "%.1f %.1f"%sol.percentage(), 
    print "wall", args.wall, "workers", num_workers, "op", args.op, 'best_k',
    try:
        print "%.1f" % sol.mem['best_k_found'],
    except KeyError:
        print 'NO',
    print 'fps', "%.1f" % sum(fpss)
                
    #map(Process.join, workers)
    print "\nTotal time elapsed:", time.time()-began

@command
def initials(args):
    """Produce initial solutions in all available ways, and 10x randomly."""
    sol = VrptwSolution(VrptwTask(args.test))
    results = []
    for k in sort_keys.keys():
        VrptwTask.sort_order = k
        build_first(sol)
        results.append((sol.percentage(), k, sol.k))
    VrptwTask.sort_order = 'by_random_ord'
    for i in xrange(9):
        build_first(sol)
        results.append((sol.percentage(), 'by_random_ord', sol.k))
    rank = 1
    for prec, k, sol_k in sorted(results):
        print "%-20s %.2f %.2f  routes %d  rank %02d %s" % (
              (k+':',)+prec+(sol_k, rank, sol.task.name))
        rank += 1

def get_argument_parser():
    """Create and configure an argument parser.
    Used by main function; may be used for programmatic access."""
    try:
        from argparse import ArgumentParser, Action
        parser = ArgumentParser(
            description="Optimizing VRPTW instances with some heuristics")
            
        parser.add_argument(
            "test", type=file, nargs='?', default=os.path.join(
                os.path.dirname(__file__), 'hombergers','rc210_1.txt'),
            help="the test instance: txt format as by M. Solomon")
        parser.add_argument(
            "command", choices=commands, nargs="?", default="poolchain",
            help="main command to execute (when omitted: poolchain)")
            
        parser.add_argument(
            "--op", choices=operations.keys(), nargs="?",
            default="op_fight_shortest", help="neighbourhood operator to use")
        
        parser.add_argument(
            "--runs", "-n", type=int, default=0,
            help="repeat (e.g. optimization) n times, or use n processes")
        parser.add_argument(
            "--glob", "-g", default="hombergers/*.txt",
            help="glob expression for run_all, defaults to all H")
            
        parser.add_argument(
            "--wall", "-w", type=int, default=600,
            help="approximate walltime (real) in seconds")
        parser.add_argument(
            "--intvl", type=int, default=3,
            help="approximate refresh rate (delay between messages)")
        parser.add_argument(
            "--strive", action="store_true",
            help="run for best known route count, and then only short")
                    
        parser.add_argument(
            "--multi", "-p", action="store_true",
            help="use multiprocessing for parallelism e.g. with run_all")
        parser.add_argument(
            "--prof", action="store_true",
            help="profile the code (don't do that), 10x slower")
    
        class OptionAction(Action):
            """A dispatching action for option parser - global configs"""
            def __call__(self, parser, namespace, values,
                         option_string=None):
                if option_string in ['-o', '--output']:
                    VrptwSolution.outdir = values
                elif option_string == '--order':
                    VrptwTask.sort_order = values
                elif option_string in ['-s', '--seed']:
                    global r_seed
                    r_seed = int(values)
                    r.seed(r_seed)

        parser.add_argument(
            "--seed", "-s", action=OptionAction,
            help="Set a custom RNG seed")    
                
        parser.add_argument(
            "--output", "-o", default="output", action=OptionAction,
            help="output directory for saving solutions")
        
        parser.add_argument(
            "--order", action=OptionAction, choices=sort_keys.keys(),
            help="choose specific order for initial customers")
        
        parser.add_argument(
            "--newer", action="store_true",
            help="use untested extra features (caution)")
            
        return parser
    except ImportError:
        print "Install argparse module"
        raise

def main(can_profile = False):
    """Entry point when this module is ran at top-level."""
    args = get_argument_parser().parse_args()
    if can_profile and args.prof:
        import cProfile
        args.test.close()
        cProfile.run('main()', 'profile.bin')
        return
    # execute the selected command
    globals()[args.command](args)

if __name__ == '__main__':
    main(True)
    

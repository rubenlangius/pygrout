#!/usr/bin/env python

from math import hypot
from random import Random
from itertools import count, izip, dropwhile
import sys
import os

# for imports in some environments
sys.path.append('.')

# additional modules
from consts import *
from compat import *

from undo import UndoStack

u = UndoStack()
"""Global undo - may be later made possible to override."""

class VrptwTask(object):
    """Data loader - holds data of a VRPTW Solomon-formatted test."""
    
    # Possible customer ordering (when inserting into initial solution)
    sort_keys = dict(
        by_timewin_asc  = staticmethod(lambda x: x[B]-x[A]), # ascending TW
        by_timewin_desc = staticmethod(lambda x: x[A]-x[B]), # descending TW
        by_id           = staticmethod(lambda x: 0),         # unsorted
        by_random_ord   = staticmethod(lambda x: Random().random) # random order
    )

    sort_order = sort_keys['by_timewin_asc']
    
    def __init__(self, stream):
        lines = stream.readlines()
        self.name = lines[0].strip()
        self.Kmax, self.capa = map(int, lines[4].split())
        self.cust = [ tuple(map(int, x.split())) for x in lines[9:] ]
        self.N = len(self.cust)-1
        self.precompute()
        self.load_best()
        
    def precompute(self):
        """Initialize or update computed members: distances and times."""
        try:
            # numpy version - faster! 
            if NO_NUMPY:
                raise ImportError
            from numpy import tile
            from operator import itemgetter
            # transpose customers, get Xs and Ys and SRVs
            x, y, srv = itemgetter(X, Y, SRV)(zip(*self.cust))
            # make squares
            xx, yy = tile(x, (len(x),1)), tile(y, (len(y), 1))
            # compute hypots - distances
            self.dist = ((xx-xx.T)**2+(yy-yy.T)**2)**0.5
            # compute travel times (including service)
            self.time = self.dist + tile(srv, (len(srv),1)).T
        except ImportError:
            # no numpy - list fallback
            # distances between customers
            sys.stderr.write("Warning: Not using NumPy\n")
            self.dist = [
                [ hypot(self.cust[i][X]-self.cust[j][X],
                        self.cust[i][Y]-self.cust[j][Y])
                  for j in xrange(len(self.cust))
                ] for i in xrange(len(self.cust))
            ]
            # travel times including service
            self.time = [
                [ elem  + self.cust[j][SRV] for elem in self.dist[j] ]
                 for j in xrange(len(self.dist))
            ]
            
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
        return sorted(self.cust[1:], key=VrptwTask.sort_order)

    def load_best(self):
        """Look for saved best solution values in the bestknown/ dir."""
        try:
            self.best_k, self.best_dist = map(eval,  open(
                os.path.join('bestknown', self.name+'.txt')).read().split())
            print("Best known solution for test %(name)s: %(best_k)d routes,"
                  " %(best_dist).2f total distance." % self.__dict__)
        except:
            self.best_k, self.best_dist = None, None
            print >>sys.stderr, ("Best known solution not found for test: "
                                 +self.name)
            raise
            
def error(msg):
    """A function to print or suppress errors."""
    print msg
    
class VrptwSolution(object):
    """A routes (lists of customer IDs) collection, basically."""
    def __init__(self, task):
        """The task could be used to keep track of it."""
        self.task = task
        self.r = []
        self.dist = 0.
        self.k = 0
    def val(self):
        """Return a tuple to represent the solution value; less is better."""
        return (self.k, self.dist)
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
    # TODO: make remove_... use it.
    edges = sol.r[r][R_EDG]
    a, b, arr_a, _ = edges[pos]
    for idx in xrange(pos+1, len(edges)):
        b, _, old_arrival, _ = edges[idx]
        new_arrival = max(arr_a + sol.t(a, b), sol.a(b))
        # check, if there is a modification
        if new_arrival == old_arrival:
            break
        u.set(edges[idx], E_ARF, new_arrival)
        a = b
        arr_a = new_arrival
        
    
def propagate_deadline(sol, r, pos):
    """Update deadlines (latest legal service begin) on a route before pos."""
    # TODO: check, make insert_... and remove_... use it.
    edges = sol.r[r][R_EDG]
    _, b, _, larr_b = edges[pos]
    for idx in xrange(pos-1, -1, -1):
        _, a, _, old_deadline = edges[idx]
        new_deadline = min(larr_b-sol.t(a, b), sol.b(a))
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
    assert arr_c <= larr_c
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

def find_bestpos_on(sol, c, r, forbid=None):
    """Finds best position to insert customer on existing route."""
    pos, mininc = None, None
    # check capacity
    if sol.r[r][R_CAP] + sol.dem(c) > sol.task.capa:
        return None, None
    # check route edges
    for (a, b, arr_a, larr_b), i in izip(sol.r[r][R_EDG], count()):
        arr_c = max(arr_a + sol.t(a, c), sol.a(c)) # earliest possible
        larr_c = min(sol.b(c), larr_b-sol.t(c, b)) # latest if c WAS here
        larr_a = min(sol.b(a), larr_c-sol.t(a, c))
        if  arr_c <= larr_c and arr_a <= larr_a:
            distinc = -(sol.d(a, c) + sol.d(c, b) - sol.d(a, b))
            if mininc < distinc and i <> forbid:
                pos, mininc = i, distinc
    return mininc, pos

def find_bestpos(sol, c):
    """Find best positions on any route, return the route pos and distance.
    The exact format is a nested tuple: ((-distance increase, position), route)"""
    return max((find_bestpos_on(sol, c, rn), rn) for rn in xrange(len(sol.r)))


def find_bestpos_except_pos(sol, c, r, pos):
    """Find best position on routes other than position pos on route r.
    Most likely to be called with customer previous position."""
    return max((find_bestpos_on(sol, c, rn, [pos if rn==r else None]), rn) for rn in xrange(len(sol.r)))
    
# TODO: maybe return where we finally inserted him
def insert_customer(sol, c):
    """Insert customer at best position or new route."""
    if not len(sol.r):
        insert_new(sol, c)
        if DEBUG_INIT:
            print c, ">new", len(sol.r)-1
        return len(sol.r)-1, 0
    else:
        # best distinc, best pos, best route
        (bd, bp), br = find_bestpos(sol, c)
        # found some route to insert
        if not bd is None:
            insert_at_pos(sol, c, br, bp)
            if DEBUG_INIT:
                print "Cust %d to route %d after %d distinc %.3f" % (
                    c, br, sol.r[br][R_EDG][bp][E_FRO], -bd )
            return br, bp
        else:
            insert_new(sol, c)
            if DEBUG_INIT:
                print c, ">new", len(sol.r)-1
            return len(sol.r)-1, 0

def remove_customer(sol, r, pos):
    """Remove customer at pos from a route and return his ID."""
    assert pos < sol.r[r][R_LEN]
    edges = sol.r[r][R_EDG]
    a, b, arr_a, larr_b = u.pop(edges, pos)
    d, c, arr_b, larr_c = u.pop(edges, pos)
    assert b == d
    
    if sol.r[r][R_LEN] == 2: # last customer - remove route
        rt = u.pop(sol.r, r)
        # solution route count decrease
        u.ada(sol, 'k', -1)
        # solution distance decrease
        u.ada(sol, 'dist', -rt[R_DIS])
        return b

    assert arr_a + sol.t(a, c) < larr_c
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

operations = set()
def operation(func):
    """A decorator for single solution operations (maybe lengthy)."""
    operations.add(func.__name__)
    return func

@operation
def solution_diag(sol):
    if not sol.check():
        badroute = dropwhile(sol.check_route, xrange(len(sol.r))).next()
        print "Bad route:", badroute
        sol.task.routeInfo(sol.r[badroute])
        u.undo_last()
        print "----\n"*3, "Was before:"
        sol.task.routeInfo(sol.r[badroute])
        exit()

@operation
def build_first(sol):
    """Greedily construct the first solution."""
    for c in sol.task.getSortedCustomers():
        insert_customer(sol, c[ID])
        solution_diag(sol)
        u.checkpoint()
    u.commit()

def op_rand_remove_greedy_ins(sol, randint = Random().randint):
    """Neighbourhood operator - remove random customer and insert back."""
    # pick a route
    r = randint(0, sol.k-1)
    # _, r = min( (sol.r[i][R_LEN], i) for i in xrange(sol.k) )
    pos = randint(0, sol.r[r][R_LEN]-2)
    c = remove_customer(sol, r, pos)  
    insert_customer(sol, c)

@operation
def local_search(sol, oper=op_rand_remove_greedy_ins, ci=u.commit, undo=u.undo, verbose=False, listener=None):
    """Optimize solution by local search."""
    oldval = sol.val()
    for j in xrange(20): # thousands of iterations
        value_before_batch = sol.val()
        for x in xrange(1000):
            oper(sol)
            if sol.val() < oldval:
                if verbose:
                    print "From (%d, %.4f) to (%d, %.4f)" % (oldval + sol.val())
                if listener:
                    listener(sol)
                solution_diag(sol)
                oldval = sol.val()
                if not sol.check():
                    print "ERR"
                ci()
            else:
                undo()
        print oldval
        if value_before_batch[0] == oldval[0] and abs(value_before_batch[1]-oldval[1])< 1e-6:
            print "No further changes. Quitting."
            break

@operation
def parallel_search(sol):
    import multiprocessing
    
@operation
def print_bottomline(sol):
    print "%d %.2f" % (sol.k, sol.dist)
    return sol

# MAIN COMMANDS
        
commands = set()
def command(func):
    """A command decorator - the decoratee should be a valid command."""
    commands.add(func.__name__)
    return func

@operation
def save_solution(sol):
    from cPickle import dump
    from uuid import uuid1
    print sol.r
    dump(sol.r, open(sol.task.name+'-'+str(uuid1())+'.p', 'wb'), 2)
    return sol
    
@command
def optimize(args):
    """Perform optimization of a VRPTW instance according to the arguments."""
    sol = VrptwSolution(VrptwTask(args.test))
    op_list = args.run if args.run else presets[args.preset]
    # TODO: maybe check if this is marked as operation at all ;)
    for op in op_list:
        globals()[op](sol)
    return sol

@command
def run_all(args):
    """As optimize, but runs all instances."""
    from multiprocessing import Pool
    p = Pool(2)
    from glob import glob
    from copy import deepcopy
    def process(fname):
        a = deepcopy(args)
        a.test = open(fname)
        optimize(a)
        
    #p.map(process, glob('solomons/*.txt')+glob('hombergers/*.txt'))
    map(process, glob('solomons/*.txt')+glob('hombergers/*.txt'))
    
# OPERATION PRESETS

presets = {
    'default': "build_first local_search print_like_Czarnas save_solution".split(),
    'brief': "build_first local_search print_bottomline save_solution".split(),
    'initial': "build_first print_like_Czarnas save_solution".split()
}

def get_argument_parser():
    """Create and configure an argument parser.
    Used by main function; may be used for programmatic access."""
    try:
        from argparse import ArgumentParser, Action
        parser = ArgumentParser(
            epilog="The default presets are: "+str(presets['default']),
            description="Optimizing VRPTW instances with some heuristics")
        parser.add_argument(
            "command", choices=commands, nargs="?", default="optimize",
            help="choose operation mode (currently only optimize)")
        parser.add_argument(
            "test", type=file, nargs='?', default='solomons/c101.txt',
            help="the test instance: txt format as by M. Solomon")
        parser.add_argument(
            "--run", "-e", choices=operations, action="append",
            help="perform specific available operation on the solution")
        parser.add_argument(
            "--preset", choices=presets.keys(), default="default",
            help="choose a preset of operations (for optimize command)")
        parser.add_argument(
            "--tkview", action="store_true",
            help="display routes on a Tkinter canvas")
        class OrderSwitcher(Action):
            def __call__(self, parser, namespace, values,
                         option_string=None):
                VrptwTask.sort_order = VrptwTask.sort_keys[values]
        parser.add_argument(
            "--order", action=OrderSwitcher,
            choices=VrptwTask.sort_keys.keys(),
            help="choose specific order for initial customers")
        return parser
    except ImportError:
        print "Install argparse module"
        raise
    
def main():    
    """Entry point when this module is ran at top-level.
    This function may change, testing some current new functionality."""
    parser = get_argument_parser()
    args = parser.parse_args()
    # execute the selected command
    globals()[args.command](args)

if __name__=='__main__':
    main()
    

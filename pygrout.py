#!/usr/bin/env python

from math import hypot
from itertools import count, izip, dropwhile
import sys

# for imports in some environments
sys.path.append('.')

# additional modules
from consts import *
from compat import *

from undo import UndoStack

u = UndoStack()
"""Global undo - may be later made possible to override."""

class VrptwTask(object):
    """Data loader - holds data of a VRPTW Solomon-formatted instance."""
    def __init__(self, filename):
        lines = open(filename).readlines()
        self.name = lines[0].strip()
        self.Kmax, self.capa = map(int, lines[4].split())
        self.cust = [ tuple(map(int, x.split())) for x in lines[9:] ]
        self.N = len(self.cust)-1
        # Computed stuff
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
    prev_arr  = arr_c + sol.t(c, b)
    for i in range(pos+2, len(edges)): # starts with (b, x, arr_b, larr_x)
        p, n, arr, larr = edges[i]
        if prev_arr < arr: # first wait for time window
            break
        u.set(edges[i], E_ARF, prev_arr)
        prev_arr = prev_arr+sol.t(p, n)

    # propagate time window constraints - backward
    next_larr, next = larr_c - sol.t(a,c), c
    for i in range(pos-1, -1, -1):
        p, n, arr, larr = edges[i]
        if next_larr > larr: # first early close
            break
        u.set(edges[i], E_LAT, next_larr)
        next_larr, next = next_larr-sol.t(n, next), n

    # update distances
    dinc = sol.d(a, c)+sol.d(c, b)-sol.d(a, b)
    sol.r[r][R_DIS] += dinc
    sol.dist += dinc
    # update capacity
    sol.r[r][R_CAP] += sol.dem(c)
    # update count
    sol.r[r][R_LEN] += 1

def find_bestpos_on(sol, c, r):
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
            if mininc < distinc:
                pos, mininc = i, distinc
    return mininc, pos
    
def insert_customer(sol, c):
    """Insert customer at best position or new route."""
    if not len(sol.r):
        insert_new(sol, c)
        if DEBUG_INIT:
                print c, ">new", len(sol.r)-1
    else:
        # best distinc, best pos, best route
        (bd, bp), br = max(
            (find_bestpos_on(sol, c, rn), rn) for rn in xrange(len(sol.r)) )
        # found some route to insert
        if not bd is None:
            insert_at_pos(sol, c, br, bp)
            if DEBUG_INIT:
                print "Cust %d to route %d after %d distinc %.3f" % (
                    c, br, sol.r[br][R_EDG][bp][E_FRO], -bd )
        else:
            insert_new(sol, c)
            if DEBUG_INIT:
                print c, ">new", len(sol.r)-1

def build_first(sol, sortkey = lambda c: c[B]-c[A]):
    """Greedily construct the first solution."""
    for c in sorted(sol.task.cust[1:], key=sortkey):
        insert_customer(sol, c[ID])
        if not sol.check():
            badroute = dropwhile(lambda x: sol.check_route(x), xrange(len(sol.r))).next()
            print "Bad route:", badroute
            sol.task.routeInfo(sol.r[badroute])
            u.undo_last()
            print "----\n"*3, "Was before:"
            sol.task.routeInfo(sol.r[badroute])
            exit()
        u.checkpoint()
    
def check_initial_sorting(test):
    sorters = [ 
               lambda x: x[B]-x[A], # ascending TW
               lambda x: x[A]-x[B], # descending TW
               lambda x: 0 # unsorted
               ]
    task = VrptwTask(test)
    s = [ VrptwSolution(task) for x in sorters ]
    for data in zip(s, sorters):
        build_first(*data)
    results = [ (sol.k, sol.dist) for sol in s ]
    best = map(lambda x: x[1], sorted(zip(results, range(1,4))))
    print task.name, results, best 
    
def test_initial_creation():
    """Unit test for creating solutions to all included benchmarks."""
    def check_one(test):
        s = VrptwSolution(VrptwTask(test))
        build_first(s)
        assert s.check()==True, 'Benchmark %s failed at initial solution' % test
    from glob import iglob
    from itertools import chain
    completed = 0
    # Homberger's are too heavy
    # tests = chain(iglob("solomons/*.txt"), iglob('hombergers/*.txt'))
    tests = iglob("solomons/*.txt")
    for test in tests:
        yield check_one, test
        completed += 1
    assert completed == 56, 'Wrong number of checked benchmarks'
    
def main():
    """Entry point when this module is ran at top-level.
    This function may change, testing some current new functionality."""
    test = 'solomons/c101.txt'
    
    if len(sys.argv) < 2:
        sys.stderr.write("usage: %s benchmark_file.txt\nUsing default benchmark: %s\n" % (sys.argv[0], test))
    else:
        test = sys.argv[1]
        
    test_initial_creation()

if __name__=='__main__':
    main()
    
"""
# BENCHMARK
import timeit
t = timeit.Timer(
    "VrptwTask('hombergers/c110_1.txt')",
    'from __main__ import VrptwTask'
)
t1 = t.timeit(3)/3
NO_NUMPY = True
t2 = t.timeit(3)/3
print t1,t2,t2/t1
"""


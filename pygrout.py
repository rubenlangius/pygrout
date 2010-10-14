#!/usr/bin/env python

from math import hypot
from itertools import count, izip
import pprint
import sys
import operator

# additional modules
from compat import *

NO_NUMPY = False
DEBUG_INIT = True

ID, X, Y, DEM, A, B, SRV = range(7)

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
    def checkRoute(self, route):
        """Displays a route summary."""
        prev, arr, dist = 0, 0.0, 0.0
        for next in route:
            n_arr = max(arr+self.time[prev][next], self.cust[next][A])
            print self.cust[next], n_arr, dist
            prev, arr, dist = next, n_arr, dist + self.dist[prev][next]
            
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
            now, dist, cap, l = 0, 0, 0, 0
            prevd = None
            for fro, to, afro, lato in self.r[i][R_EDG]:
                actual = max(now, self.a(fro))
                if afro <> actual:
                    error("Wrong time: %.2f (expected %.2f, err %.3f) on rt %d"
                          " edge %d from %d to %d, a(from) %d" 
                          % (afro, actual, actual-afro, i, l, fro, to, self.a(fro)))
                    print self.route(i)
                    print prevd
                    pprint.pprint(self.r[i][R_EDG])
                    return False
                if fro:
                    if not fro in unserviced:
                        error("Customer %d serviced again on route %d" % (fro, i))
                    else:
                        unserviced.remove(fro)
                dist += self.d(fro, to)
                prevd = self.d(fro, to)
                cap += self.dem(fro)
                if cap > self.task.capa:
                    error("Vehicle capacity exceeded on route %d with customer %d" % (i, fro))
                l += 1
                now = actual + self.t(fro, to)
            if l != self.r[i][R_LEN]:
                error("Wrong length %d (actual %d) for route %d" % (self.r[i][R_LEN], l, i))
        if len(unserviced) and complete:
            error("Unserviced customers left: " + ", ".join(str(x) for x in sorted(unserviced)))
        print "Check OK"
        return True
    
    def check_full(self):
        """Check full solution - shorthand method."""
        return self.check(True)
        

class UndoStack(object):
    """Holds description of a sequence of operations, possibly separated by checkpoints."""

# route    
R_LEN, R_CAP, R_DIS, R_EDG = range(4)
# customer
E_FRO, E_TOW, E_ARF, E_LAT = range(4)

def insert_new(sol, c):
    """Inserts customer C on a new route."""
    sol.r.append( [
        2,                # number of edges
        sol.dem(c),       # demand on route 
        sol.d(0,c)+sol.d(c,0), # distance there and back
        [
            [0, c,                         0, sol.b(c)], # depot -> c
            [c, 0, max(sol.t(0,c), sol.a(c)), sol.b(0)]  # c -> depot
        ]
    ] )
    sol.k += 1                   # route no inc
    sol.dist += sol.r[-1][R_DIS] # total distance inc

def insert_at_pos(sol, c, r, pos):
    """Inserts c into route ad pos. Does no checks."""
    # update edges (with arival times)
    edges = sol.r[r][R_EDG]
    # old edge
    a, b, arr_a, larr_b = edges.pop(pos)
    # arrival time to middle
    arr_c = max(arr_a + sol.t(a, c), sol.a(c))
    # latest arrival to middle
    larr_c = min(sol.b(c), larr_b-sol.t(c, b))
    # new edges - second then first
    edges.insert(pos, [c, b, arr_c, larr_b])
    edges.insert(pos, [a, c, arr_a, larr_c])
    # propagate time window constraints - forward
    prev_arr, prev  = arr_c + sol.t(c, b), c
    for i in range(pos+2, len(edges)):
        p, n, arr, larr = edges[i]
        if prev_arr < arr: # first wait for time window
            break
        edges[i][E_ARF], prev_arr, prev = prev_arr, prev_arr+sol.t(prev, p), p
    # propagate time window constraints - backward
    next_larr, next = larr_c - sol.t(a,c), c
    for i in range(pos-1, -1, -1):
        p, n, arr, larr = edges[i]
        if next_larr > larr: # first early close
            break
        edges[i][E_LAT], next_larr, next = next_larr, next_larr-sol.t(n, next), n
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
        arr_c = max(arr_a + sol.t(a, c), sol.a(c))
        if  arr_c <= sol.b(c) and arr_c + sol.t(c, b) <= larr_b:
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
        pprint.pprint(sol.r[min(2,len(sol.r)-1)][R_EDG])
        if not sol.check():
            exit()
    
def test_initial_sorting(test):
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
    s[0].check()
    pprint.pprint(s[0].r[11])
    
def test_initial_creation(test):
    global DEBUG_INIT
    s = VrptwSolution(VrptwTask(test))
    DEBUG_INIT = True
    build_first(s)
    print_like_Czarnas(s)    
    
def main():
    """Entry point when this module is ran at top-level.
    This function may change, testing some current new functionality."""
    test = 'solomons/c101.txt'
    
    if len(sys.argv) < 2:
        sys.stderr.write("usage: %s benchmark_file.txt\nUsing default benchmark: %s\n" % (sys.argv[0], test))
    else:
        test = sys.argv[1]
        
    test_initial_sorting(test)

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


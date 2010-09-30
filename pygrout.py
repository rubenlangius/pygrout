#!/usr/bin/env python

from math import hypot
from itertools import count, izip
import pprint

NO_NUMPY = False

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
            self.dist = [
                [ hypot(self.cust[i][X]-self.cust[j][X],
                        self.cust[i][Y]-self.cust[j][Y])
                  for j in xrange(len(self.cust))
                ] for i in xrange(len(self.cust))
            ]
            # travel times including 
            self.time = [
                [ elem  + self.cust[j][SRV] for elem in self.dist[j] ]
                 for j in xrange(len(self.dist))
            ]
    
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
    prev_arr, prev  = arr_a + sol.t(a, c), c
    for i in range(pos+1, len(edges)):
        p, n, arr, larr = edges[i]
        if prev_arr < arr: # first wait for time window
            break
        edges[i][E_ARF], prev_arr, prev = prev_arr, prev_arr+sol.t(prev, p), p
    # propagate time window constraints - backward
            
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
        if arr_a + sol.t(a, c) + sol.t(c, b) <= larr_b: #TODO: too weak cond!
            distinc = sol.d(a, c) + sol.d(c, b) - sol.d(a, b)
            if mininc < -distinc:
                pos, mininc = i, -distinc
    return mininc, pos
    
def insert_customer(sol, c):
    """Insert customer at best position or new route."""
    if not len(sol.r):
        insert_new(sol, c)
    else:
        # best distinc, best pos, best route
        (bd, bp), br = max(
            (find_bestpos_on(sol, c, rn), rn) for rn in xrange(len(sol.r)) )
        #
        if bd:
            insert_at_pos(sol, c, br, bp)
        else:
            insert_new(sol, c)

def build_first(sol):
    """Greedily construct the first solution."""
    for c in sorted(sol.task.cust[1:], key=lambda c: c[A]-c[B]):
        print c, c[A]-c[B]
        insert_customer(sol, c[ID])

def symbol(i):
    """Return a suitable symbol for displaying the customer"""
    for t, f in [
        (62, lambda x: '+'),
        (36, lambda x: chr(ord('a')+x-36)),
        (10, lambda x: chr(ord('A')+x-10)),
        ( 0, lambda x: chr(ord('0')+x)),
        (None, lambda x: '?') ]:
        if i >= t:
            return f(i)
    
def describe(sol, cols=50, onlyrouted=True):
    """Produces a textual representation of the task."""
    customers = [ sol.task.cust[c] for c in
                  set(x[E_FRO] for r in sol.r for x in r[R_EDG])
                ] if onlyrouted else sol.task.cust
    minx, maxx, miny, maxy = [
        op( x[k] for x in customers ) for k in X, Y for op in min, max ]
    sx, sy = (maxx - minx), (maxy-miny)
    rows = sy * cols // sx
    board = [ [ ' ' for i in xrange(cols+1) ] for j in xrange(rows+1) ]
    for y, x, i in [ ((c[Y]-miny)*rows//sy, (c[X]-minx)*cols//sx, c[ID])
                     for c in customers ]:
        board[y][x] = symbol(i)
    print "\n".join("".join(row) for row in board[::-1])

def print_like_Czarnas(sol):
    """Prints the solution in a form compatible (and diffable) with Czarnas.
       The Czarnas' code features a 'routeCostMultipiler', which is used like
       this: 
    (Solution.h)
    routeCostMultiplier = ROUTE_COST_WEIGHT * MAX_CUSTOMERS;
    FLOAT result = routeCostMultiplier * routes + TO_FLOAT(totalDistance);
    (constants.h)
    #define ROUTE_COST_WEIGHT (2.0*((MAX_X - MIN_X)+(MAX_Y - MIN_Y)))
    #define MAX_CUSTOMERS 100
    #define MIN_X 0.0
    #define MAX_X 100.0
    #define MIN_Y 0.0
    #define MAX_Y 100.0
    -> This formula is - arguably - bad, because it depends on the number
       of customers and the coordinates range, which for Homberger tests
       is different and quite large and can overflow integers
    for 100 - routeCostMultiplier = 40 000, >2**15
    for 1000 - 4 000 000 > 2**21, multiplied by 6 decimal places (20 bits)
    additionally, TODO
    """
    print "Solution:\nRoutes: %d\n"
    
def main():
    """Entry point when this module is ran at top-level.
    This function may change, testing some current new functionality."""
    s = VrptwSolution(VrptwTask('solomons/c101.txt'))
    build_first(s)
    print_like_Czarnas(s)

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


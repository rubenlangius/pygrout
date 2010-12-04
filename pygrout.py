#!/usr/bin/env python

from math import hypot
from random import Random
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
        self.precompute()
        
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

def find_bestpos(sol, c):
    """Find best positions on any route, return the route pos and distance.
    
    The exact format is a nested tuple: ((-distance increase, position), route)"""
    return max((find_bestpos_on(sol, c, rn), rn) for rn in xrange(len(sol.r)))

# TODO: maybe return where we finally inserted him
def insert_customer(sol, c):
    """Insert customer at best position or new route."""
    if not len(sol.r):
        insert_new(sol, c)
        if DEBUG_INIT:
            print c, ">new", len(sol.r)-1
    else:
        # best distinc, best pos, best route
        (bd, bp), br = find_bestpos(sol, c)
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

def solution_diag(sol):
    if not sol.check():
        badroute = dropwhile(lambda x: sol.check_route(x), xrange(len(sol.r))).next()
        print "Bad route:", badroute
        sol.task.routeInfo(sol.r[badroute])
        u.undo_last()
        print "----\n"*3, "Was before:"
        sol.task.routeInfo(sol.r[badroute])
        exit()
    
def build_first(sol, sortkey = lambda c: c[B]-c[A]):
    """Greedily construct the first solution."""
    for c in sorted(sol.task.cust[1:], key=sortkey):
        insert_customer(sol, c[ID])
        solution_diag(sol)
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

    completed = 0
    # Homberger's are too heavy
    # from itertools import chain
    # tests = chain(iglob("solomons/*.txt"), iglob('hombergers/*.txt'))
    tests = iglob("solomons/*.txt")
    for test in tests:
        yield check_one, test
        completed += 1
    assert completed == 56, 'Wrong number of checked benchmarks'

def op_rand_remove_greedy_ins(sol, randint = Random().randint):
    """Neighbourhood operator - remove random customer and insert back."""
    # pick a route
    r = randint(0, sol.k-1)
    # _, r = min( (sol.r[i][R_LEN], i) for i in xrange(sol.k) )
    pos = randint(0, sol.r[r][R_LEN]-2)
    c = remove_customer(sol, r, pos)  
    insert_customer(sol, c)
    
def local_search(sol, oper=op_rand_remove_greedy_ins, ci=u.commit, undo=u.undo):
    """Optimize solution by local search."""
    ci()
    oldval = sol.val()
    for j in xrange(20): # thousands of iterations
        value_before_batch = sol.val()
        for x in xrange(1000):
            oper(sol)
            if sol.val() < oldval:
                print "From (%d, %.4f) to (%d, %.4f)" % (oldval + sol.val())
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

def local_search_filename(filename):
    sol = VrptwSolution(VrptwTask(filename))
    build_first(sol)
    local_search(sol)
    print_like_Czarnas(sol)
    
def main():    
    """Entry point when this module is ran at top-level.
    This function may change, testing some current new functionality."""
    test = 'solomons/c101.txt'
    
    if len(sys.argv) < 2:
        sys.stderr.write("usage: %s benchmark_file.txt\nUsing default benchmark: %s\n" % (sys.argv[0], test))
    else:
        test = sys.argv[1]
    local_search_filename(test)
        

if __name__=='__main__':
    try:
        raise ImportError
        import profile
        profile.run('main()')
    except ImportError:
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


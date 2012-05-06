from vrptw.consts import *
from itertools import count

def pairs(iterable):
    """A generator for adjacent elements of an iterable."""
    it = iter(iterable)
    prev = it.next()
    for next_ in it:
        yield (prev, next_)
        prev = next_

def test_pairs():
    """Unit test for pairs() generator."""
    for actual, expected in zip(pairs(range(5)), [(i, i+1) for i in range(4)]):
        assert actual == expected

def d(s):
    """Debug print with a sleep."""
    import time
    print s
    time.sleep(1)

def dd(s):
    """Debug print, no sleep."""
    print s

def solution_header(sol):
    """
    The Czarnas' code features a 'routeCostMultipiler', which is used like this:
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
    routeCostMultiplier = 40000
    value = sol.dist + len(sol.r) * routeCostMultiplier
    result = "Solution:\nRoutes: %d\n" % (len(sol.r))
    result += "Vehicle capacity: %.2f\nSolution value: %.3f\n" % (sol.task.capa, value)
    result += "Total travel distance: %.3f\n" % sol.dist
    return result

def print_like_Czarnas(sol, sparse=False):
    """Prints the solution in a form compatible (and diffable) with Czarnas."""
    result = solution_header(sol)

    for rt, num in zip(sol.r, count(1)):
        if (not sparse) or rt[R_LEN] > 2:
            result += "Route: %d, len: %d, dist: %.3f, max cap: %.2f" % (
                num, rt[R_LEN], rt[R_DIS], rt[R_CAP])
            result += ", route: "+"-".join(
                str(e[E_FRO]) for e in rt[R_EDG][1:] )+"\n"
    if sparse and any(rt[R_LEN]==2 for rt in sol.r):
        result += "Single routes: " + ", ".join(str(rt[R_EDG][1][E_FRO]) for rt in sol.r if rt[R_LEN]==2)+"\n"
    print result

def print_like_Czarnas_long(sol):
    """Prints a verbose description of the solution (one line per customer).
    Compatible with the printSolutionAllData() method in the reference code

            DATATYPE dist = data.getDistance(DEPOT, getRouteStart(r));
            for (int c = getRouteStart(r); c != DEPOT; c = cust[c].getNext()) {
                initCap -= TO_FLOAT(data.getDemand(c));
                fprintf(output, "(%2d, %7.2f, %7.2f, %7.2f, %7.2f, %5.2f, %6.2f, %6.2f, %4.1f)\n", c,
                        TO_FLOAT(cust[c].getArrival()),
                        TO_FLOAT(cust[c].getLatestArrival()),
                        TO_FLOAT(data.getBeginTime(c)), TO_FLOAT(data.getEndTime(c)),
                        TO_FLOAT(data.getServiceTime(c)),
                        TO_FLOAT(data.getDistance(cust[c].getPrev(), c)), initCap,
                        TO_FLOAT(data.getDemand(c)));
                if (initCap > TO_FLOAT(data.getVehicleCapacity()) || initCap < 0.0)
                    fprintf(output, "************* vehicle capacity violated!!!\n");
                dist += data.getDistance(c, cust[c].getNext());
            }
    """
    result = solution_header(sol)

    for rt, num in zip(sol.r, count(1)):
        result += (
            "Route: %d\nRoute length: %d\nRoute cost: %.3f\n"
            "Init capacity: %.2f, max capacity = %.2f\n" %
            (num, rt[R_LEN], rt[R_DIS], rt[R_CAP], rt[R_CAP]) +
            "Route \n"
            "(cust, arriv, ltstArr, bgnWind, endWind, srvcT, dstPrv, weight, dem):\n"
            " ------------------------------------------------------------------\n"
            )
        wgt = 0
        for bef, aft in pairs(rt[R_EDG]):
            cust = bef[E_TOW]
            result += (
                "(%2d, %7.2f, %7.2f, %7.2f, %7.2f, %5.2f, %6.2f, %6.2f, %4.1f)\n" %
                 ( cust, aft[E_ARF], bef[E_LAT], sol.a(cust), sol.b(cust),
                  sol.task.cust[cust][SRV], sol.d(bef[E_FRO], cust), wgt, sol.dem(cust) )
                )
        result += "\n"
    print result

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

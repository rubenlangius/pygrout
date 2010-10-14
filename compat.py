
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
    
def print_like_Czarnas(sol):
    """Prints the solution in a form compatible (and diffable) with Czarnas."""
    result = solution_header(sol)
    
    for rt, num in zip(sol.r, count(1)):
        result += "Route: %d, len: %d, dist: %.3f, max cap: %.2f" % (
                num, rt[R_LEN], rt[R_DIS], rt[R_CAP])
        result += ", route: "+"-".join(
                str(e[E_FRO]) for e in rt[R_EDG][1:] )+"\n"
    print result

def print_like_Czarnas_long(sol):
    """Prints a verbose description of the solution (one line per customer).
    Compatible with the printSolutionAllData() method in the reference code"""
    result = solution_header(sol)
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

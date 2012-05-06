#!/usr/bin/env python

# may perhaps even work on systems without numpy
try:
    import numpy
except:
    numpy = type('dummy', (object,), dict(float64=float))()

from vrptw import VrptwTask

class DummyTask(VrptwTask):
    def __init__(self, cust = [
            [0, 0, 0, 0, 0, 20, 0],
            [1, 1, 1, 20, 1, 5, 1],
            [2, 2, 2, 10, 0, 8, 1],
            [3, 1, 2, 15, 4, 15, 1],
            [4, 0, 1, 5, 10, 18, 1]
                               ], Kmax = 10, capa = 200):
        self.name = 'test'
        self.cust = cust
        self.Kmax = Kmax
        self.capa = capa
        self.N = len(self.cust)-1
        self.precompute()
        self.best_k, self.best_dist = None, None

def test_savings():
    """Check the savings (Clarke, Wright) construction method."""
    from vrptw import VrptwSolution
    sol = VrptwSolution(DummyTask())
    from pygrout import build_by_savings
    build_by_savings(sol)
    assert sol.check()

def test_deepmap():
    """Check the utility for mapping nested lists and dictionaries."""
    from organize import deepmap
    from math import sqrt
    plain_list = [1, 3, 2]
    assert deepmap(lambda x: x+1, plain_list) == [2, 4, 3]
    plain_dict = dict(a='hi', b='there')
    assert deepmap(str.upper, plain_dict)==dict(a='HI', b='THERE')
    nested = {'a': [0, 4, 16], 'b': [1, 9, 25]}
    assert deepmap(sqrt, nested)=={'a': [0., 2., 4.], 'b':[1., 3., 5.]}

def _rec_assert_simmilar(a, b):
    assert type(a)==type(b), 'wrong types: %s and %s' % (type(a), type(b))
    if type(a) == list or type(a)==tuple:
        for pair in zip(a, b):
            _rec_assert_simmilar(*pair)
    elif type(a) == int:
        assert a == b
    elif type(a) == float or type(a)==numpy.float64:
        assert abs(a-b) < 1e-4
    else:
        assert False, 'unexpected type: '+str(type(a))

def test_flattening():
    """Checks the format for interchange with other programs, like grout."""
    from pygrout import (VrptwSolution, VrptwTask, build_first,
        print_like_Czarnas)
    task = VrptwTask('solomons/rc208.txt')
    s1 = VrptwSolution(task)
    build_first(s1)
    print_like_Czarnas(s1)
    data1 = s1.flatten()
    print data1
    s2 = VrptwSolution(task)
    s2.inflate(data1)
    print "Ok, inflated... Let's see:"
    print_like_Czarnas(s2)
    print s2.flatten()
    assert s2.check()
    assert s2.flatten()==data1
    _rec_assert_simmilar(s1.get_essence(), s2.get_essence())

# possible similar tests: test for assign, copy,
# {get,set}_essence of Solution. But these work already.

def test_find_pos():
    """Check consistency of finding the best position in a route."""
    from pygrout import (VrptwSolution, VrptwTask, build_first,
        print_like_Czarnas, find_bestpos_on, find_allpos_on, R_EDG)
    sol = VrptwSolution(VrptwTask('solomons/rc206.txt'))
    build_first(sol)
    for i in xrange(sol.k):
        for c in sol.r[i][R_EDG][1:]:
            for j in xrange(sol.k):
                if i <> j:
                    best = find_bestpos_on(sol, c[0], j)
                    allp = list(find_allpos_on(sol, c[0], j))
                    print "Best:", best, "all:", allp
                    if best == (None, None):
                        assert allp == []
                    else:
                        assert best in allp
                        assert best == max(allp)

def test_argmap():
    """Test of the class ArgMap from helper.py"""
    try:
        from helper import ArgMap
    except ImportError:
        print "Missing something: PyQt or matplotlib etc."
        return
    from glob import glob
    m = ArgMap()
    files = glob("solomons/r1*")
    files.sort()
    m.add(files)
    assert m(files[0]) == 1
    assert m.ticklabels == ['r101', 'r106']
    assert m.ticks == [1, 6]

# Test left out, reenable in case of trouble ;)
def _test_initial_creation():
    """Unit test for creating solutions to all included benchmarks."""
    from pygrout import VrptwSolution, VrptwTask, build_first
    def check_one(test):
        s = VrptwSolution(VrptwTask(test))
        build_first(s)
        assert s.check()==True, 'Benchmark %s failed at initial solution' % test
    from glob import iglob

    # Homberger's are too heavy
    # from itertools import chain
    # tests = chain(iglob("solomons/*.txt"), iglob('hombergers/*.txt'))
    tests = iglob("solomons/*.txt")
    for test in tests:
        yield check_one, test

if __name__ == '__main__':
    test_savings()

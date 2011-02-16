#!/usr/bin/env python   

# may perhaps even work on systems without numpy
try:
    import numpy
except: 
    numpy = type('dummy', (object,), dict(float64=float))

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

# Test left out, reenable in case of trouble ;)
def _test_initial_creation():
    """Unit test for creating solutions to all included benchmarks."""
    from pygrout import VrptwSolution, VrptwTask, build_first
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

if __name__ == '__main__':
    test_flattening()

#!/usr/bin/env python
    
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


'''

from multiprocessing import Process, Lock
from time import sleep

def f(l, i):
    l.acquire()
    sleep(1)
    print('hello world', i)
    l.release()

if __name__ == '__main__':
    lock = Lock()

    for num in range(10):
        Process(target=f, args=(lock, num)).start()


optimize(args)

Co dalej?

- spokojnie:

1. Konfiguracja operatorów i algorytmów

init_greedy
init_rep_greedy
init_rand_greedy

'''

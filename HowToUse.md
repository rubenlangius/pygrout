# Introduction #

First of all, currently there is no true installation of the pygrout program. It can be installed as a Python extension with distutils. But for optimizing the known, and included Solomon's and Homberger's benchmark instances, it is recommended to check out the [source](http://code.google.com/p/pygrout/source/checkout) from Mercurial and run the program `pygrout.py` from its main directory (i.e. `pygrout/`).

When run without parameters, `pygrout.py` will start optimizing one default problem instance (for example the rc210\_1), using a default command (`poolchain` at the time of writing).

To see more options, the list of commands etc., execute:
```
python pygrout.py -h
```
or
```
python pygrout.py --help
```

which will produce the typical help output as compiled by the `argparse` module. Feel free to experiment with various options, however some of them require specific modules to be installed (like [NumPy](http://numpy.scipy.org) or [mpi4py](http://mpi4py.scipy.org/)).

When using the MPI mode it is advised to run the interpreter through the the `mpiexec` or `mpirun` utility (as provided by your MPI implementation). It may be pointless or impossible to run the cluster mode as a standalone process.

# Details #

# Additional programs #

Except for `pygrout.py` this project has other tools for purposes other than optimizing problem instances. They are:

  * `test.py` - a tiny test suite for `py.test` unit test framework. Covers little, but sometimes used to test new features
  * `organize.py` - processing the results of `pygrout.py` runs, see [ProcessingResults](ProcessingResults.md)
  * `get_best_results.py` - a script to refresh local best known results for the instances (mainly Homberger's, the Solomon's ones are unlikely to change often)
  * `show_profile.py` - a little tool for displaying parts of a profiling statistics. If `pygrout.py` is run with profiling, we can display a sorted summary (it's a CLI wrapper around the `pstats` module)
  * `pbs_helper.py` - _(deprecated)_ a not very successful tool for running `pygrout.py` through a Portable Batch System (PBS) implementation (like Torque). It is not very general at present, but it features walltime calculation for runs of packs of benchmark instances (to make it more supercomputer-friendly: reduce the number of jobs in the queue).
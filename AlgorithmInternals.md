# Introduction #

This is a description of the underlying data structures and API which power `pygrout`. It is a work in progres, but it is to be kept in sync with current development status.

# Data structures #

## VrptwTask (class) ##

This class is an encapsulation of a typical benchmark of VRPTW, as published by Solomon or Homberger.

It is constructed by passing an open stream (file-like object) to the constructor. It is better to provide an actual file, as the filename associated with this string is remembered in the object. This is used by functions like `load_solution` to recreate the task object of the stored solution.

## VrptwSolution (class) ##

This class is a representation of a solution to a VRPTW instance - one described by an object of VrptwTask.

The VrptwTask this solutions refers to must be passed to its constructor, like:

`sol = VrptwSolution(VrptwTask('hombergers/c1_2_1.txt')`

Which creates an _empty_ (having no routes, i.e. not correct yet) solution object. To obtain a solution now, we have to use the function `build_first(sol)`. This way produces a solution created with a greedy heuristic.


# Parallel scheme #

The scheme is as follows:

![https://lh3.googleusercontent.com/_pindFWIOGO4/TZCGB8EjGbI/AAAAAAAABKs/ze9Y_nmYqjk/s400/text4225.png](https://lh3.googleusercontent.com/_pindFWIOGO4/TZCGB8EjGbI/AAAAAAAABKs/ze9Y_nmYqjk/s400/text4225.png)

The master has some storage for solutions (to be more precise: solution essences). Every worker has a capacity for one current solution. Workers try to obtain a solution for further processing from the first queue, if none are currently available - a new one is created, and later processed for a specified amount of time. Then the results are placed on the second queue. The second queue is, in the mean time, read by the master and the resulting solutions can be sent to the first queue again (for another round of optimization) or discarded. The best 15 are kept, and if a result is worse than all of them, it doesn't go to the first queue, and one of the 15 is sent instead.

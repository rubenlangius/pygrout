

# Introduction #

This page is meant as a longer descriptions of the development of this program. It will not describe plans, but actual changes and improvements. Some of them may finally not be useful.

It can be a companion to the terse hg log, which is not a good place to write about less important, or usage related stuff.

Even if not that often, this page may change significantly more times than other pages.

# Changes #

## 2012 ##

Around April, 1'st (not joking), new data structures were created in C++, and are under development. A wrapper for use in Python is planned only after the implementation of Nagata's Route Minimization is completed (which may take a while). Why is this neccessary? Please take a look at the best results table at sintef, and count the entries labelled BC, BC2, BC3 and BC4.

## 2011-08 ##

### 2011-08-31 ###

The "maybe faster savings" are indeed faster - by factor of 20x in case of all Solomon problems. This is due to the fact, that mfsavings is O(n<sup>2</sup>), while "classic" savings is about O(n<sup>3</sup>). The latter needs far to much to apply it to Homberger tests.

Below is a plot of 200-customer problems, solved with mf\_savings:

<table><tr><td><a href='https://picasaweb.google.com/lh/photo/hvuYRa_uwdi24uPmP9e0Gw?feat=embedwebsite'><img src='https://lh4.googleusercontent.com/-a9RRDY_d3QI/Tl3sgqeYYrI/AAAAAAAABNs/wRhrpUoQCsE/s288/mfs.png' height='91' width='288' /></a></td></tr><tr><td></td></tr></table>

### 2011-08-18 ###

The saving of plots required less hassle than thought before - it's all there together with zooming, thanks to NavigationToolbar2QTAgg. Current plans: implementing other methods + offloading the data processing.

### 2011-08-17 ###

The plotting of best solutions was added - and the UI now is almost ready for comparing different methods of solution construction (currently it can compare the best known solution to savings method with different wait time limits - or without limit). Plans: saving the images (an example could then be shown here).

### 2011-08-11 ###

The `helper.py` now has a progress bar - mainly for tranquilizing the user. I haven't tested it on a machine with a significant number of cores/cpus. Some sort of distributing the computation could also be desirable (like a headless server, not requiring matplotlib and PyQt4...).

### 2011-08-03 ###

The plotting was moved to a separate thread, and made parallel with `multiprocessing.Pool`. Still very slow. Benchmark group selection by glob patterns (predefined in a list) - working.

### 2011-08-02 ###

The plotting for savings method (which previously was buggy, because of not obeying the capacity constraints), is _almost_ ready: it only blocks the UI of the `helper.py` program (which is a GUI in Qt for investigating the savings method -- at least for now).

On an AMD P320 it currently takes 270 s to process the Solomon benchmarks, which is far too long (while loading all of them in a loop is instantaneous). The loading of all Homberger benchmarks in a loop lasts about 35 seconds (if there are 2GB of RAM free, with no need to swap out, with swapping it goes up to ~50 s, user time about identical). A little test will be performed for the Hombergers to plot, but this is risky (will it end before the dusk?).

For measuring such times, now there is a special toylike class - [StopWatch](http://code.google.com/p/pygrout/source/browse/stopwatch.py), it is inspired by a typical stopwatch - you just start it, and then can print the current time (since the start). Haven't found such a class in the standard library.

## 2011-07 ##

### 2011-07-12 ###

The API has now been a little refactored: much was moved into a new package-module `vrptw`, which includes the data structures and basic operations on them. Some are currently not undoable, while most are; I'll implement the other versions if they are needed (not undoable = permanent change to solution, may have better performance).

The `pygrout` module and entry point to the "main application" contains the specific algorithms, along with application managing code (commandline options parsing, etc.).

A new method for initializing solutions is `build_by_savings(VrptwSolution: sol)`. It is a simplification of the Clarke and Wright's (1964) method. It came last in Solomons (1987) comparison of methods. Despite some manouvre possible through two parameters to it, I expect it to be the worst available method. But this needs actual tests to be proven.

## 2011-05 ##

### 2011-05-28 ###

There will be a slight change in saving of results - the name now includes a part of an MD5 digest of the solution. This can prevent overwriting of solutions with different routes, but same value (if they ever happen). Besides - if the solution is already there, it will not get overwritten, but a message will be emitted. There may be (later) another variant - for counting the multiplicity of some solution obtained - like another suffix in the name.

### 2011-05-21 ###

The new commands for organize.py: `k_map` and `dist_map` display a summary of results in the current directory.

Below is an example of the k\_map for the present results:

![https://lh5.googleusercontent.com/-VKQkGLUJEBA/TfYu3YKtGMI/AAAAAAAABMM/4NB3cbR0hDY/s640/kmap.png](https://lh5.googleusercontent.com/-VKQkGLUJEBA/TfYu3YKtGMI/AAAAAAAABMM/4NB3cbR0hDY/s640/kmap.png)

The right number of routes is easily obtained for [R2](https://code.google.com/p/pygrout/source/detail?r=2) family, moderately difficult for [R1](https://code.google.com/p/pygrout/source/detail?r=1) and RC1, hard for RC2, and -- paradoxally -- hardest for C1 and C2...

## 2011-04 ##

### 2011-04-21 ###
The `setup.py` (a `distutils` script) was developed and it installs pygrout as a Python extension. The necessary data (best known solutions, the benchmark problems) are installed along. This is one option to have an "installer", still problematic is the loading of the test cases (tied to argument parsing...). Installed are a few modules (`pygrout`, `compat` and `undo`), a real package (`vrptw`), with `consts` as a member module, and two fake packages with benchmark instances.

### 2011-04-07 ###
The `organize.py` can now do plots of the number of routes above "optimum", which is the currently best known solution.

Example (output from `python organize.py plot_excess_routes`):
![![](https://lh5.googleusercontent.com/_pindFWIOGO4/TZ2oBsAZRHI/AAAAAAAABLU/YLgNcs_t6YM/s144/eg_excess_routes.png)](https://lh5.googleusercontent.com/_pindFWIOGO4/TZ2oBsAZRHI/AAAAAAAABLU/YLgNcs_t6YM/s800/eg_excess_routes.png)
This shows the route counts for H group, which were obtained around 24<sup>th</sup> Feb. It searches for results in the current directory (where the script is run), and parses their filename. A special page, ProcessingResults will contain more details about it.

## 2011-03 (in HPC-E2) ##

### 2011-03-03 (Thu) ###

Today a MPI-based parallel scheme was developed. It uses passive workers, central master commanding them, and blocking single communications (the blocking communication was neccessary because of lack of variable-length non-blocking receive).

The method is not yet at its full strength, however it already produced some results. It wasn't executed as long as the `poolchain` version: only 10 minutes or until the exhaustion of further ideas by the master. Actually, the master only commanded the workers to eliminate different routes by using the `op_route_min` function. At first the workers construct different solutions (using different orderings of customers), and then the master produces _k_ new tasks of each 'successful' result, i.e. every initial solution not worse by more than 1 route than current best, and every solution, that successfully had one of its routes eliminated.

This method found best-_k_ solutions for 26/56 S, and 92/300 H. That's 10 non-trivial S and 34 non-trivial H. There are new H: `c2_4_5 c2_6_5`, which is quite good news, because there were few results in the H's C-group.

So, current state is: S 42/56, H 143/300.

## 2011-02 (in HPC-E2) ##

### 2011-02-28 (Mon) ###

An update of the minimal-known-k results:

S: 42/56, new: `r202 rc202`

H: 141/300, new:
```
r1_4_5 r1_8_9 r1_810 
r2_2_2 r2_4_1 r2_6_5 r2_8_1 r210_1 r210_5 
rc1_2_5 rc11010 
rc2_2_2 rc2_2_3 rc2_4_3 rc2_6_3 rc2_610 
rc2_8_8 rc2_810 rc210_3 rc210_8 rc210_9 rc21010
```

These results were accomplished by employing the new operator of (attempted) shortest route elimination (program option "`-- op op_tabu_shortest`", which had a previous error in determining the number of customers to remove and crashed during the computations.

The operator was tested specifically on the benchmark for which no best-k were found yet.

### resume command ###
The resume command is a way to execute the function `op_route_min` (not yet a true "operator" by pygrouts terms), on a solution which has been saved previously (i.e., a `.p` file with pickled solution data). It is a try to remove a single route at once, and then to insert them into the remaining routes. The removed customers are held in a data structure known as the ejection pool (EP), which was so named by Lim and Zhang in their 2007 paper.

### grout interface ###
The simulated annealing engine `grout`, which is written in C++, is a derivative work of the code by Czech and Czarnas, which they used for solving Solomons tests successfully. I obtained the code from one of the authors, but I don't know the actual copyright status, so I am not distributing the source code. I created a SWIG wrapper around the code, which can be used in pygrout e.g. for post-processing a result (the old C++ code is good at reducing the distance of routes). Thanks to the wrapper, the code can be parallelized using `multiprocessing` or `mpi4py`. Now there is a command `grout`, which can be used as the resume command - to post-process a
saved solution. This feature may be removed in the future.

### 2011-02-27 (Sun) ###

[Initial solutions](BuildFirstSolution.md) where analyzed for finding the best metod of constructing them (most beneficial arrangement of inserted customers). The results are [here](BuildFirstSolution#Arrangements_performances.md).

### 2011-02-25 (Fri) ###

After analysing the results (with improved `organize.py`, which can show progress differentially), the current state of tests, for which the algorithms found vehicle-number-acceptable solutions is:

Solomons: 40 (new: `r207`)

Hombergers: 119 (new: `r1_4_2 r1_6_3 r1_6_4 r1_6_7 r1_6_8 rc1_6_2 rc110_2 rc2_2_1`)

The op\_tabu\_shortest had a problem with randomly choosing removed customer range, and cost processes hung, which lead to no results. The simple op\_tabu\_single produced one route-best solution.

The current status of solutions with best route count will be presented on a dedicated [page](MinimalRouteCount.md).

### 2011-02-24 (Thu) ###

The new subcommand of pygrout, `initials` was used to create initial solutions of all tests (several times, with random order being tested 10x for each test case). The subcommand itself only does build one solution sequentially, but it can be carried out in parallel by calling it from a [special script](http://code.google.com/p/meats/source/browse/trunk/dopar/dopar2.py). Although it has much overhead, the use of multiprocessing can still provide significant speedup, for example 13.18 on 16 processes (0.82 efficiency).
```
time dopar2.py "./pygrout.py ARG initials" solomons/* hombergers/*
real    7m35.010s
user    99m36.451s
sys 0m21.717s
```
The results are for a Xeon(R) X5560. Ram usage during this computation was high, because of running 16 processes simultaneously. The 16 threads are not actual cores, so the results will be different for a single, long run. It's result will be added soon, however there should be a smaller user-time than right now, not only because of using a full core (unshared between threads) but also because of the automatic [overclocking technology](http://en.wikipedia.org/wiki/Intel_Turbo_Boost) in this chip.

A local search with restrictions (inserting only to different routes, prefering longer routes as targets) was prepared. Results will be on friday.

### 2011-02-23 (Wed) ###

Because of a dramatic bug the results which came overnight have very little actual value. The tested nb-hood operator was not modifying the solution, being effectively a no-op. This incident was, however, an occasion to study the quality of initial solutions more thorougly. Read more on the wiki page: [EasyTests](EasyTests.md).

### 2011-02-22 (Tue) ###

There were a number of changes to the program itself (`pygrout.py`), as well as to its utilities (`pbs_helper.py`, `organize.py` and `show_profile.py`). The PBS helper is still not useful - creating a separate job for every run turned out not to work in practice (queue length limit). Organizing of results now is able to show difference and intersection of files; and profile can sort the results by different criteria (cmdline i-face).

The changes encompass removal of --run/-e and --preset options (@operation decorated functions). The decorator was left, but it now decorates "operators" - like op\_greedy\_multiple etc., and can be selected for local\_search, run\_all, and poolchain with the --op option.

Using a poolchain, 16 workers (1 8-core Nehalem blade), and 35 min wall time, further results with the right vehicle count were found.

The current state is:

Solomon collection: found 39 out of 56.
```
c101 c102 c103  c104  c105  c106  c107  c108 c109 
c201 c202 c203  c204  c205  c206  c207  c208 
r101 r102 r103        r105  r106 
r201      r203        r205  r206        r208 r209 r210 
          rc103 rc104             rc107 
rc201     rc203 rc204 rc205 rc206 rc207 rc208
```

Homberger collection: found 102 of 300.
```
c1_2_1 c1_2_3 c1_2_4 c1_2_5 c1_2_6 c1_2_7 
c1_4_4 
-
-
-

c2_2_1 c2_2_2 c2_2_3 c2_2_4 c2_2_5 c2_2_6 c2_2_7 c2_2_8 c2_2_9 c2_210 
-
-
-
-

r1_2_2 r1_2_3 r1_2_4 r1_2_5 r1_2_6 r1_2_7 r1_2_8 r1_2_9 r1_210 
r1_4_3 r1_4_4 r1_4_6 r1_4_7 r1_4_8 r1_410 
r1_6_1 
r1_8_3 r1_8_4 r1_8_6 r1_8_7 r1_8_8 
r110_1  

r2_2_3 r2_2_4 r2_2_5 r2_2_6 r2_2_7 r2_2_8 r2_2_9 r2_210 
r2_4_2 r2_4_3 r2_4_4 r2_4_5 r2_4_6 r2_4_7 r2_4_8 r2_4_9 r2_410 
r2_6_3 r2_6_4 r2_6_6 r2_6_7 r2_6_8 r2_6_9 r2_610 
r2_8_2 r2_8_3 r2_8_4 r2_8_5 r2_8_6 r2_8_7 r2_8_8 r2_8_9 r2_810
r210_2 r210_3 r210_4 r210_6 r210_7 r210_8 

rc1_2_2 rc1_2_3 rc1_2_4 rc1_2_6 rc1_2_7 rc1_2_8 rc1_2_9 rc1_210 
rc1_4_3 rc1_4_4 
rc1_6_3 rc1_6_4 
-
rc110_4

rc2_2_4 rc2_2_8 rc2_2_9 rc2_210 
rc2_4_4 rc2_4_8 rc2_4_9 rc2_410 
rc2_6_4 
rc2_8_4
rc210_4 
```
Of course, the distance has not been matched in most cases.

### 2011-02-20 (Sun) ###

A next feature of this project is a utility for separating "good" results from bad ones: `organize.py`. It should be executed inside a directory containing pickled solutions (`*.p`) files). It will create a subdirectory named '100s' and move files which contain the string '-100.0-' into that subdirectory. Currently these are the files, which have the right number of routes (i.e. the number of world's best known solutions).

With its help it was possible to determine, how many of the known benchmarks (by Solomon, and Gehring and Homberger), were solved by `pygrout` so far.

S: 35 out of 56

H: about 50, of various lengths, out of 300
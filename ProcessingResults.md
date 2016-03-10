# Introduction #

After a typical run, the pygrout program will produce a file with the solution that was found. The default directory for placing these files is `output/` and the files name contains useful information about the final value: routes and distance, together with their percentages of best solution's corresponding values.

# Details #

After running pygrout a number of times, we can use the script `organize.py` to separate the solutions for which the best route count was achieved from the rest.

It is often recommended to put the results of one specific 'session' into a separate directory, for example like this:

```
$ cd output/
$ mkdir YYYY_mm_dd_testing_algo_X
$ mv *.p !$
$ cd !$
$ ../../organize.py main
```

The last command sorts out the solutions with optimal (supposedly optimal) solutions into a subfolder `100s`. A summary is displayed, and text files are created -- they contain the names of the problem instances which were solved: a file named `bad.txt` contains problems which have had suboptimal results in the current directory (in terms of route count), and for example the `100s/good.txt` lists instances which had optimal results.

## Searching for specific solutions ##

Except for that, we can perform some different kinds of analysis, for example to use standard posix tools to search for certain information. If the current directory is either the program's main directory or the output/ directory, we can issue a command:

```
$ find -name rc1_4_6*.p | sed 's/\(.*\/\)\(.*\)/\2   (\1)/g' | sort
```

to search for all solutions of the Homberger's rc1\_4\_6 problem, sorted according to value, and also showing the directory where the file resides (which may be related to the method which led to its obtainment).
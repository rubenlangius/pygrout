# Introduction #

The mechanism for building the initial solution is a simple greedy procedure. A simple explanation would be: "insert customers by one, choosing the best place in existing routes, and inserting customer to a new route only if no such place exists".

# Algorithm details #

The pseudocode of it is as follows:

```
existing_routes = {}

for c in *customers*, sorted by arbitrary *order*, do:

    route, position = find_best_position_in(existing_routes)

    if not_found(route) then:
        insert_coustomer_to_route_at_pos(c, route, position)

    else:
        route = create_new_route()
        existing_routes += route
        insert_coustomer_to_route_at_pos(c, route, 1) 

    endif
endfor
```

The **order** is important, and changes the final outcome, leading to better or worse route counts and distances.

# Debugging initial creation #

Pygrout has a command, `initials`, which can help to debug the quality of created initial solutions. At the time of writing, a typical output of this command is as follows:
```
Sum of q: 17940 (k_min >= 90), Q(0..4) = 10 10 20 20 40
Best known solution for test c11010: 90 routes, 40586.60 total distance.
by_closing:          147.78 249.32    routes 133   rank 01
by_midtime:          150.00 244.35    routes 135   rank 02
by_opening:          155.56 245.18    routes 140   rank 03
by_opening_desc:     181.11 279.38    routes 163   rank 04
by_random_ord:       185.56 347.84    routes 167   rank 05
by_timewin:          185.56 353.25    routes 167   rank 06
by_random_ord:       186.67 348.25    routes 168   rank 07
by_id:               187.78 352.46    routes 169   rank 08
by_random_ord:       187.78 354.18    routes 169   rank 09
by_weight_desc:      188.89 352.06    routes 170   rank 10
by_random_ord:       188.89 356.81    routes 170   rank 11
by_random_ord:       190.00 355.71    routes 171   rank 12
by_random_ord:       190.00 361.60    routes 171   rank 13
by_timewin_desc:     192.22 356.45    routes 173   rank 14
by_weight:           192.22 358.94    routes 173   rank 15
by_random_ord:       192.22 361.97    routes 173   rank 16
by_random_ord:       193.33 362.89    routes 174   rank 17
by_random_ord:       194.44 370.96    routes 175   rank 18
by_random_ord:       196.67 383.95    routes 177   rank 19
by_midtime_desc:     207.78 309.49    routes 187   rank 20
by_closing_desc:     225.56 346.60    routes 203   rank 21
```

# Arrangements performances #

The command `initials` from `pygrout` builds the initial solution repeatedly for different order functions (arrangement of the inserted customers). There were, at the time of writing, 12 predefined sorting ways:
```
by_closing
by_closing_desc
by_id
by_midtime
by_midtime_desc
by_opening
by_opening_desc
by_random_ord
by_timewin
by_timewin_desc
by_weight
by_weight_desc
```

The `by_timewin` criteria means by the length of the time window, mid-time is the middle of the time window, closing and opening also concern the time windows. The random order is completely random, and it was tested multiple times, ranking differently, but usually at the bottom. Weight was also used, and the criteria `by_id` (which misses its `_desc` counterpart), means no sorting - this causes the arrangement from the test case being used.

To analyze the results of multiple runs of `pygrout <file> initials` concatenated into one file -- in this case `initi_alls.o156925` -- we can issue a following pipeline of shell commands:
```
$ grep "rank 01" initi_alls.o156925 | awk '{print $1;}' | sort | uniq -c | sort -rn
```
This shows the methods which ranked first (in any of the cases, so `by_random_id` has a 10-fold chance). The output was (higher number = better):
```
    233 by_closing:
     40 by_timewin:
     19 by_opening_desc:
     18 by_id:
     16 by_midtime:
     10 by_random_ord:
      9 by_closing_desc:
      6 by_opening:
      4 by_midtime_desc:
      1 by_weight:
```

For initial solutions ranked second (higher number = better):
```
    130 by_midtime:
     66 by_timewin:
     51 by_closing:
     48 by_opening_desc:
     22 by_opening:
     18 by_random_ord:
     13 by_midtime_desc:
      4 by_closing_desc:
      3 by_id:
      1 by_weight:
```

Now, at the other end of the spectrum: ranking 19. (lower number = better):
```
    135 by_random_ord:
     44 by_closing_desc:
     36 by_weight_desc:
     33 by_timewin_desc:
     30 by_midtime_desc:
     19 by_id:
     14 by_opening_desc:
     14 by_opening:
     13 by_weight:
     10 by_midtime:
      6 by_timewin:
      2 by_closing:
```
and 20. (lower number = better):
```
    133 by_random_ord:
     67 by_midtime_desc:
     48 by_timewin_desc:
     34 by_weight_desc:
     20 by_closing_desc:
     11 by_midtime:
     10 by_opening_desc:
     10 by_opening:
      8 by_timewin:
      8 by_id:
      7 by_weight:
```

This, however, doesn't produce an actual image (although it is the 'overall' ranking, but transposed), because additional information, quantitative, must be taken into account.

To rank the methods by created total number of routes, we can use more power of the `awk` tool -- to make sums, and then to scale the `by_random_ord` to fit all other methods:
```
awk '/rank/ {stat[$1]+=$5} END {stat["by_random_ord:"]/=10; for(x in stat) print stat[x] " " x;}' initi_alls.o156925 | sort -n
```

This time the output is (lower number = better):
```
12203 by_closing:
13231 by_midtime:
13548 by_opening:
14445 by_timewin:
14584 by_opening_desc:
15265.4 by_random_ord:
15281 by_weight:
15285 by_id:
16004 by_weight_desc:
16032 by_closing_desc:
16099 by_midtime_desc:
16294 by_timewin_desc:
```

And now, it is easy to create also "average ranking" ranking, by changing the column above to `$7` in the `awk` script:
```
awk '/rank/ {stat[$1]+=$7} END {stat["by_random_ord:"]/=10; for(x in stat) print stat[x]/356 " " x;}' initi_alls.o156925 | sort -n
```

The division by `356` is normalizing to "ranking position", because it is the number of tests, which were performed. Not very suprisingly, the order is the same, as above, but it has a dramatic feature (lower number = better):
```
2.25843 by_closing:
6.74719 by_midtime:
6.74719 by_timewin:
7.53652 by_opening_desc:
7.92697 by_opening:
11.3258 by_id:
11.9157 by_weight:
11.943 by_random_ord:
11.9635 by_closing_desc:
12.8567 by_midtime_desc:
15.7781 by_weight_desc:
16.514 by_timewin_desc:
```

So, average ranking is 1. or 2. for `by_closing`, and then... no prize! The next pretendent `by_midtime` and `by_timewin` (approx. _ex aequo_), are ranked 5. to 6. (that is: on average). Random ordering is in the middle, which is very natural. In reality its spread - most of it are the positions 15-22 (not included int the ranking), but some of it reaches the missing positions 2-4.

(The above observations were made for a population of Homberger tests. The Solomon tests have a bit different characteristics, e.g. by\_weight is not the very worst. However, the ranking is only a bit shifted).

Finally, the average percentage of routes compared against the number of them in the best solution, can be determined by an identical script:
```
awk '/rank/ {stat[$1]+=$2} END {stat["by_random_ord:"]/=10; for(x in stat) print stat[x]/356 " " x;}' initi_alls.o156925 | sort -n
```

```
119.191 by_closing:
130.755 by_midtime:
134.838 by_opening:
136.345 by_opening_desc:
138.547 by_timewin:
148.865 by_id:
148.908 by_weight:
150.053 by_random_ord:
150.985 by_closing_desc:
151.278 by_midtime_desc:
156.132 by_weight_desc:
160.402 by_timewin_desc:
```

So, the best solutions have on average still ~20% more routes than there should be. That is why the route number optimization phase is [usually](EasyTests.md) neccessary.

See also: [EasyTests](EasyTests.md)
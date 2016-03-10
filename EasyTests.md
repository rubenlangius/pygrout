# Introduction #

Finding the minimal number of routes is usually the hardest part of solving VRPTW instances. However, a few instances from the collections by Solomon, and the one by Homberger and Gehring, produce initial solutions with the right number of routes immediately - by using the greedy [heuristic of insertions](BuildFirstSolution.md).

Of course - this does not mean, that these problems are thus solved to optimality. It only means, that they don't require the first stage (of route reduction), which is typical for many succesful method of VRPTW solving.

# Details #

## Solomon 100-customer tests ##
(16)
```
c102 c103 c104 c107 c109 
c201 c202 c203 c205 c207 

-
r201 r203 r206 r210

-
rc204 rc205
```

## Homberger tests ##
(58)
```
r1_2_3 r1_2_4 r1_2_6 r1_2_7 r1_2_8 
r1_4_3 r1_4_4 r1_4_7 r1_4_8 
-
r1_8_2 r1_8_3 r1_8_4 r1_8_6 r1_8_7 r1_8_8 
-

r2_2_3 r2_2_4 r2_2_6 r2_2_7 r2_2_8 r2_2_9 r2_210 
r2_4_2 r2_4_3 r2_4_4 r2_4_5 r2_4_6 r2_4_7 r2_4_8 r2_4_9 r2_410
r2_6_2 r2_6_3 r2_6_4 r2_6_6 r2_6_7 r2_6_8 
r2_8_2 r2_8_3 r2_8_4 r2_8_6 r2_8_7 r2_8_8 r2_8_9
r210_2 r210_3 r210_4 r210_6 r210_7 r210_8 

rc1_2_4 
-
rc1_6_3 
-
rc110_4 

rc2_2_4 
rc2_4_4 
rc2_6_4 
rc2_8_4
rc210_4 
```

# Interpretation #

In the Solomon group, there are several 'easy' test in both the C groups (1 and 2), and then only a few for [R2](https://code.google.com/p/pygrout/source/detail?r=2) and RC2 (and none from their "1" counterparts). It may be conjectured, that the big capacity and long time windows of the "2" family make it easier to obtain good initial solutions by a greedy heuristic.

In Homberger group there are, interestingly, no easy C test cases. The easy tests are among the R group (mostly, with a majority in the "2" group and greatest numbers of them among the 400- and 800-customer collections). The RC subset of Homberger's test cases has only occasional easy tests, and they are mostly the fourth ones (or third, which is similar) - these are always with a long, weakly-constraining time windows, with a large number of open time windows throughout the scheduling time.
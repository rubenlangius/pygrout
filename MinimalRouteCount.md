# Introduction #

The VRPTW solutions are compared by a hierarchical criterion, which prefers route count over total distance. This means, that a solution with 10 routes and 9999 total distance is still better than one with 11 routes and 1000 total distance.

The route count has, arguably, high importance - for a given fleet of vehicles it is much easier to increase the distance, than to find a next vehicle.

# Test cases list #
These are the tests, for which `pygrout` found feasible solutions that had the same number of routes as the currently best-known solutions.
The total distance in not taken into account here.



## Solomon 100-customer tests ##
(42)
```
c101 c102 c103 c104 c105 c106 c107 c108 c109 
c201 c202 c203 c204 c205 c206 c207 c208 

r101 r102 r103 -    r105 r106 -    -    -    -    -    - 
r201 r202 r203 -    r205 r206 r207 r208 r209 r210 -

-     -     rc103 rc104 -     -     rc107 -
rc201 rc202 rc203 rc204 rc205 rc206 rc207 rc208
```

## Homberger tests ##
(141)
```
c1_2_1 -      c1_2_3 c1_2_4 c1_2_5 c1_2_6 c1_2_7 - - -
c1_4_1 -      -      c1_4_4 
-      -      -      c1_6_4 
- - - - - - - - - -
- - - - - - - - - -

c2_2_1 c2_2_2 c2_2_3 c2_2_4 c2_2_5 c2_2_6 c2_2_7 c2_2_8 c2_2_9 c2_210 
-      -      -      -      -      -      -      c2_4_8 - - 
- - - - - - - - - -
- - - - - - - - - -
- - - - - - - - - -

-      r1_2_2 r1_2_3 r1_2_4 r1_2_5 r1_2_6 r1_2_7 r1_2_8 r1_2_9 r1_210 
-      r1_4_2 r1_4_3 r1_4_4 r1_4_5 r1_4_6 r1_4_7 r1_4_8 r1_4_9 r1_410 
r1_6_1 -      r1_6_3 r1_6_4 -      -      r1_6_7 r1_6_8 - -
-      r1_8_2 r1_8_3 r1_8_4 -      r1_8_6 r1_8_7 r1_8_8 r1_8_9 r1_810 
r110_1 - - - - - - - - -

-      r2_2_2 r2_2_3 r2_2_4 r2_2_5 r2_2_6 r2_2_7 r2_2_8 r2_2_9 r2_210 
r2_4_1 r2_4_2 r2_4_3 r2_4_4 r2_4_5 r2_4_6 r2_4_7 r2_4_8 r2_4_9 r2_410 
-      r2_6_2 r2_6_3 r2_6_4 r2_6_5 r2_6_6 r2_6_7 r2_6_8 r2_6_9 r2_610
r2_8_1 r2_8_2 r2_8_3 r2_8_4 r2_8_5 r2_8_6 r2_8_7 r2_8_8 r2_8_9 r2_810
r210_1 r210_2 r210_3 r210_4 r210_5 r210_6 r210_7 r210_8 r210_9 r21010

- rc1_2_2 rc1_2_3 rc1_2_4 rc1_2_5 rc1_2_6 rc1_2_7 rc1_2_8 rc1_2_9 rc1_210
- -       rc1_4_3 rc1_4_4 - - - - - -
- rc1_6_2 rc1_6_3 rc1_6_4 - - - - - - 
- - - - - - - - - -
- rc110_2 rc110_3 rc110_4 -       -       -       -       -       rc11010 

rc2_2_1 rc2_2_2 rc2_2_3 rc2_2_4 -    -    -       rc2_2_8 rc2_2_9 rc2_210 
                rc2_4_3 rc2_4_4                   rc2_4_8 rc2_4_9 rc2_410 
                rc2_6_3 rc2_6_4                                   rc2_610 
                        rc2_8_4                   rc2_8_8         rc2_810 
                rc210_3 rc210_4                   rc210_8 rc210_9 rc21010
```

See also: [EasyTests](EasyTests.md)
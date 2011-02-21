#!/usr/bin/env python

import sys
import pstats
import os

def main():
    if len(sys.argv) == 1:
        print "\nUsage: %s profile_output [sort_order] [profile_output...]" % sys.argv[0]
        print "Where sort order may be: time (default), cumulative, ..."
        print """ (taken from documentation:)

        Valid Arg                Meaning
        'calls'         call count
        'cumulative'    cumulative time
        'file'          file name
        'module'        file name
        'pcalls'        primitive call count
        'line'          line number
        'name'          function name
        'nfl'           name/file/line
        'stdname'       standard name
        'time'          internal time 
"""
        exit()
    s = pstats.Stats(sys.argv[1])
    order = 'time'
    for f in sys.argv[2:]:
        if os.path.exists(f):
            s.add(f)
        else:
            order = f
    s.sort_stats(order).print_stats(20)
    
if __name__ == '__main__':
    main()

#!/usr/bin/env python

import sys
import pstats
import os

def main():
    if len(sys.argv) == 1:
        print "\nUsage: %s profile_output [sort_order] [profile_output...]" % sys.argv[0]
        print """ 
        Where sort order may be: time (default), cumulative, ...
        (taken from documentation:)

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

    # the process

    order = 'time'
    files = filter(os.path.exists, sys.argv[1:])
    
    extra = list(set(sys.argv[1:])-files)
    if len(extra) > 0:
        order = extra[0]
        if len(extra) > 1:
            print "Warning: excess args ignored:", extra[1:]

    s = pstats.Stats(files[0])
    map(s.add, files[1:])

    num_rows = int(os.getenv('ROWS', '20'))
    s.sort_stats(order).print_stats(num_rows)
    
if __name__ == '__main__':
    main()

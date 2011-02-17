#!/usr/bin/env python

import sys
import pstats

def main():
    print sys.argv
    if len(sys.argv) == 0:
        print "Usage: %s profile_output [profile_output...]" % sys.argv[0]
        exit()
    s = pstats.Stats(sys.argv[1])
    for f in sys.argv[2:]:
        s.add(f)
    s.sort_stats('time').print_stats(20)
    
if __name__ == '__main__':
    main()
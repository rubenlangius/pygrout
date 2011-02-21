#!/usr/bin/env python

import os
import re
import glob
import textwrap

# regex to remove the non-setname part of name
cutoff = re.compile('-.*')

def find_medium(test):
    """Glob and return all but the 'smallest' and 'largest' files."""
    # missing case-insensitive glob. Besides, this Solomon's mess
    # with capital C, R and RC might be worth cleaning up...
    return sorted(glob.glob(test+'*.p')+glob.glob(test.upper()+'*.p'))[1:-1]

def read_as_set(f):
    """Read the file and return set of lines."""
    return set(map(str.strip, open(f)))

def printf(set_):
    """Used to display a set, with count, sorted and textwrapped."""
    print "(%d)"%len(set_)
    print textwrap.fill(" ".join(sorted(set_)))

def compare(*args):
    """Read in the passed files and display differences."""
    if len(args) < 2:
        print "Provide at least two filenames to compare."
        return
    if len(args) > 2:
        print "Warning: only 2 files work now."
    first, secnd = map(read_as_set, args[:2])
    print "Only in%s:" % args[1]
    printf(first.difference(secnd))
    print "Only in %s:" % args[2]
    printf(secnd.difference(first))
    print "In both:"
    printf(first.intersection(secnd))    
    
def union(*args):
    """Read in the passed files and display the union (set sum)."""
    
    # helpers (maybe later global)
    
    def sel_solomons(set_):
        """Select Solomon test names (only full 100 customer)."""
        return set(filter(re.compile('r?c?\d{3}$').match, set_))

    def sel_homberger(set_):
        """Select Solomon test names (only full 100 customer)."""
        return set(filter(re.compile('r?c?[0-9_]{5}$').match, set_))
    
    if len(args) < 1:
        print "Provide at least two filenames to add together."
        return
        
    sets = map(read_as_set, args)
    sum_of_all = set.union(*sets)
    print "All found results are:"
    printf(sum_of_all)
    print "Including junk:"
    printf(sum_of_all.difference(
        sel_solomons(sum_of_all), sel_homberger(sum_of_all)))
    print "Full Solomon tests:"
    printf(sel_solomons(sum_of_all))
    print "Homberger tests:"
    printf(sel_homberger(sum_of_all))
    
def missing(*args):
    """List problem sets which are missing from all the arguments."""
    
    def gen_hombergers():
        """Set of all Homberger instance names."""
        return set([ c+n+s+x 
            for c in ['c','r','rc'] 
            for n in ['1', '2']
            for s in ['_2','_4','_6','_8','10'] 
            for x in (['_%d' % i for i in xrange(1,10)]+['10']) ])
    
    sum_of_all = set.union(*map(read_as_set, args))
    
    hombergers = gen_hombergers()
    print "Missing Homberger tests:"
    difference = hombergers.difference(sum_of_all)
    if difference == hombergers:
        print "(ALL %d)" % len(hombergers)
    else:
        printf(difference)

    
def main():
    """Main function - clean up a typical /output (sub)directory."""
    
    # helpers 
    
    def create_file(fn, set_):
        if not os.path.exists(fn):
            open(fn, 'w').write("\n".join(sorted(set_)))
        else:
            present = read_as_set(fn)
            if present <> set_:
                print "File %s present, but inconsistent, differences" % fn
                printf(present.symmetric_difference(set_))

    # ensure directory for best results (k == 100%)
    
    if not os.path.exists('100s') and os.path.basename(os.getcwd()) <> '100s':
        print "Creating directory 100s (best-k results)"
        os.makedirs('100s')
    else:
        print "Directory 100s already present"
        
    # move best results to their directory (also their .vrp companions)
    
    solved = re.compile('[^-]+-100.0-.*')
    sol_ok = filter(solved.match, glob.glob('*.*'))
    if len(sol_ok):
        print "Moving %d best-k results to 100s:" % len(sol_ok)
        for f in sol_ok:
            print f
            os.rename(f, os.path.join('100s',f))
    else:
        print "No best-k results found here."
    
    # ensure there is an up-to-date all_list.txt, read results
    
    present = set(glob.glob('*.p'))
    if os.path.exists('all_list.txt'):
        files = read_as_set('all_list.txt')
        if not files >= present:
            print "all_list.txt missing files:"
            printf(present.difference(files))
            files = files.union(present)
            open('all_list.txt', 'w').write("\n".join(sorted(files)))
    else:
        # there was no all_list.txt
        open('all_list.txt', 'w').write("\n".join(sorted(present)))
        files = present
            
    # grouping of the results to different sets

    sets_bad = set(cutoff.sub('', f).lower() for f in files)
    
    # good sets are always in the 
    
    sets_good = set(cutoff.sub('', f.replace('100s/','')).lower()
                    for f in glob.glob("100s/*.p"))
    sets_sometimes = sets_bad.intersection(sets_good)
    sets_always = sets_good.difference(sets_bad)
    sets_never = sets_bad.difference(sets_good)
    
    # print summaries (for every run)
        
    print "\nBad results:"
    printf(sets_bad)
    print "\nGood results:"
    printf(sets_good)
    print "\nSolved sometimes:"
    printf(sets_sometimes)
    print "\nSolved never:"
    printf(sets_never)
    print "\nSolved always:"
    printf(sets_always)
    
    # remove junk - medium solutions (conditionally)
    
    if len(present) > 2*len(sets_bad):
        if 'y' == raw_input('Delete medium solutions (y/N)?'):
            for i in sets_bad:
                moritures = find_medium(i)
                print i, len(moritures)
                for f in moritures:
                    print "Removing", f, "..."
                    os.unlink(f)

    # create lists for bad, never and sometimes
    
    create_file('never.txt', sets_never)
    create_file('bad.txt', sets_bad) # broadest
    create_file('sometimes.txt', sets_sometimes)
    create_file('100s/sometimes.txt', sets_sometimes)
    create_file('100s/good.txt', sets_good) # broadest
    create_file('100s/always.txt', sets_always)

# global list of functions
from types import FunctionType
funcs = filter(lambda k: type(globals()[k])==FunctionType, globals().keys())

if __name__ == '__main__': 
    # my well-known "call-function-from-argv" design pattern
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in funcs:
        # call function passing other args as params
        print globals()[sys.argv[1]](*sys.argv[2:])
    else:
        main()


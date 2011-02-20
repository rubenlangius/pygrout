#!/usr/bin/env python

import os
import re
import glob
import textwrap

# regex to remove the non-setname part of name
cutoff = re.compile('-.*')

def find_medium(test):
    """Glob and return all but the 'smallest' and 'largest' files."""
    return sorted(glob.glob(test+'*.p'))[1:-1]
    
def main():
    """Main function - clean up a typical /output (sub)directory."""
    
    # helpers 
    
    def printf(set_):
        print "(%d)"%len(set_)
        print textwrap.fill(" ".join(sorted(set_)))
        
    def read_as_set(f):
        return set(map(str.strip, open(f)))
        
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
        
    # move best results to their directory
    
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
    
if __name__ == '__main__': main()


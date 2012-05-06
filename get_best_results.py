#!/usr/bin/env python

homberger_urls = [
    'http://www.sintef.no/Projectweb/TOP/Problems/VRPTW/Homberger-benchmark/%d00-customers/' % n
    for n in xrange(2,11,2)
]

solomon_urls = [
'http://web.cba.neu.edu/~msolomon/c1c2solu.htm',
'http://web.cba.neu.edu/~msolomon/r1r2solu.htm',
'http://web.cba.neu.edu/~msolomon/rc12solu.htm',
'http://web.cba.neu.edu/~msolomon/heuristi.htm'
]

import re
import urllib2
import time

homb = re.compile(r'''<td style.*?([rc]{1,2}[12][0-9_]{4,6}).*?<td.*?(\d+).*?<td.*?([\d\.]+)''', re.DOTALL)
solo = re.compile('''([RC]{1,2}[12]\d{2}\.?\d{0,3})\s*(\d+)\s*(\d+\.\d+)''', re.DOTALL)

# download function
get = lambda url: urllib2.urlopen(url).read()

def save(match):
    name, vehicles, distance = match.groups()
    name = name.replace('_10', '10')
    fname = 'vrptw/bestknown/%s.txt' % name
    old_data = open(fname).read()
    new_data = '%s %s\n' % (vehicles, distance)
    summary = '%-7s %3s %s' % (name, vehicles, distance)
    if old_data <> new_data:
        open(fname, 'w').write(new_data)
        print summary, 'CHANGED, from:', old_data
    else:
        print summary
    return summary

def sanitize(dta):
    """Prepare some bad HTML for easier regexp scanning."""
    dta = re.compile("<!--.*?-->", re.DOTALL).sub('', dta)
    dta = re.compile("<style.*?</style>", re.DOTALL).sub('', dta)
    dta = re.compile("<.*?>", re.DOTALL).sub('', dta)
    dta = re.sub('[^\d\n \.RC]+', ' ', dta)
    return dta

def get_hombergers_sintef():
    """Download best result for Hombergers tests from SINTEF site."""
    summary = []
    for u in homberger_urls:
        print "Visiting", u
        start = time.time()
        data = get(u)
        s, kb = time.time()-start, len(data)/1024.0
        found =  homb.finditer(data)
        for m in found:
            summary.append(save(m))
        print "Downloaded %.1f KB in %.1f s (%.1f KB/s)" % (kb, s, kb/s)
    open('vrptw/bestknown/summary_H.txt', 'w').write("\n".join(sorted(summary,
         key=lambda x: x.replace('_', '0'))))

def get_solomons():
    """Download best results for Solomons tests as published by himself."""
    import os.path
    # local debug proxy for downloading - uncomment below
    # get = lambda url: open(os.path.basename(url)).read()
    summary = []
    for u in solomon_urls:
        print "Visiting", u
        start = time.time()
        data = sanitize(get(u))
        found =  solo.finditer(data)
        for m in found:
            summary.append(save(m))
        s, kb = time.time()-start, len(data)/1024.0
        print "Downloaded %.1f KB in %.1f s (%.1f KB/s)" % (kb, s, kb/s)
    open('vrptw/bestknown/summary_S.txt', 'w').write("\n".join(sorted(summary,
         key=lambda x: x.replace('_', '0'))))

if __name__ == '__main__':
    get_hombergers_sintef()
    # get_solomons()

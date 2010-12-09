#!/usr/bin/env python

homberger_urls = [
    'http://www.sintef.no/Projectweb/TOP/Problems/VRPTW/Homberger-benchmark/%d00-customers/' % n
    for n in xrange(2,11,2)
]

solomon_urls = [
'http://web.cba.neu.edu/~msolomon/r1r2solu.htm',
'http://web.cba.neu.edu/~msolomon/c1c2solu.htm',
'http://web.cba.neu.edu/~msolomon/rc12solu.htm',
'http://web.cba.neu.edu/~msolomon/heuristi.htm'
]

import re
import urllib2

get = lambda url: urllib2.urlopen(url).read()

homb = re.compile(r'''<td style.*?([rc]{1,2}[12][0-9_]{4,6}).*?<td.*?(\d+).*?<td.*?([\d\.]+)''', re.DOTALL)

import time

summary = []
for u in homberger_urls:
    print "Visiting", u
    start = time.time()
    data = get(u)
    s, kb = time.time()-start, len(data)/1024.0
    found =  homb.findall(data)
    for name, vehicles, distance in found:
        name = name.replace('_10', '10')
        open('bestknown/hombergers/%s.txt' % name, 'w').write(
            '%s %s' % (vehicles, distance))
        summary.append('%-7s %3s %s' % (name, vehicles, distance))
        print summary[-1]
    print "Downloaded %.1f KB in %.1f s (%.1f KB/s)" % (kb, s, kb/s)
open('bestknown/summary_H.txt', 'w').write("\n".join(summary))


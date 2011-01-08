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
][:1] # others don't work yet...

import re
import urllib2
import time

homb = re.compile(r'''<td style.*?([rc]{1,2}[12][0-9_]{4,6}).*?<td.*?(\d+).*?<td.*?([\d\.]+)''', re.DOTALL)
solo = re.compile('''([RC]{1,2}[12]\d{2}\.\d{2,3}).*?(\d+).*?<.*?(\d+\.\d+)''', re.DOTALL)

# download function
get = lambda url: urllib2.urlopen(url).read()

def get_hombergers_sintef():
    """Download best result for Hombergers tests from SINTEF site."""
    summary = []
    for u in homberger_urls:
        print "Visiting", u
        start = time.time()
        data = get(u)
        s, kb = time.time()-start, len(data)/1024.0
        found =  homb.findall(data)
        for name, vehicles, distance in found:
            name = name.replace('_10', '10')
            open('bestknown/%s.txt' % name, 'w').write(
                '%s %s' % (vehicles, distance))
            summary.append('%-7s %3s %s' % (name, vehicles, distance))
            print summary[-1]
        print "Downloaded %.1f KB in %.1f s (%.1f KB/s)" % (kb, s, kb/s)
    open('bestknown/summary_H.txt', 'w').write("\n".join(summary))

def get_solomons():
    """Download best results for Solomons tests as published by himself."""
    import os.path
    clean_styles = re.compile(" ?style='[^']*'")
    # local debug proxy for downloading - uncomment below
    # get = lambda url: open(os.path.basename(url)).read()
    summary = []
    
    for u in solomon_urls:
        print "Visiting", u
        start = time.time()
        data = clean_styles.sub('', get(u))
        for noise in ['&nbsp;', ' class=MsoNormal', ' align=right']:
            data = data.replace(noise, '')
        s, kb = time.time()-start, len(data)/1024.0
        found =  solo.findall(data)
        for name, vehicles, distance in found:
            name = name.replace('.100', '')
            open('bestknown/%s.txt' % name, 'w').write(
                '%s %s' % (vehicles, distance))
            summary.append('%-7s %3s %s' % (name, vehicles, distance))
            print summary[-1]
        print "Downloaded %.1f KB in %.1f s (%.1f KB/s)" % (kb, s, kb/s)
    open('bestknown/summary_S.txt', 'w').write("\n".join(summary))

if __name__ == '__main__':
    get_hombergers_sintef()
    get_solomons()

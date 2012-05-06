#!/usr/bin/env python

try:
    import argparse
except ImportError:
    import zipfile
    import urllib2
    import os

    # A non-seekable stream doesn't suffice, downloading
    open('argparse.zip', 'wb').write(urllib2.urlopen(
        'http://argparse.googlecode.com/files/argparse-1.1.zip'
    ).read())

    # Extract the module file
    f = zipfile.ZipFile('argparse.zip')
    open('argparse.py', 'w').write(f.read('argparse-1.1/argparse.py'))
    f.close()

    # Compile
    import argparse

    # Cleanup
    os.remove('argparse.py')
    os.remove('argparse.zip')
else:
    print "Error: argparse is already available"

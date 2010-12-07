import zipfile
import urllib2
import os

if os.path.exists('argparse.pyc'):
    exit()

open('argparse.zip', 'wb').write(
    urllib2.urlopen(
        'http://argparse.googlecode.com/files/argparse-1.1.zip').read())


f = zipfile.ZipFile('argparse.zip')

open('argparse.py', 'w').write(f.open('argparse-1.1/argparse.py', 'r').read())

f.close()

# compile 
import argparse

os.remove('argparse.py')
os.remove('argparse.zip')

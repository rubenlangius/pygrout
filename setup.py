from distutils.core import setup
from glob import glob

setup(name='Pygrout',
      version='0.1',
      description='VRPTW solving utility',
      author='Tomasz Gandor',
      url='http://code.google.com/p/pygrout/',
      packages=['vrptw', 'solomons', 'hombergers'],
      package_data = { 'vrptw': ['bestknown/*.txt'],
                       'hombergers': ['*.txt'],
                       'solomons': ['*.txt'] },
      py_modules=['pygrout', 'compat', 'undo']
     )

from distutils.core import setup
from glob import glob

setup(name='Pygrout',
      version='0.1',
      description='VRPTW solving utility',
      author='Tomasz Gandor',
      url='http://code.google.com/p/pygrout/',
      py_modules=['pygrout', 'consts', 'compat', 'undo']
     )

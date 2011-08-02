#!/usr/bin/env python

import time

class StopWatch(object):
    """A small object to simulate a typical stopwatch."""
    def __init__(self):
        self.restart()
    def __str__(self):
        return str(time.time()-self.start)
    def restart(self):
        self.start = time.time()

# Functions used in batch processing
import os

import psutil

## Process managing
def killtree(pid, including_parent=False, verbose=False):
    """Function to kill all children of a given process"""
    parent = psutil.Process(pid)
    for child in parent.children(recursive=True):
        if verbose:
            print ("child", child)
        child.kill()
    if including_parent:
        parent.kill()

def killchild(verbose=False):
    """Easy function to kill children of current process"""
    _pid = os.getpid()
    killtree(_pid, False, verbose)


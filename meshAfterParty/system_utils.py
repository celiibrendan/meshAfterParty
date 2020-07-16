import sys, os

# # ************ warning this will disable all printing until turned off *************
# # Disable
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')

# # Restore
# def enablePrint():
#     sys.stdout = sys.__stdout__
# # ************ warning this will disable all printing until turned off *************

    
#better way of turning off printing: 
import os, sys

class HiddenPrints:
    """
    Example of how to use: 
    with HiddenPrints():
        print("This will not be printed")

    print("This will be printed as before")
    
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
    
import warnings

def ignore_warnings():
    """
    This will ignore warnings but not the meshlab warnings
    
    """
    warnings.filterwarnings('ignore')
    

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull

@contextmanager
def suppress_stdout_stderr():
    """
    Purpose: Will suppress all print outs
    and pinky warning messages:
    --> will not suppress the output of all the widgets like tqdm
    
    Ex: How to suppress warning messages in Poisson
    import soma_extraction_utils as sm
with su.suppress_stdout_stderr():
    sm.soma_volume_ratio(my_neuron.concept_network.nodes["S0"]["data"].mesh)
    
    
    A context manager that redirects stdout and stderr to devnull
    Example of how to use: 
    import sys

    def rogue_function():
        print('spam to stdout')
        print('important warning', file=sys.stderr)
        1 + 'a'
        return 42

    with suppress_stdout_stderr():
        rogue_function()
    
    
    """
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
            


"""
#for creating a conditional with statement around some code (to suppress outputs)


Example: (used in neuron init)
if minimal_output:
            print("Processing Neuorn in minimal output mode...please wait")

with su.suppress_stdout_stderr() if minimal_output else su.dummy_context_mgr():
    #do the block of node

"""
class dummy_context_mgr():
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_value, traceback):
        return False

import contextlib

@contextlib.contextmanager
def dummy_context_mgr():

    yield None
    
    

# How to save and load objects
"""
*** Warning *****
The way you import a class can affect whether it was picklable or not

Example: 

---Way that works---: 

su = reload(su)

from neuron import Neuron
another_neuron = Neuron(new_neuron)
su.save_object(another_neuron,"inhibitory_saved_neuron")

---Way that doesn't work---
su = reload(su)

import neuron
another_neuron = neuron.Neuron(new_neuron)
su.save_object(another_neuron,"inhibitory_saved_neuron")

"""
from pathlib import Path
import pickle
def save_object(obj, filename):
    if type(filename) == type(Path()):
        filename = str(filename.absolute())
    if filename[-4:] != ".pkl":
        filename += ".pkl"
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    if filename[-4:] != ".pkl":
        filename += ".pkl"
    with open(filename, 'rb') as input:
        retrieved_obj = pickle.load(input)
    return retrieved_obj




#--------------- Less memory pickling options -----------------
# Pickle a file and then compress it into a file with extension 
def compressed_pickle(title, data):
    """
    compressed_pickle('example_cp', data) 
    """
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)


# Load any compressed pickle file
def decompress_pickle(file):
    """
    Example: 
    data = decompress_pickle('example_cp.pbz2') 
    """
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


import os
def get_file_size(filepath):
    return os.path.getsize(filepath)
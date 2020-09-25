import sys, os

# # ************ warning this will disable all printing until turned off *************
# # Disable
# def blockPrint():
#     sys.stdout = open(os.devnull, 'w')

# # Restore
# def enablePrint():
#     sys.stdout = sys.__stdout__
# # ************ warning this will disable all printing until turned off *************

    
"""
How to get a reference to the current module

import sys
current_module = sys.modules[__name__]
"""
    
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
import logging, sys
def ignore_warnings():
    """
    This will ignore warnings but not the meshlab warnings
    
    """
    warnings.filterwarnings('ignore')
    logging.disable(sys.maxsize)
    

from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
import tqdm_utils as tqu
from tqdm_utils import tqdm
import copy

@contextmanager
def suppress_stdout_stderr(suppress_tqdm=True):
    """
    Purpose: Will suppress all print outs
    and pinky warning messages:
    --> will now suppress the output of all the widgets like tqdm outputs
    if suppress_tqdm = True
    
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
    #will supress the warnings:
    ignore_warnings()
    
    
    #get the original setting of the tqdm.disable
    if suppress_tqdm:
        original_tqdm = copy.copy(tqdm.disable)
        tqu.turn_off_tqdm()
    
    
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
            
    if not original_tqdm:
        tqu.turn_on_tqdm()
            
    
            


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
def save_object(obj, filename,return_size=False):
    """
    Purpose: to save a pickled object of a neuron
    
    ** Warning ** do not reload the module of the 
    object you are compressing before compression
    or else it will not work***
    
    """
    if type(filename) == type(Path()):
        filename = str(filename.absolute())
    if filename[-4:] != ".pkl":
        filename += ".pkl"
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    print(f"Saved object at {Path(filename).absolute()}")
    
    file_size = get_file_size(filename)/1000000
    print(f"File size is {file_size} MB")
    
    if return_size:
        return file_size
    

def load_object(filename):
    if filename[-4:] != ".pkl":
        filename += ".pkl"
    with open(filename, 'rb') as input:
        retrieved_obj = pickle.load(input)
    return retrieved_obj




#--------------- Less memory pickling options -----------------
# Pickle a file and then compress it into a file with extension 
import bz2
import _pickle as cPickle
def compressed_pickle(obj,filename,return_size=False):
    """
    compressed_pickle(data,'example_cp') 
    """
    if type(filename) == type(Path()):
        filename = str(filename.absolute())
    if filename[-5:] != ".pbz2":
        filename += ".pbz2"
 
    with bz2.BZ2File(filename, 'w') as f: 
        cPickle.dump(obj, f)
        
    print(f"Saved object at {Path(filename).absolute()}")
    file_size = get_file_size(filename)/1000000
    print(f"File size is {file_size} MB")
    
    if return_size:
        return file_size


# Load any compressed pickle file
def decompress_pickle(filename):
    """
    Example: 
    data = decompress_pickle('example_cp.pbz2') 
    """
    if type(filename) == type(Path()):
        filename = str(filename.absolute())
    if filename[-5:] != ".pbz2":
        filename += ".pbz2"
        
    data = bz2.BZ2File(filename, 'rb')
    data = cPickle.load(data)
    return data


import os
def get_file_size(filepath):
    return os.path.getsize(filepath)
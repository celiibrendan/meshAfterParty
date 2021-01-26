#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Purpose: To decompose the multi-somas for splitting
using the new decomposition method



"""


# In[2]:


import numpy as np
import datajoint as dj
import trimesh
from tqdm.notebook import tqdm
from pathlib import Path

from os import sys
sys.path.append("/meshAfterParty/")

import datajoint_utils as du
from importlib import reload


# In[3]:


#so that it will have the adapter defined
from datajoint_utils import *


# In[4]:


test_mode = False


# # Debugging the contains method

# In[5]:


import system_utils as su


# In[6]:


import minfig
import time
import numpy as np
#want to add in a wait for the connection part
random_sleep_sec = np.random.randint(0, 200)
print(f"Sleeping {random_sleep_sec} sec before conneting")
if not test_mode:
    time.sleep(random_sleep_sec)
print("Done sleeping")

du.config_celii()
du.set_minnie65_config_segmentation(minfig)
du.print_minnie65_config_paths(minfig)

#configuring will include the adapters
minnie,schema = du.configure_minnie_vm()


# # Defining the Table

# In[7]:


import neuron_utils as nru
import neuron
import trimesh_utils as tu
import numpy as np


# In[8]:


import meshlab
meshlab.set_meshlab_port(current_port=None)


# In[57]:


import numpy as np
import time

verbose = True
spine_version = 0

up_to_date_spine_process = 3
@schema
class SpineRecalculation(dj.Computed):
    definition="""
    -> minnie.Decomposition()
    split_index          : tinyint unsigned             # the index of the neuron object that resulted AFTER THE SPLITTING ALGORITHM
    ---    
    spine_version          : tinyint unsigned             # the version of the spine algorithm
    updated_spines          : bool          # whether or not the spines were updated (1 = yes, 0 = no)
    n_spines_old: int unsigned                 # number of spines before recalculation
    n_spines_new: int unsigned                 # number of spines after recalculation
    spine_data=NULL : longblob     #stores the newly computes spines that were used for the classification
    run_time=NULL : double                   # the amount of time to run (seconds)
    """
                             
    key_source = minnie.Decomposition() & minnie.NucleiSegmentsRun2()

    def make(self,key):
        """
        Pseudocode: 
        0) Download the possible neurons from either Decomposition or DecompositionSplit (using datajoint function)
        
        For Each Neuron
        1) Get the number of spines currently
        3) Run the calculate spines function
        4) get the new spines as a data structure
        5) Calculate the new number of spines
        6) Save in dictionary to write
        
        7) Write all keys
        """
        
        whole_pass_time = time.time()
        
        segment_id = key["segment_id"]
        
        
        #0) Download the possible neurons from either Decomposition or DecompositionSplit
        

        neuron_objs,split_indexes,table_name,process_version = du.decomposition_by_segment_id(segment_id,
                                                                                              return_split_indexes=True,
                                                                                              return_process_version=True,
                                                                                              return_table_name=True,
                                                                              verbose=verbose)
            
            
            
            
        new_keys = []
        for neuron_obj,split_index in zip(neuron_objs,split_indexes):     
        
            print(f"\n\n\n---- Working on Neuron {neuron_obj.segment_id}_{neuron_obj.description} ----")
            
            #1) Get the number of spines currently
            n_spines_old = neuron_obj.n_spines
            
            #2) Run the calculate spines function
            if process_version < up_to_date_spine_process:
                neuron_obj.calculate_spines()
                updated_spines = True
            else:
                if verbose:
                    print(f"Skipping re-calculation because process version {process_version} is equal or above the required version {up_to_date_spine_process}")
                updated_spines = False
            
            #3) Run the calculate spines function
            n_spines_new = neuron_obj.n_spines
            
            #4) get the new spines as a data structure
            spine_data = neuron_obj.get_computed_attribute_data(attributes=["spines","spines_volume"])
        
            
            
            
            #7) Pass stats and file location to insert
            new_key = dict(key,
                           split_index = split_index,
                           spine_version=spine_version,
                           updated_spines=updated_spines,
                           n_spines_old = n_spines_old,
                           n_spines_new = n_spines_new,
                           spine_data = spine_data,
                           run_time=np.round(time.time() - whole_pass_time,4)
                          )



            new_keys.append(new_key)

        
        self.insert(new_keys, allow_direct_insert=True, skip_duplicates=True)

        print(f"\n\n ------ Total time for {key['segment_id']} = {time.time() - whole_pass_time} ------")


# # Running the Populate

# In[58]:


curr_table = (minnie.schema.jobs & "table_name='__spine_recalculation'")
(curr_table).delete()# & "status='error'")
#curr_table.delete()
#(curr_table & "error_message = 'ValueError: need at least one array to concatenate'").delete()


# In[59]:


import time
import random
import neuron
neuron = reload(neuron)

start_time = time.time()
if not test_mode:
    time.sleep(random.randint(0, 800))
print('Populate Started')
if not test_mode:
    SpineRecalculation.populate(reserve_jobs=True, suppress_errors=True)
else:
    SpineRecalculation.populate(reserve_jobs=True, suppress_errors=False)
print('Populate Done')

print(f"Total time for SpineRecalculation populate = {time.time() - start_time}")


# In[ ]:





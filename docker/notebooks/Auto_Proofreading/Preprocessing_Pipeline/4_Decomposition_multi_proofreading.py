#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Purpose: To decompose the multi-somas for splitting
using the new decomposition method



"""


# In[1]:


import numpy as np
import datajoint as dj
import trimesh
from tqdm.notebook import tqdm
from pathlib import Path

from os import sys
sys.path.append("/meshAfterParty/")

import datajoint_utils as du
from importlib import reload


# # Debugging the seg fault

# In[2]:


#so that it will have the adapter defined
from datajoint_utils import *


# In[3]:


test_mode = False


# In[4]:


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

# In[5]:


import neuron_utils as nru
import neuron
import trimesh_utils as tu
import numpy as np


# In[6]:


import meshlab
meshlab.set_meshlab_port(current_port=None)


# In[10]:


decimation_version = 0
decimation_ratio = 0.25
# key_source = (minnie.Decimation().proj(decimation_version='version')  & 
#                   dict(decimation_version=decimation_version,decimation_ratio=decimation_ratio)  
#                   & minnie.MultiSomaProofread2() & (dj.U("segment_id") & (minnie.BaylorSegmentCentroid()).proj()))

key_source = ((minnie.Decimation).proj(decimation_version='version') & 
                            "decimation_version=" + str(decimation_version) &
                       f"decimation_ratio={decimation_ratio}" &  (minnie.BaylorSegmentCentroid() & "multiplicity>0") & minnie.MultiSomaProofread2())
key_source


# In[ ]:


#schema.external['decomposition'].delete(delete_external_files=True)


# In[12]:


import numpy as np
import time
decimation_version = 0
decimation_ratio = 0.25
process_version = 1

@schema
class Decomposition(dj.Computed):
    definition="""
    -> minnie.Decimation.proj(decimation_version='version')
    ver : decimal(6,2) #the version number of the materializaiton
    process_version : int unsigned #the version of the preprocessing pipeline run
    index : tinyint unsigned  #the index of the neuron object that resulted from that mesh (indexed starting at 0)
    ---
    multiplicity=null    : tinyint unsigned             # the number of somas found for this base segment
    decomposition: <decomposition>
    n_vertices           : int unsigned                 # number of vertices
    n_faces              : int unsigned                 # number of faces
    n_not_processed_soma_containing_meshes : int unsigned  #the number of meshes with somas that were not processed
    n_error_limbs: int #the number of limbs that are touching multiple somas or 1 soma in multiple places
    n_same_soma_multi_touching_limbs: int # number of limbs that touch the same soma multiple times
    n_multi_soma_touching_limbs: int # number of limbs that touch multiple somas
    n_somas: int #number of soma meshes detected
    n_limbs: int
    n_branches: int
    max_limb_n_branches=NULL:int
    
    skeletal_length=NULL: double
    max_limb_skeletal_length=NULL:double
    median_branch_length=NULL:double #gives information on average skeletal length to next branch point
    
    
    width_median=NULL: double #median width from mesh center without spines removed
    width_no_spine_median=NULL: double #median width from mesh center with spines removed
    width_90_perc=NULL: double # 90th percentile for width without spines removed
    width_no_spine_90_perc=NULL: double  # 90th percentile for width with spines removed
    
    
    n_spines: bigint

    spine_density=NULL: double # n_spines/ skeletal_length
    spines_per_branch=NULL: double
    
    skeletal_length_eligible=NULL: double # the skeletal length for all branches searched for spines
    n_spine_eligible_branches=NULL: int # the number of branches that were checked for spines because passed width threshold
    
    spine_density_eligible=NULL:double # n_spines/skeletal_length_eligible
    spines_per_branch_eligible=NULL:double # n_spines/n_spine_eligible_branches
    
    total_spine_volume=NULL: double # the sum of all spine volume
    spine_volume_median=NULL: double # median of the spine volume for those spines with able to calculate volume
    spine_volume_density=NULL: double #total_spine_volume/skeletal_length
    spine_volume_density_eligible=NULL: double #total_spine_volume/skeletal_length_eligible
    spine_volume_per_branch_eligible=NULL: double #total_spine_volume/n_spine_eligible_branches
    
    run_time=NULL : double                   # the amount of time to run (seconds)

    
    """


    
    key_source = ((minnie.Decimation).proj(decimation_version='version') & 
                            "decimation_version=" + str(decimation_version) &
                       f"decimation_ratio={decimation_ratio}" &  (minnie.BaylorSegmentCentroid() & "multiplicity>0") & minnie.MultiSomaProofread2())
                                                                  
    

    def make(self,key):
        """
        Pseudocode for process:

        1) Get the segment id from the key
        2) Get the decimated mesh
        3) Get the somas info
        4) Run the preprocessing
        5) Calculate all starter stats
        6) Save the file in a certain location
        7) Pass stats and file location to insert
        """
        whole_pass_time = time.time()
        #1) Get the segment id from the key
        segment_id = key["segment_id"]
        description = str(key['decimation_version']) + "_25"
        print(f"\n\n\n---- Working on Neuron {key['segment_id']} ----")
        global_start = time.time()
        
        #2) Get the decimated mesh
        current_neuron_mesh = du.fetch_segment_id_mesh(segment_id)
        

        #3) Get the somas info 
        somas = du.get_soma_mesh_list(segment_id) 
        soma_ver = du.get_soma_mesh_list_ver(segment_id)
                  
        
        print(f"somas = {somas}")
                  
        #3b) Get the glia and nuclei information 
        glia_faces,nuclei_faces = du.get_segment_glia_nuclei_faces(segment_id,return_empty_list=True)
        
                  
        #4) Run the preprocessing


        total_neuron_process_time = time.time()

        print(f"\n--- Beginning preprocessing of {segment_id}---")
        recovered_neuron = neuron.Neuron(
        mesh = current_neuron_mesh,
        somas = somas,
        segment_id=segment_id,
        description=description,
        suppress_preprocessing_print=False,
        suppress_output=False,
        calculate_spines=True,
        widths_to_calculate=["no_spine_median_mesh_center"],
        glia_faces=glia_faces,
        nuclei_faces = nuclei_faces,

                )

        print(f"\n\n\n---- Total preprocessing time = {time.time() - total_neuron_process_time}")


        #5) Don't have to do any of the processing anymore because will do in the neuron object
        stats_dict = recovered_neuron.neuron_stats()



        #6) Save the file in a certain location
        save_time = time.time()
        ret_file_path = recovered_neuron.save_compressed_neuron(output_folder=str(du.get_decomposition_path()),
                                          return_file_path=True,
                                         export_mesh=False,
                                         suppress_output=True)

        ret_file_path_str = str(ret_file_path.absolute()) + ".pbz2"
        print(f"Save time = {time.time() - save_time}")

        #7) Pass stats and file location to insert
        new_key = dict(key,
                       ver = soma_ver,
                       process_version = process_version,
                       index = 0,
                       multiplicity=1,
                       decomposition=ret_file_path_str,
                       n_vertices=len(current_neuron_mesh.vertices),
                       n_faces=len(current_neuron_mesh.faces),
                       run_time=np.round(time.time() - whole_pass_time,4)
                      )
        new_key.update(stats_dict)

        self.insert1(new_key, allow_direct_insert=True, skip_duplicates=True)

        print(f"\n\n ------ Total time for {segment_id} = {time.time() - global_start} ------")
    


# # Running the Populate

# In[13]:


curr_table = (minnie.schema.jobs & "table_name='__decomposition'")
(curr_table)#.delete()# & "status='error'")
#curr_table.delete()
#(curr_table & "error_message = 'ValueError: need at least one array to concatenate'").delete()


# In[ ]:


import time
import random
import compartment_utils as cu
cu = reload(cu)
import preprocessing_vp2 as pre
pre = reload(pre)
import skeleton_utils as sk
sk = reload(sk)
nru = reload(nru)

start_time = time.time()
if not test_mode:
    time.sleep(random.randint(0, 800))
print('Populate Started')
if not test_mode:
    Decomposition.populate(reserve_jobs=True, suppress_errors=True)
else:
    Decomposition.populate(reserve_jobs=True, suppress_errors=False)
print('Populate Done')

print(f"Total time for DecompositionMultiSoma populate = {time.time() - start_time}")


# In[ ]:





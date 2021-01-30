"""
Purpose: Module provides tools for helping to find the interesting branches
and limbs according to the query and functions that you define

"""

import neuron_utils as nru
import skeleton_utils as sk
import numpy as np
import pandas as pd
import numpy_utils as nu
import pandas_utils as pu
import networkx_utils as xu

def convert_neuron_to_branches_dataframe(current_neuron,
                                        limbs_to_process=None,
                                        limb_branch_dict_restriction=None):
    """
    Purpose: 
    How to turn a concept map into a pandas table with only the limb_idx and node_idx
    
    Example: 
    neuron_df = convert_neuron_to_branches_dataframe(current_neuron = recovered_neuron)
    
    """
    curr_concept_network = nru.return_concept_network(current_neuron)
    
    if limbs_to_process is None:
        limb_idxs = nru.get_limb_names_from_concept_network(curr_concept_network)
    else:
        limb_idxs = [nru.get_limb_string_name(k) for k in limbs_to_process]
    
    
    limb_node_idx_dicts = []

    for l in limb_idxs:
        if limb_branch_dict_restriction is None:
            limb_node_idx_dicts += [dict(limb=l,node=int(k)) for k in 
                                curr_concept_network.nodes[l]["data"].concept_network.nodes()]
        else:
            if l in limb_branch_dict_restriction.keys():
                limb_node_idx_dicts += [dict(limb=l,node=int(k)) for k in 
                                    limb_branch_dict_restriction[l]]

    df = pd.DataFrame(limb_node_idx_dicts)
    return df

#wrapper to help with classifying funciton uses
class run_options:
    def __init__(self,run_type="Limb"):
        self.run_type = run_type

    def __call__(self, f):
        f.run_type = self.run_type
        return f

#------------------------------- Branch Functions ------------------------------------#

"""
Want to have functions that just operate off of branch characteristics or limb characteristics
"""

#Branch Functions
import skeleton_utils as sk

@run_options(run_type="Branch")
def n_faces_branch(curr_branch,name=None,branch_name=None,**kwargs):
    return len(curr_branch.mesh.faces)

@run_options(run_type="Branch")
def width(curr_branch,name=None,branch_name=None,**kwargs):
    return curr_branch.width

@run_options(run_type="Branch")
def skeleton_distance_branch(curr_branch,name=None,branch_name=None,**kwargs):
    try:
        #print(f"curr_branch.skeleton = {curr_branch.skeleton.shape}")
        return sk.calculate_skeleton_distance(curr_branch.skeleton)
    except:
        print(f"curr_branch.skeleton = {curr_branch.skeleton}")
        raise Exception("")
        
@run_options(run_type="Branch")
def skeletal_length(curr_branch,name=None,branch_name=None,**kwargs):
    try:
        #print(f"curr_branch.skeleton = {curr_branch.skeleton.shape}")
        return sk.calculate_skeleton_distance(curr_branch.skeleton)
    except:
        print(f"curr_branch.skeleton = {curr_branch.skeleton}")
        raise Exception("")
        

@run_options(run_type="Branch")
def n_spines(branch,limb_name=None,branch_name=None,**kwargs):
    if not branch.spines is None:
        return len(branch.spines)
    else:
        return 0

@run_options(run_type="Branch")
def width_new(branch,limb_name=None,branch_name=None,width_new_name="no_spine_mean_mesh_center",**kwargs):
    return branch.width_new[width_new_name]




# ---------- new possible widths ----------------- #
@run_options(run_type="Branch")
def mean_mesh_center(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.width_new["mean_mesh_center"]

@run_options(run_type="Branch")
def median_mesh_center(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.width_new["median_mesh_center"]

@run_options(run_type="Branch")
def no_spine_mean_mesh_center(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.width_new["no_spine_mean_mesh_center"]

@run_options(run_type="Branch")
def no_spine_median_mesh_center(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.width_new["no_spine_median_mesh_center"]

# ---------- new possible widths ----------------- #




@run_options(run_type="Branch")
def no_spine_width(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.width_new["no_spine_average"]

@run_options(run_type="Branch")
def no_spine_average_mesh_center(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.width_new["no_spine_average_mesh_center"]

@run_options(run_type="Branch")
def spines_per_skeletal_length(branch,limb_name=None,branch_name=None,**kwargs):
    curr_n_spines = n_spines(branch)
    curr_skeleton_distance = sk.calculate_skeleton_distance(branch.skeleton)
    return curr_n_spines/curr_skeleton_distance

# --------- spine densities ---------------- #
@run_options(run_type="Branch")
def spine_density(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.spine_density


# ----------- working with different labels -----------------#
@run_options(run_type="Branch")
def matching_label(branch,limb_name=None,branch_name=None,**kwargs):
    poss_labels = kwargs["labels"]
    match_flag = False
    for l in poss_labels:
        if l in branch.labels:
            match_flag = True
            break
    return match_flag

@run_options(run_type="Branch")
def labels_restriction(branch,limb_name=None,branch_name=None,**kwargs):
    poss_labels = kwargs.get("matching_labels",[])
    not_possible_labels = kwargs.get("not_matching_labels",[])
    
#     print(f"poss_labels = {poss_labels}")
#     print(f"not_possible_labels = {not_possible_labels}")
#     print(f"np.intersect1d(branch.labels,poss_labels) = {np.intersect1d(branch.labels,poss_labels)}")
   
    if len(poss_labels) == len(np.intersect1d(branch.labels,poss_labels)):
        if 0 == len(np.intersect1d(branch.labels,not_possible_labels)):
            return True
    return False

@run_options(run_type="Branch")
def labels(branch,limb_name=None,branch_name=None,**kwargs):
    return branch.labels


def query_neuron_by_labels(neuron_obj,
    matching_labels=[],
    not_matching_labels=[]):
    
    return ns.query_neuron(neuron_obj,
               query="labels_restriction == True",
               functions_list=["labels_restriction"],
               function_kwargs=dict(matching_labels=matching_labels,
                                    not_matching_labels=not_matching_labels
                                    
                                   ))


#------------------------------- Limb Functions ------------------------------------#
"""
Rule: For the limb functions will either return
1) 1 number
2) 1 True/False Value
3) Array of 1 or 2 that matches the number of branches
4) a list of nodes that it applies to

"""
def convert_limb_function_return_to_dict(function_return,
                                        curr_limb_concept_network):
    """
    purpose: to take the returned
    value of a limb function and convert it to 
    a dictionary that maps all of the nodes to a certain value
    - capable of handling both a dictionary and a scalar value
    
    """
    
    #curr_limb_concept_network = curr_neuron_concept_network.nodes[limb_name]["data"].concept_network
    function_mapping_dict = dict()
    if np.isscalar(function_return):
        for branch_idx in curr_limb_concept_network.nodes():
            function_mapping_dict[branch_idx] = function_return
    elif set(list(function_return.keys())) == set(list(curr_limb_concept_network.nodes())):
        function_mapping_dict = function_return
    else:
        raise Exception("The value returned from limb function was not a scalar nor did it match the keys of the limb branches")  
    
    return function_mapping_dict

def convert_limb_function_return_to_limb_branch_dict(function_return,
                                        curr_limb_concept_network,
                                                    limb_name):
    """
    Purpose: returns a dictionary that maps limb to valid branches
    according to a function return that is True or False
    (only includes the branches that are true from the function_return)
    
    Result: retursn a dictionary like dict(L1=[3,5,8,9,10])
    """
    new_dict = convert_limb_function_return_to_dict(function_return,
                                        curr_limb_concept_network)
    return {limb_name:[k for k,v in new_dict.items() if v == True]}
    


@run_options(run_type="Limb")
def skeleton_distance_limb(curr_limb,limb_name=None,**kwargs):
    curr_skeleton = curr_limb.get_skeleton()
    return sk.calculate_skeleton_distance(curr_skeleton)

@run_options(run_type="Limb")
def n_faces_limb(curr_limb,limb_name=None,**kwargs):
    return len(curr_limb.mesh.faces)

@run_options(run_type="Limb")
def merge_limbs(curr_limb,limb_name=None,**kwargs):
    return "MergeError" in curr_limb.labels

@run_options(run_type="Limb")
def limb_error_branches(curr_limb,limb_name=None,**kwargs):
    error_nodes = nru.classify_endpoint_error_branches_from_limb_concept_network(curr_limb.concept_network)
    node_names = np.array(list(curr_limb.concept_network.nodes()))
    return dict([(k,k in error_nodes) for k in node_names])

@run_options(run_type="Limb")
def average_branch_length(curr_limb,limb_name=None,**kwargs):
    return np.mean([sk.calculate_skeleton_distance(curr_limb.concept_network.nodes[k]["data"].skeleton) for k in curr_limb.concept_network.nodes()])

@run_options(run_type="Limb")
def test_limb(curr_limb,limb_name=None,**kwargs):
    return 5

from copy import deepcopy
import networkx as nx
import skeleton_utils as sk

@run_options(run_type="Limb")
@run_options(run_type="Limb")

def skeletal_distance_from_soma_excluding_node(curr_limb,
                    limb_name = None,
                    somas = None,
                    error_if_all_nodes_not_return=True,
                    print_flag = False,
                    **kwargs
                            
    ):
    
    return nru.skeletal_distance_from_soma(curr_limb,
                    limb_name = None,
                    somas = None,
                    error_if_all_nodes_not_return=True,
                    include_node_skeleton_dist=False,
                    print_flag = False,
                    **kwargs)

@run_options(run_type="Limb")
def skeletal_distance_from_soma(curr_limb,
                    limb_name = None,
                    somas = None,
                    error_if_all_nodes_not_return=True,
                    include_node_skeleton_dist=True,
                    print_flag = False,
                    **kwargs
                            
    ):
    
    return nru.skeletal_distance_from_soma(curr_limb,
                    limb_name = None,
                    somas = None,
                    error_if_all_nodes_not_return=True,
                    include_node_skeleton_dist=True,
                    print_flag = False,
                    **kwargs)


from copy import deepcopy
import networkx_utils as xu



axon_width_like_requirement = "(median_mesh_center < 200)"# or no_spine_median_mesh_center < 150)"
ais_axon_width_like_requirement = "(median_mesh_center < 600)" #this will get all of axons including the axon intial segment

def axon_width_like_query(width_to_use):
    axon_width_like_query = (
                    #f"(n_spines < 4 and {width_to_use} and skeleton_distance_branch <= 15000)"
                    f"(n_spines < 4 and {width_to_use} and skeleton_distance_branch <= 25000)"
                    #f" or (skeleton_distance_branch > 15000 and {width_to_use} and spines_per_skeletal_length < 0.00023)"
                    f" or (skeleton_distance_branch > 25000 and {width_to_use} and spines_per_skeletal_length < 0.00015)"
                            
                            )
    return axon_width_like_query

def axon_merge_error_width_like_query(width_to_use):
   

    axon_width_like_query = (f"((n_spines < 2) and ((n_spines < 3 and {width_to_use} and skeleton_distance_branch <= 25000)"
    #f" or (skeleton_distance_branch > 15000 and {width_to_use} and spines_per_skeletal_length < 0.00023)"
    f" or (skeleton_distance_branch > 25000 and {width_to_use} and spines_per_skeletal_length < 0.00015)))")
    return axon_width_like_query
    

axon_width_like_functions_list = [
    "width",
    "median_mesh_center",
    "n_spines",
    "n_faces_branch",
    "skeleton_distance_branch",
    "spines_per_skeletal_length",
    "no_spine_median_mesh_center",
]


def axon_width_like_segments_old(current_neuron,
                      current_query=None,
                      current_functions_list=None,
                             include_ais=False,
                             axon_merge_error=False,
                             verbose=False,
                             width_to_use=None):
    """
    Will get all of
    
    """
    
    if current_functions_list is None:
        current_functions_list = axon_width_like_functions_list
    if current_query is None:
        if not width_to_use is None:
            width_expression  = f"(median_mesh_center < {width_to_use})"
        else:
            if include_ais:
                width_expression = ais_axon_width_like_requirement
            else:
                width_expression = axon_width_like_requirement
        current_query = axon_width_like_query(width_expression)
        
    if axon_merge_error:
        print(f"width_expression = {width_expression}")
        current_query = axon_merge_error_width_like_query(width_expression)
        
    if verbose:
        print(f"current_query = {current_query}")
        
    limb_branch_dict = query_neuron(current_neuron,
                                       #query="n_spines < 4 and no_spine_average_mesh_center < 400",
                                       query=current_query,
                                       #return_dataframe=True,
                   functions_list=current_functions_list)
    return limb_branch_dict

# ------------- 1//22: Addition that accounts for newer higher fidelity spining --------- #
def axon_width_like_query_revised(width_to_use,spine_limit):
   
    axon_width_like_query = f"((n_spines < 7) and ({width_to_use}) and spine_density <= 0.00008)"
    return axon_width_like_query


def axon_width_like_segments(current_neuron,
                      current_query=None,
                      current_functions_list=None,
                             include_ais=False,
                             verbose=False,
                             width_to_use=None):
    """
    Purpose: Will get all of the branches that look like spines
    based on width and spine properties
    
    """
    
    functions_list = ["median_mesh_center",
                     "n_spines",
                     "spine_density"]
    
    if current_functions_list is None:
        current_functions_list = functions_list
        
    if current_query is None:
        if not width_to_use is None:
            width_expression  = f"(median_mesh_center < {width_to_use})"
        else:
            if include_ais:
                spine_limit = 5
                width_expression = ais_axon_width_like_requirement
            else:
                spine_limit = 3
                width_expression = axon_width_like_requirement
                
        current_query = axon_width_like_query_revised(width_expression,spine_limit)
        
        
    if verbose:
        print(f"current_query = {current_query}")
        
    limb_branch_dict = query_neuron(current_neuron,
                                       #query="n_spines < 4 and no_spine_average_mesh_center < 400",
                                       query=current_query,
                                       #return_dataframe=True,
                   functions_list=current_functions_list)
    return limb_branch_dict

import neuron_visualizations as nviz
def axon_segments_after_checks(neuron_obj,
                               include_ais=True,
                               downstream_face_threshold=3000,
                                width_match_threshold=50,
                               plot_axon=False,
                               **kwargs):
    
    axon_like_limb_branch_dict = axon_width_like_segments(neuron_obj,
                                                         include_ais=include_ais,
                                                         **kwargs)
    
    current_functions_list = ["axon_segment"]
    final_axon_like_classification = ns.query_neuron(neuron_obj,

                                       query="axon_segment==True",
                                       function_kwargs=dict(limb_branch_dict =axon_like_limb_branch_dict,
                                                            downstream_face_threshold=downstream_face_threshold,
                                                            width_match_threshold=width_match_threshold,
                                                           print_flag=False),
                                       functions_list=current_functions_list)
    
    if plot_axon:
        nviz.visualize_neuron(neuron_obj,
                              visualize_type=["mesh"],
                             limb_branch_dict=final_axon_like_classification,
                             mesh_color="red",
                              mesh_color_alpha=1,
                             mesh_whole_neuron=True)
    return final_axon_like_classification




@run_options(run_type="Limb")
def axon_segment(curr_limb,limb_branch_dict=None,limb_name=None,
                 
                 #the parameters for the axon_segment_downstream_dendrites function
                 downstream_face_threshold=5000,
                 downstream_non_axon_percentage_threshold=0.3,
                 max_skeletal_length_can_flip=20000,
                 distance_for_downstream_check=40000,
                 print_flag=False,
                 
                 
                 #the parameters for the axon_segment_clean_false_positives function
                 width_match_threshold=50,
               width_type = "no_spine_median_mesh_center",
               must_have_spine=True,
             **kwargs):
    """
    Function that will go through and hopefully label all of the axon pieces on a limb
    """
    
    curr_limb_concept_network = curr_limb.concept_network
    #print(f"limb_branch_dict BEFORE = {limb_branch_dict}")
    downstream_filtered_limb_branch_dict = axon_segment_downstream_dendrites(curr_limb,limb_branch_dict,limb_name=limb_name,
             downstream_face_threshold=downstream_face_threshold,
             downstream_non_axon_percentage_threshold=downstream_non_axon_percentage_threshold,                
            max_skeletal_length_can_flip=max_skeletal_length_can_flip,                      
            distance_for_downstream_check=distance_for_downstream_check,
             print_flag=print_flag,
             **kwargs)
    """ Old way of doing before condensed
    #print(f"limb_branch_dict AFTER = {limb_branch_dict}")
    #unravel the output back into a dictionary mapping every node to a value
    #print(f"downstream_filtered_limb_branch_dict BEFORE= {downstream_filtered_limb_branch_dict}")
    downstream_filtered_limb_branch_dict =  convert_limb_function_return_to_dict(downstream_filtered_limb_branch_dict,
                                                curr_limb_concept_network)
    #print(f"downstream_filtered_limb_branch_dict AFTER= {downstream_filtered_limb_branch_dict}")
    #convert the dictionary mapping to a new limb_branch_dict just for that limb
    limb_branch_dict_downstream_filtered = {limb_name:[k for k,v in downstream_filtered_limb_branch_dict.items() if v == True]}
    #print(f"limb_branch_dict_downstream_filtered = {limb_branch_dict_downstream_filtered}")
    
    """
    
    limb_branch_dict_downstream_filtered = convert_limb_function_return_to_limb_branch_dict(downstream_filtered_limb_branch_dict,
                                                                                           curr_limb_concept_network,
                                                                                           limb_name)
    
    
    return axon_segment_clean_false_positives(curr_limb=curr_limb,
                                       limb_branch_dict=limb_branch_dict_downstream_filtered,
                                       limb_name=limb_name,
                                    width_match_threshold=width_match_threshold,
                                   width_type = width_type,
                                   must_have_spine=must_have_spine,
                                 print_flag=print_flag,
                                 **kwargs)

@run_options(run_type="Limb")
def axon_segment_downstream_dendrites(curr_limb,limb_branch_dict,limb_name=None,downstream_face_threshold=5000,
                                      downstream_non_axon_percentage_threshold = 0.5,
                                      max_skeletal_length_can_flip=20000,
                                      distance_for_downstream_check = 50000,
                 print_flag=False,
                 #return_limb_branch_dict=False,
                 **kwargs):
    """
    Purpose: To filter the aoxn-like segments (so that does not mistake dendritic branches)
    based on the criteria that an axon segment should not have many non-axon upstream branches
    
    Example on how to run: 
    
    curr_limb_name = "L1"
    curr_limb = uncompressed_neuron.concept_network.nodes[curr_limb_name]["data"]
    ns = reload(ns)

    return_value = ns.axon_segment(curr_limb,limb_branch_dict=limb_branch_dict,
                 limb_name=curr_limb_name,downstream_face_threshold=5000,
                     print_flag=False)
    return_value
    """
    
   
    
    
    if print_flag:
        print(f"downstream_face_threshold= {downstream_face_threshold}")
        print(f"downstream_non_axon_percentage_threshold = {downstream_non_axon_percentage_threshold}")
        print(f"max_skeletal_length_can_flip = {max_skeletal_length_can_flip}")
        print(f"distance_for_downstream_check = {distance_for_downstream_check}")
        #print(f"limb_branch_dict= {limb_branch_dict}")
        
    #curr_limb_branch_dict = kwargs["function_kwargs"]["limb_branch_dict"]
    
    curr_limb_branch_dict = limb_branch_dict
    
    if limb_name not in list(curr_limb_branch_dict.keys()):
        return False
    
    curr_axon_nodes = curr_limb_branch_dict[limb_name]
    
#     if print_flag:
#         print(f"curr_axon_nodes = {curr_axon_nodes}")
    
    curr_limb_copy = deepcopy(curr_limb) #deepcopying so don't change anything
    
    non_axon_nodes = []
    #1) Get all of the concept maps (by first getting all of the somas)
    touching_somas = [k["starting_soma"] for k in curr_limb_copy.all_concept_network_data]
    #2) For each of the concept maps: 
    for sm_start in touching_somas:
        curr_limb_copy.set_concept_network_directional(sm_start)
        curr_directional_network = curr_limb_copy.concept_network_directional
        
        #- For each node: 
        for n in curr_axon_nodes:
            
            # ----------- 1/28: Only want to go downstream for a certain extent, and only want to flip back to dendrite if small segments
            if curr_limb[n].skeletal_length > max_skeletal_length_can_flip:
                if print_flag:
                    print(f"Skipping a possible flip because the length is too long for threshold ({max_skeletal_length_can_flip}): {curr_limb[n].skeletal_length}")
                continue
            #a. Get all of the downstream nodes
            
            curr_downstream_nodes = nru.branches_within_skeletal_distance(limb_obj = curr_limb_copy,
                            start_branch = n,
                            max_distance_from_start = distance_for_downstream_check,
                            verbose = False,
                            include_start_branch_length = False,
                            include_node_branch_length = False,
                            only_consider_downstream = True)
            
            #curr_downstream_nodes = xu.downstream_edges(curr_directional_network,n)
            
            # if there are any downstream nodes
            if len(curr_downstream_nodes) > 0:
                #curr_downstream_nodes = np.concatenate(curr_downstream_nodes)
                #b. Get the total number of faces for all upstream non-axon nodes
                curr_non_axon_nodes = set([k for k in curr_downstream_nodes if k not in curr_axon_nodes])
                downstream_axon_nodes = set([k for k in curr_downstream_nodes if k in curr_axon_nodes])
                
                if len(curr_non_axon_nodes) > 0: 
                    non_axon_face_count = np.sum([len(curr_limb_copy.concept_network.nodes[k]["data"].mesh.faces) for k in curr_non_axon_nodes])
                    axon_face_count = np.sum([len(curr_limb_copy.concept_network.nodes[k]["data"].mesh.faces) for k  in downstream_axon_nodes])
                    perc_non_axon = non_axon_face_count / (non_axon_face_count + axon_face_count)
                    if print_flag:
                        print(f"Soma {sm_start}, limb {limb_name}, node {n} had {non_axon_face_count} non-axon downstream faces, {axon_face_count} axon downstream for a percentage of {perc_non_axon}")  
                    #if non_axon_face_count > downstream_face_threshold:
                    
                    # ----------- 1/20 addition: That factors in percentages and not just raw face count ------- #
                    if downstream_non_axon_percentage_threshold is not None:
                        if print_flag:
                            print(f"perc_non_axon for limb_{limb_name}_node_{n},  = {perc_non_axon}")
                        reverse_label = perc_non_axon > downstream_non_axon_percentage_threshold and non_axon_face_count > downstream_face_threshold
                    else:
                        reverse_label = non_axon_face_count > downstream_face_threshold
                    
                    if reverse_label:
                        non_axon_nodes.append(n)
                        if print_flag:
                            print(f"     Added {n} to non-axon list")
                else:
                    if print_flag:
                        print(f"Soma {sm_start}, limb {limb_name}, node {n} did not hae any NON-AXON downstream targets")
            else:
                if print_flag:
                    print(f"Soma {sm_start}, limb {limb_name}, node {n} did not hae any downstream targets")
    
    #compile all of the non-axon nodes
    total_non_axon_nodes = set(non_axon_nodes)
    
    if print_flag:
        print(f"total_non_axon_nodes = {total_non_axon_nodes}")
    #make a return dictionary that shows the filtered down axons
    return_dict = dict()
    for n in curr_limb_copy.concept_network.nodes():
        if n in curr_axon_nodes and n not in total_non_axon_nodes:
            return_dict[n] = True
        else:
            return_dict[n] = False
    
    return return_dict

import width_utils as wu
@run_options(run_type="Limb")
def axon_segment_clean_false_positives(curr_limb,
                                       limb_branch_dict,
                                       limb_name=None,
                                    width_match_threshold=50,
                                   width_type = "no_spine_average_mesh_center",
                                   must_have_spine=True,
                                       interest_nodes=[],
                                    #return_limb_branch_dict=False,
                                 print_flag=False,
                                 **kwargs):
    """
    Purpose: To help prevent the false positives
    where small end dendritic segments are mistaken for axon pieces
    by checking if the mesh transition in width is very constant between an upstream 
    node (that is a non-axonal piece) and the downstream node that is an axonal piece
    then this will change the axonal piece to a non-axonal piece label: 
    
    
    Idea: Can look for where width transitions are pretty constant with preceeding dendrite and axon
    and if very similar then keep as non-dendrite

    *** only apply to those with 1 or more spines

    Pseudocode: 
    1) given all of the axons

    For each axon node:
    For each of the directional concept networks
    1) If has an upstream node that is not an axon --> if not then continue
    1b) (optional) Has to have at least one spine or continues
    2) get the upstream nodes no_spine_average_mesh_center width array
    2b) find the endpoints of the current node
    3) Find which endpoints match from the node and the upstream node
    4) get the tangent part of the no_spine_average_mesh_center width array from the endpoints matching
    (this is either the 2nd and 3rd from front or last depending on touching AND that it is long enough)

    5) get the tangent part of the node based on touching

    6) if the average of these is greater than upstream - 50

    return an updated dictionary
    
    """
    if limb_name not in list(limb_branch_dict.keys()):
        if print_flag:
            print(f"{limb_name} not in curr_limb_branch_dict.keys so returning False")
        return False
    
    curr_axon_nodes = limb_branch_dict[limb_name]
    
    curr_limb_copy = deepcopy(curr_limb)
    
    non_axon_nodes = []
    
    #a) Get all of the concept maps (by first getting all of the somas)
    touching_somas = [k["starting_soma"] for k in curr_limb_copy.all_concept_network_data]
    
    #b) For each of the concept maps: 
    for sm_start in touching_somas:
        curr_limb_copy.set_concept_network_directional(sm_start)
        curr_directional_network = curr_limb_copy.concept_network_directional
        
        #- For each node: 
        for n in curr_axon_nodes:
            
            #if already added to the non-axons nodes then don't need to check anymore
            if n in non_axon_nodes:
                continue
            
            #1) If has an upstream node that is not an axon --> if not then continue
            curr_upstream_nodes = xu.upstream_edges_neighbors(curr_directional_network,n)
            
            if len(curr_upstream_nodes) == 0:
                continue
            if len(curr_upstream_nodes) > 1:
                raise Exception(f"More than one upstream node for node {n}: {curr_upstream_nodes}")
            
            upstream_node = curr_upstream_nodes[0][0]
            if print_flag:
                print(f"n = {n}, upstream_node= {upstream_node}")
            
            if upstream_node in curr_axon_nodes:
                if print_flag:
                    print("Skipping because the upstream node is not a non-axon piece")
                continue
                
        
            #1b) (optional) Has to have at least one spine or continues
            if must_have_spine:
                if not (curr_limb_copy.concept_network.nodes[n]["data"].spines) is None:
                    if len(curr_limb_copy.concept_network.nodes[n]["data"].spines) == 0:
                        if print_flag:
                            print(f"Not processing node {n} because there were no spines and  must_have_spine set to {must_have_spine}")
                        continue
                else:
                    if print_flag:
                        print(f"Not processing node {n} because spines were NONE and must_have_spine set to {must_have_spine}")
                    continue #if spines were None
                    
            
            #2- 5) get the tangent touching parts of the mesh
            width_array_1,width_array_2 = wu.find_mesh_width_array_border(curr_limb=curr_limb_copy,
                                 node_1 = n,
                                 node_2 = upstream_node,
                                width_name=width_type,
                                segment_start = 1,
                                segment_end = 6,
                                skeleton_segment_size = 500,
                                print_flag=False,
                                **kwargs
                                )
            
            #6) if the average of these is greater than upstream - 50 then add to the list of non axons
            #interest_nodes = [56,71]`x
            if n in interest_nodes or upstream_node in interest_nodes:
                print(f"width_array_1 = {width_array_1}")
                print(f"width_array_2 = {width_array_2}")
                print(f"np.mean(width_array_1) = {np.mean(width_array_1)}")
                print(f"np.mean(width_array_2) = {np.mean(width_array_2)}")
            
            if np.mean(width_array_1) >= (np.mean(width_array_2)-width_match_threshold):
                
                non_axon_nodes.append(n)
                if print_flag:
                    print(f"Adding node {n} to non_axon list with threshold {width_match_threshold} because \n"
                          f"   np.mean(width_array_1)  = {np.mean(width_array_1) }"
                          f"   np.mean(width_array_2)  = {np.mean(width_array_2) }")
                 
    #after checking all nodes and concept networks
    #compile all of the non-axon nodes
    total_non_axon_nodes = set(non_axon_nodes)
    
    if print_flag:
        print(f"total_non_axon_nodes = {total_non_axon_nodes}")
    #make a return dictionary that shows the filtered down axons
    return_dict = dict()
    for n in curr_limb_copy.concept_network.nodes():
        if n in curr_axon_nodes and n not in total_non_axon_nodes:
            return_dict[n] = True
        else:
            return_dict[n] = False
    
    return return_dict
            
    
# --------- 1/15: Additions to help find axon -----------------#
import neuron_statistics as nst

@run_options(run_type="Limb")
def soma_starting_angle(curr_limb,limb_name=None,**kwargs):
    """
    will compute the angle in degrees from the vector pointing
    straight to the volume and the vector pointing from 
    the middle of the soma to the starting coordinate of the limb
    
    """
    print_flag = kwargs.get("verbose",False)
    curr_soma_angle = nst.soma_starting_angle(limb_obj=curr_limb,
                        soma_center = kwargs["soma_center"],
                        )
    
    if print_flag:
        print(f"Limb {limb_name} soma angle: {curr_soma_angle} ")
        
    return curr_soma_angle
    


#------------------------------- Creating the Data tables from the neuron and functions------------------------------

def apply_function_to_neuron(current_neuron,current_function,function_kwargs=None,
                            ):
    """
    Purpose: To retrieve a dictionary mapping every branch on every node
    to a certain value as defined by the function passed
    
    Example: 
    curr_function = ns.width
    curr_function_mapping = ns.apply_function_to_neuron(recovered_neuron,curr_function)
    
    """
    if function_kwargs is None:
        function_kwargs = dict()
    
    curr_neuron_concept_network = nru.return_concept_network(current_neuron)
    
    if "limbs_to_process" in function_kwargs.keys():
        curr_limb_names = [nru.get_limb_string_name(k) for k in function_kwargs["limbs_to_process"]]
    else:
        curr_limb_names = nru.get_limb_names_from_concept_network(curr_neuron_concept_network)
        
    limb_branch_dict_restriction =  function_kwargs.get("limb_branch_dict_restriction",None)
    
    if limb_branch_dict_restriction is None:
        limb_branch_dict_restriction = nru.neuron_limb_branch_dict(current_neuron)
    else:
        print(f"Using the limb_branch_dict_restriction = {limb_branch_dict_restriction}")
        
    
    
    function_mapping = dict([(limb_name,dict()) for limb_name in curr_limb_names if limb_name in limb_branch_dict_restriction.keys()])

    #if it was a branch function that was passed
    if current_function.run_type=="Branch":
        for limb_name in function_mapping.keys():
            curr_limb_concept_network = curr_neuron_concept_network.nodes[limb_name]["data"].concept_network
            for branch_idx in curr_limb_concept_network.nodes():
                
                # -- 1/29 Addition: Will only look at nodes in the limb branch dict restriction -- #
                if branch_idx not in limb_branch_dict_restriction[limb_name]:
                    continue
                    
#                 if limb_name == "L0" and branch_idx == 73:
#                     print(f"Computing {current_function} for !!!!!")
                    
                function_mapping[limb_name][branch_idx] = current_function(curr_limb_concept_network.nodes[branch_idx]["data"],limb_name=limb_name,branch_name=branch_idx,**function_kwargs)
                
    elif current_function.run_type=="Limb":
        #if it was a limb function that was passed
        """
        - for each limb:
          i) run the function and recieve the returned result
          2) If only a single value is returned --> make dict[limb_idx][node_idx] = value all with the same value
          3) if dictionary of values: 
             a. check that keys match the node_names
             b. make dict[limb_idx][node_idx] = value for all nodes using the dictionary
        
        """
        for limb_name in function_mapping.keys():
            function_return = current_function(curr_neuron_concept_network.nodes[limb_name]["data"],limb_name=limb_name,**function_kwargs)
            curr_limb_concept_network = curr_neuron_concept_network.nodes[limb_name]["data"].concept_network
            
            function_mapping[limb_name] =  convert_limb_function_return_to_dict(function_return,
                                                        curr_limb_concept_network)
            
            """ Older way of doing this before functionality was moved out to function
            if np.isscalar(function_return):
                for branch_idx in curr_limb_concept_network.nodes():
                    function_mapping[limb_name][branch_idx] = function_return
            elif set(list(function_return.keys())) == set(list(curr_limb_concept_network.nodes())):
                function_mapping[limb_name] = function_return
            else:
                raise Exception("The value returned from limb function was not a scalar nor did it match the keys of the limb branches")
            """
        
    else:
        raise Exception("Function recieved was neither a Branch nor a Limb")
        
    return function_mapping


def map_new_limb_node_value(current_df,mapping_dict,value_name):
    """
    To apply a dictionary to a neuron dataframe table
    
    mapping_dict = dict()
    for x,y in zip(neuron_df["limb"].to_numpy(),neuron_df["node"].to_numpy()):
        if x not in mapping_dict.keys():
            mapping_dict[x]=dict()
        mapping_dict[x][y] = np.random.randint(10)
        
    map_new_limb_node_value(neuron_df,mapping_dict,value_name="random_number")
    neuron_df
    """
    if len(current_df) > 0:
        current_df[value_name] = current_df.apply(lambda x: mapping_dict[x["limb"]][x["node"]], axis=1)
    return current_df

import system_utils as su
def generate_neuron_dataframe(current_neuron,
                              functions_list,
                              check_nans=True,
                              function_kwargs=dict()):
    """
    Purpose: With a neuron and a specified set of functions generate a dataframe
    with the values computed
    
    Arguments:
    current_neuron: Either a neuron object or the concept network of a neuron
    functions_list: List of functions to process the limbs and branches of the concept network
    check_nans : whether to check and raise an Exception if any nans in run
    
    Application: We will then later restrict using df.eval()
    
    Pseudocode: 
    1) convert the functions_list to a list
    2) Create a dataframe for the neuron
    3) For each function:
    a. get the dictionary mapping of limbs/branches to values
    b. apply the values to the dataframe
    4) return the dataframe
    
    Example: 
    returned_df = ns.generate_neuron_dataframe(recovered_neuron,functions_list=[
    ns.n_faces_branch,
    ns.width,
    ns.skeleton_distance_branch,
    ns.skeleton_distance_limb,
    ns.n_faces_limb,
    ns.merge_limbs,
    ns.limb_error_branches
    ])

    returned_df[returned_df["merge_limbs"] == True]
    """
    
    if not nu.is_array_like(functions_list):
        functions_list = [functions_list]
        
    #print(f"functions_list = {functions_list}")
    
    
    limbs_to_process = function_kwargs.get("limbs_to_process",None)
    limb_branch_dict_restriction=function_kwargs.get("limb_branch_dict_restriction",None)
    
    #2) Create a dataframe for the neuron
    curr_df = convert_neuron_to_branches_dataframe(current_neuron,
                                                  limbs_to_process=limbs_to_process,
                                                  limb_branch_dict_restriction=limb_branch_dict_restriction)
    
    """
    3) For each function:
    a. get the dictionary mapping of limbs/branches to values
    b. apply the values to the dataframe
    
    """
    
    for curr_function in functions_list:
        curr_function_mapping = apply_function_to_neuron(current_neuron,curr_function,function_kwargs)
        map_new_limb_node_value(curr_df,curr_function_mapping,value_name=curr_function.__name__)
        
    if check_nans:
        if pu.n_nans_total(curr_df) > 0:
            print(f"Number of nans = {pu.n_nans_per_column(curr_df)}")
            su.compressed_pickle(curr_df,"curr_df")
            raise Exception("Some fo the data in the dataframe were incomplete")
            
    
    #4) return the dataframe
    return curr_df



# -------------------- Function that does full querying of neuron -------------------------- #


import sys
current_module = sys.modules[__name__]

def query_neuron(concept_network,
                         functions_list,
                          query,
                         function_kwargs=None,
                          query_variables_dict=None,
                          return_dataframe=False,
                          return_limbs=False,
                          return_limb_grouped_branches=True,
                         limb_branch_dict_restriction = None,
                         print_flag=False,
                 
                         ):
    """
    *** to specify "limbs_to_process" to process just put in the function kwargs
    
    Purpose: Recieve a neuron object or concept map 
    representing a neuron and apply the query
    to find the releveant limbs, branches
    
    
    Possible Ouptuts: 
    1) filtered dataframe
    2) A list of the [(limb_idx,branches)] ** default
    3) A dictionary that makes limb_idx to the branches that apply (so just grouping them)
    4) Just a list of the limbs
    
    Arguments
    concept_network,
    feature_functios, #the list of str/functions that specify what metrics want computed (so can use in query)
    query, #df.query string that specifies how to filter for the desired branches/limbs
    local_dict=dict(), #if any variables in the query string whose values can be loaded into query (variables need to start with @)
    return_dataframe=False, #if just want the filtered dataframe
    return_limbs=False, #just want limbs in query returned
    return_limb_grouped_branches=True, #if want dictionary with keys as limbs and values as list of branches in the query
    print_flag=True,
    
    
    Example: 
    from os import sys
    sys.path.append("../../meshAfterParty/meshAfterParty/")
    from importlib import reload
    
    import pandas_utils as pu
    import pandas as pd
    from pathlib import Path
    
    
    compressed_neuron_path = Path("../test_neurons/test_objects/12345_2_soma_practice_decompress")

    import neuron_utils as nru
    nru = reload(nru)
    import neuron
    neuron=reload(neuron)

    import system_utils as su

    with su.suppress_stdout_stderr():
        recovered_neuron = nru.decompress_neuron(filepath=compressed_neuron_path,
                          original_mesh=compressed_neuron_path)

    recovered_neuron
    
    ns = reload(ns)
    nru = reload(nru)

    list_of_faces = [1038,5763,7063,11405]
    branch_threshold = 31000
    current_query = "n_faces_branch in @list_of_faces or skeleton_distance_branch > @branch_threshold"
    local_dict=dict(list_of_faces=list_of_faces,branch_threshold=branch_threshold)


    functions_list=[
    ns.n_faces_branch,
    "width",
    ns.skeleton_distance_branch,
    ns.skeleton_distance_limb,
    "n_faces_limb",
    ns.merge_limbs,
    ns.limb_error_branches
    ]

    returned_output = ns.query_neuron(recovered_neuron,
                             functions_list,
                              current_query,
                              local_dict=local_dict,
                              return_dataframe=False,
                              return_limbs=False,
                              return_limb_grouped_branches=True,
                             print_flag=False)
    
    
    
    Example 2:  How to use the local dictionary with a list
    
    ns = reload(ns)

    current_functions_list = [
        "skeletal_distance_from_soma",
        "no_spine_average_mesh_center",
        "n_spines",
        "n_faces_branch",

    ]

    function_kwargs=dict(somas=[0],print_flag=False)
    query="skeletal_distance_from_soma > -1 and (limb in @limb_list)"
    query_variables_dict = dict(limb_list=['L1','L2',"L3"])

    limb_branch_dict_df = ns.query_neuron(uncompressed_neuron,
                                       query=query,
                                          function_kwargs=function_kwargs,
                                          query_variables_dict=query_variables_dict,
                   functions_list=current_functions_list,
                                      return_dataframe=True)

    limb_branch_dict = ns.query_neuron(uncompressed_neuron,
                                       query=query,
                   functions_list=current_functions_list,
                                       query_variables_dict=query_variables_dict,
                                       function_kwargs=function_kwargs,
                                      return_dataframe=False)
    
    
    """
    if function_kwargs is None:
        function_kwargs=dict()
        
    if query_variables_dict is None:
        query_variables_dict=dict()
    
    
    local_dict = query_variables_dict
    concept_network_old = concept_network
    #any preprocessing work
    if concept_network.__class__.__name__ == "Neuron":
        if print_flag:
            print("Extracting concept network from neuron object")
        concept_network = concept_network.concept_network
        
    if print_flag:
        print(f"functions_list = {functions_list}")
        
    final_feature_functions = []
    for f in functions_list:
        if type(f) == str:
            try:
                curr_feature = getattr(current_module, f)
                #curr_feature = 0
            except:
                raise Exception(f"The funciton {f} specified by string was not a pre-made funciton in neuron_searching module")
        elif callable(f):
            curr_feature = f
        else:
            raise Exception(f"Function item {f} was not a string or callable function")
        
        final_feature_functions.append(curr_feature)
    
    if print_flag:
        print(f"final_feature_functions = {final_feature_functions}")
        
        
    
    if limb_branch_dict_restriction is not None:
        function_kwargs["limb_branch_dict_restriction"] = limb_branch_dict_restriction
        
    #0) Generate a pandas table that originally has the limb index and node index
    returned_df = generate_neuron_dataframe(concept_network,functions_list=final_feature_functions,
                                           function_kwargs=function_kwargs)
    if len(returned_df)>0:
        filtered_returned_df = returned_df.query(query,
                      local_dict=local_dict)
    else:
        filtered_returned_df = returned_df
    
    """
    Preparing output for returning
    """

    if return_dataframe:
        return filtered_returned_df
    
    
    if len(filtered_returned_df)>0:
        limb_branch_pairings = filtered_returned_df[["limb","node"]].to_numpy()

        #gets a dictionary where key is the limb and value is a list of all the branches that were in the filtered dataframe
        limb_to_branch = dict([(k,np.sort(limb_branch_pairings[:,1][np.where(limb_branch_pairings[:,0]==k)[0]]).astype("int")) 
                               for k in np.unique(limb_branch_pairings[:,0])])
        
        if limb_branch_dict_restriction is not None:
            limb_to_branch = nru.limb_branch_intersection([limb_branch_dict_restriction,limb_to_branch])
        
        if return_limbs:
            return list(limb_to_branch.keys())

        if return_limb_grouped_branches:
            return limb_to_branch

        return limb_branch_pairings
    else:
        if return_limbs:
            return []
        if return_limb_grouped_branches:
            return dict()
        if return_limb_grouped_branches:
            return np.array([]).reshape(-1,2)
        


import neuron_searching as ns
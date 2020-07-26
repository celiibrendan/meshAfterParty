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

def convert_neuron_to_branches_dataframe(current_neuron):
    """
    Purpose: 
    How to turn a concept map into a pandas table with only the limb_idx and node_idx
    
    Example: 
    neuron_df = convert_neuron_to_branches_dataframe(current_neuron = recovered_neuron)
    
    """
    curr_concept_network = nru.return_concept_network(current_neuron)
    limb_idxs = nru.get_limb_names_from_concept_network(curr_concept_network)
    
    
    limb_node_idx_dicts = []

    for l in limb_idxs:
        limb_node_idx_dicts += [dict(limb=l,node=int(k)) for k in 
                                curr_concept_network.nodes[l]["data"].concept_network.nodes()]

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
def n_faces_branch(curr_branch):
    return len(curr_branch.mesh.faces)

@run_options(run_type="Branch")
def width(curr_branch):
    return curr_branch.width

@run_options(run_type="Branch")
def skeleton_distance_branch(curr_branch):
    try:
        #print(f"curr_branch.skeleton = {curr_branch.skeleton.shape}")
        return sk.calculate_skeleton_distance(curr_branch.skeleton)
    except:
        print(f"curr_branch.skeleton = {curr_branch.skeleton}")
        raise Exception("")



#------------------------------- Limb Functions ------------------------------------#
"""
Rule: For the limb functions will either return
1) 1 number
2) 1 True/False Value
3) Array of 1 or 2 that matches the number of branches
4) a list of nodes that it applies to

"""
@run_options(run_type="Limb")
def skeleton_distance_limb(curr_limb):
    curr_skeleton = curr_limb.get_skeleton()
    return sk.calculate_skeleton_distance(curr_skeleton)

@run_options(run_type="Limb")
def n_faces_limb(curr_limb):
    return len(curr_limb.mesh.faces)

@run_options(run_type="Limb")
def merge_limbs(curr_limb):
    return "MergeError" in curr_limb.labels

@run_options(run_type="Limb")
def limb_error_branches(curr_limb):
    error_nodes = nru.classify_endpoint_error_branches_from_limb_concept_network(curr_limb.concept_network)
    node_names = np.array(list(curr_limb.concept_network.nodes()))
    return dict([(k,k in error_nodes) for k in node_names])

@run_options(run_type="Limb")
def average_branch_length(curr_limb):
    return np.mean([sk.calculate_skeleton_distance(curr_limb.concept_network.nodes[k]["data"].skeleton) for k in curr_limb.concept_network.nodes()])

#------------------------------- Creating the Data tables from the neuron and functions------------------------------

def apply_function_to_neuron(current_neuron,current_function):
    """
    Purpose: To retrieve a dictionary mapping every branch on every node
    to a certain value as defined by the function passed
    
    Example: 
    curr_function = ns.width
    curr_function_mapping = ns.apply_function_to_neuron(recovered_neuron,curr_function)
    
    """
    
    curr_neuron_concept_network = nru.return_concept_network(current_neuron)
    curr_limb_names = nru.get_limb_names_from_concept_network(curr_neuron_concept_network)
    
    function_mapping = dict([(limb_name,dict()) for limb_name in curr_limb_names])

    #if it was a branch function that was passed
    if current_function.run_type=="Branch":
        for limb_name in function_mapping.keys():
            curr_limb_concept_network = curr_neuron_concept_network.nodes[limb_name]["data"].concept_network
            for branch_idx in curr_limb_concept_network.nodes():
                function_mapping[limb_name][branch_idx] = current_function(curr_limb_concept_network.nodes[branch_idx]["data"])
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
            function_return = current_function(curr_neuron_concept_network.nodes[limb_name]["data"])
            curr_limb_concept_network = curr_neuron_concept_network.nodes[limb_name]["data"].concept_network
            if np.isscalar(function_return):
                for branch_idx in curr_limb_concept_network.nodes():
                    function_mapping[limb_name][branch_idx] = function_return
            elif set(list(function_return.keys())) == set(list(curr_limb_concept_network.nodes())):
                function_mapping[limb_name] = function_return
            else:
                raise Exception("The value returned from limb function was not a scalar nor did it match the keys of the limb branches")  
        
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
    current_df[value_name] = current_df.apply(lambda x: mapping_dict[x["limb"]][x["node"]], axis=1)
    return current_df

def generate_neuron_dataframe(current_neuron,functions_list,check_nans=True):
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
    
    #2) Create a dataframe for the neuron
    curr_df = convert_neuron_to_branches_dataframe(current_neuron)
    
    """
    3) For each function:
    a. get the dictionary mapping of limbs/branches to values
    b. apply the values to the dataframe
    
    """
    
    for curr_function in functions_list:
        curr_function_mapping = apply_function_to_neuron(current_neuron,curr_function)
        map_new_limb_node_value(curr_df,curr_function_mapping,value_name=curr_function.__name__)
        
    if check_nans:
        if pu.n_nans_total(curr_df) > 0:
            print(f"Number of nans = {pu.n_nans_per_column(neuron_df)}")
            raise Exception("Some fo the data in the dataframe were incomplete")
    
    #4) return the dataframe
    return curr_df



# -------------------- Function that does full querying of neuron -------------------------- #


import sys
current_module = sys.modules[__name__]

def query_neuron(concept_network,
                         functions_list,
                          query,
                          local_dict=dict(),
                          return_dataframe=False,
                          return_limbs=False,
                          return_limb_grouped_branches=True,
                         print_flag=False,
                         ):
    """
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
    
    
    
    """
    
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
        
    #0) Generate a pandas table that originally has the limb index and node index
    returned_df = generate_neuron_dataframe(concept_network,functions_list=final_feature_functions)
    
    filtered_returned_df = returned_df.query(query,
                  local_dict=local_dict)
    
    """
    Preparing output for returning
    """

    if return_dataframe:
        return filtered_returned_df
    
    limb_branch_pairings = filtered_returned_df[["limb","node"]].to_numpy()
    
    #gets a dictionary where key is the limb and value is a list of all the branches that were in the filtered dataframe
    limb_to_branch = dict([(k,np.sort(limb_branch_pairings[:,1][np.where(limb_branch_pairings[:,0]==k)[0]]).astype("int")) 
                           for k in np.unique(limb_branch_pairings[:,0])])
    if return_limbs:
        return list(limb_to_branch.keys())
    
    if return_limb_grouped_branches:
        return limb_to_branch
    
    return limb_branch_pairings



import numpy as np
import copy

import neuron_searching as ns
import neuron_visualizations as nviz
import networkx_utils as xu
import skeleton_utils as sk

def filter_axon_limb_false_positive_end_nodes(curr_limb,curr_limb_axon_like_nodes,verbose=False,skeleton_length_threshold=30000):
    """
    Purpose: Will remove end nodes that were accidentally mistaken as axons
    
    """
    final_axon_like_nodes = np.array(copy.copy(curr_limb_axon_like_nodes))

    if len(curr_limb_axon_like_nodes)>0:
        #2) Filter for only those that are end nodes
        axon_node_degrees = np.array(xu.get_node_degree(curr_limb.concept_network,curr_limb_axon_like_nodes))
        end_node_idx = np.where(axon_node_degrees == 1)[0]

        if len(end_node_idx) == 0:
            pass
        else:
            nodes_to_check = curr_limb_axon_like_nodes[end_node_idx]
            for n_name in nodes_to_check:
                curr_sk_length = sk.calculate_skeleton_distance(curr_limb[n_name].skeleton) 
                if curr_sk_length > skeleton_length_threshold:
                    if verbose:
                        print(f"Skipping because skeleton too long: {curr_sk_length}")
                    continue
                
                curr_neighbors = xu.get_neighbors(curr_limb.concept_network,n_name)
                if verbose:
                    print(f"curr_neighbors = {curr_neighbors}")
                if len(curr_neighbors) == 0:
                    if verbose:
                        print("skipping because no neighbors")
                    pass
                elif curr_neighbors[0] in curr_limb_axon_like_nodes:
                    if verbose:
                        print("skipping because neighbor axon")
                    pass
                else:
                    if verbose:
                        print(f"Skipping end node {n_name} because neighbor was dendrite")
                    final_axon_like_nodes = final_axon_like_nodes[final_axon_like_nodes != n_name]
    return final_axon_like_nodes



def filter_axon_neuron_false_positive_end_nodes(neuron_obj,current_axons_dict):
    filtered_axon_dict = dict()
    for limb_name_key,curr_limb_axon_like_nodes in current_axons_dict.items():

        curr_limb_idx = int(limb_name_key[1:])

        curr_limb = neuron_obj[curr_limb_idx]
        filtered_axon_dict[limb_name_key] = filter_axon_limb_false_positive_end_nodes(curr_limb,curr_limb_axon_like_nodes)
    return filtered_axon_dict

def axon_like_segments(neuron_obj,include_ais=False,filter_away_end_false_positives=True,visualize_at_end=False,width_to_use=None,
                       verbose=False):
    current_neuron = neuron_obj
    axon_like_limb_branch_dict = ns.axon_width_like_segments(current_neuron,
                                                        include_ais=include_ais,
                                                             width_to_use=width_to_use,
                                                            verbose=verbose)
    
    
    
    current_functions_list = ["axon_segment"]
    limb_branch_dict_upstream_filter = ns.query_neuron(current_neuron,
                                       query="axon_segment==True",
                                       function_kwargs=dict(limb_branch_dict =axon_like_limb_branch_dict,
                                                            downstream_face_threshold=3000,
                                                            width_match_threshold=50,
                                                           print_flag=False),
                                       functions_list=current_functions_list)
    
    if filter_away_end_false_positives:
        if verbose:
            print("Using filter_away_end_false_positives")
        limb_branch_dict_upstream_filter = filter_axon_neuron_false_positive_end_nodes(neuron_obj,
                                                                                      limb_branch_dict_upstream_filter)
        
    if visualize_at_end:
        colors_dict_returned = nviz.visualize_neuron(current_neuron,
                                             visualize_type=["mesh"],
                      limb_branch_dict=limb_branch_dict_upstream_filter,
                     mesh_color="red",
                     mesh_color_alpha=1,
                     mesh_whole_neuron=True,
                     return_color_dict=True)
    
    return limb_branch_dict_upstream_filter



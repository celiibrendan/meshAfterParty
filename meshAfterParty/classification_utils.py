"""
Utils for helping with the classification of a neuron
for compartments like axon, apical, basal...

"""

import neuron_utils as nru
import numpy as np
import neuron_statistics as nst
import neuron_visualizations as nviz

import networkx_utils as xu
import networkx as nx
import matplotlib_utils as mu

import neuron_searching as ns

top_volume_vector = np.array([0,-1,0])

def axon_candidates(neuron_obj,
                    possible_axon_limbs = None,
                   ais_threshold=20000,
                   plot_close_branches=False,
                    plot_candidats_after_elimination=False,
                    plot_candidates_after_adding_back=False,
                   verbose=False,
                   **kwargs):
    """
    Purpose: To return with a list of the possible 
    axon subgraphs of the limbs of a neuron object
    
    Pseudocode: 
    1) Find all the branches in the possible ais range and delete them from the concept networks
    2) Collect all the leftover branches subgraph as candidates
    3) Add back the candidates that were deleted
    4) Combining all the candidates in one list
    """
    curr_neuron_obj = neuron_obj
    
    if possible_axon_limbs is None:
        possible_axon_limbs = nru.get_limb_names_from_concept_network(curr_neuron_obj.concept_network)
    
    
    
    close_limb_branch_dict = ns.query_neuron(curr_neuron_obj,
                                            functions_list=[ns.skeletal_distance_from_soma],
                                            query=f"skeletal_distance_from_soma<{ais_threshold}",
                                            function_kwargs=dict(limbs_to_process=possible_axon_limbs),
                                             #return_dataframe=False


                                            )
    outside_bubble_limb_branch_dict = ns.query_neuron(curr_neuron_obj,
                                            functions_list=[ns.skeletal_distance_from_soma],
                                            query=f"skeletal_distance_from_soma>={ais_threshold}",
                                            function_kwargs=dict(limbs_to_process=possible_axon_limbs),
                                             #return_dataframe=False


                                            )
    
    if plot_close_branches:
        colors_dict_returned = nviz.visualize_neuron(curr_neuron_obj,
                              limb_branch_dict=close_limb_branch_dict,
                             mesh_color="red",
                             mesh_color_alpha=1,
                             mesh_whole_neuron=True,
                             return_color_dict=True)

        
        
        
    
    # 2) --------Delete the nodes from the branch graph and then group into connected ocmponents
    # into candidates

    limbs_to_check = [nru.get_limb_string_name(k) for k in possible_axon_limbs]

    sub_limb_color_dict = dict()
    total_sub_limbs = dict() #will map the limbs to the connected components


    for limb_idx in limbs_to_check:
        print(f"\nPhase 2: Working on Limb {limb_idx}")

        #initializing the candidate list and the color dictionary for visualization
        total_sub_limbs[limb_idx] = []
        sub_limb_color_dict[limb_idx] = dict()



        curr_limb = curr_neuron_obj[limb_idx]

        if limb_idx in close_limb_branch_dict.keys():
            nodes_to_eliminate = close_limb_branch_dict[limb_idx]
        else:
            nodes_to_eliminate = []

        #the nodes that were eliminated we need to show deleted colors
        for n in nodes_to_eliminate:
            sub_limb_color_dict[limb_idx][n] = mu.color_to_rgba("black", alpha=1)

        if verbose:
            print(f"nodes_to_eliminate = {nodes_to_eliminate}")

        curr_filt_network = nx.Graph(curr_limb.concept_network_directional)
        curr_filt_network.remove_nodes_from(nodes_to_eliminate)

        if len(curr_filt_network) == 0:
            if verbose:
                print("The filtered network is empty so just leaving the candidates as empty lists")
            continue

        curr_limb_conn_comp = list(nx.connected_components(curr_filt_network))


        total_sub_limbs[limb_idx] = [list(k) for k in curr_limb_conn_comp]

        colors_to_use = mu.generate_unique_random_color_list(n_colors=len(curr_limb_conn_comp),colors_to_omit=["black","midnightblue"])
        for j,(c_comp,curr_random_color) in enumerate(zip(curr_limb_conn_comp,colors_to_use)):

            for n in c_comp:
                sub_limb_color_dict[limb_idx][n] = curr_random_color


                
    if plot_candidats_after_elimination:
        nviz.visualize_neuron(curr_neuron_obj,
                              visualize_type=["mesh"],
                             limb_branch_dict="all",
                             mesh_color=sub_limb_color_dict,
                             mesh_fill_color="green")
        
        
        
        
        
        
        
    # ----------- Part 3: ------------------#
        
        
    """
    3) Adding back all endpoints that were eliminated in step 2: Process is below

    For each limb

    0) Get all of the endpoint nodes in the whole directed concept network
    - remove the starting node from the list
    1) Find the shortest path from every endpoint to the starting node
    2) Concatenate shortest paths into dictionary mapping node to number of
    occurances in the shortest paths
    3) Find all of the endpoints that were eliminated with the restriction
    4) Filter those endpoint paths for nodes that only have an 
    occurance of one for the lookup dictionary
    5) Add all deleted endpoint filtered shortest paths as candidates

    How to handle corner cases:
    1) If only starting node that got deleted
    --> just add that as a candidate
    2) If all of network got deleted, current way will work

    """

    removed_candidates = dict()

    for limb_idx in limbs_to_check:
        if verbose:
            print(f"\n----Working on Limb {limb_idx}-----")

        curr_limb = curr_neuron_obj[limb_idx]    

        removed_candidates[limb_idx] = []

        if limb_idx in close_limb_branch_dict.keys():
            nodes_to_eliminate = close_limb_branch_dict[limb_idx]
        else:
            nodes_to_eliminate = []
            if verbose:
                print("No nodes were eliminated so don't need to add back any candidates")
            continue


        curr_network = nx.Graph(curr_limb.concept_network_directional)
        curr_starting_node = curr_limb.current_starting_node

        #covering the corner case that only the root node existed
        #and it was deleted
        if len(nodes_to_eliminate) == 1 and len(curr_network)==1:
            if verbose:
                print("network was only of size 1 and that node was eliminated so returning that as the only candidate")
            removed_candidates[limb_idx] = [[curr_starting_node]]

            #adding the color
            curr_random_color = mu.generate_unique_random_color_list(n_colors=1,colors_to_omit=["black","midnightblue"])[0]
            sub_limb_color_dict[limb_idx][n] = curr_random_color

        else:
            #0) Get all of the endpoint nodes in the whole directed concept network
            #- remove the starting node from the list
            curr_endpoints = xu.get_nodes_of_degree_k(curr_network,1)
            if curr_starting_node in curr_endpoints:
                curr_endpoints.remove(curr_starting_node)


            #3) Find all of the endpoints that were eliminated with the restriction
            endpoints_eliminated = [k for k in curr_endpoints if k in nodes_to_eliminate]

            if len(endpoints_eliminated) == 0:
                if verbose:
                    print("No endpoints were eliminated so don't need to add back any candidates")
                continue

            #1) Find the shortest path from every endpoint to the starting node
            shortest_paths_endpoints = dict()
            for en in curr_endpoints:
                en_shortest_path = nx.shortest_path(curr_network,
                                source = en,
                                 target = curr_starting_node)
                shortest_paths_endpoints[en] = en_shortest_path

            #2) Concatenate shortest paths into dictionary mapping node to number of
            #occurances in the shortest paths
            node_occurance = dict()
            for curr_path in shortest_paths_endpoints.values():
                for n in curr_path:
                    if n not in node_occurance.keys():
                        node_occurance[n] = 1
                    else:
                        node_occurance[n] += 1

            #4) Filter those endpoint paths for nodes that only have an 
            #occurance of one for the lookup dictionary
            added_back_candidates = []
            for en_elim in endpoints_eliminated:
                filtered_path = [k for k in shortest_paths_endpoints[en_elim] if node_occurance[k] == 1]
                added_back_candidates.append(filtered_path)

            if verbose:
                print(f"New candidates added back: {added_back_candidates}")

            removed_candidates[limb_idx] = added_back_candidates

        #5) Adding the new paths to the color dictionary for visualization 
        colors_to_use = mu.generate_unique_random_color_list(n_colors=len(removed_candidates[limb_idx]),colors_to_omit=["black","midnightblue"])
        for add_path,curr_random_color in zip(removed_candidates[limb_idx],colors_to_use):
            for n in add_path:
                sub_limb_color_dict[limb_idx][n] = curr_random_color

    # checking that adding back the candidates went well

    if plot_candidates_after_adding_back:
        nviz.visualize_neuron(curr_neuron_obj,
                              visualize_type=["mesh"],
                             limb_branch_dict="all",
                             mesh_color=sub_limb_color_dict,
                             mesh_fill_color="green")
        
        
        
        
    # --------- Part 4: Combining All the Candidates ------------ #
    
    all_candidates = dict()
    for limb_name in limbs_to_check:
        all_candidates[int(limb_name[1:])] = total_sub_limbs[limb_name] + removed_candidates[limb_name]

    if verbose:
        print("Final Candidates")
        for limb_name, list_of_subgraphs in all_candidates.items():
            print(f"\nLimb {limb_name}")
            for sg in list_of_subgraphs:
                print(repr(np.array(sg)))
                
    return all_candidates


import skeleton_utils as sk
import numpy_utils as nu
def candidate_starting_skeletal_angle(limb_obj,candidate_nodes,
                                      offset = 20000,
                                    axon_sk_direction_comparison_distance = 5000,
                                    buffer_for_skeleton = 5000,
                                      top_volume_vector = np.array([0,-1,0]),
                                      plot_skeleton_paths_before_restriction=False,
                                      plot_skeleton_paths_after_restriction=False,
                                      return_restricted_skeletons=False,
                                      branches_not_to_consider_for_end_nodes = None,
                                      verbose=False,
                                     ):
    
    """
    Purpose: To get the skeleton that represents the starting skeleton
    --> and then find the projection angle to filter it away or not

    Pseudocode: 
    1) convert the graph into a skeleton (this is when self touches could be a problem)
    2) Find all skeleton points that are within a certain distance of the starting coordinate
    3) Find all end-degree nodes (except for the start)
    4) Find path back to start for all end-nodes
    5) Find paths that are long enough for the offset plus test --> if none then don't filter
    anyway

    For each valid path (make them ordered paths):
    6) Get the offset + test subskeletons for all valid paths
    7) Get the angle of the sksletons vectors
    """
    
    # -- Renaming the variables --
    subgraph_branches = candidate_nodes
    curr_limb = limb_obj
    
    
    sk_size_to_compare = axon_sk_direction_comparison_distance + offset
    total_distance = offset + axon_sk_direction_comparison_distance + buffer_for_skeleton
    
    
    
    #1) convert the graph into a skeleton (this is when self touches could be a problem)
    candidate_sk= sk.stack_skeletons([curr_limb[k].skeleton for k in subgraph_branches])
    candidate_sk_graph = sk.convert_skeleton_to_graph(candidate_sk)

    #2) Find all skeleton points that are within a certain distance of the starting coordinate
    starting_sk_coord = curr_limb.current_starting_coordinate
    starting_sk_node = xu.get_graph_node_by_coordinate(candidate_sk_graph,starting_sk_coord)
    skeletons_nodes_for_comparison = xu.find_nodes_within_certain_distance_of_target_node(
        candidate_sk_graph,
        starting_sk_node,total_distance)
    np.array(list(skeletons_nodes_for_comparison))
    comparison_subgraph = candidate_sk_graph.subgraph(skeletons_nodes_for_comparison)
    
    #3) Find all edn-degree nodes (except for the start)
    all_endnodes = xu.get_nodes_of_degree_k(comparison_subgraph,1)
    starting_coordinate_endnode = xu.get_graph_node_by_coordinate(
        comparison_subgraph,
        starting_sk_coord)
    
    
        
    
    endnodes_to_test = np.setdiff1d(all_endnodes,[starting_coordinate_endnode])
    
    # ------------ 1/24 Addition: Will get rid of end nodes that are on dendritic portions ----- #
    if branches_not_to_consider_for_end_nodes is not None:
        """
        Pseudocode: 
        1) Get coordinate of end node
        2) Get the branches that belong to that coordinate
        3) Subtract off the branches that shouldn't be considered
        4) If empty then skip, if not then add
        
        """
        debug_cancellation = False
        new_endnodes_to_test = []
        
        if debug_cancellation:
            print(f"branches_not_to_consider_for_end_nodes: {branches_not_to_consider_for_end_nodes}")
            print(f"endnodes_to_test BEFORE FILTERING= {endnodes_to_test}")
        
        for curr_endpoint in endnodes_to_test:
            if debug_cancellation:
                print(f"working on endpoint: {curr_endpoint}")
            curr_endpoint_coordinate = xu.get_coordinate_by_graph_node(comparison_subgraph,
                                                                      curr_endpoint)
            
            branches_of_endnode = nru.find_branch_with_specific_coordinate(limb_obj,
                                                                           curr_endpoint_coordinate)
            viable_branches = np.setdiff1d(branches_of_endnode,branches_not_to_consider_for_end_nodes)
            
            if debug_cancellation:
                print(f"branches_of_endnode: {branches_of_endnode}")
                print(f"viable_branches: {viable_branches}")
                
            
            if len(viable_branches)>0:
                new_endnodes_to_test.append(curr_endpoint)
        
        endnodes_to_test = np.array(new_endnodes_to_test)
        
        if debug_cancellation:
            print(f"endnodes_to_test AFTER FILTERING= {endnodes_to_test}")

    if verbose:
        print(f"endnodes_to_test = {endnodes_to_test}")
        
        
    # nviz.plot_objects(curr_limb.mesh,
    #             skeletons=[sk.convert_graph_to_skeleton(comparison_subgraph)],
    #                  )
    
    
    if len(endnodes_to_test) == 0:
        if return_restricted_skeletons:
            return None,None
        else:
            return None
        
        
        
    #4) Find path back to start for all end-nodes
    
    paths_to_test = [nx.shortest_path(comparison_subgraph,
                                      starting_coordinate_endnode,
                                      k
                                     ) for k in endnodes_to_test]
    sk_paths_to_test = [sk.convert_graph_to_skeleton(comparison_subgraph.subgraph(k))
                           for k in paths_to_test]
    sk_paths_to_test_ordered = [sk.order_skeleton(k,
                                                  start_endpoint_coordinate = starting_sk_coord)
                               for k in sk_paths_to_test]

    if len(sk_paths_to_test_ordered) <= 0: 
        raise Exception("Found no skeleton paths")
        
    if plot_skeleton_paths_before_restriction:
        endpoint_scatters = xu.get_coordinate_by_graph_node(comparison_subgraph,endnodes_to_test)
        for k,sc_point in zip(sk_paths_to_test_ordered,endpoint_scatters):
            nviz.plot_objects(curr_limb.mesh,
                             skeletons=[k],
                             scatters=[sc_point.reshape(-1,3)])
            
            
    #5) Find paths that are long enough for the offset plus test --> if none then don't filter any
    sk_distances = np.array([sk.calculate_skeleton_distance(k) for k in sk_paths_to_test_ordered])
    filtered_indexes = np.where(sk_distances>=sk_size_to_compare)[0]


    if len(filtered_indexes)> 0:
        filtered_skeletons = [sk_paths_to_test_ordered[k] for k in filtered_indexes]
    else:
        filtered_skeletons = sk_paths_to_test_ordered

    if verbose:
        print(f"Skeleton paths distances = {sk_distances}")
        print(f"Filtered indexes = {filtered_indexes}")
        print(f"len(filtered_skeletons) = {len(filtered_skeletons)}")
        
        
        
    
    #6) Get the offset + test subskeletons for all valid paths
    filtered_skeletons_restricted = [sk.restrict_skeleton_from_start_plus_offset(k,
                                        offset=offset,
                                        comparison_distance=axon_sk_direction_comparison_distance)
             for k in filtered_skeletons]

    if plot_skeleton_paths_after_restriction:
        endpoint_scatters = xu.get_coordinate_by_graph_node(comparison_subgraph,endnodes_to_test)
        for k,sc_point in zip(filtered_skeletons_restricted,endpoint_scatters):
            nviz.plot_objects(curr_limb.mesh,
                             skeletons=[k],
                             scatters=[sc_point.reshape(-1,3)])
            
    #7) Get the angle of the sletons vectors

    #angle between going down and skeleton vector
    sk_vectors = [sk.skeleton_endpoint_vector(k) for k in filtered_skeletons_restricted]
    sk_angles = np.array([nu.angle_between_vectors(top_volume_vector,k) for k in sk_vectors])

    

    if verbose:
        print(f"sk_angles = {sk_angles}")
        
        
    if return_restricted_skeletons:
        return sk_angles,filtered_skeletons_restricted
    else:
        return sk_angles
    
    
import neuron_searching as ns

def filter_axon_candiates(neuron_obj,
    axon_subgraph_candidates,
    axon_angle_threshold_relaxed = 110,#90,
    axon_angle_threshold = 120,
    relaxation_percentage = 0.85,
                          
    #parameters for computing the skeletal angle
     
    skeletal_angle_offset = 10000,
    skeletal_angle_comparison_distance = 10000,
    skeletal_angle_buffer = 5000,
                          
    axon_like_limb_branch_dict = None,
                          
                          min_ais_width=85,
    verbose = False,
    
                          
    return_axon_angles = True,
    **kwargs
    ):

    """
    Pseudocode: 

    For each candidate: 

    0) If all Axon? (Have a more relaxed threshold for the skeleton angle)
    1) Find the starting direction, and if not downwards --> then not axon
    2) ------------- Check if too thin at the start --> Not Axon (NOT GOING TO DO THIS) -------------
    3) If first branch is axon --> classify as axon
    4) Trace back to starting node and add all branches that are axon like

    """
    
    
    if axon_like_limb_branch_dict is None:
        axon_like_limb_branch_dict = ns.query_neuron(neuron_obj,
                functions_list=["matching_label"],
               query="matching_label==True",
               function_kwargs=dict(labels=["axon-like"]),
               )
        
    final_axon_like_classification = axon_like_limb_branch_dict
    curr_neuron_obj = neuron_obj
    
    axon_candidate_filtered = dict()
    axon_candidate_filtered_angles = dict()
    
    for curr_limb_idx,limb_candidate_grouped_branches in axon_subgraph_candidates.items():
    
        curr_limb_name = nru.get_limb_string_name(curr_limb_idx)
        curr_limb = curr_neuron_obj[curr_limb_idx]
        
        for curr_candidate_idx,curr_candidate_subgraph in enumerate(limb_candidate_grouped_branches):
            curr_candidate_subgraph = np.array(curr_candidate_subgraph)
            
            if verbose:
                print(f"\n\n --- Working on limb {curr_limb_idx}, candidate # {curr_candidate_idx}")
                
            
            # ------------- Part A: Filtering For Axon Composition ------------------

            """
            Pseudocode: 
            1) Get the number of branches in the candidate that are axons
            2a) If all are axons --> choose the relaxed axon angle threshold
            2b) If none are axons --> remove as not a candidate
            2c) if some are --> use standard axon threshold

            """
            if curr_limb_name in final_axon_like_classification.keys():
                axon_branches_on_limb = final_axon_like_classification[curr_limb_name]
            else:
                axon_branches_on_limb = []
                
                
            axon_branches_on_subgraph = np.intersect1d(axon_branches_on_limb,curr_candidate_subgraph)
            

            axon_percentage = len(axon_branches_on_subgraph)/len(curr_candidate_subgraph)

            if verbose:
                print(f"{len(axon_branches_on_subgraph)} out of {len(curr_candidate_subgraph)} branches are axons")
                print(f"Axon percentage = {axon_percentage}")
                
            if axon_percentage > relaxation_percentage:
                curr_axon_angle_threshold = axon_angle_threshold_relaxed
                
            elif len(axon_branches_on_subgraph) == 0:
                if verbose:
                    print(f"Not adding candidate no axon branches detected ")
                continue
            else:
                curr_axon_angle_threshold = axon_angle_threshold

            if verbose:
                print(f"curr_axon_angle_threshold = {curr_axon_angle_threshold}")






            # ---------  Part B: Filtering For Starting Skeleton Angle -------------

            curr_limb.set_concept_network_directional(starting_soma = 0)

            undirectional_limb_graph = nx.Graph(curr_limb.concept_network_directional)

            current_shortest_path,st_node,end_node = xu.shortest_path_between_two_sets_of_nodes(
                undirectional_limb_graph,[curr_limb.current_starting_node],
                curr_candidate_subgraph)

            candidate_nodes = np.unique(np.hstack([curr_candidate_subgraph,current_shortest_path]))
            
            # ----- 1/24: Filtering out the nodes that are on branches that are not axons --------- #
            non_axon_branches_on_subgraph = np.setdiff1d(candidate_nodes,axon_branches_on_limb)
            
            
            
            
            

            candidate_angles,restr_skels = clu.candidate_starting_skeletal_angle(limb_obj=curr_limb,
                              candidate_nodes=candidate_nodes,
                                  offset = skeletal_angle_offset,
                                axon_sk_direction_comparison_distance = skeletal_angle_comparison_distance,
                                buffer_for_skeleton = skeletal_angle_buffer,
                                  top_volume_vector = np.array([0,-1,0]),
#                                   plot_skeleton_paths_before_restriction=False,
#                                   plot_skeleton_paths_after_restriction=False,
                                                 return_restricted_skeletons=True,
                                  verbose=verbose,
                                   branches_not_to_consider_for_end_nodes = non_axon_branches_on_subgraph,
                                **kwargs
                                 )
            
            if candidate_angles is not None:
                sk_passing_threshold = np.where(candidate_angles>curr_axon_angle_threshold)[0]
            else:
                sk_passing_threshold = []

            if len(sk_passing_threshold) == 0:
                if verbose:
                    print(f"Not adding candidate because no angles ({candidate_angles})"
                          f" passed the threhold {curr_axon_angle_threshold} ")
                continue





            # -----------Part C: Filtering by Axon Being the Current Starting Piece -------------

            candidate_starting_node = current_shortest_path[-1]
            if candidate_starting_node not in axon_branches_on_limb:
                if verbose:
                    print(f"Not adding candidate the first branch was not an axon ")
                continue
                
            if (curr_limb[candidate_starting_node].width_new["no_spine_median_mesh_center"] < min_ais_width):
                if verbose:
                    print(f'Not adding candidate the because AIS width was not higher than threshold ({min_ais_width}): {curr_limb[candidate_starting_node].width_new["no_spine_median_mesh_center"]} ')
                continue
                
            




            # ----Part D: Add all of the candidates branches and those backtracking to mesh that are axon-like
            extra_nodes_to_add = np.intersect1d(axon_branches_on_limb,current_shortest_path[:-1])
            true_axon_branches = np.hstack([curr_candidate_subgraph,extra_nodes_to_add])

            if verbose:
                print(f"Adding the following branches as true axons: {true_axon_branches}")

            if curr_limb_idx not in axon_candidate_filtered:
                axon_candidate_filtered[curr_limb_idx] = dict()
                axon_candidate_filtered_angles[curr_limb_idx] = dict()

            axon_candidate_filtered[curr_limb_idx][curr_candidate_idx] = true_axon_branches
            axon_candidate_filtered_angles[curr_limb_idx][curr_candidate_idx] = np.max(candidate_angles)
    
    #compiling list into limb_branch dict that is easy to use
    limb_branch_dict = dict()
    
    for limb_idx, limb_info in axon_candidate_filtered.items():
        curr_branches = []
        for cand_idx,cand_list in limb_info.items():
            curr_branches.append(cand_list)
        limb_branch_dict[f"L{limb_idx}"] = np.concatenate(curr_branches)
    
    if return_axon_angles:
        return limb_branch_dict, axon_candidate_filtered_angles
    else:
        return limb_branch_dict


def axon_classification(neuron_obj,
                        
                        
    error_on_multi_soma = True,
    ais_threshold = 5000,#10000,
                        
                        #Part 1: for axon-like classification
                        downstream_face_threshold=3000,
                        width_match_threshold=50,
                        plot_axon_like_segments=False,
                        
                        #Part 2: Filter limbs by starting angle
                        axon_soma_angle_threshold = 70,
                        
                        #Part 3: Creating Candidates
                        plot_candidates = False,
                        
                        #Part 4: Filtering Candidates
                        plot_axons = False,
                        plot_axon_errors=False,
                        
                        axon_angle_threshold_relaxed = 110,
                        axon_angle_threshold = 120,
                        
                        
    clean_prior_axon_labels=True,  
    label_axon_errors =True,
                        
    error_width_max = 140,
                        
    return_axon_labels=True,
    return_axon_angles=False,
                        
    return_error_labels=True,
                        
                        
    verbose = True,
                        
    **kwargs
    ):
    """
    Purpose: 
    To put the whole axon classificatoin steps 
    together into one function that will labels
    branches as axon-like, axon and error
    
    1) Classify All Axon-Like Segments
    2) Filter Limbs By Starting Angle
    3) Get all of the Viable Candidates
    4) Filter Candidates
    5) Apply Labels
    
    """
    
    curr_neuron_obj = neuron_obj
    
    soma_names = curr_neuron_obj.get_soma_node_names()
    
    if len(soma_names)>1:
        soma_print = f"More than 1 soma: {soma_names}"
        if error_on_multi_soma:
            raise Exception(soma_print)
        else:
            print(soma_print)

    soma_name = soma_names[0]
    
    
    
    
    
    #  ------------- Part 1: Classify All Axon-Like Segments ----------------------------
    
    
    axon_like_limb_branch_dict = ns.axon_width_like_segments(curr_neuron_obj,
                                                        include_ais=True)

    # nviz.visualize_neuron(curr_neuron_obj,
    #                       visualize_type=["mesh"],
    #                      limb_branch_dict=axon_like_limb_branch_dict,
    #                      mesh_color="red",
    #                       mesh_color_alpha=1,
    #                      mesh_whole_neuron=True)

    current_functions_list = ["axon_segment"]
    final_axon_like_classification = ns.query_neuron(curr_neuron_obj,

                                       query="axon_segment==True",
                                       function_kwargs=dict(limb_branch_dict =axon_like_limb_branch_dict,
                                                            downstream_face_threshold=downstream_face_threshold,
                                                            width_match_threshold=width_match_threshold,
                                                           print_flag=False),
                                       functions_list=current_functions_list)

    if plot_axon_like_segments:
        nviz.visualize_neuron(curr_neuron_obj,
                              visualize_type=["mesh"],
                             limb_branch_dict=final_axon_like_classification,
                             mesh_color="red",
                              mesh_color_alpha=1,
                             mesh_whole_neuron=True)
        
        
    
    if verbose:
        print(f"\nPart 1: Axon like branchese \n{final_axon_like_classification}")
        
        
        
    #------------------ Part 2: Filter Limbs By Starting Angle  ------------------
    
    
    

    soma_center = curr_neuron_obj["S0"].mesh_center

    possible_axon_limbs_dict = ns.query_neuron(curr_neuron_obj,
                    query=f"soma_starting_angle>{axon_soma_angle_threshold}",
                   functions_list=[ns.soma_starting_angle],
                   function_kwargs=dict(soma_center=soma_center,
                                       verbose=verbose))

    possible_axon_limbs = list(possible_axon_limbs_dict.keys())
    possible_axon_limbs = [nru.get_limb_int_name(k) for k in possible_axon_limbs]
    
    if verbose: 
        print(f'\nPart 2: possible_axon_limbs = {possible_axon_limbs}')
              
              
                
                
    
    #---------------------- Part 3: Get all of the Viable Candidates ----------------------
    
    
    axon_subgraph_candidates = clu.axon_candidates(curr_neuron_obj,
                   possible_axon_limbs=possible_axon_limbs,
                        ais_threshold=ais_threshold,
                   plot_candidates_after_adding_back=plot_candidates,
                   verbose=verbose,
                                                   
                                                  **kwargs)
              
    if verbose:
        print(f"Part 3: axon_subgraph_candidates = {axon_subgraph_candidates}")
        
        
        
        
        
    
    #---------------------- Part 4: Filtering The Candidates ----------------------
    
    curr_result = clu.filter_axon_candiates(
                            curr_neuron_obj,
                            axon_subgraph_candidates,
                            verbose = verbose,
                            axon_like_limb_branch_dict=final_axon_like_classification,
                                return_axon_angles=return_axon_angles,
                                axon_angle_threshold_relaxed = axon_angle_threshold_relaxed,
                                axon_angle_threshold = axon_angle_threshold,
                            **kwargs
                            )
    
    if return_axon_angles:
        final_true_axons, axon_angles = curr_result
    else:
        final_true_axons = curr_result
    
    if verbose:
        print(f"\n\nPart 4: final_true_axons = {final_true_axons}")
    
    if plot_axons:
        if len(final_true_axons)>0:
            nviz.visualize_neuron(curr_neuron_obj,
                                  visualize_type=["mesh"],
                                 limb_branch_dict=final_true_axons,
                                 mesh_color="red",
                                  mesh_color_alpha=1,
                                 mesh_whole_neuron=True)
        else:
            print("NO AXON DETECTED FOR PLOTTING")
        
        
        
    #---------------------- Part 5: Adding Labels ----------------------
    """
    Pseudocode: 
    1) Clear the labels if option set
    2) Label all the true axon branches
    
    
    """
    
    if clean_prior_axon_labels:
        nru.clear_all_branch_labels(curr_neuron_obj,["axon","axon-like","axon-error"])
    
#     nru.add_branch_label(curr_neuron_obj,
#                     limb_branch_dict=final_axon_like_classification,
#                     labels="axon-like")
    
    nru.add_branch_label(curr_neuron_obj,
                    limb_branch_dict=final_true_axons,
                    labels="axon")
    
    if label_axon_errors or return_error_labels:
        axon_error_like_limb_branch_dict = ns.axon_width_like_segments(curr_neuron_obj,
                                                        include_ais=False)


        current_functions_list = ["axon_segment"]
        final_axon_error_like_classification = ns.query_neuron(curr_neuron_obj,

                                           query="axon_segment==True",
                                           function_kwargs=dict(limb_branch_dict =axon_error_like_limb_branch_dict,
                                                                downstream_face_threshold=downstream_face_threshold,
                                                                width_match_threshold=width_match_threshold,
                                                               print_flag=False),
                                           functions_list=current_functions_list)
        nru.add_branch_label(curr_neuron_obj,
                limb_branch_dict=final_axon_error_like_classification,
                labels="axon-like")
        
        if error_width_max is None:
            error_limb_branch_dict = ns.query_neuron_by_labels(curr_neuron_obj,
                                 matching_labels = ["axon-like"],
                                 not_matching_labels = ["axon"]
                                 )
        else:
            error_limb_branch_dict = ns.query_neuron(neuron_obj,
                            query=f"(labels_restriction == True) and (median_mesh_center < {error_width_max})",
                   functions_list=["labels_restriction","median_mesh_center"],
                   function_kwargs=dict(matching_labels=["axon-like"],
                                        not_matching_labels=["axon"]
                                       )
                           )

        if label_axon_errors:
            nru.add_branch_label(curr_neuron_obj,
                            limb_branch_dict=error_limb_branch_dict,
                            labels="axon-error")
    
        if plot_axon_errors:
            if len(error_limb_branch_dict) > 0:
                nviz.visualize_neuron(curr_neuron_obj,
                                      visualize_type=["mesh"],
                                     limb_branch_dict=error_limb_branch_dict,
                                     mesh_color="red",
                                      mesh_color_alpha=1,
                                     mesh_whole_neuron=True)
            else:
                print("NO AXON ERRORS DETECTED FOR PLOTTING!!")
        

    if return_axon_labels and return_error_labels:
        if return_axon_angles:
            return final_true_axons,axon_angles,error_limb_branch_dict
        else:
            return final_true_axons,error_limb_branch_dict
    elif return_axon_labels:
        if return_axon_angles:
            return final_true_axons,axon_angles
        else:
            return final_true_axons
    elif return_error_labels:
        return error_limb_branch_dict
                                                    
        

        
        
        
        
        
        
# ----------- 1/22: Apical Classification (The beginning parts): ----------------- #

import networkx_utils as xu
import skeleton_utils as sk
import networkx as nx
import numpy_utils as nu

def apical_branch_candidates_on_limb(limb_obj,
                                     
                                     
                                    apical_check_distance_max = 90000,
                                    apical_check_distance_min = 25000,
                                    plot_restricted_skeleton = False,
                                    plot_restricted_skeleton_with_endnodes=False,
                                     
                                     
                                    angle_threshold = 30,
                                    top_volume_vector = np.array([0,-1,0]),
                                     
                                     spine_density_threshold = 0.00001,
                                    total_skeleton_distance_threshold_multiplier = 0.5,
                                    apical_width_threshold = 350,
                                    upward_distance_to_skeletal_distance_ratio_threshold = 0.85,
                                    
                                    verbose=False,
                                    **kwargs):
    """
    Purpose: To identify the branches on the limb that are most likely 
    part of a large upward apical branch
    
    
    Psuedoode:
    0a) Getting the subskeleton region to analyze
    0b) Divided the Restricted Skeleton into components to analyze
    
    For each connected component
    1) Get all the end nodes of the subgraph
    2) Subtract of the closest subgraph node to limb start
    For each end node
    3) Look at the vector between end nodes and closest node 
        (continue if not approximately straight up) and not long enough
    4) Find the branches that contain the two ends of the path

    For all combinations of branches:

    5) Find the shortest path between the two branches on the context network
    6) Get the subskeleton:
    - Analyze for width and spine density (and if too thin or not spiny enough then continue)
    7) If passed all tests then add the branch path as possible candidate
    
    """
    
    
    
    curr_limb = limb_obj
    apical_branches = []
    
    total_skeleton_distance_threshold = total_skeleton_distance_threshold_multiplier*(apical_check_distance_max - apical_check_distance_min)
    
    
    #0a) Getting the subskeleton region to analyze
    
    
    limb_gr = sk.convert_skeleton_to_graph(curr_limb.skeleton)
    st_node = xu.get_graph_node_by_coordinate(limb_gr,curr_limb.current_starting_coordinate)
    nodes_max_distance = xu.find_nodes_within_certain_distance_of_target_node(limb_gr,st_node,apical_check_distance_max)
    nodes_min_distance = xu.find_nodes_within_certain_distance_of_target_node(limb_gr,st_node,apical_check_distance_min)
    nodes_with_distance_range = np.setdiff1d(list(nodes_max_distance),list(nodes_min_distance))


    restricted_limb_gr = limb_gr.subgraph(nodes_with_distance_range)
    restricted_limb_sk = sk.convert_graph_to_skeleton(restricted_limb_gr)
    
    if plot_restricted_skeleton:
        nviz.plot_objects(curr_limb.mesh,
                         skeletons=[restricted_limb_sk])
        
        
    #0b) Divided the Restricted Skeleton into components to analyze
        
    conn_comp = list([np.array(list(k)) for k in nx.connected_components(restricted_limb_gr)])
    conn_comp_closest_nodes = [xu.shortest_path_between_two_sets_of_nodes(limb_gr,[st_node],k)[2]
                               for k in conn_comp]
    
    if plot_restricted_skeleton_with_endnodes:
        nviz.plot_objects(curr_limb.mesh,
                     skeletons=[restricted_limb_sk],
                     scatters=[xu.get_coordinate_by_graph_node(limb_gr,conn_comp_closest_nodes)],
                     scatter_size=1)
        
        
    
    for component_idx in range(len(conn_comp)):
        
        # 1) Get all the end nodes of the subgraph
        curr_cmpnt = conn_comp[component_idx]
        closest_node = conn_comp_closest_nodes[component_idx]
        closest_node_coordinate = xu.get_coordinate_by_graph_node(limb_gr,closest_node)

        c_subgraph = restricted_limb_gr.subgraph(curr_cmpnt)
        endnodes = xu.get_nodes_of_degree_k(c_subgraph,1)

        #2) Subtract of the closest subgraph node to limb start
        filtered_endnodes = np.setdiff1d(endnodes,closest_node)
        filtered_endnodes_coordinates = xu.get_coordinate_by_graph_node(limb_gr,filtered_endnodes)

        if verbose:
            print(f"Filered End nodes for component {component_idx}: {filtered_endnodes}")
            
        
        
        for e_node_idx in range(len(filtered_endnodes)):
            
            #3) Look at the vector between end nodes and closest node 
            e_node = filtered_endnodes[e_node_idx]
            e_node_coordinate = filtered_endnodes_coordinates[e_node_idx]

            # nviz.plot_objects(curr_limb.mesh,
            #                  skeletons=[restricted_limb_sk],
            #                  scatters=[xu.get_coordinate_by_graph_node(limb_gr,[closest_node,e_node])],
            #                  scatter_size=1)

            curr_vector = e_node_coordinate-closest_node_coordinate
            curr_vector_upward_distance = -curr_vector[1]
            curr_vector_len = np.linalg.norm(curr_vector)

            curr_vector_angle = nu.angle_between_vectors(top_volume_vector,curr_vector)

            if verbose:
                print(f"End Node Candidate {e_node_idx} angle = {np.round(curr_vector_angle,2)}"
                      f"\n    Upward distance {np.round(curr_vector_upward_distance,2)}")

            reject_flag = False
            if curr_vector_angle > angle_threshold:
                if verbose:
                    print(f"Rejecting candidate because did not pass angle threshold of ess than {angle_threshold}")
                continue
                
                
                
            #4) Find the branches that contain the two ends of the path
            curr_skeleton_path = sk.convert_graph_to_skeleton(limb_gr.subgraph(nx.shortest_path(limb_gr,closest_node,e_node)))
            curr_skeleton_path_len = sk.calculate_skeleton_distance(curr_skeleton_path)

            e_node_branches = nru.find_branch_with_specific_coordinate(curr_limb,e_node_coordinate)

            closest_node_branches =  nru.find_branch_with_specific_coordinate(curr_limb,closest_node_coordinate)

            #get all possible combinations
            all_branch_pairings = nu.unique_pairings_between_2_arrays(closest_node_branches,
                                                                      e_node_branches
                                                                     )
            if verbose:
                print(f"all_branch_pairings = {all_branch_pairings}")
                
                
                
            
            #for st_branch,end_branch in all_branch_pairings
            #5) Find the shortest path between the two branches on the context network

            for curr_pairing_idx  in range(len(all_branch_pairings)):

                st_branch = all_branch_pairings[curr_pairing_idx][0]
                end_branch = all_branch_pairings[curr_pairing_idx][1]

                try:
                    branch_path = nx.shortest_path(curr_limb.concept_network,st_branch,end_branch)
                except:
                    print(f"Couln't find path between branches")

                #6) Get the subskeleton:
                #- Analyze for width and spine density (and if too thin or not spiny enough then continue)

                #total_skeleton = sk.stack_skeletons([curr_limb[k].skeleton for k in branch_path])
                skeleton_distance_per_branch = np.array([sk.calculate_skeleton_distance(curr_limb[k].skeleton) for k in branch_path])
                branch_widths = np.array([curr_limb[k].width_new["median_mesh_center"] for k in branch_path])
                branch_spines = np.array([curr_limb[k].n_spines for k in branch_path])

                total_skeleton_distance = np.sum(skeleton_distance_per_branch)
                total_spine_density = np.sum(branch_spines)/np.sum(skeleton_distance_per_branch)
                scaled_branch_width = np.sum(skeleton_distance_per_branch*branch_widths)/(total_skeleton_distance)
                curr_vector_upward_distance
                upward_to_skeletal_length_ratio = curr_vector_upward_distance/curr_skeleton_path_len


                if verbose:
                    print(f"total_spine_density = {total_spine_density}")
                    print(f"scaled_branch_width = {scaled_branch_width}")
                    print(f"curr_skeleton_path_len = {curr_skeleton_path_len}")
                    print(f"curr_vector_upward_distance = {curr_vector_upward_distance}")
                    print(f"upward ratio to length = {upward_to_skeletal_length_ratio}")
                    
                # Apply the restrictions
                if ((total_spine_density > spine_density_threshold) and
                    (total_skeleton_distance > total_skeleton_distance_threshold) and 
                    (scaled_branch_width > apical_width_threshold) and
                    (upward_to_skeletal_length_ratio > upward_distance_to_skeletal_distance_ratio_threshold)):
                    
                    #print(f"Adding the following branch path as a apical pathway: {branch_path}")
                    apical_branches += list(branch_path)
                else:
                    print("Did not pass final filters to continuing")
                    continue
    
    return np.unique(apical_branches)
            
import proofreading_utils as pru
import time

def apical_classification(neuron_obj,
                          
                        skip_splitting=True,
                          
                          apical_soma_angle_threshold=40,
                          plot_viable_limbs = False,
                          label_neuron_branches=True,
                          plot_apical=True,
                          verbose=False,
                         **kwargs):
    """
    Will compute a limb branch dict of all 
    the branches that are part of a probably 
    long reaching apical branch
    
    Pseudocode: 
    1) Split the neuron and take the first neuron obj (assume only some in neuron)
    2) Check only 1 soma 
    3) Filter the limbs for viable aplical limbs based on the soma angle
    4) Iterate through the viable limbs to find the apical branches on each limb
    
    Ex:
    apical_classification(neuron_obj,
                          apical_soma_angle_threshold=40,
                          plot_viable_limbs = False,
                          label_neuron_branches=True,
                          plot_apical=True,
                          verbose=False)
    """
    
    split_time = time.time()
    
    if not skip_splitting:
        neuron_obj_list = pru.split_neuron(neuron_obj,
                                          plot_seperated_neurons=False,
                                          verbose=verbose)

        if verbose:
            print(f"Total time for split = {time.time() - split_time}")

        if len(neuron_obj_list)==0:
            raise Exception(f"Split Neurons not just one: {neuron_obj_list}")
            
        curr_neuron_obj = neuron_obj_list[0]
        
    
    else:
        curr_neuron_obj = neuron_obj
    
    
    viable_limbs = nru.viable_axon_limbs_by_starting_angle(curr_neuron_obj,
                                       soma_angle_threshold=apical_soma_angle_threshold,
                                       above_threshold=False,
                                       verbose=verbose)

    if verbose:
        print(f"viable_limbs = {viable_limbs}")
        
        
    if plot_viable_limbs:
        ret_col = nviz.visualize_neuron(curr_neuron_obj,
                     visualize_type=["mesh","skeleton"],
                     limb_branch_dict={f"L{k}":"all" for k in viable_limbs},
                     return_color_dict=True)
        
    
    apical_limb_branch_dict = dict()
    
    for limb_idx in viable_limbs:
        
        curr_limb = curr_neuron_obj[limb_idx]

        if verbose:
            print(f"Working on limb {limb_idx}")
        
        curr_limb_apical_branches = apical_branch_candidates_on_limb(curr_limb,
                                         verbose=verbose,
                                         **kwargs)
        if len(curr_limb_apical_branches) > 0:
            apical_limb_branch_dict.update({f"L{limb_idx}":curr_limb_apical_branches})
        
    
    if plot_apical:
        if len(apical_limb_branch_dict) > 0:
            nviz.visualize_neuron(curr_neuron_obj,
                                 visualize_type=["mesh"],
                                 limb_branch_dict=apical_limb_branch_dict,
                                 mesh_color="blue",
                                 mesh_whole_neuron=True,
                                 mesh_color_alpha=1)
        else:
            print("NO APICAL BRANCHES TO PLOT")
        
    if label_neuron_branches:
        nru.add_branch_label(curr_neuron_obj,
                    limb_branch_dict=apical_limb_branch_dict,
                    labels="apical")
        
    return apical_limb_branch_dict


        
        
# ---------- For inhibitory and excitatory classification ---------- #
def contains_excitatory_apical(neuron_obj,
                             plot_apical=False,
                               return_n_apicals=False,
                            **kwargs):
    apical_limb_branch_dict = clu.apical_classification(neuron_obj,
                                                    verbose=False,
                                                    plot_apical=plot_apical,
                                               **kwargs)
    if len(apical_limb_branch_dict)>0:
        apical_flag= True
    else:
        apical_flag= False
        
    if return_n_apicals:
        apical_conn_comp = nru.limb_branch_dict_to_connected_components(neuron_obj,
                                             limb_branch_dict=apical_limb_branch_dict,
                                            use_concept_network_directional=False)
        apical_flag = len(apical_conn_comp)
        
    return apical_flag
    
def contains_excitatory_axon(neuron_obj,
                             plot_axons=False,
                             return_axon_angles=True,
                             return_n_axons=False,
                            **kwargs):
    return_value = clu.axon_classification(neuron_obj,
                                                    return_error_labels=False,
                                                    verbose=False,
                                                    plot_axons=plot_axons,
                                                    label_axon_errors=False,
                                                    return_axon_angles=return_axon_angles,
                                               **kwargs)
    if return_axon_angles:
        axon_limb_branch_dict,axon_angles = return_value
    else:
        axon_limb_branch_dict = return_value
    
    
    if len(axon_limb_branch_dict)>0:
        axon_exist_flag =  True
    else:
        axon_exist_flag =  False
        
        
    if return_n_axons:
        axon_conn_comp = nru.limb_branch_dict_to_connected_components(neuron_obj,
                                             limb_branch_dict=axon_limb_branch_dict,
                                            use_concept_network_directional=False)
        axon_exist_flag = len(axon_conn_comp)
        
    if return_axon_angles:
        return axon_exist_flag,axon_angles
    else:
        return axon_exist_flag

def spine_level_classifier(neuron_obj,
                           sparsely_spiney_threshold = 0.0001,
                     spine_density_threshold = 0.0003,
                     #min_n_processed_branches=2,
                           min_processed_skeletal_length = 20000,
                           return_spine_statistics=False,
                    verbose=False,
                    **kwargs):
    """
    Purpose: To Calculate the spine density and use it to classify
    a neuron as one of the following categories based on the spine
    density of high interest branches
    
    1) no_spine
    2) sparsely_spine
    3) densely_spine
    
    """
        
    (neuron_spine_density, 
     n_branches_processed, skeletal_length_processed,
     n_branches_in_search_radius,skeletal_length_in_search_radius) = nru.neuron_spine_density(neuron_obj,
                        verbose=verbose,
                        plot_candidate_branches=False,
                        return_branch_processed_info=True,
                            **kwargs)
    
    if verbose:
        print(f"neuron_spine_density = {neuron_spine_density}")
        print(f"skeletal_length_processed = {skeletal_length_processed}")
        print(f"n_branches_processed = {n_branches_processed}")
        print(f"skeletal_length_in_search_radius = {skeletal_length_in_search_radius}")
    
#     if n_branches_processed < min_n_processed_branches:
#         final_label= "no_spined"

    if skeletal_length_processed < min_processed_skeletal_length or neuron_spine_density < sparsely_spiney_threshold:
        final_label="no_spined"
    else:
        if neuron_spine_density > spine_density_threshold:
            final_label= "densely_spined"
        else:
            final_label= "sparsely_spined"
            
    if return_spine_statistics:
        return (final_label,neuron_spine_density,
                n_branches_processed, skeletal_length_processed,
                n_branches_in_search_radius,skeletal_length_in_search_radius)
    else:
        return final_label
        
        
def inhibitory_excitatory_classifier(neuron_obj,
                                     verbose=True,
                                     return_spine_classification=False,
                                     return_axon_angles=False,
                                     return_n_axons=False,
                                     return_n_apicals=False,
                                     return_spine_statistics=False,
                                     axon_inhibitory_angle = 150,
                                     axon_inhibitory_width_threshold = 350,
                                    **kwargs):
    
    ret_value = clu.spine_level_classifier(neuron_obj,
                                                return_spine_statistics=return_spine_statistics,
                      **kwargs)
    
    if return_spine_statistics:
        (spine_category,neuron_spine_density,
        n_branches_processed, skeletal_length_processed,
                n_branches_in_search_radius,skeletal_length_in_search_radius) = ret_value
    else:
        spine_category = ret_value
    
    
    n_axons = None
    n_apicals=None
    axon_angles = None
    
    inh_exc_category = None
    
    if verbose:
        print(f"spine_category = {spine_category}")
        
    if spine_category == "no_spined":
        if verbose:
            print(f"spine_category was {spine_category} so determined as inhibitory")
        inh_exc_category = "inhibitory"
    elif spine_category == "densely_spined":
        inh_exc_category = "excitatory"
    else:
        n_apicals = clu.contains_excitatory_apical(neuron_obj,
                                                     return_n_apicals=return_n_apicals,
                                                     **kwargs)
        
        return_value = clu.contains_excitatory_axon(neuron_obj,
                                                 return_axon_angles=return_axon_angles,
                                                   return_n_axons=return_n_axons, 
                                                 **kwargs)
        if return_axon_angles:
            n_axons,axon_angles = return_value
        else:
            n_axons = return_value
        
        
        if verbose:
            print(f"n_apicals = {n_apicals}")
            print(f"n_axons = {n_axons}")
            print(f"axon_angles = {axon_angles}")
            
        if n_apicals==1 or n_axons>=1:
            #---------- 1/25 Addition: If have very bottom limb and not axon and above certain width 
                #--> means it is inhibitory
            
            inh_exc_category = "excitatory"
            
            if n_axons == 0:
                nullifying_limbs = nru.viable_axon_limbs_by_starting_angle(neuron_obj,
                                       soma_angle_threshold=axon_inhibitory_angle,
                                       above_threshold=True,
                                       verbose=True)
                
                for n_limb in nullifying_limbs:
                    
                    st_node = neuron_obj[n_limb].current_starting_node
                    st_node_width = neuron_obj[n_limb][st_node].width_new["no_spine_median_mesh_center"]
                    
                    if ( st_node_width>axon_inhibitory_width_threshold):
                        
                        print(f"Classifying as inhibitory because have large downshoot ({st_node_width}) that is not an axon")
                        inh_exc_category = "inhibitory"
                        break
         
        else:
            inh_exc_category = "inhibitory"
        
    
    if (return_axon_angles or return_n_axons) and n_axons is None:
        return_value = clu.contains_excitatory_axon(neuron_obj,
                                                 return_axon_angles=return_axon_angles,
                                                    return_n_axons=return_n_axons, 
                                                 **kwargs)
        if return_axon_angles:
            n_axons,axon_angles = return_value
        else:
            n_axons = return_value
            
    if return_n_apicals and n_apicals is None:
        n_apicals = clu.contains_excitatory_apical(neuron_obj,
                                                   return_n_apicals=return_n_apicals,
                                                     **kwargs)
        
    
    
    if (not return_spine_classification and not return_axon_angles 
        and not return_n_apicals and not return_n_axons):
        return inh_exc_category
    
    return_value = [inh_exc_category]
    
    if return_spine_classification:
        return_value.append(spine_category)
        
    if return_axon_angles:
        return_value.append(axon_angles)
        
    if return_n_axons:
        return_value.append(n_axons) 
    
    if return_n_apicals:
        return_value.append(n_apicals) 
        
    if return_spine_statistics:
        return_value += [neuron_spine_density, n_branches_processed, skeletal_length_processed,
                            n_branches_in_search_radius,skeletal_length_in_search_radius]
        
        
    return return_value
    
        
import classification_utils as clu
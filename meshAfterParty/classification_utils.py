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
        print(f"\nWorking on Limb {limb_idx}")

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
                                      verbose=False,
                                     ):
    """
    Purpose: To get the skeleton that represents the starting skeleton
    --> and then find the projection angle to filter it away or not

    Pseudocode: 
    1) convert the graph into a skeleton (this is when self touches could be a problem)
    2) Find all skeleton points that are within a certain distance of the starting coordinate
    3) Find all edn-degree nodes (except for the start)
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
    
    

    if verbose:
        print(f"endnodes_to_test = {endnodes_to_test}")
        
        
    # nviz.plot_objects(curr_limb.mesh,
    #             skeletons=[sk.convert_graph_to_skeleton(comparison_subgraph)],
    #                  )
    
    
    
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
    axon_angle_threshold_relaxed = 90,
    axon_angle_threshold = 120,
    relaxation_percentage = 0.85,
                          
    #parameters for computing the skeletal angle
     
    skeletal_angle_offset = 10000,
    skeletal_angle_comparison_distance = 10000,
    skeletal_angle_buffer = 5000,
                          
    axon_like_limb_branch_dict = None,
    verbose = False,
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

            candidate_angles,restr_skels = clu.candidate_starting_skeletal_angle(limb_obj=curr_limb,
                              candidate_nodes=candidate_nodes,
                                  offset = skeletal_angle_offset,
                                axon_sk_direction_comparison_distance = skeletal_angle_comparison_distance,
                                buffer_for_skeleton = skeletal_angle_buffer,
                                  top_volume_vector = np.array([0,-1,0]),
                                  plot_skeleton_paths_before_restriction=False,
                                  plot_skeleton_paths_after_restriction=False,
                                                 return_restricted_skeletons=True,
                                  verbose=verbose,
                                 )

            sk_passing_threshold = np.where(candidate_angles>curr_axon_angle_threshold)[0]

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




            # ----Part D: Add all of the candidates branches and those backtracking to mesh that are axon-like
            extra_nodes_to_add = np.intersect1d(axon_branches_on_limb,current_shortest_path[:-1])
            true_axon_branches = np.hstack([curr_candidate_subgraph,extra_nodes_to_add])

            if verbose:
                print(f"Adding the following branches as true axons: {true_axon_branches}")

            if curr_limb_idx not in axon_candidate_filtered:
                axon_candidate_filtered[curr_limb_idx] = dict()

            axon_candidate_filtered[curr_limb_idx][curr_candidate_idx] = true_axon_branches
    
    #compiling list into limb_branch dict that is easy to use
    limb_branch_dict = dict()
    
    for limb_idx, limb_info in axon_candidate_filtered.items():
        curr_branches = []
        for cand_idx,cand_list in limb_info.items():
            curr_branches.append(cand_list)
        limb_branch_dict[f"L{limb_idx}"] = np.concatenate(curr_branches)
    
    return limb_branch_dict


def axon_classification(neuron_obj,
                        
                        
    error_on_multi_soma = True,
    ais_threshold = 10000,
                        
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
                        
                        
    clean_prior_axon_labels=True,
    label_axon_errors=True,   
                        
    error_width_max = 140,
                        
    return_axon_labels=True,
                        
                        
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
    
    final_true_axons = clu.filter_axon_candiates(
    curr_neuron_obj,
    axon_subgraph_candidates,
    verbose = verbose,
    axon_like_limb_branch_dict=final_axon_like_classification,
    **kwargs
    )
    
    if verbose:
        print(f"\n\nPart 4: final_true_axons = {final_true_axons}")
    
    if plot_axons:
        nviz.visualize_neuron(curr_neuron_obj,
                              visualize_type=["mesh"],
                             limb_branch_dict=final_true_axons,
                             mesh_color="red",
                              mesh_color_alpha=1,
                             mesh_whole_neuron=True)
        
        
        
    #---------------------- Part 5: Adding Labels ----------------------
    """
    Pseudocode: 
    1) Clear the labels if option set
    2) Label all the true axon branches
    
    
    """
    
    if clean_prior_axon_labels:
        nru.clear_all_branch_labels(curr_neuron_obj,["axon","axon-like","axon-error"])
    
    nru.add_branch_label(curr_neuron_obj,
                    limb_branch_dict=final_axon_like_classification,
                    labels="axon-like")
    
    nru.add_branch_label(curr_neuron_obj,
                    limb_branch_dict=final_true_axons,
                    labels="axon")
        
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
    
    
    nru.add_branch_label(curr_neuron_obj,
                    limb_branch_dict=error_limb_branch_dict,
                    labels="axon-error")
    
    if plot_axon_errors:
        nviz.visualize_neuron(curr_neuron_obj,
                              visualize_type=["mesh"],
                             limb_branch_dict=error_limb_branch_dict,
                             mesh_color="red",
                              mesh_color_alpha=1,
                             mesh_whole_neuron=True)
    
    if return_axon_labels and return_error_labels:
        return final_true_axons,error_limb_branch_dict
    elif return_axon_labels:
        return final_true_axons
    elif return_error_labels:
        return error_limb_branch_dict
                                                    
        

import classification_utils as clu
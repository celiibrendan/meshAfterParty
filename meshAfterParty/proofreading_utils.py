import skeleton_utils as sk
import numpy as np
import networkx_utils as xu
import error_detection as ed
import neuron_utils as nru

import itertools
import networkx as nx
import copy

def find_high_degree_coordinates_on_path(limb_obj,curr_path_to_cut,
                                   degree_to_check=4):
    """
    Purpose: Find coordinates on a skeleton of the path speciifed (in terms of node ids)
    that are above the degree_to_check (in reference to the skeleton)
    
    
    """
    path_divergent_points = [sk.find_branch_endpoints(limb_obj[k].skeleton) for k in curr_path_to_cut]
    endpoint_coordinates = np.unique(np.concatenate(path_divergent_points),axis=0)

    limb_sk_gr = sk.convert_skeleton_to_graph(limb_obj.skeleton)
    endpoint_degrees = xu.get_coordinate_degree(limb_sk_gr,endpoint_coordinates)
    high_degree_endpoint_coordinates = endpoint_coordinates[endpoint_degrees>=degree_to_check]
    
    return high_degree_endpoint_coordinates

import skeleton_utils as sk
def get_best_cut_edge(curr_limb,
                      cut_path,
                      
                      remove_segment_threshold=1500,#the segments along path that should be combined
                      consider_path_neighbors_for_removal = True,
                      
                      #paraeters for high degree nodes
                      high_degree_offset = 2500,#1500,
                      comparison_distance = 2000,
                      match_threshold = 35,
                      
                      #parameter for both width and doubling back
                      # This will prevent the edges that were added to extend to the soma from causing the doulbing back or width threshold errors
                      skip_small_soma_connectors = True,
                      small_soma_connectors_skeletal_threshold = 2500,
                      
                      # parameters for the doubling back
                      double_back_threshold = 80,#100,# 130,
                      offset = 1000,
                      comparison_distance_double_back = 6000,
                      
                      #parameters for the width threshold
                      width_jump_threshold = 200,
                      verbose=True,
                      
                      high_degree_endpoint_coordinates_tried = [], #is a list of high degree end_nodes that have already been tried
                    **kwargs):
    """
    Purpose: To choose the best path to cut to disconnect
    a path based on the heuristic hierarchy of 
    
    Cut in descending priority
    1) high degree coordinates
    2) Doubling Back 
    3) Width Jump
    
    Pseudocode: 
    0) Combine close nodes if requested
    1) Get any high degree cordinates on path
    --> if there are then pick the first one and perform the cuts
    
    2) Check the doubling backs (and pick the highest one if above threshold)
    3) Check for width jumps (and pick the highest one)
    4) Record the cuts that will be made
    5) Make the alterations to the graph (can be adding and creating edges)
    """
    print("\n --------- START OF GET BEST EDGE --------- ")
    removed_branches=[]
    
    cut_path = np.array(cut_path)
    
    if (not remove_segment_threshold is None) and (remove_segment_threshold > 0):
        print(f"curr_limb.deleted_edges={curr_limb.deleted_edges}")
        
        path_without_ends = cut_path[1:-1]
        
        if consider_path_neighbors_for_removal:
            curr_neighbors = np.unique(np.concatenate([xu.get_neighbors(curr_limb.concept_network,k) for k in cut_path]))
            segments_to_consider = curr_neighbors[(curr_neighbors != cut_path[0]) & (curr_neighbors != cut_path[-1])]
            
            # ----------- 1/19: Have to remove the starting nodes from possible collapsing ------
            segments_to_consider = np.setdiff1d(segments_to_consider,curr_limb.all_starting_nodes)
            
            print(f"consider_path_neighbors_for_removal is set so segments_to_consider = {segments_to_consider}")
        else:
            segments_to_consider = np.array(path_without_ends)
        
        
        sk_lens = np.array([sk.calculate_skeleton_distance(curr_limb[k].skeleton) for k in segments_to_consider])
        short_segments = segments_to_consider[np.where(sk_lens<remove_segment_threshold)[0]]
        
        
        if verbose:
            print(f"Short segments to combine = {short_segments}")
            
        if len(short_segments)>0:
            if verbose:
                print("\n\n-------- Removing Segments -------------")
            curr_limb = pru.remove_branches_from_limb(curr_limb,
                                                 branch_list=short_segments,
                                                 plot_new_limb=False,
                                                verbose=verbose)
            print(f"curr_limb.deleted_edges 2={curr_limb.deleted_edges}")
            curr_limb.set_concept_network_directional(starting_node=cut_path[0],
                                                     suppress_disconnected_errors=True)
            print(f"curr_limb.deleted_edges 3={curr_limb.deleted_edges}")
            
            removed_branches = list(short_segments)
            
            for s in short_segments:
                cut_path = cut_path[cut_path != s]
            if verbose:
                print(f"Revised cut path = {cut_path}")
                print("\n-------- Done Removing Segments -------------\n\n")
                
                
            
            
        
    
    high_degree_endpoint_coordinates = find_high_degree_coordinates_on_path(curr_limb,cut_path)
    high_degree_endpoint_coordinates = nu.setdiff2d(high_degree_endpoint_coordinates,high_degree_endpoint_coordinates_tried)
    
    edges_to_create = None
    edges_to_delete = None
    
    resolve_crossover_at_end = True
    curr_high_degree_coord_list = [] #will store what high degree end-node was tried
    if verbose:
        print(f"Found {len(high_degree_endpoint_coordinates)} high degree coordinates to cut")
        
        
    # -------------- 1/27: Will organize the high degree nodes by the average width of the branches they are touching --- #
    
    if len(high_degree_endpoint_coordinates) > 0:
    
        sk_branches = [br.skeleton for br in curr_limb]

        high_degree_endpoint_coordinates_widths = []
        for coordinate in high_degree_endpoint_coordinates:
            coordinate_branches = np.sort(sk.find_branch_skeleton_with_specific_coordinate(sk_branches,coordinate))
            if len(coordinate_branches) > 0:
                mean_width = np.mean([curr_limb[b].width_new["no_spine_median_mesh_center"] for b in coordinate_branches])
            else:
                mean_width = np.inf

            high_degree_endpoint_coordinates_widths.append(mean_width)

        high_degree_order = np.argsort(high_degree_endpoint_coordinates_widths)
        
        if verbose:
            print(f"high_degree_endpoint_coordinates_widths = {high_degree_endpoint_coordinates_widths}")
            print(f"high_degree_order = {high_degree_order}")


        #for curr_high_degree_coord in high_degree_endpoint_coordinates:
        for curr_high_degree_coord_idx in high_degree_order:
            curr_high_degree_coord = high_degree_endpoint_coordinates[curr_high_degree_coord_idx]
            
            curr_high_degree_coord_list.append(curr_high_degree_coord)

            if verbose:
                print(f"Picking {curr_high_degree_coord} high degree coordinates to cut")
                print(f"curr_limb.deleted_edges 4={curr_limb.deleted_edges}")
            edges_to_delete_pre,edges_to_create_pre = ed.resolving_crossovers(curr_limb,
                                                    coordinate = curr_high_degree_coord,
                                                    offset=high_degree_offset,
                                                    comparison_distance=comparison_distance,
                                                                              match_threshold =match_threshold,
                                                    verbose = verbose,
                                                    **kwargs
                                   )
            print(f"curr_limb.deleted_edges 5 ={curr_limb.deleted_edges}")

            #  --- 1/19: Add in a check that will see if divides up the path, if not then set to empty  --- #

            if len(edges_to_delete_pre)>0:

                cut_path_edges = np.vstack([cut_path[:-1],cut_path[1:]]).T
                
                
                G = nx.from_edgelist(cut_path_edges)
                print(f"nx.number_connected_components(G) before = {nx.number_connected_components(G)}")
                G.remove_edges_from(edges_to_delete_pre)
                
                print(f"G.edges() = {G.edges()}")
                print(f"G.nodes() = {G.nodes()}")
                print(f"nx.number_connected_components(G) after = {nx.number_connected_components(G)}")

                if (len(G.nodes()) < len(cut_path)) or nx.number_connected_components(G)>1:
                    print("Using the resolve crossover delete edges because will help seperate the path")
                    edges_to_delete = edges_to_delete_pre
                    edges_to_create = edges_to_create_pre
                    resolve_crossover_at_end = False
                    break

                else:
                    print("NOT USING the resolve crossover delete edges because not help resolve the cut")
                    continue
                
            
        
    
        
    
    
    
    curr_limb.set_concept_network_directional(starting_node = cut_path[0],suppress_disconnected_errors=True)
    
    # ------------- 12 /28 addition that allows us to skip end nodes if too small ------------------
    skip_nodes = []
    if skip_small_soma_connectors:
        revised_cut_path = np.array(cut_path)
        for endnode in [cut_path[0],cut_path[-1]]:
            curr_sk_distance = sk.calculate_skeleton_distance(curr_limb[endnode].skeleton)
            if curr_sk_distance<small_soma_connectors_skeletal_threshold:
                print(f"Skipping endnode {endnode} because skeletal distance was {curr_sk_distance} and threshold was {small_soma_connectors_skeletal_threshold}")
                revised_cut_path = revised_cut_path[revised_cut_path != endnode]
                skip_nodes.append(endnode)
                
                
        if len(revised_cut_path) <2 :
            print("Could not used the revised endnodes path because empty")
            skip_nodes = []
            
    if verbose:
        print(f"skip_nodes = {skip_nodes}")
        
    if edges_to_delete is None:
        if verbose: 
            print("\nAttempting the doubling back check (symmetrical so don't need to check from both sides)")
            
        err_edges,edges,edges_double_back = ed.double_back_edges_path(curr_limb,
                                path_to_check=cut_path,
                              double_back_threshold = double_back_threshold,
                              comparison_distance = comparison_distance_double_back,
                             offset=offset,
                            verbose = verbose,
                            skip_nodes=skip_nodes)


        if len(err_edges) > 0:

            largest_double_back = np.argmax(edges_double_back)
            winning_err_edge = edges[largest_double_back]

            if verbose:
                print(f"There were {len(err_edges)} edges that passed doubling back threshold of {double_back_threshold}")
                print(f"Winning edge {winning_err_edge} had a doubling back of {edges_double_back[largest_double_back]}")
                
            edges_to_delete = [winning_err_edge]

    if edges_to_delete is None:
        if verbose: 
            print("\nAttempting the width jump check (ARTIFICIALLY ATTEMPTING FROM BOTH SIDES)")
            print(f"width_jump_threshold = {width_jump_threshold}")
            
        possible_starting_nodes = [cut_path[0],cut_path[-1]]
        
        
        first_error_edges = []
        first_error_sizes = []
        first_error_locations = []
        
        """  Old way of doing 
        for s_node in possible_starting_nodes:
            
            curr_limb.set_concept_network_directional(starting_node = s_node,suppress_disconnected_errors=True)
            
            if cut_path[0] != s_node:
                cut_path = np.flip(cut_path)
                if cut_path[0] != s_node:
                    raise Exception("Neither of cut path end nodes are starting node")
                    
            err_edges,edges,edges_width_jump = ed.width_jump_edges_path(curr_limb,
                                                                        path_to_check=cut_path,
                                                                        width_jump_threshold=width_jump_threshold,
                                                                        offset=offset,
                                                                        verbose=verbose,
                                                                        skip_nodes=skip_nodes
                                    )
            
            if verbose:
                print(f"Path starting at {s_node} had err_edges: {err_edges}")
            
            err_edges_mask = edges_width_jump>=width_jump_threshold
            
            if np.any(err_edges_mask):
                first_error_edges.append(edges[err_edges_mask][0])
                first_error_sizes.append(edges_width_jump[err_edges_mask][0])
            else:
                first_error_edges.append(None)
                first_error_sizes.append(-np.inf)
        """
        
        # ----------- 1/19: New Way of Splitting ------------- #
        curr_limb.set_concept_network_directional(starting_node = cut_path[0],suppress_disconnected_errors=True)
            


        err_edges,edges,edges_width_jump = ed.width_jump_edges_path(curr_limb,
                                                                    path_to_check=cut_path,
                                                                    width_jump_threshold=width_jump_threshold,
                                                                    offset=offset,
                                                                    verbose=verbose,
                                                                    skip_nodes=skip_nodes
                                )

        # 1) Doing the forwards way

        err_edges_mask = np.where(edges_width_jump>=width_jump_threshold)[0]

        if np.any(err_edges_mask):
            first_error_edges.append(edges[err_edges_mask[0]])
            first_error_sizes.append(edges_width_jump[err_edges_mask][0])
            first_error_locations.append(err_edges_mask[0])
        else:
            first_error_edges.append(None)
            first_error_sizes.append(-np.inf)
            first_error_locations.append(np.inf)
            
        # 2) Doing the backwards way
        
        edges_width_jump_flipped = np.flip(edges_width_jump)
        edges_flipped = np.flip(edges)
        
        err_edges_mask = np.where(edges_width_jump_flipped <=-width_jump_threshold)[0]

        if np.any(err_edges_mask):
            first_error_edges.append(edges_flipped[err_edges_mask[0]])
            first_error_sizes.append(-1*(edges_width_jump_flipped[err_edges_mask[0]]))
            first_error_locations.append(err_edges_mask[0])
        else:
            first_error_edges.append(None)
            first_error_sizes.append(-np.inf)
            first_error_locations.append(np.inf)
        
        """
        Pseudocode: 
        1) Check if both error edges are not empty
        2) Get the starting error 
        
        
        """
        if (not first_error_edges[0] is None) or (not first_error_edges[1] is None):
            """  Old way that did it on the biggest size, new way does it on the first jump
            winning_path = np.argmax(first_error_sizes)
            """
            
            winning_path = np.argmin(first_error_locations)
            winning_err_edge = first_error_edges[winning_path]
            if verbose: 
                print(f"first_error_sizes = {first_error_sizes}, first_error_locations = {first_error_locations}, winning_path = {winning_path}")
                
                
            edges_to_delete = [winning_err_edge]
        else:
            if verbose:
                print(f"Did not find an error edge in either of the paths")
                
                
    # need to resolve cross over at this point
    if resolve_crossover_at_end and (not edges_to_delete is None):
        cut_e = edges_to_delete[0]
        suggested_cut_point = sk.shared_endpoint(curr_limb[cut_e[0]].skeleton,
                                                curr_limb[cut_e[1]].skeleton)
                                             
        edges_to_delete_new,edges_to_create_new = ed.resolving_crossovers(curr_limb,
                       coordinate=suggested_cut_point,
                       return_subgraph=False)

        if len(edges_to_delete_new) > 0:
            edges_to_delete = np.vstack([edges_to_delete,edges_to_delete_new])
            edges_to_delete = list(np.unique(np.sort(np.array(edges_to_delete),axis=1),axis=0))
            
        
        
        if not edges_to_create is None:
            edges_to_create += edges_to_create_new
            edges_to_create = list(np.unique(np.sort(np.array(edges_to_create),axis=1),axis=0))
        else:
            edges_to_create = edges_to_create_new
        
        # want to limit the edges to only those with one of the disconnected edges in it
        edges_to_create_final = []
        for e_c1 in edges_to_create:
            if len(np.intersect1d(e_c1,cut_e)) == 1:
                edges_to_create_final.append(e_c1)
            else:
                if verbose:
                    print(f"Rejecting creating edge {e_c1} becuase did not involve only 1 node in the deleted edge")
        edges_to_create = edges_to_create_final
                

    
        
    curr_limb,edges_to_create_final = pru.cut_limb_network_by_edges(curr_limb,
                                                    edges_to_delete,
                                                    edges_to_create,
                                                    return_accepted_edges_to_create=True,
                                                    verbose=verbose)
    
    edges_to_create=edges_to_create_final
    
        

    if verbose:
        conn_comp = list(nx.connected_components(curr_limb.concept_network))
        print(f"Number of connected components = {len(conn_comp)}")
        for j,k in enumerate(conn_comp):
            print(f"Comp {j} = {k}")
    if nu.is_array_like(edges_to_delete):
        edges_to_delete = list(edges_to_delete)
    if nu.is_array_like(edges_to_create):
        edges_to_create = list(edges_to_create)
    if nu.is_array_like(removed_branches):
        removed_branches = list(removed_branches)
    return edges_to_delete,edges_to_create,curr_limb,removed_branches,curr_high_degree_coord_list

def get_all_coordinate_suggestions(suggestions,concatenate=True,
                                  voxel_adjustment=True):
    """
    Getting all the coordinates where there should be cuts
    
    
    """
    if len(suggestions) == 0:
        return []
    
    all_coord = []
    for limb_idx,sugg_v in suggestions.items():
        curr_coords = np.array(get_attribute_from_suggestion(suggestions,curr_limb_idx=limb_idx,
                                 attribute_name="coordinate_suggestions"))
        if voxel_adjustment and len(curr_coords)>0:
            curr_coords = curr_coords/np.array([4,4,40])
                
        if len(curr_coords) > 0:
            all_coord.append(curr_coords)
    
    if voxel_adjustment:
        all_coord
    if concatenate:
        if len(all_coord) > 0:
            return list(np.unique(np.vstack(all_coord),axis=0))
        else:
            return list(all_coord)
    else:
        return list(all_coord)
    
    
def get_n_paths_not_cut(limb_results):
    """
    Get all of the coordinates on the paths that will be cut
    
    
    """
    if len(limb_results) == 0:
        return 0
    
    n_paths_not_cut = 0

    for limb_idx, limb_data in limb_results.items():
        for path_cut_info in limb_data:
            if len(path_cut_info["paths_not_cut"]) > 0:
                n_paths_not_cut += 1   
                
    return n_paths_not_cut

def get_n_paths_cut(limb_results):
    """
    Get all of the coordinates on the paths that will be cut
    
    
    """
    if len(limb_results) == 0:
        return 0
    
    n_paths_not_cut = 0

    for limb_idx, limb_data in limb_results.items():
        for path_cut_info in limb_data:
            if len(path_cut_info["paths_cut"]) > 0:
                n_paths_not_cut += 1   
                
    return n_paths_not_cut

def get_all_cut_and_not_cut_path_coordinates(limb_results,voxel_adjustment=True,
                                            ):
    """
    Get all of the coordinates on the paths that will be cut
    
    
    """
    if len(limb_results) == 0:
        return [],[]
    
    cut_path_coordinates = []
    not_cut_path_coordinates = []

    for limb_idx, limb_data in limb_results.items():
        for path_cut_info in limb_data:
            if len(path_cut_info["paths_cut"]) > 0:
                cut_path_coordinates.append(np.vstack(path_cut_info["paths_cut"]))
            if len(path_cut_info["paths_not_cut"]) > 0:
                not_cut_path_coordinates.append(np.vstack(path_cut_info["paths_not_cut"]))

    if len(cut_path_coordinates)>0:
        total_cut_path_coordinates = np.vstack(cut_path_coordinates)
    else:
        total_cut_path_coordinates = []

    if len(not_cut_path_coordinates)>0:
        total_not_cut_path_coordinates = list(np.vstack(not_cut_path_coordinates))
    else:
        total_not_cut_path_coordinates = []


    if len(total_not_cut_path_coordinates)>0 and len(total_cut_path_coordinates)>0:
        total_cut_path_coordinates_revised = list(nu.setdiff2d(total_cut_path_coordinates,total_not_cut_path_coordinates))
    else:
        total_cut_path_coordinates_revised = list(total_cut_path_coordinates)
        
    if voxel_adjustment:
        voxel_divider = np.array([4,4,40])
        total_cut_path_coordinates_revised = [k/voxel_divider for k in total_cut_path_coordinates_revised if len(k)>0]
        total_not_cut_path_coordinates = [k/voxel_divider for k in total_not_cut_path_coordinates if len(k)>0]
        
    
        
    return list(total_cut_path_coordinates_revised),list(total_not_cut_path_coordinates)

    
    
def get_attribute_from_suggestion(suggestions,curr_limb_idx=None,
                                 attribute_name="edges_to_delete"):
    if type(suggestions) == dict:
        if curr_limb_idx is None:
            raise Exception("No specified limb idx when passed all the suggestions")
        suggestions = suggestions[curr_limb_idx]
        
    total_attribute = []
    for cut_s in suggestions:
        total_attribute += cut_s[attribute_name]
        
    return total_attribute

def get_edges_to_delete_from_suggestion(suggestions,curr_limb_idx=None):
    return get_attribute_from_suggestion(suggestions,curr_limb_idx,
                                 attribute_name="edges_to_delete")
def get_edges_to_create_from_suggestion(suggestions,curr_limb_idx=None):
    return get_attribute_from_suggestion(suggestions,curr_limb_idx,
                                 attribute_name="edges_to_create")
def get_removed_branches_from_suggestion(suggestions,curr_limb_idx=None):
    return get_attribute_from_suggestion(suggestions,curr_limb_idx,
                                 attribute_name="removed_branches")
    

def cut_limb_network_by_suggestions(curr_limb,
                                   suggestions,
                                   curr_limb_idx=None,
                                    return_copy=True,
                                   verbose=False):
    if type(suggestions) == dict:
        if curr_limb_idx is None:
            raise Exception("No specified limb idx when passed all the suggestions")
        suggestions = suggestions[curr_limb_idx]
    
    return cut_limb_network_by_edges(curr_limb,
                                    edges_to_delete=pru.get_edges_to_delete_from_suggestion(suggestions),
                                    edges_to_create=pru.get_edges_to_create_from_suggestion(suggestions),
                                     removed_branches=pru.get_removed_branches_from_suggestion(suggestions),
                                    verbose=verbose,
                                     return_copy=return_copy
                                    )
    
def cut_limb_network_by_edges(curr_limb,
                                    edges_to_delete=None,
                                    edges_to_create=None,
                                    removed_branches=[],
                                    perform_edge_rejection=False,
                                    return_accepted_edges_to_create=False,
                                    return_copy=True,
                                    verbose=False):
    if return_copy:
        curr_limb = copy.copy(curr_limb)
    if (not removed_branches is None) and len(removed_branches)>0:
        curr_limb = pru.remove_branches_from_limb(curr_limb,
                                                 branch_list=removed_branches)
    
    if not edges_to_delete is None:
        if verbose:
            print(f"edges_to_delete (cut_limb_network) = {edges_to_delete}")
        curr_limb.concept_network.remove_edges_from(edges_to_delete)
        
        curr_limb.deleted_edges += edges_to_delete
        
    #apply the winning cut
    accepted_edges_to_create = []
    if not edges_to_create is None:
        if verbose:
            print(f"edges_to_create = {edges_to_create}")
            
        if perform_edge_rejection:
            for n1,n2 in edges_to_create:
                curr_limb.concept_network.add_edge(n1,n2)
                counter = 0
                for d1,d2 in edges_to_delete:
                    try:
                        ex_path = np.array(nx.shortest_path(curr_limb.concept_network,d1,d2))
                    except:
                        pass
                    else:
                        counter += 1
                        break
                if counter > 0:
                    curr_limb.concept_network.remove_edge(n1,n2)
                    if verbose:
                        print(f"Rejected edge ({n1,n2})")
                else:
                    if verbose:
                        print(f"Accepted edge ({n1,n2})")
                    accepted_edges_to_create.append([n1,n2])
        else:
            accepted_edges_to_create = edges_to_create
            curr_limb.concept_network.add_edges_from(accepted_edges_to_create)

        #add them to the properties
        curr_limb.created_edges += accepted_edges_to_create
    
    if return_accepted_edges_to_create:
        return curr_limb,accepted_edges_to_create
    
    return curr_limb

def multi_soma_split_suggestions(neuron_obj,
                                verbose=True,
                                max_iterations=100,
                                 plot_suggestions=False,
                                 plot_suggestions_scatter_size=0.4,
                                 remove_segment_threshold = 1500,
                                 plot_cut_coordinates = False,
                                **kwargs):
    """
    Purpose: To come up with suggestions for splitting a multi-soma

    Pseudocode: 

    1) Iterate through all of the limbs that need to be processed
    2) Find the suggested cuts until somas are disconnected or failed
    3) Optional: Visualize the nodes and their disconnections

    """
    
    #sprint("inside multi_soma_split_suggestions")

    multi_soma_limbs = nru.multi_soma_touching_limbs(neuron_obj)
    multi_touch_limbs = nru.same_soma_multi_touching_limbs(neuron_obj)
    
    if verbose: 
        print(f"multi_soma_limbs = {multi_soma_limbs}")
        print(f"multi_touch_limbs = {multi_touch_limbs}")
    
    total_limbs_to_process = np.unique(np.concatenate([multi_soma_limbs,multi_touch_limbs]))

    limb_results = dict()

    for curr_limb_idx in total_limbs_to_process:
        curr_limb_idx = int(curr_limb_idx)
        if verbose:
            print(f"\n\n -------- Working on limb {curr_limb_idx}------------")
        curr_limb_copy = copy.deepcopy(neuron_obj[curr_limb_idx])

        #----- starting the path cutting ------ #
        """
        Find path to cut:
        1) Get the concept network
        2) Get all of the starting nodes for somas
        3) Get the shortest path between each combination of starting nodes

        """
        max_iterations = 10

        #2) Get all of the starting nodes for somas
        all_starting_nodes = [k["starting_node"] for k in curr_limb_copy.all_concept_network_data]

        starting_node_combinations = list(itertools.combinations(all_starting_nodes,2))
        starting_node_combinations = nu.unique_non_self_pairings(starting_node_combinations)
        

        if verbose:
            print(f"Starting combinations to process = {starting_node_combinations}")

        results = []

        for st_n_1,st_n_2 in starting_node_combinations:
            local_results = dict(starting_node_1=st_n_1,
                                starting_node_2 = st_n_2)
            st_n_1_soma,st_n_1_soma_group_idx = curr_limb_copy.get_soma_by_starting_node(st_n_1),curr_limb_copy.get_soma_group_by_starting_node(st_n_1)
            st_n_2_soma,st_n_2_soma_group_idx = curr_limb_copy.get_soma_by_starting_node(st_n_2),curr_limb_copy.get_soma_group_by_starting_node(st_n_2)

            soma_title = f"S{st_n_1_soma}_{st_n_1_soma_group_idx} from S{st_n_2_soma}_{st_n_2_soma_group_idx} "
            local_results["title"] = soma_title

            total_soma_paths_to_cut = []
            total_soma_paths_to_add = []
            total_removed_branches = []
            # need to keep cutting until no path for them
            if verbose:
                print(f"\n\n---- working on disconnecting {st_n_1} and {st_n_2}")
                print(f"---- This disconnects {soma_title} ")

            counter = 0
            success = False
            
            high_degree_endpoint_coordinates_tried = []
            local_paths_cut = []
            local_paths_not_cut = []
            
            while True:
                if verbose:
                    print(f" Cut iteration {counter}")
                seperated_graphs = list(nx.connected_components(curr_limb_copy.concept_network))
                if verbose:
                    print(f"Total number of graphs at the end of the split BEFORE DIRECTIONAL = {len(seperated_graphs)}")
                    
                    
                curr_limb_copy.set_concept_network_directional(starting_node=st_n_1,suppress_disconnected_errors=True)
                
                
                seperated_graphs = list(nx.connected_components(curr_limb_copy.concept_network))
                if verbose:
                    print(f"Total number of graphs at the end of the split AFTER DIRECTIONAL = {len(seperated_graphs)}")
                try:
                    soma_to_soma_path = np.array(nx.shortest_path(curr_limb_copy.concept_network,st_n_1,st_n_2))
                    
                    
                    
                    
                except:
                    if verbose:
                        print("No valid path so moving onto the next connection")
                    
                    success = True
                    break
                
                #try and figure out the endpoints along the path
                local_paths_cut.append(nru.skeleton_points_along_path(curr_limb_copy,soma_to_soma_path))
                
                if verbose:
                    print(f"Shortest path = {list(soma_to_soma_path)}")

                # say we found the cut node to make
                
                (cut_edges, 
                added_edges, 
                curr_limb_copy,
                removed_branches,
                curr_high_degree_coord) = pru.get_best_cut_edge(curr_limb_copy,soma_to_soma_path,
                                                                remove_segment_threshold=remove_segment_threshold,
                                                                               verbose=verbose,
                                      high_degree_endpoint_coordinates_tried=high_degree_endpoint_coordinates_tried,
                                                                              **kwargs)
                
                
                high_degree_endpoint_coordinates_tried += curr_high_degree_coord
                
                print(f"curr_limb_copy.deleted_edges = {curr_limb_copy.deleted_edges}")
                print(f"curr_limb_copy.created_edges = {curr_limb_copy.created_edges}")
                
                if verbose:
                    print(f"After get best cut: cut_edges = {cut_edges}, added_edges = {added_edges}")
                
                
                if cut_edges is None:
                    print("***** there was no suggested cut for this limb even though it is still connnected***")

                    break
                
                
                
                #------ 1/8 Addition: check if any new edges cut and if not then break
                if len(nu.intersect2d(cut_edges,total_soma_paths_to_cut)) == len(cut_edges):
                    print("**** there were no NEW suggested cuts")
                    break
                
                print(f"total_soma_paths_to_cut = {total_soma_paths_to_cut}")
                if not cut_edges is None:
                    
                    if plot_cut_coordinates:
                        suggested_cut_points = []
                        for cut_e in cut_edges:
                            high_degree_endpoint_coordinates_tried
                            shared_endpoints = sk.shared_endpoint(curr_limb_copy[cut_e[0]].skeleton,
                                                            curr_limb_copy[cut_e[1]].skeleton,
                                                                 return_possibly_two=True)
                            if shared_endpoints.ndim == 1:
                                suggested_cut_points.append(shared_endpoints)
                            else:
                                for s_e in shared_endpoints:
                                    if len(nu.matching_rows(high_degree_endpoint_coordinates_tried,s_e)) > 0:
                                        suggested_cut_points.append(s_e)
                                        
                        suggested_cut_points = np.unique(np.array(suggested_cut_points).reshape(-1,3),axis=0)
                        
                        path_skeleton_points = nru.skeleton_points_along_path(curr_limb_copy,
                                                                              skeletal_distance_per_coordinate=3000,
                                                                             branch_path=soma_to_soma_path)
                        print(f"\n\nsuggested_cut_points = {suggested_cut_points}\n\n")
                        nviz.plot_objects(curr_limb_copy.mesh,
                                          skeletons=[curr_limb_copy.skeleton],
                                          scatters = [path_skeleton_points,suggested_cut_points],
                                          scatters_colors=["blue","red"],
                                          scatter_size = 0.3
                        
                        )
                        
                    
                    
                    total_soma_paths_to_cut += cut_edges
                if not added_edges is None:
                    total_soma_paths_to_add += added_edges
                if len(removed_branches)>0:
                    total_removed_branches += removed_branches
                
                print(f"-----------counter = {counter}------------")
                counter += 1

                if counter > max_iterations:
                    print(f"Breaking because hit max iterations {max_iterations}")
                    
                    break

            if not success:
                local_paths_not_cut.append(nru.skeleton_points_along_path(curr_limb_copy,soma_to_soma_path))
                   
            local_results["edges_to_delete"] = total_soma_paths_to_cut
            local_results["edges_to_create"] = total_soma_paths_to_add
            local_results["removed_branches"] = total_removed_branches

            suggested_cut_points = []
            for cut_e in total_soma_paths_to_cut:
                high_degree_endpoint_coordinates_tried
                shared_endpoints = sk.shared_endpoint(curr_limb_copy[cut_e[0]].skeleton,
                                                curr_limb_copy[cut_e[1]].skeleton,
                                                     return_possibly_two=True)
                if shared_endpoints.ndim == 1:
                    suggested_cut_points.append(shared_endpoints)
                else:
                    for s_e in shared_endpoints:
                        if len(nu.matching_rows(high_degree_endpoint_coordinates_tried,s_e)) > 0:
                            suggested_cut_points.append(s_e)

            local_results["coordinate_suggestions"] =suggested_cut_points
            local_results["successful_disconnection"] = success
            local_results["paths_not_cut"] = local_paths_not_cut
            local_results["paths_cut"] = local_paths_cut
            results.append(local_results)

        seperated_graphs = list(nx.connected_components(curr_limb_copy.concept_network))
        if verbose:
            print(f"Total number of graphs at the end of the split = {len(seperated_graphs)}: {[np.array(list(k)) for k in seperated_graphs]}")
            

        limb_results[curr_limb_idx] = results
        
    if plot_suggestions:
        nviz.plot_split_suggestions_per_limb(neuron_obj,
                                    limb_results,
                                    scatter_size = plot_suggestions_scatter_size)
        
    return limb_results

def split_suggestions_to_concept_networks(neuron_obj,limb_results,
                                         apply_changes_to_limbs=False):
    """
    Will take the output of the multi_soma_split suggestions and 
    return the concept network with all fo the cuts applied
    
    """
    
    new_concept_networks = dict()
    for curr_limb_idx,path_cut_info in limb_results.items():
        if not apply_changes_to_limbs:
            limb_cp = copy.deepcopy(neuron_obj[curr_limb_idx])
        else:
            limb_cp = neuron_obj[curr_limb_idx]
        #limb_nx = nx.Graph(neuron_obj[curr_limb_idx].concept_network)
        for cut in path_cut_info:
            limb_cp=pru.remove_branches_from_limb(limb_cp,branch_list=cut["removed_branches"])
            limb_cp.concept_network.remove_edges_from(cut["edges_to_delete"])
            limb_cp.concept_network.add_edges_from(cut["edges_to_create"])
        new_concept_networks[curr_limb_idx] = limb_cp.concept_network
    return new_concept_networks


# --------------- Functions that do the actual limb and Neuron Splitting --------- #

import numpy as np
import preprocessing_vp2 as pre
import networkx_utils as xu
import trimesh_utils as tu
import neuron_utils as nru
import neuron
import copy
import networkx as nx

def split_neuron_limb_by_seperated_network(neuron_obj,
                     curr_limb_idx,
                    seperate_networks=None,
                    error_on_multile_starting_nodes=True,
                    delete_limb_if_empty=True,
                     verbose = True):
    """
    Purpose: To Split a neuron limb up into sepearte limb graphs specific

    Arguments:
    neuron_obj
    seperated_graphs
    limb_idx


    """
    
    
    # -------- Getting the mesh and correspondence information --------- #
    """
    1) Assemble all the faces of the nodes and concatenate them
    - copy the data into the new limb correspondence
    - save the order they were concatenated in the new limb correspondence
    - copy of 
    2) Use the concatenated faces idx to obtain the new limb mesh
    3) index the concatenated faces idx into the limb.mesh_face_idx to get the neew limb.mesh_face_idx
    """
    if seperate_networks is None:
        seperated_graphs = [neuron_obj[curr_limb_idx].concept_network]
    else:
        seperated_graphs = seperate_networks
        
        
    new_limb_data = []
    curr_limb = neuron_obj[curr_limb_idx]
    
    if len(seperated_graphs) == 0:
        curr_labels = ["empty"]
        sep_graph_data["Limb_obj"] = nru.empty_limb_object(labels=curr_labels)
        sep_graph_data["limb_labels"] = curr_labels
        sep_graph_data["limb_concept_networks"] = dict()
        sep_graph_data["limb_network_stating_info"]=dict()
        sep_graph_data["limb_meshes"] = tu.empty_mesh()
        new_limb_data.append(sep_graph_data)
    else:
        for seg_graph_idx,sep_G in enumerate(seperated_graphs):

            curr_subgraph = list(sep_G)

            #will store all of the relevant info in the 
            sep_graph_data = dict()

            if len(curr_subgraph) == 0:
                curr_labels = ["empty"]
                sep_graph_data["Limb_obj"] = nru.empty_limb_object(labels=curr_labels)
                sep_graph_data["limb_correspondence"] = dict()
                sep_graph_data["limb_labels"] = curr_labels
                sep_graph_data["limb_concept_networks"] = dict()
                sep_graph_data["limb_network_stating_info"]=dict()
                sep_graph_data["limb_meshes"] = tu.empty_mesh()
                new_limb_data.append(sep_graph_data)
                continue


            if verbose:
                print(f"\n\n----Working on seperate_graph {seg_graph_idx}----")




            fixed_node_objects = dict()

            limb_face_idx_concat = []
            face_counter = 0
            old_node_to_new_node_mapping = dict()
            for i,n_name in enumerate(curr_subgraph):
                #store the mapping for the new names
                old_node_to_new_node_mapping[n_name] = i

                fixed_node_objects[i] = copy.deepcopy(curr_limb[n_name])
                curr_mesh_face_idx = fixed_node_objects[i].mesh_face_idx
                limb_face_idx_concat.append(curr_mesh_face_idx)
                fixed_node_objects[i].mesh_face_idx = np.arange(face_counter,face_counter+len(curr_mesh_face_idx))
                face_counter += len(curr_mesh_face_idx)


            total_limb_face_idx = np.concatenate(limb_face_idx_concat)
            new_limb_mesh = curr_limb.mesh.submesh([total_limb_face_idx],append=True,repair=False)


            new_limb_mesh_face_idx = tu.original_mesh_faces_map(neuron_obj.mesh, new_limb_mesh,
                                       matching=True,
                                       print_flag=False)

            #recovered_new_limb_mesh = neuron_obj.mesh.submesh([new_limb_mesh_face_idx],append=True,repair=False)
            sep_graph_data["limb_meshes"] = new_limb_mesh

            # ------- How to get the new concept network starting info --------- #

            #get all of the starting dictionaries that match a node in the subgraph
            curr_all_concept_network_data = [k for k in curr_limb.all_concept_network_data if k["starting_node"] in list(curr_subgraph)]
            if len(curr_all_concept_network_data) > 1:
                warning_string = f"There were more not exactly one starting dictinoary: {curr_all_concept_network_data} "
                if error_on_multile_starting_nodes:
                    raise Exception(warning_string)
                else:
                    print(warning_string)


            limb_corresp_for_networks = dict([(i,dict(branch_skeleton=k.skeleton,
                                                     width_from_skeleton=k.width,
                                                     branch_mesh=k.mesh,
                                                     branch_face_idx=k.mesh_face_idx)) for i,k in fixed_node_objects.items()])

            floating_flag = False
            if len(curr_all_concept_network_data) == 0:
                #pick a random endpoint to start from the skeleton
                total_skeleton = sk.stack_skeletons([v["branch_skeleton"] for v in limb_corresp_for_networks.values()])
                all_endpoints = sk.find_skeleton_endpoint_coordinates(total_skeleton)

                if verbose:
                    print(f"There was no starting information so doing to put dummy information and random starting endpoint = {all_endpoints[0]}")
                curr_limb_network_stating_info = {-1:{-1:{"touching_verts":None,
                                                         "endpoint":all_endpoints[0]}}}
                floating_flag = True
            else:
                #curr_all_concept_network_data[0]["soma_group_idx"] = 0
                curr_all_concept_network_data = nru.clean_all_concept_network_data(curr_all_concept_network_data)
                curr_limb_network_stating_info = nru.all_concept_network_data_to_dict(curr_all_concept_network_data)

            #calculate the concept networks


            sep_graph_data["limb_correspondence"] = limb_corresp_for_networks

            sep_graph_data["limb_network_stating_info"] = curr_limb_network_stating_info

            #raise Exception("Checking on limb starting network info")

            limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(limb_corresp_for_networks,
                                                                                            curr_limb_network_stating_info,
                                                                                            run_concept_network_checks=False,
                                                                                verbose=verbose
                                                                                           )   

            sep_graph_data["limb_concept_networks"] = limb_to_soma_concept_networks


            # --------------- Making the new limb object -------------- #
            limb_str_name = f"split_limb_from_limb_{curr_limb_idx}_part_{seg_graph_idx}"
            if floating_flag:
                limb_str_name += "_floating"

            new_labels = [limb_str_name]

            new_limb_obj = neuron.Limb(mesh=new_limb_mesh,
                         curr_limb_correspondence=limb_corresp_for_networks,
                         concept_network_dict=limb_to_soma_concept_networks,
                         mesh_face_idx=new_limb_mesh_face_idx,
                        labels=new_labels,
                         branch_objects = fixed_node_objects,#this will have a dictionary mapping to the branch objects if provided
                       )


            sep_graph_data["limb_labels"] = new_labels
            sep_graph_data["Limb_obj"] = new_limb_obj

            new_limb_data.append(sep_graph_data)

    
    
    # Phase 2: ------------- Adjusting the existing neuron object --------------- #
    
    neuron_obj_cp = copy.deepcopy(neuron_obj)
    #1) map the new neuron objects to unused limb names
    new_limb_dict = dict()
    new_limb_idxs = [curr_limb_idx] + [len(neuron_obj_cp) + i for i in range(len(new_limb_data[1:]))]
    new_limb_string_names = [f"L{k}" for k in new_limb_idxs]
    for l_i,limb_data in zip(new_limb_idxs,new_limb_data):
        new_limb_dict[l_i] = limb_data


    #3) Delete the old limb data in the preprocessing dictionary (Adjust the soma_to_piece_connectivity)
    attr_to_update = ['limb_meshes', 'limb_correspondence', 'limb_network_stating_info', 'limb_concept_networks', 'limb_labels']
    for attr_upd in attr_to_update:
        del neuron_obj_cp.preprocessed_data[attr_upd][curr_limb_idx]

    # --- revise the soma_to_piece_connectivity -- #
    somas_to_delete_from = np.unique(neuron_obj_cp[curr_limb_idx].touching_somas())

    for sm_d in somas_to_delete_from:
        neuron_obj_cp.preprocessed_data["soma_to_piece_connectivity"][sm_d].remove(curr_limb_idx)

    #4) Delete the old limb from the neuron concept network   
    neuron_obj_cp.concept_network.remove_node(f"L{curr_limb_idx}")

    #5) Add the new limb nodes with edges to the somas they are touching
    for l_i,limb_data in new_limb_dict.items():
        curr_limb_obj = limb_data["Limb_obj"]
        curr_limb_touching_somas = curr_limb_obj.touching_somas()



        str_node_name = f"L{l_i}"
        
        neuron_obj_cp.concept_network.add_node(str_node_name)

        xu.set_node_data(curr_network=neuron_obj_cp.concept_network,
                                         node_name=str_node_name,
                                         curr_data=curr_limb_obj,
                                         curr_data_label="data")

        for sm_d in curr_limb_touching_somas:
            neuron_obj_cp.preprocessed_data["soma_to_piece_connectivity"][sm_d].append(l_i)
            neuron_obj_cp.concept_network.add_edge(str_node_name,f"S{sm_d}")

        for attr_upd in attr_to_update:
            if attr_upd == "limb_meshes":
                neuron_obj_cp.preprocessed_data[attr_upd].insert(l_i,limb_data[attr_upd])
            else:
                neuron_obj_cp.preprocessed_data[attr_upd][l_i] = limb_data[attr_upd]

    for sm_d in neuron_obj_cp.preprocessed_data["soma_to_piece_connectivity"]:
        neuron_obj_cp.preprocessed_data["soma_to_piece_connectivity"][sm_d] = list(np.unique(neuron_obj_cp.preprocessed_data["soma_to_piece_connectivity"][sm_d]))
    
    
    return neuron_obj_cp






# ----------------12/27: Whole Neuron Splitting ------------------------ #

import networkx as nx
import networkx_utils as xu
import trimesh_utils as tu
import preprocessing_vp2 as pre
import neuron_visualizations as nviz
import system_utils as su




def split_neuron_limbs_by_suggestions(neuron_obj,
                                split_suggestions,
                                      plot_soma_limb_network=False,
                                verbose=False):
    """
    Purpose: 
    
    Will take the suggestions of the splits and 
    split the necessary limbs of the neuron object and 
    return the split neuron
    
    """
    
    split_neuron_obj = copy.deepcopy(neuron_obj)
    limb_results = split_suggestions
    
    #this step is where applies the actual changes to the neuron obj
    new_concept_networks = pru.split_suggestions_to_concept_networks(neuron_obj,limb_results,
                                                                    apply_changes_to_limbs=True)
   
    for curr_limb_idx,curr_limb_nx in new_concept_networks.items():
        curr_limb_idx = int(curr_limb_idx)
        
        #delete the node sthat should be deleted
        removed_branches_list  = pru.get_removed_branches_from_suggestion(limb_results,curr_limb_idx)
        split_neuron_obj[curr_limb_idx] = pru.remove_branches_from_limb(split_neuron_obj[curr_limb_idx],removed_branches_list)
        
        

        conn_comp = list(nx.connected_components(curr_limb_nx))

        if verbose:
            print(f"\n\n---Working on Splitting Limb {curr_limb_idx} with {len(conn_comp)} components----")

        split_neuron_obj = pru.split_neuron_limb_by_seperated_network(split_neuron_obj,
                                                                      seperate_networks=conn_comp,
                                                                 curr_limb_idx = curr_limb_idx,
                                                                error_on_multile_starting_nodes=False,
                                                                verbose=verbose)
        
    if plot_soma_limb_network:
        nviz.plot_soma_limb_concept_network(split_neuron_obj)
        
    return split_neuron_obj

def split_disconnected_neuron(neuron_obj,
                              plot_seperated_neurons=False,
                             verbose=False,
                             save_original_mesh_idx=True,
                             filter_away_remaining_error_limbs=True,
                             return_errored_limbs_info=True):
    verbose=True
    """
    Purpose: If a neuron object has already been disconnected
    at the limbs, this function will then split the neuron object
    into a list of multiple neuron objects
    
    Pseudocode: 
    1) check that there do not exist any error limbs
    2) Do the splitting process
    3) Visualize results if requested
    
    """
    split_neuron_obj = neuron_obj
    
    
    
    #--------Part 1: check that all the limbs have beeen split so that there are no more error limbs
    same_soma_error_limbs = split_neuron_obj.same_soma_multi_touching_limbs
    multi_soma_error_limbs = split_neuron_obj.multi_soma_touching_limbs
    
    curr_error_limbs = nru.error_limbs(split_neuron_obj)

    if len(curr_error_limbs) > 0:
        if not filter_away_remaining_error_limbs:
            raise Exception(f"There were still error limbs before trying the neuron object split: error limbs = {curr_error_limbs}")
        else:
            print(f"Still remaining error limbs ({curr_error_limbs}), but will filter them away")
    
    
    
    
    # ------ Part 2: start the splitting process
    
    # get all the somas that we will split into
    soma_names = split_neuron_obj.get_soma_node_names()
    
    
    neuron_obj_list = []
    neuron_obj_errored_limbs_area = []
    neuron_obj_errored_limbs_skeletal_length = []
    neuron_obj_n_multi_soma_errors = []
    neuron_obj_n_same_soma_errors = []
    

    for curr_soma_idx,curr_soma_name in enumerate(soma_names):
        print(f"\n\n------ Working on Soma {curr_soma_idx} -------")

        neuron_cp = split_neuron_obj

        #getting all the soma information we will need for preprocessing
        soma_obj = neuron_cp[curr_soma_name]
        curr_soma_meshes = [soma_obj.mesh]
        curr_soma_sdfs = [soma_obj.sdf]
        curr_soma_volume = [soma_obj.volume]
        curr_soma_volume_ratios = [soma_obj.volume_ratio]





        # getting the limb information and new soma connectivity
        limb_neighbors = np.sort(xu.get_neighbors(neuron_cp.concept_network,curr_soma_name)).astype("int")
        limb_neighbors = [int(k) for k in limb_neighbors]
        print(f"limb_neighbors = {limb_neighbors}")

        soma_to_piece_connectivity = neuron_cp.preprocessed_data["soma_to_piece_connectivity"][curr_soma_idx]
        
        

        if len(np.intersect1d(limb_neighbors,soma_to_piece_connectivity)) < len(soma_to_piece_connectivity):
            raise Exception(f"piece connectivity ({soma_to_piece_connectivity}) not match limb neighbors ({limb_neighbors})")
            
        if filter_away_remaining_error_limbs:# and len(curr_error_limbs)>0:
            print(f"limb_neighbors BEFORE error limbs removed = {limb_neighbors}")
            original_len = len(limb_neighbors)
            
            
            # ---------- 1/28: More error limb information ----------------- #
            curr_neuron_same_soma_error_limbs = list(np.intersect1d(limb_neighbors,same_soma_error_limbs))
            curr_neuron_multi_soma_error_limbs = list(np.intersect1d(limb_neighbors,multi_soma_error_limbs))
            
            #print(f"curr_neuron_multi_soma_error_limbs = {curr_neuron_multi_soma_error_limbs}")
            
            curr_n_multi_soma_limbs_cancelled = len(curr_neuron_multi_soma_error_limbs)
            curr_n_same_soma_limbs_cancelled = len(curr_neuron_same_soma_error_limbs)
            
            unique_error_limbs = np.unique(curr_neuron_same_soma_error_limbs + curr_neuron_multi_soma_error_limbs)
            
            curr_error_limbs_cancelled_area = [split_neuron_obj[k].area/len(np.unique(split_neuron_obj[k].touching_somas()))
                                               for k in unique_error_limbs]
            
            curr_error_limbs_cancelled_skeletal_length = [split_neuron_obj[k].skeletal_length/len(np.unique(split_neuron_obj[k].touching_somas()))
                                               for k in unique_error_limbs]
            
            
            limb_neighbors = np.setdiff1d(limb_neighbors,curr_error_limbs)
            print(f"limb_neighbors AFTER error limbs removed = {limb_neighbors}")
            
        else:
            curr_n_multi_soma_limbs_cancelled = 0
            curr_n_same_soma_limbs_cancelled = 0
            curr_error_limbs_cancelled_area = []
            
        if verbose:
            print(f"curr_n_multi_soma_limbs_cancelled = {curr_n_multi_soma_limbs_cancelled}")
            print(f"curr_n_same_soma_limbs_cancelled = {curr_n_same_soma_limbs_cancelled}")
            print(f"n_errored_lims = {len(curr_error_limbs_cancelled_area)}")
            print(f"curr_error_limbs_cancelled_area = {curr_error_limbs_cancelled_area}")
            
        neuron_obj_errored_limbs_area.append(curr_error_limbs_cancelled_area)
        neuron_obj_errored_limbs_skeletal_length.append(curr_error_limbs_cancelled_skeletal_length)
        neuron_obj_n_multi_soma_errors.append(curr_n_multi_soma_limbs_cancelled)
        neuron_obj_n_same_soma_errors.append(curr_n_same_soma_limbs_cancelled)

        curr_soma_to_piece_connectivity = {0:list(np.arange(0,len(limb_neighbors)))}






        #getting the whole mesh and limb face correspondence
        mesh_list_for_whole = [soma_obj.mesh]

        #for the limb meshes
        limb_meshes = []

        #for the limb mesh faces idx
        counter = len(curr_soma_meshes[0].faces)
        face_idx_list = [np.arange(0,counter)]

        old_node_to_new_node_mapping = dict()


        for i,k in  enumerate(limb_neighbors):

            #getting the name mapping
            old_node_to_new_node_mapping[k] = i

            #getting the meshes of the limbs
            limb_mesh = neuron_cp[k].mesh
            limb_meshes.append(limb_mesh)


            mesh_list_for_whole.append(limb_mesh)
            face_length = len(limb_mesh.faces)
            face_idx_list.append(np.arange(counter,counter + face_length))
            counter += face_length

        whole_mesh = tu.combine_meshes(mesh_list_for_whole)






        # generating the new limb correspondence:
        #curr_limb_correspondence = dict([(i,neuron_cp.preprocessed_data["limb_correspondence"][k]) for i,k in enumerate(limb_neighbors)])
        curr_limb_correspondence = dict([(i,neuron_cp[k].limb_correspondence) for i,k in enumerate(limb_neighbors)])




        # concept network generation
        curr_limb_network_stating_info = dict()


        for k in limb_neighbors:

            local_starting_info = neuron_cp.preprocessed_data["limb_network_stating_info"][k]

            #making sure the soma has the right name
            soma_keys = list(local_starting_info.keys())
            if len(soma_keys) > 1:
                raise Exception("More than one soma connection")
            else:
                soma_key = soma_keys[0]

            if soma_key != 0:
                local_starting_info = {0:local_starting_info[soma_key]}


            #making sure the soma group has the right name
            starting_group_keys = list(local_starting_info[0].keys())
            if len(starting_group_keys) > 1 or starting_group_keys[0] != 0:
                raise Exception("Touching group was not equal to 0")

            #save the new starting info
            curr_limb_network_stating_info[old_node_to_new_node_mapping[k]] = local_starting_info

        # creating the new concept networks from the starting info
        curr_limb_concept_networks=dict()

        for curr_limb_idx,new_limb_correspondence_indiv in curr_limb_correspondence.items():
            limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(new_limb_correspondence_indiv,
                                                                                curr_limb_network_stating_info[curr_limb_idx],
                                                                                run_concept_network_checks=True,
                                                                               )   

            curr_limb_concept_networks[curr_limb_idx] = limb_to_soma_concept_networks








        #limb labels:
        curr_limb_labels = dict()

        for k in limb_neighbors:
            local_limb_labels = neuron_cp.preprocessed_data["limb_labels"][k]
            if local_limb_labels is None or local_limb_labels == "Unlabeled":
                local_limb_labels = []

            local_limb_labels.append(f"Soma_{curr_soma_idx}_originally")
            curr_limb_labels[old_node_to_new_node_mapping[k]] = local_limb_labels



        meshes_to_concat = [[kv["branch_mesh"] for k,kv in jv.items()] for j,jv in curr_limb_correspondence.items()]
        
        if len(meshes_to_concat) > 0:
            whole_branch_mesh = tu.combine_meshes(np.concatenate(meshes_to_concat))
            
            floating_indexes = tu.mesh_pieces_connectivity(neuron_obj.mesh,
                         whole_branch_mesh,
                        periphery_pieces=neuron_obj.non_soma_touching_meshes,
                           print_flag=False)
        
            local_floating_meshes = list(np.array(neuron_obj.non_soma_touching_meshes)[floating_indexes])

            print(f"local_floating_meshes = {local_floating_meshes}")
            whole_mesh  = whole_mesh + local_floating_meshes
        else:
            local_floating_meshes = []
            whole_mesh  = whole_mesh + local_floating_meshes
            

        
        

        #using all of the data to create new preprocessing info
        new_preprocessed_data = preprocessed_data= dict(
                #soma data
                soma_meshes = curr_soma_meshes,
                soma_sdfs = curr_soma_sdfs,
                soma_volumes = curr_soma_volume,
                soma_volume_ratios=curr_soma_volume_ratios,

                #soma connectivity
                soma_to_piece_connectivity = curr_soma_to_piece_connectivity,

                # limb info
                limb_correspondence=curr_limb_correspondence,
                limb_meshes=limb_meshes,
                limb_mehses_face_idx = face_idx_list,
                limb_labels=curr_limb_labels,

                #concept network info
                limb_concept_networks=curr_limb_concept_networks,
                limb_network_stating_info=curr_limb_network_stating_info,


                # the other mesh pieces that will not be included
                insignificant_limbs=[],
                not_processed_soma_containing_meshes=[],
                glia_faces = [],
                non_soma_touching_meshes=local_floating_meshes,
                inside_pieces=[],


                )

        limb_to_branch_objects = dict()
        for k in limb_neighbors:
            limb_obj = neuron_cp[int(k)]
            branch_dict = dict([(b,limb_obj[int(b)]) for b in limb_obj.get_branch_names()])
            limb_to_branch_objects[old_node_to_new_node_mapping[k]] = branch_dict

        segment_id = neuron_cp.segment_id
        description = f"{neuron_cp.description}_soma_{curr_soma_idx}_split"


        #---------- 1/24: Will save off the original mesh idx so can use the old mesh file to load -------- #
        if save_original_mesh_idx:
            original_mesh_idx = tu.original_mesh_faces_map(neuron_obj.mesh,whole_mesh,
                          exact_match=True)
        else:
            original_mesh_idx = None


        # new neuron object:

        single_split_neuron_obj = neuron.Neuron(mesh=whole_mesh,
                 segment_id=segment_id,
                 description=description,
                 preprocessed_data=new_preprocessed_data,
                 limb_to_branch_objects=limb_to_branch_objects,
                 widths_to_calculate=[],
                 original_mesh_idx=original_mesh_idx,
                suppress_output=not verbose)


        neuron_obj_list.append(single_split_neuron_obj)
        
        
        
    # ------ Part 3: Visualize the Results
    print(f"\n\nNumber of seperate neuron objects = {len(neuron_obj_list)}")

    if plot_seperated_neurons:
        for n_obj in neuron_obj_list:
            nviz.visualize_neuron(n_obj,
                                 visualize_type=["mesh","skeleton"],
                                 limb_branch_dict="all")

            
    
    if return_errored_limbs_info:
        return (neuron_obj_list,
                neuron_obj_errored_limbs_area,
                neuron_obj_errored_limbs_skeletal_length,
                neuron_obj_n_multi_soma_errors,
                neuron_obj_n_same_soma_errors)
    else:
        return neuron_obj_list



def split_neuron(neuron_obj,
                 limb_results=None,
                 plot_soma_limb_network=False,
                 plot_seperated_neurons=False,
                verbose=False,
                 filter_away_remaining_error_limbs=True,
                 return_error_info = False,
                **kwargs):
    """
    Purpose: To take in a whole neuron that could have any number of somas
    and then to split it into multiple neuron objects

    Pseudocode: 
    1) Get all of the split suggestions
    2) Split all of the limbs that need splitting
    3) Once have split the limbs, split the neuron object into mutliple objects


    """
    
    if neuron_obj.n_error_limbs == 0:
        print("No error limbs to processs so just returning the original neuron")
        
        neuron_list = [neuron_obj]
        
        if return_error_info:
            return (neuron_list,[0],[0],0,0)
        else:
            return neuron_list
        
    
    neuron_obj = copy.deepcopy(neuron_obj)
    
    # ---------- 1/22 Addition that cleans the concept network info ------------ #
    nru.clean_neuron_all_concept_network_data(neuron_obj)
    
    #1) Get all of the split suggestions
    if limb_results is None:
        limb_results = pru.multi_soma_split_suggestions(neuron_obj,
                                                   verbose = verbose,
                                                       **kwargs)
    else:
        print("using precomputed split suggestions")
    
    #2) Split all of the limbs that need splitting
    split_neuron_obj = pru.split_neuron_limbs_by_suggestions(neuron_obj,
                                split_suggestions=limb_results,
                                plot_soma_limb_network=plot_soma_limb_network,
                                verbose=verbose)
    
    
    
    if not filter_away_remaining_error_limbs:
        curr_error_limbs = nru.error_limbs(split_neuron_obj)

        if len(curr_error_limbs) > 0:
            raise Exception(f"There were still error limbs before the splitting: {curr_error_limbs}")
            
    
    #3) Once have split the limbs, split the neuron object into mutliple objects
    (neuron_list,
     neuron_list_errored_limbs_area,
     neuron_list_errored_limbs_skeletal_length,
    neuron_list_n_multi_soma_errors,
    neuron_list_n_same_soma_errors) = pru.split_disconnected_neuron(split_neuron_obj,
                         plot_seperated_neurons=plot_seperated_neurons,
                        filter_away_remaining_error_limbs=filter_away_remaining_error_limbs,
                         verbose =verbose)
    
    #2b) Check that all the splits occured
    for i,n_obj in enumerate(neuron_list):
        curr_error_limbs = nru.error_limbs(n_obj)

        if len(curr_error_limbs) > 0:
            raise Exception(f"There were still error limbs after splitting for neuron obj {i}")

    if return_error_info:
        return (neuron_list,neuron_list_errored_limbs_area,
                neuron_list_errored_limbs_skeletal_length,
                                        neuron_list_n_multi_soma_errors,
                                        neuron_list_n_same_soma_errors)
    else:
        return neuron_list


import numpy_utils as nu
import time
'''
def remove_branches_from_limb_old(limb_obj,branch_list,
                             plot_new_limb=False,
                              reassign_mesh=True,
                              store_placeholder_for_removed_nodes=True,
                              debug_time = True,
                             verbose=False):
    """
    Purpose: To remove 1 or more branches from the concept network
    of a limb and to adjust the underlying skeleton
    
    Application: To be used in when trying to split 
    a neuron and want to combine nodes that are really close
    
    
    *** currently does not does not reallocate the mesh part of the nodes that were deleted
    
    
    Pseudocode: 
    
    For each branch to remove
    0) Find the branches that were touching the soon to be deleted branch
    1) Alter the skeletons of those that were touching that branch
    
    After revised all nodes
    2) Remove the current node
    3) Generate the limb correspondence, network starting info to generate soma concept networks
    4) Create a new limb object and return it


    Ex: new_limb_obj = nru.remove_branches_from_limb(curr_limb,[30,31],plot_new_limb=True,verbose=True)
    """
    
    limb_time = time.time()
    
    curr_limb_cp = copy.deepcopy(limb_obj)
    
    if debug_time:
        print(f"deepcopy limb time = {time.time() - limb_time}")
        limb_time = time.time()
    
    
    if not nu.is_array_like(branch_list ):
        branch_list = [branch_list]
        
    seg_id_lookup = np.arange(0,len(curr_limb_cp))
    
    for curr_short_seg in branch_list:
        #need to look up the new seg id if there is one now based on previous deletions
        
        curr_short_seg_revised = seg_id_lookup[curr_short_seg]
        if verbose:
            print(f"curr_short_seg_revised = {curr_short_seg_revised}")
        
        
        branch_obj = curr_limb_cp[curr_short_seg_revised]
        
        touching_branches,touching_endpoints = nru.skeleton_touching_branches(curr_limb_cp,branch_idx=curr_short_seg_revised)
        
        if debug_time:
            print(f"skeleton branch touchings = {time.time() - limb_time}")
            limb_time = time.time()
        
        
        #deciding whether to take the average or just an endpoint as the new stitch point depending on if nodes on both ends
        touch_len = np.array([len(k) for k in touching_branches])
        
        if verbose:
            print(f"np.sum(touch_len>0) = {np.sum(touch_len>0)}")
            
        if np.sum(touch_len>0)==2:
            if verbose:
                print("Using average stitch point")
            new_stitch_point = np.mean(touching_endpoints,axis=0)
            middle_node = True
        else:
            if verbose:
                print("Using ONE stitch point")
            new_stitch_point = touching_endpoints[np.argmax(touch_len)]
            middle_node = False
        
        if verbose:
            print(f"touching_endpoints = {touching_endpoints}")
            print(f"new_stitch_point = {new_stitch_point}")
            
        if debug_time:
            print(f"Getting new stitch point time = {time.time() - limb_time}")
            limb_time = time.time()
        
        #1a) Decide which branch to give the mesh to
        if middle_node and reassign_mesh:
            """
            Pseudocode:
            1) Get the skeletal angles between all the touching branches and the branch to be deleted
            2) Find the branch with the smallest skeletal angle
            3) Things that need to be updated for the winning branch:
            - mesh
            -mesh_face_idx
            - n_spines
            - spines
            - spines_volume
            
            
            """
            all_touching_branches = np.concatenate(touching_branches)
            all_touching_branches_angles = [sk.offset_skeletons_aligned_parent_child_skeletal_angle(
                            curr_limb_cp[br].skeleton,
                            curr_limb_cp[curr_short_seg_revised].skeleton) for br in all_touching_branches]
            winning_branch =all_touching_branches[np.argmin(all_touching_branches_angles)]
            
            
            if verbose:
                print(f"Angles for {all_touching_branches} are {all_touching_branches_angles}")
                print(f"Branch that will absorb mesh of {curr_short_seg} is {winning_branch} ")
                
            if debug_time:
                print(f"branch angles = {time.time() - limb_time}")
                limb_time = time.time()
            
            curr_limb_cp[winning_branch].mesh_face_idx = np.concatenate([curr_limb_cp[winning_branch].mesh_face_idx,
                                                                    curr_limb_cp[curr_short_seg_revised].mesh_face_idx])
            curr_limb_cp[winning_branch].mesh = tu.combine_meshes([curr_limb_cp[winning_branch].mesh,
                                                                    curr_limb_cp[curr_short_seg_revised].mesh])
            if not curr_limb_cp[curr_short_seg_revised].spines is None:
                if curr_limb_cp[winning_branch].spines is not None:
                    curr_limb_cp[winning_branch].spines += curr_limb_cp[curr_short_seg_revised].spines
                    curr_limb_cp[winning_branch].spines_volume += curr_limb_cp[curr_short_seg_revised].spines_volume
                else:
                    curr_limb_cp[winning_branch].spines = curr_limb_cp[curr_short_seg_revised].spines
                    curr_limb_cp[winning_branch].spines_volume = curr_limb_cp[curr_short_seg_revised].spines_volume
                    
            
            if debug_time:
                print(f"Resolving spines time = {time.time() - limb_time}")
                limb_time = time.time()
            

        #1b) Alter the skeletons of those that were touching
        for t_branches,t_endpoints in zip(touching_branches,touching_endpoints):
            for br in t_branches:
                curr_limb_cp[br].skeleton = sk.add_and_smooth_segment_to_branch(curr_limb_cp[br].skeleton,
                                                                                new_stitch_point=new_stitch_point,
                                                                            skeleton_stitch_point=t_endpoints)
                curr_limb_cp[br].calculate_endpoints()
                
        if debug_time:
            print(f"smoothing the segments and calculating endpoints = {time.time() - limb_time}")
            limb_time = time.time()
        """
        
        
        
        """
        
            
        #2) Remove the current node
        if not store_placeholder_for_removed_nodes:
            curr_limb_cp.concept_network.remove_node(curr_short_seg_revised)
            #update the seg_id_lookup
            seg_id_lookup = np.insert(seg_id_lookup,curr_short_seg,-1)[:-1]
            run_concept_network_checks=True
        else:
            curr_mesh_center = curr_limb_cp[curr_short_seg_revised].mesh_center
            center_deviation = 5.2345
            curr_limb_cp[curr_short_seg_revised].skeleton = np.array([curr_mesh_center,curr_mesh_center+center_deviation]).reshape(-1,2,3)
            curr_limb_cp[curr_short_seg_revised].calculate_endpoints()
            run_concept_network_checks = False
            
        if debug_time:
            print(f" Adding placeholder branch time = {time.time() - limb_time}")
            limb_time = time.time()

        limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(curr_limb_cp.limb_correspondence,
                                                                                    curr_limb_cp.network_starting_info,
                                                                                    run_concept_network_checks=False,
                                                                              )  
        if debug_time:
            print(f"Calculating limb concept networks time = {time.time() - limb_time}")
            limb_time = time.time()
        
        #print(f"curr_limb.deleted_edges 6={curr_limb_cp.deleted_edges}")
        curr_limb_cp = neuron.Limb(curr_limb_cp.mesh,
                                 curr_limb_cp.limb_correspondence,
                                 concept_network_dict=limb_to_soma_concept_networks,
                                 mesh_face_idx = curr_limb_cp.mesh_face_idx,
                                 labels=curr_limb_cp.labels,
                                 branch_objects=curr_limb_cp.branch_objects,
                                  deleted_edges=curr_limb_cp.deleted_edges,
                                  created_edges=curr_limb_cp.created_edges)
        
        if debug_time:
                print(f"Creating new limb object time = {time.time() - limb_time}")
                limb_time = time.time()
        
        #print(f"curr_limb.deleted_edges 7={curr_limb_cp.deleted_edges}")
        
        
        
        
    
    if plot_new_limb:
        nviz.plot_limb_correspondence(curr_limb_cp.limb_correspondence)
    
    return curr_limb_cp
'''

def remove_branches_from_limb(limb_obj,branch_list,
                             plot_new_limb=False,
                              reassign_mesh=True,
                              store_placeholder_for_removed_nodes=True,
                              debug_time = False,
                             verbose=False):
    """
    Purpose: To remove 1 or more branches from the concept network
    of a limb and to adjust the underlying skeleton
    
    Application: To be used in when trying to split 
    a neuron and want to combine nodes that are really close
    
    
    *** currently does not does not reallocate the mesh part of the nodes that were deleted
    
    
    Pseudocode: 
    
    For each branch to remove
    0) Find the branches that were touching the soon to be deleted branch
    1) Alter the skeletons of those that were touching that branch
    
    After revised all nodes
    2) Remove the current node
    3) Generate the limb correspondence, network starting info to generate soma concept networks
    4) Create a new limb object and return it


    Ex: new_limb_obj = nru.remove_branches_from_limb(curr_limb,[30,31],plot_new_limb=True,verbose=True)
    """
    
    limb_time = time.time()
    
    curr_limb_cp = copy.deepcopy(limb_obj)
    
    if debug_time:
        print(f"deepcopy limb time = {time.time() - limb_time}")
        limb_time = time.time()
    
    
    if not nu.is_array_like(branch_list ):
        branch_list = [branch_list]
        
    seg_id_lookup = np.arange(0,len(curr_limb_cp))
    
    for curr_short_seg in branch_list:
        #need to look up the new seg id if there is one now based on previous deletions
        
        curr_short_seg_revised = seg_id_lookup[curr_short_seg]
        if verbose:
            print(f"curr_short_seg_revised = {curr_short_seg_revised}")
        
        
        branch_obj = curr_limb_cp[curr_short_seg_revised]
        
        touching_branches,touching_endpoints = nru.skeleton_touching_branches(curr_limb_cp,branch_idx=curr_short_seg_revised)
        
        if debug_time:
            print(f"skeleton branch touchings = {time.time() - limb_time}")
            limb_time = time.time()
        
        
        #deciding whether to take the average or just an endpoint as the new stitch point depending on if nodes on both ends
        touch_len = np.array([len(k) for k in touching_branches])
        
        if verbose:
            print(f"np.sum(touch_len>0) = {np.sum(touch_len>0)}")
            
        if np.sum(touch_len>0)==2:
            if verbose:
                print("Using average stitch point")
            new_stitch_point = np.mean(touching_endpoints,axis=0)
            middle_node = True
        else:
            if verbose:
                print("Using ONE stitch point")
            new_stitch_point = touching_endpoints[np.argmax(touch_len)]
            middle_node = False
        
        if verbose:
            print(f"touching_endpoints = {touching_endpoints}")
            print(f"new_stitch_point = {new_stitch_point}")
            
        if debug_time:
            print(f"Getting new stitch point time = {time.time() - limb_time}")
            limb_time = time.time()
        
        #1a) Decide which branch to give the mesh to
        if middle_node and reassign_mesh:
            """
            Pseudocode:
            1) Get the skeletal angles between all the touching branches and the branch to be deleted
            2) Find the branch with the smallest skeletal angle
            3) Things that need to be updated for the winning branch:
            - mesh
            -mesh_face_idx
            - n_spines
            - spines
            - spines_volume
            
            
            """
            all_touching_branches = np.concatenate(touching_branches)
            all_touching_branches_angles = [sk.offset_skeletons_aligned_parent_child_skeletal_angle(
                            curr_limb_cp[br].skeleton,
                            curr_limb_cp[curr_short_seg_revised].skeleton) for br in all_touching_branches]
            winning_branch =all_touching_branches[np.argmin(all_touching_branches_angles)]
            
            
            if verbose:
                print(f"Angles for {all_touching_branches} are {all_touching_branches_angles}")
                print(f"Branch that will absorb mesh of {curr_short_seg} is {winning_branch} ")
                
            if debug_time:
                print(f"branch angles = {time.time() - limb_time}")
                limb_time = time.time()
            
            curr_limb_cp[winning_branch].mesh_face_idx = np.concatenate([curr_limb_cp[winning_branch].mesh_face_idx,
                                                                    curr_limb_cp[curr_short_seg_revised].mesh_face_idx])
            curr_limb_cp[winning_branch].mesh = tu.combine_meshes([curr_limb_cp[winning_branch].mesh,
                                                                    curr_limb_cp[curr_short_seg_revised].mesh])
            if not curr_limb_cp[curr_short_seg_revised].spines is None:
                if curr_limb_cp[winning_branch].spines is not None:
                    curr_limb_cp[winning_branch].spines += curr_limb_cp[curr_short_seg_revised].spines
                    curr_limb_cp[winning_branch].spines_volume += curr_limb_cp[curr_short_seg_revised].spines_volume
                else:
                    curr_limb_cp[winning_branch].spines = curr_limb_cp[curr_short_seg_revised].spines
                    curr_limb_cp[winning_branch].spines_volume = curr_limb_cp[curr_short_seg_revised].spines_volume
                    
            
            if debug_time:
                print(f"Resolving spines time = {time.time() - limb_time}")
                limb_time = time.time()
            

        #1b) Alter the skeletons of those that were touching
        for t_branches,t_endpoints in zip(touching_branches,touching_endpoints):
            for br in t_branches:
                curr_limb_cp[br].skeleton = sk.add_and_smooth_segment_to_branch(curr_limb_cp[br].skeleton,
                                                                                new_stitch_point=new_stitch_point,
                                                                            skeleton_stitch_point=t_endpoints)
                curr_limb_cp[br].calculate_endpoints()
                
        if debug_time:
            print(f"smoothing the segments and calculating endpoints = {time.time() - limb_time}")
            limb_time = time.time()
        """
        
        
        
        """
        
            
        #2) Remove the current node
        if not store_placeholder_for_removed_nodes:
            curr_limb_cp.concept_network.remove_node(curr_short_seg_revised)
            #update the seg_id_lookup
            seg_id_lookup = np.insert(seg_id_lookup,curr_short_seg,-1)[:-1]
            run_concept_network_checks=True
        else:
            curr_mesh_center = curr_limb_cp[curr_short_seg_revised].mesh_center
            center_deviation = 5.2345
            curr_limb_cp[curr_short_seg_revised].skeleton = np.array([curr_mesh_center,curr_mesh_center+center_deviation]).reshape(-1,2,3)
            curr_limb_cp[curr_short_seg_revised].calculate_endpoints()
            run_concept_network_checks = False
            
        if debug_time:
            print(f" Adding placeholder branch time = {time.time() - limb_time}")
            limb_time = time.time()
            

    limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(curr_limb_cp.limb_correspondence,
                                                                                curr_limb_cp.network_starting_info,
                                                                                run_concept_network_checks=False,
                                                                          )  
    if debug_time:
        print(f"Calculating limb concept networks time = {time.time() - limb_time}")
        limb_time = time.time()

    #print(f"curr_limb.deleted_edges 6={curr_limb_cp.deleted_edges}")
    curr_limb_cp = neuron.Limb(curr_limb_cp.mesh,
                             curr_limb_cp.limb_correspondence,
                             concept_network_dict=limb_to_soma_concept_networks,
                             mesh_face_idx = curr_limb_cp.mesh_face_idx,
                             labels=curr_limb_cp.labels,
                             branch_objects=curr_limb_cp.branch_objects,
                              deleted_edges=curr_limb_cp.deleted_edges,
                              created_edges=curr_limb_cp.created_edges)

    if debug_time:
            print(f"Creating new limb object time = {time.time() - limb_time}")
            limb_time = time.time()


    if plot_new_limb:
        nviz.plot_limb_correspondence(curr_limb_cp.limb_correspondence)
    
    return curr_limb_cp



# ------------------ Neuroglancer Tools Info ---------------------------- #
import numpy as np
import numpy_utils as nu
import pandas as pd
from annotationframeworkclient import FrameworkClient
from nglui import statebuilder

def get_client():
    return FrameworkClient('minnie65_phase3_v1')

def set_state_builder(layer_name = 'split_cands',
                      color='#FFFFFF',
                     transparency=0.5):
    client = FrameworkClient('minnie65_phase3_v1')
    # The following generates a statebuilder that can turn dataframes into neuroglancer states
    img_layer = statebuilder.ImageLayerConfig(client.info.image_source(), contrast_controls=True, black=0.35, white=0.65)
    seg_layer = statebuilder.SegmentationLayerConfig(client.info.segmentation_source(), selected_ids_column='root_id',
                                                    view_kws={'alpha_3d': transparency})
    pts = statebuilder.PointMapper('split_location', set_position=True)
    anno_layer = statebuilder.AnnotationLayerConfig(layer_name, mapping_rules=pts, linked_segmentation_layer=seg_layer.name, color=color, active=True,
                                                   tags=['valid', 'not_valid','proof'])
    sb = statebuilder.StateBuilder([img_layer, seg_layer, anno_layer], state_server=client.state.state_service_endpoint)
    return sb

def set_edit_dataframe(split_locs,
                      root_ids,
                      priority=None,
                      ):
    """
    Will create a dataframe that can be fed into a statebuilder
    to generate a neuroglancer link
    
    """
    if priority is None:
        priority = list(np.arange(1,len(split_locs)+1))
        
    if not nu.is_array_like(root_ids):
        root_ids = [root_ids]*len(split_locs)
        
    edit_df = pd.DataFrame({'split_location': split_locs,
                            'root_id': root_ids,
                            'priority': priority})
    return edit_df

def split_coordinates_to_neuroglancer_link(split_locs=None,
                      root_ids=None,
                            sb=None,
                            edit_df=None,
                      priority=None):
    """
    
    
    
    Example:
    limb_results = pru.multi_soma_split_suggestions(neuron_obj)
    split_coordinates = pru.get_all_coordinate_suggestions(limb_results)
    
    neuroglancer_split_link(split_coordinates,
                          neuron_obj.segment_id)
    
    
    """
    if sb is None:
        sb = set_state_builder()
    if edit_df is None:
        edit_df = set_edit_dataframe(split_locs,
                          root_ids,
                          priority,
                          )
    
    return sb.render_state(edit_df.sort_values(by='priority'), return_as='html')

import matplotlib_utils as mu
import pandas as pd
from annotationframeworkclient import FrameworkClient
from nglui import statebuilder
import copy

def split_info_to_neuroglancer_link(segment_id,
                                   split_info=None,
                                    split_coordinates = None,
                                    cut_path_coordinates = None,
                                    not_cut_path_coordinates = None,
                                   other_annotations = dict(),
                                    output_type = "local",
                                    split_coordinates_color = "red",
                                    split_coordinates_name = "split_location",
                                    cut_path_coordinates_color = "white",
                                    cut_path_coordinates_name = "cut_path",
                                    not_cut_path_coordinates_color = "green",
                                    not_cut_path_coordinates_name = "not_cut_path",
                                    verbose = False):
    """
    Purpose: To take the split suggestions output from the 
    pru.multi_soma_split_suggestions and then to turn it into a neuroglancer link
    to be rendered locally or uploaded to the server
    
    
    """
    
    print(f"segment_id = {segment_id}")
    
    if split_coordinates is None:
        split_coordinates = pru.get_all_coordinate_suggestions(split_info)

    if cut_path_coordinates is None or not_cut_path_coordinates is None:
        cut_path_coordinates,not_cut_path_coordinates = pru.get_all_cut_and_not_cut_path_coordinates(split_info)


    annotations_info = {
                        cut_path_coordinates_name:dict(coordinates=cut_path_coordinates,
                                                   color=cut_path_coordinates_color),
                        not_cut_path_coordinates_name:dict(coordinates=not_cut_path_coordinates,
                                                   color=not_cut_path_coordinates_color),
                        split_coordinates_name:dict(coordinates=split_coordinates,
                                                   color=split_coordinates_color),

                       }

    annotations_info.update(other_annotations)

    df_list = []
    states_list = []

    for ann_name,ann_dict in annotations_info.items():
        sb = pru.set_state_builder(layer_name=ann_name,
                                         color=mu.color_to_hex(ann_dict["color"]))
        df = pru.set_edit_dataframe(ann_dict["coordinates"],segment_id)

        states_list.append(sb)
        df_list.append(df)

    if verbose:
        print(f"Number of different layers = {len(states_list)}")

    # building the chained state
    chained_sb = statebuilder.ChainedStateBuilder(states_list)

    if output_type=="local":
        return_value= chained_sb.render_state(df_list, return_as='html')
    elif output_type == "server":
        # To upload to the state server for a shortened URL:
        client = pru.get_client()
        state = chained_sb.render_state(df_list, return_as='dict')
        state_id = client.state.upload_state_json(state)
        short_url = client.state.build_neuroglancer_url(state_id, ngl_url=client.info.viewer_site())
        return_value = short_url
    else:
        raise Exception("Not implemented")

    return return_value


import re
import pandas as pd
from pathlib import Path

def ipython_html_object_to_link(html_obj):
    links = re.findall("href=[\"\'](.*?)[\"\']", html_obj.data)
    return links[0]

from tqdm_utils import tqdm
def split_suggestions_datajoint_dicts_to_neuroglancer_dataframe(split_suggestions_data,
                                                               output_type="local",
                                                               verbose=False):
    """
    Purpose: To turn the splt dictionaries into a datframe with neuroglancer links
    
    """
    spreadsheet_data = []
    for curr_data in tqdm(split_suggestions_data):
        curr_link = pru.split_info_to_neuroglancer_link(segment_id=curr_data["segment_id"],
                                            split_info = curr_data["split_results"],
                                            output_type=output_type
                                           )
        

        if type(curr_link) is str:
            curr_link_html = curr_link
        else:
            curr_link_html = ipython_html_object_to_link(curr_link)

        n_suggested_cuts = len(pru.get_all_coordinate_suggestions(curr_data["split_results"]))
        n_paths_not_cut = pru.get_n_paths_not_cut(curr_data["split_results"])

        if verbose:
            print(f"n_suggested_cuts = {n_suggested_cuts}, n_paths_not_cut = {n_paths_not_cut}")

        local_dict = dict(segment_id=curr_data["segment_id"],
                         n_suggested_cuts=n_suggested_cuts,
                         n_paths_not_cut=n_paths_not_cut,
                         link=curr_link_html)

        spreadsheet_data.append(local_dict)

    return  pd.DataFrame.from_dict(spreadsheet_data)



# ----------------- 1/26: Final Proofreading Rules splitting ------------#

import copy
import networkx as nx


def delete_branches_from_limb(neuron_obj,
                              branches_to_delete,
                              limb_idx=None,
                              limb_name=None,
                              verbose=False,
                                ):
    """
    Will delete branches from a certain limb
    
    
    
    """
    if limb_idx is None:
        limb_idx = int(limb_name[1:])
    
    if verbose:
        print(f"---- Working on Limb {limb_idx} ----")
        print(f"length of concept network BEFORE elimination = {len(neuron_obj[limb_idx].concept_network.nodes())} ")

    #1) Remove the nodes in the limb branch dict
    concept_graph = nx.Graph(neuron_obj[limb_idx].concept_network)
    concept_graph.remove_nodes_from(branches_to_delete)
    seperate_networks = [list(k) for k in nx.connected_components(concept_graph)]

    if verbose:
        print(f"length of concept network AFTER elimination = {len(concept_graph.nodes())} ")

    split_neuron = pru.split_neuron_limb_by_seperated_network(neuron_obj,
                                                             curr_limb_idx=limb_idx,
                                                              #seperate_networks = [list(concept_graph.nodes())]
                                                              seperate_networks = seperate_networks
                                                             )
    return split_neuron

def delete_branches_from_neuron(neuron_obj,
                                limb_branch_dict,
                                plot_neuron_after_cancellation = True,
                                plot_final_neuron = True,
                                verbose = False,
                               ):
    

    """
    Purpose: To eliminate the error cells and downstream targets
    given limb branch dict of nodes to eliminate

    Pseudocode: 

    For each limb in branch dict
    1) Remove the nodes in the limb branch dict

    2) Send the neuron to 
       i) split_neuron_limbs_by_suggestions
       ii) split_disconnected_neuron


    If a limb is empty or has no more connetion to the
    starting soma then it will be deleted in the end

    """



    if verbose:
        print(f"Number of branches at beginning = {neuron_obj.n_branches} ")

    #For each limb in branch dict
    n_branches_at_start = int(neuron_obj.n_branches)

    split_neuron = neuron_obj
    for limb_name,branches_to_delete in limb_branch_dict.items():
        
        split_neuron = delete_branches_from_limb(split_neuron,
                              branches_to_delete=branches_to_delete,
                              limb_name=limb_name,
                              verbose=verbose,
                                )
        

    if plot_neuron_after_cancellation:
        nviz.visualize_neuron(split_neuron,
                             limb_branch_dict="all")

    if verbose:
        print(f"Number of branches after split_neuron_limb_by_seperated_network = {split_neuron.n_branches} ")

    disconnected_neuron = pru.split_disconnected_neuron(split_neuron,return_errored_limbs_info=False)

    if len(disconnected_neuron) != 1:
        raise Exception(f"The disconnected neurons were not 1: {disconnected_neuron}")

    return_neuron = disconnected_neuron[0]

    if verbose: 
        print(f"Number of branches after split_disconnectd neuron = {return_neuron} ")

    if plot_final_neuron:
        nviz.visualize_neuron(return_neuron,
                             limb_branch_dict="all")
    
    return return_neuron
    
    
# ----------- 1/28 Proofreading Rules that will help filter a neuron object --------------- #
import classification_utils as clu

def filter_away_limb_branch_dict(neuron_obj,
                                 limb_branch_dict,
                                plot_limb_branch_filter_away=False,
                                 plot_limb_branch_filter_with_disconnect_effect=False,
                                return_error_info=True,
                                 plot_final_neuron=False,
                                 verbose=False
                                ):
    """
    Purpose: To filter away a limb branch dict from a single neuron
    
    
    """
    
    if len(limb_branch_dict) == 0:
        print("limb_branch_dict was empty so returning original neuron")
        if return_error_info:
            return neuron_obj,0,0
        else:
            return neuron_obj
    
    if plot_limb_branch_filter_away:
        print("\n\nBranches Requested to Remove (without disconnect effect)")
        nviz.plot_limb_branch_dict(neuron_obj,
                             limb_branch_dict)


    #2) Find the total branches that will be removed using the axon-error limb branch dict

    removed_limb_branch_dict = nru.limb_branch_after_limb_branch_removal(neuron_obj=neuron_obj,
                                          limb_branch_dict = limb_branch_dict,
                                 return_removed_limb_branch = True,
                                )

    if verbose:
        print(f"After disconnecte effect, removed_limb_branch_dict = {removed_limb_branch_dict}")
        
    if plot_limb_branch_filter_with_disconnect_effect:
        print("\n\nBranches Requested to Remove (WITH disconnect effect)")
        nviz.plot_limb_branch_dict(neuron_obj,
                             removed_limb_branch_dict)
    
    #3) Calculate the total skeleton length and error faces area for what will be removed
    if return_error_info:
        total_area = nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                           limb_branch_dict=removed_limb_branch_dict,
                                           feature="area")

        total_sk_distance = nru.sum_feature_over_limb_branch_dict(neuron_obj,
                                               limb_branch_dict=removed_limb_branch_dict,
                                               feature="skeletal_length")/1000
        
        if verbose:
            print(f"total_sk_distance = {total_sk_distance}, total_area = {total_area}")
        
    #4) Delete the brnaches from the neuron
    new_neuron = pru.delete_branches_from_neuron(neuron_obj,
                                    limb_branch_dict = removed_limb_branch_dict,
                                    plot_neuron_after_cancellation = False,
                                    plot_final_neuron = False,
                                    verbose = False)

    if plot_final_neuron:
        nviz.visualize_neuron(new_neuron,
                             visualize_type=["mesh"],
                             mesh_whole_neuron=True)
    
    if return_error_info:
        return new_neuron,total_area,total_sk_distance
    else:
        return new_neuron
    
    
import copy

import neuron_searching as ns
def filter_away_axon_on_dendrite_merges(
    neuron_obj,
    perform_deepcopy = True,
    
    axon_merge_error_limb_branch_dict = None,
    perform_axon_classification = False,
    use_pre_existing_axon_error_labels = False,

    return_error_info=True,

    plot_limb_branch_filter_away = False,
    plot_limb_branch_filter_with_disconnect_effect=False,
    plot_final_neuron = False,

    verbose = False):

    """
    Pseudocode: 

    If error labels not given
    1a) Apply axon classification if requested
    1b) Use the pre-existing error labels if requested

    2) Find the total branches that will be removed using the axon-error limb branch dict
    3) Calculate the total skeleton length and error faces area for what will be removed
    4) Delete the brnaches from the neuron
    5) Return the neuron

    Example: 
    
    filter_away_axon_on_dendrite_merges(
    neuron_obj = neuron_obj_1,
    perform_axon_classification = True,
    return_error_info=True,
    verbose = True)
    """
    if perform_deepcopy:
        neuron_obj = copy.deepcopy(neuron_obj)

    if axon_merge_error_limb_branch_dict is None:
        if perform_axon_classification == False and use_pre_existing_axon_error_labels == False:
            raise Exception("Need to set either perform_axon_classification or use_pre_existing_axon_error_labels because"
                           f" axon_merge_error_limb_branch_dict is None")

        if use_pre_existing_axon_error_labels:

            if verbose:
                print("using pre-existing labels for axon-error detection")

            axon_merge_error_limb_branch_dict = ns.query_neuron_by_labels(neuron_obj,matching_labels=["axon-error"])
        else:

            if verbose:
                print("performing axon classification for axon-error detection")


            axon_limb_branch_dict,axon_merge_error_limb_branch_dict = clu.axon_classification(neuron_obj,
                            return_axon_labels=True,
                            return_error_labels=True,
                           plot_axons=False,
                           plot_axon_errors=False,
                            verbose=True,
                           )


    new_neuron,total_area,total_sk_distance = filter_away_limb_branch_dict(neuron_obj,
                                     limb_branch_dict=axon_merge_error_limb_branch_dict,
                                    plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                                     plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                                    return_error_info=True,
                                     plot_final_neuron=plot_final_neuron,
                                     verbose=verbose
                                    )
    
    if return_error_info:
        return new_neuron,total_area,total_sk_distance
    else:
        return new_neuron

#---------- Rule 2: Removing Dendritic Merges on Axon ------------- #

import copy
import neuron_searching as ns
def filter_away_dendrite_on_axon_merges(
    neuron_obj,
    perform_deepcopy=True,
    
    limb_branch_dict_for_search=None,
    use_pre_existing_axon_labels=False,
    perform_axon_classification=False,
    
    dendritic_merge_on_axon_query=None,
    dendrite_merge_skeletal_length_min = 20000,
    dendrite_merge_width_min = 100,
    dendritie_spine_density_min = 0.00015,
    
    plot_limb_branch_filter_away = False,
    plot_limb_branch_filter_with_disconnect_effect = False,
    return_error_info = True,
    plot_final_neuron = False,
    verbose=False,
    ):
    """
    Purpose: To filter away the dendrite parts that are 
    merged onto axon pieces
    
    if limb_branch_dict_for_search is None then 
    just going to try and classify the axon and then 
    going to search from there
    
    
    """
    
    
    if perform_deepcopy:
        neuron_obj = copy.deepcopy(neuron_obj)

    if limb_branch_dict_for_search is None:
        if use_pre_existing_axon_labels == False and perform_axon_classification == False:
            raise Exception("Need to set either perform_axon_classification or use_pre_existing_axon_error_labels because"
                           f" limb_branch_dict_for_search is None")
        
        axon_limb_branch_dict,axon_error_limb_branch_dict = clu.axon_classification(neuron_obj,
                                    return_error_labels=True,
                                    plot_candidates=False,
                                    plot_axons=False,
                                    plot_axon_errors=False)


    if dendritic_merge_on_axon_query is None:
        dendritic_merge_on_axon_query = (f"labels_restriction == True and "
                    f"(median_mesh_center > {dendrite_merge_width_min}) and  "
                    f"(skeletal_length > {dendrite_merge_skeletal_length_min}) and "
                    f"(spine_density) > {dendritie_spine_density_min}")


    function_kwargs = dict(matching_labels=["axon"],
                          not_matching_labels=["axon-like"])

    dendritic_branches_merged_on_axon = ns.query_neuron(neuron_obj,
                    functions_list=["labels_restriction","median_mesh_center",
                                   "skeletal_length","spine_density"],
                    query=dendritic_merge_on_axon_query,
                    function_kwargs=function_kwargs)

    if verbose:
        print(f"dendritic_branches_merged_on_axon = {dendritic_branches_merged_on_axon}")
        
        
    

    (dendrite_stripped_neuron,
    total_area_dendrite_stripped,
     total_sk_distance_stripped) = pru.filter_away_limb_branch_dict(neuron_obj,
                                         limb_branch_dict=dendritic_branches_merged_on_axon,
                                        plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                                         plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                                        return_error_info=True,
                                         plot_final_neuron=plot_final_neuron,
                                         verbose=verbose
                                        )
    
    if return_error_info:
        return (dendrite_stripped_neuron,
                total_area_dendrite_stripped,
                 total_sk_distance_stripped)
    else:
        return dendrite_stripped_neuron
    
    
# ------------ Rule 3: Filtering away axon mess ----------- #

def filter_away_limb_branch_dict_with_function(
    neuron_obj,
    limb_branch_dict_function,
    perform_deepcopy=True,
    
    
    plot_limb_branch_filter_away = False,
    plot_limb_branch_filter_with_disconnect_effect = False,
    return_error_info = True,
    plot_final_neuron = False,
    verbose=False,
    
    **kwargs #These argument will be for running the function that will come up with limb branch dict
    ):
    """
    Purpose: To filter away a limb branch dict from
    a neuron using a function that generates a limb branch dict
    
    """
    
    
    if perform_deepcopy:
        neuron_obj = copy.deepcopy(neuron_obj)

    limb_branch_dict_to_cancel = limb_branch_dict_function(neuron_obj,
                             verbose=verbose,
                             **kwargs)
        

    (dendrite_stripped_neuron,
    total_area_dendrite_stripped,
     total_sk_distance_stripped) = pru.filter_away_limb_branch_dict(neuron_obj,
                                         limb_branch_dict=limb_branch_dict_to_cancel,
                                        plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                                         plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                                        return_error_info=True,
                                         plot_final_neuron=plot_final_neuron,
                                         verbose=verbose
                                        )
    
    if return_error_info:
        return (dendrite_stripped_neuron,
                total_area_dendrite_stripped,
                 total_sk_distance_stripped)
    else:
        return dendrite_stripped_neuron

def filter_away_low_branch_length_clusters(neuron_obj,
                                           max_skeletal_length=8000,
                                           min_n_nodes_in_cluster=4,
                                           return_error_info=True,
                                           plot_limb_branch_filter_with_disconnect_effect=False,
                                           plot_limb_branch_filter_away=False,
                                           plot_final_neuron=False,
                                           **kwargs):
    
    return filter_away_limb_branch_dict_with_function(neuron_obj,
                 limb_branch_dict_function=nru.low_branch_length_clusters,
                 max_skeletal_length=max_skeletal_length,
                 min_n_nodes_in_cluster=min_n_nodes_in_cluster,
                 return_error_info=return_error_info,
                plot_limb_branch_filter_away=plot_limb_branch_filter_away,
                plot_limb_branch_filter_with_disconnect_effect=plot_limb_branch_filter_with_disconnect_effect,
                 plot_final_neuron=plot_final_neuron,
                 **kwargs)
    
    
import proofreading_utils as pru
    

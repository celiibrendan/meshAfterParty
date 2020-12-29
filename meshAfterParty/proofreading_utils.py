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
                      
                      #paraeters for high degree nodes
                      high_degree_offset = 1500,
                      comparison_distance = 2000,
                      match_threshold = 35,
                      
                      #parameter for both width and doubling back
                      # This will prevent the edges that were added to extend to the soma from causing the doulbing back or width threshold errors
                      skip_small_soma_connectors = True,
                      small_soma_connectors_skeletal_threshold = 2500,
                      
                      # parameters for the doubling back
                      double_back_threshold = 100,# 130,
                      offset = 1000,
                      
                      #parameters for the width threshold
                      width_jump_threshold = 200,
                      verbose=True,
                    **kwargs):
    """
    Purpose: To choose the best path to cut to disconnect
    a path based on the heuristic hierarchy of 
    
    Cut in descending priority
    1) high degree coordinates
    2) Doubling Back 
    3) Width Jump
    
    Pseudocode: 
    1) Get any high degree cordinates on path
    --> if there are then pick the first one and perform the cuts
    
    2) Check the doubling backs (and pick the highest one if above threshold)
    3) Check for width jumps (and pick the highest one)
    4) Record the cuts that will be made
    5) Make the alterations to the graph (can be adding and creating edges)
    """
    high_degree_endpoint_coordinates = find_high_degree_coordinates_on_path(curr_limb,cut_path)
    
    edges_to_create = None
    edges_to_delete = None
    
    resolve_crossover_at_end = True
    
    if verbose:
        print(f"Found {len(high_degree_endpoint_coordinates)} high degree coordinates to cut")
        
    if len(high_degree_endpoint_coordinates)>0:
        
        curr_high_degree_coord = high_degree_endpoint_coordinates[0]

        if verbose:
            print(f"Picking {curr_high_degree_coord} high degree coordinates to cut")

        edges_to_delete_pre,edges_to_create_pre = ed.resolving_crossovers(curr_limb,
                                                coordinate = curr_high_degree_coord,
                                                offset=high_degree_offset,
                                                comparison_distance=comparison_distance,
                                                                          match_threshold =match_threshold,
                                                verbose = verbose,
                                                **kwargs
                               )
        
        if len(edges_to_delete_pre)>0:
            edges_to_delete = edges_to_delete_pre
            edges_to_create = edges_to_create_pre
            
        resolve_crossover_at_end = False
    
        
    
    
    
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
                             offset=offset,
                            verbose = verbose,
                            skip_nodes=skip_nodes)


        if len(err_edges) > 0:

            largest_double_back = np.argmax(edges_double_back)
            winning_err_edge = edges[largest_double_back]

            if verbose:
                print(f"There were {len(err_edges)} edges that passed doubling back threshold of {double_back_threshold}")
                print(f"Winning edge {winning_err_edge} had a doubling back of {largest_double_back}")
                
            edges_to_delete = [winning_err_edge]

    if edges_to_delete is None:
        if verbose: 
            print("\nAttempting the width jump check (attempting from both sides)")
            
        possible_starting_nodes = [cut_path[0],cut_path[-1]]
        
        
        first_error_edges = []
        first_error_sizes = []
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
        Pseudocode: 
        1) Check if both error edges are not empty
        2) Get the starting error 
        
        
        """
        if (not first_error_edges[0] is None) or (not first_error_edges[1] is None):
            winning_path = np.argmax(first_error_sizes)
            winning_err_edge = first_error_edges[winning_path]
            if verbose: 
                print(f"first_error_sizes = {first_error_sizes}, winning_path = {winning_path}")
                
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
            edges_to_delete += edges_to_delete_new
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
    
    return edges_to_delete,edges_to_create,curr_limb

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
                                    verbose=verbose,
                                     return_copy=return_copy
                                    )
    
def cut_limb_network_by_edges(curr_limb,
                                    edges_to_delete=None,
                                    edges_to_create=None,
                                    return_accepted_edges_to_create=False,
                                    return_copy=True,
                                    verbose=False):
    if return_copy:
        curr_limb = copy.copy(curr_limb)
        
    if not edges_to_delete is None:
        if verbose:
            print(f"edges_to_delete = {edges_to_delete}")
        curr_limb.concept_network.remove_edges_from(edges_to_delete)
        
    #apply the winning cut
    accepted_edges_to_create = []
    if not edges_to_create is None:
        if verbose:
            print(f"edges_to_create = {edges_to_create}")
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
    if return_accepted_edges_to_create:
        return curr_limb,accepted_edges_to_create
    
    return curr_limb

def multi_soma_split_suggestions(neuron_obj,
                                verbose=True,
                                max_iterations=100,
                                 plot_suggestions=False,
                                 plot_suggestions_scatter_size=0.4,
                                **kwargs):
    """
    Purpose: To come up with suggestions for splitting a multi-soma

    Pseudocode: 

    1) Iterate through all of the limbs that need to be processed
    2) Find the suggested cuts until somas are disconnected or failed
    3) Optional: Visualize the nodes and their disconnections

    """

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
        max_iterations = 100

        #2) Get all of the starting nodes for somas
        all_starting_nodes = [k["starting_node"] for k in curr_limb_copy.all_concept_network_data]

        starting_node_combinations = list(itertools.combinations(all_starting_nodes,2))

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
            # need to keep cutting until no path for them
            if verbose:
                print(f"\n\n---- working on disconnecting {st_n_1} and {st_n_2}")
                print(f"---- This disconnects {soma_title} ")

            counter = 0
            success = False
            while True:
                if verbose:
                    print(f" Cut iteration {counter}")
                try:

                    soma_to_soma_path = np.array(nx.shortest_path(curr_limb_copy.concept_network,st_n_1,st_n_2))
                except:
                    if verbose:
                        print("No valid path so moving onto the next connection")
                    success = True
                    break

                if verbose:
                    print(f"Shortest path = {list(soma_to_soma_path)}")

                # say we found the cut node to make
                cut_edges, added_edges, curr_limb_copy = pru.get_best_cut_edge(curr_limb_copy,soma_to_soma_path,
                                                                               verbose=verbose,
                                                                              **kwargs)
                if verbose:
                    print(f"After get best cut: cut_edges = {cut_edges}, added_edges = {added_edges}")
                
                if cut_edges is None:
                    print("***** there was no suggested cut for this limb even though it is still connnected***")

                    break

                if not cut_edges is None:
                    total_soma_paths_to_cut += cut_edges
                if not added_edges is None:
                    total_soma_paths_to_add += added_edges
                
                
                
                counter += 1

                if counter > max_iterations:
                    print(f"Breaking because hit max iterations {max_iterations}")

            local_results["edges_to_delete"] = total_soma_paths_to_cut
            local_results["edges_to_create"] = total_soma_paths_to_add

            suggested_cut_points = [sk.shared_endpoint(curr_limb_copy[cut_e[0]].skeleton,
                                                curr_limb_copy[cut_e[1]].skeleton)
                                             for cut_e in total_soma_paths_to_cut]

            local_results["coordinate_suggestions"] =suggested_cut_points
            local_results["successful_disconnection"] = success
            results.append(local_results)

        seperated_graphs = list(nx.connected_components(curr_limb_copy.concept_network))
        if verbose:
            print(f"Total number of graphs at the end of the split = {len(seperated_graphs)}")

        limb_results[curr_limb_idx] = results
        
    if plot_suggestions:
        nviz.plot_split_suggestions_per_limb(neuron_obj,
                                    limb_results,
                                    scatter_size = plot_suggestions_scatter_size)
        
    return limb_results

def split_suggestions_to_concept_networks(neuron_obj,limb_results):
    """
    Will take the output of the multi_soma_split suggestions and 
    return the concept network with all fo the cuts applied
    
    """
    
    new_concept_networks = dict()
    for curr_limb_idx,path_cut_info in limb_results.items():
        limb_nx = nx.Graph(neuron_obj[curr_limb_idx].concept_network)
        for cut in path_cut_info:
            limb_nx.remove_edges_from(cut["edges_to_delete"])
            limb_nx.add_edges_from(cut["edges_to_create"])
        new_concept_networks[curr_limb_idx] = limb_nx
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

def split_neuron_limb(neuron_obj,
                     seperated_graphs,
                     curr_limb_idx,
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
    new_limb_data = []
    curr_limb = neuron_obj[curr_limb_idx]

    for seg_graph_idx,sep_G in enumerate(seperated_graphs):
        if verbose:
            print(f"\n\n----Working on seperate_graph {seg_graph_idx}----")

        curr_subgraph = list(sep_G)

        #will store all of the relevant info in the 
        sep_graph_data = dict()


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
            raise (f"There were more not exactly one starting dictinoary: {curr_all_concept_network_data} ")
            
            
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
            curr_all_concept_network_data[0]["soma_group_idx"] = 0
            curr_limb_network_stating_info = nru.all_concept_network_data_to_dict(curr_all_concept_network_data)

        #calculate the concept networks


        sep_graph_data["limb_correspondence"] = limb_corresp_for_networks

        sep_graph_data["limb_network_stating_info"] = curr_limb_network_stating_info
        
        #raise Exception("Checking on limb starting network info")

        limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(limb_corresp_for_networks,
                                                                                        curr_limb_network_stating_info,
                                                                                        run_concept_network_checks=True,
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

    
    
    
    return neuron_obj_cp






# ----------------12/27: Whole Neuron Splitting ------------------------ #

import networkx as nx
import networkx_utils as xu
import trimesh_utils as tu
import preprocessing_vp2 as pre
import neuron_visualizations as nviz

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
    
    split_neuron_obj = neuron_obj
    limb_results = split_suggestions
    
    new_concept_networks = pru.split_suggestions_to_concept_networks(neuron_obj,limb_results)

    for curr_limb_idx,curr_limb_nx in new_concept_networks.items():
        curr_limb_idx = int(curr_limb_idx)

        conn_comp = list(nx.connected_components(curr_limb_nx))

        if verbose:
            print(f"\n\n---Working on Splitting Limb {curr_limb_idx} with {len(conn_comp)} components----")

        split_neuron_obj = pru.split_neuron_limb(split_neuron_obj,conn_comp,
                             curr_limb_idx = curr_limb_idx,
                                                verbose=verbose)
        
    if plot_soma_limb_network:
        nviz.plot_soma_limb_concept_network(split_neuron_obj)
        
    return split_neuron_obj



def split_disconnected_neuron(neuron_obj,
                              plot_seperated_neurons=False,
                             verbose=False):
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
    curr_error_limbs = nru.error_limbs(split_neuron_obj)

    if len(curr_error_limbs) > 0:
        raise Exception(f"There were still error limbs before trying the neuron object split: error limbs = {curr_error_limbs}")
    
    
    
    
    # ------ Part 2: start the splitting process
    
    # get all the somas that we will split into
    soma_names = split_neuron_obj.get_soma_node_names()
    
    
    neuron_obj_list = []

    for curr_soma_idx,curr_soma_name in enumerate(soma_names):
        print(f"\n\n------ Working on Soma {curr_soma_idx} -------")

        neuron_cp = split_neuron_obj

        #getting all the soma information we will need for preprocessing
        soma_obj = neuron_cp[curr_soma_name]
        curr_soma_meshes = [soma_obj.mesh]
        curr_soma_sdfs = [soma_obj.sdf]
        curr_soma_volume_ratios = [soma_obj.volume_ratio]





        # getting the limb information and new soma connectivity
        limb_neighbors = np.sort(xu.get_neighbors(neuron_cp.concept_network,curr_soma_name)).astype("int")
        limb_neighbors = [int(k) for k in limb_neighbors]

        soma_to_piece_connectivity = neuron_cp.preprocessed_data["soma_to_piece_connectivity"][curr_soma_idx]

        if len(np.intersect1d(limb_neighbors,soma_to_piece_connectivity)) < len(soma_to_piece_connectivity):
            raise Exception(f"piece connectivity ({soma_to_piece_connectivity}) not match limb neighbors ({limb_neighbors})")

        curr_soma_to_piece_connectivity = {0:np.arange(0,len(limb_neighbors))}






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
        curr_limb_correspondence = dict([(i,neuron_cp.preprocessed_data["limb_correspondence"][k]) for i,k in enumerate(limb_neighbors)])








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









        #using all of the data to create new preprocessing info
        new_preprocessed_data = preprocessed_data= dict(
                #soma data
                soma_meshes = curr_soma_meshes,
                soma_sdfs = curr_soma_sdfs,
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
                insignificant_limbs=None,
                not_processed_soma_containing_meshes=None,
                non_soma_touching_meshes=None,
                inside_pieces=None,


                )

        limb_to_branch_objects = dict()
        for k in limb_neighbors:
            limb_obj = neuron_cp[int(k)]
            branch_dict = dict([(b,limb_obj[int(b)]) for b in limb_obj.get_branch_names()])
            limb_to_branch_objects[old_node_to_new_node_mapping[k]] = branch_dict

        segment_id = neuron_cp.segment_id
        description = f"{neuron_cp.description}_soma_{curr_soma_idx}_split"





        # new neuron object:

        single_split_neuron_obj = neuron.Neuron(mesh=whole_mesh,
                 segment_id=segment_id,
                 description=description,
                 preprocessed_data=new_preprocessed_data,
                 limb_to_branch_objects=limb_to_branch_objects,
                 widths_to_calculate=[],
                suppress_output=not verbose)


        neuron_obj_list.append(single_split_neuron_obj)
        
        
        
    # ------ Part 3: Visualize the Results
    print(f"\n\nNumber of seperate neuron objects = {len(neuron_obj_list)}")

    if plot_seperated_neurons:
        for n_obj in neuron_obj_list:
            nviz.visualize_neuron(n_obj,
                                 visualize_type=["mesh","skeleton"],
                                 limb_branch_dict="all")


    return neuron_obj_list



def split_neuron(neuron_obj,
                 plot_soma_limb_network=False,
                 plot_seperated_neurons=False,
                verbose=False):
    """
    Purpose: To take in a whole neuron that could have any number of somas
    and then to split it into multiple neuron objects

    Pseudocode: 
    1) Get all of the split suggestions
    2) Split all of the limbs that need splitting
    3) Once have split the limbs, split the neuron object into mutliple objects


    """
    
    #1) Get all of the split suggestions
    limb_results = pru.multi_soma_split_suggestions(neuron_obj,plot_intermediates=False,
                                               verbose = verbose)
    
    #2) Split all of the limbs that need splitting
    split_neuron_obj = pru.split_neuron_limbs_by_suggestions(neuron_obj,
                                split_suggestions=limb_results,
                                plot_soma_limb_network=plot_soma_limb_network,
                                verbose=verbose)
        
    #2b) Check that all the splits occured
    curr_error_limbs = nru.error_limbs(split_neuron_obj)

    if len(curr_error_limbs) > 0:
        raise Exception(f"There were still error limbs before trying the neuron object split: error limbs = {curr_error_limbs}")
    
    #3) Once have split the limbs, split the neuron object into mutliple objects
    neuron_list = pru.split_disconnected_neuron(split_neuron_obj,
                         plot_seperated_neurons=True,
                         verbose =verbose)
        
    return neuron_list



import proofreading_utils as pru
    

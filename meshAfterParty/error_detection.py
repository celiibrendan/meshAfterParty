import time
import copy
import neuron
import neuron_utils as nru
from tqdm_utils import tqdm

def width_jump_edges(limb,
                    width_type = "no_spine_median_mesh_center",
                     width_jump_threshold = 100,
                     verbose=False,
                     path_to_check = None,
                    ):
    """
    Will only look to see if the width jumps up by a width_jump_threshold threshold ammount
    and if it does then will save the edges according to that starting soma group
    
    Example: 
    ed = reload(ed)
    ed.width_jump_edges(neuron_obj[5],verbose=True)
    """
    curr_limb = copy.deepcopy(limb)


    width_start_time = time.time()

    error_edges = dict()
    for k in curr_limb.all_concept_network_data:
        curr_soma = k["starting_soma"]
        curr_soma_group = k["soma_group_idx"]
        
        if verbose:
            print(f"Working on Soma {curr_soma} and Soma touching group {curr_soma_group}")
        
        
        curr_limb.set_concept_network_directional(starting_soma=curr_soma,
                                                 soma_group_idx=curr_soma_group)
        curr_net = curr_limb.concept_network_directional

        if verbose: 
            print(f'Working on soma group {k["soma_group_idx"]}')

        curr_error_edges = []
        for current_nodes in tqdm(curr_net.edges()):
            if not path_to_check is None:
                if len(np.intersect1d(current_nodes,path_to_check)) < 2:
#                     if verbose:
#                         print(f"Skipping edge {current_nodes} because not on path to check: {path_to_check}")
                    continue
            if verbose:
                print(f"  Edge: {current_nodes}")

            up_width,d_width,up_sk,d_sk = nru.branch_boundary_transition(curr_limb,
                                  edge=current_nodes,
                                #offset=0,
                                verbose=False)

            downstream_jump = d_width-up_width

            if downstream_jump > width_jump_threshold:
                if verbose:
                    print(f"Adding error edge {current_nodes} because width jump was {downstream_jump}")
                curr_error_edges.append(list(current_nodes))
        
        if curr_soma not in error_edges.keys():
            error_edges[curr_soma] = dict()
        error_edges[curr_soma][curr_soma_group] = curr_error_edges
        
    if verbose: 
        print(f"Total time for width = {time.time() - width_start_time}")
    return error_edges

def path_to_edges(path):
    return np.vstack([path[:-1],path[1:]]).T

def width_jump_edges_path(limb, #assuming the concept network is already set
                          path_to_check,
                    width_type = "no_spine_median_mesh_center",
                     width_jump_threshold = 100,
                     verbose=False,
                          return_all_edge_info = True,
                          offset=1000,
                    ):
    """
    Will only look to see if the width jumps up by a width_jump_threshold threshold ammount
    
    **but only along a certain path**
    
    
    Example: 
    curr_limb.set_concept_network_directional(starting_node = 4)
    err_edges,edges,edges_width_jump = ed.width_jump_edges_path(curr_limb,
                            path_to_check=np.flip(soma_to_soma_path),
                                        width_jump_threshold=200  )

    err_edges,edges,edges_width_jump
    """


    width_start_time = time.time()
    curr_net = limb.concept_network_directional
    edges = path_to_edges(path_to_check)
    edges_width_jump = []
    error_edges = []
    
    
    for current_nodes in edges:
        if verbose:
            print(f"  Edge: {current_nodes}")

        up_width,d_width,up_sk,d_sk = nru.branch_boundary_transition(limb,
                              edge=current_nodes,
                            offset=1000,
                            verbose=False)

        downstream_jump = d_width-up_width
        edges_width_jump.append(downstream_jump)
        
        if downstream_jump >= width_jump_threshold:
            if verbose:
                print(f"Adding error edge {current_nodes} because width jump was {downstream_jump}")
            error_edges.append(list(current_nodes))

    
    edges_width_jump = np.array(edges_width_jump)
    if verbose: 
        print(f"Total time for width = {time.time() - width_start_time}")
    if return_all_edge_info:
        return error_edges,edges,edges_width_jump
    else:
        return error_edges



import skeleton_utils as sk
import numpy_utils as nu
def double_back_edges(
    limb,
    double_back_threshold = 130,
    verbose = True,
    comparison_distance=3000,
    offset=0,
    path_to_check=None):

    """
    Purpose: To get all of the edges where the skeleton doubles back on itself

    Application: For error detection


    """

    curr_limb = copy.deepcopy(limb)


    width_start_time = time.time()

    error_edges = dict()
    for k in curr_limb.all_concept_network_data:
        curr_soma = k["starting_soma"]
        curr_soma_group = k["soma_group_idx"]

        if verbose:
            print(f"Working on Soma {curr_soma} and Soma touching group {curr_soma_group}")


        curr_limb.set_concept_network_directional(starting_soma=curr_soma,
                                                 soma_group_idx=curr_soma_group)
        curr_net = curr_limb.concept_network_directional

        if verbose: 
            print(f'Working on soma group {k["soma_group_idx"]}')

        curr_error_edges = []
        for current_nodes in tqdm(curr_net.edges()):
            if verbose:
                print(f"  Edge: {current_nodes}")
                
            if not path_to_check is None:
                if len(np.intersect1d(current_nodes,path_to_check)) < 2:
#                     if verbose:
#                         print(f"Skipping edge {current_nodes} because not on path to check: {path_to_check}")
                    continue

            up_width,d_width,up_sk,d_sk = nru.branch_boundary_transition(curr_limb,
                                  edge=current_nodes,
                                  comparison_distance = comparison_distance,
                                offset=offset,
                                verbose=False)

            """
            Pseudocode:
            1) Flip the upstream skeleton (the downstream one should be in right direction)
            2) Get the endpoints from first and last of skeleton coordinates for both to find the vectors
            3) Find the angle between them

            """
            up_sk_flipped = sk.flip_skeleton(up_sk)

            up_vec = up_sk_flipped[-1][-1] - up_sk_flipped[0][0] 
            d_vec = d_sk[-1][-1] - d_sk[0][0]

            curr_angle = nu.angle_between_vectors(up_vec,d_vec)

            if curr_angle > double_back_threshold:
                curr_error_edges.append(list(current_nodes))

        if curr_soma not in error_edges.keys():
            error_edges[curr_soma] = dict()
        error_edges[curr_soma][curr_soma_group] = curr_error_edges

    if verbose: 
        print(f"Total time for width = {time.time() - width_start_time}")
    
    return error_edges



def double_back_edges_path(
    limb,
    path_to_check,
    double_back_threshold = 130,
    verbose = True,
    comparison_distance=3000,
    offset=0,
    return_all_edge_info = True):

    """
    Purpose: To get all of the edges where the skeleton doubles back on itself
    **but only along a certain path**

    Application: For error detection
    
    
    Example: 
    curr_limb.set_concept_network_directional(starting_node = 2)
    err_edges,edges,edges_width_jump = ed.double_back_edges_path(curr_limb,
                            path_to_check=soma_to_soma_path )

    err_edges,edges,edges_width_jump


    """

    curr_limb = limb


    width_start_time = time.time()
    
    curr_net = limb.concept_network_directional
    edges = path_to_edges(path_to_check)
    edges_doubling_back = []
    error_edges = []
    
    
    for current_nodes in tqdm(edges):
        if verbose:
            print(f"  Edge: {current_nodes}")


        up_width,d_width,up_sk,d_sk = nru.branch_boundary_transition(curr_limb,
                              edge=current_nodes,
                              comparison_distance = comparison_distance,
                            offset=offset,
                            verbose=False)

        """
        Pseudocode:
        1) Flip the upstream skeleton (the downstream one should be in right direction)
        2) Get the endpoints from first and last of skeleton coordinates for both to find the vectors
        3) Find the angle between them

        """
        up_sk_flipped = sk.flip_skeleton(up_sk)

        up_vec = up_sk_flipped[-1][-1] - up_sk_flipped[0][0] 
        d_vec = d_sk[-1][-1] - d_sk[0][0]

        curr_angle = nu.angle_between_vectors(up_vec,d_vec)
        edges_doubling_back.append(curr_angle)
        
        
        if curr_angle > double_back_threshold:
            error_edges.append(list(current_nodes))

    if verbose: 
        print(f"Total time for doubling_back = {time.time() - width_start_time}")
    if return_all_edge_info:
        return error_edges,edges,edges_doubling_back
    else:
        return error_edges


            
    
import neuron_utils as nru
import numpy_utils as nu
import networkx as nx
import matplotlib.pyplot as plt
def resolving_crossovers(limb_obj,
                        coordinate,
                        match_threshold = 35,
                        verbose = True,
                         return_new_edges = True,
                        return_subgraph=False,
                        plot_intermediates=False,
                         offset=1000,
                         comparison_distance = 1000,
                        **kwargs):
    
    """
    Purpose: To determine the connectivity that should be at the location
    of a crossover (the cuts that should be made and the new connectivity)

    Pseudocode: 
    1) Get all the branches that correspond to the coordinate
    2) For each branch
    - get the boundary cosine angle between the other branches
    - if within a threshold then add edge
    3) Ge the subgraph of all these branches:
    - find what edges you have to cut
    4) Return the cuts/subgraph
    
    Ex: 
    resolving_crossovers(limb_obj = copy.deepcopy(curr_limb),
                     coordinate = high_degree_coordinates[0],
                    match_threshold = 40,
                    verbose = False,
                     return_new_edges = True,
                    return_subgraph=True,
                    plot_intermediates=False)

    """
    
    #1) Get all the branches that correspond to the coordinate
    sk_branches = [br.skeleton for br in limb_obj]

    coordinate_branches = np.sort(sk.find_branch_skeleton_with_specific_coordinate(sk_branches,coordinate))
    curr_colors = ["red","aqua","purple","green"]
    
    if verbose: 
        print(f"coordinate_branches = {coordinate_branches}")
        for c,col in zip(coordinate_branches,curr_colors):
            print(f"{c} = {col}")
    
    
    
    if plot_intermediates:
        nviz.plot_objects(meshes=[limb_obj[k].mesh for k in coordinate_branches],
                         meshes_colors=curr_colors,
                         skeletons=[limb_obj[k].skeleton for k in coordinate_branches],
                         skeletons_colors=curr_colors)
    
    # 2) For each branch
    # - get the boundary cosine angle between the other branches
    # - if within a threshold then add edge

    match_branches = []
    all_aligned_skeletons = []
    for br1_idx in coordinate_branches:
        for br2_idx in coordinate_branches:
            if br1_idx>=br2_idx:
                continue

                
            edge = [br1_idx,br2_idx]
            edge_skeletons = [sk_branches[e] for e in edge]
            aligned_sk_parts = sk.offset_skeletons_aligned_at_shared_endpoint(edge_skeletons,
                                                                             offset=offset,
                                                                             comparison_distance=comparison_distance)
            

            curr_angle = sk.parent_child_skeletal_angle(aligned_sk_parts[0],aligned_sk_parts[1])
            
            
            if verbose:
                print(f"Angle between {br1_idx} and {br2_idx} = {curr_angle} ")

            # - if within a threshold then add edge
            if curr_angle <= match_threshold:
                match_branches.append([br1_idx,br2_idx])
                
            if plot_intermediates:
                #saving off the aligned skeletons to visualize later
                all_aligned_skeletons.append(aligned_sk_parts[0])
                all_aligned_skeletons.append(aligned_sk_parts[1])
    
    if verbose: 
        print(f"Final Matches = {match_branches}")
        
    if plot_intermediates:
        print("Aligned Skeleton Parts")
        nviz.plot_objects(meshes=[limb_obj[k].mesh for k in coordinate_branches],
                         meshes_colors=curr_colors,
                         skeletons=all_aligned_skeletons)
    
    if plot_intermediates:
        for curr_match in match_branches:
            nviz.plot_objects(meshes=[limb_obj[k].mesh for k in curr_match],
                 meshes_colors=curr_colors,
                 skeletons=[limb_obj[k].skeleton for k in curr_match],
                 skeletons_colors=curr_colors)
            
    
    # find what cuts and connections need to make
    limb_subgraph = limb_obj.concept_network.subgraph(coordinate_branches)
    
    if verbose:
        print("Original graph")
        nx.draw(limb_subgraph,with_labels=True)
        plt.show()
    
    
    sorted_edges = np.sort(limb_subgraph.edges(),axis=1)
    if len(match_branches)>0:
        
        sorted_confirmed_edges = np.sort(match_branches,axis=1)


        edges_to_delete = []

        for ed in sorted_edges:
            if len(nu.matching_rows_old(sorted_confirmed_edges,ed))==0:
                edges_to_delete.append(ed)

        edges_to_create = []

        for ed in sorted_confirmed_edges:
            if len(nu.matching_rows_old(sorted_edges,ed))==0:
                edges_to_create.append(ed)
    else:
        edges_to_delete = sorted_edges
        edges_to_create = []
            
    if verbose: 
        print(f"edges_to_delete = {edges_to_delete}")
        print(f"edges_to_create = {edges_to_create}")
    
    return_value = [edges_to_delete] 
    if return_new_edges:
        return_value.append(edges_to_create)
    if return_subgraph:
        #actually creating the new sugraph
        limb_obj.concept_network.remove_edges_from(edges_to_delete)
        limb_obj.concept_network.add_edges_from(edges_to_create)
        
        if verbose:
            print(f"n_components in adjusted graph = {nx.number_connected_components(limb_obj.concept_network)}")
        return_value.append(limb_obj.concept_network)
        
    return return_value





# ------------ part that will error all floating axon pieces ----------- #

import networkx as nx
import networkx_utils as xu
import skeleton_utils as sk
import numpy_utils as nu
import copy
import trimesh_utils as tu
import numpy as np
import axon_utils as au
import neuron_utils as nru
import neuron_visualizations as nviz

def error_branches_by_axons(neuron_obj,verbose=False,visualize_errors_at_end=False,
                        min_skeletal_path_threshold = 15000,
                                sub_skeleton_length = 20000,
                                ais_angle_threshold = 110,
                                non_ais_angle_threshold = 65):
    
    if neuron_obj.n_limbs == 0:
        if return_axon_non_axon_faces:
            axon_faces = np.array([])
            non_axon_faces = np.arange(len(neuron_obj.mesh.faces))
            return np.array([]),axon_faces,non_axon_faces
        return np.array([])
    
    axon_seg_dict = au.axon_like_segments(neuron_obj,include_ais=False,
                                          filter_away_end_false_positives=True,
                                          visualize_at_end=False,
                                         )

    # Step 2: Get the branches that should not be considered for axons


    to_keep_limb_names = nru.filter_limbs_below_soma_percentile(neuron_obj,verbose=False)

    axons_to_consider = dict([(k,v) for k,v in axon_seg_dict.items() if k in to_keep_limb_names])
    axons_to_not_keep = dict([(k,v) for k,v in axon_seg_dict.items() if k not in to_keep_limb_names])
    

    if verbose:
        print(f"Axons not keeping because of soma: {axons_to_not_keep}")

    # Step 3: Erroring out the axons based on projections

    valid_axon_branches_by_limb = dict()
    not_valid_axon_branches_by_limb = dict()


    
    axon_vector = np.array([0,1,0])

    for curr_limb_name,curr_axon_nodes in axons_to_consider.items():
        if verbose:
            print(f"\n----- Working on {curr_limb_name} ------")
        # curr_limb_name = "L0"
        # curr_axon_nodes = axons_to_consider[curr_limb_name]
        curr_limb_idx = int(curr_limb_name[1:])
        curr_limb = neuron_obj[curr_limb_idx]


        #1) Get the nodes that are axons


        #2) Group into connected components
        curr_limb_network = nx.from_edgelist(curr_limb.concept_network.edges())
        axon_subgraph = curr_limb_network.subgraph(curr_axon_nodes)
        axon_connected_components = list(nx.connected_components(axon_subgraph))

        valid_axon_branches = []

        #3) Iterate through the connected components
        for ax_idx,ax_group in enumerate(axon_connected_components):
            valid_flag = False
            if verbose:
                print(f"-- Axon Group {ax_idx} of size {len(ax_group)}--")
            for soma_idx in curr_limb.touching_somas():
                all_start_node = nru.all_starting_attr_by_limb_and_soma(curr_limb,soma_idx,"starting_node")
                all_start_coord = nru.all_starting_attr_by_limb_and_soma(curr_limb,soma_idx,"starting_coordinate")
                for start_node,start_coord in zip(all_start_node,all_start_coord):
                    if verbose:
                        print(f"   Working on soma {soma_idx}, starting_node {start_node}")

                    #find the shortest path between the axon group and the starting node
                    current_shortest_path,st_node,end_node = xu.shortest_path_between_two_sets_of_nodes(curr_limb_network,[start_node],list(ax_group))
                    

                    #get the skeleton of the path
                    path_skeletons = sk.stack_skeletons([curr_limb[k].skeleton for k in current_shortest_path])

                    #order the skeleton by a certain coordinate
                    ordered_path_skeleton = sk.order_skeleton(path_skeletons,start_endpoint_coordinate=start_coord)

                    #check and see if skeletal distance is lower than distance check and if it is then use a different angle check
                    if sk.calculate_skeleton_distance(ordered_path_skeleton)< min_skeletal_path_threshold:
                        if verbose:
                            print(f"Using AIS angle threshold {ais_angle_threshold}")
                        curr_angle_threshold = ais_angle_threshold
                    else:
                        if verbose:
                            print("Not using AIS angle threshold")
                        curr_angle_threshold = non_ais_angle_threshold


                    #get the first skeletal distance of threshold
                    keep_skeleton_indices = np.where(sk.calculate_skeleton_segment_distances(ordered_path_skeleton)<=sub_skeleton_length)[0]

                    
                    restricted_skeleton = ordered_path_skeleton[keep_skeleton_indices]
                    restricted_skeleton_endpoints_sk = np.array([restricted_skeleton[0][0],restricted_skeleton[-1][-1]]).reshape(-1,2,3)
                    restricted_skeleton_vector = np.array(restricted_skeleton[-1][-1]-restricted_skeleton[0][0])
                    restricted_skeleton_vector = restricted_skeleton_vector/np.linalg.norm(restricted_skeleton_vector)

                    #angle between going down and skeleton vector
                    sk_angle = nu.angle_between_vectors(axon_vector,restricted_skeleton_vector)
                    if verbose:
                        print(f"sk_angle= {sk_angle}")

                    if sk_angle > curr_angle_threshold:
                        if verbose:
                            print("*****Path to axon group not valid******")
                    else:
                        if verbose:
                            pass
                            #print("Path to axon group valid so adding them as valid axon segments")
                        valid_axon_branches.append(list(ax_group))
                        valid_flag = True
                        break

    #                 if curr_limb_name == "L1":
    #                     raise Exception()

                if valid_flag:
                    break


        
            

        
        if len(valid_axon_branches) > 0:
            valid_axon_branches_by_limb[curr_limb_name] = np.concatenate(valid_axon_branches)
            not_valid_axon_branches_by_limb[curr_limb_name] = list(np.setdiff1d(curr_axon_nodes,np.concatenate(valid_axon_branches)))
        else:
            valid_axon_branches_by_limb[curr_limb_name] = []
            not_valid_axon_branches_by_limb[curr_limb_name] = list(curr_axon_nodes)
        
        if verbose:
            print(f"\n\nFor limb {curr_limb_idx} the valid axon branches are {valid_axon_branches_by_limb[curr_limb_name] }")
            print(f"The following are not valid: {not_valid_axon_branches_by_limb[curr_limb_name]}")

    # Step 4: Compiling all the errored faces


    final_error_axons = copy.copy(axons_to_not_keep)
    final_error_axons.update(not_valid_axon_branches_by_limb)
    
    if verbose:
        print(f"final_error_axons = {final_error_axons}")
    
    if visualize_errors_at_end:
        nviz.visualize_neuron(neuron_obj,
                              visualize_type=["mesh"],
                              limb_branch_dict=final_error_axons,
                             mesh_color="red",
                             mesh_whole_neuron=True)
        
    return final_error_axons


def error_faces_by_axons(neuron_obj,error_branches = None,
                         verbose=False,visualize_errors_at_end=False,
                        min_skeletal_path_threshold = 15000,
                                sub_skeleton_length = 20000,
                                ais_angle_threshold = 110,
                                non_ais_angle_threshold = 65,
                         return_axon_non_axon_faces=False):
    """
    Purpose: Will return the faces that are errors after computing 
    the branches that are errors
    
    
    
    """
    
    if error_branches is None:
        final_error_axons = error_branches_by_axons(neuron_obj,verbose=verbose,
                                                    visualize_errors_at_end=False,
                        min_skeletal_path_threshold = min_skeletal_path_threshold,
                                sub_skeleton_length = sub_skeleton_length,
                                ais_angle_threshold = ais_angle_threshold,
                                non_ais_angle_threshold = non_ais_angle_threshold)
    else:
        final_error_axons = error_branches
        
    
    # Step 5: Getting all of the errored faces
    error_faces = []
    for curr_limb_name,error_branch_idx in final_error_axons.items():
        curr_limb = neuron_obj[curr_limb_name]
        curr_error_faces = tu.original_mesh_faces_map(neuron_obj.mesh,
                                                            [curr_limb[k].mesh for k in error_branch_idx],
                                       matching=True,
                                       print_flag=False)


        #curr_error_faces = np.concatenate([new_limb_mesh_face_idx[curr_limb[k].mesh_face_idx] for k in error_branch_idx])
        error_faces.append(curr_error_faces)

    if len(error_faces) > 0:
        error_faces_concat = np.concatenate(error_faces)
    else:
        error_faces_concat = error_faces
        
    error_faces_concat = np.array(error_faces_concat).astype("int")
        
    if verbose:
        print(f"\n\n -------- Total number of error faces = {len(error_faces_concat)} --------------")

    if visualize_errors_at_end:
        nviz.plot_objects(main_mesh = neuron_obj.mesh,
            meshes=[neuron_obj.mesh.submesh([error_faces_concat],append=True)],
                         meshes_colors=["red"])
        
    if return_axon_non_axon_faces:
        if verbose:
            print("Computing the axon and non-axonal faces")
        axon_faces = nru.limb_branch_dict_to_faces(neuron_obj,valid_axon_branches_by_limb)
        non_axon_faces = np.setdiff1d(np.arange(len(neuron_obj.mesh.faces)),axon_faces)
        return error_faces_concat,axon_faces,non_axon_faces
        
    return error_faces_concat

''' Old Way that did not have the function split up
def error_faces_by_axons(neuron_obj,verbose=False,visualize_errors_at_end=False,
                        min_skeletal_path_threshold = 15000,
                                sub_skeleton_length = 20000,
                                ais_angle_threshold = 110,
                                non_ais_angle_threshold = 50,
                         return_axon_non_axon_faces=False):
    
    if neuron_obj.n_limbs == 0:
        if return_axon_non_axon_faces:
            axon_faces = np.array([])
            non_axon_faces = np.arange(len(neuron_obj.mesh.faces))
            return np.array([]),axon_faces,non_axon_faces
        return np.array([])
    
    axon_seg_dict = au.axon_like_segments(neuron_obj,include_ais=False,
                                          filter_away_end_false_positives=True,
                                          visualize_at_end=False,
                                         )

    # Step 2: Get the branches that should not be considered for axons


    to_keep_limb_names = nru.filter_limbs_below_soma_percentile(neuron_obj,verbose=False)

    axons_to_consider = dict([(k,v) for k,v in axon_seg_dict.items() if k in to_keep_limb_names])
    axons_to_not_keep = dict([(k,v) for k,v in axon_seg_dict.items() if k not in to_keep_limb_names])
    

    if verbose:
        print(f"Axons not keeping because of soma: {axons_to_not_keep}")

    # Step 3: Erroring out the axons based on projections

    valid_axon_branches_by_limb = dict()
    not_valid_axon_branches_by_limb = dict()


    
    axon_vector = np.array([0,1,0])

    for curr_limb_name,curr_axon_nodes in axons_to_consider.items():
        if verbose:
            print(f"\n----- Working on {curr_limb_name} ------")
        # curr_limb_name = "L0"
        # curr_axon_nodes = axons_to_consider[curr_limb_name]
        curr_limb_idx = int(curr_limb_name[1:])
        curr_limb = neuron_obj[curr_limb_idx]


        #1) Get the nodes that are axons


        #2) Group into connected components
        curr_limb_network = nx.from_edgelist(curr_limb.concept_network.edges())
        axon_subgraph = curr_limb_network.subgraph(curr_axon_nodes)
        axon_connected_components = list(nx.connected_components(axon_subgraph))

        valid_axon_branches = []

        #3) Iterate through the connected components
        for ax_idx,ax_group in enumerate(axon_connected_components):
            valid_flag = False
            if verbose:
                print(f"-- Axon Group {ax_idx} of size {len(ax_group)}--")
            for soma_idx in curr_limb.touching_somas():
                all_start_node = nru.all_starting_attr_by_limb_and_soma(curr_limb,soma_idx,"starting_node")
                all_start_coord = nru.all_starting_attr_by_limb_and_soma(curr_limb,soma_idx,"starting_coordinate")
                for start_node,start_coord in zip(all_start_node,all_start_coord):
                    if verbose:
                        print(f"   Working on soma {soma_idx}, starting_node {start_node}")

                    #find the shortest path between the axon group and the starting node
                    current_shortest_path,st_node,end_node = xu.shortest_path_between_two_sets_of_nodes(curr_limb_network,[start_node],list(ax_group))
                    

                    #get the skeleton of the path
                    path_skeletons = sk.stack_skeletons([curr_limb[k].skeleton for k in current_shortest_path])

                    #order the skeleton by a certain coordinate
                    ordered_path_skeleton = sk.order_skeleton(path_skeletons,start_endpoint_coordinate=start_coord)

                    #check and see if skeletal distance is lower than distance check and if it is then use a different angle check
                    if sk.calculate_skeleton_distance(ordered_path_skeleton)< min_skeletal_path_threshold:
                        if verbose:
                            print(f"Using AIS angle threshold {ais_angle_threshold}")
                        curr_angle_threshold = ais_angle_threshold
                    else:
                        if verbose:
                            print("Not using AIS angle threshold")
                        curr_angle_threshold = non_ais_angle_threshold


                    #get the first skeletal distance of threshold
                    keep_skeleton_indices = np.where(sk.calculate_skeleton_segment_distances(ordered_path_skeleton)<=sub_skeleton_length)[0]

                    
                    restricted_skeleton = ordered_path_skeleton[keep_skeleton_indices]
                    restricted_skeleton_endpoints_sk = np.array([restricted_skeleton[0][0],restricted_skeleton[-1][-1]]).reshape(-1,2,3)
                    restricted_skeleton_vector = np.array(restricted_skeleton[-1][-1]-restricted_skeleton[0][0])
                    restricted_skeleton_vector = restricted_skeleton_vector/np.linalg.norm(restricted_skeleton_vector)

                    #angle between going down and skeleton vector
                    sk_angle = nu.angle_between_vectors(axon_vector,restricted_skeleton_vector)
                    if verbose:
                        print(f"sk_angle= {sk_angle}")

                    if sk_angle > curr_angle_threshold:
                        if verbose:
                            print("*****Path to axon group not valid******")
                    else:
                        if verbose:
                            pass
                            #print("Path to axon group valid so adding them as valid axon segments")
                        valid_axon_branches.append(list(ax_group))
                        valid_flag = True
                        break

    #                 if curr_limb_name == "L1":
    #                     raise Exception()

                if valid_flag:
                    break


        
            

        
        if len(valid_axon_branches) > 0:
            valid_axon_branches_by_limb[curr_limb_name] = np.concatenate(valid_axon_branches)
            not_valid_axon_branches_by_limb[curr_limb_name] = list(np.setdiff1d(curr_axon_nodes,np.concatenate(valid_axon_branches)))
        else:
            valid_axon_branches_by_limb[curr_limb_name] = []
            not_valid_axon_branches_by_limb[curr_limb_name] = list(curr_axon_nodes)
        
        if verbose:
            print(f"\n\nFor limb {curr_limb_idx} the valid axon branches are {valid_axon_branches_by_limb[curr_limb_name] }")
            print(f"The following are not valid: {not_valid_axon_branches_by_limb[curr_limb_name]}")

    # Step 4: Compiling all the errored faces


    final_error_axons = copy.copy(axons_to_not_keep)
    final_error_axons.update(not_valid_axon_branches_by_limb)
    
    if verbose:
        print(f"final_error_axons = {final_error_axons}")


    # Step 5: Getting all of the errored faces

    error_faces = []
    for curr_limb_name,error_branch_idx in final_error_axons.items():
        curr_limb = neuron_obj[curr_limb_name]
        curr_error_faces = tu.original_mesh_faces_map(neuron_obj.mesh,
                                                            [curr_limb[k].mesh for k in error_branch_idx],
                                       matching=True,
                                       print_flag=False)


        #curr_error_faces = np.concatenate([new_limb_mesh_face_idx[curr_limb[k].mesh_face_idx] for k in error_branch_idx])
        error_faces.append(curr_error_faces)

    if len(error_faces) > 0:
        error_faces_concat = np.concatenate(error_faces)
    else:
        error_faces_concat = error_faces
        
    error_faces_concat = np.array(error_faces_concat).astype("int")
        
    if verbose:
        print(f"\n\n -------- Total number of error faces = {len(error_faces_concat)} --------------")

    if visualize_errors_at_end:
        nviz.plot_objects(main_mesh = neuron_obj.mesh,
            meshes=[neuron_obj.mesh.submesh([error_faces_concat],append=True)],
                         meshes_colors=["red"])
        
    if return_axon_non_axon_faces:
        if verbose:
            print("Computing the axon and non-axonal faces")
        axon_faces = nru.limb_branch_dict_to_faces(neuron_obj,valid_axon_branches_by_limb)
        non_axon_faces = np.setdiff1d(np.arange(len(neuron_obj.mesh.faces)),axon_faces)
        return error_faces_concat,axon_faces,non_axon_faces
        
    return error_faces_concat

'''


from pykdtree.kdtree import KDTree
import datajoint_utils as du
def get_error_synapse_inserts(current_mesh,
                              segment_id,
                              returned_error_faces,
                              mapping_threshold = 500,
                              minnie=None,
                              return_synapse_stats=True,
                              return_synapse_centroids=False,
                              return_synapse_ids=False,
                              use_full_Synapses=False,
                              synapse_centers=None,
                              synapse_ids=None,
                              timestamps=None,
                              verbose=False):
    """
    Purpose: To Create the synapse exclude inserts for a neuron object based on the 
    synapse table and the errored faces
    
    Pseudocode: 
    0) Create face labels for the entire mesh by makeing 0 for all of them but then changing the face labels of errors to 1
    1) Build a KDTree of the mesh
    2) Download the IDs and the centroid data
    3) Query the centroid data to find the closest face on the mesh
    4) Index the closest mesh face to the face labels and that will give label of synapse
    5) Collect all synapses that are labeled error


    """
    
    
        
    if minnie is None:
        minnie,_ = du.configure_minnie_vm()
    
    if synapse_ids is None or synapse_centers is None:
        #check if there are any synapses
        if use_full_Synapses:
            segment_synapses = minnie.Synapse() & f"presyn={segment_id} OR postsyn={segment_id}"
        else:
            segment_synapses = minnie.SynapseFiltered() & f"presyn={segment_id} OR postsyn={segment_id}"
        if len(segment_synapses)<=0:
            if verbose:
                print("Returning empty list because there were no SYNAPSES")
            if return_synapse_centroids or return_synapse_ids:
                return np.array([]),np.array([])
            if return_synapse_stats:
                return [],0,0
            else:
                return []

        #2) Download the IDs and the centroid data
        synapse_ids,timestamps, centroid_xs, centroid_ys, centroid_zs = segment_synapses.fetch("synapse_id","timestamp","centroid_x","centroid_y","centroid_z")


        synapse_centers = np.vstack([centroid_xs,centroid_ys,centroid_zs]).T
    if timestamps is None:
        timestamps = np.zeros(len(synapse_centers))
        
    
    synapse_centers_scaled = synapse_centers* [4, 4, 40]
    
    if len(returned_error_faces) == 0:
        if verbose:
            print("Returning empty list because there were no error faces")
        if return_synapse_centroids:
            #2) Download the IDs and the centroid data
            return np.array([]),synapse_centers_scaled
        if return_synapse_ids:
            return np.array([]),synapse_ids
        if return_synapse_stats:
            return [],0,0
        else:
            return []

    

    #0) Create face labels for the entire mesh by makeing 0 for all of them but then changing the face labels of errors to 1
    neuron_mesh_labels = np.zeros(len(current_mesh.faces))
    neuron_mesh_labels[returned_error_faces] = 1

    from collections import Counter
    Counter(neuron_mesh_labels)

    #1) Build a KDTree of the mesh
    
    neuron_kd = KDTree(current_mesh.triangles_center)
    
    
    if verbose:
        print(f"Processing {len(synapse_centers_scaled)} synapses")


    #3) Query the centroid data to find the closest face on the mesh
    dist,closest_face = neuron_kd.query(synapse_centers_scaled)
    if verbose:
        print(f"maximum mapping distance = {np.max(dist)}")
        
    #4) Calculate the errored synapses
    closest_face_labels = neuron_mesh_labels[closest_face]
    errored_synapses_idx = np.where((closest_face_labels==1) & (dist<mapping_threshold))[0]
    non_errored_synapses_idx = np.setdiff1d(np.arange(len(closest_face_labels)),errored_synapses_idx)

    if verbose:
        print(f"Number of errored synapses = {errored_synapses_idx.shape}")
        
    errored_synapses = synapse_ids[errored_synapses_idx]
    non_errored_synapses = synapse_ids[non_errored_synapses_idx]
    
    errored_synapse_timestamps = timestamps[errored_synapses_idx]
    data_to_write = [dict(synapse_id=syn,timestamp=t,criteria_id=0,segment_id=segment_id) for syn,t in zip(errored_synapses,errored_synapse_timestamps)]
    
    
    if return_synapse_centroids:
        print("Returning the 1) coordinates for errored synapses 2) Coordinates for non-errored synapses")
        return synapse_centers_scaled[errored_synapses_idx],synapse_centers_scaled[non_errored_synapses_idx]
    
    if return_synapse_ids:
        print("Returning the synpase ids 1) errored synpases 2) non-errored synapses")
        return errored_synapses,non_errored_synapses
        
    
    if return_synapse_stats:
        n_synapses = len(synapse_ids)
        n_errored_synapses = len(errored_synapses)
        return data_to_write,n_synapses,n_errored_synapses
    else:
        return data_to_write
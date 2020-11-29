import time
import copy
import neuron
import neuron_utils as nru
from tqdm_utils import tqdm

def width_jump_edges(limb,
                    width_type = "no_spine_median_mesh_center",
                     width_jump_threshold = 100,
                     verbose=False
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
            if verbose:
                print(f"  Edge: {current_nodes}")

            up_width,d_width,up_sk,d_sk = nru.branch_boundary_transition(curr_limb,
                                  edge=current_nodes,
                                #offset=0,
                                verbose=False)

            downstream_jump = d_width-up_width

            if downstream_jump > width_jump_threshold:
                curr_error_edges.append(list(current_nodes))
        
        if curr_soma not in error_edges.keys():
            error_edges[curr_soma] = dict()
        error_edges[curr_soma][curr_soma_group] = curr_error_edges
        
    if verbose: 
        print(f"Total time for width = {time.time() - width_start_time}")
    return error_edges



import skeleton_utils as sk
import numpy_utils as nu
def double_back_edges(
    limb,
    double_back_threshold = 130,
    verbose = True,
    comparison_distance=3000,
    offset=0,):

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
        if len(curr_all_concept_network_data) != 1:
            raise Exception(f"There were more not exactly one starting dictinoary: {curr_all_concept_network_data} ")

        curr_all_concept_network_data[0]["soma_group_idx"] = 0

        curr_limb_network_stating_info = nru.all_concept_network_data_to_dict(curr_all_concept_network_data)

        #calculate the concept networks


        limb_corresp_for_networks = dict([(i,dict(branch_skeleton=k.skeleton,
                                                 width_from_skeleton=k.width,
                                                 branch_mesh=k.mesh,
                                                 branch_face_idx=k.mesh_face_idx)) for i,k in fixed_node_objects.items()])

        sep_graph_data["limb_correspondence"] = limb_corresp_for_networks

        sep_graph_data["limb_network_stating_info"] = curr_limb_network_stating_info

        limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(limb_corresp_for_networks,
                                                                                        curr_limb_network_stating_info,
                                                                                        run_concept_network_checks=True,
                                                                                       )   

        sep_graph_data["limb_concept_networks"] = limb_to_soma_concept_networks


        # --------------- Making the new limb object -------------- #
        new_labels = ["split_limb"]
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

def error_faces_by_axons(neuron_obj,verbose=False,visualize_errors_at_end=False,
                        min_skeletal_path_threshold = 15000,
                                sub_skeleton_length = 20000,
                                ais_angle_threshold = 110,
                                non_ais_angle_threshold = 50):
    
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
        
    return error_faces_concat




from pykdtree.kdtree import KDTree
import datajoint_utils as du
def get_error_synapse_inserts(current_mesh,
                              segment_id,
                              returned_error_faces,
                              mapping_threshold = 500,
                              minnie=None,
                              return_synapse_stats=True,
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
    
    
    if len(returned_error_faces) == 0:
        if verbose:
            print("Returning empty list because there were no error faces")
        if return_synapse_stats:
            return [],0,0
        else:
            return []
    
    if minnie is None:
        minnie,_ = du.configure_minnie_vm()
    
    segment_synapses = minnie.SynapseFiltered() & f"presyn={segment_id} OR postsyn={segment_id}"
    
    if len(segment_synapses)<=0:
        if verbose:
            print("Returning empty list because there were no SYNAPSES")
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
    
    #2) Download the IDs and the centroid data
    synapse_ids,timestamps, centroid_xs, centroid_ys, centroid_zs = segment_synapses.fetch("synapse_id","timestamp","centroid_x","centroid_y","centroid_z")
    
    synapse_centers = np.vstack([centroid_xs,centroid_ys,centroid_zs]).T
    synapse_centers_scaled = synapse_centers* [4, 4, 40]
    if verbose:
        print(f"Processing {len(synapse_centers_scaled)} synapses")


    #3) Query the centroid data to find the closest face on the mesh
    dist,closest_face = neuron_kd.query(synapse_centers_scaled)
    if verbose:
        print(f"maximum mapping distance = {np.max(dist)}")
        
    #4) Calculate the errored synapses
    closest_face_labels = neuron_mesh_labels[closest_face]
    errored_synapses_idx = np.where((closest_face_labels==1) & (dist<mapping_threshold))[0]

    if verbose:
        print(f"Number of errored synapses = {errored_synapses_idx.shape}")
        
    errored_synapses = synapse_ids[errored_synapses_idx]
    errored_synapse_timestamps = timestamps[errored_synapses_idx]
    data_to_write = [dict(synapse_id=syn,timestamp=t,criteria_id=0,segment_id=segment_id) for syn,t in zip(errored_synapses,errored_synapse_timestamps)]
    
    if return_synapse_stats:
        n_synapses = len(synapse_ids)
        n_errored_synapses = len(errored_synapses)
        return data_to_write,n_synapses,n_errored_synapses
    else:
        return data_to_write
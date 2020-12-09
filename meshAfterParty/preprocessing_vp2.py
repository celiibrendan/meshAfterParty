import skeleton_utils as sk
import soma_extraction_utils as sm
import trimesh_utils as tu
import trimesh
import numpy_utils as nu
import numpy as np
from importlib import reload
import networkx as nx
import time
import compartment_utils as cu
import networkx_utils as xu
import matplotlib_utils as mu

#importing at the bottom so don't get any conflicts
import itertools
from tqdm_utils import tqdm

#for meshparty preprocessing
import meshparty_skeletonize as m_sk
import general_utils as gu
import compartment_utils as cu
from meshparty import trimesh_io
from copy import deepcopy

from neuron_utils import *
import neuron


def mesh_correspondence_first_pass(mesh,
                                   skeleton=None,
                                   skeleton_branches=None,
                                  distance_by_mesh_center=True):
    """
    Will come up with the mesh correspondences for all of the skeleton
    branches: where there can be overlaps and empty faces
    
    """
    curr_limb_mesh = mesh
    curr_limb_sk = skeleton
    
    if skeleton_branches is None:
        if skeleton is None:
            raise Exception("Both skeleton and skeleton_branches is None")
        curr_limb_branches_sk_uneven = sk.decompose_skeleton_to_branches(curr_limb_sk) #the line that is decomposing to branches
    else:
        curr_limb_branches_sk_uneven = skeleton_branches 

    #Doing the limb correspondence for all of the branches of the skeleton
    local_correspondence = dict()
    for j,curr_branch_sk in tqdm(enumerate(curr_limb_branches_sk_uneven)):
        local_correspondence[j] = dict()

        
        returned_data = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                      curr_limb_mesh,
                                     skeleton_segment_width = 1000,
                                     distance_by_mesh_center=distance_by_mesh_center)
        if len(returned_data) == 0:
            print("Got nothing from first pass so expanding the mesh correspondnece parameters ")
            returned_data = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                      curr_limb_mesh,
                                     skeleton_segment_width = 1000,
                                     distance_by_mesh_center=distance_by_mesh_center,
                                    buffer=300,
                                     distance_threshold=6000,
                                    return_closest_face_on_empty=True)
            
        # Need to just pick the closest face is still didn't get anything
        
        # ------ 12/3 Addition: Account for correspondence that does not work so just picking the closest face
        curr_branch_face_correspondence, width_from_skeleton = returned_data
        
            
#             print(f"curr_branch_sk.shape = {curr_branch_sk.shape}")
#             np.savez("saved_skeleton_branch.npz",curr_branch_sk=curr_branch_sk)
#             tu.write_neuron_off(curr_limb_mesh,"curr_limb_mesh.off")
#             #print(f"returned_data = {returned_data}")
#             raise Exception(f"The output from mesh_correspondence_adaptive_distance was nothing: curr_branch_face_correspondence")


        if len(curr_branch_face_correspondence) > 0:
            curr_submesh = curr_limb_mesh.submesh([list(curr_branch_face_correspondence)],append=True,repair=False)
        else:
            curr_submesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))


        local_correspondence[j]["branch_skeleton"] = curr_branch_sk
        local_correspondence[j]["correspondence_mesh"] = curr_submesh
        local_correspondence[j]["correspondence_face_idx"] = curr_branch_face_correspondence
        local_correspondence[j]["width_from_skeleton"] = width_from_skeleton
        
    return local_correspondence



def check_skeletonization_and_decomp(
    skeleton,
    local_correspondence):
    """
    Purpose: To check that the decomposition and skeletonization went well
    
    
    """
    #couple of checks on how the decomposition went:  for each limb
    #1) if shapes of skeletons cleaned and divided match
    #2) if skeletons are only one component
    #3) if you downsample the skeletons then still only one component
    #4) if any empty meshes
    cleaned_branch = skeleton
    empty_submeshes = []

    print(f"Limb decomposed into {len(local_correspondence)} branches")

    #get all of the skeletons and make sure that they from a connected component
    divided_branches = [local_correspondence[k]["branch_skeleton"] for k in local_correspondence]
    divided_skeleton_graph = sk.convert_skeleton_to_graph(
                                    sk.stack_skeletons(divided_branches))

    divided_skeleton_graph_recovered = sk.convert_graph_to_skeleton(divided_skeleton_graph)

    cleaned_limb_skeleton = cleaned_branch
    print(f"divided_skeleton_graph_recovered = {divided_skeleton_graph_recovered.shape} and \n"
          f"current_mesh_data[0]['branch_skeletons_cleaned'].shape = {cleaned_limb_skeleton.shape}\n")
    if divided_skeleton_graph_recovered.shape != cleaned_limb_skeleton.shape:
        print(f"****divided_skeleton_graph_recovered and cleaned_limb_skeleton shapes not match: "
                        f"{divided_skeleton_graph_recovered.shape} vs. {cleaned_limb_skeleton.shape} *****")


    #check that it is all one component
    divided_skeleton_graph_n_comp = nx.number_connected_components(divided_skeleton_graph)
    print(f"Number of connected components in deocmposed recovered graph = {divided_skeleton_graph_n_comp}")

    cleaned_limb_skeleton_graph = sk.convert_skeleton_to_graph(cleaned_limb_skeleton)
    cleaned_limb_skeleton_graph_n_comp = nx.number_connected_components(cleaned_limb_skeleton_graph)
    print(f"Number of connected components in cleaned skeleton graph= {cleaned_limb_skeleton_graph_n_comp}")

    if divided_skeleton_graph_n_comp > 1 or cleaned_limb_skeleton_graph_n_comp > 1:
        raise Exception(f"One of the decompose_skeletons or cleaned skeletons was not just one component : {divided_skeleton_graph_n_comp,cleaned_limb_skeleton_graph_n_comp}")

    #check that when we downsample it is not one component:
    curr_branch_meshes_downsampled = [sk.resize_skeleton_branch(b,n_segments=1) for b in divided_branches]
    downsampled_skeleton = sk.stack_skeletons(curr_branch_meshes_downsampled)
    curr_sk_graph_debug = sk.convert_skeleton_to_graph(downsampled_skeleton)


    con_comp = list(nx.connected_components(curr_sk_graph_debug))
    if len(con_comp) > 1:
        raise Exception(f"There were more than 1 component when downsizing: {[len(k) for k in con_comp]}")
    else:
        print(f"The downsampled branches number of connected components = {len(con_comp)}")


    for j,v in local_correspondence.items():
        if len(v["correspondence_mesh"].faces) == 0:
            empty_submeshes.append(j)

    print(f"Empty submeshes = {empty_submeshes}")

    if len(empty_submeshes) > 0:
        raise Exception(f"Found empyt meshes after branch mesh correspondence: {empty_submeshes}")
        

        
        
def correspondence_1_to_1(
                    mesh,
                    local_correspondence,
                    curr_limb_endpoints_must_keep=None,
                    curr_soma_to_piece_touching_vertices=None,
                    must_keep_labels=dict(),
                    fill_to_soma_border=True
                    ):
    """
    Will Fix the 1-to-1 Correspondence of the mesh
    correspondence for the limbs and make sure that the
    endpoints that are designated as touching the soma then 
    make sure the mesh correspondnece reaches the soma limb border
    
    has an optional argument must_keep_labels that will allow you to specify some labels that are a must keep
    
    """
    
    if len(tu.split(mesh)[0])>1:
        su.compressed_pickle(mesh,"mesh")
        raise Exception("Mesh passed to correspondence_1_to_1 is not just one mesh")
    
    mesh_start_time = time.time()
    print(f"\n\n--- Working on 1-to-1 correspondence-----")

    #geting the current limb mesh

    no_missing_labels = list(local_correspondence.keys()) #counts the number of divided branches which should be the total number of labels
    curr_limb_mesh = mesh

    #set up the face dictionary
    face_lookup = dict([(j,[]) for j in range(0,len(curr_limb_mesh.faces))])

    for j,branch_piece in local_correspondence.items():
        curr_faces_corresponded = branch_piece["correspondence_face_idx"]

        for c in curr_faces_corresponded:
            face_lookup[c].append(j)

    original_labels = set(list(itertools.chain.from_iterable(list(face_lookup.values()))))
    print(f"max(original_labels),len(original_labels) = {(max(original_labels),len(original_labels))}")


    if len(original_labels) != len(no_missing_labels):
        raise Exception(f"len(original_labels) != len(no_missing_labels) for original_labels = {len(original_labels)},no_missing_labels = {len(no_missing_labels)}")

    if max(original_labels) + 1 > len(original_labels):
        raise Exception("There are some missing labels in the initial labeling")



    #here is where can call the function that resolves the face labels
    face_coloring_copy = cu.resolve_empty_conflicting_face_labels(
                     curr_limb_mesh = curr_limb_mesh,
                     face_lookup=face_lookup,
                     no_missing_labels = list(original_labels),
                    must_keep_labels=must_keep_labels,
    )

    """  9/17 Addition: Will make sure that the desired starting node is touching the soma border """
    """
    Pseudocode:
    For each soma it is touching
    0) Get the soma border
    1) Find the label_to_expand based on the starting coordinate
    a. Get the starting coordinate

    soma_to_piece_touching_vertices=None
    endpoints_must_keep

    """

    #curr_limb_endpoints_must_keep --> stores the endpoints that should be connected to the soma
    #curr_soma_to_piece_touching_vertices --> maps soma to  a list of grouped touching vertices

    if fill_to_soma_border:
        if (not curr_limb_endpoints_must_keep is None) and (not curr_soma_to_piece_touching_vertices is None):
            for sm,soma_border_list in curr_soma_to_piece_touching_vertices.items():
                for curr_soma_border,st_coord in zip(soma_border_list,curr_limb_endpoints_must_keep[sm]):

                    #1) Find the label_to_expand based on the starting coordinate
                    divided_branches = [v["branch_skeleton"] for v in local_correspondence.values()]
                    #print(f"st_coord = {st_coord}")
                    label_to_expand = sk.find_branch_skeleton_with_specific_coordinate(divded_skeleton=divided_branches,
                                                                                       current_coordinate=st_coord)[0]


                    face_coloring_copy = cu.waterfill_starting_label_to_soma_border(curr_limb_mesh,
                                                       border_vertices=curr_soma_border,
                                                        label_to_expand=label_to_expand,
                                                       total_face_labels=face_coloring_copy,
                                                       print_flag=True)


    # -- splitting the mesh pieces into individual pieces
    divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(curr_limb_mesh,face_coloring_copy)

    #-- check that all the split mesh pieces are one component --#
    local_correspondence_revised = deepcopy(local_correspondence)
    #save off the new data as branch mesh
    for k in local_correspondence_revised.keys():
        local_correspondence_revised[k]["branch_mesh"] = divided_submeshes[k]
        local_correspondence_revised[k]["branch_face_idx"] = divided_submeshes_idx[k]

        #clean the limb correspondence that we do not need
        del local_correspondence_revised[k]["correspondence_mesh"]
        del local_correspondence_revised[k]["correspondence_face_idx"]
    
    return local_correspondence_revised



def filter_soma_touching_vertices_dict_by_mesh(mesh,
                                              curr_piece_to_soma_touching_vertices,
                                              verbose=True):
    """
    Purpose: Will take the soma to touching vertics
    and filter it for only those that touch the particular mesh piece

    Pseudocode:
    1) Build a KDTree of the mesh
    2) Create an output dictionary to store the filtered soma touching vertices
    For the original soma touching vertices, iterating through all the somas
        For each soma_touching list:
            Query the mesh KDTree and only keep the coordinates whose distance is equal to 0

    If empty dictionary then return None? (have option for this)
    
    Ex: 
    return_value = filter_soma_touching_vertices_dict_by_mesh(
    mesh = mesh_pieces_for_MAP[0],
    curr_piece_to_soma_touching_vertices = piece_to_soma_touching_vertices[1]
    )

    """
    
    if curr_piece_to_soma_touching_vertices is None:
        if verbose:
            print("In filter_soma_touching_vertices_dict_by_mesh: curr_piece_to_soma_touching_vertices was None so returning none")
        return None

    #1) Build a KDTree of the mesh
    curr_mesh_tree = KDTree(mesh.vertices)

    #2) Create an output dictionary to store the filtered soma touching vertices
    output_soma_touching_vertices = dict()

    for sm_idx,border_verts_list in curr_piece_to_soma_touching_vertices.items():
        for b_verts in border_verts_list:
            dist,closest_nodes = curr_mesh_tree.query(b_verts)
            match_verts = b_verts[dist==0]
            if len(match_verts)>0:
                if sm_idx not in output_soma_touching_vertices.keys():
                    output_soma_touching_vertices[sm_idx] = []
                output_soma_touching_vertices[sm_idx].append(match_verts)
    if len(output_soma_touching_vertices) == 0:
        return None
    else:
        return output_soma_touching_vertices
    
    
    
# ----------------- When refactoring the limb decomposition function ------ #

def find_if_stitch_point_on_end_or_branch(matched_branches_skeletons,
                                                              stitch_coordinate,
                                                              verbose=False):
                    
                    
                        # Step A: Find if stitch point is on endpt/branch point or in middle
                        stitch_point_on_end_or_branch = False
                        if len(matched_branches_skeletons) == 0:
                            raise Exception("No matching branches found for soma extending point")
                        elif len(matched_branches_skeletons)>1:
                            if verbose:
                                print(f"Multiple Branches for MP soma extending connection point {matched_branches_skeletons.shape}")
                            stitch_point_on_end_or_branch = True
                        else:# len(match_sk_branches)==1:
                            if verbose:
                                print(f"Only one Branch for MP soma Extending connection point {matched_branches_skeletons.shape}")
                            if len(nu.matching_rows(sk.find_branch_endpoints(matched_branches_skeletons[0]),
                                                    stitch_coordinate))>0:
                                stitch_point_on_end_or_branch =True

                        return stitch_point_on_end_or_branch
                    

                    



import system_utils as su
from pykdtree.kdtree import KDTree
def attach_floating_pieces_to_limb_correspondence(
        limb_correspondence,
        floating_meshes,
        floating_piece_face_threshold = 600,
        max_stitch_distance=8000,
        distance_to_move_point_threshold = 4000,
        verbose = False):

    """
    Purpose: To take a limb correspondence and add on the floating pieces
    that are significant and close enough to a limb

    Pseudocode:
    0) Filter the floating pieces for only those above certain face count
    1) Run all significant floating pieces through preprocess_limb
    2) Get all full skeleton endpoints (degree 1) for all floating pieces


    Start loop until all floating pieces have been added
    a) Get full skeletons of limbs for all limbs in limb correspondence
    b) Find the minimum distance (and the node it corresponds to) for each floating piece between their 
    endpoints and all skeleton points of limbs
    c) Find the floating piece that has the closest distance
    --> winning piece

    For the winning piece
    d) Get the closest coordinate on the matching limb
    e) Try and move closest coordinate to an endpoint or high degree node
    f) Find the branch on the main limb that corresponds to the stitch point
    g) Find whether the stitch point is on an endpoint/high degree node or will end up splitting the branch
    AKA stitch_point_on_end_or_branch
    h) Find the branch on the floating limb where the closest end point is

    At this point have
    - main limb stitch point and branches (and whether not splitting will be required)  [like MAP]
    - floating limb stitch point and branch [like MP]

    Stitching process:
    i) if not stitch_point_on_end_or_branch
    - cut the main limb branch where stitch is
    - do mesh correspondence with the new stitches
    - (just give the both the same old width)
    - replace the old entry in the limb corresondence with one of the new skeleton cuts
    and add on the other skeletons cuts to the end

    j) Add a skeletal segment from floating limb stitch point to main limb stitch point
    k) Add the floating limb branches to the end of the limb correspondence
    l) Marks the floating piece as processed


    """
    limb_correspondence_cp = limb_correspondence
    non_soma_touching_meshes = floating_meshes
    floating_limbs_above_threshold = [k for k in non_soma_touching_meshes if len(k.faces)>floating_piece_face_threshold]

    #1) Run all significant floating pieces through preprocess_limb
    with su.suppress_stdout_stderr():
        floating_limbs_correspondence = [ preprocess_limb(mesh=k,
                           soma_touching_vertices_dict = None,
                           return_concept_network = False, 
                           )  for k in floating_limbs_above_threshold]

    #2) Get all full skeleton endpoints (degree 1) for all floating pieces
    floating_limbs_skeleton = [sk.stack_skeletons([k["branch_skeleton"] for k in l_c.values()]) for l_c in floating_limbs_correspondence]
    floating_limbs_skeleton_endpoints = [sk.find_skeleton_endpoint_coordinates(k) for k in floating_limbs_skeleton]

    # nviz.plot_objects(skeletons=floating_limb_skeletons,
    #                  scatters=floating_limbs_skeleton_endpoints,
    #                  scatter_size=1)

    floating_limbs_to_process = np.arange(0,len(floating_limbs_skeleton))

    #Start loop until all floating pieces have been added
    while len(floating_limbs_to_process)>0:

        #a) Get full skeletons of limbs for all limbs in limb correspondence
        main_limb_skeletons = []
        for main_idx in np.sort(list(limb_correspondence_cp.keys())):
            main_limb_skeletons.append(sk.stack_skeletons([k["branch_skeleton"] for k in limb_correspondence_cp[main_idx].values()]))

        #b) Find the minimum distance (and the node it corresponds to) for each floating piece between their 
        #endpoints and all skeleton points of limbs 
        floating_piece_min_distance_all_main_limbs = dict([(float_idx,[]) for float_idx in floating_limbs_to_process])
        for main_idx,main_limb_sk in enumerate(main_limb_skeletons):

            main_skeleton_coordinates = sk.skeleton_unique_coordinates(main_limb_sk)
            main_kdtree = KDTree(main_skeleton_coordinates)

            for float_idx in floating_piece_min_distance_all_main_limbs.keys():

                dist,closest_node = main_kdtree.query(floating_limbs_skeleton_endpoints[float_idx])
                min_dist_idx = np.argmin(dist)
                min_dist = dist[min_dist_idx]
                min_dist_closest_node = main_skeleton_coordinates[closest_node[min_dist_idx]]
                floating_piece_min_distance_all_main_limbs[float_idx].append([min_dist,min_dist_closest_node,floating_limbs_skeleton_endpoints[float_idx][min_dist_idx]])



        winning_float = -1
        winning_float_match_main_limb = -1
        main_limb_stitch_point = None
        floating_limb_stitch_point = None
        winning_float_dist = np.inf


        #c) Find the floating piece that has the closest distance
        #--> winning piece

        #For the winning piece
        #d) Get the closest coordinate on the matching limb

        for f_idx,dist_data in floating_piece_min_distance_all_main_limbs.items():

            dist_data_array = np.array(dist_data)
            closest_main_limb = np.argmin(dist_data_array[:,0])
            closest_main_dist = dist_data_array[closest_main_limb][0]

            if closest_main_dist < winning_float_dist:

                winning_float = f_idx
                winning_float_match_main_limb = closest_main_limb
                winning_float_dist = closest_main_dist
                main_limb_stitch_point = dist_data_array[closest_main_limb][1]
                floating_limb_stitch_point = dist_data_array[closest_main_limb][2]

        winning_main_skeleton = main_limb_skeletons[winning_float_match_main_limb]
        winning_floating_correspondence = floating_limbs_correspondence[winning_float]

        if verbose:
            print(f"winning_float = {winning_float}")
            print(f"winning_float_match_main_limb = {winning_float_match_main_limb}")
            print(f"winning_float_dist = {winning_float_dist}")
            print(f"main_limb_stitch_point = {main_limb_stitch_point}")
            print(f"floating_limb_stitch_point = {floating_limb_stitch_point}")

        
        if winning_float_dist > max_stitch_distance:
            print(f"The closest float distance was {winning_float_dist} which was greater than the maximum stitch distance {max_stitch_distance}\n"
                 " --> so ending the floating mesh stitch processs")
            return limb_correspondence_cp
        else:


            #e) Try and move closest coordinate to an endpoint or high degree node

            main_limb_stitch_point,change_status = sk.move_point_to_nearest_branch_end_point_within_threshold(
                                                                skeleton=winning_main_skeleton,
                                                                coordinate=main_limb_stitch_point,
                                                                distance_to_move_point_threshold = distance_to_move_point_threshold,
                                                                verbose=verbose,
                                                                consider_high_degree_nodes=True

                                                                )
            print(f"Status of Main limb stitch point moved = {change_status}")

        #     #checking that match was right
        #     nviz.plot_objects(meshes=[floating_limbs_above_threshold[winning_float],current_mesh_data[0]["branch_meshes"][winning_float_match_main_limb]],
        #                   meshes_colors=["red","aqua"],
        #                 skeletons=[floating_limbs_skeleton[winning_float],main_limb_skeletons[winning_float_match_main_limb]],
        #                  skeletons_colors=["red","aqua"],
        #                  scatters=[floating_limb_stitch_point.reshape(-1,3),main_limb_stitch_point.reshape(-1,3)],
        #                  scatters_colors=["red","aqua"])


            #f) Find the branch on the main limb that corresponds to the stitch point
            main_limb_branches = np.array([k["branch_skeleton"] for k in limb_correspondence_cp[winning_float_match_main_limb].values()])
            match_sk_branches = sk.find_branch_skeleton_with_specific_coordinate(main_limb_branches,
                                current_coordinate=main_limb_stitch_point)

            #g) Find whether the stitch point is on an endpoint/high degree node or will end up splitting the branch
            #AKA stitch_point_on_end_or_branch
            stitch_point_on_end_or_branch = find_if_stitch_point_on_end_or_branch(
                                                                    matched_branches_skeletons= main_limb_branches[match_sk_branches],
                                                                     stitch_coordinate=main_limb_stitch_point,
                                                                      verbose=False)

            #h) Find the branch on the floating limb where the closest end point is
            winning_float_branches = np.array([k["branch_skeleton"] for k in winning_floating_correspondence.values()])
            match_float_branches = sk.find_branch_skeleton_with_specific_coordinate(winning_float_branches,
                                current_coordinate=floating_limb_stitch_point)

            if len(match_float_branches) > 1:
                raise Exception("len(match_float_branches) was greater than 1 in the floating pieces stitch")

            if verbose:
                print("\n\n")
                print(f"match_sk_branches = {match_sk_branches}")
                print(f"match_float_branches = {match_float_branches}")
                print(f"stitch_point_on_end_or_branch = {stitch_point_on_end_or_branch}")


            """
            Stitching process:
            i) if not stitch_point_on_end_or_branch
               1. cut the main limb branch where stitch is
               2. do mesh correspondence with the new stitches
               3. (just give the both the same old width)
               4. replace the old entry in the limb corresondence with one of the new skeleton cuts
                  and add on the other skeletons cuts to the end

            j) Add a skeletal segment from floating limb stitch point to main limb stitch point
            k) Add the floating limb branches to the end of the limb correspondence
            l) Marks the floating piece as processed

            """

            # ---------- Begin stitching process ---------------
            if not stitch_point_on_end_or_branch:
                main_branch = match_sk_branches[0]
                #1. cut the main limb branch where stitch is
                matching_branch_sk = sk.cut_skeleton_at_coordinate(skeleton=main_limb_branches[main_branch],
                                                                           cut_coordinate = main_limb_stitch_point)
                #2. do mesh correspondence with the new stitchess
                stitch_mesh = limb_correspondence_cp[winning_float_match_main_limb][main_branch]["branch_mesh"]

                local_correspondnece = mesh_correspondence_first_pass(mesh=stitch_mesh,
                                                          skeleton_branches=matching_branch_sk)

                local_correspondence_revised = correspondence_1_to_1(mesh=stitch_mesh,
                                                            local_correspondence=local_correspondnece)

                #3. (just give the both the same old width)
                old_width = limb_correspondence_cp[winning_float_match_main_limb][main_branch]["width_from_skeleton"]
                for branch_idx in local_correspondence_revised.keys():
                    local_correspondence_revised[branch_idx]["width_from_skeleton"] = old_width

                #4. replace the old entry in the limb corresondence with one of the new skeleton cuts
                #and add on the other skeletons cuts to the end
                print(f"main_branch = {main_branch}")
                del limb_correspondence_cp[winning_float_match_main_limb][main_branch]


                limb_correspondence_cp[winning_float_match_main_limb][main_branch] = local_correspondence_revised[0]
                limb_correspondence_cp[winning_float_match_main_limb][np.max(list(limb_correspondence_cp[winning_float_match_main_limb].keys()))+1] = local_correspondence_revised[1]
                limb_correspondence_cp[winning_float_match_main_limb] = gu.order_dict_by_keys(limb_correspondence_cp[winning_float_match_main_limb])



            #j) Add a skeletal segment from floating limb stitch point to main limb stitch point
            skeleton = winning_floating_correspondence[match_float_branches[0]]["branch_skeleton"]
            adjusted_floating_sk_branch = sk.stack_skeletons([skeleton,np.array([floating_limb_stitch_point,main_limb_stitch_point])])
            
#             adjusted_floating_sk_branch = sk.add_and_smooth_segment_to_branch(skeleton,new_seg=np.array([floating_limb_stitch_point,main_limb_stitch_point]),
#                                                                              resize_mult=0.2,n_resized_cutoff=3)

            winning_floating_correspondence[match_float_branches[0]]["branch_skeleton"] = adjusted_floating_sk_branch

            #k) Add the floating limb branches to the end of the limb correspondence
            curr_limb_key_len = np.max(list(limb_correspondence_cp[winning_float_match_main_limb].keys()))
            for float_idx,flaot_data in winning_floating_correspondence.items():
                limb_correspondence_cp[winning_float_match_main_limb][curr_limb_key_len + 1 + float_idx] = flaot_data
        



        #l) Marks the floating piece as processed
        floating_limbs_to_process = np.setdiff1d(floating_limbs_to_process,[winning_float])
        
    return limb_correspondence_cp



def calculate_limb_concept_networks(limb_correspondence,
                                    network_starting_info,
                                   run_concept_network_checks=True):
    """
    Can take a limb correspondence and the starting vertices and endpoints
    and create a list of concept networks organized by 
    [soma_idx] --> list of concept networks 
                    (because could possibly have mulitple starting points on the same soma)
    
    """
    
    limb_correspondence_individual = limb_correspondence
    divided_skeletons = np.array([limb_correspondence_individual[k]["branch_skeleton"] for k in np.sort(list(limb_correspondence_individual.keys()))])



    # -------------- Part 18: Getting Concept Networks  [soma_idx] --> list of concept networks -------#

    """
    Concept Network Pseudocode:

    Step 0: Compile the Limb correspondence into final form

    Make sure these have the same list
    limb_to_soma_touching_vertices_list,limb_to_endpoints_must_keep_list

    For every dictionary in zip(limb_to_soma_touching_vertices_list,
                            limb_to_endpoints_must_keep_list):

        #make sure the dicts have same keys
        For every key (represents the soma) in the dictionary:

            #make sure the lists have the same sizes
            For every item in the list (which would be a list of endpoints or list of groups of vertexes):
                #At this point have the soma, the endpoint and the touching vertices

                1) find the branch with the endoint that must keep
                    --> if multiple endpoints then error
                2) Call the branches_to_concept_network with the
                   a. divided skeletons
                   b. closest endpoint
                   c. endpoints of branch (from the branch found)
                   d. touching soma vertices

                3) Run the checks on the concept network

    """

    

    limb_to_soma_concept_networks = dict()

    for soma_idx in network_starting_info.keys():
        
        if soma_idx not in list(limb_to_soma_concept_networks.keys()):
            limb_to_soma_concept_networks[soma_idx] = []
        for soma_group_idx,st_dict in network_starting_info[soma_idx].items():
            t_verts = st_dict["touching_verts"]
            endpt = st_dict["endpoint"]
            print(f"\n\n---------Working on soma_idx = {soma_idx}, soma_group_idx {soma_group_idx}, endpt = {endpt}---------")



            #1) find the branch with the endoint that must keep
            # ---------------- 11/17 Addition: If the endpoint does not match a skeleton point anymore then just get the closest endpoint of mesh that has touching vertices

            start_branch = sk.find_branch_skeleton_with_specific_coordinate(divded_skeleton=divided_skeletons,
                                                            current_coordinate=endpt)[0]



            #print(f"Starting_branch = {start_branch}")
            #print(f"Start endpt = {endpt}")
            start_branch_endpoints = sk.find_branch_endpoints(divided_skeletons[start_branch])
            #print(f"Starting_branch endpoints = {start_branch_endpoints}")

            #2) Call the branches_to_concept_network with the
            curr_limb_concept_network = nru.branches_to_concept_network(curr_branch_skeletons=divided_skeletons,
                                                                  starting_coordinate=endpt,
                                                                  starting_edge=start_branch_endpoints,
                                                                  touching_soma_vertices=t_verts,
                                                                       soma_group_idx=soma_group_idx)
            print("Done generating concept network \n\n")


            run_checks = True
            check_print_flag = True

            if run_concept_network_checks:
                #3) Run the checks on the concept network
                #3.1: check to make sure the starting coordinate was recovered

                recovered_touching_piece = xu.get_nodes_with_attributes_dict(curr_limb_concept_network,dict(starting_coordinate=endpt))

                if check_print_flag:
                    print(f"recovered_touching_piece = {recovered_touching_piece}")
                if recovered_touching_piece[0] != start_branch:
                    raise Exception(f"For limb and soma {soma_idx} the recovered_touching and original touching do not match\n"
                                   f"recovered_touching_piece = {recovered_touching_piece}, original_touching_pieces = {start_branch}")


                #3.2: Check number of nodes match the number of divided skeletons
                if len(curr_limb_concept_network.nodes()) != len(divided_skeletons):
                    raise Exception("The number of nodes in the concept graph and number of branches passed to it did not match\n"
                                  f"len(curr_limb_concept_network.nodes())={len(curr_limb_concept_network.nodes())}, len(curr_limb_divided_skeletons)= {len(divided_skeletons)}")

                #3.3: Check that concept network is a connected component
                if nx.number_connected_components(curr_limb_concept_network) > 1:
                    raise Exception("There was more than 1 connected components in the concept network")


                #3.4 Make sure the oriiginal divided skeleton endpoints match the concept map endpoints
                for j,un_resized_b in enumerate(divided_skeletons):
                    """
                    Pseudocode: 
                    1) get the endpoints of the current branch
                    2) get the endpoints in the concept map
                    3) compare
                    - if not equalt then break
                    """
                    #1) get the endpoints of the current branch
                    b_endpoints = neuron.Branch(un_resized_b).endpoints
                    #2) get the endpoints in the concept map
                    graph_endpoints = xu.get_node_attributes(curr_limb_concept_network,attribute_name="endpoints",node_list=[j])[0]
                    #print(f"original_branch_endpoints = {b_endpoints}, concept graph node endpoints = {graph_endpoints}")
                    if not xu.compare_endpoints(b_endpoints,graph_endpoints):
                        raise Exception(f"The node {j} in concept graph endpoints do not match the endpoints of the original branch\n"
                                       f"original_branch_endpoints = {b_endpoints}, concept graph node endpoints = {graph_endpoints}")

            limb_to_soma_concept_networks[soma_idx].append(curr_limb_concept_network)

    return limb_to_soma_concept_networks                    

import neuron_utils as nru
import system_utils as su

def preprocess_limb(mesh,
                   soma_touching_vertices_dict = None,
                   distance_by_mesh_center=True, #how the distance is calculated for mesh correspondence
                    meshparty_segment_size = 100,
                   meshparty_n_surface_downsampling = 2,
                    combine_close_skeleton_nodes=True,
                    combine_close_skeleton_nodes_threshold=700,
                    filter_end_node_length=1500,#4001,
                    use_meshafterparty=True,
                    perform_cleaning_checks = True,
                    
                    #for controlling the pieces processed by MAP
                    width_threshold_MAP = 450,
                    size_threshold_MAP = 1000,
                    
                    #parameters for MP skeletonization,
                    
                    #Parameters for setting how the MAP skeletonization takes place
                    use_surface_after_CGAL=False,
                    surface_reconstruction_size = 500,
                    
                    #parametrers for stitching the MAP and MP pieces together
                    move_MAP_stitch_to_end_or_branch = True,
                    distance_to_move_point_threshold=500,
                    
                    #concept_network parameters
                    run_concept_network_checks = True,
                    return_concept_network = True,
                    return_concept_network_starting_info=False,
                    
                    #printing controls
                    verbose = True,
                    print_fusion_steps=True,
                    
                   ):
    curr_limb_time = time.time()

    limb_mesh_mparty = mesh


    #will store a list of all the endpoints tha tmust be kept:
    limb_to_endpoints_must_keep_list = []
    limb_to_soma_touching_vertices_list = []

    # --------------- Part 1 and 2: Getting Border Vertices and Setting the Root------------- #
    fusion_time = time.time()
    #will eventually get the current root from soma_to_piece_touching_vertices[i]
    if not soma_touching_vertices_dict is None:
        root_curr = soma_touching_vertices_dict[list(soma_touching_vertices_dict.keys())[0]][0][0]
    else:
        root_curr = None
        
    print(f"root_curr = {root_curr}")

    if print_fusion_steps:
        print(f"Time for preparing soma vertices and root: {time.time() - fusion_time }")
        fusion_time = time.time()

    # --------------- Part 3: Meshparty skeletonization and Decomposition ------------- #
    sk_meshparty_obj = m_sk.skeletonize_mesh_largest_component(limb_mesh_mparty,
                                                            root=root_curr,
                                                              filter_mesh=False)
    
    print(f"meshparty_segment_size = {meshparty_segment_size}")

    if print_fusion_steps:
        print(f"Time for 1st pass MP skeletonization: {time.time() - fusion_time }")
        fusion_time = time.time()

    (segment_branches, #skeleton branches
    divided_submeshes, divided_submeshes_idx, #mesh correspondence (mesh and indices)
    segment_widths_median) = m_sk.skeleton_obj_to_branches(sk_meshparty_obj,
                                                          mesh = limb_mesh_mparty,
                                                          meshparty_segment_size=meshparty_segment_size)





    if print_fusion_steps:
        print(f"Decomposing first pass: {time.time() - fusion_time }")
        fusion_time = time.time()


    if use_meshafterparty:
        print("Attempting to use MeshAfterParty Skeletonization and Mesh Correspondence")
        # --------------- Part 4: Find Individual Branches that could be MAP processed because of width ------------- #
        #gettin the branches that should be passed through MAP skeletonization
        pieces_above_threshold = np.where(segment_widths_median>width_threshold_MAP)[0]

        #getting the correspondnece info for those MAP qualifying
        width_large = segment_widths_median[pieces_above_threshold]
        sk_large = [segment_branches[k] for k in pieces_above_threshold]
        mesh_large_idx = [divided_submeshes_idx[k] for k in pieces_above_threshold]
    else:
        print("Only Using MeshParty Skeletonization and Mesh Correspondence")
        mesh_large_idx = []
        width_large = []
        sk_large = []


    print("Another print")
    mesh_pieces_for_MAP = []
    mesh_pieces_for_MAP_face_idx = []


    if len(mesh_large_idx) > 0: #will only continue processing if found MAP candidates

        # --------------- Part 5: Find mesh connectivity and group MAP branch candidates into MAP sublimbs ------------- #
        print(f"Found len(mesh_large_idx) MAP candidates: {[len(k) for k in mesh_large_idx]}")

        #finds the connectivity edges of all the MAP candidates
        mesh_large_connectivity = tu.mesh_list_connectivity(meshes = mesh_large_idx,
                                                            connectivity="vertices",
                                main_mesh = limb_mesh_mparty,
                                print_flag = False)
        if print_fusion_steps:
            print(f"mesh_large_connectivity: {time.time() - fusion_time }")
            fusion_time = time.time()
        """
        --------------- Grouping MAP candidates ----------------
        Purpose: Will see what mesh pieces should be grouped together
        to pass through CGAL skeletonization


        Pseudocode: 
        1) build a networkx graph with all nodes for mesh_large_idx indexes
        2) Add the edges
        3) Find the connected components
        4) Find sizes of connected components
        5) For all those connected components that are of a large enough size, 
        add the mesh branches and skeletons to the final list


        """
        G = nx.Graph()
        G.add_nodes_from(np.arange(len(mesh_large_idx)))
        G.add_edges_from(mesh_large_connectivity)
        conn_comp = list(nx.connected_components(G))

        filtered_pieces = []

        sk_large_size_filt = []
        mesh_large_idx_size_filt = []
        width_large_size_filt = []

        for cc in conn_comp:
            total_cc_size = np.sum([len(mesh_large_idx[k]) for k in cc])
            if total_cc_size>size_threshold_MAP:
                #print(f"cc ({cc}) passed the size threshold because size was {total_cc_size}")
                filtered_pieces.append(pieces_above_threshold[list(cc)])

        if print_fusion_steps:
            print(f"Finding MAP candidates connected components: {time.time() - fusion_time }")
            fusion_time = time.time()

        #filtered_pieces: will have the indexes of all the branch candidates that should  be 
        #grouped together and passed through MAP skeletonization

        if len(filtered_pieces) > 0:
            # --------------- Part 6: If Found MAP sublimbs, Get the meshes and mesh_idxs of the sublimbs ------------- #
            print(f"len(filtered_pieces) = {len(filtered_pieces)}")
            #all the pieces that will require MAP mesh correspondence and skeletonization
            #(already organized into their components)
            mesh_pieces_for_MAP = [limb_mesh_mparty.submesh([np.concatenate(divided_submeshes_idx[k])],append=True,repair=False) for k in filtered_pieces]
            mesh_pieces_for_MAP_face_idx = [np.concatenate(divided_submeshes_idx[k]) for k in filtered_pieces]



            """
            Old Way: Finding connectivity of pieces through
            mesh_idx_MP = [divided_submeshes_idx[k] for k in pieces_idx_MP]

            mesh_large_connectivity_MP = tu.mesh_list_connectivity(meshes = mesh_idx_MP,
                                    main_mesh = limb_mesh_mparty,
                                    print_flag = False)

            New Way: going to use skeleton connectivity to determine
            connectivity of pieces

            Pseudocode: 
            1)

            """
            # --------------- Part 7: If Found MAP sublimbs, Get the meshes and mesh_idxs of the sublimbs ------------- #
            # ********* if there are no pieces leftover then will automatically make all the lists below just empty (don't need to if.. else.. the case)****
            pieces_idx_MP = np.setdiff1d(np.arange(len(divided_submeshes_idx)),np.concatenate(filtered_pieces))

            skeleton_MP = [segment_branches[k] for k in pieces_idx_MP]
            skeleton_connectivity_MP = sk.skeleton_list_connectivity(
                                            skeletons=skeleton_MP
                                            )
            if print_fusion_steps:
                print(f"skeleton_connectivity_MP : {time.time() - fusion_time }")
                fusion_time = time.time()

            G = nx.Graph()
            G.add_nodes_from(np.arange(len(skeleton_MP)))
            G.add_edges_from(skeleton_connectivity_MP)
            sublimbs_MP = list(nx.connected_components(G))
            sublimbs_MP_orig_idx = [pieces_idx_MP[list(k)] for k in sublimbs_MP]


            #concatenate into sublimbs the skeletons and meshes
            sublimb_mesh_idx_branches_MP = [divided_submeshes_idx[k] for k in sublimbs_MP_orig_idx]
            sublimb_mesh_branches_MP = [[limb_mesh_mparty.submesh([ki],append=True,repair=False)
                                        for ki in k] for k in sublimb_mesh_idx_branches_MP]
            sublimb_meshes_MP = [limb_mesh_mparty.submesh([np.concatenate(k)],append=True,repair=False)
                                                         for k in sublimb_mesh_idx_branches_MP]
            sublimb_meshes_MP_face_idx = [np.concatenate(k)
                                                         for k in sublimb_mesh_idx_branches_MP]
            sublimb_skeleton_branches = [segment_branches[k] for k in sublimbs_MP_orig_idx]
            widths_MP = [segment_widths_median[k] for k in sublimbs_MP_orig_idx]

            if print_fusion_steps:
                print(f"Grouping MP Sublimbs by Graph: {time.time() - fusion_time }")
                fusion_time = time.time()


    # else: #if no pieces were determine to need MAP processing
    #     print("No MAP processing needed: just returning the Meshparty skeletonization and mesh correspondence")
    #     raise Exception("Returning MP correspondence")


    # nviz.plot_objects(main_mesh=tu.combine_meshes([limb_mesh_mparty,current_neuron["S0"].mesh]),
    #                   main_mesh_color="green",
    #     skeletons=sk_large_size_filt,
    #      meshes=[limb_mesh_mparty.submesh([k],append=True) for k in mesh_large_idx_size_filt],
    #       meshes_colors="red")








    # --------------- Part 8: If No MAP sublimbs found, set the MP sublimb lists to just the whole MP branch decomposition ------------- #

    #if no sublimbs need to be decomposed with MAP then just reassign all of the previous MP processing to the sublimb_MPs
    if len(mesh_pieces_for_MAP) == 0:
        sublimb_meshes_MP = [limb_mesh_mparty] #trimesh pieces that have already been passed through MP skeletonization (may not need)
        # -- the decomposition information ---
        sublimb_mesh_branches_MP = [divided_submeshes] #the mesh branches for all the disconnected sublimbs
        sublimb_mesh_idx_branches_MP = [divided_submeshes_idx] #The mesh branches idx that have already passed through MP skeletonization
        sublimb_skeleton_branches = [segment_branches]#the skeleton bnraches for all the sublimbs
        widths_MP = [segment_widths_median] #the mesh branches widths for all the disconnected groups

        MAP_flag = False
    else:
        MAP_flag = True



    mesh_pieces_for_MAP #trimesh pieces that should go through CGAL skeletonization
    sublimb_meshes_MP #trimesh pieces that have already been passed through MP skeletonization (may not need)

    # -- the decomposition information ---
    sublimb_mesh_branches_MP #the mesh branches for all the disconnected sublimbs
    sublimb_mesh_idx_branches_MP #The mesh branches idx that have already passed through MP skeletonization
    sublimb_skeleton_branches #the skeleton bnraches for all the sublimbs
    widths_MP #the mesh branches widths for all the disconnected groups

    if print_fusion_steps:
        print(f"Divinding into MP and MAP pieces: {time.time() - fusion_time }")
        fusion_time = time.time()



    # ------------------- At this point have the correct division between MAP and MP ------------------------

    # -------------- Part 9: Doing the MAP decomposition ------------------ #
    global_start_time = time.time()
    endpoints_must_keep = dict()



    limb_correspondence_MAP = dict()

    for sublimb_idx,(mesh,mesh_idx) in enumerate(zip(mesh_pieces_for_MAP,mesh_pieces_for_MAP_face_idx)):
        print(f"--- Working on MAP piece {sublimb_idx}---")
        #print(f"soma_touching_vertices_dict = {soma_touching_vertices_dict}")
        mesh_start_time = time.time()
        curr_soma_to_piece_touching_vertices = filter_soma_touching_vertices_dict_by_mesh(
        mesh = mesh,
        curr_piece_to_soma_touching_vertices = soma_touching_vertices_dict
        )

        if print_fusion_steps:
            print(f"MAP Filtering Soma Pieces: {time.time() - fusion_time }")
            fusion_time = time.time()

        # ---- 0) Generating the Clean skeletons  -------------------------------------------#
        if not curr_soma_to_piece_touching_vertices is None:
            curr_total_border_vertices = dict([(k,np.vstack(v)) for k,v in curr_soma_to_piece_touching_vertices.items()])
        else:
            curr_total_border_vertices = None


        cleaned_branch,curr_limb_endpoints_must_keep = sk.skeletonize_and_clean_connected_branch_CGAL(
            mesh=mesh,
            curr_soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices,
            total_border_vertices=curr_total_border_vertices,
            filter_end_node_length=filter_end_node_length,
            perform_cleaning_checks=perform_cleaning_checks,
            combine_close_skeleton_nodes = combine_close_skeleton_nodes,
            combine_close_skeleton_nodes_threshold=combine_close_skeleton_nodes_threshold,
        use_surface_after_CGAL=use_surface_after_CGAL,
        surface_reconstruction_size=surface_reconstruction_size)

        if not curr_limb_endpoints_must_keep is None:
            limb_to_endpoints_must_keep_list.append(curr_limb_endpoints_must_keep)
            limb_to_soma_touching_vertices_list.append(curr_soma_to_piece_touching_vertices)
        else:
            print("Inside MAP decomposition and curr_limb_endpoints_must_keep was None")

        if len(cleaned_branch) == 0:
            raise Exception(f"Found a zero length skeleton for limb {z} of trmesh {branch}")

        if print_fusion_steps:
            print(f"skeletonize_and_clean_connected_branch_CGAL: {time.time() - fusion_time }")
            fusion_time = time.time()

        # ---- 1) Generating Initial Mesh Correspondence -------------------------------------------#
        start_time = time.time()

        print(f"Working on limb correspondence for #{sublimb_idx} MAP piece")
        local_correspondence = mesh_correspondence_first_pass(mesh=mesh,
                                                             skeleton=cleaned_branch,
                                                             distance_by_mesh_center=distance_by_mesh_center)


        print(f"Total time for decomposition = {time.time() - start_time}")
        if print_fusion_steps:
            print(f"mesh_correspondence_first_pass: {time.time() - fusion_time }")
            fusion_time = time.time()


        #------------- 2) Doing Some checks on the initial corespondence -------- #


        if perform_cleaning_checks:
            check_skeletonization_and_decomp(skeleton=cleaned_branch,
                                            local_correspondence=local_correspondence)

        # -------3) Finishing off the face correspondence so get 1-to-1 correspondence of mesh face to skeletal piece
        local_correspondence_revised = correspondence_1_to_1(mesh=mesh,
                                        local_correspondence=local_correspondence,
                                        curr_limb_endpoints_must_keep=curr_limb_endpoints_must_keep,
                                        curr_soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices)

        # -------3b) Fixing the mesh indices to correspond to the larger mesh as a whole
        for k,v in local_correspondence_revised.items():
            local_correspondence_revised[k]["branch_face_idx"] = mesh_idx[local_correspondence_revised[k]["branch_face_idx"]]

        print(f"Total time for MAP sublimb #{sublimb_idx} mesh processing = {time.time() - mesh_start_time}")

        if print_fusion_steps:
            print(f"correspondence_1_to_1: {time.time() - fusion_time }")
            fusion_time = time.time()

        limb_correspondence_MAP[sublimb_idx] = local_correspondence_revised

    print(f"Total time for MAP sublimb processing {time.time() - global_start_time}")





    # ----------------- Part 10: Doing the MP Decomposition ---------------------- #




    sublimb_meshes_MP #trimesh pieces that have already been passed through MP skeletonization (may not need)
    # -- the decomposition information ---
    sublimb_mesh_branches_MP #the mesh branches for all the disconnected sublimbs
    sublimb_mesh_idx_branches_MP #The mesh branches idx that have already passed through MP skeletonization
    sublimb_skeleton_branches #the skeleton bnraches for all the sublimbs
    widths_MP #the mesh branches widths for all the disconnected groups

    limb_correspondence_MP = dict()

    for sublimb_idx,mesh in enumerate(sublimb_meshes_MP):
        print(f"---- Working on MP Decomposition #{sublimb_idx} ----")
        mesh_start_time = time.time()

        if len(sublimb_meshes_MP) == 1 and MAP_flag == False:
            print("Using Quicker soma_to_piece_touching_vertices because no MAP and only one sublimb_mesh piece ")
            curr_soma_to_piece_touching_vertices = soma_touching_vertices_dict
        else:
            if not soma_touching_vertices_dict is None:
                print("Computing the current soma touching verts dict manually")
                curr_soma_to_piece_touching_vertices = filter_soma_touching_vertices_dict_by_mesh(
                                                    mesh = mesh,
                                                    curr_piece_to_soma_touching_vertices = soma_touching_vertices_dict
                                                    )
            else:
                curr_soma_to_piece_touching_vertices = None

        if print_fusion_steps:
            print(f"MP filtering soma verts: {time.time() - fusion_time }")
            fusion_time = time.time()

        #creating all of the sublimb groups
        segment_branches = np.array(sublimb_skeleton_branches[sublimb_idx])
        whole_sk_MP = sk.stack_skeletons(segment_branches)
        branch = mesh
        divided_submeshes = np.array(sublimb_mesh_branches_MP[sublimb_idx])
        divided_submeshes_idx = sublimb_mesh_idx_branches_MP[sublimb_idx]
        segment_widths_median = widths_MP[sublimb_idx]


        if curr_soma_to_piece_touching_vertices is None:
            print(f"Do Not Need to Fix MP Decomposition {sublimb_idx} so just continuing")

        else:

            # ------- 11/9 addition: Fixing error where creating soma touching branch on mesh that doesn't touch border ------------------- #
            print(f"Fixing Possible Soma Extension Branch for Sublimb {sublimb_idx}")
            no_soma_extension_add = True 

            endpts_total = dict()
            curr_soma_to_piece_touching_vertices_total = dict()
            for sm_idx,sm_bord_verts_list in curr_soma_to_piece_touching_vertices.items():
                #will be used for later
                endpts_total[sm_idx] = []
                curr_soma_to_piece_touching_vertices_total[sm_idx] = []

                for sm_bord_verts in sm_bord_verts_list:
                    #1) Get the mesh pieces that are touching the border
                    matching_mesh_idx = tu.filter_meshes_by_containing_coordinates(mesh_list=divided_submeshes,
                                               nullifying_points=sm_bord_verts,
                                                filter_away=False,
                                               distance_threshold=0,
                                               return_indices=True)
                    #2) concatenate all meshes and skeletons that are touching
                    if len(matching_mesh_idx) <= 0:
                        raise Exception("None of branches were touching the border vertices when fixing MP pieces")

                    touch_mesh = tu.combine_meshes(divided_submeshes[matching_mesh_idx])
                    touch_sk = sk.stack_skeletons(segment_branches[matching_mesh_idx])

                    local_curr_soma_to_piece_touching_vertices = {sm_idx:[sm_bord_verts]}
                    new_sk,endpts,new_branch_info = sk.create_soma_extending_branches(current_skeleton=touch_sk,
                                          skeleton_mesh=touch_mesh,
                                          soma_to_piece_touching_vertices=local_curr_soma_to_piece_touching_vertices,
                                          return_endpoints_must_keep=True,
                                          return_created_branch_info=True,
                                          check_connected_skeleton=False)

                    #3) Add the info to the new running lists
                    endpts_total[sm_idx].append(endpts[sm_idx][0])
                    curr_soma_to_piece_touching_vertices_total[sm_idx].append(sm_bord_verts)


                    #4) Skip if no new branch was added
                    br_info = new_branch_info[sm_idx][0]
                    if br_info is None:
                        print("The new branch info was none so skipping \n")
                        continue

                    #4 If new branch was made then 
                    no_soma_extension_add=False

                    #1) Get the newly added branch (and the original vertex which is the first row)
                    br_new,sm_bord_verts = br_info["new_branch"],br_info["border_verts"] #this will hold the new branch and the border vertices corresponding to it

                    curr_soma_to_piece_touching_vertices_MP = {sm_idx:[sm_bord_verts]}
                    endpoints_must_keep_MP = {sm_idx:[br_new[0][1]]}


                    orig_vertex = br_new[0][0]
                    print(f"orig_vertex = {orig_vertex}")

                    #2) Find the branches that have that coordinate (could be multiple)
                    match_sk_branches = sk.find_branch_skeleton_with_specific_coordinate(segment_branches,
                        current_coordinate=orig_vertex)

                    print(f"match_sk_branches = {match_sk_branches}")



                    """ ******************* THIS NEEDS TO BE FIXED WITH THE SAME METHOD OF STITCHING ********************  """
                    """
                    Pseudocode:
                    1) Find if branch point will require split or not
                    2) If does require split then split the skeleton
                    3) Gather mesh pieces for correspondence and the skeletons
                    4) Run the mesh correspondence
                    - this case calculate the new widths after run 
                    5) Replace the old branch parts with the new ones



                    """

                    stitch_point_on_end_or_branch = find_if_stitch_point_on_end_or_branch(
                                                            matched_branches_skeletons= segment_branches[match_sk_branches],
                                                             stitch_coordinate=orig_vertex,
                                                              verbose=False)


                    if not stitch_point_on_end_or_branch:
                        matching_branch_sk = sk.cut_skeleton_at_coordinate(skeleton=segment_branches[match_sk_branches][0],
                                                                          cut_coordinate = orig_vertex)
                    else:
                        matching_branch_sk = segment_branches[match_sk_branches]


                    #3) Find the mesh and skeleton of the winning branch
                    matching_branch_meshes = np.array(divided_submeshes)[match_sk_branches]
                    matching_branch_mesh_idx = np.array(divided_submeshes_idx)[match_sk_branches]
                    extend_soma_mesh_idx = np.concatenate(matching_branch_mesh_idx)
                    extend_soma_mesh = limb_mesh_mparty.submesh([extend_soma_mesh_idx ],append=True,repair=False)

                    #4) Add newly created branch to skeleton and divide the skeleton into branches (could make 2 or 3)
                    #extended_skeleton_to_soma = sk.stack_skeletons([list(matching_branch_sk),br_new])

                    sk.check_skeleton_connected_component(sk.stack_skeletons(list(matching_branch_sk) + [br_new]))

                    #5) Run Adaptive mesh correspondnece using branches and mesh
                    local_correspondnece_MP = mesh_correspondence_first_pass(mesh=extend_soma_mesh,
                                                                             skeleton_branches = list(matching_branch_sk) + [br_new]
                                                  #skeleton=extended_skeleton_to_soma
                                                                            )

                    # GETTING MESHES THAT ARE NOT FULLY CONNECTED!!
                    local_correspondence_revised = correspondence_1_to_1(mesh=extend_soma_mesh,
                                                                local_correspondence=local_correspondnece_MP,
                                                                curr_limb_endpoints_must_keep=endpoints_must_keep_MP,
                                                                curr_soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices_MP)

                    # All the things that should be revised:
                #     segment_branches, #skeleton branches
                #     divided_submeshes, divided_submeshes_idx, #mesh correspondence (mesh and indices)
                #     segment_widths_median


                    new_submeshes = [k["branch_mesh"] for k in local_correspondence_revised.values()]
                    new_submeshes_idx = [extend_soma_mesh_idx[k["branch_face_idx"]] for k in local_correspondence_revised.values()]
                    new_skeletal_branches = [k["branch_skeleton"] for k in local_correspondence_revised.values()]

                    #calculate the new width
                    ray_inter = tu.ray_pyembree.RayMeshIntersector(limb_mesh_mparty)
                    new_widths = []
                    for new_s_idx in new_submeshes_idx:
                        curr_ray_distance = tu.ray_trace_distance(mesh=limb_mesh_mparty, 
                                            face_inds=new_s_idx,
                                           ray_inter=ray_inter)
                        curr_width_median = np.median(curr_ray_distance[curr_ray_distance!=0])
                        print(f"curr_width_median = {curr_width_median}")
                        if (not np.isnan(curr_width_median)) and (curr_width_median > 0):
                            new_widths.append(curr_width_median)
                        else:
                            print(f"USING A DEFAULT WIDTH BECAUSE THE NEWLY COMPUTED ONE WAS {curr_width_median}: {segment_widths_median[match_sk_branches[0]]}")
                            new_widths.append(segment_widths_median[match_sk_branches[0]])


                    #6) Remove the original branch and mesh correspondence and replace with the multiples
    #                     print(f"match_sk_branches BEFORE = {match_sk_branches}")
    #                     print(f"segment_branches BEFORE = {segment_branches}")
    #                     print(f"len(new_skeletal_branches) = {len(new_skeletal_branches)}")
    #                     print(f"new_skeletal_branches BEFORE= {new_skeletal_branches}")


                    #segment_branches = np.delete(segment_branches,match_sk_branches,axis=0)
                    #segment_branches = np.append(segment_branches,new_skeletal_branches,axis=0)

                    segment_branches = np.array([k for i,k in enumerate(segment_branches) if i not in match_sk_branches] + new_skeletal_branches)


                    divided_submeshes = np.delete(divided_submeshes,match_sk_branches,axis=0)
                    divided_submeshes = np.append(divided_submeshes,new_submeshes,axis=0)


                    #divided_submeshes_idx = np.delete(divided_submeshes_idx,match_sk_branches,axis=0)
                    #divided_submeshes_idx = np.append(divided_submeshes_idx,new_submeshes_idx,axis=0)
                    divided_submeshes_idx = np.array([k for i,k in enumerate(divided_submeshes_idx) if i not in match_sk_branches] + new_submeshes_idx)

                    segment_widths_median = np.delete(segment_widths_median,match_sk_branches,axis=0)
                    segment_widths_median = np.append(segment_widths_median,new_widths,axis=0)

                    try:
                        debug = False
                        if debug:
                            print(f"segment_branches.shape = {segment_branches.shape}")
                            print(f"segment_branches = {segment_branches}")
                            print(f"new_skeletal_branches = {new_skeletal_branches}")
                        sk.check_skeleton_connected_component(sk.stack_skeletons(segment_branches))
                    except:
                        su.compressed_pickle(local_correspondence_revised,"local_correspondence_revised")
                    print("checked segment branches after soma add on")
                    return_find = sk.find_branch_skeleton_with_specific_coordinate(segment_branches,
                                                 orig_vertex)



                    """ ******************* END OF HOW CAN DO STITCHING ********************  """



            limb_to_endpoints_must_keep_list.append(endpts_total)
            limb_to_soma_touching_vertices_list.append(curr_soma_to_piece_touching_vertices_total)

            # ------------------- 11/9 addition ------------------- #

            if no_soma_extension_add:
                print("No soma extending branch was added for this sublimb even though it had a soma border (means they already existed)")

            if print_fusion_steps:
                print(f"MP (because soma touching verts) soma extension add: {time.time() - fusion_time }")
                fusion_time = time.time()

        #building the limb correspondence
        limb_correspondence_MP[sublimb_idx] = dict()

        for zz,b_sk in enumerate(segment_branches):
            limb_correspondence_MP[sublimb_idx][zz] = dict(
                branch_skeleton = b_sk,
                width_from_skeleton = segment_widths_median[zz],
                branch_mesh = divided_submeshes[zz],
                branch_face_idx = divided_submeshes_idx[zz]
                )



    #limb_correspondence_MP_saved = copy.deepcopy(limb_correspondence_MP)
    #limb_correspondence_MAP_saved = copy.deepcopy(limb_correspondence_MAP)

    # ------------------------------------- Part C: Will make sure the correspondences can all be stitched together --------------- #

    # Only want to perform this step if both MP and MAP pieces
    if len(limb_correspondence_MAP)>0 and len(limb_correspondence_MP)>0:

        # -------------- Part 11: Getting Sublimb Mesh and Skeletons and Gets connectivitiy by Mesh -------#
        # -------------(filtering connections to only MP to MAP edges)--------------- #

        # ---- Doing the mesh connectivity ---------#
        sublimb_meshes_MP = []
        sublimb_skeletons_MP = []

        for sublimb_key,sublimb_v in limb_correspondence_MP.items():
            sublimb_meshes_MP.append(tu.combine_meshes([branch_v["branch_mesh"] for branch_v in sublimb_v.values()]))
            sublimb_skeletons_MP.append(sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in sublimb_v.values()]))



        sublimb_meshes_MAP = []
        sublimb_skeletons_MAP = []


        for sublimb_key,sublimb_v in limb_correspondence_MAP.items():
            sublimb_meshes_MAP.append(tu.combine_meshes([branch_v["branch_mesh"] for branch_v in sublimb_v.values()]))
            sublimb_skeletons_MAP.append(sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in sublimb_v.values()]))

        sublimb_skeletons_MP_saved = copy.deepcopy(sublimb_skeletons_MP)
        sublimb_skeletons_MAP_saved = copy.deepcopy(sublimb_skeletons_MAP)

        connectivity_type = "edges"
        for i in range(0,2):
            mesh_conn,mesh_conn_vertex_groups = tu.mesh_list_connectivity(meshes = sublimb_meshes_MP + sublimb_meshes_MAP,
                                                main_mesh = limb_mesh_mparty,
                                                connectivity=connectivity_type,
                                                min_common_vertices=1,
                                                return_vertex_connection_groups=True,
                                                return_largest_vertex_connection_group=True,
                                                print_flag = False)
            mesh_conn_old = copy.deepcopy(mesh_conn)



            #check that every MAP piece mapped to a MP piece
            mesh_conn_filt = []
            mesh_conn_vertex_groups_filt = []
            for j,(m1,m2) in enumerate(mesh_conn):
                if m1 < len(sublimb_meshes_MP) and m2 >=len(sublimb_meshes_MP):
                    mesh_conn_filt.append([m1,m2])
                    mesh_conn_vertex_groups_filt.append(mesh_conn_vertex_groups[j])
                else:
                    print(f"Edge {(m1,m2)} was not kept")
            mesh_conn_filt = np.array(mesh_conn_filt)

            mesh_conn = mesh_conn_filt
            mesh_conn_vertex_groups = mesh_conn_vertex_groups_filt

            #check that the mapping should create only one connected component
            G = nx.from_edgelist(mesh_conn)



            try:
                if len(G) != len(sublimb_meshes_MP) + len(sublimb_meshes_MAP):
                    raise Exception("Number of nodes in mesh connectivity graph is not equal to number of  MAP and MP sublimbs")

                connect_comp = list(nx.connected_components(G))
                if len(connect_comp)>1:
                    raise Exception(f"Mesh connectivity was not one component, instead it was ({len(connect_comp)}): {connect_comp} ")
            except:
                
                if connectivity_type == "vertices":
                    print(f"mesh_conn_filt = {mesh_conn_filt}")
                    print(f"mesh_conn_old = {mesh_conn_old}")
                    mesh_conn_adjusted = np.vstack([mesh_conn[:,0],mesh_conn[:,1]-len(sublimb_meshes_MP)]).T
                    print(f"mesh_conn_adjusted = {mesh_conn_adjusted}")
                    print(f"len(sublimb_meshes_MP) = {len(sublimb_meshes_MP)}")
                    print(f"len(sublimb_meshes_MAP) = {len(sublimb_meshes_MAP)}")
                    meshes = sublimb_meshes_MP + sublimb_meshes_MAP
                    #su.compressed_pickle(meshes,"meshes")
                    su.compressed_pickle(sublimb_meshes_MP,"sublimb_meshes_MP")
                    su.compressed_pickle(sublimb_meshes_MAP,"sublimb_meshes_MAP")
                    su.compressed_pickle(limb_mesh_mparty,"limb_mesh_mparty")
                    su.compressed_pickle(sublimb_skeletons_MP,"sublimb_skeletons_MP")
                    su.compressed_pickle(sublimb_skeletons_MAP,"sublimb_skeletons_MAP")




                    raise Exception("Something went wrong in the connectivity")
                else:
                    print(f"Failed on connection type {connectivity_type} ")
                    connectivity_type = "vertices"
                    print(f"so changing type to {connectivity_type}")
            else:
                print(f"Successful mesh connectivity with type {connectivity_type}")


        #adjust the connection indices for MP and MAP indices
        mesh_conn_adjusted = np.vstack([mesh_conn[:,0],mesh_conn[:,1]-len(sublimb_meshes_MP)]).T






        """
        Pseudocode:
        For each connection edge:
            For each vertex connection group:
                1) Get the endpoint vertices of the MP skeleton
                2) Find the closest endpoint vertex to the vertex connection group (this is MP stitch point)
                3) Find the closest skeletal point on MAP pairing (MAP stitch) 
                4) Find the branches that have that MAP stitch point:
                5A) If the number of branches corresponding to stitch point is multipled
                    --> then we are stitching at a branching oint
                    i) Just add the skeletal segment from MP_stitch to MAP stitch to the MP skeletal segment
                    ii) 

        """



        # -------------- STITCHING PHASE -------#
        stitch_counter = 0
        all_map_stitch_points = []
        for (MP_idx,MAP_idx),v_g in zip(mesh_conn_adjusted,mesh_conn_vertex_groups):
            print(f"\n---- Working on {(MP_idx,MAP_idx)} connection-----")

            """
            This old way of getting the endpoints was not good because could possibly just need
            a stitching done between original branch junction

            skeleton_MP_graph = sk.convert_skeleton_to_graph(curr_skeleton_MP)
            endpoint_nodes = xu.get_nodes_of_degree_k(skeleton_MP_graph,1)
            endpoint_nodes_coordinates = xu.get_node_attributes(skeleton_MP_graph,node_list=endpoint_nodes)
            """


            # -------------- Part 12: Find the MP and MAP stitching point and branches that contain the stitching point-------#

            """  OLD WAY THAT ALLOWED STITICHING POINTS TO NOT BE CONNECTED AT THE CONNECTING BRANCHES
            #getting the skeletons that should be stitched
            curr_skeleton_MP = sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in limb_correspondence_MP[MP_idx].values()])
            curr_skeleton_MAP = sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in limb_correspondence_MAP[MAP_idx].values()])

            #1) Get the endpoint vertices of the MP skeleton branches (so every endpoint or high degree node)
            #(needs to be inside loop because limb correspondence will change)
            curr_MP_branch_skeletons = [limb_correspondence_MP[MP_idx][k]["branch_skeleton"] for k in np.sort(list(limb_correspondence_MP[MP_idx].keys()))]
            endpoint_nodes_coordinates = np.array([sk.find_branch_endpoints(k) for k in curr_MP_branch_skeletons])
            endpoint_nodes_coordinates = np.unique(endpoint_nodes_coordinates.reshape(-1,3),axis=0)

            #2) Find the closest endpoint vertex to the vertex connection group (this is MP stitch point)
            av_vert = np.mean(v_g,axis=0)
            winning_vertex = endpoint_nodes_coordinates[np.argmin(np.linalg.norm(endpoint_nodes_coordinates-av_vert,axis=1))]
            print(f"winning_vertex = {winning_vertex}")


            #2b) Find the branch points where the winning vertex is located
            MP_branches_with_stitch_point = sk.find_branch_skeleton_with_specific_coordinate(
                divded_skeleton=curr_MP_branch_skeletons,
                current_coordinate = winning_vertex
            )
            print(f"MP_branches_with_stitch_point = {MP_branches_with_stitch_point}")


            #3) Find the closest skeletal point on MAP pairing (MAP stitch)
            MAP_skeleton_coords = np.unique(curr_skeleton_MAP.reshape(-1,3),axis=0)
            MAP_stitch_point = MAP_skeleton_coords[np.argmin(np.linalg.norm(MAP_skeleton_coords-winning_vertex,axis=1))]


            #3b) Consider if the stitch point is close enough to end or branch node in skeleton:
            # and if so then reassign
            if move_MAP_stitch_to_end_or_branch:
                MAP_stitch_point_new,change_status = sk.move_point_to_nearest_branch_end_point_within_threshold(
                                                        skeleton=curr_skeleton_MAP,
                                                        coordinate=MAP_stitch_point,
                                                        distance_to_move_point_threshold = distance_to_move_point_threshold,
                                                        verbose=True

                                                        )
                MAP_stitch_point=MAP_stitch_point_new


            #4) Find the branches that have that MAP stitch point:
            curr_MAP_branch_skeletons = [limb_correspondence_MAP[MAP_idx][k]["branch_skeleton"]
                                             for k in np.sort(list(limb_correspondence_MAP[MAP_idx].keys()))]

            MAP_branches_with_stitch_point = sk.find_branch_skeleton_with_specific_coordinate(
                divded_skeleton=curr_MAP_branch_skeletons,
                current_coordinate = MAP_stitch_point
            )



            MAP_stitch_point_on_end_or_branch = False
            if len(MAP_branches_with_stitch_point)>1:
                MAP_stitch_point_on_end_or_branch = True
            elif len(MAP_branches_with_stitch_point)==1:
                if len(nu.matching_rows(sk.find_branch_endpoints(curr_MAP_branch_skeletons[MAP_branches_with_stitch_point[0]]),
                                        MAP_stitch_point))>0:
                    MAP_stitch_point_on_end_or_branch=True
            else:
                raise Exception("No matching MAP values")

        """

            #*****should only get branches that are touching....****

            #getting the skeletons that should be stitched
            curr_skeleton_MP = sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in limb_correspondence_MP[MP_idx].values()])
            curr_skeleton_MAP = sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in limb_correspondence_MAP[MAP_idx].values()])


            av_vert = np.mean(v_g,axis=0)

            # ---------------- Doing the MAP part first -------------- #
            """
            The previous way did not ensure that the MAP point found will have a branch mesh that is touching the border vertices

            #3) Find the closest skeletal point on MAP pairing (MAP stitch)
            MAP_skeleton_coords = np.unique(curr_skeleton_MAP.reshape(-1,3),axis=0)

            #this does not guarentee that the MAP branch associated with the MAP stitch point is touching the border group
            MAP_stitch_point = MAP_skeleton_coords[np.argmin(np.linalg.norm(MAP_skeleton_coords-av_vert,axis=1))]
            """

            # -------------- 11/9 NEW METHOD FOR FINDING MAP STITCH POINT ------------ #
            o_keys = np.sort(list(limb_correspondence_MAP[MAP_idx].keys()))
            curr_MAP_branch_meshes = np.array([limb_correspondence_MAP[MAP_idx][k]["branch_mesh"]
                                             for k in o_keys])
            curr_MAP_branch_skeletons = np.array([limb_correspondence_MAP[MAP_idx][k]["branch_skeleton"]
                                             for k in o_keys])

            MAP_pieces_idx_touching_border = tu.filter_meshes_by_containing_coordinates(mesh_list=curr_MAP_branch_meshes,
                                           nullifying_points=v_g,
                                            filter_away=False,
                                           distance_threshold=0,
                                           return_indices=True)

            MAP_branches_considered = curr_MAP_branch_skeletons[MAP_pieces_idx_touching_border]
            curr_skeleton_MAP_for_stitch = sk.stack_skeletons(MAP_branches_considered)

            #3) Find the closest skeletal point on MAP pairing (MAP stitch)
            MAP_skeleton_coords = np.unique(curr_skeleton_MAP_for_stitch.reshape(-1,3),axis=0)

            #this does not guarentee that the MAP branch associated with the MAP stitch point is touching the border group
            MAP_stitch_point = MAP_skeleton_coords[np.argmin(np.linalg.norm(MAP_skeleton_coords-av_vert,axis=1))]

            # --------- 11/13: Making so could possibly stitch to another point that was already stitched to
            curr_br_endpts = np.array([sk.find_branch_endpoints(k) for k in MAP_branches_considered]).reshape(-1,3)
            curr_br_endpts_unique = np.unique(curr_br_endpts,axis=0)



            #3b) Consider if the stitch point is close enough to end or branch node in skeleton:
            # and if so then reassign
            if move_MAP_stitch_to_end_or_branch:
                MAP_stitch_point_new,change_status = sk.move_point_to_nearest_branch_end_point_within_threshold(
                                                        skeleton=curr_skeleton_MAP,
                                                        coordinate=MAP_stitch_point,
                                                        distance_to_move_point_threshold = distance_to_move_point_threshold,
                                                        verbose=True,
                                                        possible_node_coordinates=curr_br_endpts_unique,
                                                        )
                MAP_stitch_point=MAP_stitch_point_new


            #4) Find the branches that have that MAP stitch point:

            MAP_branches_with_stitch_point = sk.find_branch_skeleton_with_specific_coordinate(
                divded_skeleton=curr_MAP_branch_skeletons,
                current_coordinate = MAP_stitch_point
            )



            MAP_stitch_point_on_end_or_branch = False
            if len(MAP_branches_with_stitch_point)>1:
                MAP_stitch_point_on_end_or_branch = True
            elif len(MAP_branches_with_stitch_point)==1:
                if len(nu.matching_rows(sk.find_branch_endpoints(curr_MAP_branch_skeletons[MAP_branches_with_stitch_point[0]]),
                                        MAP_stitch_point))>0:
                    MAP_stitch_point_on_end_or_branch=True
            else:
                raise Exception("No matching MAP values")

            #add the map stitch point to the history
            all_map_stitch_points.append(MAP_stitch_point)

            # ---------------- Doing the MP Part --------------------- #



            ord_keys = np.sort(list(limb_correspondence_MP[MP_idx].keys()))
            curr_MP_branch_meshes = [limb_correspondence_MP[MP_idx][k]["branch_mesh"] for k in ord_keys]



            """ old way of filtering MP pieces just to those touching the MAP, but just want the ones touching the connection group

            MAP_meshes_with_stitch_point = tu.combine_meshes([limb_correspondence_MAP[MAP_idx][k]["branch_mesh"] for k in MAP_branches_with_stitch_point])

            conn = tu.mesh_pieces_connectivity(main_mesh=limb_mesh_mparty,
                                       central_piece=MAP_meshes_with_stitch_point,
                                       periphery_pieces=curr_MP_branch_meshes)
            """
            # 11/9 Addition: New way that filters meshes by their touching of the vertex connection group (this could possibly be an empty group)
            conn = tu.filter_meshes_by_containing_coordinates(mesh_list=curr_MP_branch_meshes,
                                           nullifying_points=v_g,
                                            filter_away=False,
                                           distance_threshold=0,
                                           return_indices=True)

            if len(conn) == 0:
                print("Connectivity was 0 for the MP mesh groups touching the vertex group so not restricting by that anymore")
                sk_conn = np.arange(0,len(curr_MP_branch_meshes))
            else:
                sk_conn = conn

            print(f"sk_conn = {sk_conn}")
            print(f"conn = {conn}")


            #1) Get the endpoint vertices of the MP skeleton branches (so every endpoint or high degree node)
            #(needs to be inside loop because limb correspondence will change)
            curr_MP_branch_skeletons = [limb_correspondence_MP[MP_idx][k]["branch_skeleton"] for k in sk_conn]
            endpoint_nodes_coordinates = np.array([sk.find_branch_endpoints(k) for k in curr_MP_branch_skeletons])
            endpoint_nodes_coordinates = np.unique(endpoint_nodes_coordinates.reshape(-1,3),axis=0)


            #2) Find the closest endpoint vertex to the vertex connection group (this is MP stitch point)

            winning_vertex = endpoint_nodes_coordinates[np.argmin(np.linalg.norm(endpoint_nodes_coordinates-av_vert,axis=1))]
            print(f"winning_vertex = {winning_vertex}")


            #2b) Find the branch points where the winning vertex is located
            curr_MP_branch_skeletons = [limb_correspondence_MP[MP_idx][k]["branch_skeleton"] for k in np.sort(list(limb_correspondence_MP[MP_idx].keys()))]
            MP_branches_with_stitch_point = sk.find_branch_skeleton_with_specific_coordinate(
                divded_skeleton=curr_MP_branch_skeletons,
                current_coordinate = winning_vertex
            )
            print(f"MP_branches_with_stitch_point = {MP_branches_with_stitch_point}")



            print(f"MAP_branches_with_stitch_point = {MAP_branches_with_stitch_point}")
            print(f"MAP_stitch_point_on_end_or_branch = {MAP_stitch_point_on_end_or_branch}")


            # -------- 11/13 addition: Will see if the MP stitch point was already a MAP stitch point ---- #
            if len(nu.matching_rows(np.array(all_map_stitch_points),winning_vertex)) > 0:
                keep_MP_stitch_static = True
            else:
                keep_MP_stitch_static = False





            # ------------------------- This part does the stitching -------------------- #


            """
            Pseudocode:
            1) For all MP branches
                a) Get neighbor coordinates to MP stitch points
                b) Delete the MP Stitch points on each 
                c) Add skeleton segment from neighbor to MAP stitch point
            2) Get skeletons and meshes from MP and MAP pieces
            3) Run mesh correspondence to get new meshes and mesh_idx and widths
            4a) If MAP_stitch_point_on_end_or_branch is False
            - Delete the old MAP branch parts and replace with new MAP ones
            4b) Revise the meshes,  mesh_idx, and widths of the MAP pieces
            5) Revise the meshes,  mesh_idx, and widths of the MP pieces


            """

            # -------------- Part 13: Will Adjust the MP branches that have the stitch point so extends to the MAP stitch point -------#
            curr_MP_sk = []
            for b_idx in MP_branches_with_stitch_point:
                if not keep_MP_stitch_static:
                    #a) Get neighbor coordinates to MP stitch points
                    MP_stitch_branch_graph = sk.convert_skeleton_to_graph(curr_MP_branch_skeletons[b_idx])
                    stitch_node = xu.get_nodes_with_attributes_dict(MP_stitch_branch_graph,dict(coordinates=winning_vertex))[0]
                    stitch_neighbors = xu.get_neighbors(MP_stitch_branch_graph,stitch_node)

                    if len(stitch_neighbors) != 1:
                        raise Exception("Not just one neighbor for stitch point of MP branch")
                    keep_neighbor = stitch_neighbors[0]  
                    keep_neighbor_coordinates = xu.get_node_attributes(MP_stitch_branch_graph,node_list=[keep_neighbor])[0]

                    #b) Delete the MP Stitch points on each 
                    MP_stitch_branch_graph.remove_node(stitch_node)

                    """ Old way that does not do smoothing

                    #c) Add skeleton segment from neighbor to MAP stitch point
                    new_node_name = np.max(MP_stitch_branch_graph.nodes())+1

                    MP_stitch_branch_graph.add_nodes_from([(int(new_node_name),{"coordinates":MAP_stitch_point})])
                    MP_stitch_branch_graph.add_weighted_edges_from([(keep_neighbor,new_node_name,np.linalg.norm(MAP_stitch_point - keep_neighbor_coordinates))])

                    new_MP_skeleton = sk.convert_graph_to_skeleton(MP_stitch_branch_graph)

                    """
                    try:
                        if len(MP_stitch_branch_graph)>1:
                            new_MP_skeleton = sk.add_and_smooth_segment_to_branch(skeleton=sk.convert_graph_to_skeleton(MP_stitch_branch_graph),
                                                            skeleton_stitch_point=keep_neighbor_coordinates,
                                                             new_stitch_point=MAP_stitch_point)
                        else:
                            print("Not even attempting smoothing segment because once keep_neighbor_coordinates")
                            new_MP_skeleton = np.vstack([keep_neighbor_coordinates,MAP_stitch_point]).reshape(-1,2,3)
                    except:
                        su.compressed_pickle(MP_stitch_branch_graph,"MP_stitch_branch_graph")
                        su.compressed_pickle(keep_neighbor_coordinates,"keep_neighbor_coordinates")
                        su.compressed_pickle(MAP_stitch_point,"MAP_stitch_point")


                        raise Exception("Something went wrong with add_and_smooth_segment_to_branch")





                    #smooth over the new skeleton
                    new_MP_skeleton_smooth = sk.resize_skeleton_branch(new_MP_skeleton,
                                                                      segment_width=meshparty_segment_size)

                    curr_MP_sk.append(new_MP_skeleton_smooth)
                else:
                    print(f"Not adjusting MP skeletons because keep_MP_stitch_static = {keep_MP_stitch_static}")
                    curr_MP_sk.append(curr_MP_branch_skeletons[b_idx])



            #2) Get skeletons and meshes from MP and MAP pieces
            curr_MAP_sk = [limb_correspondence_MAP[MAP_idx][k]["branch_skeleton"] for k in MAP_branches_with_stitch_point]

            #2.1) Going to break up the MAP skeleton if need be
            """
            Pseudocode:
            a) check to see if it needs to be broken up
            If it does:
            b) Convert the skeleton into a graph
            c) Find the node of the MAP stitch point (where need to do the breaking)
            d) Find the degree one nodes
            e) For each degree one node:
            - Find shortest path from stitch node to end node
            - get a subgraph from that path
            - convert graph to a skeleton and save as new skeletons

            """
            # -------------- Part 14: Breaks Up MAP skeleton into 2 pieces if Needs (because MAP stitch point not on endpoint or branch point)  -------#

            #a) check to see if it needs to be broken up
            cut_flag = False
            if not MAP_stitch_point_on_end_or_branch:
                if len(curr_MAP_sk) > 1:
                    raise Exception(f"There was more than one skeleton for MAP skeletons even though MAP_stitch_point_on_end_or_branch = {MAP_stitch_point_on_end_or_branch}")


                skeleton_to_cut = curr_MAP_sk[0]
                curr_MAP_sk = sk.cut_skeleton_at_coordinate(skeleton=skeleton_to_cut,
                                                            cut_coordinate=MAP_stitch_point)
                cut_flag=True


            # ------ 11/13 Addition: need to adjust the MAP points if have to keep MP static
            if keep_MP_stitch_static:
                curr_MAP_sk_final = []
                for map_skel in curr_MAP_sk:
                    #a) Get neighbor coordinates to MP stitch points
                    MP_stitch_branch_graph = sk.convert_skeleton_to_graph(map_skel)
                    stitch_node = xu.get_nodes_with_attributes_dict(MP_stitch_branch_graph,dict(coordinates=MAP_stitch_point))[0]
                    stitch_neighbors = xu.get_neighbors(MP_stitch_branch_graph,stitch_node)

                    if len(stitch_neighbors) != 1:
                        raise Exception("Not just one neighbor for stitch point of MP branch")
                    keep_neighbor = stitch_neighbors[0]  
                    keep_neighbor_coordinates = xu.get_node_attributes(MP_stitch_branch_graph,node_list=[keep_neighbor])[0]

                    #b) Delete the MP Stitch points on each 
                    MP_stitch_branch_graph.remove_node(stitch_node)

                    """ Old way that does not do smoothing

                    #c) Add skeleton segment from neighbor to MAP stitch point
                    new_node_name = np.max(MP_stitch_branch_graph.nodes())+1

                    MP_stitch_branch_graph.add_nodes_from([(int(new_node_name),{"coordinates":MAP_stitch_point})])
                    MP_stitch_branch_graph.add_weighted_edges_from([(keep_neighbor,new_node_name,np.linalg.norm(MAP_stitch_point - keep_neighbor_coordinates))])

                    new_MP_skeleton = sk.convert_graph_to_skeleton(MP_stitch_branch_graph)

                    """
                    try:
                        if len(MP_stitch_branch_graph)>1:
                            new_MP_skeleton = sk.add_and_smooth_segment_to_branch(skeleton=sk.convert_graph_to_skeleton(MP_stitch_branch_graph),
                                                            skeleton_stitch_point=keep_neighbor_coordinates,
                                                             new_stitch_point=winning_vertex)
                        else:
                            print("Not even attempting smoothing segment because once keep_neighbor_coordinates")
                            new_MP_skeleton = np.vstack([keep_neighbor_coordinates,MAP_stitch_point]).reshape(-1,2,3)
                    except:
                        su.compressed_pickle(MP_stitch_branch_graph,"MP_stitch_branch_graph")
                        su.compressed_pickle(keep_neighbor_coordinates,"keep_neighbor_coordinates")
                        su.compressed_pickle(MAP_stitch_point,"MAP_stitch_point")


                        raise Exception("Something went wrong with add_and_smooth_segment_to_branch")





                    #smooth over the new skeleton
                    new_MP_skeleton_smooth = sk.resize_skeleton_branch(new_MP_skeleton,
                                                                      segment_width=meshparty_segment_size)

                    curr_MAP_sk_final.append(new_MP_skeleton_smooth)
                curr_MAP_sk = copy.deepcopy(curr_MAP_sk_final)



            # -------------- Part 15: Gets all of the skeletons and Mesh to divide u and does mesh correspondence -------#
            # ------------- revise IDX so still references the whole limb mesh -----------#

            # -------------- 11/10 Addition accounting for not all MAP pieces always touching each other --------------------#
            if len(MAP_branches_with_stitch_point) > 1:
                print("\nRevising the MAP pieces index:")
                print(f"MAP_pieces_idx_touching_border = {MAP_pieces_idx_touching_border}, MAP_branches_with_stitch_point = {MAP_branches_with_stitch_point}")
                MAP_pieces_for_correspondence = nu.intersect1d(MAP_pieces_idx_touching_border,MAP_branches_with_stitch_point)
                print(f"MAP_pieces_for_correspondence = {MAP_pieces_for_correspondence}")
                curr_MAP_sk = [limb_correspondence_MAP[MAP_idx][k]["branch_skeleton"] for k in MAP_pieces_for_correspondence]
            else:
                MAP_pieces_for_correspondence = MAP_branches_with_stitch_point

            curr_MAP_meshes_idx = [limb_correspondence_MAP[MAP_idx][k]["branch_face_idx"] for k in MAP_pieces_for_correspondence]

            # Have to adjust based on if the skeleton were split

            if cut_flag:
                #Then it was cut and have to do mesh correspondence to find what label to cut
                if len(curr_MAP_meshes_idx) > 1:
                    raise Exception("MAP_pieces_for_correspondence was longer than 1 and cut flag was set")
                pre_stitch_mesh_idx = curr_MAP_meshes_idx[0]
                pre_stitch_mesh = limb_mesh_mparty.submesh([pre_stitch_mesh_idx],append=True,repair=False)
                local_correspondnece_stitch = mesh_correspondence_first_pass(mesh=pre_stitch_mesh,
                                          skeleton_branches=curr_MAP_sk)
                local_correspondence_stitch_revised = correspondence_1_to_1(mesh=pre_stitch_mesh,
                                                            local_correspondence=local_correspondnece_stitch,
                                                            curr_limb_endpoints_must_keep=None,
                                                            curr_soma_to_piece_touching_vertices=None)

                curr_MAP_meshes_idx = [pre_stitch_mesh_idx[local_correspondence_stitch_revised[nn]["branch_face_idx"]] for 
                                               nn in local_correspondence_stitch_revised.keys()]


            #To make sure that the MAP never gives up ground on the labels
            must_keep_labels_MAP = dict()
            must_keep_counter = 0
            for kk,b_idx in enumerate(curr_MAP_meshes_idx):
                #must_keep_labels_MAP.update(dict([(ii,kk) for ii in range(must_keep_counter,must_keep_counter+len(b_idx))]))
                must_keep_labels_MAP[kk] = np.arange(must_keep_counter,must_keep_counter+len(b_idx))
                must_keep_counter += len(b_idx)



            #this is where should send only the MP that apply
            MP_branches_for_correspondence,conn_idx,MP_branches_with_stitch_point_idx = nu.intersect1d(conn,MP_branches_with_stitch_point,return_indices=True)

            curr_MP_meshes_idx = [limb_correspondence_MP[MP_idx][k]["branch_face_idx"] for k in MP_branches_for_correspondence]
            curr_MP_sk_for_correspondence = [curr_MP_sk[zz] for zz in MP_branches_with_stitch_point_idx]

            stitching_mesh_idx = np.concatenate(curr_MAP_meshes_idx + curr_MP_meshes_idx)
            stitching_mesh = limb_mesh_mparty.submesh([stitching_mesh_idx],append=True,repair=False)
            stitching_skeleton_branches = curr_MAP_sk + curr_MP_sk_for_correspondence
            """

            ****** NEED TO GET THE RIGHT MESH TO RUN HE IDX ON SO GETS A GOOD MESH (CAN'T BE LIMB_MESH_MPARTY)
            BUT MUST BE THE ORIGINAL MAP MESH

            mesh_pieces_for_MAP
            sublimb_meshes_MP

            mesh_pieces_for_MAP_face_idx
            sublimb_meshes_MP_face_idx

            stitching_mesh = tu.combine_meshes(curr_MAP_meshes + curr_MP_meshes)
            stitching_skeleton_branches = curr_MAP_sk + curr_MP_sk

            """
            
            # ******************************** this is where should do thing about no mesh correspondence ***************** #

            #3) Run mesh correspondence to get new meshes and mesh_idx and widths
            local_correspondnece_stitch = mesh_correspondence_first_pass(mesh=stitching_mesh,
                                          skeleton_branches=stitching_skeleton_branches)

            try:

                local_correspondence_stitch_revised = correspondence_1_to_1(mesh=stitching_mesh,
                                                            local_correspondence=local_correspondnece_stitch,
                                                            curr_limb_endpoints_must_keep=None,
                                                            curr_soma_to_piece_touching_vertices=None,
                                                            must_keep_labels=must_keep_labels_MAP)
            except:
                su.compressed_pickle(stitching_skeleton_branches,"stitching_skeleton_branches")
                su.compressed_pickle(stitching_mesh,"stitching_mesh")
                su.compressed_pickle(local_correspondnece_stitch,"local_correspondnece_stitch")
                raise Exception("Something went wrong with 1 to 1 correspondence")


            #Need to readjust the mesh correspondence idx
            for k,v in local_correspondence_stitch_revised.items():
                local_correspondence_stitch_revised[k]["branch_face_idx"] = stitching_mesh_idx[local_correspondence_stitch_revised[k]["branch_face_idx"]]




            # -------------- Part 16: Overwrite old branch entries (and add on one new to MAP if required a split) -------#


            #4a) If MAP_stitch_point_on_end_or_branch is False
            #- Delete the old MAP branch parts and replace with new MAP ones
            if not MAP_stitch_point_on_end_or_branch:
                print("Deleting branches from dictionary")
                del limb_correspondence_MAP[MAP_idx][MAP_branches_with_stitch_point[0]]
                #adding the two new branches created from the stitching
                limb_correspondence_MAP[MAP_idx][MAP_branches_with_stitch_point[0]] = local_correspondence_stitch_revised[0]
                limb_correspondence_MAP[MAP_idx][np.max(list(limb_correspondence_MAP[MAP_idx].keys()))+1] = local_correspondence_stitch_revised[1]

                #have to reorder the keys
                #limb_correspondence_MAP[MAP_idx] = dict([(k,limb_correspondence_MAP[MAP_idx][k]) for k in np.sort(list(limb_correspondence_MAP[MAP_idx].keys()))])
                limb_correspondence_MAP[MAP_idx] = gu.order_dict_by_keys(limb_correspondence_MAP[MAP_idx])

            else: #4b) Revise the meshes,  mesh_idx, and widths of the MAP pieces if weren't broken up
                for j,curr_MAP_idx_fixed in enumerate(MAP_pieces_for_correspondence): 
                    limb_correspondence_MAP[MAP_idx][curr_MAP_idx_fixed] = local_correspondence_stitch_revised[j]
                #want to update all of the skeletons just in case was altered by keep_MP_stitch_static and not included in correspondence
                if keep_MP_stitch_static:
                    if len(MAP_branches_with_stitch_point) != len(curr_MAP_sk_final):
                        raise Exception("MAP_branches_with_stitch_point not same size as curr_MAP_sk_final")
                    for gg,map_idx_curr in enumerate(MAP_branches_with_stitch_point):
                        limb_correspondence_MAP[MAP_idx][map_idx_curr]["branch_skeleton"] = curr_MAP_sk_final[gg]


            for j,curr_MP_idx_fixed in enumerate(MP_branches_for_correspondence): #************** right here just need to make only the ones that applied
                limb_correspondence_MP[MP_idx][curr_MP_idx_fixed] = local_correspondence_stitch_revised[j+len(curr_MAP_sk)]


            #5b) Fixing the branch skeletons that were not included in the correspondence
            MP_leftover,MP_leftover_idx = nu.setdiff1d(MP_branches_with_stitch_point,MP_branches_for_correspondence)
            print(f"MP_branches_with_stitch_point= {MP_branches_with_stitch_point}")
            print(f"MP_branches_for_correspondence = {MP_branches_for_correspondence}")
            print(f"MP_leftover = {MP_leftover}, MP_leftover_idx = {MP_leftover_idx}")

            for curr_MP_leftover,curr_MP_leftover_idx in zip(MP_leftover,MP_leftover_idx):
                limb_correspondence_MP[MP_idx][curr_MP_leftover]["branch_skeleton"] = curr_MP_sk[curr_MP_leftover_idx]


            print(f" Finished with {(MP_idx,MAP_idx)} \n\n\n")
            stitch_counter += 1
    #         if cut_flag:
    #             raise Exception("Cut flag was activated")


    else:
        print("There were not both MAP and MP pieces so skipping the stitch resolving phase")

    print(f"Time for decomp of Limb = {time.time() - curr_limb_time}")
    #     # ------------- Saving the MAP and MP Decompositions ---------------- #
    #     proper_limb_mesh_correspondence_MAP[curr_limb_idx] = limb_correspondence_MAP
    #     proper_limb_mesh_correspondence_MP[curr_limb_idx] = limb_correspondence_MP






    # -------------- Part 17: Grouping the MP and MAP Correspondence into one correspondence dictionary -------#
    limb_correspondence_individual = dict()
    counter = 0

    for sublimb_idx,sublimb_branches in limb_correspondence_MAP.items():
        for branch_dict in sublimb_branches.values():
            limb_correspondence_individual[counter]= branch_dict
            counter += 1
    for sublimb_idx,sublimb_branches in limb_correspondence_MP.items():
        for branch_dict in sublimb_branches.values():
            limb_correspondence_individual[counter]= branch_dict
            counter += 1


    #info that may be used for concept networks
    network_starting_info = dict(
                touching_verts_list = limb_to_soma_touching_vertices_list,
                endpoints_must_keep = limb_to_endpoints_must_keep_list
    )

    
    
    
    
    
    # -------------- Part 18: 11-17 Addition that filters the network starting info into a more clean presentation ------------ #
    """
    Pseudocode: 
    1) Rearrange the network starting info into a ditionary mapping
      soma_idx --> branch_broder_group --> list of dict(touching_vertices,endpoint)

    2) iterate through all the somas and border vertex groups
    a. filter to only those with an endpoint that is on a branch of the skeleton
    b1: If 1 --> then keep that one
    b2: If more --> pick the one with the endpoint closest to the average fo the vertex group
    b3: If 0 --> find the best available soma extending branch endpoint

    """

    # Part 1: Rearrange network info


    t_verts_list_total,enpts_list_total = network_starting_info.values()
    network_starting_info_revised = dict()
    for j,(v_list_dict,enpts_list_dict) in enumerate(zip(t_verts_list_total,enpts_list_total)):
        #print(f"---- Working on {j} -----")
    #     print(v_list_dict)
    #     print(enpts_list_dict)
        if set(list(v_list_dict.keys())) != set(list(enpts_list_dict)):
            raise Exception("Soma keys not match for touching vertices and endpoints")
        for sm_idx in v_list_dict.keys():
            v_list_soma = v_list_dict[sm_idx]
            endpt_soma = enpts_list_dict[sm_idx]
            if len(v_list_soma) != len(endpt_soma):
                raise Exception(f"touching vertices list and endpoint list not match size for soma {sm_idx}")

            all_border_vertex_groups = soma_touching_vertices_dict[sm_idx]

            for v_l,endpt in zip(v_list_soma,endpt_soma):

                matching_border_group  = []
                for i,curr_border_group in enumerate(all_border_vertex_groups):
                    if nu.test_matching_vertices_in_lists(curr_border_group,v_l,verbose=True):
                        matching_border_group.append(i)

                if len(matching_border_group) == 0 or len(matching_border_group)>1:
                    raise Exception(f"Matching border groups was not exactly 1: {matching_border_group}")

                winning_border_group = matching_border_group[0]

                if sm_idx not in network_starting_info_revised.keys():
                    network_starting_info_revised[sm_idx] = dict()

                if winning_border_group not in network_starting_info_revised[sm_idx].keys():
                    network_starting_info_revised[sm_idx][winning_border_group] = []
                network_starting_info_revised[sm_idx][winning_border_group].append(dict(touching_verts=v_l,endpoint=endpt))


    # Part 2 Filter
    """
    2) iterate through all the somas and border vertex groups
    a. filter to only those with an endpoint that is on a branch of the skeleton
    b1: If 1 --> then keep that one
    b2: If more --> pick the one with the endpoint closest to the average fo the vertex group
    b3: If 0 --> find the best available soma extending branch endpoint

    Pseudocode for b3:
    i) get all meshes that touch the vertex group (and keep the vertices that overlap)
    --> error if none
    ii) Get all of the endpoints of all matching branches
    iii) Filter the endpoints to only those that are degree 1 in the overall skeleton
    --> if none then just keep all endpoints
    iv) Find the closest viable endpoint to the mean of the boundary group
    v) save the overlap vertices and the winning endpoint as a dictionary

    """

    sorted_keys = np.sort(list(limb_correspondence_individual.keys()))
    curr_branches = [limb_correspondence_individual[k]["branch_skeleton"] for k in sorted_keys]
    curr_meshes = [limb_correspondence_individual[k]["branch_mesh"] for k in sorted_keys]

    network_starting_info_revised_cleaned = dict()
    for soma_idx in network_starting_info_revised.keys():
        network_starting_info_revised_cleaned[soma_idx] = dict()
        for bound_g_idx,endpoint_list in network_starting_info_revised[soma_idx].items():
            endpoint_list = np.array(endpoint_list)

            filter_on_skeleton_list = []
            for zz,endpt_dict in enumerate(endpoint_list):
                #a. filter to only those with an endpoint that is on a branch of the skeleton
                sk_indices = sk.find_branch_skeleton_with_specific_coordinate(divded_skeleton=curr_branches,
                                                                            current_coordinate=endpt_dict["endpoint"])
                if len(sk_indices) > 0:
                    filter_on_skeleton_list.append(zz)

            endpoint_list_filt = endpoint_list[filter_on_skeleton_list]



            curr_border_group_coordinates = soma_touching_vertices_dict[soma_idx][bound_g_idx]
            boundary_mean = np.mean(curr_border_group_coordinates,axis=0)

            if len(endpoint_list_filt) == 1:
                print("Only one endpoint after filtering away the endpoints that are not on the skeleton")
                winning_dict = endpoint_list_filt[0]
            #b2: If more --> pick the one with the endpoint closest to the average fo the vertex group
            elif len(endpoint_list_filt) > 1:
                print(f"MORE THAN one endpoint after filtering away the endpoints that are not on the skeleton: {len(endpoint_list_filt)}")
                viable_endpoints = [endpt_dict["endpoint"] for endpt_dict in endpoint_list_filt]


                distanes_from_mean = np.linalg.norm(viable_endpoints-boundary_mean,axis=1)
                winning_endpoint_idx = np.argmin(distanes_from_mean)
                winning_dict = endpoint_list_filt[winning_endpoint_idx]

            #if there was no clear winner
            else:
                """
                Pseudocode for no viable options:
                i) get all meshes that touch the vertex group (and keep the vertices that overlap)
                --> error if none
                ii) Get all of the endpoints of all matching branches
                iii) Filter the endpoints to only those that are degree 1 in the overall skeleton
                --> if none then just keep all endpoints
                iv) Find the closest viable endpoint to the mean of the boundary group
                v) save the overlap vertices and the winning endpoint as a dictionary


                """
                print("Having to find a new branch point")
                #i) get all meshes that touch the vertex group (and keep the vertices that overlap)
                mesh_indices_on_border = tu.filter_meshes_by_containing_coordinates(curr_meshes,
                                              nullifying_points=curr_border_group_coordinates,
                                              filter_away=False,
                                              distance_threshold=0,
                                              return_indices=True)
                if len(mesh_indices_on_border) == 0:
                    raise Exception("There were no meshes that were touching the boundary group")

                total_skeleton_graph = sk.convert_skeleton_to_graph(sk.stack_skeletons(curr_branches))
                skeleton_branches_on_border = [k for n,k in enumerate(curr_branches) if n in mesh_indices_on_border]
                skeleton_branches_on_border_endpoints = np.array([sk.find_branch_endpoints(k) for k in skeleton_branches_on_border])



                viable_endpoints = []
                for enpt in skeleton_branches_on_border_endpoints.reshape(-1,3):
                    curr_enpt_node = xu.get_graph_node_by_coordinate(total_skeleton_graph,enpt,return_single_value=True)
                    curr_enpt_degree = xu.get_node_degree(total_skeleton_graph,curr_enpt_node)
                    #print(f"curr_enpt_degree = {curr_enpt_degree}")
                    if curr_enpt_degree == 1:
                        viable_endpoints.append(enpt)

                if len(viable_endpoints) == 0:
                    print("No branch endpoints were degree 1 so just using all endpoints")
                    viable_endpoints = skeleton_branches_on_border_endpoints.reshape(-1,3)

                distanes_from_mean = np.linalg.norm(viable_endpoints-boundary_mean,axis=1)
                winning_endpoint = viable_endpoints[np.argmin(distanes_from_mean)]


                sk_indices = sk.find_branch_skeleton_with_specific_coordinate(divded_skeleton=curr_branches,
                                                                                        current_coordinate=winning_endpoint)

                winning_branch = np.intersect1d(mesh_indices_on_border,sk_indices)
                if len(winning_branch) == 0:
                    raise Exception("There was no winning branch for the creation of a new soma extending branch")
                else:
                    winning_branch_single = winning_branch[0]


                winning_touching_vertices = tu.filter_vertices_by_mesh(curr_meshes[winning_branch_single],curr_border_group_coordinates)
                winning_dict = dict(touching_verts=winning_touching_vertices,endpoint=winning_endpoint)








            network_starting_info_revised_cleaned[soma_idx][bound_g_idx] = winning_dict


    # -------------- Part 18: End ------------ #
    
    
    
    
    
    
    
    
    
    
    
    
    
    if not return_concept_network:
        if return_concept_network_starting_info: #because may want to calculate the concept networks later
            return limb_correspondence_individual,network_starting_info_revised_cleaned
        else:
            return limb_correspondence_individual
    else:
        limb_to_soma_concept_networks = calculate_limb_concept_networks(limb_correspondence_individual,
                                                                        network_starting_info_revised_cleaned,
                                                                        run_concept_network_checks=run_concept_network_checks,
                                                                       )




    return limb_correspondence_individual,limb_to_soma_concept_networks






def preprocess_neuron(
                mesh=None,
                mesh_file=None,
                segment_id=None,
                 description=None,
                sig_th_initial_split=15, #for significant splitting meshes in the intial mesh split
                limb_threshold = 2000, #the mesh faces threshold for a mesh to be qualified as a limb (otherwise too small)
    
                filter_end_node_length=1500,#4001, #used in cleaning the skeleton during skeletonizations
                return_no_somas = False, #whether to error or to return an empty list for somas
    
                decomposition_type="meshafterparty",
                distance_by_mesh_center=True,
                meshparty_segment_size =100,
    
                meshparty_n_surface_downsampling = 2,

                somas=None, #the precomputed somas
                combine_close_skeleton_nodes = True,
                combine_close_skeleton_nodes_threshold=700,

                use_meshafterparty=True):
    pre_branch_connectivity = "edges"
    print(f"use_meshafterparty = {use_meshafterparty}")
    
    whole_processing_tiempo = time.time()


    """
    Purpose: To process the mesh into a format that can be loaded into the neuron class
    and used for higher order processing (how to visualize is included)
    
    This method includes the fusion

    """
    if description is None:
        description = "no_description"
    if segment_id is None:
        #pick a random segment id
        segment_id = np.random.randint(100000000)
        print(f"picking a random 7 digit segment id: {segment_id}")
        description += "_random_id"


    if mesh is None:
        if mesh_file is None:
            raise Exception("No mesh or mesh_file file were given")
        else:
            current_neuron = tu.load_mesh_no_processing(mesh_file)
    else:
        current_neuron = mesh
        
        
        
        
        
        
    # -------- Phase 1: Doing Soma Detection (if Not already done) ---------- #
    if somas is None:
        soma_mesh_list,run_time,total_soma_list_sdf = sm.extract_soma_center(segment_id,
                                                 current_neuron.vertices,
                                                 current_neuron.faces)
    else:
        soma_mesh_list,run_time,total_soma_list_sdf = somas
        print(f"Using pre-computed somas: soma_mesh_list = {soma_mesh_list}")

    # geting the soma centers
    if len(soma_mesh_list) <= 0:
        print(f"**** No Somas Found for Mesh {segment_id} so just one mesh")
        soma_mesh_list_centers = []
        if return_no_somas:
            return_value= soma_mesh_list_centers
        raise Exception("Processing of No Somas is not yet implemented yet")
    else:
        #compute the soma centers
        print(f"Soma List = {soma_mesh_list}")

        soma_mesh_list_centers = sm.find_soma_centroids(soma_mesh_list)
        print(f"soma_mesh_list_centers = {soma_mesh_list_centers}")
        
        
        
        
        
    
    #--- Phase 2: getting the soma submeshes that are connected to each soma and identifiying those that aren't 
    # ------------------ (and eliminating any mesh pieces inside the soma) ------------------------

    # -------- 11/13 Addition: Will remove the inside nucleus --------- #
    interior_time = time.time()
    main_mesh_total,inside_nucleus_pieces = tu.remove_mesh_interior(current_neuron,return_removed_pieces=True,
                                                                   try_hole_close=False)
    print(f"Total time for removing interior = {time.time() - interior_time}")


    #finding the mesh pieces that contain the soma
    #splitting the current neuron into distinct pieces
    split_time = time.time()
    split_meshes = tu.split_significant_pieces(
                                main_mesh_total,
                                significance_threshold=sig_th_initial_split,
                                print_flag=False,
                                connectivity=pre_branch_connectivity)
    print(f"Total time for splitting mesh = {time.time() - split_time}")

    print(f"# total split meshes = {len(split_meshes)}")

    #returns the index of the split_meshes index that contains each soma    
    containing_mesh_indices = sm.find_soma_centroid_containing_meshes(soma_mesh_list,
                                            split_meshes)

    # filtering away any of the inside floating pieces: 
    non_soma_touching_meshes = [m for i,m in enumerate(split_meshes)
                     if i not in list(containing_mesh_indices.values())]

    #Adding the step that will filter away any pieces that are inside the soma
    if len(non_soma_touching_meshes) > 0 and len(soma_mesh_list) > 0:
        """
        *** want to save these pieces that are inside of the soma***
        """

        non_soma_touching_meshes,inside_pieces = sm.filter_away_inside_soma_pieces(soma_mesh_list,non_soma_touching_meshes,
                                        significance_threshold=sig_th_initial_split,
                                        return_inside_pieces = True)
    
    else:
        non_soma_touching_meshes = []
        inside_pieces=[]
    
    #adding in the nuclei center to the inside pieces
    inside_pieces += inside_nucleus_pieces


    split_meshes # the meshes of the original mesh
    containing_mesh_indices #the mapping of each soma centroid to the correct split mesh
    soma_containing_meshes = sm.grouping_containing_mesh_indices(containing_mesh_indices)

    soma_touching_meshes = [split_meshes[k] for k in soma_containing_meshes.keys()]


    #     print(f"# of non soma touching seperate meshes = {len(non_soma_touching_meshes)}")
    #     print(f"# of inside pieces = {len(inside_pieces)}")
    print(f"\n-----Before filtering away multiple disconneted soma pieces-----")
    print(f"# of soma containing seperate meshes = {len(soma_touching_meshes)}")
    print(f"meshes with somas = {soma_containing_meshes}")
    
    # ------ 11/15 Addition: Part 2.b 

    """
    Pseudocode: 
    1) Get the largest of the meshes with a soma (largest in soma_touching_meshes)
    2) Save all other meshes not the largest in 
    3) Overwrite the following variables:
        soma_mesh_list
        soma_containing_meshes
        soma_touching_meshes
        total_soma_list_sdf


    """
    #1) Get the largest of the meshes with a soma (largest in soma_touching_meshes)
    soma_containing_meshes_keys = np.array(list(soma_containing_meshes.keys()))
    soma_touching_meshes = np.array([split_meshes[k] for k in soma_containing_meshes_keys])
    largest_soma_touching_mesh_idx = soma_containing_meshes_keys[np.argmax([len(kk.faces) for kk in soma_touching_meshes])]

    #2) Save all other meshes not the largest in 
    not_processed_soma_containing_meshes_idx = np.setdiff1d(soma_containing_meshes_keys,[largest_soma_touching_mesh_idx])
    not_processed_soma_containing_meshes = [split_meshes[k] for k in not_processed_soma_containing_meshes_idx]
    print(f"Number of not_processed_soma_containing_meshes = {len(not_processed_soma_containing_meshes)}")

    """
    3) Overwrite the following variables:
        soma_mesh_list
        soma_containing_meshes
        soma_touching_meshes
        total_soma_list_sdf

    """

    somas_idx_to_process = soma_containing_meshes[largest_soma_touching_mesh_idx]
    soma_mesh_list = [soma_mesh_list[k] for k in somas_idx_to_process]

    soma_containing_meshes = {largest_soma_touching_mesh_idx:list(np.arange(0,len(soma_mesh_list)))}

    soma_touching_meshes = [split_meshes[largest_soma_touching_mesh_idx]]

    total_soma_list_sdf = total_soma_list_sdf[somas_idx_to_process]

    print(f"\n-----After filtering away multiple disconneted soma pieces-----")
    print(f"# of soma containing seperate meshes = {len(soma_touching_meshes)}")
    print(f"meshes with somas = {soma_containing_meshes}")
    
    
    
    
    
    #--- Phase 3:  Soma Extraction was great (but it wasn't the original soma faces), so now need to get the original soma faces and the original non-soma faces of original pieces

    """
    for each soma touching mesh get the following:
    1) original soma meshes
    2) significant mesh pieces touching these somas
    3) The soma connectivity to each of the significant mesh pieces
    -- later will just translate the 


    Process: 

    1) Final all soma faces (through soma extraction and then soma original faces function)
    2) Subtact all soma faces from original mesh
    3) Find all significant mesh pieces
    4) Backtrack significant mesh pieces to orignal mesh and find connectivity of each to all
       the available somas
    Conclusion: Will have connectivity map


    """

    soma_touching_mesh_data = dict()

    for z,(mesh_idx, soma_idxes) in enumerate(soma_containing_meshes.items()):
        soma_touching_mesh_data[z] = dict()
        print(f"\n\n----Working on soma-containing mesh piece {z}----")

        #1) Final all soma faces (through soma extraction and then soma original faces function)
        current_mesh = split_meshes[mesh_idx]

        current_soma_mesh_list = [soma_mesh_list[k] for k in soma_idxes]

        current_time = time.time()
        mesh_pieces_without_soma = sm.subtract_soma(current_soma_mesh_list,current_mesh,
                                                    significance_threshold=250,
                                                   connectivity=pre_branch_connectivity)
        print(f"Total time for Subtract Soam = {time.time() - current_time}")
        current_time = time.time()

        mesh_pieces_without_soma_stacked = tu.combine_meshes(mesh_pieces_without_soma)

        # find the original soma faces of mesh
        soma_faces = tu.original_mesh_faces_map(current_mesh,mesh_pieces_without_soma_stacked,matching=False)
        print(f"Total time for Original_mesh_faces_map for mesh_pieces without soma= {time.time() - current_time}")
        current_time = time.time()
        soma_meshes = current_mesh.submesh([soma_faces],append=True,repair=False)

        # finding the non-soma original faces
        non_soma_faces = tu.original_mesh_faces_map(current_mesh,soma_meshes,matching=False)
        non_soma_stacked_mesh = current_mesh.submesh([non_soma_faces],append=True,repair=False)

        print(f"Total time for Original_mesh_faces_map for somas= {time.time() - current_time}")
        current_time = time.time()

        #4) Backtrack significant mesh pieces to orignal mesh and find connectivity of each to all the available somas
        # get all the seperate mesh faces

        #How to seperate the mesh faces
        seperate_soma_meshes,soma_face_components = tu.split(soma_meshes,only_watertight=False,
                                                            connectivity=pre_branch_connectivity)
        #take the top largest ones depending how many were originally in the soma list
        seperate_soma_meshes = seperate_soma_meshes[:len(soma_mesh_list)]
        soma_face_components = soma_face_components[:len(soma_mesh_list)]

        soma_touching_mesh_data[z]["soma_meshes"] = seperate_soma_meshes
        
        
        
        
        # 3) Find all significant mesh pieces
        """
        Pseudocode: 
        a) Iterate through all of the somas and get the pieces that are connected
        b) Concatenate all the results into one list and order
        c) Filter away the mesh pieces that aren't touching and add to the floating pieces
        
        """
        sig_non_soma_pieces,insignificant_limbs = tu.split_significant_pieces(non_soma_stacked_mesh,significance_threshold=limb_threshold,
                                                         return_insignificant_pieces=True,
                                                                             connectivity=pre_branch_connectivity)
        
        # a) Filter these down to only those touching the somas
        all_conneted_non_soma_pieces = []
        for i,curr_soma in enumerate(seperate_soma_meshes):
            (connected_mesh_pieces,
             connected_mesh_pieces_vertices,
             connected_mesh_pieces_vertices_idx) = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_soma,
                            periphery_pieces = sig_non_soma_pieces,
                            return_vertices = True,
                            return_vertices_idx=True)
            all_conneted_non_soma_pieces.append(connected_mesh_pieces)
        
        #b) Iterate through all of the somas and get the pieces that are connected
        t_non_soma_pieces = np.concatenate(all_conneted_non_soma_pieces)
        
        #c) Filter away the mesh pieces that aren't touching and add to the floating pieces
        sig_non_soma_pieces = [s_t for hh,s_t in enumerate(sig_non_soma_pieces) if hh in t_non_soma_pieces]
        new_floating_pieces = [s_t for hh,s_t in enumerate(sig_non_soma_pieces) if hh not in t_non_soma_pieces]
        
        print(f"new_floating_pieces = {new_floating_pieces}")
        
        non_soma_touching_meshes += new_floating_pieces
        
        

        print(f"Total time for sig_non_soma_pieces= {time.time() - current_time}")
        current_time = time.time()

        soma_touching_mesh_data[z]["branch_meshes"] = sig_non_soma_pieces
        
        
        
        
        

        print(f"Total time for split= {time.time() - current_time}")
        current_time = time.time()



        soma_to_piece_connectivity = dict()
        soma_to_piece_touching_vertices = dict()
        soma_to_piece_touching_vertices_idx = dict()
        limb_root_nodes = dict()

        m_vert_graph = tu.mesh_vertex_graph(current_mesh)

        for i,curr_soma in enumerate(seperate_soma_meshes):
            (connected_mesh_pieces,
             connected_mesh_pieces_vertices,
             connected_mesh_pieces_vertices_idx) = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_soma,
                            periphery_pieces = sig_non_soma_pieces,
                            return_vertices = True,
                            return_vertices_idx=True)
            #print(f"soma {i}: connected_mesh_pieces = {connected_mesh_pieces}")
            soma_to_piece_connectivity[i] = connected_mesh_pieces

            soma_to_piece_touching_vertices[i] = dict()
            for piece_index,piece_idx in enumerate(connected_mesh_pieces):
                limb_root_nodes[piece_idx] = connected_mesh_pieces_vertices[piece_index][0]

                """ Old way of finding vertex connected components on a mesh without trimesh function
                #find the number of touching groups and save those 
                soma_touching_graph = m_vert_graph.subgraph(connected_mesh_pieces_vertices_idx[piece_index])
                soma_con_comp = [current_mesh.vertices[np.array(list(k)).astype("int")] for k in list(nx.connected_components(soma_touching_graph))]
                soma_to_piece_touching_vertices[i][piece_idx] = soma_con_comp
                """

                soma_to_piece_touching_vertices[i][piece_idx] = tu.split_vertex_list_into_connected_components(
                                                    vertex_indices_list=connected_mesh_pieces_vertices_idx[piece_index],
                                                    mesh=current_mesh, 
                                                    vertex_graph=m_vert_graph, 
                                                    return_coordinates=True
                                                   )





    #         border_debug = False
    #         if border_debug:
    #             print(f"soma_to_piece_connectivity = {soma_to_piece_connectivity}")
    #             print(f"soma_to_piece_touching_vertices = {soma_to_piece_touching_vertices}")


        print(f"Total time for mesh_pieces_connectivity= {time.time() - current_time}")

        soma_touching_mesh_data[z]["soma_to_piece_connectivity"] = soma_to_piece_connectivity

    print(f"# of insignificant_limbs = {len(insignificant_limbs)} with trimesh : {insignificant_limbs}")
    print(f"# of not_processed_soma_containing_meshes = {len(not_processed_soma_containing_meshes)} with trimesh : {not_processed_soma_containing_meshes}")
    



    # Lets have an alert if there was more than one soma disconnected meshes
    if len(soma_touching_mesh_data.keys()) > 1:
        raise Exception("More than 1 disconnected meshes that contain somas")

    current_mesh_data = soma_touching_mesh_data
    soma_containing_idx = 0

    #doing inversion of the connectivity and touching vertices
    piece_to_soma_touching_vertices = gu.flip_key_orders_for_dict(soma_to_piece_touching_vertices)
    
    
    
    
    
    
    # Phase 4: Skeletonization, Mesh Correspondence,  

    proper_time = time.time()

    #The containers that will hold the final data for the preprocessed neuron
    limb_correspondence=dict()
    limb_network_stating_info = dict()

    # ---------- Part A: skeletonization and mesh decomposition --------- #
    skeleton_time = time.time()

    for curr_limb_idx,limb_mesh_mparty in enumerate(current_mesh_data[0]["branch_meshes"]):

        #Arguments to pass to the specific function (when working with a limb)
        soma_touching_vertices_dict = piece_to_soma_touching_vertices[curr_limb_idx]

    #     if curr_limb_idx != 10:
    #         continue

        curr_limb_time = time.time()
        print(f"\n\n----- Working on Proper Limb # {curr_limb_idx} ---------")

        print(f"meshparty_segment_size = {meshparty_segment_size}")
        limb_correspondence_individual,network_starting_info = preprocess_limb(mesh=limb_mesh_mparty,
                       soma_touching_vertices_dict = soma_touching_vertices_dict,
                       return_concept_network = False, 
                       return_concept_network_starting_info=True,
                       width_threshold_MAP=500,
                       size_threshold_MAP=2000,
                       surface_reconstruction_size=1000,  

                       #arguments added from the big preprocessing step                                                            
                       distance_by_mesh_center=distance_by_mesh_center,
                       meshparty_segment_size=meshparty_segment_size,
                       meshparty_n_surface_downsampling = meshparty_n_surface_downsampling,
                                                                               
                        use_meshafterparty=use_meshafterparty,

                       )
        #Storing all of the data to be sent to 

        limb_correspondence[curr_limb_idx] = limb_correspondence_individual
        limb_network_stating_info[curr_limb_idx] = network_starting_info
        
    print(f"Total time for Skeletonization and Mesh Correspondence = {time.time() - skeleton_time}")
        
        
        
    # ---------- Part B: Stitching on floating pieces --------- #
    print("\n\n ----- Working on Stitching ----------")

    floating_stitching_time = time.time()
    
    if len(limb_correspondence) > 0:
        non_soma_touching_meshes_to_stitch = tu.check_meshes_outside_multiple_mesh_bbox(seperate_soma_meshes,non_soma_touching_meshes,
                                 return_indices=False)
        
        limb_correspondence_with_floating_pieces = attach_floating_pieces_to_limb_correspondence(
                limb_correspondence,
                floating_meshes=non_soma_touching_meshes_to_stitch,
                floating_piece_face_threshold = 600,
                max_stitch_distance=8000,
                distance_to_move_point_threshold = 4000,
                verbose = False)
    else:
        limb_correspondence_with_floating_pieces = limb_correspondence
        



    print(f"Total time for stitching floating pieces = {time.time() - floating_stitching_time}")





    # ---------- Part C: Computing Concept Networks --------- #
    concept_network_time = time.time()

    limb_concept_networks=dict()
    limb_labels=dict()

    for curr_limb_idx,limb_mesh_mparty in enumerate(current_mesh_data[0]["branch_meshes"]):
        limb_to_soma_concept_networks = calculate_limb_concept_networks(limb_correspondence_with_floating_pieces[curr_limb_idx],
                                                                        limb_network_stating_info[curr_limb_idx],
                                                                        run_concept_network_checks=True,
                                                                           )   



        limb_concept_networks[curr_limb_idx] = limb_to_soma_concept_networks
        limb_labels[curr_limb_idx]= "Unlabeled"

    print(f"Total time for Concept Networks = {time.time() - concept_network_time}")





    preprocessed_data= dict(
        soma_meshes = current_mesh_data[0]["soma_meshes"],
        soma_to_piece_connectivity = current_mesh_data[0]["soma_to_piece_connectivity"],
        soma_sdfs = total_soma_list_sdf,
        insignificant_limbs=insignificant_limbs,
        not_processed_soma_containing_meshes=not_processed_soma_containing_meshes,
        non_soma_touching_meshes=non_soma_touching_meshes,
        inside_pieces=inside_pieces,
        limb_correspondence=limb_correspondence_with_floating_pieces,
        limb_concept_networks=limb_concept_networks,
        limb_network_stating_info=limb_network_stating_info,
        limb_labels=limb_labels,
        limb_meshes=current_mesh_data[0]["branch_meshes"],
        )



    print(f"Total time for all mesh and skeletonization decomp = {time.time() - proper_time}")
    
    return preprocessed_data
    


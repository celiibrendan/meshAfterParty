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

        try:
            returned_data = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                          curr_limb_mesh,
                                         skeleton_segment_width = 1000,
                                         distance_by_mesh_center=distance_by_mesh_center)
            curr_branch_face_correspondence, width_from_skeleton = returned_data
        except:
            print(f"curr_branch_sk.shape = {curr_branch_sk.shape}")
            np.savez("saved_skeleton_branch.npz",curr_branch_sk=curr_branch_sk)
            tu.write_neuron_off(curr_limb_mesh,"curr_limb_mesh.off")
            print(f"returned_data = {returned_data}")
            raise Exception(f"The output from mesh_correspondence_adaptive_distance was nothing: curr_branch_face_correspondence={curr_branch_face_correspondence}, width_from_skeleton={width_from_skeleton}")


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
                    curr_soma_to_piece_touching_vertices=None
                    ):
    """
    Will Fix the 1-to-1 Correspondence of the mesh
    correspondence for the limbs and make sure that the
    endpoints that are designated as touching the soma then 
    make sure the mesh correspondnece reaches the soma limb border
    
    """
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
                     no_missing_labels = list(original_labels)
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
                                              curr_piece_to_soma_touching_vertices):
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
                    
                    
import neuron_utils as nru
import system_utils as su
def preprocess_limb(mesh,
                   soma_touching_vertices_dict = None,
                   distance_by_mesh_center=True, #how the distance is calculated for mesh correspondence
                    meshparty_segment_size = 100,
                   meshparty_n_surface_downsampling = 2,
                    combine_close_skeleton_nodes=True,
                    combine_close_skeleton_nodes_threshold=700,
                    filter_end_node_length=4001,
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
        
    if print_fusion_steps:
        print(f"Time for preparing soma vertices and root: {time.time() - fusion_time }")
        fusion_time = time.time()
    
    # --------------- Part 3: Meshparty skeletonization and Decomposition ------------- #
    sk_meshparty_obj = m_sk.skeletonize_mesh_largest_component(limb_mesh_mparty,
                                                            root=root_curr,
                                                              filter_mesh=False)

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
    
    

    mesh_pieces_for_MAP = []
    mesh_pieces_for_MAP_face_idx = []


    if len(mesh_large_idx) > 0: #will only continue processing if found MAP candidates
        
        # --------------- Part 5: Find mesh connectivity and group MAP branch candidates into MAP sublimbs ------------- #
        print(f"Found len(mesh_large_idx) MAP candidates: {[len(k) for k in mesh_large_idx]}")
        
        #finds the connectivity edges of all the MAP candidates
        mesh_large_connectivity = tu.mesh_list_connectivity(meshes = mesh_large_idx,
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
        segment_branches = sublimb_skeleton_branches[sublimb_idx]
        whole_sk_MP = sk.stack_skeletons(segment_branches)
        branch = mesh
        divided_submeshes = sublimb_mesh_branches_MP[sublimb_idx]
        divided_submeshes_idx = sublimb_mesh_idx_branches_MP[sublimb_idx]
        segment_widths_median = widths_MP[sublimb_idx]

        if curr_soma_to_piece_touching_vertices is None:
            print(f"Do Not Need to Fix MP Decomposition {sublimb_idx} so just continuing")

        else:
            print(f"Fixing Possible Soma Extension Branch for Sublimb {sublimb_idx}")

            #If there is some soma touching then need to see if have to fix soma extending pieces
            return_info = sk.create_soma_extending_branches(current_skeleton=whole_sk_MP,
                                      skeleton_mesh=branch,
                                      soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices,
                                      return_endpoints_must_keep=True,
                                         return_created_branch_info=True)
            new_sk,endpts,new_branch_info = return_info

            if print_fusion_steps:
                print(f"MP (because soma touching verts) create_soma_extending_branches: {time.time() - fusion_time }")
                fusion_time = time.time()

            no_soma_extension_add = True
            
            if not endpts is None:
                limb_to_endpoints_must_keep_list.append(endpts)
                limb_to_soma_touching_vertices_list.append(curr_soma_to_piece_touching_vertices)
            
            for sm_idx in new_branch_info.keys():        
                for b_vert_idx,br_info in enumerate(new_branch_info[sm_idx]):
                    if br_info is None:
                        continue
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
                        new_widths.append(np.median(curr_ray_distance[curr_ray_distance!=0]))


                    #6) Remove the original branch and mesh correspondence and replace with the multiples

                    segment_branches = np.delete(segment_branches,match_sk_branches)
                    segment_branches = np.append(segment_branches,new_skeletal_branches,axis=0)

                    divided_submeshes = np.delete(divided_submeshes,match_sk_branches)
                    divided_submeshes = np.append(divided_submeshes,new_submeshes,axis=0)

                    divided_submeshes_idx = np.delete(divided_submeshes_idx,match_sk_branches)
                    divided_submeshes_idx = np.append(divided_submeshes_idx,new_submeshes_idx,axis=0)

                    segment_widths_median = np.delete(segment_widths_median,match_sk_branches)
                    segment_widths_median = np.append(segment_widths_median,new_widths,axis=0)
                    
                    sk.check_skeleton_connected_component(sk.stack_skeletons(segment_branches))
                    print("checked segment branches after soma add on")
                    return_find = sk.find_branch_skeleton_with_specific_coordinate(segment_branches,
                                                 orig_vertex)
                    
                    
                    
                    """ ******************* END OF HOW CAN DO STITCHING ********************  """




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



        mesh_conn,mesh_conn_vertex_groups = tu.mesh_list_connectivity(meshes = sublimb_meshes_MP + sublimb_meshes_MAP,
                                            main_mesh = limb_mesh_mparty,
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
            
            
            #getting the skeletons that should be stitched
            curr_skeleton_MP = sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in limb_correspondence_MP[MP_idx].values()])
            curr_skeleton_MAP = sk.stack_skeletons([branch_v["branch_skeleton"] for branch_v in limb_correspondence_MAP[MAP_idx].values()])

            #1) Get the endpoint vertices of the MP skeleton branches (so every endpoint or high degree node)
            #(needs to be inside loop because limb correspondence will change)
            curr_MP_branch_skeletons = [k["branch_skeleton"] for k in limb_correspondence_MP[MP_idx].values()]
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
            curr_MAP_branch_skeletons = [k["branch_skeleton"] for k in limb_correspondence_MAP[MAP_idx].values()]

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


            print(f"MAP_branches_with_stitch_point = {MAP_branches_with_stitch_point}")
            print(f"MAP_stitch_point_on_end_or_branch = {MAP_stitch_point_on_end_or_branch}")


            
            
            
            
            
            
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
                
                new_MP_skeleton = sk.add_and_smooth_segment_to_branch(skeleton=sk.convert_graph_to_skeleton(MP_stitch_branch_graph),
                                                skeleton_stitch_point=keep_neighbor_coordinates,
                                                 new_stitch_point=MAP_stitch_point)
                
                
                
                
                
                #smooth over the new skeleton
                new_MP_skeleton_smooth = sk.resize_skeleton_branch(new_MP_skeleton,
                                                                  segment_width=meshparty_segment_size)
                
                curr_MP_sk.append(new_MP_skeleton_smooth)



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
            if not MAP_stitch_point_on_end_or_branch:
                if len(curr_MAP_sk) > 1:
                    raise Exception(f"There was more than one skeleton for MAP skeletons even though MAP_stitch_point_on_end_or_branch = {MAP_stitch_point_on_end_or_branch}")

                
                skeleton_to_cut = curr_MAP_sk[0]
                curr_MAP_sk = sk.cut_skeleton_at_coordinate(skeleton=skeleton_to_cut,
                                                            cut_coordinate=MAP_stitch_point)
                

            # -------------- Part 15: Gets all of the skeletons and Mesh to divide u and does mesh correspondence -------#
            # ------------- revise IDX so still references the whole limb mesh -----------#
            curr_MAP_meshes_idx = [limb_correspondence_MAP[MAP_idx][k]["branch_face_idx"] for k in MAP_branches_with_stitch_point]

            curr_MP_sk
            curr_MP_meshes_idx = [limb_correspondence_MP[MP_idx][k]["branch_face_idx"] for k in MP_branches_with_stitch_point]

            stitching_mesh_idx = np.concatenate(curr_MAP_meshes_idx + curr_MP_meshes_idx)
            stitching_mesh = limb_mesh_mparty.submesh([stitching_mesh_idx],append=True,repair=False)
            stitching_skeleton_branches = curr_MAP_sk + curr_MP_sk
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

            #3) Run mesh correspondence to get new meshes and mesh_idx and widths
            local_correspondnece_stitch = mesh_correspondence_first_pass(mesh=stitching_mesh,
                                          skeleton_branches=stitching_skeleton_branches)

            local_correspondence_stitch_revised = correspondence_1_to_1(mesh=stitching_mesh,
                                                        local_correspondence=local_correspondnece_stitch,
                                                        curr_limb_endpoints_must_keep=None,
                                                        curr_soma_to_piece_touching_vertices=None)


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
                for j,curr_MAP_idx_fixed in enumerate(MAP_branches_with_stitch_point):
                    limb_correspondence_MAP[MAP_idx][curr_MAP_idx_fixed] = local_correspondence_stitch_revised[j]

            #5) Revise the meshes,  mesh_idx, and widths of the MP pieces
            for j,curr_MP_idx_fixed in enumerate(MP_branches_with_stitch_point):
                limb_correspondence_MP[MP_idx][curr_MP_idx_fixed] = local_correspondence_stitch_revised[j+len(curr_MAP_sk)]


            print(f" Finished with {(MP_idx,MAP_idx)} \n\n\n")


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
    
    if not return_concept_network:
        if return_concept_network_starting_info: #because may want to calculate the concept networks later
            return limb_correspondence_individual,network_starting_info
        else:
            return limb_correspondence_individual
    else:
        limb_to_soma_concept_networks = calculate_limb_concept_networks(limb_correspondence_individual,
                                                                        run_concept_network_checks=run_concept_network_checks,
                                                                       **network_starting_info)
        
                    
    return limb_correspondence_individual,limb_to_soma_concept_networks





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
                                   touching_verts_list,
                                    endpoints_must_keep,
                                   run_concept_network_checks=True):
    """
    Can take a limb correspondence and the starting vertices and endpoints
    and create a list of concept networks organized by 
    [soma_idx] --> list of concept networks 
                    (because could possibly have mulitple starting points on the same soma)
    
    """
    curr_touching_verts_list = touching_verts_list
    curr_endpoints_must_keep = endpoints_must_keep
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

    if len(curr_touching_verts_list) != len(curr_endpoints_must_keep):
        raise Exception(f"curr_touching_verts_list ({curr_touching_verts_list}) not same size as curr_endpoints_must_keep ({len(curr_endpoints_must_keep)})")

    for touch_vert_dict,endpoints_keep_dict in zip(curr_touching_verts_list,curr_endpoints_must_keep):
        if not np.array_equal(list(touch_vert_dict.keys()),list(endpoints_keep_dict.keys())):
            raise Exception(f"touch_vert_dict keys ({touch_vert_dict.keys()}) don't match endpoints_keep_dict keys ({endpoints_keep_dict.keys()})")
        for soma_idx in touch_vert_dict.keys():
            soma_t_verts_list = touch_vert_dict[soma_idx]
            soma_endpt_list = endpoints_keep_dict[soma_idx]

            if soma_idx not in list(limb_to_soma_concept_networks.keys()):
                limb_to_soma_concept_networks[soma_idx] = []

            if len(soma_t_verts_list) != len(soma_endpt_list):
                raise Exception(f"soma_t_verts_list length ({len(soma_t_verts_list)}) not equal to soma_endpt_list length ({soma_endpt_list})")
            for soma_group_idx,(t_verts,endpt) in enumerate(zip(soma_t_verts_list,soma_endpt_list)):



                #1) find the branch with the endoint that must keep
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

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


    print("Another print")
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
        stitch_counter = 0
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

            curr_skeleton_MP_for_stitch = sk.stack_skeletons(curr_MAP_branch_skeletons[MAP_pieces_idx_touching_border])

            #3) Find the closest skeletal point on MAP pairing (MAP stitch)
            MAP_skeleton_coords = np.unique(curr_skeleton_MP_for_stitch.reshape(-1,3),axis=0)

            #this does not guarentee that the MAP branch associated with the MAP stitch point is touching the border group
            MAP_stitch_point = MAP_skeleton_coords[np.argmin(np.linalg.norm(MAP_skeleton_coords-av_vert,axis=1))]


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


            # ---------------- Doing the MP Part --------------------- #



            ord_keys = np.sort(list(limb_correspondence_MP[MP_idx].keys()))
            curr_MP_branch_meshes = [limb_correspondence_MP[MP_idx][k]["branch_mesh"] for k in ord_keys]



            """ old way of filtering MP pieces just to those touching the MAP, but just want the ones touching the connection group

            MAP_meshes_with_stitch_point = tu.combine_meshes([limb_correspondence_MAP[MAP_idx][k]["branch_mesh"] for k in MAP_branches_with_stitch_point])

            conn = tu.mesh_pieces_connectivity(main_mesh=limb_mesh_mparty,
                                       central_piece=MAP_meshes_with_stitch_point,
                                       periphery_pieces=curr_MP_branch_meshes)
            """
            # 11/9 Addition: New way that filters meshes by their touching of the vertex connection group
            conn = tu.filter_meshes_by_containing_coordinates(mesh_list=curr_MP_branch_meshes,
                                           nullifying_points=v_g,
                                            filter_away=False,
                                           distance_threshold=0,
                                           return_indices=True)

            print(f"conn = {conn}")


            #1) Get the endpoint vertices of the MP skeleton branches (so every endpoint or high degree node)
            #(needs to be inside loop because limb correspondence will change)
            curr_MP_branch_skeletons = [limb_correspondence_MP[MP_idx][k]["branch_skeleton"] for k in conn]
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

            #5) Revise the meshes,  mesh_idx, and widths of the MP pieces
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



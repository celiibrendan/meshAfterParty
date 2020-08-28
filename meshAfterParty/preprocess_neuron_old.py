

# ------------------------ For the preprocessing ----------------------- #

def preprocess_neuron(mesh=None,
                     mesh_file=None,
                     segment_id=None,
                     description=None,
                     sig_th_initial_split=15, #for significant splitting meshes in the intial mesh split
                     limb_threshold = 2000, #the mesh faces threshold for a mesh to be qualified as a limb (otherwise too small)
                      filter_end_node_length=5000, #used in cleaning the skeleton during skeletonizations
                      return_no_somas = False
                     ):
    
    
    whole_processing_tiempo = time.time()
    
    """
    Purpose: To process the mesh into a format that can be loaded into the neuron class
    and used for higher order processing (how to visualize is included)
    
    """
    if description is None:
        description = "no_description"
    if segment_id is None:
        #pick a random segment id
        segment_id = np.random.randint(100000000)
        print(f"picking a random 7 digit segment id: {segment_id}")
        description += "_random_id"

    
    if mesh is None:
        if current_mesh_file is None:
            raise Exception("No mesh or mesh_file file were given")
        else:
            current_neuron = trimesh.load_mesh(current_mesh_file)
    else:
        current_neuron = mesh
        
    # ************************ Phase A ********************************
    
    print("\n\n\n\n\n****** Phase A ***************\n\n\n\n\n")
    
    
    
    
    
    # --- 1) Doing the soma detection
    
    soma_mesh_list,run_time,total_soma_list_sdf = sm.extract_soma_center(segment_id,
                                             current_neuron.vertices,
                                             current_neuron.faces)
    
    # geting the soma centers
    if len(soma_mesh_list) <= 0:
        print(f"**** No Somas Found for Mesh {segment_id} so just one mesh")
        soma_mesh_list_centers = []
        if return_no_somas:
            return soma_mesh_list_centers
        raise Exception("Processing of No Somas is not yet implemented yet")
    else:
        #compute the soma centers
        print(f"Soma List = {soma_mesh_list}")

        soma_mesh_list_centers = sm.find_soma_centroids(soma_mesh_list)
        print(f"soma_mesh_list_centers = {soma_mesh_list_centers}")
    
#     sk.graph_skeleton_and_mesh(main_mesh_verts=current_neuron.vertices,
#                           main_mesh_faces=current_neuron.faces,
#                            main_mesh_color = [0.,1.,0.,0.8]
#                           )

    # ********At this point assume that there are somas (if not would just skip to the limb skeleton stuff) *******
    
    
    
    
    
    
    
    
    #--- 2) getting the soma submeshes that are connected to each soma and identifiying those that aren't (and eliminating any mesh pieces inside the soma)
    
    main_mesh_total = current_neuron
    

    #finding the mesh pieces that contain the soma
    #splitting the current neuron into distinct pieces
    split_meshes = tu.split_significant_pieces(
                                main_mesh_total,
                                significance_threshold=sig_th_initial_split,
                                print_flag=False)

    print(f"# total split meshes = {len(split_meshes)}")


    #returns the index of the split_meshes index that contains each soma    
    containing_mesh_indices = sm.find_soma_centroid_containing_meshes(soma_mesh_list_centers,
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


    split_meshes # the meshes of the original mesh
    containing_mesh_indices #the mapping of each soma centroid to the correct split mesh
    soma_containing_meshes = sm.grouping_containing_mesh_indices(containing_mesh_indices)

    soma_touching_meshes = [split_meshes[k] for k in soma_containing_meshes.keys()]


    print(f"# of non soma touching seperate meshes = {len(non_soma_touching_meshes)}")
    print(f"# of inside pieces = {len(inside_pieces)}")
    print(f"# of soma containing seperate meshes = {len(soma_touching_meshes)}")
    print(f"meshes with somas = {soma_containing_meshes}")

   
    

    
    
    
    #--- 3)  Soma Extraction was great (but it wasn't the original soma faces), so now need to get the original soma faces and the original non-soma faces of original pieces
    
#     sk.graph_skeleton_and_mesh(other_meshes=[soma_meshes])

    

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
        print("\n\n----Working on soma-containing mesh piece {z}----")

        #1) Final all soma faces (through soma extraction and then soma original faces function)
        current_mesh = split_meshes[mesh_idx]

        current_soma_mesh_list = [soma_mesh_list[k] for k in soma_idxes]

        current_time = time.time()
        mesh_pieces_without_soma = sm.subtract_soma(current_soma_mesh_list,current_mesh,
                                                    significance_threshold=250)
        print(f"Total time for Subtract Soam = {time.time() - current_time}")
        current_time = time.time()

        mesh_pieces_without_soma_stacked = tu.combine_meshes(mesh_pieces_without_soma)

        # find the original soma faces of mesh
        soma_faces = tu.original_mesh_faces_map(current_mesh,mesh_pieces_without_soma_stacked,matching=False)
        print(f"Total time for Original_mesh_faces_map for mesh_pieces without soma= {time.time() - current_time}")
        current_time = time.time()
        soma_meshes = current_mesh.submesh([soma_faces],append=True)

        # finding the non-soma original faces
        non_soma_faces = tu.original_mesh_faces_map(current_mesh,soma_meshes,matching=False)
        non_soma_stacked_mesh = current_mesh.submesh([non_soma_faces],append=True)

        print(f"Total time for Original_mesh_faces_map for somas= {time.time() - current_time}")
        current_time = time.time()

        # 3) Find all significant mesh pieces
        sig_non_soma_pieces,insignificant_limbs = tu.split_significant_pieces(non_soma_stacked_mesh,significance_threshold=limb_threshold,
                                                         return_insignificant_pieces=True)

        print(f"Total time for sig_non_soma_pieces= {time.time() - current_time}")
        current_time = time.time()

        soma_touching_mesh_data[z]["branch_meshes"] = sig_non_soma_pieces

        #4) Backtrack significant mesh pieces to orignal mesh and find connectivity of each to all the available somas
        # get all the seperate mesh faces

        #How to seperate the mesh faces
        seperate_soma_meshes,soma_face_components = tu.split(soma_meshes,only_watertight=False)
        #take the top largest ones depending how many were originally in the soma list
        seperate_soma_meshes = seperate_soma_meshes[:len(soma_mesh_list)]
        soma_face_components = soma_face_components[:len(soma_mesh_list)]

        soma_touching_mesh_data[z]["soma_meshes"] = seperate_soma_meshes

        print(f"Total time for split= {time.time() - current_time}")
        current_time = time.time()



        soma_to_piece_connectivity = dict()
        for i,curr_soma in enumerate(seperate_soma_meshes):
            connected_mesh_pieces,connected_mesh_pieces_vertices  = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_soma,
                            periphery_pieces = sig_non_soma_pieces,
                            return_vertices = True)
            #print(f"soma {i}: connected_mesh_pieces = {connected_mesh_pieces}")
            soma_to_piece_connectivity[i] = connected_mesh_pieces

        print(f"Total time for mesh_pieces_connectivity= {time.time() - current_time}")

        soma_touching_mesh_data[z]["soma_to_piece_connectivity"] = soma_to_piece_connectivity

    print(f"# of insignificant_limbs = {len(insignificant_limbs)} with trimesh : {insignificant_limbs}")
    
    
    
    # Lets have an alert if there was more than one soma disconnected meshes
    if len(soma_touching_mesh_data.keys()) > 1:
        raise Exception("More than 1 disconnected meshes that contain somas")
    
    
    # ****Soma Touching mesh Data has the branches and the connectivity (So this is where you end up skipping if you don't have somas)***
    
    
    
    
    
    
    
    
    
    
    
    
    # ---5) Working on the Actual skeleton of all of the branches

    
    global_start_time = time.time()

    for j,(soma_containing_mesh_idx,mesh_data) in enumerate(soma_touching_mesh_data.items()):
        print(f"\n-- Working on Soma Continaing Mesh {j}--")
        current_branches = mesh_data["branch_meshes"]

        #skeletonize each of the branches
        total_skeletons = []

        for z,branch in enumerate(current_branches):
            print(f"\n    -- Working on branch {z}--")
            curren_skeleton = sk.skeletonize_connected_branch(branch)
            #clean the skeleton
                # --------  Doing the cleaning ------- #
            clean_time = time.time()
            
            new_cleaned_skeleton = sk.clean_skeleton(curren_skeleton,
                                    distance_func=sk.skeletal_distance,
                              min_distance_to_junction=filter_end_node_length, #this used to be a tuple i think when moved the parameter up to function defintion
                              return_skeleton=True,
                              print_flag=False)
            print(f"    Total time for skeleton and cleaning of branch {z}: {time.time() - clean_time}")
            if len(new_cleaned_skeleton) == 0:
                raise Exception(f"Found a zero length skeleton for limb {z} of trmesh {branch}")
            total_skeletons.append(new_cleaned_skeleton)

        soma_touching_mesh_data[j]["branch_skeletons"] = total_skeletons

    print(f"Total time for skeletonization = {time.time() - global_start_time}")
    
    
    
    
    
    
    
    
    
    
    
    
    # *************** Phase B *****************
    
    print("\n\n\n\n\n****** Phase B ***************\n\n\n\n\n")
    
    current_mesh_data = soma_touching_mesh_data
    
    
    # visualizing the original neuron
#     current_neuron = trimesh.load_mesh(current_mesh_file)
#     sk.graph_skeleton_and_mesh(main_mesh_verts=current_neuron.vertices,
#                               main_mesh_faces=current_neuron.faces,
#                                main_mesh_color = [0.,1.,0.,0.8]
#                               )
    
    
    # visualizing the somas that were extracted
#     soma_meshes = tu.combine_meshes(current_mesh_data[0]["soma_meshes"])
#     sk.graph_skeleton_and_mesh(main_mesh_verts=soma_meshes.vertices,
#                               main_mesh_faces=soma_meshes.faces,
#                                main_mesh_color = [0.,1.,0.,0.8]
#                               )


    # # Visualize the extracted branches
    # # visualize all of the branches and the meshes
    # sk.graph_skeleton_and_mesh(other_meshes=list(current_mesh_data[0]["branch_meshes"]) + list(current_mesh_data[0]["soma_meshes"]),
    #                           other_meshes_colors="random",
    #                            other_skeletons = current_mesh_data[0]["branch_skeletons"],
    #                           other_skeletons_colors="random")
    
    
    
    
    
    
    
    
    #--- 1) Cleaning each limb through distance and decomposition, checking that all cleaned branches are connected components and then visualizing
    
    skelton_cleaning_threshold = 4001
    total_cleaned = []
    for j,curr_skeleton_to_clean in enumerate(current_mesh_data[0]["branch_skeletons"]):
        print(f"\n---- Working on Limb {j} ----")
        start_time = time.time()
        print(f"before cleaning limb size of skeleton = {curr_skeleton_to_clean.shape}")
        distance_cleaned_skeleton = sk.clean_skeleton(
                                                    curr_skeleton_to_clean,
                                                    distance_func=sk.skeletal_distance,
                                                    min_distance_to_junction = skelton_cleaning_threshold,
                                                    return_skeleton=True,
                                                    print_flag=False) 
        #make sure still connected componet
        distance_cleaned_skeleton_components = nx.number_connected_components(sk.convert_skeleton_to_graph(distance_cleaned_skeleton))
        if distance_cleaned_skeleton_components > 1:
            raise Exception(f"distance_cleaned_skeleton {j} was not a single component: it was actually {distance_cleaned_skeleton_components} components")

        print(f"after DISTANCE cleaning limb size of skeleton = {distance_cleaned_skeleton.shape}")
        cleaned_branch = sk.clean_skeleton_with_decompose(distance_cleaned_skeleton)

        cleaned_branch_components = nx.number_connected_components(sk.convert_skeleton_to_graph(cleaned_branch))
        if cleaned_branch_components > 1:
            raise Exception(f"cleaned_branch {j} was not a single component: it was actually {cleaned_branch_components} components")

        #do the cleanin ghtat removes loops from branches
        print(f"After DECOMPOSITION cleaning limb size of skeleton = {cleaned_branch.shape}")
        print(f"Total time = {time.time() - start_time}")
        total_cleaned.append(cleaned_branch)

    current_mesh_data[0]["branch_skeletons_cleaned"] = total_cleaned
    
    
    
    # checking all cleaned branches are connected components

    for k,cl_sk in enumerate(current_mesh_data[0]["branch_skeletons"]): 
        n_components = nx.number_connected_components(sk.convert_skeleton_to_graph(cl_sk)) 
        if n_components > 1:
            raise Exception(f"Original limb {k} was not a single component: it was actually {n_components} components")

    for k,cl_sk in enumerate(current_mesh_data[0]["branch_skeletons_cleaned"]): 
        n_components = nx.number_connected_components(sk.convert_skeleton_to_graph(cl_sk)) 
        if n_components > 1:
            raise Exception(f"Cleaned limb {k} was not a single component: it was actually {n_components} components")
            
    
    # # visualize all of the branches and the meshes
    # sk.graph_skeleton_and_mesh(other_meshes=list(current_mesh_data[0]["branch_meshes"]) + list(current_mesh_data[0]["soma_meshes"]),
    #                           other_meshes_colors="random",
    #                            other_skeletons = current_mesh_data[0]["branch_skeletons_cleaned"],
    #                           other_skeletons_colors="random",
    #                           mesh_alpha=0.15,
    #                           html_path=f"{segment_id}_limb_skeleton.html")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # --- 2) Decomposing of limbs into branches and finding mesh correspondence (using adaptive mesh correspondence followed by a water fill for conflict and empty faces), checking that it went well with no empty meshes and all connected component graph (even when downsampling the skeleton) when constructed from branches, plus visualization at end
    
    

    start_time = time.time()

    limb_correspondence = dict()
    soma_containing_idx= 0

    for soma_containing_idx in current_mesh_data.keys():
        for limb_idx,curr_limb_mesh in enumerate(current_mesh_data[soma_containing_idx]["branch_meshes"]):
            print(f"Working on limb #{limb_idx}")
            limb_correspondence[limb_idx] = dict()
            curr_limb_sk = current_mesh_data[soma_containing_idx]["branch_skeletons_cleaned"][limb_idx]
            curr_limb_branches_sk_uneven = sk.decompose_skeleton_to_branches(curr_limb_sk) #the line that is decomposing to branches

            for j,curr_branch_sk in tqdm(enumerate(curr_limb_branches_sk_uneven)):
                limb_correspondence[limb_idx][j] = dict()


                curr_branch_face_correspondence, width_from_skeleton = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                              curr_limb_mesh,
                                             skeleton_segment_width = 1000)



                if len(curr_branch_face_correspondence) > 0:
                    curr_submesh = curr_limb_mesh.submesh([list(curr_branch_face_correspondence)],append=True)
                else:
                    curr_submesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))

                limb_correspondence[limb_idx][j]["branch_skeleton"] = curr_branch_sk
                limb_correspondence[limb_idx][j]["correspondence_mesh"] = curr_submesh
                limb_correspondence[limb_idx][j]["correspondence_face_idx"] = curr_branch_face_correspondence
                limb_correspondence[limb_idx][j]["width_from_skeleton"] = width_from_skeleton


    print(f"Total time for decomposition = {time.time() - start_time}")
    
    
    #couple of checks on how the decomposition went:  for each limb
    #1) if shapes of skeletons cleaned and divided match
    #2) if skeletons are only one component
    #3) if you downsample the skeletons then still only one component
    #4) if any empty meshes
    
    empty_submeshes = []

    for soma_containing_idx in current_mesh_data.keys():
        for limb_idx,curr_limb_mesh in enumerate(current_mesh_data[soma_containing_idx]["branch_meshes"]):
            print(f"\n---- checking limb {limb_idx}---")
            print(f"Limb {limb_idx} decomposed into {len(limb_correspondence[limb_idx])} branches")

            #get all of the skeletons and make sure that they from a connected component
            divided_branches = [limb_correspondence[limb_idx][k]["branch_skeleton"] for k in limb_correspondence[limb_idx]]
            divided_skeleton_graph = sk.convert_skeleton_to_graph(
                                            sk.stack_skeletons(divided_branches))

            divided_skeleton_graph_recovered = sk.convert_graph_to_skeleton(divided_skeleton_graph)

            cleaned_limb_skeleton = current_mesh_data[0]['branch_skeletons_cleaned'][limb_idx]
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


            for j in limb_correspondence[limb_idx].keys():
                if len(limb_correspondence[limb_idx][j]["correspondence_mesh"].faces) == 0:
                    empty_submeshes.append(dict(limb_idx=limb_idx,branch_idx = j))

    print(f"Empty submeshes = {empty_submeshes}")

    if len(empty_submeshes) > 0:
        raise Exception(f"Found empyt meshes after branch mesh correspondence: {empty_submeshes}")
        
        

    # import matplotlib_utils as mu

    # sk.graph_skeleton_and_mesh(other_meshes=total_branch_meshes,
    #                           other_meshes_colors="random",
    #                            other_skeletons=total_branch_skeletons,
    #                            other_skeletons_colors="random"
    #                           )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ---3) Finishing off the face correspondence so get 1-to-1 correspondence of mesh face to skeletal piece
    
    #--- this is the function that will clean up a limb piece so have 1-1 correspondence

    #things to prep for visualizing the axons
#     total_widths = []
#     total_branch_skeletons = []
#     total_branch_meshes = []

    soma_containing_idx = 0

    for limb_idx in limb_correspondence.keys():
        mesh_start_time = time.time()
        #clear out the mesh correspondence if already in limb_correspondecne
        for k in limb_correspondence[limb_idx].keys():
            if "branch_mesh" in limb_correspondence[limb_idx][k]:
                del limb_correspondence[limb_idx][k]["branch_mesh"]
            if "branch_face_idx" in limb_correspondence[limb_idx][k]:
                del limb_correspondence[limb_idx][k]["branch_face_idx"]
        #geting the current limb mesh
        print(f"\n\nWorking on limb_correspondence for #{limb_idx}")
        no_missing_labels = list(limb_correspondence[limb_idx].keys()) #counts the number of divided branches which should be the total number of labels
        curr_limb_mesh = current_mesh_data[soma_containing_idx]["branch_meshes"][limb_idx]

        #set up the face dictionary
        face_lookup = dict([(j,[]) for j in range(0,len(curr_limb_mesh.faces))])

        for j,branch_piece in limb_correspondence[limb_idx].items():
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


        # -- splitting the mesh pieces into individual pieces
        divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(curr_limb_mesh,face_coloring_copy)

        #-- check that all the split mesh pieces are one component --#

        #save off the new data as branch mesh
        for k in limb_correspondence[limb_idx].keys():
            limb_correspondence[limb_idx][k]["branch_mesh"] = divided_submeshes[k]
            limb_correspondence[limb_idx][k]["branch_face_idx"] = divided_submeshes_idx[k]
            
            #clean the limb correspondence that we do not need
            del limb_correspondence[limb_idx][k]["correspondence_mesh"]
            del limb_correspondence[limb_idx][k]["correspondence_face_idx"]
#             total_widths.append(limb_correspondence[limb_idx][k]["width_from_skeleton"])
#             total_branch_skeletons.append(limb_correspondence[limb_idx][k]["branch_skeleton"])
#             total_branch_meshes.append(limb_correspondence[limb_idx][k]["branch_mesh"])

        print(f"Total time for limb mesh processing = {time.time() - mesh_start_time}")
    
    
    
    
    
    # Visualizing the results of getting the mesh to skeletal segment correspondence completely 1-to-1
    
#     from matplotlib import pyplot as plt
#     fig,ax = plt.subplots(1,1)
#     bins = plt.hist(np.array(total_widths),bins=100)
#     ax.set_xlabel("Width measurement of mesh branch (nm)")
#     ax.set_ylabel("frequency")
#     ax.set_title("Width measurement of mesh branch frequency")
#     plt.show()
    
#     sk.graph_skeleton_and_mesh(other_meshes=total_branch_meshes,
#                           other_meshes_colors="random",
#                           other_skeletons=total_branch_skeletons,
#                           other_skeletons_colors="random",
#                           #html_path="two_soma_mesh_skeleton_decomp.html"
#                           )

    
#     sk.graph_skeleton_and_mesh(other_meshes=[total_branch_meshes[47]],
#                               other_meshes_colors="random",
#                               other_skeletons=[total_branch_skeletons[47]],
#                               other_skeletons_colors="random",
#                               html_path="two_soma_mesh_skeleton_decomp.html")
    
    
    
    
    
    
    
    
    
    
    
    # ********************   Phase C ***************************************
    # PART 3: LAST PART OF ANALYSIS WHERE MAKES CONCEPT GRAPHS
    
    
    print("\n\n\n\n\n****** Phase C ***************\n\n\n\n\n")
    
    
    
    
    
    # ---1) Making concept graphs:

    limb_concept_networks,limb_labels = generate_limb_concept_networks_from_global_connectivity(
        limb_correspondence = limb_correspondence,
        #limb_idx_to_branch_meshes_dict = limb_idx_to_branch_meshes_dict,
        #limb_idx_to_branch_skeletons_dict = limb_idx_to_branch_skeletons_dict,
        
        soma_meshes=current_mesh_data[0]["soma_meshes"],
        soma_idx_connectivity=current_mesh_data[0]["soma_to_piece_connectivity"],
        #soma_idx_to_mesh_dict = soma_idx_to_mesh_dict,
        #soma_idx_connectivity = soma_idx_connectivity,
        
        current_neuron=current_neuron,
        return_limb_labels=True
        )

#     #Before go and get concept maps:
#     print("Sizes of dictionaries sent")
#     for curr_limb in limb_idx_to_branch_skeletons_dict.keys():
#         print((len(limb_idx_to_branch_skeletons_dict[curr_limb]),len(limb_idx_to_branch_meshes_dict[curr_limb])))


#     print("\n\n Sizes of concept maps gotten back")
#     for curr_idx in limb_concept_networks.keys():
#         for soma_idx,concept_network in limb_concept_networks[curr_idx].items():
#             print(len(np.unique(list(concept_network.nodes()))))
            
    
    
    
    
    
    
    
    
    

    
    # ---2) Packaging the data into a dictionary that can be sent to the Neuron class to create the object
    
    #Preparing the data structure to save or use for Neuron class construction

    
    
    preprocessed_data = dict(
                            soma_meshes = current_mesh_data[0]["soma_meshes"],
                            soma_to_piece_connectivity = current_mesh_data[0]["soma_to_piece_connectivity"],
                            soma_sdfs = total_soma_list_sdf,
                            insignificant_limbs=insignificant_limbs,
                            non_soma_touching_meshes=non_soma_touching_meshes,
                            inside_pieces=inside_pieces,
                            limb_correspondence=limb_correspondence,
                            limb_concept_networks=limb_concept_networks,
                            limb_labels=limb_labels,
                            limb_meshes=current_mesh_data[0]["branch_meshes"],
                            )

    
    
    print(f"\n\n\n Total processing time = {time.time() - whole_processing_tiempo}")
    
    print(f"returning preprocessed_data = {preprocessed_data}")
    return preprocessed_data
    
    
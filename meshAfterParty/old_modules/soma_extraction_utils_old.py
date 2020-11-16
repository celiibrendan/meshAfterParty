def extract_soma_center(segment_id,
                        current_mesh_verts,
                        current_mesh_faces,
                       outer_decimation_ratio= 0.25,
                        large_mesh_threshold = 60000,
                        large_mesh_threshold_inner = 40000,
                        soma_width_threshold = 0.32,
                        soma_size_threshold = 15000,
                       inner_decimation_ratio = 0.25,
                       volume_mulitplier=8,
                       #side_length_ratio_threshold=3,
                        side_length_ratio_threshold=6,
                       soma_size_threshold_max=192000, #this puts at 12000 once decimated, another possible is 256000
                       delete_files=True,
                        backtrack_soma_mesh_to_original=True, #should either be None or 
                        boundary_vertices_threshold=None,#700 the previous threshold used
                        poisson_backtrack_distance_threshold=None,#1500 the previous threshold used
                        close_holes=False,
                        
                        #------- 11/12 Additions --------------- #
                        
                        #these arguments are for removing inside pieces
                        remove_inside_pieces = True,
                        size_threshold_to_remove=1000, #size accounting for the decimation
                        
                        
                        pymeshfix_clean=False,
                        check_holes_before_pymeshfix=False,
                        second_poisson=False,
                        segmentation_at_end=True,
                        last_size_threshold = 1300,
                        
                       ):    
    """
    Will extract the soma meshes (possible multiple) from
    a single mesh

    """
    
    global_start_time = time.time()

    #Adjusting the thresholds based on the decimations
    large_mesh_threshold = large_mesh_threshold*outer_decimation_ratio
    large_mesh_threshold_inner = large_mesh_threshold_inner*outer_decimation_ratio
    soma_size_threshold = soma_size_threshold*outer_decimation_ratio
    soma_size_threshold_max = soma_size_threshold_max*outer_decimation_ratio

    #adjusting for inner decimation
    soma_size_threshold = soma_size_threshold*inner_decimation_ratio
    soma_size_threshold_max = soma_size_threshold_max*inner_decimation_ratio
    print(f"Current Arguments Using (adjusted for decimation):\n large_mesh_threshold= {large_mesh_threshold}"
                 f" \nlarge_mesh_threshold_inner = {large_mesh_threshold_inner}"
                  f" \nsoma_size_threshold = {soma_size_threshold}"
                 f" \nsoma_size_threshold_max = {soma_size_threshold_max}"
                 f"\nouter_decimation_ratio = {outer_decimation_ratio}"
                 f"\ninner_decimation_ratio = {inner_decimation_ratio}")


    # ------------------------------


    temp_folder = f"./{segment_id}"
    temp_object = Path(temp_folder)
    #make the temp folder if it doesn't exist
    temp_object.mkdir(parents=True,exist_ok=True)

    #making the decimation and poisson objections
    Dec_outer = Decimator(outer_decimation_ratio,temp_folder,overwrite=True)
    Dec_inner = Decimator(inner_decimation_ratio,temp_folder,overwrite=True)
    Poisson_obj = Poisson(temp_folder,overwrite=True)

    #Step 1: Decimate the Mesh and then split into the seperate pieces
    new_mesh,output_obj = Dec_outer(vertices=current_mesh_verts,
             faces=current_mesh_faces,
             segment_id=segment_id,
             return_mesh=True,
             delete_temp_files=False)

    # if remove_inside_pieces:
    #     print("removing mesh interior after decimation")
    #     new_mesh = tu.remove_mesh_interior(new_mesh,size_threshold_to_remove=size_threshold_to_remove)

    #preforming the splits of the decimated mesh

    mesh_splits = new_mesh.split(only_watertight=False)

    #get the largest mesh
    mesh_lengths = np.array([len(split.faces) for split in mesh_splits])


    total_mesh_split_lengths = [len(k.faces) for k in mesh_splits]
    ordered_mesh_splits = mesh_splits[np.flip(np.argsort(total_mesh_split_lengths))]
    list_of_largest_mesh = [k for k in ordered_mesh_splits if len(k.faces) > large_mesh_threshold]

    print(f"Total found significant pieces before Poisson = {list_of_largest_mesh}")

    #if no significant pieces were found then will use smaller threshold
    if len(list_of_largest_mesh)<=0:
        print(f"Using smaller large_mesh_threshold because no significant pieces found with {large_mesh_threshold}")
        list_of_largest_mesh = [k for k in ordered_mesh_splits if len(k.faces) > large_mesh_threshold/2]

    total_soma_list = []
    total_classifier_list = []
    total_poisson_list = []
    total_soma_list_sdf = []



    #start iterating through where go through all pieces before the poisson reconstruction
    no_somas_found_in_big_loop = 0
    for i,largest_mesh in enumerate(list_of_largest_mesh):
        print(f"----- working on large mesh #{i}: {largest_mesh}")

        if remove_inside_pieces:
            print("remove_inside_pieces requested ")
            largest_mesh = tu.remove_mesh_interior(largest_mesh,size_threshold_to_remove=size_threshold_to_remove)


        if pymeshfix_clean:
            print("Requested pymeshfix_clean")
            """
            Don't have to check if manifold anymore actually just have to plug the holes
            """
            hole_groups = tu.find_border_face_groups(largest_mesh)
            if len(hole_groups) > 0:
                largest_mesh_filled_holes = tu.fill_holes(largest_mesh,max_hole_size = 10000)
            else:
                largest_mesh_filled_holes = largest_mesh

            if check_holes_before_pymeshfix:
                hole_groups = tu.find_border_face_groups(largest_mesh_filled_holes)
            else:
                print("Not checking if there are still existing holes before pymeshfix")
                hole_groups = []

            if len(hole_groups) > 0:
                #segmentation_at_end = False
                print(f"*** COULD NOT FILL HOLES WITH MAX SIZE OF {np.max([len(k) for k in hole_groups])} so not applying pymeshfix and segmentation_at_end = {segmentation_at_end}")

    #                 tu.write_neuron_off(largest_mesh_filled_holes,"largest_mesh_filled_holes")
    #                 raise Exception()
            else:
                print("Applying pymeshfix_clean because no more holes")
                largest_mesh = tu.pymeshfix_clean(largest_mesh_filled_holes,verbose=True)

        if second_poisson:
            print("Applying second poisson run")
            current_neuron_poisson = tu.poisson_surface_reconstruction(largest_mesh)
            largest_mesh = tu.split_significant_pieces(current_neuron_poisson)[0]

        somas_found_in_big_loop = False

        largest_file_name = str(output_obj.stem) + "_largest_piece.off"
        pre_largest_mesh_path = temp_object / Path(str(output_obj.stem) + "_largest_piece.off")
        pre_largest_mesh_path = pre_largest_mesh_path.absolute()
        print(f"pre_largest_mesh_path = {pre_largest_mesh_path}")
        # ******* This ERRORED AND CALLED OUR NERUON NONE: 77697401493989254 *********
        new_mesh_inner,poisson_file_obj = Poisson_obj(vertices=largest_mesh.vertices,
                   faces=largest_mesh.faces,
                   return_mesh=True,
                   mesh_filename=largest_file_name,
                   delete_temp_files=False)


        #splitting the Poisson into the largest pieces and ordering them
        mesh_splits_inner = new_mesh_inner.split(only_watertight=False)
        total_mesh_split_lengths_inner = [len(k.faces) for k in mesh_splits_inner]
        ordered_mesh_splits_inner = mesh_splits_inner[np.flip(np.argsort(total_mesh_split_lengths_inner))]

        list_of_largest_mesh_inner = [k for k in ordered_mesh_splits_inner if len(k.faces) > large_mesh_threshold_inner]
        print(f"Total found significant pieces AFTER Poisson = {list_of_largest_mesh_inner}")

        n_failed_inner_soma_loops = 0
        for j, largest_mesh_inner in enumerate(list_of_largest_mesh_inner):
            print(f"----- working on mesh after poisson #{j}: {largest_mesh_inner}")

            largest_mesh_path_inner = str(poisson_file_obj.stem) + "_largest_inner.off"

            #Decimate the inner poisson piece
            largest_mesh_path_inner_decimated,output_obj_inner = Dec_inner(
                                vertices=largest_mesh_inner.vertices,
                                 faces=largest_mesh_inner.faces,
                                mesh_filename=largest_mesh_path_inner,
                                 return_mesh=True,
                                 delete_temp_files=False)

            print(f"done exporting decimated mesh: {largest_mesh_path_inner}")

            faces = np.array(largest_mesh_path_inner_decimated.faces)
            verts = np.array(largest_mesh_path_inner_decimated.vertices)

            segment_id_new = int(str(segment_id) + f"{i}{j}")
            #print(f"Before the classifier the pymeshfix_clean = {pymeshfix_clean}")
            verts_labels, faces_labels, soma_value,classifier = wcda.extract_branches_whole_neuron(
                                    import_Off_Flag=False,
                                    segment_id=segment_id_new,
                                    vertices=verts,
                                     triangles=faces,
                                    pymeshfix_Flag=False,
                                     import_CGAL_Flag=False,
                                     return_Only_Labels=True,
                                     clusters=3,
                                     smoothness=0.2,
                                    soma_only=True,
                                    return_classifier = True
                                    )
            print(f"soma_sdf_value = {soma_value}")

            total_classifier_list.append(classifier)
            #total_poisson_list.append(largest_mesh_path_inner_decimated)

            # Save all of the portions that resemble a soma
            median_values = np.array([v["median"] for k,v in classifier.sdf_final_dict.items()])
            segmentation = np.array([k for k,v in classifier.sdf_final_dict.items()])

            #order the compartments by greatest to smallest
            sorted_medians = np.flip(np.argsort(median_values))
            print(f"segmentation[sorted_medians],median_values[sorted_medians] = {(segmentation[sorted_medians],median_values[sorted_medians])}")
            print(f"Sizes = {[classifier.sdf_final_dict[g]['n_faces'] for g in segmentation[sorted_medians]]}")
            print(f"soma_size_threshold = {soma_size_threshold}")
            print(f"soma_size_threshold_max={soma_size_threshold_max}")

            valid_soma_segments_width = [g for g,h in zip(segmentation[sorted_medians],median_values[sorted_medians]) if ((h > soma_width_threshold)
                                                                and (classifier.sdf_final_dict[g]["n_faces"] > soma_size_threshold)
                                                                and (classifier.sdf_final_dict[g]["n_faces"] < soma_size_threshold_max))]
            valid_soma_segments_sdf = [h for g,h in zip(segmentation[sorted_medians],median_values[sorted_medians]) if ((h > soma_width_threshold)
                                                                and (classifier.sdf_final_dict[g]["n_faces"] > soma_size_threshold)
                                                                and (classifier.sdf_final_dict[g]["n_faces"] < soma_size_threshold_max))]

            print("valid_soma_segments_width")
            to_add_list = []
            to_add_list_sdf = []
            if len(valid_soma_segments_width) > 0:
                print(f"      ------ Found {len(valid_soma_segments_width)} viable somas: {valid_soma_segments_width}")
                somas_found_in_big_loop = True
                #get the meshes only if signfiicant length
                labels_list = classifier.labels_list

                for v,sdf in zip(valid_soma_segments_width,valid_soma_segments_sdf):
                    submesh_face_list = np.where(classifier.labels_list == v)[0]
                    soma_mesh = largest_mesh_path_inner_decimated.submesh([submesh_face_list],append=True)

                    # ---------- No longer doing the extra checks in here --------- #


                    curr_side_len_check = side_length_check(soma_mesh,side_length_ratio_threshold)
                    curr_volume_check = soma_volume_check(soma_mesh,volume_mulitplier)
                    if curr_side_len_check and curr_volume_check:
                        to_add_list.append(soma_mesh)
                        to_add_list_sdf.append(sdf)

                    else:
                        print(f"--->This soma mesh was not added because it did not pass the sphere validation:\n "
                             f"soma_mesh = {soma_mesh}, curr_side_len_check = {curr_side_len_check}, curr_volume_check = {curr_volume_check}")
                        continue

                n_failed_inner_soma_loops = 0

            else:
                n_failed_inner_soma_loops += 1

            total_soma_list_sdf += to_add_list_sdf
            total_soma_list += to_add_list

            # --------------- KEEP TRACK IF FAILED TO FIND SOMA (IF TOO MANY FAILS THEN BREAK)
            if n_failed_inner_soma_loops >= 2:
                print("breaking inner loop because 2 soma fails in a row")
                break


        # --------------- KEEP TRACK IF FAILED TO FIND SOMA (IF TOO MANY FAILS THEN BREAK)
        if somas_found_in_big_loop == False:
            no_somas_found_in_big_loop += 1
            if no_somas_found_in_big_loop >= 2:
                print("breaking because 2 fails in a row in big loop")
                break

        else:
            no_somas_found_in_big_loop = 0





    """ IF THERE ARE MULTIPLE SOMAS THAT ARE WITHIN A CERTAIN DISTANCE OF EACH OTHER THEN JUST COMBINE THEM INTO ONE"""
    pairings = []
    for y,soma_1 in enumerate(total_soma_list):
        for z,soma_2 in enumerate(total_soma_list):
            if y<z:
                mesh_tree = KDTree(soma_1.vertices)
                distances,closest_node = mesh_tree.query(soma_2.vertices)

                if np.min(distances) < 4000:
                    pairings.append([y,z])


    #creating the combined meshes from the list
    total_soma_list_revised = []
    total_soma_list_revised_sdf = []
    if len(pairings) > 0:
        """
        Pseudocode: 
        Use a network function to find components

        """


        import networkx as nx
        new_graph = nx.Graph()
        new_graph.add_edges_from(pairings)
        grouped_somas = list(nx.connected_components(new_graph))

        somas_being_combined = []
        print(f"There were soma pairings: Connected components in = {grouped_somas} ")
        for comp in grouped_somas:
            comp = list(comp)
            somas_being_combined += list(comp)
            current_mesh = total_soma_list[comp[0]]
            for i in range(1,len(comp)):
                current_mesh += total_soma_list[comp[i]] #just combining the actual meshes

            total_soma_list_revised.append(current_mesh)
            #where can average all of the sdf values
            total_soma_list_revised_sdf.append(np.min(np.array(total_soma_list_sdf)[comp]))

        #add those that weren't combined to total_soma_list_revised
        leftover_somas = [total_soma_list[k] for k in range(0,len(total_soma_list)) if k not in somas_being_combined]
        leftover_somas_sdfs = [total_soma_list_sdf[k] for k in range(0,len(total_soma_list)) if k not in somas_being_combined]
        if len(leftover_somas) > 0:
            total_soma_list_revised += leftover_somas
            total_soma_list_revised_sdf += leftover_somas_sdfs

        print(f"Final total_soma_list_revised = {total_soma_list_revised}")
        print(f"Final total_soma_list_revised_sdf = {total_soma_list_revised_sdf}")


    if len(total_soma_list_revised) == 0:
        total_soma_list_revised = total_soma_list
        total_soma_list_revised_sdf = total_soma_list_sdf

    run_time = time.time() - global_start_time

    print(f"\n\n\n Total time for run = {time.time() - global_start_time}")
    print(f"Before Filtering the number of somas found = {len(total_soma_list_revised)}")

    #     import system_utils as su
    #     su.compressed_pickle(total_soma_list_revised,"total_soma_list_revised")
    #     su.compressed_pickle(new_mesh,"original_mesh")

    #need to erase all of the temporary files ******
    #import shutil
    #shutil.rmtree(directory)

    """
    Running the extra tests that depend on
    - border vertices
    - how well the poisson matches the backtracked soma to the real mesh
    - other size checks

    """
    filtered_soma_list = []
    filtered_soma_list_sdf = []
    for soma_mesh,curr_soma_sdf in zip(total_soma_list_revised,total_soma_list_revised_sdf):
        if backtrack_soma_mesh_to_original:
            print("Performing Soma Mesh Backtracking to original mesh")
            soma_mesh_poisson = deepcopy(soma_mesh)
            try:
                #print("About to find original mesh")
                soma_mesh = original_mesh_soma(
                                                mesh = new_mesh,
                                                soma_meshes=[soma_mesh_poisson],
                                                sig_th_initial_split=15)[0]
            except:
                import traceback
                traceback.print_exc()
                print("--->This soma mesh was not added because Was not able to backtrack soma to mesh")
                continue
            else:
                if soma_mesh is None:
                    print("--->This soma mesh was not added because Was not able to backtrack soma to mesh")
                    continue




            print(f"poisson_backtrack_distance_threshold = {poisson_backtrack_distance_threshold}")
            #do the check that tests if there is a max distance between poisson and backtrack:
            if not poisson_backtrack_distance_threshold is None and poisson_backtrack_distance_threshold > 0:

                #soma_mesh.export("soma_mesh.off")
                if close_holes: 
                    print("Using the close holes feature")
                    fill_hole_obj = meshlab.FillHoles(max_hole_size=2000,
                                                     self_itersect_faces=False)

                    soma_mesh_filled_holes,output_subprocess_obj = fill_hole_obj(   
                                                        vertices=soma_mesh.vertices,
                                                         faces=soma_mesh.faces,
                                                         return_mesh=True,
                                                         delete_temp_files=True,
                                                        )
                else:
                    soma_mesh_filled_holes = soma_mesh


                #soma_mesh_filled_holes.export("soma_mesh_filled_holes.off")



                print("APPLYING poisson_backtrack_distance_threshold CHECKS")
                mesh_1 = soma_mesh_filled_holes
                mesh_2 = soma_mesh_poisson

                poisson_max_distance = tu.max_distance_betwee_mesh_vertices(mesh_1,mesh_2,
                                                                  verbose=True)
                print(f"poisson_max_distance = {poisson_max_distance}")
                if poisson_max_distance > poisson_backtrack_distance_threshold:
                    print(f"--->This soma mesh was not added because it did not pass the poisson_backtrack_distance check:\n"
                      f" poisson_max_distance = {poisson_max_distance}")
                    continue


        #do the boundary check:
        if not boundary_vertices_threshold is None:
            print("USING boundary_vertices_threshold CHECK")
            soma_boundary_groups_sizes = np.array([len(k) for k in tu.find_border_face_groups(soma_mesh)])
            print(f"soma_boundary_groups_sizes = {soma_boundary_groups_sizes}")
            large_boundary_groups = soma_boundary_groups_sizes[soma_boundary_groups_sizes>boundary_vertices_threshold]
            print(f"large_boundary_groups = {large_boundary_groups} with boundary_vertices_threshold = {boundary_vertices_threshold}")
            if len(large_boundary_groups)>0:
                print(f"--->This soma mesh was not added because it did not pass the boundary vertices validation:\n"
                      f" large_boundary_groups = {large_boundary_groups}")
                continue

        curr_side_len_check = side_length_check(soma_mesh,side_length_ratio_threshold)
        curr_volume_check = soma_volume_check(soma_mesh,volume_mulitplier)
        if (not curr_side_len_check) or (not curr_volume_check):
            print(f"--->This soma mesh was not added because it did not pass the sphere validation:\n "
                 f"soma_mesh = {soma_mesh}, curr_side_len_check = {curr_side_len_check}, curr_volume_check = {curr_volume_check}")
            continue

        #tu.write_neuron_off(soma_mesh_poisson,"original_poisson.off")
        #If made it through all the checks then add to final list
        filtered_soma_list.append(soma_mesh)
        filtered_soma_list_sdf.append(curr_soma_sdf)


    """
    Need to delete all files in the temp folder *****
    """

    if delete_files:
        #now erase all of the files used
        from shutil import rmtree

        #remove the directory with the meshes
        rmtree(str(temp_object.absolute()))

        #removing the temporary files
        temp_folder = Path("./temp")
        temp_files = [x for x in temp_folder.glob('**/*')]
        seg_temp_files = [x for x in temp_files if str(segment_id) in str(x)]

        for f in seg_temp_files:
            f.unlink()

    # ----------- 11 /11 Addition that does a last step segmentation of the soma --------- #
    #return total_soma_list, run_time
    #return total_soma_list_revised,run_time,total_soma_list_revised_sdf

    filtered_soma_list_copy = copy.deepcopy(filtered_soma_list)
    filtered_soma_list_sdf_copy = copy.deepcopy(filtered_soma_list_sdf)

    if len(filtered_soma_list) > 0:
        filtered_soma_list_revised = []
        filtered_soma_list_sdf_revised = []
        for f_soma,f_soma_sdf in zip(filtered_soma_list,filtered_soma_list_sdf):
            if segmentation_at_end:

                if remove_inside_pieces:
                    print("removing mesh interior before segmentation")
                    f_soma = tu.remove_mesh_interior(f_soma,size_threshold_to_remove=size_threshold_to_remove)

                print("Doing the soma segmentation filter at end")

                meshes_split,meshes_split_sdf = tu.mesh_segmentation(
                    mesh = f_soma,
                    smoothness=0.5
                )
                print(f"meshes_split = {meshes_split}")
                print(f"meshes_split_sdf = {meshes_split_sdf}")

                #applying the soma width and the soma size threshold
                above_width_threshold_mask = meshes_split_sdf>=soma_width_threshold
                meshes_split_sizes = np.array([len(k.faces) for k in meshes_split])
                above_size_threshold_mask = meshes_split_sizes >= last_size_threshold

                above_width_threshold_idx = np.where(above_width_threshold_mask & above_size_threshold_mask)[0]
                if len(above_width_threshold_idx) == 0:
                    print(f"No split meshes were above the width threshold ({soma_width_threshold}) and size threshold ({last_size_threshold}) so continuing")
                    continue

                meshes_split = np.array(meshes_split)
                meshes_split_sdf = np.array(meshes_split_sdf)

                meshes_split_filtered = meshes_split[above_width_threshold_idx]
                meshes_split_sdf_filtered = meshes_split_sdf[above_width_threshold_idx]

                soma_width_threshold
                #way to choose the index of the top candidate
                top_candidate = 0
                filtered_soma_list_revised.append(meshes_split_filtered[top_candidate])
                filtered_soma_list_sdf_revised.append(meshes_split_sdf_filtered[top_candidate])


            else:
                print("Skipping the segmentatio filter at end")
                if len(f_soma.faces) >= last_size_threshold and f_soma_sdf >= soma_width_threshold:
                    filtered_soma_list_revised.append(f_soma)
                    filtered_soma_list_sdf_revised.append(f_soma_sdf)

        filtered_soma_list = np.array(filtered_soma_list_revised)
        filtered_soma_list_sdf = np.array(filtered_soma_list_sdf_revised)
                    
            
    return filtered_soma_list,run_time,filtered_soma_list_sdf
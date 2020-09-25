import numpy as np
import trimesh_utils as tu
import networkx_utils as xu

from skeleton_utils import *

from tqdm_utils import tqdm


def get_skeletal_distance_no_skipping(main_mesh,edges,
                                 buffer=0.01,
                                bbox_ratio=1.2,
                               distance_threshold=3000,
                                      distance_by_mesh_center=False,
                                print_flag=False,
                                edge_loop_print=True):
    """
    Purpose: To return the histogram of distances along a mesh subtraction process
    so that we could evenutally find an adaptive distance threshold
    
    
    """
    #print(f"distance_by_mesh_center = {distance_by_mesh_center}")
    main_mesh_bbox_restricted = main_mesh
    faces_bbox_inclusion = np.arange(0,len(main_mesh.faces))

    
    start_time = time.time()
    face_subtract_indices = []
    
    
    total_distances = []
    total_distances_std = []
    
    for i,ex_edge in tqdm(enumerate(edges)):
        #print("\n------ New loop ------")
        #print(ex_edge)
        
        # ----------- creating edge and checking distance ----- #
        loop_start = time.time()
        
        edge_line = ex_edge[1] - ex_edge[0]
        sum_threshold = 0.001
        if np.sum(np.abs(edge_line)) < sum_threshold:
            if edge_loop_print:
                print(f"edge number {i}, {ex_edge}: has sum less than {sum_threshold} so skipping")
            continue
        
        cob_edge = change_basis_matrix(edge_line)
        
        #get the limits of the example edge itself that should be cutoff
        edge_trans = (cob_edge@ex_edge.T)
        #slice_range = np.sort((cob_edge@ex_edge.T)[2,:])
        slice_range = np.sort(edge_trans[2,:])

        # adding the buffer to the slice range
        slice_range_buffer = slice_range + np.array([-buffer,buffer])

        # generate face midpoints from the triangles
        #face_midpoints = np.mean(main_mesh_bbox_restricted.vertices[main_mesh_bbox_restricted.faces],axis=1) # Old way
        face_midpoints = main_mesh_bbox_restricted.triangles_center
        
        
        #get the face midpoints that fall within the slice (by lookig at the z component)
        fac_midpoints_trans = cob_edge@face_midpoints.T
        
        slice_mask_pre_distance = ((fac_midpoints_trans[2,:]>slice_range_buffer[0]) & 
                      (fac_midpoints_trans[2,:]<slice_range_buffer[1]))

        edge_midpoint = np.mean(edge_trans.T,axis=0)
        distance_check = np.linalg.norm((fac_midpoints_trans.T)[:,:2] - edge_midpoint[:2],axis=1) < distance_threshold
        
        slice_mask = slice_mask_pre_distance & distance_check
        
        face_list = np.arange(0,len(main_mesh_bbox_restricted.faces))[slice_mask]

        #get the submesh of valid faces within the slice
        if len(face_list)>0:
            main_mesh_sub = main_mesh_bbox_restricted.submesh([face_list],append=True)
        else:
            main_mesh_sub = []

        if type(main_mesh_sub) != type(trimesh.Trimesh()):
            #print(f"total_distances = {total_distances}")
            total_distances.append(0)
            total_distances_std.append(0)
            if edge_loop_print:
                print(f"THERE WERE NO FACES THAT FIT THE DISTANCE ({distance_threshold}) and Z transform requirements")
                print("So just skipping this edge")
            continue


        #get all disconnected mesh pieces of the submesh and the face indices for lookup later
        sub_components,sub_components_face_indexes = tu.split(main_mesh_sub,only_watertight=False)
       
        
        
        if type(sub_components) != type(np.array([])) and type(sub_components) != list:
            #print(f"meshes = {sub_components}, with type = {type(sub_components)}")
            if type(sub_components) == type(trimesh.Trimesh()) :
                sub_components = [sub_components]
            else:
                raise Exception("The sub_components were not an array, list or trimesh")
        

        #getting the indices of the submeshes whose bounding box contain the edge 
        contains_points_results = np.array([s_comp.bounding_box.contains(ex_edge.reshape(-1,3)) for s_comp in sub_components])
        
        containing_indices = (np.arange(0,len(sub_components)))[np.sum(contains_points_results,axis=1) >= len(ex_edge)]
        try:
            if len(containing_indices) != 1: 
                if edge_loop_print:
                    print(f"--> Not exactly one containing mesh: {containing_indices}")
                if len(containing_indices) > 1:
                    sub_components_inner = sub_components[containing_indices]
                    sub_components_face_indexes_inner = sub_components_face_indexes[containing_indices]
                else:
                    sub_components_inner = sub_components
                    sub_components_face_indexes_inner = sub_components_face_indexes

                #get the center of the edge
                edge_center = np.mean(ex_edge,axis=0)
                #print(f"edge_center = {edge_center}")

                #find the distance between eacch bbox center and the edge center
                bbox_centers = [np.mean(k.bounds,axis=0) for k in sub_components_inner]
                #print(f"bbox_centers = {bbox_centers}")
                closest_bbox = np.argmin([np.linalg.norm(edge_center-b_center) for b_center in bbox_centers])
                #print(f"bbox_distance = {closest_bbox}")
                edge_skeleton_faces = faces_bbox_inclusion[face_list[sub_components_face_indexes_inner[closest_bbox]]]


            else:# when only one viable submesh piece and just using that sole index
                edge_skeleton_faces = faces_bbox_inclusion[face_list[sub_components_face_indexes[containing_indices[0]]]]
        except:
            print(f"sub_components = {sub_components}")
            print(f"containing_indices = {containing_indices}")
            print(f"sub_components_face_indexes (from the split) = {sub_components_face_indexes}")
            raise Exception("Error occured")


        if len(edge_skeleton_faces) < 0:
            print(f"****** Warning the edge index {i}: had no faces in the edge_skeleton_faces*******")
        face_subtract_indices.append(edge_skeleton_faces)
        
        
        #---- calculating the relevant distances ---- #
        
        face_midpoints = (main_mesh_bbox_restricted.triangles_center)[edge_skeleton_faces]
        #print(f"edge_skeleton_faces.shape = {edge_skeleton_faces.shape}")
#         print(f"cob_edge = {cob_edge}")
#         print(f"face_midpoints = {face_midpoints.shape}")
#         print(f"sub_components = {sub_components}")
#         print(f"containing_indices = {containing_indices}")
#         print(f"sub_components_face_indexes (from the split) = {sub_components_face_indexes}")
        #Exception("failed on fac_midpoints_trans")
        
        fac_midpoints_trans = cob_edge@face_midpoints.T
        
        # Will use the mesh center when calculating the distance
        if distance_by_mesh_center: 
            faces_submesh = main_mesh_bbox_restricted.submesh([edge_skeleton_faces],append=True)
            faces_submesh_center = tu.mesh_center_weighted_face_midpoints(faces_submesh)
            faces_submesh_center = faces_submesh_center.reshape(3,1)
            #print(f"cob_edge.shape = {cob_edge.shape}, faces_submesh_center.shape={faces_submesh_center.shape}")
            edge_midpoint = cob_edge@faces_submesh_center
            edge_midpoint = edge_midpoint.reshape(-1)
        
        #print(f"fac_midpoints_trans.shape = {fac_midpoints_trans.shape}")
        mesh_slice_distances = np.linalg.norm((fac_midpoints_trans.T)[:,:2] - edge_midpoint[:2],axis=1)
        #print(f"mesh_slice_distances.shape = {mesh_slice_distances.shape}")
        
        total_distances.append(np.mean(mesh_slice_distances))
        total_distances_std.append(np.std(mesh_slice_distances))
    
    
    #end of the edge loops
    
    if len(face_subtract_indices)>1:
        all_removed_faces = np.concatenate(face_subtract_indices)

        unique_removed_faces = np.array(list(set(all_removed_faces)))
        
        if len(unique_removed_faces) < 1:
            print(f"face_subtract_indices = {face_subtract_indices}")
            print(f"unique_removed_faces = {unique_removed_faces}")
            print(f"Distance of skeleton = {sk.calculate_skeleton_distance(edges)}")
            raise Exception(f"unique_removed_faces = {unique_removed_faces}")
            
            

        #faces_to_keep = set(np.arange(0,len(main_mesh.faces))).difference(unique_removed_faces)
        new_submesh = main_mesh.submesh([unique_removed_faces],only_watertight=False,append=True)
        
        split_meshes,components_faces = tu.split(new_submesh,return_components=True)
        
         #don't just want to take the biggest mesh: but want to take the one that has the most of the skeleton
        #piece corresponding to it
        
        """
        Pseudocode: 
        1) turn all of the mesh edge_skeleton_faces into meshes, have the main mesh be the whole mesh and 
        have each of the mesh pieces be a central piece
        2) Call the mesh_pieces_connectivity function and see how many of the periphery pieces are touching each of the submehses
        3) Pick the mesh that has the most 
        
        """
        
        if len(split_meshes) > 1: 
            branch_touching_number = []
            branch_correspondence_meshes = [main_mesh.submesh([k],only_watertight=False,append=True) for k in face_subtract_indices]
            for curr_central_piece in split_meshes:
                touching_periphery_pieces =tu. mesh_pieces_connectivity(
                                            main_mesh = new_submesh,
                                            central_piece = curr_central_piece,
                                            periphery_pieces = branch_correspondence_meshes,
                                            return_vertices=False)
                branch_touching_number.append(len(touching_periphery_pieces))
                if print_flag:
                    print(f"branch_touching_number = {branch_touching_number}")
            
            #find the argmax
            most_branch_containing_piece = np.argmax(branch_touching_number)
            if print_flag:
                print(f"most_branch_containing_piece = {most_branch_containing_piece}")
            
            new_submesh = split_meshes[most_branch_containing_piece]
            unique_removed_faces = unique_removed_faces[components_faces[most_branch_containing_piece]]
            
        elif len(split_meshes) == 1: 
            new_submesh = split_meshes[0]
            unique_removed_faces = unique_removed_faces[components_faces[0]]
        else:
            raise Exception("The split meshes in the mesh correspondence was 0 length")
        
        #need to further restric the unique_removed_faces to those of most significant piece
    
    else:
        unique_removed_faces = np.array([])
        new_submesh = trimesh.Trimesh()
 
    
    
    return total_distances,total_distances_std,new_submesh,np.array(unique_removed_faces)


def get_skeletal_distance(main_mesh,edges,
                                 buffer=0.01,
                                bbox_ratio=1.2,
                               distance_threshold=3000,
                               distance_by_mesh_center=True,
                                print_flag=False,
                                edge_loop_print=True):
    """
    Purpose: To return the histogram of distances along a mesh subtraction process
    so that we could evenutally find an adaptive distance threshold
    
    
    """
    print(f"INSIDE GET SKELETAL DISTANCE distance_by_mesh_center = {distance_by_mesh_center}")
    debug=False

    
    main_mesh_bbox_restricted = main_mesh
    faces_bbox_inclusion = np.arange(0,len(main_mesh.faces))

    
    start_time = time.time()
    face_subtract_indices = []
    
    
    total_distances = []
    total_distances_std = []
    for i,ex_edge in tqdm(enumerate(edges)):
        #print("\n------ New loop ------")
        #print(ex_edge)
        
        # ----------- creating edge and checking distance ----- #
        loop_start = time.time()
        
        edge_line = ex_edge[1] - ex_edge[0]
        sum_threshold = 0.001
        if np.sum(np.abs(edge_line)) < sum_threshold:
            if edge_loop_print:
                print(f"edge number {i}, {ex_edge}: has sum less than {sum_threshold} so skipping")
            continue
        
        cob_edge = change_basis_matrix(edge_line)
        
        #get the limits of the example edge itself that should be cutoff
        edge_trans = (cob_edge@ex_edge.T)
        #slice_range = np.sort((cob_edge@ex_edge.T)[2,:])
        slice_range = np.sort(edge_trans[2,:])

        # adding the buffer to the slice range
        slice_range_buffer = slice_range + np.array([-buffer,buffer])

        # generate face midpoints from the triangles
        #face_midpoints = np.mean(main_mesh_bbox_restricted.vertices[main_mesh_bbox_restricted.faces],axis=1) # Old way
        face_midpoints = main_mesh_bbox_restricted.triangles_center
        
        
        #get the face midpoints that fall within the slice (by lookig at the z component)
        fac_midpoints_trans = cob_edge@face_midpoints.T
        
        slice_mask_pre_distance = ((fac_midpoints_trans[2,:]>slice_range_buffer[0]) & 
                      (fac_midpoints_trans[2,:]<slice_range_buffer[1]))

        edge_midpoint = np.mean(edge_trans.T,axis=0)
        distance_check = np.linalg.norm((fac_midpoints_trans.T)[:,:2] - edge_midpoint[:2],axis=1) < distance_threshold
        
        slice_mask = slice_mask_pre_distance & distance_check
        
        face_list = np.arange(0,len(main_mesh_bbox_restricted.faces))[slice_mask]

        #get the submesh of valid faces within the slice
        if len(face_list)>0:
            main_mesh_sub = main_mesh_bbox_restricted.submesh([face_list],append=True)
        else:
            main_mesh_sub = []

        if type(main_mesh_sub) != type(trimesh.Trimesh()):
            if edge_loop_print:
                print(f"THERE WERE NO FACES THAT FIT THE DISTANCE ({distance_threshold}) and Z transform requirements")
                print("So just skipping this edge")
            continue

        if debug:
            print(f"face_list = {face_list}")    
            
        #get all disconnected mesh pieces of the submesh and the face indices for lookup later
        sub_components,sub_components_face_indexes = tu.split(main_mesh_sub,only_watertight=False)
       
        if debug:
            print(f"sub_components = {sub_components}")
        
        if type(sub_components) != type(np.array([])) and type(sub_components) != list:
            #print(f"meshes = {sub_components}, with type = {type(sub_components)}")
            if type(sub_components) == type(trimesh.Trimesh()) :
                sub_components = [sub_components]
            else:
                raise Exception("The sub_components were not an array, list or trimesh")
        
        if debug:
            print(f"sub_components = {sub_components}")
            print(f"sub_components_face_indexes = {sub_components_face_indexes}")
        

        #getting the indices of the submeshes whose bounding box contain the edge 
        contains_points_results = np.array([s_comp.bounding_box.contains(ex_edge.reshape(-1,3)) for s_comp in sub_components])
        
        containing_indices = (np.arange(0,len(sub_components)))[np.sum(contains_points_results,axis=1) >= len(ex_edge)]
        
        if debug:
            print(f"containing_indices = {containing_indices}")
            
        try:
            if len(containing_indices) != 1: 
                if edge_loop_print:
                    print(f"--> Not exactly one containing mesh: {containing_indices}")
                if len(containing_indices) > 1:
                    sub_components_inner = sub_components[containing_indices]
                    sub_components_face_indexes_inner = sub_components_face_indexes[containing_indices]
                else:
                    sub_components_inner = sub_components
                    sub_components_face_indexes_inner = sub_components_face_indexes

                #get the center of the edge
                edge_center = np.mean(ex_edge,axis=0)
                #print(f"edge_center = {edge_center}")

                #find the distance between eacch bbox center and the edge center
                bbox_centers = [np.mean(k.bounds,axis=0) for k in sub_components_inner]
                #print(f"bbox_centers = {bbox_centers}")
                closest_bbox = np.argmin([np.linalg.norm(edge_center-b_center) for b_center in bbox_centers])
                #print(f"bbox_distance = {closest_bbox}")
                edge_skeleton_faces = faces_bbox_inclusion[face_list[sub_components_face_indexes_inner[closest_bbox]]]


            else:# when only one viable submesh piece and just using that sole index
                edge_skeleton_faces = faces_bbox_inclusion[face_list[sub_components_face_indexes[containing_indices[0]]]]
                if debug:
                    print(f"edge_skeleton_faces = {edge_skeleton_faces}")
        except:
            print(f"sub_components = {sub_components}")
            print(f"containing_indices = {containing_indices}")
            print(f"sub_components_face_indexes (from the split) = {sub_components_face_indexes}")
            raise Exception("Error occured")


        if len(edge_skeleton_faces) < 0:
            print(f"****** Warning the edge index {i}: had no faces in the edge_skeleton_faces*******")
        face_subtract_indices.append(edge_skeleton_faces)
        
        
        #---- calculating the relevant distances ---- #
        
        face_midpoints = (main_mesh_bbox_restricted.triangles_center)[edge_skeleton_faces]
        #print(f"edge_skeleton_faces.shape = {edge_skeleton_faces.shape}")
#         print(f"cob_edge = {cob_edge}")
#         print(f"face_midpoints = {face_midpoints.shape}")
#         print(f"sub_components = {sub_components}")
#         print(f"containing_indices = {containing_indices}")
#         print(f"sub_components_face_indexes (from the split) = {sub_components_face_indexes}")
        #Exception("failed on fac_midpoints_trans")
        
        fac_midpoints_trans = cob_edge@face_midpoints.T
        
        if distance_by_mesh_center: 
            faces_submesh = main_mesh_bbox_restricted.submesh([edge_skeleton_faces],append=True)
            faces_submesh_center = tu.mesh_center_weighted_face_midpoints(faces_submesh)
            faces_submesh_center = faces_submesh_center.reshape(3,1)
            #print(f"cob_edge.shape = {cob_edge.shape}, faces_submesh_center.shape={faces_submesh_center.shape}")
            edge_midpoint = cob_edge@faces_submesh_center
            edge_midpoint = edge_midpoint.reshape(-1)
        
            
        #print(f"fac_midpoints_trans.shape = {fac_midpoints_trans.shape}")
        mesh_slice_distances = np.linalg.norm((fac_midpoints_trans.T)[:,:2] - edge_midpoint[:2],axis=1)
        #print(f"mesh_slice_distances.shape = {mesh_slice_distances.shape}")
        
        total_distances.append(np.mean(mesh_slice_distances))
        total_distances_std.append(np.std(mesh_slice_distances))
    
    debug = False
    
    if debug:
        print(f"face_subtract_indices = {face_subtract_indices}")
    
    if len(face_subtract_indices)>0:
        all_removed_faces = np.concatenate(face_subtract_indices)

        unique_removed_faces = np.array(list(set(all_removed_faces)))
        
        if len(unique_removed_faces) < 1:
            print(f"Distance of skeleton = {sk.calculate_skeleton_distance(edges)}")
            raise Exception(f"unique_removed_faces = {unique_removed_faces}")
            
        if debug:
            print(f"unique_removed_faces = {unique_removed_faces}")  

        #faces_to_keep = set(np.arange(0,len(main_mesh.faces))).difference(unique_removed_faces)
        new_submesh = main_mesh.submesh([unique_removed_faces],only_watertight=False,append=True)
        if debug:
            print("--------------------- Starting new trial --------------------")
            print(f"main_mesh.faces.shape = {main_mesh.faces.shape}")
            print(f"new_submesh.faces.shape = {new_submesh.faces.shape}")
            print(f"unique_removed_faces.shape = {unique_removed_faces.shape}")
            print(f"max(unique_removed_faces) = {max(unique_removed_faces)}")
        
        split_meshes,components_faces = tu.split(new_submesh,return_components=True)
        
        if debug:
            current_random_number = np.random.randint(10,10000)
            print()
            print("After the split has been called")
            print(f"split_meshes = {split_meshes}")
            print(f"components_faces = {[np.array(k).shape for k in components_faces]}")
            main_mesh.export(f"main_mesh_{len(main_mesh.faces)}.off")
            new_submesh.export(f"new_submesh_{len(new_submesh.faces)}.off")
            np.savez(f"unique_removed_faces_{unique_removed_faces.shape[0]}.npz",unique_removed_faces=unique_removed_faces)
            
        
         #don't just want to take the biggest mesh: but want to take the one that has the most of the skeleton
        #piece corresponding to it
        
        """
        Pseudocode: 
        1) turn all of the mesh edge_skeleton_faces into meshes, have the main mesh be the whole mesh and 
        have each of the mesh pieces be a central piece
        2) Call the mesh_pieces_connectivity function and see how many of the periphery pieces are touching each of the submehses
        3) Pick the mesh that has the most 
        
        """
        
        if len(split_meshes) > 1: 
            branch_touching_number = []
            #getting the mesh correspondence for each skeleton segment
            branch_correspondence_meshes = [main_mesh.submesh([k],only_watertight=False,append=True) for k in face_subtract_indices]
            #Out of all the submesh splits, see how many of the segment mesh correspondence it is touching
            for curr_central_piece in split_meshes:
                touching_periphery_pieces =tu.mesh_pieces_connectivity(
                                            main_mesh = new_submesh,
                                            central_piece = curr_central_piece,
                                            periphery_pieces = branch_correspondence_meshes,
                                            return_vertices=False)
                branch_touching_number.append(len(touching_periphery_pieces))
                if print_flag:
                    print(f"branch_touching_number = {branch_touching_number}")
            
            #CONCLUSION: find the submesh split piece that is touching the most skeleton segment mesh correspondences (winning mesh)
            most_branch_containing_piece = np.argmax(branch_touching_number)
            if print_flag:
                print(f"most_branch_containing_piece = {most_branch_containing_piece}")
            
            #Make this the 
            
            
            new_submesh = split_meshes[most_branch_containing_piece]
            unique_removed_faces = unique_removed_faces[components_faces[most_branch_containing_piece]]
            
            if debug:
                print()
                print(f"main_mesh.faces.shape = {main_mesh.faces.shape}")
                print("reassigning new_submesh to one of many sub pieces")
                print(f"new_submesh.faces.shape = {new_submesh.faces.shape}")
                print(f"unique_removed_faces.shape = {unique_removed_faces.shape}")
                print(f"max(unique_removed_faces) = {max(unique_removed_faces)}")
            
        elif len(split_meshes) == 1: 
            new_submesh = split_meshes[0]
            unique_removed_faces = unique_removed_faces[components_faces[0]]
            
            if debug:
                print()
                print(f"main_mesh.faces.shape = {main_mesh.faces.shape}")
                print("Assigning submesh to the only submesh")
                print(f"new_submesh.faces.shape = {new_submesh.faces.shape}")
                print(f"unique_removed_faces.shape = {unique_removed_faces.shape}")
                print(f"max(unique_removed_faces) = {max(unique_removed_faces)}")
        else:
            raise Exception("The split meshes in the mesh correspondence was 0 length")
        
        #need to further restric the unique_removed_faces to those of most significant piece
    
    else:
        unique_removed_faces = np.array([])
        new_submesh = trimesh.Trimesh()
        
    if debug:
        print(f"new_submesh = {new_submesh}")
    
    return total_distances,total_distances_std,new_submesh,np.array(unique_removed_faces)


def mesh_correspondence_adaptive_distance(curr_branch_skeleton,
                                          curr_branch_mesh,
                                         skeleton_segment_width = 1000,
                                          distance_by_mesh_center = True,
                                         print_flag=False):
    
    debug=False
    #making the skeletons resized to 1000 widths and then can use outlier finding
    
    new_skeleton  = sk.resize_skeleton_branch(curr_branch_skeleton,segment_width = skeleton_segment_width)
    if debug:
        print(f"new_skeleton = {new_skeleton}")
    if print_flag:
        print(f"new_skeleton.shape = {new_skeleton.shape}")

    (segment_skeletal_mean_distances,
     segment_skeletal_std_distances,
     mesh_correspondence,
     mesh_correspondence_indices) = get_skeletal_distance(
                        main_mesh = curr_branch_mesh,
                        edges = new_skeleton,
                        buffer=100,
                        bbox_ratio=1.2,
                        distance_threshold=3000,
                        distance_by_mesh_center=distance_by_mesh_center,
                        print_flag=False
    )
    debug=False
    if debug:
        print("\n After first skeletal distance call")
        print(f"segment_skeletal_mean_distances = {segment_skeletal_mean_distances}")
        print(f"segment_skeletal_std_distances = {segment_skeletal_std_distances}")
        print(f"mesh_correspondence = {mesh_correspondence}")
        print(f"mesh_correspondence_indices = {mesh_correspondence_indices}")
        
    if len(mesh_correspondence_indices)== 0:
        if print_flag:
            print("empty mesh_correspondence_indices returned so returning an empty array")
        return []
    
    #now use the new submesh to calculate the new threshold
    # -- Step where I compute the new threshold and and then rerun it -- #
    if len(segment_skeletal_mean_distances) > 4:
        filtered_measurements = np.array(segment_skeletal_mean_distances[1:-1])
        filtered_measurements_std = np.array(segment_skeletal_std_distances[1:-1])
    else:
        filtered_measurements = np.array(segment_skeletal_mean_distances)
        filtered_measurements_std = np.array(segment_skeletal_std_distances)



    #filter out the other outliers: do anything higher than 150% of median should be discounted
    median_value = np.median(filtered_measurements)
    outlier_mask = filtered_measurements <= median_value*1.5
    filtered_measurements = filtered_measurements[outlier_mask]
    filtered_measurements_std = filtered_measurements_std[outlier_mask]

    if print_flag:
        print(f"filtered_measurements = {filtered_measurements}")

    # try the mesh subtraction again 
    buffer = 100

    total_threshold = np.max(filtered_measurements) + 2*np.max(filtered_measurements_std)
    
    
    
    if debug:
        print(f"new_threshold = {total_threshold}")
        
    (segment_skeletal_mean_distances_2,
     filtered_measurements_std,
     mesh_correspondence_2,
     mesh_correspondence_indices_2) = get_skeletal_distance(
                        main_mesh = mesh_correspondence,
                        edges = new_skeleton,
                        buffer=100,
                        bbox_ratio=1.2,
                        distance_threshold=total_threshold,
                        distance_by_mesh_center=distance_by_mesh_center,
                        print_flag=False
    )
    
    if debug:
        print("\n After 2nd skeletal distance call")
        print(f"segment_skeletal_mean_distances_2 = {segment_skeletal_mean_distances_2}")
        print(f"filtered_measurements_std = {filtered_measurements_std}")
        print(f"mesh_correspondence_2 = {mesh_correspondence_2}")
        print(f"mesh_correspondence_indices_2 = {mesh_correspondence_indices_2}")
    
    if len(mesh_correspondence_indices_2) == 0:
        print("empty mesh_correspondence_indices_2 returned so returning original mesh correspondence")
        return mesh_correspondence_indices,np.mean(segment_skeletal_mean_distances) + 2*np.max(segment_skeletal_std_distances)
        
        
    """
    segment_skeletal_mean_distances
    segment_skeletal_std_distances,
    mesh_correspondence,
    mesh_correspondence_indices

    segment_skeletal_mean_distances_2,
    filtered_measurements_std,
    mesh_correspondence_2,
    mesh_correspondence_indices_2
    
    """
    debug = False
    if debug: 
        print(f"\n\n segment_skeletal_mean_distances.shape = {np.array(segment_skeletal_mean_distances).shape}\n"
              f"segment_skeletal_std_distances.shape = {np.array(segment_skeletal_std_distances).shape}\n"
              f"mesh_correspondence.faces.shape = {mesh_correspondence.faces.shape}\n"
              f"max(mesh_correspondence_indices)= {np.max(mesh_correspondence_indices)}\n"
              f"mesh_correspondence_indices.shape = {np.array(mesh_correspondence_indices).shape}\n"

              f"segment_skeletal_mean_distances_2.shape = {np.array(segment_skeletal_mean_distances_2).shape}\n"
              f"filtered_measurements_std.shape = {np.array(filtered_measurements_std).shape}\n"
              f"mesh_correspondence_2.faces.shape = {mesh_correspondence_2.faces.shape}\n"
              f"max(mesh_correspondence_indices_2)= {np.max(mesh_correspondence_indices_2)}\n"
              f"mesh_correspondence_indices_2.shape = {np.array(mesh_correspondence_indices_2).shape}\n")

        
    try:
        mesh_correspondence_indices[mesh_correspondence_indices_2]
    except:
        print(f"mesh_correspondence_indices = {mesh_correspondence_indices}")
        print(f"mesh_correspondence_indices_2 = {mesh_correspondence_indices_2}")
        
    #want to show the changes in mesh
#     sk.graph_skeleton_and_mesh(other_meshes = [curr_branch_mesh.submesh([mesh_correspondence_indices],append=True)])
#     sk.graph_skeleton_and_mesh(other_meshes = [curr_branch_mesh.submesh([mesh_correspondence_indices[mesh_correspondence_indices_2]],append=True)])
        
    # PROBLEM NOT PASSING BACK A CONNECTED COMPONENT
        
    return mesh_correspondence_indices[mesh_correspondence_indices_2], total_threshold


# -------- for the mesh correspondence that creates an exact 1-to1 correspondence of mesh face to skeleton branch------- #
def filter_face_coloring_to_connected_components(curr_limb_mesh,face_coloring):
    """
    Purpose: To eliminate all but the largest connected component
    of a label on the mesh face coloring 
    
    Reason for need: when cancelling out conflict pieces
    it can split up a mesh into disconnected components and we 
    want only one component so that it can later be expanded during 
    the waterfilling process
    
    """
    leftover_labels = np.unique(face_coloring)
    for curr_label in leftover_labels:
        label_indices = np.where(face_coloring==curr_label)[0]
        curr_submesh = curr_limb_mesh.submesh([label_indices],append=True)
        split_meshes,split_components = tu.split(curr_submesh,return_components=True)
        to_keep_indices = label_indices[split_components[0]]
        to_clear_indices = np.setdiff1d(label_indices, to_keep_indices)
        face_coloring[to_clear_indices] = -1
    return face_coloring

def waterfill_labeling(
                total_mesh_correspondence,
                 submesh_indices,
                 total_mesh=None,
                total_mesh_graph=None,
                 propagation_type="random",
                max_iterations = 1000,
                max_submesh_threshold = 1000
                ):
    """
    Pseudocode:
    1) check if the submesh you are propagating labels to is too large
    2) for each unmarked face get the neighbors of all of the faces, and for all these neighbors get all the labels
    3) if the neighbors label is not empty. depending on the type of progation type then pick the winning label
    a. random: just randomly choose from list
    b. .... not yet implemented
    4) revise the faces that are still empty and repeat process until all faces are empty (have a max iterations number)
    """
    
    if not total_mesh_graph:
        #finding the face adjacency:
        total_mesh_graph = nx.from_edgelist(total_mesh.face_adjacency)
    
    
    
    if len(submesh_indices)> max_submesh_threshold:
        raise Exception(f"The len of the submesh ({len(submesh_indices)}) exceeds the maximum threshold of {max_submesh_threshold} ")
    
    #check that these are unmarked
    curr_unmarked_faces = [k for k in submesh_indices if total_mesh_correspondence[k] == -1] 
    
    
    if len(curr_unmarked_faces)<len(submesh_indices):
        raise Exception(f"{len(submesh_indices)-len(curr_unmarked_faces)} submesh faces were already labeled before waterfill_labeling started")
    
    for i in range(max_iterations):
        #s2) for each unmarked face get the neighbors of all of the faces, and for all these neighbors get all the labels
        unmarked_faces_neighbors = [xu.get_neighbors(total_mesh_graph,j) for j in curr_unmarked_faces] #will be list of lists
        #print(f"unmarked_faces_neighbors = {unmarked_faces_neighbors}")
        unmarked_face_neighbor_labels = [np.array([total_mesh_correspondence[curr_neighbor] for curr_neighbor in z]) for z in unmarked_faces_neighbors]
        #print(f"unmarked_face_neighbor_labels = {unmarked_face_neighbor_labels}")
        
        if len(unmarked_face_neighbor_labels) == 0:
            print(f"curr_unmarked_faces = {curr_unmarked_faces}")
            print(f"i = {i}")
            print(f"unmarked_faces_neighbors = {unmarked_faces_neighbors}")
            print(f"unmarked_face_neighbor_labels = {unmarked_face_neighbor_labels}")
            
        #check if there is only one type of label and if so then autofil
        total_labels = list(np.unique(np.concatenate(unmarked_face_neighbor_labels)))
        
        if -1 in total_labels:
            total_labels.remove(-1)
        
        if len(total_labels) == 0:
            raise Exception("total labels does not have any marked neighbors")
        elif len(total_labels) == 1:
            #print("All surrounding labels are the same so autofilling the remainder of unlabeled labels")
            for gg in curr_unmarked_faces:
                total_mesh_correspondence[gg] = total_labels[0]
            break
        else:
            #if there are still one or more labels surrounding our unlabeled region
            for curr_face,curr_neighbors in zip(curr_unmarked_faces,unmarked_face_neighbor_labels):
                curr_neighbors = curr_neighbors[curr_neighbors != -1]
                if len(curr_neighbors) > 0:
                    if propagation_type == "random":
                        total_mesh_correspondence[curr_face] = np.random.choice(curr_neighbors)
                    else:
                        raise Exception("Not implemented propagation_type")
        
        # now replace the new curr_unmarked faces
        curr_unmarked_faces = [k for k in submesh_indices if total_mesh_correspondence[k] == -1] #old dict way
        
        
        if len(curr_unmarked_faces) == 0:
            #print(f"breaking out of loop because zero unmarked faces left after {i} iterations")
            break
        
    
    #check that no more unmarked faces or error
    end_unmarked_faces = [k for k in submesh_indices if total_mesh_correspondence[k] == -1]
    
    if len(end_unmarked_faces) > 0:
        raise Exception(f"After {i+1} iterations (with max_iterations = {max_iterations} there were still {len(end_unmarked_faces)} faces")
        
    
    return total_mesh_correspondence



def resolve_empty_conflicting_face_labels(
                     curr_limb_mesh,
                     face_lookup,
                     no_missing_labels = [],
                    max_submesh_threshold=50000,
                    max_color_filling_iterations=10):
    
    """
    Input: 
    - full mesh
    - current face coloring of mesh (could be incomplete) corresponding to skeletal pieces
    (but doesn't need the skeletal pieces to do it's jobs, those are just represented in the labels)

    Output: 
    - better face coloring which has labels that:
        a. cover entire mesh
        b. the labels exist as only 1 connected component on the mesh



    Pseudocode of what doing:
    - clearing out the branch_mesh correspondence stored in limb_correspondence[limb_idx][k]["branch_mesh"]
    - gets a list of how many subdivided branches there were (becuase this should be the number of labels) and the mesh of whole limb
    - Builds a face to skeleeton branch correspondence bassed on the current  branch_piece["correspondence_face_idx"] that already exists
        This may have overlaps or faces mapped to zero branches that we need to resolve
    - computes the percentage of empty and conflicting faces 
    - makes sure that at least one face that corresponds to each branch piece (and throws error if so)

    #Doing the resolution of the empty and conflicting faces:
    - clears out all conflicting faces and leaves them just like the empty ones
    - uses the filter_face_coloring_to_connected_components which only keeps the largest connected component of a label 
        (because the zeroing out of conflicting labels could have eliminated or split up some of the labels)
    - if  a face was totally eliminated then add it back to the face coloring
    (only does this once so a face could still be missing if one face totally overwrites another face)

    **At this point: there is at one-to-one correspondence of mesh face to skeletal piece label OR empty label (-1)

    # Using the waterfilling algorithm: designed at fixing the correspondence to empty label (-1) 
    - get a submesh of the original mesh but only for those empty faces and divide into disconnecteed mesh pieces
    - run through waterfilling algorithm to color each empty piece
    - check that there are no more empty faces
    - gets the one connected mesh component that corresponds to that label (get both the actual mesh and the mesh indexes)

    #the output of all of the algorithm: 
    - save the result back in  limb_correspondence[limb_idx][k]["branch_mesh"] so it is accurately updated


    """
    
    if len(no_missing_labels) == 0:
        no_missing_labels = list(set(list(itertools.chain.from_iterable(list(face_lookup.values())))))
         
    #get all of the faces that don't have any faces corresponding
    empty_indices = np.array([k for k,v in face_lookup.items() if len(v) == 0])

    #get all of the faces that don't have any faces corresponding
    conflict_indices = np.array([k for k,v in face_lookup.items() if len(v) >= 2])

    print(f"empty_indices % = {len(empty_indices)/len(face_lookup.keys())}\n conflict_indices % = {len(conflict_indices)/len(face_lookup.keys())}")



    #doing the face coloring (OLD WAY)
    ##face_coloring = np.array([-1 if len(v) != 1 else v[0] for v in face_lookup.values()])
    
    #doing the face coloring (new way if the keys are unordered)
    face_coloring = np.full(len(curr_limb_mesh.faces),-1)
    for k,v in face_lookup.items():
        if len(v) == 1:
            face_coloring[k] = v[0]

    # -- Need to only take the biggest piece of the non-conflicted mesh and resolve those that were eliminated--

    for i in range(0,max_color_filling_iterations):
        
        face_coloring = filter_face_coloring_to_connected_components(curr_limb_mesh,face_coloring)

        # ----fixing if there were any missing labels --- **** this still has potential for erroring ****

        leftover_labels = np.unique(face_coloring)
        missing_labels = set(np.setdiff1d(no_missing_labels.copy(), leftover_labels))

        for curr_label in missing_labels:
            labels_idx = [k for k,v in face_lookup.items() if curr_label in v]
            face_coloring[labels_idx] = curr_label

        #filter the faces again: 
        face_coloring = filter_face_coloring_to_connected_components(curr_limb_mesh,face_coloring)

        leftover_labels = np.unique(face_coloring)
        missing_labels = set(np.setdiff1d(no_missing_labels.copy(), leftover_labels))
        if len(missing_labels) == 0:
            break
        else:
            if i > 0:
                print(f"Doing No Color conflicts iteration {i+1} because missing_labels = {missing_labels} ")
                
    if len(missing_labels)>0:
        import system_utils as su
        print(f"leftover_labels = {leftover_labels}")
        print(f"no_missing_labels = {no_missing_labels}")
        print(f"missing_labels = {missing_labels}")
        su.compressed_pickle(curr_limb_mesh,"curr_limb_mesh")
        su.compressed_pickle(face_lookup,"face_lookup")
        su.compressed_pickle(no_missing_labels,"no_missing_labels")
        su.compressed_pickle(max_submesh_threshold,"max_submesh_threshold")
        raise Exception("missing labels was not resolved")
        

    # -----now just divide the groups into seperate components
    empty_faces = np.where(face_coloring==-1)[0]

    mesh_graph = nx.from_edgelist(curr_limb_mesh.face_adjacency) # creating a graph from the faces
    empty_submesh = mesh_graph.subgraph(empty_faces) #gets the empty submeshes that are disconnected
    empty_connected_components = list(nx.connected_components(empty_submesh))

    # ---- Functions that will fill in the rest of the mesh correspondence ---- #

    face_coloring_copy = face_coloring.copy()
    
    print("BEFORE face_lookup_resolved_test")
    import system_utils as su
    su.compressed_pickle(empty_connected_components,"empty_connected_components")
    
    for comp in tqdm(empty_connected_components):
        #print("len(mesh_graph) = {len(mesh_graph)}")
        face_lookup_resolved_test = waterfill_labeling(
                        #total_mesh_correspondence=face_lookup_resolved_test,
                        total_mesh_correspondence=face_coloring_copy,
                         submesh_indices=list(comp),
                         total_mesh=None,
                        total_mesh_graph=mesh_graph,
                         propagation_type="random",
                        max_iterations = 1000,
                        max_submesh_threshold = max_submesh_threshold
                        )
    print("AFTER face_lookup_resolved_test")

    # -- wheck that the face coloring did not have any empty faces --
    empty_faces = np.where(face_coloring_copy==-1)[0]
    if len(empty_faces) > 0:
        import system_utils as su
        su.compressed_pickle(curr_limb_mesh,"curr_limb_mesh")
        su.compressed_pickle(face_lookup,"face_lookup")
        su.compressed_pickle(no_missing_labels,"no_missing_labels")
        su.compressed_pickle(max_submesh_threshold,"max_submesh_threshold")

        raise Exception(f"empty faces were greater than 0 after waterfilling at: {empty_faces}")


    return face_coloring_copy

import general_utils as gu
def groups_of_labels_to_resolved_labels(current_mesh,face_correspondence_lists):
    """
    Purpose: To take a list of face correspondences of different parts
    and turn them into an array mapping every face (on the mesh) to a label
    
    """
    no_missing_labels = np.arange(0,len(face_correspondence_lists))
    
    face_lookup = dict([(j,[]) for j in range(0,len(current_mesh.faces))])
    
    for j,curr_faces_corresponded in enumerate(face_correspondence_lists):
        for c in curr_faces_corresponded:
            face_lookup[c].append(j)
    
    original_labels = gu.get_unique_values_dict_of_lists(face_lookup)
    print(f"max(original_labels),len(original_labels) = {(max(original_labels),len(original_labels))}")
    
    if len(original_labels) != len(no_missing_labels):
        raise Exception(f"len(original_labels) != len(no_missing_labels) for original_labels = {len(original_labels)},no_missing_labels = {len(no_missing_labels)}")

    if max(original_labels) + 1 > len(original_labels):
        raise Exception("There are some missing labels in the initial labeling")
        
    #here is where can call the function that resolves the face labels
    face_coloring_copy = resolve_empty_conflicting_face_labels(
                     curr_limb_mesh = current_mesh,
                     face_lookup=face_lookup,
                     no_missing_labels = no_missing_labels
        
    )
    
    divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(current_mesh,face_coloring_copy)
    divided_submeshes = list(divided_submeshes.values())
    divided_submeshes_idx = list(divided_submeshes_idx.values())
    return divided_submeshes,divided_submeshes_idx


""" ------------ 9/17 Addition: Will expand a certain label until hits the soma border -------- """
def waterfill_starting_label_to_soma_border(curr_branch_mesh,
                                           border_vertices,
                                            label_to_expand,
                                           total_face_labels,
                                           print_flag=True):

    """
    Purpose: To expand a certain label so it is touching the soma 
    border vertices
    
    """
    #0) Turn the mesh into a graph
    total_mesh_graph = nx.from_edgelist(curr_branch_mesh.face_adjacency)

    #1) Get the nodes that represent the border
    border_faces = set(tu.vertices_coordinates_to_faces(curr_branch_mesh,border_vertices))

    final_faces = np.where(total_face_labels == label_to_expand)[0]

    n_touching_soma = len(border_faces.intersection(set(final_faces)))
    counter = 0
    max_iterations = 1000
    while n_touching_soma < 10 and n_touching_soma < len(border_faces):
        final_faces = np.unique(np.concatenate([xu.get_neighbors(total_mesh_graph,k) for k in final_faces]))
        n_touching_soma = len(border_faces.intersection(set(final_faces)))
        counter+= 1
        if counter > max_iterations:
            print("Couldn't get the final faces to touch the somas border, before breaking"
                 f"\nn_touching_soma = {n_touching_soma}, final_faces = {len(final_faces)}"
                 f" border_faces = {len(border_faces)}")
            break


    if print_flag:
        print(f"Took {counter} iterations to expand the label back")
        
    total_face_labels[final_faces] = label_to_expand
    return total_face_labels


    
    
    


    
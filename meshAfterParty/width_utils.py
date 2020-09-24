import compartment_utils as cu
import skeleton_utils as sk
import trimesh_utils as tu
import numpy as np
import neuron_utils as nru
import networkx_utils as xu
import numpy_utils as nu

def calculate_new_width(branch, 
                               skeleton_segment_size=1000,
                                   width_segment_size = None,
                              return_average=False,
                                   distance_by_mesh_center=True,
                              no_spines=True,
                              summary_measure="mean",
                              print_flag=False):
    """
    Purpose: To calculate the overall width 

    """
    
    f = getattr(np,summary_measure)
    
    #ges branch mesh without any spines
    if no_spines:
        ex_branch_no_spines_mesh = nru.branch_mesh_no_spines(branch)
    else:
        ex_branch_no_spines_mesh = branch.mesh

    #resizes the branch to the desired width (HELPS WITH SMOOTHING OF THE SKELETON)
    ex_branch_skeleton_resized = sk.resize_skeleton_branch(branch.skeleton,segment_width = skeleton_segment_size)

    #The size we want the widths to be calculated at
    if not width_segment_size is None:
        ex_branch_skeleton_resized = sk.resize_skeleton_branch(ex_branch_skeleton_resized,segment_width = width_segment_size)

    (total_distances,
     total_distances_std,
     new_submesh,
     unique_faces) = cu.get_skeletal_distance_no_skipping(main_mesh=ex_branch_no_spines_mesh,
                                    edges=ex_branch_skeleton_resized,
                                     buffer=0.01,
                                    bbox_ratio=1.2,
                                   distance_threshold=branch.width,
                                    distance_by_mesh_center=distance_by_mesh_center,
                                    print_flag=False,
                                    edge_loop_print=False
                                                         )

    total_distances = np.array(total_distances)

    branch_width_average = f(total_distances)
    if branch_width_average < 0.0001:
        #just assing the old width
        print("Assigning the old width calculation because no valid new widths")
        branch_width_average = branch.width
        total_distances = np.ones(len(ex_branch_skeleton_resized))*branch_width_average
    else:
        total_distances[total_distances == 0] = branch_width_average #IF RETURNED 0 THEN FILL with 

        if print_flag:
            print(f"Overall {summary_measure} = {branch_width_average}")
            print(f"Total_distances = {total_distances}")

    if return_average:
        return total_distances,branch_width_average
    else:
        total_distances
        
def find_mesh_width_array_border(curr_limb,
                             node_1,
                             node_2,
                            width_name = "no_spine_median_mesh_center",
                            segment_start = 1,
                            segment_end = 4,
                            skeleton_segment_size = None,
                            width_segment_size = None,
                            recalculate_width_array = False, #will automatically recalculate the width array
                            default_segment_size = 1000,
                                 no_spines=True,
                                 summary_measure="mean",
                            print_flag=True,
                            **kwargs
                            ):

    """
    Purpose: To send back an array that 
    represents the widths of curent branches
    at their boundary
    - the widths may be calculated differently than currently
      stored if specified so

    Applications: 
    1) Will help with filtering out false positives
    with the axon detection
    2) For merge detections to help detect
    large width change

    Process: 
    0) make sure the two nodes are connected in the concept network
    1) if the skeleton_segment_size and width_semgent is None then recalculate the width array
    - send the 
    2) calculate the endpoints from the skeletons (to ensure they are in the right order)
    3) find the connectivity of the endpoints
    4) Get the subarrays of the width_arrays according to the start and end specified
    5) return the subarrays

    Example of Use: 
    find_mesh_width_array_border(curr_limb=curr_limb_obj,
                             #node_1=56,
                             #node_2=71,
                             node_1 = 8,
                             node_2 = 5,
                            width_name = "no_spine_average_mesh_center",
                            segment_start = 1,
                            segment_end = 4,
                            skeleton_segment_size = 50,
                            width_segment_size = None,
                            recalculate_width_array = True, #will automatically recalculate the width array
                            default_segment_size = 1000,
                            print_flag=True
                            )

    """

    # 0) make sure the two nodes are connected in the concept network
    if node_2 not in xu.get_neighbors(curr_limb.concept_network,node_1):
        raise Exception(f"Node_1 ({node_1}) and Node_2 ({node_2}) are not connected in the concept network")


    # 0) extract the branch objects
    branch_obj_1 = curr_limb.concept_network.nodes[node_1]["data"]
    branch_obj_2 = curr_limb.concept_network.nodes[node_2]["data"]
    # 1) if the skeleton_segment_size and width_semgent is then recalculate the width array
    if not skeleton_segment_size is None or recalculate_width_array:

        if "mesh_center" in width_name:
            distance_by_mesh_center = True
        else:
            distance_by_mesh_center = False
            
        if ("no_spine" in width_name) or (no_spines):
            no_spines = True
        else:
            if print_flag:
                print("Using no spines")
            
        if print_flag:
            print(f"distance_by_mesh_center = {distance_by_mesh_center}")

        if skeleton_segment_size is None:
            skeleton_segment_size = default_segment_size

        if not nu.is_array_like(skeleton_segment_size):
            skeleton_segment_size = [skeleton_segment_size]

        if width_segment_size is None:
            width_segment_size = skeleton_segment_size

        if not nu.is_array_like(width_segment_size):
            width_segment_size = [width_segment_size]


        current_width_array_1,current_width_1 = calculate_new_width(branch_obj_1, 
                                          skeleton_segment_size=skeleton_segment_size[0],
                                          width_segment_size=width_segment_size[0], 
                                          distance_by_mesh_center=distance_by_mesh_center,
                                          return_average=True,
                                          print_flag=False,
                                        no_spines=no_spines,
                                                                   summary_measure=summary_measure)

        current_width_array_2,current_width_2 = calculate_new_width(branch_obj_2, 
                                          skeleton_segment_size=skeleton_segment_size[-1],
                                          width_segment_size=width_segment_size[-1], 
                                          distance_by_mesh_center=distance_by_mesh_center,
                                            no_spines=no_spines,
                                          return_average=True,
                                          print_flag=False,
                                            summary_measure=summary_measure)
    else:
        if print_flag:
            print("**Using the default width arrays already stored**")
        current_width_array_1 = branch_obj_1.width_array[width_name]
        current_width_array_2 = branch_obj_2.width_array[width_name]

    if print_flag:
        print(f"skeleton_segment_size = {skeleton_segment_size}")
        print(f"width_segment_size = {width_segment_size}")
        print(f"current_width_array_1 = {current_width_array_1}")
        print(f"current_width_array_2 = {current_width_array_2}")
    
    
    
    
    

    #2) calculate the endpoints from the skeletons (to ensure they are in the right order)
    end_1 = sk.find_branch_endpoints(branch_obj_1.skeleton)
    end_2 = sk.find_branch_endpoints(branch_obj_2.skeleton)
    
    if print_flag:
        print(f"end_1 = {end_1}")
        print(f"end_2 = {end_2}")
    

    #3) find the connectivity of the endpoints
    node_connectivity = xu.endpoint_connectivity(end_1,end_2)

    #4) Get the subarrays of the width_arrays according to the start and end specified
    """
    Pseudocode: 

    What to do if too small? Take whole thing

    """
    if print_flag:
        print(f"node_connectivity = {node_connectivity}")
    
    return_arrays = []
    width_arrays = [current_width_array_1,current_width_array_2]

    for j,current_width_array in enumerate(width_arrays):

        if len(current_width_array)<segment_end:
            if print_flag:
                print(f"The number of segments for current_width_array_{j+1} ({len(current_width_array)}) "
                     " was smaller than the number requested, so just returning the whole width array")

            return_arrays.append(current_width_array)
        else:
            if node_connectivity[j] == 0:
                return_arrays.append(current_width_array[segment_start:segment_end])
            elif node_connectivity[j] == 1:
                return_arrays.append(current_width_array[-segment_end:-segment_start])
            else:
                raise Exception("Node connectivity was not 0 or 1")

    return return_arrays

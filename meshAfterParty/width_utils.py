import compartment_utils as cu
import skeleton_utils as sk
import trimesh_utils as tu
import numpy as np
import neuron_utils as nru

def calculate_width_without_spines(branch, 
                               skeleton_segment_size=1000,
                                   width_segment_size = None,
                              return_average=False,
                                   distance_by_mesh_center=False,
                              print_flag=False):
   
        
    ex_branch_no_spines_mesh = nru.branch_mesh_no_spines(branch)
    
    ex_branch_skeleton_resized = sk.resize_skeleton_branch(branch.skeleton,segment_width = skeleton_segment_size)
    
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
    # 
    total_distances = np.array(total_distances)

    branch_width_average = np.mean(total_distances)
    if branch_width_average < 0.0001:
        #just assing the old width
        print("Assigning the old width calculation because no valid new widths")
        branch_width_average = branch.width
        total_distances = np.ones(len(ex_branch_skeleton_resized))*branch_width_average
    else:
        total_distances[total_distances == 0] = branch_width_average

        if print_flag:
            print(f"Overall Average = {branch_width_average}")
            print(f"Total_distances = {total_distances}")
    
    if return_average:
        return total_distances,branch_width_average
    else:
        total_distances
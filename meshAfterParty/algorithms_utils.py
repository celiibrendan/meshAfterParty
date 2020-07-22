"""
Purpose: defining general algorithms that help with processing

"""
import numpy as np

def compare_uneven_groups(group1,
                          group2,
                          comparison_func,
                          group_name="no_name",
                          return_differences=False,
                         print_flag=False):
    """
    Pseudocode: will return lists from each groups
    that are not in the other group
    
    Example: 
    import algorithms_utils as au
    au = reload(au)
    au.compare_uneven_groups(obj1.inside_pieces[:10],obj2.inside_pieces,
                             comparison_func = tu.compare_meshes_by_face_midpoints,
                             group_name="inside_pieces",
                             return_differences=True)

    def equal_func(a,b):
        return a == b
    au.compare_uneven_groups([1,3,5,7,9,10],[2,4,6,8,10],
                             comparison_func = equal_func,
                             group_name="numbers_list",
                             return_differences=True)
    
    """
    differences=[]
    return_boolean = None
#     if len(group1) != len(group2):
#         differences.append(f"lengths of {group_name} did not match")
#         if return_differences:
#             return False,differences
#         else:
#             return False
        
    group1_lacking = []
    group1_pairings = []
    group2_lacking = []
    group2_pairings = []
    
    
    self_indices = np.arange(len(group1) )
    other_indices = np.arange(len(group2) )
    
    for i in self_indices:
        found_match = False
        for j in other_indices:
            if comparison_func(group1[i],
                                group2[j]):
                other_indices = other_indices[other_indices != j]
                group1_pairings.append([i,j])
                found_match=True
                break
        if not found_match:    
            #if no match was found then add to the differences list
            differences.append(f"No match found for {group_name}_1[{i}]"
                              f"\nData = {group1[i]}")
            group1_lacking.append(i)
    
    #now find the group2 indices that couldn't be found
    self_indices = np.arange(len(group2) )
    other_indices = np.arange(len(group1) )
    
    for i in self_indices:
        found_match = False
        for j in other_indices:
            if comparison_func(group2[i],
                                group1[j]):
                other_indices = other_indices[other_indices != j]
                group2_pairings.append([i,j])
                found_match=True
                break
        if not found_match:    
            #if no match was found then add to the differences list
            differences.append(f"No match found for {group_name}_2[{i}]"
                              f"\nData = {group2[i]}")
            group2_lacking.append(i)

    if print_flag:
        print(f"group2_lacking = {group2_lacking}")
        print(f"group1_lacking = {group1_lacking}")
        print(f"group1_pairings = {group1_pairings}")
        print(f"group2_pairings = {group2_pairings}")
    if len(group2_lacking)>0 or len(group1_lacking)>0:
        return_boolean=False
    else:
        return_boolean=True
    
    if return_differences:
        return return_boolean,differences
    else:
        return return_boolean
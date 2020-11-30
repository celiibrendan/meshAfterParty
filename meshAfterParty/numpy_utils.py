import numpy as np

"""
Notes on functionality: 
np.concatenate: combines list of lists into one list like itertools does
np.ptp: gives range from maximum-minimum

np.diff #gets the differences between subsequent elements (turns n element --> n-1 elements)

np.insert(array,indexes of where you want insertion,what you want inserted before the places you specified) --> can do multiple insertions: 

Ex: 
x = np.array([1,4,5,10])
np.insert(x,slice(0,5),2)
>> output: array([ 2,  1,  2,  4,  2,  5,  2, 10])


If want to find the indexes of what is common between 2 1D arrray use
same_ids,x_ind,y_ind = np.intersect1d(soma_segment_id,connectivity_ids,return_indices=True)


"""
def compare_threshold(item1,item2,threshold=0.0001,print_flag=False):
    """
    Purpose: Function that will take a scalar or 2D array and subtract them
    if the distance between them is less than the specified threshold
    then consider equal
    
    Example: 
    nu = reload(nu)

    item1 = [[1,4,5,7],
             [1,4,5,7],
             [1,4,5,7]]
    item2 = [[1,4,5,8.00001],
            [1,4,5,7.00001],
            [1,4,5,7.00001]]

    # item1 = [1,4,5,7]
    # item2 = [1,4,5,9.0000001]

    print(nu.compare_threshold(item1,item2,print_flag=True))
    """
    item1 = np.array(item1)
    item2 = np.array(item2)

    if item1.ndim != item2.ndim:
        raise Exception(f"Dimension for item1.ndim ({item1.ndim}) does not equal item2.ndim ({item2.ndim})")
    if item1.ndim > 2 or item2.ndim > 2:
        raise Exception(f"compare_threshold does not handle items with greater than 2 dimensions: item1.ndim ({item1.ndim}), item2.ndim ({item2.ndim}) ")

    if item1.ndim < 2:
        difference = np.linalg.norm(item1-item2)
    else:
        difference = np.sum(np.linalg.norm(item1 - item2,axis=1))
    
    if print_flag:
        print(f"difference = {difference}")
        
    #compare against threshold and return result
    return difference <= threshold

def concatenate_lists(list_of_lists):
    try:
        return np.concatenate(list_of_lists)
    except:
        return []

def is_array_like(current_data):
    return type(current_data) in [type(np.ndarray([])),type(np.array([])),list]

def non_empty_or_none(current_data):
    if current_data is None:
        return False
    else:
        if len(current_data) == 0:
            return False
        return True

def array_after_exclusion(
                        original_array=[],                    
                        exclusion_list=[],
                        n_elements=0):
    """
    To efficiently get the difference between 2 lists:
    
    original_list = [1,5,6,10,11]
    exclusion = [10,6]
    n_elements = 20

    array_after_exclusion(n_elements=n_elements,exclusion_list=exclusion)
    
    
    ** pretty much the same thing as : 
    np.setdiff1d(array1, array2)

    """
    
    
    if len(exclusion_list) == 0: 
        return original_array
    
    if len(original_array)==0:
        if n_elements > 0:
            original_array = np.arange(n_elements)
        else:
            raise Exceptino("No original array passed")
    else:
        original_array = np.array(original_array)
            
    mask = ~np.isin(original_array,exclusion_list)
    #print(f"mask = {mask}")
    return original_array[mask]

from pathlib import Path
def load_dict(file_path):
    if file_path == type(Path()):
        file_path = str(file_path.absolute())
      
    my_dict = np.load(file_path,allow_pickle=True)
    return my_dict[my_dict.files[0]][()]


from scipy.spatial.distance import pdist,squareform
def get_coordinate_distance_matrix(coordinates):
    distance_matrix_condensed = pdist(coordinates,'euclidean')
    distance_matrix = squareform(distance_matrix_condensed)
    return distance_matrix

def get_matching_vertices(possible_vertices,ignore_diagonal=True,
                         equiv_distance=0,
                         print_flag=False):
    """
    ignore_diagonal is not implemented yet 
    """
    possible_vertices = possible_vertices.reshape(-1,3)
    
    dist_matrix = get_coordinate_distance_matrix(possible_vertices)
    
    dist_matrix_copy = dist_matrix.copy()
    dist_matrix_copy[np.eye(dist_matrix.shape[0]).astype("bool")] = np.inf
    if print_flag:
        print(f"The smallest distance (not including diagonal) = {np.min(dist_matrix_copy)}")
    
    matching_vertices = np.array(np.where(dist_matrix <= equiv_distance)).T
    if ignore_diagonal:
        left_side = matching_vertices[:,0]
        right_side = matching_vertices[:,1]

        result = matching_vertices[left_side != right_side]
    else:
        result = matching_vertices
        
    if len(result) > 0:
        return np.unique(np.sort(result,axis=1),axis=0)
    else:
        return result

def number_matching_vertices_between_lists(arr1,arr2,verbose=False):
    stacked_vertices = np.vstack([np.unique(arr1,axis=0),np.unique(arr2,axis=0)])
    stacked_vertices_unique = np.unique(stacked_vertices,axis=0)
    n_different = len(stacked_vertices) - len(stacked_vertices_unique)
    return n_different

def test_matching_vertices_in_lists(arr1,arr2,verbose=False):
    n_different = number_matching_vertices_between_lists(arr1,arr2)
    if verbose:
        print(f"Number of matching vertices = {n_different}")
    if n_different > 0:
        return True
    elif n_different == 0:
        return False
    else:
        raise Exception("More vertices in unique list")

"""
How can find pairwise distance:

example_skeleton = current_mesh_data[0]["branch_skeletons"][0]
ex_skeleton = example_skeleton.reshape(-1,3)


#sk.convert_skeleton_to_graph(ex_skeleton)

from scipy.spatial.distance import pdist
import time 
start_time = time.time()
distance_matrix = pdist(ex_skeleton,'euclidean')
print(f"Total time for pdist = {time.time() - start_time}")

returns a matrix that is a lower triangular matrix of size n*(n-1)/2
that gives the distance



"""
def find_matching_endpoints_row(branch_idx_to_endpoints,end_coordinates):
    match_1 = (branch_idx_to_endpoints.reshape(-1,3) == end_coordinates[0]).all(axis=1).reshape(-1,2)
    match_2 = (branch_idx_to_endpoints.reshape(-1,3) == end_coordinates[1]).all(axis=1).reshape(-1,2)
    return np.where(np.sum(match_1 + match_2,axis=1)>1)[0]

def matching_rows_old(vals,row,print_flag=False):

    if len(vals) == 0:
        return np.array([])
    vals = np.array(vals)
    if print_flag:
        print(f"vals = {vals}")
        print(f"row = {row}")
    return np.where((np.array(vals) == np.array(row)).all(axis=1))[0]

def matching_rows(vals,row,
                      print_flag=False,
                      equiv_distance = 0.0001):

    if len(vals) == 0:
        return np.array([])
    vals = np.array(vals)
    row = np.array(row).reshape(-1,3)
    if print_flag:
        print(f"vals = {vals}")
        print(f"row = {row}")
    return np.where(np.linalg.norm(vals-row,axis=1)<equiv_distance)[0]




# ----------- made when developing the neuron class ------------- #
def sort_multidim_array_by_rows(edge_array,order_row_items=False):
    """
    Purpose: To sort an array along the 0 axis where you maintain the row integrity
    (with possibly sorting the individual elements along a row)
    
    Example: On how to get sorted edges
    import numpy_utils as nu
    nu = reload(nu)
    nu.sort_multidim_array_by_rows(limb_concept_network.edges(),order_row_items=True)
    
    """
    #print(f'edge_array = {edge_array} with type = {type(edge_array)}')
    
    #make sure it is an array
    edge_array = np.array(edge_array)
    
    #check that multidimensional
    if len(edge_array.shape ) < 2:
        print(f"edge_array = {edge_array}")
        raise Exception("array passed did not have at least 2 dimensions")
        
    #will rearrange the items to be in a row if not care about the order here
    if order_row_items:
        edge_array = np.sort(edge_array,axis=1)

    #sort by the x and then y of the egde
    def sorting_func(k):
        return [k[i] for i,v in enumerate(edge_array.shape)]

    #sorted_edge_array = np.array(sorted(edge_array , key=lambda k: [k[0], k[1]]))
    sorted_edge_array = np.array(sorted(edge_array , key=sorting_func))
    
    return sorted_edge_array



def sort_elements_in_every_row(current_array):
    return np.array([np.sort(yi) for yi in current_array])
# --------- Functions pulled from trimesh.grouping ---------- #


def intersect1d(arr1,arr2,assume_unique=False,return_indices=False):
    """
    Will return the common elements from 2 possibly different sized arrays
    
    If select the return indices = True,
    will also return the indexes of the common elements
    
    
    """
    return np.intersect1d(arr1,arr2,
                         assume_unique=assume_unique,
                         return_indices=return_indices)

def setdiff1d(arr1,arr2,assume_unique=False,return_indices=True):
    """
    Purpose: To get the elements in arr1 that aren't in arr2
    and then to possibly return the indices of those that were
    unique in the first array
    
    
    
    """
    
    arr1 = np.array(arr1)
    leftout = np.setdiff1d(arr1,arr2,assume_unique=assume_unique)
    _, arr_1_indices, _ = np.intersect1d(arr1,leftout,return_indices=True)
    arr_1_indices_sorted= np.sort(arr_1_indices)
    if return_indices:
        return arr1[arr_1_indices_sorted],arr_1_indices_sorted
    else:
        return arr1[arr_1_indices_sorted]
    
    
def divide_into_label_indexes(mapping):
    """
    Purpose: To take an array that attributes labels to indices
    and divide it into a list of the arrays that correspond to the indices of
    all of the labels
    
    """
    unique_labels = np.sort(np.unique(mapping))
    final_list = [np.where(mapping==lab)[0] for lab in unique_labels]
    return final_list

def turn_off_scientific_notation():
    np.set_printoptions(suppress=True)
    
def average_by_weights(values,weights):
    weights_normalized = weights/np.sum(weights)
    return np.sum(values*weights_normalized)

def angle_between_vectors(v1, v2, acute=True,degrees=True,verbose=False):
    """
    vec1 = np.array([0,0,1])
    vec2 = np.array([1,1,-0.1])
    angle(vec1,vec2,verbose=True)
    """

    dot_product = np.dot(v1, v2)
    if verbose:
        print(f"dot_product = {dot_product}")
    angle = np.arccos(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    if acute == True:
        rad_angle =  angle
    else:
        rad_angle =  2 * np.pi - angle
        
    if degrees:
        return  180* rad_angle/np.pi
    else:
        return rad_angle
            
    
    return return_angle

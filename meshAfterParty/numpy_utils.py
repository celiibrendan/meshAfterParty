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

def matching_rows(vals,row):
    if len(vals) == 0:
        return np.array([])
    return np.where((np.array(vals) == np.array(row)).all(axis=1))[0]

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
import numpy_utils as nu
import numpy as np
def invert_mapping(my_map):
    """
    Will invert a dictionary mapping that is not unique
    (Also considers an array of a mapping of the indices to the value)
    
    Ex: 
    input: [8,1,4,5,4,6,8]
    output: {8: [0, 6], 1: [1], 4: [2, 4], 5: [3], 6: [5]}
    """
    if type(my_map) == dict:
        pass
    if nu.is_array_like(my_map):
        my_map = dict([(i,k) for i,k in enumerate(my_map)])
    else:
        raise Exception("Non dictionary or array type recieved")

    inv_map = {}
    
    #handling the one-dimensional case where dictionary just maps to numbers
    if np.isscalar(list(my_map.values())[0]):
        for k, v in my_map.items():
            inv_map[v] = inv_map.get(v, []) + [k]
    else: #2-D case where dictionary maps to list of numbers
        for k,v1 in my_map.items():
            for v in v1:
                inv_map[v] = inv_map.get(v, []) + [k]
        
    return inv_map

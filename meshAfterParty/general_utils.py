
"""
How to recieve a single values tuple 
and unpack it into its element --> use (__ , ) on recieving end

def ex_function(ex_bool = True,second_bool=False):
    
    
    return_value = (5,)
    if ex_bool:
        return_value += (6,)
    if second_bool:
        return_value += (7,)
        

    return return_value

(y,) = ex_function(ex_bool=False,second_bool=False)
type(y)

"""


import numpy_utils as nu
import numpy as np
import itertools
def invert_mapping(my_map,total_keys=None):
    """
    Will invert a dictionary mapping that is not unique
    (Also considers an array of a mapping of the indices to the value)
    
    Ex: 
    input: [8,1,4,5,4,6,8]
    output: {8: [0, 6], 1: [1], 4: [2, 4], 5: [3], 6: [5]}
    """
    if type(my_map) == dict:
        pass
    elif nu.is_array_like(my_map):
        my_map = dict([(i,k) for i,k in enumerate(my_map)])
    else:
        raise Exception("Non dictionary or array type recieved")
        
    if total_keys is None:
        inv_map = {}
    else:
        inv_map = dict([(k,[]) for k in total_keys])
    
    #handling the one-dimensional case where dictionary just maps to numbers
    if np.isscalar(list(my_map.values())[0]):
        for k, v in my_map.items():
            inv_map[v] = inv_map.get(v, []) + [k]
    else: #2-D case where dictionary maps to list of numbers
        for k,v1 in my_map.items():
            for v in v1:
                inv_map[v] = inv_map.get(v, []) + [k]
        
    return inv_map

def get_unique_values_dict_of_lists(dict_of_lists):
    """
    Purpose: If have a dictionary that maps the keys to lists,
    this function will give the unique values of all the elements of all the lists
    
    
    """
    return set(list(itertools.chain.from_iterable(list(dict_of_lists.values()))))

def flip_key_orders_for_dict(curr_dict):
    """
    To flip the order of keys in dictionarys with multiple
    levels of keys:
    
    Ex: 
    test_dict = {0:{1:['a','b','c'],2:['c','d','e']},
            1:{0:['i','j','k'],2:['f','g','h']}}
    
    output:
    {0: {1: ['i', 'j', 'k']},
     1: {0: ['a', 'b', 'c']},
     2: {0: ['c', 'd', 'e'], 1: ['f', 'g', 'h']}}
     
     Pseudocode: 
     How to flip the soma to piece touching dictionaries
    1) get all the possible limb keys
    2) Create a dictionary with empty list
    3) Iterate through all of the somas
    - if the limb is in the keys then add the info (if not then skip)

    
    """
    test_dict = curr_dict
    all_limbs = np.unique(np.concatenate([list(v.keys()) for v in test_dict.values()]))
    flipped_dict = dict()
    for l_idx in all_limbs:
        flipped_dict[l_idx]=dict()
        for sm_idx,sm_dict in test_dict.items():
            if l_idx in sm_dict.keys():
                flipped_dict[l_idx].update({sm_idx:sm_dict[l_idx]})
    return flipped_dict


#have to reorder the keys
def order_dict_by_keys(current_dict):
    current_dict_new = dict([(k,current_dict[k]) for k in np.sort(list(current_dict.keys()))])
    return current_dict_new


def dict_to_array(current_dict):
    return np.vstack([list(current_dict.keys()),list(current_dict.values())]).T




# ------------ Help  with raising errors --------------- #
class Error(Exception): 
  
    # Error is derived class for Exception, but 
    # Base class for exceptions in this module 
    pass
  
class CGAL_skel_error(Error): 
  
    # Raised when an operation attempts a state  
    # transition that's not allowed. 
    def __init__(self,  msg): 
  
        # Error message thrown is saved in msg 
        self.msg = msg 
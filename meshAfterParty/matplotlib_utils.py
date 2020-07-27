from matplotlib import colors
import numpy as np

"""
Notes on other functions: 
eventplot #will plot 1D data as lines, can stack multiple 1D events
-- if did a lot of these gives the characteristic neuron spikes
   all stacked on top of each other


matplot colors can be described with 
"C102" where C{number} --> there are only 10 possible colors
but the number can go as high as you want it just repeats after 10
Ex: C100  = C110



"""

graph_color_list = ["blue","green","red","cyan","magenta",
     "black","grey","midnightblue","pink","crimson",
     "orange","olive","sandybrown","tan","gold","palegreen",
    "darkslategray","cadetblue","brown","forestgreen"]



def generate_random_color(print_flag=False):
    rand_color = np.random.choice(graph_color_list,1)
    if print_flag:
        print(f"random color chosen = {rand_color}")
    return colors.to_rgba(rand_color[0])

def generate_color_list(
                        user_colors=[], #if user sends a prescribed list
                        n_colors=-1,
                        colors_to_omit=[],
                        alpha_level=0.2):
    """
    Can specify the number of colors that you want
    Can specify colors that you don't want
    accept what alpha you want
    
    Example of how to use
    colors_array = generate_color_list(colors_to_omit=["green"])
    """
    #print(f"user_colors = {user_colors}")
    # if user_colors is defined then use that 
    if len(user_colors)>0:
        current_color_list = user_colors
    else:
        current_color_list = graph_color_list.copy()
    
    #remove any colors that shouldn't belong
    current_color_list = [k for k in current_color_list if k not in colors_to_omit]
    
    #print(f"current_color_list = {current_color_list}")
    
    if len(current_color_list) < len(user_colors):
        raise Exception(f"one of the colors you specified was part of unallowed colors {colors_to_omit}for a skeleton (because reserved for main)")
    
    #make a list as long as we need
    if n_colors > 0:
        current_color_list = (current_color_list*np.ceil(n_colors/len(current_color_list)).astype("int"))[:n_colors]
    
    #print(f"current_color_list = {current_color_list}")
    #now turn the color names all into rgb
    color_list_rgb = np.array([colors.to_rgba(k) for k in current_color_list])
    
    #changing the alpha level to the prescribed value
    color_list_rgb[:,3] = alpha_level
    
    return color_list_rgb


    
#----------------------------- Functions that were made for new graph visualization ------------------- #

def color_to_rgb(color_str):
    """
    To turn a string of a color into an RGB value
    
    Ex: color_to_rgb("red")
    """
    if type(color_str) == str:
        return colors.to_rgb(color_str)
    else:
        return np.array(color_str)

def color_to_rgba(current_color,alpha=0.2):
    curr_rgb = color_to_rgb(current_color)
    return apply_alpha_to_color_list(curr_rgb,alpha=alpha)
    
from copy import copy
def get_graph_color_list():
    return copy(graph_color_list)

def generate_random_rgba(print_flag=False):
    rand_color = np.random.choice(graph_color_list,1)
    if print_flag:
        print(f"random color chosen = {rand_color}")
    return colors.to_rgb(rand_color[0])

import numpy_utils as nu
def generate_color_list_no_alpha_change(
                        user_colors=[], #if user sends a prescribed list
                        n_colors=-1,
                        colors_to_omit=[],
                        alpha_level=0.2):
    """
    Can specify the number of colors that you want
    Can specify colors that you don't want
    accept what alpha you want
    
    Example of how to use
    colors_array = generate_color_list(colors_to_omit=["green"])
    """
    if len(user_colors)>0:
        current_color_list = user_colors
    else:
        current_color_list = graph_color_list.copy()
    
    if len(colors_to_omit) > 0:
        colors_to_omit_converted = np.array([color_to_rgb(k) for k in colors_to_omit])
        #print(f"colors_to_omit_converted = {colors_to_omit_converted}")
        colors_to_omit_converted = colors_to_omit_converted[:,:3]

        #remove any colors that shouldn't belong
        colors_to_omit = []
        current_color_list = [k for k in current_color_list if len(nu.matching_rows(colors_to_omit_converted,k[:3])) == 0]
    
    #print(f"current_color_list = {current_color_list}")
    
    if len(current_color_list) == 0:
        raise Exception(f"No colors remaining in color list after colors_to_omit applied ({current_color_list})")
    
    #make a list as long as we need

    current_color_list = (current_color_list*np.ceil(n_colors/len(current_color_list)).astype("int"))[:n_colors]
    
    return current_color_list


def process_non_dict_color_input(color_input):
    """
    Will return a color list that is as long as n_items
    based on a diverse set of options for how to specify colors
    
    - string
    - list of strings
    - 1D np.array
    - list of strings and 1D np.array
    - list of 1D np.array or 2D np.array
    
    *Warning: This will not be alpha corrected*
    """
    
    if color_input == "random": #if just string that says random
        graph_color_list = get_graph_color_list()
        color_list = [color_to_rgb(k) for k in graph_color_list]
    elif type(color_input) == str: #if just give a string then turn into list with string
        color_list = [color_to_rgb(color_input)]
    elif all(type(elem)==str for elem in color_input): #if just list of strings
        color_list = [color_to_rbg(k) for k in color_input]
    elif any(nu.is_array_like(elem) for elem in color_input): #if there is an array in the list 
        color_list = [color_to_rgb(k) if type(k)==str else k for k in  color_input]
    else:
        color_list = [color_input]
    
    return color_list

def apply_alpha_to_color_list(color_list,alpha=0.2,print_flag=False):
    single_input = False
    if not nu.is_array_like(color_list):
        color_list = [color_list]
        single_input = True
    color_list_alpha_fixed = []
    
    for c in color_list:
        if len(c) == 3:
            color_list_alpha_fixed.append(np.concatenate([c,[alpha]]))
        elif len(c) == 4:
            color_list_alpha_fixed.append(c)
        else:
            raise Exception(f"Found color that was not 3 or 4 length array in colors list: {c}")
    if print_flag:
        print(f"color_list_alpha_fixed = {color_list_alpha_fixed}")
    
    if single_input:
        return color_list_alpha_fixed[0]
    
    return color_list_alpha_fixed
    


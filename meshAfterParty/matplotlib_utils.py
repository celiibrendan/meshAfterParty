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
        color_list = [color_to_rgb(k) for k in color_input]
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



import webcolors
import numpy as np

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def convert_rgb_to_name(rgb_value):
    """
    Example: convert_rgb_to_name(np.array([[1,0,0,0.5]]))
    """
    rgb_value = np.array(rgb_value)
    if not nu.is_array_like(rgb_value[0]):
        rgb_value = rgb_value.reshape(1,-1)
    
    #print(f"rgb_value.shape = {rgb_value.shape}")

    output_colors = []
    for k in rgb_value:
        if len(k) > 3:
            k = k[:3]
        adjusted_color_value = np.array(k)*255
        output_colors.append(get_colour_name(adjusted_color_value)[-1])
    
    if len(output_colors) == 1:
        return output_colors[0]
    elif len(output_colors) > 1:
        return output_colors
    else:
        raise Exception("len(output_colors) == 0")
        
def convert_dict_rgb_values_to_names(color_dict):
    """
    Purpose: To convert dictonary with colors as values to the color names
    instead of the rgb equivalents
    
    Application: can be used on the color dictionary returned by the 
    neuron plotting function
    
    Example: 
    import matplotlib_utils as mu
    mu = reload(mu)
    nviz=reload(nviz)


    returned_color_dict = nviz.visualize_neuron(uncompressed_neuron,
                                                visualize_type=["network"],
                                                network_resolution="branch",
                                                network_directional=True,
                                                network_soma=["S1","S0"],
                                                network_soma_color = ["black","red"],       
                                                limb_branch_dict=dict(L1="all",
                                                L2="all"),
                                                node_size = 1,
                                                arrow_size = 1,
                                                return_color_dict=True)
                                                
    color_info = mu.convert_dict_rgb_values_to_names(returned_color_dict)
    
    
    """
    return dict([(k,convert_rgb_to_name(v)) for k,v in color_dict.items()])
    

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

base_colors_dict = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)


def plot_color_dict(colors,sorted_names=None, 
                    hue_sort=False,
                    ncols = 4,
                    figure_width = 20,
                    figure_height = 8,
                   print_flag=True):

    """
    Ex: 
    
    #how to plot the base colors
    Examples: 
    mu.plot_color_dict(mu.base_colors_dict,figure_height=20)
    mu.plot_color_dict(mu.base_colors_dict,hue_sort=True,figure_height=20)
    
    How to plot colors returned from the plotting function:
    import matplotlib_utils as mu
    mu = reload(mu)
    nviz=reload(nviz)


    returned_color_dict = nviz.visualize_neuron(uncompressed_neuron,
                                                visualize_type=["network"],
                                                network_resolution="branch",
                                                network_directional=True,
                                                network_soma=["S1","S0"],
                                                network_soma_color = ["black","red"],       
                                                limb_branch_dict=dict(L1="all",
                                                L2="all"),
                                                node_size = 1,
                                                arrow_size = 1,
                                                return_color_dict=True)
                                                
    
    mu.plot_color_dict(returned_color_dict,hue_sort=False,figure_height=20)
    
    """
    if sorted_names is None:
        if hue_sort:
            # Sort colors by hue, saturation, value and then by name.
            by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                            for name, color in colors.items())
            #getting the names of the 
            sorted_names = [name for hsv, name in by_hsv]
        else:
            sorted_names = sorted(list(colors.keys()))
    n = len(sorted_names)
     #will always have 4 columns
    nrows = n // ncols + 1

    if print_flag:
        print(f"nrows = {nrows}")
        print(f"n-ncols*nrows = {n-ncols*nrows}")
    #creates figure
    fig, ax = plt.subplots(figsize=(figure_width, figure_height))

    # Get height and width
    X, Y = fig.get_dpi() * fig.get_size_inches()
    h = Y / (nrows + 1)
    w = X / ncols

    for i, name in enumerate(sorted_names):
        row = i % nrows
        col = i // nrows
        y = Y - (row * h) - h

        xi_line = w * (col + 0.05)
        xf_line = w * (col + 0.25)
        xi_text = w * (col + 0.3)

        ax.text(xi_text, y, name, fontsize=(h * 0.5),
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y + h * 0.1, xi_line, xf_line,
                  #gets the color by name
                  color=colors[name], linewidth=(h * 0.6)
                 #color=[1,0,0,1],linewidth=(h * 0.6)
                 )

    ax.set_xlim(0, X)
    ax.set_ylim(0, Y)
    ax.set_axis_off()

    fig.subplots_adjust(left=0, right=1,
                        top=1, bottom=0,
                        hspace=0, wspace=0)
    plt.show()

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
    
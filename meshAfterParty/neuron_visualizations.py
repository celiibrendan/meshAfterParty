import ipyvolume as ipv
import skeleton_utils as sk
import numpy as np
import networkx as nx
import neuron_utils as nru
import networkx_utils as xu
import time
from importlib import reload

def plot_soma_limb_concept_network(neuron_obj,
                                  soma_color="red",
                                  limb_color="aqua",
                                   multi_touch_color = "brown",
                                  node_size=500,
                                  font_color="black",
                                  node_colors=dict(),
                                  **kwargs):
    """
    Purpose: To plot the connectivity of the soma and the meshes in the neuron

    How it was developed: 

    import networkx_utils as xu
    xu = reload(xu)
    node_list = xu.get_node_list(my_neuron.concept_network)
    node_list_colors = ["red" if "S" in n else "blue" for n in node_list]
    nx.draw(my_neuron.concept_network,with_labels=True,node_color=node_list_colors,
           font_color="white",node_size=500)

    """

    node_list = xu.get_node_list(neuron_obj.concept_network)
    multi_touch_nodes = neuron_obj.same_soma_multi_touching_limbs
    node_list_colors = []
    for n in node_list:
        if n in list(node_colors.keys()):
            curr_color = node_colors[n]
        else:
            if "S" in n:
                curr_color = soma_color
            else:
                if int(n[1:]) in multi_touch_nodes:
                    curr_color = multi_touch_color
                else:
                    curr_color = limb_color
        node_list_colors.append(curr_color)
    
    #print(f"font_color = {font_color}")
    nx.draw(neuron_obj.concept_network,with_labels=True,node_color=node_list_colors,
           font_color=font_color,node_size=node_size)
    
from copy import deepcopy
import neuron
import matplotlib.pyplot as plt
def plot_limb_concept_network_2D(neuron_obj,
                                 node_colors=dict(),
                                 limb_name=None,
                                 somas=None,
                                 starting_soma=None,
                                 starting_soma_group=0,
                                 default_color = "green",
                                  node_size=2000,
                                  font_color="white",
                                 font_size=30,
                                 directional=True,
                                 print_flag=False,
                                 plot_somas=True,
                                 soma_color="red",
                                 pos=None,
                                 pos_width = 3,
                                 width_min = 0.3,
                                 width_noise_ampl=0.2,
                                 pos_vertical_gap=0.05,
                                 fig_width=40,
                                 fig_height=20,
                                 suppress_disconnected_errors=True,
                                  **kwargs):
    """
    Purpose: To plot the concept network as a 2D networkx graph
    
    Pseudocode: 
    0) If passed a neuron object then use the limb name to get the limb object
    - make copy of limb object
    1) Get the somas that will be used for concept network
    2) Assemble the network by concatenating (directional or undirectional)
    3) Assemble the color list to be used for the coloring of the nodes. Will take:
    a. dictionary
    b. List
    c. Scalar value for all nodes
    
    4) Add on the soma to the graphs if asked for it
    5) Generate a hierarchical positioning for graph if position argument not specified
    
    for all the starting somas
    4) Use the nx.draw function
    
    Ex: 
    nviz = reload(nviz)
    xu = reload(xu)
    limb_idx = "L3"
    nviz.plot_limb_concept_network_2D(neuron_obj=uncompressed_neuron,
                                     limb_name=limb_idx,
                                     node_colors=color_dictionary)
    """
    
    #0) If passed a neuron object then use the limb name to get the limb object
    #- make copy of limb object
    
    if limb_name is None and len(node_colors)>0:
        #just strip the L name of the first key that is not the soma
        limb_name = [k for k in node_colors.keys() if "S" not in k][0].split("_")[0]
        print(f"No limb name was given so using {limb_name} because was the limb in the first key")
        
    if str(type(neuron_obj)) == str(neuron.Neuron):
        if not limb_name is None:
            limb_obj = deepcopy(neuron_obj.concept_network.nodes[limb_name]["data"])
        else:
            raise Exception("Neuron object recieved but no limb name specified")
    elif str(type(neuron_obj)) == str(neuron.Limb):
        limb_obj = deepcopy(neuron_obj)
    else:
        raise Exception(f"Non Limb or Neuron object recieved: {type(neuron_obj)}")
    
    #1) Get the somas that will be used for concept network
    if somas is None:
        somas = limb_obj.touching_somas()
        somas = [somas[0]]
    
    #2) Assemble the network by concatenating (directional or undirectional)
    # (COULD NOT END UP CONCATENATING AND JUST USE ONE SOMA AS STARTING POINT)
    if directional:
        graph_list = []
        if starting_soma is not None:
            limb_obj.set_concept_network_directional(starting_soma=starting_soma,
                                                     soma_group_idx = starting_soma_group,
                                                     suppress_disconnected_errors=suppress_disconnected_errors)
            full_concept_network = limb_obj.concept_network_directional
        else:
            for s in somas:
                limb_obj.set_concept_network_directional(starting_soma=s,suppress_disconnected_errors=suppress_disconnected_errors)
                graph_list.append(limb_obj.concept_network_directional)
            full_concept_network = xu.combine_graphs(graph_list)
    else:
        full_concept_network = limb_obj.concept_network

    
    #3) Assemble the color list to be used for the coloring of the nodes. Will take:
    #a. dictionary
    #b. List
    #c. Scalar value for all nodes
    color_list = []
    node_list = xu.get_node_list(full_concept_network)
    
    if type(node_colors) == dict:
        #check to see if it is a limb_branch_dict
        L_check = np.any(["L" in k for k in node_colors.keys()])
        
        if L_check:
            if limb_name is None:
                raise Exception("Limb_branch dictionary given for node_colors but no limb name given to specify color mappings")
            node_colors = dict([(int(float(k.split("_")[-1])),v) for k,v in node_colors.items() if limb_name in k])
        
        if set(list(node_colors.keys())) != set(node_list):
            if print_flag:
                print(f"Node_colors dictionary does not have all of the same keys so using default color ({default_color}) for missing nodes")
        for n in node_list:
            if n in node_colors.keys():
                color_list.append(node_colors[n])
            else:
                color_list.append(default_color)
    elif type(node_colors) == list:
        if len(node_list) != len(node_colors):
            raise Exception(f"List of node_colors {(len(node_colors))} passed does not match list of ndoes in limb graph {(len(node_list))}")
        else:
            color_list = node_colors
    elif type(node_colors) == str:
        color_list = [node_colors]*len(node_list)
    else:
        raise Exception(f"Recieved invalid node_list type of {type(node_colors)}")
    
    #4) Add on the soma to the graphs if asked for it
    if plot_somas:
        #adding the new edges
        new_edge_list = []
        for k in limb_obj.all_concept_network_data:
            curr_soma = k["starting_soma"]
            curr_soma_group = k["soma_group_idx"]
            sm_name = f'S{k["starting_soma"]}_{k["soma_group_idx"]}'
            if curr_soma == limb_obj.current_starting_soma and curr_soma_group == limb_obj.current_soma_group_idx:
                new_edge_list.append((sm_name,k["starting_node"]))
            else:
                new_edge_list.append((k["starting_node"],sm_name))
        #new_edge_list = [(f'S{k["starting_soma"]}',k["starting_node"]) for k in limb_obj.all_concept_network_data]
        full_concept_network.add_edges_from(new_edge_list)
        #adding the new colors
        color_list += [soma_color]*len(new_edge_list)
    
    #print(f"full_concept_network.nodes = {full_concept_network.nodes}")
    #5) Generate a hierarchical positioning for graph if position argument not specified
    if pos is None:
        sm_name = f'S{limb_obj.current_starting_soma}_{limb_obj.current_soma_group_idx}'
        if plot_somas:
            starting_hierarchical_node = sm_name
        else:
            starting_hierarchical_node = {limb_obj.current_starting_node}
        #print(f"full_concept_network.nodes() = {full_concept_network.nodes()}")
        pos = xu.hierarchy_pos(full_concept_network,starting_hierarchical_node,
                              width=pos_width,width_min=width_min,width_noise_ampl=width_noise_ampl, vert_gap = pos_vertical_gap, vert_loc = 0, xcenter = 0.5)    
        #print(f"pos = {pos}")
    
    if print_flag:
        print(f"node_colors = {node_colors}")
        
    #6) Use the nx.draw function
    #print(f"pos={pos}")
    
    
    plt.figure(1,figsize=(fig_width,fig_height))
    nx.draw(full_concept_network,
            pos=pos,
            with_labels=True,
            node_color=color_list,
           font_color=font_color,
            node_size=node_size,
            font_size=font_size,
           **kwargs)
    


def plot_concept_network(curr_concept_network,
                            arrow_size = 0.5,
                            arrow_color = "maroon",
                            edge_color = "black",
                            node_color = "red",
                            scatter_size = 0.1,
                            starting_node_color="pink",
                            show_at_end=True,
                            append_figure=False,
                            highlight_starting_node=True,
                            starting_node_size=-1,
                                 flip_y=True,
                        suppress_disconnected_errors=False):
    
    if starting_node_size == -1:
        starting_node_size = scatter_size*3
    
    """
    Purpose: 3D embedding plot of concept graph
    
    
    Pseudocode: 

    Pseudocode for visualizing direction concept graphs
    1) Get a dictionary of the node locations
    2) Get the edges of the graph
    3) Compute the mipoints and directions of all of the edges
    4) Plot a quiver plot using the midpoints and directions for the arrows
    5) Plot the nodes and edges of the graph

    
    Example of how to use with background plot of neuron:
    
    my_neuron #this is the curent neuron object
    plot_concept_network(curr_concept_network = curr_limb_concept_network_directional,
                        show_at_end=False,
                        append_figure=False)

    # Just graphing the normal edges without

    curr_neuron_mesh =  my_neuron.mesh
    curr_limb_mesh =  my_neuron.concept_network.nodes[f"L{curr_limb_idx}"]["data"].mesh

    sk.graph_skeleton_and_mesh(other_meshes=[curr_neuron_mesh,curr_limb_mesh],
                              other_meshes_colors=["olive","brown"],
                              show_at_end=True,
                              append_figure=True)
                              
                              
    Another example wen testing: 
    import neuron_visualizations as nviz
    nviz = reload(nviz)
    nru = reload(nru)
    sk = reload(sk)

    nviz.plot_concept_network(curr_concept_network = curr_limb_concept_network_directional,
                            scatter_size=0.3,
                            show_at_end=True,
                            append_figure=False)
    
    """
    
    
    if not append_figure:
        ipv.pylab.clear()
        ipv.figure(figsize=(15,15))
    
    node_locations = dict([(k,curr_concept_network.nodes[k]["data"].mesh_center) for k in curr_concept_network.nodes()])

    node_edges = np.array(list(curr_concept_network.edges))



    if type(curr_concept_network) == type(nx.DiGraph()):
        #print("plotting a directional concept graph")
        #getting the midpoints then the directions of arrows for the quiver
        midpoints = []
        directions = []
        for n1,n2 in curr_concept_network.edges:
            difference = node_locations[n2] - node_locations[n1]
            directions.append(difference)
            midpoints.append(node_locations[n1] + difference/2)
        directions = np.array(directions)
        midpoints = np.array(midpoints)



        ipv.pylab.quiver(midpoints[:,0],midpoints[:,1],midpoints[:,2],
                        directions[:,0],directions[:,1],directions[:,2],
                        size=arrow_size,
                        size_selected=20,
                        color = arrow_color)

    #graphing the nodes

    # nodes_mesh = ipv.pylab.scatter(node_locations_array[:,0], 
    #                                 node_locations_array[:,1], 
    #                                 node_locations_array[:,2],
    #                                 size = 0.01,
    #                                 marker = "sphere")

    node_locations_array = np.array([v for v in node_locations.values()])
    #print(f"node_locations_array = {node_locations_array}")

    
    
    if highlight_starting_node:
        starting_node_num = xu.get_starting_node(curr_concept_network,only_one=False)
        starting_node_num_coord = [curr_concept_network.nodes[k]["data"].mesh_center for k in starting_node_num]
    
        #print(f"Highlighting starting node {starting_node_num} with coordinate = {starting_node_num_coord}")
        
        for k in starting_node_num_coord:
            sk.graph_skeleton_and_mesh(
                                       other_scatter=[k],
                                       other_scatter_colors=starting_node_color,
                                       scatter_size=starting_node_size,
                                       show_at_end=False,
                                       append_figure=True
                                      )
    
    #print(f"Current scatter size = {scatter_size}")
    concept_network_skeleton = nru.convert_concept_network_to_skeleton(curr_concept_network)
    sk.graph_skeleton_and_mesh(other_skeletons=[concept_network_skeleton],
                              other_skeletons_colors=edge_color,
                               other_scatter=[node_locations_array.reshape(-1,3)],
                               other_scatter_colors=node_color,
                               scatter_size=scatter_size,
                               show_at_end=False,
                               append_figure=True,
                               flip_y=flip_y,
                              )
    
    

    
    
    
    if show_at_end:
        ipv.show()
        
from copy import deepcopy
def visualize_concept_map(curr_concept_network,
                            node_color="red",
                            #node_color="black",
                            node_alpha = 0.5,
                            edge_color="black",
                            node_size=0.1,

                            starting_node=True,
                            starting_node_size = 0.3,
                            starting_node_color= "pink",
                            starting_node_alpha = 0.8,

                            arrow_color = "brown",
                            arrow_alpha = 0.8,
                            arrow_size = 0.5,

                            arrow_color_reciprocal = "brown",
                            arrow_alpha_reciprocal = 0.8,
                            arrow_size_reciprocal = 0.5,
                          
                            show_at_end=True,
                            append_figure=False,
                         print_flag=False,
                         flip_y=True):

    
    """
    Purpose: To plot a concept network with more
    parameters than previous plot_concept_network
    
    Ex: 
    
    neuron = reload(neuron)
    recovered_neuron = neuron.Neuron(recovered_neuron)
    nru = reload(nru)
    nviz = reload(nviz)
    returned_network = nru.whole_neuron_branch_concept_network(recovered_neuron,
                                      directional=True,
                                     limb_soma_touch_dictionary = "all",
                                     print_flag = False)
    
    nviz.visualize_concept_map(returned_network,
                          #starting_node_size = 10,
                          arrow_color = "green")
    """
    
    if flip_y:
        curr_concept_network = deepcopy(curr_concept_network)
        for k in curr_concept_network.nodes():
            curr_concept_network.nodes[k]["data"].mesh_center[...,1] = -curr_concept_network.nodes[k]["data"].mesh_center[...,1]
    
    if not append_figure:
        ipv.pylab.clear()
        ipv.figure(figsize=(15,15))
    
    node_locations = dict([(k,curr_concept_network.nodes[k]["data"].mesh_center) for k in curr_concept_network.nodes()])
    
    node_edges = np.array(list(curr_concept_network.edges))


    #Adding the arrows for a directional graph
    if type(curr_concept_network) == type(nx.DiGraph()):
        #getting the midpoints then the directions of arrows for the quiver
        midpoints = []
        directions = []
        
        reciprocal_edges = xu.find_reciprocal_connections(curr_concept_network,redundant=True)
        
        for n1,n2 in curr_concept_network.edges:
            #going to skip reciprocal connections because will do them later
            if len(nu.matching_rows_old(reciprocal_edges,[n1,n2])) > 0:
                continue
            difference = node_locations[n2] - node_locations[n1]
            directions.append(difference)
            midpoints.append(node_locations[n1] + difference/2)
        directions = np.array(directions)
        midpoints = np.array(midpoints)

        arrow_rgba = mu.color_to_rgba(arrow_color,arrow_alpha)

        ipv.pylab.quiver(midpoints[:,0],midpoints[:,1],midpoints[:,2],
                        directions[:,0],directions[:,1],directions[:,2],
                        size=arrow_size,
                        color = arrow_rgba)
        
        
        if len(reciprocal_edges) > 0:
            #getting the midpoints then the directions of arrows for the quiver
            midpoints = []
            directions = []

            for n1,n2 in reciprocal_edges:
                #going to skip reciprocal connections because will do them later
                difference = node_locations[n2] - node_locations[n1]
                directions.append(difference)
                midpoints.append(node_locations[n1] + difference/2)
            directions = np.array(directions)
            midpoints = np.array(midpoints)

            arrow_rgba = mu.color_to_rgba(arrow_color_reciprocal,
                                          arrow_alpha_reciprocal)
            
            ipv.pylab.quiver(midpoints[:,0],midpoints[:,1],midpoints[:,2],
                            directions[:,0],directions[:,1],directions[:,2],
                            size=arrow_size_reciprocal,
                            color = arrow_rgba)

            
    if starting_node:
        starting_node_num = xu.get_starting_node(curr_concept_network,only_one=False)
        starting_node_num_coord = [curr_concept_network.nodes[k]["data"].mesh_center for k in starting_node_num]
    
        #print(f"Highlighting starting node {starting_node_num} with coordinate = {starting_node_num_coord}")
        for k in starting_node_num_coord:
#             print(f"mu.color_to_rgba(starting_node_color,starting_node_alpha) = {mu.color_to_rgba(starting_node_color,starting_node_alpha)}")
#             print(f"[k] = {[k]}")
#             print(f"scatter_size = {node_size}")
            sk.graph_skeleton_and_mesh(
                                       other_scatter=[k],
                                       other_scatter_colors=[mu.color_to_rgba(starting_node_color,starting_node_alpha)],
                                       scatter_size=starting_node_size,
                                       show_at_end=False,
                                       append_figure=True,
                                        flip_y=False
                
                                   )
    
    #print("************ Done plotting the starting nodes *******************")
    #plot all of the data points using the colors
    if type(node_color) != dict:
        color_list = mu.process_non_dict_color_input(node_color)
        #now go through and add the alpha levels to those that don't have it
        color_list_alpha_fixed = mu.apply_alpha_to_color_list(color_list,alpha=node_alpha)
        color_list_correct_size = mu.generate_color_list_no_alpha_change(user_colors=color_list_alpha_fixed,
                                                                         n_colors=len(curr_concept_network.nodes()))
        node_locations_array = [v for v in node_locations.values()]
    else:
        #if dictionary then check that all the color dictionary keys match
        node_names = list(curr_concept_network.nodes())
        if set(list(node_color.keys())) != set(node_names):
            raise Exception(f"The node_color dictionary ({node_color}) did not match the nodes in the concept network ({curr_concept_network})")
        
        #assemble the color list and the 
        color_list_correct_size = [node_color[k] for k in node_names]
        node_locations_array = [node_locations[k] for k in node_names]
        
#     print(f"node_locations = {node_locations}")
#     print(f"\n\nnode_locations_array = {node_locations_array}")
    #print("***** About to do all the other scatter points ***********")
    
    #print(f"Current scatter size = {scatter_size}")
    if print_flag:
        print(f"edge_color = {edge_color} IN SKELETON")
    concept_network_skeleton = nru.convert_concept_network_to_skeleton(curr_concept_network)
    
    plot_ipv_skeleton(concept_network_skeleton,edge_color,flip_y=False)
    sk.graph_skeleton_and_mesh(
                              #other_skeletons=[concept_network_skeleton],
                              #other_skeletons_colors=[edge_color],
                               other_scatter=node_locations_array,
                               other_scatter_colors=color_list_correct_size,
                               scatter_size=node_size,
                               show_at_end=False,
                               append_figure=True,
                                flip_y=False
                              )

    if show_at_end:
        ipv.show()
        
        
        
    
    



def plot_branch_pieces(neuron_network,
                       node_to_branch_dict,
                      background_mesh=None,
                      **kwargs):
    if background_mesh is None:
        background_mesh = trimesh.Trimesh(vertices = np.array([]),
                                         faces = np.array([]))
        
    total_branch_meshes = []
    
    for curr_limb,limb_branches in node_to_branch_dict.items():
        meshes_to_plot = [neuron_network.nodes[curr_limb]["data"].concept_network.nodes[k]["data"].mesh for k in limb_branches]
        total_branch_meshes += meshes_to_plot

    if len(total_branch_meshes) == 0:
        print("**** Warning: There were no branch meshes to visualize *******")
        return
    
    sk.graph_skeleton_and_mesh(main_mesh_verts=background_mesh.vertices,
                              main_mesh_faces=background_mesh.faces,
                              other_meshes=total_branch_meshes,
                              other_meshes_colors="red",
                              **kwargs)
    
    
    
######  Don't think need general configurations because would like for mesh, skeleton and concept_network to have different defaults
#     #the general configurations      
#     configuration_dict=None,
#     limb_branch_dict=None
#     resolution=default_resolution,
#     color_grouping="branch",
#     color="random",
#     color_alpha=default_alpha,
#     soma=False,
#     soma_color="red",
#     soma_alpha=default_alpha,
#     whole_neuron=False,
#     whole_neuron_color="grey",
#     whole_neuron_alpha=default_alpha,
    
    
import ipyvolume as ipv

def plot_ipv_mesh(elephant_mesh_sub,color=[1.,0.,0.,0.2],
                 flip_y=True):
    if len(elephant_mesh_sub.vertices) == 0:
        return
    
    if flip_y:
        elephant_mesh_sub = elephant_mesh_sub.copy()
        elephant_mesh_sub.vertices[...,1] = -elephant_mesh_sub.vertices[...,1]
    
    #check if the color is a dictionary
    if type(color) == dict:
        #get the type of values stored in there
        labels = list(color.items())
        
        #if the labels were stored as just numbers/decimals
        if type(labels[0]) == int or type(labels[0]) == float:
            #get all of the possible labels
            unique_labels = np.unique(labels)
            #get random colors for all of the labels
            colors_list =  mu.generate_color_list(n_colors)
            for lab,curr_color in zip(unique_labels,colors_list):
                #find the faces that correspond to that label
                faces_to_keep = [k for k,v in color.items() if v == lab]
                #draw the mesh with that color
                curr_mesh = elephant_mesh_sub.submesh([faces_to_keep],append=True)
                
                mesh4 = ipv.plot_trisurf(elephant_mesh_sub.vertices[:,0],
                               elephant_mesh_sub.vertices[:,1],
                               elephant_mesh_sub.vertices[:,2],
                               triangles=elephant_mesh_sub.faces)
                mesh4.color = curr_color
                mesh4.material.transparent = True
    else:          
        mesh4 = ipv.plot_trisurf(elephant_mesh_sub.vertices[:,0],
                                   elephant_mesh_sub.vertices[:,1],
                                   elephant_mesh_sub.vertices[:,2],
                                   triangles=elephant_mesh_sub.faces)

        mesh4.color = color
        mesh4.material.transparent = True
        
import skeleton_utils as sk
def plot_ipv_skeleton(edge_coordinates,color=[0,0.,1,1],
                     flip_y=True):
    if len(edge_coordinates) == 0:
        print("Edge coordinates in plot_ipv_skeleton were of 0 length so returning")
        return []
    
    if flip_y:
        edge_coordinates = edge_coordinates.copy()
        edge_coordinates[...,1] = -edge_coordinates[...,1] 
    
    unique_skeleton_verts_final,edges_final = sk.convert_skeleton_to_nodes_edges(edge_coordinates)
    mesh2 = ipv.plot_trisurf(unique_skeleton_verts_final[:,0], 
                            unique_skeleton_verts_final[:,1], 
                            unique_skeleton_verts_final[:,2], 
                            lines=edges_final)
    #print(f"color in ipv_skeleton = {color}")
    mesh2.color = color 
    mesh2.material.transparent = True
    
    #print(f"Color in skeleton ipv plot local = {color}")
    
    if flip_y:
        unique_skeleton_verts_final[...,1] = -unique_skeleton_verts_final[...,1]

    return unique_skeleton_verts_final

def plot_ipv_scatter(scatter_points,scatter_color=[1.,0.,0.,0.5],
                    scatter_size=0.4,
                    flip_y=True):
    scatter_points = np.array(scatter_points).reshape(-1,3)
    if flip_y:
        scatter_points = scatter_points.copy()
        scatter_points[...,1] = -scatter_points[...,1]
        
    if len(scatter_points) <= 0:
        print("No scatter points to plot")
        return
    
    mesh_5 = ipv.scatter(
            scatter_points[:,0], 
            scatter_points[:,1],
            scatter_points[:,2], 
            size=scatter_size, 
            color=scatter_color,
            marker="sphere")
    mesh_5.material.transparent = True    

import matplotlib_utils as mu
import numpy_utils as nu
import trimesh_utils as tu
import copy
import itertools
import numpy as np

import sys
current_module = sys.modules[__name__]
from importlib import reload

def visualize_neuron(
    #the neuron we want to visualize
    input_neuron,
    
    #the categories that will be visualized
    visualize_type=["mesh","skeleton"],
    limb_branch_dict=dict(L0=[]),
    #limb_branch_dict=dict(L0=[]),
    
    #for the mesh type:
    mesh_configuration_dict=dict(),
    mesh_limb_branch_dict=None,
    mesh_resolution="branch",
    mesh_color_grouping="branch",
    mesh_color="random",
    mesh_fill_color="brown",
    mesh_color_alpha=0.2,
    mesh_soma=True,
    mesh_soma_color="red",
    mesh_soma_alpha=0.2,
    mesh_whole_neuron=False,
    mesh_whole_neuron_color="green",
    mesh_whole_neuron_alpha=0.2,
    subtract_from_main_mesh=True,
    
    mesh_spines = False,
    mesh_spines_color = "red",
    mesh_spines_alpha = 0.8,
            
    
    #for the skeleton type:
    skeleton_configuration_dict=dict(),
    skeleton_limb_branch_dict=None,
    skeleton_resolution="branch",
    skeleton_color_grouping="branch",
    skeleton_color="random",
    skeleton_color_alpha=1,
    skeleton_soma=True,
    skeleton_fill_color = "green",
    skeleton_soma_color="red",
    skeleton_soma_alpha=1,
    skeleton_whole_neuron=False,
    skeleton_whole_neuron_color="blue",
    skeleton_whole_neuron_alpha=1,
    
    #for concept_network 
    network_configuration_dict=dict(),
    network_limb_branch_dict=None,
    network_resolution="branch",
    network_color_grouping="branch",
    network_color="random",
    network_color_alpha=0.5,
    network_soma=True,
    network_fill_color = "brown",
    network_soma_color="red",
    network_soma_alpha=0.5,
    network_whole_neuron=False,
    network_whole_neuron_color="black",
    network_whole_neuron_alpha=0.5,
    network_whole_neuron_node_size=0.15,
    
    # ------ specific arguments for the concept_network -----
    network_directional=True,
    limb_to_starting_soma="all",
    
    edge_color = "black",
    node_size = 0.15,
    
    starting_node=True,
    starting_node_size=0.3,
    starting_node_color= "pink",
    starting_node_alpha=0.5,
    
    arrow_color = "brown",
    arrow_alpha = 0.8,
    arrow_size = 0.3,
    
    arrow_color_reciprocal = "pink",#"brown",
    arrow_alpha_reciprocal = 1,#0.8,
    arrow_size_reciprocal = 0.7,#0.3,
    
    # arguments for plotting other meshes associated with neuron #
    
    inside_pieces = False,
    inside_pieces_color = "red",
    inside_pieces_alpha = 1,
    
    insignificant_limbs = False,
    insignificant_limbs_color = "red",
    insignificant_limbs_alpha = 1,
    
    non_soma_touching_meshes = False, #whether to graph the inside pieces
    non_soma_touching_meshes_color = "red",
    non_soma_touching_meshes_alpha = 1,
    
    
    
    # arguments for how to display/save ipyvolume fig
    buffer=1000,
    axis_box_off=True,
    html_path="",
    show_at_end=True,
    append_figure=False,
    
    # arguments that will help with random colorization:
    colors_to_omit= [],
    
    #whether to return the color dictionary in order to help
    #locate certain colors
    return_color_dict = False, #if return this then can use directly with plot_color_dict to visualize the colors of certain branches
    
    
    print_flag = False,
    print_time = False,
    flip_y=True,
    
    #arguments for scatter
    scatters=[],
    scatters_colors=[],
    scatter_size=0.3,
    main_scatter_color = "red",
    
    soma_border_vertices = False,
    soma_border_vertices_color="random",
    
    verbose=True,
    subtract_glia = True
    
    ):
    
    """
    ** tried to optimize for speed but did not find anything that really sped it up**
    ipv.serialize.performance = 0/1/2 was the only thing I really found but this didn't help
    (most of the time is spent on compiling the visualization and not on the python,
    can see this by turning on print_time=True, which only shows about 2 seconds for runtime
    but is really 45 seconds for large mesh)
    
    How to plot the spines:
    nviz.visualize_neuron(uncompressed_neuron,
                      limb_branch_dict = dict(),
                     #mesh_spines=True,
                      mesh_whole_neuron=True,
                      mesh_whole_neuron_alpha = 0.1,
                      
                    mesh_spines = True,
                    mesh_spines_color = "red",
                    mesh_spines_alpha = 0.8,
                      
                     )
    Examples: 
    How to do a concept_network graphing: 
    nviz=reload(nviz)
    returned_color_dict = nviz.visualize_neuron(uncompressed_neuron,
                                                visualize_type=["network"],
                                                network_resolution="branch",
                                                network_whole_neuron=True,
                                                network_whole_neuron_node_size=1,
                                                network_whole_neuron_alpha=0.2,
                                                network_directional=True,

                                                #network_soma=["S1","S0"],
                                                #network_soma_color = ["black","red"],       
                                                limb_branch_dict=dict(L1=[11,15]),
                                                network_color=["pink","green"],
                                                network_color_alpha=1,
                                                node_size = 5,
                                                arrow_size = 1,
                                                return_color_dict=True)
    
    
    
    Cool facts: 
    1) Can specify the soma names and not just say true so will
    only do certain somas
    
    Ex: 
    returned_color_dict = nviz.visualize_neuron(uncompressed_neuron,
                     visualize_type=["network"],
                     network_resolution="limb",
                                            network_soma=["S0"],
                    network_soma_color = ["red","black"],       
                     limb_branch_dict=dict(L1=[],L2=[]),
                     node_size = 5,
                     return_color_dict=True)
    
    2) Can put "all" for limb_branch_dict or can put "all"
    for the lists of each branch
    
    3) Can specify the somas you want to graph and their colors
    by sending lists
    
    
    Ex 3: How to specifically color just one branch and fill color the rest of limb
    limb_idx = "L0"
    ex_limb = uncompressed_neuron.concept_network.nodes[limb_idx]["data"]
    branch_idx = 3
    ex_branch = ex_limb.concept_network.nodes[2]["data"]

    nviz.visualize_neuron(double_neuron_processed,
                          visualize_type=["mesh"],
                         limb_branch_dict=dict(L0="all"),
                          mesh_color=dict(L1={3:"red"}),
                          mesh_fill_color="green"

                         )
    
    
    """
    import neuron_visualizations as nviz
    nviz = reload(nviz)
    
    total_time = time.time()
    #print(f"print_time = {print_time}")
    import ipyvolume as ipv
    
    current_neuron = copy.deepcopy(input_neuron)
    
    local_time = time.time()
    #To uncomment for full graphing
    if not append_figure:
        ipv.pylab.clear()
        ipv.figure(figsize=(15,15))

    if print_time:
        print(f"Time for setting up figure = {time.time() - local_time}")
        local_time = time.time()
        
    main_vertices = []
    
    #do the mesh visualization type
    for viz_type in visualize_type:
        local_time = time.time()
        if verbose:
            print(f"\n Working on visualization type: {viz_type}")
        if viz_type=="mesh":
            current_type = "mesh"
            
            
            #configuring the parameters
            configuration_dict = mesh_configuration_dict
            configuration_dict.setdefault("limb_branch_dict",mesh_limb_branch_dict)
            configuration_dict.setdefault("resolution",mesh_resolution)
            configuration_dict.setdefault("color_grouping",mesh_color_grouping)
            configuration_dict.setdefault("color",mesh_color)
            configuration_dict.setdefault("fill_color",mesh_fill_color)
            configuration_dict.setdefault("color_alpha",mesh_color_alpha)
            configuration_dict.setdefault("soma",mesh_soma)
            configuration_dict.setdefault("soma_color",mesh_soma_color)
            configuration_dict.setdefault("soma_alpha",mesh_soma_alpha)
            configuration_dict.setdefault("whole_neuron",mesh_whole_neuron)
            configuration_dict.setdefault("whole_neuron_color",mesh_whole_neuron_color)
            configuration_dict.setdefault("whole_neuron_alpha",mesh_whole_neuron_alpha)
            
            configuration_dict.setdefault("mesh_spines",mesh_spines)
            configuration_dict.setdefault("mesh_spines_color",mesh_spines_color)
            configuration_dict.setdefault("mesh_spines_alpha",mesh_spines_alpha)
            
        elif viz_type == "skeleton":
            current_type="skeleton"
            
            #configuring the parameters
            configuration_dict = skeleton_configuration_dict
            configuration_dict.setdefault("limb_branch_dict",skeleton_limb_branch_dict)
            configuration_dict.setdefault("resolution",skeleton_resolution)
            configuration_dict.setdefault("color_grouping",skeleton_color_grouping)
            configuration_dict.setdefault("color",skeleton_color)
            configuration_dict.setdefault("fill_color",skeleton_fill_color)
            configuration_dict.setdefault("color_alpha",skeleton_color_alpha)
            configuration_dict.setdefault("soma",skeleton_soma)
            configuration_dict.setdefault("soma_color",skeleton_soma_color)
            configuration_dict.setdefault("soma_alpha",skeleton_soma_alpha)
            configuration_dict.setdefault("whole_neuron",skeleton_whole_neuron)
            configuration_dict.setdefault("whole_neuron_color",skeleton_whole_neuron_color)
            configuration_dict.setdefault("whole_neuron_alpha",skeleton_whole_neuron_alpha)
            
        elif viz_type == "network":
            current_type="mesh_center"
            
            #configuring the parameters
            configuration_dict = network_configuration_dict
            configuration_dict.setdefault("limb_branch_dict",network_limb_branch_dict)
            configuration_dict.setdefault("resolution",network_resolution)
            configuration_dict.setdefault("color_grouping",network_color_grouping)
            configuration_dict.setdefault("color",network_color)
            configuration_dict.setdefault("fill_color",network_fill_color)
            configuration_dict.setdefault("color_alpha",network_color_alpha)
            configuration_dict.setdefault("soma",network_soma)
            configuration_dict.setdefault("soma_color",network_soma_color)
            configuration_dict.setdefault("soma_alpha",network_soma_alpha)
            configuration_dict.setdefault("whole_neuron",network_whole_neuron)
            configuration_dict.setdefault("whole_neuron_color",network_whole_neuron_color)
            configuration_dict.setdefault("whole_neuron_alpha",network_whole_neuron_alpha)
            configuration_dict.setdefault("whole_neuron_node_size",network_whole_neuron_node_size)
            
            # ------ specific arguments for the concept_network -----
            configuration_dict.setdefault("network_directional",network_directional)
            configuration_dict.setdefault("limb_to_starting_soma",limb_to_starting_soma)
            
            configuration_dict.setdefault("node_size",node_size)
            configuration_dict.setdefault("edge_color",edge_color)
            
            
            configuration_dict.setdefault("starting_node",starting_node)
            configuration_dict.setdefault("starting_node_size",starting_node_size)
            configuration_dict.setdefault("starting_node_color",starting_node_color)
            configuration_dict.setdefault("starting_node_alpha",starting_node_alpha)
            
            configuration_dict.setdefault("arrow_color",arrow_color)
            configuration_dict.setdefault("arrow_alpha",arrow_alpha)
            configuration_dict.setdefault("arrow_size",arrow_size)
            
            configuration_dict.setdefault("arrow_color_reciprocal",arrow_color_reciprocal)
            configuration_dict.setdefault("arrow_alpha_reciprocal",arrow_alpha_reciprocal)
            configuration_dict.setdefault("arrow_size_reciprocal",arrow_size_reciprocal)
            
            
            
        else:
            raise Exception(f"Recieved invalid visualization type: {viz_type}")
        
        
        #sets the limb branch dict specially  (uses overall one if none assigned)

        #print(f"current_type = {current_type}")
        
        #handle if the limb_branch_dict is "all"
        if configuration_dict["limb_branch_dict"] is None:
            #print("limb_branch_dict was None")
            configuration_dict["limb_branch_dict"] = limb_branch_dict
        
        if configuration_dict["limb_branch_dict"] == "all":
            configuration_dict["limb_branch_dict"] = dict([(k,"all") for k in current_neuron.get_limb_node_names()])
            
        
        if print_time:
            print(f"Extracting Dictionary = {time.time() - local_time}")
            local_time = time.time()
        
        #------------------------- Done with collecting the parameters ------------------------
        
        if print_flag:
            for k,v in configuration_dict.items():
                print(k,v)
        
        #get the list of items specific
        limbs_to_plot = sorted(list(configuration_dict["limb_branch_dict"].keys()))
        plot_items = []
        plot_items_order = []
        if configuration_dict["resolution"] == "limb":
            
            plot_items = [getattr(current_neuron.concept_network.nodes[li]["data"],current_type) for li in limbs_to_plot]
            plot_items_order = [[li] for li in limbs_to_plot]
        elif configuration_dict["resolution"] == "branch":
            
            for li in limbs_to_plot:
                curr_limb_obj = current_neuron.concept_network.nodes[li]["data"]
                #handle if "all" is the key
                if ((configuration_dict["limb_branch_dict"][li] == "all") or 
                   ("all" in configuration_dict["limb_branch_dict"][li])):
                    #gather all of the branches: 
                    plot_items += [getattr(curr_limb_obj.concept_network.nodes[k]["data"],current_type) for k in sorted(curr_limb_obj.concept_network.nodes())]
                    plot_items_order += [[li,k] for k in sorted(curr_limb_obj.concept_network.nodes())]
                else:
                    for branch_idx in sorted(configuration_dict["limb_branch_dict"][li]):
                        plot_items.append(getattr(curr_limb_obj.concept_network.nodes[branch_idx]["data"],current_type))
                        plot_items_order.append([li,branch_idx])
        else:
            raise Exception("The resolution specified was neither branch nore limb")
            
        
        #getting the min and max of the plot items to set the zoom later (could be empty)
        
        
        if print_time:
            print(f"Creating Plot Items = {time.time() - local_time}")
            local_time = time.time()
        
        
#         print(f"plot_items_order= {plot_items_order}")
#         print(f"plot_items= {plot_items}")
        
     
        # Now need to build the colors dictionary
        """
        Pseudocode:
        if color is a dictionary then that is perfect and what we want:
        
        -if color grouping or resolution at limb then this dictionary should be limb --> color
        -if resolution adn color grouping aat branch should be limb --> branch --> color
        
        if not then generate a dictionary like that where 
        a) if color is random: generate list of random colors for length needed and then store in dict
        b) if given one color or list of colors:
        - make sure it is a list of colors
        - convert all of the strings into rgb colors
        - for all the colors in the list that do not have an alpha value fill it in with the default alpha
        - repeat the list enough times to give every item a color
        - assign the colors to the limb or limb--> branch dictionary

        """
        
        #need to do preprocessing of colors if not a dictionary
        if type(configuration_dict["color"]) != dict:
            color_list = mu.process_non_dict_color_input(configuration_dict["color"])
        else:
            #if there was a dictionary given then compile a color list and fill everything not specified with mesh_fill_color
            color_list = []
            for dict_keys in plot_items_order:
                #print(f"dict_keys = {dict_keys}")
                first_key = dict_keys[0]
                if first_key not in configuration_dict["color"]:
                    color_list.append(mu.color_to_rgb(configuration_dict["fill_color"]))
                    continue
                if len(dict_keys) == 1:
                    color_list.append(mu.color_to_rgb(configuration_dict["color"][first_key]))
                elif len(dict_keys) == 2:
                    second_key = dict_keys[1]
                    if second_key not in configuration_dict["color"][first_key]:
                        color_list.append(mu.color_to_rgb(configuration_dict["fill_color"]))
                        continue
                    else:
                        color_list.append(mu.color_to_rgb(configuration_dict["color"][first_key][second_key]))
                else:
                    raise Exception(f"plot_items_order item is greater than size 2: {dict_keys}")
        
        if print_flag:
            print(f"color_list = {color_list}")
            

            
        #now go through and add the alpha levels to those that don't have it
        color_list_alpha_fixed = mu.apply_alpha_to_color_list(color_list,alpha=configuration_dict["color_alpha"])
        
        color_list_correct_size = mu.generate_color_list_no_alpha_change(user_colors=color_list_alpha_fixed,
                                                                         n_colors=len(plot_items),
                                                                        colors_to_omit=colors_to_omit)
        
        if print_flag:
            print(f"color_list_correct_size = {color_list_correct_size}")
            print(f"plot_items = {plot_items}")
            print(f"plot_items_order = {plot_items_order}")
            
            
        if print_time:
            print(f"Creating Colors list = {time.time() - local_time}")
            local_time = time.time()
        #------at this point have a list of colors for all the things to plot -------
        
        
        
        #4) If soma is requested then get the some items
        
        
        soma_names = current_neuron.get_soma_node_names()
        if nu.is_array_like(configuration_dict["soma"]):
            soma_names = [k for k in soma_names if k in configuration_dict["soma"]]
            
        if viz_type == "mesh":
            local_time = time.time()
            #add the vertices to plot to main_vertices list
            if len(plot_items)>0:
                min_max_vertices = np.array([[np.min(k.vertices,axis=0),np.max(k.vertices,axis=0)] for k in  plot_items]).reshape(-1,3)
                min_vertices = np.min(min_max_vertices,axis=0)
                max_vertices = np.max(min_max_vertices,axis=0)
                main_vertices.append(np.array([min_vertices,max_vertices]).reshape(-1,3))
                
            if print_time:
                print(f"Collecting vertices for mesh = {time.time() - local_time}")
                local_time = time.time()
                
            
            #Can plot the meshes now
            for curr_mesh,curr_mesh_color in zip(plot_items,color_list_correct_size):
                plot_ipv_mesh(curr_mesh,color=curr_mesh_color,flip_y=flip_y)
            
            if print_time:
                print(f"Plotting mesh pieces= {time.time() - local_time}")
                local_time = time.time()
                
            #Plot the soma if asked for it
            if configuration_dict["soma"]:
                """
                Pseudocode: 
                1) Get the soma meshes
                2) for the color specified: 
                - if string --> convert to rgba
                - if numpy array --> 
                
                configuration_dict.setdefault("soma",mesh_soma)
                configuration_dict.setdefault("soma_color",mesh_soma_color)
                configuration_dict.setdefault("soma_alpha",mesh_soma_alpha)

                """
                
                soma_meshes = [current_neuron.concept_network.nodes[k]["data"].mesh for k in soma_names]
                
                soma_colors_list = mu.process_non_dict_color_input(configuration_dict["soma_color"])
                soma_colors_list_alpha = mu.apply_alpha_to_color_list(soma_colors_list,alpha=configuration_dict["soma_alpha"])
                soma_colors_list_alpha_fixed_size = mu.generate_color_list_no_alpha_change(soma_colors_list_alpha,
                                                                                          n_colors=len(soma_meshes))
                soma_names,soma_colors_list_alpha_fixed_size
                for curr_soma_mesh,curr_soma_color in zip(soma_meshes,soma_colors_list_alpha_fixed_size):
                    plot_ipv_mesh(curr_soma_mesh,color=curr_soma_color,flip_y=flip_y)
                    main_vertices.append(curr_soma_mesh.vertices)
                
                if print_time:
                    print(f"plotting mesh somas= {time.time() - local_time}")
                    local_time = time.time()
                    
            #will add the background mesh if requested
            if configuration_dict["whole_neuron"]:
                whole_neuron_colors_list = mu.process_non_dict_color_input(configuration_dict["whole_neuron_color"])
                whole_neuron_colors_list_alpha = mu.apply_alpha_to_color_list(whole_neuron_colors_list,alpha=configuration_dict["whole_neuron_alpha"])
                
                if subtract_glia:
                    if (current_neuron.glia_faces is not None) and (len(current_neuron.glia_faces) > 0):
                        whole_mesh = current_neuron.mesh.submesh([np.delete(np.arange(len(current_neuron.mesh.faces)),
                                                                                 current_neuron.glia_faces)],append=True,repair=False)
                    else:
                        whole_mesh = current_neuron.mesh
                else:
                    whole_mesh = current_neuron.mesh
                    
                    
                # Will do the erroring of the mesh
                if (subtract_from_main_mesh and (len(plot_items)>0)):
                    main_mesh_to_plot = tu.subtract_mesh(original_mesh=whole_mesh,
                                              subtract_mesh=plot_items)
                else:
                    main_mesh_to_plot = whole_mesh
                    
                
        
                
                # will do the plotting
                plot_ipv_mesh(main_mesh_to_plot,color=whole_neuron_colors_list_alpha[0],flip_y=flip_y)
                main_vertices.append([np.min(main_mesh_to_plot.vertices,axis=0),
                                      np.max(main_mesh_to_plot.vertices,axis=0)])
                
                if print_time:
                    print(f"Plotting mesh whole neuron = {time.time() - local_time}")
                    local_time = time.time()
                
            
            #plotting the spines
            if configuration_dict["mesh_spines"]:
                #plotting the spines
                spine_meshes = []
                
                for limb_names in current_neuron.get_limb_node_names():
                    #iterate through all of the branches
                    curr_limb_obj = current_neuron.concept_network.nodes[limb_names]["data"]
                    for branch_name in curr_limb_obj.concept_network.nodes():
                        curr_spines = curr_limb_obj.concept_network.nodes[branch_name]["data"].spines
                        if not curr_spines is None:
                            spine_meshes += curr_spines
                
                spines_color_list = mu.process_non_dict_color_input(configuration_dict["mesh_spines_color"])
                
                
                
                spines_color_list_alpha = mu.apply_alpha_to_color_list(spines_color_list,alpha=configuration_dict["mesh_spines_alpha"])
                #print(f"spines_color_list_alpha = {spines_color_list_alpha}")
                if len(spines_color_list_alpha) == 1:
                    #concatenate the meshes
                    #print("Inside spine meshes combined")
                    combined_spine_meshes = tu.combine_meshes(spine_meshes)
                    plot_ipv_mesh(combined_spine_meshes,color=spines_color_list_alpha[0],flip_y=flip_y)
                    main_vertices.append(combined_spine_meshes.vertices)
                    
                else:
                    spines_colors_list_alpha_fixed_size = mu.generate_color_list_no_alpha_change(spines_color_list_alpha,
                                                                                              n_colors=len(spine_meshes))


                    for curr_spine_mesh,curr_spine_color in zip(spine_meshes,spines_colors_list_alpha_fixed_size):
                        plot_ipv_mesh(curr_spine_mesh,color=curr_spine_color,flip_y=flip_y)
                        main_vertices.append(curr_spine_mesh.vertices)
                if print_time:
                    print(f"Plotting mesh spines= {time.time() - local_time}")
                    local_time = time.time()
                
            
        elif viz_type == "skeleton":
            local_time = time.time()
            #add the vertices to plot to main_vertices list
            if len(plot_items)>0:
                reshaped_items = np.concatenate(plot_items).reshape(-1,3)
                min_vertices = np.min(reshaped_items,axis=0)
                max_vertices = np.max(reshaped_items,axis=0)
                main_vertices.append(np.array([min_vertices,max_vertices]).reshape(-1,3))
                
            if print_time:
                print(f"Gathering vertices for skeleton= {time.time() - local_time}")
                local_time = time.time()
            
            #Can plot the meshes now
            for curr_skeleton,curr_skeleton_color in zip(plot_items,color_list_correct_size):
                plot_ipv_skeleton(curr_skeleton,color=curr_skeleton_color,flip_y=flip_y)
            
            if print_time:
                print(f"Plotting skeleton pieces = {time.time() - local_time}")
                local_time = time.time()
            
            
            if configuration_dict["soma"]:
                    
                soma_colors_list = mu.process_non_dict_color_input(configuration_dict["soma_color"])
                soma_colors_list_alpha = mu.apply_alpha_to_color_list(soma_colors_list,alpha=configuration_dict["soma_alpha"])
                soma_colors_list_alpha_fixed_size = mu.generate_color_list_no_alpha_change(soma_colors_list_alpha,
                                                                                          n_colors=len(soma_names))
                #get the somas associated with the neurons
                soma_skeletons = [nru.get_soma_skeleton(current_neuron,k) for k in soma_names]
                
                for curr_soma_sk,curr_soma_sk_color in zip(soma_skeletons,soma_colors_list_alpha_fixed_size):
                    sk_vertices = plot_ipv_skeleton(curr_soma_sk,color=curr_soma_sk_color,flip_y=flip_y)
                    main_vertices.append(sk_vertices) #adding the vertices
                
                if print_time:
                    print(f"Plotting skeleton somas = {time.time() - local_time}")
                    local_time = time.time()
            
            if configuration_dict["whole_neuron"]:
                whole_neuron_colors_list = mu.process_non_dict_color_input(configuration_dict["whole_neuron_color"])
                whole_neuron_colors_list_alpha = mu.apply_alpha_to_color_list(whole_neuron_colors_list,alpha=configuration_dict["whole_neuron_alpha"])
                
                #graph
                sk_vertices = plot_ipv_skeleton(current_neuron.skeleton,color=whole_neuron_colors_list_alpha[0],flip_y=flip_y)
                main_vertices.append([np.min(sk_vertices,axis=0),
                                      np.max(sk_vertices,axis=0)])
                
                if print_time:
                    print(f"Plotting skeleton whole neuron = {time.time() - local_time}")
                    local_time = time.time()
                
                
        elif viz_type == "network":
            local_time = time.time()
            """
            Pseudocode: 
            0) get the mesh_centers of all of the nodes in the concept_network sent and add to the main vertices
            1) get the current concept network (limb or branch) based on the resolution
            - if branch level then use the function that assembles
            2) get a list of all the nodes in the plot_items_order and assemble into a dictionary (have to fix the name)
            3) For all the somas to be added, add them to the dictionary of label to color (and add vertices to main vertices)
            4) Use that dictionary to send to the visualize_concept_map function and call the function
            with all the other parameters in the configuration dict
            
            5) get the mesh_centers of all of the nodes in the concept_network sent and add to the main vertices
            
            """
            
            
            #0) get the mesh_centers of all of the nodes in the concept_network sent and add to the main vertices
            if len(plot_items)>0:
                reshaped_items = np.concatenate(plot_items).reshape(-1,3)
                min_vertices = np.min(reshaped_items,axis=0)
                max_vertices = np.max(reshaped_items,axis=0)
                main_vertices.append(np.array([min_vertices,max_vertices]).reshape(-1,3))
                
            if print_time:
                print(f"Gathering vertices for network = {time.time() - local_time}")
                local_time = time.time()
                
            
            #1) get the current concept network (limb or branch) based on the resolution
            #- if branch level then use the function that assembles
            if configuration_dict["resolution"] == "branch":

                curr_concept_network = nru.whole_neuron_branch_concept_network(current_neuron,
                                                          directional= configuration_dict["network_directional"],
                                                         limb_soma_touch_dictionary = configuration_dict["limb_to_starting_soma"],
                                                         print_flag = False)

                    
                
                
                #2) get a list of all the nodes in the plot_items_order and assemble into a dictionary for colors (have to fix the name)
                item_to_color_dict = dict([(f"{name[0]}_{name[1]}",col) for name,col in zip(plot_items_order,color_list_correct_size)])
            else:
                #2) get a list of all the nodes in the plot_items_order and assemble into a dictionary for colors (have to fix the name)
                curr_concept_network = current_neuron.concept_network
                item_to_color_dict = dict([(f"{name[0]}",col) for name,col in zip(plot_items_order,color_list_correct_size)])
                
            if print_time:
                print(f"Getting whole concept network and colors = {time.time() - local_time}")
                local_time = time.time()
            
            
            #3) For all the somas to be added, add them to the dictionary of label to color
            if soma_names:
                soma_colors_list = mu.process_non_dict_color_input(configuration_dict["soma_color"])
                soma_colors_list_alpha = mu.apply_alpha_to_color_list(soma_colors_list,alpha=configuration_dict["soma_alpha"])
                soma_colors_list_alpha_fixed_size = mu.generate_color_list_no_alpha_change(soma_colors_list_alpha,
                                                                                          n_colors=len(soma_names))
                
                for s_name,s_color in zip(soma_names,soma_colors_list_alpha_fixed_size):
                    item_to_color_dict[s_name] = s_color
                    main_vertices.append(current_neuron.concept_network.nodes[s_name]["data"].mesh_center)
                    
                if print_time:
                    print(f"Adding soma items to network plotting = {time.time() - local_time}")
                    local_time = time.time()
                    
            #print(f"plot_items_order = {plot_items_order}")
            #print(f"item_to_color_dict = {item_to_color_dict}")
                
            curr_concept_network_subgraph = nx.subgraph(curr_concept_network,list(item_to_color_dict.keys()))
            
            
            
            if print_time:
                print(f"Getting Subgraph of concept network = {time.time() - local_time}")
                local_time = time.time()
            
            #4) Use that dictionary to send to the visualize_concept_map function and call the function
            #with all the other parameters in the configuration dict
            
            visualize_concept_map(curr_concept_network_subgraph,
                            node_color=item_to_color_dict,
                            edge_color=configuration_dict["edge_color"],
                            node_size=configuration_dict["node_size"],

                            starting_node=configuration_dict["starting_node"],
                            starting_node_size = configuration_dict["starting_node_size"],
                            starting_node_color= configuration_dict["starting_node_color"],
                            starting_node_alpha = configuration_dict["starting_node_alpha"],

                            arrow_color = configuration_dict["arrow_color"] ,
                            arrow_alpha = configuration_dict["arrow_alpha"],
                            arrow_size = configuration_dict["arrow_size"],

                            arrow_color_reciprocal = configuration_dict["arrow_color_reciprocal"] ,
                            arrow_alpha_reciprocal = configuration_dict["arrow_alpha_reciprocal"],
                            arrow_size_reciprocal = configuration_dict["arrow_size_reciprocal"],
                          
                            show_at_end=False,
                            append_figure=True)
            
            if print_time:
                print(f"Graphing concept network pieces = {time.time() - local_time}")
                local_time = time.time()
            
            # plot the entire thing if asked for it
            if configuration_dict["whole_neuron"]:
                #compute the new color
                whole_neuron_network_color = mu.color_to_rgba(configuration_dict["whole_neuron_color"],
                                                             configuration_dict["whole_neuron_alpha"])
                
                whole_neuron_network_edge_color = mu.color_to_rgba(configuration_dict["edge_color"],
                                                             configuration_dict["whole_neuron_alpha"])
                print(f"whole_neuron_network_edge_color = {whole_neuron_network_edge_color}")
                
                visualize_concept_map(curr_concept_network,
                            node_color=whole_neuron_network_color,
                            edge_color=whole_neuron_network_edge_color,
                            node_size=configuration_dict["whole_neuron_node_size"],

                            starting_node=configuration_dict["starting_node"],
                            starting_node_size = configuration_dict["starting_node_size"],
                            starting_node_color= configuration_dict["starting_node_color"],
                            starting_node_alpha = configuration_dict["starting_node_alpha"],

                            arrow_color = configuration_dict["arrow_color"] ,
                            arrow_alpha = configuration_dict["arrow_alpha"],
                            arrow_size = configuration_dict["arrow_size"],

                            arrow_color_reciprocal = configuration_dict["arrow_color_reciprocal"] ,
                            arrow_alpha_reciprocal = configuration_dict["arrow_alpha_reciprocal"],
                            arrow_size_reciprocal = configuration_dict["arrow_size_reciprocal"],
                          
                            show_at_end=False,
                            append_figure=True)
                if print_time:
                    print(f"Graphing whole neuron concept network = {time.time() - local_time}")
                    local_time = time.time()
            
        else:
            raise Exception("Invalid viz_type")
        
        
    # -------------- plotting the insignificant meshes, floating meshes and non-significant limbs ----- #
    """
    Pseudocode: for [inside_piece,insignificant_limbs,non_soma_touching_meshes]
    
    1) get whether the argument was True/False or a list
    2) If True or list, assemble the color
    3) for each mesh plot it with the color

    """
    local_time = time.time()
    
    other_mesh_dict = dict(
        inside_pieces=inside_pieces,
        inside_pieces_color=inside_pieces_color,
        inside_pieces_alpha=inside_pieces_alpha,
        
        insignificant_limbs=insignificant_limbs,
        insignificant_limbs_color=insignificant_limbs_color,
        insignificant_limbs_alpha=insignificant_limbs_alpha,
        
        non_soma_touching_meshes=non_soma_touching_meshes,
        non_soma_touching_meshes_color=non_soma_touching_meshes_color,
        non_soma_touching_meshes_alpha=non_soma_touching_meshes_alpha
    
    
    )

    
    other_mesh_types = ["inside_pieces","insignificant_limbs","non_soma_touching_meshes"]
    
    for m_type in other_mesh_types:
        if other_mesh_dict[m_type]:
            if type(other_mesh_dict[m_type]) is bool:
                current_mesh_list = getattr(current_neuron,m_type)
            elif "all" in other_mesh_dict[m_type]:
                current_mesh_list = getattr(current_neuron,m_type)
            else:
                total_mesh_list = getattr(current_neuron,m_type)
                current_mesh_list = [k for i,k in enumerate(total_mesh_list) if i in other_mesh_dict[m_type]]
                
            #get the color
            curr_mesh_colors_list = mu.process_non_dict_color_input(other_mesh_dict[m_type + "_color"])
            curr_mesh_colors_list_alpha = mu.apply_alpha_to_color_list(curr_mesh_colors_list,alpha=other_mesh_dict[m_type + "_alpha"])

            #graph
            for curr_mesh in current_mesh_list:
                plot_ipv_mesh(curr_mesh,color=curr_mesh_colors_list_alpha,flip_y=flip_y)
                main_vertices.append(curr_mesh.vertices)
        if print_time:
            print(f"Plotting mesh pieces of {m_type} = {time.time() - local_time}")
            local_time = time.time()
    
    # ----- doing any extra scatter plotting you may need ---- #
    """
    scatters=[],
    scatters_colors=[],
    scatter_size=0.3,
    main_scatter_color="red"
    
    soma_border_vertices
    soma_border_vertices_color
    """
    
    if soma_border_vertices:
        if len(plot_items_order) > 0:
            if verbose:
                print("working on soma border vertices")
            unique_limb_names = np.unique([k[0] for k in plot_items_order])
            all_soma_verts = [[k["touching_soma_vertices"] for k in 
                                        input_neuron[curr_limb_idx].all_concept_network_data] for curr_limb_idx in unique_limb_names]

            
            new_borders = list(itertools.chain.from_iterable(all_soma_verts))
            if soma_border_vertices_color != "random":
                new_borders_colors = [soma_border_vertices_color]*len(new_borders)
            else:
                new_borders_colors = mu.generate_color_list(n_colors=len(new_borders),alpha_level=1)
            
            for curr_scatter,curr_color in zip(new_borders,new_borders_colors):
                plot_ipv_scatter(curr_scatter,scatter_color=curr_color,
                            scatter_size=scatter_size,flip_y=flip_y)
                main_vertices.append(curr_scatter)

    if type(scatters_colors) == str:
        scatters_colors = [scatters_colors]
    
    if len(scatters) > 0 and len(scatters_colors) == 0:
        scatters_colors = [main_scatter_color]*len(scatters)
    
    for curr_scatter,curr_color in zip(scatters,scatters_colors):
        
        plot_ipv_scatter(curr_scatter,scatter_color=curr_color,
                    scatter_size=scatter_size,flip_y=flip_y)
        main_vertices.append(curr_scatter)        
        
        
    #To uncomment for full graphing
    
    #create the main mesh vertices for setting the bounding box
    if len(main_vertices) == 0:
        raise Exception("No vertices plotted in the entire function")
    elif len(main_vertices) == 1:
        main_vertices = main_vertices[0]
    else:
        #get rid of all empty ones
        main_vertices = np.vstack([np.array(k).reshape(-1,3) for k in main_vertices if len(k)>0])
    
    if len(main_vertices) == 0:
        raise Exception("No vertices plotted in the entire function (after took out empty vertices)")
    
    main_vertices = np.array(main_vertices).reshape(-1,3)
    
    if flip_y:
        main_vertices = main_vertices.copy()
        main_vertices[...,1] = -main_vertices[...,1]
        
    volume_max = np.max(main_vertices.reshape(-1,3),axis=0)
    volume_min = np.min(main_vertices.reshape(-1,3),axis=0)
    
    if print_time:
        print(f"Getting volume min/max = {time.time() - local_time}")
        local_time = time.time()
        
    #setting the min/max of the plots
    ranges = volume_max - volume_min
    index = [0,1,2]
    max_index = np.argmax(ranges)
    min_limits = [0,0,0]
    max_limits = [0,0,0]


    for i in index:
        if i == max_index:
            min_limits[i] = volume_min[i] - buffer
            max_limits[i] = volume_max[i] + buffer 
            continue
        else:
            difference = ranges[max_index] - ranges[i]
            min_limits[i] = volume_min[i] - difference/2  - buffer
            max_limits[i] = volume_max[i] + difference/2 + buffer
    
    if print_time:
        print(f"calculating max limits = {time.time() - local_time}")
        local_time = time.time()
    
    ipv.xlim(min_limits[0],max_limits[0])
    ipv.ylim(min_limits[1],max_limits[1])
    ipv.zlim(min_limits[2],max_limits[2])
    
    if print_time:
        print(f"setting ipyvolume max limits = {time.time() - local_time}")
        local_time = time.time()
    
    ipv.style.set_style_light()
    if axis_box_off:
        ipv.style.axes_off()
        ipv.style.box_off()
    else:
        ipv.style.axes_on()
        ipv.style.box_on()
    
    if print_time:
        print(f"Setting axis and box on/off = {time.time() - local_time}")
        local_time = time.time()
    
    if show_at_end:
        ipv.show()
    
    if print_time:
        print(f"ipv.show= {time.time() - local_time}")
        local_time = time.time()
    
    if html_path != "":
        ipv.pylab.save(html_path)
    
    if print_time:
        print(f"saving html = {time.time() - local_time}")
        local_time = time.time()
    
        


    
    if return_color_dict:
        #build the color dictionary
        if len(plot_items_order) == 0 or len(color_list_correct_size)==0:
            print("No color dictionary to return because plot_items_order or color_list_correct_size empty")
            if print_time:
                print(f"Total time for run = {time.time() - total_time}")
            return dict()
        
        if len(plot_items_order[0]) == 1:
            color_dict_to_return = dict([(k[0],v) for k,v in zip(plot_items_order,color_list_correct_size)])
        elif len(plot_items_order[0]) == 2:
            color_dict_to_return = dict([(f"{k[0]}_{k[1]}",v) for k,v in zip(plot_items_order,color_list_correct_size)])
        else:
            raise Exception("Length of first element in plot_items order is greater than 2 elements")
        
        #whether to add soma mappings to the list of colors:
        #soma_names, soma_colors_list_alpha_fixed_size
        try: 
            color_dict_to_return_soma = dict([(k,v) for k,v in zip(soma_names,soma_colors_list_alpha_fixed_size)])
            color_dict_to_return.update(color_dict_to_return_soma)
        except:
            pass
        
        if print_time:
            print(f"Preparing color dictionary = {time.time() - local_time}")
            local_time = time.time()
        if print_time:
            print(f"Total time for run = {time.time() - total_time}")
        
        return color_dict_to_return
        
        
    
    if print_time:
        print(f"Total time for run = {time.time() - total_time}")
    return


def plot_spines(current_neuron,flip_y=True):
    visualize_neuron(current_neuron,
                          limb_branch_dict = dict(),
                          mesh_whole_neuron=True,
                          mesh_whole_neuron_alpha = 0.1,

                        mesh_spines = True,
                        mesh_spines_color = "red",
                        mesh_spines_alpha = 0.8,
                     flip_y=flip_y

                         )
# -------  9/24: Wrapper for the sk.graph function that is nicer to interface with ----#
"""
def graph_skeleton_and_mesh(main_mesh_verts=[],
                            main_mesh_faces=[],
                            unique_skeleton_verts_final=[],
                            edges_final=[],
                            edge_coordinates=[],
                            other_meshes=[],
                            other_meshes_colors =  [],
                            mesh_alpha=0.2,
                            other_meshes_face_components = [],
                            other_skeletons = [],
                            other_skeletons_colors =  [],
                            return_other_colors = False,
                            main_mesh_color = [0.,1.,0.,0.2],
                            main_skeleton_color = [0,0.,1,1],
                            main_mesh_face_coloring = [],
                            other_scatter=[],
                            scatter_size = 0.3,
                            other_scatter_colors=[],
                            main_scatter_color=[1.,0.,0.,0.5],
                            buffer=1000,
                           axis_box_off=True,
                           html_path="",
                           show_at_end=True,
                           append_figure=False):
                           
things that need to be changed:
1) main_mesh combined
2) edge_coordinates is just the main_skeleton
other_scatter --> scatters
3) change all the other_[]_colors names


*if other inputs aren't list then make them list
                           
"""
import skeleton_utils as sk

import numpy_utils as nu
def plot_objects(main_mesh=None,
                 main_skeleton=None,
                 main_mesh_color = [0.,1.,0.,0.2],
                main_skeleton_color = [0,0.,1,1],
                meshes=[],
                meshes_colors =  [],
                mesh_alpha=0.2,
                            
                skeletons = [],
                skeletons_colors =  [],
                            
                scatters=[],
                scatters_colors=[],
                scatter_size = 0.3,
                            
                main_scatter_color=[1.,0.,0.,0.5],
                buffer=1000,
                axis_box_off=True,
                html_path="",
                show_at_end=True,
                append_figure=False,
                flip_y=True,
                
                subtract_from_main_mesh=True):
    import neuron_visualizations as nviz
    nviz = reload(nviz)
    

        
    if main_skeleton is None:
        edge_coordinates = []
    else:
        edge_coordinates=main_skeleton
        
        
    convert_to_list_vars = [meshes,meshes_colors,skeletons,
                            skeletons_colors,scatters,scatters_colors]
    
    def convert_to_list(curr_item):
        if type(curr_item) != list:
            if nu.is_array_like(curr_item):
                return list(curr_item)
            else:
                return [curr_item]
        else:
            return curr_item
    
    meshes =  convert_to_list(meshes)
    meshes_colors =  convert_to_list(meshes_colors)
    skeletons =  convert_to_list(skeletons)
    skeletons_colors =  convert_to_list(skeletons_colors)
    scatters =  convert_to_list(scatters)
    scatters_colors =  convert_to_list(scatters_colors)
    
    
    if (subtract_from_main_mesh and (not main_mesh is None) and (len(meshes)>0)):
        main_mesh = tu.subtract_mesh(original_mesh=main_mesh,
                                  subtract_mesh=meshes,exact_match=False)
        
    
    if main_mesh is None:
        main_mesh_verts = []
        main_mesh_faces= []
    else:
        main_mesh_verts = main_mesh.vertices
        main_mesh_faces= main_mesh.faces
        
    return sk.graph_skeleton_and_mesh(main_mesh_verts=main_mesh_verts,
                           main_mesh_faces=main_mesh_faces,
                           edge_coordinates=edge_coordinates,
                           other_meshes=meshes,
                                      mesh_alpha=mesh_alpha,
                            other_meshes_colors=meshes_colors,
                            other_skeletons=skeletons,
                            other_skeletons_colors=skeletons_colors,
                            other_scatter=scatters,
                            other_scatter_colors=scatters_colors,
                            scatter_size=scatter_size,
                            main_scatter_color=main_scatter_color,
                            buffer=buffer,
                            axis_box_off=axis_box_off,
                            html_path=html_path,
                            show_at_end=show_at_end,
                            append_figure=append_figure,
                            flip_y=flip_y
                           )
        
        
def plot_branch_spines(curr_branch):
    import trimesh_utils as tu
    """
    Purpose: To plot a branch with certain spines
    """
    shaft_mesh = tu.subtract_mesh(curr_branch.mesh,curr_branch.spines,exact_match=False)
    nviz.plot_objects(main_mesh=shaft_mesh,
                     meshes=curr_branch.spines,
                      meshes_colors="red",
                     mesh_alpha=1)
    
    
import numpy as np
import copy
import proofreading_utils as pru
def plot_split_suggestions_per_limb(neuron_obj,
                                    limb_results,
                                   scatter_color = "red",
                                    scatter_alpha = 0.3,
                                   scatter_size=0.3,
                                   add_components_colors=True,
                                   component_colors = "random"):

    """
    
    
    """
    for curr_limb_idx,path_cut_info in limb_results.items():
        component_colors_cp = copy.copy(component_colors)
        print(f"\n\n-------- Suggestions for Limb {curr_limb_idx}------")
        
        curr_scatters = []
        for path_i in path_cut_info:
            if len(path_i["coordinate_suggestions"])>0:
                curr_scatters.append(np.concatenate(path_i["coordinate_suggestions"]).reshape(-1,3))
                
        if len(curr_scatters) == 0:
            print("\n\n No suggested cuts for this limb!!")
            
            nviz.visualize_neuron(neuron_obj,
                             visualize_type=["mesh","skeleton"],
                             limb_branch_dict={f"L{curr_limb_idx}":"all"},
                             mesh_color="green",
                             skeleton_color="blue",
                             )
            continue
            
        curr_scatters = np.vstack(curr_scatters)
        scatter_color_list = [mu.color_to_rgba(scatter_color,scatter_alpha)]*len(curr_scatters)
        
        # will create a dictionary that will show all of the disconnected components in different colors
        if add_components_colors:
            curr_limb = pru.cut_limb_network_by_suggestions(copy.deepcopy(neuron_obj[curr_limb_idx]),
                                                      path_cut_info)
            limb_nx = curr_limb.concept_network
            
#             for cut in path_cut_info:
#                 limb_nx.remove_edges_from(cut["edges_to_cut"])
#                 limb_nx.add_edges_from(cut["edges_to_add"])
            
            conn_comp= list(nx.connected_components(limb_nx))
            
            if component_colors_cp == "random":
                component_colors_cp = mu.generate_color_list(n_colors = len(conn_comp))
            elif type(component_colors_cp) == list:
                component_colors_cp = component_colors_cp*np.ceil(len(conn_comp)/len(component_colors_cp)).astype("int")
            else:
                component_colors_cp = ["green"]*len(conn_comp)

            color_dict = dict()
            for groud_ids,c in zip(conn_comp,component_colors_cp):
                for i in groud_ids:
                    color_dict[i] = c
                    
            mesh_component_colors = color_dict
            skeleton_component_colors = color_dict
            #print(f"skeleton_component_colors = {color_dict}")
        else:
            mesh_component_colors = "green"
            skeleton_component_colors = "blue"
            
        #at this point have all of the scatters we want
        nviz.visualize_neuron(neuron_obj,
                             visualize_type=["mesh","skeleton"],
                             limb_branch_dict={f"L{curr_limb_idx}":"all"},
                             mesh_color={f"L{curr_limb_idx}":mesh_component_colors},
                             skeleton_color={f"L{curr_limb_idx}":skeleton_component_colors},
                             scatters=[curr_scatters],
                             scatters_colors=scatter_color_list,
                             scatter_size=scatter_size,
                             )
        
        
def visualize_neuron_path(neuron_obj,
                          limb_idx,
                          path,
                          path_mesh_color="red",
                          path_skeleton_color = "blue",
                          mesh_fill_color="green",
                          skeleton_fill_color="green",
                         visualize_type=["mesh","skeleton"],
                         scatters=[],
                         scatter_color_list=[],
                         scatter_size=0.3):
    
    curr_limb_idx = limb_idx
    

    mesh_component_colors = dict([(k,path_mesh_color) for k in path])
    skeleton_component_colors = dict([(k,path_skeleton_color) for k in path])
    
    nviz.visualize_neuron(neuron_obj,
                             visualize_type=visualize_type,
                             limb_branch_dict={f"L{curr_limb_idx}":"all"},
                             mesh_color={f"L{curr_limb_idx}":mesh_component_colors},
                              mesh_fill_color=mesh_fill_color,
                          
                             skeleton_color={f"L{curr_limb_idx}":skeleton_component_colors},
                          skeleton_fill_color=skeleton_fill_color,
                             scatters=scatters,
                             scatters_colors=scatter_color_list,
                             scatter_size=scatter_size,
                             )

def limb_correspondence_plottable(limb_correspondence,mesh_name="branch_mesh"):
    """
    Extracts the meshes and skeleton parts from limb correspondence so can be plotted
    
    """
    keys = list(limb_correspondence.keys())
    if list(limb_correspondence[keys[0]].keys())[0] == 0:
        # then we have a limb correspondence with multiple objects
        meshes=gu.combine_list_of_lists([[k[mesh_name] for k in ki.values()] for ki in limb_correspondence.values()])
        skeletons=gu.combine_list_of_lists([[k["branch_skeleton"] for k in ki.values()] for ki in limb_correspondence.values()])
    else:
        meshes=[k[mesh_name] for k in limb_correspondence.values()]
        skeletons=[k["branch_skeleton"] for k in limb_correspondence.values()]
    
    return meshes,skeletons

import general_utils as gu
def plot_limb_correspondence(limb_correspondence,
                            meshes_colors="random",
                            skeleton_colors="random",
                            mesh_name="branch_mesh",
                            scatters=[],
                            scatter_size=0.3,
                            **kwargs):
    meshes,skeletons = limb_correspondence_plottable(limb_correspondence,mesh_name=mesh_name)
        
    nviz.plot_objects(
                      meshes=meshes,
                     meshes_colors=meshes_colors,
                     skeletons=skeletons,
                     skeletons_colors=skeleton_colors,
        scatters=scatters,
        scatter_size = scatter_size,
        **kwargs
                     )
    
    
def plot_limb_path(limb_obj,path):
    """
    Purpose: To highlight the nodes on a path
    with just given a limb object
    
    Pseudocode: 
    1) Get the entire limb mesh will be the main mesh
    2) Get the meshes corresponding to the path
    3) Get all of the skeletons
    4) plot
    
    """
    
    nviz.plot_objects(main_mesh = limb_obj.mesh,
                        meshes=[limb_obj[k].mesh for k in path],
                      meshes_colors="red",
                     skeletons=[limb_obj[k].skeleton for k in path])
    
import neuron_visualizations as nviz
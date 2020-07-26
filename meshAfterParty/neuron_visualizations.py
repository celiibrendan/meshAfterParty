import ipyvolume as ipv
import skeleton_utils as sk
import numpy as np
import networkx as nx
import neuron_utils as nru
import networkx_utils as xu

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
                            starting_node_size=-1):
    
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
        starting_node_num = xu.get_starting_node(curr_concept_network)
        starting_node_num_coord = curr_concept_network.nodes[starting_node_num]["data"].mesh_center
    
        #print(f"Highlighting starting node {starting_node_num} with coordinate = {starting_node_num_coord}")
        
        sk.graph_skeleton_and_mesh(
                                   other_scatter=[starting_node_num_coord],
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
                               append_figure=True
                              )
    
    

    
    
    
    if show_at_end:
        ipv.show()
        
        
        
        
        
        
def plot_scatter_and_edges(scatter_points, #will help define the edges
                           edges,
                           directional,
                           starting_coordinate,
                           
                            arrow_size = 0.5,
                            arrow_color = "maroon",
                            edge_color = "black",
                            node_color = "red",
                            scatter_size = 0.1,
                            starting_node_color="pink",
                            show_at_end=True,
                            append_figure=False,
                            highlight_starting_node=True,
                            starting_node_size=-1):
    
    if starting_node_size == -1:
        starting_node_size = scatter_size*3
    
    """
    
    """
    
    
    if not append_figure:
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


    node_locations_array = np.array([v for v in node_locations.values()])
    #print(f"node_locations_array = {node_locations_array}")

    
    
    if highlight_starting_node:
        starting_node_num = xu.get_starting_node(curr_concept_network)
        starting_node_num_coord = curr_concept_network.nodes[starting_node_num]["data"].mesh_center
    
        #print(f"Highlighting starting node {starting_node_num} with coordinate = {starting_node_num_coord}")
        
        sk.graph_skeleton_and_mesh(
                                   other_scatter=[starting_node_num_coord],
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
                               append_figure=True
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

def plot_ipv_mesh(elephant_mesh_sub,color=[1.,0.,0.,0.2]):
    if len(elephant_mesh_sub.vertices) == 0:
        return
    
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
def plot_ipv_skeleton(edge_coordinates,color=[0,0.,1,1]):
    if len(edge_coordinates) == 0:
        print("Edge coordinates in plot_ipv_skeleton were of 0 length so returning")
        return []
    unique_skeleton_verts_final,edges_final = sk.convert_skeleton_to_nodes_edges(edge_coordinates)
    mesh2 = ipv.plot_trisurf(unique_skeleton_verts_final[:,0], 
                            unique_skeleton_verts_final[:,1], 
                            unique_skeleton_verts_final[:,2], 
                            lines=edges_final)
    #print(f"color in ipv_skeleton = {color}")
    mesh2.color = color 
    mesh2.material.transparent = True

    return unique_skeleton_verts_final

def plot_ipv_scatter(scatter_points,scatter_color=[1.,0.,0.,0.5],
                    scatter_size=0.4):
    scatter_points = np.array(scatter_points).reshape(-1,3)
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
def visualize_neuron(
    #the neuron we want to visualize
    current_neuron,
    
    #the categories that will be visualized
    visualize_type=["mesh"],
    limb_branch_dict="all",
    
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
    
    # ------ specific arguments for the concept_network -----
    network_directional=True,
    limb_to_starting_soma=None,
    
    node_size = 1,
    
    starting_node=True,
    starting_node_size=0.3,
    starting_node_color= "pink",
    starting_node_alpha=0.5,
    
    arrow_color = "maroon",
    arrow_alpha = 0.5,
    arrow_size = 0.5,
    
    
    # arguments for plotting other meshes associated with neuron #
    
    inside_pieces = False,
    inside_pieces_color = "black",
    inside_piece_alpha = 0.5,
    
    insignificant_limbs = False,
    insignificant_limbs_color = "black",
    insignificant_limbs_alpha = 0.5,
    
    non_soma_touching_meshes = False, #whether to graph the inside pieces
    non_soma_touching_meshes_color = "black",
    non_soma_touching_meshes_alpha = 0.5,
    
    
    
    # arguments for how to display/save ipyvolume fig
    buffer=1000,
    axis_box_off=True,
    html_path="",
    show_at_end=True,
    append_figure=False,
    
    # arguments that will help with random colorization:
    colors_to_omit= [],
    
    print_flag = False
    
    ):
    
    #To uncomment for full graphing
    if not append_figure:
        ipv.figure(figsize=(15,15))
    
        
    main_vertices = []
    
    print(f"mesh_resolution = {mesh_resolution}")
    print(f"mesh_color = {mesh_color}")
    #do the mesh visualization type
    for viz_type in visualize_type:
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
            
            # ------ specific arguments for the concept_network -----
            configuration_dict.setdefault("network_directional",network_directional)
            configuration_dict.setdefault("limb_to_starting_soma",limb_to_starting_soma)
            
            configuration_dict.setdefault("node_size",node_size)
            
            configuration_dict.setdefault("starting_node",starting_node)
            configuration_dict.setdefault("starting_node_size",starting_node_size)
            configuration_dict.setdefault("starting_node_color",starting_node_color)
            configuration_dict.setdefault("starting_node_alpha",starting_node_alpha)
            
            configuration_dict.setdefault("arrow_color",arrow_color)
            configuration_dict.setdefault("arrow_alpha",arrow_alpha)
            configuration_dict.setdefault("arrow_size",arrow_size)
            
        else:
            raise Exception(f"Recieved invalid visualization type: {viz_type}")
        
        
        #sets the limb branch dict specially  (uses overall one if none assigned)

        
        #handle if the limb_branch_dict is "all"
        if configuration_dict["limb_branch_dict"] is None:
            configuration_dict["limb_branch_dict"] = limb_branch_dict
        
        if configuration_dict["limb_branch_dict"] == "all":
            configuration_dict["limb_branch_dict"] = dict([(k,"all") for k in current_neuron.get_limb_node_names()])
        
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
                if configuration_dict["limb_branch_dict"][li] == "all":
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
        
        
        
        
            
#         if print_flag:
#             print(f"plot_items_order= {plot_items_order}")
#             print(f"plot_items= {plot_items}")
        
     
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
            print(f"plot_items_order = {plot_items_order}")
            
        #------at this point have a list of colors for all the things to plot -------
        
        
        
        #4) If soma is requested then get the some items
        
        
    
        if viz_type == "mesh":
            #add the vertices to plot to main_vertices list
            if len(plot_items)>0:
                min_max_vertices = np.array([[np.min(k.vertices,axis=0),np.max(k.vertices,axis=0)] for k in  plot_items]).reshape(-1,3)
                min_vertices = np.min(min_max_vertices,axis=0)
                max_vertices = np.max(min_max_vertices,axis=0)
                main_vertices.append(np.array([min_vertices,max_vertices]).reshape(-1,3))
                
            
            #Can plot the meshes now
            for curr_mesh,curr_mesh_color in zip(plot_items,color_list_correct_size):
                plot_ipv_mesh(curr_mesh,color=curr_mesh_color)
                
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
                soma_names = current_neuron.get_soma_node_names()
                
                if nu.is_array_like(type(configuration_dict["soma"])):
                    soma_names = [k for k in soma_names if k in configuration_dict["soma"]]
                soma_meshes = [current_neuron.concept_network.nodes[k]["data"].mesh for k in soma_names]
                
                soma_colors_list = mu.process_non_dict_color_input(configuration_dict["soma_color"])
                soma_colors_list_alpha = mu.apply_alpha_to_color_list(soma_colors_list,alpha=configuration_dict["soma_alpha"])
                soma_colors_list_alpha_fixed_size = mu.generate_color_list_no_alpha_change(soma_colors_list_alpha,
                                                                                          n_colors=len(soma_meshes))
                
                for curr_soma_mesh,curr_soma_color in zip(soma_meshes,soma_colors_list_alpha_fixed_size):
                    plot_ipv_mesh(curr_soma_mesh,color=curr_soma_color)
                    main_vertices.append(curr_soma_mesh.vertices)
                    
            #will add the background mesh if requested
            if configuration_dict["whole_neuron"]:
                whole_neuron_colors_list = mu.process_non_dict_color_input(configuration_dict["whole_neuron_color"])
                whole_neuron_colors_list_alpha = mu.apply_alpha_to_color_list(whole_neuron_colors_list,alpha=configuration_dict["whole_neuron_alpha"])
                
                #graph
                plot_ipv_mesh(current_neuron.mesh,color=whole_neuron_colors_list_alpha[0])
                main_vertices.append([np.min(current_neuron.mesh.vertices,axis=0),
                                      np.max(current_neuron.mesh.vertices,axis=0)])
                
            
        elif viz_type == "skeleton":
            #add the vertices to plot to main_vertices list
            if len(plot_items)>0:
                reshaped_items = np.concatenate(plot_items).reshape(-1,3)
                min_vertices = np.min(reshaped_items,axis=0)
                max_vertices = np.max(reshaped_items,axis=0)
                main_vertices.append(np.array([min_vertices,max_vertices]).reshape(-1,3))
            
            #Can plot the meshes now
            for curr_skeleton,curr_skeleton_color in zip(plot_items,color_list_correct_size):
                plot_ipv_skeleton(curr_skeleton,color=curr_skeleton_color)
            
            
            
            if configuration_dict["soma"]:
                soma_names = current_neuron.get_soma_node_names()
                
                if nu.is_array_like(type(configuration_dict["soma"])):
                    soma_names = [k for k in soma_names if k in configuration_dict["soma"]]
                    
                soma_colors_list = mu.process_non_dict_color_input(configuration_dict["soma_color"])
                soma_colors_list_alpha = mu.apply_alpha_to_color_list(soma_colors_list,alpha=configuration_dict["soma_alpha"])
                soma_colors_list_alpha_fixed_size = mu.generate_color_list_no_alpha_change(soma_colors_list_alpha,
                                                                                          n_colors=len(soma_names))
                #get the somas associated with the neurons
                soma_skeletons = [nru.get_soma_skeleton(current_neuron,k) for k in soma_names]
                
                for curr_soma_sk,curr_soma_sk_color in zip(soma_skeletons,soma_colors_list_alpha_fixed_size):
                    sk_vertices = plot_ipv_skeleton(curr_soma_sk,color=curr_soma_sk_color)
                    main_vertices.append(sk_vertices) #adding the vertices
            
            if configuration_dict["whole_neuron"]:
                whole_neuron_colors_list = mu.process_non_dict_color_input(configuration_dict["whole_neuron_color"])
                whole_neuron_colors_list_alpha = mu.apply_alpha_to_color_list(whole_neuron_colors_list,alpha=configuration_dict["whole_neuron_alpha"])
                
                #graph
                sk_vertices = plot_ipv_skeleton(current_neuron.skeleton,color=whole_neuron_colors_list_alpha[0])
                main_vertices.append([np.min(sk_vertices,axis=0),
                                      np.max(sk_vertices,axis=0)])
                
        elif viz_type == "network":
            #add the vertices to plot to main_vertices list
            if len(plot_items)>0:
                reshaped_items = np.concatenate(plot_items).reshape(-1,3)
                min_vertices = np.min(reshaped_items,axis=0)
                max_vertices = np.max(reshaped_items,axis=0)
                main_vertices.append(np.array([min_vertices,max_vertices]).reshape(-1,3))
                
            #this will look very different if the limbs or the branches want to be plotted
            
            """
            configuration_dict.setdefault("network_directional",network_directional)
            configuration_dict.setdefault("limb_to_starting_soma",limb_to_starting_soma)
            
            configuration_dict.setdefault("node_size",node_size)
            
            configuration_dict.setdefault("starting_node",starting_node)
            configuration_dict.setdefault("starting_node_size",starting_node_size)
            configuration_dict.setdefault("starting_node_color",starting_node_color)
            configuration_dict.setdefault("starting_node_alpha",starting_node_alpha)
            
            configuration_dict.setdefault("arrow_color",arrow_color)
            configuration_dict.setdefault("arrow_alpha",arrow_alpha)
            configuration_dict.setdefault("arrow_size",arrow_size)
            
            """
            
            raise Exception("Plotting of configuration network not implemented yet")
#             plot_concept_network(curr_concept_network,
#                             arrow_size = configuration_dict[,
#                             arrow_color = "maroon",
#                             edge_color = "black",
#                             node_color = "red",
#                             scatter_size = 0.1,
#                             starting_node_color="pink",
#                             show_at_end=True,
#                             append_figure=False,
#                             highlight_starting_node=True,
#                             starting_node_size=-1)
            
            
            
                                                        
            
            if configuration_dict["soma"]:
                raise Exception("Soma for skeleton Not implemented yet")
        else:
            raise Exception("Invalid viz_type")
        
        
        
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
    
    main_vertices = np.array(main_vertices)
    volume_max = np.max(main_vertices.reshape(-1,3),axis=0)
    volume_min = np.min(main_vertices.reshape(-1,3),axis=0)
        
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
    
    
    ipv.xlim(min_limits[0],max_limits[0])
    ipv.ylim(min_limits[1],max_limits[1])
    ipv.zlim(min_limits[2],max_limits[2])
    
    
    ipv.style.set_style_light()
    if axis_box_off:
        ipv.style.axes_off()
        ipv.style.box_off()
    else:
        ipv.style.axes_on()
        ipv.style.box_on()
        
    if show_at_end:
        ipv.show()
    
    if html_path != "":
        ipv.pylab.save(html_path)

    
    
    
    return
    
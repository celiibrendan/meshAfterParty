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
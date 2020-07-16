

"""
Purpose of this file: To help the development of the neuron object
1) Concept graph methods
2) Preprocessing pipeline for creating the neuron object from a meshs

"""

import skeleton_utils as sk
import soma_extraction_utils as sm
import trimesh_utils as tu
import trimesh
import numpy_utils as nu
import numpy as np
from importlib import reload
import networkx as nx
import time
import compartment_utils as cu
import networkx_utils as xu
import matplotlib_utils as mu

#importing at the bottom so don't get any conflicts
import neuron #package where can use the Branches class to help do branch skeleton analysis


from tqdm.notebook import tqdm
import itertools


# tools for restricting 


# -------------- tools for the concept graphs ------------------ #
def get_starting_info_from_concept_network(concept_networks):
    """
    Pseudocode: 
    1) get the soma it's connect to
    2) get the node that has the starting coordinate 
    3) get the endpoints and starting coordinate for that nodes
    """
    
    
    output_dicts = []
    for current_soma,curr_concept_network in concept_networks.items():
        curr_output_dict = dict()
        # 1) get the soma it's connect to
        curr_output_dict["starting_soma"] = current_soma
        
        # 2) get the node that has the starting coordinate 
        starting_node = xu.get_starting_node(curr_concept_network)
        curr_output_dict["starting_node"] = starting_node
        
        endpoints_dict = xu.get_node_attributes(curr_concept_network,attribute_name="endpoints",node_list=[starting_node],
                       return_array=False)
        
        curr_output_dict["starting_endpoints"] = endpoints_dict[starting_node]
        
        starting_node_dict = xu.get_node_attributes(curr_concept_network,attribute_name="starting_coordinate",node_list=[starting_node],
                       return_array=False)
        #get the starting coordinate of the starting dict
        curr_output_dict["starting_coordinate"] = starting_node_dict[starting_node]

        
        #curr_output_dict["concept_network"] = curr_concept_network
        output_dicts.append(curr_output_dict)
    
    return output_dicts

def convert_concept_network_to_skeleton(curr_concept_network):
    #get the midpoints
    node_locations = dict([(k,curr_concept_network.nodes[k]["data"].mesh_center) for k in curr_concept_network.nodes()])
    curr_edges = curr_concept_network.edges()
    graph_nodes_skeleton = np.array([(node_locations[n1],node_locations[n2]) for n1,n2 in curr_edges]).reshape(-1,2,3)
    return graph_nodes_skeleton


def convert_concept_network_to_undirectional(concept_network):
    return nx.Graph(concept_network)

def convert_concept_network_to_directional(concept_network,
                                        node_widths=None,
                                          no_cycles=True):
    """
    Pseudocode: 
    0) Create a dictionary with the keys as all the nodes and empty list as values
    1) Get the starting node
    2) Find all neighbors of starting node
    2b) Add the starting node to the list of all the nodes it is neighbors to
    3) Add starter node to the "procesed_nodes" so it is not processed again
    4) Add each neighboring node to the "to_be_processed" list

    5) Start loop that will continue until "to_be_processed" is done
    a. Get the next node to be processed
    b. Get all neighbors
    c. For all nodes who are not currently in the curr_nodes's list from the lookup dictionary
    --> add the curr_node to those neighbor nodes lists
    d. For all nodes not already in the to_be_processed or procesed_nodes, add them to the to_be_processed list
    ...
    z. when no more nodes in to_be_processed list then reak

    6) if the no_cycles option is selected:
    - for every neruong with multiple neurons in list, choose the one that has the branch width that closest matches

    7) convert the incoming edges dictionary to edge for a directional graph
    
    Example of how to use: 
    
    example_concept_network = nx.from_edgelist([[1,2],[2,3],[3,4],[4,5],[2,5],[2,6]])
    nx.draw(example_concept_network,with_labels=True)
    plt.show()
    xu.set_node_attributes_dict(example_concept_network,{1:dict(starting_coordinate=np.array([1,2,3]))})

    directional_ex_concept_network = nru.convert_concept_network_to_directional(example_concept_network,no_cycles=True)
    nx.draw(directional_ex_concept_network,with_labels=True)
    plt.show()

    node_widths = {1:0.5,2:0.61,3:0.73,4:0.88,5:.9,6:0.4}
    directional_ex_concept_network = nru.convert_concept_network_to_directional(example_concept_network,no_cycles=True,node_widths=node_widths)
    nx.draw(directional_ex_concept_network,with_labels=True)
    plt.show()
    """

    curr_limb_concept_network = concept_network
    mesh_widths = node_widths

    #if only one node in concept_network then return
    if len(curr_limb_concept_network.nodes()) <= 1:
        print("Concept graph size was 1 or less so returning original")
        return nx.DiGraph(curr_limb_concept_network)

    #0) Create a dictionary with the keys as all the nodes and empty list as values
    incoming_edges_to_node = dict([(k,[]) for k in curr_limb_concept_network.nodes()])
    to_be_processed_nodes = []
    processed_nodes = []
    max_iterations = len(curr_limb_concept_network.nodes()) + 100

    #1) Get the starting node 
    starting_node = xu.get_starting_node(curr_limb_concept_network)

    #2) Find all neighbors of starting node
    curr_neighbors = xu.get_neighbors(curr_limb_concept_network,starting_node)

    #2b) Add the starting node to the list of all the nodes it is neighbors to
    for cn in curr_neighbors:
        incoming_edges_to_node[cn].append(starting_node)

    #3) Add starter node to the "procesed_nodes" so it is not processed again
    processed_nodes.append(starting_node)

    #4) Add each neighboring node to the "to_be_processed" list
    to_be_processed_nodes.extend([k for k in curr_neighbors if k not in processed_nodes ])
    # print(f"incoming_edges_to_node AT START= {incoming_edges_to_node}")
    # print(f"processed_nodes_AT_START = {processed_nodes}")
    # print(f"to_be_processed_nodes_AT_START = {to_be_processed_nodes}")

    #5) Start loop that will continue until "to_be_processed" is done
    for i in range(max_iterations):
    #     print("\n")
    #     print(f"processed_nodes = {processed_nodes}")
    #     print(f"to_be_processed_nodes = {to_be_processed_nodes}")

        #a. Get the next node to be processed
        curr_node = to_be_processed_nodes.pop(0)
        #print(f"curr_node = {curr_node}")
        #b. Get all neighbors
        curr_node_neighbors = xu.get_neighbors(curr_limb_concept_network,curr_node)
        #print(f"curr_node_neighbors = {curr_node_neighbors}")
        #c. For all nodes who are not currently in the curr_nodes's list from the lookup dictionary
        #--> add the curr_node to those neighbor nodes lists
        for cn in curr_node_neighbors:
            if cn == curr_node:
                raise Exception("found a self connection in network graph")
            if cn not in incoming_edges_to_node[curr_node]:
                incoming_edges_to_node[cn].append(curr_node)

            #d. For all nodes not already in the to_be_processed or procesed_nodes, add them to the to_be_processed list
            if cn not in to_be_processed_nodes and cn not in processed_nodes:
                to_be_processed_nodes.append(cn)


        # add the nodes to those been processed
        processed_nodes.append(curr_node)


        #z. when no more nodes in to_be_processed list then reak
        if len(to_be_processed_nodes) == 0:
            break

    #print(f"incoming_edges_to_node = {incoming_edges_to_node}")
    #6) if the no_cycles option is selected:
    #- for every neruong with multiple neurons in list, choose the one that has the branch width that closest matches

    incoming_lengths = [k for k,v in incoming_edges_to_node.items() if len(v) >= 1]
    if len(incoming_lengths) != len(curr_limb_concept_network.nodes())-1:
        raise Exception("after loop in directed concept graph, not all nodes have incoming edges (except starter node)")

    if no_cycles == True:
        print("checking and resolving cycles")
        #get the nodes with multiple incoming edges
        multi_incoming = dict([(k,v) for k,v in incoming_edges_to_node.items() if len(v) >= 2])


        if len(multi_incoming) > 0:
            print("There are loops to resolve and 'no_cycles' parameters set requires us to fix eliminate them")
            #find the mesh widths of all the incoming edges and the current edge

            #if mesh widths are available then go that route
            if not mesh_widths is None:
                print("Using mesh_widths for resolving loops")
                for curr_node,incoming_nodes in multi_incoming.items():
                    curr_node_width = mesh_widths[curr_node]
                    incoming_nodes_width_difference = [np.linalg.norm(mesh_widths[k]- curr_node_width) for k in incoming_nodes]
                    winning_incoming_node = incoming_nodes[np.argmin(incoming_nodes_width_difference).astype("int")]
                    incoming_edges_to_node[curr_node] = [winning_incoming_node]
            else: #if not mesh widths available then just pick the longest edge
                """
                Get the coordinates of all of the nodes
                """
                node_coordinates_dict = xu.get_node_attributes(curr_limb_concept_network,attribute_name="coordinates",return_array=False)
                if set(list(node_coordinates_dict.keys())) != set(list(incoming_edges_to_node.keys())):
                    print("The keys of the concept graph with 'coordinates' do not match the keys of the edge dictionary")
                    print("Just going to use the first incoming edge by default")
                    for curr_node,incoming_nodes in multi_incoming.items():
                        winning_incoming_node = incoming_nodes[0]
                        incoming_edges_to_node[curr_node] = [winning_incoming_node]
                else: #then have coordinate information
                    print("Using coordinate distance to pick the winning node")
                    curr_node_coordinate = node_coordinates_dict[curr_node]
                    incoming_nodes_distance = [np.linalg.norm(node_coordinates_dict[k]- curr_node_coordinate) for k in incoming_nodes]
                    winning_incoming_node = incoming_nodes[np.argmax(incoming_nodes_distance).astype("int")]
                    incoming_edges_to_node[curr_node] = [winning_incoming_node]
        else:
            print("No cycles to fix")


        #check that all have length of 1
        multi_incoming = dict([(k,v) for k,v in incoming_edges_to_node.items() if len(v) == 1])
        if len(multi_incoming) != len(curr_limb_concept_network.nodes()) - 1:
            raise Exception("Inside the no_cycles but at the end all of the nodes only don't have one incoming cycle"
                           f"multi_incoming = {multi_incoming}")

    #7) convert the incoming edges dictionary to edge for a directional graph
    total_edges = []

    for curr_node,incoming_nodes in multi_incoming.items():
        curr_incoming_edges = [(j,curr_node) for j in incoming_nodes]
        total_edges += curr_incoming_edges

    #creating the directional network
    curr_limb_concept_network_directional = nx.DiGraph(nx.create_empty_copy(curr_limb_concept_network,with_data=True))
    curr_limb_concept_network_directional.add_edges_from(total_edges)

    return curr_limb_concept_network_directional


def branches_to_concept_network(curr_branch_skeletons,
                             starting_coordinate,
                              starting_edge,
                             max_iterations= 1000000):
    """
    Will change a list of branches into 
    """
    
    print(f"Starting_edge inside branches_to_conept = {starting_edge}")
    
    start_time = time.time()
    processed_nodes = []
    edge_endpoints_to_process = []
    concept_network_edges = []

    """
    If there is only one branch then just pass back a one-node graph 
    with no edges
    """
    if len(curr_branch_skeletons) == 0:
        raise Exception("Passed no branches to be turned into concept network")
    
    if len(curr_branch_skeletons) == 1:
        concept_network = xu.GraphOrderedEdges()
        concept_network.add_node(0)
        
        starting_node = 0
        attrs = {starting_node:{"starting_coordinate":starting_coordinate,"endpoints":neuron.Branch(starting_edge).endpoints}}
        xu.set_node_attributes_dict(concept_network,attrs)
        
        #add the endpoints 
        return concept_network

    # 0) convert each branch to one segment and build a graph from it
    
    
    curr_branch_meshes_downsampled = [sk.resize_skeleton_branch(b,n_segments=1) for b in curr_branch_skeletons]
    
    """
    In order to solve the problem that once resized there could be repeat edges
    
    Pseudocode: 
    1) predict the branches that are repeats and then create a map 
    of the non-dom (to be replaced) and dominant (the ones to replace)
    2) Get an arange list of the branch idxs and then delete the non-dominant ones
    3) Run the whole concept map process
    4) At the end for each non-dominant one, at it in (with it's idx) and copy
    the edges of the dominant one that it was mapped to
    
    
    """
    
    downsampled_skeleton = sk.stack_skeletons(curr_branch_meshes_downsampled)
    # curr_sk_graph_debug = sk.convert_skeleton_to_graph_old(downsampled_skeleton)
    # nx.draw(curr_sk_graph_debug,with_labels = True)

    #See if touching row matches the original: 
    

    all_skeleton_vertices = downsampled_skeleton.reshape(-1,3)
    unique_rows,indices = np.unique(all_skeleton_vertices,return_inverse=True,axis=0)
    
    reshaped_indices = np.sort(indices.reshape(-1,2),axis=1)
    unique_edges,unique_edges_indices = np.unique(reshaped_indices,axis = 0,return_inverse=True)
    from collections import Counter
    multiplicity_edge_counter = dict(Counter(unique_edges_indices))
    #this will give the unique edge that appears multiple times
    duplicate_edge_identifiers = [k for k,v in multiplicity_edge_counter.items() if v > 1] 
    
    #for keeping track of original indexes
    original_idxs = np.arange(0,len(curr_branch_meshes_downsampled))
    
    if len(duplicate_edge_identifiers) > 0:
        print(f"There were {len(duplicate_edge_identifiers)} duplication nodes found")
        all_conn_comp = []
        for d in duplicate_edge_identifiers:
            all_conn_comp.append(list(np.where(unique_edges_indices == [d] )[0]))

        domination_map = dict()
        for curr_comp in all_conn_comp:
            dom_node = curr_comp[0]
            non_dom_nodes = curr_comp[1:]
            for n_dom in non_dom_nodes:
                domination_map[n_dom] = dom_node
        print(f"domination_map = {domination_map}")
        

        to_delete_rows = list(domination_map.keys())

        #delete all of the non dominant rows from the indexes and the skeletons
        original_idxs = np.delete(original_idxs,to_delete_rows,axis=0)
        curr_branch_meshes_downsampled = [k for i,k in enumerate(curr_branch_meshes_downsampled) if i not in to_delete_rows]
    
    #print(f"curr_branch_meshes_downsampled[24] = {curr_branch_meshes_downsampled[24]}")
    curr_stacked_skeleton = sk.stack_skeletons(curr_branch_meshes_downsampled)
    #print(f"curr_stacked_skeleton[24] = {curr_stacked_skeleton[24]}")

    branches_graph = sk.convert_skeleton_to_graph(curr_stacked_skeleton) #can recover the original skeleton
#     print(f"len(curr_stacked_skeleton) = {len(curr_stacked_skeleton)}")
#     print(f"len(branches_graph.edges_ordered()) = {len(branches_graph.edges_ordered())}")
#     print(f"(branches_graph.edges_ordered())[24] = {(branches_graph.edges_ordered())[24]}")
#     print(f"coordinates = (branches_graph.edges_ordered())[24] = {xu.get_node_attributes(branches_graph,node_list=(branches_graph.edges_ordered())[24])}")


    #************************ need to just make an edges lookup dictionary*********#


    #1) Identify the starting node on the starting branch
    starting_node = xu.get_nodes_with_attributes_dict(branches_graph,dict(coordinates=starting_coordinate))
    print(f"At the start, starting_node (in terms of the skeleton, that shouldn't match the starting edge) = {starting_node}")
    if len(starting_node) != 1:
        raise Exception(f"The number of starting nodes found was not exactly one: {starting_node}")
    #1b) Add all edges incident and their other node label to a list to check (add the first node to processed nodes list)
    incident_edges = xu.node_to_edges(branches_graph,starting_node)
    #print(f"incident_edges = {incident_edges}")
    # #incident_edges_idx = edge_to_index(incident_edges)

    # #adding them to the list to be processed
    edge_endpoints_to_process = [(edges,edges[edges != starting_node ]) for edges in incident_edges]
    processed_nodes.append(starting_node)

    #need to add all of the newly to look edges and the current edge to the concept_network_edges
    """
    Pseudocode: 
    1) convert starting edge to the node identifiers
    2) iterate through all the edges to process and add the combos where the edge does not match
    """
    edge_coeff= []
    for k in starting_edge:
        edge_coeff.append(xu.get_nodes_with_attributes_dict(branches_graph,dict(coordinates=k))[0])
    
    
    for curr_edge,edge_enpt in edge_endpoints_to_process:
        if not np.array_equal(np.sort(curr_edge),np.sort(edge_coeff)):
            #add to the concept graph
            concept_network_edges += [(np.array(curr_edge),np.array(edge_coeff))]
        else:
            starting_node_edge = curr_edge
            print("printing out current edge:")
            print(xu.get_node_attributes(branches_graph,node_list=starting_node_edge))
        
    
    for i in range(max_iterations):
        #print(f"==\n\n On iteration {i}==")
        if len(edge_endpoints_to_process) == 0:
            print(f"edge_endpoints_to_process was empty so exiting loop after {i} iterations")
            break

        #2) Pop the edge edge number,endpoint of the stack
        edge,endpt = edge_endpoints_to_process.pop(0)
        #print(f"edge,endpt = {(edge,endpt)}")
        #- if edge already been processed then continue
        if endpt in processed_nodes:
            #print(f"Already processed endpt = {endpt} so skipping")
            continue
        #a. Find all edges incident on this node
        incident_edges = xu.node_to_edges(branches_graph,endpt)
        #print(f"incident_edges = {incident_edges}")

        considering_edges = [k for k in incident_edges if not np.array_equal(k,edge) and not np.array_equal(k,np.flip(edge))]
        #print(f"considering_edges = {considering_edges}")
        #b. Create edges from curent edge to those edges incident with it
        concept_network_edges += [(edge,k) for k in considering_edges]

        #c. Add the current node as processed
        processed_nodes.append(endpt)

        #d. For each edge incident add the edge and the other connecting node to the list
        new_edge_processing = [(e,e[e != endpt ]) for e in considering_edges]
        edge_endpoints_to_process = edge_endpoints_to_process + new_edge_processing
        #print(f"edge_endpoints_to_process = {edge_endpoints_to_process}")

    if len(edge_endpoints_to_process)>0:
        raise Exception(f"Reached max_interations of {max_iterations} and the edge_endpoints_to_process not empty")

    #flattening the connections so we can get the indexes of these edges
    flattened_connections = np.array(concept_network_edges).reshape(-1,2)
    orders = xu.get_edge_attributes(branches_graph,edge_list=flattened_connections)
    #******
    fixed_idx_orders = original_idxs[orders]
    concept_network_edges_fixed = np.array(fixed_idx_orders).reshape(-1,2)

    
    # # edge_endpoints_to_process
    #print(f"concept_network_edges_fixed = {concept_network_edges_fixed}")
    concept_network = xu.GraphOrderedEdges()
    #print("type(concept_network) = {type(concept_network)}")
    concept_network.add_edges_from([k for k in concept_network_edges_fixed])
    
    #add the endpoints as attributes to each of the nodes
    node_endpoints_dict = dict()
    old_ordered_edges = branches_graph.edges_ordered()
    for edge_idx,curr_branch_graph_edge in enumerate(old_ordered_edges):
        new_edge_idx = original_idxs[edge_idx]
        curr_enpoints = np.array(xu.get_node_attributes(branches_graph,node_list=curr_branch_graph_edge)).reshape(-1,3)
        node_endpoints_dict[new_edge_idx] = dict(endpoints=curr_enpoints)
        xu.set_node_attributes_dict(concept_network,node_endpoints_dict)
    
    
    
    
    #add the starting coordinate to the corresponding node
    #print(f"starting_node_edge right before = {starting_node_edge}")
    starting_order = xu.get_edge_attributes(branches_graph,edge_list=[starting_node_edge]) 
    #print(f"starting_order right before = {starting_order}")
    if len(starting_order) != 1:
        raise Exception(f"Only one starting edge index was not found,starting_order={starting_order} ")
    
    starting_edge_index = original_idxs[starting_order[0]]
    print(f"starting_node in concept map (that should match the starting edge) = {starting_edge_index}")
    #attrs = {starting_node[0]:{"starting_coordinate":starting_coordinate}} #old way that think uses the wrong starting_node
    attrs = {starting_edge_index:{"starting_coordinate":starting_coordinate}} 
    
    xu.set_node_attributes_dict(concept_network,attrs)
    
    #want to set all of the edge endpoints on the nodes as well just for a check
    
    
    
    print(f"Total time for branches to concept conversion = {time.time() - start_time}\n")
    
    
    # Add back the nodes that were deleted
    if len(duplicate_edge_identifiers) > 0:
        print("Working on adding back the edges that were duplicates")
        for non_dom,dom in domination_map.items():
            #print(f"Re-adding: {non_dom}")
            #get the endpoints attribute
            # local_node_endpoints_dict
            
            curr_neighbors = xu.get_neighbors(concept_network,dom)  
            new_edges = np.vstack([np.ones(len(curr_neighbors))*non_dom,curr_neighbors]).T
            concept_network.add_edges_from(new_edges)
            
            curr_endpoint = xu.get_node_attributes(concept_network,attribute_name="endpoints",node_list=[dom])[0]
            #print(f"curr_endpoint in add back = {curr_endpoint}")
            add_back_attribute_dict = {non_dom:dict(endpoints=curr_endpoint)}
            #print(f"To add dict = {add_back_attribute_dict}")
            xu.set_node_attributes_dict(concept_network,add_back_attribute_dict)
            
    return concept_network




def generate_limb_concept_networks_from_global_connectivity(
    limb_idx_to_branch_meshes_dict,
    limb_idx_to_branch_skeletons_dict,
    soma_idx_to_mesh_dict,
    soma_idx_connectivity,
    current_neuron,
    return_limb_labels=True
    ):
    """
    ****** Could significantly speed this up if better picked the 
    periphery meshes (which now are sending all curr_limb_divided_meshes)
    sent to 
    
    tu.mesh_pieces_connectivity(main_mesh=current_neuron,
                                        central_piece = curr_soma_mesh,
                                    periphery_pieces=curr_limb_divided_meshes)
    *********
    
    
    Purpose: To create concept networks for all of the skeletons
             based on our knowledge of the mesh

        Things it needs: 
        - branch_mehses
        - branch skeletons
        - soma meshes
        - whole neuron
        - soma_to_piece_connectivity

        What it returns:
        - concept networks
        - branch labels
        
    
    Pseudocode: 
    1) Get all of the meshes for that limb (that were decomposed)
    2) Use the entire neuron, the soma meshes and the list of meshes and find out shich one is touching the soma
    3) With the one that is touching the soma, find the enpoints of the skeleton
    4) Find the closest matching endpoint
    5) Send the deocmposed skeleton branches to the branches_to_concept_network function
    6) Graph the concept graph using the mesh centers

    Example of Use: 
    
    import neuron
    neuron = reload(neuron)

    #getting mesh and skeleton dictionaries
    limb_idx_to_branch_meshes_dict = dict()
    limb_idx_to_branch_skeletons_dict = dict()
    for k in limb_correspondence.keys():
        limb_idx_to_branch_meshes_dict[k] = [limb_correspondence[k][j]["branch_mesh"] for j in limb_correspondence[k].keys()]
        limb_idx_to_branch_skeletons_dict[k] = [limb_correspondence[k][j]["branch_skeleton"] for j in limb_correspondence[k].keys()]      

    #getting the soma dictionaries
    soma_idx_to_mesh_dict = dict()
    for k,v in enumerate(current_mesh_data[0]["soma_meshes"]):
        soma_idx_to_mesh_dict[k] = v

    soma_idx_connectivity = current_mesh_data[0]["soma_to_piece_connectivity"]


    limb_concept_networkx,limb_labels = neuron.generate_limb_concept_networks_from_global_connectivity(
        limb_idx_to_branch_meshes_dict = limb_idx_to_branch_meshes_dict,
        limb_idx_to_branch_skeletons_dict = limb_idx_to_branch_skeletons_dict,
        soma_idx_to_mesh_dict = soma_idx_to_mesh_dict,
        soma_idx_connectivity = soma_idx_connectivity,
        current_neuron=current_neuron,
        return_limb_labels=True
        )
    

    """

    
    
    if set(list(limb_idx_to_branch_meshes_dict.keys())) != set(list(limb_idx_to_branch_skeletons_dict.keys())):
        raise Exception("There was a difference in the keys for the limb_idx_to_branch_meshes_dict and limb_idx_to_branch_skeletons_dict")
        
    global_concept_time = time.time()
    
    total_limb_concept_networks = dict()
    total_limb_labels = dict()
    soma_mesh_faces = dict()
    for limb_idx in limb_idx_to_branch_meshes_dict.keys():
        local_concept_time = time.time()
        print(f"\n\n------Working on limb {limb_idx} -------")
        curr_concept_network = dict()
        
        curr_limb_divided_meshes = limb_idx_to_branch_meshes_dict[limb_idx]
        #curr_limb_divided_meshes_idx = [v["branch_face_idx"] for v in limb_correspondence[limb_idx].values()]
        curr_limb_divided_skeletons = limb_idx_to_branch_skeletons_dict[limb_idx]
        print(f"inside loop len(curr_limb_divided_meshes) = {len(curr_limb_divided_meshes)}"
             f" len(curr_limb_divided_skeletons) = {len(curr_limb_divided_skeletons)}")
        
        #find what mesh piece was touching
        touching_soma_indexes = []
        for k,v in soma_idx_connectivity.items():
            if limb_idx in v:
                touching_soma_indexes.append(k)
        
        if len(touching_soma_indexes) == 0:
            raise Exception("Did not find touching soma index")
        if len(touching_soma_indexes) >= 2:
            print("Merge limb detected")
            
        
        for soma_idx in touching_soma_indexes:
            print(f"--- Working on soma_idx: {soma_idx}----")
            curr_soma_mesh = soma_idx_to_mesh_dict[soma_idx]
            
            
            
            if soma_idx in list(soma_mesh_faces.keys()):
                soma_info = soma_mesh_faces[soma_idx]
            else:
                soma_info = curr_soma_mesh
                
            #filter the periphery pieces
            original_idxs = np.arange(0,len(curr_limb_divided_meshes))
            
            periph_filter_time = time.time()
            distances_periphery_to_soma = np.array([tu.closest_distance_between_meshes(curr_soma_mesh,k) for k in curr_limb_divided_meshes])
            periphery_distance_threshold = 2000
            
            original_idxs = original_idxs[distances_periphery_to_soma<periphery_distance_threshold]
            filtered_periphery_meshes = np.array(curr_limb_divided_meshes)[distances_periphery_to_soma<periphery_distance_threshold]
            
            print(f"Total time for filtering periphery meshes = {time.time() - periph_filter_time}")
            periph_filter_time = time.time()
            
            if len(filtered_periphery_meshes) == 0:
                raise Exception("There were no periphery meshes within a threshold distance of the mesh")
            

            touching_pieces,touching_vertices,central_piece_faces = tu.mesh_pieces_connectivity(main_mesh=current_neuron,
                                        central_piece = soma_info,
                                        periphery_pieces = filtered_periphery_meshes,
                                                         return_vertices=True,
                                                        return_central_faces=True
                                                                                 )
            soma_mesh_faces[soma_idx] = central_piece_faces
            
            #fixing the indexes so come out the same
            touching_pieces = original_idxs[touching_pieces]
            print(f"touching_pieces = {touching_pieces}")
            print(f"Total time for mesh connectivity = {time.time() - periph_filter_time}")
            
            
            if len(touching_pieces) >= 2:
                print("**More than one touching point to soma, touching_pieces = {touching_pieces}**")
                # picking the piece with the most shared vertices
                len_touch_vertices = [len(k) for k in touching_vertices]
                winning_piece_idx = np.argmax(len_touch_vertices)
                print(f"winning_piece_idx = {winning_piece_idx}")
                touching_piece = [touching_pieces[winning_piece_idx]]
                print(f"Winning touching piece = {touching_piece}")
            if len(touching_pieces) < 1:
                raise Exception("No touching pieces")
            
            #print out the endpoints of the winning touching piece
            
                
            #3) With the one that is touching the soma, find the enpoints of the skeleton
            
            touching_branch = neuron.Branch(curr_limb_divided_skeletons[touching_pieces[0]])
            endpoints = touching_branch.endpoints
            print(f"Touching piece endpoints = {endpoints}")
            soma_midpoint = np.mean(curr_soma_mesh.vertices,axis=0)

            #4) Find the closest matching endpoint
            closest_idx = np.argmin([np.linalg.norm(soma_midpoint-k) for k in endpoints])
            closest_endpoint = endpoints[closest_idx]
            
            print(f"inside inner loop "
             f"len(curr_limb_divided_skeletons) = {len(curr_limb_divided_skeletons)}")
            print(f"closest_endpoint = {closest_endpoint}")
            curr_limb_concept_network = branches_to_concept_network(curr_limb_divided_skeletons,closest_endpoint,np.array(endpoints).reshape(-1,3))
            curr_concept_network[soma_idx] = curr_limb_concept_network
            
            
            # ----- Some checks that make sure concept mapping went well ------ #
            #get the node that has the starting coordinate:
            recovered_touching_piece = xu.get_nodes_with_attributes_dict(curr_limb_concept_network,dict(starting_coordinate=closest_endpoint))
            
            print(f"recovered_touching_piece = {recovered_touching_piece}")
            if recovered_touching_piece[0] != touching_pieces[0]:
                raise Exception(f"For limb {limb_idx} and soma {soma_idx} the recovered_touching and original touching do not match\n"
                               f"recovered_touching_piece = {recovered_touching_piece}, original_touching_pieces = {touching_pieces}")
                                                                         

            print(f"After concept mapping size = {len(curr_limb_concept_network.nodes())}")
            
            if len(curr_limb_concept_network.nodes()) != len(curr_limb_divided_skeletons):
                   raise Exception("The number of nodes in the concept graph and number of branches passed to it did not match\n"
                                  f"len(curr_limb_concept_network.nodes())={len(curr_limb_concept_network.nodes())}, len(curr_limb_divided_skeletons)= {len(curr_limb_divided_skeletons)}")

            if nx.number_connected_components(curr_limb_concept_network) > 1:
                raise Exception("There was more than 1 connected components in the concept network")
            
#             #for debugging: 
#             endpoints_dict = xu.get_node_attributes(curr_limb_concept_network,attribute_name="endpoints",return_array=False)
#             print(f"endpoints_dict = {endpoints_dict}")
#             print(f"{curr_limb_concept_network.nodes()}")
            
            
            #make sure that the original divided_skeletons endpoints match the concept map endpoints
            for j,un_resized_b in enumerate(curr_limb_divided_skeletons):
                """
                Pseudocode: 
                1) get the endpoints of the current branch
                2) get the endpoints in the concept map
                3) compare
                - if not equalt then break
                """
                #1) get the endpoints of the current branch
                b_endpoints = neuron.Branch(un_resized_b).endpoints
                #2) get the endpoints in the concept map
                graph_endpoints = xu.get_node_attributes(curr_limb_concept_network,attribute_name="endpoints",node_list=[j])[0]
                #print(f"original_branch_endpoints = {b_endpoints}, concept graph node endpoints = {graph_endpoints}")
                if not xu.compare_endpoints(b_endpoints,graph_endpoints):
                    raise Exception(f"The node {j} in concept graph endpoints do not match the endpoints of the original branch\n"
                                   f"original_branch_endpoints = {b_endpoints}, concept graph node endpoints = {graph_endpoints}")
                
                
        total_limb_concept_networks[limb_idx] = curr_concept_network
        
        if len(curr_concept_network) > 1:
            total_limb_labels[limb_idx] = "MergeError"
        else:
            total_limb_labels[limb_idx] = "Normal"
            
        print(f"Local time for concept mapping = {time.time() - local_concept_time}")

    print(f"Total time for concept mapping = {time.time() - global_concept_time}")
        
        
    #returning from the function    
        
    if return_limb_labels:
        return total_limb_concept_networks,total_limb_labels
    else:
        return total_limb_concept_networks









# ------------------------ For the preprocessing ----------------------- #

def preprocess_neuron(mesh=None,
                     mesh_file=None,
                     segment_id=None,
                     description=None,
                     sig_th_initial_split=15, #for significant splitting meshes in the intial mesh split
                     limb_threshold = 2000, #the mesh faces threshold for a mesh to be qualified as a limb (otherwise too small)
                      filter_end_node_length=5000, #used in cleaning the skeleton during skeletonizations
                     ):
    
    
    whole_processing_tiempo = time.time()
    
    """
    Purpose: To process the mesh into a format that can be loaded into the neuron class
    and used for higher order processing (how to visualize is included)
    
    """
    
    if segment_id is None:
        #pick a random segment id
        segment_id = np.random.randint(100000000)
        print(f"picking a random 7 digit segment id: {segment_id}")
    if description is None:
        description = "no_description"
    
    if mesh is None:
        if current_mesh_file is None:
            raise Exception("No mesh or mesh_file file were given")
        else:
            current_neuron = trimesh.load_mesh(current_mesh_file)
    else:
        current_neuron = mesh
        
    # ************************ Phase A ********************************
    
    print("\n\n\n\n\n****** Phase A ***************\n\n\n\n\n")
    
    
    
    
    
    # --- 1) Doing the soma detection
    
    soma_mesh_list,run_time,total_soma_list_sdf = sm.extract_soma_center(segment_id,
                                             current_neuron.vertices,
                                             current_neuron.faces)
    
    # geting the soma centers
    if len(soma_mesh_list) <= 0:
        print(f"**** No Somas Found for Mesh {segment_id} so just one mesh")
        soma_mesh_list_centers = []
        raise Exception("Processing of No Somas is not yet implemented yet")
    else:
        #compute the soma centers
        print(f"Soma List = {soma_mesh_list}")

        soma_mesh_list_centers = sm.find_soma_centroids(soma_mesh_list)
        print(f"soma_mesh_list_centers = {soma_mesh_list_centers}")
    
#     sk.graph_skeleton_and_mesh(main_mesh_verts=current_neuron.vertices,
#                           main_mesh_faces=current_neuron.faces,
#                            main_mesh_color = [0.,1.,0.,0.8]
#                           )

    # ********At this point assume that there are somas (if not would just skip to the limb skeleton stuff) *******
    
    
    
    
    
    
    
    
    #--- 2) getting the soma submeshes that are connected to each soma and identifiying those that aren't (and eliminating any mesh pieces inside the soma)
    
    main_mesh_total = current_neuron
    

    #finding the mesh pieces that contain the soma
    #splitting the current neuron into distinct pieces
    split_meshes = tu.split_significant_pieces(
                                main_mesh_total,
                                significance_threshold=sig_th_initial_split,
                                print_flag=False)

    print(f"# total split meshes = {len(split_meshes)}")


    #returns the index of the split_meshes index that contains each soma    
    containing_mesh_indices = sm.find_soma_centroid_containing_meshes(soma_mesh_list_centers,
                                            split_meshes)
    
    # filtering away any of the inside floating pieces: 
    non_soma_touching_meshes = [m for i,m in enumerate(split_meshes)
                     if i not in list(containing_mesh_indices.values())]


    #Adding the step that will filter away any pieces that are inside the soma
    if len(non_soma_touching_meshes) > 0 and len(soma_mesh_list) > 0:
        """
        *** want to save these pieces that are inside of the soma***
        """

        non_soma_touching_meshes,inside_pieces = sm.filter_away_inside_soma_pieces(soma_mesh_list,non_soma_touching_meshes,
                                        significance_threshold=sig_th_initial_split,
                                        return_inside_pieces = True)                                                      


    split_meshes # the meshes of the original mesh
    containing_mesh_indices #the mapping of each soma centroid to the correct split mesh
    soma_containing_meshes = sm.grouping_containing_mesh_indices(containing_mesh_indices)

    soma_touching_meshes = [split_meshes[k] for k in soma_containing_meshes.keys()]


    print(f"# of non soma touching seperate meshes = {len(non_soma_touching_meshes)}")
    print(f"# of inside pieces = {len(inside_pieces)}")
    print(f"# of soma containing seperate meshes = {len(soma_touching_meshes)}")
    print(f"meshes with somas = {soma_containing_meshes}")

   
    

    
    
    
    #--- 3)  Soma Extraction was great (but it wasn't the original soma faces), so now need to get the original soma faces and the original non-soma faces of original pieces
    
#     sk.graph_skeleton_and_mesh(other_meshes=[soma_meshes])

    

    """
    for each soma touching mesh get the following:
    1) original soma meshes
    2) significant mesh pieces touching these somas
    3) The soma connectivity to each of the significant mesh pieces
    -- later will just translate the 


    Process: 

    1) Final all soma faces (through soma extraction and then soma original faces function)
    2) Subtact all soma faces from original mesh
    3) Find all significant mesh pieces
    4) Backtrack significant mesh pieces to orignal mesh and find connectivity of each to all
       the available somas
    Conclusion: Will have connectivity map


    """

    soma_touching_mesh_data = dict()

    for z,(mesh_idx, soma_idxes) in enumerate(soma_containing_meshes.items()):
        soma_touching_mesh_data[z] = dict()
        print("\n\n----Working on soma-containing mesh piece {z}----")

        #1) Final all soma faces (through soma extraction and then soma original faces function)
        current_mesh = split_meshes[mesh_idx]

        current_soma_mesh_list = [soma_mesh_list[k] for k in soma_idxes]

        current_time = time.time()
        mesh_pieces_without_soma = sm.subtract_soma(current_soma_mesh_list,current_mesh,
                                                    significance_threshold=250)
        print(f"Total time for Subtract Soam = {time.time() - current_time}")
        current_time = time.time()

        mesh_pieces_without_soma_stacked = tu.combine_meshes(mesh_pieces_without_soma)

        # find the original soma faces of mesh
        soma_faces = tu.original_mesh_faces_map(current_mesh,mesh_pieces_without_soma_stacked,matching=False)
        print(f"Total time for Original_mesh_faces_map for mesh_pieces without soma= {time.time() - current_time}")
        current_time = time.time()
        soma_meshes = current_mesh.submesh([soma_faces],append=True)

        # finding the non-soma original faces
        non_soma_faces = tu.original_mesh_faces_map(current_mesh,soma_meshes,matching=False)
        non_soma_stacked_mesh = current_mesh.submesh([non_soma_faces],append=True)

        print(f"Total time for Original_mesh_faces_map for somas= {time.time() - current_time}")
        current_time = time.time()

        # 3) Find all significant mesh pieces
        sig_non_soma_pieces,insignificant_limbs = tu.split_significant_pieces(non_soma_stacked_mesh,significance_threshold=limb_threshold,
                                                         return_insignificant_pieces=True)

        print(f"Total time for sig_non_soma_pieces= {time.time() - current_time}")
        current_time = time.time()

        soma_touching_mesh_data[z]["branch_meshes"] = sig_non_soma_pieces

        #4) Backtrack significant mesh pieces to orignal mesh and find connectivity of each to all the available somas
        # get all the seperate mesh faces

        #How to seperate the mesh faces
        seperate_soma_meshes,soma_face_components = tu.split(soma_meshes,only_watertight=False)
        #take the top largest ones depending how many were originally in the soma list
        seperate_soma_meshes = seperate_soma_meshes[:len(soma_mesh_list)]
        soma_face_components = soma_face_components[:len(soma_mesh_list)]

        soma_touching_mesh_data[z]["soma_meshes"] = seperate_soma_meshes

        print(f"Total time for split= {time.time() - current_time}")
        current_time = time.time()



        soma_to_piece_connectivity = dict()
        for i,curr_soma in enumerate(seperate_soma_meshes):
            connected_mesh_pieces,connected_mesh_pieces_vertices  = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_soma,
                            periphery_pieces = sig_non_soma_pieces,
                            return_vertices = True)
            #print(f"soma {i}: connected_mesh_pieces = {connected_mesh_pieces}")
            soma_to_piece_connectivity[i] = connected_mesh_pieces

        print(f"Total time for mesh_pieces_connectivity= {time.time() - current_time}")

        soma_touching_mesh_data[z]["soma_to_piece_connectivity"] = soma_to_piece_connectivity

    print(f"# of insignificant_limbs = {len(insignificant_limbs)} with trimesh : {insignificant_limbs}")
    
    
    
    # Lets have an alert if there was more than one soma disconnected meshes
    if len(soma_touching_mesh_data.keys()) > 1:
        raise Exception("More than 1 disconnected meshes that contain somas")
    
    
    # ****Soma Touching mesh Data has the branches and the connectivity (So this is where you end up skipping if you don't have somas)***
    
    
    
    
    
    
    
    
    
    
    
    
    # ---5) Working on the Actual skeleton of all of the branches

    
    global_start_time = time.time()

    for j,(soma_containing_mesh_idx,mesh_data) in enumerate(soma_touching_mesh_data.items()):
        print(f"\n-- Working on Soma Continaing Mesh {j}--")
        current_branches = mesh_data["branch_meshes"]

        #skeletonize each of the branches
        total_skeletons = []

        for z,branch in enumerate(current_branches):
            print(f"\n    -- Working on branch {z}--")
            curren_skeleton = sk.skeletonize_connected_branch(branch)
            #clean the skeleton
                # --------  Doing the cleaning ------- #
            clean_time = time.time()
            
            new_cleaned_skeleton = sk.clean_skeleton(curren_skeleton,
                                    distance_func=sk.skeletal_distance,
                              min_distance_to_junction=filter_end_node_length, #this used to be a tuple i think when moved the parameter up to function defintion
                              return_skeleton=True,
                              print_flag=False)
            print(f"    Total time for skeleton and cleaning of branch {z}: {time.time() - clean_time}")
            if len(new_cleaned_skeleton) == 0:
                raise Exception(f"Found a zero length skeleton for limb {z} of trmesh {branch}")
            total_skeletons.append(new_cleaned_skeleton)

        soma_touching_mesh_data[j]["branch_skeletons"] = total_skeletons

    print(f"Total time for skeletonization = {time.time() - global_start_time}")
    
    
    
    
    
    
    
    
    
    
    
    
    # *************** Phase B *****************
    
    print("\n\n\n\n\n****** Phase B ***************\n\n\n\n\n")
    
    current_mesh_data = soma_touching_mesh_data
    
    
    # visualizing the original neuron
#     current_neuron = trimesh.load_mesh(current_mesh_file)
#     sk.graph_skeleton_and_mesh(main_mesh_verts=current_neuron.vertices,
#                               main_mesh_faces=current_neuron.faces,
#                                main_mesh_color = [0.,1.,0.,0.8]
#                               )
    
    
    # visualizing the somas that were extracted
#     soma_meshes = tu.combine_meshes(current_mesh_data[0]["soma_meshes"])
#     sk.graph_skeleton_and_mesh(main_mesh_verts=soma_meshes.vertices,
#                               main_mesh_faces=soma_meshes.faces,
#                                main_mesh_color = [0.,1.,0.,0.8]
#                               )


    # # Visualize the extracted branches
    # # visualize all of the branches and the meshes
    # sk.graph_skeleton_and_mesh(other_meshes=list(current_mesh_data[0]["branch_meshes"]) + list(current_mesh_data[0]["soma_meshes"]),
    #                           other_meshes_colors="random",
    #                            other_skeletons = current_mesh_data[0]["branch_skeletons"],
    #                           other_skeletons_colors="random")
    
    
    
    
    
    
    
    
    #--- 1) Cleaning each limb through distance and decomposition, checking that all cleaned branches are connected components and then visualizing
    
    skelton_cleaning_threshold = 4001
    total_cleaned = []
    for j,curr_skeleton_to_clean in enumerate(current_mesh_data[0]["branch_skeletons"]):
        print(f"\n---- Working on Limb {j} ----")
        start_time = time.time()
        print(f"before cleaning limb size of skeleton = {curr_skeleton_to_clean.shape}")
        distance_cleaned_skeleton = sk.clean_skeleton(
                                                    curr_skeleton_to_clean,
                                                    distance_func=sk.skeletal_distance,
                                                    min_distance_to_junction = skelton_cleaning_threshold,
                                                    return_skeleton=True,
                                                    print_flag=False) 
        #make sure still connected componet
        distance_cleaned_skeleton_components = nx.number_connected_components(sk.convert_skeleton_to_graph(distance_cleaned_skeleton))
        if distance_cleaned_skeleton_components > 1:
            raise Exception(f"distance_cleaned_skeleton {j} was not a single component: it was actually {distance_cleaned_skeleton_components} components")

        print(f"after DISTANCE cleaning limb size of skeleton = {distance_cleaned_skeleton.shape}")
        cleaned_branch = sk.clean_skeleton_with_decompose(distance_cleaned_skeleton)

        cleaned_branch_components = nx.number_connected_components(sk.convert_skeleton_to_graph(cleaned_branch))
        if cleaned_branch_components > 1:
            raise Exception(f"cleaned_branch {j} was not a single component: it was actually {cleaned_branch_components} components")

        #do the cleanin ghtat removes loops from branches
        print(f"After DECOMPOSITION cleaning limb size of skeleton = {cleaned_branch.shape}")
        print(f"Total time = {time.time() - start_time}")
        total_cleaned.append(cleaned_branch)

    current_mesh_data[0]["branch_skeletons_cleaned"] = total_cleaned
    
    
    
    # checking all cleaned branches are connected components

    for k,cl_sk in enumerate(current_mesh_data[0]["branch_skeletons"]): 
        n_components = nx.number_connected_components(sk.convert_skeleton_to_graph(cl_sk)) 
        if n_components > 1:
            raise Exception(f"Original limb {k} was not a single component: it was actually {n_components} components")

    for k,cl_sk in enumerate(current_mesh_data[0]["branch_skeletons_cleaned"]): 
        n_components = nx.number_connected_components(sk.convert_skeleton_to_graph(cl_sk)) 
        if n_components > 1:
            raise Exception(f"Cleaned limb {k} was not a single component: it was actually {n_components} components")
            
    
    # # visualize all of the branches and the meshes
    # sk.graph_skeleton_and_mesh(other_meshes=list(current_mesh_data[0]["branch_meshes"]) + list(current_mesh_data[0]["soma_meshes"]),
    #                           other_meshes_colors="random",
    #                            other_skeletons = current_mesh_data[0]["branch_skeletons_cleaned"],
    #                           other_skeletons_colors="random",
    #                           mesh_alpha=0.15,
    #                           html_path=f"{segment_id}_limb_skeleton.html")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # --- 2) Decomposing of limbs into branches and finding mesh correspondence (using adaptive mesh correspondence followed by a water fill for conflict and empty faces), checking that it went well with no empty meshes and all connected component graph (even when downsampling the skeleton) when constructed from branches, plus visualization at end
    
    

    start_time = time.time()

    limb_correspondence = dict()
    soma_containing_idx= 0

    for soma_containing_idx in current_mesh_data.keys():
        for limb_idx,curr_limb_mesh in enumerate(current_mesh_data[soma_containing_idx]["branch_meshes"]):
            print(f"Working on limb #{limb_idx}")
            limb_correspondence[limb_idx] = dict()
            curr_limb_sk = current_mesh_data[soma_containing_idx]["branch_skeletons_cleaned"][limb_idx]
            curr_limb_branches_sk_uneven = sk.decompose_skeleton_to_branches(curr_limb_sk) #the line that is decomposing to branches

            for j,curr_branch_sk in tqdm(enumerate(curr_limb_branches_sk_uneven)):
                limb_correspondence[limb_idx][j] = dict()


                curr_branch_face_correspondence, width_from_skeleton = cu.mesh_correspondence_adaptive_distance(curr_branch_sk,
                                              curr_limb_mesh,
                                             skeleton_segment_width = 1000)



                if len(curr_branch_face_correspondence) > 0:
                    curr_submesh = curr_limb_mesh.submesh([list(curr_branch_face_correspondence)],append=True)
                else:
                    curr_submesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))

                limb_correspondence[limb_idx][j]["branch_skeleton"] = curr_branch_sk
                limb_correspondence[limb_idx][j]["correspondence_mesh"] = curr_submesh
                limb_correspondence[limb_idx][j]["correspondence_face_idx"] = curr_branch_face_correspondence
                limb_correspondence[limb_idx][j]["width_from_skeleton"] = width_from_skeleton


    print(f"Total time for decomposition = {time.time() - start_time}")
    
    
    #couple of checks on how the decomposition went:  for each limb
    #1) if shapes of skeletons cleaned and divided match
    #2) if skeletons are only one component
    #3) if you downsample the skeletons then still only one component
    #4) if any empty meshes
    
    empty_submeshes = []

    for soma_containing_idx in current_mesh_data.keys():
        for limb_idx,curr_limb_mesh in enumerate(current_mesh_data[soma_containing_idx]["branch_meshes"]):
            print(f"\n---- checking limb {limb_idx}---")
            print(f"Limb {limb_idx} decomposed into {len(limb_correspondence[limb_idx])} branches")

            #get all of the skeletons and make sure that they from a connected component
            divided_branches = [limb_correspondence[limb_idx][k]["branch_skeleton"] for k in limb_correspondence[limb_idx]]
            divided_skeleton_graph = sk.convert_skeleton_to_graph(
                                            sk.stack_skeletons(divided_branches))

            divided_skeleton_graph_recovered = sk.convert_graph_to_skeleton(divided_skeleton_graph)

            cleaned_limb_skeleton = current_mesh_data[0]['branch_skeletons_cleaned'][limb_idx]
            print(f"divided_skeleton_graph_recovered = {divided_skeleton_graph_recovered.shape} and \n"
                  f"current_mesh_data[0]['branch_skeletons_cleaned'].shape = {cleaned_limb_skeleton.shape}\n")
            if divided_skeleton_graph_recovered.shape != cleaned_limb_skeleton.shape:
                print(f"****divided_skeleton_graph_recovered and cleaned_limb_skeleton shapes not match: "
                                f"{divided_skeleton_graph_recovered.shape} vs. {cleaned_limb_skeleton.shape} *****")


            #check that it is all one component
            divided_skeleton_graph_n_comp = nx.number_connected_components(divided_skeleton_graph)
            print(f"Number of connected components in deocmposed recovered graph = {divided_skeleton_graph_n_comp}")

            cleaned_limb_skeleton_graph = sk.convert_skeleton_to_graph(cleaned_limb_skeleton)
            cleaned_limb_skeleton_graph_n_comp = nx.number_connected_components(cleaned_limb_skeleton_graph)
            print(f"Number of connected components in cleaned skeleton graph= {cleaned_limb_skeleton_graph_n_comp}")

            if divided_skeleton_graph_n_comp > 1 or cleaned_limb_skeleton_graph_n_comp > 1:
                raise Exception(f"One of the decompose_skeletons or cleaned skeletons was not just one component : {divided_skeleton_graph_n_comp,cleaned_limb_skeleton_graph_n_comp}")

            #check that when we downsample it is not one component:
            curr_branch_meshes_downsampled = [sk.resize_skeleton_branch(b,n_segments=1) for b in divided_branches]
            downsampled_skeleton = sk.stack_skeletons(curr_branch_meshes_downsampled)
            curr_sk_graph_debug = sk.convert_skeleton_to_graph(downsampled_skeleton)


            con_comp = list(nx.connected_components(curr_sk_graph_debug))
            if len(con_comp) > 1:
                raise Exception(f"There were more than 1 component when downsizing: {[len(k) for k in con_comp]}")
            else:
                print(f"The downsampled branches number of connected components = {len(con_comp)}")


            for j in limb_correspondence[limb_idx].keys():
                if len(limb_correspondence[limb_idx][j]["correspondence_mesh"].faces) == 0:
                    empty_submeshes.append(dict(limb_idx=limb_idx,branch_idx = j))

    print(f"Empty submeshes = {empty_submeshes}")

    if len(empty_submeshes) > 0:
        raise Exception(f"Found empyt meshes after branch mesh correspondence: {empty_submeshes}")
        
        

    # import matplotlib_utils as mu

    # sk.graph_skeleton_and_mesh(other_meshes=total_branch_meshes,
    #                           other_meshes_colors="random",
    #                            other_skeletons=total_branch_skeletons,
    #                            other_skeletons_colors="random"
    #                           )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # ---3) Finishing off the face correspondence so get 1-to-1 correspondence of mesh face to skeletal piece
    
    #--- this is the function that will clean up a limb piece so have 1-1 correspondence

    #things to prep for visualizing the axons
#     total_widths = []
#     total_branch_skeletons = []
#     total_branch_meshes = []

    soma_containing_idx = 0

    for limb_idx in limb_correspondence.keys():
        mesh_start_time = time.time()
        #clear out the mesh correspondence if already in limb_correspondecne
        for k in limb_correspondence[limb_idx].keys():
            if "branch_mesh" in limb_correspondence[limb_idx][k]:
                del limb_correspondence[limb_idx][k]["branch_mesh"]
            if "branch_face_idx" in limb_correspondence[limb_idx][k]:
                del limb_correspondence[limb_idx][k]["branch_face_idx"]
        #geting the current limb mesh
        print(f"\n\nWorking on limb_correspondence for #{limb_idx}")
        no_missing_labels = list(limb_correspondence[limb_idx].keys()) #counts the number of divided branches which should be the total number of labels
        curr_limb_mesh = current_mesh_data[soma_containing_idx]["branch_meshes"][limb_idx]

        #set up the face dictionary
        face_lookup = dict([(j,[]) for j in range(0,len(curr_limb_mesh.faces))])

        for j,branch_piece in limb_correspondence[limb_idx].items():
            curr_faces_corresponded = branch_piece["correspondence_face_idx"]

            for c in curr_faces_corresponded:
                face_lookup[c].append(j)

        original_labels = set(list(itertools.chain.from_iterable(list(face_lookup.values()))))
        print(f"max(original_labels),len(original_labels) = {(max(original_labels),len(original_labels))}")


        if len(original_labels) != len(no_missing_labels):
            raise Exception(f"len(original_labels) != len(no_missing_labels) for original_labels = {len(original_labels)},no_missing_labels = {len(no_missing_labels)}")

        if max(original_labels) + 1 > len(original_labels):
            raise Exception("There are some missing labels in the initial labeling")



        #here is where can call the function that resolves the face labels
        face_coloring_copy = cu.resolve_empty_conflicting_face_labels(
                         curr_limb_mesh = curr_limb_mesh,
                         face_lookup=face_lookup,
                         no_missing_labels = list(original_labels)
        )


        # -- splitting the mesh pieces into individual pieces
        divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(curr_limb_mesh,face_coloring_copy)

        #-- check that all the split mesh pieces are one component --#

        #save off the new data as branch mesh
        for k in limb_correspondence[limb_idx].keys():
            limb_correspondence[limb_idx][k]["branch_mesh"] = divided_submeshes[k]
            limb_correspondence[limb_idx][k]["branch_face_idx"] = divided_submeshes_idx[k]
#             total_widths.append(limb_correspondence[limb_idx][k]["width_from_skeleton"])
#             total_branch_skeletons.append(limb_correspondence[limb_idx][k]["branch_skeleton"])
#             total_branch_meshes.append(limb_correspondence[limb_idx][k]["branch_mesh"])

        print(f"Total time for limb mesh processing = {time.time() - mesh_start_time}")
    
    
    
    
    
    # Visualizing the results of getting the mesh to skeletal segment correspondence completely 1-to-1
    
#     from matplotlib import pyplot as plt
#     fig,ax = plt.subplots(1,1)
#     bins = plt.hist(np.array(total_widths),bins=100)
#     ax.set_xlabel("Width measurement of mesh branch (nm)")
#     ax.set_ylabel("frequency")
#     ax.set_title("Width measurement of mesh branch frequency")
#     plt.show()
    
#     sk.graph_skeleton_and_mesh(other_meshes=total_branch_meshes,
#                           other_meshes_colors="random",
#                           other_skeletons=total_branch_skeletons,
#                           other_skeletons_colors="random",
#                           #html_path="two_soma_mesh_skeleton_decomp.html"
#                           )

    
#     sk.graph_skeleton_and_mesh(other_meshes=[total_branch_meshes[47]],
#                               other_meshes_colors="random",
#                               other_skeletons=[total_branch_skeletons[47]],
#                               other_skeletons_colors="random",
#                               html_path="two_soma_mesh_skeleton_decomp.html")
    
    
    
    
    
    
    
    
    
    
    
    # ********************   Phase C ***************************************
    # PART 3: LAST PART OF ANALYSIS WHERE MAKES CONCEPT GRAPHS
    
    
    print("\n\n\n\n\n****** Phase C ***************\n\n\n\n\n")
    
    
    
    
    
    # ---1) Making concept graphs:

    #getting mesh and skeleton dictionaries
    limb_idx_to_branch_meshes_dict = dict()
    limb_idx_to_branch_skeletons_dict = dict()
    for k in limb_correspondence.keys():
        limb_idx_to_branch_meshes_dict[k] = [limb_correspondence[k][j]["branch_mesh"] for j in limb_correspondence[k].keys()]
        limb_idx_to_branch_skeletons_dict[k] = [limb_correspondence[k][j]["branch_skeleton"] for j in limb_correspondence[k].keys()]      

    #getting the soma dictionaries
    soma_idx_to_mesh_dict = dict()
    for k,v in enumerate(current_mesh_data[0]["soma_meshes"]):
        soma_idx_to_mesh_dict[k] = v

    soma_idx_connectivity = current_mesh_data[0]["soma_to_piece_connectivity"]




    limb_concept_networks,limb_labels = generate_limb_concept_networks_from_global_connectivity(
        limb_idx_to_branch_meshes_dict = limb_idx_to_branch_meshes_dict,
        limb_idx_to_branch_skeletons_dict = limb_idx_to_branch_skeletons_dict,
        soma_idx_to_mesh_dict = soma_idx_to_mesh_dict,
        soma_idx_connectivity = soma_idx_connectivity,
        current_neuron=current_neuron,
        return_limb_labels=True
        )

    #Before go and get concept maps:
    print("Sizes of dictionaries sent")
    for curr_limb in limb_idx_to_branch_skeletons_dict.keys():
        print((len(limb_idx_to_branch_skeletons_dict[curr_limb]),len(limb_idx_to_branch_meshes_dict[curr_limb])))


    print("\n\n Sizes of concept maps gotten back")
    for curr_idx in limb_concept_networks.keys():
        for soma_idx,concept_network in limb_concept_networks[curr_idx].items():
            print(len(np.unique(list(concept_network.nodes()))))
            
    
    
    
    
    
    
    
    
    
    

    
    # ---2) Packaging the data into a dictionary that can be sent to the Neuron class to create the object
    
    #Preparing the data structure to save or use for Neuron class construction

    preprocessed_data = dict(
                            soma_meshes = current_mesh_data[0]["soma_meshes"],
                            soma_to_piece_connectivity = current_mesh_data[0]["soma_to_piece_connectivity"],
                            soma_sdfs = total_soma_list_sdf,
                            insignificant_limbs=insignificant_limbs,
                            non_soma_touching_meshes=non_soma_touching_meshes,
                            inside_pieces=inside_pieces,
                            limb_correspondence=limb_correspondence,
                            limb_concept_networks=limb_concept_networks,
                            limb_labels=limb_labels,
                            limb_meshes=current_mesh_data[0]["branch_meshes"],
                            )

    
    
    print(f"\n\n\n Total processing time = {time.time() - whole_processing_tiempo}")
    
    print(f"returning preprocessed_data = {preprocessed_data}")
    return preprocessed_data
    
    


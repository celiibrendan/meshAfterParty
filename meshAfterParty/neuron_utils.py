

"""
Purpose of this file: To help the development of the neuron object
1) Concept graph methods
2) Preprocessing pipeline for creating the neuron object from a meshs

"""

import soma_extraction_utils as sm
import skeleton_utils as sk
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
import itertools
from tqdm_utils import tqdm

#for meshparty preprocessing
import meshparty_skeletonize as m_sk
import general_utils as gu
import compartment_utils as cu
from meshparty import trimesh_io

# tools for restricting 

# -------------- 7/22 Help Filter Bad Branches ------------------ #
def classify_error_branch(curr_branch,width_to_face_ratio=5):
    curr_width = curr_branch.width
    curr_face_count = len(curr_branch.mesh.faces)
    if curr_width/curr_face_count > width_to_face_ratio:
        return True
    else:
        return False
    
def classify_endpoint_error_branches_from_limb_concept_network(curr_concept_network,**kwargs):
    """
    Purpose: To identify all endpoints of concept graph where the branch meshes/skeleton
    are likely a result of bad skeletonization or meshing:
    
    Applications: Can get rid of these branches later
    
    Pseudocode: 
    1) Get all of the endpoints of the concept network
    2) Get all of the branch objects for the endpoints
    3) Return the idx's of the branch objects that test positive for being an error branch
    """
    
    #1) Get all of the endpoints of the concept network
    end_nodes = xu.get_nodes_of_degree_k(curr_concept_network,1)
    
    #2) Get all of the branch objects for the endpoints
    end_node_branches = [curr_concept_network.nodes[k]["data"] for k in end_nodes]
    
    #3) Return the idx's of the branch objects that test positive for being an error branch
    total_error_end_node_branches = []
    
    for en_idx,e_branch in zip(end_nodes,end_node_branches):
        if classify_error_branch(e_branch):
            total_error_end_node_branches.append(en_idx)
    
    return total_error_end_node_branches
    

# -------------- tools for the concept networks ------------------ #

import copy
import networkx as nx

def whole_neuron_branch_concept_network(input_neuron,
                                  directional=True,
                                 limb_soma_touch_dictionary = None,
                                 print_flag = False):
    
    """
    Purpose: To return the entire concept network with all of the limbs and 
    somas connected of an entire neuron
    
    Arguments:
    input_neuron: neuron object
    directional: If want a directional or undirectional concept_network returned
    limb_soma_touch_dictionary: a dictionary mapping the limb to the starting soma
    you want it to start if directional option is set
    Ex:  {"L1":[0,1]})
    
    
    Pseudocode:  
    1) Get the soma subnetwork from the concept network of the neuron
    2) For each limb network:
    - if directional: 
    a) if no specific starting soma picked --> use the soma with the smallest index as starting one
    - if undirectional
    a2) if undirectional then just choose the concept network
    b) Rename all of the nodes to L#_#
    c) Add the network to the soma/total network and add an edge from the soma to the starting node
    (do so for all)

    3) Then take a subgraph of the concept network based on the nodes you want
    4) Send the subgraph to a function that graphs the networkx graph

    
    """

    

    current_neuron = copy.deepcopy(input_neuron)
    
    if limb_soma_touch_dictionary is None:
        limb_soma_touch_dictionary=dict()
    elif type(limb_soma_touch_dictionary) == dict:
        pass
    elif limb_soma_touch_dictionary == "all":
        limb_soma_touch_dictionary = dict([(limb_idx,xu.get_neighbors(current_neuron.concept_network,limb_idx,int_label=False)) for limb_idx in current_neuron.get_limb_node_names()])
    else:
        raise Exception(f"Recieved invalid input for  limb_soma_touch_dictionary: {limb_soma_touch_dictionary}")

    total_network= nx.DiGraph(current_neuron.concept_network.subgraph(current_neuron.get_soma_node_names()))

    for limb_idx in current_neuron.get_limb_node_names():
        if print_flag:
            print(f"Working on Limb: {limb_idx}")


        if limb_idx in limb_soma_touch_dictionary.keys():
            touching_soma = limb_soma_touch_dictionary[limb_idx]
        else:
            touching_soma = []

        curr_limb_obj = current_neuron.concept_network.nodes[limb_label(limb_idx)]["data"]
        curr_network = None
        if not directional:
            curr_network = curr_limb_obj.concept_network
        else:
            if len(touching_soma) > 0:
                """
                For all somas specified: get the network
                1) if this is first one then just copy the network
                2) if not then get the edges and add to existing network
                """
                for starting_soma in touching_soma:
                    if print_flag:
                        print(f"---Working on soma: {starting_soma}")
                    curr_limb_obj.set_concept_network_directional(starting_soma)
                    soma_specific_network = curr_limb_obj.concept_network_directional
                    if curr_network is None:
                        curr_network = copy.deepcopy(soma_specific_network)
                    else:
                        #get the edges
                        curr_network.add_edges_from(soma_specific_network.edges())

                        #get the specific starting node for that network and add it to the current one
                        #print(f"For limb_idx {limb_idx}, curr_limb_obj.all_concept_network_data = {curr_limb_obj.all_concept_network_data}")
                        matching_concept_network_data = [k for k in curr_limb_obj.all_concept_network_data if 
                                                         ((soma_label(k["starting_soma"]) == starting_soma) or (["starting_soma"] == starting_soma))]

                        if len(matching_concept_network_data) != 1:
                            raise Exception(f"The concept_network data for the starting soma ({starting_soma}) did not have exactly one match: {matching_concept_network_data}")

                        matching_concept_network_dict = matching_concept_network_data[0]
                        curr_starting_node = matching_concept_network_dict["starting_node"]
                        curr_starting_coordinate= matching_concept_network_dict["starting_coordinate"]

                        #set the starting coordinate in the concept network
                        attrs = {curr_starting_node:{"starting_coordinate":curr_starting_coordinate}}
                        if print_flag:
                            print(f"attrs = {attrs}")
                        xu.set_node_attributes_dict(curr_network,attrs)

            else:
                curr_network = curr_limb_obj.concept_network_directional

        #At this point should have the desired concept network

        #print(curr_network.nodes())
        mapping = dict([(k,f"{limb_label(limb_idx)}_{k}") for k in curr_network.nodes()])
        curr_network = nx.relabel_nodes(curr_network,mapping)
        #print(curr_network.nodes())
#         if print_flag:
#             print(f'current network edges = {curr_network["L0_17"],curr_network["L0_20"]}')


        #need to get all connections from soma to limb:
        soma_to_limb_edges = []
        for soma_connecting_dict in curr_limb_obj.all_concept_network_data:
            soma_to_limb_edges.append((soma_label(soma_connecting_dict["starting_soma"]),
                                      f"{limb_label(limb_idx)}_{soma_connecting_dict['starting_node']}"))

        total_network = nx.compose(total_network,curr_network)
        total_network.add_edges_from(soma_to_limb_edges)
        
        if print_flag:
            print(f'current network edges = {total_network["L0_17"],total_network["L0_20"]}')
        
    if directional:
        return nx.DiGraph(total_network)
    
    return total_network



def get_limb_names_from_concept_network(concept_network):
    """
    Purpose: Function that takes in either a neuron object
    or the concept network and returns just the concept network
    depending on the input
    
    """
    return [k for k in concept_network.nodes() if "L" in k]

def get_soma_names_from_concept_network(concept_network):
    """
    Purpose: Function that takes in either a neuron object
    or the concept network and returns just the concept network
    depending on the input
    
    """
    return [k for k in concept_network.nodes() if "S" in k]
    

def return_concept_network(current_neuron):
    """
    Purpose: Function that takes in either a neuron object
    or the concept network and returns just the concept network
    depending on the input
    
    """
    if current_neuron.__class__.__name__ == "Neuron":
        curr_concept_network = current_neuron.concept_network
    #elif type(current_neuron) == type(xu.GraphOrderedEdges()):
    elif current_neuron.__class__.__name__ == "GraphOrderedEdges":
        curr_concept_network = current_neuron
    else:
        exception_string = (f"current_neuron not a Neuron object or Graph Ordered Edges instance: {type(current_neuron)}"
                       f"\n {current_neuron.__class__.__name__}"
                       f"\n {xu.GraphOrderedEdges().__class__.__name__}"
                           f"\n {current_neuron.__class__.__name__ == xu.GraphOrderedEdges().__class__.__name__}")
        print(exception_string)
        raise Exception("")
    return curr_concept_network
    

def convert_limb_concept_network_to_neuron_skeleton(curr_concept_network,check_connected_component=True):
    """
    Purpose: To take a concept network that has the branch 
    data within it to the skeleton for that limb
    
    Pseudocode: 
    1) Get the nodes names of the branches 
    2) Order the node names
    3) For each node get the skeletons into an array
    4) Stack the array
    5) Want to check that skeleton is connected component
    
    Example of how to run: 
    full_skeleton = convert_limb_concept_network_to_neuron_skeleton(recovered_neuron.concept_network.nodes["L1"]["data"].concept_network)
    
    """
    sorted_nodes = np.sort(list(curr_concept_network.nodes()))
    #print(f"sorted_nodes = {sorted_nodes}")
    full_skeleton = sk.stack_skeletons([curr_concept_network.nodes[k]["data"].skeleton for k in sorted_nodes])
    if check_connected_component:
        sk.check_skeleton_one_component(full_skeleton)
    return full_skeleton
    
def get_starting_info_from_concept_network(concept_networks):
    """
    Purpose: To turn a dictionary that maps the soma indexes to a concept map
    into just a list of dictionaries with all the staring information
    
    Ex input:
    concept_networks = {0:concept_network, 1:concept_network,}
    
    Ex output:
    [dict(starting_soma=..,starting_node=..
            starting_endpoints=...,starting_coordinate=...,touching_soma_vertices=...)]
    
    Pseudocode: 
    1) get the soma it's connect to
    2) get the node that has the starting coordinate 
    3) get the endpoints and starting coordinate for that nodes
    """
    
    
    output_dicts = []
    for current_soma,curr_concept_network_list in concept_networks.items():
        for curr_concept_network in curr_concept_network_list:
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

            if "touching_soma_vertices" in curr_concept_network.nodes[starting_node].keys():
                curr_output_dict["touching_soma_vertices"] = curr_concept_network.nodes[starting_node]["touching_soma_vertices"]
            else:
                curr_output_dict["touching_soma_vertices"] = None
                
            #soma starting group
            if "soma_group_idx" in curr_concept_network.nodes[starting_node].keys():
                curr_output_dict["soma_group_idx"] = curr_concept_network.nodes[starting_node]["soma_group_idx"]
            else:
                curr_output_dict["soma_group_idx"] = None
                
            

            curr_output_dict["concept_network"] = curr_concept_network
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
                                touching_soma_vertices=None,
                                soma_group_idx=None,
                                starting_soma=None,
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
        #print("setting touching_soma_vertices 1")
        attrs = {starting_node:{"starting_coordinate":starting_coordinate,
                                "endpoints":neuron.Branch(starting_edge).endpoints,
                               "touching_soma_vertices":touching_soma_vertices,
                                "soma_group_idx":soma_group_idx,
                               "starting_soma":starting_soma}
                                }
        
        xu.set_node_attributes_dict(concept_network,attrs)
        #print(f"Recovered touching vertices after 1 = {xu.get_all_nodes_with_certain_attribute_key(concept_network,'touching_soma_vertices')}")
        
        #add the endpoints 
        return concept_network

    # 0) convert each branch to one segment and build a graph from it
    
    
    # 8-29 debug
    #curr_branch_meshes_downsampled = [sk.resize_skeleton_branch(b,n_segments=1) for b in curr_branch_skeletons]
    curr_branch_meshes_downsampled = []
    for i,b in enumerate(curr_branch_skeletons):
        try:
            curr_branch_meshes_downsampled.append(sk.resize_skeleton_branch(b,n_segments=1))
        except:
            print(f"The following branch {i} could not be downsampled: {b}")
            raise Exception("not downsampled branch")
        
    
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
    attrs = {starting_edge_index:{"starting_coordinate":starting_coordinate,"touching_soma_vertices":touching_soma_vertices,"soma_group_idx":soma_group_idx,"starting_soma":starting_soma}} 
    #print("setting touching_soma_vertices 2")
    xu.set_node_attributes_dict(concept_network,attrs)
    #print(f"Recovered touching vertices after 2 = {xu.get_all_nodes_with_certain_attribute_key(concept_network,'touching_soma_vertices')}")
    
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



""" Older function definition
def generate_limb_concept_networks_from_global_connectivity(
    limb_idx_to_branch_meshes_dict,
    limb_idx_to_branch_skeletons_dict,
    soma_idx_to_mesh_dict,
    soma_idx_connectivity,
    current_neuron,
    return_limb_labels=True
    ): 
"""

def check_concept_network(curr_limb_concept_network,closest_endpoint,
                          curr_limb_divided_skeletons,print_flag=True,
                         return_touching_piece=True):
    recovered_touching_piece = xu.get_nodes_with_attributes_dict(curr_limb_concept_network,dict(starting_coordinate=closest_endpoint))
    if print_flag:
        print(f"recovered_touching_piece = {recovered_touching_piece}")
        print(f"After concept mapping size = {len(curr_limb_concept_network.nodes())}")
    if len(curr_limb_concept_network.nodes()) != len(curr_limb_divided_skeletons):
        raise Exception("The number of nodes in the concept graph and number of branches passed to it did not match\n"
                      f"len(curr_limb_concept_network.nodes())={len(curr_limb_concept_network.nodes())}, len(curr_limb_divided_skeletons)= {len(curr_limb_divided_skeletons)}")
    if nx.number_connected_components(curr_limb_concept_network) > 1:
        raise Exception("There was more than 1 connected components in the concept network")


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
    if return_touching_piece:
        return recovered_touching_piece
            

from pykdtree.kdtree import KDTree #for finding the closest endpoint
def generate_limb_concept_networks_from_global_connectivity(
        limb_correspondence,
        soma_meshes,
        soma_idx_connectivity,
        current_neuron,
        limb_to_soma_starting_endpoints = None,
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
    print("********************************** generate_limb_concept_networks_from_global_connectivity****************************")
    # ------------ 7/17 Added preprocessing Step so can give the function more generic arguments ---------------- #
    #getting mesh and skeleton dictionaries
    limb_idx_to_branch_meshes_dict = dict()
    limb_idx_to_branch_skeletons_dict = dict()
    for k in limb_correspondence.keys():
        limb_idx_to_branch_meshes_dict[k] = [limb_correspondence[k][j]["branch_mesh"] for j in limb_correspondence[k].keys()]
        limb_idx_to_branch_skeletons_dict[k] = [limb_correspondence[k][j]["branch_skeleton"] for j in limb_correspondence[k].keys()]      

    #getting the soma dictionaries
    soma_idx_to_mesh_dict = dict()
    for k,v in enumerate(soma_meshes):
        soma_idx_to_mesh_dict[k] = v




    
    
    
    

    
    
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
                
                """ 9/17: Want to pick the one with the starting endpoint if exists
                Pseudocode: 
                1) Get the starting endpoint if exists
                2) Get the endpoints of all the touching branches
                3) Get the index (if any ) of the branch that has this endpoint in skeleton
                4) Make that the winning index
                
                
                
                """
                if not limb_to_soma_starting_endpoints is None:
                    print("Using new winning piece based on starting coordinate")
                    ideal_starting_coordinate = limb_to_soma_starting_endpoints[limb_idx][soma_idx]
                    touching_pieces_branches = [neuron.Branch(curr_limb_divided_skeletons[k]).endpoints for k in touching_pieces]
                    print("trying to use new find_branch_skeleton_with_specific_coordinate")
                    winning_piece_idx = sk.find_branch_skeleton_with_specific_coordinate(touching_pieces_branches,
                                                                  current_coordinate=ideal_starting_coordinate)[0]
                    
                else:
                    # picking the piece with the most shared vertices
                    len_touch_vertices = [len(k) for k in touching_vertices]
                    winning_piece_idx = np.argmax(len_touch_vertices)
                    
                print(f"winning_piece_idx = {winning_piece_idx}")
                touching_pieces = [touching_pieces[winning_piece_idx]]
                print(f"Winning touching piece = {touching_pieces}")
                touching_pieces_soma_vertices = touching_vertices[winning_piece_idx]
            else:
                touching_pieces_soma_vertices = touching_vertices[0]
            if len(touching_pieces) < 1:
                raise Exception("No touching pieces")
            
            #print out the endpoints of the winning touching piece
            
                
            #3) With the one that is touching the soma, find the enpoints of the skeleton
            print(f"Using touching_pieces[0] = {touching_pieces[0]}")
            touching_branch = neuron.Branch(curr_limb_divided_skeletons[touching_pieces[0]])
            endpoints = touching_branch.endpoints
            
            
            """  OLDER WAY OF FINDING STARTING ENDPOINT WHERE JUST COMPARES TO SOMA CENTER
            print(f"Touching piece endpoints = {endpoints}")
            soma_midpoint = np.mean(curr_soma_mesh.vertices,axis=0)

            #4) Find the closest matching endpoint
            closest_idx = np.argmin([np.linalg.norm(soma_midpoint-k) for k in endpoints])
            closest_endpoint = endpoints[closest_idx]
            
            """
            

            closest_endpoint = None
            if not limb_to_soma_starting_endpoints is None:
                """  # -----------  9/16 -------------- #
                Will pick the starting coordinate that was given if it was on the winning piece
                """
                ideal_starting_coordinate = limb_to_soma_starting_endpoints[limb_idx][soma_idx]
                endpoints_list = endpoints.reshape(-1,3)
                match_result = nu.matching_rows(endpoints_list,ideal_starting_coordinate)
                if len(match_result)>0:
                    closest_endpoint = endpoints_list[match_result[0]]
            if closest_endpoint is None:
                """  # -----------  9/1 -------------- #
                New method for finding 
                1) Build a KDTree of the winning touching piece soma boundary vertices
                2) query the endpoints against the vertices
                3) pick the endpoint that has the closest match
                """
                ex_branch_KDTree = KDTree(touching_pieces_soma_vertices)
                distances,closest_nodes = ex_branch_KDTree.query(endpoints)
                closest_endpoint = endpoints[np.argmin(distances)]
            
            

            
            
            
            print(f"inside inner loop "
             f"len(curr_limb_divided_skeletons) = {len(curr_limb_divided_skeletons)}")
            print(f"closest_endpoint WITH NEW KDTREE METHOD= {closest_endpoint}")
            
            print(f"About to send touching_soma_vertices = {touching_pieces_soma_vertices}")
            curr_limb_concept_network = branches_to_concept_network(curr_limb_divided_skeletons,closest_endpoint,np.array(endpoints).reshape(-1,3),
                                                                   touching_soma_vertices=touching_pieces_soma_vertices)
            
            #print(f"Recovered touching vertices = {xu.get_all_nodes_with_certain_attribute_key(curr_limb_concept_network,'touching_soma_vertices')}")
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

    print(f"\n\n ----- Total time for concept mapping = {time.time() - global_concept_time} ----")
        
        
    #returning from the function    
        
    if return_limb_labels:
        return total_limb_concept_networks,total_limb_labels
    else:
        return total_limb_concept_networks

# ----------------------- End of Concept Networks ------------------------------------- #
    
    

# -----------------------  For the compression of a neuron object ---------------------- #
def find_face_idx_and_check_recovery(original_mesh,submesh_list,print_flag=False):
    debug = False
    if len(submesh_list) == 0:
        if print_flag:
            print("Nothing in submesh_list sent to find_face_idx_and_check_recovery so just returning empty list")
            return []
    submesh_list_face_idx = []
    for i,sm in enumerate(submesh_list):
        
        sm_faces_idx = tu.original_mesh_faces_map(original_mesh=original_mesh, 
                                   submesh=sm,
                               matching=True,
                               print_flag=False,
                                                 exact_match=True)
        submesh_list_face_idx.append(sm_faces_idx)
        if debug:
            print(f"For submesh {i}: sm.faces.shape = {sm.faces.shape}, sm_faces_idx.shape = {sm_faces_idx.shape}")
        
    recovered_submesh_meshes = [original_mesh.submesh([sm_f],append=True,repair=False) for sm_f in submesh_list_face_idx]
    #return recovered_submesh_meshes
    for j,(orig_sm,rec_sm) in enumerate(zip(submesh_list,recovered_submesh_meshes)):
        result = tu.compare_meshes_by_face_midpoints(orig_sm,rec_sm,print_flag=False)
        if not result:
            tu.compare_meshes_by_face_midpoints(orig_sm,rec_sm,print_flag=True)
            raise Exception(f"Submesh {j} was not able to be accurately recovered")
    
    return submesh_list_face_idx
    

import copy


def smaller_preprocessed_data(neuron_object,print_flag=False):
    double_soma_obj = neuron_object
    
    total_compression_time = time.time()
    
    
    
    
    
    # doing the soma recovery more streamlined
    compression_time = time.time()
    soma_meshes_face_idx = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=double_soma_obj.preprocessed_data["soma_meshes"])
    if print_flag:
        print(f"Total time for soma meshes compression = {time.time() - compression_time }")
    compression_time = time.time()
    #insignificant, non_soma touching and inside pieces just mesh pieces ()
    insignificant_limbs_face_idx = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=double_soma_obj.preprocessed_data["insignificant_limbs"])
    not_processed_soma_containing_meshes_face_idx = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=double_soma_obj.preprocessed_data["not_processed_soma_containing_meshes"])

    inside_pieces_face_idx = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=double_soma_obj.preprocessed_data["inside_pieces"])

    non_soma_touching_meshes_face_idx = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=double_soma_obj.preprocessed_data["non_soma_touching_meshes"])
    
    if print_flag:
        print(f"Total time for insignificant_limbs,inside_pieces,non_soma_touching_meshes,not_processed_soma_containing_meshes compression = {time.time() - compression_time }")
    compression_time = time.time()
    
    # recover the limb meshes from the original
    #------------------------------- THERE IS SOME DISCONNECTED IN THE MESH THAT IS IN PREPROCESSED DATA AND THE ACTUAL LIMB MESH ------------------ #
    #------------------------------- MAKES SENSE BECAUSE DOING MESH CORRESPONDENCE AFTERWARDS THAT ALTERS THE MESH BRANCHES A BIT, SO JUST PULL FROM THE MESH_FACES_IDX ------------------ #

    limb_meshes_face_idx = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=double_soma_obj.preprocessed_data["limb_meshes"])
    if print_flag:
        print(f"Total time for limb_meshes compression = {time.time() - compression_time }")
    compression_time = time.time()    
    
    # limb_correspondence can get rid of branch mesh and just recover from branch_face_idx
    
    
    
    
    """
    Pseudocode: 
    1) Want to keep skeleton and width
    2) Generate new branch_face_idx based on the original mesh
    --> later can recover the branch_mesh from the whole neuron mesh and the new branch_face_idx
    --> regenerate the and branch_face_idx from the recovered limb mesh and he recovered mesh


    """
    if print_flag:
        print(f"    Starting Limb Correspondence Compression")
    new_limb_correspondence = copy.deepcopy(double_soma_obj.preprocessed_data["limb_correspondence"])
    

    for k in new_limb_correspondence:
        for j in tqdm(new_limb_correspondence[k]):
            new_limb_correspondence[k][j]["branch_face_idx_whole_neuron"] = find_face_idx_and_check_recovery(original_mesh=double_soma_obj.mesh,
                                                           submesh_list=[new_limb_correspondence[k][j]["branch_mesh"]])[0]

            if "branch_face_idx" in new_limb_correspondence[k][j].keys():
                del new_limb_correspondence[k][j]["branch_face_idx"]
            if "branch_mesh" in new_limb_correspondence[k][j].keys():
                del new_limb_correspondence[k][j]["branch_mesh"]
                
    if print_flag:
        print(f"Total time for new_limb_correspondence compression = {time.time() - compression_time }")
    compression_time = time.time() 
    
    
    # all of the data will be 
    soma_meshes_face_idx

    # soma_to_piece_connectivity is already small dictionary
    double_soma_obj.preprocessed_data["soma_to_piece_connectivity"]
    double_soma_obj.preprocessed_data["soma_sdfs"]
    

    insignificant_limbs_face_idx
    inside_pieces_face_idx
    non_soma_touching_meshes_face_idx

    limb_meshes_face_idx

    new_limb_correspondence

    double_soma_obj.preprocessed_data['limb_labels']
    double_soma_obj.preprocessed_data['limb_concept_networks']
    
    """
    Algorithm for how to save off the following: 
    1) width_new (the dictionary where keyword maps to scalar) #save as dict of dict
    2) width_array (the dictionary where keyword maps to array)# 
    3) spines(list or none): 
    4) branch_labels (list): dict of dict
    
    How to store the width_new (just a dictionary for all of the widths)
    width_new_key = just anraveled
    
    """

    
    computed_attribute_dict = double_soma_obj.get_computed_attribute_data()
    
    #geting the labels data
    labels_lookup =double_soma_obj.get_attribute_dict("labels")

    if "soma_volume_ratios" not in double_soma_obj.preprocessed_data.keys():
        double_soma_obj.preprocessed_data["soma_volume_ratios"] = [double_soma_obj[ll].volume_ratio for ll in double_soma_obj.get_soma_node_names()]
        
    compressed_dict = dict(
                          #saving the original number of faces and vertices to make sure reconstruciton doesn't happen with wrong mesh
                          original_mesh_n_faces = len(double_soma_obj.mesh.faces),
                          original_mesh_n_vertices = len(double_soma_obj.mesh.vertices), 
        
                          soma_meshes_face_idx=soma_meshes_face_idx,

                          soma_to_piece_connectivity=double_soma_obj.preprocessed_data["soma_to_piece_connectivity"],
                          soma_sdfs=double_soma_obj.preprocessed_data["soma_sdfs"],
                          soma_volume_ratios=double_soma_obj.preprocessed_data["soma_volume_ratios"],

                          insignificant_limbs_face_idx=insignificant_limbs_face_idx,
                          not_processed_soma_containing_meshes_face_idx = not_processed_soma_containing_meshes_face_idx,
                          inside_pieces_face_idx=inside_pieces_face_idx,
                          non_soma_touching_meshes_face_idx=non_soma_touching_meshes_face_idx,

                          limb_meshes_face_idx=limb_meshes_face_idx,

                          new_limb_correspondence=new_limb_correspondence,
                            
                          segment_id=double_soma_obj.segment_id,
                          description=double_soma_obj.description,
                          decomposition_type=double_soma_obj.decomposition_type,
            
                          # don't need these any more because will recompute them when decompressing
                          #limb_labels= double_soma_obj.preprocessed_data['limb_labels'],
                          #limb_concept_networks=double_soma_obj.preprocessed_data['limb_concept_networks']
        
                          #new spine/width/labels compression
                          computed_attribute_dict = computed_attribute_dict,
                          
                          #For concept network creation
                          limb_network_stating_info = double_soma_obj.preprocessed_data["limb_network_stating_info"]
        
        
                         
    )
    
    if print_flag:
        print(f"Total time for compression = {time.time() - total_compression_time }")
    
    return compressed_dict

from pathlib import Path
import system_utils as su

def save_compressed_neuron(neuron_object,output_folder,file_name="",return_file_path=False,export_mesh=False):
    output_folder = Path(output_folder)
    
    if file_name == "":
        file_name = f"{neuron_object.segment_id}_{neuron_object.description}"
    
    output_path = output_folder / Path(file_name)
    
    output_path = Path(output_path)
    output_path.parents[0].mkdir(parents=True, exist_ok=True)
    
    inhib_object_compressed_preprocessed_data = smaller_preprocessed_data(neuron_object,print_flag=True)
    compressed_size = su.compressed_pickle(inhib_object_compressed_preprocessed_data,output_path,return_size=True)
    
    print(f"\n\n---Finished outputing neuron at location: {output_path.absolute()}---")
    
    if export_mesh:
        neuron_object.mesh.export(str(output_path.absolute()) +".off")
    
    if return_file_path:
        return output_path
    
    

#For decompressing the neuron

import preprocessing_vp2 as pre
def decompress_neuron(filepath,original_mesh,
                     suppress_output=True):
    if suppress_output:
        print("Decompressing Neuron in minimal output mode...please wait")
    
    with su.suppress_stdout_stderr() if suppress_output else su.dummy_context_mgr():
        
        loaded_compression = su.decompress_pickle(filepath)
        print(f"Inside decompress neuron and decomposition_type = {loaded_compression['decomposition_type']}")

        #creating dictionary that will be used to construct the new neuron object
        recovered_preprocessed_data = dict()

        """
        a) soma_meshes: use the 
        Data: soma_meshes_face_idx 
        Process: use submesh on the neuron mesh for each

        """
        if type(original_mesh) == type(Path()) or type(original_mesh) == str:
            original_mesh = tu.load_mesh_no_processing(original_mesh)

        if len(original_mesh.faces) != loaded_compression["original_mesh_n_faces"]:
            raise Exception(f"Number of faces in mesh used for compression ({loaded_compression['original_mesh_n_faces']})"
                            f" does not match the number of faces in mesh passed to decompress_neuron function "
                            f"({len(original_mesh.faces)})")
        else:
            print("Passed faces original mesh check")

        if len(original_mesh.vertices) != loaded_compression["original_mesh_n_vertices"]:
            raise Exception(f"Number of vertices in mesh used for compression ({loaded_compression['original_mesh_n_vertices']})"
                            f" does not match the number of vertices in mesh passed to decompress_neuron function "
                            f"({len(original_mesh.vertices)})")
        else:
            print("Passed vertices original mesh check")


        recovered_preprocessed_data["soma_meshes"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["soma_meshes_face_idx"]]

        """
        b) soma_to_piece_connectivity
        Data: soma_to_piece_connectivity
        Process: None

        c) soma_sdfs
        Data: soma_sdfs
        Process: None
        """
        recovered_preprocessed_data["soma_to_piece_connectivity"] = loaded_compression["soma_to_piece_connectivity"]
        recovered_preprocessed_data["soma_sdfs"] = loaded_compression["soma_sdfs"]
        if "soma_volume_ratios" in  recovered_preprocessed_data.keys():
            recovered_preprocessed_data["soma_volume_ratios"] = loaded_compression["soma_volume_ratios"]
        else:
            recovered_preprocessed_data["soma_volume_ratios"] = None

        """
        d) insignificant_limbs
        Data: insignificant_limbs_face_idx
        Process: use submesh on the neuron mesh for each

        d) non_soma_touching_meshes
        Data: non_soma_touching_meshes_face_idx
        Process: use submesh on the neuron mesh for each

        d) inside_pieces
        Data: inside_pieces_face_idx
        Process: use submesh on the neuron mesh for each
        """

        recovered_preprocessed_data["insignificant_limbs"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["insignificant_limbs_face_idx"]]
        
        
        
        recovered_preprocessed_data["not_processed_soma_containing_meshes"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["not_processed_soma_containing_meshes_face_idx"]]

        recovered_preprocessed_data["non_soma_touching_meshes"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["non_soma_touching_meshes_face_idx"]]

        recovered_preprocessed_data["inside_pieces"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["inside_pieces_face_idx"]]

        """
        e) limb_meshes
        Data: limb_meshes_face_idx
        Process: use submesh on the neuron mesh for each

        """

        recovered_preprocessed_data["limb_meshes"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["limb_meshes_face_idx"]]


        """

        f) limb_correspondence
        Data: new_limb_correspondence
        Process: 
        -- get branch mesh for each item
        --> later can recover the branch_mesh from the whole neuron mesh and the new branch_face_idx
        -- get branch_face_idx for each itme
        --> regenerate the and branch_face_idx from the recovered limb mesh and he recovered mesh

        """

        new_limb_correspondence = loaded_compression["new_limb_correspondence"]

        for k in new_limb_correspondence:
            print(f"Working on limb {k}")
            for j in tqdm(new_limb_correspondence[k]):
                print(f"  Working on branch {j}")
                
                new_limb_correspondence[k][j]["branch_mesh"] = original_mesh.submesh([new_limb_correspondence[k][j]["branch_face_idx_whole_neuron"]],append=True,repair=False)
                
                try:
                    
                    new_limb_correspondence[k][j]["branch_face_idx"] = tu.original_mesh_faces_map(original_mesh=recovered_preprocessed_data["limb_meshes"][k], 
                                               submesh=new_limb_correspondence[k][j]["branch_mesh"] ,
                                           matching=True,
                                           print_flag=False,
                                           exact_match=True)
                except:
                    #Then try using the stitched meshes
                    #possible_non_touching_meshes = [c for c in recovered_preprocessed_data["non_soma_touching_meshes"] if len(c.faces) == len(new_limb_correspondence[k][j]["branch_mesh"].faces)]
                    possible_non_touching_meshes = [c for c in recovered_preprocessed_data["non_soma_touching_meshes"] if len(c.faces) >= len(new_limb_correspondence[k][j]["branch_mesh"].faces)]
                    found_match = False
                    for zz,t_mesh in enumerate(possible_non_touching_meshes):
                        try:
                            new_limb_correspondence[k][j]["branch_face_idx"] = tu.original_mesh_faces_map(original_mesh=t_mesh, 
                                                   submesh=new_limb_correspondence[k][j]["branch_mesh"] ,
                                               matching=True,
                                               print_flag=False,
                                               exact_match=True)
                            found_match=True
                            break
                        except:
                            print(f"Viable Non soma touching mesh({zz}): {t_mesh} was not a match")
                    if not found_match:
                        raise Exception(f'Could Not find matching faces on decompression of mesh {new_limb_correspondence[k][j]["branch_mesh"]}')
                    


                if "branch_face_idx_whole_neuron" in new_limb_correspondence[k][j].keys():
                    del new_limb_correspondence[k][j]["branch_face_idx_whole_neuron"]

        recovered_preprocessed_data["limb_correspondence"] = new_limb_correspondence


        # ------------------ This is old way of restoring the limb concept networks but won't work now ------------- #
        
        '''
        """
        g) limb_concept_networks, limb_labels:
        Data: All previous data
        Process: Call the funciton that creates the concept_networks using all the data above
        """

        limb_concept_networks,limb_labels = generate_limb_concept_networks_from_global_connectivity(
                limb_correspondence = recovered_preprocessed_data["limb_correspondence"],
                #limb_idx_to_branch_meshes_dict = limb_idx_to_branch_meshes_dict,
                #limb_idx_to_branch_skeletons_dict = limb_idx_to_branch_skeletons_dict,

                soma_meshes=recovered_preprocessed_data["soma_meshes"],
                soma_idx_connectivity=recovered_preprocessed_data["soma_to_piece_connectivity"] ,
                #soma_idx_to_mesh_dict = soma_idx_to_mesh_dict,
                #soma_idx_connectivity = soma_idx_connectivity,

                current_neuron=original_mesh,
                return_limb_labels=True
                )
        '''
        
        # ----------------- ------------- #
        """
        Pseudocode for limb concept networks
        
        
        
        """
        
        limb_network_stating_info = loaded_compression["limb_network_stating_info"]
        
        limb_concept_networks=dict()
        limb_labels=dict()

        for curr_limb_idx,new_limb_correspondence_indiv in new_limb_correspondence.items():
            limb_to_soma_concept_networks = pre.calculate_limb_concept_networks(new_limb_correspondence_indiv,
                                                                                run_concept_network_checks=True,
                                                                               **limb_network_stating_info[curr_limb_idx])   



            limb_concept_networks[curr_limb_idx] = limb_to_soma_concept_networks
            limb_labels[curr_limb_idx]= "Unlabeled"
        

        recovered_preprocessed_data["limb_concept_networks"] = limb_concept_networks
        recovered_preprocessed_data["limb_labels"] = limb_labels


        """
        h) get the segment ids and the original description

        """
        if "computed_attribute_dict" in loaded_compression.keys():
            computed_attribute_dict = loaded_compression["computed_attribute_dict"]
        else:
            computed_attribute_dict = None
        #return computed_attribute_dict

        # Now create the neuron from preprocessed data
        decompressed_neuron = neuron.Neuron(mesh=original_mesh,
                     segment_id=loaded_compression["segment_id"],
                     description=loaded_compression["description"],
                     decomposition_type = loaded_compression["decomposition_type"],
                     preprocessed_data=recovered_preprocessed_data,
                     computed_attribute_dict = computed_attribute_dict,
                     suppress_output=suppress_output,
                                            calculate_spines=False,
                                            
                                           widths_to_calculate=[])
    
    return decompressed_neuron

# --------------  END OF COMPRESSION OF NEURON ---------------- #

# --------------  7/23 To help with visualizations of neuron ---------------- #

def get_whole_neuron_skeleton(current_neuron,
                             check_connected_component=True,
                             print_flag=False):
    """
    Purpose: To generate the entire skeleton with limbs stitched to the somas
    of a neuron object
    
    Example Use: 
    
    total_neuron_skeleton = nru.get_whole_neuron_skeleton(current_neuron = recovered_neuron)
    sk.graph_skeleton_and_mesh(other_meshes=[current_neuron.mesh],
                              other_skeletons = [total_neuron_skeleton])
                              
    Ex 2: 
    nru = reload(nru)
    returned_skeleton = nru.get_whole_neuron_skeleton(recovered_neuron,print_flag=True)
    sk.graph_skeleton_and_mesh(other_skeletons=[returned_skeleton])
    """
    limb_skeletons_total = []
    for limb_idx in current_neuron.get_limb_node_names():
        if print_flag:
            print(f"\nWorking on limb: {limb_idx}")
        curr_limb_obj = current_neuron.concept_network.nodes[limb_idx]["data"]
        #stack the new skeleton pieces with the current skeleton 
        curr_limb_skeleton = curr_limb_obj.get_skeleton(check_connected_component=True)
        if print_flag:
            print(f"curr_limb_skeleton.shape = {curr_limb_skeleton.shape}")
        
        limb_skeletons_total.append(curr_limb_skeleton)

    #get the soma skeletons
    soma_skeletons_total = []
    for soma_idx in current_neuron.get_soma_node_names():
        if print_flag:
            print(f"\nWorking on soma: {soma_idx}")
        #get the soma skeletons
        curr_soma_skeleton = get_soma_skeleton(current_neuron,soma_name=soma_idx)
        
        if print_flag:
            print(f"for soma {soma_idx}, curr_soma_skeleton.shape = {curr_soma_skeleton.shape}")
        
        soma_skeletons_total.append(curr_soma_skeleton)

    total_neuron_skeleton = sk.stack_skeletons(limb_skeletons_total + soma_skeletons_total)

    if check_connected_component:
        sk.check_skeleton_one_component(total_neuron_skeleton)

    return total_neuron_skeleton

def get_soma_skeleton(current_neuron,soma_name):
    """
    Purpose: to return the skeleton for a soma that goes from the 
    soma center to all of the connecting limb
    
    Pseudocode: 
    1) get all of the limbs connecting to the soma (through the concept network)
    2) get the starting coordinate for that soma
    For all of the limbs connected
    3) Make the soma center to that starting coordinate a segment
    

    
    """
    #1) get all of the limbs connecting to the soma (through the concept network)
    limbs_connected_to_soma = xu.get_neighbors(current_neuron.concept_network,soma_name,int_label=False)
    #2) get the starting coordinate for that soma
    curr_soma_center = current_neuron.concept_network.nodes[soma_name]["data"].mesh_center
    
    #For all of the limbs connected
    #3) Make the soma center to that starting coordinate a segment
    soma_skeleton_pieces = []
    for limb_idx in limbs_connected_to_soma:
        curr_limb_obj = current_neuron.concept_network.nodes[limb_idx]["data"]
        
        curr_starting_coordinate = [cn_data["starting_coordinate"] for cn_data in curr_limb_obj.all_concept_network_data
                                                    if f"S{cn_data['starting_soma']}" == soma_name]
#         if len(curr_starting_coordinate) != 1:
#             raise Exception(f"curr_starting_coordinate not exactly one element: {curr_starting_coordinate}")
        
        for curr_endpoint in curr_starting_coordinate:
            new_skeleton_piece = np.vstack([curr_soma_center,curr_endpoint]).reshape(-1,2,3)
            soma_skeleton_pieces.append(new_skeleton_piece)
    
    return sk.stack_skeletons(soma_skeleton_pieces)


# def get_soma_skeleton_for_limb(current_neuron,limb_idx):
#     """
#     Purpose: To get the extra piece of skeleton
#     associated with that limb for all of those soma it connects to
    

#     """
    
#     #
    
#     soma_to_starting_dict = dict()
#     for cn_data in curr_limb_obj.all_concept_network_data:
#         soma_to_starting_dict[cn_data["starting_soma"]] = cn_data["starting_coordinate"]

#     """
#     will generate the new skeleton stitches


#     """
#     new_skeleton_pieces = []
#     for curr_soma,curr_endpoint in soma_to_starting_dict.items():
#         curr_soma_center = current_neuron.concept_network.nodes[f"S{curr_soma}"]["data"].mesh_center
#         #print(f"curr_soma_center = {curr_soma_center}")
#         new_skeleton_piece = np.vstack([curr_soma_center,curr_endpoint]).reshape(-1,2,3)
#         new_skeleton_pieces.append(new_skeleton_piece)
#         #print(f"new_skeleton_piece = {new_skeleton_piece}")

    
#     return new_skeleton_pieces


def soma_label(name_input):
    if type(name_input) == str:
        return name_input
    elif type(name_input) == int:
        return f"S{name_input}"
    else:
        raise Exception(f"Recieved unexpected type ({type(name_input)}) for soma name")

def limb_label(name_input):
    if type(name_input) == str:
        return name_input
    elif type(name_input) == int:
        return f"L{name_input}"
    else:
        raise Exception(f"Recieved unexpected type ({type(name_input)}) for limb name")

    
    
# --------------- 8/5 --------------------------#
def branch_mesh_no_spines(branch):
    """
    Purpose: To return the branch mesh without any spines
    """
    original_mesh_flag = False
    if not branch.spines is None:
        if len(branch.spines) > 0:
            ex_branch_no_spines_mesh = tu.original_mesh_faces_map(branch.mesh,
                                    tu.combine_meshes(branch.spines),
                                   matching=False,
                                   print_flag=False,
                                   match_threshold = 0.001,
                                                            return_mesh=True,
                                                                 )
        else:
            original_mesh_flag = True
    else: 
        original_mesh_flag = True
    
    if original_mesh_flag:
        ex_branch_no_spines_mesh = branch.mesh
        
    return ex_branch_no_spines_mesh

#xu.endpoint_connectivity(end_1,end_2)

import width_utils as wu

# ---------------------- 8/31: For querying and axon searching --------------------------- #
from copy import deepcopy
def branch_skeletal_distance_from_soma(curr_limb,
                                       branch_idx,
                                    somas = None,
                                      dict_return=True,
                                      print_flag=False):
    """
    Purpose: Will find the distance of a branch from the specified somas
    as measured by the skeletal distance
    
    Pseudocode
    1) Make a copy of the current limb
    2) Get all of the somas that will be processed 
    (either specified or by default will )
    3) For each soma, find the skeletal distance from that branch to that soma and save in dictioanry
    4) if not asked to return dictionary then just return the minimum distance
    """
    
    curr_limb_copy =  deepcopy(curr_limb)
    
    #0) Create dictionary that will store all of the results
    return_dict = dict()
    
    #For each directional concept network
    if somas is None:
        touching_somas = [k["starting_soma"] for k in curr_limb_copy.all_concept_network_data]
    else:
        if not nu.is_array_like(somas):
            somas = [somas]
        touching_somas = somas
        
    if print_flag:
        print(f"touching_somas = {touching_somas}")
    
    for sm_start in touching_somas:
        if print_flag:
            print(f"--> Working on soma {sm_start}")
        try:
            curr_limb_copy.set_concept_network_directional(sm_start)
        except:
            raise Exception(f"Limb ({limb_name}) was not connected to soma {sm_start} accordinag to all concept networks")
        
        curr_directional_network = curr_limb_copy.concept_network_directional
        starting_node = curr_limb_copy.current_starting_node
        
        if print_flag:
            print(f"starting_node = {starting_node}")
        
        try:
            curr_shortest_path = nx.shortest_path(curr_directional_network,starting_node,branch_idx)
        except:
            if print_flag:
                print(f"branch_idx {branch_idx} did not have a path to soma {sm}, so making distance np.inf")
            return_dict[sm_start] = np.inf
            continue
            
        path_length = np.sum([sk.calculate_skeleton_distance(curr_directional_network.nodes[k]["data"].skeleton)
                           for k in curr_shortest_path[:-1]])
        
        if print_flag:
            print(f"path_length = {path_length}")
        
        return_dict[sm_start] = path_length
    
    #once have the final dictionary either return the dictionary or the minimum path
    if dict_return:
        return return_dict
    else: #return the minimum path length
        return np.min(list(return_dict.values()))
    

# ------------------------------ 9/1 To help with mesh correspondence -----------------------------------------------------#
import itertools
import general_utils as gu

from trimesh.ray import ray_pyembree

def apply_adaptive_mesh_correspondence_to_neuron(current_neuron,
                                                apply_sdf_filter=False,
                                                n_std_dev=1):

    
    for limb_idx in np.sort(current_neuron.get_limb_node_names()):
        
        ex_limb = current_neuron.concept_network.nodes[limb_idx]["data"]
        if apply_sdf_filter:
            print("Using SDF filter")
            ray_inter = ray_pyembree.RayMeshIntersector(ex_limb.mesh)
            
        
        segment_mesh_faces = dict()
        for branch_idx in np.sort(ex_limb.concept_network.nodes()):
            print(f"---- Working on limb {limb_idx} branch {branch_idx} ------")
            ex_branch = ex_limb.concept_network.nodes[branch_idx]["data"]

            #1) get all the neighbors 1 hop away in connectivity
            #2) Assemble a mesh of all the surrounding neighbors
            one_hop_neighbors = xu.get_neighbors(ex_limb.concept_network,branch_idx)
            if len(one_hop_neighbors) > 0:
                two_hop_neighbors = np.concatenate([xu.get_neighbors(ex_limb.concept_network,k) for k in one_hop_neighbors])
                branches_for_surround = np.unique([branch_idx] + list(one_hop_neighbors) + list(two_hop_neighbors))


                surround_mesh_faces = np.concatenate([ex_limb.concept_network.nodes[k]["data"].mesh_face_idx for k in branches_for_surround])
                surrounding_mesh = ex_limb.mesh.submesh([surround_mesh_faces],append=True,repair=False)

                #3) Send the skeleton and the surrounding mesh to the mesh adaptive distance --> gets back indices
                return_value = cu.mesh_correspondence_adaptive_distance(curr_branch_skeleton=ex_branch.skeleton,
                                                     curr_branch_mesh=surrounding_mesh)

                if len(return_value) == 2:
                    remaining_indices, width = return_value
                    final_limb_indices = surround_mesh_faces[remaining_indices]
                else: #if mesh correspondence couldn't be found
                    print("Mesh correspondence couldn't be found so using defaults")
                    final_limb_indices = ex_branch.mesh_face_idx
                    width = ex_branch.width

            else: #if mesh correspondence couldn't be found
                print("Mesh correspondence couldn't be found so using defaults")
                final_limb_indices = ex_branch.mesh_face_idx
                width = ex_branch.width



            """  How we would get the final mesh  
            branch_mesh_filtered = ex_limb.mesh.submesh([final_limb_indices],append=True,repair=False) 

            """
            #5b) store the width measurement based back in the mesh object
            ex_branch.width_new["adaptive"] = width

            if apply_sdf_filter:
                #---------- New step:  Further filter the limb indices
                new_branch_mesh = ex_limb.mesh.submesh([final_limb_indices],append=True,repair=False)
                new_branch_obj = copy.deepcopy(ex_branch)
                new_branch_obj.mesh = new_branch_mesh
                new_branch_obj.mesh_face_idx = final_limb_indices

                filtered_branch_mesh,filtered_branch_mesh_idx,filtered_branch_sdf_mean= sdf_filter(curr_branch=new_branch_obj,
                                                                                                       curr_limb=ex_limb,
                                                                                                       return_sdf_mean=True,
                                                                                                       ray_inter=ray_inter,
                                                                                                      n_std_dev=n_std_dev)
                final_limb_indices = final_limb_indices[filtered_branch_mesh_idx]
            
            segment_mesh_faces[branch_idx] = final_limb_indices

        #This ends up fixing any conflicts in labeling
        face_lookup = gu.invert_mapping(segment_mesh_faces,total_keys=np.arange(0,len(ex_limb.mesh.faces)))
        #original_labels = set(list(itertools.chain.from_iterable(list(face_lookup.values()))))
        #original_labels = gu.get_unique_values_dict_of_lists(face_lookup)
        original_labels = np.arange(0,len(ex_limb))

        face_coloring_copy = cu.resolve_empty_conflicting_face_labels(curr_limb_mesh = ex_limb.mesh,
                                                                                        face_lookup=face_lookup,
                                                                                        no_missing_labels = list(original_labels),
                                                                     max_submesh_threshold=50000)

        divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(ex_limb.mesh,face_coloring_copy)

        #now reassign the new divided supmeshes
        for branch_idx in ex_limb.concept_network.nodes():
            ex_branch = ex_limb.concept_network.nodes[branch_idx]["data"]

            ex_branch.mesh = divided_submeshes[branch_idx]
            ex_branch.mesh_face_idx = divided_submeshes_idx[branch_idx]
            ex_branch.mesh_center = tu.mesh_center_vertex_average(ex_branch.mesh)
            
            #need to change the preprocessed_data to reflect the change
            limb_idx_used = int(limb_idx[1:])
            current_neuron.preprocessed_data["limb_correspondence"][limb_idx_used][branch_idx]["branch_mesh"] = ex_branch.mesh 
            current_neuron.preprocessed_data["limb_correspondence"][limb_idx_used][branch_idx]["branch_face_idx"] = ex_branch.mesh_face_idx
            
            

# --- 9/2: Mesh correspondence that helps deal with the meshparty data  ----
import trimesh_utils as tu
def sdf_filter(curr_branch,curr_limb,size_threshold=20,
               return_sdf_mean=False,
               ray_inter=None,
              n_std_dev = 1):
    """
    Purpose: to eliminate edge parts of meshes that should
    not be on the branch mesh correspondence
    
    Pseudocode
    The filtering step (Have a size threshold for this maybe?):
    1) Calculate the sdf values for all parts of the mesh
    2) Restrict the faces to only thos under mean + 1.5*std_dev
    3) split the mesh and only keep the biggest one

    Example: 
    
    limb_idx = 0
    branch_idx = 20
    branch_idx = 3
    #branch_idx=36
    filtered_branch_mesh, filtered_branch_mesh_idx = sdf_filter(double_neuron_processed[limb_idx][branch_idx],double_neuron_processed[limb_idx],
                                                               n_std_dev=1)
    filtered_branch_mesh.show()

    """
    

    
    #1) Calculate the sdf values for all parts of the mesh
    ray_trace_width_array = tu.ray_trace_distance(curr_limb.mesh,face_inds=curr_branch.mesh_face_idx,ray_inter=ray_inter)
    ray_trace_width_array_mean = np.mean(ray_trace_width_array[ray_trace_width_array>0])
    #apply the size threshold
    if len(curr_branch.mesh.faces)<20:
        if return_sdf_mean:
            return curr_branch.mesh,np.arange(0,len(curr_branch.mesh_face_idx)),ray_trace_width_array_mean
        else:
            return curr_branch.mesh,np.arange(0,len(curr_branch.mesh_face_idx))
    
    #2) Restrict the faces to only thos under mean + 1.5*std_dev
    ray_trace_mask = ray_trace_width_array < (ray_trace_width_array_mean + n_std_dev*np.std(ray_trace_width_array))
    filtered_mesh = curr_limb.mesh.submesh([curr_branch.mesh_face_idx[ray_trace_mask]],append=True,repair=False)

    
    #3) split the mesh and only keep the biggest one
    filtered_split_meshes, filtered_split_meshes_idx = tu.split(filtered_mesh)
    
    if return_sdf_mean:
        return filtered_split_meshes[0],filtered_split_meshes_idx[0],ray_trace_width_array_mean
    else:
        return filtered_split_meshes[0],filtered_split_meshes_idx[0]
    
    
# --------- 9/9 Helps with splitting the mesh limbs ------------ #
import time
import trimesh_utils as tu
def get_limb_to_soma_border_vertices(current_neuron,print_flag=False):
    """
    Purpose: To create a lookup dictionary indexed by 
    - soma
    - limb name
    The will return the vertex coordinates on the border of the soma and limb

    
    """
    start_time = time.time()

    limb_to_soma_border_by_soma = dict()

    for soma_name in current_neuron.get_soma_node_names():

        soma_idx = int(soma_name[1:])


        curr_soma_mesh = current_neuron[soma_label(soma_idx)].mesh
        touching_limbs = current_neuron.get_limbs_touching_soma(soma_idx)
        touching_limb_objs = [current_neuron[k] for k in touching_limbs]

        touching_limbs_meshes = [k.mesh for k in touching_limb_objs]
        touching_pieces,touching_vertices = tu.mesh_pieces_connectivity(main_mesh=current_neuron.mesh,
                                                central_piece = curr_soma_mesh,
                                                periphery_pieces = touching_limbs_meshes,
                                                                 return_vertices=True,
                                                                return_central_faces=False,
                                                                        print_flag=False
                                                                                         )
        limb_to_soma_border = dict([(k,v) for k,v in zip(np.array(touching_limbs)[touching_pieces],touching_vertices)])
        limb_to_soma_border_by_soma[soma_idx] = limb_to_soma_border
    if print_flag:
        print(time.time() - start_time)
    return limb_to_soma_border_by_soma

def compute_all_concept_network_data_from_limb(curr_limb,current_neuron_mesh,soma_meshes,soma_restriction=None,
                                              print_flag=False):
    ex_limb = curr_limb
    
    curr_limb_divided_meshes = [ex_limb[k].mesh for k in ex_limb.get_branch_names()]
    curr_limb_divided_skeletons = [ex_limb[k].skeleton for k in ex_limb.get_branch_names()]


    """ Old way of doing it which required the neuron
    if soma_restriction is None:
        soma_restriction_names = current_neuron.get_soma_node_names()
    else:
        soma_restriction_names = [soma_label(k) for k in soma_restriction]

    soma_restriction_names_ints = [int(k[1:]) for k in soma_restriction_names]
    soma_mesh_list = [current_neuron.concept_network.nodes[k]["data"].mesh for k in soma_restriction_names]
    """
    
    if soma_restriction is None:
        soma_mesh_list = soma_meshes
        soma_restriction_names_ints = list(np.arange(0,len(soma_mesh_list)))
    else:
        soma_mesh_list = [k for i,k in soma_meshes if i in soma_restriction]
        soma_restriction_names_ints = soma_restriction


    derived_concept_network_data = []
    for soma_idx,soma_mesh in zip(soma_restriction_names_ints,soma_mesh_list):
        periph_filter_time = time.time()

        original_idxs = np.arange(0,len(curr_limb_divided_meshes))


        distances_periphery_to_soma = np.array([tu.closest_distance_between_meshes(soma_mesh,k) for k in curr_limb_divided_meshes])
        periphery_distance_threshold = 2000

        original_idxs = original_idxs[distances_periphery_to_soma<periphery_distance_threshold]
        filtered_periphery_meshes = np.array(curr_limb_divided_meshes)[distances_periphery_to_soma<periphery_distance_threshold]


        touching_pieces,touching_vertices,central_piece_faces = tu.mesh_pieces_connectivity(main_mesh=current_neuron_mesh,
                                    central_piece = soma_mesh,
                                    periphery_pieces = filtered_periphery_meshes,
                                                     return_vertices=True,
                                                    return_central_faces=True
                                                                             )
        if print_flag:
            print(f"Total time for mesh connectivity = {time.time() - periph_filter_time}")
        #print(f"touching_pieces = {original_idxs[touching_pieces[0]]}")
        if len(touching_pieces) > 0:
            touching_pieces = original_idxs[touching_pieces]
            if print_flag:
                print(f"touching_pieces = {touching_pieces}")


            if len(touching_pieces) >= 2:
                if print_flag:
                    print("**More than one touching point to soma, touching_pieces = {touching_pieces}**")
                # picking the piece with the most shared vertices
                len_touch_vertices = [len(k) for k in touching_vertices]
                winning_piece_idx = np.argmax(len_touch_vertices)
                if print_flag:
                    print(f"winning_piece_idx = {winning_piece_idx}")
                touching_pieces = [touching_pieces[winning_piece_idx]]
                if print_flag:
                    print(f"Winning touching piece = {touching_pieces}")
                touching_pieces_soma_vertices = touching_vertices[winning_piece_idx]
            else:
                touching_pieces_soma_vertices = touching_vertices[0]
            if len(touching_pieces) < 1:
                raise Exception("No touching pieces")

            #3) With the one that is touching the soma, find the enpoints of the skeleton
            if print_flag:
                print(f"Using touching_pieces[0] = {touching_pieces[0]}")
            touching_branch = neuron.Branch(curr_limb_divided_skeletons[touching_pieces[0]])
            endpoints = touching_branch.endpoints

            """  # -----------  9/1 -------------- #
            New method for finding 
            1) Build a KDTree of the winning touching piece soma boundary vertices
            2) query the endpoints against the vertices
            3) pick the endpoint that has the closest match
            """
            ex_branch_KDTree = KDTree(touching_pieces_soma_vertices)
            distances,closest_nodes = ex_branch_KDTree.query(endpoints)
            closest_endpoint = endpoints[np.argmin(distances)]

            derived_concept_network_data.append(dict(starting_soma=soma_idx,
                                                    starting_node=touching_pieces[0],
                                                     starting_endpoints=endpoints,
                                                     starting_coordinate=closest_endpoint,
                                                    touching_soma_vertices=touching_pieces_soma_vertices
                                               ))
    return derived_concept_network_data

def error_limb_indexes(neuron_obj):
    return np.where(np.array([len(limb.all_concept_network_data) for limb in neuron_obj])>1)[0]


import neuron #package where can use the Branches class to help do branch skeleton analysis



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

def whole_neuron_branch_concept_network_old(input_neuron,
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
                    # now need to iterate through all touching groups
                    if print_flag:
                        print(f"---Working on soma: {starting_soma}")
                    curr_limb_obj.set_concept_network_directional(starting_soma,soma_group_idx=-1)
                    soma_specific_network = curr_limb_obj.concept_network_directional
                    
                    #Just making sure that curr_network already exists to add things to
                    if curr_network is None:
                        curr_network = copy.deepcopy(soma_specific_network)
                    else:
                        # ---------- Will go through and set the edges and the network data ---------- #
                        
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


def whole_neuron_branch_concept_network(input_neuron,
                                    directional=True,
                                    limb_soma_touch_dictionary = "all",
                                    print_flag = True):

    """
    Purpose: To return the entire concept network with all of the limbs and 
    somas connected of an entire neuron

    Arguments:
    input_neuron: neuron object
    directional: If want a directional or undirectional concept_network returned
    limb_soma_touch_dictionary: a dictionary mapping the limb to the starting soma and soma_idx
    you want visualize if directional is chosen

    This will visualize multiple somas and multiple soma touching groups
    Ex:  {1:[{0:[0,1],1:[0]}]})


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
        #make sure that the limb labels are numbers
        pass
    elif limb_soma_touch_dictionary == "all":
        """
        Pseudocode: 
        Iterate through all of the limbs
            Iterate through all of the soma starting info
                Build the dictionary for all possible touches

        """
        limb_soma_touch_dictionary = limb_to_soma_mapping(current_neuron)
    else:
        raise Exception(f"Recieved invalid input for  limb_soma_touch_dictionary: {limb_soma_touch_dictionary}")

    total_network= nx.DiGraph(current_neuron.concept_network.subgraph(current_neuron.get_soma_node_names()))

    for limb_idx,soma_info_dict in limb_soma_touch_dictionary.items():
        curr_limb = current_neuron[limb_idx]

        curr_network = None
        if not directional:
            curr_network = curr_limb.concept_network
        else:
            for starting_soma,soma_group_info in soma_info_dict.items():
                """
                For all somas specified: get the network
                1) if this is first one then just copy the network
                2) if not then get the edges and add to existing network
                """
                for soma_group_idx in soma_group_info:
                    if print_flag:
                        print(f"---Working on soma: {starting_soma}, group = {soma_group_idx}")
                    curr_limb.set_concept_network_directional(starting_soma,soma_group_idx=soma_group_idx)
                    soma_specific_network = curr_limb.concept_network_directional

                    #Just making sure that curr_network already exists to add things to
                    if curr_network is None:
                        curr_network = copy.deepcopy(soma_specific_network)
                    else:
                        # ---------- Will go through and set the edges and the network data ---------- #

                        #get the edges
                        curr_network.add_edges_from(soma_specific_network.edges())

                        matching_concept_network_dict = curr_limb.get_concept_network_data_by_soma_and_idx(starting_soma,
                                                                                                           soma_group_idx)


                        curr_starting_node = matching_concept_network_dict["starting_node"]
                        curr_starting_coordinate= matching_concept_network_dict["starting_coordinate"]

                        #set the starting coordinate in the concept network
                        attrs = {curr_starting_node:{"starting_coordinate":curr_starting_coordinate}}
                        if print_flag:
                            print(f"attrs = {attrs}")
                        xu.set_node_attributes_dict(curr_network,attrs)
                        
                        


        #At this point should have the desired concept network

        mapping = dict([(k,f"{limb_label(limb_idx)}_{k}") for k in curr_network.nodes()])
        curr_network = nx.relabel_nodes(curr_network,mapping)

        #need to get all connections from soma to limb:
        soma_to_limb_edges = []
        for soma_connecting_dict in curr_limb.all_concept_network_data:
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
                                          no_cycles=True,
                                          suppress_disconnected_errors=False,
                                          verbose=False):
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

        if len(to_be_processed_nodes) == 0:
            break
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
        

    #print(f"incoming_edges_to_node = {incoming_edges_to_node}")
    #6) if the no_cycles option is selected:
    #- for every neruong with multiple neurons in list, choose the one that has the branch width that closest matches

    incoming_lengths = [k for k,v in incoming_edges_to_node.items() if len(v) >= 1]
    
    if not suppress_disconnected_errors:
        if len(incoming_lengths) != len(curr_limb_concept_network.nodes())-1:
            raise Exception("after loop in directed concept graph, not all nodes have incoming edges (except starter node)")

    if no_cycles == True:
        if verbose:
            print("checking and resolving cycles")
        #get the nodes with multiple incoming edges
        multi_incoming = dict([(k,v) for k,v in incoming_edges_to_node.items() if len(v) >= 2])


        if len(multi_incoming) > 0:
            if verbose:
                print("There are loops to resolve and 'no_cycles' parameters set requires us to fix eliminate them")
            #find the mesh widths of all the incoming edges and the current edge

            #if mesh widths are available then go that route
            if not mesh_widths is None:
                if verbose:
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
                    if verbose:
                        print("The keys of the concept graph with 'coordinates' do not match the keys of the edge dictionary")
                        print("Just going to use the first incoming edge by default")
                    for curr_node,incoming_nodes in multi_incoming.items():
                        winning_incoming_node = incoming_nodes[0]
                        incoming_edges_to_node[curr_node] = [winning_incoming_node]
                else: #then have coordinate information
                    if verbose:
                        print("Using coordinate distance to pick the winning node")
                    curr_node_coordinate = node_coordinates_dict[curr_node]
                    incoming_nodes_distance = [np.linalg.norm(node_coordinates_dict[k]- curr_node_coordinate) for k in incoming_nodes]
                    winning_incoming_node = incoming_nodes[np.argmax(incoming_nodes_distance).astype("int")]
                    incoming_edges_to_node[curr_node] = [winning_incoming_node]
        else:
            if verbose:
                print("No cycles to fix")


        #check that all have length of 1
        multi_incoming = dict([(k,v) for k,v in incoming_edges_to_node.items() if len(v) == 1])
        
        if not suppress_disconnected_errors:
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
                             max_iterations= 1000000,
                               verbose=False):
    """
    Will change a list of branches into 
    """
    if verbose:
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
        if verbose:
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
        if verbose:
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
    
    if verbose:
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
            if verbose:
                print("printing out current edge:")
                print(xu.get_node_attributes(branches_graph,node_list=starting_node_edge))
        
    
    for i in range(max_iterations):
        #print(f"==\n\n On iteration {i}==")
        if len(edge_endpoints_to_process) == 0:
            if verbose:
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
    if verbose:
        print(f"starting_node in concept map (that should match the starting edge) = {starting_edge_index}")
    #attrs = {starting_node[0]:{"starting_coordinate":starting_coordinate}} #old way that think uses the wrong starting_node
    attrs = {starting_edge_index:{"starting_coordinate":starting_coordinate,"touching_soma_vertices":touching_soma_vertices,"soma_group_idx":soma_group_idx,"starting_soma":starting_soma}} 
    #print("setting touching_soma_vertices 2")
    xu.set_node_attributes_dict(concept_network,attrs)
    #print(f"Recovered touching vertices after 2 = {xu.get_all_nodes_with_certain_attribute_key(concept_network,'touching_soma_vertices')}")
    
    #want to set all of the edge endpoints on the nodes as well just for a check
    
    
    if verbose:
        print(f"Total time for branches to concept conversion = {time.time() - start_time}\n")
    
    
    # Add back the nodes that were deleted
    if len(duplicate_edge_identifiers) > 0:
        if verbose:
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
                          curr_limb_divided_skeletons,print_flag=False,
                         return_touching_piece=True,
                         verbose=False):
    
    recovered_touching_piece = xu.get_nodes_with_attributes_dict(curr_limb_concept_network,dict(starting_coordinate=closest_endpoint))
    
    
    if verbose:
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
def find_face_idx_and_check_recovery(original_mesh,submesh_list,print_flag=False,check_recovery=True):
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
        
    if check_recovery:
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
        
    if hasattr(double_soma_obj,"original_mesh_idx"):
        original_mesh_idx=double_soma_obj.original_mesh_idx
    else:
        original_mesh_idx = None
    
    soma_names = double_soma_obj.get_soma_node_names()
    
    compressed_dict = dict(
                          #saving the original number of faces and vertices to make sure reconstruciton doesn't happen with wrong mesh
                          original_mesh_n_faces = len(double_soma_obj.mesh.faces),
                          original_mesh_n_vertices = len(double_soma_obj.mesh.vertices), 
        
                          soma_meshes_face_idx=soma_meshes_face_idx,

                          soma_to_piece_connectivity=double_soma_obj.preprocessed_data["soma_to_piece_connectivity"],
                          soma_volumes=[double_soma_obj[k].volume for k in soma_names],
                          soma_sdfs = double_soma_obj.preprocessed_data["soma_sdfs"],
                          soma_volume_ratios=double_soma_obj.preprocessed_data["soma_volume_ratios"],

                          insignificant_limbs_face_idx=insignificant_limbs_face_idx,
                          not_processed_soma_containing_meshes_face_idx = not_processed_soma_containing_meshes_face_idx,
                          glia_faces = double_soma_obj.preprocessed_data["glia_faces"],
                          labels = double_soma_obj.labels,
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
                          limb_network_stating_info = double_soma_obj.preprocessed_data["limb_network_stating_info"],
        
                          #for storing the faces indexed into the original mesh
                          original_mesh_idx=original_mesh_idx
        
        
                         
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
import time
import preprocessing_vp2 as pre
def decompress_neuron(filepath,original_mesh,
                     suppress_output=True,
                     debug_time = False,
                      using_original_mesh=True,
                     #error_on_original_mesh_faces_vertices_mismatch=False
                     ):
    if suppress_output:
        print("Decompressing Neuron in minimal output mode...please wait")
    
    with su.suppress_stdout_stderr() if suppress_output else su.dummy_context_mgr():
        
        decompr_time = time.time()
        
        loaded_compression = su.decompress_pickle(filepath)
        print(f"Inside decompress neuron and decomposition_type = {loaded_compression['decomposition_type']}")

        if debug_time:
            print(f"Decompress pickle time = {time.time() - decompr_time}")
            decompr_time = time.time()
        
        #creating dictionary that will be used to construct the new neuron object
        recovered_preprocessed_data = dict()

        """
        a) soma_meshes: use the 
        Data: soma_meshes_face_idx 
        Process: use submesh on the neuron mesh for each

        """
        if type(original_mesh) == type(Path()) or type(original_mesh) == str:
            if str(Path(original_mesh).absolute())[-3:] == '.h5':
                original_mesh = tu.load_mesh_no_processing_h5(original_mesh)
            else:
                original_mesh = tu.load_mesh_no_processing(original_mesh)
        elif type(original_mesh) == type(trimesh.Trimesh()):
            print("Recieved trimesh as orignal mesh")
        else:
            raise Exception(f"Got an unknown type as the original mesh: {original_mesh}")
            
        if debug_time:
            print(f"Getting mesh time = {time.time() - decompr_time}")
            decompr_time = time.time()
            

        # ------- 1/23 Addition: where using a saved mesh face idx to index into an original mesh ------#
        original_mesh_idx = loaded_compression.get("original_mesh_idx",None) 
        
        if using_original_mesh:
            if original_mesh_idx is None:
                print("The flag for using original mesh was set but no original_mesh_faces_idx stored in compressed data")
            else:
                print(f"Original mesh BEFORE using original_mesh_idx = {original_mesh}")
                original_mesh = original_mesh.submesh([original_mesh_idx],append=True,repair=False)
                print(f"Original mesh AFTER using original_mesh_idx = {original_mesh}")
            
            
        error_on_original_mesh_faces_vertices_mismatch=False
        
        if len(original_mesh.faces) != loaded_compression["original_mesh_n_faces"]:
            warning_string = (f"Number of faces in mesh used for compression ({loaded_compression['original_mesh_n_faces']})"
                            f" does not match the number of faces in mesh passed to decompress_neuron function "
                            f"({len(original_mesh.faces)})")
            if error_on_original_mesh_faces_vertices_mismatch:
                raise Exception(warning_string)
            else:
                print(warning_string)
        else:
            print("Passed faces original mesh check")

        if len(original_mesh.vertices) != loaded_compression["original_mesh_n_vertices"]:
            warning_string = (f"Number of vertices in mesh used for compression ({loaded_compression['original_mesh_n_vertices']})"
                            f" does not match the number of vertices in mesh passed to decompress_neuron function "
                            f"({len(original_mesh.vertices)})")
            
            if error_on_original_mesh_faces_vertices_mismatch:
                raise Exception(warning_string)
            else:
                print(warning_string)
        else:
            print("Passed vertices original mesh check")
            
            
        if debug_time:
            print(f"Face and Vertices check time = {time.time() - decompr_time}")
            decompr_time = time.time()


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
        
        recovered_preprocessed_data["soma_volumes"] = loaded_compression.get("soma_volumes",None)
        
        if "soma_volume_ratios" in  loaded_compression.keys():
            print("using precomputed soma_volume_ratios")
            recovered_preprocessed_data["soma_volume_ratios"] = loaded_compression["soma_volume_ratios"]
        else:
            recovered_preprocessed_data["soma_volume_ratios"] = None
            
        if debug_time:
            print(f"Original Soma mesh time = {time.time() - decompr_time}")
            decompr_time = time.time()

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
        
        
        
        
        if "glia_faces" in loaded_compression.keys():
            curr_glia = loaded_compression["glia_faces"]
        else:
            curr_glia = np.array([])
        
        recovered_preprocessed_data["glia_faces"] = curr_glia
        
        
        if "labels" in loaded_compression.keys():
            curr_labels = loaded_compression["labels"]
        else:
            curr_labels = np.array([])
        
        recovered_preprocessed_data["labels"] = curr_labels
        
        
        

        recovered_preprocessed_data["non_soma_touching_meshes"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["non_soma_touching_meshes_face_idx"]]

        recovered_preprocessed_data["inside_pieces"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["inside_pieces_face_idx"]]
        
        if debug_time:
            print(f"Insignificant and Not-processed and glia time = {time.time() - decompr_time}")
            decompr_time = time.time()

        """
        e) limb_meshes
        Data: limb_meshes_face_idx
        Process: use submesh on the neuron mesh for each

        """

        recovered_preprocessed_data["limb_meshes"] = [original_mesh.submesh([k],append=True,repair=False) for k in loaded_compression["limb_meshes_face_idx"]]

        if debug_time:
            print(f"Limb meshes time = {time.time() - decompr_time}")
            decompr_time = time.time()
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

        if debug_time:
            print(f"Limb Correspondence = {time.time() - decompr_time}")
            decompr_time = time.time()

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
                                                                                limb_network_stating_info[curr_limb_idx],
                                                                                run_concept_network_checks=True,
                                                                               )   



            limb_concept_networks[curr_limb_idx] = limb_to_soma_concept_networks
            limb_labels[curr_limb_idx]= None
        
        if debug_time:
            print(f"calculating limb networks = {time.time() - decompr_time}")
            decompr_time = time.time()

        recovered_preprocessed_data["limb_concept_networks"] = limb_concept_networks
        recovered_preprocessed_data["limb_labels"] = limb_labels
        recovered_preprocessed_data["limb_network_stating_info"] = limb_network_stating_info


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
                                           widths_to_calculate=[],
                                           original_mesh_idx=original_mesh_idx)
        if debug_time:
            print(f"Sending to Neuron Object = {time.time() - decompr_time}")
            decompr_time = time.time()
    
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
    
    for curr_st_data in curr_limb_copy.all_concept_network_data:
        sm_start = curr_st_data["starting_soma"]
        sm_group_start = curr_st_data["soma_group_idx"]
        
        if sm_start not in touching_somas:
            continue
            
        
        if print_flag:
            print(f"--> Working on soma {sm_start}")
        try:
            curr_limb_copy.set_concept_network_directional(sm_start,sm_group_start)
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

def same_soma_multi_touching_limbs(neuron_obj,return_n_touches=False):
    same_soma_multi_touch_limbs = []

    touch_dict = dict()
    for curr_limb_idx, curr_limb in enumerate(neuron_obj):
        if len(curr_limb.all_concept_network_data) > 0:
            touching_somas = [k["starting_soma"] for k in curr_limb.all_concept_network_data]
            soma_mapping = gu.invert_mapping(touching_somas)

            for soma_idx,touch_idxs in soma_mapping.items():
                if len(touch_idxs) > 1:
                    if curr_limb_idx not in touch_dict.keys():
                        touch_dict[curr_limb_idx] = dict()

                    same_soma_multi_touch_limbs.append(curr_limb_idx)
                    touch_dict[curr_limb_idx][soma_idx] = len(touch_idxs)
                    break
                
    if return_n_touches:
        return touch_dict
    else:
        return np.array(same_soma_multi_touch_limbs)
                     

def multi_soma_touching_limbs(neuron_obj):
    multi_soma_touch_limbs = []

    for curr_limb_idx, curr_limb in enumerate(neuron_obj):
        if len(curr_limb.all_concept_network_data) > 0:
            touching_somas = [k["starting_soma"] for k in curr_limb.all_concept_network_data]
            soma_mapping = gu.invert_mapping(touching_somas)
            if len(soma_mapping.keys()) > 1:
                multi_soma_touch_limbs.append(curr_limb_idx)

    return np.array(multi_soma_touch_limbs)

def error_limbs(neuron_obj):
    """
    Purpose: Will return all of the 
    
    """
    multi_soma_limbs = nru.multi_soma_touching_limbs(neuron_obj)
    multi_touch_limbs = nru.same_soma_multi_touching_limbs(neuron_obj)
    return np.unique(np.concatenate([multi_soma_limbs,multi_touch_limbs]))


# ---- 11/20 functions that will help compute statistics of the neuron object -----------



def n_error_limbs(neuron_obj):
    return len(error_limb_indexes(neuron_obj))

def n_somas(neuron_obj):
    return len(neuron_obj.get_soma_node_names())

def n_limbs(neuron_obj):
    return len(neuron_obj.get_limb_node_names())

def n_branches_per_limb(neuron_obj):
    return [len(ex_limb.get_branch_names()) for ex_limb in neuron_obj]

def n_branches(neuron_obj):
    return np.sum(neuron_obj.n_branches_per_limb)

def skeleton_length_per_limb(neuron_obj):
    return [sk.calculate_skeleton_distance(limb.skeleton) for limb in neuron_obj]

def skeletal_length(neuron_obj):
    return np.sum(neuron_obj.skeleton_length_per_limb)


def max_limb_n_branches(neuron_obj):
    if len(neuron_obj.n_branches_per_limb)>0:
        return np.max(neuron_obj.n_branches_per_limb)
    else:
        return None

def max_limb_skeletal_length(neuron_obj):
    if len(neuron_obj.skeleton_length_per_limb) > 0:
        return np.max(neuron_obj.skeleton_length_per_limb)
    else:
        return None

def all_skeletal_lengths(neuron_obj):
    all_skeletal_lengths = []
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            curr_branch_sk_len = sk.calculate_skeleton_distance(curr_branch.skeleton)
            all_skeletal_lengths.append(curr_branch_sk_len)
    return np.array(all_skeletal_lengths)

def median_branch_length(neuron_obj):
    if len(all_skeletal_lengths(neuron_obj))>0:
        return np.round(np.median(all_skeletal_lengths(neuron_obj)),3)
    else:
        return None
    

# -- width data --
def all_medain_mesh_center_widths(neuron_obj):
    all_widths = []
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            all_widths.append(curr_branch.width_new["median_mesh_center"])
    return np.array(all_widths)

def all_no_spine_median_mesh_center_widths(neuron_obj):
    all_widths = []
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            all_widths.append(curr_branch.width_new["no_spine_median_mesh_center"])
    return np.array(all_widths)

def width_median(neuron_obj):
    if len(all_medain_mesh_center_widths(neuron_obj)) > 0:
        return np.round(np.median(all_medain_mesh_center_widths(neuron_obj)),3)
    else:
        return None

def width_no_spine_median(neuron_obj):
    if len(all_no_spine_median_mesh_center_widths(neuron_obj)) > 0:
        return np.round(np.median(all_no_spine_median_mesh_center_widths(neuron_obj)),3)
    else:
        return None

def width_perc(neuron_obj,perc=90):
    if len(all_medain_mesh_center_widths(neuron_obj)) > 0:
        return np.round(np.percentile(all_medain_mesh_center_widths(neuron_obj),perc),3)
    else:
        return None

def width_no_spine_perc(neuron_obj,perc=90):
    if len(all_no_spine_median_mesh_center_widths(neuron_obj)) > 0:
        return np.round(np.percentile(all_no_spine_median_mesh_center_widths(neuron_obj),perc),3)
    else:
        return None



# -- spine data --

def n_spines(neuron_obj):
    if neuron_obj.spines is None:
        return 0
    else:
        return len(neuron_obj.spines)

def spine_density(neuron_obj):
    skeletal_length = neuron_obj.skeletal_length
    if skeletal_length > 0:
        spine_density = neuron_obj.n_spines/skeletal_length
    else:
        spine_density = 0
    return spine_density

def spines_per_branch(neuron_obj):
    if neuron_obj.n_branches > 0:
        spines_per_branch = neuron_obj.n_spines/neuron_obj.n_branches
    else:
        spines_per_branch = 0
    return spines_per_branch
    
def n_spine_eligible_branches(neuron_obj):
    n_spine_eligible_branches = 0
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            if not curr_branch.spines is None:
                n_spine_eligible_branches += 1
    return n_spine_eligible_branches

def spine_eligible_branch_lengths(neuron_obj):
    spine_eligible_branch_lengths = []
    for curr_limb in neuron_obj:
        for curr_branch in curr_limb:
            if not curr_branch.spines is None:
                curr_branch_sk_len = sk.calculate_skeleton_distance(curr_branch.skeleton)
                spine_eligible_branch_lengths.append(curr_branch_sk_len)
    return spine_eligible_branch_lengths

def skeletal_length_eligible(neuron_obj):
    return np.round(np.sum(neuron_obj.spine_eligible_branch_lengths),3)

def spine_density_eligible(neuron_obj):
    #spine eligible density and per branch
    if neuron_obj.skeletal_length_eligible > 0:
        spine_density_eligible = neuron_obj.n_spines/neuron_obj.skeletal_length_eligible
    else:
        spine_density_eligible = 0
    
    return spine_density_eligible

def spines_per_branch_eligible(neuron_obj):
    if neuron_obj.n_spine_eligible_branches > 0:
        spines_per_branch_eligible = np.round(neuron_obj.n_spines/neuron_obj.n_spine_eligible_branches,3)
    else:
        spines_per_branch_eligible = 0
    
    return spines_per_branch_eligible


# ------- all the spine volume stuff -----------
def total_spine_volume(neuron_obj):
    if neuron_obj.n_spines > 0:
        spines_vol = np.array(neuron_obj.spines_volume)
        return np.sum(spines_vol)
        
    else:
        return 0

def spine_volume_median(neuron_obj):
    spines_vol = np.array(neuron_obj.spines_volume)
    if neuron_obj.n_spines > 0:
        #spine_volume_median
        valid_spine_vol = spines_vol[spines_vol>0]

        if len(valid_spine_vol) > 0:
            spine_volume_median = np.median(valid_spine_vol)
        else:
            spine_volume_median = 0
        
        return spine_volume_median
        
    else:
        return 0
    
def spine_volume_density(neuron_obj):
    if neuron_obj.n_spines > 0:
        if neuron_obj.skeletal_length_eligible > 0:
            spine_volume_density_eligible = neuron_obj.total_spine_volume/neuron_obj.skeletal_length
        else:
            spine_volume_density_eligible = 0
        
        return spine_volume_density_eligible
        
    else:
        return 0


def spine_volume_density_eligible(neuron_obj):
    if neuron_obj.n_spines > 0:
        if neuron_obj.skeletal_length > 0:
            spine_volume_density = neuron_obj.total_spine_volume/neuron_obj.skeletal_length_eligible
        else:
            spine_volume_density = 0
        
        return spine_volume_density
        
    else:
        return 0
    
def spine_volume_per_branch_eligible(neuron_obj):
    if neuron_obj.n_spines > 0:
        if neuron_obj.n_spine_eligible_branches > 0:
            spine_volume_per_branch_eligible = neuron_obj.total_spine_volume/neuron_obj.n_spine_eligible_branches
        else:
            spine_volume_per_branch_eligible = 0
        
        return spine_volume_per_branch_eligible
        
    else:
        return 0
    
    
# -------------- 11 / 26 To help with erroring------------------------------#

import copy
import numpy as np

def align_and_restrict_branch(base_branch,
                              common_endpoint=None,
                              width_name= "no_spine_median_mesh_center",
                             offset=500,
                             comparison_distance=2000,
                             skeleton_segment_size=1000,
                              verbose=False,
                             ):
    

    #Now just need to do the resizing (and so the widths calculated will match this)
    base_skeleton_ordered = sk.resize_skeleton_branch(base_branch.skeleton,skeleton_segment_size)

    if not common_endpoint is None:
        #figure out if need to flip or not:
        if np.array_equal(common_endpoint,base_skeleton_ordered[-1][-1]):

            base_width_ordered = np.flip(base_branch.width_array[width_name])
            base_skeleton_ordered = sk.flip_skeleton(base_skeleton_ordered)
            flip_flag = True
            if verbose:
                print("Base needs flipping")
                print(f"Skeleton after flip = {base_skeleton_ordered}")
        elif np.array_equal(common_endpoint,base_skeleton_ordered[0][0]):
            base_width_ordered = base_branch.width_array[width_name]
            flip_flag = False
        else:
            raise Exception("No matching endpoint")
    else:
        base_width_ordered = base_branch.width_array[width_name]
        
    # apply the cutoff distance
    if verbose:
        print(f"Base offset = {offset}")
        
    
    (skeleton_minus_buffer,
     offset_indexes,
     offset_success) = sk.restrict_skeleton_from_start(base_skeleton_ordered,
                                                                    offset,
                                                                     subtract_cutoff=True)
   
    
    base_final_skeleton = None
    base_final_indexes = None

    if offset_success:
        
        (skeleton_comparison,
         comparison_indexes,
         comparison_success) = sk.restrict_skeleton_from_start(skeleton_minus_buffer,
                                                                        comparison_distance,
                                                                         subtract_cutoff=False)
        
        if comparison_success:
            if verbose:
                print("Base: Long enough for offset and comparison length")
            base_final_skeleton = skeleton_comparison
            base_final_indexes = offset_indexes[comparison_indexes]

        else:
            if verbose:
                print("Base: Passed the offset phase but was not long enough for comparison")
    else:
        if verbose:
            print("Base: Was not long enough for offset")


    if base_final_skeleton is None:
        if verbose:
            print("Base: Not using offset ")
        (base_final_skeleton,
         base_final_indexes,
         _) = sk.restrict_skeleton_from_start(base_skeleton_ordered,
                                                                        comparison_distance,
                                                                         subtract_cutoff=False)
        

    
    base_final_widths = base_width_ordered[np.clip(base_final_indexes,0,len(base_width_ordered)-1)]
    base_final_seg_lengths = sk.calculate_skeleton_segment_distances(base_final_skeleton,cumsum=False)
    
    return base_final_skeleton,base_final_widths,base_final_seg_lengths

import copy
def branch_boundary_transition(curr_limb,
                              edge,
                              width_name= "no_spine_median_mesh_center",
                              offset=500,
                              comparison_distance=2000,
                              skeleton_segment_size=1000,
                              return_skeletons=True,
                              verbose=False):
    """
    Purpose: Will find the boundary skeletons and width average at the boundary
    with some specified boundary skeletal length (with an optional offset)


    """

    base_node = edge[-1]
    upstream_node= edge[0]
    upstream_node_original = upstream_node

    base_branch = curr_limb[base_node]
    upstream_branch = curr_limb[upstream_node]


    # 0) make sure the two nodes are connected in the concept network
    if base_node not in xu.get_neighbors(curr_limb.concept_network,upstream_node):
        raise Exception(f"base_node ({base_node}) and upstream_node ({upstream_node}) are not connected in the concept network")

    # ----- Part 1: Do the processing on the base node -------------- #
    common_endpoint = sk.shared_endpoint(base_branch.skeleton,upstream_branch.skeleton)
    common_endpoint_original = copy.copy(common_endpoint)
    if verbose:
        print(f"common_endpoint = {common_endpoint}")
    
    (base_final_skeleton,
    base_final_widths,
    base_final_seg_lengths) = nru.align_and_restrict_branch(base_branch,
                              common_endpoint=common_endpoint,
                                 width_name=width_name,
                             offset=offset,
                             comparison_distance=comparison_distance,
                             skeleton_segment_size=skeleton_segment_size,
                              verbose=verbose,
                             )
    
    
    
    
    
    

    # ----- Part 2: Do the processing on the upstream nodes -------------- #
    upstream_offset = offset
    upstream_comparison = comparison_distance
    upstream_node = edge[0]
    previous_node = edge[1]
    upstream_skeleton = []
    upstream_seg_lengths = []
    upstream_seg_widths = []

    count = 0
    while upstream_comparison > 0:
        """
        Pseudocode:
        1) Get shared endpoint of upstream and previous node
        2) resize the upstream skeleton to get it ordered and right scale of width
        3) Flip the skeleton and width array if needs to be flipped
        4) if current offset is greater than 0, then restrict skeelton to offset:
        5a) if it was not long enough:
            - subtact total length from buffer
        5b) If successful:
            - restrit skeleton by comparison distance
            - Add skeleton, width and skeelton lengths to list
            - subtract new distance from comparison distance
            - if comparison distance is 0 or less then break
        6)  change out upstream node and previous node (because at this point haven't broken outside loop)

        """
        if verbose:
            print(f"--- Upstream iteration: {count} -----")
        prev_branch = curr_limb[previous_node]
        upstream_branch = curr_limb[upstream_node]

        #1) Get shared endpoint of upstream and previous node
        common_endpoint = sk.shared_endpoint(prev_branch.skeleton,upstream_branch.skeleton)

        #2) resize the upstream skeleton to get it ordered and right scale of width
        upstream_skeleton_ordered = sk.resize_skeleton_branch(upstream_branch.skeleton,skeleton_segment_size)
        if verbose:
            print(f"upstream_skeleton_ordered {sk.calculate_skeleton_distance(upstream_skeleton_ordered)} = {upstream_skeleton_ordered}")
            
        
          # ----------- 1 /5 : To prevent from erroring when indexing into width
#         #accounting for the fact that the skeleton might be a little longer thn the width array now
#         upstream_width = upstream_branch.width_array[width_name]
#         extra_width_segment = [upstream_width[-1]]*(len(upstream_skeleton_ordered)-len(upstream_width))
#         upstream_width = np.hstack([upstream_width,extra_width_segment])
         

        #3) Flip the skeleton and width array if needs to be flipped
        if np.array_equal(common_endpoint,upstream_skeleton_ordered[-1][-1]):
            upstream_width_ordered = np.flip(upstream_branch.width_array[width_name])
            upstream_skeleton_ordered = sk.flip_skeleton(upstream_skeleton_ordered)
            flip_flag = True
        elif np.array_equal(common_endpoint,upstream_skeleton_ordered[0][0]):
            upstream_width_ordered = upstream_branch.width_array[width_name]
            flip_flag = False
        else:
            raise Exception("No matching endpoint")

            
        if verbose: 
            print(f"flip_flag = {flip_flag}")
            print(f"upstream_offset = {upstream_offset}")

        #4) if current offset is greater than 0, then restrict skeelton to offset:
        if upstream_offset > 0:
            if verbose:
                print("Restricting to offset")
            (skeleton_minus_buffer,
             offset_indexes,
             offset_success) = sk.restrict_skeleton_from_start(upstream_skeleton_ordered,
                                                                            upstream_offset,
                                                                             subtract_cutoff=True)
        else:
            if verbose:
                print("Skipping the upstream offset because 0")
            skeleton_minus_buffer = upstream_skeleton_ordered
            offset_indexes = np.arange(len(upstream_skeleton_ordered))
            offset_success = True
        
        
        #print(f"skeleton_minus_buffer {sk.calculate_skeleton_distance(skeleton_minus_buffer)} = {skeleton_minus_buffer}")

        """
        5a) if it was not long enough:
        - subtact total length from buffer
        """
        if not offset_success:
            upstream_offset -= sk.calculate_skeleton_distance(upstream_skeleton_ordered)
            if verbose:
                print(f"Subtracting the offset was not successful so changing to {upstream_offset} and reiterating")
        else:
            """
            5b) If successful:
            - restrit skeleton by comparison distance
            - Add skeleton, width and skeelton lengths to list
            - subtract new distance from comparison distance
            - if comparison distance is 0 or less then break

            """
            #making sure the upstream offset is 0 if we were successful
            upstream_offset = 0
            
            if verbose:
                print(f"After subtracting the offset the length is: {sk.calculate_skeleton_distance(skeleton_minus_buffer)}")

            #- restrit skeleton by comparison distance
            (skeleton_comparison,
             comparison_indexes,
             comparison_success) = sk.restrict_skeleton_from_start(skeleton_minus_buffer,
                                                                            upstream_comparison,
                                                                             subtract_cutoff=False)
            #- Add skeleton, width and skeelton lengths to list
            upstream_skeleton.append(skeleton_comparison)
            upstream_seg_lengths.append(sk.calculate_skeleton_segment_distances(skeleton_comparison,cumsum=False))

            
            upstream_indices = offset_indexes[comparison_indexes]
            upstream_seg_widths.append(upstream_width_ordered[np.clip(upstream_indices,0,len(upstream_width_ordered)-1) ])

            # - subtract new distance from comparison distance
            upstream_comparison -= sk.calculate_skeleton_distance(skeleton_comparison)

            if comparison_success:
                if verbose:
                    print(f"Subtracting the comparison was successful and exiting")
                break
            else:
                if verbose:
                    print(f"Subtracting the comparison was not successful so changing to {upstream_comparison} and reiterating")

        #6)  change out upstream node and previous node (because at this point haven't broken outside loop)
        previous_node = upstream_node
        upstream_node = xu.upstream_node(curr_limb.concept_network_directional,upstream_node)

        if verbose:
            print(f"New upstream_node = {upstream_node}")

        if upstream_node is None:
            if verbose:
                print("Breaking because hit None upstream node")
            break

        count += 1

    upstream_final_skeleton = sk.stack_skeletons(upstream_skeleton)
    if verbose:
        print(f"upstream_final_skeleton = {upstream_final_skeleton}")

    # Do a check at the very end and if no skeleton then just take that branches
    if len(upstream_final_skeleton) <= 0:
        print("No upstream skeletons so doing backup")
        resize_sk = sk.resize_skeleton_branch(curr_limb[upstream_node_original].skeleton,
                                                       skeleton_segment_size)
        upstream_skeleton = [resize_sk]
        upstream_seg_lengths = [sk.calculate_skeleton_segment_distances(resize_sk,cumsum=False)]
        upstream_seg_widths = [curr_limb[upstream_node_original].width_array[width_name]]
        
        (upstream_final_skeleton,
         upstream_final_widths,
        upstream_final_seg_lengths) = nru.align_and_restrict_branch(curr_limb[upstream_node_original],
                                  common_endpoint=common_endpoint_original,
                                width_name=width_name,
                                 offset=offset,
                                 comparison_distance=comparison_distance,
                                 skeleton_segment_size=skeleton_segment_size,
                                  verbose=verbose,
                                 )
    else:
        upstream_final_seg_lengths = np.concatenate(upstream_seg_lengths)
        upstream_final_widths = np.concatenate(upstream_seg_widths)




    #Final results
    base_final_skeleton
    base_final_widths
    base_final_seg_lengths

    upstream_skeleton 
    upstream_seg_lengths 
    upstream_seg_widths

    base_final_skeleton
    

    base_width_average = nu.average_by_weights(weights = base_final_seg_lengths,
                                values = base_final_widths)
    upstream_width_average = nu.average_by_weights(weights = upstream_final_seg_lengths,
                            values = upstream_final_widths)

    if return_skeletons:
        return upstream_width_average,base_width_average,upstream_final_skeleton,base_final_skeleton
    else:
        return upstream_width_average,base_width_average
    

def find_parent_child_skeleton_angle(curr_limb_obj,
                            child_node,   
                            parent_node=None,
                           comparison_distance=3000,
                            offset=0,
                           verbose=False,
                           check_upstream_network_connectivity=True):
    
    if parent_node is None:
        parent_node = xu.upstream_node(curr_limb_obj.concept_network_directional,child_node)
        
    # -------Doing the parent calculation---------
    parent_child_edge = [parent_node,child_node]

    up_width,d_width,up_sk,d_sk = branch_boundary_transition(curr_limb_obj,
                                      edge=parent_child_edge,
                                      comparison_distance = comparison_distance,
                                    offset=offset,
                                    verbose=False,               
                                    check_upstream_network_connectivity=check_upstream_network_connectivity)
    up_sk_flipped = sk.flip_skeleton(up_sk)

    up_vec = up_sk_flipped[-1][-1] - up_sk_flipped[0][0] 
    d_vec_child = d_sk[-1][-1] - d_sk[0][0]

    parent_child_angle = np.round(nu.angle_between_vectors(up_vec,d_vec_child),2)

    if verbose:
        print(f"parent_child_angle = {parent_child_angle}")
        
    return parent_child_angle    



def find_sibling_child_skeleton_angle(curr_limb_obj,
                            child_node,
                            parent_node=None,
                           comparison_distance=3000,
                            offset=0,
                           verbose=False):
    
    
    # -------Doing the parent calculation---------
    if parent_node is None:
        parent_node = xu.upstream_node(curr_limb_obj.concept_network_directional,child_node)
        
    parent_child_edge = [parent_node,child_node]

    up_width,d_width,up_sk,d_sk = branch_boundary_transition(curr_limb_obj,
                                      edge=parent_child_edge,
                                      comparison_distance = comparison_distance,
                                    offset=offset,
                                    verbose=False)
    
    d_vec_child = d_sk[-1][-1] - d_sk[0][0]

    # -------Doing the child calculation---------
    sibling_nodes = xu.sibling_nodes(curr_limb_obj.concept_network_directional,
                                    child_node)
    
    sibl_angles = dict()
    for s_n in sibling_nodes:
        sibling_child_edge = [parent_node,s_n]

        up_width,d_width,up_sk,d_sk = branch_boundary_transition(curr_limb_obj,
                                          edge=sibling_child_edge,
                                          comparison_distance = comparison_distance,
                                        offset=offset,
                                        verbose=False)

        up_vec = up_sk[-1][-1] - up_sk[0][0] 
        d_vec_sibling = d_sk[-1][-1] - d_sk[0][0]

        sibling_child_angle = np.round(nu.angle_between_vectors(d_vec_child,d_vec_sibling),2)
        
        sibl_angles[s_n] = sibling_child_angle
        
    return sibl_angles
    

def all_concept_network_data_to_dict(all_concept_network_data):
    return_dict = dict()
    for st_info in all_concept_network_data:
        curr_soma_idx = st_info["starting_soma"]
        curr_soma_group_idx = st_info["soma_group_idx"]
        curr_endpoint = st_info["starting_coordinate"]
        curr_touching_soma_vertices = st_info["touching_soma_vertices"]
        
        if curr_soma_idx not in return_dict.keys():
            return_dict[curr_soma_idx] = dict()
        
        return_dict[curr_soma_idx][curr_soma_group_idx] = dict(touching_verts=curr_touching_soma_vertices,
                                                         endpoint=curr_endpoint
                                                        )
        

            
    return return_dict
            
    
def limb_to_soma_mapping(current_neuron):
    """
    Purpose: Will create a mapping of 
    limb --> soma_idx --> list of soma touching groups
    
    """
    limb_soma_touch_dictionary = dict()
    for curr_limb_idx,curr_limb in enumerate(current_neuron):
        limb_soma_touch_dictionary[curr_limb_idx] = dict()
        for st_info in curr_limb.all_concept_network_data:
            curr_soma_idx = st_info["starting_soma"]
            curr_soma_group_idx = st_info["soma_group_idx"]
            if curr_soma_idx not in limb_soma_touch_dictionary[curr_limb_idx].keys():
                limb_soma_touch_dictionary[curr_limb_idx][curr_soma_idx] = []
            limb_soma_touch_dictionary[curr_limb_idx][curr_soma_idx].append(curr_soma_group_idx)
            
    return limb_soma_touch_dictionary

    
    
def all_starting_dicts_by_soma(curr_limb,soma_idx):
    return [k for k in curr_limb.all_concept_network_data if k["starting_soma"] == soma_idx]
def all_starting_attr_by_limb_and_soma(curr_limb,soma_idx,attr="starting_node"):
    starting_dicts = all_starting_dicts_by_soma(curr_limb,soma_idx)
    return [k[attr] for k in starting_dicts]

def convert_int_names_to_string_names(limb_names,start_letter="L"):
    return [f"{start_letter}{k}" for k in limb_names]

def convert_string_names_to_int_names(limb_names):
    return [int(k[1:]) for k in limb_names]

def get_limb_string_name(limb_idx,start_letter="L"):
    if type(limb_idx) == int:
        return f"{start_letter}{limb_idx}" 
    elif type(limb_idx) == str:
        return limb_idx
    else:
        raise Exception("Not int or string input")
        
def get_limb_int_name(limb_name):
    if type(limb_name) == int:
        return limb_name
    elif type(limb_name) == str:
        return int(limb_name[1:])
    else:
        raise Exception("Not int or string input")

import neuron_visualizations as nviz
def filter_limbs_below_soma_percentile(neuron_obj,
                                        above_percentile = 70,
                                         return_string_names=True,
                                       visualize_remianing_neuron=False,
                                        verbose = True):
    """
    Purpose: Will only keep those limbs that have 
    a mean touching vertices lower than the soma faces percentile specified
    
    Pseudocode: 
    1) Get the soma mesh
    2) Get all of the face midpoints
    3) Get only the y coordinates of the face midpoints  and turn negative
    4) Get the x percentile of those y coordinates
    5) Get all those faces above that percentage
    6) Get those faces as a submesh and show

    -- How to cancel out the the limbs

    """
    keep_limb_idx = []
    for curr_limb_idx,curr_limb in enumerate(neuron_obj):

        touching_somas = curr_limb.touching_somas()

        keep_limb = False
        for sm_idx in touching_somas:
            if not keep_limb :
                sm_mesh = neuron_obj[f"S{sm_idx}"].mesh

                tri_centers_y = -sm_mesh.triangles_center[:,1]
                perc_y_position = np.percentile(tri_centers_y,above_percentile)


                """ Don't need this: just for verification that was working with soma
                kept_faces = np.where(tri_centers_y <= perc_y_position)[0]

                soma_top = sm_mesh.submesh([kept_faces],append=True)
                """

                """
                Pseudocode for adding limb as possible:
                1) Get all starting dictionaries for that soma
                For each starting dict:
                a) Get the mean of the touching_soma_vertices (and turn negative)
                b) If mean is less than the perc_y_position then set keep_limb to True and break


                """
                all_soma_starting_dicts = all_starting_dicts_by_soma(curr_limb,sm_idx)
                for j,curr_start_dict in enumerate(all_soma_starting_dicts):
                    if verbose:
                        print(f"Working on touching group {j}")

                    t_verts_mean = -1*np.mean(curr_start_dict["touching_soma_vertices"][:,1])

                    if t_verts_mean <= perc_y_position:
                        if verbose:
                            print("Keeping limb because less than y position")
                        keep_limb = True
                        break

                if keep_limb:
                    break
                    
        #decide whether or not to keep limb
        if keep_limb:
            if verbose:
                print(f"Keeping Limb {curr_limb_idx}")
            
            keep_limb_idx.append(curr_limb_idx)
            
    if visualize_remianing_neuron:
        remaining_limbs = convert_int_names_to_string_names(keep_limb_idx)
        ret_col = nviz.visualize_neuron(neuron_obj,
                     visualize_type=["mesh","skeleton"],
                     limb_branch_dict=dict([(k,"all") for k in remaining_limbs]),
                     return_color_dict=True)
            
    if verbose:
        print(f"\n\nTotal removed Limbs = {np.delete(np.arange(len(neuron_obj.get_limb_node_names())),keep_limb_idx)}")
    if return_string_names:
        return convert_int_names_to_string_names(keep_limb_idx)
    else:
        return keep_limb_idx

def limb_branch_dict_to_faces(neuron_obj,limb_branch_dict):
    """
    Purpose: To return the face indices of the main
    mesh that correspond to the limb/branches indicated by dictionary
    
    Pseudocode: 
    0) Have a final face indices list
    
    Iterate through all of the limbs
        Iterate through all of the branches
            1) Get the original indices of the branch on main mesh
            2) Add to the list
            
    3) Concatenate List and return
    
    ret_val = nru.limb_branch_dict_to_faces(neuron_obj,dict(L1=[0,1,2]))
    """
    final_face_indices = []
    
    for limb_name,branch_names in limb_branch_dict.items():
        
        all_branch_meshes = [neuron_obj[limb_name][k].mesh for k in branch_names]
        
        if len(all_branch_meshes)>0:
            match_faces = tu.original_mesh_faces_map(neuron_obj.mesh,
                                                        all_branch_meshes,
                                                           matching=True,
                                                           print_flag=False)
        else:
            match_faces = []
        
        final_face_indices.append(match_faces)
    
    if len(final_face_indices)>0:
        match_faces_idx = np.concatenate(final_face_indices).astype("int")
    else:
        match_faces_idx = np.array([])
        
    return match_faces_idx
 
    
    
def skeleton_touching_branches(limb_obj,branch_idx,
                              return_endpoint_groupings=True):
    """
    Purpose: Can find all the branch numbers
    that touch a certain branch object based on the skeleton endpoints
    
    """
    curr_short_seg = branch_idx
    curr_limb = limb_obj
    branch_obj = limb_obj[branch_idx]

    network_nodes = np.array(curr_limb.concept_network.nodes())
    network_nodes = network_nodes[network_nodes!= curr_short_seg]

    network_branches = [curr_limb[k].skeleton for k in network_nodes]
    neighbor_branches_by_endpoint = [network_nodes[sk.find_branch_skeleton_with_specific_coordinate(network_branches,e)] for e in branch_obj.endpoints]
    
    
    
    if return_endpoint_groupings:
        return neighbor_branches_by_endpoint,branch_obj.endpoints
    else:
        return np.concatenate(neighbor_branches_by_endpoint)
    
    
def all_soma_connnecting_endpionts_from_starting_info(starting_info):
    all_endpoints = []
    try:
        for limb_idx,limb_start_v in starting_info.items():
            for soma_idx,soma_v in limb_start_v.items():
                for soma_group_idx,group_v in soma_v.items():
                    all_endpoints.append(group_v["endpoint"])
    except:
        for soma_idx,soma_v in starting_info.items():
            for soma_group_idx,group_v in soma_v.items():
                all_endpoints.append(group_v["endpoint"])
        
    if len(all_endpoints) > 0:
        all_endpoints = np.unique(np.vstack(all_endpoints),axis=0)
    return all_endpoints
    
    
import copy

def skeleton_points_along_path(limb_obj,branch_path,
                               skeletal_distance_per_coordinate=4000,
                               return_unique=True):
    """
    Purpose: Will give skeleton coordinates for the endpoints of the 
    branches along the specified path
    
    if skeletal_distance_per_coordinate is None then will just endpoints
    """
    if skeletal_distance_per_coordinate is None:
        skeleton_coordinates = np.array([sk.find_branch_endpoints(limb_obj[k].skeleton) for k in branch_path]).reshape(-1,3)
    else:
        skeleton_coordinates = np.concatenate([sk.resize_skeleton_branch(
                                        limb_obj[k].skeleton,
                                        segment_width=skeletal_distance_per_coordinate) for k in branch_path]).reshape(-1,3)
        
    if return_unique:
        return np.unique(skeleton_coordinates,axis=0)
    else:
        return skeleton_coordinates
    
    
def get_matching_concept_network_data(limb_obj,soma_idx=None,soma_group_idx=None,
                                     starting_node=None,
                                     verbose=False):
    
    if type(soma_idx) == str:
        soma_idx = int(soma_idx[1:])
    
    if soma_idx is None and (soma_group_idx is None) and starting_node is None:
        raise Exception("All soma, soma_group and starting node descriptions are None")
        
    matching_concept_network_dicts_idx = np.arange(len(limb_obj.all_concept_network_data))
  
    if soma_idx is not None:
        soma_matches = np.array([i for i,k in enumerate(limb_obj.all_concept_network_data) if k["starting_soma"] == soma_idx])
        matching_concept_network_dicts_idx = np.intersect1d(matching_concept_network_dicts_idx,soma_matches)
        
    if soma_group_idx is not None:
        soma_matches = np.array([i for i,k in enumerate(limb_obj.all_concept_network_data) if k["soma_group_idx"] == soma_group_idx])
        matching_concept_network_dicts_idx = np.intersect1d(matching_concept_network_dicts_idx,soma_matches)
        
    if starting_node is not None:
        soma_matches = np.array([i for i,k in enumerate(limb_obj.all_concept_network_data) if k["starting_node"] == starting_node])
        matching_concept_network_dicts_idx = np.intersect1d(matching_concept_network_dicts_idx,soma_matches)
        
    if verbose:
        print(f"matching_concept_network_dicts_idx = {matching_concept_network_dicts_idx}")
        
    return [limb_obj.all_concept_network_data[k] for k in matching_concept_network_dicts_idx]
    
    
    
# ----------- 1/15: For Automatic Axon and Apical Classification ---------------#
import numpy_utils as nu
def add_branch_label(neuron_obj,limb_branch_dict,
                    labels):
    """
    Purpose: Will go through and apply a label to the branches
    specified
    
    """
    if not nu.is_array_like(labels):
        labels = [labels]
    
    for limb_name ,branch_array in limb_branch_dict.items():
        for b in branch_array:
            branch_obj = neuron_obj[limb_name][b]
            
            for l in labels:
                if l not in branch_obj.labels:
                    branch_obj.labels.append(l)
                    
def clear_all_branch_labels(neuron_obj,labels_to_clear="all"):
    if labels_to_clear != "all" and not nu.is_array_like(labels_to_clear):
        labels_to_clear = [labels_to_clear]
        
    for l in neuron_obj:
        for b in l:
            if labels_to_clear == "all":
                b.labels=[]
            else:
                b.labels = list(np.setdiff1d(b.labels,labels_to_clear))
            
            
import neuron_statistics as nst
def viable_axon_limbs_by_starting_angle_old(neuron_obj,
                                       axon_soma_angle_threshold=70,
                                       return_starting_angles=False):
    """
    This is method that does not use neuron querying (becuase just simple iterating through limbs)
    """
    
    possible_axon_limbs = []
    # Find the limb find the soma angle AND Filter away all limbs with a soma starting angle above threshold
    limb_to_starting_angle = dict()
    for curr_limb_idx,curr_limb in enumerate(curr_neuron_obj):
        curr_soma_angle = nst.soma_starting_angle(curr_neuron_obj,curr_limb_idx)
        limb_to_starting_angle[curr_limb_idx] = curr_soma_angle

        if curr_soma_angle > axon_soma_angle_threshold:
            possible_axon_limbs.append(curr_limb_idx)
    
    if return_starting_angles:
        return possible_axon_limbs,limb_to_starting_angle
    else:
        return possible_axon_limbs
    
import neuron_searching as ns
def viable_axon_limbs_by_starting_angle(neuron_obj,
                                       soma_angle_threshold,
                                        above_threshold=False,
                                        soma_name="S0",
                                        return_int_name=True,
                                       verbose=False):
    
    curr_neuron_obj = neuron_obj
    soma_center = curr_neuron_obj[soma_name].mesh_center

    if above_threshold:
        curr_query = f"soma_starting_angle>{soma_angle_threshold}"
    else:
        curr_query = f"soma_starting_angle<{soma_angle_threshold}"
    
    possible_axon_limbs_dict = ns.query_neuron(curr_neuron_obj,
                        query=curr_query,
                       functions_list=["soma_starting_angle"],
                       function_kwargs=dict(soma_center=soma_center,
                                           verbose=verbose))

    possible_axon_limbs = list(possible_axon_limbs_dict.keys())
    if return_int_name:
        return [nru.get_limb_int_name(k) for k in possible_axon_limbs]
    else:
        return possible_axon_limbs
    
    
def skeletal_distance_from_soma(curr_limb,
                    limb_name = None,
                    somas = None,
                    error_if_all_nodes_not_return=True,
                    include_node_skeleton_dist=True,
                    print_flag = False,
                    **kwargs
                            
    ):

    """
    Purpose: To determine the skeletal distance away from 
    a soma a branch piece is
    
    Pseudocode: 
    0) Create dictionary that will store all of the results
    For each directional concept network
    1) Find the starting node
    For each node: 
    1)find the shortest path from starting node to that node
    2) convert the path into skeletal distance of each node 
    and then add up
    3) Map of each of distances to the node in a dictionary and return
    - replace a previous one if smaller
    
    Example: 
    skeletal_distance_from_soma(
                    limb_name = "L1"
                    curr_limb = uncompressed_neuron.concept_network.nodes[limb_name]["data"]
                    print_flag = True
                    #soma_list=None
                    somas = [0,1]
                    check_all_nodes_in_return=True
    )

    """
    if print_flag:
        print(f"\n\n------Working on Limb ({limb_name})-------")
        print(f"Starting nodes BEFORE copy = {xu.get_starting_node(curr_limb.concept_network,only_one=False)}")

    curr_limb_copy =  deepcopy(curr_limb)
    
    if print_flag:
        print(f"Starting nodes after copy = {xu.get_starting_node(curr_limb_copy.concept_network,only_one=False)}")

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
        print(f"Performing analysis for somas: {touching_somas}")

    for sm_start in touching_somas:
        #1) Find the starting node
        if print_flag:
            print(f"--> Working on soma {sm_start}")
        try:
            curr_limb_copy.set_concept_network_directional(sm_start)
        except:
            if print_flag:
                print(f"Limb ({limb_name}) was not connected to soma {sm_start} accordinag to all concept networks")
            continue
        curr_directional_network = curr_limb_copy.concept_network_directional
        starting_node = curr_limb_copy.current_starting_node

        #For each node: 
        for n in curr_directional_network.nodes():
            #1)find the shortest path from starting node to that node
            #( could potentially not be there because it is directional)
            try:
                curr_shortest_path = nx.shortest_path(curr_directional_network,starting_node,n)
            except:
                #return_dict[n] = np.inf
                continue
            #2) convert the path into skeletal distance of each node and then add up
            if not include_node_skeleton_dist:
                path_length = np.sum([sk.calculate_skeleton_distance(curr_directional_network.nodes[k]["data"].skeleton)
                               for k in curr_shortest_path[:-1]])
            else:
                path_length = np.sum([sk.calculate_skeleton_distance(curr_directional_network.nodes[k]["data"].skeleton)
                               for k in curr_shortest_path])


            #3) Map of each of distances to the node in a dictionary and return
            #- replace a previous one if smaller

            if n in return_dict.keys():
                if path_length < return_dict[n]:
                    return_dict[n] = path_length
            else:
                return_dict[n] = path_length
    if print_flag:
        print(f"\nBefore Doing the dictionary correction, return_dict={return_dict}\n")
    #check that the return dict has all of the nodes
    for n in list(curr_limb_copy.concept_network.nodes()):
        if n not in return_dict.keys():
            return_dict[n] = np.inf
   
    if error_if_all_nodes_not_return:
        if set(list(return_dict.keys())) != set(list(curr_limb_copy.concept_network.nodes())):
            raise Exception("return_dict keys do not exactly match the curr limb nodes")
            
    return return_dict
    

def find_branch_with_specific_coordinate(limb_obj,
                                        coordinates):
    """
    Purpose: To find all branch idxs whos skeleton contains a certain coordinate
    
    """
    
    coordinates = np.array(coordinates).reshape(-1,3)
    
    network_branches = [k.skeleton for k in limb_obj]
    
    final_branch_idxs = []
    for e in coordinates:
        curr_branch_idx = sk.find_branch_skeleton_with_specific_coordinate(network_branches,e)
        if len(curr_branch_idx) > 0:
            final_branch_idxs.append(curr_branch_idx)
    
    if len(final_branch_idxs) > 0:
        final_branch_idxs = np.concatenate(final_branch_idxs)
    
    return final_branch_idxs



def neuron_spine_density(neuron_obj,
                        lower_width_bound = 140,
                        upper_width_bound = 520,#380,
                        spine_threshold = 2,
                        skeletal_distance_threshold = 110000,#30000,
                        skeletal_length_threshold = 15000,#10000
                        verbose=False,
                        plot_candidate_branches=False,
                        return_branch_processed_info=True,
                        **kwargs):
    """
    Purpose: To Calculate the spine density used to classify
    a neuron as one of the following categories based on the spine
    density of high interest branches
    
    1) no_spine
    2) sparsely_spine
    3) densely_spine
    
    
    """
    curr_neuron_obj= neuron_obj
    
    
    
    
    if plot_candidate_branches:
        return_dataframe=False
        close_limb_branch_dict = ns.query_neuron(curr_neuron_obj,
                                            functions_list=["skeletal_distance_from_soma_excluding_node","no_spine_median_mesh_center",
                                                            "n_spines","spine_density","skeletal_length"],
                                            query=(f"(skeletal_distance_from_soma_excluding_node<{skeletal_distance_threshold})"
                                                   f" and (no_spine_median_mesh_center > {lower_width_bound})"
                                                   f" and (no_spine_median_mesh_center < {upper_width_bound})"
                                                  f" and (n_spines > {spine_threshold})"
                                                   f" and skeletal_length > {skeletal_length_threshold} "
                                                  ),
                                             return_dataframe=return_dataframe


                                          )
        
        nviz.visualize_neuron(curr_neuron_obj,
                              visualize_type=["mesh"],
                             limb_branch_dict=close_limb_branch_dict,
                              mesh_color="red",
                              mesh_whole_neuron=True)
        
        
    
    
    return_dataframe = True
    close_limb_branch_dict = ns.query_neuron(curr_neuron_obj,
                                            functions_list=["skeletal_distance_from_soma_excluding_node","no_spine_median_mesh_center",
                                                            "n_spines","spine_density","skeletal_length"],
                                            query=(f"(skeletal_distance_from_soma_excluding_node<{skeletal_distance_threshold})"
                                                   f" and (no_spine_median_mesh_center > {lower_width_bound})"
                                                   f" and (no_spine_median_mesh_center < {upper_width_bound})"
                                                  f" and (n_spines > {spine_threshold})"
                                                   f" and skeletal_length > {skeletal_length_threshold} "
                                                  ),
                                             return_dataframe=return_dataframe


                                          )
    
    total_branches_in_search_radius = ns.query_neuron(curr_neuron_obj,
                                            functions_list=["skeletal_distance_from_soma_excluding_node","skeletal_length"],
                                            query=(f"(skeletal_distance_from_soma_excluding_node<{skeletal_distance_threshold})"
                                                  ),
                                             return_dataframe=return_dataframe


                                          )
    
    
    # ---- 1/24: Calculating the skeletal length of the viable branches --- #
    total_skeletal_length_in_search_radius = np.sum(total_branches_in_search_radius["skeletal_length"].to_numpy())
    processed_skeletal_length = np.sum(close_limb_branch_dict["skeletal_length"].to_numpy())

    if len(close_limb_branch_dict)>0:
        median_spine_density = np.median(close_limb_branch_dict["spine_density"].to_numpy())
    else:
        median_spine_density = 0
        
    if verbose:
        print(f'median spine density = {median_spine_density}')
        print(f"Number of branches = {len(close_limb_branch_dict)}")
        print(f"Number of branches in radius = {len(total_branches_in_search_radius)}")
        print(f"processed_skeletal_length = {processed_skeletal_length}")
        print(f"total_skeletal_length_in_search_radius = {total_skeletal_length_in_search_radius}")
        
        
    if return_branch_processed_info:
        return (median_spine_density,
                len(close_limb_branch_dict),processed_skeletal_length,
                len(total_branches_in_search_radius),total_skeletal_length_in_search_radius)
    else:
        return median_spine_density
    
def all_concept_network_data_to_limb_network_stating_info(all_concept_network_data):
    """
    Purpose: Will conver the concept network data list of dictionaries into a 
    the dictionary representation of only the limb touching vertices and
    endpoints of the limb_network_stating_info in the preprocessed data
    
    Pseudocode: 
    Iterate through all of the network dicts and store as
    soma--> soma_group_idx --> dict(touching_verts,
                                    endpoint)
                                    
    stored in the concept network as 
    touching_soma_vertices
    starting_coordinate
    
    """
    limb_network = dict()
    for k in all_concept_network_data:
        soma_idx = k["starting_soma"]
        soma_group_idx = k["soma_group_idx"]
        
        if soma_idx not in limb_network.keys():
            limb_network[soma_idx] = dict()
            
        limb_network[soma_idx][soma_group_idx] = dict(touching_verts=k["touching_soma_vertices"],
                                                     endpoint = k["starting_coordinate"])
        
    return limb_network
    

def clean_all_concept_network_data(all_concept_network_data,
                                  verbose=False):
    
    """
    Purpose: To make sure that there are
    no duplicate entries of that starting nodes
    and either to combine the soma touching points
    or just keep the largest one

    Pseudocode: 
    1) Start with an empty dictionary
    For all the dictionaries:
    2)  store the result
    indexed by starting soma and starting node
    3) If an entry already existent --> then either add the soma touching
    vertices (and unique) to the list or replace it if longer

    4) Turn the one dictionary into a list of dictionaries
    like the all_concept_network_data attribute

    5) Replace the all_concept_network_data


    """

    new_network_data = dict()

    for n_dict in all_concept_network_data:
        starting_soma = n_dict["starting_soma"]
        starting_node = n_dict["starting_node"]

        if starting_soma not in new_network_data.keys():
            new_network_data[starting_soma] = dict()

        if starting_node in new_network_data[starting_soma].keys():
            if (len(new_network_data[starting_soma][starting_node]["touching_soma_vertices"]) < 
                len(n_dict["touching_soma_vertices"])):
                if verbose:
                    print(f"Replacing the Soma_{starting_soma}_Node_{starting_node} dictionary")
                new_network_data[starting_soma][starting_node] = n_dict
            else:
                if verbose:
                    print(f"Skipping the Soma_{starting_soma}_Node_{starting_node} dictionary because smaller")
        else:
            new_network_data[starting_soma][starting_node] = n_dict

    #4) Turn the one dictionary into a list of dictionaries
    #like the all_concept_network_data attribute

    new_network_list = []
    for soma_idx,soma_info in new_network_data.items():
        for idx,(starting_node,node_info) in enumerate(soma_info.items()):
            node_info["soma_group_idx"] = idx
            new_network_list.append(node_info)

    return new_network_list



def clean_neuron_all_concept_network_data(neuron_obj,verbose=False):
    """
    Will go through and clean all of the concept network data
    in all the limbs of a Neuron
    """
    for j,curr_limb in enumerate(neuron_obj):
        if verbose:
            print(f"\n\n---- Working on Limb {j} ----")
            
            
        cleaned_network = nru.clean_all_concept_network_data(curr_limb.all_concept_network_data,
                                                                          verbose=verbose)
        
        if verbose:
            print(f"cleaned_network = {cleaned_network}\n\n")
        
        curr_limb.all_concept_network_data = cleaned_network
        
        #setting the concept network
        st_soma = curr_limb.all_concept_network_data[0]["starting_soma"]
        st_node = curr_limb.all_concept_network_data[0]["starting_node"]
        curr_limb.set_concept_network_directional(starting_soma=st_soma,
                                                 starting_node=st_node)
        
        # --------- 1/24: Cleaning the preprocessed data as well -----------#
        if verbose:
            print(f"cleaned_network = {cleaned_network}")
            
        new_limb_network = nru.all_concept_network_data_to_limb_network_stating_info(cleaned_network)
        
        if verbose:
            print(f"\n---------\nnew_limb_network = {new_limb_network}\n---------\n")
        neuron_obj.preprocessed_data["limb_network_stating_info"][j] = new_limb_network
        
        if verbose:
            print(f"curr_limb.all_concept_network_data = {curr_limb.all_concept_network_data}\n\n")
            
#         neuron_obj[j] = curr_limb
    
#     return neuron_obj


def limb_branch_dict_to_connected_components(neuron_obj,
                                             limb_branch_dict,
            use_concept_network_directional=False):
    """
    Purpose: To turn the limb branch dict into a
    list of all the connected components described by the
    limb branch dict
    
    """
    
    axon_connected_comps = []
    for limb_name, axon_branches in limb_branch_dict.items():
        
        if use_concept_network_directional:
            curr_network = neuron_obj[limb_name].concept_network_directional
        else:
            curr_network = neuron_obj[limb_name].concept_network
            
        axon_subgraph = curr_network.subgraph(axon_branches)
        conn_comp = [(limb_name,np.array(list(k))) for k in nx.connected_components(axon_subgraph)]
        axon_connected_comps += conn_comp

    return axon_connected_comps
        
def empty_limb_object(labels=["empty"]):
    curr_limb = neuron.Limb(mesh=None,
                        curr_limb_correspondence=dict(),
                         concept_network_dict=dict(),
                        labels=labels)
    curr_limb.concept_network = nx.Graph()
    curr_limb.concept_network_directional = nx.DiGraph()
    return curr_limb


def sum_feature_over_limb_branch_dict(neuron_obj,
                                       limb_branch_dict,
                                       feature,
                                     feature_function=None):
    """
    Purpose: To sum the value of some feature over the branches
    specified by the limb branch dict
    """
    
    feature_total = 0
    
    for limb_name, branch_list in limb_branch_dict.items():
        for b in branch_list:
            feature_value = getattr(neuron_obj[limb_name][b],feature)
            if feature_function is not None:
                feature_value = feature_function(feature_value)
            feature_total += feature_value
            
    return feature_total

def limb_branch_after_limb_branch_removal(neuron_obj,
                                      limb_branch_dict,
                             return_removed_limb_branch = False,
                             verbose=False
                            ):

    """
    Purpose: To take a branches that should be deleted from
    different limbs in a limb branch dict then to determine the leftover branches
    of each limb that are still connected to the starting node



    Pseudocode:
    For each starting node
    1) Get the starting node
    2) Get the directional conept network and turn it undirected
    3) Find the total branches that will be deleted and kept
    once the desired branches are removed (only keeping the ones 
    still connected to the starting branch)
    4) add the removed and kept branches to the running limb branch dict

    """
    

    limb_branch_dict_kept = dict()
    limb_branch_dict_removed = dict()

    for limb_name in neuron_obj.get_limb_node_names():
        limb_obj = neuron_obj[limb_name]
        branch_names = limb_obj.get_branch_names()

        if limb_name not in limb_branch_dict.keys():
            limb_branch_dict_kept[limb_name] = branch_names
            continue



        nodes_to_remove = limb_branch_dict[limb_name]

        G = nx.Graph(limb_obj.concept_network_directional)
        nodes_to_keep = limb_obj.current_starting_node

        kept_branches,removed_branches = xu.nodes_in_kept_groups_after_deletion(G,
                                            nodes_to_keep,
                                               nodes_to_remove=nodes_to_remove,
                                            return_removed_nodes = True
                                               ) 
        if len(kept_branches)>0:
            limb_branch_dict_kept[limb_name] = kept_branches
        if len(removed_branches) > 0:
            limb_branch_dict_removed[limb_name] = removed_branches

    if return_removed_limb_branch:
        return limb_branch_dict_removed
    else:
        return limb_branch_dict_kept


import networkx as nx
def branches_within_skeletal_distance(limb_obj,
                                    start_branch,
                                    max_distance_from_start,
                                    verbose = False,
                                    include_start_branch_length = False,
                                    include_node_branch_length = False,
                                    only_consider_downstream = False):

    """
    Purpose: to find nodes within a cetain skeletal distance of a certain 
    node (can be restricted to only those downstream)

    Pseudocode: 
    1) Get the directed concept grpah
    2) Get all of the downstream nodes of the node
    3) convert directed concept graph into an undirected one
    4) Get a subgraph using all of the downstream nodes
    5) For each node: 
    - get the shortest path from the node to the starting node
    - add up the skeleton distance (have options for including each endpoint)
    - if below the max distance then add
    6) Return nodes


    Ex: 
    start_branch = 53
        
    viable_downstream_nodes = nru.branches_within_skeletal_distance(limb_obj = current_neuron[6],
                                start_branch = start_branch,
                                max_distance_from_start = 50000,
                                verbose = False,
                                include_start_branch_length = False,
                                include_node_branch_length = False,
                                only_consider_downstream = True)

    limb_branch_dict=dict(L6=viable_downstream_nodes+[start_branch])

    nviz.plot_limb_branch_dict(current_neuron,
                              limb_branch_dict)

    """

    curr_limb = limb_obj



    viable_downstream_nodes = []

    dir_nx = curr_limb.concept_network_directional

    #2) Get all of the downstream nodes of the node

    if only_consider_downstream:
        all_downstream_nodes = list(xu.all_downstream_nodes(dir_nx,start_branch))
    else:
        all_downstream_nodes = list(dir_nx.nodes())
        all_downstream_nodes.remove(start_branch)

    if len(all_downstream_nodes) == 0:
        if verbose:
            print(f"No downstream nodes to test")

        return []

    if verbose:
        print(f"Number of downstream nodes = {all_downstream_nodes}")

    #3) convert directed concept graph into an undirected one
    G_whole = nx.Graph(dir_nx)

    #4) Get a subgraph using all of the downstream nodes
    G = G_whole.subgraph(all_downstream_nodes + [start_branch])

    for n in all_downstream_nodes:

        #- get the shortest path from the node to the starting node
        try:
            curr_shortest_path = nx.shortest_path(G,start_branch,n)
        except:
            if verbose:
                print(f"Continuing because No path between start node ({start_branch}) and node {n}")
            continue 


        if not include_node_branch_length:
            curr_shortest_path = curr_shortest_path[:-1]

        if not include_start_branch_length:
            curr_shortest_path = curr_shortest_path[1:]

        total_sk_length_of_path = np.sum([curr_limb[k].skeletal_length for k in curr_shortest_path])

        if total_sk_length_of_path <= max_distance_from_start:
            viable_downstream_nodes.append(n)
        else:
            if verbose:
                print(f"Branch {n} was too far from the start node : {total_sk_length_of_path} (threshold = {max_distance_from_start})")

    return viable_downstream_nodes

import classification_utils as clu
def low_branch_length_clusters(neuron_obj,
                              max_skeletal_length = 8000,
                                min_n_nodes_in_cluster = 4,
                               width_max = None,
                               use_axon_like_restriction = False,
                               verbose=False,
                               **kwargs
                                ):

    """
    Purpose: To find parts of neurons with lots of nodes
    close together on concept network with low branch length
    
    Pseudocode:
    1) Get the concept graph of a limb 
    2) Eliminate all of the nodes that are too long skeletal length
    3) Divide the remaining axon into connected components
    - if too many nodes are in the connected component then it is
    an axon mess and should delete all those nodes
    
    Application: Helps filter away axon mess

    """
    
    use_deletion=False
    
    curr_neuron_obj=neuron_obj

    if width_max is None:
        width_max = np.inf

    limb_branch_dict = dict()
    
    
    
    # ---------- Getting the restriction that we will check over ---- #
    if use_axon_like_restriction:
        axon_limb_branch_dict = clu.axon_like_limb_branch_dict(curr_neuron_obj)
    else:
        axon_limb_branch_dict = None
        
    

    if not use_deletion:
        limb_branch_restriction = ns.query_neuron(curr_neuron_obj,
                        functions_list=["skeletal_length","median_mesh_center"],
                       query = ( f" (skeletal_length < {max_skeletal_length}) and "
                               f" (median_mesh_center < {width_max})"),
                       limb_branch_dict_restriction=axon_limb_branch_dict)
        if verbose:
            print(f"limb_branch_restriction = {limb_branch_restriction}")
    else:
        limb_branch_restriction = nru.neuron_limb_branch_dict(curr_neuron_obj)


    for limb_name,nodes_to_keep in limb_branch_restriction.items():
        curr_limb = curr_neuron_obj[limb_name]
        if verbose:
            print(f"--- Working on Limb {limb_name} ---")

        if use_deletion:
        #1) Get the branches that are below a certain threshold
            nodes_to_delete = [jj for jj,branch in enumerate(curr_limb) 
                               if ((curr_limb[jj].skeletal_length > max_skeletal_length ))]

            if verbose:
                print(f"nodes_to_delete = {nodes_to_delete}")

            #2) Elimnate the nodes from the concept graph
            G_short = nx.Graph(curr_limb.concept_network)
            G_short.remove_nodes_from(nodes_to_delete)
        
        else:
            #2) Elimnate the nodes from the concept graph
            G= nx.Graph(curr_limb.concept_network)
            G_short = G.subgraph(nodes_to_keep)

            if verbose:
                print(f"nodes_to_keep = {nodes_to_keep}")

        
        #3) Divide the remaining graph into connected components
        conn_comp = [list(k) for k in nx.connected_components(G_short)]

        potential_error_branches = []

        for c in conn_comp:
            if len(c) > min_n_nodes_in_cluster:
                potential_error_branches += c

        #4)  If found any error nodes then add to limb branch dict
        if len(potential_error_branches) > 0:
            limb_branch_dict[limb_name] = potential_error_branches

    return limb_branch_dict

def neuron_limb_branch_dict(neuron_obj):
    """
    Purpose: To develop a limb branch dict represnetation
    of the limbs and branchs of a neuron
    
    """
    limb_branch_dict_new = dict()
    
    if neuron_obj.__class__.__name__ == "Neuron":
        for limb_name in neuron_obj.get_limb_node_names():
            limb_branch_dict_new[limb_name] = neuron_obj[limb_name].get_branch_names()
    else:
        net = neuron_obj
        curr_limb_names = [k for k in net.nodes() if "L" in k]
        for limb_name in curr_limb_names:
            limb_branch_dict_new[limb_name] = np.array(list(net.nodes[limb_name]["data"].concept_network.nodes()))

        
    return limb_branch_dict_new

def limb_branch_invert(neuron_obj,
                           limb_branch_dict,
                           verbose=False):
    """
    Purpose: To get every node that is not in limb branch dict
    
    Ex: 
    invert_limb_branch_dict(curr_neuron_obj,limb_branch_return,
                       verbose=True)
    """
    
    limb_branch_dict_new = dict()
    for j,curr_limb in enumerate(neuron_obj):
        
        limb_name = f"L{j}"
        
        if verbose:
            print(f"\n--- Working on limb {limb_name}")
        
        if limb_name in limb_branch_dict:
            curr_branches = limb_branch_dict[limb_name]
        else:
            curr_branches = []
            
        
            
        leftover_branches = np.setdiff1d(curr_limb.get_branch_names(),curr_branches)
        if verbose:
            print(f"curr_branches = {curr_branches}")
            print(f"leftover_branches = {leftover_branches}")
            print(f"total combined branches = {len(curr_branches) +len(leftover_branches) }, len(limb) = {len(curr_limb)}")
        if len(leftover_branches)>0:
            limb_branch_dict_new[limb_name] = leftover_branches
            
    return limb_branch_dict_new

def limb_branch_combining(
                           limb_branch_dict_list,
                           combining_function,
                           verbose=False):
    """
    Purpose: To get every node that is not in limb branch dict
    
    Ex: 
    invert_limb_branch_dict(curr_neuron_obj,limb_branch_return,
                       verbose=True)
    """
    all_keys = nu.union1d_multi_list([list(k.keys()) for k in limb_branch_dict_list])
    
    
    limb_branch_dict_new = dict()
    for limb_name in all_keys:
        
        if verbose:
            print(f"\n--- Working on limb {limb_name}")
        
        curr_branches = [k.get(limb_name,[]) for k in limb_branch_dict_list]
        
        leftover_branches = nu.function_over_multi_lists(curr_branches,combining_function)
        
        if verbose:
            print(f"combining_function = {combining_function}")
            print(f"curr_branches = {curr_branches}")
            print(f"leftover_branches = {leftover_branches}")
            
        if len(leftover_branches)>0:
            limb_branch_dict_new[limb_name] = leftover_branches
            
    return limb_branch_dict_new

def limb_branch_setdiff(limb_branch_dict_list):
    
    return limb_branch_combining(
                           limb_branch_dict_list,
                           np.setdiff1d,
                           verbose=False)

def limb_branch_union(limb_branch_dict_list):
    
    return limb_branch_combining(
                           limb_branch_dict_list,
                           np.union1d,
                           verbose=False)

def limb_branch_intersection(limb_branch_dict_list):
    
    return limb_branch_combining(
                           limb_branch_dict_list,
                           np.intersect1d,
                           verbose=False
    )

            

import neuron_utils as nru
import neuron #package where can use the Branches class to help do branch skeleton analysis


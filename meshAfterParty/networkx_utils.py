import networkx as nx
import numpy as np
import numpy_utils as nu
import time


def unpickle_graph(path):
    G_loaded = nx.read_gpickle(path)
    return G_loaded

def pickle_graph(path):
    nx.write_gpickle(path)

def find_reciprocal_connections(G,redundant=False):
    """
    Will give a list of the edges that are reciprocal connections
    ** only gives one version of the reciprocal connections so doesn't repeat**
    
    Arguments: 
    G: the graph to look for reciprocal connections
    redundant: whether to return a list with redundant connections or not (Ex: [(b,a)]  or [(b,a),(a,b)]
    
    Ex: 
    import networkx_utils as xu
    xu = reload(xu)
    xu.find_reciprocal_connections(returned_network)
    """
    reciprocal_pairs = np.array([(u,v) for (u,v) in G.edges() if G.has_edge(v,u)])
    if redundant:
        return reciprocal_pairs
    
    filtered_reciprocal_pairs = []

    for a,b in reciprocal_pairs:       
        if len(nu.matching_rows(filtered_reciprocal_pairs,[b,a])) == 0:
            filtered_reciprocal_pairs.append([a,b])

    filtered_reciprocal_pairs = np.array(filtered_reciprocal_pairs)
    return filtered_reciprocal_pairs


def compare_endpoints(endpoints_1,endpoints_2,**kwargs):
    """
    comparing the endpoints of a graph: 
    
    Ex: 
    import networkx_utils as xu
    xu = reload(xu)mess
    end_1 = np.array([[2,3,4],[1,4,5]])
    end_2 = np.array([[1,4,5],[2,3,4]])

    xu.compare_endpoints(end_1,end_2)
    """
    #this older way mixed the elements of the coordinates together to just sort the columns
    #return np.array_equal(np.sort(endpoints_1,axis=0),np.sort(endpoints_2,axis=0))
    
    #this is correct way to do it (but has to be exact to return true)
    #return np.array_equal(nu.sort_multidim_array_by_rows(endpoints_1),nu.sort_multidim_array_by_rows(endpoints_2))

    return nu.compare_threshold(nu.sort_multidim_array_by_rows(endpoints_1),
                                nu.sort_multidim_array_by_rows(endpoints_2),
                                **kwargs)


def endpoint_connectivity(endpoints_1,endpoints_2,
                         exceptions_flag=True,
                         print_flag=False):
    """
    Pupose: To determine where the endpoints of two branches are connected
    
    Example: 
    end_1 = np.array([[759621., 936916., 872083.],
       [790891., 913598., 806043.]])
    end_2 = np.array([[790891., 913598., 806043.],
       [794967., 913603., 797825.]])
       
    endpoint_connectivity(end_1,end_2)
    >> {0: 1, 1: 0}
    """
    connections_dict = dict()
    
    stacked_endpoints = np.vstack([endpoints_1,endpoints_2])
    endpoints_match = nu.get_matching_vertices(stacked_endpoints)
    
    if len(endpoints_match) == 0:
        print_string = f"No endpoints matching: {endpoints_match}"
        if exceptions_flag:
            raise Exception(print_string)
        else:
            print(print_string)
        return connections_dict
    
    if len(endpoints_match) > 1:
        print_string = f"Multiple endpoints matching: {endpoints_match}"
        if exceptions_flag:
            raise Exception(print_string)
        else:
            print(print_string)
    
    
    #look at the first connection
    first_match = endpoints_match[0]
    first_endpoint_match = first_match[0]
    
    if print_flag:
        print(f"first_match = {first_match}")
        print(f"first_endpoint_match = {endpoints_1[first_endpoint_match]}")
    
    
    if 0 != first_endpoint_match and 1 != first_endpoint_match:
        raise Exception(f"Non 0,1 matching node in first endpoint: {first_endpoint_match}")
    else:
        connections_dict.update({0:first_endpoint_match})
        
    second_endpoint_match = first_match[-1]
    
    if 2 != second_endpoint_match and 3 != second_endpoint_match:
        raise Exception(f"Non 2,3 matching node in second endpoint: {second_endpoint_match}")
    else:
        connections_dict.update({1:second_endpoint_match-2})
    
    return connections_dict



def combine_graphs(list_of_graphs):
    """
    Purpose: Will combine graphs, but if they have the same name
    then will combine the nodes
    
    Example: 
    xu = reload(xu)
    G1 = nx.from_edgelist([[1,2],[3,4],[2,3]])
    nx.draw(G1)
    plt.show()

    G2 = nx.from_edgelist([[3,4],[2,3],[2,5]])
    nx.draw(G2)
    plt.show()

    G3 = nx.compose_all([G1,G2])
    nx.draw(G3)
    plt.show()

    nx.draw(xu.combine_graphs([G1,G2,G3]))
    plt.show()

    """
    if len(list_of_graphs) == 1:
        return list_of_graphs[0]
    elif len(list_of_graphs) == 0:
        raise Exception("List of graphs is empty")
    else:
        return nx.compose_all(list_of_graphs)

def edge_to_index(G,curr_edge):
    matching_edges_idx = nu.matching_rows(G.edges_ordered(),curr_edge)
    if len(matching_edges_idx) == 1:
        return nu.matching_rows(G.edges_ordered(),curr_edge)[0]
    else: 
        return nu.matching_rows(G.edges_ordered(),curr_edge) 

def index_to_edge(G,edge_idx):
    return np.array(G.edges_ordered())[edge_idx]

def node_to_edges(G,node_number):
#     if type(node_number) != list:
#         node_number = [node_number]
    #print(f"node_number={node_number}")
    if type(G) == type(nx.Graph()):
        return list(G.edges(node_number))
    elif type(G) == type(GraphOrderedEdges()):
        return G.edges_ordered(node_number)
    else:
        raise Exception("not expected type for G")

        
def get_node_list(G,exclude_list = []):
    return [n for n in list(G.nodes()) if n not in exclude_list]

import numpy_utils as nu
def get_nodes_with_attributes_dict(G,attribute_dict):
    """
    
    
    """
    
    # --- 11/4 An aleration that instead calles the more efficient method ---
    if len(attribute_dict.keys()) == 1 and "coordinates" in attribute_dict.keys():
        return get_graph_node_by_coordinate(G,attribute_dict["coordinates"],return_single_value=False)
    
    node_list = []
    total_search_keys = list(attribute_dict.keys())
    for x,y in G.nodes(data=True):
        if len(set(total_search_keys).intersection(set(list(y.keys())))) < len(total_search_keys):
            #there were not enough keys in the node we were searching
            continue
        else:
            add_flag = True
            for search_key in total_search_keys:
                #print(f"y[search_key] = {y[search_key]}")
                #print(f"attribute_dict[search_key] = {attribute_dict[search_key]}")
                curr_search_val= y[search_key]
                if type(curr_search_val) in [type(np.array([])),type(np.ndarray([])),list]:
                    if not nu.compare_threshold(np.array(curr_search_val),attribute_dict[search_key]):
                        add_flag=False
                        break
                else:
                    if y[search_key] != attribute_dict[search_key]:
                        add_flag=False
                        break
            if add_flag:
                #print("Added!")
                node_list.append(x)
    return node_list

def get_graph_node_by_coordinate_old(G,coordinate):
    match_nodes = get_nodes_with_attributes_dict(G,dict(coordinates=coordinate))
    #print(f"match_nodes = {match_nodes}")
    if len(match_nodes) != 1:
        raise Exception(f"Not just one node in graph with coordinate {coordinate}: {match_nodes}")
    else:
        return match_nodes[0]
    
def get_graph_node_by_coordinate(G,coordinate,return_single_value=True):
    """
    Much faster way of searching for nodes by coordinates
    
    """
    graph_nodes = np.array(list(G.nodes()))
    node_coordinates = get_node_attributes(G,node_list = graph_nodes)
    match_nodes = nu.matching_rows(node_coordinates,coordinate)
    if return_single_value:
        if len(match_nodes) != 1:
            raise Exception(f"Not just one node in graph with coordinate {coordinate}: {match_nodes}")
        else:
            return graph_nodes[match_nodes[0]]
    else:
        return graph_nodes[match_nodes]

def get_all_nodes_with_certain_attribute_key(G,attribute_name):
    return nx.get_node_attributes(G,attribute_name)

import numpy_utils as nu
def get_node_attributes(G,attribute_name="coordinates",node_list=[],
                       return_array=True):
    #print(f"attribute_name = {attribute_name}")
    if not nu.is_array_like(node_list):
        node_list = [node_list]
    attr_dict = nx.get_node_attributes(G,attribute_name)
    #print(f"attr_dict= {attr_dict}")
    if len(node_list)>0:
        #print("inside re-ordering")
        #attr_dict = dict([(k,v) for k,v in attr_dict.items() if k in node_list])
        attr_dict = dict([(k,attr_dict[k]) for k in node_list])
    
    if return_array:
        return np.array(list(attr_dict.values()))
    else: 
        return attr_dict
    
def remove_selfloops(UG):
    self_edges = nx.selfloop_edges(UG)
    #print(f"self_edges = {self_edges}")
    UG.remove_edges_from(self_edges)
    return UG

def get_neighbors(G,node,int_label=True):
    if int_label:
        return [int(n) for n in G[node]]
    else:
        return [n for n in G[node]]
    

def get_nodes_of_degree_k(G,degree_choice):
    return [k for k,v in dict(G.degree).items() if v == degree_choice]

def get_nodes_greater_or_equal_degree_k(G,degree_choice):
    return [k for k,v in dict(G.degree).items() if v >= degree_choice]

def get_nodes_less_or_equal_degree_k(G,degree_choice):
    return [k for k,v in dict(G.degree).items() if v <= degree_choice]

def get_node_degree(G,node_name):
    if not nu.is_array_like(node_name):
        node_name = [node_name]
    degree_dict = dict(G.degree)
    node_degrees = [degree_dict[k] for k in node_name]
    if len(node_degrees) > 1:
        return node_degrees
    else:
        return node_degrees[0]


def set_node_attributes_dict(G,attrs):
    """
    Can set the attributes of nodes with dictionaries 
    
    ex: 
    G = nx.path_graph(3)
    attrs = {0: {'attr1': 20, 'attr2': 'nothing'}, 1: {'attr2': 3}}
    nx.set_node_attributes(G, attrs)
    G.nodes[0]['attr1']

    G.nodes[0]['attr2']

    G.nodes[1]['attr2']

    G.nodes[2]
    """
    nx.set_node_attributes(G, attrs)

def relabel_node_names(G,mapping,copy=False):
    nx.relabel_nodes(G, mapping, copy=copy)
    print("Finished relabeling nodes")
    
    
def get_all_attributes_for_nodes(G,node_list=[],
                       return_dict=False):
    if len(node_list) == 0:
        node_list = list(G.nodes())
    
    attributes_list = [] 
    attributes_list_dict = dict()
    for n in node_list:
        attributes_list.append(G.nodes[n])
        attributes_list_dict.update({n:G.nodes[n]})
    
    if return_dict:
        return attributes_list_dict
    else:
        return attributes_list
    
# -------------- start of functions to help with edges ---------------#
def get_all_attributes_for_edges(G1,edges_list=[],return_dict=False):
    """
    Ex: 
    G1 = limb_concept_network
    xu.get_all_attributes_for_edges(G1,return_dict=True)
    """
    if len(edges_list) == 0:
        print("Edge list was 0 so generating sorted edges")
        edges_list = nu.sort_multidim_array_by_rows(G1.edges(),order_row_items=isinstance(G1,(nx.Graph)))
    elif len(edges_list) != len(G1.edges()):
        print(f"**Warning the length of edges_list ({len(edges_list)}) is less than the total number of edges for Graph**")
    else:
        pass

    attributes_list = [] 
    attributes_list_dict = dict()
    for u,v in edges_list:
        attributes_list.append(G1[u][v])
        attributes_list_dict.update({(u,v):G1[u][v]})
    
    
    if return_dict:
        return attributes_list_dict
    else:
        return attributes_list


def get_edges_with_attributes_dict(G,attribute_dict):
    if type(attribute_dict) != dict:
        raise Exception("Did not recieve dictionary for searching")
    total_edges = []
    total_searching_keys = list(attribute_dict.keys())
    for (u,v) in G.edges():
        if len(set(total_searching_keys).intersection(set(list(G[u][v])))) < len(total_searching_keys):
            continue
        else:
            match = True
            for k in total_searching_keys:
                if G[u][v][k] !=  attribute_dict[k]:
                    match = False
                    break
            if match:
                total_edges.append((u,v))
    return total_edges
               


def get_edge_attributes(G,attribute="order",edge_list=[],undirected=True):
    #print(f"edge_list = {edge_list}, type={type(edge_list)}, shape = {edge_list.shape}")
    #print("")
    edge_attributes = nx.get_edge_attributes(G,"order")
    #print(f"edge_attributes = {edge_attributes}")
    if len(edge_list) > 0:
        if undirected:
            total_attributes = []
            for e in edge_list:
                try:
                    total_attributes.append(edge_attributes[tuple(e)])
                except:
                    #try the other way around to see if it exists
                    try:
                        total_attributes.append(edge_attributes[tuple((e[-1],e[0]))])
                    except: 
                        print(f"edge_attributes = {edge_attributes}")
                        print(f"(e[-1],e[0]) = {(e[-1],e[0])}")
                        raise Exception("Error in get_edge_attributes")
            return total_attributes
        else:
            return [edge_attributes[tuple(k)] for k in edge_list]
    else:
        return edge_attributes


import copy
# how you can try to remove a cycle from a graph
def remove_cycle(branch_subgraph, max_cycle_iterations=1000): 
    
    #branch_subgraph_copy = copy.copy(branch_subgraph)
    for i in range(max_cycle_iterations): 
        try:
            edges_in_cycle = nx.find_cycle(branch_subgraph)
        except:
            break
        else:
            print(f"type(branch_subgraph) = {type(branch_subgraph)}")
            #make a copy to unfreeze
            branch_subgraph = GraphOrderedEdges(branch_subgraph)
            #print(f"edges_in_cycle = {edges_in_cycle}")
            #not doing random deletion just picking first edge
            picked_edge_to_delete = edges_in_cycle[-1]
            #print(f"picked_edge_to_delete = {picked_edge_to_delete}")
            branch_subgraph.remove_edge(picked_edge_to_delete[0],picked_edge_to_delete[-1])
            #nx.draw(G)
            #plt.show()
            

    try:
        edges_in_cycle = nx.find_cycle(branch_subgraph)
    except:
        pass
    else:
        raise Exception("There was still a cycle left after cleanup")
    
    return branch_subgraph


def find_skeletal_distance_along_graph_node_path(G,node_path):
    """
    Purpose: To find the skeletal distance along nodes of
    a graph that represents a skeleton
    
    Pseudocode: 
    1) Get the coordinates of the nodes
    2) Find the distances between consecutive coordinates
    
    Ex: 
    find_skeletal_distance_along_graph_node_path(
                                                G = skeleton_graph,
                                                node_path = cycles_list[0]
                                                )
    
    """
    coordinates = get_node_attributes(G,node_list=node_path)
    total_distance = np.sum(np.linalg.norm(coordinates[:-1] - coordinates[1:],axis=1))
    return total_distance


def find_all_cycles(G, source=None, cycle_length_limit=None,time_limit = 1000):
    import system_utils as su
    try:
        with su.time_limit(time_limit):
            """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
            types which work with min() and > ."""
            if source is None:
                # produce edges for all components
                comp_list = [list(k) for k in list(nx.connected_components(G))]
                nodes=[i[0] for i in comp_list]
            else:
                # produce edges for components with source
                nodes=[source]
            # extra variables for cycle detection:
            cycle_stack = []
            output_cycles = set()

            def get_hashable_cycle(cycle):
                """cycle as a tuple in a deterministic order."""
                m = min(cycle)
                mi = cycle.index(m)
                mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
                if cycle[mi-1] > cycle[mi_plus_1]:
                    result = cycle[mi:] + cycle[:mi]
                else:
                    result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
                return tuple(result)

            for start in nodes:
                #print(f"start = {start}")
                if start in cycle_stack:
                    continue
                cycle_stack.append(start)

                stack = [(start,iter(G[start]))]
                while stack:
                    #print(f"len(stack) = {len(stack)}")
                    parent,children = stack[-1]
                    try:
                        child = next(children)

                        if child not in cycle_stack:
                            cycle_stack.append(child)
                            stack.append((child,iter(G[child])))
                        else:
                            i = cycle_stack.index(child)
                            if i < len(cycle_stack) - 2: 
                                output_cycles.add(get_hashable_cycle(cycle_stack[i:]))

                    except StopIteration:
                        stack.pop()
                        cycle_stack.pop()
    except su.TimeoutException as e:
        print("Timed out when trying to find the cycles!")
        return []
        
    
    cycles_list = [list(i) for i in output_cycles]
    cycles_list_array = np.array(cycles_list)
    sorted_list = np.argsort([len(k) for k in cycles_list_array])[::-1]
    cycles_list_sorted = cycles_list_array[sorted_list]
    return list(cycles_list_sorted) 


def set_node_data(curr_network,node_name,curr_data,curr_data_label):
    
        node_attributes_dict = dict()
        if node_name not in list(curr_network.nodes()):
                raise Exception(f"Node {node_name} not in the concept map of the curent neuron before trying to add {node_name} to the concept graph")
        else:
            node_attributes_dict[node_name] = {curr_data_label:curr_data}
                
        #setting the actual attributes
        set_node_attributes_dict(curr_network,node_attributes_dict)

class GraphOrderedEdges(nx.Graph):
    """
    Example of how to use:
    - graph that has ordered edges
    
    xu = reload(xu)

    ordered_Graph = xu.GraphEdgeOrder()
    ordered_Graph.add_edge(1,2)
    ordered_Graph.add_edge(4,3)
    ordered_Graph.add_edge(1,3)
    ordered_Graph.add_edge(3,4)

    ordered_Graph.add_edges_from([(5,6),(2,3),(3,8)])
    xu.get_edge_attributes(ordered_Graph)

    xu.get_edge_attributes(ordered_Graph,"order")
    """
    def __init__(self,data=None,edge_order=dict()):
        super().__init__(data)
        if len(edge_order) > 0 and len(self.edges()) > 0 :
            #set the edge order
            nx.set_edge_attributes(self,name="order",values=dict([(tuple(k),edge_order[tuple(k)]) for k in list(self.edges())]))
            
        
    
    #make sure don't lose properties when turning to undirected
#     def to_undirected(self):
#         edge_order = get_edge_attributes(self)
#         #super().to_undirected()
#         #self.__init__(self,edge_order=edge_order)
        
    
    #just want to add some functions that ordered edges 
    def add_edge(self,u,v):
        """
        Will add the edge plus an order index
        """
        total_edges = len(self.edges())
        #print(f"Total edges already = {total_edges}")
        super().add_edge(u,v,order=total_edges)
    
    #will do the adding edges
    def add_edges_from(self,ebunch_to_add, **kwargs):
        
        #get the total edges
        total_edges = len(self.edges())
        #get the new labels
        
        ebunch_to_add = list(ebunch_to_add)
        
        if len(ebunch_to_add) > 0:
            #add the edges
            super().add_edges_from(ebunch_to_add,**kwargs)
            
            #changes the ebunch if has dictionary associated with it
            if len(ebunch_to_add[0]) == 3:
                ebunch_to_add = [(k,v) for k,v,z in ebunch_to_add]
                
            ending_edge_count = len(self.edges())
            
            new_orders= list(range(total_edges,total_edges + len(ebunch_to_add)))
            
#             print(f"total_edges = {total_edges}")
#             print(f"len(new_orders) = {len(new_orders)}")
#             print(f"len(ebunch_to_add) = {len(ebunch_to_add)}")
#             print(f"ending_edge_count = {ending_edge_count}")
            
            nx.set_edge_attributes(self,name="order",values=dict([(tuple(k),v) for v,k in zip(new_orders,ebunch_to_add)]))
        
    def add_weighted_edges_from(self, ebunch_to_add, weight='weight', **kwargs):
        
        self.add_edges_from(((u, v, {weight: d}) for u, v, d in ebunch_to_add),
                            **kwargs)
    
    #will get the edges in an ordered format
    def edges_ordered(self,*attr):
        """
        nbunch: 
        """
        try:
            returned_edges = np.array(list(super().edges(*attr))).astype("int")
        except:
            returned_edges = np.array(list(super().edges(*attr)))
        #print(f"returned_edges = {returned_edges}")
        #get the order of all of these edges
        if len(returned_edges)>0:
            try:
                orders = np.array(get_edge_attributes(self,attribute="order",edge_list=returned_edges)).astype("int")
            except:
                orders = np.array(get_edge_attributes(self,attribute="order",edge_list=returned_edges))
            return returned_edges[np.argsort(orders)]
        else:
            return returned_edges
        
    def reorder_edges(self):
        
        ord_ed = self.edges_ordered()
        if len(ord_ed)>0:
            nx.set_edge_attributes(self,name="order",values=dict([(tuple(k),v) for v,k in enumerate(ord_ed)]))
        else:
            pass #do nothing because no edges to reorder
        
    #functions that will do the deleting of edges and then reordering
    def remove_edge(self,u,v):
        print("in remove edge")
        super().remove_edge(u,v)
        self.reorder_edges()
    
    
    #the add_weighted_edges_from will already use the add_edges_from
    def remove_edges_from(self,ebunch):
        super().remove_edges_from(ebunch)
        self.reorder_edges()
    
    #*************** overload delete vertex************
    def remove_node(self,n):
        super().remove_node(n)
        self.reorder_edges()
    
    def remove_nodes_from(self,nodes):
        super().remove_nodes_from(nodes)
        self.reorder_edges()
        
# ------------------ for neuron package -------------- #
def get_starting_node(G,attribute_for_start="starting_coordinate",only_one=True):
    starting_node_dict = get_all_nodes_with_certain_attribute_key(G,attribute_for_start)
    
    if only_one:
        if len(starting_node_dict) != 1:
            raise Exception(f"The number of starting nodes was not equal to 1: starting_node_dict = {starting_node_dict}")

        starting_node = list(starting_node_dict.keys())[0]
        return starting_node
    else:
        return list(starting_node_dict.keys())


def compare_networks(
    G1,
    G2,
    compare_edge_attributes=["all"],
    compare_edge_attributes_exclude=[],
    edge_threshold_attributes = ["weight"],
    edge_comparison_threshold=0,
    compare_node_attributes=["all"], 
    compare_node_attributes_exclude=[],
    node_threshold_attributes = ["coordinates","starting_coordinate","endpoints"],
    node_comparison_threshold=0,
    return_differences=False,
    print_flag=False
    ):
    """
    Purpose: To customly compare graphs based on the edges attributes and nodes you want to compare
    AND TO MAKE SURE THEY HAVE THE SAME NODE NAMES
    
    
    G1,G2,#the 2 graphs that will be compared
    compare_edge_attributes=[],#whether to consider the edge attributes when comparing
    edge_threshold_attributes = [], #the names of attributes that will be considered close if below edge_comparison_threshold
    edge_comparison_threshold=0, #the threshold for comparing the attributes named in edge_threshold_attributes
    compare_node_attributes=[], #whether to consider the node attributes when comparing
    node_threshold_attributes = [], #the names of attributes that will be considered close if below node_comparison_threshold
    node_comparison_threshold=0, #the threshold for comparing the attributes named in node_threshold_attributes
    print_flag=False
    
    
    Pseudocode:
    0) check that number of edges and nodes are the same, if not then return false
    1) compare the sorted edges array to see if equal
    2) compare the edge weights are the same (maybe within a threshold) (BUT MUST SPECIFY THRESHOLD)
    3) For each node name: 
    - check that the attributes are the same
    - can specify attribute names that can be within a certian threshold (BUT MUST SPECIFY THRESHOLD)
    
    
    Example: 
    # Testing of the graph comparison 
    network_copy = limb_concept_network.copy()
    network_copy_2 = network_copy.copy()

    #changing and seeing if we can pick up on the difference
    network_copy[1][2]["order"] = 55
    network_copy_2.nodes[2]["endpoints"] = np.array([[1,2,3],[4,5,6]])
    network_copy_2.nodes[3]["endpoints"] = np.array([[1,2,5],[4,5,6]])
    network_copy_2.remove_edge(1,2)

    xu.compare_networks(
        G1=network_copy,
        G2=network_copy_2,
        compare_edge_attributes=["all"],
        edge_threshold_attributes = [],
        edge_comparison_threshold=0,
        compare_node_attributes=["endpoints"], 
        node_threshold_attributes = ["endpoints"],
        node_comparison_threshold=0.1,
        print_flag=True
        )
        
    Example with directional: 
    #directional test 

    network_copy = limb_concept_network.copy()
    network_copy_2 = network_copy.copy()

    #changing and seeing if we can pick up on the difference

    network_copy_2.nodes[2]["endpoints"] = np.array([[1,2,3],[4,5,6]])
    network_copy_2.nodes[3]["endpoints"] = np.array([[1,2,5],[4,5,6]])
    del network_copy_2.nodes[12]["starting_coordinate"]
    #network_copy_2.remove_edge(1,2)

    xu.compare_networks(
        G1=network_copy,
        G2=network_copy_2,
        compare_edge_attributes=["all"],
        edge_threshold_attributes = [],
        edge_comparison_threshold=0,
        compare_node_attributes=["all"], 
        node_threshold_attributes = ["endpoints"],
        compare_node_attributes_exclude=["data"],
        node_comparison_threshold=0.1,
        print_flag=True
        )
    
    Example on how to use to compare skeletons: 
    skeleton_1 = copy.copy(total_skeleton)
    skeleton_2 = copy.copy(total_skeleton)
    skeleton_1[0][0] = np.array([558916.8, 1122107. ,  842972.8])

    sk_1_graph = sk.convert_skeleton_to_graph(skeleton_1)
    sk_2_graph = sk.convert_skeleton_to_graph(skeleton_2)

    xu.compare_networks(sk_1_graph,sk_2_graph,print_flag=True,
                     edge_comparison_threshold=2,
                     node_comparison_threshold=2)
    
    """
    
    total_compare_time = time.time()
    
    local_compare_time = time.time()
    if not nu.is_array_like(compare_edge_attributes):
        compare_edge_attributes = [compare_edge_attributes]
    
    if not nu.is_array_like(compare_edge_attributes):
        compare_edge_attributes = [compare_edge_attributes]
      
    differences_list = []
    return_value = None
    for i in range(0,1):
        #0) check that number of edges and nodes are the same, if not then return false
        if str(type(G1)) != str(type(G1)):
            differences_list.append(f"Type of G1 graph ({type(G1)}) does not match type of G2 graph({type(G2)})")
            break

        if len(G1.edges()) != len(G2.edges()):
            differences_list.append(f"Number of edges in G1 ({len(G1.edges())}) does not match number of edges in G2 ({len(G2.edges())})")
            break

        if len(G1.nodes()) != len(G2.nodes()):
            differences_list.append(f"Number of nodes in G1 ({len(G1.nodes())}) does not match number of nodes in G2 ({len(G2.nodes())})")
            break

        if set(list(G1.nodes())) != set(list(G2.nodes())):
            differences_list.append(f"Nodes in G1 ({set(list(G1.nodes()))}) does not match nodes in G2 ({set(list(G2.nodes()))})")
            break

        if print_flag: 
            print(f"Total time for intial checks: {time.time() - local_compare_time}")
        local_compare_time = time.time()

        #1) compare the sorted edges array to see if equal
        unordered_bool = str(type(G1)) == str(type(nx.Graph())) or str(type(G1)) == str(type(GraphOrderedEdges()))

        if print_flag:
            print(f"unordered_bool = {unordered_bool}")

        #already checked for matching edge length so now guard against the possibility that no edges:
        if len(list(G1.edges())) > 0:


            G1_sorted_edges = nu.sort_multidim_array_by_rows(list(G1.edges()),order_row_items=unordered_bool)
            G2_sorted_edges = nu.sort_multidim_array_by_rows(list(G2.edges()),order_row_items=unordered_bool)


            if not np.array_equal(G1_sorted_edges,G2_sorted_edges):
                differences_list.append("The edges array are not equal ")
                break

            if print_flag: 
                print(f"Total time for Sorting and Comparing Edges: {time.time() - local_compare_time}")
            local_compare_time = time.time()

            #2) compare the edge weights are the same (maybe within a threshold) (BUT MUST SPECIFY THRESHOLD)
            if print_flag:
                print(f"compare_edge_attributes = {compare_edge_attributes}")
            if len(compare_edge_attributes)>0:

                G1_edge_attributes = get_all_attributes_for_edges(G1,edges_list=G1_sorted_edges,return_dict=True)
                G2_edge_attributes = get_all_attributes_for_edges(G2,edges_list=G2_sorted_edges,return_dict=True)

                """
                loop that will go through each edge and compare the dictionaries:
                - only compare the attributes selected (compare all if "all" in list)
                - if certain attributes show up in the edge_threshold_attributes then compare then against the edge_comparison_threshold
                """

                for z,curr_edge in enumerate(G1_edge_attributes.keys()):
                    G1_edge_dict = G1_edge_attributes[curr_edge]
                    G2_edge_dict = G2_edge_attributes[curr_edge]
                    #print(f"G1_edge_dict.keys() = {G1_edge_dict.keys()}")

                    if "all" not in compare_edge_attributes:
                        G1_edge_dict = dict([(k,v) for k,v in G1_edge_dict.items() if k in compare_edge_attributes])
                        G2_edge_dict = dict([(k,v) for k,v in G2_edge_dict.items() if k in compare_edge_attributes])

                    #do the exclusion of some attributes:
                    G1_edge_dict = dict([(k,v) for k,v in G1_edge_dict.items() if k not in compare_edge_attributes_exclude])
                    G2_edge_dict = dict([(k,v) for k,v in G2_edge_dict.items() if k not in compare_edge_attributes_exclude])

                    if z ==1:
                        if print_flag:
                            print(f"Example G1_edge_dict = {G1_edge_dict}")


                    #check that they have the same number of keys
                    if set(list(G1_edge_dict.keys())) != set(list(G2_edge_dict.keys())):
                        differences_list.append(f"The dictionaries for the edge {curr_edge} did not have same keys in G1 ({G1_edge_dict.keys()}) as G2 ({G2_edge_dict.keys()})")
                        continue
                        #return False
                    #print(f"G1_edge_dict.keys() = {G1_edge_dict.keys()}")
                    #check that all of the values for each key match
                    for curr_key in G1_edge_dict.keys():
                        #print(f"{(G1_edge_dict[curr_key],G2_edge_dict[curr_key])}")
                        if curr_key in edge_threshold_attributes:
                            value_difference = np.linalg.norm(G1_edge_dict[curr_key]-G2_edge_dict[curr_key])
                            if value_difference > edge_comparison_threshold:
                                differences_list.append(f"The edge {curr_edge} has a different value for {curr_key} in G1 ({G1_edge_dict[curr_key]}) and in G2 ({G2_edge_dict[curr_key]}) "
                                     f"that was above the current edge_comparison_threshold ({edge_comparison_threshold}) ")
                                #return False
                        else:
                            if nu.is_array_like(G1_edge_dict[curr_key]):
                                if not np.array_equal(G1_edge_dict[curr_key],G2_edge_dict[curr_key]):
                                    differences_list.append(f"The edge {curr_edge} has a different value for {curr_key} in G1 ({G1_edge_dict[curr_key]}) and in G2 ({G2_edge_dict[curr_key]}) ")
                            else:
                                if G1_edge_dict[curr_key] != G2_edge_dict[curr_key]:
                                    differences_list.append(f"The edge {curr_edge} has a different value for {curr_key} in G1 ({G1_edge_dict[curr_key]}) and in G2 ({G2_edge_dict[curr_key]}) ")
                                    #return False

        #if no discrepancy has been detected then return True
        if len(differences_list) == 0:
            if print_flag:
                print("Made it through edge comparison without there being any discrepancies")

        if print_flag: 
            print(f"Total time for Checking Edges Attributes : {time.time() - local_compare_time}")
        local_compare_time = time.time()

        """
        3) For each node name: 
        - check that the attributes are the same
        - can specify attribute names that can be within a certian threshold (BUT MUST SPECIFY THRESHOLD)
        """

        if len(compare_node_attributes)>0:

            G1_node_attributes = get_all_attributes_for_nodes(G1,return_dict=True)
            G2_node_attributes = get_all_attributes_for_nodes(G2,return_dict=True)

            """
            loop that will go through each node and compare the dictionaries:
            - only compare the attributes selected (compare all if "all" in list)
            - if certain attributes show up in the node_threshold_attributes then compare then against the node_comparison_threshold
            """
            if print_flag:
                print(f"compare_node_attributes = {compare_node_attributes}")
            for z,n in enumerate(G1_node_attributes.keys()):
                G1_node_dict = G1_node_attributes[n]
                G2_node_dict = G2_node_attributes[n]

                if "all" not in compare_node_attributes:
                    G1_node_dict = dict([(k,v) for k,v in G1_node_dict.items() if k in compare_node_attributes])
                    G2_node_dict = dict([(k,v) for k,v in G2_node_dict.items() if k in compare_node_attributes])


                #doing the exlusion
                G1_node_dict = dict([(k,v) for k,v in G1_node_dict.items() if k not in compare_node_attributes_exclude])
                G2_node_dict = dict([(k,v) for k,v in G2_node_dict.items() if k not in compare_node_attributes_exclude])

                if z ==1:
                    if print_flag:
                        print(f"Example G1_edge_dict = {G1_node_dict}")


                #check that they have the same number of keys
                if set(list(G1_node_dict.keys())) != set(list(G2_node_dict.keys())):
                    differences_list.append(f"The dictionaries for the node {n} did not have same keys in G1 ({G1_node_dict.keys()}) as G2 ({G2_node_dict.keys()})")
                    continue
                    #return False

                #check that all of the values for each key match
                for curr_key in G1_node_dict.keys():
                    #print(f"curr_key = {curr_key}")

                    if curr_key in node_threshold_attributes:
                        value_difference = np.linalg.norm(G1_node_dict[curr_key]-G2_node_dict[curr_key])
                        if value_difference > node_comparison_threshold:
                            differences_list.append(f"The node {n} has a different value for {curr_key} in G1 ({G1_node_dict[curr_key]}) and in G2 ({G2_node_dict[curr_key]}) "
                                 f"that was above the current node_comparison_threshold ({node_comparison_threshold}) ")
                            #return False
                    else:

                        if nu.is_array_like(G1_node_dict[curr_key]):
                            #print((set(list(G1_node_dict.keys())),set(list(G2_node_dict.keys()))))
                            if not np.array_equal(G1_node_dict[curr_key],G2_node_dict[curr_key]):
                                differences_list.append(f"The node {n} has a different value for {curr_key} in G1 ({G1_node_dict[curr_key]}) and in G2 ({G2_node_dict[curr_key]}) ")
                                #return False
                        else:
                            #print(f"curr_key = {curr_key}")
                            #print(f"G1_node_dict[curr_key] != G2_node_dict[curr_key] = {G1_node_dict[curr_key] != G2_node_dict[curr_key]}")
                            #print(f"G1_node_dict[curr_key] = {G1_node_dict[curr_key]}, G2_node_dict[curr_key] = {G2_node_dict[curr_key]}")
                            if G1_node_dict[curr_key] != G2_node_dict[curr_key]:
                                differences_list.append(f"The node {n} has a different value for {curr_key} in G1 ({G1_node_dict[curr_key]}) and in G2 ({G2_node_dict[curr_key]}) ")
                                #return False
                        #print(f"differences_list = {differences_list}")

        if print_flag: 
            print(f"Total time for Comparing Node Attributes: {time.time() - local_compare_time}")
        local_compare_time = time.time()
    
    #if no discrepancy has been detected then return True
    
    if len(differences_list) == 0:
        if print_flag:
            print("Made it through Node comparison without there being any discrepancies")
        return_boolean = True
    else:
        if print_flag:
            print("Differences List:")
            for j,diff in enumerate(differences_list):
                print(f"{j})   {diff}")
        return_boolean = False
    
    if return_differences is None:
        raise Exception("return_differences is None!!!")
        
    if return_differences:
        return return_boolean,differences_list
    else:
        return return_boolean
    
# -------------- 8/4 additions ----------------------- #
"""
How to determine upstream and downstream targets

Example: 
import networkx as nx
import matplotlib.pyplot as plt
G = nx.DiGraph()
G.add_edges_from(
    [('A', 'B'), ('A', 'C'), ('D', 'B'), ('E', 'C'), ('E', 'F'),('F','Z'),
     ('B', 'H'), ('B', 'G'), ('B', 'F'), ('C', 'G'), ('Q', 'D')])

pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),node_size = 50)
nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)
nx.draw_networkx_labels(G, pos)
plt.show()

print("Downstream Edges of 'B' (just example)-->")
print(list(nx.dfs_edges(G,'B')))
print(downstream_edges(G,"B"))
print(downstream_edges_neighbors(G,"B"))


print("\nUpstream Edges of 'B' (just example)-->")
print(list(nx.edge_dfs(G,'B', orientation='reverse')))
print(upstream_edges(G,"B"))
print(upstream_edges_neighbors(G,"B"))

"""
def downstream_edges(G,node):
    return list(nx.dfs_edges(G,node))
def downstream_edges_neighbors(G,node):
    return [k for k in list(nx.dfs_edges(G,node)) if node in k]

def upstream_edges(G,node):
    return list(nx.edge_dfs(G,node, orientation='reverse'))
def upstream_edges_neighbors(G,node):
    return [k for k in list(nx.edge_dfs(G,node, orientation='reverse')) if node in k]
    
def upstream_node(G,node,return_single=True):
    curr_upstream_nodes = upstream_edges_neighbors(G,node)
        
    if len(curr_upstream_nodes) == 0:
        return None
    elif len(curr_upstream_nodes) > 1:
        raise Exception(f"More than one upstream node for node {node}: {curr_upstream_nodes}")
    else:
        return curr_upstream_nodes[0][0]
       
    
# --------------------- 8/31 -------------------------- #
import random


def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5,width_min = 0.3,width_noise_ampl=0.2):
    '''
    
    Old presets: 
    width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5
    
    
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
    Licensed under Creative Commons Attribution-Share Alike 

    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch 
    - if the tree is directed and this is not given, 
      the root will be found and used
    - if the tree is directed and this is given, then 
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given, 
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''
        
        if np.abs(width) < width_min:
            width = np.sign(width)*(width_min + width_noise_ampl*np.random.rand(1)[0])
            #width = width_min 
            #print(f"width_noise_ampl = {width_noise_ampl}")
        
        #print(f"root {root}: inside _hierarchy_pos: width = {width}, xcenter={xcenter}")

        if pos is None:
            pos = {root:(xcenter,vert_loc)} #if no previous position then start dictioanry
        else:
            pos[root] = (xcenter, vert_loc) #if dictionary already exists then add position to dictionary
        children = list(G.neighbors(root)) #get all children of current root
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent) #remove parent from possible neighbors (so only get ones haven't used) 
        if len(children)!=0: #if have children to graph
            dx = width/len(children)  #take whole width and allocates for how much each child gets
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                """
                How recursive call works: 
                1) the dx allocated for each child becomes the next width
                2) same vertical gap
                3) New vertical location is original but stepped down vertical gap
                3) New x location is the nextx
                
                """
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos


    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    
    
# --------- 9/17 Addition ----------------------- #
from copy import deepcopy
def shortest_path_between_two_sets_of_nodes(G,node_list_1,node_list_2,
                                           return_node_pairs=True):
    """
    Algorithm that will find the shortest path from a set of one
    list of nodes on a graph and another set of nodes:
    
    Returns: The shortest path, the nodes from each set that were paired
    
    Things to think about:
    - could possibly have non-overlapping groups
    
    Pseudocode:
    0) Make a copy of the graph
    1) Add a new node to graph that is connected to all nodes in node_list_1 (s)
    2) Add a new node to graph that is connected to all nodes in node_list_2 (t)
    3) Find shortest path from s to t
    4) remove s and t from path and return the two endpoints of path as node pair
    
    Example: 
    import networkx as nx
    G = nx.path_graph(10)
    node_list_1 = [1,2]
    node_list_2 = [9,5]
    
    shortest_path_between_two_sets_of_nodes(G,node_list_1,node_list_2,
                                           return_node_pairs=True)
    
    will return [ 2, 3, 4, 5],2,5
    """
    #0) Make a copy of the graph
    G_copy = deepcopy(G)
    node_number_max = np.max(G.nodes())

    #1) Add a new node to graph that is connected to all nodes in node_list_1 (s)
    s = node_number_max + 1
    G_copy.add_weighted_edges_from([(s,k,0.0001) for k in node_list_1])

    #2) Add a new node to graph that is connected to all nodes in node_list_2 (t)
    t = node_number_max + 2
    G_copy.add_weighted_edges_from([(t,k,0.0001) for k in node_list_2])

    #3) Find shortest path from s to t
    shortest_path = nx.shortest_path(G_copy,s,t,weight="weight")
    

    #node_pair
    curr_shortest_path = shortest_path[1:-1]
    end_node_1 = shortest_path[1]
    end_node_2 = shortest_path[-2]
    
    #make sure it is the shortest path between end nodes
    curr_shortest_path = nx.shortest_path(G,end_node_1,end_node_2)
    
    if return_node_pairs:
        return curr_shortest_path,end_node_1,end_node_2
    else:
        return curr_shortest_path
    
    
def find_nodes_within_certain_distance_of_target_node(G,
                                                      target_node,
                                                        cutoff_distance = 10000,
                                                        return_dict=False):
    """
    Purpose: To Find the node values that are within a certain 
    distance of a target node 
    
    """
    distance_dict = nx.single_source_dijkstra_path_length(G,target_node,
                                                          cutoff=cutoff_distance
                                                         )
    if return_dict:
        return distance_dict
    
    close_nodes = set(np.array(list(distance_dict)).astype("int"))
    return close_nodes
    
    
def add_new_coordinate_node(G,
    node_coordinate,
    replace_nodes=None,
    replace_coordinates=None,
    neighbors=None,
    node_id=None,
                           return_node_id=True):
    """
    To add a node to a graph
    with just a coordinate and potentially replacing 
    another node
    """
    
    #G = copy.deepcopy(G)
    
    if not replace_coordinates is None:
        if len(replace_coordinates.shape) < 2:
            replace_coordinates=replace_coordinates.reshape(-1,3)
            
        replace_nodes = [get_graph_node_by_coordinate(G,k) for k in replace_coordinates]

#     print(f"replace_coordinates = {replace_coordinates}")
#     print(f"replace_nodes = {replace_nodes}")
#     print(f"len(G) = {len(G)}")

    if not replace_nodes is None:
        if not nu.is_array_like(replace_nodes):
            replace_nodes = [replace_nodes]
        neighbors = np.unique(np.concatenate([get_neighbors(G,k) for k in replace_nodes]))
    
    if node_id is None:
        node_id = np.max(G.nodes()) + 1
        
    G.add_node(node_id,coordinates=node_coordinate)

    G.add_weighted_edges_from([(node_id,k,
             np.linalg.norm(G.nodes[k]["coordinates"] - node_coordinate)) for k in neighbors])
    
    if not replace_nodes is None:
        G.remove_nodes_from(replace_nodes)
        
    if return_node_id:
        return G,node_id
    else:
        return G
    
    

    
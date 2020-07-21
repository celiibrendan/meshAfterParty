import networkx as nx
import numpy as np
import numpy_utils as nu
import time

def compare_endpoints(endpoints_1,endpoints_2,**kwargs):
    """
    comparing the endpoints of a graph: 
    
    Ex: 
    import networkx_utils as xu
    xu = reload(xu)
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
def get_nodes_with_attributes_dict(G,attribute_dict):
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
                    if not np.array_equiv(np.array(curr_search_val),attribute_dict[search_key]):
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

def get_all_nodes_with_certain_attribute_key(G,attribute_name):
    return nx.get_node_attributes(G,attribute_name)

def get_node_attributes(G,attribute_name="coordinates",node_list=[],
                       return_array=True):
    #print(f"attribute_name = {attribute_name}")
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
        returned_edges = np.array(list(super().edges(*attr))).astype("int")
        #print(f"returned_edges = {returned_edges}")
        #get the order of all of these edges
        if len(returned_edges)>0:
            orders = np.array(get_edge_attributes(self,attribute="order",edge_list=returned_edges)).astype("int")
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
def get_starting_node(G,attribute_for_start="starting_coordinate"):
    starting_node_dict = get_all_nodes_with_certain_attribute_key(G,attribute_for_start)
        
    if len(starting_node_dict) != 1:
        raise Exception(f"The number of starting nodes was not equal to 1: starting_node_dict = {starting_node_dict}")

    starting_node = list(starting_node_dict.keys())[0]
    return starting_node


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
        
    
    
    
    
    
    

    
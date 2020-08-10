import networkx as nx
import numpy as np

#neuron module specific imports
import compartment_utils as cu
import matplotlib_utils as mu
import networkx_utils as xu
import numpy_utils as nu
import skeleton_utils as sk
import trimesh_utils as tu
import time
import soma_extraction_utils as sm
import system_utils as su

import neuron_visualizations as nviz
import neuron_utils as nru
from pathlib import Path
import algorithms_utils as au
import neuron_searching as ns

import width_utils as wu

import copy 

import sys
current_module = sys.modules[__name__]


import spine_utils as spu


def export_skeleton(self,subgraph_nodes=None):
    """
    
    """
    if subgraph is None:
        total_graph = self.neuron_graph
    else:
        pass
        # do some subgraphing
        #total_graph = self.neuron_graph.subgraph([])
    
    #proce


def export_mesh_labels(self):
    """
    
    """
    pass


def convert_soma_to_piece_connectivity_to_graph(soma_to_piece_connectivity):
    """
    Pseudocode: 
    1) Create the edges with the new names from the soma_to_piece_connectivity
    2) Create a GraphOrderedEdges from the new edges
    
    Ex: 
        
    concept_network = convert_soma_to_piece_connectivity_to_graph(current_mesh_data[0]["soma_to_piece_connectivity"])
    nx.draw(concept_network,with_labels=True)
    """
    
    total_edges = []
    for soma_key,list_of_limbs in soma_to_piece_connectivity.items():
        total_edges += [[f"S{soma_key}",f"L{curr_limb}"] for curr_limb in list_of_limbs]
    
    print(f"total_edges = {total_edges}")
    concept_network = xu.GraphOrderedEdges()
    concept_network.add_edges_from(total_edges)
    return concept_network 

    
from copy import deepcopy as dc

def dc_check(current_object,attribute):
    try:
        return getattr(current_object,attribute)
    except:
        return None

def copy_concept_network(curr_network):
    copy_network = dc(curr_network)

    for n in copy_network.nodes():
        """ Old way that didn't account for dynamic updates
        current_node_class = copy_network.nodes[n]["data"].__class__
        #print(f"current_node_class = {current_node_class}")
        copy_network.nodes[n]["data"] = current_node_class(copy_network.nodes[n]["data"])
        
        """
        
        """
        New way:
        1) get the name of the class
        2) get a reference to the definition based on the current module definition 
        3) Use that instantiation
        
        """
        current_node_class_name = copy_network.nodes[n]["data"].__class__.__name__
        class_constructor = getattr(current_module,current_node_class_name)
        copy_network.nodes[n]["data"] = class_constructor(copy_network.nodes[n]["data"])
        
        
    return copy_network

class Branch:
    """
    Class that will hold one continus skeleton
    piece that has no branching
    """
    
    #def __init__(self,branch_skeleton,mesh=None,mesh_face_idx=None,width=None):
        
    def __init__(self,
                skeleton,
                width=None,
                mesh=None,
                mesh_face_idx=None,
                 labels=[] #for any labels of that branch
        ):
        
        if str(type(skeleton)) == str(Branch):
            #print("Recived Branch object so copying object")
            #self = copy.deepcopy(skeleton)
            self.skeleton = dc(skeleton.skeleton).reshape(-1,2,3)
            self.mesh=dc(skeleton.mesh)
            self.width = dc(skeleton.width)
            self.mesh_face_idx = dc(skeleton.mesh_face_idx)
            self.endpoints = dc(skeleton.endpoints)
            self.mesh_center = dc(skeleton.mesh_center)
            
            
            
            if not self.mesh is None:
                self.mesh_center = tu.mesh_center_vertex_average(self.mesh)
            self.labels=dc(skeleton.labels)
            if not nu.is_array_like(self.labels):
                self.labels=[self.labels]
                
                
            self.spines = dc_check(skeleton,"spines")
            self.width_new = dc_check(skeleton,"width_new")
            if self.width_new is None:
                self.width_new = dict()
            self.width_array = dc_check(skeleton,"width_array")
            if self.width_array is None:
                self.width_array = dict()
            return 
            
        self.skeleton=skeleton.reshape(-1,2,3)
        self.mesh=mesh
        self.width=width
        self.mesh_face_idx = mesh_face_idx
        
        #calculate the end coordinates of skeleton branch
        self.endpoints = sk.find_branch_endpoints(skeleton)
        self.mesh_center = None
        if not self.mesh is None:
            self.mesh_center = tu.mesh_center_vertex_average(self.mesh)
        self.labels=labels
        if not nu.is_array_like(self.labels):
            self.labels=[self.labels]
            
        
        self.spines = None
        self.width_new = dict()
        self.width_array = dict()
        
        
        
    """
    --- Branch comparison:
    'endpoints', #sort them by row and then subtract and then if distance is within threshold then equal
    'labels',
    'mesh',
    'mesh_center',
    'skeleton', : how do we compare skeletons? (compare just like a networkx graph)
    --> convert to networkx graph
    --> define functions that make nodes equal if within a certain distance
    --> define function that makes edges equal if have same distance
    'width'
    """
        
    def __eq__(self,other):
        #print("inside equality function")
        """
        Purpose: Computes equality of all members except the face_idx (will print out if not equal)
        
        Examples: 
        
        tu = reload(tu)
        neuron= reload(neuron)
        B1 = neuron.Branch(copy.deepcopy(double_soma_obj.concept_network.nodes["L1"]["data"].concept_network.nodes[0]["data"]))
        B2= copy.deepcopy(B1)

        #initial comparison was equal
        #B1 == B2


        #when changed the skeleton was not equal
        # B2.skeleton[0][0] = [2,3,4]
        # B1 == B2

        #when changed the labels was not equal
        # B1.labels=["new Labels"]
        # B1 == B2

        B2.mesh = B2.mesh.submesh([np.arange(0,len(B2.mesh.faces)-5)],append=True)
        B1 != B2

        # B1 != B2
        
        """
        print_flag = True
        differences = []
        
        #comparing the meshes
        if not tu.compare_meshes_by_face_midpoints(self.mesh,other.mesh):
            differences.append(f"mesh didn't match"
                               f"\n    self.mesh = {self.mesh},"
                               f" other.mesh = {other.mesh}") 
        
        if not xu.compare_endpoints(self.endpoints,other.endpoints):
            differences.append(f"endpoints didn't match: "
                               f"\n    self.endpoints = {self.endpoints}, other.endpoints = {other.endpoints}")
        
        if set(self.labels) != set(other.labels):
            differences.append(f"labels didn't match: "
                               f"\n    self.labels = {self.labels}, other.labels = {other.labels}")
            
        if not nu.compare_threshold(self.mesh_center,other.mesh_center):
            differences.append(f"mesh_center didn't match: "
                               f"\n    self.mesh_center = {self.mesh_center}, other.mesh_center = {other.mesh_center}")
            
        if not nu.compare_threshold(self.width,other.width):
            differences.append(f"width didn't match: "
                               f"\n    self.width = {self.width}, other.width = {other.width}")
            
        if not sk.compare_skeletons_ordered(self.skeleton,other.skeleton):
            differences.append(f"Skeleton didn't match")
        
        #print out if face idx was different but not make part of the comparison
        if set(self.mesh_face_idx) != set(other.mesh_face_idx):
            #print("*** Warning: mesh_face_idx didn't match (but not factored into equality comparison)")
            pass
        
        
        if len(differences) == 0:
            return True
        else:
            if print_flag:
                print("Differences List:")
                for j,diff in enumerate(differences):
                    print(f"{j})   {diff}")
            return False
        
    
    def __ne__(self,other):
        #print(f"self.__eq__(other) = {self.__eq__(other)}")
        return not self.__eq__(other)
    
            




class Limb:
    """
    Class that will hold one continus skeleton
    piece that has no branching (called a limb)
    
    3) Limb Process: For each limb made 
    a. Build all the branches from the 
        - mesh
        - skeleton
        - width
        - branch_face_idx
    b. Pick the top concept graph (will use to store the nodes)
    c. Put the branches as "data" in the network
    d. Get all of the starting coordinates and starting edges and put as member attributes in the limb
    """
    def get_skeleton(self,check_connected_component=True):
        """
        Purpose: Will return the entire skeleton of all the branches
        stitched together
        
        """
        return nru.convert_limb_concept_network_to_neuron_skeleton(self.concept_network,
                             check_connected_component=check_connected_component)
    
    @property
    def skeleton(self,check_connected_component=False):
        """
        Purpose: Will return the entire skeleton of all the branches
        stitched together
        
        """
        return nru.convert_limb_concept_network_to_neuron_skeleton(self.concept_network,
                             check_connected_component=check_connected_component)
    
    @property
    def concept_network_data_by_soma(self):
        #compile a dictionary of all of the starting material
        return_dict = dict()
        for curr_data in self.all_concept_network_data:
            return_dict[curr_data["starting_soma"]] = dict([(k,v) for k,v in curr_data.items() if k != "starting_soma"])
        return return_dict
    
    def touching_somas(self):
        return [k["starting_soma"] for k in self.all_concept_network_data]
    
    
    '''    
        def get_soma_starting_coordinate(self,starting_soma,print_flag=False):
            """
            This function can now be replaced by 
            curr_limb_obj.concept_network_data_by_soma[soma_idx]["starting_coordinate"]

            """
            if starting_soma not in self.touching_somas():
                raise Exception(f"Current limb does not touch soma {starting_soma}")

            matching_concept_network_data = [k for k in self.all_concept_network_data if ((k["starting_soma"] == starting_soma) or (nru.soma_label(k["starting_soma"]) == starting_soma))]

            if len(matching_concept_network_data) != 1:
                raise Exception(f"The concept_network data for the starting soma ({starting_soma}) did not have exactly one match: {matching_concept_network_data}")
            else:
                return matching_concept_network_data[0]["starting_coordinate"]

        '''
    
    def get_skeleton_soma_starting_node(self,soma,print_flag=False):
        """
        Purpose: from the all
        
        """
        #limb_starting_coordinate = self.get_soma_starting_coordinate(soma)
        limb_starting_coordinate = self.concept_network_data_by_soma[soma]["starting_coordinate"]

        if print_flag:
            print(f"limb_starting_coordinate = {limb_starting_coordinate}")
        limb_skeleton_graph = sk.convert_skeleton_to_graph(self.skeleton)

        sk_starting_node = xu.get_nodes_with_attributes_dict(limb_skeleton_graph,
                                      attribute_dict=dict(coordinates=limb_starting_coordinate))
        if len(sk_starting_node) != 1:
            raise Exception(f"Not exactly one skeleton starting node: sk_starting_node = {sk_starting_node}")
        return sk_starting_node[0]
    
    def find_branch_by_skeleton_coordinate(self,target_coordinate):
    
        """
        Purpose: To be able to find the branch where the skeleton point resides

        Pseudocode: 
        For each branch:
        1) get the skeleton
        2) ravel the skeleton into a numpy array
        3) searh for that coordinate:
        - if returns a non empty list then add to list

        """
        matching_node = []
        for n in self.concept_network.nodes():
            curr_skeleton_points = self.concept_network.nodes[n]["data"].skeleton.reshape(-1,3)
            row_matches = nu.matching_rows(curr_skeleton_points,target_coordinate)
            if len(row_matches) > 0:
                matching_node.append(n)

        if len(matching_node) > 1: 
            print(f"***Warning More than one branch skeleton matches the desired corrdinate: {matching_node}")
        elif len(matching_node) == 1:
            matching_node = matching_node[0]
        else:
            raise Exception("No matching branches found")

        return matching_node
    
    
    def convert_concept_network_to_directional(self,no_cycles = True,width_source=None,print_flag=False):
        """
        
        
        Example on how it was developed: 
        
        import numpy as np
        import networkx_utils as xu
        xu = reload(xu)
        import matplotlib.pyplot as plt
        import neuron_utils as nru
        
        curr_limb_idx = 0
        no_cycles = True
        curr_limb_concept_network = my_neuron.concept_network.nodes[f"L{curr_limb_idx}"]["data"].concept_network 
        curr_neuron_mesh =  my_neuron.mesh
        curr_limb_mesh =  my_neuron.concept_network.nodes[f"L{curr_limb_idx}"]["data"].mesh
        nx.draw(curr_limb_concept_network,with_labels=True)
        plt.show()


        mesh_widths = dict([(k,curr_limb_concept_network.nodes[k]["data"].width) for k in curr_limb_concept_network.nodes() ])

        directional_concept_network = nru.convert_concept_network_to_directional(curr_limb_concept_network,no_cycles=True)


        nx.draw(directional_concept_network,with_labels=True)
        plt.show()
        """
        if self.concept_network is None:
            raise Exception("Cannot use convert_concept_nextwork_to_directional on limb if concept_network is None")
            
        curr_limb_concept_network = self.concept_network    
        
        #make sure that there is one and only one starting node embedded in the graph
        try: 
            xu.get_starting_node(curr_limb_concept_network)
        except:
            print("There was not exactly one starting nodes in the current self.concept_network"
                  " when trying to convert to concept network ")
            xu.get_starting_node(curr_limb_concept_network)
        

        if width_source is None:
            #check to see if the mesh center width is available
            try:
                if "no_spine_average_mesh_center" in curr_limb_concept_network.nodes[0]["data"].width_new.keys():
                    width_source = "no_spine_average_mesh_center"
                else:
                    width_source = "width"
            except:
                width_source = "width"
        
        if print_flag:
            print(f"width_source = {width_source}, type = {type(width_source)}")
        
        if width_source == "width":
            if print_flag:
                print("Using the default width")
            node_widths = dict([(k,curr_limb_concept_network.nodes[k]["data"].width) for k in curr_limb_concept_network.nodes() ])
        else:
            if print_flag:
                print(f"Using the {width_source} in width_new that was calculated")
            node_widths = dict([(k,curr_limb_concept_network.nodes[k]["data"].width_new[width_source]) for k in curr_limb_concept_network.nodes() ])
            
        if print_flag:
            print(f"node_widths= {node_widths}")
        directional_concept_network = nru.convert_concept_network_to_directional(
            curr_limb_concept_network,
            node_widths=node_widths,                                                    
            no_cycles=True,
                                                                            )
        
        return directional_concept_network
        
    
    def set_concept_network_directional(self,starting_soma,print_flag=False,**kwargs):
        """
        Pseudocode: 
        1) Get the current concept_network
        2) Delete the current starting coordinate
        3) Use the all_concept_network_data to find the starting node and coordinate for the
        starting soma specified
        4) set the starting coordinate of that node
        5) rerun the convert_concept_network_to_directional and set the output to the self attribute
        Using: 
        self.concept_network_directional = self.convert_concept_network_to_directional(no_cycles = True)
        
        Example: 
        
        import neuron_visualizations as nviz

        curr_limb_obj = recovered_neuron.concept_network.nodes["L1"]["data"]
        print(xu.get_starting_node(curr_limb_obj.concept_network_directional))
        print(curr_limb_obj.current_starting_coordinate)
        print(curr_limb_obj.current_starting_node)
        print(curr_limb_obj.current_starting_endpoints)
        print(curr_limb_obj.current_starting_soma)
        
        nviz.plot_concept_network(curr_limb_obj.concept_network_directional,
                         arrow_size=5,
                         scatter_size=3)
                         
        curr_limb_obj.set_concept_network_directional(starting_soma=1,print_flag=False)
        
        print(xu.get_starting_node(curr_limb_obj.concept_network_directional))
        print(curr_limb_obj.current_starting_coordinate)
        print(curr_limb_obj.current_starting_node)
        print(curr_limb_obj.current_starting_endpoints)
        print(curr_limb_obj.current_starting_soma)

        nviz.plot_concept_network(curr_limb_obj.concept_network_directional,
                                 arrow_size=5,
                                 scatter_size=3)
        
        Example 8/4:
        uncompressed_neuron_revised.concept_network.nodes["L1"]["data"].set_concept_network_directional(starting_soma=0,width_source="width",print_flag=True)
        
        """
        matching_concept_network_data = [k for k in self.all_concept_network_data if ((k["starting_soma"] == starting_soma) or (nru.soma_label(k["starting_soma"]) == starting_soma))]

        if len(matching_concept_network_data) != 1:
            raise Exception(f"The concept_network data for the starting soma ({starting_soma}) did not have exactly one match: {matching_concept_network_data}")
        
        
        
        #find which the starting_coordinate and starting_node

        previous_starting_node = xu.get_starting_node(self.concept_network,only_one=False)
        if len(previous_starting_node) > 1:
            print("**** Warning there were more than 1 starting nodes in concept networks"
                 f"\nprevious_starting_node = {previous_starting_node}")
        if len(previous_starting_node) == 0:
            print("**** Warning there were 0 starting nodes in concept networks"
                 f"\nprevious_starting_node = {previous_starting_node}")
        
        if print_flag:
            print(f"Deleting starting coordinate from nodes: {previous_starting_node}")
            
        for prev_st_node in previous_starting_node:
            del self.concept_network.nodes[prev_st_node]["starting_coordinate"]
        
        

        matching_concept_network_dict = matching_concept_network_data[0]
        curr_starting_node = matching_concept_network_dict["starting_node"]
        curr_starting_coordinate= matching_concept_network_dict["starting_coordinate"]

        #set the starting coordinate in the concept network
        attrs = {curr_starting_node:{"starting_coordinate":curr_starting_coordinate}}
        if print_flag:
            print(f"attrs = {attrs}")
        xu.set_node_attributes_dict(self.concept_network,attrs)

        #make sure only one starting coordinate
        new_starting_coordinate = xu.get_starting_node(self.concept_network)
        if print_flag:
            print(f"New starting coordinate at node {new_starting_coordinate}")
        
        self.current_starting_coordinate = matching_concept_network_dict["starting_coordinate"]
        self.current_starting_node = matching_concept_network_dict["starting_node"]
        self.current_starting_endpoints = matching_concept_network_dict["starting_endpoints"]
        self.current_starting_soma = matching_concept_network_dict["starting_soma"]
        
        if print_flag:
            self.concept_network_directional = self.convert_concept_network_to_directional(no_cycles = True,print_flag=print_flag,**kwargs)
        else:
            with su.suppress_stdout_stderr():
                self.concept_network_directional = self.convert_concept_network_to_directional(no_cycles = True,print_flag=print_flag,**kwargs)
        
        
    
    def __init__(self,
                             mesh,
                             curr_limb_correspondence=None,
                             concept_network_dict=None,
                             mesh_face_idx=None,
                            labels=[]
                            ):
        
        
        """
        Allow for an initialization of a limb with another limb object
        
        Parts that need to be copied over:
        'all_concept_network_data',
         'concept_network',
         'concept_network_directional',
         'current_starting_coordinate',
         'current_starting_endpoints',
         'current_starting_node',
         
         'current_starting_soma',
         'label',
         'mesh',
         'mesh_center',
         'mesh_face_idx'
         
        """
        
        
        if str(type(mesh)) == str(Limb):
            #print("Recived Limb object so copying object")
            # properties we are copying: [k for k in dir(example_limb) if "__" not in k]
            
            self.all_concept_network_data = dc(mesh.all_concept_network_data)
            self.concept_network=copy_concept_network(mesh.concept_network)
            #want to do and copy the meshes in each of the networks to make sure their properties update
            self.concept_network_directional = copy_concept_network(mesh.concept_network_directional)
            #want to do and copy the meshes in each of the networks to make sure their properties update
            self.current_starting_coordinate = dc(mesh.current_starting_coordinate)
            self.current_starting_endpoints = dc(mesh.current_starting_endpoints)
            self.current_starting_node = dc(mesh.current_starting_node)
            
            self.current_starting_soma = dc(mesh.current_starting_soma)
            self.labels = dc(mesh.labels)
            if not nu.is_array_like(self.labels):
                self.labels=[self.labels]
            self.mesh = dc(mesh.mesh)
            self.mesh_center = dc(mesh.mesh_center)
            self.mesh_face_idx = dc(mesh.mesh_face_idx)
            return 
        
        
        
        self.mesh=mesh
        
        #checking that some of arguments aren't None
        if curr_limb_correspondence is None:
            raise Exception("curr_limb_correspondence is None before start of Limb processing in init")
        if concept_network_dict is None:
            raise Exception("concept_network_dict is None before start of Limb processing in init")
        
        #make sure it is a list
        if not nu.is_array_like(labels):
            labels=[labels]
            
        self.labels=labels
        
        #All the stuff dealing with the concept graph
        self.current_starting_coordinate=None
        self.concept_network = None
        self.all_concept_network_data = None
        if len(concept_network_dict) > 0:
            concept_network_data = nru.get_starting_info_from_concept_network(concept_network_dict)
            
            current_concept_network = concept_network_data[0]
            
            self.current_starting_coordinate = current_concept_network["starting_coordinate"]
            self.current_starting_node = current_concept_network["starting_node"]
            self.current_starting_endpoints = current_concept_network["starting_endpoints"]
            self.current_starting_soma = current_concept_network["starting_soma"]
            self.concept_network = concept_network_dict[self.current_starting_soma]
            self.all_concept_network_data = concept_network_data
        
        #get all of the starting coordinates an
        self.mesh_face_idx = mesh_face_idx
        self.mesh_center = tu.mesh_center_vertex_average(self.mesh)
        
        #just adding these in case could be useful in the future (what we computed for somas)
        #self.volume_ratio = sm.soma_volume_ratio(self.mesh)
        #self.side_length_ratios = sm.side_length_ratios(self.mesh)
        
        #Start with the branch stuff
        """
        a. Build all the branches from the 
        - mesh
        - skeleton
        - width
        - branch_face_idx
        b. Pick the top concept graph (will use to store the nodes)
        c. Put the branches as "data" in the network
        """
        
        for j,branch_data in curr_limb_correspondence.items():
            curr_skeleton = branch_data["branch_skeleton"]
            curr_width = branch_data["width_from_skeleton"]
            curr_mesh = branch_data["branch_mesh"]
            curr_face_idx = branch_data["branch_face_idx"]
            
            branch_obj = Branch(
                                skeleton=curr_skeleton,
                                width=curr_width,
                                mesh=curr_mesh,
                               mesh_face_idx=curr_face_idx,
                                labels=[],
            )
            
            #Set all  of the branches as data in the nodes
            xu.set_node_data(self.concept_network,
                            node_name=j,
                            curr_data=branch_obj,
                             curr_data_label="data"
                            )
            
        self.concept_network_directional = self.convert_concept_network_to_directional(no_cycles = True)
    
    
    def __eq__(self,other):
        """
        Purpose: Computes equality of all members except the face_idx
        
        Things we want to compare: 
        'all_concept_network_data', #array of dictionary: inside dictionary
        
        How to compare this array of dictionaries (because may not be in order)
        Pseudocode
        0) check that arrays are the same size (if not register as a difference)
        1) make an array of all the indexes in the self and than other arrays
        2) Start with the first index in self array:
        a. Iterate with those left in the other array to see if can match a dictionary 
        b. Once find a match, eliminate those indices from the lists and add as a pairings
        c. Go to next one in list
        d. If can't find pairing, add to differnces list and keep going
        
        3) At end if no differences then make sure self and others indicies list is empty

        
         'concept_network', #networkx graph
         'concept_network_directional', #networkx graph
         'current_starting_coordinate', #np.array
         'current_starting_endpoints', #np.array (for endpoints)
         'current_starting_node', #int
         
         'current_starting_soma',#int
         'label', #list
         'mesh', #mesh (compare_meshes_by_face_midpoints)
         'mesh_center', #1D array (compare threshold)
         'mesh_face_idx' #set comparison 
         
         
         Example: How tested the comparison
         tu = reload(tu)
        neuron= reload(neuron)
        xu = reload(xu)
        example_limb = double_soma_obj.concept_network.nodes["L1"]["data"]
        example_limb.labels = example_limb.label
        [k for k in dir(example_limb) if "__" not in k]
        L1 = neuron.Limb(example_limb)

        L2 = neuron.Limb(L1)

        #----testing the all_concept_network_data
        # L1.all_concept_network_data = [L1.all_concept_network_data[1],L1.all_concept_network_data[0]]
        # L2.all_concept_network_data[0]["starting_soma"] = 10

        #---testing the concept network comparison
        #L1.concept_network.nodes[1]["data"].skeleton[0][0][0] = 1
        #L2.concept_network.remove_node(1)
        #L2.concept_network.nodes[1]["data"].mesh = L2.concept_network.nodes[1]["data"].mesh.submesh([np.arange(len(L2.concept_network.nodes[1]["data"].mesh.faces)-1)],append=True)

        #---testing concept_network_directional
        #L1.concept_network_directional.nodes[1]["data"].skeleton[0][0][0] = 1
        #L2.concept_network_directional.remove_node(1)
        #L2.concept_network_directional.nodes[1]["data"].mesh = L2.concept_network.nodes[1]["data"].mesh.submesh([np.arange(len(L2.concept_network.nodes[1]["data"].mesh.faces)-1)],append=True)

        #----testing current_starting_endpoints
        #L2.current_starting_endpoints = np.array([[1,2,3],[4,5,6]])

        #---- testing current_starting_soma
        #L1.current_starting_soma=10

        #---- testing current_starting_soma
        #L2.labels=["new_labels"]

        # --- mesh_face_idx

        #L1.mesh_face_idx= np.array([])

        #----testing mesh_center
        #L2.mesh_center = np.array([1,2,3])
        
        """
        
        print_flag = True
        differences = []
        
        #----------comparing the network dictionaries-------------
        def __compare_concept_network_dicts(dict1,dict2):
            endpoints_compare = xu.compare_endpoints(dict1["starting_endpoints"],dict2["starting_endpoints"])
            starting_node_compare = dict1["starting_node"] == dict2["starting_node"]
            starting_soma_compare = dict1["starting_soma"] == dict2["starting_soma"]
            starting_coordinate_compare = nu.compare_threshold(dict1["starting_coordinate"],dict2["starting_coordinate"])
            
            if endpoints_compare and starting_node_compare and starting_soma_compare and starting_coordinate_compare:
                return True
            else:
                return False
        
        if len(self.all_concept_network_data) != len(other.all_concept_network_data):
            differences.append(f"lengths of all_concept_network_data did not match")
        else:
            self_indices = np.arange(len(self.all_concept_network_data) )
            other_indices = np.arange(len(self.all_concept_network_data) )
            
            pairings = []
            for i in self_indices:
                found_match = False
                for j in other_indices:
                    if __compare_concept_network_dicts(self.all_concept_network_data[i],
                                                      other.all_concept_network_data[j]):
                        #if match was found then remove the matching indices from other indices and break
                        other_indices = other_indices[other_indices != j]
                        pairings.append([i,j])
                        found_match=True
                        break
                
                if not found_match:    
                    #if no match was found then add to the differences list
                    differences.append(f"No match found for self.all_concept_network_data[{i}]"
                                      f"\nDictionary = {self.all_concept_network_data[i]}")
            
            #should have pairings for all all indices
            #print(f"pairings = {pairings}")

        #----------END OF comparing the network dictionaries-------------
        
        
        # Compare'concept_network', #networkx graph
        nx_compare_result,nx_diff_list = xu.compare_networks(self.concept_network,
                                                          other.concept_network,return_differences=True)
        if not nx_compare_result:
            differences.append(f"concept_network didn't match"
                              f"\n    Differences in compare_networks = {nx_diff_list}")
            
            
        #Comparing concept_network_directional
        nx_compare_result,nx_diff_list = xu.compare_networks(self.concept_network_directional,
                                                          other.concept_network_directional,return_differences=True)
        if not nx_compare_result:
            differences.append(f"concept_network_directional didn't match"
                              f"\n    Differences compare_networks = {nx_diff_list}")
            
        #Compare current_starting_coordinate 
        if not nu.compare_threshold(self.current_starting_coordinate,other.current_starting_coordinate):
            differences.append(f"current_starting_coordinate didn't match: "
                               f"\n    self.current_starting_coordinate = {self.current_starting_coordinate},"
                               f" other.current_starting_coordinate = {other.current_starting_coordinate}")
            
        #comparing the endpoints
        if not xu.compare_endpoints(self.current_starting_endpoints,other.current_starting_endpoints):
            differences.append(f"endpoints didn't match: "
                               f"\n    self.endpoints = {self.current_starting_endpoints}, other.endpoints = {other.current_starting_endpoints}") 
        
        #comparing current_starting_node
        if self.current_starting_node != other.current_starting_node:
            differences.append(f"current_starting_node didn't match: "
                               f"\n    self.current_starting_node = {self.current_starting_node},"
                               f" other.current_starting_node = {other.current_starting_node}") 
            
        #comparing the current_starting_soma
        if self.current_starting_soma != other.current_starting_soma:
            differences.append(f"current_starting_soma didn't match: "
                               f"\n    self.current_starting_soma = {self.current_starting_soma},"
                               f" other.current_starting_soma = {other.current_starting_soma}") 
        
        #comparing the labels:
        if set(self.labels) != set(other.labels):
            differences.append(f"labels didn't match: "
                               f"\n    self.labels = {self.labels}, other.labels = {other.labels}")
        
        #comparing the meshes
        if not tu.compare_meshes_by_face_midpoints(self.mesh,other.mesh):
            differences.append(f"mesh didn't match"
                               f"\n    self.mesh = {self.mesh},"
                               f" other.mesh = {other.mesh}") 
        
    
        #comparing the mesh centers    
        if not nu.compare_threshold(self.mesh_center,other.mesh_center):
            differences.append(f"mesh_center didn't match: "
                               f"\n    self.mesh_center = {self.mesh_center}, other.mesh_center = {other.mesh_center}")

        
        #print out if face idx was different but not make part of the comparison
        if set(self.mesh_face_idx) != set(other.mesh_face_idx):
#             print("*** Warning: mesh_face_idx didn't match (but not factored into equality comparison)\n"
#                  f"set(self.mesh_face_idx) = {self.mesh_face_idx}, set(other.mesh_face_idx) = {other.mesh_face_idx}")
            pass
        
        
        if len(differences) == 0:
            return True
        else:
            if print_flag:
                print("Differences List:")
                for j,diff in enumerate(differences):
                    print(f"{j})   {diff}")
            return False
        
    
    def __ne__(self,other):
        return not self.__eq__(other)
        


class Soma:
    """
    Class that will hold one continus skeleton
    piece that has no branching
    
    Properties that are housed:
     'mesh',
     'mesh_center',
     'mesh_face_idx',
     'sdf',
     'side_length_ratios',
     'volume_ratio'
    
    """
    
    def __init__(self,mesh,mesh_face_idx=None,sdf=None):
        #Accounting for the fact that could recieve soma object
        if str(type(mesh)) == str(Soma):
            #print("Recived Soma object so copying object")
            # properties we are copying: [k for k in dir(example_limb) if "__" not in k]
            
            self.mesh = dc(mesh.mesh)
            self.sdf=dc(mesh.sdf)
            self.mesh_face_idx = dc(mesh.mesh_face_idx)
            self.volume_ratio = dc(mesh.volume_ratio)
            self.side_length_ratios = dc(mesh.side_length_ratios)
            self.mesh_center = dc(mesh.mesh_center)
            
            return 
        
        #print("bypassing soma object initialization")
        self.mesh=mesh
        self.sdf=sdf
        self.mesh_face_idx = mesh_face_idx
        self.volume_ratio = sm.soma_volume_ratio(self.mesh)
        self.side_length_ratios = sm.side_length_ratios(self.mesh)
        self.mesh_center = tu.mesh_center_vertex_average(self.mesh)
        
    def __eq__(self,other):
        #print("inside equality function")
        """
        Purpose: Computes equality of all members except the face_idx (will print out if not equal)
        
        Properties that need to be checked:
        'mesh',
         'mesh_center',
         'mesh_face_idx',
         'sdf',
         'side_length_ratios',
         'volume_ratio'
        
        Example of How tested it: 
        
        tu = reload(tu)
        neuron= reload(neuron)
        xu = reload(xu)
        example_soma = double_soma_obj.concept_network.nodes["S0"]["data"]
        S1 = neuron.Soma(example_soma)
        S2 = neuron.Soma(example_soma)

        #----testing mesh_center
        S2.mesh_center = np.array([1,2,3])

        #---- testing side_length_ratios
        S1.side_length_ratios=[10,19,20]

        #---- testing current_starting_soma
        S2.volume_ratio = 14

        # --- mesh_face_idx
        S2.mesh_face_idx= np.array([])

        #----testing mesh_center
        S2.sdf = 14

        """
        print_flag = True
        differences = []
        
        #comparing the meshes
        if not tu.compare_meshes_by_face_midpoints(self.mesh,other.mesh):
            differences.append(f"mesh didn't match"
                               f"\n    self.mesh = {self.mesh},"
                               f" other.mesh = {other.mesh}") 
        
            
        if not nu.compare_threshold(self.mesh_center,other.mesh_center):
            differences.append(f"mesh_center didn't match: "
                               f"\n    self.mesh_center = {self.mesh_center}, other.mesh_center = {other.mesh_center}")
            
        #print out if face idx was different but not make part of the comparison
        if set(self.mesh_face_idx) != set(other.mesh_face_idx):
#             print("*** Warning: mesh_face_idx didn't match (but not factored into equality comparison)")
            pass
            
        if not nu.compare_threshold(self.sdf,other.sdf):
            differences.append(f"sdf didn't match: "
                               f"\n    self.sdf = {self.sdf}, other.sdf = {other.sdf}")
            
        if not nu.compare_threshold(self.side_length_ratios,other.side_length_ratios):
            differences.append(f"side_length_ratios didn't match: "
                               f"\n    self.side_length_ratios = {self.side_length_ratios}, other.side_length_ratios = {other.side_length_ratios}")
            
        if not nu.compare_threshold(self.volume_ratio,other.volume_ratio):
            differences.append(f"volume_ratio didn't match: "
                               f"\n    self.volume_ratio = {self.volume_ratio}, other.volume_ratio = {other.volume_ratio}")
        
    
        
        if len(differences) == 0:
            return True
        else:
            if print_flag:
                print("Differences List:")
                for j,diff in enumerate(differences):
                    print(f"{j})   {diff}")
            return False
        
    
    def __ne__(self,other):
        #print(f"self.__eq__(other) = {self.__eq__(other)}")
        return not self.__eq__(other)
    
                               
                               
# def preprocess_neuron(current_neuron,segment_id=None,
#                      description=None):
#     if segment_id is None:
#         #pick a random segment id
#         segment_id = np.random.randint(100000000)
#         print(f"picking a random 7 digit segment id: {segment_id}")
#     if description is None:
#         description = "no_description"
    
#     raise Exception("prprocessing pipeline not finished yet")
    

    

from neuron_utils import preprocess_neuron

class Neuron:
    """
    Neuron class docstring: 
    Will 
    
    Purpose: 
    An object oriented approach to housing the data
    about a single neuron mesh and the secondary 
    data that can be gleamed from this. For instance
    - skeleton
    - compartment labels
    - soma centers
    - subdivided mesh into cable pieces
    
    
    Pseudocode: 
    
    1) Create Neuron Object (through __init__)
    a. Add the small non_soma_list_meshes
    b. Add whole mesh
    c. Add soma_to_piece_connectivity as concept graph and it will be turned into a concept map

    2) Creat the soma meshes
    a. Create soma mesh objects
    b. Add the soma objects as ["data"] attribute of all of the soma nodes

    3) Limb Process: For each limb (use an index to iterate through limb_correspondence,current_mesh_data and limb_concept_network/lables) 
    a. Build all the branches from the 
        - mesh
        - skeleton
        - width
        - branch_face_idx
    b. Pick the top concept graph (will use to store the nodes)
    c. Put the branches as "data" in the network
    d. Get all of the starting coordinates and starting edges and put as member attributes in the limb

    
    """
    def get_total_n_branches(self):
        return np.sum([len(self.concept_network.nodes[li]["data"].concept_network.nodes()) for li in self.get_limb_node_names()])
    
    def get_skeleton(self,check_connected_component=True):
        return nru.get_whole_neuron_skeleton(self,
                                 check_connected_component=check_connected_component)
    
    @property
    def skeleton(self,check_connected_component=False):
        return nru.get_whole_neuron_skeleton(self,
                                 check_connected_component=check_connected_component)
    
    def __init__(self,mesh,
                 segment_id=None,
                 description=None,
                 preprocessed_data=None,
                suppress_preprocessing_print=True,
                ignore_warnings=True,
                minimal_output=False):
#                  concept_network=None,
#                  non_graph_meshes=dict(),
#                  pre_processed_mesh = dict()
#                 ):
        """here would be calling any super classes inits
        Ex: Parent.__init(self)
        
        Class can act like a dictionary and can d
        """
    
        #covering the scenario where the data was recieved was actually another neuron class
        #print(f"type of mesh = {mesh.__class__}")
        #print(f"type of self = {self.__class__}")
        
        neuron_creation_time = time.time()
        
        if minimal_output:
            print("Processing Neuorn in minimal output mode...please wait")
        
        
        with su.suppress_stdout_stderr() if minimal_output else su.dummy_context_mgr():

            if str(mesh.__class__) == str(self.__class__):
                print("Recieved another instance of Neuron class in init -- so just copying data")
                self.segment_id=dc(mesh.segment_id)
                self.description = dc(mesh.description)
                self.preprocessed_data = dc(mesh.preprocessed_data)
                self.mesh = dc(mesh.mesh)
                self.concept_network = copy_concept_network(mesh.concept_network)
                
                #mesh pieces
                self.inside_pieces = dc(mesh.inside_pieces)
                self.insignificant_limbs = dc(mesh.insignificant_limbs)
                self.non_soma_touching_meshes = dc(mesh.non_soma_touching_meshes)
                
                return 
                
                

            if ignore_warnings: 
                su.ignore_warnings()

            self.mesh = mesh

            if description is None:
                description = ""
            if segment_id is None:
                #pick a random segment id
                segment_id = np.random.randint(100000000)
                print(f"picking a random 7 digit segment id: {segment_id}")
                description += "_random_id"
            


            self.segment_id = segment_id
            self.description = description


            neuron_start_time =time.time()
            if preprocessed_data is None: 
                print("--- 0) Having to preprocess the Neuron becuase no preprocessed data\nPlease wait this could take a while.....")
                if suppress_preprocessing_print:
                    with su.suppress_stdout_stderr():
                        preprocessed_data = nru.preprocess_neuron(mesh,
                                         segment_id=segment_id,
                                         description=description)
                        print(f"preprocessed_data inside with = {preprocessed_data}")
                else:
                    preprocessed_data = nru.preprocess_neuron(mesh,
                                         segment_id=segment_id,
                                         description=description)

                print(f"--- 0) Total time for preprocessing: {time.time() - neuron_start_time}\n\n\n\n")
                neuron_start_time = time.time()
            else:
                print("Already have preprocessed data")

            #print(f"preprocessed_data inside with = {preprocessed_data}")

            #this is for if ever you want to copy the neuron from one to another or save it off?
            self.preprocessed_data = preprocessed_data


            #self.non_graph_meshes = preprocessed_data["non_graph_meshes"]
            limb_concept_networks = preprocessed_data["limb_concept_networks"]
            limb_correspondence = preprocessed_data["limb_correspondence"]
            limb_meshes = preprocessed_data["limb_meshes"]
            limb_labels = preprocessed_data["limb_labels"]

            self.insignificant_limbs = preprocessed_data["insignificant_limbs"]
            self.non_soma_touching_meshes = preprocessed_data["non_soma_touching_meshes"]
            self.inside_pieces = preprocessed_data["inside_pieces"]

            soma_meshes = preprocessed_data["soma_meshes"]
            soma_to_piece_connectivity = preprocessed_data["soma_to_piece_connectivity"]
            soma_sdfs = preprocessed_data["soma_sdfs"]
            print(f"--- 1) Finished unpacking preprocessed materials: {time.time() - neuron_start_time}")
            neuron_start_time =time.time()

            # builds the networkx graph where we will store most of the data
            if type(soma_to_piece_connectivity) == type(nx.Graph()):
                self.concept_network = soma_to_piece_connectivity
            elif type(soma_to_piece_connectivity) == dict:
                concept_network = convert_soma_to_piece_connectivity_to_graph(soma_to_piece_connectivity)
                self.concept_network = concept_network
            else:
                raise Exception(f"Recieved an incompatible type of {type(soma_to_piece_connectivity)} for the concept_network")

            print(f"--- 2) Finished creating neuron connectivity graph: {time.time() - neuron_start_time}")
            neuron_start_time =time.time()

            """
            2) Creat the soma meshes
            a. Create soma mesh objects
            b. Add the soma objects as ["data"] attribute of all of the soma nodes
            """

            if "soma_meshes_face_idx" in list(preprocessed_data.keys()):
                soma_meshes_face_idx = preprocessed_data["soma_meshes_face_idx"]
                print("Using already existing soma_meshes_face_idx in preprocessed data ")
            else:
                print("Having to generate soma_meshes_face_idx because none in preprocessed data")
                soma_meshes_face_idx = []
                for curr_soma in soma_meshes:
                    curr_soma_meshes_face_idx = tu.original_mesh_faces_map(mesh, curr_soma,
                           matching=True,
                           print_flag=False)
                    soma_meshes_face_idx.append(curr_soma_meshes_face_idx)

                print(f"--- 3a) Finshed generating soma_meshes_face_idx: {time.time() - neuron_start_time}")
                neuron_start_time =time.time()

            for j,(curr_soma,curr_soma_face_idx,current_sdf) in enumerate(zip(soma_meshes,soma_meshes_face_idx,soma_sdfs)):
                Soma_obj = Soma(curr_soma,mesh_face_idx=curr_soma_face_idx,sdf=current_sdf)
                soma_name = f"S{j}"
                #Add the soma object as data in 
                xu.set_node_data(curr_network=self.concept_network,
                                     node_name=soma_name,
                                     curr_data=Soma_obj,
                                     curr_data_label="data")
            print(f"--- 3) Finshed generating soma objects and adding them to concept graph: {time.time() - neuron_start_time}")
            neuron_start_time =time.time()


            """
            3) Add the limbs to the graph:
            a. Create the limb objects and their associated names
            (use an index to iterate through limb_correspondence,current_mesh_data and limb_concept_network/lables) 
            b. Add the limbs to the neuron concept graph nodes

            """

            if "limb_mehses_face_idx" in list(preprocessed_data.keys()):
                limb_mehses_face_idx = preprocessed_data["limb_mehses_face_idx"]
                print("Using already existing limb_mehses_face_idx in preprocessed data ")
            else:
                limb_mehses_face_idx = []
                for curr_limb in limb_meshes:
                    curr_limb_meshes_face_idx = tu.original_mesh_faces_map(mesh, curr_limb,
                           matching=True,
                           print_flag=False)
                    limb_mehses_face_idx.append(curr_limb_meshes_face_idx)

                print(f"--- 4a) Finshed generating curr_limb_meshes_face_idx: {time.time() - neuron_start_time}")
                neuron_start_time =time.time()

    #         print("Returning so can debug")
    #         return

            for j,(curr_limb_mesh,curr_limb_mesh_face_idx) in enumerate(zip(limb_meshes,limb_mehses_face_idx)):
                """
                will just find the curr_limb_concept_network, curr_limb_label by indexing
                """
                curr_limb_correspondence = limb_correspondence[j]
                curr_limb_concept_networks = limb_concept_networks[j]
                curr_limb_label = limb_labels[j]




                Limb_obj = Limb(
                                 mesh=curr_limb_mesh,
                                 curr_limb_correspondence=curr_limb_correspondence,
                                 concept_network_dict=curr_limb_concept_networks,
                                 mesh_face_idx=curr_limb_mesh_face_idx,
                                 labels=curr_limb_label
                                )


                limb_name = f"L{j}"
                #Add the soma object as data in
                xu.set_node_data(curr_network=self.concept_network,
                                     node_name=limb_name,
                                     curr_data=Limb_obj,
                                     curr_data_label="data")

                xu.set_node_data(self.concept_network,node_name=soma_name,curr_data=Soma_obj,curr_data_label="data")

            print(f"--- 4) Finshed generating Limb objects and adding them to concept graph: {time.time() - neuron_start_time}")
            neuron_start_time =time.time()

        print(f"Total time for neuron instance creation = {time.time() - neuron_creation_time}")
        
    
    
    #Overloading the Comparison 
    def __eq__(self,other):
        """
        Purpose: Computes equality of all members of the neuron object
        
        Things that need to compare: 
        concept_network
        segment_id
        description
        mesh
        inside_pieces #we should do these in the same order
        insignificant_limbs #same order
        non_soma_touching_meshes #same order
        
        Testing: 
        import numpy as np
        nru = reload(nru)
        neuron = reload(neuron)
        xu = reload(xu)
        sk = reload(sk)
        nu= reload(nu)
        tu = reload(tu)

        import soma_extraction_utils as sm
        sm = reload(sm)

        obj1 = neuron.Neuron(double_soma_obj,minimal_output=False)
        obj2 = neuron.Neuron(double_soma_obj,minimal_output=False)

        #obj1 == obj2

        #testing the changing of different things
        #obj1.concept_network.nodes["S0"]["data"].mesh = obj1.concept_network.nodes["S0"]["data"].mesh.submesh([np.arange(0,len(obj1.concept_network.nodes["S0"]["data"].mesh.faces)-5)],append=True)
        #obj1.concept_network.nodes["S0"]["data"].sdf = 100
        #obj1.description = "hello"
        #obj1.segment_id = 1234567
        #obj1.inside_pieces = obj1.inside_pieces[:10]
        #obj2.non_soma_touching_meshes =  obj2.non_soma_touching_meshes[6:17]
        #curr_mesh = obj1.concept_network.nodes["L1"]["data"].concept_network.nodes[1]["data"].mesh
        #obj1.concept_network.nodes["L1"]["data"].concept_network.nodes[1]["data"].mesh = curr_mesh.submesh([np.arange(5,50)],append=True)
        
        au = reload(au)
        obj1 == obj2
        
        """
        
        print_flag = True
        differences = []
      
        
        # Compare'concept_network', #networkx graph
        nx_compare_result,nx_diff_list = xu.compare_networks(self.concept_network,
                                                          other.concept_network,return_differences=True)
        if not nx_compare_result:
            differences.append(f"concept_network didn't match"
                              f"\n    Differences in compare_networks = {nx_diff_list}")
        
        #comparing the segment_id
        if self.segment_id != other.segment_id:
            differences.append(f"segment_id didn't match: "
                               f"\n    self.segment_id = {self.segment_id},"
                               f" other.segment_id = {other.segment_id}") 
        
        #comparing the description
        if self.description != other.description:
            differences.append(f"description didn't match: "
                               f"\n    self.description = {self.description},"
                               f" other.description = {other.description}") 
        
        
        #comparing the meshes
        if not tu.compare_meshes_by_face_midpoints(self.mesh,other.mesh):
            differences.append(f"mesh didn't match"
                               f"\n    self.mesh = {self.mesh},"
                               f" other.mesh = {other.mesh}") 
        
        #comparing the mesh lists
        mesh_lists_to_check = ["inside_pieces",
                                "insignificant_limbs",
                                "non_soma_touching_meshes"]
        
        
        for curr_mesh_attr in mesh_lists_to_check:
            curr_self_attr = getattr(self, curr_mesh_attr)
            curr_other_attr = getattr(other, curr_mesh_attr)
            
            """
            Older method of comparison that did not account for lists of different sizes: 
            mesh_list_comparisons = tu.compare_meshes_by_face_midpoints_list(curr_self_attr,
                                                                             curr_other_attr)
            
            """
            comparison_result, comparison_differences = au.compare_uneven_groups(curr_self_attr,curr_other_attr,
                             comparison_func = tu.compare_meshes_by_face_midpoints,
                             group_name=curr_mesh_attr,
                             return_differences=True)
        
            
            
            if not comparison_result:
                """
                #didnt account for different size comparisons
                differences.append(f"{curr_mesh_attr} didn't match"
                                   f"\n    self.{curr_mesh_attr} different = {[k for truth,k in zip(mesh_list_comparisons,curr_self_attr) if truth == False]},"
                                   f" other.{curr_mesh_attr} different = {[k for truth,k in zip(mesh_list_comparisons,curr_other_attr) if truth == False]}") 
                """
                differences += comparison_differences
        
        if len(differences) == 0:
            return True
        else:
            if print_flag:
                print("Differences List:")
                for j,diff in enumerate(differences):
                    print(f"{j})   {diff}")
            return False
        
    
    def __ne__(self,other):
        return not self.__eq__(other)
    
    
    
    """
    What visualizations to neuron do: 
    1) Show the soma/limb concept network with colors (or any subset of that)

    * Be able to pick a 
    2) Show the entire skeleton
    3) show the entire mesh


    Ideal: 
    1) get a submesh: By
    - names 
    - properties
    - or both
    2) Be able to describe what feature want to see with them:
    - skeleton
    - mesh: 
        branch or limb color specific
    - concept network 
        directed or undirected
        branch or limb color specific

    3) Have some feature of the whole mesh in the background


    Want to specify certian colors of specific groups

    Want to give back the colors with the names of the things if did random

    """
    def get_limb_node_names(self):
        return [k for k in self.concept_network.nodes() if "L" in k]
    def get_branch_node_names(self,limb_idx):
        limb_idx = nru.limb_label(limb_idx)
        curr_limb_obj = self.concept_network.nodes[limb_idx]["data"]
        return list(curr_limb_obj.concept_network.nodes())
    def get_soma_node_names(self):
        return [k for k in self.concept_network.nodes() if "S" in k]
    
    #how to save neuron object
    def save_neuron_object(self,
                          filename=""):
        if filename == "":
            print("No filename/location given so creating own")
            filename = f"{self.segment_id}_{self.description}.pkl"
        file = Path(filename)
        print(f"Saving Object at: {file.absolute()}")
        
        su.save_object(self,file)
    
    
    def calculate_width_without_spines(self,
                                      skeleton_segment_size = 1000,
                                       width_segment_size=None,
                                      width_name = "no_spine_average",
                                      **kwargs):


        for limb_idx in self.get_limb_node_names():
            for branch_idx in self.get_branch_node_names(limb_idx):
                print(f"Working on limb {limb_idx} branch {branch_idx}")
                curr_branch_obj = self.concept_network.nodes[nru.limb_label(limb_idx)]["data"].concept_network.nodes[branch_idx]["data"]
                if "distance_by_mesh_center" not in kwargs.keys():
                    if "mesh_center" in width_name:
                        distance_by_mesh_center = True
                    else:
                        distance_by_mesh_center = False
                
                
                current_width_array,current_width = wu.calculate_width_without_spines(curr_branch_obj, 
                                      skeleton_segment_size=skeleton_segment_size,
                                      width_segment_size=width_segment_size, 
                                      distance_by_mesh_center=distance_by_mesh_center,
                                      return_average=True,
                                      print_flag=False,
                                    **kwargs)


                curr_branch_obj.width_new[width_name] = current_width
                curr_branch_obj.width_array[width_name] = current_width_array
    
        
    def calculate_spines(self,
                        query="width > 400 and n_faces_branch>100",
                        clusters_threshold=2,
                        smoothness_threshold=0.08,
                        shaft_threshold=300,
                        cgal_path=Path("./cgal_temp")):
        
        print(f"smoothness_threshold = {smoothness_threshold}")
        if type(query) == dict():
            functions_list = query["functions_list"]
            current_query = query["query"]
        else:
            functions_list = ["width","n_faces_branch"]
            current_query = query

        new_branch_dict = ns.query_neuron(self,
                       functions_list=functions_list,
                       query=current_query)
        
        
        for limb_idx in self.get_limb_node_names():
            for branch_idx in self.get_branch_node_names(limb_idx):
                curr_branch = self.concept_network.nodes[nru.limb_label(limb_idx)]["data"].concept_network.nodes[branch_idx]["data"]
                if limb_idx in new_branch_dict.keys():
                    if branch_idx in new_branch_dict[limb_idx]:
                
                        print(f"Working on limb {limb_idx} branch {branch_idx}")
                        #calculate the spines
                        spine_submesh_split= spu.get_spine_meshes_unfiltered(current_neuron = self,
                                                                limb_idx=limb_idx,
                                                                branch_idx=branch_idx,
                                                                clusters=clusters_threshold,
                                                                smoothness=smoothness_threshold,
                                                                cgal_folder = cgal_path,
                                                                delete_temp_file=True,
                                                                return_sdf=False,
                                                                print_flag=False,
                                                                shaft_threshold=shaft_threshold)

        #                 if limb_idx == "L0":
        #                     if branch_idx == 0:
        #                         print(f"spine_submesh_split = {spine_submesh_split}")

                        spine_submesh_split_filtered = spu.filter_spine_meshes(spine_submesh_split,
                                                                              spine_n_face_threshold=20)
        #                 if limb_idx == "L0":
        #                     if branch_idx == 0:
        #                         print(f"spine_submesh_split_filtered = {spine_submesh_split_filtered}")


                        curr_branch.spines = spine_submesh_split_filtered
                else:
                    curr_branch.spines = None

    def plot_soma_limb_concept_network(self,
                                      soma_color="red",
                                      limb_color="blue",
                                      node_size=500,
                                      font_color="white",
                                      node_colors=dict()):
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
        
        node_list = xu.get_node_list(self.concept_network)
        node_list_colors = []
        for n in node_list:
            if n in list(node_colors.keys()):
                curr_color = node_colors[n]
            else:
                if "S" in n:
                    curr_color = soma_color
                else:
                    curr_color = limb_color
            node_list_colors.append(curr_color)
        #node_list_colors = [soma_color if "S" in n else limb_color for n in node_list]
        nx.draw(self.concept_network,with_labels=True,node_color=node_list_colors,
               font_color=font_color,node_size=node_size)
        
    def plot_limb_concept_network(self,
                                  limb_name="",
                                 limb_idx=-1,
                                  node_size=0.3,
                                  directional=True,
                                 append_figure=False,
                                 show_at_end=True,**kwargs):
        if limb_name == "":
            if limb_idx == -1:
                raise Exception("Limb name and limb_idx both not specified")
            limb_name = f"L{limb_idx}"
            
        if directional:
            curr_limb_concept_network_directional = self.concept_network.nodes[limb_name]["data"].concept_network_directional
        else:
            curr_limb_concept_network_directional = self.concept_network.nodes[limb_name]["data"].concept_network
        nviz.plot_concept_network(curr_concept_network = curr_limb_concept_network_directional,
                            scatter_size=node_size,
                            show_at_end=show_at_end,
                            append_figure=append_figure,**kwargs)
        

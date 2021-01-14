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
from pykdtree.kdtree import KDTree #to be used for the soma_vertex nullification

import width_utils as wu
import meshlab

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
            self.spines_volume = dc_check(skeleton,"spines_volume")
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
        self.calculate_endpoints()
        self.mesh_center = None
        if not self.mesh is None:
            self.mesh_center = tu.mesh_center_vertex_average(self.mesh)
        self.labels=labels
        if not nu.is_array_like(self.labels):
            self.labels=[self.labels]
            
        
        self.spines = None
        self.spines_volume = None
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
    
    def calculate_endpoints(self):
        self.endpoints = sk.find_branch_endpoints(self.skeleton)
    
    @property
    def skeletal_length(self):
        return sk.calculate_skeleton_distance(self.skeleton)
    
    @property
    def n_spines(self):
        return nru.n_spines(self)
    
    @property
    def spine_density(self):
        return nru.spine_density(self)
    
    @property
    def total_spine_volume(self):
        return nru.total_spine_volume(self)
    
    @property
    def spine_volume_median(self):
        return nru.spine_volume_median(self)
    
    @property
    def spine_volume_density(self):
        return nru.spine_volume_density(self)
    
    @property
    def skeletal_length_eligible(self):
        return sk.calculate_skeleton_distance(self.skeleton) 
    
    def compute_spines_volume(self,
              max_hole_size=2000,
              self_itersect_faces=False):
        if self.spines is None:
            self.spines_volume = None
        else:
            self.spines_volume = [tu.mesh_volume(sp,verbose=False) for sp in self.spines]
    
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
    @property
    def branch_objects(self):
        """
        dictionary mapping the index to the 
        """
        return dict([(i,k) for i,k in enumerate(self)])
    
    
    @property
    def network_starting_info(self):
        """
        Purpose: will generate the dictionary that is organized
        soma_idx --> soma_group_idx --> dict(touching_verts,endpoint)
        
        that can be used to generate a concept network from 
        
        """
        st_dict = dict()
        for st in self.all_concept_network_data:
            soma_idx = st["starting_soma"] 
            soma_group_idx = st["soma_group_idx"] 
            if st["starting_soma"] not in st_dict.keys():
                st_dict[soma_idx] = dict()

            st_dict[soma_idx][soma_group_idx] = dict(touching_verts=st["touching_soma_vertices"],
                                                    endpoint=st["starting_coordinate"])
        return st_dict
    
    @property
    def limb_correspondence(self):
        self._index = -1
        limb_corr = dict()
        for idx,b in enumerate(self):
            curr_width = b.width    
            try:
                curr_width = b.width_new["median_mesh_center"]
            except:
                pass

            limb_corr[idx] = dict(branch_skeleton=b.skeleton,
                                  width_from_skeleton = curr_width,
                                 branch_mesh = b.mesh,
                                 branch_face_idx = b.mesh_face_idx,
                                 )
        
            
        return limb_corr
    
    @property
    def divided_skeletons(self):
        curr_corr = self.limb_correspondence
        return np.array([curr_corr[k]["branch_skeleton"] for k in np.sort(list(curr_corr.keys()))])
        
    @property
    def spines(self):
        self._index = -1
        total_spines = []
        for b in self:
            if not b.spines is None:
                total_spines += b.spines
        return total_spines
    
    @property
    def spines_volume(self):
        self._index = -1
        total_spines_volume = []
        for b in self:
            if not b.spines_volume is None:
                total_spines_volume += b.spines_volume
        return total_spines_volume
    
    def compute_spines_volume(self):
        self._index = -1
        for b in self:
            b.compute_spines_volume()
    
    def get_branch_names(self,ordered=True,return_int=True):
        node_names = np.sort(list(self.concept_network.nodes()))
        if return_int:
            return node_names.astype("int")
        else:
            return node_names
    
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
    
    def get_concept_network_data_by_soma(self,soma_idx=None):
        #compile a dictionary of all of the starting material
        return_dict = dict()
        for curr_data in self.all_concept_network_data:
            return_dict[curr_data["starting_soma"]] = dict([(k,v) for k,v in curr_data.items() if k != "starting_soma"])
        if not soma_idx is None:
            return return_dict[soma_idx]
        else:
            return return_dict
    
    
    
    def get_concept_network_data_by_soma_and_idx(self,soma_idx,soma_group_idx):
        return_dict = []
        for curr_data in self.all_concept_network_data:
            if curr_data["soma_group_idx"] == soma_group_idx and curr_data["starting_soma"] == soma_idx:
                return_dict.append(curr_data)
        
        if len(return_dict) != 1:
            raise Exception(f"Did not find exactly one starting dictionary for soma_idx {soma_idx}, soma_group_idx {soma_group_idx}: {len(return_dict)} ")
        else:
            return return_dict[0]
         
    
    @property
    def concept_network_data_by_soma(self):
        #compile a dictionary of all of the starting material
        return_dict = dict()
        for curr_data in self.all_concept_network_data:
            return_dict[curr_data["starting_soma"]] = dict([(k,v) for k,v in curr_data.items() if k != "starting_soma"])
        return return_dict
    
    @property
    def concept_network_data_by_starting_node(self):
        #compile a dictionary of all of the starting material
        return_dict = dict()
        for curr_data in self.all_concept_network_data:
            return_dict[curr_data["starting_node"]] = dict([(k,v) for k,v in curr_data.items() if k != "starting_node"])
        return return_dict
    
    def touching_somas(self):
        return [k["starting_soma"] for k in self.all_concept_network_data if k["starting_soma"] >= 0]
    
    
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
        if type(soma) == str:
            soma = int(soma[1:])
        
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
    
    def get_starting_branch_by_soma(self,soma,print_flag=False):
        """
        Purpose: from the all
        
        """
        if type(soma) == str:
            soma = int(soma[1:])
        
        return self.concept_network_data_by_soma[soma]["starting_node"]
    
    def get_soma_by_starting_node(self,starting_node,print_flag=False):
        """
        Purpose: from the all
        
        """
        
        return self.concept_network_data_by_starting_node[starting_node]["starting_soma"]
    
    def get_soma_group_by_starting_node(self,starting_node,print_flag=False):
        """
        Purpose: from the all
        
        """
        
        return self.concept_network_data_by_starting_node[starting_node]["soma_group_idx"]
    
    
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
    
    
    def convert_concept_network_to_directional(self,no_cycles = True,width_source=None,print_flag=False,
                                              suppress_disconnected_errors=False):
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
            suppress_disconnected_errors =suppress_disconnected_errors)
                                                                            
        
        return directional_concept_network
    
    
    def set_concept_network_directional(self,starting_soma=None,soma_group_idx=0,starting_node=None,print_flag=False,
                                       suppress_disconnected_errors=False,**kwargs):
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
        debug = False
        
        if not starting_node is None: 
            soma_group_idx = self.get_soma_group_by_starting_node(starting_node)
            starting_soma = self.get_soma_by_starting_node(starting_node)
            
        if soma_group_idx is None:
            soma_group_idx = 0
        
        if soma_group_idx == -1:
            soma_group_idx = self.current_soma_group_idx
        
        matching_concept_network_data = [k for k in self.all_concept_network_data if ((k["starting_soma"] == starting_soma) or (nru.soma_label(k["starting_soma"]) == starting_soma))]

        if len(matching_concept_network_data) < 1:
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
            del self.concept_network.nodes[prev_st_node]["touching_soma_vertices"]
            del self.concept_network.nodes[prev_st_node]["soma_group_idx"]
            del self.concept_network.nodes[prev_st_node]["starting_soma"]
        
        
        
        matching_concept_network_dict = matching_concept_network_data[soma_group_idx]
        #print(f"matching_concept_network_dict = {matching_concept_network_dict}")
        curr_starting_node = matching_concept_network_dict["starting_node"]
        curr_starting_coordinate= matching_concept_network_dict["starting_coordinate"]
        curr_touching_soma_vertices = matching_concept_network_dict["touching_soma_vertices"]
        curr_soma_group_idx = matching_concept_network_dict["soma_group_idx"]
        
        if debug:
            print("Applying the set_directional change!!!!")
            print(f"curr_touching_soma_vertices = {curr_touching_soma_vertices}")

        #set the starting coordinate in the concept network
        attrs = {curr_starting_node:{"starting_coordinate":curr_starting_coordinate,
                                    "touching_soma_vertices":curr_touching_soma_vertices,
                                    "soma_group_idx":curr_soma_group_idx,
                                    "starting_soma":starting_soma}
                }
        if print_flag:
            print(f"attrs = {attrs}")
        xu.set_node_attributes_dict(self.concept_network,attrs)
        
        if debug:
            print(f'self.concept_network.nodes[curr_starting_node] = {self.concept_network.nodes[curr_starting_node] }')

        #make sure only one starting coordinate
        new_starting_coordinate = xu.get_starting_node(self.concept_network)
        if print_flag:
            print(f"New starting coordinate at node {new_starting_coordinate}")
            
        
        self.current_starting_coordinate = matching_concept_network_dict["starting_coordinate"]
        self.current_starting_node = matching_concept_network_dict["starting_node"]
        self.current_starting_endpoints = matching_concept_network_dict["starting_endpoints"]
        self.current_starting_soma = matching_concept_network_dict["starting_soma"]
        self.current_touching_soma_vertices = matching_concept_network_dict["touching_soma_vertices"]
        self.current_soma_group_idx = matching_concept_network_dict["soma_group_idx"]
        
        """
        --- 1/4/2021 Change: Making so redoes the edges of the concept network when resetting the source
        
        
        """
        #Now need to reset the edges according to the new starting info
        self.set_concept_network_edges_from_current_starting_data()

        
        if print_flag:
            self.concept_network_directional = self.convert_concept_network_to_directional(no_cycles = True,print_flag=print_flag,
                                                                                           suppress_disconnected_errors=suppress_disconnected_errors,
                                                                                           **kwargs)
        else:
            with su.suppress_stdout_stderr():
                self.concept_network_directional = self.convert_concept_network_to_directional(no_cycles = True,print_flag=print_flag,
                                                                                               suppress_disconnected_errors=suppress_disconnected_errors,
                                                                                               **kwargs)
        
        
    def set_concept_network_edges_from_current_starting_data(self,verbose=False):
        new_concept_network = nru.branches_to_concept_network(curr_branch_skeletons= self.divided_skeletons,
                                                                  starting_coordinate=self.current_starting_coordinate,
                                                                  starting_edge=self.current_starting_endpoints,
                                                                  touching_soma_vertices=self.current_touching_soma_vertices,
                                                                       soma_group_idx=self.current_soma_group_idx,
                                                                       verbose=False)
        
        
        
        self.concept_network.remove_edges_from(list(self.concept_network.edges()))
        self.concept_network.add_edges_from(list(new_concept_network.edges()))
        self.concept_network.remove_edges_from(self.deleted_edges)
        self.concept_network.add_edges_from(self.created_edges)
        
        
    def __init__(self,
                             mesh,
                             curr_limb_correspondence=None,
                             concept_network_dict=None,
                             mesh_face_idx=None,
                            labels=[],
                             branch_objects = dict(),#this will have a dictionary mapping to the branch objects if provided
                             deleted_edges = [],
                             created_edges = [],
                            verbose=False):
        
        
        """
        Allow for an initialization of a limb with another limb oconcept_network_dictbject
        
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
            
            
            
            attributes_to_set = dict(current_touching_soma_vertices=None,
                                current_soma_group_idx = None,
                                deleted_edges = [],
                                created_edges = [])
            for attr,attr_v in attributes_to_set.items():
                if hasattr(mesh,attr):
                    setattr(self,attr,dc(getattr(mesh,attr)))
                else:
                    setattr(self,attr,attr_v)

            self.current_starting_soma = dc(mesh.current_starting_soma)
            self.labels = dc(mesh.labels)
            if not nu.is_array_like(self.labels):
                self.labels=[self.labels]
            self.mesh = dc(mesh.mesh)
            self.mesh_center = dc(mesh.mesh_center)
            self.mesh_face_idx = dc(mesh.mesh_face_idx)
            self._index = -1
            
            
            return 
        
        debug_edges = False
        
        if debug_edges:
            print(f"before set: deleted_edges={deleted_edges}")
            print(f"before set: created_edges={created_edges}")
        
        self._index = -1
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
        
        #print(f"Inside the Limb constructor and concept_network_dict = {concept_network_dict}")
        
        if verbose:
            print(f"concept_network_dict = {concept_network_dict}")
        if len(concept_network_dict) > 0:
            concept_network_data = nru.get_starting_info_from_concept_network(concept_network_dict)
            #print(f"concept_network_data = {concept_network_data}")
            current_concept_network = concept_network_data[0]
            
            self.current_starting_coordinate = current_concept_network["starting_coordinate"]
            self.current_starting_node = current_concept_network["starting_node"]
            self.current_starting_endpoints = current_concept_network["starting_endpoints"]
            self.current_starting_soma = current_concept_network["starting_soma"]
            self.current_touching_soma_vertices = current_concept_network["touching_soma_vertices"]
            self.current_soma_group_idx = current_concept_network["soma_group_idx"]
            self.concept_network = concept_network_dict[self.current_starting_soma][self.current_soma_group_idx]
            
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
        suppress_disconnected_errors=False
        for j,branch_data in curr_limb_correspondence.items():
            if (not branch_objects is None) and j in branch_objects:
                #print(f"using existing branch object for node {j}")
                branch_obj = branch_objects[j]
            else:
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
            
            if j not in self.concept_network:
                self.concept_network.add_node(j)
                suppress_disconnected_errors=True
            
            
            #Set all  of the branches as data in the nodes
            xu.set_node_data(self.concept_network,
                            node_name=j,
                            curr_data=branch_obj,
                             curr_data_label="data"
                            )
            
        #Setting the concept network
        self.deleted_edges =deleted_edges
        self.created_edges = created_edges
        
        if debug_edges:
            print(f"self.deleted_edges = {self.deleted_edges}")
            print(f"self.created_edges = {self.created_edges}")
        
        self.set_concept_network_edges_from_current_starting_data()
        self.concept_network_directional = self.convert_concept_network_to_directional(no_cycles = True,
                                                                            suppress_disconnected_errors=suppress_disconnected_errors)
        
        if debug_edges:
            print(f"self.deleted_edges = {self.deleted_edges}")
            print(f"self.created_edges = {self.created_edges}")

        
        
    # ----------------- 9/2 To help with compression ------------------------- #
    def get_attribute_dict(self,attribute_name):
        attribute_dict = dict()
        for branch_idx,curr_branch in enumerate(self):
            if attribute_name == "spines":
                if not curr_branch.spines is None:
                    attribute_dict[branch_idx] = [tu.original_mesh_faces_map(curr_branch.mesh,k) for k in curr_branch.spines]
                else:
                    attribute_dict[branch_idx] = None
            else:
                attribute_dict[branch_idx] = getattr(curr_branch,attribute_name)
            
        return attribute_dict
    
    def set_attribute_dict(self,attribute_name,attribute_dict):
        for branch_idx,curr_branch in enumerate(self):
            if branch_idx in list(attribute_dict.keys()):
                if attribute_name == "spines":
                    print(f"     Branch {branch_idx}")
                    if not attribute_dict[branch_idx] is None:
                        print(f"curr_branch.mesh = {curr_branch.mesh}")
                        curr_branch.spines = [curr_branch.mesh.submesh([k],append=True,repair=False) for k in attribute_dict[branch_idx]]
                    else:
                        curr_branch.spines = None
                else:
                    setattr(curr_branch,attribute_name,attribute_dict[branch_idx])
            else:
                print(f"Skipping attributes for Branch {branch_idx} because not in dictionary")
                
    
    # Defining some useful built in functions
    def __getitem__(self,key):
        return self.concept_network.nodes[key]["data"]
    def __setitem__(self,key,newvalue):
        self.concept_network.nodes[key]["data"] = newvalue
    def __len__(self):
        return len(list(self.concept_network.nodes()))
    #for the iterable
    def __iter__(self):
        return self
    def __next__(self):
        self._index += 1
        #print(f"Limb self._index = {self._index}")
        sorted_node_indexes = np.sort(list(self.concept_network.nodes()))
        if self._index >= len(self):
            self._index = -1
            raise StopIteration
        else:
            return self[sorted_node_indexes[self._index]]
    
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
    
    def __init__(self,mesh,mesh_face_idx=None,sdf=None,volume_ratio=None):
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
        if volume_ratio is None:
            self.volume_ratio = sm.soma_volume_ratio(self.mesh,
                                                     #watertight_method="fill_holes"
                                                    )
        else:
            print("Using precomputed volume ratio")
            self.volume_ratio = volume_ratio
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
    


import preprocess_neuron as pn
import preprocessing_vp2 as pre

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

    Example 1:
    How you could generate completely from mesh to help with debugging:
    
    # import trimesh_utils as tu
    # mesh_file_path = Path("/notebooks/test_neurons/multi_soma_example.off")
    # mesh_file_path.exists()
    # current_neuron_mesh = tu.load_mesh_no_processing(str(mesh_file_path.absolute()))

    # # picking a random segment id
    # segment_id = 12345
    # description = "double_soma_meshafterparty"

    # # --------------------- Processing the Neuron ----------------- #
    # import soma_extraction_utils as sm

    # somas = sm.extract_soma_center(segment_id,
    #                              current_neuron_mesh.vertices,
    #                              current_neuron_mesh.faces)

    # import time
    # meshparty_time = time.time()
    # import compartment_utils as cu
    # cu = reload(cu)

    # import meshparty_skeletonize as m_sk
    # import preprocess_neuron as pn
    # pn = reload(pn)
    # m_sk = reload(m_sk)

    # somas = somas

    # nru = reload(nru)
    # neuron = reload(neuron)
    # current_neuron = neuron.Neuron(
    #     mesh=current_neuron_mesh,
    #     segment_id=segment_id,
    #     description=description,
    #     decomposition_type="meshafterparty",
    #     somas = somas,
    #     #branch_skeleton_data=branch_skeleton_data,
    #     suppress_preprocessing_print=False,
    # )
    # print(f"Total time for processing: {time.time() - meshparty_time}")


    # # ----------------- Calculating the Spines and Width ----------- #
    # current_neuron.calculate_spines(print_flag=True)
    # #nviz.plot_spines(current_neuron)

    # current_neuron.calculate_new_width(no_spines=False,
    #                                        distance_by_mesh_center=True)

    # current_neuron.calculate_new_width(no_spines=False,
    #                                        distance_by_mesh_center=True,
    #                                        summary_measure="median")

    # current_neuron.calculate_new_width(no_spines=True,
    #                                        distance_by_mesh_center=True,
    #                                        summary_measure="mean")

    # current_neuron.calculate_new_width(no_spines=True,
    #                                        distance_by_mesh_center=True,
    #                                        summary_measure="median")

    # # ------------------ Saving off the Neuron --------------- #
    # current_neuron.save_compressed_neuron(output_folder=Path("/notebooks/test_neurons/meshafterparty_processed/"),
    #                                      export_mesh=True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    
    
    def get_total_n_branches(self):
        return np.sum([len(self.concept_network.nodes[li]["data"].concept_network.nodes()) for li in self.get_limb_node_names()])
    
    def get_skeleton(self,check_connected_component=True):
        return nru.get_whole_neuron_skeleton(self,
                                 check_connected_component=check_connected_component)
    
    def get_attribute_dict(self,attribute_name):
        attribute_dict = dict()
        for limb_idx,curr_limb in enumerate(self):
            attribute_dict[limb_idx] = curr_limb.get_attribute_dict(attribute_name)
            
        return attribute_dict
    
    def set_attribute_dict(self,attribute_name,attribute_dict):
        for limb_idx,curr_limb in enumerate(self):
            print(f"Working on Limb {limb_idx}:")
            if limb_idx in list(attribute_dict.keys()):
                curr_limb.set_attribute_dict(attribute_name,attribute_dict[limb_idx])
            else:
                print(f"Limb {limb_idx} not in attribute dict so skipping")
            
    def set_computed_attribute_data(self,computed_attribute_data,print_flag=False):
        start_time = time.time()
        
        if computed_attribute_data is None:
            return
        
        for k,v in computed_attribute_data.items():
            self.set_attribute_dict(k,v)

        """  
        # Old way 1 of Setting 
        width_array_lookup = dict()
        width_new_lookup = dict()
        spines_lookup = dict()
        
        for limb_idx,ex_limb in self:
            width_array_lookup = 

        sorted_limb_labels = np.sort(self.get_limb_node_names())
        for limb_idx in sorted_limb_labels:
            ex_limb = self.concept_network.nodes[limb_idx]["data"]

            sorted_branch_labels = np.sort(ex_limb.concept_network.nodes())
            for branch_idx in sorted_branch_labels:
                if print_flag:
                    print(f"---- Working on limb {limb_idx} branch {branch_idx} ------")
                ex_branch = ex_limb.concept_network.nodes[branch_idx]["data"]

                ex_branch.width_array = width_array_lookup[limb_idx][branch_idx] 
                
                if not spines_lookup[limb_idx][branch_idx] is None:
                    ex_branch.spines = [ex_branch.submesh([k],append=True,repair=False) for k in spines_lookup[limb_idx][branch_idx]]
                else:
                    ex_branch.spines = None
                
                ex_branch.width_new = width_new_lookup[limb_idx][branch_idx]
        
        
        # Old Way 2: 
                
        width_array_lookup = spine_width_data["width_array_lookup"]
        width_new_lookup = spine_width_data["width_new_lookup"]
        spines_lookup = spine_width_data["spines_lookup"]
        
        set_attribute_dict("width_array",width_array_lookup)
        set_attribute_dict("width_new",width_new_lookup)
        set_attribute_dict("spines",spines_lookup)
        
        """
        
        if print_flag:
            print(f"Total time for spine/width compression = {time.time() - start_time}")
    
    def get_computed_attribute_data(self,
                                    attributes = ["width_array","width_new","spines","spines_volume","labels"],
                                    one_dict=True,
                                    print_flag=False):
        start_time = time.time()
        
        lookup_values = []
        lookup_dict = dict()
        for a in attributes: 
            current_lookup_value = self.get_attribute_dict(a)
            lookup_values.append(current_lookup_value)
            if one_dict:
                lookup_dict[a] = current_lookup_value

        if print_flag:
            print(f"Total time for spine/width compression = {time.time() - start_time}")
        
        """  
        # OLD WAY 1: OF DOING THIS WITHOUT ITERABLE
        width_array_lookup = dict()
        width_new_lookup = dict()
        spines_lookup = dict()

        sorted_limb_labels = np.sort(self.get_limb_node_names())
        for limb_idx in sorted_limb_labels:
            ex_limb = self.concept_network.nodes[limb_idx]["data"]

            spines_lookup[limb_idx] = dict()
            width_array_lookup[limb_idx] = dict()
            width_new_lookup[limb_idx] = dict()

            sorted_branch_labels = np.sort(ex_limb.concept_network.nodes())
            for branch_idx in sorted_branch_labels:
                if print_flag:
                    print(f"---- Working on limb {limb_idx} branch {branch_idx} ------")
                ex_branch = ex_limb.concept_network.nodes[branch_idx]["data"]

                width_array_lookup[limb_idx][branch_idx] = ex_branch.width_array
                if not ex_branch.spines is None:
                    spines_lookup[limb_idx][branch_idx] = [tu.original_mesh_faces_map(ex_branch.mesh,k) for k in ex_branch.spines]
                else:
                    spines_lookup[limb_idx][branch_idx] = None
                width_new_lookup[limb_idx][branch_idx] = ex_branch.width_new
                
        # Old Way #2: where not generic
        width_array_lookup = self.get_attribute_dict("width_array")
        width_new_lookup = self.get_attribute_dict("width_new")
        spines_lookup = self.get_attribute_dict("spines")
        
        spine_width_data = dict(
                width_array_lookup = width_array_lookup,
                width_new_lookup = width_new_lookup,
                spines_lookup = spines_lookup
               )
        """

        if one_dict:
            return lookup_dict
        else:
            return lookup_values
    
    @property
    def skeleton(self,check_connected_component=False):
        return nru.get_whole_neuron_skeleton(self,
                                 check_connected_component=check_connected_component)
    

    
    def __init__(self,mesh,
                 segment_id=None,
                 description=None,
                 preprocessed_data=None,
                 
                 decomposition_type="meshafterparty",
                 mesh_correspondence="meshparty", #meshafterparty_adaptive
                 distance_by_mesh_center=True, #how the distance is calculated for mesh correspondence
                 meshparty_segment_size = 100,
                 meshparty_n_surface_downsampling = 2,
                 meshparty_adaptive_correspondence_after_creation=False,
                suppress_preprocessing_print=True,
                 computed_attribute_dict=None,
                 somas = None,
                 branch_skeleton_data=None,
                 combine_close_skeleton_nodes = True,
                combine_close_skeleton_nodes_threshold=700,
                 
                ignore_warnings=True,
                suppress_output=False,
                calculate_spines=True,
                widths_to_calculate=["no_spine_median_mesh_center"],
                fill_hole_size=0,# The old value for the parameter when performing 2000,
                 
                 preprocessing_version=2,
                 limb_to_branch_objects=None,
                 
                 glia_faces=None,
                 nuclei_faces = None,
                ):
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
        
        if suppress_output:
            print("Processing Neuorn in minimal output mode...please wait")
        
        
        with su.suppress_stdout_stderr() if suppress_output else su.dummy_context_mgr():

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
                self.not_processed_soma_containing_meshes = dc(mesh.not_processed_soma_containing_meshes)
                self.glia_faces = dc(mesh.glia_faces)
                self.non_soma_touching_meshes = dc(mesh.non_soma_touching_meshes)
                
                if hasattr(mesh,"decomposition_type"):
                    self.decomposition_type = dc(mesh.decomposition_type)
                else:
                    self.decomposition_type = None
                    
                #in order to become an iterable
                self._index = -1
                
                return 
                
                
        
            if ignore_warnings: 
                su.ignore_warnings()
                
            #in order to become an iterable
            self._index = -1

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
            self.decomposition_type = decomposition_type


            neuron_start_time =time.time()
            if preprocessed_data is None: 
                print("--- 0) Having to preprocess the Neuron becuase no preprocessed data\nPlease wait this could take a while.....")
                
                with su.suppress_stdout_stderr() if suppress_preprocessing_print else su.dummy_context_mgr():
                    if fill_hole_size == -1:
                        vert_holes = tu.find_border_vertex_groups(self.mesh)
                        vert_holes_size = np.array([len(k) for k in vert_holes])
                        fill_hole_size = np.max(fill_hole_size) + 10
                        print(f"Calculating max hole filling size as {fill_hole_size} ")

                        
                    if fill_hole_size > 0 and len(tu.find_border_vertex_groups(self.mesh))>0:
                        try:
                            mesh = tu.fill_holes(mesh,max_hole_size=fill_hole_size)
                            self.mesh = mesh
                        except:
                            print("**** Tried to fill holes but was unable to, just preceeding on*****")
                        else:
                            vert_holes = tu.find_border_vertex_groups(self.mesh)
                            vert_holes_size = np.array([len(k) for k in vert_holes])
                            print(f"Successfully filled all holes up to size {fill_hole_size}")
                            print(f"Still existing holes = {vert_holes_size}")
                    else:
                        print("Skipping the hole filling")
                        
                    if preprocessing_version == 1:
                        preprocessed_data = pn.preprocess_neuron(mesh,
                                         segment_id=segment_id,
                                         description=description,
                                          decomposition_type=decomposition_type,
                                            mesh_correspondence=mesh_correspondence,
                                            distance_by_mesh_center=distance_by_mesh_center,
                                            meshparty_segment_size =meshparty_segment_size,
                                             meshparty_n_surface_downsampling = meshparty_n_surface_downsampling,
                                          somas=somas,
                                            branch_skeleton_data=branch_skeleton_data,
                                            combine_close_skeleton_nodes = combine_close_skeleton_nodes,
                                            combine_close_skeleton_nodes_threshold=combine_close_skeleton_nodes_threshold)
                    elif preprocessing_version == 2:
                        
                        if "meshafterparty" in decomposition_type.lower():
                            use_meshafterparty = True
                        else:
                            use_meshafterparty = False
                        
                        preprocessed_data = pre.preprocess_neuron(
                                                mesh,
                                        segment_id=segment_id,
                                         description=description,
                                          decomposition_type=decomposition_type,
                                        distance_by_mesh_center=distance_by_mesh_center,
                                        meshparty_segment_size =meshparty_segment_size,
                                        
                                        somas=somas, #the precomputed somas
                                        combine_close_skeleton_nodes = combine_close_skeleton_nodes,
                                        combine_close_skeleton_nodes_threshold=combine_close_skeleton_nodes_threshold,
                        
                                        use_meshafterparty=use_meshafterparty,
                                        glia_faces=glia_faces,
                                         nuclei_faces = nuclei_faces)
                        
                    
                    
                    
                    #print(f"preprocessed_data inside with = {preprocessed_data}")
                        
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
            self.not_processed_soma_containing_meshes = preprocessed_data["not_processed_soma_containing_meshes"]
            
            if "glia_faces" in preprocessed_data.keys():
                self.glia_faces = preprocessed_data["glia_faces"]
            else:
                self.glia_faces = []
            
            self.non_soma_touching_meshes = preprocessed_data["non_soma_touching_meshes"]
            self.inside_pieces = preprocessed_data["inside_pieces"]

            soma_meshes = preprocessed_data["soma_meshes"]
            soma_to_piece_connectivity = preprocessed_data["soma_to_piece_connectivity"]
            soma_sdfs = preprocessed_data["soma_sdfs"]
            
            if "soma_volume_ratios" in preprocessed_data.keys() and (not preprocessed_data["soma_volume_ratios"] is None):
                pass
            else:
                print("No soma volume ratios so computing them now")                                                          
                preprocessed_data["soma_volume_ratios"] = [sm.soma_volume_ratio(j) for j in soma_meshes]
                
            soma_volume_ratios = preprocessed_data["soma_volume_ratios"]
                
            
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

            for j,(curr_soma,curr_soma_face_idx,current_sdf,curr_volume_ratio) in enumerate(zip(soma_meshes,soma_meshes_face_idx,soma_sdfs,soma_volume_ratios)):
                Soma_obj = Soma(curr_soma,mesh_face_idx=curr_soma_face_idx,sdf=current_sdf,volume_ratio=curr_volume_ratio)
                soma_name = f"S{j}"
                #Add the soma object as data in 
                
                # --- 11/21 adaption that accounts for if soma is not in the concept network
                if soma_name in self.concept_network.nodes():
                    xu.set_node_data(curr_network=self.concept_network,
                                         node_name=soma_name,
                                         curr_data=Soma_obj,
                                         curr_data_label="data")
                else:
                    print(f"Did not have {soma_name} in concept network so adding it")
                    self.concept_network.add_node(soma_name,data=Soma_obj)
                    
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

                
                if not (limb_to_branch_objects is None) and j in limb_to_branch_objects.keys():
                    branch_objects = limb_to_branch_objects[j]
                else:
                    branch_objects = None
                    

                print(f"curr_limb_concept_networks= {curr_limb_concept_networks}")
                Limb_obj = Limb(
                                 mesh=curr_limb_mesh,
                                 curr_limb_correspondence=curr_limb_correspondence,
                                 concept_network_dict=curr_limb_concept_networks,
                                 mesh_face_idx=curr_limb_mesh_face_idx,
                                 labels=curr_limb_label,
                                branch_objects = branch_objects
                                )


                limb_name = f"L{j}"
                #Add the soma object as data in
                if limb_name not in self.concept_network.nodes():
                    self.concept_network.add_node(limb_name)
                
                xu.set_node_data(curr_network=self.concept_network,
                                     node_name=limb_name,
                                     curr_data=Limb_obj,
                                     curr_data_label="data")

                #xu.set_node_data(self.concept_network,node_name=soma_name,curr_data=Soma_obj,curr_data_label="data")

            print(f"--- 4) Finshed generating Limb objects and adding them to concept graph: {time.time() - neuron_start_time}")
            

            if decomposition_type == "meshparty" and meshparty_adaptive_correspondence_after_creation:
                neuron_start_time =time.time()
                print(f"--- 5) Doing the adaptive mesh correspondence on the meshparty preprocessing ---")
                nru.apply_adaptive_mesh_correspondence_to_neuron(self)
                print(f"--- 5) Finished Doing the adaptive mesh correspondence on the meshparty preprocessing: {time.time() - neuron_start_time}")
            else:
                print(f"--- 5) SKIPPING Doing the adaptive mesh correspondence on the meshparty preprocessing ---")
            
            if not computed_attribute_dict is None:
                neuron_start_time =time.time()
                print(f"--- 6) Using the computed_attribute_dict to populate neuron attributes ---")
                self.set_computed_attribute_data(computed_attribute_dict)
                print(f"--- 6) FINISHED Using the computed_attribute_dict to populate neuron attributes: {time.time() - neuron_start_time}")
            else:
                print(f"--- 6) SKIPPING Using the computed_attribute_dict to populate neuron attributes ---")
                
            # printing what concept network looks like 
            print(f"self.n_limbs = {self.n_limbs}")
            
            if self.n_limbs > 0:    
                if calculate_spines:
                    #check to see that spines don't already exist
                    print("7) Calculating the spines for the neuorn if do not already exist")
                    if not self.spines_already_computed():
                        print("7a) calculating spines because didn't exist")
                        self.calculate_spines()

                for w in widths_to_calculate:
                    self.calculate_new_width(width_name=w)
            else:
                print("Skipping the width and spine calculation because no limbs")
                
            
            
        print(f"Total time for neuron instance creation = {time.time() - neuron_creation_time}")

    # ------------ 9/24: Function that will see if spines are already comuted ------------ #
    def spines_already_computed(self):
        """
        Pseudocode:
        1) Iterate through all of limbs and branches
        2) If find one instance where spines not None, return True
        3) If none found, return False

        """
        found_spines=False

        self._index = -1

        for limb in self:
            limb._index = -1
            if found_spines == True:
                break
            for branch in limb:
                if not branch.spines is None:
                    print(f"Found non-null branch = {branch.spines}")
                    found_spines=True
                    break
            limb._index = -1
        self._index = -1
        return found_spines
    #------------------ some useful built in functions ------------------ #
    
    def __getitem__(self,key):
        if type(key) == int:
            key = f"L{key}"
        return self.concept_network.nodes[key]["data"]
    def __setitem__(self,key,newvalue):
        if type(key) == int:
            key = f"L{key}"
        self.concept_network.nodes[key]["data"] = newvalue
    def __len__(self):
        return len(list(self.get_limb_node_names()))
    def __iter__(self):
        return self
    def __next__(self):
        self._index += 1
        #print(f"Neuron self._index = {self._index}")
        if self._index >= len(self):
            self._index = -1
            raise StopIteration
        else:
            return self[self._index]
        
    
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

        obj1 = neuron.Neuron(double_soma_obj,suppress_output=False)
        obj2 = neuron.Neuron(double_soma_obj,suppress_output=False)

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
                               "not_processed_soma_containing_meshes",
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
    def get_soma_meshes(self):
        """
        Gives the same output that running the soma identifier would
        
        Retunrs: a list containing the following elements
        1) list of soma meshes (N)
        2) scalar value of time it took to process (dummy 0)
        3) list of soma sdf values (N)
        
        """
        soma_meshes = [self.concept_network.nodes[k]["data"].mesh for k in sorted(self.get_soma_node_names())]
        return soma_meshes
    
    def get_somas(self):
        """
        Gives the same output that running the soma identifier would
        
        Retunrs: a list containing the following elements
        1) list of soma meshes (N)
        2) scalar value of time it took to process (dummy 0)
        3) list of soma sdf values (N)
        
        """
        soma_meshes = [self.concept_network.nodes[k]["data"].mesh for k in sorted(self.get_soma_node_names())]
        somas_sdfs = [self.concept_network.nodes[k]["data"].sdf for k in sorted(self.get_soma_node_names())]
        somas =[soma_meshes,0,somas_sdfs]
        return somas
    
    
    def get_limb_node_names(self,return_int=False):
        with_l_names = [k for k in self.concept_network.nodes() if "L" in k]
        if return_int:
            sorted_numbers = np.sort([int(k[1:]) for k in with_l_names])
            return [int(k) for k in sorted_numbers]
        else:
            return with_l_names
        
    def get_branch_node_names(self,limb_idx):
        limb_idx = nru.limb_label(limb_idx)
        curr_limb_obj = self.concept_network.nodes[limb_idx]["data"]
        return list(curr_limb_obj.concept_network.nodes())
    def get_soma_node_names(self,int_label=False):
        soma_names = [k for k in self.concept_network.nodes() if "S" in k]
        if int_label:
            return [int(k[1:]) for k in soma_names]
        else:
            return soma_names
    
    def get_limbs_touching_soma(self,soma_idx):
        """
        Purpose: To get all of the limb names contacting a certain soma
        
        Example:
        current_neuron.get_limbs_touching_soma(0)
        
        """
        return xu.get_neighbors(self.concept_network,nru.soma_label(soma_idx),int_label=False)
    
    def get_somas_touching_limbs(self,limb_idx,return_int=True):
        """
        Purpose: To get all of the limb names contacting a certain soma
        
        Example:
        current_neuron.get_limbs_touching_soma(0)
        
        """
        soma_neighbors = xu.get_neighbors(self.concept_network,nru.limb_label(limb_idx),int_label=False)
        if return_int:
            return [int(k[1:]) for k in soma_neighbors]
        else:
            return soma_neighbors
    
    # --------------------- For saving the neuron -------------------- #
    def save_compressed_neuron(self,output_folder,file_name="",return_file_path=False,
                               export_mesh=False,
                              suppress_output=True):
        """
        Will save the neuron in a compressed format:
        
        
        Ex: How to save compressed neuron
        double_neuron_preprocessed.save_compressed_neuron("/notebooks/test_neurons/preprocessed_neurons/meshafterparty/",export_mesh=True,
                                         file_name=f"{double_neuron_preprocessed.segment_id}_{double_neuron_preprocessed.description}_meshAfterParty",
                                         return_file_path=True)
        
        Ex: How to reload compressed neuron
        nru.decompress_neuron(filepath="/notebooks/test_neurons/preprocessed_neurons/meshafterparty/12345_double_soma_meshAfterParty",
                     original_mesh='/notebooks/test_neurons/preprocessed_neurons/meshafterparty/12345_double_soma_meshAfterParty')
        """
        if suppress_output:
            print("Saving Neuorn in suppress_output mode...please wait")
        
        
        with su.suppress_stdout_stderr() if suppress_output else su.dummy_context_mgr():
            returned_file_path = nru.save_compressed_neuron(self,
                                       output_folder=output_folder,
                                       file_name=file_name,
                                       return_file_path=True,
                                       export_mesh=export_mesh)
        print(f"Saved File at location: {returned_file_path}")
        
        if  return_file_path:
            return returned_file_path
    
    #how to save neuron object
    def save_neuron_object(self,
                          filename=""):
        if filename == "":
            print("No filename/location given so creating own")
            filename = f"{self.segment_id}_{self.description}.pkl"
        file = Path(filename)
        print(f"Saving Object at: {file.absolute()}")
        
        su.save_object(self,file)
    
    def calculate_new_width(self,
                          skeleton_segment_size = 1000,
                           width_segment_size=None,
                          width_name = None,
                            distance_by_mesh_center=True,
                           no_spines=True,
                            summary_measure="mean",
                            
                          **kwargs):
        """
        Purpose: To calculate new width definitions based on if
        1) Want to use skeleton center or mesh center
        2) Want to include spines or not
        
        Examples:
        current_neuron.calculate_new_width(no_spines=False,
                                               distance_by_mesh_center=True)
                                               
        current_neuron.calculate_new_width(no_spines=False,
                                       distance_by_mesh_center=True,
                                       summary_measure="median")
                                       
        current_neuron.calculate_new_width(no_spines=True,
                                       distance_by_mesh_center=True,
                                       summary_measure="mean")
        
        current_neuron.calculate_new_width(no_spines=True,
                                       distance_by_mesh_center=True,
                                       summary_measure="median")
        
        """
        #print(f"width_name = {width_name}")
        if width_name is None:
            width_name = str(summary_measure)
        else:
            if "mean" in width_name:
                summary_measure = "mean"
            elif "median" in width_name:
                summary_measure = "median"
            else: 
                raise Exception("No summary statistic was specified in the name")
        
        if summary_measure != "mean":
            width_name = width_name.replace("mean",summary_measure)
            if summary_measure not in width_name:
                width_name = f"{width_name}_{summary_measure}"
                
        if ("no_spine" not in width_name) and (no_spines):
            width_name = f"no_spine_{width_name}"
        if ("mesh_center" not in width_name) and (distance_by_mesh_center):
            width_name = f"{width_name}_mesh_center"
        
        
        for limb_idx in self.get_limb_node_names():
            for branch_idx in self.get_branch_node_names(limb_idx):
                print(f"Working on limb {limb_idx} branch {branch_idx}")
                curr_branch_obj = self.concept_network.nodes[nru.limb_label(limb_idx)]["data"].concept_network.nodes[branch_idx]["data"]
                
                if "mesh_center" in width_name:
                    distance_by_mesh_center = True
                else:
                    distance_by_mesh_center = False
                
                if "no_spine" in width_name:
                    no_spines=True
                else:
                    no_spine=False
                    
                if "mean" in width_name:
                    summary_measure = "mean"
                elif "median" in width_name:
                    summary_measure = "median"
                
                
#                 print(f"Before width call width_name =  {width_name} with parameters:\n "
#                       f"distance_by_mesh_center={distance_by_mesh_center}, no_spines = {no_spines}, summary_measure={summary_measure}")
                
                
                #Add rule that will help skip segment if has no spines
                already_computed = False
                
                
                if (curr_branch_obj.spines is None or len(curr_branch_obj.spines) == 0) and no_spines:
                    
                    #see if we can skip
                    new_width_name = width_name.replace("no_spine_","")
                    if new_width_name in curr_branch_obj.width_new.keys():
                        print("No spines and using precomputed width")
                        curr_branch_obj.width_new[width_name] = curr_branch_obj.width_new[new_width_name]
                        curr_branch_obj.width_array[width_name] = curr_branch_obj.width_array[new_width_name]
                        
                        already_computed=True
                
        
                if not already_computed:
                    current_width_array,current_width = wu.calculate_new_width(curr_branch_obj, 
                                          skeleton_segment_size=skeleton_segment_size,
                                          width_segment_size=width_segment_size, 
                                          distance_by_mesh_center=distance_by_mesh_center,
                                          no_spines=no_spines,
                                          summary_measure=summary_measure,
                                          return_average=True,
                                          print_flag=False,
                                        **kwargs)


                    curr_branch_obj.width_new[width_name] = current_width
                    curr_branch_obj.width_array[width_name] = current_width_array
    
    
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
                
                
                current_width_array,current_width = wu.calculate_new_width(curr_branch_obj, 
                                      skeleton_segment_size=skeleton_segment_size,
                                      width_segment_size=width_segment_size, 
                                      distance_by_mesh_center=distance_by_mesh_center,
                                      return_average=True,
                                      print_flag=False,
                                    **kwargs)


                curr_branch_obj.width_new[width_name] = current_width
                curr_branch_obj.width_array[width_name] = current_width_array
    
    
    
    import system_utils as su
    def calculate_spines(self,
                        #query="width > 400 and n_faces_branch>100",
                         query="median_mesh_center > 140 and n_faces_branch>100",#previous used median_mesh_center > 140
                        clusters_threshold=3,#2,
                        smoothness_threshold=0.1,#0.08,
                        shaft_threshold=300,
                        cgal_path=Path("./cgal_temp"),
                        print_flag=False,
                        filter_out_border_spines=True,
                        skeleton_endpoint_nullification=True,
                         soma_vertex_nullification = True,
                         border_percentage_threshold=0.3,
                        check_spine_border_perc=0.4,
                        calculate_spine_volume=True):
        
        print(f"query = {query}")
        print(f"smoothness_threshold = {smoothness_threshold}")
        if type(query) == dict():
            functions_list = query["functions_list"]
            current_query = query["query"]
        else:
            functions_list = ["median_mesh_center","n_faces_branch"]
            current_query = query
            
        
        #check that have calculated the median mesh center if required
        if "median_mesh_center" in functions_list:
            if "median_mesh_center" not in self[0][0].width_new.keys():
                print("The median_mesh_center was requested but has not already been calculated so calculating now.... ")
                
                self.calculate_new_width(no_spines=False,
                                       distance_by_mesh_center=True,
                                       summary_measure="median")
            else:
                print("The median_mesh_center was requested and HAS already been calculated")

        new_branch_dict = ns.query_neuron(self,
                       functions_list=functions_list,
                       query=current_query)
        if print_flag:
            print(f"new_branch_dict = {new_branch_dict}")
 
        
        
        self._index = -1
        for limb_idx,curr_limb in enumerate(self):
            limb_idx = f"L{limb_idx}"
            
            #to be used for endpoint nullification
            if skeleton_endpoint_nullification:
                curr_limb_end_coords = sk.find_skeleton_endpoint_coordinates(curr_limb.skeleton)
            
            
            if soma_vertex_nullification:
                soma_verts = np.concatenate([self[f"S{k}"].mesh.vertices for k in curr_limb.touching_somas()])
                soma_kdtree = KDTree(soma_verts)
            
            curr_limb._index = -1
            for branch_idx,curr_branch in enumerate(curr_limb):
                if limb_idx in new_branch_dict.keys():
                    #print(f"new_branch_dict[{limb_idx}] = {new_branch_dict[limb_idx]}")
                    #print(f"branch_idx = {branch_idx}")
                    if branch_idx in new_branch_dict[limb_idx]:
                        if print_flag:
                            print(f"Working on limb {limb_idx} branch {branch_idx}")
                        #calculate the spines
                        
                        #su.compressed_pickle(curr_branch,"curr_branch_before_spines")
                        
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
                        
                            
                        #print(f"curr_branch.mesh = {curr_branch.mesh}")
#                         spine_submesh_split = spu.get_spine_meshes_unfiltered_from_mesh(curr_branch.mesh,
#                                                                 segment_name=f"{limb_idx}_{branch_idx}",
#                                                                 clusters=clusters_threshold,
#                                                                 smoothness=smoothness_threshold,
#                                                                 cgal_folder = cgal_path,
#                                                                 delete_temp_file=True,
#                                                                 return_sdf=False,
#                                                                 print_flag=False,
#                                                                 shaft_threshold=shaft_threshold)
                        
                        if print_flag:
                            print(f"--> n_spines found before filtering = {len(spine_submesh_split)}")

        #                 if limb_idx == "L0":
        #                     if branch_idx == 0:
        #                         print(f"spine_submesh_split = {spine_submesh_split}")

                        spine_submesh_split_filtered = spu.filter_spine_meshes(spine_submesh_split,
                                                                              spine_n_face_threshold=20)
        #                 if limb_idx == "L0":
        #                     if branch_idx == 0:
        #                         print(f"spine_submesh_split_filtered = {spine_submesh_split_filtered}")
        
                        if filter_out_border_spines:
                            if print_flag:
                                print("Using the filter_out_border_spines option")
                            spine_submesh_split_filtered = spu.filter_out_border_spines(self[limb_idx][branch_idx].mesh,
                                                                                        spine_submesh_split_filtered,
                                                                                        border_percentage_threshold=border_percentage_threshold,
                                                                                        check_spine_border_perc=check_spine_border_perc,
                                                                                        verbose=print_flag
                                                                                       )
                        if skeleton_endpoint_nullification:
                            if print_flag:
                                print("Using the skeleton_endpoint_nullification option")
                                
                            spine_submesh_split_filtered = tu.filter_meshes_by_containing_coordinates(spine_submesh_split_filtered,
                                                                        curr_limb_end_coords,
                                                                        distance_threshold=500)
                        
                        if soma_vertex_nullification:
                            if print_flag:
                                print("Using the soma_vertex_nullification option")
                                
                            spine_submesh_split_filtered = spu.filter_out_soma_touching_spines(spine_submesh_split_filtered,
                                                                        soma_kdtree=soma_kdtree)
                            
                        
                

                        if print_flag:
                            print(f"--> n_spines found = {len(spine_submesh_split_filtered)}")
                        curr_branch.spines = spine_submesh_split_filtered
                        
                    else:
                        curr_branch.spines = None
                else:
                    curr_branch.spines = None
                
                # will compute the spine volumes if asked for 
                if calculate_spine_volume:
                    curr_branch.compute_spines_volume()
                    
    @property
    def spines(self):
        self._index = -1
        total_spines = []
        for b in self:
            if not b.spines is None:
                total_spines += b.spines
        return total_spines
    
    @property
    def spines_volume(self):
        self._index = -1
        total_spines_volume = []
        for b in self:
            if not b.spines_volume is None:
                total_spines_volume += b.spines_volume
        return total_spines_volume
    
    def compute_spines_volume(self):
        self._index = -1
        for b in self:
            b.compute_spines_volume()
        
    
    def plot_soma_limb_concept_network(self,
                                      soma_color="red",
                                      limb_color="aqua",
                                      node_size=800,
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
        
        nviz.plot_soma_limb_concept_network(self,
                                              soma_color=soma_color,
                                              limb_color=limb_color,
                                              node_size=node_size,
                                              font_color=font_color,
                                              node_colors=node_colors,
                                              **kwargs
                                            
                                           )
        
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
    
    # ---- 11/20 functions that will help compute statistics of the neuron object ----------
    
    # -- skeleton and branch data ---
    @property
    def n_error_limbs(self):
        return nru.n_error_limbs(self)
    @property
    def same_soma_multi_touching_limbs(self):
        return nru.same_soma_multi_touching_limbs(self)
    @property
    def multi_soma_touching_limbs(self):
        return nru.multi_soma_touching_limbs(self)
    
    @property
    def n_somas(self):
        return nru.n_somas(self)
    
    @property
    def n_limbs(self):
        return nru.n_limbs(self)
    
    @property
    def n_limbs(self):
        return nru.n_limbs(self)
    
    @property
    def n_branches_per_limb(self):
        return nru.n_branches_per_limb(self)
    
    @property
    def n_branches(self):
        return nru.n_branches(self)
    
    @property
    def skeleton_length_per_limb(self):
        return nru.skeleton_length_per_limb(self)
    
    @property
    def skeletal_length(self):
        return nru.skeletal_length(self)
    
    @property
    def max_limb_skeletal_length(self):
        return nru.max_limb_skeletal_length(self)
    
    @property 
    def max_limb_n_branches(self):
        return nru.max_limb_n_branches(self)
    
    @property
    def median_branch_length(self):
        return nru.median_branch_length(self)
    

    # -- width data --
    @property
    def width_median(self):
        return nru.width_median(self)
    @property
    def width_no_spine_median(self):
        return nru.width_no_spine_median(self)

    @property
    def width_90_perc(self):
        return nru.width_perc(self,perc=90)
    @property
    def width_no_spine_90_perc(self):
        return nru.width_no_spine_perc(self,perc=90)
    

    
    # -- spine entries--
    @property
    def n_spines(self):
        return nru.n_spines(self)

    @property
    def spine_density(self):
        return nru.spine_density(self)
    
    @property
    def spines_per_branch(self):
        return nru.spines_per_branch(self)

    @property
    def n_spine_eligible_branches(self):
        return nru.n_spine_eligible_branches(self)

    @property
    def spine_eligible_branch_lengths(self):
        return nru.spine_eligible_branch_lengths(self)
    @property
    def skeletal_length_eligible(self):
        return nru.skeletal_length_eligible(self)
    
    @property
    def spine_density_eligible(self):
        return nru.spine_density_eligible(self)

    @property
    def spines_per_branch_eligible(self):
        return nru.spines_per_branch_eligible(self)
    
    # ------ spine volume issues ----
    @property
    def total_spine_volume(self):
        return nru.total_spine_volume(self)

    @property
    def spine_volume_median(self):
        return nru.spine_volume_median(self)
    
    @property
    def spine_volume_density(self):
        return nru.spine_volume_density(self)
    
    @property
    def spine_volume_density_eligible(self):
        return nru.spine_volume_density_eligible(self)
    
    @property
    def spine_volume_per_branch_eligible(self):
        return nru.spine_volume_per_branch_eligible(self)
    
    def neuron_stats(self):
        stats_dict = dict(
                        n_not_processed_soma_containing_meshes = len(self.not_processed_soma_containing_meshes),
                        n_error_limbs=self.n_error_limbs,
                        n_same_soma_multi_touching_limbs=len(self.same_soma_multi_touching_limbs),
                        n_multi_soma_touching_limbs = len(self.multi_soma_touching_limbs),
                        n_somas=self.n_somas,
                        n_limbs=self.n_limbs,
                        n_branches=self.n_branches,
                        max_limb_n_branches=self.max_limb_n_branches,
                       
                        skeletal_length=self.skeletal_length,
                        max_limb_skeletal_length=self.max_limb_skeletal_length,
                        median_branch_length=self.median_branch_length,

                        width_median=self.width_median, #median width from mesh center without spines removed
                        width_no_spine_median=self.width_no_spine_median, #median width from mesh center with spines removed
                        width_90_perc=self.width_90_perc, # 90th percentile for width without spines removed
                        width_no_spine_90_perc=self.width_no_spine_90_perc,  # 90th percentile for width with spines removed

                        n_spines=self.n_spines,

                        spine_density=self.spine_density, # n_spines/ skeletal_length
                        spines_per_branch=self.spines_per_branch,

                        skeletal_length_eligible=self.skeletal_length_eligible, # the skeletal length for all branches searched for spines
                        n_spine_eligible_branches=self.n_spine_eligible_branches,
                        spine_density_eligible = self.spine_density_eligible,
                        spines_per_branch_eligible = self.spines_per_branch_eligible,

                        total_spine_volume=self.total_spine_volume, # the sum of all spine volume
                        spine_volume_median = self.spine_volume_median,
                        spine_volume_density=self.spine_volume_density, #total_spine_volume/skeletal_length
                        spine_volume_density_eligible=self.spine_volume_density_eligible, #total_spine_volume/skeletal_length_eligible
                        spine_volume_per_branch_eligible=self.spine_volume_per_branch_eligible, #total_spine_volume/n_spine_eligible_branche
        
        
        
        )
        
        return stats_dict
    
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

import copy 



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
        
        self.skeleton=skeleton
        self.mesh=mesh
        self.width=width
        self.mesh_face_idx = mesh_face_idx
        
        #calculate the end coordinates of skeleton branch
        self.endpoints = sk.find_branch_endpoints(skeleton)
        self.mesh_center = None
        if not self.mesh is None:
            self.mesh_center = tu.mesh_center_vertex_average(self.mesh)
        self.labels=labels
            




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
    
    def convert_concept_network_to_directional(self,no_cycles = True):
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
        
        node_widths = dict([(k,curr_limb_concept_network.nodes[k]["data"].width) for k in curr_limb_concept_network.nodes() ])
        
        directional_concept_network = nru.convert_concept_network_to_directional(
            curr_limb_concept_network,
            node_widths=node_widths,                                                    
            no_cycles=True,
                                                                            )
        
        return directional_concept_network
        
        
    
    def __init__(self,
                             mesh,
                             curr_limb_correspondence,
                             concept_network_dict,
                             mesh_face_idx=None,
                            labels=[]
                            ):
        
        
        self.mesh=mesh
        if nu.is_array_like(labels):
            labels=[labels]
            
        self.label=labels
        
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
        


class Soma:
    """
    Class that will hold one continus skeleton
    piece that has no branching
    """
    
    def __init__(self,mesh,mesh_face_idx=None,sdf=None):
        self.mesh=mesh
        self.sdf=sdf
        self.mesh_face_idx = mesh_face_idx
        self.volume_ratio = sm.soma_volume_ratio(self.mesh)
        self.side_length_ratios = sm.side_length_ratios(self.mesh)
        self.mesh_center = tu.mesh_center_vertex_average(self.mesh)

        
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
                segment_id=copy.deepcopy(mesh.segment_id)
                description = copy.deepcopy(mesh.description)
                preprocessed_data = copy.deepcopy(mesh.preprocessed_data)
                mesh = copy.deepcopy(mesh.mesh)

            if ignore_warnings: 
                su.ignore_warnings()

            self.mesh = mesh

            if segment_id is None:
                #pick a random segment id
                segment_id = np.random.randint(100000000)
                print(f"picking a random 7 digit segment id: {segment_id}")
            if description is None:
                description = ""


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
                                 append_figure=False,
                                 show_at_end=True,**kwargs):
        if limb_name == "":
            limb_name = f"L{limb_idx}"
            
        curr_limb_concept_network_directional = self.concept_network.nodes[limb_name]["data"].concept_network_directional
        nviz.plot_concept_network(curr_concept_network = curr_limb_concept_network_directional,
                            scatter_size=node_size,
                            show_at_end=show_at_end,
                            append_figure=append_figure,**kwargs)
        
            

            
            
        
            
        
        
        
        
        
            



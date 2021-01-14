"""
Example of how to do spine analysis: 

Pseudocode: 
1) make sure the cgal temp folder exists
2) run the segmentation command
3) Read int csv 
4) Visualize the results using the graph function

import cgal_Segmentation_Module as csm

clusters=2
smoothness = 0.03

from pathlib import Path
cgal_folder = Path("./cgal_temp")
if not cgal_folder.exists():
    cgal_folder.mkdir(parents=True,exist_ok=False)

check_index = 66
current_mesh = total_branch_meshes[check_index]

file_to_write = cgal_folder / Path(f"segment_{check_index}.off")

written_file_location = tu.write_neuron_off(current_mesh,file_to_write)

if written_file_location[-4:] == ".off":
    cgal_mesh_file = written_file_location[:-4]
else:
    cgal_mesh_file = written_file_location
    
print(f"Going to run cgal segmentation with:"
     f"\nFile: {cgal_mesh_file} \nclusters:{clusters} \nsmoothness:{smoothness}")
    
csm.cgal_segmentation(cgal_mesh_file,clusters,smoothness)

#read in the csv file
cgal_output_file = Path(cgal_mesh_file + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + ".csv" )

cgal_data = np.genfromtxt(str(cgal_output_file.absolute()), delimiter='\n')

#get a look at how many groups and what distribution:
from collections import Counter
print(f"Counter of data = {Counter(cgal_data)}")

split_meshes,split_meshes_idx = tu.split_mesh_into_face_groups(current_mesh,cgal_data,return_idx=True,
                               check_connect_comp = False)

split_meshes,split_meshes_idx
# plot the face mapping 
sk.graph_skeleton_and_mesh(other_meshes=[k for k in split_meshes.values()],
                          other_meshes_colors="random")

"""


"""
Pseudocode: 
1) make sure the cgal temp folder exists
2) run the segmentation command
3) Read int csv 
4) Visualize the results using the graph function

"""
import networkx as nx
import cgal_Segmentation_Module as csm
from pathlib import Path
import trimesh_utils as tu
import numpy as np
import numpy_utils as nu
import skeleton_utils as sk
import copy
import neuron_utils as nru
import networkx_utils as xu

connectivity = "edges"

def cgal_segmentation(written_file_location,
                      clusters=2,
                      smoothness=0.03,
                      return_sdf=True,
                     print_flag=False,
                     delete_temp_file=True):
    
    if written_file_location[-4:] == ".off":
        cgal_mesh_file = written_file_location[:-4]
    else:
        cgal_mesh_file = written_file_location
    if print_flag:
        print(f"Going to run cgal segmentation with:"
             f"\nFile: {cgal_mesh_file} \nclusters:{clusters} \nsmoothness:{smoothness}")

    csm.cgal_segmentation(cgal_mesh_file,clusters,smoothness)

    #read in the csv file
    cgal_output_file = Path(cgal_mesh_file + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + ".csv" )
    cgal_output_file_sdf = Path(cgal_mesh_file + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + "_sdf.csv" )

    cgal_data = np.genfromtxt(str(cgal_output_file.absolute()), delimiter='\n')
    cgal_sdf_data = np.genfromtxt(str(cgal_output_file_sdf.absolute()), delimiter='\n')
    
    if delete_temp_file:
        cgal_output_file.unlink()
        cgal_output_file_sdf.unlink()
        
    
    if return_sdf:
        return cgal_data,cgal_sdf_data
    else:
        return cgal_data

def split_mesh_into_spines_shaft(current_mesh,
                           segment_name="",
                           clusters=2,
                          smoothness=0.03,
                          cgal_folder = Path("./cgal_temp"),
                          delete_temp_file=True,
                          shaft_threshold = 300,
                                 return_sdf = True,
                                print_flag = True):

    if not cgal_folder.exists():
        cgal_folder.mkdir(parents=True,exist_ok=False)

    file_to_write = cgal_folder / Path(f"segment_{segment_name}.off")

    written_file_location = tu.write_neuron_off(current_mesh,file_to_write)
    
    cgal_data,cgal_sdf_data = cgal_segmentation(written_file_location,
                                             clusters,
                                             smoothness,
                                             return_sdf=True,
                                               delete_temp_file=delete_temp_file)
    #print(f"file_to_write = {file_to_write.absolute()}")
    if delete_temp_file:
        #print("attempting to delete file")
        file_to_write.unlink()
    
    #get a look at how many groups and what distribution:
    from collections import Counter
    if print_flag:
        print(f"Counter of data = {Counter(cgal_data)}")

    #gets the meshes that are split using the cgal labels
    split_meshes,split_meshes_idx = tu.split_mesh_into_face_groups(current_mesh,cgal_data,return_idx=True,
                                   check_connect_comp = False)
    
    
    if len(split_meshes.keys()) <= 1:
        print("There was only one mesh found from the spine process and mesh split, returning empty array")
        if return_sdf:
            return [],[],[],[],[]
        else:
            return [],[],[],[]
        
    
#     # How to identify just one shaft
#     shaft_index = -1
#     shaft_total = -1
#     for k,v in split_meshes.items():
#         curr_length = len(v.faces)
#         if  curr_length > shaft_total:
#             shaft_index = k
#             shaft_total = curr_length
    
#     shaft_mesh = split_meshes.pop(shaft_index)
#     shaft_mesh_idx = split_meshes_idx.pop(shaft_index)
    
#     print(f"shaft_index = {shaft_index}")
    
    shaft_meshes = []
    shaft_meshes_idx = []
    
    spine_meshes = []
    spine_meshes_idx = []
    
    #Applying a length threshold to get all other possible shaft meshes
    for spine_id,spine_mesh in split_meshes.items():
        if len(spine_mesh.faces) < shaft_threshold:
            spine_meshes.append(spine_mesh)
            spine_meshes_idx.append(split_meshes_idx[spine_id])
        else:
            shaft_meshes.append(spine_mesh)
            shaft_meshes_idx.append(split_meshes_idx[spine_id])
 
    if len(shaft_meshes) == 0:
        if print_flag:
            print("No shaft meshes detected")
        if return_sdf:
            return [],[],[],[],[]
        else:
            return [],[],[],[]
 
    if len(spine_meshes) == 0:
        if print_flag:
            print("No spine meshes detected")

    if return_sdf:
        return spine_meshes,spine_meshes_idx,shaft_meshes,shaft_meshes_idx,cgal_sdf_data
    else:
        return spine_meshes,spine_meshes_idx,shaft_meshes,shaft_meshes_idx
    
    
import numpy as np
import system_utils as su
def get_spine_meshes_unfiltered_from_mesh(current_mesh,
                                          segment_name=None,
                                        clusters=2,
                                        smoothness=0.05,
                                        cgal_folder = Path("./cgal_temp"),
                                        delete_temp_file=True,
                                        return_sdf=False,
                                        print_flag=False,
                                        shaft_threshold=300):
    
    if segment_name is None:
        segment_name = f"{np.random.randint(10,1000)}_{np.random.randint(10,1000)}"
    
    print(f"segment_name before cgal = {segment_name}")
    
    (spine_meshes,
     spine_meshes_idx,
     shaft_meshes,
     shaft_meshes_idx,
    cgal_sdf_data) = spine_data_returned= split_mesh_into_spines_shaft(current_mesh,
                               segment_name=segment_name,
                               clusters=clusters,
                              smoothness=smoothness,
                              cgal_folder = cgal_folder,
                              delete_temp_file=delete_temp_file,
                              return_sdf = True,
                              print_flag=print_flag,
                              shaft_threshold = shaft_threshold)
    if len(spine_meshes) == 0:
        if return_sdf:
            return  [],[]
        else:
            return []
    else:
        spine_mesh_names = [f"s{i}" for i,mesh in enumerate(spine_meshes)]
        shaft_mesh_names = [f"b{i}" for i,mesh in enumerate(shaft_meshes)]

        total_meshes = spine_meshes + shaft_meshes
        total_meshes_idx = spine_meshes_idx + shaft_meshes_idx
        total_names = spine_mesh_names + shaft_mesh_names

        total_edges = []
        for j,(curr_mesh,curr_mesh_idx) in enumerate(zip(total_meshes,total_meshes_idx)):
            touching_meshes = tu.mesh_pieces_connectivity(
                            main_mesh=current_mesh,
                            central_piece=curr_mesh_idx,
                            periphery_pieces=total_meshes_idx,
                            connectivity = "vertices")
            try:
                touching_meshes.remove(j)
            except:
                print(f"j = {j}")
                su.compressed_pickle(current_mesh,"current_mesh")
                su.compressed_pickle(curr_mesh_idx,"curr_mesh_idx")
                su.compressed_pickle(total_meshes_idx,"total_meshes_idx")
                su.compressed_pickle(total_names,"total_names")
                raise Exception("didn't do remove")
                
            #construct the edges
            curr_edges = [[total_names[j],total_names[h]] for h in touching_meshes]
            total_edges += curr_edges

        spine_graph = xu.remove_selfloops(nx.from_edgelist(total_edges))
        #nx.draw(spine_graph,with_labels=True)


        """
        How to determine whih parts that are the shaft
        1) start with biggest shaft
        2) Find the shoftest paths to all shaft parts
        3) add all the nodes that aren't already in the shaft category to the shaft category
        """

        #find the biggest shaft
        biggest_shaft = f"b{np.argmax([len(k.faces) for k in shaft_meshes])}"
        non_biggest_shaft = [k for k in shaft_mesh_names if k != biggest_shaft]

        final_shaft_mesh_names = shaft_mesh_names.copy()
        if len(non_biggest_shaft) > 0:
            #find all shortest paths from biggest shaft to non_biggest_shaft
            shaft_shortest_paths = [nx.shortest_path(spine_graph,
                                                     source=biggest_shaft,target=curr_shaft) for curr_shaft in non_biggest_shaft]

            new_shaft_meshes = [int(k[1:]) for k in np.unique(np.concatenate(shaft_shortest_paths)) if "s" in k]
            #print(f"new_shaft_meshes = {new_shaft_meshes}")
            final_shaft_mesh_names += [k for k in np.unique(np.concatenate(shaft_shortest_paths)) if "s" in k]
            final_shaft_meshes = shaft_meshes + [spine_meshes[k] for k in new_shaft_meshes]
            final_shaft_meshes_idx = np.unique(np.concatenate(shaft_meshes_idx + [spine_meshes_idx[k] for k in new_shaft_meshes]))
        else:
            final_shaft_meshes = shaft_meshes
            final_shaft_meshes_idx = np.unique(np.concatenate(shaft_meshes_idx))

        final_shaft_mesh_names = np.unique(final_shaft_mesh_names)

        final_spine_faces_idx = np.delete(np.arange(0,len(current_mesh.faces)), final_shaft_meshes_idx)

        """
        #Old way of getting all of the spines: by just dividing the mesh using disconnected components
        #after subtracting the shaft mesh

        spine_submesh = current_mesh.submesh([final_spine_faces_idx],append=True)
        spine_submesh_split = spine_submesh.split(only_watertight=False)

        """

        """
        #New way of extracting the spines using graphical methods

        Pseudocode:
        1) remove the shaft meshes from the graph
        2) get the connected components
        3) assemble the connected components total face_idx:
        a. get the sdf values that correspond to those
        b. get the submesh that corresponds to those

        """
        spine_graph.remove_nodes_from(final_shaft_mesh_names)

        spine_submesh_split=[]
        spine_submesh_split_sdf = []
        for sp_list in list(nx.connected_components(spine_graph)):
            curr_spine_face_idx_split = np.concatenate([spine_meshes_idx[int(sp[1:])] for sp in sp_list ])
            spine_submesh_split_sdf.append(cgal_sdf_data[curr_spine_face_idx_split])
            spine_submesh_split.append(current_mesh.submesh([curr_spine_face_idx_split],append=True))


        if print_flag:
            print(f"\n\nTotal Number of Spines Found = {len(spine_submesh_split)}")


        #sort the list by size
        spine_length_orders = [len(k.faces) for k in spine_submesh_split]
        greatest_to_least = np.flip(np.argsort(spine_length_orders))
        spines_greatest_to_least =  np.array(spine_submesh_split)[greatest_to_least]
        spines_sdf_greatest_to_least = np.array(spine_submesh_split_sdf)[greatest_to_least]
        if return_sdf:
            return spines_greatest_to_least,spines_sdf_greatest_to_least
        else:
            return spines_greatest_to_least


def get_spine_meshes_unfiltered(current_neuron,
                 limb_idx,
                branch_idx,
                clusters=2,
                smoothness=0.05,
                cgal_folder = Path("./cgal_temp"),
                delete_temp_file=True,
                return_sdf=False,
                print_flag=False,
                shaft_threshold=300):
    
    get_spine_meshes_unfiltered
    current_mesh = current_neuron.concept_network.nodes[nru.limb_label(limb_idx)]["data"].concept_network.nodes[branch_idx]["data"].mesh
    
    return get_spine_meshes_unfiltered_from_mesh(current_mesh,
                                        segment_name=f"{limb_idx}_{branch_idx}",
                                        clusters=clusters,
                                        smoothness=smoothness,
                                        cgal_folder = cgal_folder,
                                        delete_temp_file=delete_temp_file,
                                        return_sdf=return_sdf,
                                        print_flag=print_flag,
                                        shaft_threshold=shaft_threshold)
    
        
        
"""
These filters didn't seem to work very well...

"""
def sdf_median_mean_difference(sdf_values):
    return np.abs(np.median(sdf_values) - np.mean(sdf_values)) 

def apply_sdf_filter(sdf_values,sdf_median_mean_difference_threshold = 0.025,
                    return_not_passed=False):
    pass_filter = []
    not_pass_filter = []
    for j,curr_sdf in enumerate(sdf_values):
        if sdf_median_mean_difference(curr_sdf)< sdf_median_mean_difference_threshold:
            pass_filter.append(j)
        else:
            not_pass_filter.append(j)
    if return_not_passed:
        return not_pass_filter
    else:
        return pass_filter

def surface_area_to_volume(current_mesh):
    """
    Method to try and differentiate false from true spines
    conclusion: didn't work
    
    Even when dividing by the number of faces
    """
    return current_mesh.bounding_box_oriented.volume/current_mesh.area


def filter_spine_meshes(spine_meshes,
                        spine_n_face_threshold=20):
    return [k for k in spine_meshes if len(k.faces)>=spine_n_face_threshold]


#------------ 9/23 Addition -------------- #
import trimesh_utils as tu
def filter_out_border_spines(mesh,spine_submeshes,
                            border_percentage_threshold=0.3,                                                    
                             check_spine_border_perc=0.9,
                            verbose=False):
    return tu.filter_away_border_touching_submeshes_by_group(mesh,spine_submeshes,
                                                             border_percentage_threshold=border_percentage_threshold,
                                                             inverse_border_percentage_threshold=check_spine_border_perc,
                                                             verbose = verbose,
                                                            )

from pykdtree.kdtree import KDTree
def filter_out_soma_touching_spines(spine_submeshes,soma_vertices=None,soma_kdtree=None,
                                   verbose=False,):
    """
    Purpose: To filter the spines that are touching the somae
    Because those are generally false positives picked up 
    by cgal segmentation
    
    Pseudocode
    1) Create a KDTree from the soma vertices
    2) For each spine:
    a) Do a query against the KDTree with vertices
    b) If any of the vertices have - distance then nullify


    """
    if soma_kdtree is None and not soma_vertices is None:
        soma_kdtree = KDTree(soma_vertices)
    if soma_kdtree is None and soma_verties is None:
        raise Exception("Neither a soma kdtree or soma vertices were given")

    if verbose:
        print(f"Number of spines before soma border filtering = {len(spine_submeshes)}")
    final_spines = []
    for j,sp_mesh in enumerate(spine_submeshes):
        sp_dist,sp_closest = soma_kdtree.query(sp_mesh.vertices)
        n_match_vertices = np.sum(sp_dist==0)
        
        if n_match_vertices == 0:
            final_spines.append(sp_mesh)
        else:
            if verbose:
                print(f"Spine {j} was removed because had {n_match_vertices} border vertices")
    if verbose:
        print(f"Number of spines before soma border filtering = {len(final_spines)}")
    
    return final_spines

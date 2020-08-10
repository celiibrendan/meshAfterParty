"""
These functions just help with generically 
helping with trimesh mesh manipulation
"""

import trimesh
import numpy as np
import networkx as nx
from pykdtree.kdtree import KDTree
import time

import numpy_utils as nu
from pathlib import Path
from tqdm_utils import tqdm

#loading a mesh safely without any processing to mess up the vertices/faces
def load_mesh_no_processing(current_mesh_file):
    if type(current_mesh_file) == type(Path()):
        current_mesh_file = str(current_mesh_file.absolute())
    if current_mesh_file[-4:] != ".off":
        current_mesh_file += ".off"
    return trimesh.load_mesh(current_mesh_file,process=False)


def mesh_center_vertex_average(mesh_list):
    if not nu.is_array_like(mesh_list):
        mesh_list = [mesh_list]
    mesh_list_centers = [np.array(np.mean(k.vertices,axis=0)).astype("float")
                           for k in mesh_list]
    if len(mesh_list) == 1:
        return mesh_list_centers[0]
    else:
        return mesh_list_centers
    
def mesh_center_weighted_face_midpoints(mesh):
    """
    Purpose: calculate a mesh center point
    
    Pseudocode: 
    a) get the face midpoints
    b) get the surface area of all of the faces and total surface area
    c) multiply the surface area percentage by the midpoints
    d) sum up the products
    """
    #a) get the face midpoints
    face_midpoints = mesh.triangles_center
    #b) get the surface area of all of the faces and total surface area
    total_area = mesh.area
    face_areas = mesh.area_faces
    face_areas_prop = face_areas/total_area

    #c) multiply the surface area percentage by the midpoints
    mesh_center = np.sum(face_midpoints*face_areas_prop.reshape(-1,1),axis=0)
    return mesh_center
        

def write_neuron_off(current_mesh,main_mesh_path):
    if type(main_mesh_path) != str:
        main_mesh_path = str(main_mesh_path.absolute())
    if main_mesh_path[-4:] != ".off":
        main_mesh_path += ".off"
    current_mesh.export(main_mesh_path)
    with open(main_mesh_path,"a") as f:
        f.write("\n")
    return main_mesh_path


def combine_meshes(mesh_pieces):
    leftover_mesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))
    for m in mesh_pieces:
        leftover_mesh += m
    return leftover_mesh

"""
def bbox_mesh_restriction(curr_mesh,bbox_upper_corners,
                         mult_ratio = 1):
    bbox_center = np.mean(bbox_upper_corners,axis=0)
    bbox_distance = np.max(bbox_upper_corners,axis=0)-bbox_center
    
    #face_midpoints = np.mean(curr_mesh.vertices[curr_mesh.faces],axis=1)
    face_midpoints = curr_mesh.triangles_center
    
    sum_totals = np.invert(np.sum((np.abs(face_midpoints-bbox_center)-mult_ratio*bbox_distance) > 0,axis=1).astype("bool").reshape(-1))
    #total_face_indexes = set(np.arange(0,len(sum_totals)))
    faces_bbox_inclusion = (np.arange(0,len(sum_totals)))[sum_totals]
    
    try:
        curr_mesh_bbox_restriction = curr_mesh.submesh([faces_bbox_inclusion],append=True)
        return curr_mesh_bbox_restriction,faces_bbox_inclusion
    except:
        #print(f"faces_bbox_inclusion = {faces_bbox_inclusion}")
        #print(f"curr_mesh = {curr_mesh}")
        #raise Exception("failed bbox_mesh")
        return curr_mesh,np.arange(0,len(curr_mesh.faces))
    
"""


# New bounding box method able to accept multiple
def bbox_mesh_restriction(curr_mesh,bbox_upper_corners,
                         mult_ratio = 1):
    """
    Purpose: Can send multiple bounding box corners to the function
    and it will restrict your mesh to only the faces that are within
    those bounding boxs
    ** currently doing bounding boxes that are axis aligned
    
    -- Future work --
    could get an oriented bounding box by doing
    
    elephant_skeleton_verts_mesh = trimesh.Trimesh(vertices=el_verts,faces=np.array([]))
    elephant_skeleton_verts_mesh.bounding_box_oriented 
    
    but would then have to do a projection into the oriented bounding box
    plane to get all of the points contained within
    
    
    """
    
    
    
    if type(bbox_upper_corners) != list:
        bbox_upper_corners = [bbox_upper_corners]
    
    sum_totals_list = []
    for bb_corners in bbox_upper_corners:
    
        bbox_center = np.mean(bb_corners,axis=0)
        bbox_distance = np.max(bb_corners,axis=0)-bbox_center

        #face_midpoints = np.mean(curr_mesh.vertices[curr_mesh.faces],axis=1)
        face_midpoints = curr_mesh.triangles_center

        current_sums = np.invert(np.sum((np.abs(face_midpoints-bbox_center)-mult_ratio*bbox_distance) > 0,axis=1).astype("bool").reshape(-1))
        sum_totals_list.append(current_sums)
    
    sum_totals = np.logical_or.reduce(sum_totals_list)
    #print(f"sum_totals = {sum_totals}")
    
    faces_bbox_inclusion = (np.arange(0,len(sum_totals)))[sum_totals]
    
    try:
        curr_mesh_bbox_restriction = curr_mesh.submesh([faces_bbox_inclusion],append=True)
        return curr_mesh_bbox_restriction,faces_bbox_inclusion
    except:
        #print(f"faces_bbox_inclusion = {faces_bbox_inclusion}")
        #print(f"curr_mesh = {curr_mesh}")
        #raise Exception("failed bbox_mesh")
        return curr_mesh,np.arange(0,len(curr_mesh.faces))
    

    

# main mesh cancellation

def split_significant_pieces(new_submesh,
                            significance_threshold=100,
                            print_flag=False,
                            return_insignificant_pieces=False):
    
    if type(new_submesh) != type(trimesh.Trimesh()):
        print("Inside split_significant_pieces and was passed empty mesh so retruning empty list")
        return []
    
    if print_flag:
        print("------Starting the mesh filter for significant outside pieces-------")

    mesh_pieces = new_submesh.split(only_watertight=False)
    if type(mesh_pieces) not in [type(np.ndarray([])),type(np.array([])),list]:
        mesh_pieces = [mesh_pieces]
    
    if print_flag:
        print(f"There were {len(mesh_pieces)} pieces after mesh split")

    significant_pieces = [m for m in mesh_pieces if len(m.faces) >= significance_threshold]
    if return_insignificant_pieces:
        insignificant_pieces = [m for m in mesh_pieces if len(m.faces) < significance_threshold]

    if print_flag:
        print(f"There were {len(significant_pieces)} pieces found after size threshold")
    if len(significant_pieces) <=0:
        print("THERE WERE NO MESH PIECES GREATER THAN THE significance_threshold")
        return []
    
    #arrange the significant pieces from largest to smallest
    x = [len(k.vertices) for k in significant_pieces]
    sorted_indexes = sorted(range(len(x)), key=lambda k: x[k])
    sorted_indexes = sorted_indexes[::-1]
    sorted_significant_pieces = [significant_pieces[k] for k in sorted_indexes]
    
    if return_insignificant_pieces:
        #arrange the significant pieces from largest to smallest
        x = [len(k.vertices) for k in insignificant_pieces]
        sorted_indexes = sorted(range(len(x)), key=lambda k: x[k])
        sorted_indexes = sorted_indexes[::-1]
        sorted_significant_pieces_insig = [insignificant_pieces[k] for k in sorted_indexes]
    if return_insignificant_pieces:
        return sorted_significant_pieces,sorted_significant_pieces_insig
    else:
        return sorted_significant_pieces


    
from trimesh.graph import *
def split(mesh, only_watertight=False, adjacency=None, engine=None, return_components=True, **kwargs):
    """
    Split a mesh into multiple meshes from face
    connectivity.
    If only_watertight is true it will only return
    watertight meshes and will attempt to repair
    single triangle or quad holes.
    Parameters
    ----------
    mesh : trimesh.Trimesh
    only_watertight: bool
      Only return watertight components
    adjacency : (n, 2) int
      Face adjacency to override full mesh
    engine : str or None
      Which graph engine to use
    Returns
    ----------
    meshes : (m,) trimesh.Trimesh
      Results of splitting
      
    ----------------***** THIS VERSION HAS BEEN ALTERED TO PASS BACK THE COMPONENTS INDICES TOO ****------------------
    
    if return_components=True then will return an array of arrays that contain face indexes for all the submeshes split off
    Ex: 
    
    tu.split(elephant_and_box)
    meshes = array([<trimesh.Trimesh(vertices.shape=(2775, 3), faces.shape=(5558, 3))>,
        <trimesh.Trimesh(vertices.shape=(8, 3), faces.shape=(12, 3))>],
       dtype=object)
    components = array([array([   0, 3710, 3709, ..., 1848, 1847, 1855]),
        array([5567, 5566, 5565, 5564, 5563, 5559, 5561, 5560, 5558, 5568, 5562,
        5569])], dtype=object)
    
    """
    if adjacency is None:
        adjacency = mesh.face_adjacency

    # if only watertight the shortest thing we can split has 3 triangles
    if only_watertight:
        min_len = 4
    else:
        min_len = 1

    components = connected_components(
        edges=adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=min_len,
        engine=engine)
    meshes = mesh.submesh(
        components, only_watertight=only_watertight, **kwargs)
    
    """ 6 19, old way of doing checking that did not resolve anything
    if type(meshes) != type(np.array([])):
        print(f"meshes = {meshes}, with type = {type(meshes)}")
    """
        
    if type(meshes) != type(np.array([])) and type(meshes) != list:
        #print(f"meshes = {sub_components}, with type = {type(sub_components)}")
        if type(meshes) == type(trimesh.Trimesh()) :
            
            print("list was only one so surrounding them with list")
            #print(f"meshes_before = {meshes}")
            #print(f"components_before = {components}")
            meshes = [meshes]
            
        else:
            raise Exception("The sub_components were not an array, list or trimesh")
            
    #make sure they are in order from least to greatest size
    current_array = [len(c) for c in components]
    ordered_indices = np.flip(np.argsort(current_array))
    
    
    ordered_meshes = np.array([meshes[i] for i in ordered_indices])
    ordered_components = np.array([components[i] for i in ordered_indices])
    
    if len(ordered_meshes)>=2:
        if (len(ordered_meshes[0].faces) < len(ordered_meshes[1].faces)) and (len(ordered_meshes[0].vertices) < len(ordered_meshes[1].vertices)) :
            #print(f"ordered_meshes = {ordered_meshes}")
            raise Exception(f"Split is not passing back ordered faces:"
                            f" ordered_meshes = {ordered_meshes},  "
                           f"components= {components},  "
                           f"meshes = {meshes},  "
                            f"current_array={current_array},  "
                            f"ordered_indices={ordered_indices},  "
                           )
    
    #control if the meshes is iterable or not
    if return_components:
        return ordered_meshes,ordered_components
    else:
        return ordered_meshes

def closest_distance_between_meshes(original_mesh,submesh,print_flag=False):
    global_start = time.time()
    original_mesh_midpoints = original_mesh.triangles_center
    submesh_midpoints = submesh.triangles_center
    
    #1) Put the submesh face midpoints into a KDTree
    submesh_mesh_kdtree = KDTree(submesh_midpoints)
    #2) Query the fae midpoints of submesh against KDTree
    distances,closest_node = submesh_mesh_kdtree.query(original_mesh_midpoints)
    
    if print_flag:
        print(f"Total time for mesh distance: {time.time() - global_start}")
        
    return np.min(distances)

def compare_meshes_by_face_midpoints_list(mesh1_list,mesh2_list,**kwargs):
    match_list = []
    for mesh1,mesh2 in zip(mesh1_list,mesh2_list):
        match_list.append(compare_meshes_by_face_midpoints(mesh1,mesh2,**kwargs))
    
    return match_list

def compare_meshes_by_face_midpoints(mesh1,mesh2,match_threshold=0.001,print_flag=False):
    #0) calculate the face midpoints of each of the faces for original and submesh
    global_start = time.time()
    total_faces_greater_than_treshold = dict()
    starting_meshes = [mesh1,mesh2]
    for i in range(0,2):
        
        original_mesh_midpoints = starting_meshes[i].triangles_center
        submesh_midpoints = starting_meshes[np.abs(i-1)].triangles_center


        #1) Put the submesh face midpoints into a KDTree
        submesh_mesh_kdtree = KDTree(submesh_midpoints)
        #2) Query the fae midpoints of submesh against KDTree
        distances,closest_node = submesh_mesh_kdtree.query(original_mesh_midpoints)

        faces_greater_than_treshold = (np.arange(len(original_mesh_midpoints)))[distances >= match_threshold]
        total_faces_greater_than_treshold[i] = faces_greater_than_treshold
    
    if print_flag:
        print(f"Total time for mesh mapping: {time.time() - global_start}")
    
    
    if len(total_faces_greater_than_treshold[0])>0 or len(total_faces_greater_than_treshold[1])>0:
        if print_flag:
            print(f"{len(total_faces_greater_than_treshold[0])} face midpoints of mesh1 were farther than {match_threshold} "
                  f"from the face midpoints of mesh2")
            print(f"{len(total_faces_greater_than_treshold[1])} face midpoints of mesh2 were farther than {match_threshold} "
                  f"from the face midpoints of mesh1")
        return False
    else:
        if print_flag:
            print("Meshes are equal!")
        return True
    

def original_mesh_faces_map(original_mesh, submesh,
                           matching=True,
                           print_flag=False,
                           match_threshold = 0.001,
                           return_mesh=False):
    """
    PUrpose: Given a base mesh and mesh that was a submesh of that base mesh
    - find the original face indices of the submesh
    
    Pseudocode: 
    0) calculate the face midpoints of each of the faces for original and submesh
    1) Put the base mesh face midpoints into a KDTree
    2) Query the fae midpoints of submesh against KDTree
    3) Only keep those that correspond to the faces or do not correspond to the faces
    based on the parameter setting
    
    Can be inversed so can find the mapping of all the faces that not match a mesh
    """
    global_start = time.time()
    
    if type(original_mesh) != type(trimesh.Trimesh()):
        raise Exception("original mesh must be trimesh object")
    
    if type(submesh) != type(trimesh.Trimesh()):
        if not nu.non_empty_or_none(submesh):
            if matching:
                return_faces = np.array([])
                if return_mesh:
                    return trimesh.Trimesh(faces=np.array([]),
                                          vertices=np.array([]))
            else:
                return_faces = np.arange(0,len(original_mesh.faces))
                if return_mesh:
                    return original_mesh
                
            return return_faces
                
        else:
            submesh = combine_meshes(submesh)
    
    
    
    #0) calculate the face midpoints of each of the faces for original and submesh
    original_mesh_midpoints = original_mesh.triangles_center
    submesh_midpoints = submesh.triangles_center
    
    #1) Put the submesh face midpoints into a KDTree
    submesh_mesh_kdtree = KDTree(submesh_midpoints)
    #2) Query the fae midpoints of submesh against KDTree
    distances,closest_node = submesh_mesh_kdtree.query(original_mesh_midpoints)
    
    if print_flag:
        print(f"Total time for mesh mapping: {time.time() - global_start}")
    
    #3) Only keep those that correspond to the faces or do not correspond to the faces
    #based on the parameter setting
    if matching:
        return_faces = (np.arange(len(original_mesh_midpoints)))[distances < match_threshold]
        
    else:
        return_faces = (np.arange(len(original_mesh_midpoints)))[distances >= match_threshold]
        
    if return_mesh:
        return original_mesh.submesh([return_faces],append=True)
    else:
        return return_faces
    
    
def mesh_pieces_connectivity(
                main_mesh,
                central_piece,
                periphery_pieces,
                return_vertices=False,
                return_central_faces=False):
    """
    purpose: function that will determine if certain pieces of mesh are touching in reference
    to a central mesh

    Pseudocde: 
    1) Get the original faces of the central_piece and the periphery_pieces
    2) For each periphery piece, find if touching the central piece at all
    
    - get the vertices belonging to central mesh
    - get vertices belonging to current periphery
    - see if there is any overlap
    
    
    2a) If yes then add to the list to return
    2b) if no, don't add to list
    
    Example of How to use it: 
    
    connected_mesh_pieces = mesh_pieces_connectivity(
                    main_mesh=current_mesh,
                    central_piece=seperate_soma_meshes[0],
                    periphery_pieces = sig_non_soma_pieces)
    print(f"connected_mesh_pieces = {connected_mesh_pieces}")

    Application: For finding connectivity to the somas


    """
    
    """
    # 7-8 change: wanted to adapt so could give face ids as well instead of just meshes
    """
    #1) Get the original faces of the central_piece and the periphery_pieces
    if type(central_piece) == type(trimesh.Trimesh()):
        central_piece_faces = original_mesh_faces_map(main_mesh,central_piece)
    else:
        #then what was passed were the face ids
        central_piece_faces = central_piece.copy()
    
    periphery_pieces_faces = []
    #periphery_pieces_faces = [original_mesh_faces_map(main_mesh,k) for k in periphery_pieces]
    #print(f"periphery_pieces = {len(periphery_pieces)}")
    for k in periphery_pieces:
        if type(k) == type(trimesh.Trimesh()):
            #print("using trimesh pieces")
            periphery_pieces_faces.append(original_mesh_faces_map(main_mesh,k))
        else:
            #print("just using face idxs")
            periphery_pieces_faces.append(k)
    
    #2) For each periphery piece, find if touching the central piece at all
    touching_periphery_pieces = []
    touching_periphery_pieces_intersecting_vertices= []
    
    #the faces have the vertices indices stored so just comparing vertices indices!
    central_p_verts = np.unique(main_mesh.faces[central_piece_faces].ravel())
    
    for j,curr_p_faces in enumerate(periphery_pieces_faces):
        
        curr_p_verts = np.unique(main_mesh.faces[curr_p_faces].ravel())
        
        intersecting_vertices = np.intersect1d(central_p_verts,curr_p_verts)
        
        if len(np.intersect1d(central_p_verts,curr_p_verts)) > 0:
            touching_periphery_pieces.append(j)
            touching_periphery_pieces_intersecting_vertices.append(main_mesh.vertices[intersecting_vertices])
    
    
    
    if not return_vertices and not return_central_faces:
        return touching_periphery_pieces
    else:
        if return_vertices and return_central_faces:
            return touching_periphery_pieces,touching_periphery_pieces_intersecting_vertices,central_piece_faces
        elif return_vertices:
            return touching_periphery_pieces,touching_periphery_pieces_intersecting_vertices
        elif return_central_faces:
            touching_periphery_pieces,central_piece_faces
        else:
            raise Exception("Soething messed up with return in mesh connectivity")
            



def split_mesh_into_face_groups(base_mesh,face_mapping,return_idx=True,
                               check_connect_comp = True):
    """
    Will split a mesh according to a face coloring of labels to split into 
    """
    if type(face_mapping) == dict:
        sorted_dict = dict(sorted(face_mapping.items()))
        face_mapping = list(sorted_dict.values())
    
    if len(face_mapping) != len(base_mesh.faces):
        raise Exception("face mapping does not have same length as mesh faces")
    
    unique_labels = np.sort(np.unique(face_mapping))
    total_submeshes = dict()
    total_submeshes_idx = dict()
    for lab in tqdm(unique_labels):
        faces = np.where(face_mapping==lab)[0]
        total_submeshes_idx[lab] = faces
        if not check_connect_comp:
            total_submeshes[lab] = base_mesh.submesh([faces],append=True,only_watertight=False)
        else: 
            curr_submeshes = base_mesh.submesh([faces],append=False,only_watertight=False)
            #print(f"len(curr_submeshes) = {len(curr_submeshes)}")
            if len(curr_submeshes) == 1:
                total_submeshes[lab] = curr_submeshes[0]
            else:
                raise Exception(f"Label {lab} has {len(curr_submeshes)} disconnected submeshes"
                                "\n(usually when checking after the waterfilling algorithm)")
    if return_idx:
        return total_submeshes,total_submeshes_idx
    else:
        return total_submeshes




"""    
An algorithm that could be used to find sdf values    
    

import numpy as np
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

from mesh_to_sdf import get_surface_point_cloud, scale_to_unit_sphere
import trimesh
import skimage, skimage.measure
import os



mesh = current_mesh
mesh = scale_to_unit_sphere(mesh)

print("Scanning...")
cloud = get_surface_point_cloud(mesh, surface_point_method='scan', scan_count=20, scan_resolution=400)

cloud.show()

os.makedirs("test", exist_ok=True)
for i, scan in enumerate(cloud.scans):
    scan.save("test/scan_{:d}.png".format(i))

print("Voxelizing...")
voxels = cloud.get_voxels(128, use_depth_buffer=True)

print("Creating a mesh using Marching Cubes...")
vertices, faces, normals, _ = skimage.measure.marching_cubes_lewiner(voxels, level=0)
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
mesh.show()"""
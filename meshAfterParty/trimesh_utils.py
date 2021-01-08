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
    """
    will load a mesh from .off file format
    """
    if type(current_mesh_file) == type(Path()):
        current_mesh_file = str(current_mesh_file.absolute())
    if current_mesh_file[-4:] != ".off":
        current_mesh_file += ".off"
    return trimesh.load_mesh(current_mesh_file,process=False)

# --------- Dealing with h5 files
import h5py
def load_mesh_no_processing_h5(current_mesh_file):
    """
    Will load a mesh from h5py file format
    
    """
    if type(current_mesh_file) == type(Path()):
        current_mesh_file = str(current_mesh_file.absolute())
    if current_mesh_file[-3:] != ".h5":
        current_mesh_file += ".h5"
        
    with h5py.File(current_mesh_file, 'r') as hf:
        vertices = hf['vertices'][()].astype(np.float64)
        faces = hf['faces'][()].reshape(-1, 3).astype(np.uint32)
        
    return trimesh.Trimesh(vertices=vertices,faces=faces)

def write_h5_file(mesh=None,vertices=None,faces=None,segment_id=12345,
                  filepath="./",
                 filename=None,
                 return_file_path=True):
    """
    Purpose: Will write a h5 py file to store a mesh
    
    Pseudocode:
    1) Extract the vertices and the faces
    2) Create the complete file path with the write extension
    3) Write the .h5 file
    4) return the filepath 
    """
    
    #1) Extract the vertices and the faces
    if (vertices is None) or (faces is None):
        if mesh is None:
            raise Exception("mesh none and vertices or faces are none ")
        vertices=mesh.vertices
        faces=mesh.faces
        
    #2) Create the complete file path with the write extension
    curr_path = Path(filepath)
    
    assert curr_path.exists()
    
    if filename is None:
        filename = f"{segment_id}.h5"
    
    if str(filename)[-3:] != ".h5":
        filename = str(filename) + ".h5"
    
    total_path = str((curr_path / Path(filename)).absolute())
    
    with h5py.File(total_path, 'w') as hf:
        hf.create_dataset('segment_id', data=segment_id)
        hf.create_dataset('vertices', data=vertices)
        hf.create_dataset('faces', data=faces)
        
    if return_file_path:
        return total_path
    



# --------- Done with h5 files ---------------- #


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


def combine_meshes(mesh_pieces,merge_vertices=True):
    leftover_mesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))
    for m in mesh_pieces:
        leftover_mesh += m
        
    if merge_vertices:
        leftover_mesh.merge_vertices()
    
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
        curr_mesh_bbox_restriction = curr_mesh.submesh([faces_bbox_inclusion],append=True,repair=False)
        return curr_mesh_bbox_restriction,faces_bbox_inclusion
    except:
        #print(f"faces_bbox_inclusion = {faces_bbox_inclusion}")
        #print(f"curr_mesh = {curr_mesh}")
        #raise Exception("failed bbox_mesh")
        return curr_mesh,np.arange(0,len(curr_mesh.faces))
    


# -------------- 11/21 More bounding box functions ----- #
def bounding_box_corners(mesh,bbox_multiply_ratio=1):
    bbox_verts = mesh.bounding_box.vertices
    bb_corners = np.array([np.min(bbox_verts,axis=0),np.max(bbox_verts,axis=0)]).reshape(2,3)
    if bbox_multiply_ratio == 1:
        return bb_corners
    
    bbox_center = np.mean(bb_corners,axis=0)
    bbox_distance = np.max(bb_corners,axis=0)-bbox_center
    new_corners = np.array([bbox_center - bbox_multiply_ratio*bbox_distance,
                            bbox_center + bbox_multiply_ratio*bbox_distance
                           ]).reshape(-1,3)
    return new_corners
        

def check_meshes_outside_mesh_bbox(main_mesh,test_meshes,
                                  return_indices=False):
    return check_meshes_inside_mesh_bbox(main_mesh,test_meshes,
                                  return_indices=return_indices,
                                  return_inside=False)

def check_meshes_inside_mesh_bbox(main_mesh,test_meshes,
                                  return_indices=False,
                                  return_inside=True,
                                 bbox_multiply_ratio=1):
    """
    Purpose: Will check to see if any of the vertices
    of the test meshes are inside the bounding box of the main mesh
    
    Pseudocode: 
    1) Get the bounding box corners of the main mesh
    2) For each test mesh
    - send the vertices to see if inside bounding box
    - if any are then add indices to the running list
    
    3) Return either the meshes/indices of the inside/outside pieces
    based on the parameters set
    
    """
    #1) Get the bounding box corners of the main mesh
    main_mesh_bbox_corners = bounding_box_corners(main_mesh,bbox_multiply_ratio)
    
    #2) Iterate through test meshes
    inside_meshes_idx = []
    for j,tm in enumerate(test_meshes):
        inside_results = trimesh.bounds.contains(main_mesh_bbox_corners,tm.vertices.reshape(-1,3))
        if np.any(inside_results):
            inside_meshes_idx.append(j)
    
    #3) Set the return values
    if not return_inside:
        return_idx = np.setdiff1d(np.arange(len(test_meshes)),inside_meshes_idx)
    else:
        return_idx = np.array(inside_meshes_idx)
    
    if return_indices:
        return return_idx
    else:
        return [k for i,k in enumerate(test_meshes) if i in return_idx]
    
import numpy_utils as nu
import numpy as np
def check_meshes_outside_multiple_mesh_bbox(main_meshes,test_meshes,
                                  return_indices=False):
    return check_meshes_inside_multiple_mesh_bbox(main_meshes,test_meshes,
                                  return_indices=return_indices,
                                  return_inside=False)

def check_meshes_inside_multiple_mesh_bbox(main_meshes,test_meshes,
                                  return_indices=False,
                                  return_inside=True):
    """
    Purpose: will return all of the pieces inside or outside of 
    multiple seperate main mesh bounding boxes
    
    Pseudocode: 
    For each main mesh
    1) Run the check_meshes_inside_mesh_bbox and collect the resulting indexes
    2) Combine the results based on the following:
    - If outside, then do intersetion of results (becuase need to be outside of all)
    - if inside, then return union of results (because if inside at least one then should be considered inside)
    3) Return either the meshes or indices
    
    Ex: 
    import trimesh_utils as tu
    tu = reload(tu)
    tu.check_meshes_inside_multiple_mesh_bbox([soma_mesh,soma_mesh,soma_mesh],neuron_obj.non_soma_touching_meshes,
                                 return_indices=False)
    
    """
    if not nu.is_array_like(main_meshes):
        raise Exception("Was expecting a list of main meshes")
    
    #1) Run the check_meshes_inside_mesh_bbox and collect the resulting indexes
    
    all_results = []
    for main_mesh in main_meshes:
        curr_results = check_meshes_inside_mesh_bbox(main_mesh,test_meshes,
                                  return_indices=True,
                                  return_inside=return_inside)
        
        all_results.append(curr_results)
    
    #2) Combine the results based on the following:
    if return_inside:
        joining_function = np.union1d
    else:
        joining_function = np.intersect1d
    
    final_indices = all_results[0]
    
    for i in range(1,len(all_results)):
        final_indices = joining_function(final_indices,all_results[i])
    
    #3) Return either the meshes or indices
    if return_indices:
        return final_indices
    else:
        return [k for i,k in enumerate(test_meshes) if i in final_indices]

    
    

# main mesh cancellation
import numpy as  np
import system_utils as su
# --------------- 12/3 Addition: Made the connectivity matrix from the vertices by default ------------- #
def split_significant_pieces(new_submesh,
                            significance_threshold=100,
                            print_flag=False,
                            return_insignificant_pieces=False,
                            connectivity="vertices"):
    
    if type(new_submesh) != type(trimesh.Trimesh()):
        print("Inside split_significant_pieces and was passed empty mesh so retruning empty list")
        if return_insignificant_pieces:
            return [],[]
        else:
            return []
    
    if print_flag:
        print("------Starting the mesh filter for significant outside pieces-------")
#     import system_utils as su
#     su.compressed_pickle(new_submesh,f"new_submesh_{np.random.randint(10,1000)}")
    if connectivity=="edges":
        mesh_pieces = new_submesh.split(only_watertight=False,repair=False)
    else:
        mesh_pieces = split_by_vertices(new_submesh,return_components=False)
        
    if print_flag:
        print(f"Finished splitting mesh_pieces into = {mesh_pieces}")
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
        if return_insignificant_pieces:
            return [],[]
        else:
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


"""
******* 
The submesh function if doesn't have repair = False might
end up adding on some faces that you don't want!
*******
"""

def sort_meshes_largest_to_smallest(meshes,
                                    sort_attribute="faces",
                                    return_idx=False):
    x = [len(getattr(k,sort_attribute)) for k in meshes]
    sorted_indexes = sorted(range(len(x)), key=lambda k: x[k])
    sorted_indexes = sorted_indexes[::-1]
    sorted_meshes = [meshes[k] for k in sorted_indexes]
    if return_idx:
        return sorted_meshes,sorted_indexes
    else:
        return sorted_meshes
    
    
from trimesh.graph import *
def split(mesh, only_watertight=False, adjacency=None, engine=None, return_components=True, connectivity="vertices", **kwargs):
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
    if connectivity == "vertices":
        return split_by_vertices(mesh,return_components=return_components)
    
    if adjacency is None:
        adjacency = mesh.face_adjacency

    # if only watertight the shortest thing we can split has 3 triangles
    if only_watertight:
        min_len = 4
    else:
        min_len = 1
        
    #print(f"only_watertight = {only_watertight}")

    components = connected_components(
        edges=adjacency,
        nodes=np.arange(len(mesh.faces)),
        min_len=min_len,
        engine=engine)
    
    #print(f"components = {[c.shape for c in components]}")
    meshes = mesh.submesh(
        components, only_watertight=only_watertight, repair=False, **kwargs)
    #print(f"meshes = {meshes}")
    
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
#     current_array = [len(c) for c in components]
#     ordered_indices = np.flip(np.argsort(current_array))
    
    # order according to number of faces in meshes (SO DOESN'T ERROR ANYMORE)
    current_array = [len(c.faces) for c in meshes]
    ordered_indices = np.flip(np.argsort(current_array))
    
    
    ordered_meshes = np.array([meshes[i] for i in ordered_indices])
    ordered_components = np.array([components[i] for i in ordered_indices],dtype="object")
    
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
    try:
        ordered_comp_indices = np.array([k.astype("int") for k in ordered_components])
    except:
        import system_utils as su
        su.compressed_pickle(ordered_components,"ordered_components")
        print(f"ordered_components = {ordered_components}")
        raise Exception("ordered_components")
    
    if return_components:
        return ordered_meshes,ordered_comp_indices
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
    debug = False
    global_start = time.time()
    total_faces_greater_than_threshold = dict()
    starting_meshes = [mesh1,mesh2]
    if debug:
        print(f"mesh1.faces.shape = {mesh1.faces.shape},mesh2.faces.shape = {mesh2.faces.shape}")
    for i in range(0,len(starting_meshes)):
        
        original_mesh_midpoints = starting_meshes[i].triangles_center
        submesh_midpoints = starting_meshes[np.abs(i-1)].triangles_center


        #1) Put the submesh face midpoints into a KDTree
        submesh_mesh_kdtree = KDTree(submesh_midpoints)
        #2) Query the fae midpoints of submesh against KDTree
        distances,closest_node = submesh_mesh_kdtree.query(original_mesh_midpoints)

        faces_greater_than_treshold = (np.arange(len(original_mesh_midpoints)))[distances >= match_threshold]
        total_faces_greater_than_threshold[i] = faces_greater_than_treshold
    
    if print_flag:
        print(f"Total time for mesh mapping: {time.time() - global_start}")
    
    
    if len(total_faces_greater_than_threshold[0])>0 or len(total_faces_greater_than_threshold[1])>0:
        if print_flag:
            print(f"{len(total_faces_greater_than_threshold[0])} face midpoints of mesh1 were farther than {match_threshold} "
                  f"from the face midpoints of mesh2")
            
            print(f"{len(total_faces_greater_than_threshold[1])} face midpoints of mesh2 were farther than {match_threshold} "
                  f"from the face midpoints of mesh1")
        if debug:
            mesh1.export("mesh1_failed.off")
            mesh2.export("mesh2_failed.off")
        return False
    else:
        if print_flag:
            print("Meshes are equal!")
        return True
    
import time
def original_mesh_vertices_map(original_mesh, submesh=None,
                               vertices_coordinates=None,
                               matching=True,
                               match_threshold = 0.001,
                               print_flag=False):
    """
    Purpose: Given an original_mesh and either a 
        i) submesh
        ii) list of vertices coordinates
    Find the indices of the original vertices in the
    original mesh
    
    Pseudocode:
    1) Get vertices to map to original
    2) Construct a KDTree of the original mesh vertices
    3) query the closest vertices on the original mesh
    
    
    """
    
    if not submesh is None:
        vertices_coordinates = submesh.vertices
    elif vertices_coordinates is None:
        raise Exception("Both Submesh and vertices_coordinates are None")
    else:
        pass
    
    global_start = time.time()
    #1) Put the submesh face midpoints into a KDTree
    original_mesh_kdtree = KDTree(original_mesh.vertices)
    #2) Query the fae midpoints of submesh against KDTree
    distances,closest_node = original_mesh_kdtree.query(vertices_coordinates)

    #check that all of them matched below the threshold
    if np.any(distances> match_threshold):
        raise Exception(f"There were {np.sum(distances> match_threshold)} faces that did not have an exact match to the original mesh")

    if print_flag:
        print(f"Total time for mesh mapping: {time.time() - global_start}")

    return closest_node
    
def subtract_mesh(original_mesh,subtract_mesh,
                    return_mesh=True
                   ):
    
    if nu.is_array_like(subtract_mesh):
        subtract_mesh = combine_meshes(subtract_mesh)
        
    return original_mesh_faces_map(original_mesh=original_mesh,
                                   submesh=subtract_mesh,
                                   matching=False,
                                   return_mesh=return_mesh
                                  )

def restrict_mesh(original_mesh,restrict_meshes,
                    return_mesh=True
                   ):
    
    if nu.is_array_like(restrict_meshes):
        restrict_meshes = combine_meshes(restrict_meshes)
        
    return original_mesh_faces_map(original_mesh=original_mesh,
                                   submesh=restrict_meshes,
                                   matching=True,
                                   return_mesh=return_mesh
                                  )

def original_mesh_faces_map(original_mesh, submesh,
                           matching=True,
                           print_flag=False,
                           match_threshold = 0.001,
                            exact_match=False,
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
    
    #pre-check for emppty meshes
    if len(submesh.vertices) == 0 or len(submesh.faces) == 0:
        if matching:
            return np.array([])
        else:
            return np.arange(0,len(original_mesh.faces))
        
    
    
    #0) calculate the face midpoints of each of the faces for original and submesh
    original_mesh_midpoints = original_mesh.triangles_center
    submesh_midpoints = submesh.triangles_center
    
    if not exact_match:
        #This was the old way which was switching the order the new faces were found
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
    else:        
        #1) Put the submesh face midpoints into a KDTree
        original_mesh_kdtree = KDTree(original_mesh_midpoints)
        #2) Query the fae midpoints of submesh against KDTree
        distances,closest_node = original_mesh_kdtree.query(submesh_midpoints)
        
        #check that all of them matched below the threshold
        if np.any(distances> match_threshold):
            raise Exception(f"There were {np.sum(distances> match_threshold)} faces that did not have an exact match to the original mesh")
        
        if print_flag:
            print(f"Total time for mesh mapping: {time.time() - global_start}")
        
        if matching:
            return_faces = closest_node

    if return_mesh:
        return original_mesh.submesh([return_faces],append=True)
    else:
        return return_faces

def shared_edges_between_faces_on_mesh(mesh,faces_a,faces_b,
                                 return_vertices_idx=False):
    """
    Given two sets of faces, find the edges which are in both sets of faces.
    Parameters
    ---------
    faces_a : (n, 3) int
      Array of faces
    faces_b : (m, 3) int
      Array of faces
    Returns
    ---------
    shared : (p, 2) int
      Edges shared between faces
      
      
    Pseudocode:
    1) Get the unique edges of each of the faces
    """
    faces_a_edges = np.unique(mesh.faces_unique_edges[faces_a].ravel())
    faces_b_edges = np.unique(mesh.faces_unique_edges[faces_b].ravel())
    shared_edges_idx = np.intersect1d(faces_a_edges,faces_b_edges)
    
    if return_vertices_idx:
        return np.unique(mesh.edges_unique[shared_edges_idx].ravel())
    else:
        return shared_edges_idx
    
def mesh_pieces_connectivity(
                main_mesh,
                central_piece,
                periphery_pieces,
                connectivity="edges",
                return_vertices=False,
                return_central_faces=False,
                return_vertices_idx = False,
                print_flag=False,
                merge_vertices=False):
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


    Example: How to use merge vertices option
    import time

    start_time = time.time()

    #0) Getting the Soma border

    tu = reload(tu)
    new_mesh = tu.combine_meshes(touching_limbs_meshes + [curr_soma_mesh])

    soma_idx = 1
    curr_soma_mesh = current_neuron[nru.soma_label(soma_idx)].mesh
    touching_limbs = current_neuron.get_limbs_touching_soma(soma_idx)
    touching_limb_objs = [current_neuron[k] for k in touching_limbs]

    touching_limbs_meshes = [k.mesh for k in touching_limb_objs]
    touching_pieces,touching_vertices = tu.mesh_pieces_connectivity(main_mesh=new_mesh,
                                            central_piece = curr_soma_mesh,
                                            periphery_pieces = touching_limbs_meshes,
                                                             return_vertices=True,
                                                            return_central_faces=False,
                                                                    print_flag=False,
                                                                    merge_vertices=True,
                                                                                     )
    limb_to_soma_border = dict([(k,v) for k,v in zip(np.array(touching_limbs)[touching_pieces],touching_vertices)])
    limb_to_soma_border

    print(time.time() - start_time)

    """
    
    """
    # 7-8 change: wanted to adapt so could give face ids as well instead of just meshes
    """
    if merge_vertices:
        main_mesh.merge_vertices()
    
    #1) Get the original faces of the central_piece and the periphery_pieces
    if type(central_piece) == type(trimesh.Trimesh()):
        central_piece_faces = original_mesh_faces_map(main_mesh,central_piece)
    else:
        #then what was passed were the face ids
        central_piece_faces = central_piece.copy()
        
    if print_flag:
        print(f"central_piece_faces = {central_piece_faces}")
    
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
    
    if print_flag:
        print(f"periphery_pieces_faces = {periphery_pieces_faces}")
    
    #2) For each periphery piece, find if touching the central piece at all
    touching_periphery_pieces = []
    touching_periphery_pieces_intersecting_vertices= []
    touching_periphery_pieces_intersecting_vertices_idx = []
    
    #the faces have the vertices indices stored so just comparing vertices indices!
    
    if connectivity!="edges":
        central_p_verts = np.unique(main_mesh.faces[central_piece_faces].ravel())
    
    for j,curr_p_faces in enumerate(periphery_pieces_faces):
        
        if connectivity=="edges": #will do connectivity based on edges
            intersecting_vertices = shared_edges_between_faces_on_mesh(main_mesh,
                                                                       faces_a=central_piece_faces,
                                                                       faces_b=curr_p_faces,
                                                                       return_vertices_idx=True)
            
        else:
            curr_p_verts = np.unique(main_mesh.faces[curr_p_faces].ravel())
            intersecting_vertices = np.intersect1d(central_p_verts,curr_p_verts)
        
        if print_flag:
            print(f"intersecting_vertices = {intersecting_vertices}")
        
        if len(intersecting_vertices) > 0:
            touching_periphery_pieces.append(j)
            touching_periphery_pieces_intersecting_vertices.append(main_mesh.vertices[intersecting_vertices])
            touching_periphery_pieces_intersecting_vertices_idx.append(intersecting_vertices)
    
    
    
    #redoing the return structure
    return_value = [touching_periphery_pieces]
    if return_vertices:
        return_value.append(touching_periphery_pieces_intersecting_vertices)
    if return_central_faces:
        return_value.append(central_piece_faces)
    if return_vertices_idx:
        return_value.append(touching_periphery_pieces_intersecting_vertices_idx)
    
    if len(return_value) == 1:
        return return_value[0]
    else:
        return return_value
    
    
#     if not return_vertices and not return_central_faces:
#         return touching_periphery_pieces
#     else:
#         if return_vertices and return_central_faces:
#             return touching_periphery_pieces,touching_periphery_pieces_intersecting_vertices,central_piece_faces
#         elif return_vertices:
#             return touching_periphery_pieces,touching_periphery_pieces_intersecting_vertices
#         elif return_central_faces:
#             touching_periphery_pieces,central_piece_faces
#         else:
#             raise Exception("Soething messed up with return in mesh connectivity")
            


    
def mesh_list_connectivity(meshes,
                        main_mesh,
                           connectivity="edges",
                           min_common_vertices=1,
                           return_vertex_connection_groups=False,
                           return_largest_vertex_connection_group=False,
                           return_connected_components=False,
                        print_flag = False):
    """
    Pseudocode:
    1) Build an edge list
    2) Use the edgelist to find connected components

    Arguments:
    - meshes (list of trimesh.Trimesh) #
    - retrun_vertex_connection_groups (bool): whether to return the touching vertices


    """

    periphery_pieces = meshes
    meshes_connectivity_edge_list = []
    meshes_connectivity_vertex_connection_groups = []
    
    vertex_graph = None
    
        

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


    
        """
        Pseudocode:
        Iterates through all combinations of meshes
        1) get the faces of both meshes in the pair
        2) using the faces get the shared edges between them (if any)
        3) If there were shared edges then save them as intersecting vertices
        
        
        """
    
    for j,central_p_faces in enumerate(periphery_pieces_faces):
        if connectivity!="edges":
            central_p_verts = np.unique(main_mesh.faces[central_p_faces].ravel())
        
        for i in range(0,j):
            curr_p_faces = periphery_pieces_faces[i]
            if connectivity=="edges": #will do connectivity based on edges
                intersecting_vertices = shared_edges_between_faces_on_mesh(main_mesh,
                                                                           faces_a=central_p_faces,
                                                                           faces_b=curr_p_faces,
                                                                           return_vertices_idx=True)
                
            else: #then do the vertex way
                
                curr_p_verts = np.unique(main_mesh.faces[curr_p_faces].ravel())

                intersecting_vertices = np.intersect1d(central_p_verts,curr_p_verts)
                
            if print_flag:
                print(f"intersecting_vertices = {intersecting_vertices}")


            if len(intersecting_vertices) >= min_common_vertices:
                if return_vertex_connection_groups:

                    if vertex_graph is None:
                        vertex_graph = mesh_vertex_graph(main_mesh)

                    curr_vertex_connection_groups = split_vertex_list_into_connected_components(
                                                    vertex_indices_list=intersecting_vertices,
                                                    mesh=main_mesh,
                                                    vertex_graph=vertex_graph,
                                                    return_coordinates=True,
                                                   )
                    if return_largest_vertex_connection_group:
                        curr_vertex_connection_groups_len = [len(k) for k in curr_vertex_connection_groups]
                        largest_group = np.argmax(curr_vertex_connection_groups_len)
                        curr_vertex_connection_groups = curr_vertex_connection_groups[largest_group]

                    meshes_connectivity_vertex_connection_groups.append(curr_vertex_connection_groups)

                meshes_connectivity_edge_list.append((j,i))

    meshes_connectivity_edge_list = nu.sort_elements_in_every_row(meshes_connectivity_edge_list)
    if return_vertex_connection_groups:
        return meshes_connectivity_edge_list,meshes_connectivity_vertex_connection_groups
    elif return_connected_components:
        return xu.connected_components_from_nodes_edges(np.arange(len(meshes)),meshes_connectivity_edge_list)
    else:
        return meshes_connectivity_edge_list

'''
Saved method before added in vertex options


def mesh_list_connectivity(meshes,
                        main_mesh,
                           min_common_vertices=1,
                           return_vertex_connection_groups=False,
                           return_largest_vertex_connection_group=False,
                        print_flag = False):
    """
    Pseudocode:
    1) Build an edge list
    2) Use the edgelist to find connected components

    Arguments:
    - meshes (list of trimesh.Trimesh) #
    - retrun_vertex_connection_groups (bool): whether to return the touching vertices


    """

    periphery_pieces = meshes
    meshes_connectivity_edge_list = []
    meshes_connectivity_vertex_connection_groups = []
    
    vertex_graph = None
    
        

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


    for j,central_p_faces in enumerate(periphery_pieces_faces):
        central_p_verts = np.unique(main_mesh.faces[central_p_faces].ravel())
        for i in range(0,j):
            curr_p_faces = periphery_pieces_faces[i]
            curr_p_verts = np.unique(main_mesh.faces[curr_p_faces].ravel())

            intersecting_vertices = np.intersect1d(central_p_verts,curr_p_verts)
            if print_flag:
                print(f"intersecting_vertices = {intersecting_vertices}")
            
            
            

            if len(intersecting_vertices) >= min_common_vertices:
                if return_vertex_connection_groups:
                    
                    if vertex_graph is None:
                        vertex_graph = mesh_vertex_graph(main_mesh)
                    
                    curr_vertex_connection_groups = split_vertex_list_into_connected_components(
                                                    vertex_indices_list=intersecting_vertices,
                                                    mesh=main_mesh,
                                                    vertex_graph=vertex_graph,
                                                    return_coordinates=True,
                                                   )
                    if return_largest_vertex_connection_group:
                        curr_vertex_connection_groups_len = [len(k) for k in curr_vertex_connection_groups]
                        largest_group = np.argmax(curr_vertex_connection_groups_len)
                        curr_vertex_connection_groups = curr_vertex_connection_groups[largest_group]
                
                    meshes_connectivity_vertex_connection_groups.append(curr_vertex_connection_groups)
                    
                meshes_connectivity_edge_list.append((j,i))

    meshes_connectivity_edge_list = nu.sort_elements_in_every_row(meshes_connectivity_edge_list)
    if return_vertex_connection_groups:
        return meshes_connectivity_edge_list,meshes_connectivity_vertex_connection_groups
    else:
        return meshes_connectivity_edge_list
    






'''
    
    


def split_vertex_list_into_connected_components(
                                                vertex_indices_list, #list of vertices referencing the mesh
                                                mesh=None, #the main mesh the vertices list references
                                                vertex_graph=None, # a precomputed vertex graph if available
                                                return_coordinates=True, #whether to return the groupings as coordinates (if False the returns them as indices)
                                               ):
    """
    Purpose: 
    Given a list of vertices (in reference to a main mesh),
    returns the vertices divided into connected components on the graph
    
    Pseudocode:
    1) Build graph from vertex and edges of mesh
    2) Create a subgraph from the vertices list
    3) Find the connected components of the subgraph
    4) Either return the vertex coordinates or indices
    """
    if vertex_graph is None:
        #1) Build graph from vertex and edges of mesh
        if mesh is None:
            raise Exception("Neither the vertex graph or mesh argument were non None")
            
        vertex_graph = mesh_vertex_graph(mesh)
    
    #2) Create a subgraph from the vertices list
    vertex_subgraph = vertex_graph.subgraph(vertex_indices_list)
    
    vertex_groups = [np.array(list(k)).astype("int") for k in list(nx.connected_components(vertex_subgraph))]
    
    if return_coordinates:
        return [mesh.vertices[k] for k in vertex_groups]
    else:
        return vertex_groups


def split_mesh_into_face_groups(base_mesh,face_mapping,return_idx=True,
                               check_connect_comp = True,
                                return_dict=True):
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
            total_submeshes[lab] = base_mesh.submesh([faces],append=True,only_watertight=False,repair=False)
        else: 
            curr_submeshes = base_mesh.submesh([faces],append=False,only_watertight=False,repair=False)
            #print(f"len(curr_submeshes) = {len(curr_submeshes)}")
            if len(curr_submeshes) == 1:
                total_submeshes[lab] = curr_submeshes[0]
            else:
                raise Exception(f"Label {lab} has {len(curr_submeshes)} disconnected submeshes"
                                "\n(usually when checking after the waterfilling algorithm)")
    
    if not return_dict:
        total_submeshes = np.array(list(total_submeshes.values()))
        total_submeshes_idx =np.array(list(total_submeshes_idx.values()))
        
    if return_idx:
        return total_submeshes,total_submeshes_idx
    else:
        return total_submeshes
    
def split_mesh_by_closest_skeleton(mesh,skeletons,return_meshes=False):
    """
    Pseudocode: 
    For each N branch: 
    1) Build a KDTree of the skeleton
    2) query the mesh against the skeleton, get distances

    3) Concatenate all the distances and turn into (DxN) matrix 
    4) Find the argmin of each row and that is the assignment

    """


    dist_list = []
    for s in skeletons:
        sk_kd = KDTree(s.reshape(-1,3))
        dist, _ = sk_kd.query(mesh.triangles_center)
        dist_list.append(dist)

    dist_matrix = np.array(dist_list).T
    face_assignment = np.argmin(dist_matrix,axis=1)
    
    split_meshes_faces = [np.where(face_assignment == s_i)[0] for s_i in range(len(skeletons))]
    
    if return_meshes:
        split_meshes = [mesh.submesh([k],append=True,repair=False) for k in split_meshes_faces]
        return split_meshes,split_meshes_faces 
    else:
        return split_meshes_faces
        
#     split_meshes,split_meshes_faces = tu.split_mesh_into_face_groups(mesh,face_assignment,return_dict=False)
#     return split_meshes,split_meshes_faces 


 
"""
https://github.com/GPUOpen-LibrariesAndSDKs/RadeonProRenderUSD/issues/2


apt-get update
apt-get install -y wget

#explains why has to do this so can see the shared library: 
#https://stackoverflow.com/questions/1099981/why-cant-python-find-shared-objects-that-are-in-directories-in-sys-path
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc 
source ~/.bashrc



https://github.com/embree/embree#linux-and-macos (for the dependencies)

#for the dependencies
sudo apt-get install -y cmake-curses-gui
sudo apt-get install -y libtbb-dev
sudo apt-get install -y libglfw3-dev

Then run the following bash script (bash embree.bash)

trimesh bash file

---------------------------
set -xe
​
# Fetch the archive from GitHub releases.
wget https://github.com/embree/embree/releases/download/v2.17.7/embree-2.17.7.x86_64.linux.tar.gz -O /tmp/embree.tar.gz -nv
echo "2c4bdacd8f3c3480991b99e85b8f584975ac181373a75f3e9675bf7efae501fe  /tmp/embree.tar.gz" | sha256sum --check
tar -xzf /tmp/embree.tar.gz --strip-components=1 -C /usr/local
# remove archive
rm -rf /tmp/embree.tar.gz
​
# Install python bindings for embree (and upstream requirements).
pip3 install --no-cache-dir numpy cython
pip3 install --no-cache-dir https://github.com/scopatz/pyembree/releases/download/0.1.6/pyembree-0.1.6.tar.gz

-------------------------------





"""    
from trimesh.ray import ray_pyembree
def ray_trace_distance(mesh, 
                    face_inds=None, 
                   vertex_inds=None,
                   ray_origins=None,
                   ray_directions=None,
                   max_iter=10, 
                   rand_jitter=0.001, 
                   verbose=False,
                   ray_inter=None,
                      debug=False):
    """
    Purpose: To calculate the distance from a vertex or face
    midpoint to an intersecting side of the mesh
    - To help with width calculations
    
    Pseudocode: 
    1) Get the ray origins and directions
    2) Create a mask that tells which ray origins we have
    calculated a valid width for and an array to store the widths (start as -1)
    3) Start an iteration loop that will only stop
    when have a valid width for all origin points
        a. get the indices of which points we don't have valid sdfs for
        and restrict the run to only those 
        b. Add some gaussian noise to the normals of these rays
        c. Run the ray intersection to get the (multiple=False)
            - locations of intersections (mx3)
            - index_ray responsible for that intersection (m,)
            - mesh face that was intersected (m,)
        d. Update the width array for the origins that returned
           a valid width (using the diagonal_dot instead of linalg.norm because faster )
        e. Update the mask that shows which ray_origins have yet to be processed
    
    4) Return the width array
    
    
    """
    
    if not trimesh.ray.has_embree:
        logging.warning(
            "calculating rays without pyembree, conda install pyembree for large speedup")

    #initializing the obejct that can perform ray tracing
    if ray_inter is None:
        ray_inter = ray_pyembree.RayMeshIntersector(mesh)
    
    if not face_inds is None:
        ray_origins = mesh.triangles_center[face_inds]
        ray_directions = mesh.face_normals[face_inds]
    elif not vertex_inds is None:
        ray_origins = mesh.vertices[vertex_inds]
        ray_directions = mesh.vertex_normals[vertex_inds]
    elif (not ray_origins is None) and (not ray_directions is None):
        passs
    else:
        face_inds = np.arange(0,len(mesh.faces))
        ray_origins = mesh.triangles_center[face_inds]
        ray_directions = mesh.face_normals[face_inds]
        
    
    rs = np.zeros(len(ray_origins)) #array to hold the widths when calculated
    good_rs = np.full(len(rs), False) #mask that keeps track of how many widths have been calculated

    it = 0
    while not np.all(good_rs): #continue until all sdf widths are calculated
        if debug:
            print(f"\n --- Iteration = {it} -----")
        if debug:
            print(f"Number of non_good rs = {np.sum(~good_rs)}")
        
        #this is the indices of where the mask [~good_rs,:] is true
        blank_inds = np.where(~good_rs)[0] #the vertices who still need widths calculated
        
        #
        starts = ray_origins[blank_inds] - ray_directions[blank_inds]
        
        #gets the opposite of the vertex normal so is pointing inwards
        #then adds jitter that gets bigger and bigger
        ray_directions_with_jitter = -ray_directions[blank_inds] \
            + (1.2**it)*rand_jitter*np.random.rand(* #the * is to expand out the shape tuple
                                                   ray_directions[blank_inds].shape)
        
        #computes the locations, index_ray and index of hit mesh
        intersect_locations,intersect_ray_index,intersect_mesh_index = ray_inter.intersects_location(starts, ray_directions_with_jitter, multiple_hits=False)
        
        if debug:
            print(f"len(intersect_locations) = {len(intersect_locations)}")
            
        if len(intersect_locations) > 0:
            
            #rs[blank_inds[intersect_ray_index]] = np.linalg.norm(starts[intersect_ray_index]-intersect_locations,axis=1)
            depths = trimesh.util.diagonal_dot(intersect_locations - starts[intersect_ray_index],
                                      ray_directions_with_jitter[intersect_ray_index])
            if debug:
                print(f"Number of dephts that are 0 = {len(np.where(depths == 0)[0])}")
            rs[blank_inds[intersect_ray_index]] = depths
            
            if debug:
                print(f"Number of rs == 0: {len(np.where(rs==0)[0]) }")
                print(f"np.sum(~good_rs) BEFORE UPDATE= {np.sum(~good_rs) }")
                if len(depths)<400:
                    print(f"depths = {depths}")
                    print(f"blank_inds[intersect_ray_index] = {blank_inds[intersect_ray_index]}")
                    print(f"np.where(rs==0)[0] = {np.where(rs==0)[0]}")
            good_rs[blank_inds[intersect_ray_index]] = True
            if debug:
                print(f"np.sum(~good_rs) AFTER UPDATE = {np.sum(~good_rs) }")
            
        if debug: 
            print(f"np.all(good_rs) = {np.all(good_rs)}")
        it += 1
        if it > max_iter:
            if verbose:
                print(f"hit max iterations {max_iter}")
            break
    return rs
        
def vertices_coordinates_to_vertex_index(mesh,vertex_coordinates,error_on_unmatches=True):
    """
    Purpose: To map the vertex coordinates to vertex indices
    
    """
    m_kd = KDTree(mesh.vertices)
    dist,closest_vertex = m_kd.query(vertex_coordinates)
    zero_dist = np.where(dist == 0)[0]
    if error_on_unmatches:
        mismatch_number = len(vertex_coordinates)-len(zero_dist)
        if mismatch_number > 0:
            raise Exception(f"{mismatch_number} of the vertices coordinates were not a perfect match")
    
    return closest_vertex[zero_dist]

def vertices_coordinates_to_faces(mesh,vertex_coordinates,error_on_unmatches=False,concatenate_unique_list=True):
    vertex_indices = vertices_coordinates_to_vertex_index(mesh,vertex_coordinates,error_on_unmatches)
    return vertices_to_faces(mesh,vertex_indices,concatenate_unique_list)

def vertices_to_faces(current_mesh,vertices,
                     concatenate_unique_list=False):
    """
    Purpose: If have a list of vertex indices, to get the face indices associated with them
    """
    try:
        vertices = np.array(vertices)

        intermediate_face_list = current_mesh.vertex_faces[vertices]
        faces_list = [k[k!=-1] for k in intermediate_face_list]
        if concatenate_unique_list:
            return np.unique(np.concatenate(faces_list))
        else:
            return faces_list
    except:
        print(f"vertices = {vertices}")
        su.compressed_pickle(current_mesh,"current_mesh_error_v_to_f")
        su.compressed_pickle(vertices,"vertices_error_v_to_f")
        raise Exception("Something went wrong in vertices to faces")

import numpy_utils as nu
import system_utils as su
def vertices_coordinates_to_faces_old(current_mesh,vertex_coordinates):
    """
    
    Purpose: If have a list of vertex coordinates, to get the face indices associated with them
    
    Example: To check that it worked well with picking out border
    sk.graph_skeleton_and_mesh(other_meshes=[curr_branch.mesh,curr_branch.mesh.submesh([unique_border_faces],append=True)],
                              other_meshes_colors=["red","black"],
                              mesh_alpha=1)

    
    
    """
    try:
        border_vertices_idx = []
        for v in vertex_coordinates:
            curr_match_idx = nu.matching_rows(current_mesh.vertices,v)
            if len(curr_match_idx) > 0:
                border_vertices_idx.append(curr_match_idx)
        border_vertices_idx = np.array(border_vertices_idx)
    except:
        su.compressed_pickle(current_mesh,"current_mesh")
        su.compressed_pickle(vertex_coordinates,"vertex_coordinates")
        raise Exception("Something went from for matching_rows")
        
    border_faces = vertices_to_faces(current_mesh,vertices=border_vertices_idx)
    unique_border_faces = np.unique(np.concatenate(border_faces))
    return unique_border_faces


import networkx as nx
import networkx_utils as xu
def mesh_vertex_graph(mesh):
    """
    Purpose: Creates a weighted connectivity graph from the vertices and edges
    
    """
    curr_weighted_edges = np.hstack([mesh.edges_unique,mesh.edges_unique_length.reshape(-1,1)])
    vertex_graph = nx.Graph()  
    vertex_graph.add_weighted_edges_from(curr_weighted_edges)
    return vertex_graph

# ------------ Algorithms used for checking the spines -------- #

from trimesh.grouping import *
def waterfilling_face_idx(mesh,
                      starting_face_idx,
                      n_iterations=10,
                         return_submesh=False,
                         connectivity="vertices"):
    """
    Will extend certain faces by infecting neighbors 
    for a certain number of iterations:
    
    Example:
    curr_border_faces = tu.find_border_faces(curr_branch.mesh)
    expanded_border_mesh = tu.waterfilling_face_idx(curr_branch.mesh,
                                                    curr_border_faces,
                                                     n_iterations=10,
                                                    return_submesh=True)
    sk.graph_skeleton_and_mesh(other_meshes=[curr_branch.mesh,expanded_border_mesh],
                              other_meshes_colors=["black","red"])
    """
    #1) set the starting faces
    final_faces = starting_face_idx
    
    #0) Turn the mesh into a graph
    if connectivity=="edges":
        total_mesh_graph = nx.from_edgelist(mesh.face_adjacency)
        #2) expand the faces
        for i in range(n_iterations):
            final_faces = np.unique(np.concatenate([xu.get_neighbors(total_mesh_graph,k) for k in final_faces]))
    else:
        for i in range(n_iterations):
            final_faces = face_neighbors_by_vertices(mesh,final_faces)

    
    if return_submesh:
        return mesh.submesh([final_faces],append=True,repair=False)
    else:
        return final_faces
    
    
def find_border_vertices(mesh):
    if len(mesh.faces) < 3:
        return []

    if mesh.is_watertight:
        return []

    # we know that in a watertight mesh every edge will be included twice
    # thus every edge which appears only once is part of a hole boundary
    boundary_groups = group_rows(
        mesh.edges_sorted, require_count=1)

    return mesh.edges_sorted[boundary_groups].ravel()

def find_border_faces(mesh):
    border_verts = find_border_vertices(mesh)
    border_faces = np.unique(np.concatenate(vertices_to_faces(mesh,find_border_vertices(mesh).ravel())))
    return border_faces


def find_border_vertex_groups(mesh,return_coordinates=False,
                              return_cycles=False,
                             return_sizes=False):
    """
    Will return all borders as faces and grouped together
    """
    if len(mesh.faces) < 3:
        return []

    if mesh.is_watertight:
        return []

    # we know that in a watertight mesh every edge will be included twice
    # thus every edge which appears only once is part of a hole boundary
    boundary_groups = group_rows(
        mesh.edges_sorted, require_count=1)

    # mesh is not watertight and we have too few edges
    # edges to do a repair
    # since we haven't changed anything return False
    if len(boundary_groups) < 3:
        return []

    boundary_edges = mesh.edges[boundary_groups]
    index_as_dict = [{'index': i} for i in boundary_groups]

    # we create a graph of the boundary edges, and find cycles.
    g = nx.from_edgelist(
        np.column_stack((boundary_edges,
                         index_as_dict)))
    if return_cycles:
        border_edge_groups = xu.find_all_cycles(g,time_limit=20)
        if len(border_edge_groups)  == 0:
            print("Finding the cycles did not work when doing the border vertex edge so using connected components")
        border_edge_groups = list(nx.connected_components(g))
    else:
        border_edge_groups = list(nx.connected_components(g))

    """
    Psuedocode on converting list of edges to 
    list of faces

    """
    if return_coordinates:
        return_value = [mesh.vertices[list(k)] for k in border_edge_groups]
    else:
        return_value = [list(k) for k in border_edge_groups]
    
    if return_sizes:
        border_groups_len = np.array([len(k) for k in return_value])
        return return_value,border_groups_len
    else:
        return return_value
    
    

def find_border_face_groups(mesh,return_sizes=False):
    """
    Will return all borders as faces and grouped together
    """
    if len(mesh.faces) < 3:
        return []

    if mesh.is_watertight:
        return []

    # we know that in a watertight mesh every edge will be included twice
    # thus every edge which appears only once is part of a hole boundary
    boundary_groups = group_rows(
        mesh.edges_sorted, require_count=1)

    # mesh is not watertight and we have too few edges
    # edges to do a repair
    # since we haven't changed anything return False
    if len(boundary_groups) < 3:
        return []

    boundary_edges = mesh.edges[boundary_groups]
    index_as_dict = [{'index': i} for i in boundary_groups]

    # we create a graph of the boundary edges, and find cycles.
    g = nx.from_edgelist(
        np.column_stack((boundary_edges,
                         index_as_dict)))
    border_edge_groups = list(nx.connected_components(g))

    """
    Psuedocode on converting list of edges to 
    list of faces

    """
    border_face_groups = [vertices_to_faces(mesh,list(j),concatenate_unique_list=True) for j in border_edge_groups]
    if return_sizes:
        border_groups_len = np.array([len(k) for k in border_face_groups])
        return border_face_groups,border_groups_len
    else:
        return border_face_groups
    
def border_euclidean_length(border):
    """
    The border does have to be specified as ordered coordinates
    
    """
    ex_border_shift = np.roll(border,1,axis=0)
    return np.sum(np.linalg.norm(border - ex_border_shift,axis=1))

def largest_hole_length(mesh,euclidean_length=True):
    """
    Will find either the vertex count or the euclidean distance
    of the largest hole in a mesh
    
    """
    border_vert_groups,border_vert_sizes = find_border_vertex_groups(mesh,
                                return_coordinates=True,
                                 return_cycles=True,
                                return_sizes=True,
                                                                       )
    if euclidean_length:
        border_lengths = [border_euclidean_length(k) for k in border_vert_groups]
        largest_border_idx = np.argmax(border_lengths)
        largest_border_size = border_lengths[largest_border_idx]
        return largest_border_size
    else:
        return np.max(border_vert_sizes)

def expand_border_faces(mesh,n_iterations=10,return_submesh=True):
    curr_border_faces_groups = find_border_face_groups(mesh)
    expanded_border_face_groups = []
    for curr_border_faces in curr_border_faces_groups:
        expanded_border_mesh = waterfilling_face_idx(mesh,
                                                    curr_border_faces,
                                                     n_iterations=n_iterations,
                                                    return_submesh=return_submesh)
        expanded_border_face_groups.append(expanded_border_mesh)
    return expanded_border_face_groups
    
def mesh_with_ends_cutoff(mesh,n_iterations=5,
                         return_largest_mesh=True,
                         significance_threshold=100,
                         verbose=False):
    """
    Purpose: Will return a mesh with the ends with a border
    that are cut off by finding the border, expanding the border
    and then removing these faces and returning the largest piece
    
    Pseudocode:
    1) Expand he border meshes
    2) Get a submesh without the border faces
    3) Split the mesh into significants pieces
    3b) Error if did not find any significant meshes
    4) If return largest mesh is True, only return the top one
    
    """
    #1) Expand he border meshes
    curr_border_faces = expand_border_faces(mesh,n_iterations=n_iterations,return_submesh=False)
    
    #2) Get a submesh without the border faces
    if verbose:
        print(f"Removing {len(curr_border_faces)} border meshes of sizes: {[len(k) for k in curr_border_faces]} ")
    faces_to_keep = np.setdiff1d(np.arange(len(mesh.faces)),np.concatenate(curr_border_faces))
    leftover_submesh = mesh.submesh([faces_to_keep],append=True,repair=False)

    if verbose:
        printf("Leftover submesh size: {leftover_submesh}")
        
    #3) Split the mesh into significants pieces
    sig_leftover_pieces = split_significant_pieces(leftover_submesh,significance_threshold=significance_threshold)
    
    #3b) Error if did not find any significant meshes
    if len(sig_leftover_pieces) <= 0:
        raise Exception("No significant leftover pieces were detected after border subtraction")
        
    #4) If return largest mesh is True, only return the top one
    if return_largest_mesh:
        return sig_leftover_pieces[0]
    else:
        return sig_leftover_pieces
    
'''
# Old method that only computed percentage of total number of border vertices
from pykdtree.kdtree import KDTree
def filter_away_border_touching_submeshes(
                            mesh,
                            submesh_list,
                            border_percentage_threshold=0.5,#would make 0.00001 if wanted to enforce nullification if at most one touchedss
                            verbose = False,
                            return_meshes=True,
                            ):
    """
    Purpose: Will return submeshes or indices that 
    do not touch a border edge of the parenet mesh

    Pseudocode:
    1) Get the border vertices of mesh
    2) For each submesh
    - do KDTree between submesh vertices and border vertices
    - if one of distances is equal to 0 then nullify

    Ex: 
    
    return_value = filter_away_border_touching_submeshes(
                                mesh = eraser_branch.mesh,
                                submesh_list = eraser_branch.spines,
                                verbose = True,
                                return_meshes=True)
    sk.graph_skeleton_and_mesh(main_mesh_verts=mesh.vertices,
                           main_mesh_faces=mesh.faces,
                            other_meshes=eraser_branch.spines,
                                                  other_meshes_colors="red")
    sk.graph_skeleton_and_mesh(main_mesh_verts=mesh.vertices,
                           main_mesh_faces=mesh.faces,
                            other_meshes=return_value,
                                other_meshes_colors="red")
    """

    #1) Get the border vertices of mesh
    border_verts_idx = find_border_vertices(mesh)
    if len(border_verts_idx) == 0:
        if verbose:
            print("There were no border edges for the main mesh")
        passed_idx = np.arange(len(submesh_list))
    else:
        """
        Want to just find a matching border group and then look 
        at percentage
        """

        passed_idx = []
        for i,subm in enumerate(submesh_list):
            spine_kdtree = KDTree(subm.vertices)
            dist,closest_vert_idx = spine_kdtree.query(mesh.vertices[border_verts_idx])
            
            if len(dist[dist == 0])/len(border_verts_idx) < border_percentage_threshold:
                passed_idx.append(i)


        passed_idx = np.array(passed_idx)

    if return_meshes:
        return [k for i,k in enumerate(submesh_list) if i in passed_idx]
    else:
        return passed_idx
'''
from pykdtree.kdtree import KDTree
import numpy as np

def filter_away_border_touching_submeshes_by_group(
                            mesh,
                            submesh_list,
                            border_percentage_threshold=0.3,#would make 0.00001 if wanted to enforce nullification if at most one touchedss
                            inverse_border_percentage_threshold=0.9,
                            verbose = False,
                            return_meshes=True,
                    
                            ):
    """
    Purpose: Will return submeshes or indices that 
    do not touch a border edge of the parenet mesh

    Pseudocode:
    1) Get the border vertices of mesh grouped
    2) For each submesh
       a. Find which border group the vertices overlap with (0 distances)
       b. For each group that it is touching 
          i) Find the number of overlap
          ii) if the percentage is greater than threshold then nullify
    - 

    Ex: 
    
    return_value = filter_away_border_touching_submeshes(
                                mesh = eraser_branch.mesh,
                                submesh_list = eraser_branch.spines,
                                verbose = True,
                                return_meshes=True)
    sk.graph_skeleton_and_mesh(main_mesh_verts=mesh.vertices,
                           main_mesh_faces=mesh.faces,
                            other_meshes=eraser_branch.spines,
                                                  other_meshes_colors="red")
    sk.graph_skeleton_and_mesh(main_mesh_verts=mesh.vertices,
                           main_mesh_faces=mesh.faces,
                            other_meshes=return_value,
                                other_meshes_colors="red")
                                
    Ex 2:
    tu = reload(tu)
    tu.filter_away_border_touching_submeshes_by_group(
        mesh=curr_branch.mesh,
        submesh_list=curr_branch.spines
    )
    """
    
    if verbose:
        print(f"border_percentage_threshold = {border_percentage_threshold}")

    #1) Get the border vertices of mesh
    border_vertex_groups = find_border_vertex_groups(mesh)
    if len(border_vertex_groups) == 0:
        if verbose:
            print("There were no border edges for the main mesh")
        passed_idx = np.arange(len(submesh_list))
    else:
        """
        Want to just find a matching border group and then look 
        at percentage
        """

        passed_idx = []
        for i,subm in enumerate(submesh_list):
            #creates KDTree for the submesh
            spine_kdtree = KDTree(subm.vertices)
            
            not_touching_significant_border=True
            
            for z,b_verts in enumerate(border_vertex_groups):
                dist,closest_vert_idx = spine_kdtree.query(mesh.vertices[list(b_verts)])
                touching_perc = len(dist[dist == 0])/len(b_verts)
                if verbose:
                    print(f"Submesh {i} touching percentage for border {z} = {touching_perc}")
                if touching_perc > border_percentage_threshold:
                    if verbose:
                        print(f"Submesh {z} was touching a greater percentage ({touching_perc}) of border vertices than threshold ({border_percentage_threshold})")
                    not_touching_significant_border=False
                    break
            
            #apply the spine check that will see if percentage of border vertices of spine touching mesh border vertices
            #is above some threshold
            if inverse_border_percentage_threshold > 0:
                if verbose:
                    print(f"Applying inverse_border_percentage_threshold = {inverse_border_percentage_threshold}")
                    print(f"border_vertex_groups = {border_vertex_groups}")
                all_border_verts = np.concatenate([list(k) for k in border_vertex_groups])
                whole_border_kdtree= KDTree(mesh.vertices[all_border_verts])
                dist,closest_vert_idx = whole_border_kdtree.query(subm.vertices)
                touching_perc = len(dist[dist == 0])/len(dist)
                if touching_perc > inverse_border_percentage_threshold:
                    not_touching_significant_border = False
            
            if not_touching_significant_border:
                passed_idx.append(i)


        passed_idx = np.array(passed_idx)
        if verbose:
            print(f"At end passed_idx = {passed_idx} ")

    if return_meshes:
        return [k for i,k in enumerate(submesh_list) if i in passed_idx]
    else:
        return passed_idx

    
def max_distance_betwee_mesh_vertices(mesh_1,mesh_2,
                                      verbose=False,
                                     max_distance_threshold=None):
    """
    Purpose: Will calculate the maximum distance between vertices of two meshes
    
    Application: Can be used to see how well a poisson reconstruction
    estimate of a soma and the actual soma that was backtracked to 
    the mesh are in order to identify true somas and not
    get fooled by the glia / neural error checks
    
    Pseudocode:
    1) Make a KDTree from the new backtracked soma
    2) Do a query of the poisson soma vertices
    3) If a certain distance is too far then fail
    
    """
    
    #print(f"mesh_1={mesh_1},mesh_2 = {mesh_2}")
    #1) Make a KDTree from the new backtracked soma
    backtrack_mesh_kdtree = KDTree(mesh_1.vertices)
    #2) Do a query of the poisson soma vertices
    check_mesh_distances,closest_nodes = backtrack_mesh_kdtree.query(mesh_2.vertices)
    #print(f"check_mesh_distances = {check_mesh_distances}")
    max_dist = np.max(check_mesh_distances)
    
    if verbose:
        print(f"maximum distance from mesh_2 vertices to mesh_1 vertices is = {max_dist}")
    
    if max_distance_threshold is None:
        return max_dist
    else:
        if max_dist > max_distance_threshold:
            return False
        else:
            return True

import meshlab

def fill_holes(mesh,
              max_hole_size=2000,
              self_itersect_faces=False,
              ):
    
    mesh.merge_vertices()
    
    if tu.is_manifold(mesh):
        print("Mesh was manifold")
        if len(tu.find_border_face_groups(mesh))==0 and tu.is_manifold(mesh):
            print("No holes needed to fill and mesh was manifold so returning original mesh")
            return mesh
        
    lrg_mesh = mesh
    with meshlab.FillHoles(max_hole_size=max_hole_size,self_itersect_faces=self_itersect_faces) as fill_hole_obj:

        mesh_filled_holes,fillholes_file_obj = fill_hole_obj(   
                                            vertices=lrg_mesh.vertices,
                                             faces=lrg_mesh.faces,
                                             return_mesh=True,
                                             delete_temp_files=True,
                                            )
    return mesh_filled_holes

def filter_meshes_by_containing_coordinates(mesh_list,nullifying_points,
                                                filter_away=True,
                                           method="distance",
                                           distance_threshold=500,
                                           verbose=False,
                                           return_indices=False):
    """
    Purpose: Will either filter away or keep meshes from a list of meshes
    based on points based to the function
    
    Application: Can filter away spines that are too close to the endpoints of skeletons
    
    Ex: 
    import trimesh
    import numpy as np
    tu = reload(tu)

    curr_limb = recovered_neuron[2]
    curr_limb_end_coords = find_skeleton_endpoint_coordinates(curr_limb.skeleton)


    kept_spines = []

    for curr_branch in curr_limb:
        #a) get the spines
        curr_spines = curr_branch.spines

        #For each spine:
        if not curr_spines is None:
            curr_kept_spines = tu.filter_meshes_by_bbox_containing_coordinates(curr_spines,
                                                                            curr_limb_end_coords)
            print(f"curr_kept_spines = {curr_kept_spines}")
            kept_spines += curr_kept_spines

    nviz.plot_objects(meshes=kept_spines)
    """
    if not nu.is_array_like(mesh_list):
        mesh_list = [mesh_list]
        
    nullifying_points = np.array(nullifying_points).reshape(-1,3)
    
    containing_meshes = []
    containing_meshes_idx = []
    
    non_containing_meshes = []
    non_containing_meshes_idx = []
    for j,sp_m in enumerate(mesh_list):
        # tried filling hole and using contains
        #sp_m_filled = tu.fill_holes(sp_m)
        #contains_results = sp_m.bounds.contains(currc_limb_end_coords)

        #tried using the bounds method
        #contains_results = trimesh.bounds.contains(sp_m.bounds,currc_limb_end_coords.reshape(-1,3))

        #final version
        if method=="bounding_box":
            contains_results = sp_m.bounding_box_oriented.contains(nullifying_points.reshape(-1,3))
        elif method == "distance":
            sp_m_kdtree = KDTree(sp_m.vertices)
            distances,closest_nodes = sp_m_kdtree.query(nullifying_points.reshape(-1,3))
            contains_results = distances <= distance_threshold
            if verbose:
                print(f"Submesh {j} ({sp_m}) distances = {distances}")
                print(f"Min distance {np.min(distances)}")
                print(f"contains_results = {contains_results}\n")
        else:
            raise Exception(f"Unimplemented method ({method}) requested")
            
        if np.sum(contains_results) > 0:
            containing_meshes.append(sp_m)
            containing_meshes_idx.append(j)
            
        else:
            non_containing_meshes.append(sp_m)
            non_containing_meshes_idx.append(j)
    
    if filter_away:
        if return_indices:
            return non_containing_meshes_idx
        else:
            return non_containing_meshes
    else:
        if return_indices:
            return containing_meshes_idx
        else:
            return containing_meshes
    

# --------------- 11/11 ---------------------- #
import meshlab 
def poisson_surface_reconstruction(mesh,
                                   output_folder="./temp",
                                  delete_temp_files=True,
                                   name=None,
                                  verbose=False):
    if type(output_folder) != type(Path()):
        output_folder = Path(str(output_folder))
        output_folder.mkdir(parents=True,exist_ok=True)

    # CGAL Step 1: Do Poisson Surface Reconstruction
    Poisson_obj = meshlab.Poisson(output_folder,overwrite=True)

    if name is None:
        name = f"mesh_{np.random.randint(10,1000)}"

    
    skeleton_start = time.time()
    
    if verbose:
        print("     Starting Screened Poisson")
    new_mesh,output_subprocess_obj = Poisson_obj(   
                                vertices=mesh.vertices,
                                 faces=mesh.faces,
                                mesh_filename=name + ".off",
                                 return_mesh=True,
                                 delete_temp_files=delete_temp_files,
                                )
    if verbose:
        print(f"-----Time for Screened Poisson= {time.time()-skeleton_start}")
        
    return new_mesh


def decimate(mesh,
               decimation_ratio=0.25,
               output_folder="./temp",
              delete_temp_files=True,
               name=None,
              verbose=False):
    if type(output_folder) != type(Path()):
        output_folder = Path(str(output_folder))
        output_folder.mkdir(parents=True,exist_ok=True)

    # CGAL Step 1: Do Poisson Surface Reconstruction
    Decimator_obj = meshlab.Decimator(decimation_ratio,output_folder,overwrite=True)

    if name is None:
        name = f"mesh_{np.random.randint(10,1000)}"

    skeleton_start = time.time()
    
    if verbose:
        print("     Starting Screened Poisson")
    #Step 1: Decimate the Mesh and then split into the seperate pieces
    new_mesh,output_obj = Decimator_obj(vertices=mesh.vertices,
             faces=mesh.faces,
             segment_id=None,
             return_mesh=True,
             delete_temp_files=False)
    
    if verbose:
        print(f"-----Time for Screened Poisson= {time.time()-skeleton_start}")
        
    return new_mesh
        

import pymeshfix
import time

def pymeshfix_clean(mesh,
                    joincomp = True,
                   remove_smallest_components = False,
                   verbose=False):
    """
    Purpose: Will apply the pymeshfix algorithm
    to clean the mesh
    
    Application: Can help with soma identificaiton
    because otherwise nucleus could interfere with the segmentation
    
    
    """
    if verbose:
        print(f"Staring pymeshfix on {mesh}")
    start_time = time.time()

    meshfix = pymeshfix.MeshFix(mesh.vertices,mesh.faces)
    
    meshfix.repair(
                   verbose=False,
                   joincomp=joincomp,
                   remove_smallest_components=remove_smallest_components
                  )
    current_neuron_poisson_pymeshfix = trimesh.Trimesh(vertices=meshfix.v,faces=meshfix.f)

    if verbose:
        print(f"Total time for pymeshfix = {time.time() - start_time}")
    return current_neuron_poisson_pymeshfix


from pathlib import Path
import trimesh_utils as tu
import numpy as np
import time


import cgal_Segmentation_Module as csm
def mesh_segmentation(
        mesh = None,
        filepath = None,
        clusters=2,
        smoothness=0.2,
        cgal_folder = Path("./"),
        return_sdf = True,


        delete_temp_files = True,
        return_meshes = True ,
        check_connect_comp = True, #will only be used if returning meshes
        return_ordered_by_size = True,

        verbose = False,

    ):
    """
    Function tha segments the mesh and then 
    either returns:
    1) Face indexes of different mesh segments
    2) The cut up mesh into different mesh segments
    3) Can optionally return the sdf values of the different mesh

    Example: 
    tu = reload(tu)

    meshes_split,meshes_split_sdf = tu.mesh_segmentation(
        mesh = real_soma
    )
    
    """
    
    if not cgal_folder.exists():
        cgal_folder.mkdir(parents=True,exist_ok=False)

    mesh_temp_file = False
    if filepath is None:
        if mesh is None:
            raise Exception("Both mesh and filepath are None")
        file_dest = cgal_folder / Path(f"{np.random.randint(10,1000)}_mesh.off")
        filepath = write_neuron_off(mesh,file_dest)
        mesh_temp_file = True

    filepath = Path(filepath)

    assert(filepath.exists())
    filepath_no_ext = filepath.absolute().parents[0] / filepath.stem


    start_time = time.time()

    if verbose:
        print(f"Going to run cgal segmentation with:"
             f"\nFile: {str(filepath_no_ext)} \nclusters:{clusters} \nsmoothness:{smoothness}")

    csm.cgal_segmentation(str(filepath_no_ext),clusters,smoothness)

    #read in the csv file
    cgal_output_file = Path(str(filepath_no_ext) + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + ".csv" )
    cgal_output_file_sdf = Path(str(filepath_no_ext) + "-cgal_" + str(np.round(clusters,2)) + "_" + "{:.2f}".format(smoothness) + "_sdf.csv" )

    cgal_data = np.genfromtxt(str(cgal_output_file.absolute()), delimiter='\n')
    cgal_sdf_data = np.genfromtxt(str(cgal_output_file_sdf.absolute()), delimiter='\n')

    if return_meshes:
        if mesh is None:
            mesh = load_mesh_no_processing(filepath)
        split_meshes,split_meshes_idx = split_mesh_into_face_groups(mesh,cgal_data,return_idx=True,
                                       check_connect_comp = check_connect_comp,
                                                                      return_dict=False)

        if return_ordered_by_size:
            split_meshes,split_meshes_sort_idx = sort_meshes_largest_to_smallest(split_meshes,return_idx=True)

        if return_sdf:
            #will return sdf data for all of the meshes
            sdf_medains_for_mesh = np.array([np.median(cgal_sdf_data[k]) for k in split_meshes_idx])

            if return_ordered_by_size:
                sdf_medains_for_mesh = sdf_medains_for_mesh[split_meshes_sort_idx]
            return_value= split_meshes,sdf_medains_for_mesh
        else:
            return_value= split_meshes
    else:
        if return_sdf:
            return_value= cgal_data,cgal_sdf_data
        else:
            return_value= cgal_data

    if delete_temp_files:
        cgal_output_file.unlink()
        cgal_output_file_sdf.unlink()
        if mesh_temp_file:
            filepath.unlink()

    return return_value


"""Purpose: crude check to see if mesh is manifold:

https://gamedev.stackexchange.com/questions/61878/how-check-if-an-arbitrary-given-mesh-is-a-single-closed-mesh
"""

import trimesh
import open3d as o3d
def convert_trimesh_to_o3d(mesh):
    if not type(mesh) == type(o3d.geometry.TriangleMesh()):
        new_o3d_mesh = o3d.geometry.TriangleMesh()
        new_o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        new_o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    else:
        new_o3d_mesh = mesh
    return new_o3d_mesh

def convert_o3d_to_trimesh(mesh):
    if not type(mesh) == type(trimesh.Trimesh()):
        new_mesh = trimesh.Trimesh(
                                    vertices=np.asarray(mesh.vertices),
                                   faces=np.asarray(mesh.triangles)
                                  )
    else:
        new_mesh = mesh
    return new_mesh
    
def mesh_volume_o3d(mesh):
    mesh_o3d = convert_trimesh_to_o3d(mesh)
    return mesh_o3d.get_volume()
    
def is_manifold(mesh):
    mesh_o3d = convert_trimesh_to_o3d(mesh)  
    return mesh_o3d.is_vertex_manifold()

def is_watertight(mesh):
    return mesh.is_watertight

def get_non_manifold_edges(mesh):
    mesh_o3d = convert_trimesh_to_o3d(mesh)  
    return np.asarray(mesh_o3d.get_non_manifold_edges())

def get_non_manifold_vertices(mesh):
    mesh_o3d = convert_trimesh_to_o3d(mesh)  
    return np.asarray(mesh_o3d.get_non_manifold_vertices())


def mesh_interior(mesh,
                    return_interior=True,
                    quality_max=0.1,
                  try_hole_close=True,
                      max_hole_size = 10000,
                     self_itersect_faces=False,
                  verbose=True,
                    
                    **kwargs
              ):
    
    if try_hole_close:
        try:
            mesh = fill_holes(mesh,
                  max_hole_size=max_hole_size,
                  self_itersect_faces=self_itersect_faces)
        except:
            if verbose: 
                print("The hole closing did not work so continuing without")
            pass
                
    with meshlab.Interior(return_interior=return_interior,
                                quality_max=quality_max,
                                 **kwargs) as remove_obj:

        mesh_remove_interior,remove_file_obj = remove_obj(   
                                            vertices=mesh.vertices,
                                             faces=mesh.faces,
                                             return_mesh=True,
                                             delete_temp_files=True,
                                            )
    return mesh_remove_interior

def remove_mesh_interior(mesh,
                         size_threshold_to_remove=700,
                         verbose=True,
                         return_removed_pieces=False,
                         connectivity="vertices",
                         try_hole_close=True,
                         return_face_indices=False,
                         **kwargs):
    """
    Will remove interior faces of a mesh with a certain significant size
    
    """
    curr_interior_mesh = mesh_interior(mesh,return_interior=True,
                                       try_hole_close=try_hole_close,
                                       **kwargs)
    
    sig_inside = tu.split_significant_pieces(curr_interior_mesh,significance_threshold=size_threshold_to_remove,
                                            connectivity=connectivity)
    if len(sig_inside) == 0:
        sig_meshes_no_threshold = split_significant_pieces(curr_interior_mesh,significance_threshold=1)
        meshes_sizes = np.array([len(k.faces) for k in sig_meshes_no_threshold])
        if verbose:
            print(f"No significant ({size_threshold_to_remove}) interior meshes present")
            if len(meshes_sizes)>0:
                print(f"largest is {(np.max(meshes_sizes))}")
        if return_face_indices:
            return_mesh = np.arange(len(mesh.faces))
        else:
            return_mesh= mesh
    else:
        if verbose:
            print(f"Removing the following inside neurons: {sig_inside}")
        
        if return_face_indices:
            return_mesh= subtract_mesh(mesh,sig_inside,
                    return_mesh=False
                   )
        else:
            return_mesh= subtract_mesh(mesh,sig_inside)
        
        
    if return_removed_pieces:
        # --- 11/15: Need to only return inside pieces that are mapped to the original face ---
        sig_inside_remapped = [tu.original_mesh_faces_map(mesh,jj,
                                                          return_mesh=True) for jj in sig_inside]
        sig_inside_remapped = [k for k in sig_inside_remapped if len(k.faces) >= 1] 
        return return_mesh,sig_inside_remapped
    else:
        return return_mesh
    
    
def filter_vertices_by_mesh(mesh,vertices):
    """
    Purpose: To restrict the vertices to those
    that only lie on a mesh
    
    """
    
    #1) Build a KDTree of the mesh
    curr_mesh_tree = KDTree(mesh.vertices)
    
    #2) Query the vertices against the mesh
    dist,closest_nodes = curr_mesh_tree.query(vertices)
    match_verts = vertices[dist==0]
    
    return match_verts
    


import time
import copy
def fill_holes_trimesh(mesh):
    mesh_copy = copy.deepcopy(mesh)
    trimesh.repair.fill_holes(mesh_copy)
    return mesh_copy

import system_utils as su
def mesh_volume(mesh,watertight_method="trimesh",
                return_closed_mesh=False,
                zero_out_not_closed_meshes=True,
                poisson_obj=None,
                fill_holes_obj=None,
               verbose=False):
    """
    Purpose: To try and compute the volume of spines 
    with an optional argumet to try and close the mesh beforehand
    """
    start_time = time.time()
    if watertight_method is None:
        closed_mesh = mesh
    else:
        try: 
            if watertight_method == "trimesh":
                closed_mesh = fill_holes_trimesh(mesh)
                if not closed_mesh.is_watertight:
                    with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                        print("Trimesh closing holes did not work so using meshlab fill holes")
                        _, closed_mesh = mesh_volume(mesh=mesh,
                                                     watertight_method="fill_holes",
                                                     return_closed_mesh=True,
                                                     poisson_obj=poisson_obj,
                                                     fill_holes_obj=fill_holes_obj,
                                                     verbose=verbose)
            elif watertight_method == "poisson":
                if poisson_obj is None:
                    with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                        closed_mesh = poisson_surface_reconstruction(mesh)
                else:
                    with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                        print("Using premade object for poisson")
                        closed_mesh,output_subprocess_obj = poisson_obj(   
                                vertices=mesh.vertices,
                                 faces=mesh.faces,
                                 return_mesh=True,
                                 delete_temp_files=True,
                                )
                    
            elif watertight_method == "fill_holes":
                try:
                    if fill_holes_obj is None:
                        with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                            closed_mesh = fill_holes(mesh)
                    else:
                        with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                            print("Using premade object for fill holes")
                            closed_mesh,fillholes_file_obj = fill_holes_obj(   
                                            vertices=mesh.vertices,
                                             faces=mesh.faces,
                                             return_mesh=True,
                                             delete_temp_files=True,
                                            )
                except:
                    if verbose:
                        print("Filling holes did not work so using poisson reconstruction")
                    if poisson_obj is None:
                        with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                            closed_mesh = poisson_surface_reconstruction(mesh)
                    else:
                        with su.suppress_stdout_stderr() if (not verbose) else su.dummy_context_mgr():
                            print("Using premade object for poisson")
                            closed_mesh,output_subprocess_obj = Poisson_obj(   
                                    vertices=mesh.vertices,
                                     faces=mesh.faces,
                                     return_mesh=True,
                                     delete_temp_files=True,
                                    )
            else:
                raise Exception(f"The watertight method ({watertight_method}) is not one of implemented ones")
        except:
            print(f"The watertight method {watertight_method} could not run so not closing mesh")
            closed_mesh = mesh
            
    if verbose:
        print(f"Total time for mesh closing = {time.time() - start_time}")
        
    
    if not closed_mesh.is_watertight or closed_mesh.volume < 0:
        if zero_out_not_closed_meshes:
            final_volume = 0
        else:
            raise Exception(f"mesh {mesh} was not watertight ({mesh.is_watertight}) or volume is 0, vol = {closed_mesh.volume}")
    else:
        final_volume = closed_mesh.volume
    
    if return_closed_mesh:
        return final_volume,closed_mesh
    else:
        return final_volume
    

import networkx as nx
def vertex_components(mesh):
    return [list(k) for k in nx.connected_components(mesh.vertex_adjacency_graph)]

def components_to_submeshes(mesh,components,return_components=True,only_watertight=False,**kwargs):
    meshes = mesh.submesh(
        components, only_watertight=only_watertight, repair=False, **kwargs)
    

        
    if type(meshes) != type(np.array([])) and type(meshes) != list:
        #print(f"meshes = {sub_components}, with type = {type(sub_components)}")
        if type(meshes) == type(trimesh.Trimesh()) :
            
            print("list was only one so surrounding them with list")
            #print(f"meshes_before = {meshes}")
            #print(f"components_before = {components}")
            meshes = [meshes]
            
        else:
            raise Exception("The sub_components were not an array, list or trimesh")
            
    # order according to number of faces in meshes (SO DOESN'T ERROR ANYMORE)
    current_array = [len(c.faces) for c in meshes]
    ordered_indices = np.flip(np.argsort(current_array))
    
    
    ordered_meshes = np.array([meshes[i] for i in ordered_indices])
    ordered_components = np.array([components[i] for i in ordered_indices],dtype="object")
    
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
    try:
        ordered_comp_indices = np.array([k.astype("int") for k in ordered_components])
    except:
        import system_utils as su
        su.compressed_pickle(ordered_components,"ordered_components")
        print(f"ordered_components = {ordered_components}")
        raise Exception("ordered_components")
    
    if return_components:
        return ordered_meshes,ordered_comp_indices
    else:
        return ordered_meshes


def split_by_vertices(mesh,return_components=False,verbose=False):
    
    local_time = time.time()
    conn_verts = vertex_components(mesh)
    if verbose:
        print(f"for vertex components = {time.time() - local_time}")
        local_time = time.time()
    faces_per_component = [np.unique(np.concatenate(mesh.vertex_faces[k])) for k in conn_verts]
    if verbose:
        print(f"for faces_per_component = {time.time() - local_time}")
        local_time = time.time()
    
    faces_per_component = [k[k!=-1] for k in faces_per_component]
    if verbose:
        print(f"filtering faces_per_component = {time.time() - local_time}")
        local_time = time.time()
        
    ordered_meshes,ordered_comp_indices = components_to_submeshes(mesh,faces_per_component,return_components=True)
    if verbose:
        print(f"for components_to_submeshes = {time.time() - local_time}")
        local_time = time.time()
    
    if return_components:
        return ordered_meshes,ordered_comp_indices
    else:
        return ordered_meshes
    
    
import itertools
def mesh_face_graph_by_vertex(mesh):
    """
    Create a connectivity graph based on the faces that touch the same vertex have a connection edge
    
    """
    faces_adj_by_vertex = np.concatenate([np.array(list(itertools.combinations(k[k!=-1],2))) for k in mesh.vertex_faces if len(k[k!=-1])>1])
    if len(faces_adj_by_vertex) == 0:
        return nx.Graph()
    else:
        unique_edges = np.unique(faces_adj_by_vertex,axis=0)
        return nx.from_edgelist(unique_edges)
    
    
def find_closest_face_to_coordinates(mesh,coordinates,return_closest_distance=False,verbose=False):
    """
    Given a list of coordinates will find the closest
    face on a mesh
    
    """
    #2) get the closest point from the nodes to face centers of mesh
    mesh_kd = KDTree(mesh.triangles_center)
    dist,closest_faces = mesh_kd.query(coordinates)
    
    #3) Get the lowest distance
    closest_index = np.argmin(dist)
    min_distance = dist[closest_index]
    
    if verbose:
        print(f"Closest_distance = {min_distance}")
        
    if return_closest_distance:
        return closest_index,min_distance
    else:
        return closest_index
    
    
def face_neighbors_by_vertices(mesh,faces_list,
                              concatenate_unique_list=True):
    """
    Find the neighbors of face where neighbors are
    faces that touch the same vertices
    
    Pseudocode: 
    1) Change the faces to vertices
    2) Find all the faces associated with the vertices
    """
    if concatenate_unique_list:
        return vertices_to_faces(mesh,mesh.faces[faces_list].ravel(),concatenate_unique_list=concatenate_unique_list)
    else:
        return [vertices_to_faces(mesh,mesh.faces[k].ravel(),concatenate_unique_list=True) for k in faces_list]
    
    
def face_neighbors_by_vertices_seperate(mesh,faces_list):
    f_verts = mesh.faces[faces_list]
    return [np.unique(k[k!=-1]) for k in mesh.vertex_faces[f_verts]]

import compartment_utils as cu
def skeleton_to_mesh_correspondence(mesh,
                                    skeletons,
                                    remove_inside_pieces_threshold = 100,
                                    return_meshes=True,
                                    distance_by_mesh_center=True,
                                    connectivity="edges",
                                    verbose=False):
    """
    Purpose: To get the first pass mesh 
    correspondence of a skeleton or list of skeletons
    in reference to a mesh

    Pseudocode: 
    1) If requested, remove the interior of the mesh (if this is set then can't return indices)
    - if return indices is set then error if interior also set
    2) for each skeleton:
        a. Run the mesh correspondence adaptive function
        b. check to see if got any output (if did not then return empty list or empty mesh)
        c. If did add a submesh or indices to the return list


    Example:
    return_value = tu.skeleton_to_mesh_correspondence( mesh = debug_mesh,
                                                skeletons = viable_end_node_skeletons
                                   )

    nviz.plot_objects(meshes=return_value,
                      meshes_colors="random",
                      skeletons=viable_end_node_skeletons,
                     skeletons_colors="random")
    """

    if type(skeletons) != list:
        skeletons = [skeletons]

    return_indices = []

    if remove_inside_pieces_threshold > 0:
        curr_limb_mesh_indices = tu.remove_mesh_interior(mesh,
                                                 size_threshold_to_remove=remove_inside_pieces_threshold,
                                                 try_hole_close=False,
                                                 return_face_indices=True,
                                                )
        curr_limb_mesh_indices = np.array(curr_limb_mesh_indices)
        curr_mesh = mesh.submesh([curr_limb_mesh_indices],append=True,repair=False)
    else:
        curr_limb_mesh_indices = np.arange(len(mesh.faces))
        curr_mesh = mesh


    #1) Run the first pass mesh correspondence
    for curr_sk in skeletons:
        returned_data = cu.mesh_correspondence_adaptive_distance(curr_sk,
                                  curr_mesh,
                                 skeleton_segment_width = 1000,
                                 distance_by_mesh_center=distance_by_mesh_center,
                                                            connectivity=connectivity)

        if len(returned_data) == 0:
            return_indices.append([])
        else:

            curr_branch_face_correspondence, width_from_skeleton = returned_data
            return_indices.append(curr_limb_mesh_indices[curr_branch_face_correspondence])

    if return_meshes:
        return_value = []
        for ind in return_indices:
            if len(ind)>0:
                return_value.append(mesh.submesh([ind],append=True,repair=False))
            else:
                return_value.append(trimesh.Trimesh(vertices=np.array([]),
                                                   faces=np.array([])))
    else:
        return_value = return_indices

    if verbose:
        if not return_meshes:
            ret_val_sizes = [len(k) for k in return_value]
        else:
            ret_val_sizes = [len(k.faces) for k in return_value]

        print(f"Returned value sizes = {ret_val_sizes}")
        
    return return_value
    
    
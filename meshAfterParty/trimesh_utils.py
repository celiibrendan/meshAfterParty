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
        curr_mesh_bbox_restriction = curr_mesh.submesh([faces_bbox_inclusion],append=True,repair=False)
        return curr_mesh_bbox_restriction,faces_bbox_inclusion
    except:
        #print(f"faces_bbox_inclusion = {faces_bbox_inclusion}")
        #print(f"curr_mesh = {curr_mesh}")
        #raise Exception("failed bbox_mesh")
        return curr_mesh,np.arange(0,len(curr_mesh.faces))
    

    

# main mesh cancellation
import numpy as  np
import system_utils as su
def split_significant_pieces(new_submesh,
                            significance_threshold=100,
                            print_flag=False,
                            return_insignificant_pieces=False):
    
    if type(new_submesh) != type(trimesh.Trimesh()):
        print("Inside split_significant_pieces and was passed empty mesh so retruning empty list")
        return []
    
    if print_flag:
        print("------Starting the mesh filter for significant outside pieces-------")
#     import system_utils as su
#     su.compressed_pickle(new_submesh,f"new_submesh_{np.random.randint(10,1000)}")
    mesh_pieces = new_submesh.split(only_watertight=False,repair=False)
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
   
   
    
def mesh_pieces_connectivity(
                main_mesh,
                central_piece,
                periphery_pieces,
                return_vertices=False,
                return_central_faces=False,
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
    
    #the faces have the vertices indices stored so just comparing vertices indices!
    central_p_verts = np.unique(main_mesh.faces[central_piece_faces].ravel())
    
    for j,curr_p_faces in enumerate(periphery_pieces_faces):
        
        curr_p_verts = np.unique(main_mesh.faces[curr_p_faces].ravel())
        
        intersecting_vertices = np.intersect1d(central_p_verts,curr_p_verts)
        if print_flag:
            print(f"intersecting_vertices = {intersecting_vertices}")
        
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
            total_submeshes[lab] = base_mesh.submesh([faces],append=True,only_watertight=False,repair=False)
        else: 
            curr_submeshes = base_mesh.submesh([faces],append=False,only_watertight=False,repair=False)
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
        
    

def vertices_to_faces(current_mesh,vertices,
                     concatenate_unique_list=False):
    """
    Purpose: If have a list of vertex indices, to get the face indices associated with them
    """
    
    intermediate_face_list = current_mesh.vertex_faces[vertices]
    faces_list = [k[k!=-1] for k in intermediate_face_list]
    if concatenate_unique_list:
        return np.unique(np.concatenate(faces_list))
    else:
        return faces_list

import numpy_utils as nu
def vertices_coordinates_to_faces(current_mesh,vertex_coordinates):
    """
    
    Purpose: If have a list of vertex coordinates, to get the face indices associated with them
    
    Example: To check that it worked well with picking out border
    sk.graph_skeleton_and_mesh(other_meshes=[curr_branch.mesh,curr_branch.mesh.submesh([unique_border_faces],append=True)],
                              other_meshes_colors=["red","black"],
                              mesh_alpha=1)

    
    
    """
    border_vertices_idx = [nu.matching_rows(current_mesh.vertices,v)[0] for v in vertex_coordinates]
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
                         return_submesh=False):
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
    #0) Turn the mesh into a graph
    total_mesh_graph = nx.from_edgelist(mesh.face_adjacency)
    
    #1) set the starting faces
    final_faces = starting_face_idx
    
    #2) expand the faces
    for i in range(n_iterations):
        final_faces = np.unique(np.concatenate([xu.get_neighbors(total_mesh_graph,k) for k in final_faces]))
    
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


def find_border_vertex_groups(mesh):
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
    return border_edge_groups
    

def find_border_face_groups(mesh):
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
    return border_face_groups

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
                            border_percentage_threshold=0.5,#would make 0.00001 if wanted to enforce nullification if at most one touchedss
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
                if touching_perc > border_percentage_threshold:
                    if verbose:
                        print(f"Submesh {z} was touching a greater percentage ({touching_perc}) of border vertices than threshold ({border_percentage_threshold})")
                    not_touching_significant_border=False
                    break
            
            
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
              self_itersect_faces=False):
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
                                           verbose=False):
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
    non_containing_meshes = []
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
            contains_results = distances < distance_threshold
            if verbose:
                print(f"Submesh {j} ({sp_m}) distances = {distances}")
                print(f"Min distance {np.min(distances)}")
                print(f"contains_results = {contains_results}\n")
        else:
            raise Exception(f"Unimplemented method ({method}) requested")
            
        if np.sum(contains_results) > 0:
            containing_meshes.append(sp_m)
        else:
            non_containing_meshes.append(sp_m)
    
    if filter_away:
        return non_containing_meshes
    else:
        return containing_meshes
    
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
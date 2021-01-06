import numpy as np
import datajoint as dj
from tqdm.notebook import tqdm
from pathlib import Path
import time

"""
Good link: https://medium.com/kubernetes-tutorials/learn-how-to-assign-pods-to-nodes-in-kubernetes-using-nodeselector-and-affinity-features-e62c437f3cf8

Kubernetes help: 
kubernetes.io/hostname=at-compute005

To show labels: 
kubectl get nodes --show-labels

gil-desktop
nikosk

#how to delete the 
for p in $(kubectl get pods | grep Terminating | awk '{print $1}'); do kubectl delete pod $p --grace-period=0 --force;done


"""

"""
How to clear the external

schema.external['decomposition'].delete(delete_external_files=True)

"""
decomposition_folder = "decomposition"

attributes_need_resetting = ["external_segmentation_path",
                             "external_mesh_path",
                             "external_decimated_mesh_path",
                             "external_skeleton_path",
                            ]

def print_minnie65_config_paths(minfig):
    """
    Check the relevant paths of the minfig to make 
    sure they are set to the right segmentation
    
    """
    for at in attributes_need_resetting:
        curr_at_path = getattr(minfig.minnie65_config,at)
        print(f"Current path for {at} = {curr_at_path}")
        

def config_celii():
    dj.config['database.host'] = 'at-database.ad.bcm.edu'
    dj.config['database.user'] = 'celiib'
    dj.config['database.password'] = 'newceliipass'
    


def set_minnie65_config_segmentation(minfig,
                                 curr_seg="02",
                                verbose=False):
    
    

    if verbose:
        #check that went well
        for at in attributes_need_resetting:
            curr_at_path = getattr(minfig.minnie65_config,at)
            print(f"Current path for {at} = {curr_at_path}")
            
    
    curr_seg_path = getattr(minfig.minnie65_config,"external_segmentation_path")
    external_segmentation_path = curr_seg_path.parent / Path(curr_seg)
    setattr(minfig.minnie65_config,"external_segmentation_path",external_segmentation_path)

    external_mesh_path = external_segmentation_path / 'meshes'
    setattr(minfig.minnie65_config,"external_mesh_path",external_mesh_path)
    external_decimated_mesh_path = external_segmentation_path / 'decimated_meshes'
    setattr(minfig.minnie65_config,"external_decimated_mesh_path",external_decimated_mesh_path)
    external_skeleton_path = external_segmentation_path / 'skeletons'
    setattr(minfig.minnie65_config,"external_skeleton_path",external_skeleton_path)
    
    if verbose:
        #check that went well
        for at in attributes_need_resetting:
            curr_at_path = getattr(minfig.minnie65_config,at)
            print(f"Current path for {at} = {curr_at_path}")



        
        
# ------ Functions that will help decimate meshes ------------ #

# --------- Adapter that will be used for decomposition ----------- #
import neuron_utils as nru
import os
class DecompositionAdapter(dj.AttributeAdapter):
    # Initialize the correct attribute type (allows for use with multiple stores)
    def __init__(self, attribute_type):
        self.attribute_type = attribute_type
        super().__init__()

    #?
    attribute_type = '' # this is how the attribute will be declared
    has_version = False # used for file name recognition
    
    def put(self, filepath):
        # save the filepath to the mesh
        filepath = os.path.abspath(filepath)
        assert os.path.exists(filepath)
        return filepath
    
    def get(self,filepath):
        """
        1) Get the filepath of the decimated mesh
        2) Make sure that both file paths exist
        3) use the decompress method
        
        """
        
        #1) Get the filepath of the decimated mesh
        
        filepath = Path(filepath)
        assert os.path.exists(filepath)
        
        """Old way where used the file path
        dec_filepath = get_decimated_mesh_path_from_decomposition_path(filepath)
        assert os.path.exists(dec_filepath)
        print(f"Attempting to get the following files:\ndecomp = {filepath}\ndec = {dec_filepath} ")
        """
        
        
        #2) get the decimated mesh 
        segment_id = int(filepath.stem.split("_")[0])
        dec_mesh = fetch_segment_id_mesh(segment_id)
        
        
        #3) use the decompress method
        recovered_neuron = nru.decompress_neuron(filepath=filepath,
                     original_mesh=dec_mesh)
        
        return recovered_neuron
        
        
decomposition = DecompositionAdapter('filepath@decomposition')

adapter_decomp_obj = {
    'decomposition':decomposition
}


"""
other adapters that are available for use:

# instantiate for use as a datajoint type
mesh = MeshAdapter('filepath@meshes')
decimated_mesh = DecimatedMeshAdapter('filepath@decimated_meshes')

# also store in one object for ease of use with virtual modules
adapter_objects = {
    'mesh': mesh,
    'decimated_mesh': decimated_mesh
}



"""


# --------- DONE Adapter that will be used for decomposition ----------- #

# ---------- Soma Adapter ------------#
import h5py
import os

from collections import namedtuple


class SomasAdapter(dj.AttributeAdapter):
    # Initialize the correct attribute type (allows for use with multiple stores)
    def __init__(self, attribute_type):
        self.attribute_type = attribute_type
        super().__init__()

    attribute_type = '' # this is how the attribute will be declared

    TriangularMesh = namedtuple('TriangularMesh', ['segment_id', 'vertices', 'faces'])
    
    def put(self, filepath):
        # save the filepath to the mesh
        filepath = os.path.abspath(filepath)
        assert os.path.exists(filepath)
        return filepath

    def get(self, filepath):
        # access the h5 file and return a mesh
        assert os.path.exists(filepath)

        with h5py.File(filepath, 'r') as hf:
            vertices = hf['vertices'][()].astype(np.float64)
            faces = hf['faces'][()].reshape(-1, 3).astype(np.uint32)
        
        segment_id = os.path.splitext(os.path.basename(filepath))[0]

        return self.TriangularMesh(
            segment_id=int(segment_id),
            vertices=vertices,
            faces=faces
        )
    
somas = SomasAdapter('filepath@somas')

adapter_somas_obj = {
    'somas':somas
}



# ----------- DONE Soma Adapter


from minfig import adapter_objects
def get_adapter_object():
    if "decomposition" not in adapter_objects.keys():
        adapter_objects.update(adapter_decomp_obj)
    if "somas" not in adapter_objects.keys():
        adapter_objects.update(adapter_somas_obj)
    return adapter_objects

import datajoint as dj
import minfig
def get_decomposition_path():
    return minfig.minnie65_config.external_segmentation_path / Path(f"{decomposition_folder}/")
def get_somas_path():
    return minfig.minnie65_config.external_segmentation_path / Path("somas/")

def get_decimated_mesh_path_from_decomposition_path(filepath):
    filepath = Path(filepath)
    dec_filepath = filepath.parents[1] / Path(f"decimation_meshes/{filepath.stem}.h5")
    return dec_filepath
    
def configure_minnie_vm():
    
    set_minnie65_config_segmentation(minfig)
    minnie = minfig.configure_minnie(return_virtual_module=True)

    # Old way of getting access to the virtual modules
    # m65 = dj.create_virtual_module('minnie', 'microns_minnie65_02')

    #New way of getting access to module
    
     # included with wildcard imports
    
    minnie = dj.create_virtual_module('minnie', 'microns_minnie65_02', add_objects=get_adapter_object())

    schema = dj.schema("microns_minnie65_02")
    dj.config["enable_python_native_blobs"] = True
    
    #confiugre the storage
    decomp_path = get_decomposition_path()
    somas_path = get_somas_path()
    
    assert decomp_path.exists()
    assert somas_path.exists()
    
#     if not decomp_path.exists():
#         raise Exception("The decomposition path does not exist")


    stores_config = {'decomposition': {
                'protocol': 'file',
                'location': str(decomp_path),
                'stage': str(decomp_path)
            },
            'somas': {
                'protocol': 'file',
                'location': str(somas_path),
                'stage': str(somas_path)
            }       
                    
                    }    

    if 'stores' not in dj.config:
        dj.config['stores'] = stores_config
    else:
        dj.config['stores'].update(stores_config)
        
        
    return minnie,schema

import trimesh
def get_decimated_mesh(seg_id,decimation_ratio=0.25):
    key = dict(segment_id=seg_id,decimation_ratio=decimation_ratio)
    new_mesh = (minnie.Decimation() & key).fetch1("mesh")
    current_mesh_verts,current_mesh_faces = new_mesh.vertices,new_mesh.faces
    return trimesh.Trimesh(vertices=current_mesh_verts,faces=current_mesh_faces)

# def get_seg_extracted_somas(seg_id,minnie=None):
#     key = dict(segment_id=seg_id)  
#     soma_vertices, soma_faces = (minnie.BaylorSegmentCentroid() & key).fetch("soma_vertices","soma_faces")
#     return [trimesh.Trimesh(vertices=v,faces=f) for v,f in zip(soma_vertices, soma_faces)]

def get_seg_extracted_somas(seg_id):
    key = dict(segment_id=seg_id)  
    soma_meshes = (minnie.BaylorSegmentCentroid() & key).fetch("mesh")
    return [trimesh.Trimesh(vertices=v.vertices,faces=v.faces) for v in soma_meshes]


def get_soma_mesh_list(seg_id):
    key = dict(segment_id=seg_id)  
    soma_vertices, soma_faces,soma_run_time,soma_sdf = (minnie.BaylorSegmentCentroid() & key).fetch("soma_vertices","soma_faces","run_time","sdf")
    s_meshes = [trimesh.Trimesh(vertices=v,faces=f) for v,f in zip(soma_vertices, soma_faces)]
    s_times = np.array(soma_run_time)
    s_sdfs = np.array(soma_sdf)
    return [s_meshes,s_times,s_sdfs]

def get_soma_mesh_list(seg_id):
    key = dict(segment_id=seg_id)  
    soma_meshes,soma_run_time,soma_sdf = (minnie.BaylorSegmentCentroid() & key).fetch("mesh","run_time","sdf")
    s_meshes = [trimesh.Trimesh(vertices=v.vertices,faces=v.faces) for v in soma_meshes]
    s_times = np.array(soma_run_time)
    s_sdfs = np.array(soma_sdf)
    return [s_meshes,s_times,s_sdfs]

def nucleus_id_to_seg_id(nucleus_id):
    """
    Pseudocode:
    1) restrict the nucleus id table
    2) fetch for the segment id
    """
    nucl_key = dict(nucleus_id=nucleus_id)
    nucl_seg_id = (minnie.NucleusID()  & nucl_key).fetch1("segment_id")
    if nucl_seg_id == 0:
        raise Exception(f"No segment id for nucleus Id {nucleus_id}")
    return nucl_seg_id
    
import trimesh
def fetch_segment_id_mesh(seg_id,decimation_ratio=0.25):
    key = dict(segment_id=seg_id,decimation_ratio=decimation_ratio)
    new_mesh = (minnie.Decimation() & key).fetch1("mesh")
    current_mesh_verts,current_mesh_faces = new_mesh.vertices,new_mesh.faces
    return trimesh.Trimesh(vertices=current_mesh_verts,faces=current_mesh_faces)

def fetch_undecimated_segment_id_mesh(seg_id,decimation_ratio=0.25):
    key = dict(segment_id=seg_id,decimation_ratio=decimation_ratio)
    new_mesh = (minnie.Mesh() & key).fetch1("mesh")
    current_mesh_verts,current_mesh_faces = new_mesh.vertices,new_mesh.faces
    return trimesh.Trimesh(vertices=current_mesh_verts,faces=current_mesh_faces)


def fetch_nucleus_id_mesh(nucleus_id,verbose=False):
    seg_id = nucleus_id_to_seg_id(nucleus_id)
    print(f"Attempting to fetch segment_id: {seg_id}")
    return fetch_segment_id_mesh(seg_id)





import skeleton_utils as sk
def plot_decimated_mesh_with_somas(seg_id):
    """
    To visualize a decimated mesh with the somas
    """
    print(f"Segment_id = {seg_id}")
#     multi_soma_seg_ids = np.unique(multi_soma_seg_ids)
#     seg_id_idx = -2
#     seg_id = multi_soma_seg_ids[seg_id_idx]


    dec_mesh = get_decimated_mesh(seg_id)
    print(f"vertices = {len(dec_mesh.vertices)}, faces= = {len(dec_mesh.faces)}")
    curr_soma_meshes = get_seg_extracted_somas(seg_id)
    curr_soma_mesh_list = get_soma_mesh_list(seg_id)

    import skeleton_utils as sk
    sk.graph_skeleton_and_mesh(main_mesh_verts=dec_mesh.vertices,
                               main_mesh_faces=dec_mesh.faces,
                            other_meshes=curr_soma_meshes,
                              other_meshes_colors="red")
    
import trimesh_utils as tu
import neuron_visualizations as nviz
import error_detection as ed
def plot_errored_faces(segment_id,
                       plot_synapses=False,
                       current_mesh=None,
                       neuron_obj=None,
                       return_obj=False,
                       valid_synapse_color = "yellow",
                       error_color = "red",**kwargs):
    """
    Function that will plot the neuron, the errored mesh part
    and the synapses if requested (distinguishing between errored and non-errored synapses)
    
    du.plot_errored_faces(segment_id=864691134884745210,
                       plot_synapses=True,
                       current_mesh=None,
                       neuron_obj=None,
                       valid_synapse_color = "yellow",
                       error_color = "red")
    
    """
    
    #1) Pull down the mesh
    if current_mesh is None:
        current_mesh = fetch_segment_id_mesh(segment_id)

    #2) PUll down the synapse data and the error faces
    n_synapses,n_errored_synapses,errored_faces = (minnie.AutoProofreadLabels() &
                                                   dict(segment_id=segment_id)).fetch1("n_synapses","n_errored_synapses","face_idx_for_error")
    
    error_submesh = current_mesh.submesh([errored_faces],append=True)
    valid_mesh = tu.subtract_mesh(current_mesh,error_submesh)
    
    if plot_synapses:
        if neuron_obj is None:
            neuron_obj = (minnie.Decomposition() & dict(segment_id=segment_id)).fetch1("decomposition")
        
        err_synapses,non_err_synapses = ed.get_error_synapse_inserts(current_mesh,segment_id,
                                                             errored_faces,return_synapse_centroids=True)
        
        nviz.plot_objects(main_mesh=valid_mesh,
                  meshes=[error_submesh],
                          mesh_alpha=1,
                  meshes_colors=[error_color],
                scatters=[err_synapses,non_err_synapses],
                 scatters_colors=[error_color,valid_synapse_color])
        if return_obj:
            return neuron_obj
    
    else:
        nviz.plot_objects(main_mesh=valid_mesh,
                 meshes=error_submesh,
                  meshes_colors=error_color,
                 mesh_alpha=1,
                     **kwargs)
    
    
# ---------------- 1/5/21 Modules that are used for downloading --------------------------------------- #
def adapt_mesh_hdf5(segment_id=None, filepath=None, basepath=None, return_type='namedtuple', as_lengths=False):
    """
    Reads from a mesh hdf5 and returns it in the form of a numpy array with labeled dtype or optionally
        as a dictionary, or optionally as separate variables.
    :param segment_id: Segment ID will be used in conjunction with the `m65.external_mesh_path`
        to find the mesh if filepath is not set.
    :param filepath: File path pointing to the hdf5 mesh file. If filepath is None it will use
        `m65.external_mesh_path` joined with the segment_id.
    :param return_type: Options = {
        namedtuple = return the vertices and triangles in a namedtuple format
        dict = return the vertices and triangles as arrays in a dictionary referrenced by those names,
        separate = return the vertex array and triangle array as separate variables in that order
    :param as_lengths: Overrides return_type and instead returns the length of the vertex and face arrays.
        This is done without pulling the mesh into memory, which makes it far more space and time efficient.
    }
    """
    
    if basepath is None:
        basepath = minfig.minnie65_config.external_mesh_path
        
    # File manipulation
    if filepath is None:
        if segment_id is not None and basepath is not None:
            filepath = os.path.join(basepath, f'{segment_id}.h5')
        else:
            raise TypeError('Both segment_id and filepath cannot be None.')
    elif segment_id is None:
        segment_id = os.path.splitext(os.path.basename(filepath))[0]
    else:
        raise TypeError('Both segment_id and filepath cannot be None.')
    filepath = os.path.abspath(filepath)

    # Load the mesh data
    with h5py.File(filepath, 'r') as f:
        if as_lengths:
            return f['vertices'].shape[0], int(f['faces'].shape[0] / 3)
        vertices = f['vertices'][()].astype(np.float64)
        faces = f['faces'][()].reshape(-1, 3).astype(np.uint32)
    
    # Return options
    if return_type == 'namedtuple':
        return Mesh(
            segment_id=segment_id,
            vertices=vertices,
            faces=faces
        )        
    elif return_type == 'dict':
        return dict(
            segment_id=segment_id,
            vertices=vertices,
            faces=faces
        )
    elif return_type == 'separate':
        return vertices, faces
    else:
        raise TypeError(f'return_type does not accept {return_type} argument')
        

def download_meshes(
    segment_ids=None,
    segment_order=None,
    target_dir=None,
    # cloudvolume_path="precomputed://gs://microns-seunglab/minnie65/seg_minnie65_0",
    cloudvolume_path=r"graphene://https://minniev1.microns-daf.com/segmentation/table/minnie3_v1",
    overwrite=False,
    n_threads=1,
    verbose=True,
    stitch_mesh_chunks=True,
    download_each_supress_errors=False
):
    """
    Cloudvolume also requires credentials:
    :param segment_order: Options = {
        None = leaves the order alone,
        'reverse' = reverse the order,
        'asc' = sorts in ascending order,
        'desc' = sorts in descending order,
        'shuffle' = shuffles the order
    }
    """
    
    if target_dir is None:
        target_dir = minfig.minnie65_config.external_mesh_path
        
    
    from meshparty import trimesh_io

    if segment_ids is None:
        segment_ids = (Segment - Mesh).fetch('segment_id')

    if segment_order is not None:
        segment_order = segment_order.lower()
    if segment_order == 'reverse':
        segment_ids = segment_ids[::-1]
    elif segment_order == 'asc':
        segment_ids = np.sort(segment_ids)
    elif segment_order == 'desc':
        segment_ids = np.sort(segment_ids)[::-1]
    elif segment_order == 'shuffle':
        np.random.shuffle(segment_ids)

    mesh_download_time = time.time()

    if not download_each_supress_errors:
        trimesh_io.download_meshes(
            seg_ids=segment_ids,
            target_dir=target_dir,
            cv_path=cloudvolume_path,
            overwrite=overwrite,
            n_threads=n_threads,
            verbose=verbose,
            stitch_mesh_chunks=stitch_mesh_chunks
        )
    else:
        for seg in segment_ids:
            try:
                trimesh_io.download_meshes(
                    seg_ids=[seg],
                    target_dir=target_dir,
                    cv_path=cloudvolume_path,
                    overwrite=overwrite,
                    n_threads=n_threads,
                    verbose=verbose,
                    stitch_mesh_chunks=stitch_mesh_chunks
                )
            except ValueError as e:
                print(e)

    total_time = time.time()-mesh_download_time
    print(f'Done in {total_time:0f} seconds.')
    return total_time


def fill_from_ids(segment_ids, skip_duplicates=True):
    for segment_id in segment_ids:
        mesh_path = os.path.join(minfig.minnie65_config.external_mesh_path, f'{segment_id}.h5')
        n_vertices, n_faces = adapt_mesh_hdf5(filepath=mesh_path, as_lengths=True)
        minnie.Mesh.insert1(dict(
            segment_id=segment_id,
            n_vertices=n_vertices,
            n_faces=n_faces,
            mesh=mesh_path
        ),
        skip_duplicates=skip_duplicates,
        allow_direct_insert=True)
        
def download_and_insert_allen_meshes(segment_ids,n_threads=1,
                                    insert_in_multi_soma_table=False):
    """
    Purpose: To Download the meshes from the allen institute
    and then insert the segment ids into the Segment
    and Mesh table in Datajoint
    """
    
    # 1) Fill segment table with segment ids
    download_meshes(segment_ids = segment_ids,n_threads=12)
    
    #2) Manually add segmnet ids to segment tables
    insert_keys = [dict(segment_id=k) for k in segment_ids]
    minnie.Segment.insert(insert_keys,skip_duplicates=True)
    
    #3) Fill in the Mesh Table
    fill_from_ids(segment_ids=segment_ids)
    
    if insert_in_multi_soma_table:
        minnie.MultiSomaProofread.insert(insert_keys,skip_duplicates=True)
    
    
#runs the configuration
config_celii()
minnie,_ = configure_minnie_vm()
from minfig.adapters import *


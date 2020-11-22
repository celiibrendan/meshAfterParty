import numpy as np
import datajoint as dj
from tqdm.notebook import tqdm
from pathlib import Path

"""
How to clear the external

schema.external['decomposition'].delete(delete_external_files=False)

"""

attributes_need_resetting = ["external_segmentation_path",
                             "external_mesh_path",
                             "external_decimated_mesh_path",
                             "external_skeleton_path",
                            ]

def config_celii():
    dj.config['database.host'] = 'at-database.ad.bcm.edu'
    dj.config['database.user'] = 'celiib'
    dj.config['database.password'] = 'newceliipass'
    
#runs the configuration
config_celii()

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
            
def print_minnie65_config_paths(minfig):
    """
    Check the relevant paths of the minfig to make 
    sure they are set to the right segmentation
    
    """
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
    return minfig.minnie65_config.external_segmentation_path / Path("decomposition/")
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
def get_decimated_mesh(seg_id,decimation_ratio=0.25,minnie=None):
    if minnie is None:
        minnie,_ = configure_minnie_vm()
    key = dict(segment_id=seg_id,decimation_ratio=decimation_ratio)
    new_mesh = (minnie.Decimation() & key).fetch1("mesh")
    current_mesh_verts,current_mesh_faces = new_mesh.vertices,new_mesh.faces
    return trimesh.Trimesh(vertices=current_mesh_verts,faces=current_mesh_faces)

# def get_seg_extracted_somas(seg_id,minnie=None):
#     if minnie is None:
#         minnie,_ = configure_minnie_vm()
#     key = dict(segment_id=seg_id)  
#     soma_vertices, soma_faces = (minnie.BaylorSegmentCentroid() & key).fetch("soma_vertices","soma_faces")
#     return [trimesh.Trimesh(vertices=v,faces=f) for v,f in zip(soma_vertices, soma_faces)]

def get_seg_extracted_somas(seg_id,minnie=None):
    if minnie is None:
        minnie,_ = configure_minnie_vm()
    key = dict(segment_id=seg_id)  
    soma_meshes = (minnie.BaylorSegmentCentroid() & key).fetch("mesh")
    return [trimesh.Trimesh(vertices=v.vertices,faces=v.faces) for v in soma_meshes]


def get_soma_mesh_list(seg_id,minnie=None):
    if minnie is None:
        minnie,_ = configure_minnie_vm()
    key = dict(segment_id=seg_id)  
    soma_vertices, soma_faces,soma_run_time,soma_sdf = (minnie.BaylorSegmentCentroid() & key).fetch("soma_vertices","soma_faces","run_time","sdf")
    s_meshes = [trimesh.Trimesh(vertices=v,faces=f) for v,f in zip(soma_vertices, soma_faces)]
    s_times = np.array(soma_run_time)
    s_sdfs = np.array(soma_sdf)
    return [s_meshes,s_times,s_sdfs]

def get_soma_mesh_list(seg_id,minnie=None):
    if minnie is None:
        minnie,_ = configure_minnie_vm()
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
def fetch_segment_id_mesh(seg_id,decimation_ratio=0.25,minnie=None):
    if minnie is None:
        minnie,_ = configure_minnie_vm()
    key = dict(segment_id=seg_id,decimation_ratio=decimation_ratio)
    new_mesh = (minnie.Decimation() & key).fetch1("mesh")
    current_mesh_verts,current_mesh_faces = new_mesh.vertices,new_mesh.faces
    return trimesh.Trimesh(vertices=current_mesh_verts,faces=current_mesh_faces)

def fetch_undecimated_segment_id_mesh(seg_id,decimation_ratio=0.25):
    minnie,schema = configure_minnie_vm()
    key = dict(segment_id=seg_id,decimation_ratio=decimation_ratio)
    new_mesh = (minnie.Mesh() & key).fetch1("mesh")
    current_mesh_verts,current_mesh_faces = new_mesh.vertices,new_mesh.faces
    return trimesh.Trimesh(vertices=current_mesh_verts,faces=current_mesh_faces)


def fetch_nucleus_id_mesh(nucleus_id,verbose=False):
    seg_id = nucleus_id_to_seg_id(nucleus_id)
    print(f"Attempting to fetch segment_id: {seg_id}")
    return fetch_segment_id_mesh(seg_id)





import skeleton_utils as sk
def plot_decimated_mesh_with_somas(seg_id,minnie=None):
    """
    To visualize a decimated mesh with the somas
    """
#     multi_soma_seg_ids = np.unique(multi_soma_seg_ids)
#     seg_id_idx = -2
#     seg_id = multi_soma_seg_ids[seg_id_idx]

    if minnie is None:
        minnie,_ = configure_minnie_vm()

    dec_mesh = get_decimated_mesh(seg_id)
    curr_soma_meshes = get_seg_extracted_somas(seg_id)
    curr_soma_mesh_list = get_soma_mesh_list(seg_id)

    import skeleton_utils as sk
    sk.graph_skeleton_and_mesh(main_mesh_verts=dec_mesh.vertices,
                               main_mesh_faces=dec_mesh.faces,
                            other_meshes=curr_soma_meshes,
                              other_meshes_colors="red")
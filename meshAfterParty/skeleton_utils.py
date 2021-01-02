"""
These functions will help with skeletonization

"""
import numpy_utils as nu
import trimesh_utils as tu
from trimesh_utils import split_significant_pieces,split,combine_meshes,write_neuron_off
import networkx_utils as xu
import matplotlib_utils as mu
import meshparty_skeletonize as m_sk
import soma_extraction_utils as sm

import numpy as np
import trimesh

from tqdm_utils import tqdm



def compare_endpoints(endpoints_1,endpoints_2,**kwargs):
    """
    comparing the endpoints of a graph: 
    
    Ex: 
    import networkx_utils as xu
    xu = reload(xu)mess
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

def save_skeleton_cgal(surface_with_poisson_skeleton,largest_mesh_path):
    """
    surface_with_poisson_skeleton (np.array) : nx2 matrix with the nodes
    """
    first_node = surface_with_poisson_skeleton[0][0]
    end_nodes =  surface_with_poisson_skeleton[:,1]
    
    skeleton_to_write = str(len(end_nodes) + 1) + " " + str(first_node[0]) + " " +  str(first_node[1]) + " " +  str(first_node[2])
    
    for node in end_nodes:
        skeleton_to_write +=  " " + str(node[0]) + " " +  str(node[1]) + " " +  str(node[2])
    
    output_file = largest_mesh_path
    if output_file[-5:] != ".cgal":
        output_file += ".cgal"
        
    f = open(output_file,"w")
    f.write(skeleton_to_write)
    f.close()
    return 

#read in the skeleton files into an array
def read_skeleton_edges_coordinates(file_path):
    if type(file_path) == str or type(file_path) == type(Path()):
        file_path = [file_path]
    elif type(file_path) == list:
        pass
    else:
        raise Exception("file_path not a string or list")
    new_file_path = []
    for f in file_path:
        if type(f) == type(Path()):
            new_file_path.append(str(f.absolute()))
        else:
            new_file_path.append(str(f))
    file_path = new_file_path
    
    total_skeletons = []
    for fil in file_path:
        try:
            with open(fil) as f:
                bones = np.array([])
                for line in f.readlines():
                    #print(line)
                    line = (np.array(line.split()[1:], float).reshape(-1, 3))
                    #print(line[:-1])
                    #print(line[1:])

                    #print(bones.size)
                    if bones.size <= 0:
                        bones = np.stack((line[:-1],line[1:]),axis=1)
                    else:
                        bones = np.vstack((bones,(np.stack((line[:-1],line[1:]),axis=1))))
                    #print(bones)
                total_skeletons.append(np.array(bones).astype(float))
        except:
            print(f"file {fil} not found so skipping")
    
    return stack_skeletons(total_skeletons)
#     if len(total_skeletons) > 1:
#         returned_skeleton = np.vstack(total_skeletons)
#         return returned_skeleton
#     if len(total_skeletons) == 0:
#         print("There was no skeletons found for these files")
#     return np.array(total_skeletons).reshape(-1,2,3)

#read in the skeleton files into an array
def read_skeleton_verts_edges(file_path):
    with open(file_path) as f:
        bones = np.array([])
        for line in f.readlines():
            #print(line)
            line = (np.array(line.split()[1:], float).reshape(-1, 3))
            #print(line[:-1])
            #print(line[1:])

            #print(bones.size)
            if bones.size <= 0:
                bones = np.stack((line[:-1],line[1:]),axis=1)
            else:
                bones = np.vstack((bones,(np.stack((line[:-1],line[1:]),axis=1))))
            #print(bones)
    
    bones_array = np.array(bones).astype(float)
    
    #unpacks so just list of vertices
    vertices_unpacked  = bones_array.reshape(-1,3)

    #reduce the number of repeat vertices and convert to list
    unique_rows = np.unique(vertices_unpacked, axis=0)
    unique_rows_list = unique_rows.tolist()

    #assigns the number to the vertex (in the original vertex list) that corresponds to the index in the unique list
    vertices_unpacked_coefficients = np.array([unique_rows_list.index(a) for a in vertices_unpacked.tolist()])

    #reshapes the vertex list to become an edge list (just with the labels so can put into netowrkx graph)
    edges_with_coefficients =  np.array(vertices_unpacked_coefficients).reshape(-1,2)

    return unique_rows, edges_with_coefficients


def skeleton_unique_coordinates(skeleton):
    return np.unique(skeleton.reshape(-1,3),axis=0)

def convert_nodes_edges_to_skeleton(nodes,edges):
    return nodes[edges]

def convert_skeleton_to_nodes_edges(bones_array):
    #unpacks so just list of vertices
    vertices_unpacked  = bones_array.reshape(-1,3)

    #reduce the number of repeat vertices and convert to list
    unique_rows = np.unique(vertices_unpacked, axis=0)
    unique_rows_list = unique_rows.tolist()

    #assigns the number to the vertex (in the original vertex list) that corresponds to the index in the unique list
    vertices_unpacked_coefficients = np.array([unique_rows_list.index(a) for a in vertices_unpacked.tolist()])

    #reshapes the vertex list to become an edge list (just with the labels so can put into netowrkx graph)
    edges_with_coefficients =  np.array(vertices_unpacked_coefficients).reshape(-1,2)

    return unique_rows, edges_with_coefficients

def calculate_skeleton_segment_distances(my_skeleton,cumsum=True):
    segment_distances = np.sqrt(np.sum((my_skeleton[:,0] - my_skeleton[:,1])**2,axis=1)).astype("float")
    if cumsum:
        return np.cumsum(segment_distances)
    else:
        return segment_distances

def calculate_skeleton_distance(my_skeleton):
    if len(my_skeleton) == 0:
        return 0
    total_distance = np.sum(np.sqrt(np.sum((my_skeleton[:,0] - my_skeleton[:,1])**2,axis=1)))
    return float(total_distance)


import ipyvolume as ipv
from copy import deepcopy
def plot_ipv_mesh(mesh,color=[1.,0.,0.,0.2],
                 flip_y=True):
    
    if len(mesh.vertices) == 0:
        return
    
    if flip_y:
        #print("inside elephant flipping copy")
        elephant_mesh_sub = mesh.copy()
        elephant_mesh_sub.vertices[...,1] = -elephant_mesh_sub.vertices[...,1]
    else:
        elephant_mesh_sub = mesh
    
    
    #check if the color is a dictionary
    if type(color) == dict:
        #get the type of values stored in there
        labels = list(color.items())
        
        #if the labels were stored as just numbers/decimals
        if type(labels[0]) == int or type(labels[0]) == float:
            #get all of the possible labels
            unique_labels = np.unique(labels)
            #get random colors for all of the labels
            colors_list =  mu.generate_color_list(n_colors)
            for lab,curr_color in zip(unique_labels,colors_list):
                #find the faces that correspond to that label
                faces_to_keep = [k for k,v in color.items() if v == lab]
                #draw the mesh with that color
                curr_mesh = elephant_mesh_sub.submesh([faces_to_keep],append=True)
                
                mesh4 = ipv.plot_trisurf(elephant_mesh_sub.vertices[:,0],
                               elephant_mesh_sub.vertices[:,1],
                               elephant_mesh_sub.vertices[:,2],
                               triangles=elephant_mesh_sub.faces)
                mesh4.color = curr_color
                mesh4.material.transparent = True
    else:          
        mesh4 = ipv.plot_trisurf(elephant_mesh_sub.vertices[:,0],
                                   elephant_mesh_sub.vertices[:,1],
                                   elephant_mesh_sub.vertices[:,2],
                                   triangles=elephant_mesh_sub.faces)
        mesh4.color = color
        mesh4.material.transparent = True
        

def plot_ipv_skeleton(edge_coordinates,color=[0,0.,1,1],
                     flip_y=True):
    
    if len(edge_coordinates) == 0:
        print("Edge coordinates in plot_ipv_skeleton were of 0 length so returning")
        return []
    
    if flip_y:
        edge_coordinates = edge_coordinates.copy()
        edge_coordinates[...,1] = -edge_coordinates[...,1] 
    
    #print(f"edge_coordinates inside after change = {edge_coordinates}")
    unique_skeleton_verts_final,edges_final = convert_skeleton_to_nodes_edges(edge_coordinates)
    mesh2 = ipv.plot_trisurf(unique_skeleton_verts_final[:,0], 
                            unique_skeleton_verts_final[:,1], 
                            unique_skeleton_verts_final[:,2], 
                            lines=edges_final)
    #print(f"color in ipv_skeleton = {color}")
    mesh2.color = color 
    mesh2.material.transparent = True
    
    #print(f"Color in skeleton ipv plot = {color}")

    if flip_y:
        unique_skeleton_verts_final[...,1] = -unique_skeleton_verts_final[...,1]
    
    return unique_skeleton_verts_final

def plot_ipv_scatter(scatter_points,scatter_color=[1.,0.,0.,0.5],
                    scatter_size=0.4,
                    flip_y=True):
    
    scatter_points = (np.array(scatter_points).reshape(-1,3)).astype("float")
    if flip_y:
        scatter_points = scatter_points.copy()
        scatter_points[...,1] = -scatter_points[...,1]
#     print(f"scatter_points[:,0] = {scatter_points[:,0]}")
#     print(f"scatter_points[:,1] = {scatter_points[:,1]}")
#     print(f"scatter_points[:,2] = {scatter_points[:,2]}")
#     print(f"scatter_size = {scatter_size}")
#     print(f"scatter_color = {scatter_color}")
    mesh_5 = ipv.scatter(
            scatter_points[:,0], 
            scatter_points[:,1],
            scatter_points[:,2], 
            size=scatter_size, 
            color=scatter_color,
            marker="sphere")
    mesh_5.material.transparent = True

def graph_skeleton_and_mesh(main_mesh_verts=[],
                            main_mesh_faces=[],
                            unique_skeleton_verts_final=[],
                            edges_final=[],
                            edge_coordinates=[],
                            other_meshes=[],
                            other_meshes_colors =  [],
                            mesh_alpha=0.2,
                            other_meshes_face_components = [],
                            other_skeletons = [],
                            other_skeletons_colors =  [],
                            return_other_colors = False,
                            main_mesh_color = [0.,1.,0.,0.2],
                            main_skeleton_color = [0,0.,1,1],
                            main_mesh_face_coloring = [],
                            other_scatter=[],
                            scatter_size = 0.3,
                            other_scatter_colors=[],
                            main_scatter_color=[1.,0.,0.,0.5],
                            buffer=1000,
                           axis_box_off=True,
                           html_path="",
                           show_at_end=True,
                           append_figure=False,
                           flip_y=True):
    """
    Graph the final result of skeleton and mesh
    
    Pseudocode on how to do face colorings :
    could get a dictionary mapping faces to colors or groups
    - if mapped to groups then do random colors (and generate them)
    - if mapped to colors then just do submeshes and send the colors
    """
    #print(f"other_scatter = {other_scatter}")
    #print(f"mesh_alpha = {mesh_alpha}")
    
    if not append_figure:
        ipv.figure(figsize=(15,15))
    
    main_mesh_vertices = []
    
    
    #print("Working on main skeleton")
    if (len(unique_skeleton_verts_final) > 0 and len(edges_final) > 0) or (len(edge_coordinates)>0):
        if flip_y:
            edge_coordinates = edge_coordinates.copy()
            edge_coordinates[...,1] = -edge_coordinates[...,1]
        if (len(edge_coordinates)>0):
            unique_skeleton_verts_final,edges_final = convert_skeleton_to_nodes_edges(edge_coordinates)
        mesh2 = ipv.plot_trisurf(unique_skeleton_verts_final[:,0], 
                                unique_skeleton_verts_final[:,1], 
                                unique_skeleton_verts_final[:,2], 
                                lines=edges_final, color='blue')

        mesh2.color = main_skeleton_color 
        mesh2.material.transparent = True
        
        if flip_y:
            unique_skeleton_verts_final[...,1] = -unique_skeleton_verts_final[...,1]
            
        main_mesh_vertices.append(unique_skeleton_verts_final)
    
    #print("Working on main mesh")
    if len(main_mesh_verts) > 0 and len(main_mesh_faces) > 0:
        if len(main_mesh_face_coloring) > 0:
            #will go through and color the faces of the main mesh if any sent
            for face_array,face_color in main_mesh_face_coloring:
                curr_mesh = main_mesh.submesh([face_array],append=True)
                plot_ipv_mesh(curr_mesh,face_color,flip_y=flip_y)
        else:
            if flip_y:
                main_mesh_verts = main_mesh_verts.copy()
                main_mesh_verts[...,1] = -main_mesh_verts[...,1]
            
            main_mesh = trimesh.Trimesh(vertices=main_mesh_verts,faces=main_mesh_faces)

            mesh3 = ipv.plot_trisurf(main_mesh.vertices[:,0],
                                   main_mesh.vertices[:,1],
                                   main_mesh.vertices[:,2],
                                   triangles=main_mesh.faces)
            mesh3.color = main_mesh_color
            mesh3.material.transparent = True
            
            #flipping them back
            if flip_y:
                main_mesh_verts[...,1] = -main_mesh_verts[...,1]
            
        main_mesh_vertices.append(main_mesh_verts)
        
    
    # cast everything to list type
    if type(other_meshes) != list and type(other_meshes) != np.ndarray:
        other_meshes = [other_meshes]
    if type(other_meshes_colors) != list and type(other_meshes_colors) != np.ndarray:
        other_meshes_colors = [other_meshes_colors]
    if type(other_skeletons) != list and type(other_skeletons) != np.ndarray:
        other_skeletons = [other_skeletons]
    if type(other_skeletons_colors) != list and type(other_skeletons_colors) != np.ndarray:
        other_skeletons_colors = [other_skeletons_colors]
        
#     if type(other_scatter) != list and type(other_scatter) != np.ndarray:
#         other_scatter = [other_scatter]
#     if type(other_scatter_colors) != list and type(other_scatter_colors) != np.ndarray:
#         other_scatter_colors = [other_scatter_colors]

    if not nu.is_array_like(other_scatter):
        other_scatter = [other_scatter]
    if not nu.is_array_like(other_scatter_colors):
        other_scatter_colors = [other_scatter_colors]
    
        
    
    
    if len(other_meshes) > 0:
        if len(other_meshes_face_components ) > 0:
            other_meshes_colors = other_meshes_face_components
        elif len(other_meshes_colors) == 0:
            other_meshes_colors = [main_mesh_color]*len(other_meshes)
        else:
            #get the locations of all of the dictionaries
            if "random" in other_meshes_colors:
                other_meshes_colors = mu.generate_color_list(
                            user_colors=[], #if user sends a prescribed list
                            n_colors=len(other_meshes),
                            #colors_to_omit=["green","blue"], #because that is the one used for the main mesh
                            alpha_level=mesh_alpha)
            else:
                other_meshes_colors = mu.generate_color_list(
                            user_colors=other_meshes_colors, #if user sends a prescribed list
                            n_colors=len(other_meshes),
                            #colors_to_omit=["green","blue"], #because that is the one used for the main mesh
                            alpha_level=mesh_alpha)
            
    
       
    #print("Working on other meshes")
    for curr_mesh,curr_color in zip(other_meshes,other_meshes_colors):
        #print(f"flip_y = {flip_y}")
        plot_ipv_mesh(curr_mesh,color=curr_color,flip_y=flip_y)
        
        main_mesh_vertices.append(curr_mesh.vertices)
    
    
    #print("Working on other skeletons")
    if len(other_skeletons) > 0:
        if len(other_skeletons_colors) == 0:
            other_skeletons_colors = [main_skeleton_color]*len(other_skeletons)
        elif "random" in other_skeletons_colors:
            other_skeletons_colors = mu.generate_color_list(
                        user_colors=[], #if user sends a prescribed list
                        n_colors=len(other_skeletons),
                        #colors_to_omit=["green","blue"], #because that is the one used for the main mesh
                        alpha_level=1)
        else:
            
            other_skeletons_colors = mu.generate_color_list(
                        user_colors=other_skeletons_colors, #if user sends a prescribed list
                        n_colors=len(other_skeletons),
                        #colors_to_omit=["green","blue"], #because that is the one used for the main mesh
                        alpha_level=1)
            #print(f"user colors picked for other_skeletons_colors = {other_skeletons_colors}")
    
        
    for curr_sk,curr_color in zip(other_skeletons,other_skeletons_colors):
        sk_vertices = plot_ipv_skeleton(curr_sk,color=curr_color,flip_y=flip_y)
        
        main_mesh_vertices.append(sk_vertices)
        
        
    #printing the scatter plots
    #print("Working on other scatter plots")
    if len(other_scatter) > 0 and len(other_scatter_colors) == 0:
        other_scatter_colors = [main_scatter_color]*len(other_scatter)
        
    for curr_scatter,curr_color in zip(other_scatter,other_scatter_colors):
        plot_ipv_scatter(curr_scatter,scatter_color=curr_color,
                    scatter_size=scatter_size,flip_y=flip_y)
        main_mesh_vertices.append(curr_scatter)
    

    #create the main mesh vertices for setting the bounding box
    if len(main_mesh_vertices) == 0:
        raise Exception("No meshes or skeletons passed to the plotting funciton")
    elif len(main_mesh_vertices) == 1:
        main_mesh_vertices = main_mesh_vertices[0]
    else:
        #get rid of all empt
        #print(f"main_mesh_vertices = {main_mesh_vertices}")
        main_mesh_vertices = np.vstack([k.reshape(-1,3) for k in main_mesh_vertices if len(k)>0])
    
    if len(main_mesh_vertices) == 0:
        raise Exception("There is nothing to grpah")
    

    
    if flip_y:
        main_mesh_vertices = main_mesh_vertices.copy()
        main_mesh_vertices = main_mesh_vertices.reshape(-1,3)
        main_mesh_vertices[...,1] = -main_mesh_vertices[...,1]
    

    volume_max = np.max(main_mesh_vertices.reshape(-1,3),axis=0)
    volume_min = np.min(main_mesh_vertices.reshape(-1,3),axis=0)
    
#     if len(main_mesh_vertices) < 10:
#         print(f"main_mesh_vertices = {main_mesh_vertices}")
#     print(f"volume_max= {volume_max}")
#     print(f"volume_min= {volume_min}")

    ranges = volume_max - volume_min
    index = [0,1,2]
    max_index = np.argmax(ranges)
    min_limits = [0,0,0]
    max_limits = [0,0,0]


    for i in index:
        if i == max_index:
            min_limits[i] = volume_min[i] - buffer
            max_limits[i] = volume_max[i] + buffer 
            continue
        else:
            difference = ranges[max_index] - ranges[i]
            min_limits[i] = volume_min[i] - difference/2  - buffer
            max_limits[i] = volume_max[i] + difference/2 + buffer

    #ipv.xyzlim(-2, 2)
    ipv.xlim(min_limits[0],max_limits[0])
    ipv.ylim(min_limits[1],max_limits[1])
    ipv.zlim(min_limits[2],max_limits[2])
    
    
    ipv.style.set_style_light()
    if axis_box_off:
        ipv.style.axes_off()
        ipv.style.box_off()
    else:
        ipv.style.axes_on()
        ipv.style.box_on()
        
    if show_at_end:
        ipv.show()
    
    if html_path != "":
        ipv.pylab.save(html_path)
    
    if return_other_colors:
        return other_meshes_colors
        


""" ------------------- Mesh subtraction ------------------------------------"""
import numpy as np
#make sure pip3 install trimesh --upgrade so can have slice
import trimesh 
import matplotlib.pyplot as plt
import ipyvolume as ipv
import calcification_Module as cm
from pathlib import Path
import time
import skeleton_utils as sk

#  Utility functions
angle = np.pi/2
rotation_matrix = np.array([[np.cos(angle),-np.sin(angle),0],
                            [np.sin(angle),np.cos(angle),0],
                            [0,0,1]
                           ])

def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q
def change_basis_matrix(v):
    """
    This just gives change of basis matrix for a basis 
    that has the vector v as its 3rd basis vector
    and the other 2 vectors orthogonal to v 
    (but not necessarily orthogonal to each other)
    *** not make an orthonormal basis ***
    
    -- changed so now pass the non-orthogonal components
    to the QR decomposition to get them as orthogonal
    
    """
    a,b,c = v
    #print(f"a,b,c = {(a,b,c)}")
    if np.abs(c) > 0.00001:
        v_z = v/np.linalg.norm(v)
        v_x = np.array([1,0,-a/c])
        #v_x = v_x/np.linalg.norm(v_x)
        v_y = np.array([0,1,-b/c])
        #v_y = v_y/np.linalg.norm(v_y)
        v_x, v_y = gram_schmidt_columns(np.vstack([v_x,v_y]).T).T
        return np.vstack([v_x,v_y,v_z])
    else:
        #print("Z coeffienct too small")
        #v_z = v
        v[2] = 0
        #print(f"before norm v_z = {v}")
        v_z = v/np.linalg.norm(v)
        #print(f"after norm v_z = {v_z}")
        
        v_x = np.array([0,0,1])
        v_y = rotation_matrix@v_z
        
    return np.vstack([v_x,v_y,v_z])

def mesh_subtraction_by_skeleton(main_mesh,edges,
                                 buffer=0.01,
                                bbox_ratio=1.2,
                                 distance_threshold=2000,
                             significance_threshold=500,
                                print_flag=False):
    """
    Purpose: Will return significant mesh pieces that are
    not already accounteed for by the skeleton
    
    Example of how to run
    
    main_mesh_path = Path("./Dustin/Dustin.off")
    main_mesh = trimesh.load_mesh(str(main_mesh_path.absolute()))
    skeleton_path = main_mesh_path.parents[0] / Path(main_mesh_path.stem + "_skeleton.cgal")
    edges = sk.read_skeleton_edges_coordinates(str(skeleton_path.absolute()))

    # turn this into nodes and edges
    main_mesh_nodes, main_mesh_edges = sk.read_skeleton_verts_edges(str(skeleton_path.absolute()))
    sk.graph_skeleton_and_mesh(
                main_mesh_verts=main_mesh.vertices,
                main_mesh_faces=main_mesh.faces,
                unique_skeleton_verts_final = main_mesh_nodes,
                edges_final=main_mesh_edges,
                buffer = 0
                              )
                              
    leftover_pieces =  mesh_subtraction_by_skeleton(main_mesh,edges,
                                 buffer=0.01,
                                bbox_ratio=1.2,
                                 distance_threshold=500,
                             significance_threshold=500,
                                print_flag=False)
                                
    # Visualize the results: 
    pieces_mesh = trimesh.Trimesh(vertices=np.array([]),
                                 faces=np.array([]))

    for l in leftover_pieces:
        pieces_mesh += l

    sk.graph_skeleton_and_mesh(
                main_mesh_verts=pieces_mesh.vertices,
                main_mesh_faces=pieces_mesh.faces,
                unique_skeleton_verts_final = main_mesh_nodes,
                edges_final=main_mesh_edges,
                buffer = 0
                              )
    
    """
    
    skeleton_nodes = edges.reshape(-1,3)
    skeleton_bounding_corners = np.vstack([np.max(skeleton_nodes,axis=0),
               np.min(skeleton_nodes,axis=0)])
    
    main_mesh_bbox_restricted, faces_bbox_inclusion = tu.bbox_mesh_restriction(main_mesh,
                                                                        skeleton_bounding_corners,
                                                                        bbox_ratio)

    if type(main_mesh_bbox_restricted) == type(trimesh.Trimesh()):
        print(f"Inside mesh subtraction, len(main_mesh_bbox_restricted.faces) = {len(main_mesh_bbox_restricted.faces)}")
    else:
        print("***** Bounding Box Restricted Mesh is empty ****")
        main_mesh_bbox_restricted = main_mesh
        faces_bbox_inclusion = np.arange(0,len(main_mesh.faces))
    
    start_time = time.time()

    #face_subtract_color = []
    face_subtract_indices = []

    #distance_threshold = 2000
    
    edge_loop_print=False
    for i,ex_edge in tqdm(enumerate(edges)):
        #print("\n------ New loop ------")
        #print(ex_edge)
        
        # ----------- creating edge and checking distance ----- #
        loop_start = time.time()
        
        edge_line = ex_edge[1] - ex_edge[0]
        sum_threshold = 0.001
        if np.sum(np.abs(edge_line)) < sum_threshold:
            if edge_loop_print:
                print(f"edge number {i}, {ex_edge}: has sum less than {sum_threshold} so skipping")
            continue
#         if edge_loop_print:
#             print(f"Checking Edge Distance = {time.time()-loop_start}")
#         loop_start = time.time()
        
        cob_edge = change_basis_matrix(edge_line)
        
#         if edge_loop_print:
#             print(f"Change of Basis Matrix calculation = {time.time()-loop_start}")
#         loop_start - time.time()
        
        #get the limits of the example edge itself that should be cutoff
        edge_trans = (cob_edge@ex_edge.T)
        #slice_range = np.sort((cob_edge@ex_edge.T)[2,:])
        slice_range = np.sort(edge_trans[2,:])

        # adding the buffer to the slice range
        slice_range_buffer = slice_range + np.array([-buffer,buffer])
        
#         if edge_loop_print:
#             print(f"Calculate slice= {time.time()-loop_start}")
#         loop_start = time.time()

        # generate face midpoints from the triangles
        #face_midpoints = np.mean(main_mesh_bbox_restricted.vertices[main_mesh_bbox_restricted.faces],axis=1) # Old way
        face_midpoints = main_mesh_bbox_restricted.triangles_center
        
#         if edge_loop_print:
#             print(f"Face midpoints= {time.time()-loop_start}")
#         loop_start = time.time()
        
        #get the face midpoints that fall within the slice (by lookig at the z component)
        fac_midpoints_trans = cob_edge@face_midpoints.T
        
#         if edge_loop_print:
#             print(f"Face midpoints transform= {time.time()-loop_start}")
#         loop_start = time.time()
        
        
        
#         if edge_loop_print:
#             print(f"edge midpoint= {time.time()-loop_start}")
#         loop_start = time.time()
        
        slice_mask_pre_distance = ((fac_midpoints_trans[2,:]>slice_range_buffer[0]) & 
                      (fac_midpoints_trans[2,:]<slice_range_buffer[1]))

#         if edge_loop_print:
#             print(f"Applying slice restriction = {time.time()-loop_start}")
#         loop_start = time.time()
        
        
        """ 6/18 change
        # apply the distance threshold to the slice mask
        edge_midpoint = np.mean(ex_edge,axis=0)
        #raise Exception("Add in part for distance threshold here")
        distance_check = np.linalg.norm(face_midpoints[:,:2] - edge_midpoint[:2],axis=1) < distance_threshold
        
        """
        
#         edge_midpoint = np.mean(cob_edge.T,axis=0)
#         distance_check = np.linalg.norm((fac_midpoints_trans.T)[:,:2] - edge_midpoint[:2],axis=1) < distance_threshold

        edge_midpoint = np.mean(edge_trans.T,axis=0)
        distance_check = np.linalg.norm((fac_midpoints_trans.T)[:,:2] - edge_midpoint[:2],axis=1) < distance_threshold
        

        slice_mask = slice_mask_pre_distance & distance_check
        
#         if edge_loop_print:
#             print(f"Applying distance restriction= {time.time()-loop_start}")
#         loop_start = time.time()


        face_list = np.arange(0,len(main_mesh_bbox_restricted.faces))[slice_mask]

        #get the submesh of valid faces within the slice
        if len(face_list)>0:
            main_mesh_sub = main_mesh_bbox_restricted.submesh([face_list],append=True)
        else:
            main_mesh_sub = []
        
        

        if type(main_mesh_sub) != type(trimesh.Trimesh()):
            if edge_loop_print:
                print(f"THERE WERE NO FACES THAT FIT THE DISTANCE ({distance_threshold}) and Z transform requirements")
                print("So just skipping this edge")
            continue

#         if edge_loop_print:
#             print(f"getting submesh= {time.time()-loop_start}")
#         loop_start = time.time()
        
        #get all disconnected mesh pieces of the submesh and the face indices for lookup later
        sub_components,sub_components_face_indexes = tu.split(main_mesh_sub,only_watertight=False)
        if type(sub_components) != type(np.array([])) and type(sub_components) != list:
            #print(f"meshes = {sub_components}, with type = {type(sub_components)}")
            if type(sub_components) == type(trimesh.Trimesh()) :
                sub_components = [sub_components]
            else:
                raise Exception("The sub_components were not an array, list or trimesh")
        
#         if edge_loop_print:
#             print(f"splitting the mesh= {time.time()-loop_start}")
#         loop_start = time.time()

        #getting the indices of the submeshes whose bounding box contain the edge 
        """ 6-19: might want to use bounding_box_oriented? BUT THIS CHANGE COULD SLOW IT DOWN
        contains_points_results = np.array([s_comp.bounding_box_oriented.contains(ex_edge.reshape(-1,3)) for s_comp in sub_components])
        """
        contains_points_results = np.array([s_comp.bounding_box.contains(ex_edge.reshape(-1,3)) for s_comp in sub_components])
        
        containing_indices = (np.arange(0,len(sub_components)))[np.sum(contains_points_results,axis=1) >= len(ex_edge)]
        
#         if edge_loop_print:
#             print(f"containing indices= {time.time()-loop_start}")
#         loop_start = time.time()

        try:
            if len(containing_indices) != 1: 
                if edge_loop_print:
                    print(f"--> Not exactly one containing mesh: {containing_indices}")
                if len(containing_indices) > 1:
                    sub_components_inner = sub_components[containing_indices]
                    sub_components_face_indexes_inner = sub_components_face_indexes[containing_indices]
                else:
                    sub_components_inner = sub_components
                    sub_components_face_indexes_inner = sub_components_face_indexes

                #get the center of the edge
                edge_center = np.mean(ex_edge,axis=0)
                #print(f"edge_center = {edge_center}")

                #find the distance between eacch bbox center and the edge center
                bbox_centers = [np.mean(k.bounds,axis=0) for k in sub_components_inner]
                #print(f"bbox_centers = {bbox_centers}")
                closest_bbox = np.argmin([np.linalg.norm(edge_center-b_center) for b_center in bbox_centers])
                #print(f"bbox_distance = {closest_bbox}")
                edge_skeleton_faces = faces_bbox_inclusion[face_list[sub_components_face_indexes_inner[closest_bbox]]]

    #             if edge_loop_print:
    #                 print(f"finding closest box when 0 or 2 or more containing boxes= {time.time()-loop_start}")
    #             loop_start = time.time()
            elif len(containing_indices) == 1:# when only one viable submesh piece and just using that sole index
                #print(f"only one viable submesh piece so using index only number in: {containing_indices}")

                edge_skeleton_faces = faces_bbox_inclusion[face_list[sub_components_face_indexes[containing_indices[0]].astype("int")]]
    #             if edge_loop_print:
    #                 print(f"only 1 containig face getting the edge_skeleton_faces= {time.time()-loop_start}")
    #             loop_start = time.time()
            else:
                raise Exception("No contianing indices")
        except:
            import system_utils as su
            su.compressed_pickle(main_mesh_sub,"main_mesh_sub")
            su.compressed_pickle(ex_edge,"ex_edge")
            su.compressed_pickle(sub_components_face_indexes,"sub_components_face_indexes")
            su.compressed_pickle(containing_indices,"containing_indices")
            su.compressed_pickle(face_list,"face_list")
            su.compressed_pickle(faces_bbox_inclusion,"faces_bbox_inclusion")
            
            raise Exception("Indexing not work in mesh subtraction")

            
            
            
            

        if len(edge_skeleton_faces) < 0:
            print(f"****** Warning the edge index {i}: had no faces in the edge_skeleton_faces*******")
        face_subtract_indices.append(edge_skeleton_faces)
#         if edge_loop_print:
#                 print(f"check and append for face= {time.time()-loop_start}")
        #face_subtract_color.append(viable_colors[i%len(viable_colors)])
        
    print(f"Total Mesh subtraction time = {np.round(time.time() - start_time,4)}")
    
    if len(face_subtract_indices)>0:
        all_removed_faces = np.concatenate(face_subtract_indices)

        unique_removed_faces = set(all_removed_faces)

        faces_to_keep = set(np.arange(0,len(main_mesh.faces))).difference(unique_removed_faces)
        new_submesh = main_mesh.submesh([list(faces_to_keep)],only_watertight=False,append=True)
    else:
        new_submesh = main_mesh
    
    significant_pieces = split_significant_pieces(new_submesh,
                                                         significance_threshold,
                                                         print_flag=False)


    return significant_pieces

""" ------------------- End of Mesh Subtraction ------------------------------------"""



""" ----------Start of Surface Skeeltonization -- """

import networkx as nx
import time 
import numpy as np
import trimesh
import random


# # Older version that was not working properly
# def generate_surface_skeleton(vertices,
#                               faces, 
#                               surface_samples=1000,
#                           print_flag=False):
    
#     #return surface_with_poisson_skeleton,path_length
    
#     mesh = trimesh.Trimesh(vertices=vertices,
#                                   faces = faces,
#                            )


#     start_time = time.time()

#     ga = nx.from_edgelist(mesh.edges)

#     if surface_samples<len(vertices):
#         k = surface_samples
#     else:
#         k = len(vertices)
#     sampled_nodes = random.sample(ga.nodes, k)


#     lp_end_list = []
#     lp_magnitude_list = []

#     for s in sampled_nodes: 
#         sp_dict = nx.single_source_shortest_path_length(ga,s)

#         list_keys = list(sp_dict.keys())
#         longest_path_node = list_keys[len(list_keys)-1]
#         longest_path_magnitude = sp_dict[longest_path_node]


#         lp_end_list.append(longest_path_node)
#         lp_magnitude_list.append(longest_path_magnitude)

#     #construct skeleton from shortest path
#     final_start = sampled_nodes[np.argmax(lp_magnitude_list)]
#     final_end = sampled_nodes[np.argmax(lp_end_list)]

#     node_list = nx.shortest_path(ga,final_start,final_end)
#     if len(node_list) < 2:
#         print("node_list len < 2 so returning empty list")
#         return np.array([])
#     #print("node_list = " + str(node_list))

#     final_skeleton = mesh.vertices[np.vstack([node_list[:-1],node_list[1:]]).T]
#     if print_flag:
#         print(f"   Final Time for surface skeleton with sample size = {k} = {time.time() - start_time}")

#     return final_skeleton


def generate_surface_skeleton_slower(vertices,
                              faces, 
                              surface_samples=1000,
                              n_surface_downsampling=0,
                          print_flag=False):
    """
    Purpose: Generates a surface skeleton without using
    the root method and instead just samples points
    """
    
    #return surface_with_poisson_skeleton,path_length
    
    mesh = trimesh.Trimesh(vertices=vertices,
                                  faces = faces,
                           )


    start_time = time.time()

    ga = nx.from_edgelist(mesh.edges)

    if surface_samples<len(vertices):
        sampled_nodes = np.random.choice(len(vertices),surface_samples , replace=False)
    else:
        if print_flag:
            print("Number of surface samples exceeded number of vertices, using len(vertices)")
        sampled_nodes = np.arange(0,len(vertices))
        
    lp_end_list = []
    lp_magnitude_list = []

    for s in sampled_nodes: 
        #gives a dictionary where the key is the end node and the value is the number of
        # edges on the shortest path
        sp_dict = nx.single_source_shortest_path_length(ga,s)

        #
        list_keys = list(sp_dict.keys())
        
        #gets the end node that would make the longest shortest path 
        longest_path_node = list_keys[-1]
        
        #get the number of edges for the path
        longest_path_magnitude = sp_dict[longest_path_node]


        #add the ending node and the magnitude of it to lists
        lp_end_list.append(longest_path_node)
        lp_magnitude_list.append(longest_path_magnitude)

    lp_end_list = np.array(lp_end_list)
    #construct skeleton from shortest path
    max_index = np.argmax(lp_magnitude_list)
    final_start = sampled_nodes[max_index]
    final_end = lp_end_list[max_index]

    node_list = nx.shortest_path(ga,final_start,final_end)
    if len(node_list) < 2:
        print("node_list len < 2 so returning empty list")
        return np.array([])
    #print("node_list = " + str(node_list))

    final_skeleton = mesh.vertices[np.vstack([node_list[:-1],node_list[1:]]).T]
    if print_flag:
        print(f"   Final Time for surface skeleton with sample size = {k} = {time.time() - start_time}")
        
    for i in range(n_surface_downsampling):
        final_skeleton = downsample_skeleton(final_skeleton)

    return final_skeleton

import meshparty

from meshparty_skeletonize import *
def setup_root(mesh, is_soma_pt=None, soma_d=None, is_valid=None):
    """ function to find the root index to use for this mesh
    
    Purpose: The output will be used to find the path for a 
    surface skeletonization (aka: longest shortest path)
    
    The output:
    1) root: One of the end points
    2) target: The other endpoint:
    3) root_ds: (N,) matrix of distances from root to all other vertices
    4) : predecessor matrix for root to shortest paths of all other vertices
    --> used to find surface path
    5) valid: boolean mask (NOT USED)
    
    """
    if is_valid is not None:
        valid = np.copy(is_valid)
    else:
        valid = np.ones(len(mesh.vertices), np.bool)
    assert(len(valid) == mesh.vertices.shape[0])

    root = None
    # soma mode
    if is_soma_pt is not None:
        # pick the first soma as root
        assert(len(soma_d) == mesh.vertices.shape[0])
        assert(len(is_soma_pt) == mesh.vertices.shape[0])
        is_valid_root = is_soma_pt & valid
        valid_root_inds = np.where(is_valid_root)[0]
        if len(valid_root_inds) > 0:
            min_valid_root = np.nanargmin(soma_d[valid_root_inds])
            root = valid_root_inds[min_valid_root]
            root_ds, pred = sparse.csgraph.dijkstra(mesh.csgraph,
                                                    directed=False,
                                                    indices=root,
                                                    return_predecessors=True)
        else:
            start_ind = np.where(valid)[0][0]
            root, target, pred, dm, root_ds = utils.find_far_points(mesh,
                                                                    start_ind=start_ind)
        valid[is_soma_pt] = False

    if root is None:
        # there is no soma close, so use far point heuristic
        start_ind = np.where(valid)[0][0]
        root, target, pred, dm, root_ds = utils.find_far_points(
            mesh, start_ind=start_ind)
    valid[root] = False
    assert(np.all(~np.isinf(root_ds[valid])))
    return root, target,root_ds, pred, valid

def generate_surface_skeleton(vertices,
                              faces, 
                              surface_samples=1000,
                              n_surface_downsampling=0,
                          print_flag=False):
    """
    Purpose: To generate a surface skeleton
    
    Specifics: New implementation that uses meshparty 
    method of finding root that optimally finds 
    longest shortest path
    
    """
    
    meshparty_skeleton_time = time.time()
    branch_obj_tr_io  = meshparty.trimesh_io.Mesh(vertices = vertices,
                                   faces=faces)
    
    root, target,root_ds, root_pred, valid = setup_root(branch_obj_tr_io)

    current_path = utils.get_path(root,target,root_pred)

    surface_sk_edges = np.vstack([current_path[:-1],current_path[1:]]).T
    meshparty_surface_skeleton = branch_obj_tr_io.vertices[surface_sk_edges]
    
    if print_flag: 
        print(f"Total time for surface skeletonization = {time.time() - meshparty_skeleton_time}")
    
    for i in range(n_surface_downsampling):
        meshparty_surface_skeleton = downsample_skeleton(meshparty_surface_skeleton)
    
    return meshparty_surface_skeleton


def downsample_skeleton(current_skeleton):
    #print("current_skeleton = " + str(current_skeleton.shape))
    """
    Downsamples the skeleton by 50% number of edges
    """
    extra_segment = []
    if current_skeleton.shape[0] % 2 != 0:
        extra_segment = np.array([current_skeleton[0]])
        current_skeleton = current_skeleton[1:]
        #print("extra_segment = " + str(extra_segment))
        #print("extra_segment.shape = " + str(extra_segment.shape))
    else:
        #print("extra_segment = " + str(extra_segment))
        pass

    even_indices = [k for k in range(0,current_skeleton.shape[0]) if k%2 == 0]
    odd_indices = [k for k in range(0,current_skeleton.shape[0]) if k%2 == 1]
    even_verts = current_skeleton[even_indices,0,:]
    odd_verts = current_skeleton[odd_indices,1,:]

    downsampled_skeleton = np.hstack([even_verts,odd_verts]).reshape(even_verts.shape[0],2,3)
    #print("dowsampled_skeleton.shape = " + str(downsampled_skeleton.shape))
    if len(extra_segment) > 0:
        #print("downsampled_skeleton = " + str(downsampled_skeleton.shape))
        final_downsampled_skeleton = np.vstack([extra_segment,downsampled_skeleton])
    else:
        final_downsampled_skeleton = downsampled_skeleton
    return final_downsampled_skeleton


# ----- Stitching Algorithm ----- #
import networkx as nx

from pykdtree.kdtree import KDTree

import scipy
def stitch_skeleton(
                                          staring_edges,
                                          max_stitch_distance=18000,
                                          stitch_print = False,
                                          main_mesh = []
                                        ):

    stitched_time = time.time()

    stitch_start = time.time()

    all_skeleton_vertices = staring_edges.reshape(-1,3)
    unique_rows,indices = np.unique(all_skeleton_vertices,return_inverse=True,axis=0)
    edges_with_coefficients = indices.reshape(-1,2)

    if stitch_print:
        print(f"Getting the unique rows and indices= {time.time()-stitch_start}")
    stitch_start = time.time()

    #create the graph from the edges
    #B = nx.Graph() #old way
    B = xu.GraphOrderedEdges()
    B.add_nodes_from([(x,{"coordinates":y}) for x,y in enumerate(unique_rows)])
    
    
    B.add_edges_from(edges_with_coefficients)
    
    if stitch_print:
        print(f"Putting edges into networkx graph= {time.time()-stitch_start}")
    stitch_start = time.time()

    # find the shortest distance between the two different subgraphs:
    from scipy.spatial import distance_matrix

    # UG = B.to_undirected() #no longer need this
    UG = B
    
    UG.edges_ordered()
    
    if stitch_print:
        print(f"Making undirected graph= {time.time()-stitch_start}")
    stitch_start = time.time()

    #get all of the coordinates

    print("len_subgraphs AT BEGINNING of the loop")
    counter = 0
    print_flag = True

    n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=nx.adjacency_matrix(UG), directed=False, return_labels=True)
    #print(f"At beginning n_components = {n_components}, unique labels = {np.unique(labels)}")
    
    
    
    subgraph_components = np.where(labels==0)[0]
    outside_components = np.where(labels !=0)[0]

    for j in tqdm(range(n_components)):
        
        counter+= 1
        if stitch_print:
            print(f"Starting Loop {counter}")
        start_time = time.time()
        """
        1) Get the indexes of the subgraph
        2) Build a KDTree from those not in the subgraph (save the vertices of these)
        3) Query against the nodes in the subgraph  and get the smallest distance
        4) Create this new edge

        """
        stitch_time = time.time()
        #1) Get the indexes of the subgraph
        #n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=nx.adjacency_matrix(UG), directed=False, return_labels=True)
        if stitch_print:
            print(f"Finding Number of Connected Components= {time.time()-stitch_start}")
        stitch_start = time.time()

        subgraph_components = np.where(labels==0)[0]

        if stitch_print:
            print(f"Faces belonging to largest component= {time.time()-stitch_start}")
        stitch_start = time.time()
        #print("subgraph_components = " + str(subgraph_components))
        if len(subgraph_components) == len(UG.nodes):
            print("all graph is one component!")
            #print(f"unique labels = {np.unique(labels)}")
            break

        if stitch_print:
            print(f"n_components = {n_components}")

        outside_components = np.where(labels !=0)[0]

        if stitch_print:
            print(f"finding faces not in largest component= {time.time()-stitch_start}")
        stitch_start = time.time()
        #print("outside_components = " + str(outside_components))

        #2) Build a KDTree from those not in the subgraph (save the vertices of these)
        mesh_tree = KDTree(unique_rows[outside_components])
        if stitch_print:
            print(f"Building KDTree= {time.time()-stitch_start}")
        stitch_start = time.time()


        #3) Query against the nodes in the subgraph  and get the smallest distance
        """
        Conclusion:
        Distance is of the size of the parts that are in the KDTree
        The closest nodes represent those that were queryed

        """
        distances,closest_node = mesh_tree.query(unique_rows[subgraph_components])
        if stitch_print:
            print(f"Mesh Tree query= {time.time()-stitch_start}")
        stitch_start = time.time()
        min_index = np.argmin(distances)
        
        #check if the distance is too far away 
        if distances[min_index] > max_stitch_distance:
            print(f"**** The distance exceeded max stitch distance of {max_stitch_distance}"
                   f" and still had {n_components} left\n"
                  f"   Actual distance was {distances[min_index]} ")
        

        if stitch_print:
            print(f"Finding closest distance= {time.time()-stitch_start}")
        stitch_start = time.time()


        closest_outside_node = outside_components[closest_node[min_index]]
        closest_subgraph_node = subgraph_components[min_index]

        if stitch_print:
            print(f"Getting nodes to be paired up= {time.time()-stitch_start}")
        stitch_start = time.time()

        
        
        #get the edge distance of edge about to create:

    #         graph_coordinates=nx.get_node_attributes(UG,'coordinates')
    #         prospective_edge_length = np.linalg.norm(np.array(graph_coordinates[closest_outside_node])-np.array(graph_coordinates[closest_subgraph_node]))
    #         print(f"Edge distance going to create = {prospective_edge_length}")

        #4) Create this new edge
        UG.add_edge(closest_subgraph_node,closest_outside_node)

        #get the label of the closest outside node 
        closest_outside_label = labels[closest_outside_node]

        #get all of the nodes with that same label
        new_subgraph_components = np.where(labels==closest_outside_label)[0]

        #relabel the nodes so now apart of the largest component
        labels[new_subgraph_components] = 0

        #move the newly relabeled nodes out of the outside components into the subgraph components
        ## --- SKIP THIS ADDITION FOR RIGHT NOW -- #


        if stitch_print:
            print(f"Adding Edge = {time.time()-stitch_start}")
        stitch_start = time.time()

        n_components -= 1

        if stitch_print:
            print(f"Total Time for loop = {time.time() - start_time}")


    # get the largest subgraph!!! in case have missing pieces

    #add all the new edges to the 

#     total_coord = nx.get_node_attributes(UG,'coordinates')
#     current_coordinates = np.array(list(total_coord.values()))

    current_coordinates = unique_rows
    
    try:
        #total_edges_stitched = current_coordinates[np.array(list(UG.edges())).reshape(-1,2)] #old way of edges
        total_edges_stitched = current_coordinates[UG.edges_ordered().reshape(-1,2)]
    except:
        print("getting the total edges stitched didn't work")
        print(f"current_coordinates = {current_coordinates}")
        print(f"UG.edges_ordered() = {list(UG.edges_ordered())} with type = {type(list(UG.edges_ordered()))}")
        print(f"np.array(UG.edges_ordered()) = {UG.edges_ordered()}")
        print(f"np.array(UG.edges_ordered()).reshape(-1,2) = {UG.edges_ordered().reshape(-1,2)}")
        
        raise Exception(" total_edges_stitched not calculated")
        #print("returning ")
        #total_edges_stitched
    

    print(f"Total time for skeleton stitching = {time.time() - stitched_time}")
    
    return total_edges_stitched


def stack_skeletons(sk_list,graph_cleaning=False):
    list_of_skeletons = [np.array(k).reshape(-1,2,3) for k in sk_list if len(k)>0]
    if len(list_of_skeletons) == 0:
        print("No skeletons to stack so returning empty list")
        return []
    elif len(list_of_skeletons) == 1:
        #print("only one skeleton so no stacking needed")
        return np.array(list_of_skeletons).reshape(-1,2,3)
    else:
        final_sk = (np.vstack(list_of_skeletons)).reshape(-1,2,3)
        if graph_cleaning:
            final_sk = sk.convert_graph_to_skeleton(sk.convert_skeleton_to_graph(final_sk))
        return final_sk

#------------ The actual skeletonization from mesh contraction----------- #
from calcification_param_Module import calcification_param
def calcification(
                    location_with_filename,
                    max_triangle_angle =1.91986,
                    quality_speed_tradeoff=0.2,#0.1,
                    medially_centered_speed_tradeoff=0.2,#0.2,
                    area_variation_factor=0.0001,#0.0001,
                    max_iterations=500,#500,
                    is_medially_centered=True,
                    min_edge_length = 75,
                    edge_length_multiplier = 0.002,
                    print_parameters=True
                ):
    
    if type(location_with_filename) == type(Path()):
        location_with_filename = str(location_with_filename.absolute())
    
    if location_with_filename[-4:] == ".off":
        location_with_filename = location_with_filename[:-4]
    
    #print(f"location_with_filename = {location_with_filename}")
    print(f"min_edge_length = {min_edge_length}")
    
    return_value = calcification_param(
        location_with_filename,
        max_triangle_angle,
        quality_speed_tradeoff,
        medially_centered_speed_tradeoff,
        area_variation_factor,
        max_iterations,
        is_medially_centered,
        min_edge_length,
        edge_length_multiplier,
        print_parameters
    )
    
    return return_value,location_with_filename+"_skeleton.cgal"

def skeleton_cgal(mesh=None,
                  mesh_path=None,
                  filepath = "./temp.off",
                  remove_temp_files=True,
                  verbose=False,
                  **kwargs):
    """
    Pseudocode: 
    1) Write the mesh to a file
    2) Pass the file to the calcification
    3) Delete the temporary file
    """
    #1) Write the mesh to a file
    if not mesh is None:
        written_path = write_neuron_off(mesh,filepath)
    else:
        written_path = mesh_path
    
    #2) Pass the file to the calcification
    sk_time = time.time()
    skeleton_results,sk_file = calcification(written_path,**kwargs)
    
    significant_poisson_skeleton = read_skeleton_edges_coordinates([sk_file])
    
    #3) Delete the temporary file
    if remove_temp_files:
        if mesh_path is None:
            Path(written_path).unlink()
        Path(sk_file).unlink()
    
    if verbose:
        print(f"Total time for skeletonizing {time.time() - sk_time}")
        print(f"Returning skeleton of size {significant_poisson_skeleton.shape}")
    
    return significant_poisson_skeleton
    


# ---------- Does the cleaning of the skeleton -------------- #

#old way that didnt account for the nodes that are close together
def convert_skeleton_to_graph_old(staring_edges,
                             stitch_print=False):
    stitch_start = time.time()

    all_skeleton_vertices = staring_edges.reshape(-1,3)
    unique_rows,indices = np.unique(all_skeleton_vertices,return_inverse=True,axis=0)
    #need to merge unique indices so if within a certain range of each other then merge them together
    
    edges_with_coefficients = indices.reshape(-1,2)

    if stitch_print:
        print(f"Getting the unique rows and indices= {time.time()-stitch_start}")
    stitch_start = time.time()

    #create the graph from the edges
    #B = nx.Graph() #old way
    B = xu.GraphOrderedEdges()
    B.add_nodes_from([(int(x),{"coordinates":y}) for x,y in enumerate(unique_rows)])
    #print("just added the nodes")
    
    #print(f"xu.get_edge_attributes(B)= {xu.get_edge_attributes(B)}")
    
    #B.add_edges_from(edges_with_coefficients) #older weights without weights
    #adds weights for the edges
    weights = np.linalg.norm(unique_rows[edges_with_coefficients[:,0]] - unique_rows[edges_with_coefficients[:,1]],axis=1)
    edges_with_weights = np.hstack([edges_with_coefficients,weights.reshape(-1,1)])
    B.add_weighted_edges_from(edges_with_weights)
    #print("right after add_weighted_edges_from")
    #print(f"xu.get_edge_attributes(B)= {xu.get_edge_attributes(B)}")

    print(f"len(B.edges()) = {len(B.edges())}")
    if stitch_print:
        print(f"Putting edges into networkx graph= {time.time()-stitch_start}")
    stitch_start = time.time()

    # find the shortest distance between the two different subgraphs:
    from scipy.spatial import distance_matrix
    #print(f"B.__class__ = {B.__class__}")
    #print(f"xu.get_edge_attributes(B)= {xu.get_edge_attributes(B)}")
    UG = B
    #UG = B.to_undirected()
    
    if stitch_print:
        print(f"Making undirected graph= {time.time()-stitch_start}")
    stitch_start = time.time()
    
    #UG.remove_edges_from(nx.selfloop_edges(UG))
    UG = xu.remove_selfloops(UG)
    print(f"len(UG.edges()) = {len(UG.edges())}")
    #print(f"UG.__class__ = {UG.__class__}")
    #make sure the edges are ordered 
    UG.reorder_edges()
    print(f"len(UG.edges()) = {len(UG.edges())}")
    #print(f"UG.__class__ = {UG.__class__}")
    return UG


def convert_skeleton_to_graph(staring_edges,
                             stitch_print=False,
                                   combine_node_dist = 0.0001,
                             node_matching_size_threshold=10000):
    """
    Purpose: To automatically convert a skeleton to a graph
    
    * 7/9 adjustments: make so slight differences in coordinates not affect the graph
    
    Pseudocode for how you could apply the closeness to skeletons
    1) Get the unique rows
    2) See if there are any rows that are the same (that gives you what to change them to)
    3) put those into a graph and find the connected components
    4) Pick the first node in the component to be the dominant one
    a. for each non-dominant node, replace every occurance of the non-dominant one with the dominant one in indices
    - add the non-dominant ones to a list to delete 
    
    ** this will result in an indices that doesn't have any of the repeat nodes, but the repeat nodes are still 
    taking up the numbers that they were originally in order with ***
    
    np.delete(x,[1,3],axis=0)) # to delete the rows 
    
    5) remap the indices and delete the unique rows that were not used
    
    
    5) Do everything like normal

"""
    stitch_start = time.time()

    all_skeleton_vertices = staring_edges.reshape(-1,3)
    unique_rows,indices = np.unique(all_skeleton_vertices,return_inverse=True,axis=0)
    
    #need to merge unique indices so if within a certain range of each other then merge them together
    reshaped_indices = indices.reshape(-1,2)
    
    #This is fine because know there might be but fix it later on! (in terms of concept graph)
    if len(reshaped_indices) != len(np.unique(reshaped_indices,axis=0)):
        print("**** Warning: There were redundant edges in the skeleton*****")
    
    #part where will combine nodes that are very close
    
    #only do this if small enough, if too big then must skip (because will get memory error)
    if len(unique_rows) < node_matching_size_threshold:
    
        matching_nodes = nu.get_matching_vertices(unique_rows,equiv_distance=combine_node_dist)

        if len(matching_nodes) > 0:
            """
            Overall this loop will change the unique_rows and indices to account for nodes that should be merged
            """
            # Example graph for finding components
            ex_edges = matching_nodes.reshape(-1,2)
            ex_graph = nx.from_edgelist(ex_edges)


            #get the connected components
            all_conn_comp = list(nx.connected_components(ex_graph))

            to_delete_nodes = []
            for c_comp in all_conn_comp:
                curr_comp = list(c_comp)
                dom_node = curr_comp[0]
                non_dom_nodes = curr_comp[1:]
                for n_dom in non_dom_nodes:
                    indices[indices==n_dom] = dom_node
                    to_delete_nodes.append(n_dom)

            unique_leftovers = np.sort(np.unique(indices.ravel()))
            #construct a dictionary for mapping
            map_dict = dict([(v,k) for k,v in enumerate(unique_leftovers)])

            print(f"Gettng rid of {len(to_delete_nodes)} nodes INSIDE SKELETON TO GRAPH CONVERSION")

            def vec_translate(a):    
                return np.vectorize(map_dict.__getitem__)(a)

            indices = vec_translate(indices)

            #now delete the rows that were ignored
            unique_rows = np.delete(unique_rows,to_delete_nodes,axis=0)

            #do a check to make sure everything is working
            if len(np.unique(indices.ravel())) != len(unique_rows) or max(np.unique(indices.ravel())) != len(unique_rows) - 1:
                raise Exception("The indices list does not match the size of the unique rows"
                               f"np.unique(indices.ravel()) = {np.unique(indices.ravel())}, len(unique_rows)= {len(unique_rows) }")
    
    #resume regular conversion
    edges_with_coefficients = indices.reshape(-1,2)
    
    

    if stitch_print:
        print(f"INSIDE CONVERT_SKELETON_TO_GRAPH Getting the unique rows and indices= {time.time()-stitch_start}")
    stitch_start = time.time()

    #create the graph from the edges
    #B = nx.Graph() #old way
    B = xu.GraphOrderedEdges()
    B.add_nodes_from([(int(x),{"coordinates":y}) for x,y in enumerate(unique_rows)])
    #print("just added the nodes")
    #print(f"xu.get_edge_attributes(B)= {xu.get_edge_attributes(B)}")
    
    #B.add_edges_from(edges_with_coefficients) #older weights without weights
    #adds weights for the edges
    weights = np.linalg.norm(unique_rows[edges_with_coefficients[:,0]] - unique_rows[edges_with_coefficients[:,1]],axis=1)
    edges_with_weights = np.hstack([edges_with_coefficients,weights.reshape(-1,1)])
    B.add_weighted_edges_from(edges_with_weights)
    #print("right after add_weighted_edges_from")
    #print(f"xu.get_edge_attributes(B)= {xu.get_edge_attributes(B)}")

    if stitch_print:
        print(f"Putting edges into networkx graph= {time.time()-stitch_start}")
    stitch_start = time.time()

    # find the shortest distance between the two different subgraphs:
    from scipy.spatial import distance_matrix
    #print(f"B.__class__ = {B.__class__}")
    #print(f"xu.get_edge_attributes(B)= {xu.get_edge_attributes(B)}")
    UG = B
    #UG = B.to_undirected()
    
    if stitch_print:
        print(f"Making undirected graph= {time.time()-stitch_start}")
    stitch_start = time.time()
    
    #UG.remove_edges_from(nx.selfloop_edges(UG))
    UG = xu.remove_selfloops(UG)
    #print(f"UG.__class__ = {UG.__class__}")
    #make sure the edges are ordered 
    UG.reorder_edges()
    #print(f"UG.__class__ = {UG.__class__}")
    return UG


def convert_graph_to_skeleton(UG):
    UG = nx.convert_node_labels_to_integers(UG)
    total_coord = nx.get_node_attributes(UG,'coordinates')
    current_coordinates = np.array(list(total_coord.values()))
    
    try:
        #total_edges_stitched = current_coordinates[np.array(list(UG.edges())).reshape(-1,2)] # old way
        total_edges_stitched = current_coordinates[UG.edges_ordered().reshape(-1,2)]
    except:
        UG.edges_ordered()
        print("getting the total edges stitched didn't work")
        print(f"current_coordinates = {current_coordinates}")
        print(f"UG.edges() = {UG.edges_ordered()} with type = {type(UG.edges_ordered)}")
        print(f"np.array(UG.edges()) = {UG.edges_ordered()}")
        print(f"np.array(UG.edges()).reshape(-1,2) = {UG.edges_ordered().reshape(-1,2)}")
        
        raise Exception(" total_edges_stitched not calculated")
        
    return total_edges_stitched

def list_len_measure(curr_list,G):
    return len(curr_list)

def skeletal_distance(curr_list,G,coordinates_dict):
    
    #clean_time = time.time()
    #coordinates_dict = nx.get_node_attributes(G,'coordinates')
    #print(f"Extracting attributes = {time.time() - clean_time}")
    #clean_time = time.time()
    coor = [coordinates_dict[k] for k in curr_list]
    #print(f"Reading dict = {time.time() - clean_time}")
    #clean_time = time.time()
    norm_values =  [np.linalg.norm(coor[i] - coor[i-1]) for i in range(1,len(coor))]
    #print(f"Calculating norms = {time.time() - clean_time}")
    #print(f"norm_values = {norm_values}")
    return np.sum(norm_values)

def clean_skeleton(G,
                   distance_func,
                  min_distance_to_junction = 3,
                  return_skeleton=True,
                   endpoints_must_keep = None, #must be the same size as soma_border_vertices
                  print_flag=False,
                  return_removed_skeletons=False):
    """
    Example of how to use: 
    
    Simple Example:  
    def distance_measure_func(path,G=None):
    #print("new thing")
    return len(path)

    new_G = clean_skeleton(G,distance_measure_func,return_skeleton=False)
    nx.draw(new_G,with_labels=True)
    
    More complicated example:
    
    import skeleton_utils as sk
    from importlib import reload
    sk = reload(sk)

    from pathlib import Path
    test_skeleton = Path("./Dustin_vp6/Dustin_soma_0_branch_0_0_skeleton.cgal")
    if not test_skeleton.exists():
        print(str(test_skeleton)[:-14])
        file_of_skeleton = sk.calcification(str(test_skeleton.absolute())[:-14])
    else:
        file_of_skeleton = test_skeleton

    # import the skeleton
    test_sk = sk.read_skeleton_edges_coordinates(test_skeleton)
    import trimesh
    test_mesh = trimesh.load_mesh(str(str(test_skeleton.absolute())[:-14] + ".off"))
    sk.graph_skeleton_and_mesh(test_mesh.vertices,
                              test_mesh.faces,
                              edge_coordinates=test_sk)

    # clean the skeleton and then visualize
    import time
    clean_time = time.time()
    cleaned_skeleton = clean_skeleton(test_sk,
                        distance_func=skeletal_distance,
                  min_distance_to_junction=10000,
                  return_skeleton=True,
                  print_flag=True)
    print(f"Total time for skeleton clean {time.time() - clean_time}")

    # see what skeleton looks like now
    test_mesh = trimesh.load_mesh(str(str(test_skeleton.absolute())[:-14] + ".off"))
    sk.graph_skeleton_and_mesh(test_mesh.vertices,
                              test_mesh.faces,
                              edge_coordinates=cleaned_skeleton)
                              
                              
    --------------- end of example -----------------
    """
    
    
    """ --- old way which had index error when completley straight line 
    def end_node_path_to_junciton(curr_G,end_node):
        curr_node = end_node
        node_list = [curr_node]
        for i in range(len(curr_G)):
            neighbors = list(curr_G[curr_node])
            if len(neighbors) <= 2:
                curr_node = [k for k in neighbors if k not in node_list][0]
                node_list.append(curr_node)
                #print(f"node_list = {node_list}")
            else:
                break
        return node_list
    """
    
    def end_node_path_to_junciton(curr_G,end_node):
        curr_node = end_node
        node_list = [curr_node]
        for i in range(len(curr_G)):
            #print(f"\nloop #{i} with curr_node = {curr_node}")

            neighbors = list(curr_G[curr_node])
            #print(f"neighbors = {neighbors}")
            #print(f"node_list = {node_list}")
            if len(neighbors) <= 2:
                #print(f"[k for k in neighbors if k not in node_list] = {[k for k in neighbors if k not in node_list]}")
                possible_curr_nodes = [k for k in neighbors if k not in node_list]
                if len(possible_curr_nodes) <= 0: #this is when it is just one straight line
                    break
                else:
                    curr_node = possible_curr_nodes[0]
                    node_list.append(curr_node)
                    #print(f"node_list = {node_list}")
            else:
                break
        return node_list
    
    print(f"Using Distance measure {distance_func.__name__}")
    
    
    if type(G) not in [type(nx.Graph()),type(xu.GraphOrderedEdges())]:
        G = convert_skeleton_to_graph(G)
        
    kwargs = dict()
    kwargs["coordinates_dict"] = nx.get_node_attributes(G,'coordinates')
    
    
    end_nodes = np.array([k for k,n in dict(G.degree()).items() if n == 1])
    
    
    if (not endpoints_must_keep is None) and len(endpoints_must_keep)>0:
        print(f"endpoints_must_keep = {endpoints_must_keep}")
        all_single_nodes_to_eliminate = []
        endpoints_must_keep = np.array(endpoints_must_keep).reshape(-1,3)
        
        print(f"Number of end_nodes BEFORE filtering = {len(end_nodes)}")
        end_nodes_coordinates = xu.get_node_attributes(G,node_list=end_nodes)
        
        for end_k in endpoints_must_keep:
            end_node_idx = xu.get_nodes_with_attributes_dict(G,dict(coordinates=end_k))
            if len(end_node_idx)>0:
                end_node_idx = end_node_idx[0]
                try:
                    end_node_must_keep_idx = np.where(end_nodes==end_node_idx)[0][0]
                    
                except:
                    print(f"end_nodes = {end_nodes}")
                    print(f"end_node_idx = {end_node_idx}")
                    raise Exception("Something went wrong when trying to find end nodes")
                all_single_nodes_to_eliminate.append(end_node_must_keep_idx)
            else:
                raise Exception("Passed end node to keep that wasn't in the graph")
            
        print(f"all_single_nodes_to_eliminate = {all_single_nodes_to_eliminate}")
        new_end_nodes = np.array([k for i,k in enumerate(end_nodes) if i not in all_single_nodes_to_eliminate])

        #doing the reassigning
        end_nodes = new_end_nodes
        
            
    #clean_time = time.time()
    paths_to_j = [end_node_path_to_junciton(G,n) for n in end_nodes]
    #print(f"Total time for node path to junction = {time.time() - clean_time}")
    #clean_time = time.time()
    end_nodes_dist_to_j = np.array([distance_func(n,G,**kwargs) for n in paths_to_j])
    #print(f"Calculating distances = {time.time() - clean_time}")
    #clean_time = time.time()
    
    end_nodes = end_nodes[end_nodes_dist_to_j<min_distance_to_junction]
    end_nodes_dist_to_j = end_nodes_dist_to_j[end_nodes_dist_to_j<min_distance_to_junction]
    
    skeletons_removed = []
    if len(end_nodes) == 0 or len(end_nodes_dist_to_j) == 0:
        #no end nodes so need to return 
        print("no small end nodes to get rid of so returning whole skeleton")
    else:
        
        
        
        
        current_end_node = end_nodes[np.argmin(end_nodes_dist_to_j)]
        #print(f"Ordering the nodes = {time.time() - clean_time}")
        clean_time = time.time()
        if print_flag:
            print(f"total end_nodes = {end_nodes}")
        #current_end_node = ordered_end_nodes[0]
        paths_removed = 0

        for i in tqdm(range(len(end_nodes))):
            current_path_to_junction = end_node_path_to_junciton(G,current_end_node)
            if print_flag:
                #print(f"ordered_end_nodes = {ordered_end_nodes}")
                print(f"\n\ncurrent_end_node = {current_end_node}")
                print(f"current_path_to_junction = {current_path_to_junction}")
            if distance_func(current_path_to_junction,G,**kwargs) <min_distance_to_junction:
                if print_flag:
                    print(f"the current distance that was below was {distance_func(current_path_to_junction,G,**kwargs)}")
                #remove the nodes
                
                path_to_rem = current_path_to_junction[:-1]
                skeletons_removed.append(convert_graph_to_skeleton( G.subgraph(path_to_rem)))
                
                paths_removed += 1
                G.remove_nodes_from(current_path_to_junction[:-1])
                end_nodes = end_nodes[end_nodes != current_end_node]
                end_nodes_dist_to_j = np.array([distance_func(end_node_path_to_junciton(G,n),G,**kwargs) for n in end_nodes])

                end_nodes = end_nodes[end_nodes_dist_to_j<min_distance_to_junction]
                end_nodes_dist_to_j = end_nodes_dist_to_j[end_nodes_dist_to_j<min_distance_to_junction]

                if len(end_nodes_dist_to_j)<= 0:
                    break
                current_end_node = end_nodes[np.argmin(end_nodes_dist_to_j)]
    #             if print_flag:
    #                 print(f"   insdie if statement ordered_end_nodes = {ordered_end_nodes}")

                #current_end_node = ordered_end_nodes[0]

            else:
                break
            
    G = xu.remove_selfloops(G)
    if print_flag:
        print(f"Done cleaning networkx graph with {paths_removed} paths removed")
    if return_skeleton:
        if return_removed_skeletons:
            return convert_graph_to_skeleton(G),skeletons_removed
        else:
            return convert_graph_to_skeleton(G)
    else:
        if return_removed_skeletons:
            return G,skeletons_removed
        else:
            return G
    





def clean_skeleton_with_soma_verts(G,
                   distance_func,
                  min_distance_to_junction = 3,
                  return_skeleton=True,
                   soma_border_vertices=None, #should be list of soma vertices
                   distance_to_ignore_end_nodes_close_to_soma_border=5000,
                   skeleton_mesh=None,
                   endpoints_must_keep = None, #must be the same size as soma_border_vertices
                  print_flag=False):
    """
    Example of how to use: 
    
    Simple Example:  
    def distance_measure_func(path,G=None):
    #print("new thing")
    return len(path)

    new_G = clean_skeleton(G,distance_measure_func,return_skeleton=False)
    nx.draw(new_G,with_labels=True)
    
    More complicated example:
    
    import skeleton_utils as sk
    from importlib import reload
    sk = reload(sk)

    from pathlib import Path
    test_skeleton = Path("./Dustin_vp6/Dustin_soma_0_branch_0_0_skeleton.cgal")
    if not test_skeleton.exists():
        print(str(test_skeleton)[:-14])
        file_of_skeleton = sk.calcification(str(test_skeleton.absolute())[:-14])
    else:
        file_of_skeleton = test_skeleton

    # import the skeleton
    test_sk = sk.read_skeleton_edges_coordinates(test_skeleton)
    import trimesh
    test_mesh = trimesh.load_mesh(str(str(test_skeleton.absolute())[:-14] + ".off"))
    sk.graph_skeleton_and_mesh(test_mesh.vertices,
                              test_mesh.faces,
                              edge_coordinates=test_sk)

    # clean the skeleton and then visualize
    import time
    clean_time = time.time()
    cleaned_skeleton = clean_skeleton(test_sk,
                        distance_func=skeletal_distance,
                  min_distance_to_junction=10000,
                  return_skeleton=True,
                  print_flag=True)
    print(f"Total time for skeleton clean {time.time() - clean_time}")

    # see what skeleton looks like now
    test_mesh = trimesh.load_mesh(str(str(test_skeleton.absolute())[:-14] + ".off"))
    sk.graph_skeleton_and_mesh(test_mesh.vertices,
                              test_mesh.faces,
                              edge_coordinates=cleaned_skeleton)
                              
                              
    --------------- end of example -----------------
    """
    
    
    """ --- old way which had index error when completley straight line 
    def end_node_path_to_junciton(curr_G,end_node):
        curr_node = end_node
        node_list = [curr_node]
        for i in range(len(curr_G)):
            neighbors = list(curr_G[curr_node])
            if len(neighbors) <= 2:
                curr_node = [k for k in neighbors if k not in node_list][0]
                node_list.append(curr_node)
                #print(f"node_list = {node_list}")
            else:
                break
        return node_list
    """
    
    def end_node_path_to_junciton(curr_G,end_node):
        curr_node = end_node
        node_list = [curr_node]
        for i in range(len(curr_G)):
            #print(f"\nloop #{i} with curr_node = {curr_node}")

            neighbors = list(curr_G[curr_node])
            #print(f"neighbors = {neighbors}")
            #print(f"node_list = {node_list}")
            if len(neighbors) <= 2:
                #print(f"[k for k in neighbors if k not in node_list] = {[k for k in neighbors if k not in node_list]}")
                possible_curr_nodes = [k for k in neighbors if k not in node_list]
                if len(possible_curr_nodes) <= 0: #this is when it is just one straight line
                    break
                else:
                    curr_node = possible_curr_nodes[0]
                    node_list.append(curr_node)
                    #print(f"node_list = {node_list}")
            else:
                break
        return node_list
    
    print(f"Using Distance measure {distance_func.__name__}")
    
    
    if type(G) not in [type(nx.Graph()),type(xu.GraphOrderedEdges())]:
        G = convert_skeleton_to_graph(G)
        
    kwargs = dict()
    kwargs["coordinates_dict"] = nx.get_node_attributes(G,'coordinates')
    
    
    end_nodes = np.array([k for k,n in dict(G.degree()).items() if n == 1])
    """ 9/16 Addition: Will ignore certain end nodes whose distance is too close to soma border"""
    if not soma_border_vertices is None: #assuming that has the correct length
        if len(end_nodes) > 0:


            """
            OLD METHOD THAT DID NOT TRAVERSE ACROSS MESH GRAPH 
            
            Pseducode:
            1) Get the coordinates for all of the end nodes
            2) Put the soma border vertices in a KDTree
            3) Query the KDTree with the node endpoints coordinates
            4) Filter endpoints for only those farther than distance_to_ignore_end_nodes_close_to_soma_border
            
            
            print(f"Number of end_nodes BEFORE filtering = {len(end_nodes)}")
            end_nodes_coordinates = xu.get_node_attributes(G,node_list=end_nodes)
            soma_KD = KDTree(soma_border_vertices)
            distances,closest_nodes = soma_KD.query(end_nodes_coordinates)
            end_nodes = end_nodes[distances>distance_to_ignore_end_nodes_close_to_soma_border]
            print(f"Number of end_nodes AFTER filtering = {len(end_nodes)}")
        
            """
            print(f"Going to ignore certain endnodes that are {distance_to_ignore_end_nodes_close_to_soma_border} nm close to soma border vertices")
            #New method that traverses across mesh graph
            if skeleton_mesh is None:
                raise Exception("Skeleton_mesh is None when trying to account for soma_border_vertices in cleaning")
                
            print(f"Number of end_nodes BEFORE filtering = {len(end_nodes)}")
            end_nodes_coordinates = xu.get_node_attributes(G,node_list=end_nodes)
            
            #0) Get the mesh vertices and create a KDTree from them
            mesh_KD = KDTree(skeleton_mesh.vertices)
            
            #3) Create Weighted Graph from vertex edges
            vertex_graph = tu.mesh_vertex_graph(skeleton_mesh)
            
            all_single_nodes_to_eliminate = []
            
            if endpoints_must_keep is None:
                endpoints_must_keep = [[]]*len(soma_border_vertices)
            for sm_idx, sbv in soma_border_vertices.items():
                # See if there is a node for use to keep
                
                """ OLD WAY BEFORE MAKING MULTIPLE POSSIBLE SOMA TOUCHES
                end_k = endpoints_must_keep[sm_idx]
                """
                
                end_k_list = endpoints_must_keep[sm_idx]
                for end_k in end_k_list:
                    if not end_k is None:
                        end_node_must_keep = xu.get_nodes_with_attributes_dict(G,dict(coordinates=end_k))[0]
                        end_node_must_keep_idx = np.where(end_nodes==end_node_must_keep)[0][0]
                        all_single_nodes_to_eliminate.append(end_node_must_keep_idx)
                        print(f"Using an already specified end node: {end_node_must_keep} with index {end_node_must_keep_idx}"
                             f"checking was correct node end_nodes[index] = {end_nodes[end_node_must_keep_idx]}")
                    continue
            
                #1) Map the soma border vertices to the mesh vertices
                soma_border_distances,soma_border_closest_nodes = mesh_KD.query(sbv[0].reshape(-1,3))

                #2) Map the endpoints to closest mesh vertices
                end_nodes_distances,end_nodes_closest_nodes = mesh_KD.query(end_nodes_coordinates)



                #4) For each endpoint, find shortest distance from endpoint to a soma border along graph
                # for en,en_mesh_vertex in zip(end_nodes,end_nodes_closest_nodes):
                #     #find the shortest path to a soma border vertex
                node_idx_to_keep = []
                node_idx_to_eliminate = []
                node_idx_to_eliminate_len = []
                for en_idx,en in enumerate(end_nodes_closest_nodes):
                    try:
                        path_len, path = nx.single_source_dijkstra(vertex_graph,
                                                                   source = en,
                                                                   target=soma_border_closest_nodes[0],
                                                                   cutoff=distance_to_ignore_end_nodes_close_to_soma_border
                                                                  )
                    except:
                        node_idx_to_keep.append(en_idx)
                    else: #a valid path was found
                        node_idx_to_eliminate.append(en_idx)
                        node_idx_to_eliminate_len.append(path_len)
                        print(f"May Eliminate end_node {en_idx}: {end_nodes[en_idx]} because path_len to soma border was {path_len}")

                if len(node_idx_to_eliminate_len) > 0:
                    #see if there matches a node that we must keep
                    
                    
                    single_node_idx_to_eliminate = node_idx_to_eliminate[np.argmin(node_idx_to_eliminate_len)]
                    print(f"single_node_to_eliminate = {single_node_idx_to_eliminate}")
                    all_single_nodes_to_eliminate.append(single_node_idx_to_eliminate)
                else:
                    print("No close endpoints to choose from for elimination")
            
            print(f"all_single_nodes_to_eliminate = {all_single_nodes_to_eliminate}")
            new_end_nodes = np.array([k for i,k in enumerate(end_nodes) if i not in all_single_nodes_to_eliminate])

            #doing the reassigning
            end_nodes = new_end_nodes
            
    #clean_time = time.time()
    paths_to_j = [end_node_path_to_junciton(G,n) for n in end_nodes]
    #print(f"Total time for node path to junction = {time.time() - clean_time}")
    #clean_time = time.time()
    end_nodes_dist_to_j = np.array([distance_func(n,G,**kwargs) for n in paths_to_j])
    #print(f"Calculating distances = {time.time() - clean_time}")
    #clean_time = time.time()
    
    end_nodes = end_nodes[end_nodes_dist_to_j<min_distance_to_junction]
    end_nodes_dist_to_j = end_nodes_dist_to_j[end_nodes_dist_to_j<min_distance_to_junction]
    
    if len(end_nodes) == 0 or len(end_nodes_dist_to_j) == 0:
        #no end nodes so need to return 
        print("no small end nodes to get rid of so returning whole skeleton")
    else:
        
        
        
        
        current_end_node = end_nodes[np.argmin(end_nodes_dist_to_j)]
        #print(f"Ordering the nodes = {time.time() - clean_time}")
        clean_time = time.time()
        if print_flag:
            print(f"total end_nodes = {end_nodes}")
        #current_end_node = ordered_end_nodes[0]
        paths_removed = 0

        for i in tqdm(range(len(end_nodes))):
            current_path_to_junction = end_node_path_to_junciton(G,current_end_node)
            if print_flag:
                #print(f"ordered_end_nodes = {ordered_end_nodes}")
                print(f"\n\ncurrent_end_node = {current_end_node}")
                print(f"current_path_to_junction = {current_path_to_junction}")
            if distance_func(current_path_to_junction,G,**kwargs) <min_distance_to_junction:
                if print_flag:
                    print(f"the current distance that was below was {distance_func(current_path_to_junction,G,**kwargs)}")
                #remove the nodes
                paths_removed += 1
                G.remove_nodes_from(current_path_to_junction[:-1])
                end_nodes = end_nodes[end_nodes != current_end_node]
                end_nodes_dist_to_j = np.array([distance_func(end_node_path_to_junciton(G,n),G,**kwargs) for n in end_nodes])

                end_nodes = end_nodes[end_nodes_dist_to_j<min_distance_to_junction]
                end_nodes_dist_to_j = end_nodes_dist_to_j[end_nodes_dist_to_j<min_distance_to_junction]

                if len(end_nodes_dist_to_j)<= 0:
                    break
                current_end_node = end_nodes[np.argmin(end_nodes_dist_to_j)]
    #             if print_flag:
    #                 print(f"   insdie if statement ordered_end_nodes = {ordered_end_nodes}")

                #current_end_node = ordered_end_nodes[0]

            else:
                break
            
    G = xu.remove_selfloops(G)
    if print_flag:
        print(f"Done cleaning networkx graph with {paths_removed} paths removed")
    if return_skeleton:
        return convert_graph_to_skeleton(G)
    else:
        return G
    
import copy
import time
def combine_close_branch_points(skeleton=None,
                                combine_threshold = 700,
                               print_flag=False,
                                skeleton_branches=None):
    """
    Purpose: To take a skeleton or graph and return a skelton/graph 
    where close branch points are combined
    
    
    Example Code of how could get the orders: 
    # How could potentially get the edge order we wanted
    endpoint_neighbors_to_order_map = dict()
    for k in endpoint_neighbors:
        total_orders = []
        total_orders_neighbors = []
        for j in p:
            try:
                total_orders.append(curr_sk_graph[j][k]["order"])
                total_orders_neighbors.append(j)
            except:
                pass
        order_index = np.argmin(total_orders)
        endpoint_neighbors_to_order_map[(k,total_orders_neighbors[order_index])] = total_orders[order_index]
    endpoint_neighbors_to_order_map
    
    
    
    Ex: 
    sk = reload(sk)
    import numpy_utils as nu
    nu = reload(nu)
    branch_skeleton_data_cleaned = []
    for i,curr_sk in enumerate(branch_skeleton_data):
        print(f"\n----- Working on skeleton {i} ---------")
        new_sk = sk.combine_close_branch_points(curr_sk,print_flag=True)
        print(f"Original Sk = {curr_sk.shape}, Cleaned Sk = {new_sk.shape}")
        branch_skeleton_data_cleaned.append(new_sk)
        
        
        
    """
    
    
    debug_time = False
    combine_close_time = time.time()
    
    branches_flag = False    
    if not skeleton_branches is None:
        #Create an array that maps the branch idx to the endpoints and make a copy
        branch_idx_to_endpoints = np.array([find_branch_endpoints(k) for k in skeleton_branches])
        branch_idx_to_endpoints_original = copy.deepcopy(branch_idx_to_endpoints)
        branch_keep_idx = np.arange(0,len(skeleton_branches))
        
        skeleton = sk.stack_skeletons(skeleton_branches)
        branches_flag = True
    
    
    convert_back_to_skeleton = False
    #1) convert the skeleton to a graph
    
    if nu.is_array_like(skeleton):
        curr_sk_graph = sk.convert_skeleton_to_graph(skeleton)
        convert_back_to_skeleton=True
    else:
        curr_sk_graph = skeleton

    #2) Get all of the high degree nodes
    high_degree_nodes = np.array(xu.get_nodes_greater_or_equal_degree_k(curr_sk_graph,3))
    
    """
    Checked that thes high degree nodes were correctly retrieved
    high_degree_coordinates = xu.get_node_attributes(curr_sk_graph,node_list = high_degree_nodes)
    sk.graph_skeleton_and_mesh(other_skeletons=[curr_sk],
                              other_scatter=[high_degree_coordinates])

    """
    
    #3) Get paths between all of them high degree nodes
    valid_paths = []
    valid_path_lengths = []
    for s in high_degree_nodes:
        degree_copy = high_degree_nodes[high_degree_nodes != s]

        for t in degree_copy:
            try:
                path_len, path = nx.single_source_dijkstra(curr_sk_graph,source = s,target=t,cutoff=combine_threshold)
            except:
                continue
            else: #a valid path was found
                degree_no_endpoints = degree_copy[degree_copy != t]

                if len(set(degree_no_endpoints).intersection(set(path))) > 0:
                    continue
                else:
                    match_path=False
                    for v_p in valid_paths:
                        if set(v_p) == set(path):
                            match_path = True
                            break
                    if not match_path:
                        valid_paths.append(np.array(list(path)))
                        valid_path_lengths.append(path_len)
                        
                    
                    
#                     sorted_path = np.sort(path)
#                     try:
                        
                        
#                         if len(nu.matching_rows(valid_paths,sorted_path)) > 0:
#                             continue
#                         else:
#                             valid_paths.append(sorted_path)
#                     except:
#                         print(f"valid_paths = {valid_paths}")
#                         print(f"sorted_path = {sorted_path}")
#                         print(f"nu.matching_rows(valid_paths,sorted_path) = {nu.matching_rows(valid_paths,sorted_path)}")
#                         raise Exception()

    if print_flag:
        print(f"Found {len(valid_paths)} valid paths to replace")
        print(f"valid_paths = {(valid_paths)}")
        print(f"valid_path_lengths = {valid_path_lengths}")
                     
    if debug_time:
        print(f"Finding all paths = { time.time() - combine_close_time}")
        combine_close_time = time.time()
        
    if len(valid_paths) == 0:
        if print_flag:
            print("No valid paths found so just returning the original")
        if branches_flag:
            skeleton_branches,branch_keep_idx
        else:
            return skeleton
    
    # Need to combine paths if there is any overlapping:
    valid_paths

    """
    # --------------If there were valid paths found -------------
    
    
    """
    curr_sk_graph_cp = copy.deepcopy(curr_sk_graph)
    if debug_time:
        print(f"Copying graph= { time.time() - combine_close_time}")
        combine_close_time = time.time()
        
    print(f"length of Graph = {len(curr_sk_graph_cp)}")
    for p_idx,p in enumerate(valid_paths):
        
        print(f"Working on path {p}")
        path_degrees = xu.get_node_degree(curr_sk_graph_cp,p)
        print(f"path_degrees = {path_degrees}")
        
        p_end_nodes = p[[0,-1]]
        #get endpoint coordinates
        path_coordinates = xu.get_node_attributes(curr_sk_graph_cp,node_list=p)
        end_coordinates = np.array([path_coordinates[0],path_coordinates[-1]]).reshape(-1,3)
        
        if debug_time:
            print(f"Getting coordinates = { time.time() - combine_close_time}")
            combine_close_time = time.time()
        
#         print(f"end_coordinates = {end_coordinates}")
#         print(f"branch_idx_to_endpoints = {branch_idx_to_endpoints}")
#         print(f"branch_idx_to_endpoints.shape = {branch_idx_to_endpoints.shape}")


        debug = False
        if debug:
            end_coordinates_try_2 = xu.get_node_attributes(curr_sk_graph_cp,node_list=p_end_nodes)
            print(f"end_coordinates = {end_coordinates}")
            print(f"end_coordinates_try_2 = {end_coordinates_try_2}")
            print(f"branch_idx_to_endpoints = {branch_idx_to_endpoints}")
            
        if branches_flag:
            #find the branch_idx with the found endpoints (if multiple then continue)
            branch_idxs = nu.find_matching_endpoints_row(branch_idx_to_endpoints,end_coordinates)
        
            if len(branch_idxs)>1:
                continue
            elif len(branch_idxs) == 0:
                raise Exception("No matching endpoints for branch")
            else:
                branch_position_to_delete = branch_idxs[0]
        
        if debug_time:
            print(f"Finding matching endpoints = { time.time() - combine_close_time}")
            combine_close_time = time.time()
    
        #get the coordinates of the path and average them for new node
        average_endpoints = np.mean(path_coordinates,axis=0)
        
        #replace the old end nodes with the new one
        curr_sk_graph_cp,new_node_id = xu.add_new_coordinate_node(curr_sk_graph_cp,node_coordinate=average_endpoints,replace_nodes=p_end_nodes)
        
        if debug_time:
            print(f"Adding new coordinate node = { time.time() - combine_close_time}")
            combine_close_time = time.time()
        
        #go through and change all remaining paths to now include the new combined node id
        for p_idx_curr in range(p_idx+1,len(valid_paths)):
            if debug:
                print(f"valid_paths[p_idx_curr] = {valid_paths[p_idx_curr]}")
                print(f"p_end_nodes = {p_end_nodes}")
                print(f"new_node_id = {new_node_id}")
                print(f"(valid_paths[p_idx_curr]==p_end_nodes[0]) | (valid_paths[p_idx_curr]==p_end_nodes[1]) = {(valid_paths[p_idx_curr]==p_end_nodes[0]) | (valid_paths[p_idx_curr]==p_end_nodes[1])}")
                
                
            valid_paths[p_idx_curr][(valid_paths[p_idx_curr]==p_end_nodes[0]) | (valid_paths[p_idx_curr]==p_end_nodes[1])] = new_node_id
        
        if debug_time:
            print(f"Changing all remaining paths = { time.time() - combine_close_time}")
            combine_close_time = time.time()
        
        if branches_flag:
            #delete the branch id from the index
            branch_idx_to_endpoints = np.delete(branch_idx_to_endpoints, branch_position_to_delete, 0)
            branch_keep_idx = np.delete(branch_keep_idx,branch_position_to_delete)



            #go through and replace and of the endpoints list that were the old endpoints now with the new one
            match_1 = (branch_idx_to_endpoints.reshape(-1,3) == end_coordinates[0]).all(axis=1).reshape(-1,2)
            match_2 = (branch_idx_to_endpoints.reshape(-1,3) == end_coordinates[1]).all(axis=1).reshape(-1,2)
            replace_mask = match_1 | match_2
            branch_idx_to_endpoints[replace_mask] = average_endpoints
            
        if debug_time:
            print(f"Replacing branch index = { time.time() - combine_close_time}")
            combine_close_time = time.time()
        
        #delete the nodes that were on the path
        curr_sk_graph_cp.remove_nodes_from(p)
        
        if debug_time:
            print(f"Removing nodes = { time.time() - combine_close_time}")
            combine_close_time = time.time()
        
    
    
    if branches_flag:
        """
        1) Find the branches that were not filtered away
        2) Get the new and original endpoints of the filtered branches
        3) For all filtered branches:
           i) convert skeleton into a graph
           For each endpoint
               ii) send the original endpoint to get replaced by new one
           iii) convert back into skeleton and save

        """
        #1) Find the branches that were not filtered away
        skeleton_branches = np.array(skeleton_branches)
        filtered_skeleton_branches = skeleton_branches[branch_keep_idx]
        
        #2) Get the new and original endpoints of the filtered branches
        original_endpoints = branch_idx_to_endpoints_original[branch_keep_idx]
        new_endpoints = branch_idx_to_endpoints
        
        #3) For all filtered branches:
        edited_skeleton_branches = []
        for f_sk,f_old_ep,f_new_ep in zip(filtered_skeleton_branches,original_endpoints,new_endpoints):
            f_sk_graph = convert_skeleton_to_graph(f_sk)
            for old_ep,new_ep in zip(f_old_ep,f_new_ep):
                f_sk_graph = xu.add_new_coordinate_node(f_sk_graph,
                                           node_coordinate=new_ep,
                                           replace_coordinates=old_ep,
                                          return_node_id=False)
            edited_skeleton_branches.append(convert_graph_to_skeleton(f_sk_graph))
        
        if debug_time:
            print(f"Filtering branches = { time.time() - combine_close_time}")
            combine_close_time = time.time()
        
        return edited_skeleton_branches,branch_keep_idx
        

    if convert_back_to_skeleton:
        return sk.convert_graph_to_skeleton(curr_sk_graph_cp)
    else:
        return curr_sk_graph_cp






def old_combine_close_branch_points(skeleton,
                                combine_threshold = 700,
                               print_flag=False):
    """
    Purpose: To take a skeleton or graph and return a skelton/graph 
    where close branch points are combined
    
    
    Example Code of how could get the orders: 
    # How could potentially get the edge order we wanted
    endpoint_neighbors_to_order_map = dict()
    for k in endpoint_neighbors:
        total_orders = []
        total_orders_neighbors = []
        for j in p:
            try:
                total_orders.append(curr_sk_graph[j][k]["order"])
                total_orders_neighbors.append(j)
            except:
                pass
        order_index = np.argmin(total_orders)
        endpoint_neighbors_to_order_map[(k,total_orders_neighbors[order_index])] = total_orders[order_index]
    endpoint_neighbors_to_order_map
    
    
    
    Ex: 
    sk = reload(sk)
    import numpy_utils as nu
    nu = reload(nu)
    branch_skeleton_data_cleaned = []
    for i,curr_sk in enumerate(branch_skeleton_data):
        print(f"\n----- Working on skeleton {i} ---------")
        new_sk = sk.combine_close_branch_points(curr_sk,print_flag=True)
        print(f"Original Sk = {curr_sk.shape}, Cleaned Sk = {new_sk.shape}")
        branch_skeleton_data_cleaned.append(new_sk)
        
        
        
    """
    
    convert_back_to_skeleton = False
    #1) convert the skeleton to a graph
    
    if nu.is_array_like(skeleton):
        curr_sk_graph = sk.convert_skeleton_to_graph(skeleton)
        convert_back_to_skeleton=True
    else:
        curr_sk_graph = skeleton

    #2) Get all of the high degree nodes
    high_degree_nodes = np.array(xu.get_nodes_greater_or_equal_degree_k(curr_sk_graph,3))
    
    """
    Checked that thes high degree nodes were correctly retrieved
    high_degree_coordinates = xu.get_node_attributes(curr_sk_graph,node_list = high_degree_nodes)
    sk.graph_skeleton_and_mesh(other_skeletons=[curr_sk],
                              other_scatter=[high_degree_coordinates])

    """
    
    #3) Get paths between all of them high degree nodes
    valid_paths = []
    valid_path_endpoints = []
    valid_path_lengths = []
    for s in high_degree_nodes:
        degree_copy = high_degree_nodes[high_degree_nodes != s]

        for t in degree_copy:
            try:
                path_len, path = nx.single_source_dijkstra(curr_sk_graph,source = s,target=t,cutoff=combine_threshold)
            except:
                continue
            else: #a valid path was found
                degree_no_endpoints = degree_copy[degree_copy != t]

                if len(set(degree_no_endpoints).intersection(set(path))) > 0:
                    continue
                else:
                    match_path=False
                    for v_p in valid_paths:
                        if set(v_p) == set(path):
                            match_path = True
                            break
                    if not match_path:
                        valid_paths.append(np.sort(path))
                        valid_path_lengths.append(path_len)
                        valid_path_endpoints.append([s,t])
                        
                    
                    
#                     sorted_path = np.sort(path)
#                     try:
                        
                        
#                         if len(nu.matching_rows(valid_paths,sorted_path)) > 0:
#                             continue
#                         else:
#                             valid_paths.append(sorted_path)
#                     except:
#                         print(f"valid_paths = {valid_paths}")
#                         print(f"sorted_path = {sorted_path}")
#                         print(f"nu.matching_rows(valid_paths,sorted_path) = {nu.matching_rows(valid_paths,sorted_path)}")
#                         raise Exception()

    if print_flag:
        print(f"Found {len(valid_paths)} valid paths to replace")
        print(f"valid_paths = {(valid_paths)}")
        print(f"valid_path_lengths = {valid_path_lengths}")
                        
    if len(valid_paths) == 0:
        if print_flag:
            print("No valid paths found so just returning the original")
        return skeleton
    
    # Need to combine paths if there is any overlapping:
    valid_paths

    """
    # --------------If there were valid paths found -------------
    
    5) For the paths that past the thresholding:
    With a certain path
    a. take the 2 end high degree nodes and get all of their neighbors
    a2. get the coordinates of the endpoints and average them for new node
    b. Create a new node with all the neighbors and averaged coordinate
    c. replace all of the other paths computed (if they had the 2 end high degree nodes) replace with the new node ID
    d. Delete the old high degree ends and all nodes on the path
    Go to another path
    """
    curr_sk_graph_cp = copy.deepcopy(curr_sk_graph)
    for p,p_end_nodes in zip(valid_paths,valid_path_endpoints):
        print(f"Working on path {p}")
        
        #a. take the 2 end high degree nodes and get all of their neighbors
        endpoint_neighbors = np.unique(np.concatenate([xu.get_neighbors(curr_sk_graph_cp,k) for k in p]))
        endpoint_neighbors = np.setdiff1d(endpoint_neighbors,list(p))
        print(f"endpoint_neighbors = {endpoint_neighbors}")
        print(f"p_end_nodes = {p_end_nodes}")
        
        #a2. get the coordinates of the endpoints and average them for new node
        #endpoint_coordinates = np.vstack([xu.get_node_attributes(curr_sk_graph_cp,node_list=k) for k in p])
        path_coordinates = xu.get_node_attributes(curr_sk_graph_cp,node_list=p)
        end_coordinates = np.array([path_coordinates[0],path_coordinates[-1]]).reshape(-1,3)
        print(f"end_coordinates = {end_coordinates}")
#         print(f"endpoint_coordinates = {endpoint_coordinates}")
#         print(f"endpoint_coordinates_try_2 = {endpoint_coordinates_try_2}")
        average_endpoints = np.mean(path_coordinates,axis=0)
    
        #b. Create a new node with all the neighbors and averaged coordinate
        new_node_id = np.max(curr_sk_graph_cp.nodes()) + 1
        curr_sk_graph_cp.add_node(new_node_id,coordinates=average_endpoints)
        
        #c. replace all of the other paths computed (if they had the 2 end high degree nodes) replace with the new node ID
        print(f"endpoint_neighbors = {endpoint_neighbors}")
        curr_sk_graph_cp.add_weighted_edges_from([(new_node_id,k,
                    np.linalg.norm(curr_sk_graph_cp.nodes[k]["coordinates"] - average_endpoints)) for k in endpoint_neighbors])
        
    #d. Delete the old high degree ends and all nodes on the path
    concat_valid_paths = np.unique(np.concatenate(valid_paths))
    print(f"Concatenating all paths and deleting: {concat_valid_paths}")
    
    curr_sk_graph_cp.remove_nodes_from(concat_valid_paths)
    
    if convert_back_to_skeleton:
        return sk.convert_graph_to_skeleton(curr_sk_graph_cp)
    else:
        return curr_sk_graph_cp

    
    
    
# ---------------------- Full Skeletonization Function --------------------- #
from pykdtree.kdtree import KDTree
import time
import trimesh
import numpy as np
from pathlib import Path

import time
import os
import pathlib


import meshlab
import skeleton_utils as sk

from shutil import rmtree
from pathlib import Path

import soma_extraction_utils as soma_utils
from pathlib import Path
import trimesh

def load_somas(segment_id,main_mesh_total,
              soma_path):
    soma_path = str(soma_path)
    try:
        current_soma = trimesh.load_mesh(str(soma_path))
        return [current_soma]
    except:
        print("No Soma currently available so must compute own")
        (total_soma_list, 
             run_time, 
             total_soma_list_sdf) = soma_utils.extract_soma_center(
                                segment_id,
                                main_mesh_total.vertices,
                                main_mesh_total.faces,
                                outer_decimation_ratio= 0.25,
                                large_mesh_threshold = 60000,
                                large_mesh_threshold_inner = 40000,
                                soma_width_threshold = 0.32,
                                soma_size_threshold = 20000,
                               inner_decimation_ratio = 0.25,
                               volume_mulitplier=7,
                               side_length_ratio_threshold=3,
                                soma_size_threshold_max=192000,
                                delete_files=True
            )
        
        # save the soma
        print(f"Found {len(total_soma_list)} somas")
        soma_mesh = combine_meshes(total_soma_list)
        soma_mesh.export(soma_path)
        
        return total_soma_list
    else:
        return []

import meshparty_skeletonize as m_sk
import time
def skeletonize_connected_branch_meshparty(mesh,
                                           segment_size = 100,
                                           verbose=False,
                                          ):

    fusion_time = time.time()

    # --------------- Part 3: Meshparty skeletonization and Decomposition ------------- #
    sk_meshparty_obj = m_sk.skeletonize_mesh_largest_component(mesh,
                                                            root=None,
                                                              filter_mesh=False)

    if verbose:
        print(f"meshparty_segment_size = {meshparty_segment_size}")



    new_skeleton = m_sk.skeleton_obj_to_branches(sk_meshparty_obj,
                                                          mesh = mesh,
                                                          meshparty_segment_size=segment_size,
                                                          return_skeleton_only=True)

    if verbose:
        print(f"Time meshparty skeletonization: {time.time() - fusion_time }")

    return new_skeleton

def skeletonize_connected_branch(current_mesh,
                        output_folder="./temp",
                        delete_temp_files=True,
                        name="None",
                        surface_reconstruction_size=50,
                        surface_reconstruction_width = 250,
                        poisson_stitch_size = 4000,
                        n_surface_downsampling = 1,
                        n_surface_samples=1000,
                        skeleton_print=False,
                        mesh_subtraction_distance_threshold=3000,
                        mesh_subtraction_buffer=50,
                        max_stitch_distance = 18000,
                        current_min_edge = 75,
                        close_holes=True,
                        limb_name=None,
                        use_surface_after_CGAL = True,
                        remove_cycles=True,
                        connectivity="edges",
                        verbose=False,
                                 
                        ):
    """
    Purpose: To take a mesh and construct a full skeleton of it
    (Assuming the Soma is already extracted)
    
    1) Poisson Surface Reconstruction
    2) CGAL skeletonization of all signfiicant pieces 
        (if above certain size ! threshold) 
                --> if not skip straight to surface skeletonization
    3) Using CGAL skeleton, find the leftover mesh not skeletonized
    4) Do surface reconstruction on the parts that are left over
    - with some downsampling
    5) Stitch the skeleton 
    """
    debug = True
    
    
    print(f"inside skeletonize_connected_branch and use_surface_after_CGAL={use_surface_after_CGAL}, surface_reconstruction_size={surface_reconstruction_size}")
    #check that the mesh is all one piece
    current_mesh_splits = split_significant_pieces(current_mesh,
                               significance_threshold=1,
                                                  connectivity=connectivity)
    if len(current_mesh_splits) > 1:
        print(f"The mesh passed has {len(current_mesh_splits)} pieces so just taking the largest one {current_mesh_splits[0]}")
        current_mesh = current_mesh_splits[0]

    # check the size of the branch and if small enough then just do
    # Surface Skeletonization
    if len(current_mesh.faces) < surface_reconstruction_size:
        #do a surface skeletonization
        print("Doing skeleton surface reconstruction")
        surf_sk = generate_surface_skeleton(current_mesh.vertices,
                                    current_mesh.faces,
                                    surface_samples=n_surface_samples,
                                             n_surface_downsampling=n_surface_downsampling )
        return surf_sk
    else:
    
        #if can't simply do a surface skeletonization then 
        #use cgal method that requires temp folder

        if type(output_folder) != type(Path()):
            output_folder = Path(str(output_folder))
            output_folder.mkdir(parents=True,exist_ok=True)
            
        # CGAL Step 1: Do Poisson Surface Reconstruction
        Poisson_obj = meshlab.Poisson(output_folder,overwrite=True)
        

        skeleton_start = time.time()
        print("     Starting Screened Poisson")
        new_mesh,output_subprocess_obj = Poisson_obj(   
                                    vertices=current_mesh.vertices,
                                     faces=current_mesh.faces,
                                    mesh_filename=name + ".off",
                                     return_mesh=True,
                                     delete_temp_files=delete_temp_files,
                                    )
        
        if close_holes: 
            print("Using the close holes feature")
            
            new_mesh = tu.fill_holes(new_mesh)
            """
            Old Way 
            
            FillHoles_obj = meshlab.FillHoles(output_folder,overwrite=True)

            new_mesh,output_subprocess_obj = FillHoles_obj(   
                                                vertices=new_mesh.vertices,
                                                 faces=new_mesh.faces,
                                                 return_mesh=True,
                                                 delete_temp_files=delete_temp_files,
                                                )
            """
        
        
        print(f"-----Time for Screened Poisson= {time.time()-skeleton_start}")
        
        
        #2) Filter away for largest_poisson_piece:
        if use_surface_after_CGAL:
            restriction_threshold = surface_reconstruction_size
        else:
            
            restriction_threshold = poisson_stitch_size
        if verbose:
            print(f"restriction_threshold = {restriction_threshold}")
            
        mesh_pieces = split_significant_pieces(new_mesh,
                                            significance_threshold=restriction_threshold,
                                              connectivity=connectivity)
        
        if skeleton_print:
            print(f"Signifiant mesh pieces of {surface_reconstruction_size} size "
                 f"after poisson = {len(mesh_pieces)}")
        skeleton_ready_for_stitching = np.array([])
        skeleton_files = [] # to be erased later on if need be
        if len(mesh_pieces) <= 0:
            if skeleton_print:
                print("No signficant skeleton pieces so just doing surface skeletonization")
            # do surface skeletonization on all of the pieces
            surface_mesh_pieces = split_significant_pieces(new_mesh,
                                            significance_threshold=2,
                                                          connectivity=connectivity)
            
            #get the skeletons for all those pieces
            current_mesh_skeleton_list = [
                generate_surface_skeleton(p.vertices,
                                    p.faces,
                                    surface_samples=n_surface_samples,
                                    n_surface_downsampling=n_surface_downsampling )
                for p in surface_mesh_pieces
            ]
            
            skeleton_ready_for_stitching = stack_skeletons(current_mesh_skeleton_list)
            
            #will stitch them together later
        else: #if there are parts that can do the cgal skeletonization
            skeleton_start = time.time()
            print(f"mesh_pieces = {mesh_pieces}")
            print("     Starting Calcification (Changed back where stitches large poissons)")
            for zz,piece in enumerate(mesh_pieces):
                current_mesh_path = output_folder / f"{name}_{zz}"
                if skeleton_print:
                    print(f"current_mesh_path = {current_mesh_path}")
                written_path = write_neuron_off(piece,current_mesh_path)
                
                #print(f"Path sending to calcification = {written_path[:-4]}")
                returned_value, sk_file_name = calcification(written_path,
                                                               min_edge_length = current_min_edge)
                if skeleton_print:
                    print(f"returned_value = {returned_value}")
                    print(f"sk_file_name = {sk_file_name}")
                #print(f"Time for skeletonizatin = {time.time() - skeleton_start}")
                skeleton_files.append(sk_file_name)
                
            if skeleton_print:
                print(f"-----Time for Running Calcification = {time.time()-skeleton_start}")
            
            #collect the skeletons and subtract from the mesh
            
            significant_poisson_skeleton = read_skeleton_edges_coordinates(skeleton_files)
            
            
            
            if len(significant_poisson_skeleton) > 0:
                if remove_cycles:
                    significant_poisson_skeleton = remove_cycles_from_skeleton(significant_poisson_skeleton)
                
                
                if use_surface_after_CGAL:
                    boolean_significance_threshold=5

                    print(f"Before mesh subtraction number of skeleton edges = {significant_poisson_skeleton.shape[0]+1}")
                    mesh_pieces_leftover =  mesh_subtraction_by_skeleton(current_mesh,
                                                                significant_poisson_skeleton,
                                                                buffer=mesh_subtraction_buffer,
                                                                bbox_ratio=1.2,
                                                                distance_threshold=mesh_subtraction_distance_threshold,
                                                                significance_threshold=boolean_significance_threshold,
                                                                print_flag=False
                                                               )

                    # *****adding another significance threshold*****
                
                    leftover_meshes_sig = [k for k in mesh_pieces_leftover if len(k.faces) > surface_reconstruction_size]
                else:
                    leftover_meshes_sig = []
                
                
                #want to filter these significant pieces for only those below a certain width
                if not surface_reconstruction_width is None and len(leftover_meshes_sig) > 0:
                    if skeleton_print:
                        print("USING THE SDF WIDTHS TO FILTER SURFACE SKELETONS")
                        print(f"leftover_meshes_sig before filtering = {len(leftover_meshes_sig)}")
                    leftover_meshes_sig_new = []
                    from trimesh.ray import ray_pyembree
                    ray_inter = ray_pyembree.RayMeshIntersector(current_mesh)
                    """
                    Pseudocode:
                    For each leftover significant mesh
                    1) Map the leftover piece back to the original face
                    2) Get the widths fo the piece
                    3) get the median of the non-zero values
                    4) if greater than the surface_reconstruction_width then add to list
                    
                    """
                    for lm in leftover_meshes_sig:
                        face_indices_leftover_0 = tu.original_mesh_faces_map(current_mesh,lm)
                        curr_width_distances = tu.ray_trace_distance(mesh=current_mesh,
                          face_inds=face_indices_leftover_0,
                                                 ray_inter=ray_inter
                        )
                        filtered_widths = curr_width_distances[curr_width_distances>0]
                        if len(filtered_widths) == 0:
                            continue
                        if np.mean(filtered_widths) < surface_reconstruction_width:
                            leftover_meshes_sig_new.append(lm)
                            
                    leftover_meshes_sig = leftover_meshes_sig_new
                    if skeleton_print:
                        print(f"leftover_meshes_sig AFTER filtering = {len(leftover_meshes_sig)}")
                    
                leftover_meshes = combine_meshes(leftover_meshes_sig)
            else:
#                 if not use_surface_after_CGAL:
#                     surf_sk = generate_surface_skeleton(m.vertices,
#                                                m.faces,
#                                                surface_samples=n_surface_samples,
#                                     n_surface_downsampling=n_surface_downsampling )
#                     return surf_sk
#                     raise gu.CGAL_skel_error(f"No CGAL skeleton was generated when the {use_surface_after_CGAL} flag was set")
                
                """------------------ 1 / 2 /2021 Addition ------------------------"""
    
                print("No recorded skeleton so skipping"
                     " to meshparty skeletonization")
                
                return sk.skeletonize_connected_branch_meshparty(current_mesh)
                
                #leftover_meshes_sig = [current_mesh]
            
            
            if skeleton_print:
                print(f"len(leftover_meshes_sig) = {leftover_meshes_sig}")
                for zz,curr_m in enumerate(leftover_meshes_sig):
                    tu.write_neuron_off(curr_m,f"./leftover_test/limb_{limb_name}_{zz}.off")
                    
            leftover_meshes_sig_surf_sk = []
            for m in tqdm(leftover_meshes_sig):
                surf_sk = generate_surface_skeleton(m.vertices,
                                               m.faces,
                                               surface_samples=n_surface_samples,
                                    n_surface_downsampling=n_surface_downsampling )
                if len(surf_sk) > 0:
                    leftover_meshes_sig_surf_sk.append(surf_sk)
            leftovers_stacked = stack_skeletons(leftover_meshes_sig_surf_sk)
            #print(f"significant_poisson_skeleton = {significant_poisson_skeleton}")
            #print(f"leftover_meshes_sig_surf_sk = {leftover_meshes_sig_surf_sk}")
            skeleton_ready_for_stitching = stack_skeletons([significant_poisson_skeleton,leftovers_stacked])
            
        #now want to stitch together whether generated from 
        if skeleton_print:
            print(f"After cgal process the un-stitched skeleton has shape {skeleton_ready_for_stitching.shape}")
            #su.compressed_pickle(skeleton_ready_for_stitching,"sk_before_stitiching")
        
        # Now want to always do the skeleton stitching
        #if use_surface_after_CGAL:
        stitched_skeletons_full = stitch_skeleton(
                                                  skeleton_ready_for_stitching,
                                                  max_stitch_distance=max_stitch_distance,
                                                  stitch_print = False,
                                                  main_mesh = []
                                                )
#         else:
#             stitched_skeletons_full = skeleton_ready_for_stitching
            
        #stitched_skeletons_full_cleaned = clean_skeleton(stitched_skeletons_full)
        
        # erase the skeleton files if need to be
        if delete_temp_files:
            for sk_fi in skeleton_files:
                if Path(sk_fi).exists():
                    Path(sk_fi).unlink()
        
        # if created temp folder then erase if empty
        if str(output_folder.absolute()) == str(Path("./temp").absolute()):
            print("The process was using a temp folder")
            if len(list(output_folder.iterdir())) == 0:
                print("Temp folder was empty so deleting it")
                if output_folder.exists():
                    rmtree(str(output_folder.absolute()))
        
        return stitched_skeletons_full
    
def soma_skeleton_stitching(total_soma_skeletons,soma_mesh):
    """
    Purpose: Will stitch together the meshes that are touching
    the soma 
    
    Pseudocode: 
    1) Compute the soma mesh center point
    2) For meshes that were originally connected to soma
    a. Find the closest skeletal point to soma center
    b. Add an edge from closest point to soma center
    3) Then do stitching algorithm on all of remaining disconnected
        skeletons
    
    
    """
    # 1) Compute the soma mesh center point
    soma_center = np.mean(soma_mesh.vertices,axis=0)
    
    soma_connecting_skeleton = []
    for skel in total_soma_skeletons:
        #get the unique vertex points
        unique_skeleton_nodes = np.unique(skel.reshape(-1,3),axis=0)
        
        # a. Find the closest skeletal point to soma center
        # b. Add an edge from closest point to soma center
        mesh_tree = KDTree(unique_skeleton_nodes)
        distances,closest_node = mesh_tree.query(soma_center.reshape(-1,3))
        closest_skeleton_vert = unique_skeleton_nodes[closest_node[np.argmin(distances)]]
        soma_connecting_skeleton.append(np.array([closest_skeleton_vert,soma_center]).reshape(-1,2,3))
    
    print(f"soma_connecting_skeleton[0].shape = {soma_connecting_skeleton[0].shape}")
    print(f"total_soma_skeletons[0].shape = {total_soma_skeletons[0].shape}")
    # stith all of the ekeletons together
    soma_stitched_sk = stack_skeletons(total_soma_skeletons + soma_connecting_skeleton)
    
    return soma_stitched_sk



# ---- Util functions to be used for the skeletonization of soma containing meshes ------ #
def recursive_soma_skeletonization(main_mesh,
                                  soma_mesh_list,
                                soma_mesh_list_indexes,
                                   mesh_base_path="./temp_folder",
                                  soma_mesh_list_centers=[],
                                   current_name="segID"
                                  ):
    """
    Parameters:
    Mesh piece 
    The soma centers list and meshes list contained somewhere within the mesh piece
    
        Algorithm
    1) Start with the first soma and subtract from mesh
    2) Find all of the disconnected mesh pieces
    3) If there is still a soma piece that has not been processed, 
    find mesh pieces and all the somas that are contained within that
    4) Send Each one of those mesh pieces and soma lists
    to the recursive_soma_skeletonization (these will return skeletons)
    5) For all other pieces that do not have a soma do skeleton of branch
    6) Do soma skeleton stitching using all the branches and those returning
    from step 4
    7) return skeleton

    """
    print("\n\n Inside New skeletonization recursive calls\n\n")
    
    if len(soma_mesh_list) == 0:
        raise Exception("soma_mesh_list was empty")
    else:
        soma_mesh_list = list(soma_mesh_list)
    
    #0) If don't have the soma_mesh centers then calculate
    if len(soma_mesh_list_centers) != len(soma_mesh_list):
        soma_mesh_list_centers = sm.find_soma_centroids(soma_mesh_list)
    
    #1) Start with the first soma and subtract from mesh
    #2) Find all of the disconnected mesh pieces
    current_soma = soma_mesh_list.pop(0)
    current_soma_index = soma_mesh_list_indexes.pop(0)
    current_soma_center = soma_mesh_list_centers.pop(0)
    mesh_pieces = sm.subtract_soma(current_soma,main_mesh)
    print(f"currently working on soma index {current_soma_index}")
    
    print(f"mesh_pieces after the soma subtraction = {len(mesh_pieces)}")
    
    if len(mesh_pieces) < 1:
        #just return an empty list
        print("No significant pieces after soma cancellation so just using the soma center as the skeleton")
        return np.vstack([current_soma_center,current_soma_center]).reshape(-1,2,3)
        
    
    #3) If there is still a soma piece that has not been processed, 
    total_soma_skeletons = []
    
    if len(soma_mesh_list) > 0:
        #find mesh pieces and all the somas that are contained within that
        containing_mesh_indices = sm.find_soma_centroid_containing_meshes(
                                            soma_mesh_list_centers,
                                            mesh_pieces
        )
        
        # rearrange into lists of somas per mesh soma 
        meshes_mapped_to_somas = sm.grouping_containing_mesh_indices(containing_mesh_indices)
        
        #get all of the other mesh pieces that weren't a part of the soma containing
        mesh_pieces_with_soma = list(meshes_mapped_to_somas.keys())
        non_soma_branches = [k for i,k in enumerate(mesh_pieces) if i not in mesh_pieces_with_soma]

        print(f"meshes_mapped_to_somas = {meshes_mapped_to_somas}")
        
        #recursive call to the function to find all those skeletons for the 
        #mesh groupings 
        for mesh_idx,soma_list in meshes_mapped_to_somas.items():
            mesh_soma_list = [k for i,k in enumerate(soma_mesh_list) if i in soma_list]
            mesh_soma_list_indexes = [k for i,k in enumerate(soma_mesh_list_indexes) if i in soma_list]
            mesh_soma_list_centers = [k for i,k in enumerate(soma_mesh_list_centers) if i in soma_list]
            
            print(f"mesh_soma_list = {mesh_soma_list}\n"
                f"mesh_soma_list_indexes = {mesh_soma_list_indexes}\n"
                 f"mesh_soma_list_centers = {mesh_soma_list_centers}\n")
            
            soma_mesh_skeleton = recursive_soma_skeletonization(
                                  mesh_pieces[mesh_idx],
                                  soma_mesh_list=mesh_soma_list,
                                    soma_mesh_list_indexes = mesh_soma_list_indexes,
                                  soma_mesh_list_centers=mesh_soma_list_centers,
                                mesh_base_path=mesh_base_path,
                                current_name=current_name
            )
            
            total_soma_skeletons.append(soma_mesh_skeleton)
        
        
        
    
    else:
        non_soma_branches = mesh_pieces
    
    
    print(f"non_soma_branches = {len(non_soma_branches)}")
    print(f"mesh_pieces = {len(mesh_pieces)}")

    
    #5) For all other pieces that do not have a soma do skeleton of branch
    for dendrite_index,picked_dendrite in enumerate(non_soma_branches):
        dendrite_name=current_name + f"_soma_{current_soma_index}_branch_{dendrite_index}"
        
        print(f"\n\nWorking on {dendrite_name}")
        stitched_dendrite_skeleton = skeletonize_connected_branch(picked_dendrite,
                                                       output_folder=mesh_base_path,
                                                       name=dendrite_name,
                                                        skeleton_print = True)
        
        if len(stitched_dendrite_skeleton)<=0:
                print(f"*** Dendrite {dendrite_index} did not have skeleton computed***")
        else: 
            total_soma_skeletons.append(stitched_dendrite_skeleton)
    
    #stitching together the soma parts:
    soma_stitched_skeleton = soma_skeleton_stitching(total_soma_skeletons,current_soma)
    
    #return the stitched skeleton
    return soma_stitched_skeleton
    




    
def skeletonize_neuron(main_mesh_total,
                        segment_id = 12345,
                        soma_mesh_list = [],
                       mesh_base_path="",
                       current_name="",
                       filter_end_node_length=5000,
                       sig_th_initial_split=15,

                        ):
    """
    Purpose: to skeletonize a neuron
    
    Example of How to Use:
    
    neuron_file = '/notebooks/test_neurons/91216997676870145_excitatory_1.off'
    current_mesh = trimesh.load_mesh(neuron_file)
    segment_id = 91216997676870145
    html_path = neuron_file[:-4] + "_skeleton.html"
    current_mesh
    
    new_cleaned_skeleton = skeletonize_neuron(main_mesh_total=current_mesh,
                            segment_id = segment_id,
                           mesh_base_path="",
                           current_name="",

                            )

    new_cleaned_skeleton.shape
    
    """
    import skeleton_utils as sk
    global_time = time.time()
    
    #if no soma is provided then do own soma finding
    if len(soma_mesh_list) == 0:
        print("\nComputing Soma because none given")
        (soma_mesh_list, 
             run_time, 
             total_soma_list_sdf) = soma_utils.extract_soma_center(
                                segment_id,
                                main_mesh_total.vertices,
                                main_mesh_total.faces,
                                outer_decimation_ratio= 0.25,
                                large_mesh_threshold = 60000,
                                large_mesh_threshold_inner = 40000,
                                soma_width_threshold = 0.32,
                                soma_size_threshold = 20000,
                               inner_decimation_ratio = 0.25,
                               volume_mulitplier=7,
                               side_length_ratio_threshold=3,
                                soma_size_threshold_max=192000,
                                delete_files=True
            )
    else:
        print(f"Not computing soma because list already given: {soma_mesh_list}")
        
        
    
    if len(soma_mesh_list) <= 0:
        print(f"**** No Somas Found for Mesh {segment_id} so just one mesh")
        soma_mesh_list_centers = []
    else:
        #compute the soma centers
        print(f"Soma List = {soma_mesh_list}")
        
        soma_mesh_list_centers = sm.find_soma_centroids(soma_mesh_list)
        print(f"soma_mesh_list_centers = {soma_mesh_list_centers}")

    
    split_meshes = split_significant_pieces(
                            main_mesh_total,
                            significance_threshold=sig_th_initial_split,
                            print_flag=False)
    
    
    """
    Pseudocode: 
    For all meshes in list
    1) compute soma center
    2) Find all the bounding boxes that contain the soma center
    3) Find the mesh with the closest distance from 
       one vertex to the soma center and tht is winner
    """
    
    
    #returns the index of the split_meshes index that contains each soma    
    containing_mesh_indices = sm.find_soma_centroid_containing_meshes(soma_mesh_list_centers,
                                            split_meshes)
    
    non_soma_touching_meshes = [m for i,m in enumerate(split_meshes)
                     if i not in list(containing_mesh_indices.values())]
    
    
    #Adding the step that will filter away any pieces that are inside the soma
    if len(non_soma_touching_meshes) > 0 and len(soma_mesh_list) > 0:
        non_soma_touching_meshes = soma_utils.filter_away_inside_soma_pieces(soma_mesh_list,non_soma_touching_meshes,
                                        significance_threshold=sig_th_initial_split)
        
    
    print(f"# of non soma touching seperate meshes = {len(non_soma_touching_meshes)}")
    print(f"# of soma containing seperate meshes = {len(np.unique(list(containing_mesh_indices.values())))}")
    
    print(f"contents of soma containing seperate meshes = {np.unique(list(containing_mesh_indices.values()))}")
    
    
    # setting the base path and the current name
    if mesh_base_path == "":
        mesh_base_path = Path(f"./{segment_id}")
    else:
        mesh_base_path = Path(mesh_base_path)
        
    if current_name == "":
        current_name = f"{segment_id}"
        
    if mesh_base_path.exists():
        rmtree(str(mesh_base_path.absolute()))
    mesh_base_path.mkdir(parents=True,exist_ok=True)
    print(list(mesh_base_path.iterdir()))
    
    """
    Pseudocode for better skeletonization of the multi-soma cases:
    Have containing_mesh_indices that has the indices of the mesh that contain each soma
    
    Recursive function: 
    0) divide into soma indices that correspond to the same mesh indices
    For each group that corresponds to same mesh indices
    1) Start with the first soma and subtract from mesh
    2) Find all of the disconnected mesh pieces
    3) If there is still a soma piece that has not been processed, find the soma piece that is containing each soma and make into groups
    4) skeletonize all of the pieces that not have somas associated with them
    - if have lists from step 3, call the function for each of them, 
    4b) once recieve all of the skeletons then stitch together on that soma
    
    """
    
    
    
    #------ do the skeletonization of the soma touchings --------#
    print("\n\n ---- Working on soma touching skeletons ------")

    soma_touching_time = time.time()
    
    
    
    """ OLD WAY OF DOING THE SKELETONS FOR THE SOMA TOUCHING THAT DOES DOUBLE SKELETONIZATION 
    
    # ***** this part will have a repeat of the meshes that contain the soma *** #
    soma_touching_meshes = dict([(i,split_meshes[m_i]) 
                                 for i,m_i in containing_mesh_indices.items()])
    soma_touching_meshes_skeletons = []
    
    
    for s_i,main_mesh in soma_touching_meshes.items():
        #Do the mesh subtraction to get the disconnected pieces
        current_soma = soma_mesh_list[s_i]

        mesh_pieces = sm.subtract_soma(current_soma,main_mesh)
        print(f"mesh_pieces after the soma subtraction = {len(mesh_pieces)}")
        #get each branch skeleton
        total_soma_skeletons = []
        for dendrite_index,picked_dendrite in enumerate(mesh_pieces):
            dendrite_name=current_name + f"_soma_{s_i}_branch_{dendrite_index}"
            print(f"\n\nWorking on {dendrite_name}")
            stitched_dendrite_skeleton = skeletonize_connected_branch(picked_dendrite,
                                                           output_folder=mesh_base_path,
                                                           name=dendrite_name,
                                                            skeleton_print = True)

            if len(stitched_dendrite_skeleton)<=0:
                print(f"*** Dendrite {dendrite_index} did not have skeleton computed***")
            else: 
                total_soma_skeletons.append(stitched_dendrite_skeleton)

    
    
    #stitching together the soma parts:
    soma_stitched_skeleton = soma_skeleton_stitching(total_soma_skeletons,current_soma)
    
    """
    
    
    # ---------------------- NEW WAY OF DOING THE SKELETONIZATION OF THE SOMA CONTAINING PIECES ------- #
    # rearrange into lists of somas per mesh soma 
    meshes_mapped_to_somas = sm.grouping_containing_mesh_indices(containing_mesh_indices)

    print(f"meshes_mapped_to_somas = {meshes_mapped_to_somas}")

    soma_stitched_skeleton = []
    soma_mesh_list_indexes = list(np.arange(len(soma_mesh_list_centers)))
    
    #recursive call to the function to find all those skeletons for the 
    #mesh groupings 
    for mesh_idx,soma_list in meshes_mapped_to_somas.items():
        mesh_soma_list = [k for i,k in enumerate(soma_mesh_list) if i in soma_list]
        mesh_soma_list_indexes = [k for i,k in enumerate(soma_mesh_list_indexes) if i in soma_list]
        mesh_soma_list_centers = [k for i,k in enumerate(soma_mesh_list_centers) if i in soma_list]

        print(f"mesh_soma_list = {mesh_soma_list}\n"
            f"mesh_soma_list_indexes = {mesh_soma_list_indexes}\n"
             f"mesh_soma_list_centers = {mesh_soma_list_centers}\n")


        soma_mesh_skeleton = recursive_soma_skeletonization(
                                      split_meshes[mesh_idx],
                                      soma_mesh_list=mesh_soma_list,
                                        soma_mesh_list_indexes = mesh_soma_list_indexes,
                                      soma_mesh_list_centers=mesh_soma_list_centers,
                                    mesh_base_path=mesh_base_path,
                                    current_name=current_name
        )

        soma_stitched_skeleton.append(soma_mesh_skeleton)
        
    print(f"Total time for soma touching skeletons: {time.time() - soma_touching_time}")
    # ----------------------DONE WITH SKELETONIZATION OF THE SOMA CONTAINING PIECES ------- #
    
    
    #------ do the skeletonization of the NON soma touchings --------#
    print("\n\n ---- Working on non-soma touching skeletons ------")
    non_soma_time = time.time()

    non_soma_touching_meshes

    total_non_soma_skeletons = []
    for j,picked_non_soma_branch in enumerate(non_soma_touching_meshes):
    #     if j<66:
    #         continue
        dendrite_name=current_name + f"_non_soma_{j}"
        print(f"\n\nWorking on {dendrite_name}")
        stitched_dendrite_skeleton = skeletonize_connected_branch(picked_non_soma_branch,
                                                       output_folder=mesh_base_path,
                                                       name=dendrite_name,
                                                        skeleton_print = True)

        if len(stitched_dendrite_skeleton)<=0:
            print(f"*** Dendrite {dendrite_index} did not have skeleton computed***")
        else: 
            total_non_soma_skeletons.append(stitched_dendrite_skeleton)


    print(f"Time for non-soma skeletons = {time.time() - non_soma_time}")
    
    # --------- Doing the stitching of the skeletons -----------#
    try:
        stacked_non_soma_skeletons = stack_skeletons(total_non_soma_skeletons)
    except:
        print(f"stacked_non_soma_skeletons = {stacked_non_soma_skeletons}")
        raise Exception("stacked_non_soma_skeletons stack failed ")
    
    try:
        stacked_soma_skeletons = stack_skeletons(soma_stitched_skeleton)
    except:
        print(f"soma_stitched_skeleton = {soma_stitched_skeleton}")
        raise Exception("soma_stitched_skeleton stack failed ")
    
    
    try:
        whole_skeletons_for_stitching = stack_skeletons([stacked_non_soma_skeletons,stacked_soma_skeletons])
    except: 
        print(f"[stacked_non_soma_skeletons,stacked_soma_skeletons] = {[stacked_non_soma_skeletons,stacked_soma_skeletons]}")
        raise Exception("[stacked_non_soma_skeletons,stacked_soma_skeletons] stack failed")

    final_skeleton_pre_clean = stitch_skeleton(
                                                      whole_skeletons_for_stitching,
                                                      stitch_print = False,
                                                      main_mesh = []
                                                    )
    
    # --------  Doing the cleaning ------- #
    clean_time = time.time()
    new_cleaned_skeleton = clean_skeleton(final_skeleton_pre_clean,
                            distance_func=skeletal_distance,
                      min_distance_to_junction=filter_end_node_length,
                      return_skeleton=True,
                      print_flag=False)
    print(f"Total time for skeleton clean {time.time() - clean_time}")
    
    print(f"\n\n\n\nTotal time for whole skeletonization of neuron = {time.time() - global_time}")
    return new_cleaned_skeleton



# ------ Functions to help with the compartment ---- #
# converted into a function
import networkx_utils as xu
import networkx as nx
import matplotlib.pyplot as plt

def get_ordered_branch_nodes_coordinates(skeleton_graph,nodes=False,coordinates=True):

    """Purpose: want to get ordered skeleton coordinates:
    1) get both end nodes
    2) count shortest path between them (to get order)
    3) then use the get node attributes function

    """
    #find the 2 endpoints:
    sk_graph_clean = xu.remove_selfloops(skeleton_graph)
    enpoints = [k for k,v in dict(sk_graph_clean.degree).items() if v == 1]
    #print(f"enpoints= {enpoints}")
    if len(enpoints) != 2:
        nx.draw(sk_graph_clean)
        print(f"sk_graph_clean.degree = {dict(sk_graph_clean.degree).items() }")
        nx.draw(skeleton_graph,with_labels=True)
        plt.show()
        raise Exception("The number of endpoints was not 2 for a branch")

    # gets the shortest path
    shortest_path = nx.shortest_path(sk_graph_clean,enpoints[0],enpoints[1],weight="weight")
    #print(f"shortest_path = {shortest_path}")

    skeleton_node_coordinates = xu.get_node_attributes(skeleton_graph,node_list=shortest_path)
    #print(f"skeleton_node_coordinates = {skeleton_node_coordinates}")

    if nodes==False and coordinates==True:
        return skeleton_node_coordinates
    elif nodes==True and coordinates==False:
        return shortest_path
    elif nodes==True and coordinates==True:
        return shortest_path,skeleton_node_coordinates
    else:
        raise Exception("neither nodes or coordinates set to return from get_ordered_branch_nodes_coordinates")


def split_skeleton_into_edges(current_skeleton):
    """
    Purpose: Will split a skeleton into a list of skeletons where each skeleton is just
    one previous edge of the skeleton before
    
    Example of how to use: 
    
    returned_split = split_skeleton_into_edges(downsampled_skeleton)
    print(len(returned_split), downsampled_skeleton.shape)
    returned_split
    
    """
    
    total_skeletons = [k for k in current_skeleton]
    return total_skeletons
    
        
def decompose_skeleton_to_branches(current_skeleton,
                                   max_branch_distance=-1,
                                  skip_branch_threshold=20000,
                                  return_indices=False,
                                  remove_cycles=True):
    """
    Example of how to run: 
    elephant_skeleton = sk.read_skeleton_edges_coordinates("../test_neurons/elephant_skeleton.cgal")
    elephant_skeleton_branches = sk.decompose_skeleton_to_branches(elephant_skeleton)
    sk.graph_skeleton_and_mesh(other_skeletons=[sk.stack_skeletons(elephant_skeleton_branches)])
    
    ***** Future error possibly: there could be issues in the future where had triangles of degree > 2 in your skeleton******
    """
    
    if type(current_skeleton) not in [type(nx.Graph()),type(xu.GraphOrderedEdges())]:
        el_sk_graph = convert_skeleton_to_graph(current_skeleton)
    else:
        el_sk_graph = current_skeleton
    
    
    
    el_sk_graph = xu.remove_selfloops(el_sk_graph)
    degree_dict = dict(el_sk_graph.degree)
    branch_nodes = [k for k,v in degree_dict.items() if v <= 2]
    seperated_branch_graph = el_sk_graph.subgraph(branch_nodes)
    
    branch_skeletons = []
    branch_skeleton_indices = []
    max_cycle_iterations = 1000

    seperated_branch_graph_comp = list(nx.connected_components(seperated_branch_graph))
    # now add back the nodes that were missing for each branch and collect all of the skeletons
    for curr_branch in seperated_branch_graph_comp:
        """
        new method of decomposing that avoids keeping loops (but will error if getting rid of large loop)
        
        # old way 
        all_neighbors = [xu.get_neighbors(el_sk_graph,n) for n in curr_branch] 
        all_neighbors.append(list(curr_branch))
        total_neighbors = np.unique(np.hstack(all_neighbors))
        branch_subgraph = el_sk_graph.subgraph(total_neighbors)
        branch_skeletons.append(sk.convert_graph_to_skeleton(branch_subgraph))
        
        New method: only if the two endpoints are connected together, then we give 
        back a skeleton just of those endpoints (so this will skip the current branch alltogether)
        --> but if skipping a branch that is too big then error
        - else do the old method
        
        """
        
        all_neighbors = [xu.get_neighbors(el_sk_graph,n) for n in curr_branch] 
        all_neighbors.append(list(curr_branch))
        total_neighbors = np.unique(np.hstack(all_neighbors))
        
        #check to see if the t junctions are connected
        high_degree_neigh = [k for k in total_neighbors if degree_dict[k]>2]
        if len(high_degree_neigh) > 2:
            raise Exception("Too many high degree nodes found in branch of decomposition")
        if len(high_degree_neigh) == 2:
            if high_degree_neigh[1] in xu.get_neighbors(el_sk_graph,high_degree_neigh[0]):
                print("high-degree endpoints were connected so just using that connection")
                
                #check that what skipping isn't too big
                print(f"curr_branch = {curr_branch}")
                if len(curr_branch) >= 2:
                    branch_subgraph = el_sk_graph.subgraph(list(curr_branch))
                    skip_distance = sk.calculate_skeleton_distance( sk.convert_graph_to_skeleton(branch_subgraph))
                    if  skip_distance > skip_branch_threshold:
                        raise Exception(f"Branch that we are skipping is too large with skip distance: {skip_distance}")

                #save this for later when add back all high degree branches that are connected
#                 branch_skeletons.append((xu.get_node_attributes(el_sk_graph,attribute_name="coordinates"
#                                                                 ,node_list=high_degree_neigh,
#                                                                return_array=True)).reshape(1,2,3))
                continue

        
        
        
        
        branch_subgraph = el_sk_graph.subgraph(total_neighbors)
        
        #12/17 NO LONGER attempting to eliminate any cycles
        if remove_cycles:
            branch_subgraph = xu.remove_cycle(branch_subgraph)
        
        branch_skeletons.append(sk.convert_graph_to_skeleton(branch_subgraph))
        branch_skeleton_indices.append(list(branch_subgraph.nodes()))
        
    #observation: seem to be losing branches that have two high degree nodes connected to each other and no other loop around it
        
    #add back all of the high degree branches that form subgraphs
    high_degree_branch_nodes = [k for k,v in degree_dict.items() if v > 2]
    seperated_branch_graph = el_sk_graph.subgraph(high_degree_branch_nodes)
    #get the connected components
    high_degree_conn_comp = nx.connected_components(seperated_branch_graph)
    
    """
    Here is where need to make a decision about what to do with high degree nodes: 
    I say just split all of the edges just into branches and then problem is solved (1 edge branches)
    """
    
    
    for c_comp in high_degree_conn_comp:
        if len(c_comp) >= 2:
            #add the subgraph to the branches
            branch_subgraph = el_sk_graph.subgraph(list(c_comp))
            branch_subgraph = nx. nx.minimum_spanning_tree(branch_subgraph)
            #and constant loop that check for cycle in this complexand if there is one then delete a random edge from the cycle

            """
            Might have to add in more checks for more complicated high degree node complexes

            """
            
            #new method that will delete any cycles might find in the branches
        
            #12/17 NO LONGER attempting to eliminate any cycles
            if remove_cycles:
                branch_subgraph = xu.remove_cycle(branch_subgraph)
        
            high_degree_branch_complex = sk.convert_graph_to_skeleton(branch_subgraph)
            seperated_high_degree_edges = split_skeleton_into_edges(high_degree_branch_complex)
                    
            #branch_skeletons.append(sk.convert_graph_to_skeleton(branch_subgraph)) #old way
            branch_skeletons += seperated_high_degree_edges
            branch_skeleton_indices += list(branch_subgraph.edges())
            
            
            #check if there every was a cycle: 
            

    for br in branch_skeletons:
        try:
            #print("Testing for cycle")
            edges_in_cycle = nx.find_cycle(sk.convert_skeleton_to_graph(br))
        except:
            pass
        else:
            raise Exception("There was a cycle found in the branch subgraph")
        
    branch_skeletons = [b.reshape(-1,2,3) for b in branch_skeletons]
    
    if return_indices:
        return branch_skeletons,branch_skeleton_indices
    else:
        return branch_skeletons

def convert_branch_graph_to_skeleton(skeleton_graph):
    """ Want an ordered skeleton that is only a line 
    Pseudocode: 
    1) Get the ordered node coordinates
    2) Create an edge array like [(0,1),(1,2).... (n_nodes-1,n_nodes)]
    3) index the edges intot he node coordinates and return
    """
    skeleton_node_coordinates = get_ordered_branch_nodes_coordinates(skeleton_graph)
    #print(f"skeleton_node_coordinates.shape = {skeleton_node_coordinates.shape}")
    s = np.arange(0,len(skeleton_node_coordinates)).T
    edges = np.vstack([s[:-1],s[1:]]).T
    return skeleton_node_coordinates[edges]    


# def divide_branch(curr_branch_skeleton,
#                            segment_width):
#     """
#     When wanting to divide up one branch into multiple branches that 
#     don't exceed a certain size
    
#     Pseudocode: 
#     1) Resize the skee
    
#     """

def resize_skeleton_branch(
                            curr_branch_skeleton,
                           segment_width = 0,
                          n_segments = 0,
                            print_flag=False):
    
    """
    sk = reload(sk)
    cleaned_skeleton = sk.resize_skeleton_branch(curr_branch_skeleton,segment_width=1000)

    sk.graph_skeleton_and_mesh(other_meshes=[curr_branch_mesh],
                              other_skeletons=[cleaned_skeleton])
    """
    
    if segment_width<=0 and n_segments<=0:
        raise Exception("Both segment_width and n_segments are non-positive")
    
    
    #curr_branch_nodes_coordinates = np.vstack([curr_branch_skeleton[:,0,:].reshape(-1,3),curr_branch_skeleton[-1,1,:].reshape(-1,3)])
    #print(f"curr_branch_nodes_coordinates = {curr_branch_nodes_coordinates}")  

    #final product of this is it gets a skeleton that goes in a line from one endpoint to the other 
    #(because skeleton can possibly be not ordered)
    skeleton_graph = sk.convert_skeleton_to_graph(curr_branch_skeleton)
    skeleton_node_coordinates = get_ordered_branch_nodes_coordinates(skeleton_graph)
    cleaned_skeleton = convert_branch_graph_to_skeleton(skeleton_graph)

    # #already checked that these were good                 
    # print(f"skeleton_node_coordinates = {skeleton_node_coordinates}")
    # print(f"cleaned_skeleton = {cleaned_skeleton}")


    # gets the distance markers of how far have traveled from end node for each node
    seg_bins = np.hstack([np.array([0]),sk.calculate_skeleton_segment_distances(cleaned_skeleton,cumsum=True)])

    if n_segments > 0:
            segment_width = seg_bins[-1]/n_segments #sets the width to 
            if print_flag:
                print(f"segment_width = {segment_width}")
    else:
        if segment_width>seg_bins[-1]:
            #print("Skeletal width required was longer than the current skeleton so just returning the endpoints")
            return np.vstack([cleaned_skeleton[0][0],cleaned_skeleton[-1][-1]]).reshape(1,2,3)
    

    #gets the distance of each segment
    segment_widths = sk.calculate_skeleton_segment_distances(cleaned_skeleton,cumsum=False)
    #print(f"total_distance = {sk.calculate_skeleton_distance(cleaned_skeleton)}")

    n_full_segs = int(seg_bins[-1]/segment_width)
    new_seg_endpoints = np.arange(segment_width,segment_width*n_full_segs+0.01,segment_width)

    if new_seg_endpoints[-1] > seg_bins[-1]:
        raise Exception("End of new_seg_endpoints is greater than original skeleton ")

    #accounts for the perfect fit
    if new_seg_endpoints[-1] == seg_bins[-1]:
        #print("exact match so eliminating last new bin")
        new_seg_endpoints = new_seg_endpoints[:-1] #remove the last one because will readd it back

    #print(f"seg_bins = {seg_bins}")
    #print(f"new_seg_endpoints = {new_seg_endpoints}")

    #getting the vertices

    """
    3) for each new segment endpoint, 
    a) calculate between which two existing skeleton segment end points it would exist
    (using just a distnace measurement from each end point to the next)
    b)calculate the coordinate that is a certain distance in middle based on which endpoints between

    new_vector * (new_seg_endpoint - lower_bin_distance)/seg_width + lower_bin_vector
    # """

    bin_indices = np.digitize(new_seg_endpoints, seg_bins)
    #print(f"bin_indices = {bin_indices}")
    # print(f"bin_indices = {bin_indices}")
    # print(f"seg_bins[bin_indices-1]= {seg_bins[bin_indices-1]}")
    # print(f"new_seg_endpoints - seg_bins[bin_indices-1] = {(new_seg_endpoints - seg_bins[bin_indices-1]).astype('int')}")
    #print(f"skeleton_node_coordinates (SHOULD BE ORDERED) = {skeleton_node_coordinates}")
    new_coordinates = (((skeleton_node_coordinates[bin_indices] - skeleton_node_coordinates[bin_indices-1])
                       *((new_seg_endpoints - seg_bins[bin_indices-1])/segment_widths[bin_indices-1]).reshape(-1,1)) + skeleton_node_coordinates[bin_indices-1])

    #print(f"new_coordinates = {new_coordinates.shape}")

    #add on the ending coordinates
    final_new_coordinates = np.vstack([skeleton_node_coordinates[0].reshape(-1,3),new_coordinates,skeleton_node_coordinates[-1].reshape(-1,3)])
    #print(f"final_new_coordinates = {final_new_coordinates.shape}")

    #make a new skeleton from the coordinates
    new_skeleton = np.stack((final_new_coordinates[:-1],final_new_coordinates[1:]),axis=1)
    if print_flag:
        print(f"new_skeleton = {new_skeleton.shape}")

    return new_skeleton


from scipy.spatial.distance import pdist,squareform
def skeleton_graph_nodes_to_group(skeleton_grpah):
    """
    Checks that no nodes in graph are in the same coordinates and need to be combined
    
    Example Use Case: 
    
    example_skeleton = current_mesh_data[0]["branch_skeletons"][0]
    skeleton_grpah = sk.convert_skeleton_to_graph(example_skeleton)
    limb_nodes_to_group = sk.skeleton_graph_nodes_to_group(skeleton_grpah)
    limb_nodes_to_group

    #decompose the skeleton and then recompose and see if any nodes to group
    decomposed_branches = sk.decompose_skeleton_to_branches(example_skeleton)
    decomposed_branches_stacked = sk.stack_skeletons(example_skeleton)
    restitched_decomposed_skeleton = sk.convert_graph_to_skeleton(sk.convert_skeleton_to_graph(decomposed_branches_stacked))
    sk.skeleton_graph_nodes_to_group(restitched_decomposed_skeleton)

    #shows that the restitched skeleton is still just one connected componet
    connected_components = nx.connected_components(sk.convert_skeleton_to_graph(decomposed_branches_stacked))
    len(list(connected_components))

    sk.graph_skeleton_and_mesh(other_skeletons = [restitched_decomposed_skeleton])
    
    
    """
    if type(skeleton_grpah)  not in [type(nx.Graph()),type(xu.GraphOrderedEdges())]:
        skeleton_grpah = convert_skeleton_to_graph(skeleton_grpah)
    #get all of the vertices
    coordinates = xu.get_node_attributes(skeleton_grpah,attribute_name="coordinates")
    #get the distances between coordinates
    distance_matrix = nu.get_coordinate_distance_matrix(coordinates)
    
    #great a graph out of the distance matrix with a value of 0
    nodes_to_combine = nx.from_edgelist(np.array(np.where(distance_matrix==0)).T)
    #clean graph for any self loops
    nodes_to_combine  = xu.remove_selfloops(nodes_to_combine)
    
    grouped_nodes = nx.connected_components(nodes_to_combine)
    nodes_to_group = [k for k in list(grouped_nodes) if len(k)>1]
    
    return nodes_to_group

def recompose_skeleton_from_branches(decomposed_branches):
    """
    Takes skeleton branches and stitches them back together without any repeating nodes
    """
    decomposed_branches_stacked = sk.stack_skeletons(decomposed_branches)
    restitched_decomposed_skeleton = sk.convert_graph_to_skeleton(sk.convert_skeleton_to_graph(decomposed_branches_stacked))
    return restitched_decomposed_skeleton

def clean_skeleton_with_decompose(distance_cleaned_skeleton):
    """
    Purpose: to eliminate the loops that are cleaned in the decompose process from the skeleton and then reconstruct
    Pseudocode: 
    1) decompose skeleton
    2) recompose skeleton (was checked that no nodes to recombine)
    
    """
    branches = decompose_skeleton_to_branches(distance_cleaned_skeleton)
    return recompose_skeleton_from_branches(branches)

def divide_branch(curr_branch_skeleton,
                            segment_width = 1000,
                           equal_width=True,
                           n_segments = 0):


    """
    When wanting to divide up one branch into multiple branches that 
    don't exceed a certain size

    Example of how to use: 
    
    sk = reload(sk)

    curr_index = 1
    ex_branch = total_branch_skeletons[curr_index]
    ex_mesh = total_branch_meshes[curr_index]
    # sk.graph_skeleton_and_mesh(other_skeletons=[ex_branch],
    #                           other_meshes=[ex_mesh])



    #there were empty arrays which is causing the error!
    returned_branches = sk.divide_branch(curr_branch_skeleton=ex_branch,
                                segment_width = 1000,
                                equal_width=False,
                                n_segments = 0)

    print(len(returned_branches))
    lengths = [sk.calculate_skeleton_distance(k) for k in returned_branches]
    print(f"lengths = {lengths}")


    sk.graph_skeleton_and_mesh(
                                other_skeletons=returned_branches[:10],
                            other_skeletons_colors=["black"],
                              #other_skeletons=[ex_branch],
                              other_meshes=[ex_mesh])

    """

    if segment_width<=0 and n_segments<=0:
        raise Exception("Both segment_width and n_segments are non-positive")

    skeleton_graph = sk.convert_skeleton_to_graph(curr_branch_skeleton)
    skeleton_node_coordinates = get_ordered_branch_nodes_coordinates(skeleton_graph)
    cleaned_skeleton = convert_branch_graph_to_skeleton(skeleton_graph)

    seg_bins = np.hstack([np.array([0]),sk.calculate_skeleton_segment_distances(cleaned_skeleton,cumsum=True)])



    if n_segments > 0:
            segment_width = seg_bins[-1]/n_segments
    else:
        if segment_width>seg_bins[-1]:
            #print("Skeletal width required was longer than the current skeleton so just returning the endpoints")
            return [np.vstack([cleaned_skeleton[0][0],cleaned_skeleton[-1][-1]]).reshape(1,2,3)]


    segment_widths = sk.calculate_skeleton_segment_distances(cleaned_skeleton,cumsum=False)
    #print(f"total_distance = {sk.calculate_skeleton_distance(cleaned_skeleton)}")

    if equal_width and n_segments <= 0:
        #print("making all of the branch segments equal width")
        n_segments_that_fit = seg_bins[-1]/segment_width
        #print(f"n_segments_that_fit = {n_segments_that_fit}")
        if n_segments_that_fit > int(n_segments_that_fit): #if there is some leftover 
            segment_width = seg_bins[-1]/np.ceil(n_segments_that_fit)
            #print(f"New segment width in order to make them equal = {segment_width}\n")

    n_full_segs = int(seg_bins[-1]/segment_width)
    #print(f"n_full_segs = {n_full_segs}")

    #old way
    new_seg_endpoints = np.arange(segment_width,segment_width*n_full_segs+0.01,segment_width)
    
    #print(f"new_seg_endpoints[-1] - seg_bins[-1] = {new_seg_endpoints[-1] - seg_bins[-1]}")
    if new_seg_endpoints[-1] > seg_bins[-1]:
        if new_seg_endpoints[-1] - seg_bins[-1] > 0.01:
            raise Exception("End of new_seg_endpoints is greater than original skeleton ")
        else:
            new_seg_endpoints[-1] =  seg_bins[-1]

    #accounts for the perfect fit
    if new_seg_endpoints[-1] == seg_bins[-1]:
        #print("exact match so eliminating last new bin")
        new_seg_endpoints = new_seg_endpoints[:-1] #remove the last one because will readd it back

    #print(f"seg_bins = {seg_bins}")
    #print(f"new_seg_endpoints = {new_seg_endpoints}")

    #getting the vertices

    """
    3) for each new segment endpoint, 
    a) calculate between which two existing skeleton segment end points it would exist
    (using just a distnace measurement from each end point to the next)
    b)calculate the coordinate that is a certain distance in middle based on which endpoints between

    new_vector * (new_seg_endpoint - lower_bin_distance)/seg_width + lower_bin_vector
    # """

    bin_indices = np.digitize(new_seg_endpoints, seg_bins)

    new_coordinates = (((skeleton_node_coordinates[bin_indices] - skeleton_node_coordinates[bin_indices-1])
                       *((new_seg_endpoints - seg_bins[bin_indices-1])/segment_widths[bin_indices-1]).reshape(-1,1)) + skeleton_node_coordinates[bin_indices-1])

    #these should be the same size
    #     print(f"bin_indices = {bin_indices}")
    #     print(f"new_coordinates = {new_coordinates}")
    #     return bin_indices,new_coordinates

    """
    Using the bin_indices and new_coordinates construct a list of branches with the original vertices plus the new cuts
    Pseudocode:

    indices mean that they are greater than or equal to the bin below but absolutely less than the bin indices value
    --> need to make sure that the new cut does not fall on current cut
    --> do this by checking that the last node before the cut isn't equal to the cut

    1) include all of the skeleton points but not including the bin idexed numer
    """
    returned_branches = []
    skeleton_node_coordinates #these are the original coordinates
    for z,(curr_bin,new_c) in enumerate(zip(bin_indices,new_coordinates)):
        if z==0:
#             print(f"curr_bin = {curr_bin}")
#             print(f"bin_indices = {bin_indices}")
            
            previous_nodes = skeleton_node_coordinates[:curr_bin]
#             print(f"previous_nodes = {previous_nodes}")
#             print(f"previous_nodes[-1] = {previous_nodes[-1]}")
#             print(f"new_c = {new_c}")
#             print(f"np.linalg.norm(previous_nodes[:-1]- new_c) = {np.linalg.norm(previous_nodes[-1]- new_c)}")
            if np.linalg.norm(previous_nodes[-1]- new_c) > 0.001:
                #print("inside linalg_norm")
                previous_nodes = np.vstack([previous_nodes,new_c.reshape(-1,3)])
            
            #print(f"previous_nodes = {previous_nodes}")
            #now create the branch
            returned_branches.append(np.stack((previous_nodes[:-1],previous_nodes[1:]),axis=1).reshape(-1,2,3))
            #print(f"returned_branches = {returned_branches}")
        else:
            #if this was not the first branch
            previous_nodes = new_coordinates[z-1].reshape(-1,3)
            if curr_bin > bin_indices[z-1]:
                previous_nodes = np.vstack([previous_nodes,skeleton_node_coordinates[bin_indices[z-1]:curr_bin].reshape(-1,3)])
            if np.linalg.norm(previous_nodes[-1]- new_c) > 0.001:
                previous_nodes = np.vstack([previous_nodes,new_c.reshape(-1,3)])

            returned_branches.append(np.stack((previous_nodes[:-1],previous_nodes[1:]),axis=1).reshape(-1,2,3))


    #     if np.array_equal(returned_branches[-1],np.array([], dtype="float64").reshape(-1,2,3)):
    #         print(f"previous_nodes= {previous_nodes}")
    #         print(f"new_c = {new_c}")
    #         print(f"curr_bin = {curr_bin}")
    #         print(f"bin_indices = {bin_indices}")
    #         print(f"z = {z}")
    #         raise Exception("stopping")

    #add this last section to the skeleton
    if np.linalg.norm(new_c - skeleton_node_coordinates[-1]) > 0.001: #so the last node has not been added yet
        previous_nodes = new_coordinates[-1].reshape(-1,3)
        if bin_indices[-1]<len(seg_bins):
            previous_nodes = np.vstack([previous_nodes,skeleton_node_coordinates[bin_indices[-1]:len(skeleton_node_coordinates)].reshape(-1,3)])
        else:
            previous_nodes = np.vstack([previous_nodes,skeleton_node_coordinates[-1].reshape(-1,3)])
        returned_branches.append(np.stack((previous_nodes[:-1],previous_nodes[1:]),axis=1).reshape(-1,2,3))
    
    #check 1: that the first and last of original branch is the same as the decomposed
    first_coord = returned_branches[0][0][0]
    last_coord = returned_branches[-1][-1][-1]
    
#     print(f"first original coord = {skeleton_node_coordinates[0]}")
#     print(f"last original coord = {skeleton_node_coordinates[-1]}")
#     print(f"first_coord = {first_coord}")
#     print(f"last_coord = {last_coord}")
    
    
    if not np.array_equal(skeleton_node_coordinates[0],first_coord):
        print(f"skeleton_node_coordinates = {skeleton_node_coordinates}")
        print(f"first_coord = {first_coord}")
        raise Exception("First coordinate does not match")
        
    if not np.array_equal(skeleton_node_coordinates[-1],last_coord):
        print(f"skeleton_node_coordinates = {skeleton_node_coordinates}")
        print(f"last_coord = {last_coord}")
        raise Exception("Last coordinate does not match")

    
    #check 2: that it is all one connected branch
    total_skeleton = sk.stack_skeletons(returned_branches)
    total_skeleton_graph = sk.convert_skeleton_to_graph(total_skeleton)
    n_comps = nx.number_connected_components(total_skeleton_graph)
    
    
    #print(f"Number of connected components is {n_comps}")
    
    if n_comps > 1:
        raise Exception(f"Number of connected components is {n_comps}")

    print(f"Total number of returning branches = {len(returned_branches)}")
    return returned_branches

# -------- for the mesh correspondence -------
# def waterfill_labeling(
#                 total_mesh_correspondence,
#                  submesh_indices,
#                  total_mesh=None,
#                 total_mesh_graph=None,
#                  propagation_type="random",
#                 max_iterations = 1000,
#                 max_submesh_threshold = 1000
#                 ):
#     """
#     Pseudocode:
#     1) check if the submesh you are propagating labels to is too large
#     2) for each unmarked face get the neighbors of all of the faces, and for all these neighbors get all the labels
#     3) if the neighbors label is not empty. depending on the type of progation type then pick the winning label
#     a. random: just randomly choose from list
#     b. .... not yet implemented
#     4) revise the faces that are still empty and repeat process until all faces are empty (have a max iterations number)
#     """
    
#     if not total_mesh_graph:
#         #finding the face adjacency:
#         total_mesh_graph = nx.from_edgelist(total_mesh.face_adjacency)
    
    
    
#     if len(submesh_indices)> max_submesh_threshold:
#         raise Exception(f"The len of the submesh ({len(submesh_indices)}) exceeds the maximum threshold of {max_submesh_threshold} ")
    
#     #check that these are unmarked
#     curr_unmarked_faces = [k for k in submesh_indices if total_mesh_correspondence[k] == -1] 
    
    
#     if len(curr_unmarked_faces)<len(submesh_indices):
#         raise Exception(f"{len(submesh_indices)-len(curr_unmarked_faces)} submesh faces were already labeled before waterfill_labeling started")
    
#     for i in range(max_iterations):
#         #s2) for each unmarked face get the neighbors of all of the faces, and for all these neighbors get all the labels
#         unmarked_faces_neighbors = [xu.get_neighbors(total_mesh_graph,j) for j in curr_unmarked_faces] #will be list of lists
#         #print(f"unmarked_faces_neighbors = {unmarked_faces_neighbors}")
#         unmarked_face_neighbor_labels = [np.array([total_mesh_correspondence[curr_neighbor] for curr_neighbor in z]) for z in unmarked_faces_neighbors]
#         #print(f"unmarked_face_neighbor_labels = {unmarked_face_neighbor_labels}")
        
#         if len(unmarked_face_neighbor_labels) == 0:
#             print(f"curr_unmarked_faces = {curr_unmarked_faces}")
#             print(f"i = {i}")
#             print(f"unmarked_faces_neighbors = {unmarked_faces_neighbors}")
#             print(f"unmarked_face_neighbor_labels = {unmarked_face_neighbor_labels}")
            
#         #check if there is only one type of label and if so then autofil
#         total_labels = list(np.unique(np.concatenate(unmarked_face_neighbor_labels)))
        
#         if -1 in total_labels:
#             total_labels.remove(-1)
        
#         if len(total_labels) == 0:
#             raise Exception("total labels does not have any marked neighbors")
#         elif len(total_labels) == 1:
#             print("All surrounding labels are the same so autofilling the remainder of unlabeled labels")
#             for gg in curr_unmarked_faces:
#                 total_mesh_correspondence[gg] = total_labels[0]
#             break
#         else:
#             #if there are still one or more labels surrounding our unlabeled region
#             for curr_face,curr_neighbors in zip(curr_unmarked_faces,unmarked_face_neighbor_labels):
#                 curr_neighbors = curr_neighbors[curr_neighbors != -1]
#                 if len(curr_neighbors) > 0:
#                     if propagation_type == "random":
#                         total_mesh_correspondence[curr_face] = np.random.choice(curr_neighbors)
#                     else:
#                         raise Exception("Not implemented propagation_type")
        
#         # now replace the new curr_unmarked faces
#         curr_unmarked_faces = [k for k in submesh_indices if total_mesh_correspondence[k] == -1] #old dict way
        
        
#         if len(curr_unmarked_faces) == 0:
#             print(f"breaking out of loop because zero unmarked faces left after {i} iterations")
#             break
        
    
#     #check that no more unmarked faces or error
#     end_unmarked_faces = [k for k in submesh_indices if total_mesh_correspondence[k] == -1]
    
#     if len(end_unmarked_faces) > 0:
#         raise Exception(f"After {i+1} iterations (with max_iterations = {max_iterations} there were still {len(end_unmarked_faces)} faces")
        
    
#     return total_mesh_correspondence


# ----- functions to help with the Neuron class ---- #
def find_branch_endpoints(db):
    db_graph = sk.convert_skeleton_to_graph(db)
    end_node_coordinates = xu.get_node_attributes(db_graph,node_list=xu.get_nodes_of_degree_k(db_graph,1))

    if len(end_node_coordinates) != 2:
        raise Exception("Not exactly 2 end nodes in the passed branch")
    else:
        return end_node_coordinates
    
def compare_skeletons_ordered(skeleton_1,skeleton_2,
                             edge_threshold=0.01, #how much the edge distances can vary by
                              node_threshold = 0.01, #how much the nodes can vary by
                              print_flag = False
                             ):
    """
    Purpose: To compare skeletons where the edges are ordered (not comparing overall skeletons)
    Those would be isomorphic graphs (not yet developed)
    
    Example of how to use: 
    skeletons_idx_to_stack = [0,1,2,3]
    total_skeleton = sk.stack_skeletons([double_soma_obj.concept_network.nodes["L1"]["data"].concept_network.nodes[k]["data"].skeleton for k in skeletons_idx_to_stack])
    #sk.graph_skeleton_and_mesh(other_skeletons=[total_skeleton])
    
    skeleton_1 = copy.copy(total_skeleton)
    skeleton_2 = copy.copy(total_skeleton)
    skeleton_1[0][0] = np.array([558916.8, 1122107. ,  842972.8]) #change so there will be error
    
    sk.compare_skeletons_ordered(skeleton_1,
                          skeleton_2,
                             edge_threshold=0.01, #how much the edge distances can vary by
                              node_threshold = 0.01, #how much the nodes can vary by
                              print_flag = True
                             )

    
    """
    sk_1_graph = convert_skeleton_to_graph(skeleton_1)
    sk_2_graph = convert_skeleton_to_graph(skeleton_2)

    return xu.compare_networks(sk_1_graph,sk_2_graph,print_flag=print_flag,
                     edge_comparison_threshold=edge_threshold,
                     node_comparison_threshold=node_threshold)
    
    
# ----------------- 7/22 Functions made to help with graph searching and visualizaiton ------------ #
def skeleton_n_components(curr_skeleton):
    """
    Purpose: To get the number of connected components represented by 
    the current skeleton
    """
    cleaned_branch_components = nx.number_connected_components(convert_skeleton_to_graph(curr_skeleton))
    return cleaned_branch_components

def check_skeleton_one_component(curr_skeleton):
    cleaned_branch_components = skeleton_n_components(curr_skeleton)
    if cleaned_branch_components > 1:
        raise Exception(f"Skeleton is not one component: n_components = {cleaned_branch_components}")
    

# ---------------- 9/17: Will help with creating branch points extending towards soma if not already exist ---
from pykdtree.kdtree import KDTree
import networkx_utils as xu
import time

def create_soma_extending_branches(
    current_skeleton, #current skeleton that was created
    skeleton_mesh, #mesh that was skeletonized
    soma_to_piece_touching_vertices,#dictionary mapping a soma it is touching to the border vertices,
    return_endpoints_must_keep=True,
    return_created_branch_info=False,
    try_moving_to_closest_sk_to_endpoint=True, #will try to move the closest skeleton point to an endpoint
    distance_to_move_point_threshold = 1500, #maximum distance willling to move closest skeleton point to get to an endpoint
    check_connected_skeleton=True
                                    ):
    """
    Purpose: To make sure there is one singular branch extending towards the soma
    
    Return value:
    endpoints_must_keep: dict mapping soma to array of the vertex points that must be kept
    because they are the soma extending branches of the skeleton
    
    Pseudocode: 
    Iterating through all of the somas and all of the groups of touching vertices
    a) project the skeleton and the soma boundary vertices on to the vertices of the mesh
    b) Find the closest skeleton point to the soma boundary vetices
    c) check the degree of the closest skeleton point:
    - if it is a degree one then leave alone
    - if it is not a degree one then create a new skeleton branch from the 
    closest skeleton point and the average fo the border vertices and add to 
    
    Extension: (this is the same method that would be used for adding on a floating skeletal piece)
    If we made a new skeleton branch then could pass back the closest skeleton point coordinates
    and the new skeleton segment so we could:
    1) Find the branch that it was added to 
    2) Divide up the mesh correspondnece between the new resultant branches
    -- would then still reuse the old mesh correspondence
    
    """
    endpoints_must_keep = dict()
    new_branches = dict()
    
    #0) Create a graph of the mesh from the vertices and edges and a KDTree
    start_time = time.time()
    vertex_graph = tu.mesh_vertex_graph(skeleton_mesh)
    mesh_KD = KDTree(skeleton_mesh.vertices)
    print(f"Total time for mesh KDTree = {time.time() - start_time}")
    
    for s_index,v in soma_to_piece_touching_vertices.items():
        
        endpoints_must_keep[s_index] = []
        new_branches[s_index]=[]
        
        for j,sbv in enumerate(v):


            #1)  Project all skeleton points and soma boundary vertices onto the mesh
            all_skeleton_points = np.unique(current_skeleton.reshape(-1,3),axis=0)
            sk_points_distances,sk_points_closest_nodes = mesh_KD.query(all_skeleton_points)

            #sbv = soma_to_piece_touching_vertices[s_index]
            print(f"sbv[0].reshape(-1,3) = {sbv[0].reshape(-1,3)}")
            soma_border_distances,soma_border_closest_nodes = mesh_KD.query(sbv[0].reshape(-1,3))

            
            ''' old way that relied on soley paths on the mesh graph
            start_time = time.time()
            #2) Find the closest skeleton point to the soma border (for that soma), find shortest path from many to many
            path,closest_sk_point,closest_soma_border_point = xu.shortest_path_between_two_sets_of_nodes(vertex_graph,sk_points_closest_nodes,soma_border_closest_nodes)
            print(f"Shortest path between 2 nodes = {time.time() - start_time}")

            #3) Find closest skeleton point
            closest_sk_pt = np.where(sk_points_closest_nodes==closest_sk_point)[0][0]
            closest_sk_pt_coord = all_skeleton_points[closest_sk_pt]
            '''
            
            """New Method 10/27
            1) applies a mesh filter for only those within a certian distance along mesh graph (filter)
            2) Of filtered vertices, finds one closest to soma border average
            
            """
            curr_cut_distane = 10000
            
            for kk in range(0,5):
                close_nodes = xu.find_nodes_within_certain_distance_of_target_node(vertex_graph,target_node=soma_border_closest_nodes[0],cutoff_distance=curr_cut_distane)
                filter_1_skeleton_points = np.array([sk_pt for sk_pt,sk_pt_node in zip(all_skeleton_points,sk_points_closest_nodes) if sk_pt_node in close_nodes])
                if len(filter_1_skeleton_points) >0:
                    break
                print(f"On iteration {kk} the filter points were empty with close_nodes len = {len(close_nodes)}, len(all_skeleton_points) = {len(all_skeleton_points)}, len(sk_points_closest_nodes) = {len(sk_points_closest_nodes)}")
                
                curr_cut_distane = curr_cut_distane*3
            
            if len(filter_1_skeleton_points) == 0:
                raise Exception (f"Still No filter nodes with curr_cut_distane = {curr_cut_distane}")
                    
                
            

            border_average_coordinate = np.mean(sbv,axis=0)

            closest_sk_point_idx = np.argmin(np.linalg.norm(filter_1_skeleton_points-border_average_coordinate,axis=1))
            closest_sk_pt_coord = filter_1_skeleton_points[closest_sk_point_idx]
            
            
            
            sk_graph = sk.convert_skeleton_to_graph(current_skeleton)
            
            distance_to_move_point_threshold
            if try_moving_to_closest_sk_to_endpoint:
                print(f"closest_sk_pt_coord BEFORE = {closest_sk_pt_coord}")
                print(f"current_skeleton.shape = {current_skeleton.shape}")
                closest_sk_pt_coord,change_status = move_point_to_nearest_branch_end_point_within_threshold(
                                                        skeleton=current_skeleton,
                                                        coordinate=closest_sk_pt_coord,
                                                        distance_to_move_point_threshold = distance_to_move_point_threshold,
                                                        verbose=True,
                                                        consider_high_degree_nodes=False

                                                        )
                print(f"change_status for create soma extending pieces = {change_status}")
                print(f"closest_sk_pt_coord AFTER = {closest_sk_pt_coord}")
            
            #find the node that has the desired vertices and its' degree
            sk_node = xu.get_nodes_with_attributes_dict(sk_graph,dict(coordinates=closest_sk_pt_coord))[0]
            sk_node_degree = sk_graph.degree()[sk_node]
    

            if sk_node_degree == 0:
                raise Exception("Found 0 degree node in skeleton")
                
            elif sk_node_degree == 1: #3a) If it is a node of degree 1 --> do nothing
                print(f"skipping soma {s_index} because closest skeleton node was already end node")
                endpoints_must_keep[s_index].append(closest_sk_pt_coord)
                new_branches[s_index].append(None)
                continue
            else:
                #3b) If Not endpoint:
                #Add an edge from the closest skeleton point coordinate to vertex average of all soma boundaries
                print("Adding new branch to skeleton")
                print(f"border_average_coordinate = {border_average_coordinate}")
                
                new_branch_sk = np.vstack([closest_sk_pt_coord,border_average_coordinate]).reshape(-1,2,3)
                current_skeleton = sk.stack_skeletons([current_skeleton,new_branch_sk])
                endpoints_must_keep[s_index].append(border_average_coordinate)
                
                #will store the newly added branch and the corresponding border vertices
                new_branches[s_index].append(dict(new_branch = new_branch_sk,border_verts=sbv))
                
        endpoints_must_keep[s_index] = np.array(endpoints_must_keep[s_index])
        
    print(f"endpoints_must_keep = {endpoints_must_keep}")
    #check if skeleton is connected component when finishes
    if check_connected_skeleton:
        if nx.number_connected_components(convert_skeleton_to_graph(current_skeleton)) != 1:
            su.compressed_pickle(current_skeleton,"current_skeleton")
            raise Exception("The skeleton at end wasn't a connected component")
    
    return_value = [current_skeleton]
    
    if return_endpoints_must_keep:
        return_value.append(endpoints_must_keep)
    if return_created_branch_info:
        return_value.append(new_branches)
    return return_value


import numpy_utils as nu
def find_branch_skeleton_with_specific_coordinate(divded_skeleton,current_coordinate):
    """
    Purpose: From list of skeletons find the ones that have a certain coordinate
    
    Example: 
    curr_limb = current_neuron[0]
    branch_names = curr_limb.get_branch_names(return_int=True)
    curr_limb_divided_skeletons = [curr_limb[k].skeleton for k in branch_names]
    ideal_starting_endpoint = curr_limb.current_starting_coordinate
    
    sk = reload(sk)
    sk.find_branch_skeleton_with_specific_coordinate(curr_limb_divided_skeletons,ideal_starting_endpoint)

    """
    matching_branch = []
    for b_idx,b_sk in enumerate(divded_skeleton):
        match_result = nu.matching_rows(b_sk.reshape(-1,3),current_coordinate)
        #print(f"match_result = {match_result}")
        if len(match_result)>0:
            matching_branch.append(b_idx)
    
    return matching_branch

#----------- 9/24 -------------- #
import networkx_utils as xu
def find_skeleton_endpoint_coordinates(skeleton):
    """
    Purpose: To find the endpoint coordinates 
    of a skeleton
    
    Application: 
    1) Can get the endpoints of a skeleton and 
    then check that none of the spines contain 
    an endpoint coordinate to help 
    guard against false spines at the endpoints
    
    Pseudocode:
    1) convert the skeleton to a graph
    2) Find the endpoint nodes of the graph (with degree 1)
    3) return the coordinates of the graph nodes
    
    """
    G = convert_skeleton_to_graph(skeleton)
    endpoint_nodes = xu.get_nodes_of_degree_k(G,degree_choice=1)
    endpoint_coordinates = xu.get_node_attributes(G,node_list=endpoint_nodes)
    return endpoint_coordinates


def path_ordered_skeleton(skeleton):
    """
    Purpose: To order the edges in sequential order in
    a skeleton so skeleton[0] is one end edge
    and skeleton[-1] is the other end edge
    
    Pseudocode: 
    How to order a skeleton: 
    1) turn the skeleton into a graph
    2) start at an endpoint node
    3) output the skeleton edges for the edges of the graph until hit the other end node


    
    Ex: 
    skeleton = big_neuron[0][30].skeleton
    new_skeleton_ordered = path_ordered_skeleton(skeleton)
    
    
    
    """

    #1) turn the skeleton into a graph
    G = convert_skeleton_to_graph(skeleton)
    #2) start at an endpoint node
    end_nodes = xu.get_nodes_of_degree_k(G,1)

    sk_node_path = nx.shortest_path(G,source=end_nodes[0],target=end_nodes[-1])
    sk_node_path_coordinates = xu.get_node_attributes(G,node_list=sk_node_path)

    ordered_skeleton = np.hstack([sk_node_path_coordinates[:-1],
                                  sk_node_path_coordinates[1:]]).reshape(-1,2,3)

    return ordered_skeleton



# ---------------------- For preprocessing of neurons revised ------------------ #
import system_utils as su
def skeletonize_and_clean_connected_branch_CGAL(mesh,
                       curr_soma_to_piece_touching_vertices=None,
                       total_border_vertices=None,
                        filter_end_node_length=4001,
                       perform_cleaning_checks=False,
                       combine_close_skeleton_nodes = True,
                        combine_close_skeleton_nodes_threshold=700,
                                               verbose=False,
                                                remove_cycles_at_end = True,
                                                **kwargs):
    """
    Purpose: To create a clean skeleton from a mesh
    (used in the neuron preprocessing package)
    """
    branch = mesh
    clean_time = time.time()
    current_skeleton = skeletonize_connected_branch(branch,**kwargs)
    
    print("Checking connected components after skeletonize_connected_branch")
    check_skeleton_connected_component(current_skeleton)

    if not remove_cycles_at_end:
        current_skeleton = remove_cycles_from_skeleton(current_skeleton)
    



#                     sk_debug = True
#                     if sk_debug:
#                         import system_utils as su
#                         print("**Saving the skeletons**")
#                         su.compressed_pickle(branch,
#                                             "curr_branch_saved")
#                     if sk_debug:
#                         import system_utils as su
#                         print("**Saving the skeletons**")
#                         su.compressed_pickle(current_skeleton,
#                                             "current_skeleton")

    print(f"    Total time for skeletonizing branch: {time.time() - clean_time}")
    clean_time = time.time()
    
    print("Checking connected components after removing cycles")
    check_skeleton_connected_component(current_skeleton)
    

    
    if not curr_soma_to_piece_touching_vertices is None:

        current_skeleton, curr_limb_endpoints_must_keep = create_soma_extending_branches(
                        current_skeleton=current_skeleton, #current skeleton that was created
                        skeleton_mesh=branch, #mesh that was skeletonized
                        soma_to_piece_touching_vertices=curr_soma_to_piece_touching_vertices,#dictionary mapping a soma it is touching to the border vertices,
                        return_endpoints_must_keep=True,
                                                        )
    else:
        if verbose:
            print("Not Creating soma extending branches because curr_soma_to_piece_touching_vertices is None")
        curr_limb_endpoints_must_keep = None




    print(f"    Total time for Fixing Skeleton Soma Endpoint Extension : {time.time() - clean_time}")
    """  --------- END OF 9/17 Addition:  -------- """

    #                     sk_debug = True
    #                     if sk_debug:
    #                         import system_utils as su
    #                         print("**Saving the skeletons**")
    #                         su.compressed_pickle(current_skeleton,
    #                                             "current_skeleton_after_addition")



        # --------  Doing the cleaning ------- #
    clean_time = time.time()
    print(f"filter_end_node_length = {filter_end_node_length}")

    """ 9/16 Edit: Now send the border vertices and don't want to clean anyy end nodes that are within certain distance of border"""

    #soma_border_vertices = total_border_vertices,
    #skeleton_mesh=branch,
    
    #gathering the endpoints to send to skeleton cleaning
    if not curr_limb_endpoints_must_keep is None:
        coordinates_to_keep = np.vstack(list(curr_limb_endpoints_must_keep.values())).reshape(-1,3)
    else:
        coordinates_to_keep = None
    
    check_skeleton_connected_component(current_skeleton)
    new_cleaned_skeleton = clean_skeleton(current_skeleton,
                            distance_func=skeletal_distance,
                      min_distance_to_junction=filter_end_node_length, #this used to be a tuple i think when moved the parameter up to function defintion
                      return_skeleton=True,
#                         soma_border_vertices = total_border_vertices,
#                         skeleton_mesh=branch,
                        endpoints_must_keep = coordinates_to_keep,
                      print_flag=False)

#                     sk_debug = True
#                     if sk_debug:
#                         import system_utils as su
#                         print("**Saving the skeletons**")
#                         su.compressed_pickle(new_cleaned_skeleton,
#                                             "new_cleaned_skeleton")
    
    print("Checking connected components after clean_skeleton")
    try:
        check_skeleton_connected_component(new_cleaned_skeleton)
    except:
        print("No connected skeleton after cleaning so just going with older skeleton")
        new_cleaned_skeleton = current_skeleton
    
    #--- 1) Cleaning each limb through distance and decomposition, checking that all cleaned branches are connected components and then visualizing
    distance_cleaned_skeleton = new_cleaned_skeleton

    if perform_cleaning_checks:
        #make sure still connected componet
        distance_cleaned_skeleton_components = nx.number_connected_components(convert_skeleton_to_graph(distance_cleaned_skeleton))
        if distance_cleaned_skeleton_components > 1:
            raise Exception(f"distance_cleaned_skeleton {j} was not a single component: it was actually {distance_cleaned_skeleton_components} components")

        print(f"after DISTANCE cleaning limb size of skeleton = {distance_cleaned_skeleton.shape}")

    cleaned_branch = clean_skeleton_with_decompose(distance_cleaned_skeleton)

    if perform_cleaning_checks:
        cleaned_branch_components = nx.number_connected_components(convert_skeleton_to_graph(cleaned_branch))
        if cleaned_branch_components > 1:
            raise Exception(f"BEFORE COMBINE: cleaned_branch {j} was not a single component: it was actually {cleaned_branch_components} components")



    if combine_close_skeleton_nodes:
        print(f"********COMBINING CLOSE SKELETON NODES WITHIN {combine_close_skeleton_nodes_threshold} DISTANCE**********")
        cleaned_branch = combine_close_branch_points(cleaned_branch,
                                                            combine_threshold = combine_close_skeleton_nodes_threshold,
                                                            print_flag=True) 

        
    
    if remove_cycles_at_end:
        cleaned_branch = remove_cycles_from_skeleton(cleaned_branch)
        
    cleaned_branch = clean_skeleton_with_decompose(cleaned_branch)


    if perform_cleaning_checks:
        n_components = nx.number_connected_components(convert_skeleton_to_graph(cleaned_branch)) 
        if n_components > 1:
            raise Exception(f"After combine: Original limb was not a single component: it was actually {n_components} components")
            
        divided_branches = sk.decompose_skeleton_to_branches(cleaned_branch)
        
        #check that when we downsample it is not one component:
        curr_branch_meshes_downsampled = [sk.resize_skeleton_branch(b,n_segments=1) for b in divided_branches]
        downsampled_skeleton = sk.stack_skeletons(curr_branch_meshes_downsampled)
        curr_sk_graph_debug = sk.convert_skeleton_to_graph(downsampled_skeleton)


        con_comp = list(nx.connected_components(curr_sk_graph_debug))
        if len(con_comp) > 1:
            raise Exception(f"There were more than 1 component when downsizing: {[len(k) for k in con_comp]}")

    return cleaned_branch,curr_limb_endpoints_must_keep

def check_skeleton_connected_component(skeleton):
    sk_graph = convert_skeleton_to_graph(skeleton)
    n_comp = nx.number_connected_components(sk_graph)
    if n_comp != 1:
        raise Exception(f"There were {n_comp} number of components detected in the skeleton")

def skeleton_connected_components(skeleton):
    total_limb_sk_graph = sk.convert_skeleton_to_graph(skeleton)
    conn_comp_graph = list(nx.connected_components(total_limb_sk_graph))
    conn_comp_sk = [sk.convert_graph_to_skeleton(total_limb_sk_graph.subgraph(list(k))) for k in conn_comp_graph]
    return conn_comp_sk
        
def remove_cycles_from_skeleton(skeleton,
    max_cycle_distance = 5000,
    verbose = False,
    check_cycles_at_end=True,
    return_original_if_error=False,
    error_on_more_than_two_paths_between_high_degree_nodes=False):
    
    """
    Purpose: To remove small cycles from a skeleton
    

    Pseudocode: How to resolve a cycle
    A) Convert the skeleton into a graph
    B) Find all cycles in the graph

    For each cycle
    -------------
    1) Get the length of the cycle 
    --> if length if too big then skip
    2) If only 1 high degree node, then just delete the other non high degree nodes
    3) Else, there should only be 2 high degree nodes in the vertices of the cycle
    --> if more or less then skip
    3) Get the 2 paths between the high degree nodes
    4) Delete nodes on the path for the longer distance one
    ------------

    C) convert the graph back into a skeleton


    
    
    Ex: 
    remove_cycles_from_skeleton(skeleton = significant_poisson_skeleton)
    
    
    """
    
    try:

        #A) Convert the skeleton into a graph
        skeleton_graph = convert_skeleton_to_graph(skeleton)
        #B) Find all cycles in the graph
        cycles_list = xu.find_all_cycles(skeleton_graph)

        number_skipped = 0
        for j,cyc in enumerate(cycles_list):
            if verbose:
                print(f"\n ---- Working on cycle {j}: {cyc} ----")
            #1) Get the length of the cycle 
            #--> if length if too big then skip
            cyc = np.array(cyc)
            
            if len(np.setdiff1d(cyc,skeleton_graph.nodes()))>0:
                print(f"--- cycle {j} has nodes that don't exist anymore so skipping --")
                continue

            sk_dist_of_cycle = xu.find_skeletal_distance_along_graph_node_path(skeleton_graph,cyc)

            if max_cycle_distance < sk_dist_of_cycle:
                if verbose:
                    print(f"Skipping cycle {j} because total distance ({sk_dist_of_cycle}) is larger than max_cycle_distance ({max_cycle_distance}): {cyc} ")
                number_skipped += 1
                continue


            #Find the degrees of all of the nodes
            node_degrees = np.array([xu.get_node_degree(skeleton_graph,c) for c in cyc])
            print(f"node_degrees = {node_degrees}")

            #2) If only 1 high degree node, then just delete the other non high degree nodes
            if np.sum(node_degrees>2) == 1:
                if verbose:
                    print(f"Deleting non-high degree nodes in cycle {j}: {cyc} becuase there was only one high degree node: {node_degrees}")
                nodes_to_delete = cyc[np.where(node_degrees<=2)[0]]

                skeleton_graph.remove_nodes_from(nodes_to_delete)
                continue


            #3) Else, there should only be 2 high degree nodes in the vertices of the cycle
            #--> if more or less then skip

            if np.sum(node_degrees>2) > 2:
                if verbose:
                    print(f"Skipping cycle {j} because had {np.sum(node_degrees>2)} number of high degree nodes: {node_degrees} ")
                number_skipped += 1
                continue

            high_degree_nodes = cyc[np.where(node_degrees>2)[0]]
            
            if len(high_degree_nodes) == 0:
                print("No higher degree (above 2) nodes detected")
                continue
            
            
            cycle_graph = skeleton_graph.subgraph(cyc)

            #3) Get the 2 paths between the high degree nodes
            both_paths = list(nx.all_simple_paths(cycle_graph,high_degree_nodes[0],high_degree_nodes[1],len(cycle_graph)))

            if len(both_paths) != 2:
                if error_on_more_than_two_paths_between_high_degree_nodes:
                    su.compressed_pickle(skeleton,"skeleton")
                    raise Exception(f"Did not come up with only 2 paths between high degree nodes: both_paths = {both_paths} ")
                else:
                    print(f"Did not come up with only 2 paths between high degree nodes: both_paths = {both_paths} ")

            path_lengths = [xu.find_skeletal_distance_along_graph_node_path(skeleton_graph,g) for g in both_paths]


            #4) Delete nodes on the path for the longer distance one
            longest_path_idx = np.argmax(path_lengths)
            longest_path = both_paths[longest_path_idx]
            if len(longest_path) <= 2:
                raise Exception(f"Longest path for deletion was only of size 2 or less: both_paths = {both_paths}, longest_path = {longest_path}")

            if verbose:
                print(f"For cycle {j} deleting the following path because longest distance {path_lengths[longest_path_idx]}: {longest_path[1:-1]}")

            skeleton_graph.remove_nodes_from(longest_path[1:-1])


        #C) check that all cycles removed except for those ones
        if check_cycles_at_end:
            cycles_at_end = xu.find_all_cycles(skeleton_graph)
            if number_skipped != len(cycles_at_end):
                print(f"The number of cycles skipped ({number_skipped}) does not equal the number of cycles at the end ({len(cycles_at_end)})")
        #C) convert the graph back into a skeleton
        skeleton_removed_cycles = convert_graph_to_skeleton(skeleton_graph)
        
        if len(skeleton_removed_cycles) == 0:
            #su.compressed_pickle(skeleton,"remove_cycles_skeleton")
            #raise Exception("Removing the cycles made the skeleton of 0 size so returning old skeleton")
            print("Removing the cycles made the skeleton of 0 size so returning old skeleton")
            return skeleton
        
        return skeleton_removed_cycles
    except:
        if return_original_if_error:
            return skeleton
        else:
            su.compressed_pickle(skeleton,"remove_cycles_skeleton")
            raise Exception("Something went wrong in remove_cycles_from_skeleton (12/2 found because had disconnected skeleton)")



import itertools
def skeleton_list_connectivity(skeletons,
    print_flag = False):
    """
    Will find the edge list for the connectivity of 
    branches in a list of skeleton branches
    
    
    """
    
    sk_endpoints = np.array([find_branch_endpoints(k) for k in skeletons]).reshape(-1,3)
    unique_endpoints,indices,counts= np.unique(sk_endpoints,return_inverse=True,return_counts=True,axis=0)
    
    total_edge_list = []
    repeated_indices = np.where(counts>1)[0]
    for ri in repeated_indices:
        connect_branches = np.where(indices == ri)[0]
        connect_branches_fixed = np.floor(connect_branches/2)
        total_edge_list += list(itertools.combinations(connect_branches_fixed,2))
    total_edge_list = np.array(total_edge_list).astype("int")
    
    return total_edge_list

def skeleton_list_connectivity_slow(
    skeletons,
    print_flag = False
    ):

    """
    Purpose: To determine which skeletons
    branches are connected to which and to
    record an edge list

    Pseudocode:
    For all branches i:
        a. get the endpoints
        For all branches j:
            a. get the endpoints
            b. compare the endpoints to the first
            c. if matching then add an edge


    """
    skeleton_connectivity_edge_list = []
    
    for j,sk_j in enumerate(skeletons):
        
        sk_j_ends = sk.find_branch_endpoints(sk_j)
        for i,sk_i in enumerate(skeletons):
            if i<=j:
                continue
            sk_i_ends = sk.find_branch_endpoints(sk_i)

            stacked_endpoints = np.vstack([sk_j_ends,sk_i_ends])
            endpoints_match = nu.get_matching_vertices(stacked_endpoints)

            if len(endpoints_match)>0:
                skeleton_connectivity_edge_list.append((j,i))

    return skeleton_connectivity_edge_list
            

    
def move_point_to_nearest_branch_end_point_within_threshold(
        skeleton,
        coordinate,
        distance_to_move_point_threshold = 1000,
        return_coordinate=True,
        return_change_status=True,
        verbose=False,
        consider_high_degree_nodes=True,
        possible_node_coordinates=None,
        excluded_node_coordinates=None
        ):
    """
    Purpose: To pick a branch or endpoint node that
    is within a certain a certain distance of the original 
    node (if none in certain distance then return original)
    
    Arguments: 
    possible_node_coordinates: this allows you to specify nodes that you want to select
    
    """
    
    curr_skeleton_MAP = skeleton
    MAP_stitch_point = coordinate

    #get a network of the skeleton
    curr_skeleton_MAP_graph = sk.convert_skeleton_to_graph(curr_skeleton_MAP)
    #get the node where the stitching will take place
    node_for_stitch = xu.get_nodes_with_attributes_dict(curr_skeleton_MAP_graph,dict(coordinates=MAP_stitch_point))[0]
    #get all of the endnodes or high degree nodes
    
    # ----- 11/13 addition: Use the node locations sent or just use the high degree or end nodes from the graph
    if possible_node_coordinates is None:
        curr_MAP_end_nodes = xu.get_nodes_of_degree_k(curr_skeleton_MAP_graph,1)
        if consider_high_degree_nodes:
            curr_MAP_branch_nodes = xu.get_nodes_greater_or_equal_degree_k(curr_skeleton_MAP_graph,3)
        else:
            curr_MAP_branch_nodes = []
        possible_node_loc = np.array(curr_MAP_end_nodes + curr_MAP_branch_nodes)
    else:
        possible_node_loc = np.array([xu.get_graph_node_by_coordinate(curr_skeleton_MAP_graph,zz) for zz in possible_node_coordinates])
        
    #removing the high degree coordinates that should not be there
    if not (excluded_node_coordinates is None):
        possible_node_loc_to_exclude = np.array([xu.get_graph_node_by_coordinate(curr_skeleton_MAP_graph,zz,return_neg_one_if_not_find=True) for zz in excluded_node_coordinates])
        possible_node_loc = np.setdiff1d(possible_node_loc,possible_node_loc_to_exclude)
        

    #get the distance along the skeleton from the stitch point to all of the end or branch nodes
    curr_shortest_path,end_node_1,end_node_2 = xu.shortest_path_between_two_sets_of_nodes(curr_skeleton_MAP_graph,
                                                                node_list_1=[node_for_stitch],
                                                                node_list_2=possible_node_loc)

#     if verbose:
#         print(f"curr_shortest_path = {curr_shortest_path}")

        
    changed_node = False
    if len(curr_shortest_path) == 1:
        if verbose:
             print(f"Current stitch point was a branch or endpoint")
        MAP_stitch_point_new = coordinate
    else:
        
        #get the length of the path
        shortest_path_length = nx.shortest_path_length(curr_skeleton_MAP_graph,
                           end_node_1,
                           end_node_2,
                           weight="weight")

        if verbose:
            print(f"Current stitch point was not a branch or endpoint, shortest_path_length to one = {shortest_path_length}")
            
        if shortest_path_length < distance_to_move_point_threshold:
            if verbose:
                print(f"Changing the stitch point becasue the distance to end or branch node was {shortest_path_length}"
                     f"\nNew stitch point has degree {xu.get_node_degree(curr_skeleton_MAP_graph,end_node_2)}")
            
            MAP_stitch_point_new = end_node_2
            changed_node=True
        else:
            MAP_stitch_point_new = coordinate
    
    if return_coordinate and changed_node:
        MAP_stitch_point_new = xu.get_node_attributes(curr_skeleton_MAP_graph,node_list=MAP_stitch_point_new)[0]
    
    return_value = [MAP_stitch_point_new]
    if return_change_status:
        return_value.append(changed_node)
        
    return return_value


def cut_skeleton_at_coordinate(skeleton,
                        cut_coordinate,
                              tolerance = 0.001, #if have to find cut point that is not already coordinate
                               verbose=False
                        ):
    """
    Purpose: 
    To cut a skeleton into 2 pieces at a certain cut coordinate
    
    Application: Used when the MP skeleton pieces 
    connect to the middle of a MAP branch and have to split it
    
    Example:
    ex_sk = neuron_obj[1][0].skeleton
    cut_coordinate = np.array([ 560252., 1121040.,  842599.])

    new_sk_cuts = sk.cut_skeleton_at_coordinate(skeleton=ex_sk,
                              cut_coordinate=cut_coordinate)

    nviz.plot_objects(skeletons=new_sk_cuts,
                     skeletons_colors="random",
                     scatters=[cut_coordinate])
    
    
    """
    curr_MAP_sk_new = []
    #b) Convert the skeleton into a graph
    curr_MAP_sk_graph = sk.convert_skeleton_to_graph(skeleton)
    #c) Find the node of the MAP stitch point (where need to do the breaking)
    
    
    MP_stitch_node = xu.get_nodes_with_attributes_dict(curr_MAP_sk_graph,dict(coordinates=cut_coordinate))
    
    # --------- New Addition that accounts for if cut point is not an actual node but can interpolate between nodes -------#
    if len(MP_stitch_node) == 0: #then have to add the new stitch point
        current_point = cut_coordinate
        winning_edge = None
        
        for node_a,node_b in curr_MAP_sk_graph.edges:
            node_a_coord,node_b_coord = xu.get_node_attributes(curr_MAP_sk_graph,node_list=[node_a,node_b])

            AB = np.linalg.norm(node_a_coord-node_b_coord)
            AC = np.linalg.norm(node_a_coord-cut_coordinate)
            CB = np.linalg.norm(cut_coordinate-node_b_coord)

            if np.abs(AB - AC - CB) < tolerance:
                winning_edge = [node_a,node_b]
                winning_edge_coord = [node_a_coord,node_b_coord]
                if verbose:
                    print(f"Found winning edge: {winning_edge}")
                break
        if winning_edge is None:
            raise Exception("Cut point was neither a matching node nor a coordinate between 2 nodes ")
            
        new_node_name = np.max(curr_MAP_sk_graph.nodes()) + 1

        curr_MAP_sk_graph.add_nodes_from([(new_node_name,{"coordinates":cut_coordinate})])
        curr_MAP_sk_graph.add_weighted_edges_from([(winning_edge[k],
                                            new_node_name,
                                            np.linalg.norm(winning_edge_coord[k] - cut_coordinate)
                                           ) for k in range(0,2)])
        curr_MAP_sk_graph.remove_edge(winning_edge[0],winning_edge[1])

        MP_stitch_node = new_node_name
    else:
        MP_stitch_node = MP_stitch_node[0]
        
    # --------- End of Addition -------#
    
    #d) Find the degree one nodes
    curr_end_nodes_for_break = xu.get_nodes_of_degree_k(curr_MAP_sk_graph,1)

    #e) For each degree one node:
    for e_n in curr_end_nodes_for_break:
        #- Find shortest path from stitch node to end node
        stitch_to_end_path = nx.shortest_path(curr_MAP_sk_graph,MP_stitch_node,e_n)
        #- get a subgraph from that path
        stitch_to_end_path_graph = curr_MAP_sk_graph.subgraph(stitch_to_end_path)
        #- convert graph to a skeleton and save as new skeletons
        new_sk = sk.convert_graph_to_skeleton(stitch_to_end_path_graph)
        curr_MAP_sk_new.append(new_sk)

    return curr_MAP_sk_new



def smooth_skeleton_branch(skeleton,
                    neighborhood=2,
                    iterations=100,
                    coordinates_to_keep=None,
                    keep_endpoints=True,
    ):
    
    """
    Purpose: To smooth skeleton of branch while keeping the same endpoints

    Pseudocode:
    1) get the endpoint coordinates of the skeleton
    2) turn the skeleton into nodes and edges
    - if number of nodes is less than 3 then return
    3) Find the indexes that are the end coordinates
    4) Send the coordinates and edges off to get smoothed
    5) Replace the end coordinate smooth vertices with original
    6) Convert nodes and edges back to a skeleton
    
    Ex: 
    orig_smoothed_sk = smooth_skeleton(neuron_obj[limb_idx][branch_idx].skeleton,
                                  neighborhood=5)

    """
    
    



    #2) turn the skeleton into nodes and edges
    nodes,edges = sk.convert_skeleton_to_nodes_edges(skeleton)

    #- if number of nodes is less than 3 then return
    if len(nodes) < 3:
        print("Only 2 skeleton nodes so cannot do smoothing")
        return skeleton

    if not coordinates_to_keep is None:
        coordinates_to_keep = np.array(coordinates_to_keep).reshape(-1,3)
        
    if keep_endpoints:
        #1) get the endpoint coordinates of the skeleton
        curr_endpoints = sk.find_branch_endpoints(skeleton).reshape(-1,3)
        if not coordinates_to_keep is None:
            coordinates_to_keep = np.vstack([curr_endpoints,coordinates_to_keep]).reshape(-1,3)
        else:
            coordinates_to_keep = curr_endpoints
            

        
    #3) Find the indexes that are the end coordinates
    coordinates_to_keep_idx = [nu.matching_rows(nodes,k)[0] for k in coordinates_to_keep]
        

    #4) Send the coordinates and edges off to get smoothed
    
    smoothed_nodes = m_sk.smooth_graph(nodes,edges,neighborhood=neighborhood,iterations=iterations)

    #5) Replace the end coordinate smooth vertices with original
    for endpt_idx,endpt in zip(coordinates_to_keep_idx,coordinates_to_keep):
        smoothed_nodes[endpt_idx] = endpt

    #6) Convert nodes and edges back to a skeleton
    final_sk_smooth = sk.convert_nodes_edges_to_skeleton(smoothed_nodes,edges)

    return final_sk_smooth



def add_and_smooth_segment_to_branch(skeleton,
                              skeleton_stitch_point=None,
                              new_stitch_point=None,
                              new_seg=None,
                              resize_mult= 0.2,
                               n_resized_cutoff=3,
                               smooth_branch_at_end=True,
                                n_resized_cutoff_to_smooth=None,
                                     smooth_width = 100,
                                max_stitch_distance_for_smoothing=300,
                                verbose=False,
                               **kwargs,
                              ):
    """
    Purpose: To add on a new skeletal segment to a branch that will
    prevent the characteristic hooking when stitching a new point
    
    Pseudocode: 
    1) Get the distance of the stitch point = A
    2) Resize the skeleton to B*A (where B < 1)
    3) Find all nodes that are CA away from the MP stitch point
    4) Delete those nodes (except the last one and set that as the new stitch point)
    5) Make the new stitch

    Ex: When using a new segment
    orig_sk_func_smoothed = add_and_smooth_segment_to_branch(orig_sk,
                           new_seg = np.array([stitch_point_MAP,stitch_point_MP]).reshape(-1,2,3))
    """
    # 12/21 Addition: If the point you are trying to stitch to is already there then just return the skeleton
    if not new_stitch_point is None:
        sk_graph_at_beginning = sk.convert_skeleton_to_graph(skeleton)
        match_nodes_to_new_stitch_point = xu.get_nodes_with_attributes_dict(sk_graph_at_beginning,dict(coordinates=new_stitch_point))
        if len(match_nodes_to_new_stitch_point)>0:
            if verbose:
                print("New stitch point was already on the skeleton so don't need to add it")
            return skeleton
    
    
    if len(skeleton) == 0:
        raise Exception("The skeleton passed to the smoothing function was empty")
    
    orig_sk = skeleton
    orig_sk_segment_width = np.mean(sk.calculate_skeleton_segment_distances(orig_sk,cumsum=False))
    
    if skeleton_stitch_point is None or new_stitch_point is None:
        new_seg_reshaped = new_seg.reshape(-1,3)
        #try to get the stitch points from the new seg
        if ((len(sk.find_branch_skeleton_with_specific_coordinate([orig_sk],new_seg_reshaped[0])) > 0) and
            (len(sk.find_branch_skeleton_with_specific_coordinate([orig_sk],new_seg_reshaped[1])) == 0)):
            stitch_point_MP = new_seg_reshaped[0]
            stitch_point_MAP = new_seg_reshaped[1]
        elif ((len(sk.find_branch_skeleton_with_specific_coordinate([orig_sk],new_seg_reshaped[1])) > 0) and
            (len(sk.find_branch_skeleton_with_specific_coordinate([orig_sk],new_seg_reshaped[0])) == 0)):
            stitch_point_MP = new_seg_reshaped[1]
            stitch_point_MAP = new_seg_reshaped[0]
        else:
            raise Exception("Could not find a stitch point that was on the existing skeleton and one that was not")
    else:
        stitch_point_MAP = new_stitch_point
        stitch_point_MP = skeleton_stitch_point

    #1) Get the distance of the stitch point = A
    stitch_distance = np.linalg.norm(stitch_point_MAP-stitch_point_MP)
    if stitch_distance > max_stitch_distance_for_smoothing:
        if verbose:
            print(f"Using max stitch distance ({max_stitch_distance_for_smoothing}) for smoothing because stitch_distance greater ({stitch_distance}) ")
        stitch_distance = max_stitch_distance_for_smoothing
    
        

    #2) Resize the skeleton to B*A (where B < 1)
    
    orig_sk_resized = sk.resize_skeleton_branch(orig_sk,segment_width = resize_mult*stitch_distance)

    #3) Find all nodes that are CA away from the MP stitch point
    orig_resized_graph = sk.convert_skeleton_to_graph(orig_sk_resized)
    MP_stitch_node = xu.get_nodes_with_attributes_dict(orig_resized_graph,dict(coordinates=stitch_point_MP))

    if len(MP_stitch_node) == 1:
        MP_stitch_node = MP_stitch_node[0]
    else:
        raise Exception(f"MP_stitch_node not len = 1: len = {len(MP_stitch_node)}")

    nodes_within_dist = gu.dict_to_array(xu.find_nodes_within_certain_distance_of_target_node(orig_resized_graph,
                                                         target_node=MP_stitch_node,
                                                        cutoff_distance=n_resized_cutoff*stitch_distance,
                                                                            return_dict=True))
    farthest_node_idx = np.argmax(nodes_within_dist[:,1])
    farthest_node = nodes_within_dist[:,0][farthest_node_idx]
    new_stitch_point_MP = xu.get_node_attributes(orig_resized_graph,node_list=farthest_node)[0]

    #need to consider if the farthest node is an endpoint
    farthest_node_degree = xu.get_node_degree(orig_resized_graph,farthest_node)

    keep_branch = None
    if farthest_node_degree > 1:#then don't have to worry about having reached branch end
        cut_branches = sk.cut_skeleton_at_coordinate(orig_sk,cut_coordinate=new_stitch_point_MP)
        #find which branch had the original cut point
        branch_to_keep_idx = 1 - sk.find_branch_skeleton_with_specific_coordinate(cut_branches,stitch_point_MP)[0]
        keep_branch = cut_branches[branch_to_keep_idx]
    else:
        keep_branch = np.array([])

    # nodes_to_delete = np.delete(nodes_within_dist[:,0],farthest_node_idx)
    new_seg = np.array([[new_stitch_point_MP],[stitch_point_MAP]]).reshape(-1,2,3)
    final_sk = sk.stack_skeletons([new_seg,keep_branch]) 
    final_sk=sk.convert_graph_to_skeleton(sk.convert_skeleton_to_graph(final_sk))
    
    

    if smooth_branch_at_end:
        #resize the skeleton
        coordinates_to_keep=None
        #skeleton_reshaped = sk.resize_skeleton_branch(final_sk,segment_width=smooth_width)
        skeleton_reshaped = final_sk
        
        if len(keep_branch)>0:
            if n_resized_cutoff_to_smooth is None:
                n_resized_cutoff_to_smooth = n_resized_cutoff + 2
  
            """
            Pseudocode for smoothing only a certain portion:
            1) Convert the branch to a graph
            2) Find the node with the MP stitch point
            3) Find all nodes within n_resized_cutoff_to_smooth*stitch_distance
            4) Get all nodes not in that list
            5) Get all the coordinates of those nodes
            """
            #1) Convert the branch to a graph
            sk_gr = convert_skeleton_to_graph(skeleton_reshaped)
            #2) Find the node with the MP stitch point
            MP_stitch_node = xu.get_graph_node_by_coordinate(sk_gr,new_stitch_point_MP)
            MAP_stitch_node = xu.get_graph_node_by_coordinate(sk_gr,stitch_point_MAP)

            #3) Find all nodes within n_resized_cutoff_to_smooth*stitch_distance
            distance_to_smooth = n_resized_cutoff_to_smooth*stitch_distance
            nodes_to_smooth_pre = xu.find_nodes_within_certain_distance_of_target_node(sk_gr,target_node=MP_stitch_node,
                                                                cutoff_distance=distance_to_smooth)
            
            #need to add in nodes to endpoint in case the distance_to_smooth doesn't exend there
            nodes_to_MAP = nx.shortest_path(sk_gr,MP_stitch_node,MAP_stitch_node)

            nodes_to_smooth = np.unique(list(nodes_to_smooth_pre) + list(nodes_to_MAP))
            #print(f"nodes_to_smooth = {nodes_to_smooth}")

            #4) Get all nodes not in that list
            nodes_to_not_smooth = np.setdiff1d(list(sk_gr.nodes()),list(nodes_to_smooth))
            #print(f"nodes_to_smooth = {nodes_to_not_smooth}")

            #5) Get all the coordinates of those nodes
            coordinates_to_keep = xu.get_node_attributes(sk_gr,node_list=nodes_to_not_smooth)
            
        
        final_sk = smooth_skeleton_branch(skeleton_reshaped,coordinates_to_keep=coordinates_to_keep,**kwargs)
        
    #need to resize the final_sk
    if len(final_sk) == 0:
        """
        Pseudocode: 
        3) Create a skeleton segment from the skeleton_stitch_point to the new point
        4) Stack the skeletons
        5) Return 
        
        """
        print("The Skeleton at the end of smoothing was empty so just going to stitch the new point to skeleton without stitching")
        
        
        new_sk_seg = np.array([skeleton_stitch_point,new_stitch_point])
        final_sk = sk.stack_skeletons([skeleton,new_sk_seg])
        return final_sk
        
    else: 
        final_sk = sk.resize_skeleton_branch(final_sk,segment_width=orig_sk_segment_width)
        return final_sk

def number_connected_components(skeleton):
    """
    Will find the number of connected components in a whole skeleton
    
    """
    return nx.number_connected_components(convert_skeleton_to_graph(skeleton))

def number_connected_components_branches(skeleton_branches):
    """
    Will find the number of connected components in a list of skeleton branches
    
    """
    return nx.number_connected_components(convert_skeleton_to_graph(stack_skeletons(skeleton_branches)))

    
# ---------------- 11/26 Extra Utils for the Error Detection------------------
def endpoint_connectivity(endpoints_1,endpoints_2,
                         exceptions_flag=True,
                          return_coordinate=False,
                         print_flag=False):
    """
    Pupose: To determine where the endpoints of two branches are connected
    
    Example: 
    end_1 = np.array([[759621., 936916., 872083.],
       [790891., 913598., 806043.]])
    end_2 = np.array([[790891., 913598., 806043.],
       [794967., 913603., 797825.]])
       
    endpoint_connectivity(end_1,end_2)
    >> {0: 1, 1: 0}
    """
    connections_dict = dict()
    
    stacked_endpoints = np.vstack([endpoints_1,endpoints_2])
    endpoints_match = nu.get_matching_vertices(stacked_endpoints)
    
    if len(endpoints_match) == 0:
        print_string = f"No endpoints matching: {endpoints_match}"
        if exceptions_flag:
            raise Exception(print_string)
        else:
            print(print_string)
        return connections_dict
    
    if len(endpoints_match) > 1:
        print_string = f"Multiple endpoints matching: {endpoints_match}"
        if exceptions_flag:
            raise Exception(print_string)
        else:
            print(print_string)
    
    
    #look at the first connection
    first_match = endpoints_match[0]
    first_endpoint_match = first_match[0]
    
    if print_flag:
        print(f"first_match = {first_match}")
        print(f"first_endpoint_match = {endpoints_1[first_endpoint_match]}")
    
    if return_coordinate:
        return endpoints_1[first_endpoint_match]
    
    if 0 != first_endpoint_match and 1 != first_endpoint_match:
        raise Exception(f"Non 0,1 matching node in first endpoint: {first_endpoint_match}")
    else:
        connections_dict.update({0:first_endpoint_match})
        
    second_endpoint_match = first_match[-1]
    
    if 2 != second_endpoint_match and 3 != second_endpoint_match:
        raise Exception(f"Non 2,3 matching node in second endpoint: {second_endpoint_match}")
    else:
        connections_dict.update({1:second_endpoint_match-2})
    
    return connections_dict

def shared_endpoint(skeleton_1,skeleton_2):
    """
    Will return the endpoint that joins two branches
    """
    end_1 = find_branch_endpoints(skeleton_1)
    end_2 = find_branch_endpoints(skeleton_2)
    node_connectivity = endpoint_connectivity(end_1,end_2,print_flag=False,return_coordinate=True)
    return node_connectivity
    

def flip_skeleton(current_skeleton):
    """
    Will flip the absolute order of a skeleton
    """
    new_sk = np.flip(current_skeleton,0)
    return np.flip(new_sk,1)


def order_skeleton(skeleton,start_endpoint_coordinate=None,verbose=False,return_indexes=False):
    """
    Purpose: to get the skeleton in ordered vertices
    1) Convert to graph
    2) Find the endpoint nodes
    3) Find the shortest path between endpoints
    4) Get the coordinates of all of the nodes
    5) Create the skeleton by indexing into the coordinates by the order of the path

    """
    #1) Convert to graph
    sk_graph = convert_skeleton_to_graph(skeleton)
    #2) Find the endpoint nodes
    sk_graph_endpt_nodes = np.array(xu.get_nodes_of_degree_k(sk_graph,1))
    if verbose:
        print(f"sk_graph_endpt_nodes = {sk_graph_endpt_nodes}")
    
    #2b) If a starting endpoint coordinate was picked then use that
    if not start_endpoint_coordinate is None:
        if verbose:
            print(f"Using start_endpoint_coordinate = {start_endpoint_coordinate}")
        curr_st_node = xu.get_graph_node_by_coordinate(sk_graph,start_endpoint_coordinate)
        start_node_idx = np.where(sk_graph_endpt_nodes==curr_st_node)[0]
        if len(start_node_idx) == 0:
            #raise Exception(f"The start endpoint was not an end node: {start_endpoint_coordinate}")
            print(f"Warning: start endpoint was not an end node: {start_endpoint_coordinate} but not erroring")
            first_start_node = curr_st_node
        else:
            if verbose:
                print(f"start_node_idx = {start_node_idx}")
            start_node_idx = start_node_idx[0]
            first_start_node = sk_graph_endpt_nodes[start_node_idx]
    else:
        start_node_idx = 0
        first_start_node = sk_graph_endpt_nodes[start_node_idx]
        
    
    leftover_start_nodes = sk_graph_endpt_nodes[sk_graph_endpt_nodes!=first_start_node]
    if len(leftover_start_nodes) == 1:
        other_end_node = leftover_start_nodes[0]
    else:
        #find the 
        shortest_path,orig_st,other_end_node = xu.shortest_path_between_two_sets_of_nodes(sk_graph,[first_start_node],list(leftover_start_nodes))

    #3) Find the shortest path between endpoints
    shortest_path = np.array(nx.shortest_path(sk_graph,first_start_node,other_end_node)).astype("int")
    
    if verbose:
        print(f"shortest_path = {shortest_path}")


    #4) Get the coordinates of all of the nodes
    node_coordinates = xu.get_node_attributes(sk_graph,node_list = shortest_path)

    #5) Create the skeleton by indexing into the coordinates by the order of the path
    
    ordered_skeleton = np.stack((node_coordinates[:-1],node_coordinates[1:]),axis=1)
    
    if return_indexes:
        new_edges = np.sort(np.stack((shortest_path[:-1],shortest_path[1:]),axis=1),axis=1)
        original_edges = np.sort(sk_graph.edges_ordered(),axis=1)
        
        orig_indexes = [nu.matching_rows_old(original_edges,ne)[0] for ne in new_edges]
        return ordered_skeleton,orig_indexes
    
    return ordered_skeleton


def align_skeletons_at_connectivity(sk_1,sk_2):
    """
    To align 2 skeletons where both starts with the endpoint
    that they share
    
    Ex: 
    
    
    """
    common_coordinate = shared_endpoint(sk_1,sk_2)
    sk_1 = order_skeleton(sk_1,start_endpoint_coordinate=common_coordinate)
    sk_2 = order_skeleton(sk_2,start_endpoint_coordinate=common_coordinate)
    return sk_1,sk_2


def restrict_skeleton_from_start(skeleton,
                     cutoff_distance,
                    subtract_cutoff = False,
                    return_indexes = True,
                    return_success = True,
                    tolerance = 10):
    """
    To restrict a skeleton to a certain cutoff distance from the start
    which keeps that distance or subtracts it (and does not resize or reorder the skeleton but keeps the existing segment lengths)
    
    Ex: 
    restrict_skeleton_from_start(skeleton = base_skeleton_ordered,
    cutoff_distance = offset)

    """
    #handling if the cutof is 0
    if cutoff_distance <= 0:
            return_values = [skeleton]
            if return_indexes:
                return_values.append(np.arange(0,len(skeleton)))
            if return_success:
                return_values.append(True)
            return return_values

    distance_of_segs = calculate_skeleton_segment_distances(skeleton,cumsum=False)
    offset_idxs = np.where(np.cumsum(distance_of_segs)>=(cutoff_distance-tolerance))[0]
    if len(offset_idxs)>0:
        offset_idxs = offset_idxs[1:]

    subtract_idxs = np.setdiff1d(np.arange(len(distance_of_segs)),offset_idxs)
        
    subtract_sk = skeleton[subtract_idxs]
    subtract_sk_len = calculate_skeleton_distance(subtract_sk)
#     print(f"subtract_sk_len = {subtract_sk_len}")
#     print(f"(cutoff_distance-tolerance) = {(cutoff_distance-tolerance)}")
    success_subtraction = subtract_sk_len >= (cutoff_distance-tolerance)
#     print(f"success_subtraction = {success_subtraction}")

    #flip the indexes if want to keep the segment
    if not subtract_cutoff: 
        keep_indexes = np.setdiff1d(np.arange(len(distance_of_segs)),offset_idxs)
    else:
        keep_indexes = offset_idxs

    #restrict the skeleton
    return_sk = skeleton[keep_indexes]

    return_values = [return_sk]

    if return_indexes:
        return_values.append(keep_indexes)
    if return_success:
        return_values.append(success_subtraction)

    return return_values


from tqdm_utils import tqdm
from pykdtree.kdtree import KDTree

def matching_skeleton_branches_by_vertices(branches):
    
    decomposed_branches = branches
    kdtree_branches = [KDTree(k.reshape(-1,3)) for k in decomposed_branches]
    matching_edges_kdtree = []
    for i,d_br_1 in tqdm(enumerate(decomposed_branches)):
        for j,d_br_2 in enumerate(decomposed_branches):
            if i < j:
                dist, nearest = kdtree_branches[i].query(d_br_2.reshape(-1,3))
                if sum(dist) == 0:
                    matching_edges_kdtree.append([i,j])
                    
    return matching_edges_kdtree
                    
                    
def matching_skeleton_branches_by_endpoints(branches):
    matching_edges = []
    decomposed_branches = branches
    
    for i,d_br_1 in tqdm(enumerate(decomposed_branches)):
        for j,d_br_2 in enumerate(decomposed_branches):
            if i < j:
                c_t = time.time()
                br_1_end = sk.find_branch_endpoints(d_br_1)
                br_2_end = sk.find_branch_endpoints(d_br_2)
                #print(f"branch: {time.time() - c_t}")
                c_t = time.time()
                if sk.compare_endpoints(br_1_end,br_2_end):
                    matching_edges.append([i,j])
    return matching_edges
    

def check_correspondence_branches_have_2_endpoints(correspondence,
                                                  verbose=True,
                                                  raise_error= True):
    """
    Purpose: check that all branches have 2 endpoints
    """

    irregular_branches = []
    for piece_idx,piece_correspondence in correspondence.items():
        
        if "branch_skeleton" in piece_correspondence.keys():
            k = piece_idx
            v = piece_correspondence
            
            curr_sk = v["branch_skeleton"]
            curr_sk_endpoint_coord = sk.find_skeleton_endpoint_coordinates(curr_sk)
            if len(curr_sk_endpoint_coord) != 2:
                if verbose:
                    print(f"Branch {k} had {len(curr_sk_endpoint_coord)} endpoints")
                irregular_branches.append([piece_idx,k,len(curr_sk_endpoint_coord)])
        else:
            for k,v in piece_correspondence.items():
                curr_sk = v["branch_skeleton"]
                curr_sk_endpoint_coord = sk.find_skeleton_endpoint_coordinates(curr_sk)
                if len(curr_sk_endpoint_coord) != 2:
                    if verbose:
                        print(f"Piece {piece_idx}, Branch {k} had {len(curr_sk_endpoint_coord)} endpoints")
                    irregular_branches.append([piece_idx,k,len(curr_sk_endpoint_coord)])
    if raise_error and len(irregular_branches)>0:
        raise Exception(f"Found the following irregular branches: {irregular_branches}")
        
    return irregular_branches


# ---------------- 12/23 -------------------- #
def offset_skeletons_aligned_at_shared_endpoint(skeletons,
                                               offset=1000,
                                            comparison_distance=2000,
                                                min_comparison_distance=1000,
                                            verbose=True,
                                               ):

    """
    Pseudocode: 

    1) Get the shared endpoint of the branches
    2) Reorder the branches so both start with the endpoint and then resize

    For each edge skeleton (in order to get the final edge skeletons):
    3) Use the restrict skeelton function to subtract the offset
    - if not then add whole skeleton to final skeleton
    4) if it was a sucess, see if the distance is greater than comparison distance
    - if not then add current skeleton to final
    5) Use the subtract skeleton to only keep the comparison distance of skeleton
    6) Add to final skeleton

    offset_skeletons_aligned_at_shared_endpoint()
    
    Ex: 
    vis_branches_idx = [7,9]
    vis_branches = [curr_limb[k] for k in vis_branches_idx]
    vis_branches


    curr_skeletons = [k.skeleton for k in vis_branches]
    stripped_skeletons = sk.offset_skeletons_aligned_at_shared_endpoint(curr_skeletons)

    curr_colors = ["red","black"]
    nviz.plot_objects(meshes=[k.mesh for k in vis_branches],
                      meshes_colors=curr_colors,
                      skeletons=stripped_skeletons,
                      skeletons_colors=curr_colors,
                      scatters=[np.array([stripped_skeletons[0][-1][-1],stripped_skeletons[1][-1][-1]])],
                      scatter_size=1
                     )


    sk.parent_child_skeletal_angle(stripped_skeletons[1],stripped_skeletons[0])


    """
    edge_skeletons = skeletons
    seg_size = 100


    common_endpoint = sk.shared_endpoint(edge_skeletons[0],edge_skeletons[1])
    
    edge_skeletons_ordered = [sk.order_skeleton(sk.resize_skeleton_branch(e,seg_size),common_endpoint) for e in edge_skeletons]
    
    final_skeletons = []
    for e in edge_skeletons_ordered:
        
        # -------- Making sure that we don't take off too much so it's just a spec
        original_sk_length = sk.calculate_skeleton_distance(e)
        if original_sk_length < offset + min_comparison_distance:
            offset_adjusted = original_sk_length - min_comparison_distance
            if offset_adjusted < 0:
                offset_adjusted = 0
                
            #print(f" Had to Adjust Offset to {offset_adjusted}")
        else:
            offset_adjusted  = offset
            
        ret_sk,_,success = sk.restrict_skeleton_from_start(e,
                                                           cutoff_distance = offset_adjusted,
                                                           subtract_cutoff=True)
        if not success:
            final_skeletons.append(e)
        else:
            if sk.calculate_skeleton_distance(ret_sk) > comparison_distance:
                ret_sk,_,success = sk.restrict_skeleton_from_start(ret_sk,
                                                           cutoff_distance = comparison_distance,
                                                           subtract_cutoff=False)
            final_skeletons.append(ret_sk)
    return final_skeletons


def parent_child_skeletal_angle(parent_skeleton,child_skeleton):
    """
    To find the angle from continuation that the
    second skeleton deviates from the parent angle 
    
    angles are just computed from the vectors of the endpoints
    
    """
    up_sk = parent_skeleton
    d_sk = child_skeleton
    
    up_sk_flipped = sk.flip_skeleton(up_sk)

    up_vec = up_sk_flipped[-1][-1] - up_sk_flipped[0][0] 
    d_vec_child = d_sk[-1][-1] - d_sk[0][0]

    parent_child_angle = np.round(nu.angle_between_vectors(up_vec,d_vec_child),2)
    return parent_child_angle


def offset_skeletons_aligned_parent_child_skeletal_angle(skeleton_1,skeleton_2,
                                                        offset=1000,
                                                        comparison_distance=2000,
                                                        min_comparison_distance=1000):
    """
    Purpose: To determine the parent child skeletal angle
    of 2 skeletons while using the offset and comparison distance
    
    
    
    """
    
    edge_skeletons = [skeleton_1,skeleton_2]
    aligned_sk_parts = sk.offset_skeletons_aligned_at_shared_endpoint(edge_skeletons,
                                                                     offset=offset,
                                                        comparison_distance=comparison_distance,
                                                        min_comparison_distance=min_comparison_distance)


    curr_angle = sk.parent_child_skeletal_angle(aligned_sk_parts[0],aligned_sk_parts[1])
    return curr_angle


from tqdm_utils import tqdm
from pykdtree.kdtree import KDTree

def map_between_branches_lists(branches_1,branches_2,check_all_matched=True,
                              min_to_match = 2):
    """
    Purpose: 
    Will create a unique mapping of a branch
    in the first list to the best fitting branch in the second
    in terms of the most matching coordinates with a distance of 0
    
    min_to_match is the number of vertices that must match in order to
    be considered for the matching 
    Ex:
    cleaned_branches = sk.decompose_skeleton_to_branches(curr_limb_sk_cleaned)
    original_branches = [k.skeleton for k in curr_limb]
    map_between_branches_lists(original_branches,cleaned_branches)
    """
    original_branches = branches_1
    cleaned_branches = branches_2
    
    old_to_new_branch_mapping = []

    for o_br in tqdm(original_branches):
        o_br_kd = KDTree(o_br.reshape(-1,3))

        n_matches = [len(np.where(o_br_kd.query(c_br.reshape(-1,3))[0]==0)[0]) for c_br in cleaned_branches]
        
        max_matched_index = np.argmax(n_matches)
        max_matched_index_number = n_matches[max_matched_index]
        
        if max_matched_index_number >= min_to_match:
            old_to_new_branch_mapping.append(max_matched_index)
        else:
            old_to_new_branch_mapping.append(-1)
        
    old_to_new_branch_mapping = np.array(old_to_new_branch_mapping)
    if check_all_matched:
        if len(np.unique(old_to_new_branch_mapping[old_to_new_branch_mapping!=-1])) < len(branches_2):
            raise Exception("Not all of the new branches had at least one mapping")
    return old_to_new_branch_mapping
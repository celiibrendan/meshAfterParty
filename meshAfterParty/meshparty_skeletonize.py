from scipy import sparse, spatial
import numpy as np
import time
from meshparty import trimesh_vtk, utils, mesh_filters
import pandas as pd
from pykdtree.kdtree import KDTree
from tqdm_utils import tqdm
from tqdm import trange 
from meshparty.trimesh_io import Mesh
from meshparty.skeleton import Skeleton
from collections import defaultdict
from pykdtree.kdtree import KDTree as pyKDTree
import trimesh.ray
from trimesh.ray import ray_pyembree
import logging


# --------------------- Functions from skeleton.py ------------------- #
def compute_segments(sk):
        """Precompute segments between branches and end points"""
        segments = []
        segment_map = np.zeros(len(sk.vertices))-1

        path_queue = sk.end_points.tolist()
        bp_all = sk.branch_points
        bp_seen = []
        seg_ind = 0

        
        while len(path_queue)>0:
            ind = path_queue.pop()
            segment = [ind]
            ptr = sk.path_to_root(ind)
            if len(ptr)>1:
                for pind in ptr[1:]:
                    if pind in bp_all:
                        segment.append(pind)
                        segments.append(np.array(segment))
                        segment_map[segment] = seg_ind
                        seg_ind += 1
                        if pind not in bp_seen:
                            path_queue.append(pind)
                            bp_seen.append(pind)
                        break
                    else:
                        segment.append(pind)
                else: #after the for loop which means it did not experience a break
                    segments.append(np.array(segment))
                    segment_map[segment] = seg_ind
                    seg_ind += 1
            else:
                if len(segment) < 2:
                    continue
                #print(f"segment for length 1 = {segment}")
                segments.append(np.array(segment))
                segment_map[segment] = seg_ind
                seg_ind += 1
        return segments, segment_map.astype(int)


# --------------------- Functions from skeletonize.py ------------------- #
def skeletonize_mesh(mesh, soma_pt=None, soma_radius=7500, collapse_soma=True,
                     invalidation_d=12000, smooth_vertices=False, smooth_neighborhood=5,compute_radius=True,
                     compute_original_index=True, cc_vertex_thresh=100, verbose=True):
    '''
    Build skeleton object from mesh skeletonization

    Parameters
    ----------
    mesh: meshparty.trimesh_io.Mesh
        the mesh to skeletonize, defaults assume vertices in nm
    soma_pt: np.array
        a length 3 array specifying to soma location to make the root
        default=None, in which case a heuristic root will be chosen
        in units of mesh vertices. 
    soma_radius: float
        distance in mesh vertex units over which to consider mesh 
        vertices close to soma_pt to belong to soma
        these vertices will automatically be invalidated and no
        skeleton branches will attempt to reach them.
        This distance will also be used to collapse all skeleton
        points within this distance to the soma_pt root if collpase_soma
        is true. (default=7500 (nm))
    collapse_soma: bool
        whether to collapse the skeleton around the soma point (default True)
    invalidation_d: float
        the distance along the mesh to invalidate when applying TEASAR
        like algorithm.  Controls how detailed a structure the skeleton
        algorithm reaches. default (12000 (nm))
    smooth_vertices: bool
        whether to smooth the vertices of the skeleton
    compute_radius: bool
        whether to calculate the radius of the skeleton at each point on the skeleton
        (default True)
    compute_original_index: bool
        whether to calculate how each of the mesh nodes maps onto the skeleton
        (default True)
    cc_vertex_thresh : int, optional
        Smallest number of vertices in a connected component to skeletonize.
    verbose: bool
        whether to print verbose logging

    Returns
    -------
    :obj:`meshparty.skeleton.Skeleton`
           a Skeleton object for this mesh
    '''
    
    if verbose:
        print(f"smooth_neighborhood = {smooth_neighborhood}")
    skel_verts, skel_edges, smooth_verts, orig_skel_index, skel_map = calculate_skeleton_paths_on_mesh(mesh,
                                                                                                       soma_pt=soma_pt,
                                                                                                       soma_thresh=soma_radius,
                                                                                                       invalidation_d=invalidation_d,
                                                                                                       cc_vertex_thresh=cc_vertex_thresh,
                                                                                                       smooth_neighborhood=smooth_neighborhood,
                                                                                                      return_map=True)
    if verbose:
        print(f"skel_verts.shape = {skel_verts.shape}")
        print(f"orig_skel_index.shape = {orig_skel_index.shape}")
        

    if smooth_vertices is True:
        skel_verts = smooth_verts
        
    if verbose:
        print(f"After smooth verts skel_verts.shape = {skel_verts.shape}")

    if collapse_soma is True and soma_pt is not None:
        soma_verts = mesh_filters.filter_spatial_distance_from_points(
            mesh, [soma_pt], soma_radius)
        new_v, new_e, new_skel_map, vert_filter, root_ind = collapse_soma_skeleton(soma_pt, skel_verts, skel_edges,
                                                                                   soma_d_thresh=soma_radius, mesh_to_skeleton_map=skel_map,
                                                                                   soma_mesh_indices=soma_verts, return_filter=True,
                                                                                   return_soma_ind=True)
    else:
        new_v, new_e, new_skel_map = skel_verts, skel_edges, skel_map
        vert_filter = np.arange(len(orig_skel_index))
        
        if verbose:
            print(f"new_v.shape = {new_v.shape}")
            print(f"vert_filter.shape = {vert_filter.shape}")

        if soma_pt is None:
            sk_graph = utils.create_csgraph(new_v, new_e)
            root_ind = utils.find_far_points_graph(sk_graph)[0]
        else:
            # Still try to root close to the soma
            _, qry_inds = pyKDTree(new_v).query(soma_pt[np.newaxis, :])
            root_ind = qry_inds[0]

    skel_map_full_mesh = np.full(mesh.node_mask.shape, -1, dtype=np.int64)
    skel_map_full_mesh[mesh.node_mask] = new_skel_map
    ind_to_fix = mesh.map_boolean_to_unmasked(np.isnan(new_skel_map))
    skel_map_full_mesh[ind_to_fix] = -1

    props = {}
    
    if verbose:
        print(f"orig_skel_index[vert_filter].shape = {orig_skel_index[vert_filter].shape}")
        
    if compute_original_index is True:
        props['mesh_index'] = np.append(
            mesh.map_indices_to_unmasked(orig_skel_index[vert_filter]), -1)
    if compute_radius is True:

        rs = ray_trace_distance(orig_skel_index[vert_filter], mesh)
        rs = np.append(rs, soma_radius)
        props['rs'] = rs

    sk = Skeleton(new_v, new_e, mesh_to_skel_map=skel_map_full_mesh,
                  vertex_properties=props, root=root_ind)
    return sk, skel_map


def calculate_skeleton_paths_on_mesh(mesh, soma_pt=None, soma_thresh=7500,
                                     invalidation_d=10000, smooth_neighborhood=5,
                                     large_skel_path_threshold=5000,
                                     cc_vertex_thresh=50,  return_map=False):
    """ function to turn a trimesh object of a neuron into a skeleton, without running soma collapse,
    or recasting result into a Skeleton.  Used by :func:`meshparty.skeletonize.skeletonize_mesh` and
    makes use of :func:`meshparty.skeletonize.skeletonize_components`

    Parameters
    ----------
    mesh: meshparty.trimesh_io.Mesh
        the mesh to skeletonize, defaults assume vertices in nm
    soma_pt: np.array
        a length 3 array specifying to soma location to make the root
        default=None, in which case a heuristic root will be chosen
        in units of mesh vertices
    soma_thresh: float
        distance in mesh vertex units over which to consider mesh 
        vertices close to soma_pt to belong to soma
        these vertices will automatically be invalidated and no
        skeleton branches will attempt to reach them.
        This distance will also be used to collapse all skeleton
        points within this distance to the soma_pt root if collpase_soma
        is true. (default=7500 (nm))
    invalidation_d: float
        the distance along the mesh to invalidate when applying TEASAR
        like algorithm.  Controls how detailed a structure the skeleton
        algorithm reaches. default (10000 (nm))
    smooth_neighborhood: int
        the neighborhood in edge hopes over which to smooth skeleton locations.
        This controls the smoothing of the skeleton
        (default 5)
    large_skel_path_threshold: int
        the threshold in terms of skeleton vertices that skeletons will be
        nominated for tip merging.  Smaller skeleton fragments 
        will not be merged at their tips (default 5000)
    cc_vertex_thresh: int
        the threshold in terms of vertex numbers that connected components
        of the mesh will be considered for skeletonization. mesh connected
        components with fewer than these number of vertices will be ignored
        by skeletonization algorithm. (default 100)
    return_map: bool
        whether to return a map of how each mesh vertex maps onto each skeleton vertex
        based upon how it was invalidated.

    Returns
    -------
        skel_verts: np.array
            a Nx3 matrix of skeleton vertex positions
        skel_edges: np.array
            a Kx2 matrix of skeleton edge indices into skel_verts
        smooth_verts: np.array
            a Nx3 matrix of vertex positions after smoothing
        skel_verts_orig: np.array
            a N long index of skeleton vertices in the original mesh vertex index
        (mesh_to_skeleton_map): np.array
            a Mx2 map of mesh vertex indices to skeleton vertex indices

    """
    debug = False
    if debug:
        print(f"smooth_neighborhood = {smooth_neighborhood}")
        print(f"soma_pt = {soma_pt}")
        print(f"invalidation_d = {invalidation_d}")
        cc_vertex_thresh = 25
        print(f"cc_vertex_thresh = {cc_vertex_thresh}")

        
    cc_vertex_thresh = 10
    print(f"cc_vertex_thresh = {cc_vertex_thresh}")
    skeletonize_output = skeletonize_components(mesh,
                                                soma_pt=soma_pt,
                                                soma_thresh=soma_thresh,
                                                invalidation_d=invalidation_d,
                                                cc_vertex_thresh=cc_vertex_thresh,
                                                return_map=return_map)
    if return_map is True:
        all_paths, roots, tot_path_lengths, mesh_to_skeleton_map = skeletonize_output
    else:
        all_paths, roots, tot_path_lengths = skeletonize_output
        
    if debug:
        print(f"all_paths = {all_paths}")
        print(f"roots = {roots}")
        print(f"tot_path_lengths = {tot_path_lengths}")
        

    all_edges = []
    for comp_paths in all_paths:
        all_edges.append(utils.paths_to_edges(comp_paths))
    if len(all_edges) > 0:
        tot_edges = np.vstack(all_edges)
    else:
        tot_edges = np.zeros((3, 0))

    skel_verts, skel_edges, skel_verts_orig = reduce_verts(
        mesh.vertices, tot_edges)
    smooth_verts = smooth_graph(
        skel_verts, skel_edges, neighborhood=smooth_neighborhood)

    if return_map:
        mesh_to_skeleton_map = utils.nanfilter_shapes(
            np.unique(tot_edges.ravel()), mesh_to_skeleton_map)
    else:
        mesh_to_skeleton_map = None

    output_tuple = (skel_verts, skel_edges, smooth_verts, skel_verts_orig)

    if return_map:
        output_tuple = output_tuple + (mesh_to_skeleton_map,)

    return output_tuple


def reduce_verts(verts, faces):
    """removes unused vertices from a graph or mesh

    Parameters
    ----------
    verts : numpy.array
        NxD numpy array of vertex locations
    faces : numpy.array
        MxK numpy array of connected shapes (i.e. edges or tris)
        (entries are indices into verts)

    Returns
    ------- 
    np.array
        new_verts, a filtered set of vertices 
    np.array
        new_face, a reindexed set of faces (or edges)
    np.array
        used_verts, the index of the new_verts in the old verts

    """
    try:
        used_verts = np.unique(faces.ravel())
        new_verts = verts[used_verts, :]
        new_face = np.zeros(faces.shape, dtype=faces.dtype)
        for i in range(faces.shape[1]):
            new_face[:, i] = np.searchsorted(used_verts, faces[:, i])
    except:
        print(f"verts={verts}")
        print(f"faces={faces}")
        print(f"used_verts = {used_verts}")
        raise Exception
    return new_verts, new_face, used_verts


def skeletonize_components(mesh,
                           soma_pt=None,
                           soma_thresh=10000,
                           invalidation_d=10000,
                           cc_vertex_thresh=100,
                           return_map=False):
    """ core skeletonization routine, used by :func:`meshparty.skeletonize.calculate_skeleton_paths_on_mesh`
    to calculcate skeleton on all components of mesh, with no post processing """
    # find all the connected components in the mesh
    n_components, labels = sparse.csgraph.connected_components(mesh.csgraph,
                                                               directed=False,
                                                               return_labels=True)
    comp_labels, comp_counts = np.unique(labels, return_counts=True)

    if return_map:
        mesh_to_skeleton_map = np.full(len(mesh.vertices), np.nan)

    # variables to collect the paths, roots and path lengths
    all_paths = []
    roots = []
    tot_path_lengths = []

    if soma_pt is not None:
        soma_d = mesh.vertices - soma_pt[np.newaxis, :]
        soma_d = np.linalg.norm(soma_d, axis=1)
        is_soma_pt = soma_d < soma_thresh
    else:
        is_soma_pt = None
        soma_d = None
    # loop over the components
    for k in trange(n_components):
        if comp_counts[k] > cc_vertex_thresh:

            # find the root using a soma position if you have it
            # it will fall back to a heuristic if the soma
            # is too far away for this component
            root, root_ds, pred, valid = setup_root(mesh,
                                                    is_soma_pt,
                                                    soma_d,
                                                    labels == k)
            # run teasar on this component
            teasar_output = mesh_teasar(mesh,
                                        root=root,
                                        root_ds=root_ds,
                                        root_pred=pred,
                                        valid=valid,
                                        invalidation_d=invalidation_d,
                                        return_map=return_map)
            if return_map is False:
                paths, path_lengths = teasar_output
            else:
                paths, path_lengths, mesh_to_skeleton_map_single = teasar_output
                mesh_to_skeleton_map[~np.isnan(
                    mesh_to_skeleton_map_single)] = mesh_to_skeleton_map_single[~np.isnan(mesh_to_skeleton_map_single)]

            if len(path_lengths) > 0:
                # collect the results in lists
                tot_path_lengths.append(path_lengths)
                all_paths.append(paths)
                roots.append(root)

    if return_map:
        return all_paths, roots, tot_path_lengths, mesh_to_skeleton_map
    else:
        return all_paths, roots, tot_path_lengths


def setup_root(mesh, is_soma_pt=None, soma_d=None, is_valid=None):
    """ function to find the root index to use for this mesh """
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
    return root, root_ds, pred, valid


def mesh_teasar(mesh, root=None, valid=None, root_ds=None, root_pred=None, soma_pt=None,
                soma_thresh=7500, invalidation_d=10000, return_timing=False, return_map=False):
    """core skeletonization function used to skeletonize a single component of a mesh"""
    # if no root passed, then calculation one
    if root is None:
        root, root_ds, root_pred, valid = setup_root(mesh,
                                                     soma_pt=soma_pt,
                                                     soma_thresh=soma_thresh)
    # if root_ds have not be precalculated do so
    if root_ds is None:
        root_ds, root_pred = sparse.csgraph.dijkstra(mesh.csgraph,
                                                     False,
                                                     root,
                                                     return_predecessors=True)
    # if certain vertices haven't been pre-invalidated start with just
    # the root vertex invalidated
    if valid is None:
        valid = np.ones(len(mesh.vertices), np.bool)
        valid[root] = False
    else:
        if (len(valid) != len(mesh.vertices)):
            raise Exception("valid must be length of vertices")

    if return_map == True:
        mesh_to_skeleton_dist = np.full(len(mesh.vertices), np.inf)
        mesh_to_skeleton_map = np.full(len(mesh.vertices), np.nan)

    total_to_visit = np.sum(valid)
    if np.sum(np.isinf(root_ds) & valid) != 0:
        print(np.where(np.isinf(root_ds) & valid))
        raise Exception("all valid vertices should be reachable from root")

    # vector to store each branch result
    paths = []

    # vector to store each path's total length
    path_lengths = []

    # keep track of the nodes that have been visited
    visited_nodes = [root]

    # counter to track how many branches have been counted
    k = 0

    # arrays to track timing
    start = time.time()
    time_arrays = [[], [], [], [], []]

    with tqdm(total=total_to_visit) as pbar:
        # keep looping till all vertices have been invalidated
        while(np.sum(valid) > 0):
            k += 1
            t = time.time()
            # find the next target, farthest vertex from root
            # that has not been invalidated
            target = np.nanargmax(root_ds*valid)
            if (np.isinf(root_ds[target])):
                raise Exception('target cannot be reached')
            time_arrays[0].append(time.time()-t)

            t = time.time()
            # figure out the longest this branch could be
            # by following the route from target to the root
            # and finding the first already visited node (max_branch)
            # The dist(root->target) - dist(root->max_branch)
            # is the maximum distance the shortest route to a branch
            # point from the target could possibly be,
            # use this bound to reduce the djisktra search radius for this target
            max_branch = target
            while max_branch not in visited_nodes:
                max_branch = root_pred[max_branch]
            max_path_length = root_ds[target]-root_ds[max_branch]

            # calculate the shortest path to that vertex
            # from all other vertices
            # up till the distance to the root
            ds, pred_t = sparse.csgraph.dijkstra(
                mesh.csgraph,
                False,
                target,
                limit=max_path_length,
                return_predecessors=True)

            # pick out the vertex that has already been visited
            # which has the shortest path to target
            min_node = np.argmin(ds[visited_nodes])
            # reindex to get its absolute index
            branch = visited_nodes[min_node]
            # this is in the index of the point on the skeleton
            # we want this branch to connect to
            time_arrays[1].append(time.time()-t)

            t = time.time()
            # get the path from the target to branch point
            path = utils.get_path(target, branch, pred_t)
            visited_nodes += path[0:-1]
            # record its length
            assert(~np.isinf(ds[branch]))
            path_lengths.append(ds[branch])
            # record the path
            paths.append(path)
            time_arrays[2].append(time.time()-t)

            t = time.time()
            # get the distance to all points along the new path
            # within the invalidation distance
            dm, _, sources = sparse.csgraph.dijkstra(
                mesh.csgraph, False, path, limit=invalidation_d,
                min_only=True, return_predecessors=True)
            time_arrays[3].append(time.time()-t)

            t = time.time()
            # all such non infinite distances are within the invalidation
            # zone and should be marked invalid
            nodes_to_update = ~np.isinf(dm)
            marked = np.sum(valid & ~np.isinf(dm))
            if return_map == True:
                new_sources_closer = dm[nodes_to_update] < mesh_to_skeleton_dist[nodes_to_update]
                mesh_to_skeleton_map[nodes_to_update] = np.where(new_sources_closer,
                                                                 sources[nodes_to_update],
                                                                 mesh_to_skeleton_map[nodes_to_update])
                mesh_to_skeleton_dist[nodes_to_update] = np.where(new_sources_closer,
                                                                  dm[nodes_to_update],
                                                                  mesh_to_skeleton_dist[nodes_to_update])

            valid[~np.isinf(dm)] = False

            # print out how many vertices are still valid
            pbar.update(marked)
            time_arrays[4].append(time.time()-t)
    # record the total time
    dt = time.time() - start

    out_tuple = (paths, path_lengths)
    if return_map:
        out_tuple = out_tuple + (mesh_to_skeleton_map,)
    if return_timing:
        out_tuple = out_tuple + (time_arrays, dt)

    return out_tuple


def smooth_graph(values, edges, mask=None, neighborhood=2, iterations=100, r=.1):
    """ smooths a spatial graph via iterative local averaging
        calculates the average value of neighboring values
        and relaxes the values toward that average

        Parameters
        ----------
        values : numpy.array
            a NxK numpy array of values, for example xyz positions
        edges : numpy.array
            a Mx2 numpy array of indices into values that are edges
        mask : numpy.array
            NOT yet implemented
            optional N boolean vector of values to mask
            the vert locations.  the result will return a result at every vert
            but the values that are false in this mask will be ignored and not
            factored into the smoothing.
        neighborhood : int
            an integer of how far in the graph to relax over
            as being local to any vertex (default = 2)
        iterations : int
            number of relaxation iterations (default = 100)
        r : float
            relaxation factor at each iteration
            new_vertex = (1-r)*old_vertex*mask + (r+(1-r)*(1-mask))*(local_avg)
            default = .1

        Returns
        -------
        np.array
            new_verts, a Nx3 list of new smoothed vertex positions

    """
    N = len(values)
    E = len(edges)

    # setup a sparse matrix with the edges
    sm = sparse.csc_matrix(
        (np.ones(E), (edges[:, 0], edges[:, 1])), shape=(N, N))

    # an identity matrix
    eye = sparse.csc_matrix((np.ones(N, dtype=np.float32),
                             (np.arange(0, N), np.arange(0, N))),
                            shape=(N, N))
    # for undirected graphs we want it symettric
    sm = sm + sm.T

    # this will store our relaxation matrix
    C = sparse.csc_matrix(eye)
    # multiple the matrix and add to itself
    # to spread connectivity along the graph
    for i in range(neighborhood):
        C = C + sm @ C
    # zero out the diagonal elements
    C.setdiag(np.zeros(N))
    # don't overweight things that are connected in more than one way
    C = C.sign()
    # measure total effective neighbors per node
    neighbors = np.sum(C, axis=1)

    # normalize the weights of neighbors according to number of neighbors
    neighbors = 1/neighbors
    C = C.multiply(neighbors)
    # convert back to csc
    C = C.tocsc()

    # multiply weights by relaxation term
    C *= r

    # construct relaxation matrix, adding on identity with complementary weight
    A = C + (1-r)*eye

    # make a copy of original vertices to no destroy inpuyt
    new_values = np.copy(values)

    # iteratively relax the vertices
    for i in range(iterations):
        new_values = A*new_values
    return new_values


def collapse_soma_skeleton(soma_pt, verts, edges, soma_d_thresh=12000, mesh_to_skeleton_map=None,
                           soma_mesh_indices=None, return_filter=False, only_soma_component=True, return_soma_ind=False):
    """function to adjust skeleton result to move root to soma_pt 

    Parameters
    ----------
    soma_pt : numpy.array
        a 3 long vector of xyz locations of the soma (None to just remove duplicate )
    verts : numpy.array
        a Nx3 array of xyz vertex locations
    edges : numpy.array
        a Kx2 array of edges of the skeleton
    soma_d_thresh : float
        distance from soma_pt to collapse skeleton nodes
    mesh_to_skeleton_map : np.array
        a M long array of how each mesh index maps to a skeleton vertex
        (default None).  The function will update this as it collapses vertices to root.
    soma_mesh_indices : np.array
         a K long array of indices in the mesh that should be considered soma
         Any  skeleton vertex on these vertices will all be collapsed to root.
    return_filter : bool
        whether to return a list of which skeleton vertices were used in the end
        for the reduced set of skeleton vertices
    only_soma_component : bool
        whether to collapse only the skeleton connected component which is closest to the soma_pt
        (default True)
    return_soma_ind : bool
        whether to return which skeleton index that is the soma_pt

    Returns
    -------
    np.array
        verts, Px3 array of xyz skeleton vertices
    np.array
        edges, Qx2 array of skeleton edges
    (np.array)
        new_mesh_to_skeleton_map, returned if mesh_to_skeleton_map and soma_pt passed 
    (np.array)
        used_vertices, if return_filter this contains the indices into the passed verts which the return verts is using
    int
        an index into the returned verts that is the root of the skeleton node, only returned if return_soma_ind is True

    """
    if soma_pt is not None:
        if only_soma_component:
            closest_soma_ind = np.argmin(np.linalg.norm(verts-soma_pt, axis=1))
            close_inds = np.linalg.norm(verts-soma_pt, axis=1) < soma_d_thresh
            orig_graph = utils.create_csgraph(
                verts, edges, euclidean_weight=False)
            speye = sparse.diags(close_inds.astype(int))
            _, compids = sparse.csgraph.connected_components(
                orig_graph * speye)
            soma_verts = np.flatnonzero(compids[closest_soma_ind] == compids)
        else:
            dv = np.linalg.norm(verts - soma_pt_m, axis=1)
            soma_verts = np.where(dv < soma_d_thresh)[0]

        soma_pt_m = soma_pt[np.newaxis, :]
        new_verts = np.vstack((verts, soma_pt_m))
        soma_i = verts.shape[0]
        edges_m = edges.copy()
        edges_m[np.isin(edges, soma_verts)] = soma_i

        simple_verts, simple_edges = trimesh_vtk.remove_unused_verts(
            new_verts, edges_m)
        good_edges = ~(simple_edges[:, 0] == simple_edges[:, 1])

        if mesh_to_skeleton_map is not None:
            new_mesh_to_skeleton_map = mesh_to_skeleton_map.copy()
            remap_rows = np.isin(mesh_to_skeleton_map, soma_verts)
            new_mesh_to_skeleton_map[remap_rows] = soma_i
            new_mesh_to_skeleton_map = utils.nanfilter_shapes(np.unique(edges_m.ravel()),
                                                              new_mesh_to_skeleton_map)
            if soma_mesh_indices is not None:
                new_mesh_to_skeleton_map[soma_mesh_indices] = len(
                    simple_verts)-1

        output = [simple_verts, simple_edges[good_edges]]
        if mesh_to_skeleton_map is not None:
            output.append(new_mesh_to_skeleton_map)
        if return_filter:
            # Remove the largest value which is soma_i
            used_vertices = np.unique(edges_m.ravel())[:-1]
            output.append(used_vertices)
        if return_soma_ind:
            output.append(len(simple_verts)-1)
        return output

    else:
        simple_verts, simple_edges = trimesh_vtk.remove_unused_verts(
            verts, edges)
        return simple_verts, simple_edges


def ray_trace_distance(vertex_inds, mesh, max_iter=10, rand_jitter=0.001, verbose=False, ray_inter=None):
    '''
    Compute distance to opposite side of the mesh for specified vertex indices on the mesh.

    Parameters
    ----------
    vertex_inds : np.array
        a K long set of indices into the mesh.vertices that you want to perform ray tracing on
    mesh : :obj:`meshparty.trimesh_io.Mesh`
        mesh to perform ray tracing on
    max_iter : int
        maximum retries to attempt in order to get a proper sdf measure (default 10)
    rand_jitter : float
        the amplitude of gaussian jitter on the vertex normal to add on each iteration (default .001)
    verbose : bool
        whether to print debug statements (default False)
    ray_inter: ray_pyembree.RayMeshIntersector
        a ray intercept object pre-initialized with a mesh, in case y ou are doing this many times
        and want to avoid paying initialization costs. (default None) will initialize it for you

    Returns
    -------
    np.array
        rs, a K long array of sdf values. rays with no result after max_iters will contain zeros.

    '''
    if not trimesh.ray.has_embree:
        logging.warning(
            "calculating rays without pyembree, conda install pyembree for large speedup")

    if ray_inter is None:
        ray_inter = ray_pyembree.RayMeshIntersector(mesh)

    rs = np.zeros(len(vertex_inds))
    good_rs = np.full(len(rs), False)

    it = 0
    while not np.all(good_rs):
        if verbose:
            print(np.sum(~good_rs))
        blank_inds = np.where(~good_rs)[0]
        starts = (mesh.vertices -
                  mesh.vertex_normals)[vertex_inds, :][~good_rs, :]
        vs = -mesh.vertex_normals[vertex_inds, :] \
            + (1.2**it)*rand_jitter*np.random.rand(*
                                                   mesh.vertex_normals[vertex_inds, :].shape)
        vs = vs[~good_rs, :]

        rtrace = ray_inter.intersects_location(starts, vs, multiple_hits=False)

        if len(rtrace[0] > 0):
            # radius values
            rs[blank_inds[rtrace[1]]] = np.linalg.norm(
                mesh.vertices[vertex_inds, :][rtrace[1]]-rtrace[0], axis=1)
            good_rs[blank_inds[rtrace[1]]] = True
        it += 1
        if it > max_iter:
            break
    return rs

# ================= 10/12 added skeletonization for decomposition=================================== #

from copy import deepcopy
from importlib import reload
import numpy as np
import networkx as nx
import time
from meshparty import trimesh_io

def skeletonize_mesh_largest_component(mesh,
                                    root=None,
                                      verbose=False,
                                      filter_mesh=True, #will filter the mesh for just one connected piece
                                      ):
    """
    To run the skeletonization on the 
    largest connected component of one mesh
    
    Example 1:
    #How to get the skeleton from the skeleton object
    sk_meshparty = sk_meshparty_obj.vertices[sk_meshparty_obj.edges]
    
    """
    limb_mesh_mparty = deepcopy(mesh)

#     # just getting the largest connected component
#     connect_comp = list(nx.connected_components(nx.from_edgelist(limb_mesh_mparty.face_adjacency)))
#     max_connect_comp = np.argmax([len(k) for k in connect_comp])
#     limb_mesh_mparty = limb_mesh_mparty.submesh([list(connect_comp[max_connect_comp])],append=True)
    if filter_mesh:
        limb_mesh_mparty = limb_mesh_mparty.split(only_watertight=False,repair=False)[0]

    limb_obj_tr_io  = trimesh_io.Mesh(vertices = limb_mesh_mparty.vertices,
                                                           faces = limb_mesh_mparty.faces,
                                                           normals=limb_mesh_mparty.face_normals)

    meshparty_time = time.time()
    if verbose:
        print("\nStep 1: Starting Skeletonization")
    sk_meshparty_obj, v = skeletonize_mesh(limb_obj_tr_io,
                          soma_pt = root,
                          soma_radius = 0,
                          collapse_soma = False,
                          invalidation_d=12000,
                          smooth_vertices=True,
                           smooth_neighborhood = 1,
                          compute_radius = True, #Need the pyembree list
                          compute_original_index=True,
                          verbose=verbose)
    if verbose:
        print(f"Total time for meshParty skeletonization = {time.time() - meshparty_time}")
    if filter_mesh:
        return sk_meshparty_obj,limb_mesh_mparty
    else:
        return sk_meshparty_obj


import time
import skeleton_utils as sk
import general_utils as gu
import numpy as np
import compartment_utils as cu
import itertools
import trimesh_utils as tu

def skeleton_obj_to_branches(sk_meshparty_obj,
                             mesh,
                            meshparty_n_surface_downsampling = 0,
                            meshparty_segment_size = 100,
                            verbose=False
                            ):
    debug = False
    
    if debug:
        print(f"mesh = {mesh}")
        print(f"meshparty_segment_size = {meshparty_segment_size}")
        print(f"meshparty_n_surface_downsampling = {meshparty_n_surface_downsampling}")
    

    limb_mesh_mparty = mesh
    #Step 2: Getting the branches
    if verbose:
        print("\nStep 2: Decomposing Branches")
    meshparty_time = time.time()

    segments, segment_maps = compute_segments(sk_meshparty_obj)
    # getting the skeletons that go with them
    segment_branches = np.array([sk_meshparty_obj.vertices[np.vstack([k[:-1],k[1:]]).T] for k in segments])
    
    if debug:
        print(f"segments = {segments}")
        print(f"segment_branches = {segment_branches}")
    
    branches_touching_root = sk.find_branch_skeleton_with_specific_coordinate(segment_branches,
                                        current_coordinate=sk_meshparty_obj.vertices[sk_meshparty_obj.root])
    
    
    
    #combine segments that are connected at the root if there are only 2
    if len(branches_touching_root) ==2:
        print("connecting at the root")
        keep_ind = np.setdiff1d(np.arange(len(segments)),branches_touching_root).astype("int")
        
        #calculating the new segment
        b_touch_seg_1 = segments[branches_touching_root[0]]
        b_touch_seg_2 = segments[branches_touching_root[1]]

        b_touch_seg_1_ends = b_touch_seg_1[[0,-1]]
        b_touch_seg_2_ends = b_touch_seg_2[[0,-1]]
        b_1_root_end = np.where(b_touch_seg_1_ends == sk_meshparty_obj.root)[0]
        b_2_root_end = np.where(b_touch_seg_2_ends == sk_meshparty_obj.root)[0]
        b_1_root_end,b_2_root_end

        if b_1_root_end == 0:
            b_touch_seg_1 = np.flip(b_touch_seg_1)
        if b_2_root_end == 1:
            b_touch_seg_2 = np.flip(b_touch_seg_2)
            
        new_seg = np.concatenate([b_touch_seg_1[:-1],b_touch_seg_2])
        
        
        #adding the new segment to the older segments
        new_segments = [segments[k] for k in keep_ind]
        new_segments.append(new_seg)
        
        new_segment_branches = np.array([sk_meshparty_obj.vertices[np.vstack([k[:-1],k[1:]]).T] for k in new_segments])
        
        segments = new_segments
        segment_branches = new_segment_branches
        
    branches_touching_root = sk.find_branch_skeleton_with_specific_coordinate(segment_branches,
                                        current_coordinate=sk_meshparty_obj.vertices[sk_meshparty_obj.root])
    print(f"branches_touching_root = {branches_touching_root}")
    #raise Exception("")
        

    #------------ Add in the downsampling and resizing ----------------- #


    if meshparty_n_surface_downsampling > 0:
        if verbose:
            print(f"Going to downsample the meshparty segments {meshparty_n_surface_downsampling}")
        for j,s in enumerate(segment_branches):
            for i in range(n_surface_downsampling):
                s = sk.downsample_skeleton(s)
            segment_branches[j] = s

    
    if meshparty_segment_size > 0:
        new_segment_branches = []
        if verbose:
            print(f"Resizing meshparty skeletal segments to length {meshparty_segment_size} nm")
        for j,s in enumerate(segment_branches):
            new_segment_branches.append(sk.resize_skeleton_branch(s,segment_width = meshparty_segment_size))
            
        if debug:
            print(f"new_segment_branches = {new_segment_branches}")
        segment_branches = np.array(new_segment_branches)

    #------------ END OF downsampling and resizing ----------------- #

    if verbose:
        print(f"Total time for meshParty decomposition = {time.time() - meshparty_time}")


    # -- Step 3: Creating the mesh correspondence --
    if verbose:
        print("\nStep 3: Mesh correspondence")
    meshparty_time = time.time()

    sk_vertices_to_mesh_vertices = gu.invert_mapping(sk_meshparty_obj.mesh_to_skel_map)
    #getting a list of all the original vertices that belong to each segment
    segment_mesh_vertices = [np.unique(np.concatenate([sk_vertices_to_mesh_vertices[k] for k in segment_list])) for segment_list in segments]
    #getting a list of all the original vertices that belong to each segment
    segment_mesh_faces = [np.unique(limb_mesh_mparty.vertex_faces[k]) for k in segment_mesh_vertices]
    segment_mesh_faces = [k[k>=0] for k in segment_mesh_faces]
    
    # --------------- 10/29: Adding in the part that combines the branch points that are close ----------- #
    
    segment_branches_filtered,kept_branches_idx = sk.combine_close_branch_points(
                                                            skeleton_branches=segment_branches)
    
    segment_branches_filtered = np.array(segment_branches_filtered)
    #print(f"kept_branches_idx = {kept_branches_idx}")
    print(f"max(kept_branches_idx) = {max(kept_branches_idx)}, len(kept_branches_idx) = {len(kept_branches_idx)}")
    segment_mesh_faces_filtered = [k for i,k in enumerate(segment_mesh_faces) if i in set(kept_branches_idx)]
    
    

    #face_lookup = gu.invert_mapping(segment_mesh_faces)
    face_lookup = gu.invert_mapping(segment_mesh_faces_filtered)

    curr_limb_mesh = limb_mesh_mparty


    original_labels = set(list(itertools.chain.from_iterable(list(face_lookup.values()))))
    if verbose:
        print(f"max(original_labels),len(original_labels) = {(max(original_labels),len(original_labels))}")

    face_coloring_copy = cu.resolve_empty_conflicting_face_labels(curr_limb_mesh = curr_limb_mesh,
                                                                face_lookup=face_lookup,
                                                                no_missing_labels = list(original_labels))


    # -- splitting the mesh pieces into individual pieces
    divided_submeshes,divided_submeshes_idx = tu.split_mesh_into_face_groups(curr_limb_mesh,face_coloring_copy,
                                                                            return_dict=False)


    if verbose:
        print(f"Total time for meshParty mesh correspondence = {time.time() - meshparty_time}")

    # -- Step 4: Getting the Widths ---
    if verbose:
        print("\nStep 4: Retrieving Widths")
    meshparty_time = time.time()

    segment_width_measurements = [sk_meshparty_obj.vertex_properties["rs"][k] for k in segments]
    segment_widths_median = np.array([np.median(k) for k in segment_width_measurements])
    segment_widths_median_filtered = segment_widths_median[kept_branches_idx]

    if verbose:
        print(f"Total time for meshParty Retrieving Widths = {time.time() - meshparty_time}")
    # ---- Our Final Products -----
    
    segment_branches = np.array(segment_branches)

#     return (segment_branches, #skeleton branches
#             divided_submeshes, divided_submeshes_idx, #mesh correspondence (mesh and indices)
#             segment_widths_median) #widths

    return (segment_branches_filtered, #skeleton branches
            divided_submeshes, divided_submeshes_idx, #mesh correspondence (mesh and indices)
            segment_widths_median_filtered) #widths
    

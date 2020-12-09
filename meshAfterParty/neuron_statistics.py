import numpy as np
import numpy_utils as nu
import trimesh_utils as tu
import error_detection as ed
import networkx as nx
import networkx_utils as xu
import neuron_utils as nru
import neuron_visualizations as nviz

def neuron_path_analysis(neuron_obj,
                        N = 3,
                        plot_paths = False,
                        return_dj_inserts = True,
                        verbose = False):

    """
    Pseudocode: 
    1) Get all the errored branches
    For Each Limb:
    2) Remove the errored branches
    3) Find all branches that are N steps away from starting node (and get the paths)
    4) Filter away paths that do not have all degrees of 2 on directed network
    *** Those are the viable paths we would analyze***
    5) Extract the statistics

    """

    # 0) Compute the mesh center of the soma
    curr_soma = neuron_obj["S0"].mesh
    curr_soma_center = tu.mesh_center_vertex_average(curr_soma)
    y_vector = np.array([0,-1,0])

    # 1) Getting all Errored Branches 
    error_branches = ed.error_branches_by_axons(neuron_obj,visualize_errors_at_end=False)

    # ----------- Loop that will iterate through all branches ----------- #
    neuron_path_inserts_by_limb = dict()


    total_paths = dict()
    for curr_limb_idx,curr_limb_obj in enumerate(neuron_obj):
        l_name = f"L{curr_limb_idx}"
        if l_name in error_branches.keys():
            curr_limb_error_branches = error_branches[l_name]
        else:
            curr_limb_error_branches = []

        # 2) Remove the errored branches
        if len(curr_limb_obj.all_concept_network_data)>1:
            raise Exception(f"More than one starting node for limb {curr_limb_idx}")

        st_node = curr_limb_obj.current_starting_node
        st_coordinates = curr_limb_obj.current_starting_coordinate



        #2-4: Getting the paths we want

        G = nx.Graph(curr_limb_obj.concept_network)

        target_to_path = nx.single_source_shortest_path(G, source=st_node)#, cutoff=N+1) 
        paths_of_certain_length = [v for k,v in target_to_path.items() if (len(v) == N) ]
        if verbose:
            print(f"Number of paths with {N} nodes = {len(paths_of_certain_length)}")

        #remove the paths with errors on
        paths_of_certain_length_no_errors = [v for v in paths_of_certain_length if len(np.intersect1d(curr_limb_error_branches,
                                                                                                     v))==0]

        if verbose:
            print(f"Number of paths with {N} nodes and no Errors = {len(paths_of_certain_length_no_errors)}")

        #need to filter away for high degree nodes along path
        """
        1) Turn network into directional
        2) Find all of the upstream nodes
        3) Filter those paths away where existence of greater than 2 degrees
        """
        #1) Turn network into directional
        G_directional = nx.DiGraph(curr_limb_obj.concept_network_directional)

        final_paths = []
        for ex_path in paths_of_certain_length_no_errors:
            path_degree = np.array([len(xu.downstream_edges_neighbors(G_directional,k)) for k in ex_path[:-1]] )
            if np.sum(path_degree!=2) == 0:
                final_paths.append(ex_path)
            else:
                if verbose:
                    print(f"Ignoring path because path degrees are {path_degree}")

        if verbose: 
            print(f"Number of paths after filtering away high degree nodes = {len(final_paths)} ")

        if plot_paths:
            if len(final_paths) > 0:
                curr_nodes = np.unique(np.concatenate(final_paths))
            else:
                curr_nodes = []
            total_paths.update({f"L{curr_limb_idx}":curr_nodes})


        # Step 5: Calculating the Statistics on the branches


        """
        Pseudocode: 
        1) Get starting angle of branch

        For all branches not in the starting node
        a) Get the width (all of them)
        b) Get the number of spines, spines_volume, and spine density


        d) Skeletal distance (distance to next branch point)
        e) Angle between parent branch and current branch
        f) Angle between sibling branch and current


        """

        #1) Get starting angle of branch
        st_vector = st_coordinates - curr_soma_center
        st_vector_norm = st_vector/np.linalg.norm(st_vector)
        angle_from_top = np.round(nu.angle_between_vectors(y_vector,st_vector_norm),2)



        limb_path_dict = dict()
        for zz,curr_path in enumerate(final_paths):

            local_dict = dict(soma_angle=angle_from_top)
            for j,n in enumerate(curr_path[1:]):
                curr_name = f"n{j}_"
                curr_node = curr_limb_obj[n]

                #a) Get the width (all of them)
                for w_name,w_value in curr_node.width_new.items():
                    local_dict[curr_name +"width_"+ w_name] = np.round(w_value,2)

                #b) Get the number of spines, spines_volume, and spine density
                attributes_to_export = ["n_spines","total_spine_volume","spine_volume_median",
                                       "spine_volume_density","skeletal_length"]
                for att in attributes_to_export:
                    local_dict[curr_name + att] = np.round(getattr(curr_node,att),2)

                #e) Angle between parent branch and current branch
                local_dict[curr_name + "parent_angle"] = nru.find_parent_child_skeleton_angle(curr_limb_obj,child_node=n)

                #f) Angle between sibling branch and current
                local_dict[curr_name + "sibling_angle"]= list(nru.find_sibling_child_skeleton_angle(curr_limb_obj,child_node=n).values())[0]

            limb_path_dict[zz] = local_dict


        neuron_path_inserts_by_limb[curr_limb_idx] =  limb_path_dict

    if plot_paths:
        nviz.visualize_neuron(neuron_obj,
                          visualize_type=["mesh"],
                          limb_branch_dict=total_paths,
                         mesh_color="red",
                         mesh_whole_neuron=True)   

    if return_dj_inserts:
        # Need to collapse this into a list of dictionaries to insert
        dj_inserts = []
        for limb_idx,limb_paths in neuron_path_inserts_by_limb.items():
            for path_idx,path_dict in limb_paths.items():
                dj_inserts.append(dict(path_dict,limb_idx=limb_idx,path_idx=path_idx))
        return dj_inserts
    else:
        return neuron_path_inserts_by_limb
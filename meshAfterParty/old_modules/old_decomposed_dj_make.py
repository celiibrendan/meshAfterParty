#5) Calculate all starter stats
        # ----------- calculating the attributes --------- #
        n_error_limbs = len(nru.error_limb_indexes(recovered_neuron))

        n_somas = len(recovered_neuron.get_soma_node_names())

        n_limbs = len(recovered_neuron.get_limb_node_names())

        n_branches_per_limb = [len(ex_limb.get_branch_names()) for ex_limb in recovered_neuron]
        n_branches = np.sum(n_branches_per_limb)

        n_spines = len(recovered_neuron.spines)

        #for total skeletal length
        sk_len_per_limb = [sk.calculate_skeleton_distance(limb.skeleton) for limb in recovered_neuron]
        skeletal_length = np.sum(sk_len_per_limb)

        if skeletal_length > 0:
            spine_density = n_spines/skeletal_length
        else:
            spine_density = 0

        max_limb_skeletal_length = np.max(sk_len_per_limb)

        max_limb_n_branches = np.max(n_branches_per_limb)
        
        
        
        
        #5b) --------------- Compute the stats suggested by Jake -------------------------
        
        
        # --> preprocessing
        all_skeletal_lengths = []
        all_widths_no_spine = []
        all_widths = []

        n_spine_eligible_branches = 0
        spine_eligible_branch_lengths = []

        for curr_limb in recovered_neuron:
            for curr_branch in curr_limb:
                curr_branch_sk_len = sk.calculate_skeleton_distance(curr_branch.skeleton)
                all_skeletal_lengths.append(curr_branch_sk_len)
                all_widths_no_spine.append(curr_branch.width_new["no_spine_median_mesh_center"])
                all_widths.append(curr_branch.width_new["median_mesh_center"])

                if not curr_branch.spines is None:
                    spine_eligible_branch_lengths.append(curr_branch_sk_len)
                    n_spine_eligible_branches += 1

        all_skeletal_lengths = np.array(all_skeletal_lengths)
        median_branch_length = np.round(np.median(all_skeletal_lengths),3)
        
        
        #--> width data
        width_median = np.round(np.median(all_widths),3)
        width_no_spine_median = np.round(np.median(all_widths_no_spine),3)

        width_90_perc = np.round(np.percentile(all_widths,90),3)
        width_no_spine_90_perc = np.round(np.percentile(all_widths_no_spine,90),3)
        
        
        
        # -->  spine data
        if n_branches > 0:
            spines_per_branch = np.round(n_spines/n_branches,3)
        else:
            spines_per_branch = 0
        
        #spine eligible density and per branch
        skeletal_length_eligible = np.sum(spine_eligible_branch_lengths)
        if skeletal_length_eligible > 0:
            spine_density_eligible = n_spines/skeletal_length_eligible
        else:
            spine_density_eligible = 0

        if n_branches > 0:
            spines_per_branch_eligible = np.round(n_spines/n_spine_eligible_branches,3)
        else:
            spines_per_branch_eligible = 0

        
        
        spine_time = time.time()
        total_spines = recovered_neuron.spines
        total_volume =[]
        for ts in total_spines:
            vol = tu.mesh_volume(total_spines[0],
                                     watertight_method=None,
                                     return_closed_mesh=False,
                             verbose=False)
            total_volume.append(vol)

        print(f"Total time for calculating spine volume = {time.time() - spine_time}")
        total_volume = np.array(total_volume)

        spine_volume_median = np.median(total_volume)

        total_spine_volume = np.sum(total_volume)

        if skeletal_length > 0:
            spine_volume_density = total_spine_volume/skeletal_length
        else:
            spine_volume_density = 0


        if skeletal_length_eligible > 0:
            spine_volume_density_eligible = total_spine_volume/skeletal_length_eligible
        else:
            spine_volume_density_eligible = 0

        if n_branches > 0:
            spine_volume_per_branch_eligible = np.round(total_spine_volume/n_spine_eligible_branches,3)
        else:
            spine_volume_per_branch_eligible = 0

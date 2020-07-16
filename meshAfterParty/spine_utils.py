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
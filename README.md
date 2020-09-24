# meshAfterParty
BCM Microns Mesh Tools

# Status Quo Notes: 
- visualizations using ipyvolume only currently work in the jupyter notebook and not jupyter lab
(to access just add /tree to the localhost name in the URL, Ex: http://localhost:8890/tree)
- specific versions of python packages were used

# Installation: 
If cloning using git and on a Windows computer, if you want to run the docker environment then 
you must prevent the conversion of line ending formats to Windows, to do this: 

1) git config --global core.autocrlf false #ensures no auto-conversion will occur

2) git clone [____] #clones the repo

3) git config --global core.autocrlf true #returns the git behavior to auto-conversion

** 
if you do not do this then you will get the following error: 

notebook_1  | /usr/bin/env: 'bash\r': No such file or directory
docker_notebook_1 exited with code 127

**

To Build docker image:
0) Have docker or docker desktop installed
1) navigate to docker folder inside of the meshAfterParty repo
2) docker-compose build --no-cache
3) docker-compose up
4) The jupyter notebook will be served at http://localhost:8890/tree
docker

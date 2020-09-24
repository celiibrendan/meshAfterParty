FROM celiib/decimation_skeletonization:latest
LABEL maintainer="Brendan Papadopoulos"

#--7/21: python modules that needed to be added for the neuron package functionality--

#the skeletonization function that allows for parameter passing
ADD ./CGAL/cgal_skeleton_param /src/CGAL/cgal_skeleton_param 
RUN pip3 install -e /src/CGAL/cgal_skeleton_param 

#the missing pythongn modules
RUN pip3 install bz2file ipywidgets pandasql tqdm

#maintaining certain versions of python modules
RUN pip3 install --upgrade --force-reinstall trimesh==3.6.15
RUN pip3 install --upgrade --force-reinstall networkx==2.4
RUN pip3 install --upgrade --force-reinstall ipyvolume==0.5.2

#---- 8/7 addition --- #
RUN pip3 install webcolors

#enabling the visualization of widgets and ipyvolume
RUN apt update
RUN apt install curl -y
RUN apt install nodejs
RUN npm install n -g
RUN n stable
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install ipyvolume

#----------------#enabling the pyembree module-----------------------
RUN apt-get update
RUN apt-get install -y wget

#explains why has to do this so can see the shared library: 
#https://stackoverflow.com/questions/1099981/why-cant-python-find-shared-objects-that-are-in-directories-in-sys-path
#------------- MAY NEED TO FIX THIS --------------- #
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

ADD ./CGAL/cgal_skeleton_param /src/CGAL/cgal_skeleton_param 

#https://github.com/embree/embree#linux-and-macos (for the dependencies)
#for the dependencies
RUN apt-get install -y cmake-curses-gui
RUN apt-get install -y libtbb-dev
RUN apt-get install -y libglfw3-dev


#Then run the following bash script (bash embree.bash)
#trimesh bash file: https://github.com/mikedh/trimesh/blob/master/docker/builds/embree.bash
ADD ./python_bash_files/embree.bash /src/embree.bash
#may need to change this
RUN chmod +x /src/embree.bash && /src/embree.bash 

# This will add in the meshlabserver functionality for cleaning
ARG QMAKE_FLAGS="-spec linux-g++ CONFIG+=release CONFIG+=qml_release CONFIG+=c++11 QMAKE_CXXFLAGS+=-fPIC QMAKE_CXXFLAGS+=-std=c++11 QMAKE_CXXFLAGS+=-fpermissive INCLUDEPATH+=/usr/include/eigen3 LIBS+=-L/meshlab/src/external/lib/linux-g++"

ARG MAKE_FLAGS="-j"

ADD ./meshlab_patch_files/meshlab_mini.pro /meshlab/src/meshlab_mini.pro
WORKDIR /meshlab/src
RUN qmake -qt=5 meshlab_mini.pro $QMAKE_FLAGS && make $MAKE_FLAGS


#fixing the trimesh version
RUN pip3 install --upgrade --force-reinstall trimesh==3.8.1

#creating starting folder as the root
WORKDIR /


ADD ./jupyter/run_jupyter_unix.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter_unix.sh"]
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

#enabling the visualization of widgets and ipyvolume
RUN apt update
RUN apt install curl -y
RUN npm install n -g
RUN n stable
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager
RUN jupyter labextension install ipyvolume

#creating starting folder as the root
WORKDIR /


ADD ./jupyter/run_jupyter_unix.sh /scripts/
ADD ./jupyter/jupyter_notebook_config.py /root/.jupyter/
ADD ./jupyter/custom.css /root/.jupyter/custom/
RUN chmod -R a+x /scripts
ENTRYPOINT ["/scripts/run_jupyter_unix.sh"]
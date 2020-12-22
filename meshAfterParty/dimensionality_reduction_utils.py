import matplotlib.pyplot as plt
import numpy as np
from sklearn import decomposition
from matplotlib.ticker import MaxNLocator



# ------------ Problem 1 -------------- #

def plot_variance_explained(data_var,title=None, title_prefix=None):
    """
    Create a square root eigenvalue plot from pca analysis
    
    """
    plt.clf()
    if title is None:
        title = "Percent Variance Explained vs PC Index" 
    if not title_prefix is None:
        title = title_prefix + ": " + title
        
    fig,ax = plt.subplots(1,1)
    
    if type(data_var) == dict:
        data_var = data_var["percent_variance_explained"]
    
    ax.plot(data_var)
    ax.set_ylabel("Percent Variance Explained")
    ax.set_xlabel("PC Index")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

def plot_sq_root_eigvals(eigVals,title=None, title_prefix=None):
    """
    Create a square root eigenvalue plot from pca analysis
    
    """
    plt.clf()
    if title is None:
        title = "Eigenvalue Square Root vs PC Index" 
    if not title_prefix is None:
        title = title_prefix + ": " + title
        
    fig,ax = plt.subplots(1,1)
    
    square_root_eigenvalues = np.sqrt(eigVals)
    ax.plot(square_root_eigenvalues)
    ax.set_ylabel("Square Root of ith PC eigenvalue")
    ax.set_xlabel("PC Index")
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

def data_mean(data):
    return np.mean(data,axis=0).reshape(1,data.shape[-1])

def data_covariance(data):
    #1) Get Covariance of data
    data_zero_mean = data - data_mean(data)
    mat_mul = data_zero_mean.T@(data_zero_mean)
    cov_matrix = mat_mul*(1/(data.shape[0]-1)) #using the population cov
    return cov_matrix
    
def eigen_decomp(data=None,cov_matrix=None):
    if cov_matrix is None:
        cov_matrix = data_covariance(data)
    
    #eigenvectors are stored as column vectors
    eigenValues,eigenVectors = np.linalg.eig(cov_matrix)

    #3) Sort eigenvectors by the eigenvalues
    eigenValues_sorted_idx = eigenValues.argsort()[::-1]
    eigenValues_sorted = eigenValues[eigenValues_sorted_idx] 
    eigenVectors_sorted = eigenVectors[:,eigenValues_sorted_idx].T
    return eigenValues_sorted,eigenVectors_sorted
    
def explained_variance(data=None,return_cumulative=True,
                       eigenValues_sorted=None):
    
    if eigenValues_sorted is None:
        eigenValues_sorted,eigenVectors_sorted = eigen_decomp(data)
    

    explained_variance_by_comp = eigenValues_sorted/np.sum(eigenValues_sorted)

    if return_cumulative:
        explained_variance_up_to_M_comp = np.cumsum(eigenValues_sorted)/np.sum(eigenValues_sorted)
        return explained_variance_by_comp,explained_variance_up_to_M_comp
    else:
        return explained_variance_by_comp
    

def projected_and_backprojected_data(data,
                                eigenVectors_sorted_filt):
#     if eigenVectors_sorted_filt is None:
#         eigenValues_sorted_filt,eigenVectors_sorted_filt = eigen_decomp(data)
    
    data_zero_mean = data - data_mean(data)
    proj_data = data_zero_mean @ eigenVectors_sorted_filt.T
    
    back_projection = ((eigenVectors_sorted_filt.T)@proj_data.T).T + data_mean(data)
    
    return proj_data,back_projection
    
def pca_analysis(data,
                 n_components = None,
                 whiten=False,
                 method="sklearn",
                 plot_sqrt_eigvals=True,
                 **kwargs):
    """
    Arguments:
    - data, #where each data point is stored as a column vector
    - method: whether performed with sklearn or by manual implmenetation
    - n_components: the number of principal components to anlayze
    - plot_sqrt_eigvals: whether to plot the square root of eigenvalues at end
    
    purpose: Will compute the following parts of PCA analysis
    
    mean
    covaraince_matrix
    eigenvalues (the variance explained)
    eigenvectors (the principal of components), as row vectors
    percent_variance_explained
    percent_variance_explained_up_to_n_comp
    data_proj = data projected onto n_components PC
    data_backproj = the data points projected into PC space reprojected 
                    back into the original R^N space 
                    (but may be with reduced components use for reconstruction)
    
    Ex:
    
    #pracitice on iris data
    from sklearn import datasets
    iris = datasets.load_iris()
    test_data = iris.data
    
    pca_dict_sklearn = pca_analysis(data=test_data,
                                    n_components = 3,
                                     method="sklearn")
                                     
    pca_dict_manual = pca_analysis(data=test_data,
                    n_components = 3,
                     method="manual")

    """
    
    if n_components is None:
        n_components = data.shape[-1]
    
    print(f"n_components = {n_components}")
    print(f"whiten = {whiten}")
    
    pca_dict = dict()
    if method == "sklearn":
        print("---- Using sklearn method ----")
        #perform the pca decomposition
        pca = decomposition.PCA(n_components=n_components,whiten=whiten)
        pca.fit(data)
        data_proj = pca.transform(data)
        data_backproj = pca.inverse_transform(data_proj)
        
        pca_dict["mean"] = pca.mean_
        pca_dict["covaraince_matrix"] = pca.get_covariance()
        pca_dict["eigenVectors"] = pca.components_
        pca_dict["eigenValues"] = pca.explained_variance_
        pca_dict["percent_variance_explained"] = pca.explained_variance_ratio_
        pca_dict["percent_variance_explained_up_to_n_comp"] = np.cumsum(pca_dict["percent_variance_explained"])
        
        pca_dict["data_proj"] = data_proj
        pca_dict["data_backproj"] = data_backproj
        pca_dict["pca_obj"] = pca
        pca_dict["data"] = data
    
    
    elif method=="manual":
        print("---- Using manual method ----")
        #Get Covariance of and mean of data
        pca_dict["mean"] = data_mean(data)
        pca_dict["covaraince_matrix"] = data_covariance(data)
        
        #calculate the eigenvalues and explained variance
        (eigenValues,
         eigenVectors) = eigen_decomp(
                                        cov_matrix=pca_dict["covaraince_matrix"]
                                      )
        
        
        (explained_variance_by_comp,
         explained_variance_up_to_M_comp) = explained_variance(
                                            eigenValues_sorted=eigenValues
                                                            )
        
        
         
        # restrict the data to only n_components PC
        eigenValues_sorted_filt = eigenValues[:n_components]
        eigenVectors_sorted_filt = eigenVectors[:n_components]
        explained_variance_up_to_M_comp_filt = explained_variance_up_to_M_comp[:n_components]
        explained_variance_by_comp_filt = explained_variance_by_comp[:n_components]
        
        #store the values in the dictionary
        pca_dict["eigenVectors"] = eigenVectors_sorted_filt
        pca_dict["eigenValues"] = eigenValues_sorted_filt
        pca_dict["percent_variance_explained"] = explained_variance_by_comp_filt
        pca_dict["percent_variance_explained_up_to_n_comp"] = explained_variance_up_to_M_comp_filt
        
        
        #find the projected and backprojected data
        pca_dict["data_proj"],pca_dict["data_backproj"] = projected_and_backprojected_data(data,
                                eigenVectors_sorted_filt=pca_dict["eigenVectors"])
        
        
    else:
        raise Exception("Method not implemented")

    if plot_sqrt_eigvals:
        plot_sq_root_eigvals(pca_dict["eigenValues"],**kwargs)

    return pca_dict



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot_projected_data(data_proj,labels):
    """
    To plot the PC projection in 3D
    
    """

    X_proj = data_proj
    y = labels

    fig = plt.figure(1, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2], 
               c=y, cmap=plt.cm.nipy_spectral,
               edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")

    plt.show()
    

import matplotlib.pyplot as plt
def plot_um(UM,height=8,width=4,title="Imshow of the UM matrix"):
    fig,ax = plt.subplots(1,1)
    fig.set_size_inches(width,height)
    c = ax.imshow(UM,interpolation='nearest',aspect='auto')

    plt.colorbar(c)
    plt.title(title)
    plt.show()
    
    
    
# ------------ Problem 2 -------------- #
def plot_top_2_PC_and_mean_waveform(data,
                                    spikewaves_pca=None,
                                    title="Waveforms for Mean waveform and top 2 PC",
                                    title_prefix=None,
                                   scale_by_sq_root_eigVal=True,
                                   return_spikewaves_pca=False,
                                   mean_scale=1):
    if not title_prefix is None:
        title = f"{title_prefix} : {title}"
    
    if spikewaves_pca is None:
        spikewaves_pca = pca_analysis(data=data,
                                 n_components=2,
                method='sklearn',
                plot_sqrt_eigvals=False)
    
    plt.clf()
    #PC1,PC2 = spikewaves_pca["eigenVectors"]
    if scale_by_sq_root_eigVal:
        PC1,PC2 = spikewaves_pca["eigenVectors"]*np.sqrt(spikewaves_pca["eigenValues"].reshape(2,-1))
    else:
        PC1,PC2 = spikewaves_pca["eigenVectors"]

    mean_waveform = np.mean(data,axis=0)*mean_scale
    time_samples = np.arange(len(PC1))

    fig,ax = plt.subplots(1,1)
    labels=["PC1","PC2","mean"]
    colors = ["r","g","k"]
    waveforms = [PC1,PC2 ,mean_waveform]

    for j,wv in enumerate(waveforms):
        ax.plot(time_samples,wv,c=colors[j],label=labels[j])

    ax.set_title(title)
    ax.set_ylabel(r"Voltage ($ \mu V$)")
    ax.set_xlabel("Time Samples")
    ax.legend()
    
    if return_spikewaves_pca:
        return spikewaves_pca
    
    
def compute_total_variance(data):
    curr_data_mean = np.mean(data,axis=0)
    total_var = (1/(len(data)-1))*np.sum([np.linalg.norm(xi-curr_data_mean) for xi in data])
    return total_var

def fraction_of_variance_after_proj_back_proj():
    pass
    
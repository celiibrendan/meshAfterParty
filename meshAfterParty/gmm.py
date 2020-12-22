from sklearn import mixture

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import time
from collections import Counter
import copy

import matplotlib_utils as mu
import pandas_utils as pu
import dimensionality_reduction_utils as dr

from os import sys
sys.path.append("/notebooks/Neurosignal_Final/PRML/")
from prml.rv import VariationalGaussianMixture


# =------------- Functions to Help with Plotting -------------- #

def plot_BIC_and_Likelihood(gmm_data,
                       fig_width=12,
                       fig_height=5,
                           title_suffix = ""):
    """
    Pupose
    
    """
    cluster_values = list(gmm_data.keys())
    cluster_labels = "Number of Clusters"
    
    
    fig1,_ = plt.subplots(1,2)

    fig1 = mu.plot_graph(
        title = "Average Log Likelihood Of Data vs. Number of Clusters\n" + title_suffix,
        y_values = [gmm_data[K]["log_likelihood"] for K in cluster_values],
        x_values = cluster_values,
        x_axis_label = cluster_labels,
        y_axis_label = "Average Log Likelihood of Data (Per Sample)",
        figure = fig1,
        ax_index = 0,
    )


    fig1 = mu.plot_graph(
        title = "BIC vs. Number of Clusters\n" + title_suffix,
        y_values = [gmm_data[K]["bic_value"] for K in cluster_values],
        x_values = cluster_values,
        x_axis_label = cluster_labels,
        y_axis_label = "BIC (Bayesian Information Criterion)",
        figure = fig1,
        ax_index = 1
    )
    
    fig1.set_tight_layout(True)
    fig1.set_size_inches(fig_width, fig_height)
    return fig1



# ----------------- Functions for analysis ------------------- #
def gmm_analysis(X_train,
                possible_K = list(range(2,8)),
                 reg_covar = 0.00001,
                init_params = "kmeans",
                covariance_type = "full",
                 pca_obj = None,
                 scaler_obj = None,
                 column_titles = None,
                 model_type="mixture", #other type is "variational"
                 verbose=True
                ):

    """
    Purpose: Will perform gmm analysis for a specified different
    number of clusters and save the models and relevant data for further analysis
    
    """
    if column_titles is None:
        columns_picked = X_train.columns
    else:
        columns_picked = column_titles
        
        
    if type(X_train) == pd.DataFrame:
        X_train = X_train.to_numpy()
        

    scaled_original_results = dict()
    
    

    for K in possible_K:
        if verbose:
            print(f"\n\n------Working on clusters K={K}-----")
        st_time = time.time()
        scaled_original_results[K] = dict()

        reg_covar_local = copy.copy(reg_covar)
        #1) Training the GMM
        while reg_covar_local <= 0.1:
            try:
                if model_type == "mixture":
                    if verbose:
                        print("Using mixture model")
                    gmm = mixture.GaussianMixture(n_components=K, 
                                                  covariance_type=covariance_type,
                                                 reg_covar=reg_covar,
                                                 init_params=init_params)
                elif model_type == "variational":
                    if verbose:
                        print("Using variational model")
                        
                    gmm = VariationalGaussianMixture(n_components=K)
                else:
                    print("Not right model")
                    raise Exception(f"The gmm model was not picked as mixture or variational : {model_type}")
                
                
                gmm.fit(X_train)
                
            except Exception as e:
                print(f"Exception occured = {str(e)}")
                print(f"Errored on gmm for reg_cov = {reg_covar_local}")
                reg_covar_local = reg_covar_local*10
            else:
                break

        if reg_covar_local >= 1:
            raise Exception(f"No gmm converged and reg_cov was {reg_covar_local}")

        if model_type == "mixture":
            bic_value = gmm.bic(X_train)
            average_log_likelihood_train = gmm.score(X_train)
            current_means = gmm.means_
        else:
            bic_value = 0
            average_log_likelihood_train = 0
            current_means = gmm.mu

        

        # Getting the Average Log likelihood:
        
        scaled_original_results[K]["model"] = gmm
        scaled_original_results[K]["log_likelihood"] = average_log_likelihood_train
        scaled_original_results[K]["bic_value"] = bic_value
        scaled_original_results[K]["reg_covar"] = reg_covar

        
        
        if not pca_obj is None:
            if verbose:
                print("reversing the pca transformation")
            current_means = pca_obj.inverse_transform(current_means)
        
        if not scaler_obj is None:
            if verbose:
                print("reversing the normalizing transformation")
            current_means = scaler_obj.inverse_transform(current_means)
        
        recovered_means = pd.DataFrame(current_means)
        recovered_means.columns = columns_picked

        scaled_original_results[K]["recovered_means"] = recovered_means
        
        if verbose:
            if model_type == "mixture":
                print(f"Convergence status = {gmm.converged_}")
            print(f"Total time for GMM = {time.time() - st_time}")

    return scaled_original_results

def gmm_classification(gmm_model,curr_data,
                       classification="hard",
                       verbose=True,
                       return_counts=True,
                      ):
    """
    Purpose: Will use the gaussian model passed to 
    classify the data points as to which 
    cluster they belong
    
    """
    if type(gmm_model) == mixture.GaussianMixture:
        probs = gmm_model.predict_proba(curr_data)
    elif type(gmm_model) == VariationalGaussianMixture:
        probs = gmm_model.classify_proba(curr_data.to_numpy())
    else:
        raise Exception(f"The gmm model was not a mixture or Variational model: {type(gmm_model)}")
    
    
    if classification == "soft":
        count_values = np.sum(probs,axis=0)
    elif classification == "hard":
        gmm_class = np.argmax(probs,axis=1)
        counter_obj = Counter(gmm_class)
        count_values = []
        for clust_idx in range(gmm_model.n_components):
            if clust_idx in counter_obj.keys():
                count_values.append(counter_obj[clust_idx])
            else:
                count_values.append(0)
        count_values = np.array(count_values)
    if verbose:
        sorted_cluster_values = np.flip(np.argsort(count_values))
        print(f"Classification: {dict([(k,np.round(count_values[k],2)) for k in sorted_cluster_values])}")
    
    return count_values


def category_classifications(model,labeled_data,
                                       return_dataframe=True,
                                       verbose = False,
                                       classification_types = ["hard","soft"]):
    total_hard = []
    total_soft = []


    labeled_data_classification = dict()
    dicts_for_classif_df = []

    for c_type in classification_types:

        if verbose:
            print(f"\nclassification_type={c_type}")
        labeled_data_classification[c_type]=dict()

        for k,v in labeled_data.items():

            if verbose:
                print(f"{k}")

            curr_class = gmm_classification(model,v,classification=c_type,verbose=verbose)
            labeled_data_classification[c_type][k] = curr_class

            classifier_dict = dict()
            classifier_dict["classification"]=c_type
            classifier_dict["category"]=k
            classifier_dict["n_clusters"] = model.n_components
            classif_dict_up = dict([(f"cl_{i}",np.round(bb,1)) for i,bb in enumerate(curr_class)])
            classifier_dict.update(classif_dict_up)

            dicts_for_classif_df.append(classifier_dict)

    if return_dataframe:
        # Print out the classification Numbers in Easy to See Dataframe
        df_class = pd.DataFrame.from_dict(dicts_for_classif_df)
        df_class.style.set_caption(f"Clustering Numbers By Neuroscience Category for K = {model.n_components}")
        df_class = df_class.sort_values(by=['category'])
        #print(df_class.to_markdown())
        
        return labeled_data_classification,df_class
    else:
        return labeled_data_classification




def clustering_stats(data,clust_perc=0.80):
    """
    Will computer different statistics about the clusters 
    formed that will be later shown or plotting 
    
    
    Metrics: For each category and classification type
    1) highest_cluster identify
    2) highest_cluster_percentage
    3) n clusters needed to encompass clust_perc % of the category
    4) Purity statistic
    
    
    """
    # categories = ["Apical","Basal","Axon"]
    # classifications = ["hard","soft"]
    # clust_perc = 0.8

    classifications = list(data.keys())
    categories = list(data[classifications[0]].keys())

    stats_dict_by_classification = dict()

    for curr_classification in classifications:
        stats_dict = dict()

        total_per_cluster_by_category = [data[curr_classification][c] for c in categories]
        total_per_cluster = np.sum(total_per_cluster_by_category,axis=0)
        for curr_category in categories:
            local_stats_dict = dict()

            count_data = data[curr_classification][curr_category]



            """
            Statistics to find:
            1) The cluster with the most of that label and the % in that cluster
            2) The number of clusters needed to comprise 80% of labeled group
            3) The purity measurements

            Pseudocode: 
            1) get the total number items put in each cluster across all categories
            2) For each cluster:
            a. Multiply the perc in that cluster * (curent number in that cluster/total number in that cluster)


            """


            sorted_labels = np.flip(np.argsort(count_data))
            highest_cluster_perc = count_data[sorted_labels[0]]/np.sum(count_data)

            local_stats_dict["highest_cluster"]  = sorted_labels[0]
            local_stats_dict["highest_cluster_perc"] = highest_cluster_perc


            sorted_labels_cumsum_perc = np.cumsum(count_data[sorted_labels]/np.sum(count_data))
            perc_per_cluster = count_data/np.sum(count_data)

            n_clusters = np.digitize(clust_perc,sorted_labels_cumsum_perc)+1
            local_stats_dict[f"n_clusters_{np.floor(clust_perc*100)}"] = n_clusters

            #find the purity metric

            purity = np.sum(perc_per_cluster[total_per_cluster != 0]*count_data[total_per_cluster != 0]/total_per_cluster[total_per_cluster != 0])

            local_stats_dict["purity"] = purity

            stats_dict[curr_category] = local_stats_dict

        # measure the purity of each cluster
        max_per_cluster = cluster_purity = np.max(total_per_cluster_by_category,axis=0)
        cluster_purity = [m/t_c if t_c > 0 else 0 for m,t_c in zip(max_per_cluster,total_per_cluster)]
        
        stats_dict_by_classification[curr_classification] = dict(cluster_purity= cluster_purity,stats_dict=stats_dict)
    return stats_dict_by_classification



def cluster_stats_dataframe(labeled_data_classification):
    """
    Purpose: Just want to visualize the soft and the hard assignment (and show they are not that different)

    Pseudocode: 
    1) 

    """

    ret_stats = clustering_stats(labeled_data_classification)

    dict_for_df = [] 

    for cl_type,cl_data in ret_stats.items():
        k = len(cl_data["cluster_purity"])
        curr_stats_dict = cl_data["stats_dict"]

        for cat_name,cat_stats_dict in curr_stats_dict.items():
            cat_local_dict = dict()
            cat_local_dict["category"] = cat_name
            cat_local_dict["classification"] = cl_type
            cat_local_dict["n_clusters"] = k
            cat_local_dict.update(cat_stats_dict)
            
            dict_for_df.append(cat_local_dict)
            
    df = pd.DataFrame.from_dict(dict_for_df)
    return df.sort_values(by=['category'])



def plot_advanced_stats_per_k(advanced_stats_per_k,
                             stats_to_plot = ["highest_cluster_perc","purity"],
                              title_suffix="",
                             fig_width = 12,
                              fig_height = 5):
    """
    Purpose: plotting the highest cluster and purity as a function of k

    Pseudocode: 
    0) Get all the possible categories, n_clusters
    0) Sort by n_clusters
    1) Iterate through all the stats we want to plot
        2) Iterate through all of the categories
            -- for all n_clusters
            a. Restrict by category and n_clusters and pull down the statistic
            b. Add to list
            --
            c. Save full list in dictionary

        3) Plot the stat using the category dictionary (using the ax index id)




    """



    advanced_stats_df = pd.concat(list(advanced_stats_per_k.values()))

    unique_categories = np.unique(advanced_stats_df["category"].to_numpy())
    unique_n_clusters = np.unique(advanced_stats_df["n_clusters"].to_numpy())
    stats_to_plot = ["highest_cluster_perc","purity"]

    cluster_labels = "Number of Clusters"

    fig, _ = plt.subplots(1,len(stats_to_plot))

    for j,st in enumerate(stats_to_plot):
        st_cat_dict = dict()
        for cat in unique_categories:
            cat_list = []
            for k in unique_n_clusters:
                curr_st = advanced_stats_df.query(f"category=='{cat}' & n_clusters=={k}")[st].to_numpy()
                if len(curr_st) != 1:
                    raise Exception("Stat was not of size 1")
                cat_list.append(curr_st[0])
            st_cat_dict[cat] = cat_list

            fig = mu.plot_graph(
                title = f"{st} vs. Number of Clusters\n" + title_suffix,
                y_values = cat_list,
                x_values = unique_n_clusters,
                x_axis_label = cluster_labels,
                y_axis_label = f"{st}",
                figure = fig,
                ax_index = j,
                label=cat
            )
    fig.set_tight_layout(True)
    fig.set_size_inches(fig_width, fig_height)
    return fig



def gmm_pipeline(df,
                title_suffix,
                labeled_data_indices, 
                 columns_picked=None,
                 possible_K = list(range(2,8)),
                 print_tables = None, #clusters will print the clustering tables fro
                 apply_normalization=True,
                 apply_pca = True,
                 pca_whiten=True,
                 plot_sqrt_eigvals=True,
                 n_components_pca = None,
                 classification_types = ["hard"],#["hard","soft"]
                 model_type = "mixture",
                 verbose=True,
                 
                ):
    """
    Will carry out all of the clustering analysis and
    advanced stats analysis on a given dataset
    
    Arguments: 
    A data table with all of the labeled data
    """
    # ------- Initializing variables --------- #

    if print_tables is None:
        print_tables = possible_K
        
    # -------- Part 0: Preprocessing (Column restriction, Normalization, PCA) ----------- #
    if verbose:
        print(f"# -------- Part 0: Preprocessing (Column restriction, Normalization, PCA) ----------- #")
    
    if columns_picked is None:
        columns_picked = list(df.columns) 
    else:
        if verbose:
            print(f"Restricting to columns : {columns_picked}")
        df = df[columns_picked]
    
    
    # Scaling the Data
    if apply_normalization:
        if verbose:
            print(f"Applying Normalization")
            
        scaler_obj = StandardScaler()
        df_data_scaled = scaler_obj.fit_transform(df)

        #df_data_reversed = scaler.inverse_transform(df_data_scaled,copy=True)

        data_df_normalized = pd.DataFrame(df_data_scaled)
        #add on the columns
        data_df_normalized.columns = df.columns
        df = data_df_normalized
    else:
        scaler_obj = None
        
    # Applying pca to the data
    if apply_pca:
        
        if n_components_pca is None:
            n_components_pca=len(columns_picked)
            
        if verbose:
            print(f"Applying pca with {n_components_pca} components")
            
        data_analyzed = dr.pca_analysis(df.to_numpy(),
                                    n_components=n_components_pca,
                                    whiten=pca_whiten,
                                    plot_sqrt_eigvals=plot_sqrt_eigvals)
    
        if verbose:
            print(f'Explained Variance = {data_analyzed["percent_variance_explained_up_to_n_comp"]}')
            dr.plot_variance_explained(data_analyzed)
            
        df_pca = pd.DataFrame(data_analyzed["data_proj"])
        df_pca.columns = [f"PC_{j}" for j in range(n_components_pca)]
        df = df_pca
        
        pca_obj = data_analyzed["pca_obj"]
            
    else:
        pca_obj = None
    

    
    # -------- Part 1: GMM clustering with different Number of Clusters ----------- # 
    if verbose:
        print(f"# -------- Part 1: GMM clustering with different Number of Clusters ----------- # ")
    
    X_train = df
    scaled_original_results = gmm.gmm_analysis(X_train,
                    scaler_obj=scaler_obj,
                    pca_obj=pca_obj,
                     possible_K = possible_K,
                    column_titles=columns_picked,
                    model_type = model_type)
    
    if model_type == "mixture":
        fig1 = gmm.plot_BIC_and_Likelihood(scaled_original_results,title_suffix=title_suffix)
        mu.display_figure(fig1)

    
    
    # --------- Part 2: computing the advanced statistics on the clustering ------- #
    if verbose:
        print(f"# --------- Part 2: computing the advanced statistics on the clustering ------- # ")
    
    advanced_stats_per_k = dict()
    labeled_data = dict([(kk,df.iloc[vv]) for kk,vv in labeled_data_indices.items()])
    
    for curr_K in scaled_original_results.keys():
        if verbose:
            print(f"\n\n----Working on Advanced Statistics for n_clusters = {curr_K}----\n")

        model = scaled_original_results[curr_K]["model"]

        

        labeled_data_classification,df_class=gmm.category_classifications(
                                    model,
                                    labeled_data,
                                    classification_types=classification_types)
        if curr_K in print_tables:
            print("Recovered Means From Clustering")
            pu.display(scaled_original_results[curr_K]["recovered_means"])
            print("\n")
            print(f"Clustering Numbers By Neuroscience Category for K = {model.n_components}")
            pu.display_df(df_class)
            print("\n")
            
            
            
        

        cl_stats_df = gmm.cluster_stats_dataframe(labeled_data_classification)

        if curr_K in print_tables:
            print(f"Clustering Advanced Statistics By Neuroscience Category for K = {curr_K}")
            pu.display_df(cl_stats_df)

        column_restriction = ["category","highest_cluster_perc","purity","n_clusters"]
        cl_stats_restricted = cl_stats_df.query("classification=='hard'")[column_restriction]

        advanced_stats_per_k[curr_K] = cl_stats_restricted
        
    # -------- Part 3: Plotting the Advanced Cluster Statistics -------------- #
    if verbose:
        print(f"# -------- Part 3: Plotting the Advanced Cluster Statistics -------------- # ")
    fig_current = gmm.plot_advanced_stats_per_k(advanced_stats_per_k,title_suffix=title_suffix)
    mu.display_figure(fig_current)
    
    
import gmm
# import packages
from fuzzywuzzy import process, fuzz
from itertools import repeat
import miceforest as mf
import pandas as pd
import numpy as np
import gower
import re
import os

# define travel recommendation function
def get_recommendations(kernel, data_sets, user_destinations, similar="Y", recs=5):
    """
    A function which generates recommendations for travel destinations based on user input
    using Gower's distance and voting ensemble principles for aggregating results across
    imputed data sets.


    Parameters
    ----------
    kernel (miceforest.ImputationKernel) : kernel dataset created by the miceforest package 
    data_sets (int) : count of data sets created through MICE and stored in kernel
    user_destinations (np.ndarray) : array of user provided destinations
    similar (str) : 'Y' or 'N' indicating user wants similar or dissimilar recommendations
    recs (int) : number of recommendations wanted by the user
    """
    
    num_destinations = len(user_destinations)
    recommendations = np.array(list())

    for data in range(data_sets):
        imputed_df = kernel.complete_data(data)
        
        # get the df indices of the destinations provided by the user
        user_dest_idx = np.array([imputed_df.index.get_loc(loc) for loc in user_destinations])
        
        # get the computed distances for the provided destinations
        dist = gower.gower_matrix(np.asarray(imputed_df))[user_dest_idx, :]
        
        # get the indices of the user provided destinations which should not be recommended again
        excl_dest = (dist == 0).sum(axis=0).nonzero()[0]
        
        # compute the sum of the distances for the set of provided destinations
        dist_sum = dist.sum(axis=0)
        
        if similar == "Y":
            # retrieve the top n most similar destinations - minimize the sum of distances across the provided locations
            temp_recs = np.argpartition(dist.sum(axis=0), (recs+3))[:(recs+3)]
        else:
            # retrieve the top n least similar destinations - maximize the sum of distances across the provided locations
            temp_recs = np.argpartition(dist.sum(axis=0), -(recs+3))[-(recs+3):]
        
        # exclude any of the user provided destinations
        for loc in excl_dest:
            temp_recs = temp_recs[temp_recs != loc]
        
        # provide the final list of recommendations that will be output to the user
        final_recs = temp_recs[:recs]
        final_recs = np.array(imputed_df.index[final_recs])
    
        # aggregate recos across data sets
        recommendations = np.append(recommendations, final_recs)
    
    # select the most frequently recommended locations across data sets
    recommendations = np.unique(recommendations, return_counts=True)
    recommendations = recommendations[0][np.argsort(recommendations[1])[-recs:]]

    # return list of top n recommendations based on countries provided and similarity/dissimilarity 
    return recommendations


def print_recommendations(recommendations, similar="Y", recs=5):
    """
    A function which outputs the results provided by recommender system.
    
    Parameters
    ----------
    recommendations (np.ndarray) : array of recommended destinations
    similar (str) : 'Y' or 'N' indicating user wants similar or dissimilar recommendations
    recs (int) : number of recommendations wanted by the user
    """
    
    if similar == "Y":
        similar = "similarity"
    else:
        similar = "dissimilarity"
    
    print(f"Thanks for your patience, here are {recs} recommendations based on {similar}:")
    for i, loc in enumerate(recommendations):
        print(f"{i+1}. {loc}")

        
# main function that houses the body of the script
def main():
    # set path for loading aggregated data and saved kernel
    aggregate_path = "Aggregate_Destination_Data.xlsx"
    kernel_path = "mice_kernel"
    
    # load saved data and kernel, get count of imputed data sets 
    aggregate_data = pd.read_excel(aggregate_path, index_col=0)
    kernel = mf.load_kernel(kernel_path)
    datasets = kernel.dataset_count()


    # get user input
    
    # TODO: Delete example user input
    user_destinations = np.array(["Cuba", "Mexico"])  # user provided destinations
    num_destinations = len(user_destinations)
    similar = "Y"  # user desires similar recommendations
    recs = 5  # number of recommendations desired by the user

    # get recommendations for user
    recommendations = get_recommendations(
        kernel,
        datasets,
        user_destinations,
        similar, 
        recs
    ) 

    # output recommendations to user
    print_recommendations(recommendations, similar, recs)

# run program function
main()
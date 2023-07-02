# import packages
from fuzzywuzzy import process, fuzz
from miceforest import load_kernel
from sys import exit as sysexit
from gower import gower_matrix
from itertools import repeat
from tqdm import tqdm
import pandas as pd
import numpy as np


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

    for data in tqdm(range(data_sets), desc="Loading"):
        imputed_df = kernel.complete_data(data)
        
        # get the df indices of the destinations provided by the user
        user_dest_idx = np.array([imputed_df.index.get_loc(loc) for loc in user_destinations])
        
        # get the computed distances for the provided destinations
        dist = gower_matrix(np.asarray(imputed_df))[user_dest_idx, :]
        
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


def print_recommendations(recommendations):
    """
    A function which outputs the results provided by recommender system.
    
    Parameters
    ----------
    recommendations (np.ndarray) : array of recommended destinations
    similar (str) : 'Y' or 'N' indicating user wants similar or dissimilar recommendations
    recs (int) : number of recommendations wanted by the user
    """
    
    print(f"Thanks for your patience, here are {len(recommendations)} recommended destinations based on your inputs (unordered):")
    for i, loc in enumerate(recommendations):
        print(f"{i+1}. {loc}")

        
# run main program
if __name__ == "__main__":
    # set path for loading aggregated data and saved kernel
    kernel_path = "mice_kernel"
    
    # load saved data and kernel, get count of imputed data sets and original data 
    kernel = load_kernel(kernel_path)  # load kernel
    datasets = kernel.dataset_count()  # get number of imputed data sets
    destinations = kernel.working_data.index.values  # retrieve master list of destinations

    # get user input
    
    # # TODO: Delete example user input
    # user_destinations = np.array(["Canada", "Madagascar", "France"])  # user provided destinations
    # num_destinations = len(user_destinations)
    # similar = "Y"  # user desires similar recommendations
    # recs = 5  # number of recommendations desired by the user

    welcome_msg = "Thanks for using this travel recommendation system. To get started, please list up to three countries you have enjoyed travelling to or think you would (hit enter to submit your entries or skip a country).\n"
    print(welcome_msg)

    
    
    # check if we have data on the countries and validate input
    countries_valid = False
    while not countries_valid:
        # prompt user for countries
        country1 = input("Country 1: ")
        country2 = input("Country 2: ")
        country3 = input("Country 3: ")

        # check if all entries are empty
        unmatched_destinations = [country1, country2, country3]
        empty_country_count = unmatched_destinations.count("")     
        if empty_country_count == 3:
            # if all empty, prompt to quit
            print("We need at least one country to provide a recommendation to you, but the more the better.")
            exit_status = input("Would you like to try again? (Y/N): ")
            if exit_status.lower() != "y":
                # exit program
                sysexit("Thanks for giving this a try, cheers!")
            else:
                print("")
                # allow user to re-enter countries
                continue

        unmatched_destinations = np.asarray(unmatched_destinations)
        unmatched_destinations = unmatched_destinations[unmatched_destinations != ""]
        # perform fuzzy matching on provided countries 
        matched_destinations = list(map(
            process.extractOne,  # function to be applied
            unmatched_destinations,  # applied to 
            repeat(destinations),  # compared against
            repeat(process.default_processor),  # use default processor each time
            repeat(fuzz.token_sort_ratio),  # use token sort ratio instead of default
            repeat(75)  # similarity must be higher than 75% to be considered a match
        ))
        matched_destinations = np.array([None if match == None else match[0] for match in matched_destinations])

        # if the count of unmatched countries is non-zero
        unmatched = unmatched_destinations[matched_destinations == None]        
        if len(unmatched) > 0:
            print("We could not find matches for the follow countries you provided:")
            for loc in unmatched:
                print(f"{loc}")

            # if number of unmatched is the same s the number of provided destinations give option to try again or exit
            if len(unmatched) == len(matched_destinations):
                exit_status = input("Would you like to try again? (Y/N): ")
                if exit_status.lower() != "y":
                    # exit program
                    sysexit("Thanks for giving this a try, cheers!")
                else:
                    # allow user to re-enter countries
                    print("")
                    continue
            # otherwise remove unmatched entries and continue
            else:
                print("We'll provide recommendations based on the countries we did match though!")
                user_destinations = matched_destinations[matched_destinations != None]
                num_destinations = len(user_destinations)
                # countries have been validated - exit loop
                countries_valid = True
        else:
            user_destinations = matched_destinations
            num_destinations = len(user_destinations)

            # countries have been validated - exit loop
            countries_valid = True

    # get user request for similar or dissimilar recommendations
    # must be Y or N, if errneous input, default to Y
    similar = input("Would you like recommendations similar to these countries? (Y/N): ").lower()
    if similar == "y" or similar == "n":
        similar = similar.upper()
    else:
        print("We were unable to capture that entry, we'll assume you want similar recommendations.")
        similar = "Y"  
    
    # get user request for number of recommendations 
    # must be an int between 1 and 25
    recs_valid = False
    while not recs_valid:
        try:
            recs = int(input("How many recommendations would you like? (1-25): "))
            if recs > 25 or recs < 1:
                print("Requested recommendations must be between 1 and 25, we'll set a default of 5 for you.")
                recs = 5
            recs_valid = True
            print("")
        except ValueError:
            print("It seems like a valid number was not provided.")
            exit_status = input("Would you like to try again? (Y/N): ")
            if exit_status.lower() != "y":
                # exit program
                sysexit("Thanks for giving this a try, cheers!")
            else:
                print("")
                # allow user to re-enter countries
                continue
     
    # get recommendations for user
    recommendations = get_recommendations(
        kernel,
        datasets,
        user_destinations,
        similar, 
        recs
    ) 

    # output recommendations to user
    print_recommendations(recommendations)

"""
A script which takes user input of travel history and provides 
meaningful recommendations for future travel destinations.

Returns
-------
None
    Returns nothing
"""

# import packages
from itertools import repeat
from sys import exit as sysexit

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import pycountry
from fuzzywuzzy import fuzz, process
from gower import gower_matrix
from miceforest import load_kernel
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper, gen_features
from tqdm import tqdm


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

    recommendations = np.array(list())
    numeric_cols = [
        [col]
        for col in kernel.working_data.select_dtypes(include=[np.number]).columns.values
    ]

    for data in tqdm(range(data_sets), desc="Loading"):
        imputed_df = kernel.complete_data(data)

        # standardize features using feature map of numerical columns to
        # ensure distance measure is not influenced by scale of features
        feature_map = gen_features(columns=numeric_cols, classes=[StandardScaler])
        mapper = DataFrameMapper(feature_map, default=None, input_df=True, df_out=True)
        imputed_df = mapper.fit_transform(imputed_df)

        # get the df indices of the destinations provided by the user
        user_dest_idx = np.array(
            [imputed_df.index.get_loc(loc) for loc in user_destinations]
        )

        # get the computed distances for the provided destinations
        dist = gower_matrix(np.asarray(imputed_df))[user_dest_idx, :]

        # get the indices of the user provided destinations which should not be recommended again
        excl_dest = (dist == 0).sum(axis=0).nonzero()[0]

        if similar == "Y":
            # retrieve the top n most similar destinations - minimize the sum of distances across the provided locations
            temp_recs = np.argpartition(dist.sum(axis=0), (recs + 3))[: (recs + 3)]
        else:
            # retrieve the top n least similar destinations - maximize the sum of distances across the provided locations
            temp_recs = np.argpartition(dist.sum(axis=0), -(recs + 3))[-(recs + 3) :]

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
    recommendations = recommendations[0][np.argsort(recommendations[1])[-recs:]]
    print(
        f"Thanks for your patience, here are {len(recommendations)} recommended destinations based on your inputs (unordered):"
    )
    for loc in recommendations:
        print(f"* {loc}")


def show_recommendations(recommendations, user_destinations, recs):
    """
    A function which visualizes the results provided by recommender system.
    Coloring the map based on the normalized recommendation frequency a
    country achieves across the multiple imputed data sets.

    Parameters
    ----------
    recommendations (np.ndarray) : array of recommended destinations
    user_destinations (np.ndarray) : array of user provided destinations
    recs (int) : number of recommendations wanted by the user
    """
    # Instantiate list of countries not mapped from origin data in pycountry module
    unmapped = {
        "Bosnia": "Bosnia and Herzegovina",
        "Brunei": "Brunei Darussalam",
        "Cape Verde": "Cabo Verde",
        "Caribbean Netherlands": "Netherlands",
        "Côte d’Ivoire": "Côte d'Ivoire",
        "Democratic Republic of Congo": "Congo, The Democratic Republic of the",
        "East Timor": "Timor-Leste",
        "Falkland Islands": "Falkland Islands (Malvinas)",
        "U.S. Virgin Islands": "Virgin Islands, U.S.",
        "Syria": "Syrian Arab Republic",
        "São Tomé and Príncipe": "Sao Tome and Principe",
        "Sint Maarten": "Sint Maarten (Dutch part)",
        "Saint Martin": "Saint Martin (French part)",
        "Saint Helena": "Saint Helena, Ascension and Tristan da Cunha",
        "Russia": "Russian Federation",
        "Micronesia": "Micronesia, Federated States of",
        "Palestine": "Palestine, State of",
        "Laos": "Lao People's Democratic Republic",
        "Kosovo": "Serbia",
        "Iran": "Iran, Islamic Republic of",
    }

    # Get ISO 3 country codes for recommended travel destinations and compute recommendation intensity
    codes = [
        pycountry.countries.lookup(unmapped[country]).alpha_3
        if country in unmapped
        else pycountry.countries.lookup(country).alpha_3
        for country in recommendations[0]
    ]
    normalized_freq = (recommendations[1] - min(recommendations[1])) / (
        max(recommendations[1]) - min(recommendations[1])
    )
    reco_freq = pd.DataFrame(
        data={
            "Country": recommendations[0],
            "ISO_3": codes,
            "Recommendation intensity": normalized_freq,
        }
    ).iloc[np.argsort(recommendations[1])[-recs:]][:]

    # Create subtitle based on the user provided countries
    subtitle = "You have visited, or have an interest in, "
    if len(user_destinations) == 1:
        subtitle += f"{user_destinations[0]}"
    elif len(user_destinations) == 2:
        subtitle += f"{user_destinations[0]} and {user_destinations[1]}"
    else:
        subtitle += f"{user_destinations[0]}, {user_destinations[1]}, and {user_destinations[2]}"

    # Build choropleth map with recommendation intensity
    fig = px.choropleth(
        reco_freq,
        locations="ISO_3",
        color="Recommendation intensity",
        range_color=[0, 1],
        hover_name="Country",
        hover_data={"ISO_3": False, "Recommendation intensity": ":.2f"},
        projection="equirectangular",
        title="Relative strength in the recommendation of each travel destination<br><sup>"
        + subtitle,
        color_continuous_scale="viridis",
    )
    fig.update_layout(
        title=dict(font=dict(size=26), x=0.015, y=0.9625),
        margin=dict(l=15, r=10, b=30, t=70),
        coloraxis_colorbar=dict(
            title="Recommendation<br>intensity",
            thicknessmode="pixels",
            lenmode="pixels",
            yanchor="top",
            y=1,
            xanchor="left",
            x=0.01,
            tickvals=[0, 0.5, 1],
            ticktext=["Low", "Medium", "High"],
        ),
    )
    fig.show()


# run main program
if __name__ == "__main__":
    pio.renderers.default = "browser"

    # set path for loading aggregated data and saved kernel
    kernel_path = "./Data/mice_kernel"

    # load saved data and kernel, get count of imputed data sets and original data
    kernel = load_kernel(kernel_path)  # load kernel
    datasets = kernel.dataset_count()  # get number of imputed data sets
    destinations = (
        kernel.working_data.index.values
    )  # retrieve master list of destinations

    # get user input
    welcome_msg = "Thanks for using this travel recommendation system. To get started, please list up to three countries you have enjoyed travelling to or think you would (hit enter to submit your entries or skip a country).\n"
    print(welcome_msg)

    # check if we have data on the countries and validate input
    # set default user destinations to None
    countries_valid = False
    user_destinations = None

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
            print(
                "We need at least one country to provide a recommendation to you, but the more the better."
            )
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
        matched_destinations = list(
            map(
                process.extractOne,  # function to be applied
                unmatched_destinations,  # applied to
                repeat(destinations),  # compared against
                repeat(process.default_processor),  # use default processor each time
                repeat(
                    fuzz.token_sort_ratio
                ),  # use token sort ratio instead of default
                repeat(
                    75
                ),  # similarity must be higher than 75% to be considered a match
            )
        )
        matched_destinations = np.array(
            [None if match == None else match[0] for match in matched_destinations]
        )

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
                print(
                    "We'll provide recommendations based on the countries we did match though!"
                )
                user_destinations = matched_destinations[matched_destinations != None]
                # countries have been validated - exit loop
                countries_valid = True
        else:
            user_destinations = matched_destinations

            # countries have been validated - exit loop
            countries_valid = True

    # get user request for similar or dissimilar recommendations
    # must be Y or N, if errneous input, default to Y
    similar = input(
        "Would you like recommendations similar to these countries? (Y/N): "
    ).lower()
    if similar == "y" or similar == "n":
        similar = similar.upper()
    else:
        print(
            "We were unable to capture that entry, we'll assume you want similar recommendations."
        )
        similar = "Y"

    # get user request for number of recommendations
    # must be an int between 1 and 25
    # set default recommendations to 5
    recs_valid = False
    recs = 5
    while not recs_valid:
        try:
            recs = int(input("How many recommendations would you like? (1-25): "))
            if recs > 25 or recs < 1:
                print(
                    "Requested recommendations must be between 1 and 25, we'll set a default of 5 for you."
                )
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
        kernel, datasets, user_destinations, similar, recs
    )

    # output recommendations to user
    print_recommendations(recommendations)
    show_recommendations(recommendations, user_destinations, recs)

"""
Author: Peter Akioyamen

A webapp which provides travel recommendations
based on user's past experiences. Built on Streamlit.
"""

# import packages
import numpy as np
import pandas as pd
import plotly.express as px
import pycountry
import streamlit as st
from gower import gower_matrix
from miceforest import load_kernel
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper, gen_features


def checkCountries():
    """
    A callback function sanity checks the country input data
    based on the dependencies between input fields
    """
    c1 = st.session_state.country1
    c2 = st.session_state.country2
    c3 = st.session_state.country3

    if c1 == c2 and c1 != None:
        st.session_state.country2 = None
    elif c1 == c3 and c3 != None:
        st.session_state.country3 = None
    elif c2 == c3 and c2 != None:
        st.session_state.country3 = None


@st.cache_resource(show_spinner=False)
def get_kernel(kernel_path):
    """
    A function which loads the kernel from memory and caches the value

    Parameters
    ----------
    kernel_path (str) : a valid file path to the file containing the serialized imputation kernel
    """

    return load_kernel(kernel_path)


@st.cache_resource(show_spinner=False)
def get_kernel_data(_kernel, datasets):
    """
    A function which extracts the imputed datasets from the kernel

    Parameters
    ----------
    _kernel (miceforest.ImputationKernel) : imputation kernel
    datasets (int) : the number of datasets imputed and stored in the kernel
    """
    return [_kernel.complete_data(data) for data in range(datasets)]


@st.cache_data(show_spinner=False, max_entries=25)
def get_recommendations(imputed_dfs, user_destinations, similar="Y", recs=5):
    """
    A function which generates recommendations for travel destinations based on user input
    using Gower's distance and voting ensemble principles for aggregating results across
    imputed data sets.


    Parameters
    ----------
    imputed_dfs (list) : list of imputed datasets in the form of pandas dataframes
    user_destinations (np.ndarray) : array of user provided destinations
    similar (str) : 'Y' or 'N' indicating user wants similar or dissimilar recommendations
    recs (int) : number of recommendations wanted by the user
    """

    recommendations = np.array(list())
    numeric_cols = [
        [col]
        for col in imputed_dfs[0].select_dtypes(include=[np.number]).columns.values
    ]

    progress_bar = st.progress(0, "Loading, in progress, please wait...")
    for i, imputed_df in enumerate(imputed_dfs):
        progress_bar.progress(
            (i + 1) * 10, f"Generating: {(i+1/len(imputed_df))*10:.0f}%"
        )

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
        elif similar == "S":
            # retrieve the n somewhat similar destinations - minimize the sum of distances across the provided locations
            temp_recs = np.argpartition(dist.sum(axis=0), (recs + 3))[28 : (28 + recs)]
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

    progress_bar.empty()

    # return list of top n recommendations based on countries provided and similarity/dissimilarity
    return recommendations


def print_recommendations(recommendations, recs):
    """
    A function which outputs the results provided by recommender system.

    Parameters
    ----------
    recommendations (np.ndarray) : array of recommended destinations
    """
    normalized_freq = (recommendations[1] - min(recommendations[1])) / (
        max(recommendations[1]) - min(recommendations[1])
    )
    reco_freq = (
        pd.DataFrame(
            data={
                "Recommended destination": recommendations[0],
                "Recommendation intensity": normalized_freq,
            }
        )
        .iloc[np.argsort(recommendations[1])[-recs:]][:]
        .sort_values(by="Recommendation intensity", ascending=False)
        .reset_index(drop=True)
    )
    reco_freq.index += 1

    st.write(f"{len(reco_freq)} recommended destinations based on your inputs:")
    st.dataframe(reco_freq, use_container_width=True)


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
        locationmode="ISO-3",
        color="Recommendation intensity",
        range_color=[0, 1],
        hover_name="Country",
        hover_data={"ISO_3": False, "Recommendation intensity": ":.2f"},
        projection="equirectangular",
        color_continuous_scale="viridis",
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title="Recommendation<br>intensity",
            tickvals=[0, 0.5, 1],
            ticktext=["Low", "Medium", "High"],
            y=0.5,
        ),
        autosize=True,
        margin=dict(b=120, t=0, l=0, r=0),
        geo=dict(showframe=False),
    )
    st.plotly_chart(fig, use_container_width=True)


# main program
def main():
    # Set layout of streamlit page
    st.set_page_config(
        layout="wide",
        page_title="Vacation recommender system",
        page_icon="🛫",
    )

    # set path for loading aggregated data and saved kernel
    kernel_path = "./Data/mice_kernel"

    # load saved data and kernel, get count of imputed data sets and original data
    kernel = get_kernel(kernel_path)  # load kernel
    datasets = kernel.dataset_count()  # get number of imputed data sets
    destinations = (
        kernel.working_data.index.values
    )  # retrieve master list of destinations
    imputed_dfs = get_kernel_data(kernel, datasets)

    # webpage markup
    st.write(
        """
        # Vacation Recommender
        **Author: Peter Akioyamen** (see [@peter-ai](https://github.com/peter-ai/travel-recommender) for the full github repo)
    
        ### Context
        This is a recommendation system which, based on your past experiences, provides useful suggestions 
        for your next travel destination. Decisions are informed by up to 3 of your past travel experiences,
        or experiences you believe you would have enjoyed if you had went; you can query the system for similar 
        locations to those you provided, somewhat similar ones if you want a bit more excitement, or completely
        dissimilar destinations if you're looking for a complete change of scenery (no pun intended), 
        outputting up to 25 suggestions for you to consider. 
        """
    )

    with st.expander("See a few details about the analysis and implementation"):
        st.write(
            """
            Data underpinning this recommendation system was scraped from Wikipedia, considering various characteristics 
            of a destination including climate, common transportation modalities, demographics, food, political 
            atmosphere, and geograpphical features, among others. In total there were ~65 features used to assess 
            similarity between countries.

            Multiple imputation was used to deal with data missingess and fuzzy-matching to address naming
            inconsistencies across data sources. In this implementation Gower's distance was used to accommodate
            the inclusion of both categorical and continuous variables. Further exploration with clustering methods 
            could yield interesting results. For a given country that is recommended, the conviction is 
            computed as the normalized recommendation frequency across imputed datasets.

            See the [github repo](https://github.com/peter-ai/travel-recommender) if you'd like to check out 
            the source code and play around with it yourself. If interested, checkout my website 
            at [peterai.me](https://peterai.me).
            """
        )

    st.divider()
    st.write(
        """
        ### Get started
        To get started, please provide **up to** three **distinct** countries you have enjoyed travelling to 
        in the past or think you would enjoy travelling to in the future.
        """
    )

    # get the country input from user
    user_destinations = None
    country_col1, country_col2, country_col3 = st.columns(3)
    country1 = country2 = country3 = None
    with country_col1:
        country1 = st.selectbox(
            label="Country 1",
            options=destinations,
            index=35,
            key="country1",
            on_change=checkCountries,
        )

    with country_col2:
        country2 = st.selectbox(
            label="Country 2",
            options=destinations,
            index=None,
            key="country2",
            on_change=checkCountries,
        )

    with country_col3:
        country3 = st.selectbox(
            label="Country 3",
            options=destinations,
            index=None,
            key="country3",
            on_change=checkCountries,
        )

    # only keep non-None entries
    countries = np.array([country1, country2, country3])
    user_destinations = countries[countries != None]

    # get user request for similar or dissimilar recommendations
    similar = st.selectbox(
        label="Would you like recommendations that are similar to the provided destinations?",
        options=["Y", "S", "N"],
        format_func=lambda x: "No"
        if x == "N"
        else "I'm Feeling Lucky"
        if x == "S"
        else "Yes",
    )

    # get number of recommendations from user
    recs = st.slider(
        label="How many recommendations would you like?",
        min_value=3,
        max_value=25,
        value=10,
        step=1,
    )

    recommendations = get_recommendations(imputed_dfs, user_destinations, similar, recs)

    # output recommendations to user
    tab1, tab2 = st.tabs(["Choropleth map", "Recommendation table"])
    with tab1:
        show_recommendations(recommendations, user_destinations, recs)
    with tab2:
        print_recommendations(recommendations, recs)


# launch streamlit server and run recommendation engine
if __name__ == "__main__":
    main()

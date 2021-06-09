import asyncio
import json
import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Immo Eliza Project", layout="wide")


@st.cache(show_spinner=False, suppress_st_warning=True, hash_funcs={st.delta_generator.DeltaGenerator: lambda x: None})
def load_list(form):
    with open("data/ziplist.pkl", "rb") as f:
        s_zip = form.selectbox("Zipcode", pickle.load(f))
    return s_zip


@st.cache(show_spinner=False, allow_output_mutation=True, suppress_st_warning=True)
def plot_map(*args, **kwargs):
    # pio.renderers.default = 'browser'
    return px.choropleth_mapbox(*args, **kwargs)


async def plot_map_async(df_vis, geojson):
    fig = plot_map(data_frame=df_vis,
                   geojson=geojson,
                   featureidkey="properties.id",
                   color_continuous_scale='balance',
                   locations="Locality",
                   color='Color',
                   mapbox_style='dark',
                   title='Real estate by locality in Belgium (may 2021)',
                   zoom=6.5,
                   opacity=0.5,
                   hover_name="Name",
                   hover_data={
                       "Count": True,
                       "Median Price": True,
                       "Mean Price": True,
                       "Mean Price / m²": True,
                       "Median Price / m²": True,
                       "Color": False
                   },
                   center={'lat': 50.5039, 'lon': 4.4699})
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


@st.cache(hash_funcs={dict: lambda x: None}, show_spinner=False)
def load():
    with open('data/postal_district.geojson') as f:
        geojson = json.load(f)
    with open("data/model.pkl", "rb") as f, open("data/coord.json", "rt", encoding="utf-8") as jf:
        model = pickle.load(f)
        coord = json.load(jf)
    df_vis = pd.read_csv("data/database_visu.csv", index_col=0)
    px.set_mapbox_access_token(os.environ.get("MAPBOXTOKEN"))
    return model, coord, geojson, df_vis


async def load_async():
    r = load()
    return r


async def main():
    title = st.title("Immo Eliza Project")
    subtitle = st.subheader("Belgium Real estate price estimation powered by AI")
    st.markdown("---")
    loader_status = st.info("Loading Data Map ...")

    st.markdown(f"**Submit the parameters on the left sidebar to estimate the "
                f"price for any house or appartment in Belgium**")
    loading_bar = st.progress(0)
    max = 34
    loaded = load_async()

    st.sidebar.markdown("# Real Estate prediction parameters")
    form = st.sidebar.form("Search Parameters")
    s_zip = load_list(form)

    sb_type = form.radio("Property Type", ("Appartment", "House"))
    le_surface = form.number_input("Living suface (m²)", min_value=9)
    sb_building_state = form.radio("Property Type", ("Good", "Renovation required", "New"))
    le_rooms = form.number_input("Amount of Rooms", min_value=1, max_value=999)
    le_facades = form.number_input("Amount of facades", min_value=1, max_value=99)
    sb_furnished = form.radio("Sold furnished", ("No", "Yes"))
    sb_fully_e_k = form.radio("Fully Equiped Kitchen", ("Yes", "No"))
    sb_open_fire = form.radio("Equipped with an open fire", ("No", "Yes"))
    sb_terrace = form.radio("Terrace", ("No", "Yes"))
    sb_terrace_area = form.number_input("Terrace surface (m²)", min_value=0)
    sb_plot_surface = form.number_input("Surface Area of the plot of land (are)", min_value=0)
    sb_garden = form.radio("Garden", ("No", "Yes"))
    sb_garden_area = form.number_input("Garden surface (are)", min_value=1)
    sb_plot_swimming_pool = form.radio("Swimming pool", ("No", "Yes"))
    submitted = form.form_submit_button("Submit")

    model, coord, geojson, df_vis = await loaded
    loading_bar.progress(9 / max)
    zipcode = s_zip
    n_rooms = le_rooms
    long, lat = coord[str(zipcode)]["lng"], coord[str(zipcode)]["lat"]
    loading_bar.progress(10 / max)
    entry = pd.DataFrame(np.zeros((1, 15)), columns=['Number of rooms', 'Area',
                                                     'Fully equipped kitchen', 'Furnished', 'Open fire',
                                                     'Terrace Area', 'Garden', 'Surface of the land',
                                                     'Number of facades',
                                                     'Swimming pool', 'Type of property_house',
                                                     'State of the building_to renovate', 'State of the building_new',
                                                     'lat', 'lng'], dtype=np.float64)
    entry["Number of rooms"] = n_rooms
    loading_bar.progress(12 / max)

    coord.keys()
    entry["Area"] = le_surface
    loading_bar.progress(13 / max)
    entry["Fully equipped kitchen"] = 1 if sb_fully_e_k == "Yes" else 0
    loading_bar.progress(14 / max)
    entry["Furnished"] = 1 if sb_furnished == "Yes" else 0
    loading_bar.progress(15 / max)
    entry["Terrace Area"] = sb_terrace_area if sb_terrace == "Yes" else 0
    loading_bar.progress(16 / max)
    entry["Open fire"] = 1 if sb_open_fire == "Yes" else 0
    loading_bar.progress(17 / max)
    entry["Garden"] = 1 if sb_garden == "Yes" else 0
    loading_bar.progress(18 / max)
    entry["Surface of the land"] = sb_plot_surface
    loading_bar.progress(19 / max)
    entry["Number of facades"] = le_facades
    loading_bar.progress(20 / max)
    entry["Swimming pool"] = 1 if sb_plot_swimming_pool == "Yes" else 0
    loading_bar.progress(21 / max)
    entry["Type of property_house"] = 1 if sb_type == "House" else 0
    loading_bar.progress(22 / max)
    entry["State of the building_to renovate"] = 1 if sb_building_state == "Renovation required" else 0
    loading_bar.progress(23 / max)
    entry["State of the building_new"] = 1 if sb_building_state == "New" else 0
    loading_bar.progress(24 / max)
    entry["lat"] = lat
    loading_bar.progress(25 / max)
    entry["lng"] = long
    loading_bar.progress(26 / max)
    prediction = str((round(model.predict(entry.values)[0], 2)))
    loading_bar.progress(27 / max)
    integer, dec = prediction.split(".")
    loading_bar.progress(28 / max)
    pred_part = ""
    for i, char in enumerate(integer[::-1]):
        if i % 3 == 0:
            pred_part += ','
        pred_part += char
    pred_int = str(pred_part[::-1])
    if pred_int.endswith(','):
        pred_int = pred_int[:-1]
    loading_bar.progress(29 / max)

    st.success(f"The estimated price for those settings is **{pred_int}.{dec} €**")
    loading_bar.progress(30 / max)
    st.markdown("The Algorithm Root Mean Squared Error is **173,887.15**, the Mean Absolute Error is **78,291.92**")
    loading_bar.progress(31 / max)

    loading_bar.progress(32 / max)
    map_plotted = plot_map_async(df_vis, geojson)
    loading_bar.progress(33 / max)
    map_plot = st.info("Loading map display ... ")
    loading_bar.progress(34 / max)
    loader_status.markdown("")
    loading_bar.empty()
    del loader_status
    del loading_bar

    st.header("About the project")
    st.markdown(
        '''
        ---
            
        <div align='justify'>
            
        ## ImmoEliza, Challenge-Regression
        
        <div align='justify'>
            
        This project was made during an AI formation at [Becode](https://becode.org/). The main goals of the project were :
        
        1. Create a new database by scrapping real-estate websites
        2. Clean and analyse the database
        3. Preprocess the data for the machine learning algorithms
        4. Find the best algorithm and search for the optimum parameters 
            
        The different repositories for each step of the projects are:
        1. [Database Creation](https://github.com/JulienAlardot/challenge-collecting-data)
        2. [Data Analysis](https://github.com/JulienAlardot/ImmoElizaVisu)
        3. [Machine learning preprocessing and Deployment (this step)](https://github.com/JulienAlardot/ImmoElisa)
        
        The challenges for this project are extracting the necessary features in a way that would be useful for the algorithms.
        To fill  the 
        gaps in the data without negatively impacting the algorithm ability to predict from new data. And to deploy the 
        algorithm and automatise the whole process to continuously update the database and increase the algorithm ability to 
        predict .
        
        The previous steps in this project were made with different teams. See individual repositories for more informations 
        about the teams.
            
        </div>
            
        ''', unsafe_allow_html=True
    )

    st.header("About the author")
    st.markdown(
        '''
        ---

        <div align='justify'>

        ## Who am I?
        
        Hi, my name is Julien Alardot and I always loved AIs. Originally a Rigger/skinner and Python tool developer in 
        the 3D CG production, I am always curious and eager to learn new things especially in python. It is no surprise 
        that I quickly started to dig in the AI and Machine Learning fields for fun.
        
        As I decided to change my career path to a more python-oriented career, AI and Machine Learning became a 
        somewhat obvious choice. But lacking really finished projects to show in my portfolio and some knowledge 
        in certain fields. I decided to take the BeCode Training Course, and here we are two month later with the first
        really "finished" ML project I can be proud of. 
        
        Of course, this is only a beginning and I aim to continue to develop this project on the side. 
        But that is a story for another time.

        ## Links
        
        [Github](https://github.com/JulienAlardot)
        [LinkedIn](https://www.linkedin.com/in/julien-alardot-84151a150/)
        
        </div>

        ''', unsafe_allow_html=True)

    fig = await map_plotted
    map_plot.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    asyncio.run(main())

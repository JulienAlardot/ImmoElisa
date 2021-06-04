import asyncio
import json
import pickle
import random

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

@st.cache(hash_funcs={dict: lambda x: 1}, show_spinner=False)
def load():
    with open('postal_district.geojson') as f:
        geojson = json.load(f)
    with open("model.pkl", "rb") as f, open("coord.json", "rt", encoding="utf-8") as jf:
        model = pickle.load(f)
        coord = json.load(jf)
    df = pd.read_csv("database.csv", index_col=0)
    df_vis = pd.read_csv("database_visu.csv", index_col=0)
    with open("tk.pkl", "rb") as t:
        token = pickle.load(t)
        px.set_mapbox_access_token(token)
    return model, df, coord, geojson, df_vis

async def load_async():
    r= load()
    return r


async def main():

    title = st.title("Immo Eliza Project")
    subtitle = st.subheader("Belgium Real estate price estimation powered by AI")
    f" "
    f" "
    loader_status = st.markdown(
        '''
        
        Data Loading ...
    
        ''')
    loading_bar = st.progress(0)
    max = 34
    loaded = load_async()

    s_zip = st.sidebar.selectbox("Zipcode", [5101])

    # sb_type = st.sidebar.selectbox("Property Type", ("Appartment", "House"))
    sb_type = st.sidebar.radio("Property Type", ("Appartment", "House"))
    le_surface = st.sidebar.number_input("Living suface (m²)", min_value=9)
    # sb_building_state = st.sidebar.selectbox("Building state", ("Good", "Renovation required", "New"))
    sb_building_state = st.sidebar.radio("Property Type", ("Good", "Renovation required", "New"))
    le_rooms = st.sidebar.number_input("Amount of Rooms",min_value=1,max_value=999)
    le_facades = st.sidebar.number_input("Amount of facades",min_value=1,max_value=99)
    sb_furnished = st.sidebar.radio("Sold furnished", ("No", "Yes"))
    sb_fully_e_k = st.sidebar.radio("Fully Equiped Kitchen", ("Yes", "No"))
    sb_open_fire = st.sidebar.radio("Equipped with an open fire", ("No", "Yes"))
    sb_terrace = st.sidebar.radio("Terrace", ("No", "Yes"))
    sb_plot_surface = st.sidebar.number_input("Surface Area of the plot of land", min_value=0)
    sb_terrace_area = st.sidebar.number_input("Terrace surface", min_value=1) if sb_terrace == "Yes" else 0
    sb_garden = st.sidebar.radio("Garden", ("No", "Yes"))
    sb_garden_area = st.sidebar.number_input("Garden surface", min_value=1) if sb_garden == "Yes" else 0
    sb_plot_swimming_pool = st.sidebar.radio("Swimming pool", ("No", "Yes"))



    model, df, coord, geojson, df_vis = await loaded
    loading_bar.progress(9 / max)
    zipcode = s_zip
    n_rooms = le_rooms

    long, lat = coord[str(zipcode)]["lng"], coord[str(zipcode)]["lat"]
    loading_bar.progress(10 / max)
    entry= pd.DataFrame(np.zeros((1, 15)), columns=['Number of rooms', 'Area',
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
    entry["Terrace Area"] = sb_terrace_area
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
    entry["State of the building_to renovate"] = 1 if sb_building_state  == "Renovation required" else 0
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
        if i%3 == 0:
            pred_part += ','
        pred_part += char
    pred_int = str(pred_part[::-1])
    if pred_int.endswith(','):
        pred_int = pred_int[:-1]
    loading_bar.progress(29 / max)




    f"The estimated price for those settings is {pred_int}.{dec} €"
    loading_bar.progress(30 / max)
    f"The Algorithm Mean Squared Error is 30,236,741,392.02,\nthe Mean Absolute Error is 78,291.92"
    loading_bar.progress(31 / max)
    # df_vis.loc[:, ["Locality", "Median Price", "Mean Price"]]
    fig = st.cache(px.choropleth_mapbox(data_frame=df_vis,
                               geojson=geojson,
                               featureidkey="properties.id",
                               color_continuous_scale='balance',
                               locations="Locality",
                               color='Median Price',
                               mapbox_style='dark',
                               title='Real estate by locality in Belgium (may 2021)',
                               zoom=6.5,
                               hover_name="Locality",
                               hover_data={
                                   "Median Price": True,
                                   "Mean Price": True,
                                   "Mean Price / m²": True,
                                   "Median Price / m²": True,
                               },
                               center={'lat': 50.5039, 'lon': 4.4699}), show_spinner=False)
    loading_bar.progress(32 / max)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    loading_bar.progress(32 / max)
    st.plotly_chart(fig, use_container_width=True)
    loading_bar.progress(34 / max)
    loader_status.markdown(
        '''
        
        Data Loaded
    
        ''')
    del loading_bar
    #
    # f"Locality: {zipcode}"
    # f"Number of rooms {n_rooms}"""
    # f"Fully Equipped Kitcheb {f_e_k}"""



if __name__ == "__main__":

    asyncio.run(main())

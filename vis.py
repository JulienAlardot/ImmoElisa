import asyncio
import json
import pickle
import random

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


# @st.cache
async def load():
    with open('postal_district.geojson') as f:
        geojson = json.load(f)
    with open("model.pkl", "rb") as f, open("coord.json", "rt", encoding="utf-8") as jf:
        model = pickle.load(f)
        coord = json.load(jf)
    df = pd.read_csv("database.csv", index_col=0)
    with open("tk.pkl", "wb") as t:
        pickle.dump('pk.eyJ1IjoianVsaWVuYWxhcmRvdCIsImEiOiJja3BoMWFseTAybXhoMnVueGdvMGNsajgzIn0.ccK_6L4EaH1AeFM-9RIvgQ', t)
    with open("tk.pkl", "rb") as t:
        token = pickle.load(t)
        px.set_mapbox_access_token(token)
    df_vis = df.copy()
    df_vis["Locality"] = np.nan
    for i, row in df_vis.iterrows():
        for zip in coord.keys():
            if coord[zip]["lat"] == df_vis.loc[i, "lat"] and coord[zip]["lng"] == df_vis.loc[i, "lng"]:
                df_vis.loc[i, "Locality"] = zip
    mean_df = df_vis.groupby(["lat","lng"]).mean()[["Price", "Locality"]].reset_index()
    med_df = df_vis.groupby(["lat","lng"]).median()[["Price", "Locality"]].reset_index()
    df_vis["Price / m²"] = df["Price"]/df["Area"]
    prsqrm_mean_df = df_vis.groupby(["lat","lng"]).mean()[["Price / m²", "Locality"]].reset_index()
    prsqrm_median_df = df_vis.groupby(["lat","lng"]).median()[["Price / m²", "Locality"]].reset_index()

    mean_df["Mean Price"] = mean_df["Price"]
    mean_df.drop(columns=["Price"], inplace=True)

    med_df["Median Price"] = med_df["Price"]
    med_df.drop(columns=["Price"], inplace=True)

    prsqrm_mean_df["Mean Price / m²"] = mean_df["Price / m²"]
    prsqrm_mean_df.drop(columns=["Price / m²"], inplace=True)

    prsqrm_median_df["Median Price / m²"] = med_df["Price / m²"]
    prsqrm_median_df.drop(columns=["Price / m²"], inplace=True)

    df_vis

    df_vis = pd.merge(df, mean_df, "left", "Locality")
    df_vis = pd.merge(df_vis, med_df, "left", "Locality")
    df_vis = pd.merge(df_vis, prsqrm_mean_df, "left", "Locality")
    df_vis = pd.merge(df_vis, prsqrm_median_df, "left", "Locality")
    df_vis.to_csv("database_vis.csv")
    return model, df, coord, geojson, df_vis

# @st.cache
async def main():
    loaded = load()
    title = st.title("Immo Eliza Project, Belgium Real estate price estimation powered with AI\n")
    f" "
    f" "
    f" "
    f" "
    loader_status = st.markdown(
        '''
        
        Data Loading ...
    
        ''')

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
    zipcode = s_zip
    n_rooms = le_rooms

    long, lat = coord[str(zipcode)]["lng"], coord[str(zipcode)]["lat"]
    entry= pd.DataFrame(np.zeros((1, 15)), columns=['Number of rooms', 'Area',
                                                    'Fully equipped kitchen', 'Furnished', 'Open fire',
                                                    'Terrace Area', 'Garden', 'Surface of the land',
                                                    'Number of facades',
                                                    'Swimming pool', 'Type of property_house',
                                                    'State of the building_to renovate', 'State of the building_new',
                                                    'lat', 'lng'], dtype=np.float64)
    entry["Number of rooms"] = n_rooms

    coord.keys()
    entry["Area"] = le_surface
    entry["Fully equipped kitchen"] = 1 if sb_fully_e_k == "Yes" else 0
    entry["Furnished"] = 1 if sb_furnished == "Yes" else 0
    entry["Terrace Area"] = sb_terrace_area
    entry["Open fire"] = 1 if sb_open_fire == "Yes" else 0
    entry["Garden"] = 1 if sb_garden == "Yes" else 0
    entry["Surface of the land"] = sb_plot_surface
    entry["Number of facades"] = le_facades
    entry["Swimming pool"] = 1 if sb_plot_swimming_pool == "Yes" else 0
    entry["Type of property_house"] = 1 if sb_type == "House" else 0
    entry["State of the building_to renovate"] = 1 if sb_building_state  == "Renovation required" else 0
    entry["State of the building_new"] = 1 if sb_building_state == "New" else 0
    entry["lat"] = lat
    entry["lng"] = long
    f"The estimated price for those settings is {model.predict(entry.values)[0]:.2f} €"
    f"The Algorithm Mean Squared Error is 30,236,741,392.01 and the Mean Absolute Error is 78,291.92"
    # df_vis.loc[:, ["Locality", "Median Price", "Mean Price"]]
    fig = px.choropleth_mapbox(data_frame=df_vis,
                               geojson=geojson,
                               featureidkey="properties.id",
                               color_continuous_scale='balance',
                               locations="Locality",
                               color='Median Price',
                               mapbox_style='dark',
                               zoom=6.5,
                               hover_name="Locality",
                               hover_data={
                                   "Median Price": True,
                                   "Mean Price": True,
                                   "Mean Price / m²": True,
                                   "Median Price / m²": True,
                               },
                               center={'lat': 50.5039, 'lon': 4.4699})
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    loader_status.markdown(
        '''
        
        Data Loaded
    
        ''')
    #
    # f"Locality: {zipcode}"
    # f"Number of rooms {n_rooms}"""
    # f"Fully Equipped Kitcheb {f_e_k}"""



if __name__ == "__main__":

    asyncio.run(main())

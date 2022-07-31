import time
from datetime import timedelta

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium, folium_static

from crome.dashboard.database.manager import fetch_and_clean_data


def plot_map(df: pd.DataFrame):
    map_center = (36.15, -86.75)
    m = folium.Map(
        location=map_center,
        control_scale=True,
        zoom_start=11, zoom_control=True,
        tiles='OpenStreetMap',
        scrollWheelZoom=True,
    )
    for idx, row in df.iterrows():
        folium.CircleMarker(
            location=(row['lat'], row['lon']),
            radius=10,
            popup='Report({:0.2f}, {:0.2f})'.format(row['lat'], row['lon']),
            color='green' if row['report_type'] == 'pred' else 'crimson',
            fill=True,
        ).add_to(m)
    folium_static(m, width=None)


def detect_incidents_view():
    """Show incident detections

    :return:
    """
    df = fetch_and_clean_data('results')
    # filter model
    model_options = df['model'].unique()
    selected_model = st.sidebar.selectbox('Select model', model_options)
    df = df[df['model'] == selected_model]
    # filter features
    feature_options = set(df['features'].unique())
    with st.sidebar.expander('Extra features'):
        disabled = {'congestion_mean+precip_mean', 'precip_mean'}.isdisjoint(feature_options)
        if disabled:
            st.session_state['value-checkbox-precip_mean'] = False
        precip_mean = st.checkbox('Precipitation', disabled=disabled, key='value-checkbox-precip_mean')
        disabled = {'congestion_mean+precip_mean', 'congestion_mean'}.isdisjoint(feature_options)
        if disabled:
            st.session_state['value-checkbox-congestion_mean'] = False
        congestion_mean = st.checkbox('Congestion', disabled=disabled, key='value-checkbox-congestion_mean')
    selected_feature = 'base'
    if precip_mean and congestion_mean:
        selected_feature = 'congestion_mean+precip_mean'
    elif congestion_mean:
        selected_feature = 'congestion_mean'
    elif precip_mean:
        selected_feature = 'precip_mean'
    df = df[df['features'] == selected_feature]  # select only one feature
    # \Delta t
    delta_t_options = np.sort(df['delta_t'].unique())
    selected_delta_t = st.sidebar.selectbox('Select Δt (mins)', delta_t_options)
    df = df[df['delta_t'] == selected_delta_t]
    # \Delta s
    delta_s_options = np.sort(df['delta_s'].unique())
    selected_delta_s = st.sidebar.selectbox('Select Δs (m)', delta_s_options)
    df = df[df['delta_s'] == selected_delta_s]
    #
    update_rate = st.sidebar.number_input('Update rate (min)', value=1, min_value=1, max_value=30, step=1)
    retain_for = st.sidebar.number_input('Retain period (min)', value=10, min_value=5, max_value=30, step=5)
    #
    start_time = df['time'].min().to_pydatetime()
    end_time = df['time'].max().to_pydatetime()
    current_time = start_time
    # current_time = st.slider('Simulation Time', value=current_time, min_value=start_time, max_value=end_time,
    #                          format='MM/DD/YY - hh:mm')
    with st.empty():
        while current_time <= end_time:
            data_df = df[(current_time - timedelta(minutes=retain_for) <= df['time']) & (df['time'] <= current_time)]
            plot_map(data_df)
            current_time = current_time + timedelta(minutes=update_rate)
            time.sleep(1)

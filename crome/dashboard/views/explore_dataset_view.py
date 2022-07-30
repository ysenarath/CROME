import branca
import folium
import streamlit as st
import plotly.express as px
from streamlit_folium import folium_static

from crome.config import config
from crome.dashboard.database.manager import fetch_and_clean_data

PROJECT_PATH = config['DEFAULT']['project_path']


def show_incidents():
    """

    :return:
    """
    colormap = branca.colormap.linear.viridis.scale(-30, 30)
    incidents_df = fetch_and_clean_data(key='incidents')
    max_dist = st.sidebar.slider('Maximum distance from incident to Waze (meters)?', 0, 1000, 100)  # in m
    min_delta_time, max_delta_time = st.sidebar.slider('Select range of time difference to show (minutes)',
                                                       -30, 30, (-30, 30))
    incidents_df = incidents_df[incidents_df['distance'] <= (max_dist / 1000)]
    incidents_df = incidents_df[incidents_df['delta_time'] <= max_delta_time]
    incidents_df = incidents_df[min_delta_time <= incidents_df['delta_time']]
    reports_df = fetch_and_clean_data(key='reports')
    options = incidents_df['nfd_id'].unique()
    nfd_id = st.sidebar.selectbox('Select the incident to show', options)
    nfd_report = reports_df[reports_df['id'] == nfd_id].iloc[0]
    incident_reports_df = incidents_df[incidents_df['nfd_id'] == nfd_id]
    incident_location = (nfd_report['latitude'], nfd_report['longitude'])
    m = folium.Map(
        location=incident_location,
        control_scale=True,
        zoom_start=15, zoom_control=True,
        tiles='OpenStreetMap',
        scrollWheelZoom=True,
    )
    folium.RegularPolygonMarker(
        location=incident_location,
        number_of_sides=3,
        radius=10,
        popup='Incident({:0.2f}, {:0.2f})'.format(*incident_location),
        color='crimson',
        fill=True,
        fill_color='crimson',
    ).add_to(m)
    colormap.caption = 'Time difference between official incident report and Waze report.'
    colormap.add_to(m)
    for _, incident_report in incident_reports_df.iterrows():
        waze_report = reports_df[reports_df['id'] == incident_report['waze_id']].iloc[0]
        folium.CircleMarker(
            location=(waze_report['latitude'], waze_report['longitude']),
            radius=10,
            popup='Waze Report({:0.2f}, {:0.2f})'.format(waze_report['latitude'], waze_report['longitude']),
            color=colormap(incident_report['delta_time']),
            fill=True,
            fill_color=colormap(incident_report['delta_time']),
        ).add_to(m)
    folium_static(m, width=None)


def show_distributions():
    """

    :return:
    """
    max_dist = st.sidebar.slider('Maximum distance from incident to Waze (meters)?', 0, 1000, 100)  # in m
    min_delta_time, max_delta_time = st.sidebar.slider('Select range of time difference to show (minutes)',
                                                       -30, 30, (-30, 30))
    incidents_df = fetch_and_clean_data(key='incidents')
    incidents_df = incidents_df[incidents_df['distance'] <= (max_dist / 1000)]
    incidents_df = incidents_df[incidents_df['delta_time'] <= max_delta_time]
    incidents_df = incidents_df[min_delta_time <= incidents_df['delta_time']]
    data_frame = incidents_df.groupby('nfd_id').agg(lambda s: s[s.abs().idxmin()])
    fig_labels = {'delta_time': 'Δ Time (min)'}
    fig = px.histogram(data_frame, x='delta_time', nbins=60, labels=fig_labels)
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)
    data_frame = incidents_df.groupby('nfd_id').agg(lambda s: 1000 * s[s.abs().idxmin()])
    fig_labels = {'distance': 'Δ Distance (m)'}
    fig = px.histogram(data_frame, x='distance', nbins=60, labels=fig_labels)
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)


def explore_dataset_view():
    """Exploratory data analysis.

    :return: None
    """
    visualization = {
        'Incidents': show_incidents,
        'Distributions': show_distributions
    }
    with st.sidebar:
        key = st.selectbox('Select visualization:', visualization.keys())
    visualization[key]()

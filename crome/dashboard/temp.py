import streamlit as st

from crome.dataset.dataset import DatasetBuilder


@st.experimental_memo
def build_dataset(delta_t, delta_s):
    """Build dataset for provided parameters.

    :param delta_t: time step value.
    :param delta_s: spatial resolution value.
    :return: dataset
    """
    waze_df = st.session_state.waze_df
    incident_df = st.session_state.incident_df
    config = dict(delta_t=delta_t, delta_s=delta_s)
    dataset_builder = DatasetBuilder(config=config, verbose=True)
    t_index, X, y = dataset_builder(waze_df, incident_df)
    y_true = dataset_builder.get_y_true(incident_df)
    return {
        'dataset_builder': dataset_builder,
        't_index': t_index,
        'X': X,
        'y': y,
        'y_true': y_true,
    }


@st.experimental_memo
def eda_preprocess_2(delta_t, delta_s):
    dataset = build_dataset(delta_t=delta_t, delta_s=delta_s)
    y = dataset['y']
    X = dataset['X'].sum(axis=1)
    indices = zip(*np.where(y > 0))
    centered_ = np.zeros((X.shape[0], 5, 5))
    for i, j, k in indices:
        for j2 in range(j - 2, j + 3):
            for k2 in range(k - 2, k + 3):
                try:
                    v = X[i, j2, k2, 0]
                except IndexError as err:
                    v = 0
                centered_[i, j2 - j + 2, k2 - k + 2] += v
    data_df = centered_.sum(axis=0) / centered_.sum()
    return data_df

#

# data_df = eda_preprocess_2(
#     delta_t=st.session_state.time_step,
#     delta_s=st.session_state.resolution
# )
# fig, ax = plt.subplots(figsize=(10, 8))
# res = sns.heatmap(data=data_df, annot=True, annot_kws={'size': 20}, ax=ax)
# ax.tick_params(axis='both', which='major', labelsize=15)
# for t in res.texts:
#     t.set_text('{:2.1f} %'.format(float(t.get_text()) * 100))
# plt.tight_layout()
# st.pyplot(fig, dpi=300)
# #
# data_df, filtered_cross_df = eda_preprocess(
#     delta_t=st.session_state.time_step,
#     delta_s=st.session_state.resolution
# )
# filters_ = filtered_cross_df['x_Waze'] == filtered_cross_df['x_NFD']
# filters_ &= filtered_cross_df['y_Waze'] == filtered_cross_df['y_NFD']
# fig_df = filtered_cross_df[filters_]
# fig, ax = plt.subplots(figsize=(15, 5))
# sns.histplot(x='Time(min)', bins=60, data=fig_df, ax=ax)
# ax.set_ylabel('Frequency')
# ax.set_xlabel('$\Delta$Time ($min$)')
# plt.tight_layout()
# st.pyplot(fig, dpi=300)
# #
# incidents = extract_incidents(
#     delta_t=st.session_state.time_step,
#     delta_s=st.session_state.resolution
# )
# colormap = branca.colormap.linear.viridis.scale(-30, 30)
# nfd_sample, waze_reports_df = incidents[0]
# incident_location = (nfd_sample['latitude'], nfd_sample['longitude'])
# m = folium.Map(
#     location=incident_location,
#     width=600, height=600,
#     control_scale=True,
#     zoom_start=18, zoom_control=False,
#     tiles='OpenStreetMap',
#     scrollWheelZoom=False,
# )
# folium.RegularPolygonMarker(
#     location=incident_location,
#     number_of_sides=3,
#     radius=10,
#     popup='Incident({:0.2f}, {:0.2f})'.format(*incident_location),
#     color='crimson',
#     fill=True,
#     fill_color='crimson',
# ).add_to(m)
# colormap.caption = 'Time difference between official incident report and Waze report.'
# colormap.add_to(m)
# for _, waze_sample in waze_reports_df.iterrows():
#     folium.CircleMarker(
#         location=(waze_sample['latitude'], waze_sample['longitude']),
#         radius=10,
#         popup='Waze Report({:0.2f}, {:0.2f})'.format(waze_sample['latitude'], waze_sample['longitude']),
#         color=colormap(waze_sample['DeltaTime']),
#         fill=True,
#         fill_color=colormap(waze_sample['DeltaTime']),
#     ).add_to(m)
# folium_static(m)
# #
# fig, ax = plt.subplots(figsize=(15, 5))
# sns.histplot(data=filtered_cross_df, x='Time(min)', y='Distance', ax=ax)
# ax.set_ylabel(r'$\Delta Time(min)$')
# ax.set_ylabel(r'$\Delta Distance(MDS)$')
# st.pyplot(fig, dpi=300)
# #
# st.slider(
#     label='Explore Date: ',
#     min_value=datetime.datetime(2019, 9, 1, 00, 00),
#     max_value=datetime.datetime(2019, 12, 31, 00, 00),
#     value=datetime.datetime(2019, 9, 1, 00, 00),
#     format="MM/DD/YY",
#     key='start_time'
# )
# start_time = st.session_state.start_time
# end_time = start_time + datetime.timedelta(hours=24)
# st.write('Start time:', start_time, 'End time:', end_time)
# #
# fig, ax = plt.subplots(figsize=(15, 5))
# fig_data_df = data_df[data_df['Time'].between(start_time, end_time)]
# sns.scatterplot(data=fig_data_df, x='Time', y='Distance', hue='Type', ax=ax)
# ax.set_ylabel(r'$Distance(MDS)$')
# st.pyplot(fig, dpi=300)
# #
# with st.expander('Explore Data'):
#     st.dataframe(data=data_df)
#

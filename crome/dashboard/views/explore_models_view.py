import os
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from crome.dashboard.database.manager import fetch_and_clean_data
from crome.optimize import pareto
from crome.config import config

PROJECT_PATH = config['DEFAULT']['project_path']


def _show_dt_ds_plot(df):
    plot_type = st.sidebar.selectbox('Type', ['2D', '3D'], 1)
    hover_data = ['Precision', 'Recall']
    if plot_type == '3D':
        fig = px.scatter_3d(df, x='Δt (mins)', y='Δs (m)', z='F1-Score', color='Model', hover_data=hover_data)
    elif df['Base Model'].unique().shape[0] == 1:
        fig = px.scatter(df, x='Δt (mins)', y='Δs (m)', color='Model', size='F1-Score', hover_data=hover_data)
    else:
        st.write('The 2D plot is valid for single model selection only.')
        fig = None
    if fig:
        fig.update_layout(height=675)
        st.plotly_chart(fig, use_container_width=True)
    else:
        pass


def _show_precision_recall_plot(df):
    fig = px.scatter(df, x='Precision', y='Recall', color='Model', size='F1-Score', hover_data=['Δt (mins)', 'Δs (m)'])
    fig.update_layout(height=675)
    st.plotly_chart(fig, use_container_width=True)


def _show_raw_data_table(df):
    st.table(df)


@st.cache
def _detect_optimal_models(df):
    if df.shape[0] == 0:
        return df
    columns = ['Δs (m)', 'Δt (mins)', 'Model', 'Features', 'F1-Score', 'Precision', 'Recall']
    df = df[columns]
    optimal_settings = pareto.eps_sort(df.values.tolist(), objectives=[0, 1, 4], maximize=[4], epsilons=[1e3, 5, 1e-3])
    opt_df = pd.DataFrame(optimal_settings, columns=columns)
    opt_df = opt_df.assign(Rank=opt_df.index + 1, Optimal=True)
    df = pd.merge(df, opt_df, how='left') \
        .fillna({'Rank': np.NaN, 'Optimal': False}) \
        .sort_values(['Optimal', 'F1-Score', 'Rank'], ascending=False)
    model_series = df['Model']
    df = df.assign(
        Model=df['Optimal'].map({True: 'CROME (', False: ''})
            .str.cat(model_series, sep='')
            .str.cat(df['Optimal'].map({True: ')', False: ''}), sep=''),
        **{'Base Model': model_series},
    )
    return df


def explore_models_view():
    """

    :return:
    """
    visualization = {
        'Performance Analysis (Precision-Recall)': _show_precision_recall_plot,
        'Resolution Analysis (Δt-Δs)': _show_dt_ds_plot,
        'Models Table': _show_raw_data_table,
    }
    with st.sidebar:
        key = st.selectbox('Select visualization:', visualization.keys())
    df = fetch_and_clean_data('scores')
    #
    model_options = df['model'].unique()
    selected_models = st.sidebar.multiselect('Select model(s)', model_options, [model_options[0]])
    df = df[df['model'].isin(selected_models)]
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
    #
    month_options = ['September', 'October', 'November', 'December', 'Average']
    test_month_name = st.sidebar.selectbox('Select test month', month_options, 4)
    if test_month_name != 'Average':
        test_month = datetime.strptime(test_month_name, '%B').month
        df = df[df['month'] == test_month]
    #
    index_cols = 'model,features,delta_t,delta_s'.split(',')
    agg_df = df.groupby(index_cols).agg({
        'test.basic.precision_score': [np.mean, np.std],
        'test.basic.recall_score': [np.mean, np.std],
        'test.basic.f1_score': [np.mean, np.std],
    }).reset_index()
    #
    df = pd.DataFrame({
        'Model': agg_df['model'],
        'Δt (mins)': agg_df['delta_t'],
        'Δs (m)': agg_df['delta_s'],
        'Features': agg_df['features'],
        'Precision': agg_df[('test.basic.precision_score', 'mean')],
        'Recall': agg_df[('test.basic.recall_score', 'mean')],
        'F1-Score': agg_df[('test.basic.f1_score', 'mean')],
    })
    df = _detect_optimal_models(df)
    if df.shape[0] > 0:
        visualization[key](df)
    else:
        pass

from datetime import timedelta
import tqdm
from itertools import product

import pandas as pd
import streamlit as st
from haversine import haversine_vector, Unit

from crome.dataset.loader import preprocess
from crome.dataset.dataset import DatasetBuilder
from crome.config import config

PROJECT_PATH = config['DEFAULT']['project_path']


@st.cache
def fetch_and_clean_data(key):
    """Load data from managed data store.

    :param key:
    :return:
    """
    if 'reports' == key:
        return pd.read_json('{}/data/processed/reports.json'.format(PROJECT_PATH), orient='table')
    elif 'report_grids' == key:
        return pd.read_json('{}/data/processed/report_grids.json'.format(PROJECT_PATH),
                            orient='table')
    elif 'incidents' == key:
        return pd.read_json('{}/data/processed/incidents.json'.format(PROJECT_PATH),
                            orient='table')
    return None


def setup_datastore():
    """Setup database.

    :return: None
    """
    builders = []
    for i, j in list(reversed(list(product(range(1, 7), range(1, 6))))):
        delta_t, delta_s = int(i) * 5, int(j) * int(1e3)
        builder = DatasetBuilder(config=dict(delta_t=delta_t, delta_s=delta_s), verbose=True)
        builders.append(builder)
    # process Waze reports and and add in table reports
    waze_df = preprocess(input='{}/data/raw/accident_reports.csv'.format(PROJECT_PATH))
    reports = []
    report_grids = []
    report_id = 0
    for idx, row in waze_df.iterrows():
        report = {key: row[key] for key in ['longitude', 'latitude', 'time', 'reliability']}
        report['id'] = report_id
        report['type'] = 'Waze'
        for _, builder in enumerate(builders):
            grid_col, grid_row = builder.grids.search(report)
            report_grids.append({
                'report_id': report_id,
                'delta_t': builder.delta_t,
                'delta_s': builder.delta_s,
                'x': grid_col,
                'y': grid_row,
            })
        reports.append(report)
        report_id += 1
    # process NFD reports and and add in table reports
    nfd_df = preprocess(input='{}/data/raw/incident_NFD_XDSegID.pkl'.format(PROJECT_PATH))
    for idx, row in nfd_df.iterrows():
        report = {key: row[key] for key in ['longitude', 'latitude', 'time']}
        report['id'] = report_id
        report['type'] = 'NFD'
        for _, builder in enumerate(builders):
            grid_col, grid_row = builder.grids.search(report)
            report_grids.append({
                'report_id': report_id,
                'delta_t': builder.delta_t,
                'delta_s': builder.delta_s,
                'x': grid_col,
                'y': grid_row,
            })
        reports.append(report)
        report_id += 1
    report_grids_df = pd.DataFrame(report_grids)
    report_grids_df.to_json('{}/data/processed/report_grids.json'.format(PROJECT_PATH), orient='table')
    reports_df = pd.DataFrame(reports)
    reports_df.to_json('{}/data/processed/reports.json'.format(PROJECT_PATH), orient='table')
    # add processed incidents
    incidents = []
    nfd_reports_df = reports_df[reports_df['type'] == 'NFD']
    for nfd_idx in tqdm.tqdm(range(nfd_reports_df.shape[0]), desc='Iterating Incidents'):
        nfd_report = nfd_reports_df.iloc[nfd_idx]
        # Get incident location
        incident_location = [nfd_report['latitude'], nfd_report['longitude']]
        # Filter relevant Waze reports
        start_time = nfd_report['time'] + timedelta(minutes=-30)
        end_time = nfd_report['time'] + timedelta(minutes=30)
        waze_reports_df = reports_df[(
                (reports_df['type'] == 'Waze') &
                (start_time <= reports_df['time']) &
                (reports_df['time'] < end_time)
        )]
        if waze_reports_df.shape[0] == 0:
            continue
        waze_loc_arr = waze_reports_df[['latitude', 'longitude']].to_numpy()
        waze_delta_dist = haversine_vector(waze_loc_arr, [incident_location], comb=True, unit=Unit.KILOMETERS)[0]
        waze_delta_time = ((nfd_report['time'] - waze_reports_df['time']).dt.total_seconds() / 60).to_numpy()
        for idx in range(waze_reports_df.shape[0]):
            waze_report = waze_reports_df.iloc[idx]
            incidents.append({
                'nfd_id': nfd_report['id'],
                'waze_id': waze_report['id'],
                'distance': waze_delta_dist[idx],
                'delta_time': waze_delta_time[idx],
            })
    incidents_df = pd.DataFrame(incidents)
    incidents_df.to_json('{}/data/processed/incidents.json'.format(PROJECT_PATH), orient='table')


if __name__ == '__main__':
    setup_datastore()

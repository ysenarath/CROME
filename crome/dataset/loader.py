import warnings

import pandas as pd
from pickle5 import pickle

from crome.preprocessing.regions import GridArray

import logging

warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)


def is_highway_incident(waze_df, inrix_df, res=1000):
    """Determines whether the incident happened in highway.

    :param waze_df: Waze DataFame.
    :param inrix_df: Inrix DataFrame.
    :param res: resolution.
    :return: is highway.
    """
    area_of_interest = (-87.1, 35.9), (-86.6, 36.5)
    sw, ne = area_of_interest
    ga = GridArray(*sw, *ne, res)
    df1 = waze_df.assign(gridIdx=waze_df.apply(ga.search, axis=1))
    df2 = inrix_df.assign(gridIdx=inrix_df.apply(ga.search, axis=1))
    grids_with_incident = set()
    for i in range(df2.shape[0]):
        grids_with_incident.add('{}_{}'.format(*df2.iloc[i]['gridIdx']))
    is_highway = df1['gridIdx'].apply(lambda row: '{}_{}'.format(*row)).isin(grids_with_incident)
    print('Number of Highway Grids at res={}: {}'.format(res, len(grids_with_incident)))
    del df1, df2, grids_with_incident, sw, ne
    return is_highway


def validate_input_type(df):
    """Infer input type from the dataframe columns.

    :param df: input dataFrame.
    :return: input type.
    """
    input_type = 'waze_report'
    if 'emdCardNumber' in df.columns:
        input_type = 'incident'
    elif ('GPS Coordinate Longitude' in df.columns) and ('GPS Coordinate Latitude' in df.columns):
        input_type = 'incident'
    return input_type


# noinspection PyShadowingNames,PyUnusedLocal,PyShadowingBuiltins
def preprocess(input):
    """ Creates and returns a spatiotemporal DataFrame from input records/reports.

    :param input: csv file to input record/report.
    :return:
    """
    # ==================================================================================================================
    #                                            LOAD DATASET
    # ------------------------------------------------------------------------------------------------------------------
    # read from input
    if input.endswith('.csv'):
        df = pd.read_csv(input)
    elif input.endswith('.pkl'):
        with open(input, 'rb') as fp:
            df = pickle.load(fp)
    input_type = validate_input_type(df)
    # ------------------------------------------------------------------------------------------------------------------
    # add longitude and latitude to DataFrame
    if ('longitude' not in df.columns) or ('latitude' not in df.columns):
        columns = {'longitude': 'lng', 'latitude': 'lat'}
        if input_type == 'incident':
            columns = {'longitude': 'GPS Coordinate Longitude',
                       'latitude': 'GPS Coordinate Latitude'}
        for k, v in columns.items():
            df = df.assign(**{k: df.loc[:, v]})
    # ------------------------------------------------------------------------------------------------------------------
    # remove any unnamed columns if exist
    unnamed_columns = [c for c in df.columns if c.startswith('Unnamed')]
    if len(unnamed_columns) > 0:
        df.drop(unnamed_columns, axis=1, inplace=True)
    # ==================================================================================================================
    #                                            SELECT REGION
    # ------------------------------------------------------------------------------------------------------------------
    # filter data by selected region (nashville)
    region_rect = [(-87.418212890625, 36.65079252503471), (-86.0888671875, 35.523285179107816)]
    df = df[(region_rect[0][0] <= df.loc[:, 'longitude']) & (df.loc[:, 'longitude'] <= region_rect[1][0]) & (
            region_rect[1][1] <= df.loc[:, 'latitude']) & (df.loc[:, 'latitude'] <= region_rect[0][1])]
    # ==================================================================================================================
    #                                            UPDATE FORMATTING
    # ------------------------------------------------------------------------------------------------------------------
    # create time column from unix timestamp at runtime (or format existing column)
    if 'time' in df.columns:
        time_series = pd.to_datetime(df.loc[:, 'time']).dt.tz_localize(None)
    else:
        if 'timestamp' in df.columns:
            column, div = 'timestamp', 1
        elif 'pubMillis' in df.columns:
            column, div = 'pubMillis', 1000.0
        else:
            raise AttributeError('Attribute \'timestamp\' or \'pubMillis\' not found in {fn}.'.format(
                fn=input
            ))
        time_series = pd.to_datetime(df.loc[:, column].astype(float).apply(lambda x: x / div), unit='s')
    df = df.assign(time=time_series)
    # ==================================================================================================================
    #                                            SELECT TIME_RANGE
    # ------------------------------------------------------------------------------------------------------------------
    df = df[(df['time'].dt.year == 2019) & (df['time'].dt.month >= 9) & (df['time'].dt.month <= 12)]
    # ------------------------------------------------------------------------------------------------------------------
    if 'uuid' in df.columns:
        df = df.drop_duplicates(subset='uuid', keep='last')
    return df

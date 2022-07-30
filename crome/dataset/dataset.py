import datetime
import logging
import math
import numpy as np
import pandas as pd
import tqdm
from rtree.index import Index
from shapely.geometry import Polygon, Point

from crome.preprocessing.regions import GridArray

logger = logging.getLogger(__name__)


def get_grid_index(grids):
    """Returns a function that finds the grids of provided points.

    :param grids: input grids.
    :return: function.
    """
    grid_idx = Index()
    grid_features = grids.to_geojson()['features']
    grid_polygons = dict()
    index_position = {}
    for ci, co in enumerate(grid_features):
        coordinates = []
        idx = tuple(co['properties']['index'])
        for lng, lat in co['geometry']['coordinates'][0]:
            pt = grids.proj.transform(lng, lat)
            coordinates.append(pt)
        p = Polygon(coordinates)
        index_position[ci] = idx
        grid_polygons[ci] = p
        grid_idx.insert(ci, p.bounds)

    def find_func(x, r=1000):
        """Find the overlaps with grids and returns.

        :param x: points.
        :param r: buffer
        :return: all indexes of grids with intersections.
        """
        lng_p, lat_p = grids.proj.transform(*x)
        poly = Point(lng_p, lat_p).buffer(r)
        return {
            index_position[ix]: poly.intersection(grid_polygons[ix]).area
            for ix in grid_idx.intersection(poly.bounds)
        }

    return find_func


class DatasetBuilder(object):
    """DatasetBuilder"""

    def __init__(self, config=None, verbose=False):
        super().__init__()
        if config is None:
            config = dict()
        self.verbose = verbose
        # automatically provide area of interest if not provided
        area_of_interest = config.get('area_of_interest', config.get('W', None))
        if area_of_interest is None:
            # area of interest
            logger.warning('Setting \'area_of_interest\' to the default value: Nashville Area, USA.')
            area_of_interest = (-87.1, 35.9), (-86.6, 36.5)
        sw, ne = area_of_interest
        # resolution in meters
        self.delta_s = config.get('delta_s', config.get('res', 1000))
        # time step in minutes
        self.delta_t = config.get('delta_t', config.get('time_step', config.get('t_s', 5)))
        # incident time in minutes
        self.T_prime = config.get('incident_time', config.get('T_prime', 30))
        # alpha in minutes
        self.alpha = config.get('alpha', 60)
        # beta in minutes
        self.beta = config.get('beta', 60)
        # extract
        self.return_reports = config.get('return_reports', False)
        # delta - maximum distance traveled by wazer after observing the incident (in terms of grids?)
        if 'delta' in config:
            self.delta = config.get('delta')
            self.delta_grids = None
        else:
            self.delta_grids = config.get('delta[grids]', 1)
            self.delta = None
        # generate grids from
        self.grids = GridArray(*sw, *ne, self.delta_s)

    def __call__(self, xdf, ydf):
        # add grid index param
        xdf = xdf.assign(gridIdx=xdf.apply(self.grids.search, axis=1))
        ydf = ydf.assign(gridIdx=ydf.apply(self.grids.search, axis=1))
        if self.return_reports:
            return self._get_reports(xdf, ydf)
        xdf = xdf[['reliability', 'time', 'gridIdx']].resample(
            '{}T'.format(self.delta_t), on='time', label='right', origin='epoch'
        ).agg({'reliability': list, 'gridIdx': list}).reset_index()
        t_index = xdf['time'][xdf['reliability'].apply(len) > 0]
        invalid_count = 0
        dataset = ([], [], [])  # ([], [], [], [])
        enum_t_index = enumerate(t_index)
        if self.verbose:
            enum_t_index = tqdm.tqdm(enum_t_index, desc='Building dataset')
        for idx, t in enum_t_index:
            e = self._extract_record(xdf, ydf, t)
            if e is not None:
                for i in range(0, 3):
                    dataset[i].append(e[i])
            else:
                invalid_count += 1
        return np.array(dataset[0]), np.array(dataset[1]), np.array(dataset[2])

    def _extract_record(self, xdf, ydf, time):
        if self.delta_grids is not None:
            d = self.delta_grids
        else:
            d = math.ceil(self.delta / self.grids.delta)
        # <!-- extract all waze reports from start time to current time (`time`)
        start_time_waze = time - datetime.timedelta(minutes=self.T_prime)
        if not (xdf['time'] <= start_time_waze).any():
            return None
        features = xdf[(xdf['time'] > start_time_waze) & (xdf['time'] <= time)]
        tmp_x = np.zeros((features.shape[0], *self.grids.shape, 3))
        tmp_a = np.zeros(self.grids.shape)  # marks all cells that are near to Waze reports
        for j, row in enumerate(features.itertuples()):
            for (x, y), r in zip(row.gridIdx, row.reliability):
                tmp_x[j, x, y, 0] += 1  # counts number of records
                tmp_x[j, x, y, 1] += r  # sum of reliability
                # <!-- mark the near by cells in temp_a
                for xa in range(max(x - d, 0), min(x + d, tmp_a.shape[0])):
                    for ya in range(max(y - d, 0), min(y + d, tmp_a.shape[1])):
                        tmp_a[xa, ya] = 1
                # -->
        # -->
        # <!-- calculate mean of reliability
        ta, tb = tmp_x[:, :, :, 1], tmp_x[:, :, :, 0]
        tmp_x[:, :, :, 2] = np.divide(ta, tb, out=np.zeros_like(ta), where=(tb != 0))
        # -->
        # <!-- get the true label associated with Waze reports true label is determined with in\
        #     uncertain period -\alpha to +\beta and spacial resolutin of \delta
        start_time = time - datetime.timedelta(minutes=self.alpha)
        end_time = time + datetime.timedelta(minutes=self.beta)
        target = ydf[(ydf['time'] > start_time) & (ydf['time'] <= end_time)]
        tmp_y = np.zeros(self.grids.shape)
        for x, y in target.gridIdx:
            tmp_y[x, y] = 1
        tmp_y_near = tmp_y * tmp_a  # get delta close cells to waze reports observed during this time step
        # -->
        return time, tmp_x, tmp_y_near, tmp_y

    def get_y_true(self, ydf):
        """Get y true values.

        :param ydf:
        :return:
        """
        ydf = ydf.assign(gridIdx=ydf.apply(self.grids.search, axis=1))
        ydf = ydf \
            .resample('1T', on='time', label='right', origin='epoch') \
            .agg({'gridIdx': set, 'latitude': list, 'longitude': list}) \
            .reset_index()
        return list(zip(ydf['time'], ydf['gridIdx'], ydf['longitude'], ydf['latitude']))

    def _get_reports(self, xdf, ydf):
        if self.delta_grids is not None:
            d = self.delta_grids
        else:
            d = math.ceil(self.delta / self.grids.delta)
        xdf = xdf[['time', 'reliability', 'longitude', 'latitude']].resample(
            '{}T'.format(self.delta_t), on='time', label='right', origin='epoch'
        ).agg({'reliability': list, 'longitude': list, 'latitude': list}).reset_index()
        find_overlaps = get_grid_index(self.grids)
        t_index = xdf['time'][xdf['reliability'].apply(len) > 0]
        dataset = ([], [], [])
        enum_t_index = enumerate(t_index)
        if self.verbose:
            enum_t_index = tqdm.tqdm(enum_t_index, desc='')
        for idx, t in enum_t_index:
            start_time_waze = t - datetime.timedelta(minutes=self.T_prime)
            if not (xdf['time'] <= start_time_waze).any():
                continue
            dataset[0].append(t)
            features = xdf[(xdf['time'] > start_time_waze) & (xdf['time'] <= t)]
            xf = []
            tmp_a = np.zeros(self.grids.shape)  # marks all cells that are near to Waze reports
            for index, row in features.iterrows():
                xft = []
                for r, lng, lat in zip(row['reliability'], row['longitude'], row['latitude']):
                    for gridIdx, overlap in find_overlaps((lng, lat)).items():
                        xft.append({
                            'gridIdx': gridIdx,
                            'overlap': overlap,
                            'reliability': r,
                        })
                        # <!-- mark the near by cells in temp_a
                        x, y = gridIdx
                        for xa in range(max(x - d, 0), min(x + d, tmp_a.shape[0])):
                            for ya in range(max(y - d, 0), min(y + d, tmp_a.shape[1])):
                                tmp_a[xa, ya] = 1
                        # -->
                xf.append({
                    'time': row['time'],
                    'reports': xft,
                })
            dataset[1].append(xf)
            # <!--
            start_time = t - datetime.timedelta(minutes=self.alpha)
            end_time = t + datetime.timedelta(minutes=self.beta)
            target = ydf[(ydf['time'] > start_time) & (ydf['time'] <= end_time)]
            tmp_y = np.zeros(self.grids.shape)
            for x, y in target.gridIdx:
                tmp_y[x, y] = 1
            tmp_y_near = tmp_y * tmp_a
            dataset[2].append(tmp_y_near)
            # -->
        return np.array(dataset[0]), pd.Series(dataset[1]), np.array(dataset[2])

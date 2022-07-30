from collections.abc import Mapping
import json

import numpy as np
import pandas as pd
from pyproj import Proj
from shapely.geometry import box, Point

from shapely.strtree import STRtree

__all__ = [
    'load_regions',
    'RegionIndex',
    'GridArray',
]


class PolygonIndex:
    def __init__(self, data):
        """ Create index from provided array-like.

        :param data: array-like shapely Geometry objects
        """
        self.data = data
        self.tree = STRtree(self.data)
        self.index_by_id = dict((id(pt), i) for i, pt in enumerate(data))

    def __call__(self, pt):
        """ Returns the index and polygon of the nearest polygon.

        :param pt: Point(longitude, latitude)
        :return:
        """
        obj = self.tree.nearest(pt)
        return self.index_by_id[id(obj)]


def load_regions(path='../../resources/other/grids.npy'):
    """Load all grids for the region from saved numpy array.

    :param path: path to the numpy array
    :return: numpy array grid
    """
    _project = Proj(
        '+proj=lcc +lat_1=36.41666666666666 +lat_2=35.25 +lat_0=34.33333333333334 +lon_0=-86 '
        '+x_0=600000 +y_0=0 +ellps=GRS80 +datum=NAD83 +no_defs'
    )
    grid = []
    for grid_h in np.load(path, encoding='bytes', allow_pickle=True):
        h_points = []
        for grid_y in grid_h:
            y_points = []
            for point in grid_y:
                lng, lat = _project(point[0], point[1], inverse=True)
                y_points.append([lng, lat])
            h_points.append(y_points)
        grid.append(h_points)
    return np.array(grid)


class RegionIndex(PolygonIndex):
    def __init__(self, grids):
        """ Create grid index.

        :param grids: numpy array of grid boxes
        """
        data = [box(*grids[x, y][0], *grids[x, y][1])
                for x in range(len(grids)) for y in range(len(grids[x]))]
        super().__init__(data)
        self.index_position = [(x, y) for x in range(len(grids))
                               for y in range(len(grids[x]))]

    def __getitem__(self, index):
        return self.index_position[index]

    def get_distance(self, a, b, method='manhattan'):
        if method.lower() == 'manhattan':
            return abs(self[a][0] - self[b][0]) + abs(self[a][1] - self[b][1])
        else:
            return ((self[a][0] - self[b][0]) ** 2 + (self[a][1] - self[b][1]) ** 2) ** 0.5


class GridArray:
    """Grid array gets a region rectangle and splits that area to grid regions.
    Search function will help to get which grid a point belongs.

    Example usage:
    >>> from crome.preprocessing.regions import GridArray
    >>> grid_array = GridArray(*sw, *ne, res=1000)
    >>> print('Number of Regions: {}'.format(len(grid_array)))
    >>> p = (sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2
    >>> print('grid of {} is {}'.format(p, grid_array.search(p)))
    """
    proj = Proj(
        '+proj=lcc +lat_1=36.41666666666666 +lat_2=35.25 +lat_0=34.33333333333334 +lon_0=-86 '
        '+x_0=600000 +y_0=0 +ellps=GRS80 +datum=NAD83 +no_defs',
    )

    def __init__(self, x1, y1, x2, y2, delta=1000):
        """Initialize Grid Array.

        :param x1: South-West Longitude
        :param y1: South-West Latitude
        :param x2: North-East Longitude
        :param y2: North-East Latitude
        :param delta: resolution of the grid
        """
        co_sw, co_ne = Point(x1, y1), Point(x2, y2)
        sw, ne = Point(GridArray.proj.transform(co_sw.x, co_sw.y)), Point(
            GridArray.proj.transform(co_ne.x, co_ne.y))
        self._delta = delta
        self.xs = np.arange(sw.x - self._delta / 2,
                            ne.x + self._delta, self._delta)
        self.ys = np.arange(sw.y - self._delta / 2,
                            ne.y + self._delta, self._delta)

    def _validate(self, col, row):
        return (0 <= col < len(self.xs)) and (0 <= row < len(self.ys))

    def search(self, pt):
        """Search the grid where `pt` is located.

        :param pt: Point
        :return: Tuple(row, col) of pt
        """
        if isinstance(pt, Point):
            x, y = pt.x, pt.y
        elif isinstance(pt, pd.Series):
            x, y = pt.longitude, pt.latitude
        elif isinstance(pt, Mapping):
            x, y = pt['longitude'], pt['latitude']
        else:
            x, y = pt
        pj = Point(GridArray.proj(x, y))
        # `np.searchsorted` - finds indices where elements should be inserted to maintain order
        col, row = np.searchsorted(self.xs, pj.x) - 1, np.searchsorted(self.ys, pj.y) - 1
        assert self._validate(col, row), 'Point out of region. Consider changing bounding box.'
        return col, row

    # noinspection PyBroadException
    def reverse_search(self, col, row):
        """Search the South-West location of grid provided by col, and row.

        :param col: grid column
        :param row: grid row
        :return: South-West location of grid
        """
        assert self._validate(col, row), 'Grid index out of range'
        lng_a, lat_a = self.proj(self.xs[col], self.ys[row], inverse=True)
        try:
            lng_b, lat_b = self.proj(self.xs[col + 1], self.ys[row + 1], inverse=True)
        except Exception as _:
            return lng_a, lat_a
        return (lng_a + lng_b) / 2, (lat_a + lat_b) / 2

    @property
    def delta(self):
        """Gets delta

        :return: delta parameter
        """
        return self._delta

    def __len__(self):
        """Returns total number of grids in provided region"""
        return len(self.xs) * len(self.ys)

    @property
    def shape(self):
        """Gets shape of grid.

        :return: shape of grid
        """
        return len(self.xs), len(self.ys)

    @property
    def bbox(self):
        """Gets bounding box of grid.

        :return:
        """
        x0, y0, x1, y1 = min(self.xs), min(self.ys), max(self.xs), max(self.ys)
        return [*self.proj(x0, y0, inverse=True), *self.proj(x1, y1, inverse=True)]

    def __iter__(self):
        """Generator for grids.

        :return: Generator for Tuple(Grid Index, South-West Pt, North-East Pt)
        """
        for row, (y0, y1) in enumerate(zip(self.ys, self.ys[1:])):
            for col, (x0, x1) in enumerate(zip(self.xs, self.xs[1:])):
                yield [(col, row), self.proj(x0, y0, inverse=True), self.proj(x1, y1, inverse=True)]

    def to_geojson(self, path_or_buf=None, return_string=False):
        """Convert the object to a GeoJSON string.

        @path_or_buf: File path or object. If not specified, the result is only returned as a string.
        """
        features = []
        for (x, y), sw, ne in self:
            south_west = [sw[0], sw[1]]
            south_east = [ne[0], sw[1]]
            north_east = [ne[0], ne[1]]
            north_west = [sw[0], ne[1]]
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [
                        [
                            south_west, south_east, north_east, north_west, south_west
                        ]
                    ],
                },
                'properties': {
                    'index': [x, y],
                },
            })
        features = {
            'type': 'FeatureCollection',
            'bbox': self.bbox,
            'features': features
        }
        if path_or_buf is not None:
            with open(path_or_buf, 'w', encoding='utf-8') as fp:
                json.dump(features, fp)
        else:
            if return_string:
                return json.dumps(features)
            return features

#!/usr/bin/python3

__author__ = 'Noah Otte <nvethrandil@gmail.com>'
__version__= '0.1'
__license__= 'MIT'

import sys

# Forked from Eric Wieser @ https://github.com/eric-wieser/ros_numpy.git

#from .registry import converts_from_numpy, converts_to_numpy
from nav_msgs.msg import OccupancyGrid, MapMetaData
import numpy as np

#@converts_to_numpy(OccupancyGrid)
def occupancygrid_to_numpy(msg):
	data = np.asarray(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)

	return np.ma.array(data) #, mask=data==-1, fill_value=1) #fill_value = -1


#@converts_from_numpy(OccupancyGrid)
def numpy_to_occupancy_grid(arr, info=None):
	if not len(arr.shape) == 2:
		raise TypeError('Array must be 2D')
	if not arr.dtype == np.int8:
		raise TypeError('Array must be of int8s')

	grid = OccupancyGrid()
	if isinstance(arr, np.ma.MaskedArray):
		# We assume that the masked value are already -1, for speed
		arr = arr.data
	grid.data = arr.ravel()
	grid.info = info or MapMetaData()
	grid.info.height = arr.shape[0]
	grid.info.width = arr.shape[1]

	return grid


def world2map(point, msg):
        resolution = msg.info.resolution
        x = round((point[0] - msg.info.origin.position.x) / resolution, 3)
        y = round((point[1] - msg.info.origin.position.y) / resolution, 3)
        return [x, y]


def map2world(point, msg):
    resolution = msg.info.resolution
    x = round((point[0] * resolution + msg.info.origin.position.x), 3)
    y = round((point[1] * resolution + msg.info.origin.position.y), 3)
    return [x, y]


def planner2world(path, occmap):
        world_path = []
        for p in path:
                a = map2world([p[0], p[1]], occmap)
                world_path.append(a)
        return world_path


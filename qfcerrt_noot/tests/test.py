#!/usr/bin/python3

__author__ = 'Noah Otte <nvethrandil@gmail.com>'
__version__= '1.0'
__license__= 'MIT'

import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
from qfcerrt_noot.src.QFCE_RRT import QFCERRT as planner
from qfcerrt_noot.src.QFCE_RRT_Star import QFCERRTStar as planner_2
from scipy.ndimage import binary_dilation, gaussian_filter
import os 
filedir = os.path.dirname(os.path.abspath(__file__))
test_map_file = os.path.join(filedir, 'test_map2.npy')

plot_enabled = True

# Load Cspace

grid = np.load(test_map_file)
cmap="binary"
# Simulation settings
start = np.array([40.0, 3.0])  #[15.0, 80.0]
goal = np.array([40.0, 67.0])
#start = np.array([9.0, 89.0])  #[15.0, 80.0]
#goal = np.array([76.0, 4.0])  # [71, 36] [69.0, 6.0] [90.0, 10.0] [35.0, 6.0] [23, 8] [85.0, 44.0]
iterations = 1000  # 1000
stepsize = 1 # 50
no_path_found = -1
neighbour_radius = 40
no_path_found = -1
cell_sizes = [10, 100]
search_radius_increment_percentage = 0.25
max_neighbour_found = 12

rover_radius = 4 # pixels x 0.1m/pixels
noise_margin = 2 # resolution is 0.1 -> 2x that
minimum_lidar_distance = 9

mode = 2 # which mode to operate in
danger_zone = 15 # replanning is triggered if collisions might occur this many pixels ahead

bdilation_multiplier = 4 #minimum_lidar_distance + noise_margin
fov = 90
# Plot settings
fig = plt.figure("QFCE-RRT Test")
matplotlib.rcParams.update({'font.size': 20})
goalRegion = plt.Circle((goal[0], goal[1]), 3*stepsize, color='b', fill=False)
startRegion = plt.Circle((start[0], start[1]), 3*stepsize, color='r', fill=False)
# GAUSS
cost_h = 10
std_div = 5
gaussed_grid = binary_dilation(grid, iterations=bdilation_multiplier).astype(bool)
gaussed_grid = np.where(gaussed_grid > 0, cost_h, grid)
gaussed_grid = gaussian_filter(gaussed_grid, sigma=std_div, mode='wrap')

boosted_grid = np.where(grid > 0, cost_h/2, grid)
#show_grid = np.add(gaussed_grid, boosted_grid)
show_grid = gaussed_grid
plt.imshow(show_grid, cmap='plasma') #binary gray_r
plt.plot(start[0], start[1], 'ro')
plt.plot(goal[0], goal[1], 'bo')
ax = fig.gca()
ax.add_patch(goalRegion)
ax.add_patch(startRegion)
plt.xlabel("x $(0.1m/pixel)$")
plt.ylabel("y $(0.1m/pixel)$")
plt.tight_layout()

# Init RRT Algorithm
tic = time.process_time() *1000

# RRT based
#rrt = planner(grid, start, goal, iterations, stepsize, plot_enabled, max_neighbour_found, bdilation_multiplier, cell_sizes, mode, danger_zone)
#path = rrt.search()

# RRT* based
rrt = planner_2(grid, start, goal, iterations, stepsize, plot_enabled, max_neighbour_found, bdilation_multiplier, cell_sizes, mode, danger_zone, fov)
path = rrt.search_rrtstar()

toc = time.process_time()*1000
rrt.plotAllPaths(rrt.tree)

# Show results
if len(path) > 1:
    rrt.plotWaypoints(path)
    wps = len(path)
    d = rrt.getPathDistance()
    extime = toc - tic
    qt_time = rrt.quadtree_time
    pfs_time = rrt.empty_time
    [xs, ys] = np.where(grid == 0)
    free_pixels_in_map = len (xs)
    sample_ratio = (1 - (rrt.start_number_of_emptys / free_pixels_in_map)) * 100
    print("Number of waypoints: ", wps)
    print("Path Distance (pixels): ", d)
    print("Total runtime (s): ", extime)
    print("QuadTree Build-time (s): ", qt_time)
    print("PFS time (ms): ", pfs_time)
    print("Binary Dilation Time (s): ", rrt.bdil_time)
    print("Spent searching (s): ", rrt.search_time)
    print("Minimum cell side (not squared): ", stepsize)
    print("Free pixels in map: ", free_pixels_in_map)
    print("Emptys: ", rrt.start_number_of_emptys)
    print("Reduction in potential samples (%): ", sample_ratio)
else:
    rrt.plotAllPaths(rrt.goal)
    print("Goal not found")
print("Number of iterations completed: ", rrt.getIterationsDone())
m = f'QFCE-RRT* with Costmap Test'
plt.title(m)
plt.show()


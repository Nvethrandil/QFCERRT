# Quadtree-based Free-space Cell-decomposition Enhanced RRT (QFCE-RRT) path planner

The planner is a novel approach in reducing computation-time by dividing the occupancy map into a grid of convex cells.
This approach currently prevents re-sampling to improve speed at the cost of increased failure rate.

*This branch still in development and contains a lot of methods which are currently not required by the planner*


## How the planner works
The planner works as simple as:

1. The input map is subjected to binary dilation which increases the size of all objects by the user defined safety distance.
2. The planner performs edgedetection on the occupancy map and the resulting contours are fed into the quadtree datastructure. 
3. The quadtree cells which match the desired minimum and maximum area requirements are extracted. The centrepoint of each such cell are then viable sampling candidates for the upcoming RRT algorithm.
4. RRT is run on the sampling set. Entries are removed from the set after successfull sampling, preventing re-sampling and increasing exploration speed.
5. A path is returned as soon as it is found, otherwise returns -1.

## How to use (example)
Here follows some example code. Similiar setup can be found in [test.py](qfcerrt_noot/tests/test.py).
### Example parameters
```python
# Load your map as .npy file, note that this results in X and Y axis flipping
grid = np.load('YOUR_OCCUPANCYMAP_AS_NUMPY_ARRAY')
# Start coordinates with flipped X/Y
start = np.array(['START_Y_COORDINATE', 'START_X_COORDINATE'])
# Goal coordinates with flipped X/Y 
goal = np.array(['GOAL_Y_COORDINATE', 'GOAL_X_COORDINATE'])
# Set maximum iterations for RRT
iterations = 500
# Flag to enable plotting mode or not, only used in code testing
plot_enabled = False
# Defines the n x n side of smallest cell in Quadtree. Depends on size of map.
stepsize = 1 
```
**stepsize** is recommended to tune based on map size. 
- Recommend trying 5-10 for smaller maps <100x100 and 10-50 for 1000x1000 for low obstacle density
- Recommend trying 1-5 for smaller maps <100x100 and 5-20 for 1000x1000 for high obstacle density
```python
# Min.pref cells are 10x smallest size, Max.pref are 20x larger than Min.Pref
cell_sizes = [10, 20]
```
**cell_sizes** is recommended to tune based on expected map obstacle density. 
- Recommend trying [10, 20] for most cases
- The more empty space exists the higher the upper cap has to be set
- Similiar the lowest limit has to be small enough to not exclude most sampling options
```python
# Defines the % increase in search range if no neighbours are found (DEPRICATED) 
search_radius_increment_percentage = 0.25
# Amount of neighbours to find each time (ONLY USED IN RRT*)
max_neighbour_found = 8
# Extra size margin to all obstacles in occupancy map
bdilation_multiplier = 2
```
**bdilation_multiplier** should also be adjusted in reference to the map resolution and the safety margins of the desired vehicle. Here a uncertainty of 0.2m in a map of 0.1m/pixel would be accomodated with a **bdilation_multiplier** of 2.

```python
# Which post-processing mode is supposed to be enabled, 0 for none, 1 for only interpolation and 2 for Bezier + interpolation
mode = 2
# ONLY USED DURING LIVE HARDWARE TESTING. The amount of pixels a collision has to be away from the rovers position for it to count as a flawed path and to return True on a need2replan check
danger_zone = 20
# The field-of-view which is selected for the field-of-view sampler (DEPRICATED)
fov = 90
```
**mode, danger_zone, fov** are primarily used during and were implemented for difficulties during live hardware testing and can be disabled without affecting the core planner.
### Execution of planner
```python
# Initialize the planner with all the information
planner = QFCE_RRT(
      grid, 
      start, 
      goal, 
      iterations, 
      stepsize, 
      plot_enabled, 
      search_radius_increment_percentage, 
      max_neighbour_found, 
      bdilation_multiplier, 
      cell_sizes,
      mode,
      danger_zone,
      fov)
# Perform the actual search
path = planner.search()
# path contains a list of [x,y] coordinates which then can be utilized for navigation
```

### Results example plotted
This configuration has a chance of yielding the following random result on the given map in [test.py](qfcerrt_noot/tests/test.py).

![demo](https://github.com/Nvethrandil/QFCERRT/blob/main/demo.png)

## Tuning parameters
-  Based on the map size it is run on, the **stepdistance** parameter has to be adjusted. Becuase this parameter defines the smallest possible quadtree cell instance, this should be larger than a single pixel in the map, otherwise no performance gain will be made, but can also not be too large, otherwise there will be no valid samplingpoints left. 
In maps of high obstacle density, where the planner is anticipated to plan between many narrow obstacles, this value has to be small enough for at least 1 cell to fit between the minimum gap between two obstacles in which the planner shall plan in. Reversely, this value can be increased to decrease the likelyness of the planner to plan between clusters and rather prefer avoiding them altogether.

- Secondly the **bdilation_multiplier** should be taken note of as this parameter will alter the percieved map permanently for the planner and will make all obstacles by the specified amount nPixels larger in all dimensions. This can result in obstacle merging.

## Issues

The planner seems to have a failure rate of about 18.7% due to the lack of re-sampling and the proximity based evaluation of potential parent -> edgecases where the closest parent candidate is obscured by an obstacle and the sample candidate for which the parent was found will therefore never get a valid parent, preventing sampling in  adjacent regions.

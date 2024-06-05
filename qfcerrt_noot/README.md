# Quadtree-based Free-space Celldecomposition Enhanced RRT (QFCE-RRT) path planner

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

## Important parameters
These are all input parameters to the planner:

 - map (np.ndarray): 
    - The map to plan in as a numpy array

 - start (np.ndarray): 
    - The start position in map

 - goal (np.ndarray):
    - The goal position in map

 - max_iterations (int): 
    - Maximum iterations allowed for the RRT to run

 - stepdistance (int): 
    - Step distance of RRT algorithm, now defines the root of the minimum area of cells allowed in quadtree

 - plot_enabled (bool):
    - Flag to enable plotting mode or not (ONLY IN EDITOR TESTING)

 - search_radius_increment (float):
    - The percentage of radius increments performed between each neighbour search (DEPRICATED)

 - max_neighbour_found (int): 
    - Minimum neighbours to be found before stopping search, only applies after that many neighbours also exist (ONLY USED IN RRT*)

 - bdilation_multiplier (int): 
    - Amount of pixels to bloat each object in map before planning. Acts as a safety margin.

 - cell_sizes (list): 
    - Contains the minimum multiplier (min_multiplier) for preferred cells and the maximum multiplier (max_multiplier) of the maximum cells according to; 
        - min_multiplier * minimum_cell_size -> minimum_cells_preferred  
        - max_multiplier * minimum_cells_preferred  -> maximum_cells_preferred
    - This approach prevents the algorithm from over-favouring regions around obstacles and provide safer paths, incorporating more larger cells.

## Tuning parameters
Based on the map size it is run on, the **stepdistance** parameter has to be adjusted. Becuase this parameter defines the smallest possible quadtree cell instance, this should be larger than a single pixel in the map, otherwise no performance gain will be made, but can also not be too large, otherwise there will be no valid samplingpoints left. 
In maps of high obstacle density, where the planner is anticipated to plan between many narrow obstacles, this value has to be small enough for at least 1 cell to fit between the minimum gap between two obstacles in which the planner shall plan in. Reversely, this value can be increased to decrease the likelyness of the planner to plan between clusters and rather prefer avoiding them altogether.

Secondly the **bdilation_multiplier** should be taken note of as this parameter will alter the percieved map permanently for the planner and will make all obstacles by the specified amount nPixels larger in all dimensions. This can result in obstacle merging.

## Issues

The planner seems to have a failure rate of about 18.7% due to the lack of re-sampling and the proximity based evaluation of potential parent -> edgecases where the closest parent candidate is obscured by an obstacle and the sample candidate for which the parent was found will therefore never get a valid parent, preventing sampling in  adjacent regions.
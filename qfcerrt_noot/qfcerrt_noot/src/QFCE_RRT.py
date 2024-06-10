#!/usr/bin/python3

__author__ = 'Noah Otte <nvethrandil@gmail.com>'
__version__= '3.1'
__license__= 'MIT'

from typing import Tuple
# ESSENTIAL Python libraries
import numpy as np
import random
# Libraries for plotting
import matplotlib.pyplot as plt
# Libraries for benchmark statistics
import time
# ESSENTIAL special libraries
from skimage.measure import find_contours
from scipy.ndimage import binary_dilation
# ESSENTIAL custom imports
from .QT import QuadTree as Quadtree
from .QT import Circle as circle
from .QT import Rectangle as rectangle
from .QT import Point as point
from .NodeOnTree import NodeOnTree as NodeOnTree

class QFCERRT:
    """
    A class used to compute a basic RRT search algorithm utilizing quadtree map tesselation
    """
    def __init__(self, map: np.ndarray, start: np.ndarray, goal: np.ndarray, max_iterations: int, stepdistance: int, plot_enabled: bool, search_radius_increment: float, max_neighbour_found: int, bdilation_multiplier: int, cell_sizes: list, mode_select: int):
        """
        Initializes RRT algorithm

        Args:
            map (np.ndarray): 
                The map to plan in as a numpy array
            start (np.ndarray): 
                The start position in map
            goal (np.ndarray):
                The goal position in map
            max_iterations (int):
                Maximum iterations allowed
            stepdistance (int):
                Step distance of RRT algorithm
            plot_enabled (bool):
                Flag to enable plotting mode or not
            search_radius_increment (float):
                The percentage of radius increments performed between each neighbour search
            max_neighbour_found (int):
                Minimum neighbours to be found before stopping search, only applies after that many neighbours also exist
            bdilation_multiplier (int):
                Amount of pixels to bloat each object in map
            cell_sizes (list):
                Contains the divident of minimum cell size (A), preferred cell size (B) and max cell size multiplier (C) according to:
                min_cell_size = (map_width * map_height) / A
                pref_cell_size = (map_width * map_height) / B
                max_cell_size = C * pref_cell_size
            mode_select (int):
                0 - nothing is processed
                1 - interpolate points to stepsize of ~1
                2 - only smooth turns using bezier
        """
        
        # to make obstacles bigger than they actually are; binary dilation
        bdil_t1 = time.process_time()
        self.mode = mode_select
        self.bd_multi = bdilation_multiplier
        self.map = binary_dilation(map, iterations=self.bd_multi).astype(bool)
        self.bdil_time =  time.process_time() - bdil_t1
        self.max_x = self.map.shape[0]
        self.max_y = self.map.shape[1]
        self.map_center_x = self.max_x / 2
        self.map_center_y = self.max_y / 2
        print("MAP DIM (planner): ", self.max_x, self.max_y)
        # just a very high number as a starting reference for distance comparisons
        self.maxDistance = self.max_x * self.max_y
        # originally the stepdistance of algorithm, here it is the smallest cell resolution allowed
        self.stepDistance = stepdistance 
        self.maxit = max_iterations
        # tree settings and data
        self.tree = NodeOnTree(start[0], start[1])
        self.goal = NodeOnTree(goal[0], goal[1])
        self.tree.d_root = 0
        self.tree.d_parent = 0
        
        self.node_collection = []
        self.waypoint_nodes = []
        self.node_collection.append(self.tree)
        self.start = start
        # neighbour search settings
        self.search_radius_increment = search_radius_increment
        self.max_neighbour_found = max_neighbour_found
        # runtime information
        self.waypoints = []
        self.nearest_node = None
        # Define QuadTree, make sure it takes the largest value so the map fits in the square quadtree
        if self.max_x > self.max_y:
            square_bound = self.max_x
        else:
            square_bound = self.max_y
        # quadtree related main settings
        self.mapBound = rectangle(self.map_center_y, self.map_center_x, self.max_y / 2, self.max_x / 2)
        self.capacity = 4
        self.qt = Quadtree(self.mapBound, self.capacity)
        # Add the start, the root, to the tree only
        self.qt.insert(point(self.tree.x, self.tree.y, self.tree))
        # quadtree subdivision and empty cell collection, this is for empty-space calculation of map 
        self.empty_cells = []
        self.cell_scores = []
        self.normalized_scores = []
        # create boundary box containing initial start and goal to limit quadtree sampler
        y = (self.goal.x + self.tree.x) / 2
        x = (self.goal.y + self.tree.y) / 2
        w = abs(self.goal.x - self.tree.x) 
        h = abs(self.goal.y - self.tree.y) 
        if w > h:
            square = w
        else:
            square = h
        # tuning settings
        [pref_min, pref_max] = cell_sizes
        self.min_cell_allowed = stepdistance*stepdistance
        # cells at least this size have their sample-chance boosted
        self.min_cell_preferred = pref_min * self.min_cell_allowed
        # largest cells size allowed, dont't count above this size
        self.max_cell_preferred = pref_max * self.min_cell_preferred
        # bonus multiplier for cells larger than min_cell_size
        self.extra_cell_weight = 5 
        # benchmark data
        self.iterations_completed = 0
        self.pathDistance = 0
        self.numberOfWaypoints = 0
        self.start_number_of_emptys = 0
        # plot settings
        self.colour_mode_2 = False
        self.plot_enabled = plot_enabled
        self.wp_col = 'ro'
        self.plot_col = 'go'
        self.qt_color = 'magenta'
        self.info_string = f"Cells larger than {self.min_cell_preferred} are weighted {self.extra_cell_weight}x higher"
        # actually build the second quadtree on the obstacles for free-space calculation
        t1 =  time.process_time()
        bound = rectangle(x, y, square, square)
        self.qt_map = Quadtree(bound, 1)
        # keep the time it took for computation reference
        self.quadtree_time = time.process_time() - t1
        # retrieve the actual free space cells in quadtree
        t_empty = time.process_time()
        self.__processFreeSpace(self.map, self.qt_map)
        self.empty_time = time.process_time() - t_empty
        
    def need2replan(self, new_position, new_map):
        # Early exit
        if not self.waypoints:
            return True
        
        self.map = binary_dilation(new_map, iterations=self.bd_multi).astype(bool)
        
        temp_list = self.waypoints
        temp_list.sort(key=lambda e: self.distance(e, new_position), reverse=False)
        index = self.waypoints.index(temp_list[0])
        del self.waypoints[0:index]
        #self.waypoints.insert(0, new_position)
        print("Waypoints length", len(self.waypoints))
        modlist = self.waypoints
        if self.distance(new_position, modlist[0]) > 0.5:
            modlist.insert(0, new_position)
        for i in range(len(modlist)-1):
            p1 = modlist[i]
            p2 = modlist[i]
            if self.collision(p1, p2, round(self.distance(p1, p2))):
                print("REPLANNING . . .")
                return True
            
        self.waypoints = modlist
        return False
            
    def search(self) -> list:
        """
        A method which performs RRT-search

        Returns:
            (list): 
                The path found by RRT, or [-1] if no path was found
        """
        searchtime_start =  time.process_time()
        i = 0
        # Early exit in case goal is in line-of-sight
        if self.jackpot([self.tree.x, self.tree.y]) is True:
            print("Goal at once.")
            self.waypoints.insert(0, self.start)
            self.waypoints.append([self.goal.x, self.goal.y])
            self.iterations_completed = 0
            self.search_time =  time.process_time() - searchtime_start
            return self.waypoints
                    
        # Start searching    
        while i < self.maxit and self.empty_cells:
            i += 1  
            # Sample 
            sample, index = self.sampleFromEmptys()
            parent = self.sort_collection_for_nearest(sample)
            distance = round(self.distance([parent.x, parent.y], sample))
            # Check if sample worked
            if not self.collision([parent.x, parent.y], sample, distance):
                child = self.makeHeritage(parent, sample, index, distance)
                # Check if sample found goal
                if self.jackpot(sample) is True:
                    print("Goal found at: ", i)
                    # Exit procedures for found goal
                    self.adoptGoal(child)
                    self.goal.d_parent = self.distance(sample, [self.goal.x, self.goal.y])
                    self.goal.d_root = child.d_root + distance
                    self.retracePath(self.goal)
                    self.waypoints.insert(0, self.start)
                    self.apply_post_process
                    self.iterations_completed = i
                    self.search_time =  time.process_time() - searchtime_start
                    return self.waypoints
                    
        # Save parameters for failed search                
        self.iterations_completed = i
        self.search_time =  time.process_time() - searchtime_start
        print("Goal not found. Iterations completed: ", i)
        # Return failure value -1
        return [-1]
    
    
    def apply_post_process(self):
        """
        Performs post processing based on selected mode
        """
        if self.mode == 0:
            pass
        if self.mode == 1:
            self.waypoints = self.step_by_step_interpolation(self.waypoints)
        if self.mode == 2:
            self.waypoints = self.bezier_tripples(self.waypoints, 30)
    
          
    def sampleRandomEmptyNeighbour(self) -> Tuple[list, int, NodeOnTree]:
        
        parent = random.choice(self.node_collection)
        p = [parent.x, parent.y]
        samples = []
        for i in range(5):
            
            [[x, y, s]]  = random.choices(self.empty_cells, self.normalized_scores)
            samples.append([x, y, s])
            
        samples.sort(key=lambda e: self.distance([e[0], e[1]], p), reverse=False)  
        [x, y, s] = samples[0]
        index = self.empty_cells.index([x, y, s])
        sample = [x, y]     
        
        return sample, index, parent

    
    def sampleClosest(self)-> Tuple[list, int, NodeOnTree]:
        
        parent = random.choice(self.node_collection)
        p = [parent.x, parent.y]
        self.empty_cells.sort(key=lambda e: self.distance([e[0], e[1]], p), reverse=False)  
        [x, y, s] = self.empty_cells[0]
        index = self.empty_cells.index([x, y, s])
        sample = [x, y]     
        
        return sample, index, parent
    
    
    def sort_collection_for_nearest(self, point: list) -> NodeOnTree:
        """
        A method which sorts all existing nodes based on their distance to the desired reference point and returns the closest one

        Args:
            point (list): 
                The reference point

        Returns:
            NodeOnTree: 
                The closest existing node to the reference point
        """
        # Problem is there might be a point which is the closest, but is not collision free. In that case P will never find a valid Parent 
        # because the only valid ever returned was the one which leads to a collision . . .
        
        p = point
        self.node_collection.sort(key=lambda e: self.distance([e.x, e.y], p), reverse=False) 
        return self.node_collection[0]
    
    
    def adoptGoal(self, parent: NodeOnTree) -> None:
        """
        A method for a node to adopt the goal-node. 
        Sets flag goalWasFound to True.

        Args:
            parent (NodeOnTree): 
                The node that is supposed to adopt the goal
        """
        self.node_collection.append(self.goal)
        self.adoptChild(parent, self.goal)
        
        self.qt.insert(point(self.goal.x, self.goal.y, self.goal))
        self.goalWasFound = True
        
    
    def traceToRoot(self, parent, sample) -> list:
        cp = parent
        points =[sample, [parent.x, parent.y]]
        while cp.parent:
            cp = cp.parent
            #print("Appended: ", [cp.x, cp.y])
            points.append([cp.x, cp.y])
        points.append([self.tree.x, self.tree.y])
        return points
    
            
    def __processFreeSpace(self, the_map: np.ndarray, qt: Quadtree) -> None:
        """
        A method which sorts the contours of all objects in map into a Quadtree,
        stores the free space cells in global parameters and their sampling probability
        
        Args:
            the_map (np.ndarray):
                The map in which to determine the free space in
            qt (Quadtree):
                The Quadtree object in which to process the free space with and store it in
        """
        self.empty_cells = []
        self.cell_scores = []
        # Retrieve only the contours of objects in map
        boundries = find_contours(the_map, level=0.5)
        
        # Sort the contours into quadtree 
        for b in boundries:
            for p in b:
                self.qt_map.insert(point(p[0], p[1], None))
                
        new_cells = self.__findEmpty(qt)
        self.empty_cells.extend(new_cells)
        
        all_wh_values = []
        # check if there even are empty cells
        if self.empty_cells:
            for e in self.empty_cells:
                all_wh_values.append(e[2])
            avg_score = round(np.mean(all_wh_values), 2)
            #print("Boosting above: ", avg_score)
        
            # Scoring for non-uniform sampling
            for e in  self.empty_cells:
                if e[2] >= avg_score: # Trust these cells more, increase sampling probability
                    self.cell_scores.append(e[2]*self.extra_cell_weight)
                else:
                    self.cell_scores.append(e[2])
                
            self.total_score = np.sum(self.cell_scores)
            self.normalized_scores = []
        
            for e in self.cell_scores:
                self.normalized_scores.append(e / self.total_score)
            
            self.start_number_of_emptys = len(self.empty_cells)
            print("EMPTYS: ", self.start_number_of_emptys)
        else:
            print("NO EMPTYS!") 
            
                    
    def sampleFromEmptys(self) -> Tuple[list, int]:
        """
        A method which returns a random sample from the set self.empty_cells based
        on the sampling probabilities contained in self.normalized_scores

        Returns:
            sample (list):
                A list containing x and y coordinates for a point
            index (int):
                The index of the entry sampled in self.empty_cells list
        """
        [[x, y, s]] = random.choices(self.empty_cells, self.normalized_scores)
        index = self.empty_cells.index([x, y, s])
        sample = [x, y]
        return sample, index
    
        
    def __pointInMap(self, point: list) -> int:
        """
        A method to retrieve the data contained in the map to plan in, because X and Y coordinates are flipped compared to the real map
        
        Args:
            point (list):
                The coordinates of the point to return the value in the occupancy map for

        Returns:
            map_value (int):
                The value contained in the map at those coordinates
        """
        map_value = self.map[round(point[1]), round(point[0])]
        return map_value


    def getPathDistance(self) -> float:
        return self.pathDistance


    def getNumberOfWaypoints(self) -> int:
        return self.numberOfWaypoints


    def getIterationsDone(self) -> int:
        return self.iterations_completed


    def plotWaypoints(self, path) -> None:
        #for i in range(len(path)-1):
            #plt.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], 'ro', markersize=10, linestyle="-")
            #plt.pause(0.1)
        xs = []
        ys = []
        for p in self.waypoints:
            xs.append(p[0])
            ys.append(p[1])
        plt.plot(xs, ys, 'ro', linewidth=3, linestyle="-")


    def parentFirstPointThen(self) -> Tuple[list, int]:
        parent = random.choice(self.node_collection)
        angle = random.random()*2*np.pi
        x = self.stepDistance * np.cos(angle)
        y = self.stepDistance * np.sin(angle)
        sample = np.array([parent.x + x, parent.y + y])
        return sample, parent


    def findNeighbours(self, p, radius) -> list:
        r = radius
        search_space = circle(p[0], p[1], r)
        n = self.__findNeighboursInQuadTree(search_space)
        neighbour_nodes = []
        
        for neighbour in n:
            neighbour_nodes.append(neighbour.trait)
        return neighbour_nodes


    def findFixedNumberOfNeighbours(self, p: list) -> list:
        """
        A method to find a fixed number of neighbours to a given point

        Args:
            p (list):
                The reference point to find neighbours for

        Returns:
            (list):
                A list of neighbour nodes of obstacles class NodeOnTree
        """
        r = self.stepDistance
        search_space = circle(p[0], p[1], r)
        n = self.__findNeighboursInQuadTree(search_space)
        neighbour_nodes = []
        
        if len(self.node_collection) > self.max_neighbour_found:
            while len(n) < self.max_neighbour_found:
                r += self.search_radius_increment*self.stepDistance
                search_space = circle(p[0], p[1], r)
                n = self.__findNeighboursInQuadTree(search_space)

        for neighbour in n:
            neighbour_nodes.append(neighbour.trait)
        return neighbour_nodes


    def __neighbourCount(self, p) -> Tuple[float, list]:
        radius = self.stepDistance*0.99
        n = self.findNeighbours(p, radius)
        return len(n), n
    
    
    def plotAllPaths(self, root_of_tree) -> None:
        self.colour_mode_2 = False
        if not root_of_tree:
            return

        current_children = root_of_tree.children
        for child in current_children:
            if child in self.node_collection:
                self.__harryPlotter(root_of_tree, [child.x, child.y])
            self.plotAllPaths(child)


    def __harryPlotter(self, start, end) -> None:
        #plt.pause(0.01)
        plt.plot([start.x, end[0]], [start.y, end[1]], self.plot_col, markersize=0.1, linestyle=":")


    def makeHeritage(self, parent: NodeOnTree, sample: list, index_of_empty: int, parent_distance: int) -> NodeOnTree:
        """
        A method to figure out the heritage of a new point and a parent candidate

        Args:
            parent (NodeOnTree):
                The upcoming parent of the child
            sample (list):
                The coordinates for a new child
            index_of_empty (int):
                The list index of the new child from the list of self.empty_cells

        Returns:
            (NodeOnTree): 
                The newly created child node
        """
        x = sample[0]
        y = sample[1]
        index = index_of_empty
        '''if (x == self.goal.x) and (y == self.goal.y):
            child = self.goal'''
        #else: # create a new node
        child = NodeOnTree(x, y) 
        self.node_collection.append(child)
            
        self.adoptChild(parent, child)
        # Sort newly created node into Quadtree dataset
        self.qt.insert(point(child.x, child.y, child))
        
        child.d_parent = parent_distance
        child.d_root = parent.d_root + parent_distance
        
        # Delete this successfull sample from Emptys to prevent re-sampling
        del self.empty_cells[index]
        del self.normalized_scores[index]
        del self.cell_scores[index]
        
        return child


    def adoptChild(self, parent: NodeOnTree, child: NodeOnTree) -> None:
        """
        A method to assign a child to a specific parent
        
        Args:
            parent (NodeOnTree):
                The new parent of the child
            child (NodeOnTree):
                The new child of the parent
        """
        parent.children.append(child)
        child.parent = parent


    def collision(self, start: list, end: list, distance2check: int) -> bool:
        """
        Check for any collision between two points

        Args:
            start (list):
                The starting point
            end (list):
                The end point
            distance2check (int):
                The distance between these points to iterate along

        Returns:
            (bool):
                A flag which is true if collision occured, and false if not
        """
        # Terminate early if leads into obstacle
        if self.__pointInMap(end) > 0: #self.map[round(end[1]), round(end[0])]
            return True
        v_hat = self.__unitVector(start, end)
        point = np.array([0.0, 0.0])
        for i in range(distance2check):
            # Walk along the v_hat direction, in entire integers
            point[0] = round(start[0] + i * v_hat[0])
            point[1] = round(start[1] + i * v_hat[1])

            if point[0] >= self.max_y or point[1] >= self.max_x:
                return True

            if  self.__pointInMap(point) > 0:  # Obstacles are True in Map, free nodes are False
                # self.map[round(point[1]), round(point[0])]
                return True
        return False


    def __outOfBoundsCorrection(self, point) -> list:
        if round(point[0]) >= self.max_y:  # Check X coord
            point[0] = self.max_y - 1

        if round(point[1]) >= self.max_x:  # Check Y coord
            point[1] = self.max_x - 1
        if round(point[0]) <= 0:  # Check X coord
            point[0] = 0

        if round(point[1]) <= 0:  # Check Y coord
            point[1] = 0
        return point


    def __unitVector(self, start: list, end: list) -> list:
        """
        A method to compute the unit vector of two given points

        Args:
            start (list):
                The first point
            end (list):
                The second point

        Returns:
            normalized_vector (list):
                The normalized unit vector
        """
        x = end[0] - start[0]
        y = end[1] - start[1]
        if x == 0 and y == 0:
            return np.array([0, 0])
        vector = np.array([x, y])
        # Return normalized vector
        normalized_vector = vector / np.linalg.norm(vector)
        
        return normalized_vector


    def findNearest(self, point: list) -> NodeOnTree:
        """
        A method to find the closest node to a given point in a quadtree

        Args:
            point (list):
                The reference point to look from containing (x,y)

        Returns:
            (NodeOnTree):
                The closest node to the reference point
        """
        x = point[0]
        y = point[1]
        r = 2* self.stepDistance
        r_outer = r
        # Initial search space to look in
        search_space = circle(x, y, r)
        neighbours = self.__findNeighboursInQuadTree(search_space)

        # If no neighbours where found, increase search range
        while not neighbours:
            r_outer += r
            # search_space = halo(x, y, r_inner, r_outer)
            search_space = circle(x, y, r_outer)
            neighbours = self.__findNeighboursInQuadTree(search_space)

        return self.__findNearestInSet(neighbours, point)
    
    
    def __findNeighboursInQuadTree(self, search_space: circle) -> list:
        """
        A method which utilizes the quadtrees search functionality to return all entries within the search space

        Args:
            search_space (circle):
                The geometric search space to search for points in

        Returns:
            (list):
                A list of all entries found, entries are of Quadtree object type Point
        """
        neighbours = []
        return self.qt.finder(search_space, neighbours)


    def __findNearestInSet(self, neighbours: list, point: list) -> NodeOnTree:
        """
        A method to find the nearest neighbour to a given point in a given set of neighbours

        Args:
            neighbours (list):
                A list of neighbours, of quadtree object type Point
            point (list):
                The relative point to look neighbours for

        Returns:
            (NodeOnTree):
                The closest node to the reference point
        """
        best_distance = self.maxDistance
        best_node = None
        for candidate in neighbours:
            candidate_node = candidate.trait
            candidate_distance = self.distance([candidate_node.x, candidate_node.y], [point[0], point[1]])
            
            if candidate_distance <= best_distance:
                best_distance = candidate_distance
                best_node = candidate.trait
        return best_node


    def __findNearestNode(self, point) -> None:
        for node in self.node_collection:
            p1 = [node.x, node.y]
            d = self.distance(p1, point)
            if d <= self.shortestDistance:
                self.shortestDistance = d
                self.nearest_node = node
    

    def distance(self, nodel: list, point: list) -> float:
        """
        A method to calculate the Euclidean distance between two given points

        Args:
            nodel (list):
                The first point
            point (list):
                The second point

        Returns:
            (float)):
                The Euclidean distance between the two points
        """
        return np.sqrt((nodel[0] - point[0]) ** 2 + (nodel[1] - point[1]) ** 2)


    def jackpot(self, point: list) -> bool:
        """
        A method to check if a given point has collision-free line-of-sight of the goal point

        Args:
            point (list):
                The point to check for

        Returns:
            (bool):
                A flag which is true if the point had line-of-sight and false if not
        """
        #if self.distance(self.goal, point) <= self.stepDistance and not self.collision(self.goal, point, self.stepDistance):
        
        if not self.collision([self.goal.x, self.goal.y], point, round(self.distance([self.goal.x, self.goal.y], point))):
            return True
        return False
    

    def __resetNearest(self) -> None:
        self.nearest_node = None
        self.shortestDistance = self.maxDistance
    

    def retracePath(self, goal: NodeOnTree) -> None:
        """
        A method to retrace from a given node back to the root of the tree

        Args:
            goal (NodeOnTree):
                The start node to iterate back to the root from
        """
        self.waypoints.insert(0, np.array([goal.x, goal.y]))
        current_parent = goal.parent
        while current_parent.parent:
            self.numberOfWaypoints += 1
            point = np.array([current_parent.x, current_parent.y])
            self.pathDistance += self.distance([current_parent.x, current_parent.y], self.waypoints[0])
            self.waypoints.insert(0, point)
            current_parent = current_parent.parent
                

    def __findEmpty(self, the_quad_tree: Quadtree) -> list:
        """
        A method to find all empty cells in a given quadtree

        Args:
            the_quad_tree (Quadtree):
                The quadtree to search through

        Returns:
            (list):
                The list of coordinates of the centrepoints of all empty cells in quadtree
        """
        empty = []
        # Bladerunner style zoom-in on map
        self.__enhance(the_quad_tree, empty)
        return empty  
    

    def __enhance(self, cell: rectangle, empty: list) -> None:
        """
        A method to recursively search through the quadtree and store emptycells centre coordinates in a list.
        Name is based on the "enhance scene" from the movie Bladerunner, 1982

        Args:
            cell (rectangle):
                The current rectuangular cell which is checked
            empty (list):
                The list containing all centre coordinates of all empty cells found
        """
        if not cell.points:
            x = cell.boundary.x
            y = cell.boundary.y
            w = cell.boundary.w 
            h = cell.boundary.h
            if round(x) < self.max_x and round(y) < self.max_y and round(x) > 0 and round(y) > 0:
                # Y and X are swapped in this case because they where sorted swapped into the Quadtree to begin with
                '''
                Statement excludes cells with certain conditions like too small or suspiciously large cells
                '''
                if not self.__pointInMap([y, x]) and w*h > self.min_cell_allowed and w*h < self.max_cell_preferred: #and w >= self.stepDistance
                    
                    empty.append([y, x, w*h])
                    
                    if self.plot_enabled:
                        # Flipp flop the coordinates and rectangle dimensions for map plotting
                        a = y
                        b = x
                        c = h
                        d = w
                        #plt.pause(0.01)
                        plt.plot([a - c, a + c], [b + d, b + d], 'm', linestyle="--") #nw->ne
                        plt.plot([a + c, a + c], [b + d, b - d], 'm', linestyle="--") # ne->se
                        plt.plot([a + c, a - c], [b - d, b - d], 'm', linestyle="--") # se->sw
                        plt.plot([a - c, a - c], [b - d, b + d], 'm', linestyle="--") # sw->nw'''
            
        else:
            
            if cell.has_been_divided:
                self.__enhance(cell.northeast, empty)
                self.__enhance(cell.northwest, empty)
                self.__enhance(cell.southeast, empty)
                self.__enhance(cell.southwest, empty)


    def bezier_curve(self, points: np.ndarray, t: float) -> list:
        """
        A method to create a Bezier curve from a given set of control points

        Args:
            points (np.ndarray):
                The control points for the Bezier curve
            t (np.ndarray):
                An incremental position along the curve, between 0 and 1 

        Returns:
            (list):
                The Bezier curves points
        """
        N = len(points)
        curve = np.zeros(2)
        total = 0.0
        
        for i in range(N):
            weight = (t ** i) * ((1 - t) ** (N - 1 - i))
            total += weight
            curve += points[i] * weight
            
        if total != 0.0:
            curve /= total
        return curve       


    def bezier_tripples(self, path: list, max_turning_angle_allowed: int) -> list:
        """
        A method to utilize Bezier curve to smooth a path if the angle of change between points are greater than
        a specified maximum turning angle
        Args:
            path (list):
                A list of coordinates to follow along and to check
                
            max_turning_angle_allowed (int):
                The maximum turning angle allowed before Bezier-smoothing is applied

        Returns:
            (list):
                A smoothed list of coordinates 
        """
        # Double the waypoints in order to strengthen the control points for the Bezier
        path = self.interpolate_waypoints(path)
        curve = []
        t_values = np.linspace(0, 1, 10)
        i = 1
        while i < len(path)-1:
            p2 = path[i]
            p1 = [(path[i-1][0]+p2[0])/2 , (path[i-1][1]+p2[1])/2 ]
            p3 = [(path[i+1][0]+p2[0])/2 , (path[i+1][1]+p2[1])/2 ]
            first_angle = np.rad2deg(np.arctan2(p1[0]-p2[0], p1[1]-p2[1]))
            next_angle = np.rad2deg(np.arctan2(p2[0]-p3[0], p2[1]-p3[1]))
            angle_change = abs(first_angle - next_angle)
            i += 1
            # If change in angle exceeds limit, then smooth with Bezier
            if angle_change > max_turning_angle_allowed:
                points = np.asfortranarray([p1, p2, p3])
                subcurve = np.array([self.bezier_curve(points, t) for t in t_values])
                subcurve = subcurve[:-1]
                curve.extend(subcurve)
            # Else append regular waypoint
            else:
                curve.append(p1)
        curve.insert(0, path[0])
        curve.append(path[len(path)-1])
        #print(curve)
        return curve
    

    def interpolate_waypoints(self, waypoints: list) -> list:
        """
        A method to add all middle-points of all points in the list.
        Utilizing the Middle Point Formula

        Args:
            waypoints (list):
                The list of points to add middle-points to

        Returns:
            (list):
                The interpolated list
        """
        wp = waypoints
        curve = []
        for i in range(len(wp)-1):
            a = wp[i]
            c = wp[i+1]
            b = [(a[0] + c[0]) / 2, (a[1] + c[1]) / 2]
            curve.extend([a, b])
        curve.append(wp[-1])
        return curve
    
    def step_by_step_interpolation(self, path: list) -> list:
        """
        Interpolates all given points into a series of close-to-ones

        Args:
            path (list): 
                A series of points given

        Returns:
            list: 
                An interpolated list of points
        """
        curve = []
        i = 1
        for i in range(len(path)-1):
            p1 = path[i]
            p2 = path[i+1]
            d = round(np.floor(self.distance(p1, p2)))
            x_s = np.linspace(p1[0], p2[0], d)
            y_s = np.linspace(p1[1], p2[1], d)
            for j in range(len(x_s)):
                curve.append([x_s[j], y_s[j]])
        return curve


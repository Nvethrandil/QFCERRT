#!/usr/bin/python3

__author__ = 'Noah Otte <nvethrandil@gmail.com>'
__version__= '3.1'
__license__= 'MIT'

from typing import Tuple
import time
# Essential python libraries
import numpy as np
import random
# Libraries for plotting
import matplotlib.pyplot as plt
# Essential special libraries
from skimage.measure import find_contours
from scipy.ndimage import binary_dilation, gaussian_filter
from scipy.interpolate import CubicSpline, interp1d

# Essential custom imports
from .QT import QuadTree as Quadtree
from .QT import Circle as circle
from .QT import Rectangle as rectangle
from .QT import Point as point
from .NodeOnTree import NodeOnTree as NodeOnTree

class QFCERRT:
    """
    A class used to compute a basic RRT search algorithm utilizing quadtree map tesselation
    """
    def __init__(self, map: np.ndarray, 
                 start: np.ndarray, 
                 goal: np.ndarray, 
                 max_iterations: int, 
                 stepdistance: int, 
                 plot_enabled: bool, 
                 max_neighbour_found: int, 
                 bdilation_multiplier: int, 
                 cell_sizes: list, 
                 mode_select: int,
                 danger_zone: int,
                 fov: int
                 ) -> None:
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
                Step distance of RRT algorithm, here used as area indicator of minimum quadtree cell area (stepdistance^2)
            plot_enabled (bool):
                Flag to enable plotting mode or not
            max_neighbour_found (int):
                Minimum neighbours to be found before stopping search, only applies after that many neighbours also exist
            bdilation_multiplier (int):
                Amount of pixels to bloat each object in map
            cell_sizes (list):
                Contains the multiplier of preferred cell size and multiplier of preferred  max cell according to:
                cell_size = [A, B]
                pref_cell_size = (stepdistance * stepdistance) * A
                max_cell_size = pref_cell_size * B
            mode_select (int):
                0 - nothing is processed
                1 - interpolate points to stepsize of ~3
                2 - only smooth turns using bezier
            danger_zone (int):
                An integer distance (in pixels) at which detected collisions will trigger replanning
            fov (int):
                The maximum turning angle which planne paths can have -> the max field-of-view of the randomized points
        """
        
        # to make obstacles bigger than they actually are; binary dilation
        bdil_t1 = time.process_time()
        self.mode = mode_select
        self.bd_multi = bdilation_multiplier
        self.map = map #binary_dilation(map, iterations=self.bd_multi).astype(bool)
        self.bdil_time =  time.process_time() - bdil_t1
        self.max_x = self.map.shape[0]
        self.max_y = self.map.shape[1]
        self.map_center_x = self.max_x / 2
        self.map_center_y = self.max_y / 2
        print(f'Planner Message: recieved a {self.max_x}x{self.max_y} sized map.')
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
        # add the start, the root, to the tree only
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
        self.fov = (fov / 2) * np.pi / 180
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
        self.plot_col = 'wo'
        self.qt_color = 'magenta'
        self.info_string = f"Planner Message: cells larger than {self.min_cell_preferred} are weighted {self.extra_cell_weight}x higher"
        # actually build the quadtree on the obstacles for free-space calculation
        t1 =  time.process_time()
        bound = rectangle(x, y, square, square)
        bound2 = rectangle(self.map_center_y, self.map_center_x, self.max_y / 2 -1, self.max_x / 2 -1)

        self.qt_map = Quadtree(bound2, 1)
        # add the goal for many viable sampling points
        #self.qt_map.insert(point(self.goal.y, self.goal.x, None))
        '''self.qt_map.insert(point(self.goal.y +1, self.goal.x, None))
        self.qt_map.insert(point(self.goal.y -1, self.goal.x, None))
        self.qt_map.insert(point(self.goal.y, self.goal.x +1, None)) 
        self.qt_map.insert(point(self.goal.y, self.goal.x -1, None))  
        self.qt_map.insert(point(self.goal.y +1, self.goal.x+1, None))
        self.qt_map.insert(point(self.goal.y -1, self.goal.x-1, None))
        self.qt_map.insert(point(self.goal.y-1, self.goal.x +1, None)) 
        self.qt_map.insert(point(self.goal.y+1, self.goal.x -1, None)) '''
        # keep the time it took for computation reference
        self.quadtree_time = time.process_time() - t1
        # retrieve the actual free space cells in quadtree
        t_empty = time.process_time()
        self.__processFreeSpace(self.map, self.qt_map)
        self.empty_time = time.process_time() - t_empty
        # replanning settings
        self.danger_zone = danger_zone

            
    def search(self) -> list:
        """
        A method which performs RRT-search

        Returns:
            (list): 
                The path found by RRT, or [-1] if no path was found
        """
        searchtime_start =  time.process_time()
        i = 0
        # early exit in case goal is in line-of-sight
        if self.jackpot([self.tree.x, self.tree.y]) is True:
            print(f'Planner Message: goal found at once')
            self.waypoints.insert(0, self.start)
            self.waypoints.append([self.goal.x, self.goal.y])
            self.iterations_completed = 0
            self.search_time =  time.process_time() - searchtime_start
            return self.waypoints
                    
        # start searching    
        while i < self.maxit and self.empty_cells:
            i += 1  
            # perform sampling process
            sample, index = self.sampleFromEmptys()
            parent = self.sort_collection_for_nearest(sample)
            distance = round(self.distance([parent.x, parent.y], sample))
            # check if sample worked
            if not self.collision([parent.x, parent.y], sample, distance):
                child = self.makeHeritage(parent, sample, index, distance)
                # check if sample found goal
                if self.jackpot(sample) is True:
                    print(f'Planner Message: goal found at {i} iterations')
                    # exit procedures for found goal
                    self.adoptGoal(child)
                    self.goal.d_parent = self.distance(sample, [self.goal.x, self.goal.y])
                    self.goal.d_root = child.d_root + distance
                    self.retracePath(self.goal)
                    self.waypoints.insert(0, self.start)
                    self.apply_post_process
                    self.iterations_completed = i
                    self.search_time =  time.process_time() - searchtime_start
                    return self.waypoints
                    
        # save parameters for failed search                
        self.iterations_completed = i
        self.search_time =  time.process_time() - searchtime_start
        print(f'Planner Message: goal not found after {i} iterations complete')
        # return failure value -1
        return [-1]
    
    def GB_FOV_sampler(self):
        valid_angle = []     
        while not valid_angle:
            
            direction_sample, index = self.sampleFromEmptys() 
        
            for node in self.node_collection:
                # include root of tree every time
                if self.within_FOV(direction_sample, node, self.fov) and self.distance([node.x, node.y], direction_sample) < 2*self.max_size:
                    valid_angle.append(node)
            
        valid_angle.sort(key=lambda e: self.weighted_distance([e.x, e.y], direction_sample), reverse=False)
        parent = valid_angle[0]
        neighbours = valid_angle[1:]
        sample = direction_sample #self.traveller([parent.x, parent.y], direction_sample)
        
        return sample, parent, index, neighbours
    
    def within_FOV(self, point, potential_parent_node, fov):
        node = potential_parent_node
        # include root of tree every time
        if not node.parent and self.weighted_distance(point, [node.x, node.y]) < self.max_x: #OOGA BOOGA
            return True
        
        if not node.parent:
            return False
        else:
            # calculate the angle between 3 points
            a = np.array(point)
            b = np.array([node.x, node.y])
            c = np.array([node.parent.x, node.parent.y])

            ba = a - b
            bc = c - b
            try:
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)
            except:
                return False
            
            # check if the points would be in range of the FOV
            boundary = np.pi - fov
            # determine angle to the point is goal-biased
            ga = self.point2goalFOV(point, node)
            if abs(angle) >= boundary and angle < ga + np.pi/2 and angle > ga - np.pi/2:
                return True
            return False
        
    def point2goalFOV(self, point, node):
        # calculate the angle between 3 points
        a = np.array([node.x, node.y])
        b = np.array(point)
        c = np.array([self.goal.x, self.goal.y])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        
        return angle

    
    def need2replan(self, new_position: list, new_map: np.ndarray) -> bool:
        """
        Check if the planner needs to perform replanning or not

        Args:
            new_position (list): 
                The current position of the vehicle
            new_map (np.ndarray):
                The new map which might have changed from previous planning

        Returns:
            (bool):
                True if replanning is necessary, False if not
        """
        # Early exits
        if not self.waypoints:
            return True
        
        if len(self.waypoints) == 1:
            return True
        
        # perform the same Binary Dilation as on the regular map
        self.map = binary_dilation(new_map, iterations=self.bd_multi).astype(bool)
        
        # Determine the closest point in the path to the actual vehicles position [DEPRICATED]
        '''temp_list = self.waypoints
        temp_list.sort(key=lambda e: self.distance(e, new_position), reverse=False)
        index = self.waypoints.index(temp_list[0])
        del self.waypoints[0:index]
        print(f'Planner Message: {len(self.waypoints)} waypoints in current path.')'''
        modlist = self.waypoints
        
        # If current position deviates more than 0.5 from expected [DEPRICATED]
        '''if self.distance(new_position, modlist[0]) > 0.5:
            modlist.insert(0, new_position)'''
        
        # Check collisions
        for i in range(len(modlist)-1):
            p1 = modlist[i]
            p2 = modlist[i+1]
            vehicle_distance = round(self.distance(new_position, p2))
            
            # if the point to check is within the danger_zone
            if vehicle_distance < self.danger_zone:
                # if a collision occurs somewhere in that range
                if self.collision(p1, p2, round(self.distance(p1, p2))):
                    print(f'Planner Message: replanning')
                    return True
            
        #self.waypoints = modlist
        return False
    
    
    def apply_post_process(self) -> None:
        """
        Performs post processing based on selected mode
        """
        if self.mode == 0:
            pass
        if self.mode == 1:
            self.waypoints = self.step_by_step_interpolation(self.waypoints)
        if self.mode == 2:
            p = self.bezier_tripples(self.waypoints, 10)
            #p = self.some_spline_chad(self.waypoints)
            self.waypoints = self.step_by_step_interpolation(p)
    
    def some_spline_chad(self, path):
        '''
        Parametrize x and y over an imaginary third variable, imaginary_time, in order to run 2 interpolations over that 
        linearly increasing variable and then combine the x and y from the results to the final curve,
        which then does not have to be linearly increasing but can be non-monotonic
        '''
        #path = self.interpolate_linearly(path)
        x_s = []
        y_s = []
        for p in path:
            x_s.append(p[0])
            y_s.append(p[1])
            
        imaginary_time = np.linspace(0,1, len(x_s))
        r = path
        curve = CubicSpline(imaginary_time, r, axis=0, bc_type='clamped') #not-a-knot
        s2g = round(np.floor(self.distance([self.tree.x, self.tree.y], [self.goal.x, self.goal.y]) / 2))
        gd = s2g # determines the amount of points in the spline
        t = np.linspace(np.min(imaginary_time),np.max(imaginary_time),gd)
        r = curve(t)
        return r
    def some_other_spline_chad(self, path):
        x_s = []
        y_s = []
        for p in path:
            x_s.append(p[0])
            y_s.append(p[1])
            
        imaginary_time = np.linspace(0,1, len(x_s))
        r = path
        s2g = round(np.floor(self.distance([self.tree.x, self.tree.y], [self.goal.x, self.goal.y]) / 2))
        curve = interp1d(imaginary_time, r, kind='quadratic', axis=0)
        t = np.linspace(np.min(imaginary_time),np.max(imaginary_time),s2g)
        r = curve(t)
        return r
       
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
        self.node_collection.sort(key=lambda e: self.weighted_distance([e.x, e.y], p), reverse=False) #GAUSS
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
            self.max_size = np.sqrt(max(all_wh_values))
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
            print(f'Planner Message: planner has {self.start_number_of_emptys} cells to plan in')
            
        else:
            print("Planner Message: no cells to plan in") 
            
                    
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
        [[x, y, s]] = random.choices(self.empty_cells)#, self.normalized_scores)
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
        
        child = NodeOnTree(x, y) 
        self.node_collection.append(child)
        self.adoptChild(parent, child)
        
        # sort newly created node into Quadtree dataset
        self.qt.insert(point(child.x, child.y, child))
        
        child.d_parent = parent_distance
        child.d_root = parent.d_root + parent_distance
        index = index_of_empty
        if index == -1:
            return child
        else:
            # delete this successfull sample from Emptys to prevent re-sampling
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
        # terminate early if leads into obstacle
        if self.__pointInMap(end) > 0:
            return True
        
        v_hat = self.__unitVector(start, end)
        point = np.array([0.0, 0.0])
        
        for i in range(distance2check):
            # walk along the v_hat direction, in entire integers
            point[0] = round(start[0] + i * v_hat[0])
            point[1] = round(start[1] + i * v_hat[1])

            # check if out-of-bounds for the map
            if point[0] >= self.max_y or point[1] >= self.max_x:
                return True
            
            # obstacles are True in Map, free nodes are False
            if  self.__pointInMap(point) > 0:
                return True
            
        return False


    def __outOfBoundsCorrection(self, point) -> list:
        # check X coord
        if round(point[0]) >= self.max_y:  
            point[0] = self.max_y - 1
            
        # check Y coord
        if round(point[1]) >= self.max_x:  
            point[1] = self.max_x - 1
        
        # check X coord
        if round(point[0]) <= 0:  
            point[0] = 0
            
        # check Y coord
        if round(point[1]) <= 0:
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
        
        if not self.collision([self.goal.x, self.goal.y], point, round(self.distance([self.goal.x, self.goal.y], point))) and self.within_FOV(point, self.goal, self.fov): #self.distance([self.goal.x, self.goal.y], point) <= 20:#
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
            self.pathDistance += self.weighted_distance([current_parent.x, current_parent.y], self.waypoints[0])
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
                if self.valid_empty(x, y, w, h): #and w >= self.stepDistance
                    
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

    def valid_empty(self, x, y, w, h):
        if not self.__pointInMap([y, x]) and w*h > self.min_cell_allowed and w*h < self.max_cell_preferred:
            return True
        else:
           return False

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
        # interpolate the waypoints in order to strengthen the control points for the Bezier
        #path = self.step_by_step_interpolation(path)
        curve = []
        t_values = np.linspace(0, 1, 5)
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
    
    def bezier_entire_curve(self, path):
         #p = self.step_by_step_interpolation(path)
         p = path
         p_forttran =  np.asfortranarray(p)
         t_values = np.linspace(0, 1, round(len(p)))
         b_path = np.array([self.bezier_curve(p_forttran, t) for t in t_values])
         return b_path
     
    
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
        Interpolates all given points into a series of close-to-three pixels distance apart

        Args:
            path (list): 
                A series of points given

        Returns:
            (list): 
                An interpolated list of points
        """
        curve = []
        i = 1
        for i in range(len(path)-1):
            p1 = path[i]
            p2 = path[i+1]
            # get the amount of 3-pixel steps the distance is dividable by
            d = round(np.floor(self.distance(p1, p2)/3))
            if d < 3:
                #pass
                curve.append([p1[0], p1[1]])
            else:
                x_s = np.linspace(p1[0], p2[0], d)
                y_s = np.linspace(p1[1], p2[1], d)
                for j in range(len(x_s)):
                    curve.append([x_s[j], y_s[j]])
        return curve
    
    def weighted_distance(self, start, end):
        '''
        Weighted function for Gaussed map, WORK IN PROGRESS
        '''
        d = round(np.floor(self.distance(start, end)))
        d_weighted = d
        if d > 1:
            x_s = np.linspace(start[0], end[0], d)
            y_s = np.linspace(start[1], end[1], d)
            
            for i in range(len(x_s)):
                d_weighted += 100 * self.costmap[round(y_s[i]), round(x_s[i])]   
            return d_weighted
        else:
            # just add the middle point cost
            d_weighted += 50 * self.costmap[round((start[1] + end[1])/2), round((start[0] + end[0])/2)]
            return d_weighted
    
    def create_costmap(self):
        '''
        Creates a weighted costmap through a Gaussian filter
        '''
        grid = binary_dilation(self.map, iterations=self.bd_multi).astype(bool) #self.map
        cost_h = 10
        std_div = 5
        gaussed_grid = np.where(grid > 0, cost_h, grid)
        gaussed_grid = gaussian_filter(gaussed_grid, sigma=std_div, mode='wrap')
        return gaussed_grid


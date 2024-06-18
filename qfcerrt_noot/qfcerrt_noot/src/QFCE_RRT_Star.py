#!/usr/bin/python3

__author__ = 'Noah Otte <nvethrandil@gmail.com>'
__version__= '2.6'
__license__= 'MIT'

# General Python libraries
import random
import time
from matplotlib import pyplot as plt
import numpy as np
# My own libraries
from .QFCE_RRT import QFCERRT
from .NodeOnTree import NodeOnTree


class QFCERRTStar(QFCERRT):
    """
    A method to perform an RRT* search utilizing quadtree map tesselation

    Args:
        RRT (RRT):
            The parent class for the RRT algorithm, containing all methods utilized here
    """
    def __init__(self, map: np.ndarray, 
                 start: np.ndarray, 
                 goal: np.ndarray, 
                 max_iterations: int, 
                 stepdistance: int, 
                 plot_enabled: bool, 
                 neighbour_radius: int, 
                 search_radius_increment: float, 
                 max_neighbour_found: int, 
                 bdilation_multiplier: int, 
                 cell_sizes: list, 
                 mode_select: int
                 ):
        """
        Init method for RRT* class

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
            neighbour_radius (int):
                The initial neighbour radius used for search
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
        """
        
        super().__init__(map, start, goal, max_iterations, stepdistance, plot_enabled, search_radius_increment, max_neighbour_found, bdilation_multiplier, cell_sizes, mode_select)
        self.neighbour_radius = neighbour_radius 
        self.best_distance = None
        # flag to keep track if the goal was found
        self.goalWasFound = False

    def RRT_star_search(self):
        """
        A method to perform RRT* search utilizing quadtree map tesselation

        Returns:
            (list):
                A list of coordinates spanning a close-to-optimal path, or [-1] if no path was found
        """
        searchtime_start =  time.process_time()
        
        # check if the goal is in line-of-sight
        if self.jackpot([self.tree.x, self.tree.y]) is True:
            print(f'Planner Message: goal found at once')
            self.waypoints.insert(0, self.start)
            self.waypoints.append([self.goal.x, self.goal.y])
            self.iterations_completed = 0
            self.apply_post_process()
            self.search_time =  time.process_time() - searchtime_start
            return self.waypoints
        
        # determine the maximum number of iterations allowed
        if self.maxit > len(self.empty_cells):
            stop_at = len(self.empty_cells)
        else:
            stop_at = self.maxit

        i = 0
        # do while still cells have not been sampled successfully or until maximum iterations are reached
        while i <= self.maxit and len(self.empty_cells) > 1:
            i +=1
            # sample a point and get its parent-candidate
            sample, index = self.sampleFromEmptys()
            parent = self.sort_collection_for_nearest(sample)
            distance = round(self.distance([parent.x, parent.y], sample))
            
            # check if the pair worked, if so make a real node out of it and optimize
            if not self.collision([parent.x, parent.y], sample, distance):
                # declare the heritage between child and parent and find the childs neighbours
                child = self.makeHeritage(parent, sample, index, distance)
                neighbours =  self.node_collection[:self.max_neighbour_found]
                
                # if the goal was not found yet, check if it is found this time
                if not self.goalWasFound:
                    if self.jackpot(sample):
                        print(f'Planner Message: goal found at {i} iterations')
                        self.adoptGoal(child)
                        self.goal.d_parent = self.distance(sample, [self.goal.x, self.goal.y])
                        self.goal.d_root = child.d_root + distance
                        neighbours.append(self.goal)
                        
                # optimize the neighbours found
                if neighbours is not None:
                    self.__optimizeNeighbours(neighbours)
                    

        # exit procedures            
        if self.goalWasFound:
            self.retracePath(self.goal)
            self.waypoints.insert(0, self.start)
            self.apply_post_process()
        else:
            self.waypoints = [-1]
        self.iterations_completed = i
        self.search_time =  time.process_time() - searchtime_start
        
        # return waypoints
        return self.waypoints
    
    def __updateHeritage(self, new_parent: NodeOnTree, old_parent: NodeOnTree, child: NodeOnTree):
        """
        A method which changes the parent for a given child node
        
        Args:
            new_parent (NodeOnTree):
                The new parent node to connect the child to
            old_parent (NodeOnTree):
                The old parent to remove the child from
            child (NodeOnTree):
                The child whose heritage is updated
        """
        old_parent.children.remove(child)
        self.adoptChild(new_parent, child)

    def __optimizeNeighbours(self, set_of_neighbours: list):
        """
        A method to optimize the distance back to the root of the tree of all nodes contained in a given set of neighbours

        Args:
            set_of_neighbours (list):
                The list of neighbours to optimize
        """
        #set_of_neighbours.append(self.tree)
        for node in set_of_neighbours:
            subset = set_of_neighbours
            subset.remove(node)
            self.best_distance = node.d_root #self.__distance2Root(node)
            for subset_node in subset:
                if self.__isBetterDistance(node, subset_node):
                    self.__updateHeritage(subset_node, node.parent, node)

    def __isBetterDistance(self, potential_child: NodeOnTree, parent: NodeOnTree):
        """
        A method to check if a better distance to root is achieved if the given node is connected to the given parent, rather than its current parent

        Args:
            potential_child (NodeOnTree):
                The potential new child of the other node
            parent (NodeOnTree):
                The parent node to check the connection to

        Returns:
            (bool):
                A flag which is true if this was a better connection, and false if not
        """
        start_end_distance = self.distance([potential_child.x, potential_child.y], [parent.x, parent.y])
        end_root_distance = parent.d_root
        new_distance = start_end_distance + end_root_distance
        collide = self.collision([potential_child.x, potential_child.y], [parent.x, parent.y], round(start_end_distance))
        
        if new_distance < self.best_distance and not collide:
            self.best_distance = new_distance
            potential_child.d_root = new_distance
            potential_child.d_parent = start_end_distance
            return True
        else:
            return False
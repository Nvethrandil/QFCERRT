#!/usr/bin/python3

__author__ = 'Noah Otte <nvethrandil@gmail.com>'
__version__= '1.0'
__license__= 'MIT'

import numpy as npy
import matplotlib.pyplot as plt

# Inspired by The Coding Train's Coding Challenge 98.1-98.3
# Follows pseudocode from Wikipedia on QuadTree
class QuadTree():
    """
    A class for quadtree data structures.
    
    Inspired by The Coding Train's Coding Challenge 98.1-98.3
    Follows pseudocode from Wikipedia on QuadTree.
    """
    def __init__(self, boundary, capacity: int):
        """
        A method to initialize a quadtree

        Args:
            boundary (_type_): 
                A geometric object to serve as boundry condition for the tree
            capacity (int):
                The maximum amount of entries allowed in each cell of the quadtree
        """
        self.boundary = boundary
        self.capacity = capacity
        self.points = []
        self.has_been_divided = False
        # Children
        self.northeast = None
        self.northwest = None
        self.southeast = None
        self.southwest = None

        self.qt_color = 'orange'
    def subdivide(self):
        """
        A method which subdivides the quadtree into new quadtrees
        """
        # Get our coordinates
        x = self.boundary.x
        y = self.boundary.y
        w = self.boundary.w
        h = self.boundary.h
        '''
        Following code is utilized for plotting in the Y-X reversed plot
        a = y
        b = x
        c = h
        d = w
        #plt.pause(0.01)
        plt.plot([a - c, a + c], [b + d, b + d], self.qt_color, linestyle="--") #nw->ne
        plt.plot([a + c, a + c], [b + d, b - d], self.qt_color, linestyle="--") # ne->se
        plt.plot([a + c, a - c], [b - d, b - d], self.qt_color, linestyle="--") # se->sw
        plt.plot([a - c, a - c], [b - d, b + d], self.qt_color, linestyle="--") # sw->nw'''
        # Create new regions in our current region, more Rectangles()
        ne = Rectangle(x + w / 2, y - h / 2, w / 2, h / 2)
        nw = Rectangle(x - w / 2, y - h / 2, w / 2, h / 2)
        se = Rectangle(x + w / 2, y + h / 2, w / 2, h / 2)
        sw = Rectangle(x - w / 2, y + h / 2, w / 2, h / 2)
        # Each region becomes its own QuadTree
        self.northeast = QuadTree(ne, self.capacity)
        self.northwest = QuadTree(nw, self.capacity)
        self.southeast = QuadTree(se, self.capacity)
        self.southwest = QuadTree(sw, self.capacity)
        self.has_been_divided = True

    def insert(self, point):
        """
        A method which tries to insert a given point into the quadtree and which subdivides if necessary

        Args:
            point (_type_):
                The point object to sort into the tree

        Returns:
            (bool):
                Returns true if successfully stored and false if it wasnt possible
        """
        # If point is not within boundary, exit
        if not self.boundary.contains(point):
            return False
        # If not at max capacity, add point
        if len(self.points) <= self.capacity:
            self.points.append(point)
            return True
        # If region is not already divided, divide
        if not self.has_been_divided:
            self.subdivide()

        if self.northeast.insert(point) or self.northwest.insert(point) or self.southeast.insert(point) or self.southwest.insert(point):
            return True

    def finder(self, search_range, found_array: list):
        """
        A method which finds points recursively in the quadtree

        Args:
            search_range (_type_):
                The goemetric search space which defines in which boundries to search in
            found_array (list):
                A list in which to store all entries found

        Returns:
            (list):
                A list of all entries found
        """
        # If region does not intersect search_range, exit
        if not search_range.intersects(self.boundary):
            return found_array
        else:
            # Check points in current region
            for p in self.points:
                # Save them if they are within the search_range
                if search_range.contains(p):
                    found_array.append(p)
            if self.has_been_divided:
                # Concat another level in each quadrant, recursion with Finder()
                self.northwest.finder(search_range, found_array)
                self.northeast.finder(search_range, found_array)
                self.southwest.finder(search_range, found_array)
                self.southeast.finder(search_range, found_array)
        # Last exit of function, an array with all Point() found
        return found_array

class Point():
    """
    Point class used as a means to store information in the quadtree.
    More advanced information, like objects, can be stored as trait.
    """
    def __init__(self, x: float, y: float, trait):
        """
        Initialization method of Point class in quadtree

        Args:
            x (float):
                The x-coordinate of Point
            y (float): 
                The y-coordinate of Point
            trait (_type_): 
                Extra data that someone might want to store at given Point
        """
        self.x = x
        self.y = y
        self.trait = trait


class Circle():
    """
    A geometric object class, a circle. Can be used as boundary object for tree
    """
    def __init__(self, x: float, y: float, r: float):
        """
        Initialization function for Circle class in quadtree

        Args:
            x (float):
                The x-coordinate of the centre of the circle
            y (float):
                The y-coordinate of the centre of the circle
            r (float):
                The radius of the circle
        """
        self.x = x
        self.y = y
        self.r = r
        self.rSquared = self.r * self.r

    def contains(self, boundary):
        """
        A method to check if the object is contained within the given boundary object

        Args:
            boundary (_type_):
                The boundary to check against

        Returns:
            (bool):
                Returns true if the object is contained within the boundary
        """
        dx = abs(boundary.x - self.x)
        dy = abs(boundary.y - self.y)
        return self.rSquared >= (dx * dx + dy * dy)

    def intersects(self, boundary):
        """
        A method to check if the geometric object intersects with the given boundary

        Args:
            boundary (_type_):
                The boundary to check against

        Returns:
            (bool):
                Returns true if the object intersects with the boundary
        """
        dx = npy.sqrt((boundary.x - self.x) ** 2)
        dy = npy.sqrt((boundary.y - self.y) ** 2)
        r = self.r
        w = boundary.w
        h = boundary.h
        edges = (dx - w) ** 2 + (dy - h) ** 2
        if dx > (r + w) or dy > (r + h):
            return False
        if dx <= w or dy > (r + h):
            return True
        return edges <= self.rSquared

class Rectangle():
    """A geometric object class, a rectangle. Can be used as boundary object for tree
    """
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.w = width
        self.h = height

    def contains(self, boundary):
        """
        A method to check if the object is contained within the given boundary object

        Args:
            boundary (_type_):
                The boundary to check against

        Returns:
            (bool):
                Returns true if the object is contained within the boundary
        """
        # Intersection conditions 1-4
        c1 = boundary.x >= self.x - self.w
        c2 = boundary.x <= self.x + self.w
        c3 = boundary.y >= self.y - self.h
        c4 = boundary.y <= self.y + self.h
        # Check if point is within the boundary
        if c1 and c2 and c3 and c4:
            return True
        else:
            return False

    def intersects(self, boundary):
        """
        A method to check if the geometric object intersects with the given boundary

        Args:
            boundary (_type_):
                The boundary to check against

        Returns:
            (bool):
                Returns true if the object intersects with the boundary
        """
        # Intersection conditions 1-4
        c1 = boundary.x - boundary.w > self.x + self.w
        c2 = boundary.x + boundary.w < self.x - self.w
        c3 = boundary.y - boundary.h > self.y + self.h
        c4 = boundary.y + boundary.h < self.y - self.h
        if c1 and c2 and c3 and c4:
            return False
        else:
            return True

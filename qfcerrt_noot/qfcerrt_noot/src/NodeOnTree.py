#!/usr/bin/python3

__author__ = 'Noah Otte <nnvethrandil@gmail.com>'
__version__= '1.0'
__license__= 'MIT'

class NodeOnTree():
    """
    A class used for nodes of entries in RRT-trees
    """
    def __init__(self, x: float, y: float):
        """
        Init method for Nodes in an RRT-tree data structure.
        Contains the coordinates (x,y), a list of children nodes and an entry for parent node

        Args:
            x (float):
                The X coordinate of the node
            y (float):
                The Y coordinate of the node
        """
        self.x = x
        self.y = y
        self.children = []
        self.parent = None
        # Distances
        self.d_parent = None
        self.d_root = None
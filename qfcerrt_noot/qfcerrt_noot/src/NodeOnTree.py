#!/usr/bin/python3

__author__ = 'Noah Otte <nvethrandil@gmail.com>'
__version__= '1.1'
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
        # the coordinates of this node's position
        self.x = x
        self.y = y
        # list of all children to this node
        self.children = []
        # the parent of this node
        self.parent = None
        # distance to this node's parent node
        self.d_parent = None
        # distance to the root along the tree from this node
        self.d_root = None
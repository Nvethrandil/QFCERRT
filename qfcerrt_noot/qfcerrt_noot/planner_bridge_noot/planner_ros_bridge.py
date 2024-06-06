#!/usr/bin/python3

__author__ = 'Noah Otte <nvethrandil@gmail.com>'
__version__= '0.1'
__license__= 'MIT'

# Ros libraries
import rospy
import roslib
# Python libraries
import numpy as np
import time
from numpy_ros import to_numpy, to_message
# My libraries
from qfcerrt_noot.src.QFCE_RRT import QFCERRT as Planner
import path_utilities as PU
# Ros messages
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import Pose, Point, PoseStamped


# Planner to ROS System bridge
class PlannerROSBridge():
    def __init__(self, 
                 map_id: str, 
                 robotpose_id: str,
                 goal_id: str,
                 traversability_upper_boundary: int, 
                 unknown_are: int,
                 safety_buffer_in_meters: float
                 ):
        
        # Placeholders
        self.VERBOSE = False
        self.robo_coords = None
        self.goal_coords = None
        self.height_map = None
        self.mapUGV = None
        self.raw_occmap = None
        self.latest_path = None
        # Subscriber IDs
        self.map_id = map_id
        self.robotpose_id = robotpose_id
        self.goal_id = goal_id
        # Planning parameters
        self.traversability_upper_boundary = traversability_upper_boundary
        self.unknown_are = unknown_are
        self.safety_buffer = safety_buffer_in_meters
        
    def callback_pose(self, data: Pose):
        if self.VERBOSE:
            print('recieved posedata of type: "%s' % data.format)
            
        self.robo_coords = [data.pose.pose.position.x, data.pose.pose.position.y]
    
    def callback_goal(self, msg: Point):
        if self.VERBOSE:
            print('recieved posedata of type: "%s' % msg.format)
            
        self.goal_coords = [msg.point.x, msg.point.y]    
        
    def callback_map(self, msg: OccupancyGrid):
            if self.VERBOSE:
                print('recieved mapUGV of type: "%s' % msg.format)

            # Convert Occupancy grid into Numpy Array
            mapUGV =  PU.occupancygrid_to_numpy(msg)
            trav_bound_upper = self.traversability_upper_boundary
            unknown_are = self.unknown_are
            # everything above upper is 1 == nontraversable
            mapUGV = np.where(mapUGV >= trav_bound_upper, 1, mapUGV) 
            # unkown pixels are set to unknown_are value
            mapUGV = np.where(mapUGV == -1, unknown_are, mapUGV) 
            mapUGV = np.where(mapUGV > 1, 1, mapUGV)
            
            self.mapUGV = mapUGV
            self.raw_occmap = msg
    
    def subscribe_to_topics(self):
        # Subscribe to relevant topics
        self.subscriber_pose = rospy.Subscriber(self.robotpose_id, Odometry, self.callback_pose)
        self.subscriber_goal = rospy.Subscriber(self.goal_id, Odometry, self.callback_goal)
        self.subscriber_map = rospy.Subscriber(self.map_id, OccupancyGrid, self.callback_map)
        
        if self.robo_coords and self.goal_coords and self.raw_occmap:
            return True
        else:
            return False
       
    def wait_for_topics(self):
        recieved = False
        while not recieved:
            recieved = self.subscribe_to_topics()
            time.sleep(0.01)
           
    def plan_path(self):
        # Get start and goal coordinates in Map-frame
        start = self.world2map([self.robo_coords[0], self.robo_coords[1]], self.raw_occmap) 
        goal = self.world2map([self.goal_coords[0], self.goal_coords[1]], self.raw_occmap)
        
        # If goal is not reached, plan a path
        if start[0] != goal[0] and start[1] != goal[1]:
            planner = Planner(
                map = self.mapUGV,
                start = start,
                goal = goal,
                max_iterations = self.iterations,
                stepdistance = self.step_distance,
                plot_enabled = False,
                search_radius_increment = 0.5,
                max_neighbour_found = 8,
                bdilation_multiplier = round(self.safety_buffer / self.raw_occmap.info.resolution),
                cell_sizes= [10, 20])
            path = planner.search()
            if len(path) > 1:
                self.latest_path =  PU.planner2world(path, self.raw_occmap)
                return True
        return False
    
    def publish_path(self):
        pub = rospy.Publisher('path_pub', Path, queue_size=1)
        path2publish = self.latest_path
        msg = Path()
        msg.header.seq = 0 #??
        msg.header.frame_id = "/map"
        msg.header.stamp = rospy.Time.now()
        
        for i in range(len(path2publish)):
            pose_stamped = PoseStamped()
            pose_stamped.header.seq = i
            pose_stamped.header.frame_id = "/world_frame"
            pose_stamped.header.stamp = rospy.Time.now()
            
            [x, y] = path2publish[i]
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            # 2D path -> z = 0
            pose_stamped.pose.position.z = 0
            
            msg.poses.append(pose_stamped) 
                
        #rospy.loginfo(msg)
        pub.publish(msg) 

#!/usr/bin/env python3

import rospy
from nav_msgs.msg import OccupancyGrid, MapMetaData, Path
from geometry_msgs.msg import Twist, Pose2D, PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import String, Bool
import tf
import numpy as np
from numpy import linalg
from utils.utils import wrapToPi
from utils.grids import StochOccupancyGrid2D
from planners import AStar, compute_smoothed_traj
import scipy.interpolate
import matplotlib.pyplot as plt
from controllers import PoseController, TrajectoryTracker, HeadingController
from enum import Enum

from dynamic_reconfigure.server import Server
from asl_turtlebot.cfg import NavigatorConfig

from asl_turtlebot.msg import DetectedObject, DetectedObjectList
from nav_msgs.msg import Odometry

# state machine modes, not all implemented
class Mode(Enum):
    IDLE = 0
    ALIGN = 1
    TRACK = 2
    PARK = 3
    STOP = 4
    CROSS = 5
    RESCUE = 6


class Navigator:
    """
    This node handles point to point turtlebot motion, avoiding obstacles.
    It is the sole node that should publish to cmd_vel
    """

    def __init__(self):
        rospy.init_node("turtlebot_navigator", anonymous=True)
        self.mode = Mode.IDLE

        # current state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # goal state
        self.x_g = None
        self.y_g = None
        self.theta_g = None

        self.th_init = 0.0

        ## Dictionary and parameters that stores the positions of the animals and stops signs detected in the world (also helper variables for the rescue mission)
        self.obj_pose_dict = {} # dictionary 
        self.obj_pose_dict["home"] = [3.16, 1.6, 0.0] # dictionary starting with home pose 
        self.count_obj = 0 # for now we have a counter of objects detected (DEBUG: this must be changed to labels ("strings") of the objects detected with the NN)
        self.obj_thresh_dictio = 1 # threshold required to detect objects only once (DEBUG: probably tune this in the future)
        self.curr_robot_pos = Pose2D() # msg utilized for the odometry callback to get robot's current pose in the world when an object is seen
        self.animal_min_dist = 0.5 # Minimum distance from a animal to register it 
        self.rescueList = []
        ## END Dictionary

        ## Required variables to STOP at stop_signs
        # Time to stop at a stop sign
        self.stop_time = 3.0 # DEBUG: probably we want to stop for more/less time at stop_signs. Who knows?
        self.crossing_time = 8.0 # DEBUG: probably we want to cross for more/less time at stop_signs. Who knows?
        self.rescuing_time = 4.0 # DEBUG: probably we want to rescue for more/less time at animals. Who knows?
        self.stop_min_dist = 0.5 # Minimum distance from a stop sign to obey it (was formerly 0.3m)
        ## END Required variables to STOP at stop_signs

        # map parameters
        self.map_width = 0
        self.map_height = 0
        self.map_resolution = 0
        self.map_origin = [0, 0]
        self.map_probs = []
        self.occupancy = None
        self.occupancy_updated = False

        # plan parameters
        self.plan_resolution = 0.1
        self.plan_horizon = 15

        # time when we started following the plan
        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = 0
        self.plan_start = [0.0, 0.0]

        # Robot limits
        self.v_max = 0.2  # maximum velocity
        self.om_max = 0.4  # maximum angular velocity

        self.v_des = 0.12  # desired cruising velocity
        self.theta_start_thresh = 0.05  # threshold in theta to start moving forward when path-following
        self.start_pos_thresh = (
            0.2  # threshold to be far enough into the plan to recompute it
        )

        # threshold at which navigator switches from trajectory to pose control
        self.near_thresh = 0.02 # we reduced this from 0.2
        self.at_thresh = 0.02
        self.at_thresh_theta = 0.05

        # trajectory smoothing
        self.spline_alpha = 0.15
        self.spline_deg = 3  # cubic spline
        self.traj_dt = 0.1

        # trajectory tracking controller parameters
        self.kpx = 0.5
        self.kpy = 0.5
        self.kdx = 1.5
        self.kdy = 1.5

        # heading controller parameters
        self.kp_th = 2.0

        self.traj_controller = TrajectoryTracker(
            self.kpx, self.kpy, self.kdx, self.kdy, self.v_max, self.om_max
        )
        self.pose_controller = PoseController(
            0.0, 0.0, 0.0, self.v_max, self.om_max
        )
        self.heading_controller = HeadingController(self.kp_th, self.om_max)

        ###########START PUBLISHERS#################################################
        
        self.nav_planned_path_pub = rospy.Publisher(
            "/planned_path", Path, queue_size=10
        )
        self.nav_smoothed_path_pub = rospy.Publisher(
            "/cmd_smoothed_path", Path, queue_size=10
        )
        self.vis_marker_pub = rospy.Publisher(
            "/vis_marker", Marker, queue_size=10
        )
        self.nav_smoothed_path_rej_pub = rospy.Publisher(
            "/cmd_smoothed_path_rejected", Path, queue_size=10
        )
        self.nav_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        
        # Command publish object detected (this is only for debugging purposes to count how many times it adds the pose of the same obj to the dictionary)
        self.obj_pos_publisher = rospy.Publisher('/obj_pose', Pose2D , queue_size=10)
        
        # DEBUG: Print current task and dictionary keys 
        self.rescue_task_publisher = rospy.Publisher('/rescue_task', String , queue_size=10)

        # DEBUG: Publish meow and woof
        self.meow_woof_publisher = rospy.Publisher('/meow_woof', String , queue_size=10)

        # DEBUG: Publish current dictionary of detected animals
        self.animals_dictio_publisher = rospy.Publisher('/animals_dictio', String , queue_size=10)

        ###########END PUBLISHERS#################################################

        self.trans_listener = tf.TransformListener()

        self.cfg_srv = Server(NavigatorConfig, self.dyn_cfg_callback)

        ###########START SUBSCRIBERS#################################################

        rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        rospy.Subscriber("/map_metadata", MapMetaData, self.map_md_callback)
        rospy.Subscriber("/cmd_nav", Pose2D, self.cmd_nav_callback)

        # Stop sign detector (DEBUG: moved from supervisor.py to navigator.py)
        rospy.Subscriber('/detector/stop_sign', DetectedObject, self.stop_sign_detected_callback)

        # This subscriber get information of current pose from odometry (used to store robot's position when object or animal found)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # This subscriber gets information from rescue.py as a form of string to rescue a list of animals (list or just one index)
        rospy.Subscriber('/rescue', String, self.rescue_callback)
        
        # This subscriber gets an emergency stop signal
        rospy.Subscriber('/emergency_stop', Bool, self.emergencyStop_callback)

        # Animals detector (DEBUG: this function is experimental)
        rospy.Subscriber('/detector/objects', DetectedObjectList, self.animal_detected_callback)

        ###########END SUBSCRIBERS#################################################


        print("finished init")

    def rescue_callback(self, msg):
        # qqqq
        # if idx in self.obj_pose_dict:
        #     resc_pos_msg = Pose2D()
        #     resc_pos_msg.x = self.obj_pose_dict[idx][0]
        #     resc_pos_msg.y = self.obj_pose_dict[idx][1]
        #     resc_pos_msg.theta = self.obj_pose_dict[idx][2]
        #     self.obj_rescue_pose_publisher.publish(resc_pos_msg)
        try:
            listStr = msg.data
            self.rescueList = listStr.split(",")
            self.rescueList.append("home")
            # reverse
            self.rescueList.reverse()
            rospy.loginfo("DEBUG: Got callback to rescue the following list: %s", listStr)
        except:
            rospy.loginfo("ERROR: Tried to cast the rescue list but something went wrong...")

    def emergencyStop_callback(self, msg):
        try:

            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.nav_vel_pub.publish(cmd_vel)
            self.x_g = None
            self.y_g = None
            self.theta_g = None

            self.rescueList.clear() # DEBUG: see if this is necessary, ask the guys

            self.switch_mode(Mode.IDLE)
            
            rospy.loginfo("DEBUG: Got callback to emergency stop")
        except:
            rospy.loginfo("ERROR: Tried to emergency stop but something went wrong...")

    
    def odom_callback(self, msg):
        # self.curr_robot_pos = Pose2D()
        self.curr_robot_pos.x = msg.pose.pose.position.x
        self.curr_robot_pos.y = msg.pose.pose.position.y
        w = msg.pose.pose.orientation.w
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        self.curr_robot_pos.theta = np.arctan2(2*(w*z+x*y),1-2*(y**2+z**2))

    def stop_sign_detected_callback(self, msg):
        """ callback for when the detector has found a stop sign. Note that
        a distance of 0 can mean that the lidar did not pickup the stop sign at all """

        # distance of the stop sign
        dist = msg.distance

        # if close enough and in nav mode, stop
        if dist > 0 and dist < self.stop_min_dist and self.mode == Mode.TRACK: # check this with Sweekar
            self.init_stop_sign()
            rospy.loginfo("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            rospy.loginfo("DEBUG: I saw a stop sign...")
            rospy.loginfo("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            ## FIRST APPROACH WHEN WE DIDN'T HAVE ANIMALS
            
            # try:
            #     # Try adding new detected object (stop sign) or to the dictionary depending on the threshold distance
            #     if self.obj_pose_dict:
            #         arr_values = np.array(list(self.obj_pose_dict.values()))
            #         dict_x = arr_values[:, 0]
            #         dict_y = arr_values[:, 1]
            #         dist = np.sqrt((self.curr_robot_pos.x*np.ones(len(dict_x)) - dict_x)**2 + (self.curr_robot_pos.y*np.ones(len(dict_y)) - dict_y)**2)
            #         if np.min(dist) > self.obj_thresh_dictio:
            #             self.obj_pos_publisher.publish(self.curr_robot_pos)
            #             self.obj_pose_dict[str(self.count_obj)] = [self.curr_robot_pos.x, self.curr_robot_pos.y, self.curr_robot_pos.theta]
            #             self.count_obj = self.count_obj + 1
            #             rospy.loginfo("DEBUG: Added successfully new object to dictionary... Count: %s", str(self.count_obj))
            #     else:
            #         # qqqq
            #         self.obj_pos_publisher.publish(self.curr_robot_pos)
            #         self.obj_pose_dict[str(self.count_obj)] = [self.curr_robot_pos.x, self.curr_robot_pos.y, self.curr_robot_pos.theta]
            #         self.count_obj = self.count_obj + 1
            #         rospy.loginfo("DEBUG: Added successfully new object to dictionary... Count: %s", str(self.count_obj))
            # except:
            #     rospy.loginfo("ERROR: Tried to store detected stop sign pose in the dictionary but something went wrong...")

    def animal_detected_callback(self, msg):
        try:

            # distance of the stop sign
            animal_name = msg.ob_msgs[0].name
            animal_dist = msg.ob_msgs[0].distance
            animal_conf = msg.ob_msgs[0].confidence
            rospy.loginfo("**************************************************************************")
            rospy.loginfo("DEBUG: Entered animal dected callback")
            rospy.loginfo("DEBUG: Animal name: %s", str(animal_name))
            rospy.loginfo("DEBUG: Distance: %f", animal_dist)
            rospy.loginfo("DEBUG: Confidence: %f", animal_conf)

            # if close enough and in nav mode, stop
            if animal_dist > 0 and animal_dist < self.animal_min_dist and animal_conf > 0.7: #and self.mode == Mode.NAV:
                if animal_name == 'cat':
                    self.meow_woof_publisher.publish("Meow") 
                elif animal_name == 'dog':
                    self.meow_woof_publisher.publish("Woof") 


                if self.mode == Mode.RESCUE:
                    rospy.loginfo("DEBUG: We are currently at RESCUE MODE -> return")
                    rospy.loginfo("**************************************************************************")
                    return

                rospy.loginfo("DEBUG: Within distance, high confidence")
                # Try adding new detected object (stop sign) or animal to the dictionary depending on the threshold distance
                if self.obj_pose_dict:
                    rospy.loginfo("DEBUG: The dictionary is NOT empty")
                    arr_values = np.array(list(self.obj_pose_dict.values()))
                    dict_x = arr_values[:, 0]
                    dict_y = arr_values[:, 1]
                    dist = np.sqrt((self.curr_robot_pos.x*np.ones(len(dict_x)) - dict_x)**2 + (self.curr_robot_pos.y*np.ones(len(dict_y)) - dict_y)**2)
                    if np.min(dist) > self.obj_thresh_dictio and animal_name not in self.obj_pose_dict:
                        rospy.loginfo("DEBUG: Going to add the item to the dictionary")
                        self.obj_pos_publisher.publish(self.curr_robot_pos)
                        self.obj_pose_dict[str(animal_name)] = [self.curr_robot_pos.x, self.curr_robot_pos.y, self.curr_robot_pos.theta]
                        rospy.loginfo("DEBUG: Added successfully new animal to dictionary... Name: %s", str(animal_name))
                    else: 
                        rospy.loginfo("DEBUG: Didnt add new animal to dictionary because it doesnt satisfy uniqueness and dist thresh... Name: %s", str(animal_name))
                else:
                    rospy.loginfo("DEBUG: The dictionary is empty")
                    # qqqq
                    self.obj_pos_publisher.publish(self.curr_robot_pos)
                    self.obj_pose_dict[str(animal_name)] = [self.curr_robot_pos.x, self.curr_robot_pos.y, self.curr_robot_pos.theta]
                    rospy.loginfo("DEBUG: Added successfully new animal to dictionary... Name: %s", str(animal_name))
            rospy.loginfo("**************************************************************************")
        except:
            rospy.loginfo("ERROR: Tried to store detected animal pose in the dictionary but something went wrong...")

    
    def dyn_cfg_callback(self, config, level):
        rospy.loginfo(
            "Reconfigure Request: k1:{k1}, k2:{k2}, k3:{k3}".format(**config)
        )
        self.pose_controller.k1 = config["k1"]
        self.pose_controller.k2 = config["k2"]
        self.pose_controller.k3 = config["k3"]
        return config

    def cmd_nav_callback(self, data):
        """
        loads in goal if different from current goal, and replans
        """
        if (
            data.x != self.x_g
            or data.y != self.y_g
            or data.theta != self.theta_g
        ):
            rospy.logdebug(f"New command nav received:\n{data}")
            self.x_g = data.x
            self.y_g = data.y
            self.theta_g = data.theta
            self.replan()

    def map_md_callback(self, msg):
        """
        receives maps meta data and stores it
        """
        self.map_width = msg.width
        self.map_height = msg.height
        self.map_resolution = msg.resolution
        self.map_origin = (msg.origin.position.x, msg.origin.position.y)

    def map_callback(self, msg):
        """
        receives new map info and updates the map
        """
        self.map_probs = msg.data
        # if we've received the map metadata and have a way to update it:
        if (
            self.map_width > 0
            and self.map_height > 0
            and len(self.map_probs) > 0
        ):
            self.occupancy = StochOccupancyGrid2D(
                self.map_resolution,
                self.map_width,
                self.map_height,
                self.map_origin[0],
                self.map_origin[1],
                7,
                self.map_probs,
            )
            if self.x_g is not None and self.mode != Mode.STOP: # check this with Sweekar
                # if we have a goal to plan to, replan
                rospy.loginfo("replanning because of new map")
                self.replan()  # new map, need to replan

    def shutdown_callback(self):
        """
        publishes zero velocities upon rospy shutdown
        """
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.angular.z = 0.0
        self.nav_vel_pub.publish(cmd_vel)

    def near_goal(self):
        """
        returns whether the robot is close enough in position to the goal to
        start using the pose controller
        """
        try: 
            res = (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.near_thresh
            )
            return res
        except:
            rospy.loginfo("ERROR: Tried to compute near_goal but ran into an error, returning True...")
            return True

    def at_goal(self):
        """
        returns whether the robot has reached the goal position with enough
        accuracy to return to idle state
        """
        try:
            res =  (
            linalg.norm(np.array([self.x - self.x_g, self.y - self.y_g]))
            < self.at_thresh
            and abs(wrapToPi(self.theta - self.theta_g)) < self.at_thresh_theta
            )
            return res
        except:
            rospy.loginfo("ERROR: Tried to compute at_goal but ran into an error, returning True...")
            return True



    def aligned(self):
        """
        returns whether robot is aligned with starting direction of path
        (enough to switch to tracking controller)
        """
        return (
            abs(wrapToPi(self.theta - self.th_init)) < self.theta_start_thresh
        )

    def close_to_plan_start(self):
        return (
            abs(self.x - self.plan_start[0]) < self.start_pos_thresh
            and abs(self.y - self.plan_start[1]) < self.start_pos_thresh
        )

    def snap_to_grid(self, x):
        return (
            self.plan_resolution * round(x[0] / self.plan_resolution),
            self.plan_resolution * round(x[1] / self.plan_resolution),
        )

    def switch_mode(self, new_mode):
        rospy.loginfo("Switching from %s -> %s", self.mode, new_mode)
        self.mode = new_mode
    
    def publish_marker(self, publisher):
        marker = Marker()

        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time()

        # IMPORTANT: If you're creating multiple markers, 
        #            each need to have a separate marker ID.
        marker.id = 0

        marker.type = 2 # sphere
        if not self.x_g == None and not self.y_g == None:
            marker.pose.position.x = self.x_g
            marker.pose.position.y = self.y_g
            marker.pose.position.z = 0
            print("Entered to xg yg")
        else:
            print("Not entered")

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        marker.color.a = 1.0 # Don't forget to set the alpha!
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        
        publisher.publish(marker)
        print('Published marker!')
        
        # rate.sleep()

    def publish_planned_path(self, path, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for state in path:
            pose_st = PoseStamped()
            pose_st.pose.position.x = state[0]
            pose_st.pose.position.y = state[1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_smoothed_path(self, traj, publisher):
        # publish planned plan for visualization
        path_msg = Path()
        path_msg.header.frame_id = "map"
        for i in range(traj.shape[0]):
            pose_st = PoseStamped()
            pose_st.pose.position.x = traj[i, 0]
            pose_st.pose.position.y = traj[i, 1]
            pose_st.pose.orientation.w = 1
            pose_st.header.frame_id = "map"
            path_msg.poses.append(pose_st)
        publisher.publish(path_msg)

    def publish_control(self):
        """
        Runs appropriate controller depending on the mode. Assumes all controllers
        are all properly set up / with the correct goals loaded
        """
        try:
            t = self.get_current_plan_time()

            if self.mode == Mode.PARK:
                V, om = self.pose_controller.compute_control(
                    self.x, self.y, self.theta, t
                )
            elif self.mode == Mode.TRACK:
                V, om = self.traj_controller.compute_control(
                    self.x, self.y, self.theta, t
                )
            elif self.mode == Mode.ALIGN:
                V, om = self.heading_controller.compute_control(
                    self.x, self.y, self.theta, t
                )
            elif self.mode == Mode.CROSS:
                V, om = self.traj_controller.compute_control(
                    self.x, self.y, self.theta, t
                )
            else:
                V = 0.0
                om = 0.0

            cmd_vel = Twist()
            cmd_vel.linear.x = V
            cmd_vel.angular.z = om
            self.nav_vel_pub.publish(cmd_vel)
        except:
            rospy.loginfo("ERROR: Tried to publish_control but something went wrong...")
            V = 0.0
            om = 0.0


    def get_current_plan_time(self):
        t = (rospy.get_rostime() - self.current_plan_start_time).to_sec()
        return max(0.0, t)  # clip negative time to 0

    def replan(self):
        """
        loads goal into pose controller
        runs planner based on current pose
        if plan long enough to track:
            smooths resulting traj, loads it into traj_controller
            sets self.current_plan_start_time
            sets mode to ALIGN
        else:
            sets mode to PARK
        """

        # if not self.has_crossed() or not self.has_rescued():
        #     return

        # Make sure we have a map
        if not self.occupancy:
            rospy.loginfo(
                "Navigator: replanning canceled, waiting for occupancy map."
            )
            self.switch_mode(Mode.IDLE)
            return

        # Attempt to plan a path
        state_min = self.snap_to_grid((-self.plan_horizon, -self.plan_horizon))
        state_max = self.snap_to_grid((self.plan_horizon, self.plan_horizon))
        x_init = self.snap_to_grid((self.x, self.y))
        self.plan_start = x_init
        x_goal = self.snap_to_grid((self.x_g, self.y_g))
        problem = AStar(
            state_min,
            state_max,
            x_init,
            x_goal,
            self.occupancy,
            self.plan_resolution,
        )

        rospy.loginfo("Navigator: computing navigation plan")
        success = problem.solve()
        if not success:
            rospy.loginfo("Planning failed")
            return
        rospy.loginfo("Planning Succeeded")

        planned_path = problem.path

        # Check whether path is too short
        if len(planned_path) < 4 and not self.mode == Mode.RESCUE:
            rospy.loginfo("Path too short to track")
            self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
            self.switch_mode(Mode.PARK)
            return

        # Smooth and generate a trajectory
        t_new, traj_new = compute_smoothed_traj(
            planned_path, self.v_des, self.spline_deg, self.spline_alpha, self.traj_dt
        )

        # If currently tracking a trajectory, check whether new trajectory will take more time to follow
        if self.mode == Mode.TRACK:
            t_remaining_curr = (
                self.current_plan_duration - self.get_current_plan_time()
            )

            # Estimate duration of new trajectory
            th_init_new = traj_new[0, 2]
            th_err = wrapToPi(th_init_new - self.theta)
            t_init_align = abs(th_err / self.om_max)
            t_remaining_new = t_init_align + t_new[-1]

            if t_remaining_new > t_remaining_curr:
                rospy.loginfo(
                    "New plan rejected (longer duration than current plan)"
                )
                self.publish_smoothed_path(
                    traj_new, self.nav_smoothed_path_rej_pub
                )
                return

        # Otherwise follow the new plan
        self.publish_planned_path(planned_path, self.nav_planned_path_pub)
        self.publish_smoothed_path(traj_new, self.nav_smoothed_path_pub)
        # self.publish_marker(self.vis_marker_pub)

        self.pose_controller.load_goal(self.x_g, self.y_g, self.theta_g)
        self.traj_controller.load_traj(t_new, traj_new)

        self.current_plan_start_time = rospy.get_rostime()
        self.current_plan_duration = t_new[-1]

        self.th_init = traj_new[0, 2]
        self.heading_controller.load_goal(self.th_init)

        if not self.aligned():
            rospy.loginfo("Not aligned with start direction")
            self.switch_mode(Mode.ALIGN)
            return

        rospy.loginfo("Ready to track")
        self.switch_mode(Mode.TRACK)

    def rescueMission(self):
        try:
            rospy.loginfo("DEBUG: Im currently at goal position, I have rescue missions pending...")
            # set new goal for rescue mission:
            # self.obj_pose_dict[str(self.count_obj)] = [self.curr_robot_pos.x, self.curr_robot_pos.y, self.curr_robot_pos.theta]
            # rospy.loginfo("ERROR: OK1")
            rescueObj = self.rescueList[-1]
            rescueObj = rescueObj.replace(" ", "")
            # rospy.loginfo("ERROR: OK2")
            self.rescue_task_publisher.publish("Current Mission: " + str(rescueObj) + "; Current Dictionary: " + str(self.obj_pose_dict.keys()))
            if rescueObj in self.obj_pose_dict:
                # rospy.loginfo("ERROR: OK3")
                rospy.loginfo("DEBUG: Rescue object given in the list DOES exist...")
                rescuePos = self.obj_pose_dict[rescueObj]
                # rospy.loginfo("ERROR: OK4")
                self.x_g = rescuePos[0]
                # rospy.loginfo("ERROR: OK5")
                self.y_g = rescuePos[1]
                # rospy.loginfo("ERROR: OK6")
                self.theta_g = rescuePos[2]
                # rospy.loginfo("ERROR: OK7")
                if self.near_goal() and not self.mode == Mode.RESCUE:
                    # rospy.loginfo("ERROR: OK8")
                    self.switch_mode(Mode.PARK)
                    # rospy.loginfo("ERROR: OK9")
                    self.rescueList.pop()
                    # rospy.loginfo("ERROR: OK10")
                else:
                    # rospy.loginfo("ERROR: OK11")
                    self.replan() # Is this line mandatory ?????? OR is there a better way to do this????
                    # rospy.loginfo("ERROR: OK12")
            else:
                rospy.loginfo("WARNING: Rescue object given in the list DOES NOT exist... %s", rescueObj)
                # self.x_g = None
                # self.y_g = None
                # self.theta_g = None
                self.rescueList.pop()

        except:
            rospy.loginfo("ERROR: I have rescue tasks, but couldnt access dictionary to go to the next rescue mission...")


    #############START FORMER SUPERVISOR ACTIONS#############
    def init_stop_sign(self):
        """ initiates a stop sign maneuver """

        self.stop_sign_start = rospy.get_rostime()
        self.switch_mode(Mode.STOP)
        # cmd_vel = Twist()
        # cmd_vel.linear.x = 0.0
        # cmd_vel.linear.y = 0.0
        # cmd_vel.angular.z = 0.0
        # self.nav_vel_pub.publish(cmd_vel)


    def has_stopped(self):
        """ checks if stop sign maneuver is over """
        rospy.loginfo("DEBUG: Checking has_stopped...")
        rospy.loginfo("DEBUG: self.mode: %s", str(self.mode))
        rospy.loginfo("DEBUG: difference of time: %s", str(rospy.get_rostime() - self.stop_sign_start))
        rospy.loginfo("DEBUG: self.stop_time: %s", str(rospy.Duration.from_sec(self.stop_time)))
        return self.mode == Mode.STOP and \
               rospy.get_rostime() - self.stop_sign_start > rospy.Duration.from_sec(self.stop_time)

    def init_crossing(self):
        """ initiates an intersection crossing maneuver """
        rospy.loginfo("DEBUG: Checking init_crossing...")
        rospy.loginfo("DEBUG: self.cross_start: %s", str(rospy.get_rostime()))
        self.cross_start = rospy.get_rostime()
        self.switch_mode(Mode.CROSS)


    def has_crossed(self):
        """ checks if crossing maneuver is over """
        # self.nav_to_pose() check if we need this???????????????????????
        try:
            rospy.loginfo("DEBUG: Checking has_crossed...")
            rospy.loginfo("DEBUG: difference of time: %s", str(rospy.get_rostime() - self.cross_start))
            rospy.loginfo("DEBUG: self.crossing_time: %s", str(rospy.Duration.from_sec(self.crossing_time)))
            return (self.mode == Mode.CROSS and \
                rospy.get_rostime() - self.cross_start > rospy.Duration.from_sec(self.crossing_time))
        except:
            return True


    def has_rescued(self):
        """ checks if rescueing maneuver is over """
        # self.nav_to_pose() check if we need this???????????????????????
        try:
            rospy.loginfo("DEBUG: Checking has_rescued...")
            rospy.loginfo("DEBUG: self.rescue_start: %s", str(self.rescue_start))
            rospy.loginfo("DEBUG: difference of time: %s", str(rospy.get_rostime() - self.rescue_start))
            rospy.loginfo("DEBUG: self.rescuing_time: %s", str(rospy.Duration.from_sec(self.rescuing_time)))
            return self.mode == Mode.RESCUE and \
                rospy.get_rostime() - self.rescue_start > rospy.Duration.from_sec(self.rescuing_time)
        except:
            return True

    #############END FORMER SUPERVISOR ACTIONS#############

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # try to get state information to update self.x, self.y, self.theta
            try:
                (translation, rotation) = self.trans_listener.lookupTransform(
                    "/map", "/base_footprint", rospy.Time(0)
                )
                self.x = translation[0]
                self.y = translation[1]
                euler = tf.transformations.euler_from_quaternion(rotation)
                self.theta = euler[2]
            except (
                tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
            ) as e:
                self.current_plan = []
                rospy.loginfo("Navigator: waiting for state info")
                self.switch_mode(Mode.IDLE)
                print(e)
                pass

            # STATE MACHINE LOGIC
            # some transitions handled by callbacks
            if self.mode == Mode.IDLE:
                rospy.loginfo("DEBUG: Im currently in IDLE MODE...")
                if len(self.rescueList) > 0:
                    rospy.loginfo("DEBUG: Im currently at IDLE, but got callback for rescue mission and list is not empty...")
                    # self.switch_mode(Mode.PARK)
                    self.rescueMission()
                pass
            elif self.mode == Mode.ALIGN:
                if self.aligned():
                    self.current_plan_start_time = rospy.get_rostime()
                    self.switch_mode(Mode.TRACK)
            elif self.mode == Mode.TRACK:
                if self.near_goal():
                    self.switch_mode(Mode.PARK)
                elif not self.close_to_plan_start():
                    rospy.loginfo("replanning because far from start")
                    self.replan()
                elif (
                    rospy.get_rostime() - self.current_plan_start_time
                ).to_sec() > self.current_plan_duration:
                    rospy.loginfo("replanning because out of time")
                    self.replan()  # we aren't near the goal but we thought we should have been, so replan
            elif self.mode == Mode.PARK:
                if self.at_goal():
                    if len(self.rescueList) == 0:
                        rospy.loginfo("DEBUG: Im currently at goal position, switching to IDLE...")
                        # forget about goal:
                        self.x_g = None
                        self.y_g = None
                        self.theta_g = None
                        self.switch_mode(Mode.IDLE)
                elif len(self.rescueList) > 0:
                    self.rescue_start = rospy.get_rostime()
                    self.switch_mode(Mode.RESCUE)
                    # self.rescueMission()
            elif self.mode == Mode.STOP:
                rospy.loginfo("DEBUG: Im currently in STOP MODE...")
                if self.has_stopped():
                    rospy.loginfo("DEBUG: I have stopped...")
                    self.init_crossing()
            
            elif self.mode == Mode.CROSS:
                rospy.loginfo("DEBUG: Im currently in CROSS MODE...")
                if self.has_crossed():
                    rospy.loginfo("DEBUG: I have crossed...")
                    self.replan() # In case of conflict, track being changed to park incorrectly
                    self.switch_mode(Mode.ALIGN)  #safety ones

            elif self.mode == Mode.RESCUE:
                rospy.loginfo("DEBUG: Im currently in RESCUE MODE...")
                if self.has_rescued():
                    rospy.loginfo("DEBUG: Has rescued successfully...")
                    self.switch_mode(Mode.IDLE)

            self.publish_control()

            self.publish_marker(self.vis_marker_pub)
            self.animals_dictio_publisher.publish(str(self.obj_pose_dict))

            rate.sleep()


if __name__ == "__main__":
    nav = Navigator()
    rospy.on_shutdown(nav.shutdown_callback)
    nav.run()
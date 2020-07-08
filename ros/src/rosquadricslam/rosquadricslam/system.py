# import standard libraries
import os
import sys
import numpy as np
import time
sys.path.append('/home/lachness/.pyenv/versions/382_generic/lib/python3.8/site-packages/')
import cv2
import atexit
import yaml
import argparse

# import ros libraries
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from detection_msgs.msg import ObjectDetectionArray
from cv_bridge import CvBridge
import message_filters

# import custom python modules
sys.path.append('/home/lachness/git_ws/quadricslam/ros/src/rosquadricslam/rosquadricslam')
sys.path.append('/home/lachness/git_ws/quadricslam/examples/python/example_frontend')
sys.dont_write_bytecode = True
from data_association import DataAssociation
from dataset_interfaces.scenenet_dataset import SceneNetDataset
from visualization.drawing import CV2Drawing
from base.containers import Trajectory, Quadrics, Detections, ObjectDetection

# import gtsam and extension
import gtsam
import quadricslam







class ROSQuadricSLAM(Node):
    def __init__(self, args):
        # set node name
        super().__init__('ROSQuadricSLAM')

        # settings
        self.config_path = args.config_path
        self.depth = args.depth
        self.record = args.record
        self.minimum_views = args.minimum_views
        self.initialization_method = args.initialization_method

        # load camera calibration
        self.calibration = self.load_camera_calibration(self.config_path)

        # load class names
        self.class_names = self.load_class_names('/home/lachness/git_ws/PyTorch-YOLOv3/data/coco.names')

        # start subscriptions
        self.pose_subscription = message_filters.Subscriber(self, PoseStamped, 'poses')
        self.detection_subscription = message_filters.Subscriber(self, ObjectDetectionArray, 'detections')
        self.image_subscription = message_filters.Subscriber(self, Image, 'image')
        self.time_synchronizer = message_filters.TimeSynchronizer([self.image_subscription, self.pose_subscription, self.detection_subscription], self.depth)
        self.time_synchronizer.registerCallback(self.update)

        # create isam2 optimizer 
        opt_params = gtsam.ISAM2DoglegParams()
        # opt_params = gtsam.ISAM2GaussNewtonParams()
        parameters = gtsam.ISAM2Params()
        parameters.setOptimizationParams(opt_params)
        parameters.setRelinearizeSkip(1)
        parameters.setRelinearizeThreshold(0.01)
        # parameters.setEnableRelinearization(False)
        parameters.print_("ISAM2 Parameters")
        self.isam = gtsam.ISAM2(parameters)
        
        POSE_SIGMA = 0.001
        BOX_SIGMA = 10.0
        self.X = lambda i: int(gtsam.symbol(ord('x'), i))
        self.Q = lambda i: int(gtsam.symbol(ord('q'), i))
        self.bridge = CvBridge()
        self.pose_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([POSE_SIGMA]*6, dtype=np.float))
        self.bbox_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([BOX_SIGMA]*4, dtype=np.float))
        self.quadric_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([100]*9, dtype=np.float))
        self.graph = gtsam.NonlinearFactorGraph()
        self.estimate = gtsam.Values()


        # set measurement storage 
        self.images = dict()
        self.poses = Trajectory()
        self.detections = Detections()
        self.initial_quadrics = Quadrics()

        # convert from time stamp to pose_keys
        self.pose_keys = dict()

        # store current estimates to draw each frame
        self.current_trajectory = Trajectory()
        self.current_quadrics = Quadrics()

        # initialize data-association module
        self.data_association = DataAssociation(self.current_quadrics, self.calibration)

        # prepare video capture
        if self.record:
            self.video_writer = cv2.VideoWriter('good_performance.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 12.0, (640, 480))
            atexit.register(self.video_writer.release)
        print('\n~ Awaiting Measurements ~')

    def load_class_names(self, path):
        classes_fp = open(path, 'r')
        return classes_fp.read().split('\n')[:-1]

    def load_camera_calibration(self, path, no_distortion=True):
        """ Loads gtsam calibration from openvslam config format """
        camera_config = yaml.safe_load(open(path, 'r'))

        camera_model = gtsam.Cal3_S2
        calibration_list = [
            camera_config['Camera.fx'],
            camera_config['Camera.fy'],
            0.0,
            camera_config['Camera.cx'],
            camera_config['Camera.cy'],
        ]

        if no_distortion:
            return camera_model(*calibration_list)

        if 'Camera.k1' in camera_config:
            camera_model = gtsam.Cal3DS2
            calibration_list += [
                camera_config['Camera.k1'],
                camera_config['Camera.k2'],
            ]

        if 'Camera.p1' in camera_config:
            calibration_list += [
                camera_config['Camera.p1'],
                camera_config['Camera.p2'],
            ]

        return camera_model(*calibration_list)
        
    def msg2detections(self, msg, filters=None):
        detections = []
        for detection in msg.detections:
            if filters is not None:
                filter_indicies = [self.class_names.index(filter) for filter in filters]
            if filters is None or np.argmax(detection.scores) in filter_indicies:
                box = quadricslam.AlignedBox2(detection.box.xmin, detection.box.ymin, detection.box.xmax, detection.box.ymax) 
                detection = ObjectDetection(box, detection.objectness, detection.scores)
                detections.append(detection)
        return detections

    def msg2pose(self, msg):
        point = gtsam.Point3(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        rot = gtsam.Rot3.Quaternion(msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z)
        return gtsam.Pose3(rot, point)

    def msg2image(self, msg):
        return self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
    def msg2time(self, msg):
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec)*1e-9

    def time2key(self, float_time):
        try:
            return self.pose_keys[float_time]
        except KeyError as key_error:
            pose_key = len(self.pose_keys)
            self.pose_keys[float_time] = pose_key
            return pose_key







    def update(self, image_msg, pose_msg, detections_msg):
        update_start = time.time()
        self.get_logger().info('Update started')

        # convert msgs to data
        image = self.msg2image(image_msg)
        camera_pose = self.msg2pose(pose_msg).inverse()
        float_time = self.msg2time(detections_msg)
        pose_key = self.time2key(float_time)
        image_detections = self.msg2detections(detections_msg, filters=['cup', 'bowl'])
        # image_detections = self.msg2detections(detections_msg)

        # draw detections
        img = image.copy()
        drawing = CV2Drawing(img)
        for detection in image_detections:
            scores = detection.scores
            text = '{}:{:.2f}'.format(self.class_names[np.argmax(scores)], np.max(scores))
            drawing.box_and_text(detection.box, (0,0,255), text, (0,0,0))

        # draw current map 
        for quadric in self.current_quadrics.values():
            drawing.quadric(camera_pose, quadric, self.calibration, (255,0,255))
        cv2.imshow('Current view', img)
        cv2.waitKey(1)
        if self.record:
            self.video_writer.write(img)






        # associate new measurements with existing keys
        da_start = time.time()
        associated_detections = self.data_association.associate(image, image_detections, camera_pose, pose_key, visualize=True, verbose=True)
        da_end = time.time()






        # store new boxes and pose for later initialization and factor adding
        self.detections.add_detections(associated_detections)
        self.poses.add(camera_pose, pose_key)
        self.images[pose_key] = image

        # create local graph and estimate
        local_graph = gtsam.NonlinearFactorGraph()
        local_estimate = gtsam.Values()
        
        # add new pose measurements to graph / estimate
        local_estimate.insert(self.X(pose_key), camera_pose)
        prior_factor = gtsam.PriorFactorPose3(self.X(pose_key), camera_pose, self.pose_noise)
        local_graph.add(prior_factor)

        # check if we can initialize any new objects
        for object_key, object_detections in self.detections.per_object():

            # no need to re-initialize objects
            # TODO: use keys from current estimate 
            if object_key in self.initial_quadrics.keys():
                continue
            
            # initialize object if seen enough
            # TODO: use current trajectory instead of initial poses?
            quadric = self.initialize_quadric(object_key, object_detections, self.poses, local_estimate)

            # continue if not correctly initialized 
            if quadric is None: 
                continue

            # add quadric to values 
            quadric.addToValues(local_estimate, self.Q(object_key))

            # add weak quadric prior 
            prior_factor = quadricslam.PriorFactorConstrainedDualQuadric(self.Q(object_key), quadric, self.quadric_noise)
            local_graph.add(prior_factor)

            # add quadric to storage (not needed in future)
            self.initial_quadrics.add(quadric, object_key)


        # add measurements if unused
        for (pose_key, object_key), detection in self.detections.items():

            # only add new measurements
            if self.detections.is_used(pose_key, object_key):
                continue            

            # add measurements if initialized 
            # TODO: use keys from current estimate
            if object_key in self.initial_quadrics.keys():
                bbf = quadricslam.BoundingBoxFactor(detection.box, self.calibration, self.X(pose_key), self.Q(object_key), self.bbox_noise)
                bbf.addToGraph(local_graph)
                self.detections.set_used(True, pose_key, object_key)

        # append local graph / estimate to full graph
        self.graph.push_back(local_graph)
        self.estimate.insert(local_estimate)

        # use local graph / estimate to update isam2
        self.isam.update(local_graph, local_estimate)

        # calculate current estimate
        current_estimate = self.isam.calculateEstimate()
        update_end = time.time()

        
        
        
        
        # update current estimate 
        self.current_trajectory = Trajectory.from_values(current_estimate)
        self.current_quadrics.clear()
        self.current_quadrics.update(Quadrics.from_values(current_estimate))
        extracting_end = time.time()


        # print timings
        # self.get_logger().info('Update lasted {:.3f} s'.format(extracting_end-update_start))
        # print('pre-da:  {:.3f} s'.format(da_start-update_start))
        # print('da:      {:.3f} s'.format(da_end-da_start))
        # print('opt:     {:.3f} s'.format(update_end-da_end))
        # print('extract: {:.3f} s'.format(extracting_end-update_end))
        # print('')
            


    def initialize_quadric(self, object_key, object_detections, current_trajectory, local_estimate):
        """ 
        Attempts to initialize the quadric according to self.initialization_method.
        Returns None if quadric could not be initialized 
        """
        if self.initialization_method == 'SVD':
            if len(object_detections) >= self.minimum_views:

                object_boxes = [d.box for d in object_detections.values()]
                pose_keys = object_detections.keys()
                object_poses = current_trajectory.at_keys(pose_keys)
                quadric_matrix = self.quadric_SVD(object_poses, object_boxes, self.calibration)
                quadric = quadricslam.ConstrainedDualQuadric.constrain(quadric_matrix)

                # check quadric is okay
                if self.is_okay(quadric, object_poses, self.calibration):
                    return quadric

        else:
            abox = list(object_detections.values())[0]
            apose_key = list(object_detections.keys())[0]
            apose = current_trajectory.at(apose_key)
            displacement = 0.5
            quadric_pose = apose.compose(gtsam.Pose3(gtsam.Rot3(),gtsam.Point3(0,0,displacement)))
            quadric = quadricslam.ConstrainedDualQuadric(quadric_pose, np.array([0.01]*3))
            return quadric
        return None

    def quadric_SVD(self, poses, object_boxes, calibration):
        """ calculates quadric_matrix using SVD """

        # iterate through box/pose data
        planes = []
        for box, pose in zip(object_boxes, poses):

            # calculate boxes lines
            lines = box.lines()

            # convert Vector3Vector to list
            lines = [lines.at(i) for i in range(lines.size())]

            # calculate projection matrix
            P = quadricslam.QuadricCamera.transformToImage(pose, calibration).transpose()

            # project lines to planes
            planes += [P @ line for line in lines]

        # create A matrix
        A = np.asarray([np.array([p[0]**2,  2*(p[0]*p[1]),  2*(p[0]*p[2]),  2*(p[0]*p[3]),
                                                p[1]**2,  	2*(p[1]*p[2]),  2*(p[1]*p[3]),
                                                                p[2]**2,  	2*(p[2]*p[3]),
                                                                               p[3]**2]) for p in planes])

        # solve SVD for Aq = 0, which should be equal to p'Qp = 0
        _,_,V = np.linalg.svd(A, full_matrices=True)
        q = V.T[:, -1]

        # construct quadric
        dual_quadric = np.array([[q[0], q[1], q[2], q[3]],
                                [q[1], q[4], q[5], q[6]],
                                [q[2], q[5], q[7], q[8]],
                                [q[3], q[6], q[8], q[9]]])

        return dual_quadric

    def is_okay(self, quadric, poses, calibration):
        """
        Checks quadric is valid:
            quadric constrained correctly
            paralax > threshold
            reprojections valid in each frame 
                quadric infront of camera : positive depth 
                camera outside quadric
                conic is an ellipse 
            ensure views provide enough DOF (due to edges / out of frame)
        """
        for pose in poses:

            # quadric must have positive depth
            if quadric.isBehind(pose):
                return False

            # camera pose must be outside quadric 
            if quadric.contains(pose):
                return False

            # conic must be valid and elliptical 
            conic = quadricslam.QuadricCamera.project(quadric, pose, calibration)
            if conic.isDegenerate():
                return False
            if not conic.isEllipse():
                return False
                
        return True



def main(main_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', dest='config_path', type=str, required=True ,
                        help='path to the camera configuartion file')
    parser.add_argument('--depth', dest='depth', type=int, default=10, 
                        help='the queue depth to store topic messages')
    parser.add_argument('--record', dest='record', type=bool, default=False, 
                        help='boolean to record map visualization')
    parser.add_argument('--views', dest='minimum_views', type=int, default=5, 
                        help='minimum views required to initialize object')
    parser.add_argument('--init', dest='initialization_method', type=str, choices=['SVD', 'other'], default='SVD', 
                        help='method to use for initialization')
    args = parser.parse_args()
    
    # init ros
    rclpy.init(args=main_args)

    # create node
    system = ROSQuadricSLAM(args)

    # spin node
    rclpy.spin(system)

    # shutdown 
    system.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

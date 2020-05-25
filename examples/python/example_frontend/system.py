"""
QuadricSLAM Copyright 2020, ARC Centre of Excellence for Robotic Vision, Queensland University of Technology (QUT)
Brisbane, QLD 4000
All Rights Reserved

See LICENSE for the license information

Description: System (front-end) interface 
Author: Lachlan Nicholson (Python)
"""

# import standard libraries
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# import gtsam and extension
import gtsam
import quadricslam

# import custom python modules
sys.dont_write_bytecode = True
from dataset_interfaces.simulated_dataset import ManualSequence
from dataset_interfaces.scenenet_dataset import SceneNetDataset
from visualization.drawing import MPLDrawing
from base.evaluation import Evaluation
from base.containers import *


class System(object):
    """
    Python front-end to build graph/estimate from dataset. 
    """
    X = lambda i: int(gtsam.symbol(ord('x'), i))
    Q = lambda i: int(gtsam.symbol(ord('q'), i))

    @staticmethod
    def run(sequence):

        # build graph / estimate
        graph, initial_estimate = System.build_graph(sequence)

        # draw initial system
        plotting = MPLDrawing('initial_problem')
        plotting.plot_system(graph, initial_estimate)

        # check graph + estimate
        System.check_problem(graph, initial_estimate)

        # optimize using c++ back-end
        estimate = System.optimize(graph, initial_estimate, sequence.calibration)

        # draw estimation
        plotting = MPLDrawing('final_solution')
        plotting.plot_system(graph, estimate)

        # extract quadrics / trajectory 
        estimated_trajectory = Trajectory.from_values(estimate)
        estimated_quadrics = Quadrics.from_values(estimate)

        # evaluate results
        initial_ATE_H = Evaluation.evaluate_trajectory(Trajectory.from_values(initial_estimate), sequence.true_trajectory, horn=True)[0]
        estimate_ATE_H = Evaluation.evaluate_trajectory(estimated_trajectory, sequence.true_trajectory, horn=True)[0]
        initial_ATE = Evaluation.evaluate_trajectory(Trajectory.from_values(initial_estimate), sequence.true_trajectory, horn=False)[0]
        estimate_ATE = Evaluation.evaluate_trajectory(estimated_trajectory, sequence.true_trajectory, horn=False)[0]
        print('Initial ATE w/ horn alignment: {}'.format(initial_ATE_H))
        print('Final ATE w/ horn alignment: {}'.format(estimate_ATE_H))
        print('Initial ATE w/ weak alignment: {}'.format(initial_ATE))
        print('Final ATE w/ weak alignment: {}'.format(estimate_ATE))

        # # plot results
        trajectories = [Trajectory.from_values(initial_estimate), estimated_trajectory, sequence.true_trajectory]
        maps = [Quadrics.from_values(initial_estimate), estimated_quadrics, sequence.true_quadrics]
        colors = ['r', 'm', 'g']; names = ['initial_estimate', 'final_estimate', 'ground_truth']
        plotting.plot_result(trajectories, maps, colors, names)

        


    @staticmethod
    def check_problem(graph, estimate):
        # check trajectory
        # must be prior pose 
        # must be odometry between each pose variable
        # check quadrics
        # each quadric must be viewed from 3 poses

        factor_keys = []
        for i in range(graph.size()):
            factor = graph.at(i)
            int_keys = [factor.keys().at(j) for j in range(factor.keys().size())]
            factor_keys += ['{}{}'.format(chr(gtsam.symbolChr(key)), gtsam.symbolIndex(key)) for key in int_keys]

        estimate_int_keys = [estimate.keys().at(i) for i in range(estimate.keys().size())]
        estimate_keys = ['{}{}'.format(chr(gtsam.symbolChr(key)), gtsam.symbolIndex(key)) for key in estimate_int_keys]

        # check each factor has a variable
        # and each quadric is viewed 3 times
        for factor_key in factor_keys:
            
            if factor_key not in estimate_keys:
                print(factor_key, 'doesnt exist in estimate!')

        for estimate_key in estimate_keys:

            if 'q' in estimate_key:

                mkeys = [fkey for fkey in factor_keys if fkey==estimate_key]
                if len(mkeys) < 3:
                    print(estimate_key, 'doesnt have 3 bbfs')
            
    
    @staticmethod
    def optimize(graph, initial_estimate, calibration):

        # create optimizer parameters
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")    # SILENT = 0, SUMMARY, TERMINATION, LAMBDA, TRYLAMBDA, TRYCONFIG, DAMPED, TRYDELTA : VALUES, ERROR 
        params.setMaxIterations(100)
        params.setlambdaInitial(1e-5)       # defaults to 1e5
        params.setlambdaUpperBound(1e10)     # defaults to 1e5
        params.setlambdaLowerBound(1e-8)    # defaults to 0.0
        params.setRelativeErrorTol(1e-10)   # stop iterating when change in error between steps is less than this
        params.setAbsoluteErrorTol(1e-8)    # stop when cost-costchange < tol
  
        # create optimizer
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

        # run optimizer
        print('starting optimization')
        estimate = optimizer.optimize()
        print('optimization finished')

        return estimate


    @staticmethod
    def build_graph(sequence):
        """
        Sequence contains:
        * calibration
        * true_trajectory 
        * true_quadrics 
        * true_boxes 
        """

        # create empty graph / estimate
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()

        # declare noise models
        ODOM_SIGMA = 0.01; BOX_SIGMA = 5
        ODOM_NOISE = 0.01; BOX_NOISE = 0.0
        prior_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([1e-1]*6, dtype=np.float))
        odometry_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([ODOM_SIGMA]*3 + [ODOM_SIGMA]*3, dtype=np.float))
        bbox_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([BOX_SIGMA]*4, dtype=np.float))

        # get noisy odometry / boxes 
        true_odometry = sequence.true_trajectory.as_odometry()
        noisy_odometry = true_odometry.add_noise(mu=0.0, sd=ODOM_NOISE)
        noisy_boxes = sequence.true_boxes.add_noise(mu=0.0, sd=BOX_NOISE)

        # initialize trajectory
        # TODO: ensure aligned in same reference frame
        initial_trajectory = noisy_odometry.as_trajectory(sequence.true_trajectory.data()[0])
        # initial_trajectory = noisy_odometry.as_trajectory()

        # initialize quadrics
        # NOTE: careful initializing with true quadrics and noise traj as it may not make sense
        # initial_quadrics = sequence.true_quadrics
        initial_quadrics = System.initialize_quadrics(initial_trajectory, noisy_boxes, sequence.calibration)

        # add prior pose
        prior_factor = gtsam.PriorFactorPose3(System.X(0), initial_trajectory.at(0), prior_noise)
        graph.add(prior_factor)

        # add odometry measurements
        for (start_key, end_key), rpose in noisy_odometry.items():
            odometry_factor = gtsam.BetweenFactorPose3(System.X(start_key), System.X(end_key), rpose, odometry_noise)
            graph.add(odometry_factor)

        # add initial pose estimates
        for pose_key, pose in initial_trajectory.items():
            initial_estimate.insert(System.X(pose_key), pose)

        # add valid box measurements
        valid_objects = []
        initialized_quadrics = initial_quadrics.keys()
        for object_key in np.unique(noisy_boxes.object_keys()):

            # add if quadric initialized
            if object_key in initialized_quadrics:
                
                # get all views of quadric
                object_boxes = noisy_boxes.at_object(object_key)

                # add if enough views
                if len(object_boxes) > 3:

                    # add measurements
                    valid_objects.append(object_key)
                    for (pose_key, t), box in object_boxes.items():
                        bbf = quadricslam.BoundingBoxFactor(box, sequence.calibration, System.X(pose_key), System.Q(object_key), bbox_noise)
                        bbf.addToGraph(graph)

        # add initial landmark estimates
        for object_key, quadric in initial_quadrics.items():

            # add if seen > 3 times
            if (object_key in valid_objects):
                quadric.addToValues(initial_estimate, System.Q(object_key))

        return graph, initial_estimate

    @staticmethod
    def initialize_quadrics(trajectory, boxes, calibration):
        quadrics = Quadrics()

        # loop through object keys
        for object_key in np.unique(boxes.object_keys()):

            # get all detections at object key
            object_boxes = boxes.at_object(object_key)

            # get the poses associated with each detection 
            pose_keys = object_boxes.pose_keys()
            poses = trajectory.at_keys(pose_keys)

            # ensure quadric seen from > 3 views
            if len(np.unique(pose_keys)) < 3:
                continue

            # initialize quadric fomr views using svd
            quadric_matrix = System.quadric_SVD(poses, object_boxes, calibration)

            # constrain generic dual quadric to be ellipsoidal 
            quadric = quadricslam.ConstrainedDualQuadric.constrain(quadric_matrix)

            # check quadric is okay
            if (System.is_okay(quadric, poses, calibration)):
                quadrics.add(quadric, object_key)

        return quadrics

    @staticmethod
    def quadric_SVD(poses, object_boxes, calibration):
        """ calculates quadric_matrix using SVD """

        # iterate through box/pose data
        planes = []
        for box, pose in zip(object_boxes.data(), poses.data()):

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

    @staticmethod
    def is_okay(quadric, poses, calibration):
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



if __name__ == '__main__':
    """
    Options:
    - manual / scenenet 
    - seed 
    - sigmas (odom, box)
    - noise (odom, box, quadrics)
    - quadric init (svd, dataset)
    """
    np.random.seed(121)

    trainval = 'train'
    dataset_path = '/media/feyre/DATA1/Datasets/SceneNetRGBD/pySceneNetRGBD/data/{}'.format(trainval)
    protobuf_folder = '/media/feyre/DATA1/Datasets/SceneNetRGBD/pySceneNetRGBD/data/{}_protobufs'.format(trainval)
    reader_path = '/media/feyre/DATA1/Datasets/SceneNetRGBD/pySceneNetRGBD/scenenet_pb2.py'
    dataset = SceneNetDataset(dataset_path, protobuf_folder, reader_path)
    for sequence in dataset:
        System.run(sequence)

    # sequence = ManualSequence.sequence1()
    # System.run(sequence)





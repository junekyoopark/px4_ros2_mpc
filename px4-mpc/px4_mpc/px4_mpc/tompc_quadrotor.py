#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2023 PX4 Development Team
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted...
#
############################################################################

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import (
    QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
)

# Import your model & TOMPC solver
from px4_mpc.models.multirotor_rate_model import MultirotorRateModel
from px4_mpc.controllers.multirotor_rate_tompc import MultirotorRateTOMPC
from nav_msgs.msg import Path
from geometry_msgs.msg import Point

# Import your custom messages (16-parameter)
from planner_msgs.msg import FrenetPath # type: ignore


# Import other ROS messages for PX4
from visualization_msgs.msg import Marker
from px4_msgs.msg import (
    OffboardControlMode, VehicleStatus, VehicleAttitude,
    VehicleLocalPosition, VehicleRatesSetpoint
)

def vector2PoseMsg(frame_id, position, attitude):
    """
    Helper to convert [pos_x, pos_y, pos_z] + [qw, qx, qy, qz] to PoseStamped
    """
    from geometry_msgs.msg import PoseStamped
    pose_msg = PoseStamped()
    pose_msg.header.frame_id = frame_id
    pose_msg.pose.orientation.w = attitude[0]
    pose_msg.pose.orientation.x = attitude[1]
    pose_msg.pose.orientation.y = attitude[2]
    pose_msg.pose.orientation.z = attitude[3]
    pose_msg.pose.position.x = float(position[0])
    pose_msg.pose.position.y = float(position[1])
    pose_msg.pose.position.z = float(position[2])
    return pose_msg


class QuadrotorTOMPC(Node):
    def __init__(self):
        super().__init__('quadrotor_tompc')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

        # -------------------- Subscriptions --------------------
        self.trajectory_sub = self.create_subscription(
            FrenetPath,     # <--- custom message
            '/trajectory_frenet', # <--- new topic name
            self.trajectory_callback,
            10
        )
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile
        )
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.vehicle_attitude_callback,
            qos_profile
        )
        self.local_position_sub = self.create_subscription(
            VehicleLocalPosition,
            '/fmu/out/vehicle_local_position',
            self.vehicle_local_position_callback,
            qos_profile
        )

        # -------------------- Publishers --------------------
        self.publisher_offboard_mode = self.create_publisher(
            OffboardControlMode,
            '/fmu/in/offboard_control_mode',
            qos_profile
        )
        self.publisher_rates_setpoint = self.create_publisher(
            VehicleRatesSetpoint,
            '/fmu/in/vehicle_rates_setpoint',
            qos_profile
        )
        self.predicted_path_pub = self.create_publisher(
            # We'll still use nav_msgs/Path for visualization
            # but we won't subscribe to it
            # For predicted path
            # You could define your own custom msg for predicted path too.
            # We'll keep it simpler here.
            #
            # If you want to keep using nav_msgs/Path for predicted path, that's fine.
            # The difference is your reference trajectory now is TompcTrajectory.
            #
            # So we do:
            Path,
            '/px4_mpc/predicted_path',
            10
        )
        self.reference_pub = self.create_publisher(
            Marker,
            "/px4_mpc/reference",
            10
        )

        self.reference_path_pub = self.create_publisher(
            Marker,
            '/px4_mpc/reference_path',
            10
        )

        # -------------------- Timer --------------------
        timer_period = 0.02  # 20 ms
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        # -------------------- State Variables --------------------
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.vehicle_attitude = np.array([1.0, 0.0, 0.0, 0.0])  # [qw, qx, qy, qz]
        self.vehicle_local_position = np.array([0.0, 0.0, 0.0]) # [x, y, z]
        self.vehicle_local_velocity = np.array([0.0, 0.0, 0.0]) # [vx, vy, vz]

        # -------------------- MPC Setup --------------------
        MPC_HORIZON = 500
        self.model = MultirotorRateModel()

        # We'll store [px, py, pz, tx, ty, tz, nx, ny, nz, bx, by, bz, qw, qx, qy, qz]
        # 16 columns
        self.reference_trajectory = np.zeros((MPC_HORIZON, 16))

        # Create your ACADOS-based MPC object:
        self.mpc = MultirotorRateTOMPC(self.model, self.reference_trajectory)

    # --------------------------------------------------------------------------
    # ROS Callbacks
    # --------------------------------------------------------------------------
    def trajectory_callback(self, msg: FrenetPath):
        """
        Callback that receives FrenetPath, which has:
          - header
          - FrenetPoint[] waypoints  (each has 16 fields)
        We copy them into self.reference_trajectory for the MPC.
        """
        waypoints = msg.points
        horizon = min(len(waypoints), self.mpc.N)

        for i in range(horizon):
            wpt = waypoints[i]
            # convert to array of length 16
            self.reference_trajectory[i, :] = [
                wpt.px, wpt.py, wpt.pz,
                wpt.tx, wpt.ty, wpt.tz,
                wpt.nx, wpt.ny, wpt.nz,
                wpt.bx, wpt.by, wpt.bz,
                wpt.qw, wpt.qx, wpt.qy, wpt.qz
            ]

        # If fewer waypoints than N, repeat the last
        if len(waypoints) < self.mpc.N and len(waypoints) > 0:
            for i in range(len(waypoints), self.mpc.N):
                self.reference_trajectory[i, :] = self.reference_trajectory[len(waypoints) - 1, :]

        # self.get_logger().info(f"Received FrenetPath with {len(waypoints)} waypoints.")
        self.publish_reference_path()

    def publish_reference_path(self):
        """
        Publishes the reference trajectory as a line strip using visualization_msgs/Marker.
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.ns = "reference_trajectory"
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # Line thickness
        marker.color.r = 0.0
        marker.color.g = 1.0  # Green color for reference trajectory
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Add points from the reference trajectory
        for i in range(self.mpc.N):
            point = Point()
            point.x = self.reference_trajectory[i, 0]  # px
            point.y = self.reference_trajectory[i, 1]  # py
            point.z = self.reference_trajectory[i, 2]  # pz
            marker.points.append(point)

        self.reference_path_pub.publish(marker)

    def vehicle_attitude_callback(self, msg: VehicleAttitude):
        self.vehicle_attitude[0] = msg.q[0]
        self.vehicle_attitude[1] = msg.q[1]
        self.vehicle_attitude[2] = -msg.q[2]
        self.vehicle_attitude[3] = -msg.q[3]

    def vehicle_local_position_callback(self, msg: VehicleLocalPosition):
        self.vehicle_local_position[0] = msg.x
        self.vehicle_local_position[1] = -msg.y
        self.vehicle_local_position[2] = -msg.z
        self.vehicle_local_velocity[0] = msg.vx
        self.vehicle_local_velocity[1] = -msg.vy
        self.vehicle_local_velocity[2] = -msg.vz

    def vehicle_status_callback(self, msg: VehicleStatus):
        self.nav_state = msg.nav_state

    # --------------------------------------------------------------------------
    # Visualization Helper
    # --------------------------------------------------------------------------
    def publish_reference(self, pub, reference):
        """
        Publishes a small sphere marker at the reference position
        reference = [x, y, z, ...].
        """
        msg = Marker()
        msg.action = Marker.ADD
        msg.header.frame_id = "map"
        msg.ns = "arrow"
        msg.id = 1
        msg.type = Marker.SPHERE
        msg.scale.x = 0.5
        msg.scale.y = 0.5
        msg.scale.z = 0.5
        msg.color.r = 1.0
        msg.color.g = 0.0
        msg.color.b = 0.0
        msg.color.a = 1.0
        msg.pose.position.x = reference[0]
        msg.pose.position.y = reference[1]
        msg.pose.position.z = reference[2]
        msg.pose.orientation.w = 1.0
        pub.publish(msg)

    # --------------------------------------------------------------------------
    # Main Control Loop
    # --------------------------------------------------------------------------
    def cmdloop_callback(self):
        """
        Periodic function: solves MPC, publishes predicted path + control commands.
        """
        # 1) Publish offboard mode
        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = True
        self.publisher_offboard_mode.publish(offboard_msg)

        # 2) Construct current state x0
        x0 = np.array([
            self.vehicle_local_position[0],
            self.vehicle_local_position[1],
            self.vehicle_local_position[2],
            self.vehicle_local_velocity[0],
            self.vehicle_local_velocity[1],
            self.vehicle_local_velocity[2],
            self.vehicle_attitude[0],
            self.vehicle_attitude[1],
            self.vehicle_attitude[2],
            self.vehicle_attitude[3]
        ]).reshape(10, 1)

        # 3) Solve MPC
        try:
            u_pred, x_pred = self.mpc.solve(x0, self.reference_trajectory)
        except Exception as e:
            self.get_logger().error(f"MPC solver failed: {e}")
            return

        # 4) Publish predicted path as nav_msgs/Path for Rviz
        from nav_msgs.msg import Path
        predicted_path_msg = Path()
        predicted_path_msg.header.frame_id = 'map'
        for predicted_state in x_pred:
            # predicted_state[0:3] = position
            # predicted_state[6:10] = quaternion
            pose_msg = vector2PoseMsg(
                'map',
                predicted_state[0:3],
                predicted_state[6:10]
            )
            predicted_path_msg.poses.append(pose_msg)
        self.predicted_path_pub.publish(predicted_path_msg)

        # 5) Publish reference marker (just the first waypoint)
        self.publish_reference(self.reference_pub, self.reference_trajectory[0, 0:3])

        # 6) Convert MPC output to PX4 body rates
        thrust_rates = u_pred[0, :]
        thrust_command = -(thrust_rates[0] * 0.07)  # Example offset

        # 7) Publish setpoints if OFFBOARD
        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            setpoint_msg = VehicleRatesSetpoint()
            setpoint_msg.timestamp = int(Clock().now().nanoseconds / 1000)
            setpoint_msg.roll  = float(thrust_rates[1])
            setpoint_msg.pitch = float(-thrust_rates[2])
            setpoint_msg.yaw   = float(-thrust_rates[3])
            setpoint_msg.thrust_body[0] = 0.0
            setpoint_msg.thrust_body[1] = 0.0
            setpoint_msg.thrust_body[2] = float(thrust_command)
            self.publisher_rates_setpoint.publish(setpoint_msg)


def main(args=None):
    rclpy.init(args=args)
    quadrotor_mpc = QuadrotorTOMPC()
    rclpy.spin(quadrotor_mpc)
    quadrotor_mpc.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

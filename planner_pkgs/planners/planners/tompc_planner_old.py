#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
import math

class TrajectoryGenerator(Node):

    def __init__(self):
        super().__init__('trajectory_generator')
        self.publisher_ = self.create_publisher(Path, '/trajectory', 10)
        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Parameters for trajectory generation
        self.start_time = self.get_clock().now()
        self.trajectory_type = "circle"  # Options: 'line', 'circle', 'spiral'
        self.horizon = 15  # Number of steps in the trajectory
        self.step_size = 0.5  # Spacing between waypoints in meters
        self.angular_speed = 0.5  # Angular speed for circular trajectory (rad/s)

        # For circular motion, you might choose a radius:
        self.radius = 5.0

    def timer_callback(self):
        # Get the elapsed time
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

        # Generate the trajectory
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for i in range(self.horizon):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = self.get_clock().now().to_msg()

            # Calculate waypoint
            if self.trajectory_type == "line":
                # Linear trajectory (moving along x-axis)
                # position
                x = i * self.step_size
                y = 0.0
                z = 3.0
                # orientation: facing forward along +x
                yaw = 0.0

            elif self.trajectory_type == "circle":
                # Circular trajectory
                angle = elapsed_time * self.angular_speed + i * 0.1
                x = self.radius * math.cos(angle)
                y = self.radius * math.sin(angle)
                z = 3.0
                # orientation: we want to face tangentially
                # yaw = angle + pi/2 to face along tangent if desired
                # or just use angle
                yaw = angle + math.pi/2.0

            elif self.trajectory_type == "spiral":
                # Spiral trajectory
                angle = elapsed_time * self.angular_speed + i * 0.1
                r = 0.1 * (i + 1)  # radius grows slowly
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                z = 2.0 + 0.1 * i
                # orientation
                yaw = angle + math.pi/2.0

            # Convert yaw to quaternion (roll=pitch=0)
            # yaw in [0..2*pi], simple conversion
            qw = math.cos(yaw / 2.0)
            qz = math.sin(yaw / 2.0)

            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z

            # orientation with roll=0, pitch=0, yaw
            pose.pose.orientation.w = qw
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = qz

            # Add pose to the path
            path_msg.poses.append(pose)

        # Publish the trajectory
        self.publisher_.publish(path_msg)
        self.get_logger().info('Published trajectory with {} waypoints.'.format(len(path_msg.poses)))


def main(args=None):
    rclpy.init(args=args)
    trajectory_generator = TrajectoryGenerator()
    rclpy.spin(trajectory_generator)

    trajectory_generator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

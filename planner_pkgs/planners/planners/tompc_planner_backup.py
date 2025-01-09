#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np

# Import our custom messages
from planner_msgs.msg import FrenetPoint, FrenetPath

class TrajectoryGenerator(Node):

    def __init__(self):
        super().__init__('trajectory_generator')

        # Publisher for FrenetPath
        self.publisher_ = self.create_publisher(FrenetPath, '/trajectory_frenet', 10)

        self.timer_period = 0.1  # seconds
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Parameters for trajectory generation
        self.start_time = self.get_clock().now()
        self.trajectory_type = "circle"  # 'line', 'circle', 'spiral'
        self.horizon = 50               # Number of waypoints
        self.step_size = 0.5            # For line
        self.angular_speed = 0.1        # For circle or spiral
        self.radius = 15.0               # Circle radius

    def timer_callback(self):
        # 1) Compute time
        elapsed_time = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9

        # 2) Create FrenetPath message
        frenet_path_msg = FrenetPath()
        frenet_path_msg.header.stamp = self.get_clock().now().to_msg()
        frenet_path_msg.header.frame_id = "map"

        # 3) Build FrenetPoints
        for i in range(self.horizon):
            fp = FrenetPoint()

            if self.trajectory_type == "line":
                # position
                x = i * self.step_size
                y = 0.0
                z = 3.0
                yaw = 0.0

            elif self.trajectory_type == "circle":
                angle = elapsed_time * self.angular_speed + i * 0.1
                x = self.radius * math.cos(angle)
                y = self.radius * math.sin(angle)
                z = 20.0
                yaw = angle + math.pi / 2.0

            elif self.trajectory_type == "spiral":
                angle = elapsed_time * self.angular_speed + i * 0.1
                r = 0.1 * (i + 1)
                x = r * math.cos(angle)
                y = r * math.sin(angle)
                z = 2.0 + 0.1 * i
                yaw = angle + math.pi / 2.0
            else:
                # fallback: line
                x = i * self.step_size
                y = 0.0
                z = 3.0
                yaw = 0.0

            # Convert yaw to quaternion (roll = pitch = 0)
            qw = math.cos(yaw * 0.5)
            qx = 0.0
            qy = 0.0
            qz = math.sin(yaw * 0.5)

            # Fill position
            fp.px = x
            fp.py = y
            fp.pz = z

            # Dummy Frenet vectors (T, N, B)
            fp.tx, fp.ty, fp.tz = 1.0, 0.0, 0.0
            fp.nx, fp.ny, fp.nz = 0.0, 1.0, 0.0
            fp.bx, fp.by, fp.bz = 0.0, 0.0, 1.0

            # Orientation
            fp.qw = qw
            fp.qx = qx
            fp.qy = qy
            fp.qz = qz

            # Append to FrenetPath
            frenet_path_msg.points.append(fp)

        # 4) Publish FrenetPath
        self.publisher_.publish(frenet_path_msg)
        self.get_logger().info(f"Published FrenetPath with {len(frenet_path_msg.points)} points.")


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np

from planner_msgs.msg import FrenetPoint, FrenetPath
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy
)

class TrajectoryGenerator(Node):

    def __init__(self):
        super().__init__('trajectory_generator')

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

        self.publisher_ = self.create_publisher(
            FrenetPath, '/trajectory_frenet', 10
        )
        self.status_subscriber_ = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.status_callback, qos_profile
        )
        self.position_subscriber_ = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.position_callback, qos_profile
        )

        # Publish the path every 0.1 s (10 Hz).
        # Increase frequency (e.g., 0.05 for 20 Hz) if you need tighter updates.
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Trajectory parameters
        self.horizon = 20
        self.angular_step = 0.08    # Angle step (radians) between consecutive waypoints
        self.target_alt = 20.0     # Desired altitude
        self.nav_state = VehicleStatus.NAVIGATION_STATE_AUTO_LOITER
        self.current_position = np.array([0.0, 0.0, 0.0])

        # Circle info
        self.circle_center = np.array([3.0, 3.0])  # Latched once at OFFBOARD entry
        self.initialized = False

    def status_callback(self, msg):
        self.nav_state = msg.nav_state

    def position_callback(self, msg):
        self.current_position = np.array([msg.x, -1.0 * msg.y, -1.0 * msg.z])

    def timer_callback(self):
        # Only act if in OFFBOARD
        if self.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            return

        # # Latch the circle center once when first entering OFFBOARD.
        # if not self.initialized:
        #     # Use the drone’s current XY as circle center. 
        #     self.circle_center = np.array([
        #         self.current_position[0], 
        #         self.current_position[1]
        #     ])
        #     self.initialized = True
        #     self.get_logger().info(
        #         f"Latched circle center at OFFBOARD entry: "
        #         f"({self.circle_center[0]:.2f}, {self.circle_center[1]:.2f})"
        #     )

        # Create FrenetPath
        frenet_path_msg = FrenetPath()
        frenet_path_msg.header.stamp = self.get_clock().now().to_msg()
        frenet_path_msg.header.frame_id = "map"

        # Current drone position
        cur_x, cur_y, cur_z = self.current_position
        cx, cy = self.circle_center

        # Radius = distance from center -> drone
        dx_0 = cur_x - cx
        dy_0 = cur_y - cy
        radius_0 = math.sqrt(dx_0*dx_0 + dy_0*dy_0)
        # If radius is extremely small, clamp to avoid NaNs
        if radius_0 < 0.001:
            radius_0 = 0.001

        # Angle from center->drone
        angle_0 = math.atan2(dy_0, dx_0)

        # ------------------
        # First waypoint: EXACTLY drone's current position
        # so there's no "detachment" from the drone
        # ------------------
        fp0 = FrenetPoint()
        fp0.px = cur_x
        fp0.py = cur_y
        fp0.pz = cur_z

        # Yaw tangent to the circle
        yaw_0 = angle_0 + math.pi / 2.0
        fp0.qw = math.cos(yaw_0 * 0.5)
        fp0.qx = 0.0
        fp0.qy = 0.0
        fp0.qz = math.sin(yaw_0 * 0.5)

        # Dummy Frenet vectors
        fp0.tx, fp0.ty, fp0.tz = 1.0, 0.0, 0.0
        fp0.nx, fp0.ny, fp0.nz = 0.0, 1.0, 0.0
        fp0.bx, fp0.by, fp0.bz = 0.0, 0.0, 1.0

        frenet_path_msg.points.append(fp0)

        # ------------------
        # Subsequent waypoints:
        #   angle_i = angle_0 + i*self.angular_step
        #   x_i, y_i on same circle center
        #   z_i interpolates from current_z to target_alt
        # ------------------
        for i in range(1, self.horizon):
            fp = FrenetPoint()

            angle_i = angle_0 + i * self.angular_step
            # If you see it drifting in the -y direction,
            # try flipping the sign of math.sin(angle_i):
            x_i = cx + radius_0 * math.cos(angle_i)
            y_i = cy + radius_0 * math.sin(angle_i)
            # y_i = cy - radius_0 * math.sin(angle_i)  # <--- flip sign if needed

            # Linear interpolation from cur_z to 20.0
            frac = float(i) / float(self.horizon - 1) if (self.horizon > 1) else 1.0
            z_i = cur_z + frac * (self.target_alt - cur_z)

            # Yaw tangent to circle
            yaw_i = angle_i + math.pi / 2.0
            qw_i = math.cos(yaw_i * 0.5)
            qz_i = math.sin(yaw_i * 0.5)

            fp.px = x_i
            fp.py = y_i
            fp.pz = z_i

            fp.qw = qw_i
            fp.qx = 0.0
            fp.qy = 0.0
            fp.qz = qz_i

            # Dummy Frenet vectors
            fp.tx, fp.ty, fp.tz = 1.0, 0.0, 0.0
            fp.nx, fp.ny, fp.nz = 0.0, 1.0, 0.0
            fp.bx, fp.by, fp.bz = 0.0, 0.0, 1.0

            frenet_path_msg.points.append(fp)

        # Publish
        self.publisher_.publish(frenet_path_msg)
        self.get_logger().info(
            f"Published {len(frenet_path_msg.points)} pts. "
            f"First WP=({fp0.px:.2f}, {fp0.py:.2f}, {fp0.pz:.2f})"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

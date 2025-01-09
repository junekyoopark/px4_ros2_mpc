#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker

class SetPoseVisualizer(Node):
    def __init__(self):
        super().__init__('set_pose_visualizer')
        
        # Subscriber to the /set_pose topic
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/set_pose',
            self.pose_callback,
            10
        )

        # Publisher for the Marker message
        self.marker_publisher = self.create_publisher(
            Marker,
            '/visualization_marker',
            10
        )

        # Marker properties
        self.marker = Marker()
        self.marker.header.frame_id = "map"
        self.marker.ns = "set_pose_visualizer"
        self.marker.id = 0
        self.marker.type = Marker.SPHERE
        self.marker.action = Marker.ADD
        self.marker.scale.x = 0.2
        self.marker.scale.y = 0.2
        self.marker.scale.z = 0.2
        self.marker.color.r = 1.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0

    def pose_callback(self, pose_stamped):
        # Update marker position from the received pose
        self.marker.header.stamp = pose_stamped.header.stamp
        self.marker.pose = pose_stamped.pose

        # Publish the marker
        self.marker_publisher.publish(self.marker)
        self.get_logger().info(f"Visualized Pose: {pose_stamped.pose.position.x}, {pose_stamped.pose.position.y}, {pose_stamped.pose.position.z}")

def main(args=None):
    rclpy.init(args=args)
    
    visualizer = SetPoseVisualizer()
    
    rclpy.spin(visualizer)

    visualizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import math

class SetPathPublisher(Node):
    def __init__(self):
        super().__init__('set_path_publisher')

        # QoS Profile
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )

        self.path_publisher = self.create_publisher(Path, '/set_path', qos_profile)

        self.radius = 30.0  # Radius of the circular path
        self.z_height = 10.0  # Fixed height for z-axis
        self.num_waypoints = 100  # Number of waypoints in the path
        self.timer_period = 10.0  # seconds
        self.timer = self.create_timer(self.timer_period, self.publish_path)

    def generate_path(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        for i in range(self.num_waypoints):
            angle = 2 * math.pi * i / self.num_waypoints
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = 'map'
            pose.pose.position.x = self.radius * math.cos(angle)
            pose.pose.position.y = self.radius * math.sin(angle)
            pose.pose.position.z = self.z_height
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        return path

    def publish_path(self):
        path = self.generate_path()
        self.path_publisher.publish(path)
        self.get_logger().info(f'Publishing path with {len(path.poses)} waypoints.')

def main(args=None):
    rclpy.init(args=args)
    set_path_publisher = SetPathPublisher()

    rclpy.spin(set_path_publisher)

    set_path_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

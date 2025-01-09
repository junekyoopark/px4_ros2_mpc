import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from planner_msgs.srv import SendPath  # Import custom service type
import math

class SetPathPublisher(Node):
    def __init__(self):
        super().__init__('set_path_publisher')

        # Create a client for the SendPath service
        self.client = self.create_client(SendPath, '/receive_path')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /receive_path service...")

        self.timer = self.create_timer(1.0, self.send_path)

    def generate_path(self):
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = 'map'

        radius = 10.0
        z_height = 10.0
        num_waypoints = 100

        for i in range(num_waypoints):
            angle = 2 * math.pi * i / num_waypoints
            pose = PoseStamped()
            pose.pose.position.x = radius * math.cos(angle)
            pose.pose.position.y = radius * math.sin(angle)
            pose.pose.position.z = z_height
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        return path

    def send_path(self):
        # Generate the path
        path = self.generate_path()

        # Create a request and include the generated path
        request = SendPath.Request()
        request.path = path

        # Call the service
        future = self.client.call_async(request)

        rclpy.spin_until_future_complete(self, future)

        # Handle the response
        if future.result().success:
            self.get_logger().info(f"Service call successful: {future.result().message}")
            self.destroy_timer(self.timer)  # Stop sending after success
        else:
            self.get_logger().error(f"Failed to send path: {future.result().message}")

def main(args=None):
    rclpy.init(args=args)
    set_path_publisher = SetPathPublisher()
    rclpy.spin(set_path_publisher)
    set_path_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

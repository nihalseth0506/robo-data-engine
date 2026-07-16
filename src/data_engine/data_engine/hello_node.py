import rclpy
from rclpy.node import Node


class HelloNode(Node):
    def __init__(self):
        super().__init__('hello_node')
        self.get_logger().info('data_engine package is alive and talking to ROS2.')
        self.timer = self.create_timer(2.0, self.tick)
        self.count = 0

    def tick(self):
        self.count += 1
        self.get_logger().info(f'Heartbeat #{self.count}')


def main(args=None):
    rclpy.init(args=args)
    node = HelloNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
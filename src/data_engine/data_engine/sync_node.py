import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray


class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_node')

        state_sub = message_filters.Subscriber(self, JointState, '/ur5e/joint_states')
        image_sub = message_filters.Subscriber(self, Image, '/ur5e/camera/image_raw')
        cmd_sub = message_filters.Subscriber(self, JointState, '/ur5e/joint_commands')

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [state_sub, image_sub, cmd_sub],
            queue_size=30,
            slop=0.05,
        )
        self.sync.registerCallback(self.synced_callback)

        self.frame_count = 0
        self.get_logger().info('Sync node started — waiting for 3-way matched frames...')

    def synced_callback(self, state_msg, image_msg, cmd_msg):
        self.frame_count += 1

        t_state = state_msg.header.stamp.sec + state_msg.header.stamp.nanosec * 1e-9
        t_image = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
        gap_ms = abs(t_state - t_image) * 1000.0

        self.get_logger().info(
            f'Frame {self.frame_count}: 3-way match | '
            f'gap={gap_ms:.2f}ms | '
            f'joint0={state_msg.position[0]:.4f} | '
            f'cmd0={cmd_msg.position[0]:.4f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
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
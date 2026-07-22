import rclpy
from rclpy.node import Node
import message_filters
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped


class SkydioSyncNode(Node):
    def __init__(self):
        super().__init__('skydio_sync_node')

        pose_sub = message_filters.Subscriber(
            self, PoseStamped, '/skydio/pose'
        )
        image_sub = message_filters.Subscriber(
            self, Image, '/skydio/camera/image_raw'
        )
        cmd_sub = message_filters.Subscriber(
            self, JointState, '/skydio/stamped_commands'
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [pose_sub, image_sub, cmd_sub],
            queue_size=30,
            slop=0.05,
        )
        self.sync.registerCallback(self.synced_callback)

        self.frame_count = 0
        self.get_logger().info(
            'Skydio sync node started — waiting for 3-way matches...'
        )

    def synced_callback(self, pose_msg, image_msg, cmd_msg):
        self.frame_count += 1

        t_pose  = pose_msg.header.stamp.sec + pose_msg.header.stamp.nanosec * 1e-9
        t_image = image_msg.header.stamp.sec + image_msg.header.stamp.nanosec * 1e-9
        gap_ms  = abs(t_pose - t_image) * 1000.0

        pos = pose_msg.pose.position
        self.get_logger().info(
            f'Frame {self.frame_count}: 3-way match | '
            f'gap={gap_ms:.2f}ms | '
            f'pos=({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f}) | '
            f'cmd=({cmd_msg.position[0]:.1f}, '
            f'{cmd_msg.position[1]:.1f}, '
            f'{cmd_msg.position[2]:.1f}) m/s'
        )


def main(args=None):
    rclpy.init(args=args)
    node = SkydioSyncNode()
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
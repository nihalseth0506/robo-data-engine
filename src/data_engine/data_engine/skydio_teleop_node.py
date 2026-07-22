import sys
import tty
import termios
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


KEY_MAP = {
    'w': ( 1,  0,  0),   # forward  (+x)
    's': (-1,  0,  0),   # backward (-x)
    'a': ( 0,  1,  0),   # left     (+y)
    'd': ( 0, -1,  0),   # right    (-y)
    'q': ( 0,  0,  1),   # up       (+z)
    'e': ( 0,  0, -1),   # down     (-z)
}

VELOCITY_STEP = 0.5   # m/s increment per keypress


class SkydiоTeleopNode(Node):
    def __init__(self):
        super().__init__('skydio_teleop_node')

        # plain array for skydio_node's velocity subscriber
        self.vel_pub = self.create_publisher(
            Float64MultiArray,
            '/skydio/velocity_commands',
            10,
        )

        # stamped JointState for sync_node time-matching
        # we reuse JointState as a stamped container:
        # position[0]=vx, position[1]=vy, position[2]=vz
        self.cmd_pub = self.create_publisher(
            JointState,
            '/skydio/stamped_commands',
            10,
        )

        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.publish_timer = self.create_timer(0.05, self.publish_commands)

        self.get_logger().info(
            'Skydio teleop ready.\n'
            '  w/s = forward/backward\n'
            '  a/d = left/right\n'
            '  q/e = up/down\n'
            '  space = stop (zero velocity)\n'
            '  Ctrl+C to quit'
        )

        self.listener_thread = threading.Thread(
            target=self.keyboard_loop, daemon=True
        )
        self.listener_thread.start()

    def publish_commands(self):
        now = self.get_clock().now().to_msg()

        # plain array to skydio_node
        vel_msg = Float64MultiArray()
        vel_msg.data = [self.vx, self.vy, self.vz]
        self.vel_pub.publish(vel_msg)

        # stamped version for sync_node
        cmd_msg = JointState()
        cmd_msg.header.stamp = now
        cmd_msg.name = ['vx', 'vy', 'vz']
        cmd_msg.position = [self.vx, self.vy, self.vz]
        self.cmd_pub.publish(cmd_msg)

    def keyboard_loop(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                ch = sys.stdin.read(1)

                if ch == ' ':
                    self.vx = 0.0
                    self.vy = 0.0
                    self.vz = 0.0
                    self.get_logger().info('Stop — holding position')

                elif ch in KEY_MAP:
                    dvx, dvy, dvz = KEY_MAP[ch]
                    self.vx += dvx * VELOCITY_STEP
                    self.vy += dvy * VELOCITY_STEP
                    self.vz += dvz * VELOCITY_STEP
                    self.get_logger().info(
                        f'Velocity target: '
                        f'vx={self.vx:.1f} vy={self.vy:.1f} vz={self.vz:.1f} m/s'
                    )
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main(args=None):
    rclpy.init(args=args)
    node = SkydiоTeleopNode()
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
import sys
import tty
import termios
import threading
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


KEY_MAP = {
    'q': (0, +1), 'a': (0, -1),
    'w': (1, +1), 's': (1, -1),
    'e': (2, +1), 'd': (2, -1),
    'r': (3, +1), 'f': (3, -1),
    't': (4, +1), 'g': (4, -1),
    'y': (5, +1), 'h': (5, -1),
}

STEP_SIZE = 0.02

JOINT_NAMES = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
]


class TeleopNode(Node):
    def __init__(self):
        super().__init__('teleop_node')

        # publish stamped JointState so sync_node can time-match it
        self.cmd_pub = self.create_publisher(JointState, '/ur5e/joint_commands', 10)

        # also publish plain array for ur5e_node's actuator subscriber
        self.ctrl_pub = self.create_publisher(Float64MultiArray, '/ur5e/joint_ctrl', 10)

        self.targets = [-3.1416, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
        self.publish_timer = self.create_timer(0.05, self.publish_targets)

        self.get_logger().info(
            'Teleop ready. q/a w/s e/d r/f t/g y/h to jog joints 0-5. Ctrl+C to quit.'
        )

        self.listener_thread = threading.Thread(target=self.keyboard_loop, daemon=True)
        self.listener_thread.start()

    def publish_targets(self):
        now = self.get_clock().now().to_msg()

        # stamped message for sync_node
        js = JointState()
        js.header.stamp = now
        js.name = JOINT_NAMES
        js.position = self.targets
        self.cmd_pub.publish(js)

        # plain array for ur5e_node actuator control
        arr = Float64MultiArray()
        arr.data = self.targets
        self.ctrl_pub.publish(arr)

    def keyboard_loop(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                ch = sys.stdin.read(1)
                if ch in KEY_MAP:
                    idx, direction = KEY_MAP[ch]
                    self.targets[idx] += direction * STEP_SIZE
                    self.get_logger().info(
                        f'joint {idx} target -> {self.targets[idx]:.3f} rad'
                    )
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
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
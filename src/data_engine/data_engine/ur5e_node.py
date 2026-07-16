import os
import threading
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import mujoco

MODEL_PATH = os.path.expanduser(
    "~/robo-data-engine/models/ur5e_custom/ur5e.xml"
)

class UR5eNode(Node):
    def __init__(self):
        super().__init__('ur5e_node')

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        # start in home keyframe pose instead of all-zeros
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.data_lock = threading.Lock()
        self.bridge = CvBridge()

        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]

        self.renderer = mujoco.Renderer(self.model, height=240, width=320)

        self.state_pub = self.create_publisher(JointState, '/ur5e/joint_states', 10)
        self.image_pub = self.create_publisher(Image, '/ur5e/camera/image_raw', 10)

        self.cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/ur5e/joint_ctrl',
            self.command_callback,
            10,
        )

        # separate callback groups let these two timers run truly concurrently
        physics_group = MutuallyExclusiveCallbackGroup()
        camera_group = MutuallyExclusiveCallbackGroup()

        physics_dt = self.model.opt.timestep
        self.physics_timer = self.create_timer(
            physics_dt, self.physics_step, callback_group=physics_group
        )

        camera_dt = 1.0 / 30.0  # target ~30Hz, independent of physics rate
        self.camera_timer = self.create_timer(
            camera_dt, self.camera_step, callback_group=camera_group
        )

        self.get_logger().info(
            f'UR5e MuJoCo node started. {self.model.nq} joints, '
            f'physics_dt={physics_dt:.4f}s, camera_dt={camera_dt:.4f}s'
        )

    def command_callback(self, msg):
        n = min(len(msg.data), self.model.nu)
        with self.data_lock:
            for i in range(n):
                self.data.ctrl[i] = msg.data[i]

    def physics_step(self):
        with self.data_lock:
            mujoco.mj_step(self.model, self.data)
            qpos = self.data.qpos.tolist()
            qvel = self.data.qvel.tolist()

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        msg.position = qpos
        msg.velocity = qvel
        self.state_pub.publish(msg)

    def camera_step(self):
        with self.data_lock:
            self.renderer.update_scene(self.data, camera="wrist_cam")
        rgb = self.renderer.render()

        img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = UR5eNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
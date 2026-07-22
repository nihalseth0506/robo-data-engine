import os
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import mujoco


MODEL_PATH = os.path.expanduser(
    "~/robo-data-engine/mujoco_menagerie/skydio_x2/scene.xml"
)

# physics-derived constants
HOVER_THRUST = 3.2496   # N per rotor to hold altitude
MAX_THRUST   = 13.0     # N per rotor (from ctrl range)
MIN_THRUST   = 0.0      # N per rotor (rotors cannot pull)


class PIDController:
    """Simple single-axis PID controller."""
    def __init__(self, kp, ki, kd, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0

    def update(self, error, dt):
        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = (self.kp * error +
                  self.ki * self._integral +
                  self.kd * derivative)

        if self.output_limit is not None:
            output = np.clip(output, -self.output_limit, self.output_limit)

        return output


class SkydiоNode(Node):
    def __init__(self):
        super().__init__('skydio_node')

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data  = mujoco.MjData(self.model)
        self.data_lock = threading.Lock()
        self.bridge = CvBridge()

        # spawn drone at a safe hover height
        self.data.qpos[0] = 0.0   # x
        self.data.qpos[1] = 0.0   # y
        self.data.qpos[2] = 1.5   # z — 1.5m above ground
        self.data.qpos[3] = 1.0   # quaternion w
        self.data.qpos[4] = 0.0   # quaternion x
        self.data.qpos[5] = 0.0   # quaternion y
        self.data.qpos[6] = 0.0   # quaternion z
        mujoco.mj_forward(self.model, self.data)

        # set initial thrust to hover so drone doesn't drop immediately
        for i in range(self.model.nu):
            self.data.ctrl[i] = HOVER_THRUST

        # PID controllers — one per axis
        # z (altitude): most critical, needs strong response
        self.pid_z   = PIDController(kp=4.0, ki=0.5, kd=2.0, output_limit=3.0)
        # x, y (horizontal position)
        self.pid_x   = PIDController(kp=1.5, ki=0.1, kd=1.0, output_limit=1.5)
        self.pid_y   = PIDController(kp=1.5, ki=0.1, kd=1.0, output_limit=1.5)

        # target position set by teleop
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 1.5   # hover height in metres

        self.renderer = mujoco.Renderer(self.model, height=240, width=320)

        # publishers
        self.pose_pub  = self.create_publisher(PoseStamped,   '/skydio/pose',            10)
        self.twist_pub = self.create_publisher(TwistStamped,  '/skydio/twist',           10)
        self.image_pub = self.create_publisher(Image,         '/skydio/camera/image_raw', 10)
        self.ctrl_pub  = self.create_publisher(Float64MultiArray, '/skydio/thrust_commands', 10)

        # subscriber — receives velocity commands from teleop
        self.cmd_sub = self.create_subscription(
            Float64MultiArray,
            '/skydio/velocity_commands',
            self.velocity_command_callback,
            10,
        )

        # separate callback groups for physics and camera
        physics_group = MutuallyExclusiveCallbackGroup()
        camera_group  = MutuallyExclusiveCallbackGroup()

        dt = self.model.opt.timestep
        self.physics_timer = self.create_timer(dt,      self.physics_step, callback_group=physics_group)
        self.camera_timer  = self.create_timer(1/20.0,  self.camera_step,  callback_group=camera_group)

        self._dt = dt
        self.get_logger().info(
            f'Skydio X2 node started. '
            f'Hover thrust: {HOVER_THRUST:.3f} N/rotor. '
            f'Physics: {1/dt:.0f} Hz'
        )

    def velocity_command_callback(self, msg):
        """Receive [vx, vy, vz] velocity commands from teleop."""
        if len(msg.data) < 3:
            return

        # integrate velocity into position target
        # (simple position-command mode — teleop shifts the target position)
        with self.data_lock:
            self.target_x += msg.data[0] * 0.1
            self.target_y += msg.data[1] * 0.1
            self.target_z += msg.data[2] * 0.1
            self.target_z  = np.clip(self.target_z, 0.3, 5.0)

    def _compute_thrusts(self, pos, dt):
        """
        PID position controller → four rotor thrusts.
        Simple symmetric model: all four rotors share altitude correction;
        x/y corrections are applied differentially front/back and left/right.
        """
        err_x = self.target_x - pos[0]
        err_y = self.target_y - pos[1]
        err_z = self.target_z - pos[2]

        corr_z = self.pid_z.update(err_z, dt)
        corr_x = self.pid_x.update(err_x, dt)
        corr_y = self.pid_y.update(err_y, dt)

        # rotor layout (top view):
        #   1(front-left)   2(front-right)
        #   3(back-left)    4(back-right)
        #
        # +corr_x (move forward): increase back rotors, decrease front
        # +corr_y (move right):   increase left rotors, decrease right
        t1 = HOVER_THRUST + corr_z - corr_x + corr_y   # front-left
        t2 = HOVER_THRUST + corr_z - corr_x - corr_y   # front-right
        t3 = HOVER_THRUST + corr_z + corr_x + corr_y   # back-left
        t4 = HOVER_THRUST + corr_z + corr_x - corr_y   # back-right

        return [
            float(np.clip(t1, MIN_THRUST, MAX_THRUST)),
            float(np.clip(t2, MIN_THRUST, MAX_THRUST)),
            float(np.clip(t3, MIN_THRUST, MAX_THRUST)),
            float(np.clip(t4, MIN_THRUST, MAX_THRUST)),
        ]

    def physics_step(self):
        with self.data_lock:
            pos = self.data.qpos[:3].copy()
            vel = self.data.qvel[:3].copy()
            quat = self.data.qpos[3:7].copy()

            thrusts = self._compute_thrusts(pos, self._dt)
            for i, t in enumerate(thrusts):
                self.data.ctrl[i] = t

            mujoco.mj_step(self.model, self.data)

        now = self.get_clock().now().to_msg()

        # publish pose (position + orientation)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x    = float(pos[0])
        pose_msg.pose.position.y    = float(pos[1])
        pose_msg.pose.position.z    = float(pos[2])
        pose_msg.pose.orientation.w = float(quat[0])
        pose_msg.pose.orientation.x = float(quat[1])
        pose_msg.pose.orientation.y = float(quat[2])
        pose_msg.pose.orientation.z = float(quat[3])
        self.pose_pub.publish(pose_msg)

        # publish velocity
        twist_msg = TwistStamped()
        twist_msg.header.stamp = now
        twist_msg.twist.linear.x = float(vel[0])
        twist_msg.twist.linear.y = float(vel[1])
        twist_msg.twist.linear.z = float(vel[2])
        self.twist_pub.publish(twist_msg)

        # publish thrust commands (stamped via TwistStamped header trick not needed —
        # we publish a plain array here for the actuator record;
        # the sync_node will use pose which has a header)
        ctrl_msg = Float64MultiArray()
        ctrl_msg.data = thrusts
        self.ctrl_pub.publish(ctrl_msg)

    def camera_step(self):
        with self.data_lock:
            self.renderer.update_scene(self.data, camera="track")
            rgb = self.renderer.render()

        img_msg = self.bridge.cv2_to_imgmsg(rgb, encoding='rgb8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.image_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SkydiоNode()
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
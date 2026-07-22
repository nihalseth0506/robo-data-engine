import os
import argparse
import numpy as np
import pandas as pd
import cv2
import pyarrow as pa
import pyarrow.parquet as pq
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge


bridge = CvBridge()


def read_bag(bag_path):
    """Read all messages from MCAP — auto-detects embodiment from topics present."""
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {
        info.name: info.type
        for info in reader.get_all_topics_and_types()
    }

    print("Topics in bag:")
    for name, t in topic_types.items():
        print(f"  {name}  [{t}]")

    # auto-detect embodiment from which state topic exists
    if "/ur5e/joint_states" in topic_types:
        embodiment = "ur5e"
        state_topic = "/ur5e/joint_states"
        image_topic = "/ur5e/camera/image_raw"
        cmd_topic   = "/ur5e/joint_commands"
    elif "/skydio/pose" in topic_types:
        embodiment = "skydio_x2"
        state_topic = "/skydio/pose"
        image_topic = "/skydio/camera/image_raw"
        cmd_topic   = "/skydio/stamped_commands"
    else:
        raise ValueError(f"Unknown embodiment — topics: {list(topic_types.keys())}")

    print(f"\nDetected embodiment: {embodiment}")

    states   = []
    images   = []
    commands = []

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()

        if topic == state_topic:
            if embodiment == "ur5e":
                msg = deserialize_message(data, JointState)
                states.append((timestamp_ns, msg))
            else:
                msg = deserialize_message(data, PoseStamped)
                states.append((timestamp_ns, msg))

        elif topic == image_topic:
            msg = deserialize_message(data, Image)
            images.append((timestamp_ns, msg))

        elif topic == cmd_topic:
            msg = deserialize_message(data, JointState)
            commands.append((timestamp_ns, msg))

    print(f"Read: {len(states)} states, "
          f"{len(images)} images, "
          f"{len(commands)} commands")

    return states, images, commands, embodiment


def find_closest(timestamp_ns, stream, slop_ns=50_000_000):
    """Find message closest in time to timestamp_ns, within slop_ns tolerance."""
    best_msg = None
    best_gap = float("inf")

    for ts, msg in stream:
        gap = abs(ts - timestamp_ns)
        if gap < best_gap:
            best_gap = gap
            best_msg = msg

    if best_msg is None or best_gap > slop_ns:
        return None, best_gap

    return best_msg, best_gap


def extract_state(msg, embodiment):
    """Extract state vector from message, normalized per embodiment."""
    if embodiment == "ur5e":
        # JointState: first 6 values are arm joints, rest is cube free joint
        return list(msg.position)[:6], list(msg.velocity)[:6]

    else:
        # PoseStamped: [x, y, z, qw, qx, qy, qz]
        p = msg.pose.position
        q = msg.pose.orientation
        positions  = [p.x, p.y, p.z, q.w, q.x, q.y, q.z]
        velocities = [0.0] * 7   # pose msg has no velocity field
        return positions, velocities


def extract_command(msg, embodiment):
    """Extract action vector from JointState command message."""
    # both embodiments use JointState for stamped commands:
    # ur5e:    position = [joint targets × 6]
    # skydio:  position = [vx, vy, vz]
    return list(msg.position)


def export_episode(bag_path, output_dir, episode_id):
    """Convert one MCAP bag into parquet + video."""
    os.makedirs(output_dir, exist_ok=True)

    states, images, commands, embodiment = read_bag(bag_path)

    if not images:
        print("No images found — nothing to export.")
        return

    # build video from first image dimensions
    first_rgb = bridge.imgmsg_to_cv2(images[0][1], desired_encoding="rgb8")
    h, w = first_rgb.shape[:2]
    video_path = os.path.join(output_dir, f"episode_{episode_id:03d}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    duration_s = (images[-1][0] - images[0][0]) / 1e9
    fps = len(images) / duration_s if duration_s > 0 else 10.0
    fps = float(np.clip(fps, 1.0, 30.0))

    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    print(f"\nVideo: {w}x{h} @ {fps:.1f} fps")
    print(f"Matching {len(images)} images...")

    rows    = []
    skipped = 0

    for img_ts_ns, img_msg in images:
        state_msg, state_gap = find_closest(img_ts_ns, states)
        cmd_msg,   cmd_gap   = find_closest(img_ts_ns, commands)

        if state_msg is None or cmd_msg is None:
            skipped += 1
            continue

        # extract embodiment-specific vectors
        joint_positions, joint_velocities = extract_state(state_msg, embodiment)
        joint_commands = extract_command(cmd_msg, embodiment)

        # write image frame to video
        rgb = bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

        rows.append({
            "timestamp_ns":      img_ts_ns,
            "episode_id":        episode_id,
            "embodiment":        embodiment,
            "joint_positions":   joint_positions,
            "joint_velocities":  joint_velocities,
            "joint_commands":    joint_commands,
            "state_gap_ms":      state_gap / 1e6,
            "cmd_gap_ms":        cmd_gap   / 1e6,
        })

    writer.release()
    print(f"Matched {len(rows)} frames, skipped {skipped}")

    # write parquet with fixed schema
    schema = pa.schema([
        pa.field("timestamp_ns",     pa.int64()),
        pa.field("episode_id",       pa.int32()),
        pa.field("embodiment",       pa.string()),
        pa.field("joint_positions",  pa.list_(pa.float64())),
        pa.field("joint_velocities", pa.list_(pa.float64())),
        pa.field("joint_commands",   pa.list_(pa.float64())),
        pa.field("state_gap_ms",     pa.float64()),
        pa.field("cmd_gap_ms",       pa.float64()),
    ])

    df    = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, schema=schema)
    parquet_path = os.path.join(output_dir, f"episode_{episode_id:03d}.parquet")
    pq.write_table(table, parquet_path)

    print(f"\nExported:")
    print(f"  Embodiment: {embodiment}")
    print(f"  Video:      {video_path}")
    print(f"  Parquet:    {parquet_path}")
    print(f"  Rows:       {len(rows)}")

    if rows:
        print(f"\nSample row (first frame):")
        print(f"  timestamp_ns:    {rows[0]['timestamp_ns']}")
        print(f"  joint_positions: "
              f"{[f'{v:.4f}' for v in rows[0]['joint_positions']]}")
        print(f"  joint_commands:  "
              f"{[f'{v:.4f}' for v in rows[0]['joint_commands']]}")
        print(f"  state_gap_ms:    {rows[0]['state_gap_ms']:.3f}")
        print(f"  cmd_gap_ms:      {rows[0]['cmd_gap_ms']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export MCAP episode to parquet + video. "
                    "Embodiment auto-detected from topics."
    )
    parser.add_argument("bag_path",      help="Path to episode bag folder")
    parser.add_argument("--output-dir",  default="datasets/processed")
    parser.add_argument("--episode-id",  type=int, default=1)
    args = parser.parse_args()

    export_episode(args.bag_path, args.output_dir, args.episode_id)
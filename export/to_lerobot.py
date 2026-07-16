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
from cv_bridge import CvBridge


EMBODIMENT = "ur5e"
bridge = CvBridge()


def read_bag(bag_path):
    """Read all messages from MCAP bag, return three sorted lists."""
    storage_options = StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    joint_states = []   # list of (timestamp_ns, JointState)
    images = []         # list of (timestamp_ns, Image)
    joint_commands = [] # list of (timestamp_ns, JointState)

    topic_types = {
        info.name: info.type
        for info in reader.get_all_topics_and_types()
    }

    print("Topics in bag:")
    for name, t in topic_types.items():
        print(f"  {name}  [{t}]")

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()

        if topic == "/ur5e/joint_states":
            msg = deserialize_message(data, JointState)
            joint_states.append((timestamp_ns, msg))

        elif topic == "/ur5e/camera/image_raw":
            msg = deserialize_message(data, Image)
            images.append((timestamp_ns, msg))

        elif topic == "/ur5e/joint_commands":
            msg = deserialize_message(data, JointState)
            joint_commands.append((timestamp_ns, msg))

    print(f"\nRead: {len(joint_states)} joint_states, "
          f"{len(images)} images, "
          f"{len(joint_commands)} joint_commands")

    return joint_states, images, joint_commands


def find_closest(timestamp_ns, stream, slop_ns=50_000_000):
    """Find the message in stream closest to timestamp_ns within slop_ns."""
    best = None
    best_gap = float("inf")

    for ts, msg in stream:
        gap = abs(ts - timestamp_ns)
        if gap < best_gap:
            best_gap = gap
            best = (ts, msg)

    if best is None or best_gap > slop_ns:
        return None, best_gap

    return best[1], best_gap


def export_episode(bag_path, output_dir, episode_id):
    """Convert one MCAP bag into parquet + video."""
    os.makedirs(output_dir, exist_ok=True)

    joint_states, images, joint_commands = read_bag(bag_path)

    if not images:
        print("No images found — nothing to export.")
        return

    # video writer — matches image dimensions
    first_img = bridge.imgmsg_to_cv2(images[0][1], desired_encoding="rgb8")
    h, w = first_img.shape[:2]
    video_path = os.path.join(output_dir, f"episode_{episode_id:03d}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = len(images) / ((images[-1][0] - images[0][0]) / 1e9)
    fps = max(1.0, min(fps, 30.0))  # clamp to sane range
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))

    print(f"\nVideo: {w}x{h} @ {fps:.1f} fps")
    print(f"Matching {len(images)} images to closest state + command...")

    rows = []
    skipped = 0

    for img_ts_ns, img_msg in images:
        state_msg, state_gap = find_closest(img_ts_ns, joint_states)
        cmd_msg, cmd_gap = find_closest(img_ts_ns, joint_commands)

        if state_msg is None or cmd_msg is None:
            skipped += 1
            continue

        # convert image and write to video
        rgb = bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

        # build dataset row
        row = {
            "timestamp_ns":      img_ts_ns,
            "episode_id":        episode_id,
            "embodiment":        EMBODIMENT,
            "joint_positions":   list(state_msg.position)[:6],
            "joint_velocities":  list(state_msg.velocity)[:6],
            "joint_commands":    list(cmd_msg.position),
            "state_gap_ms":      state_gap / 1e6,
            "cmd_gap_ms":        cmd_gap / 1e6,
        }
        rows.append(row)

    writer.release()
    print(f"Matched {len(rows)} frames, skipped {skipped}")

    # write parquet
    df = pd.DataFrame(rows)

    # pyarrow needs list columns as fixed-size arrays
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

    table = pa.Table.from_pandas(df, schema=schema)
    parquet_path = os.path.join(output_dir, f"episode_{episode_id:03d}.parquet")
    pq.write_table(table, parquet_path)

    print(f"\nExported:")
    print(f"  Video:   {video_path}")
    print(f"  Parquet: {parquet_path}")
    print(f"  Rows:    {len(rows)}")
    print(f"\nSample row (first frame):")
    print(f"  timestamp_ns:     {rows[0]['timestamp_ns']}")
    print(f"  joint_positions:  {[f'{v:.4f}' for v in rows[0]['joint_positions']]}")
    print(f"  joint_commands:   {[f'{v:.4f}' for v in rows[0]['joint_commands']]}")
    print(f"  state_gap_ms:     {rows[0]['state_gap_ms']:.3f}")
    print(f"  cmd_gap_ms:       {rows[0]['cmd_gap_ms']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MCAP episode to parquet + video")
    parser.add_argument("bag_path", help="Path to the episode bag folder")
    parser.add_argument("--output-dir", default="datasets/processed", help="Output directory")
    parser.add_argument("--episode-id", type=int, default=2, help="Episode number")
    args = parser.parse_args()

    export_episode(args.bag_path, args.output_dir, args.episode_id)
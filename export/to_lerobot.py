import os
import json
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
#from cv_bridge import CvBridge

#bridge = CvBridge()
CHUNKS_SIZE = 1000
CODEBASE_VERSION = "v2.1"


def read_bag(bag_path):
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

    states, images, commands = [], [], []

    while reader.has_next():
        topic, data, timestamp_ns = reader.read_next()
        if topic == state_topic:
            msg = deserialize_message(
                data, JointState if embodiment == "ur5e" else PoseStamped
            )
            states.append((timestamp_ns, msg))
        elif topic == image_topic:
            msg = deserialize_message(data, Image)
            images.append((timestamp_ns, msg))
        elif topic == cmd_topic:
            msg = deserialize_message(data, JointState)
            commands.append((timestamp_ns, msg))

    print(f"Read: {len(states)} states, {len(images)} images, "
          f"{len(commands)} commands")
    return states, images, commands, embodiment


def find_closest(timestamp_ns, stream, slop_ns=50_000_000):
    best_msg, best_gap = None, float("inf")
    for ts, msg in stream:
        gap = abs(ts - timestamp_ns)
        if gap < best_gap:
            best_gap = gap
            best_msg = msg
    if best_msg is None or best_gap > slop_ns:
        return None, best_gap
    return best_msg, best_gap


def extract_state(msg, embodiment):
    if embodiment == "ur5e":
        return list(msg.position)[:6], list(msg.velocity)[:6]
    else:
        p = msg.pose.position
        q = msg.pose.orientation
        return [p.x, p.y, p.z, q.w, q.x, q.y, q.z], [0.0] * 7


def extract_command(msg):
    return list(msg.position)


def export_episode(bag_path, output_dir, episode_id, task="reach and manipulate"):
    os.makedirs(output_dir, exist_ok=True)

    states, images, commands, embodiment = read_bag(bag_path)
    if not images:
        print("No images found.")
        return

    # determine chunk and paths
    chunk = episode_id // CHUNKS_SIZE
    chunk_str = f"chunk-{chunk:03d}"
    ep_str    = f"episode_{episode_id:06d}"

    data_dir  = os.path.join(output_dir, "data",   chunk_str)
    video_dir = os.path.join(output_dir, "videos", chunk_str,
                             "observation.images.wrist")
    meta_dir  = os.path.join(output_dir, "meta")
    for d in [data_dir, video_dir, meta_dir]:
        os.makedirs(d, exist_ok=True)

    # video writer
    #first_rgb = bridge.imgmsg_to_cv2(images[0][1], desired_encoding="rgb8")

    first_msg = images[0][1]
    first_rgb = np.frombuffer(first_msg.data, dtype=np.uint8).reshape(
        first_msg.height, first_msg.width, -1
    )
    if first_msg.encoding == 'bgr8':
        first_rgb = first_rgb[:, :, ::-1]

    h, w = first_rgb.shape[:2]
    video_path = os.path.join(video_dir, f"{ep_str}.mp4")
    duration_s = (images[-1][0] - images[0][0]) / 1e9
    fps = float(np.clip(len(images) / duration_s if duration_s > 0 else 10.0,
                        1.0, 30.0))
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    print(f"\nVideo: {w}x{h} @ {fps:.1f} fps → {video_path}")
    print(f"Matching {len(images)} images...")

    rows    = []
    skipped = 0
    t0_ns   = images[0][0]   # episode start time in nanoseconds

    for frame_idx, (img_ts_ns, img_msg) in enumerate(images):
        state_msg, state_gap = find_closest(img_ts_ns, states)
        cmd_msg,   cmd_gap   = find_closest(img_ts_ns, commands)

        if state_msg is None or cmd_msg is None:
            skipped += 1
            continue

        obs_state, _ = extract_state(state_msg, embodiment)
        action        = extract_command(cmd_msg)

        rgb = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, -1
        )
        if img_msg.encoding == 'bgr8':
            bgr = rgb
        else:
            bgr = rgb[:, :, ::-1]
        writer.write(bgr)

        rows.append({
            # lerobot v2.1 required columns
            "observation.state": obs_state,
            "action":            action,
            "timestamp":         float((img_ts_ns - t0_ns) / 1e9),
            "frame_index":       frame_idx,
            "episode_index":     episode_id,
            "index":             frame_idx,   # global index — caller can offset
            "task_index":        0,
            # extra columns we keep for our own use
            "embodiment":        embodiment,
            "state_gap_ms":      state_gap / 1e6,
            "cmd_gap_ms":        cmd_gap   / 1e6,
        })

    writer.release()
    print(f"Matched {len(rows)} frames, skipped {skipped}")

    if not rows:
        print("Nothing to write.")
        return

    # parquet with lerobot v2.1 schema
    schema = pa.schema([
        pa.field("observation.state", pa.list_(pa.float64())),
        pa.field("action",            pa.list_(pa.float64())),
        pa.field("timestamp",         pa.float32()),
        pa.field("frame_index",       pa.int64()),
        pa.field("episode_index",     pa.int64()),
        pa.field("index",             pa.int64()),
        pa.field("task_index",        pa.int64()),
        pa.field("embodiment",        pa.string()),
        pa.field("state_gap_ms",      pa.float64()),
        pa.field("cmd_gap_ms",        pa.float64()),
    ])

    df    = pd.DataFrame(rows)
    table = pa.Table.from_pandas(df, schema=schema)
    parquet_path = os.path.join(data_dir, f"{ep_str}.parquet")
    pq.write_table(table, parquet_path)

    # meta/info.json
    state_dim  = len(rows[0]["observation.state"])
    action_dim = len(rows[0]["action"])
    info = {
        "codebase_version": CODEBASE_VERSION,
        "robot_type":       embodiment,
        "total_episodes":   1,
        "total_frames":     len(rows),
        "total_tasks":      1,
        "total_videos":     1,
        "total_chunks":     1,
        "chunks_size":      CHUNKS_SIZE,
        "fps":              round(fps, 1),
        "splits":           {"train": f"0:{1}"},
        "data_path":   "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path":  "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float64",
                "shape": [state_dim],
                "names": None,
            },
            "action": {
                "dtype": "float64",
                "shape": [action_dim],
                "names": None,
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [3, h, w],
                "info": {
                    "video.height":       h,
                    "video.width":        w,
                    "video.codec":        "mp4v",
                    "video.pix_fmt":      "rgb24",
                    "video.is_depth_map": False,
                    "video.fps":          round(fps, 1),
                    "video.channels":     3,
                    "has_audio":          False,
                },
            },
            "timestamp":     {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
        },
    }
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # meta/episodes.jsonl — one JSON object per line (jsonlines format)
    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        f.write(json.dumps({"episode_index": episode_id,
                            "tasks": [task],
                            "length": len(rows)}) + "\n")

    # meta/episodes_stats.jsonl — per-episode statistics for normalization
    import statistics

    positions_flat = [rows[i]["observation.state"] for i in range(len(rows))]
    actions_flat   = [rows[i]["action"] for i in range(len(rows))]

    def col_stats(matrix, dim):
        col = [matrix[r][dim] for r in range(len(matrix))]
        mean = sum(col) / len(col)
        variance = sum((x - mean) ** 2 for x in col) / len(col)
        std = variance ** 0.5
        return mean, std, min(col), max(col)

    state_dim  = len(rows[0]["observation.state"])
    action_dim = len(rows[0]["action"])

    state_stats  = [col_stats(positions_flat, d) for d in range(state_dim)]
    action_stats = [col_stats(actions_flat,   d) for d in range(action_dim)]

    ep_stats = {
        "episode_index": episode_id,
        "stats": {
            "observation.state": {
                "mean":  [state_stats[d][0] for d in range(state_dim)],
                "std":   [state_stats[d][1] for d in range(state_dim)],
                "min":   [state_stats[d][2] for d in range(state_dim)],
                "max":   [state_stats[d][3] for d in range(state_dim)],
                "count": [len(rows)],
            },
            "action": {
                "mean":  [action_stats[d][0] for d in range(action_dim)],
                "std":   [action_stats[d][1] for d in range(action_dim)],
                "min":   [action_stats[d][2] for d in range(action_dim)],
                "max":   [action_stats[d][3] for d in range(action_dim)],
                "count": [len(rows)],
            },
            "timestamp": {
                "mean":  [sum(r["timestamp"] for r in rows) / len(rows)],
                "std":   [0.0],
                "min":   [rows[0]["timestamp"]],
                "max":   [rows[-1]["timestamp"]],
                "count": [len(rows)],
            },
        }
    }

    with open(os.path.join(meta_dir, "episodes_stats.jsonl"), "w") as f:
        f.write(json.dumps(ep_stats) + "\n")

    # meta/tasks.jsonl — one JSON object per line
    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task}) + "\n")

    print(f"\nExported (lerobot v2.1 format):")
    print(f"  Embodiment: {embodiment}")
    print(f"  Parquet:    {parquet_path}")
    print(f"  Video:      {video_path}")
    print(f"  Meta:       {meta_dir}/")
    print(f"  Rows:       {len(rows)}")
    print(f"\nSample row:")
    print(f"  observation.state: {[f'{v:.4f}' for v in rows[0]['observation.state']]}")
    print(f"  action:            {[f'{v:.4f}' for v in rows[0]['action']]}")
    print(f"  timestamp:         {rows[0]['timestamp']:.4f}s")
    print(f"  state_gap_ms:      {rows[0]['state_gap_ms']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export MCAP episode to lerobot v2.1 format. "
                    "Embodiment auto-detected from bag topics."
    )
    parser.add_argument("bag_path",     help="Path to episode bag folder")
    parser.add_argument("--output-dir", default="datasets/processed")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--task",       default="reach and manipulate")
    args = parser.parse_args()

    export_episode(args.bag_path, args.output_dir,
                   args.episode_id, args.task)
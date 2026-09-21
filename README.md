# robo-data-engine

A robot data collection and synchronization pipeline built on ROS2 Jazzy and MuJoCo 3.10.

## What this is

Most robot learning projects focus on the AI model. This project focuses on the infrastructure underneath — the layer that makes robot learning at scale possible: capturing synchronized, timestamped, multi-modal sensor data from heterogeneous robot embodiments in a format any training pipeline can consume.

The pipeline is deliberately embodiment-agnostic. The same recording, synchronization, and export code runs unmodified against a 6-DOF manipulator arm and (Phase 2) a quadrotor drone — robots with fundamentally different state representations, sensor suites, and control interfaces.

## Architecture

```
teleop_node          →  /ur5e/joint_commands  (JointState,       20 Hz)
                     →  /ur5e/joint_ctrl      (Float64MultiArray, 20 Hz)

ur5e_node            →  /ur5e/joint_states    (JointState,       ~470 Hz)
                     →  /ur5e/camera/image_raw (Image,           ~20 Hz)

sync_node            ←  all three topics
                         3-way ApproximateTimeSynchronizer (slop = 50 ms)
                         verified sub-millisecond alignment

ros2 bag record      →  datasets/raw/episode_NNN/  (MCAP format)

export/to_lerobot.py →  datasets/processed/data/chunk-000/episode_NNNNNN.parquet
                     →  datasets/processed/videos/chunk-000/observation.images.wrist/episode_NNNNNN.mp4
                     →  datasets/processed/meta/{info,episodes,tasks}.json
```

## Key engineering decisions

**Multi-threaded executor with callback groups** — physics stepping (~470 Hz) and camera rendering (~20 Hz) run on genuinely separate threads, isolated by `MutuallyExclusiveCallbackGroup`. A `threading.Lock` protects shared MuJoCo `data` access, held only during the minimal critical window (scene update only, not the full render call), so neither thread blocks the other unnecessarily.

**Stamped command messages** — `Float64MultiArray` carries no timestamp, making it incompatible with `ApproximateTimeSynchronizer`. A parallel `JointState`-typed command topic on `/ur5e/joint_commands` provides the header field needed for three-way time alignment, while the actuator control path continues to receive a plain `Float64MultiArray` on `/ur5e/joint_ctrl` — decoupling the recording concern from the control concern.

**Embodiment-agnostic schema** — `joint_positions` is stored as `pa.list_(pa.float64())` (variable length) so the same parquet schema accommodates the UR5e's 6 joints and the Skydio X2's 7-DOF free-body state (xyz + quaternion) without any schema changes. The `embodiment` column carries the per-row context a downstream model needs to interpret the state vector correctly.

**Genuine lerobot v2.1 output** — export produces a fully `LeRobotDataset`-loadable dataset: correct folder structure (`data/chunk-NNN/`, `videos/chunk-NNN/observation.images.wrist/`, `meta/`), correct column names (`observation.state`, `action`, `timestamp` as float32 seconds from episode start, `frame_index`, `episode_index`, `index`, `task_index`), and all three required meta files (`info.json`, `episodes.json`, `tasks.json`). Embodiment is auto-detected from bag topic names — no flags required.

## Stack

| Layer | Tool |
|---|---|
| Communication | ROS2 Jazzy, DDS (FastRTPS) |
| Physics simulation | MuJoCo 3.10 |
| Robot models | MuJoCo Menagerie (UR5e, Skydio X2) |
| Recording | rosbag2 / MCAP |
| Data export | rosbag2_py, pandas, pyarrow, opencv-python |
| ROS2 Python | rclpy, cv_bridge, message_filters |

## Repository structure

```
robo-data-engine/
├── src/
│   └── data_engine/
│       ├── data_engine/
│       │   ├── ur5e_node.py      # MuJoCo sim → ROS2 publisher (physics + camera)
│       │   ├── sync_node.py      # 3-way time synchronizer
│       │   ├── teleop_node.py    # keyboard teleoperation
│       │   └── hello_node.py     # pipeline smoke test
│       ├── package.xml
│       └── setup.py
├── models/
│   └── ur5e_custom/              # UR5e MJCF with wrist camera, table, cube
├── export/
│   └── to_lerobot.py             # MCAP → parquet + video
├── datasets/                     # gitignored — generated locally
│   ├── raw/                      # MCAP episode bags
│   └── processed/                # parquet + MP4 exports
└── mujoco_menagerie/             # gitignored — cloned separately
```

## Setup

```bash
# 1. Clone
git clone https://github.com/nihalseth0506/robo-data-engine.git
cd robo-data-engine

# 2. Get robot models (not committed — too large)
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git

# 3. Python dependencies (ROS2 Jazzy on Ubuntu 24.04)
pip3 install mujoco "numpy>=1.26,<1.28" "opencv-python==4.9.0.80" \
    pandas pyarrow --break-system-packages

# 4. Build ROS2 package
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash

# 5. Set rendering backend (WSL2 / headless)
export MUJOCO_GL=osmesa
```

## Running a data collection session

```bash
# Terminal 1 — robot simulation
ros2 run data_engine ur5e_node

# Terminal 2 — live sync verification
ros2 run data_engine sync_node

# Terminal 3 — keyboard teleoperation
# Keys: q/a=joint0  w/s=joint1  e/d=joint2  r/f=joint3  t/g=joint4  y/h=joint5
ros2 run data_engine teleop_node

# Terminal 4 — record episode (stop this first with Ctrl+C)
cd datasets/raw
ros2 bag record \
    /ur5e/joint_states \
    /ur5e/camera/image_raw \
    /ur5e/joint_commands \
    --storage mcap \
    -o episode_001
```

## Exporting to lerobot format

```bash
source /opt/ros/jazzy/setup.bash
python3 export/to_lerobot.py \
    datasets/raw/episode_001 \
    --output-dir datasets/processed \
    --episode-id 1 \
    --task "reach toward red cube"
```

Output per episode:
- `datasets/processed/data/chunk-000/episode_000001.parquet` — lerobot v2.1 parquet
- `datasets/processed/videos/chunk-000/observation.images.wrist/episode_000001.mp4` — camera video
- `datasets/processed/meta/info.json` — dataset structure and feature schema
- `datasets/processed/meta/episodes.json` — per-episode metadata
- `datasets/processed/meta/tasks.json` — task descriptions

## Dataset schema

| Column | Type | Description |
|---|---|---|
| `observation.state` | list[float64] | Joint angles (UR5e ×6) or xyz+quaternion (Skydio ×7) |
| `action` | list[float64] | Teleop commands at capture time |
| `timestamp` | float32 | Seconds from episode start |
| `frame_index` | int64 | Frame number within episode |
| `episode_index` | int64 | Episode number |
| `index` | int64 | Global frame index across all episodes |
| `task_index` | int64 | Task identifier (0 for single-task datasets) |
| `embodiment` | string | Robot type — `ur5e` or `skydio_x2` |
| `state_gap_ms` | float64 | Time gap between image and matched state |
| `cmd_gap_ms` | float64 | Time gap between image and matched command |

## Roadmap

- [x] Phase 1: UR5e arm — simulation, wrist camera, keyboard teleop, 3-way sync, MCAP recording, lerobot v2.1 export
- [x] Phase 2: Skydio X2 drone — free-body sim, PID hover controller, velocity teleop, same pipeline proves embodiment-agnostic design

## Author

Nihal Sanjay Seth
MSc Mechatronics & Robotics, Hochschule Schmalkalden, Germany
seth.nihal.work@gmail.com

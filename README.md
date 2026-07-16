# robo-data-engine

A universal robot data collection and synchronization pipeline built on ROS2 Jazzy and MuJoCo 3.10.

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

export/to_lerobot.py →  datasets/processed/episode_NNN.parquet
                     →  datasets/processed/episode_NNN.mp4
```

## Key engineering decisions

**Multi-threaded executor with callback groups** — physics stepping (~470 Hz) and camera rendering (~20 Hz) run on genuinely separate threads, isolated by `MutuallyExclusiveCallbackGroup`. A `threading.Lock` protects shared MuJoCo `data` access, held only during the minimal critical window (scene update only, not the full render call), so neither thread blocks the other unnecessarily.

**Stamped command messages** — `Float64MultiArray` carries no timestamp, making it incompatible with `ApproximateTimeSynchronizer`. A parallel `JointState`-typed command topic on `/ur5e/joint_commands` provides the header field needed for three-way time alignment, while the actuator control path continues to receive a plain `Float64MultiArray` on `/ur5e/joint_ctrl` — decoupling the recording concern from the control concern.

**Embodiment-agnostic schema** — `joint_positions` is stored as `pa.list_(pa.float64())` (variable length) so the same parquet schema accommodates the UR5e's 6 joints and the Skydio X2's 7-DOF free-body state (xyz + quaternion) without any schema changes. The `embodiment` column carries the per-row context a downstream model needs to interpret the state vector correctly.

**lerobot-compatible export** — output format matches the Hugging Face lerobot dataset standard: one parquet file per episode with `(timestamp_ns, embodiment, joint_positions, joint_velocities, joint_commands, state_gap_ms, cmd_gap_ms)` columns, one MP4 per episode for human visual inspection.

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
    --episode-id 1
```

Output per episode:
- `datasets/processed/episode_001.parquet` — time-aligned (state, action) table
- `datasets/processed/episode_001.mp4` — wrist camera video for visual inspection

## Dataset schema

| Column | Type | Description |
|---|---|---|
| `timestamp_ns` | int64 | ROS2 system clock at image capture |
| `episode_id` | int32 | Episode number |
| `embodiment` | string | Robot type (`ur5e`, `skydio_x2`, …) |
| `joint_positions` | list[float64] | Joint angles at capture time |
| `joint_velocities` | list[float64] | Joint velocities at capture time |
| `joint_commands` | list[float64] | Teleop target angles at capture time |
| `state_gap_ms` | float64 | Time gap between image and matched state |
| `cmd_gap_ms` | float64 | Time gap between image and matched command |

## Roadmap

- [x] Phase 1: UR5e arm — simulation, teleoperation, 3-way sync, MCAP recording, lerobot export
- [ ] Phase 2: Skydio X2 drone — second embodiment, prove pipeline is genuinely embodiment-agnostic
- [ ] Phase 3: Cross-embodiment behavior cloning baseline (shared policy across both embodiments)
- [ ] Phase 4: Docker containerization + GitHub Actions CI

## Author

Nihal Sanjay Seth
MSc Mechatronics & Robotics, Hochschule Schmalkalden, Germany
seth.nihal.work@gmail.com
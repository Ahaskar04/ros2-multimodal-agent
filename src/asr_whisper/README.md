# ROS2 ASR Node

Whisper-based speech recognition.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src/asr_whisper/requirements.txt
colcon build --symlink-install
```

## Usage

```bash
source .venv/bin/activate
source install/setup.bash
ros2 run asr_whisper whisper_node
```

Transcriptions are published to `/asr/transcript`.

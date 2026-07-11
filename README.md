# ros2-multimodal-agent

A ROS 2 workspace of LLM-driven agent nodes.

## Packages

- **`turtlesim_agent`** — a natural-language agent that controls one or many
  turtles in turtlesim. A LangChain agent (default model `gpt-5.1`) turns plain
  English into turtle motion, drawing, and multi-turtle coordination.
- **`asr_whisper`** — speech-to-text node.

## Requirements

- ROS 2 (tested on Humble; also runs on Jazzy)
- Python 3.10+
- An LLM API key, e.g. `export OPENAI_API_KEY=...`
- `pip install -r src/turtlesim_agent/requirements.txt`

## Build

```bash
colcon build --symlink-install
source install/setup.bash
```

## Run the turtlesim agent (3 levels)

Each level spawns a different number of turtles and starts the agent.

```bash
ros2 launch turtlesim_agent level1.launch.py   # 1 turtle  -> "Draw a five-pointed star"
ros2 launch turtlesim_agent level2.launch.py   # 2 turtles -> "Turtle1 draw a circle, turtle2 a square"
ros2 launch turtlesim_agent level3.launch.py   # 6 turtles -> "Form a hexagon"
```

Type prompts at the `user:` prompt in the same terminal.

Two-terminal alternative (CLI):

```bash
# Terminal 1
ros2 run turtlesim turtlesim_node
# Terminal 2
ros2 run turtlesim_agent turtlesim_agent_node --ros-args -p num_turtles:=6 -p agent_model:=gpt-5.1
```

## How it works

One LangChain agent reasons over the request and calls primitive tools
(move, rotate, teleport, pen, `list_turtles`) to drive each turtle by name.
Formation geometry (e.g. hexagon vertices) is computed by the model at
runtime, not hardcoded.

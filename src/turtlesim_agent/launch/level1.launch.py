#!/usr/bin/env python3
"""Level 1 demo launch: a single turtle.

Starts turtlesim and the agent configured for 1 turtle. Try prompts like
"draw a five-pointed star".

Args:
  interface   : 'cli' (type in this terminal) or 'gui' (chat window; needs python3-tk).
  agent_model : LLM to use (default gpt-4o).

Note: in CLI mode you type your prompts in the same terminal that ran
`ros2 launch`. If stdin isn't accepted under launch on your setup, use the
two-terminal flow instead (see README/notes).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NUM_TURTLES = 1


def generate_launch_description():
    interface = LaunchConfiguration("interface")
    agent_model = LaunchConfiguration("agent_model")

    return LaunchDescription(
        [
            # Software rendering so the turtlesim window renders on machines
            # without a working GL driver.
            SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"),
            DeclareLaunchArgument("interface", default_value="cli"),
            DeclareLaunchArgument("agent_model", default_value="gpt-5.1"),
            Node(
                package="turtlesim",
                executable="turtlesim_node",
                name="turtlesim",
                output="screen",
            ),
            Node(
                package="turtlesim_agent",
                executable="turtlesim_agent_node",
                name="turtlesim_agent",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "interface": interface,
                        "agent_model": agent_model,
                        "num_turtles": NUM_TURTLES,
                    }
                ],
            ),
        ]
    )

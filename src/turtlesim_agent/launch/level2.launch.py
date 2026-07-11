#!/usr/bin/env python3
"""Level 2 demo launch: two turtles.

Starts turtlesim and the agent configured for 2 turtles (turtle1, turtle2).
Try prompts like:
  "Turtle1 draw a circle on the left. Turtle2 draw a square on the right."
  "Move turtle2 to the top."

Args:
  interface   : 'cli' (type in this terminal) or 'gui' (chat window; needs python3-tk).
  agent_model : LLM to use (default gpt-4o).
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

NUM_TURTLES = 2


def generate_launch_description():
    interface = LaunchConfiguration("interface")
    agent_model = LaunchConfiguration("agent_model")

    return LaunchDescription(
        [
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

#!/usr/bin/env python3

"""
Status Tools for TurtleSim

Tools for retrieving status information about the turtles: an individual
turtle's pose and pen status, a roster of all turtles with their positions,
and boundary checking.
"""

from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

TURTLE_NAME_FIELD = Field(
    default="",
    description="Name of the turtle to query (e.g. 'turtle1', 'turtle2'). "
    "Leave empty to target the default turtle.",
)


def make_list_turtles_tool(node):
    """Tool that lists every turtle and its current pose.

    This is the agent's situational-awareness tool for multi-turtle tasks
    (formations, 'nearest turtle', spreading out, etc.).
    """

    class ListTurtlesInput(BaseModel):
        pass

    def inner() -> str:
        poses = node.get_all_poses()
        if not poses:
            return "No turtle poses available yet."
        lines = [f"There are {len(poses)} turtle(s):"]
        for name, (x, y, theta) in poses.items():
            lines.append(f"- {name}: x={x:.2f}, y={y:.2f}, theta={theta:.2f} rad")
        return "\n".join(lines)

    return StructuredTool.from_function(
        func=inner,
        name="list_turtles",
        description="""
        Lists all turtles currently in the simulation together with their current
        position (x, y) and orientation (theta). Use this first for any task that
        involves more than one turtle (formations, nearest-turtle, spreading out).

        Returns:
        - The number of turtles and each turtle's name, position, and orientation.
        """,
        args_schema=ListTurtlesInput,
    )


def make_get_turtle_pose_tool(node):
    class GetPoseInput(BaseModel):
        turtle_name: str = TURTLE_NAME_FIELD

    def inner(turtle_name: str = "") -> str:
        name = turtle_name or node.default_turtle
        if not node.wait_for_pose(turtle_name, timeout=0.5):
            return f"Could not retrieve pose for {name}."
        p = node.get_pose(turtle_name)
        return f"Current Pose of {name}: x={p.x:.2f}, y={p.y:.2f}, yaw={p.theta:.2f} rad"

    return StructuredTool.from_function(
        func=inner,
        name="get_turtle_pose",
        description="""
        Returns the current pose of a single turtle.

        Args:
        - turtle_name (str): Which turtle to query. Empty = default turtle.

        Returns:
        - Position x, y (centimeters) and orientation yaw (radians).
        """,
        args_schema=GetPoseInput,
    )


def make_get_pen_status_tool(node):
    class GetPenInput(BaseModel):
        turtle_name: str = TURTLE_NAME_FIELD

    def inner(turtle_name: str = "") -> str:
        name = turtle_name or node.default_turtle
        pen_info = node.get_pen_info(turtle_name)
        if pen_info is None:
            return f"No such turtle: {name}"
        status = "up (not drawing)" if pen_info["off"] else "down (drawing)"
        return (
            f"Pen status for {name}:\n"
            f"- State: {status}\n"
            f"- Color: RGB({pen_info['r']}, {pen_info['g']}, {pen_info['b']})\n"
            f"- Width: {pen_info['width']}\n"
            f"- Drawing: {'No' if pen_info['off'] else 'Yes'}"
        )

    return StructuredTool.from_function(
        func=inner,
        name="get_pen_status",
        description="""
        Returns a turtle's pen status: color, width, and drawing state.

        Args:
        - turtle_name (str): Which turtle to query. Empty = default turtle.
        """,
        args_schema=GetPenInput,
    )


def make_check_bounds_tool(node):
    class CheckBoundsInput(BaseModel):
        turtle_name: str = TURTLE_NAME_FIELD
        x_min: float = Field(default=0.0, description="Minimum x coordinate")
        x_max: float = Field(default=11.0, description="Maximum x coordinate")
        y_min: float = Field(default=0.0, description="Minimum y coordinate")
        y_max: float = Field(default=11.0, description="Maximum y coordinate")

    def inner(
        turtle_name: str = "",
        x_min: float = 0.0,
        x_max: float = 11.0,
        y_min: float = 0.0,
        y_max: float = 11.0,
    ) -> str:
        name = turtle_name or node.default_turtle
        if not node.wait_for_pose(turtle_name, timeout=0.5):
            return f"Could not retrieve pose to check bounds for {name}."

        is_within = node.is_within_bounds(turtle_name, x_min, x_max, y_min, y_max)
        p = node.get_pose(turtle_name)
        verdict = "WITHIN" if is_within else "OUT OF"
        return (
            f"{name} is {verdict} bounds.\n"
            f"Current position: x={p.x:.2f}, y={p.y:.2f}\n"
            f"Allowed bounds: {x_min} < x < {x_max}, {y_min} < y < {y_max}"
        )

    return StructuredTool.from_function(
        func=inner,
        name="check_turtle_bounds",
        description="""
        Checks whether a turtle is within the specified boundaries.

        Args:
        - turtle_name (str): Which turtle to check. Empty = default turtle.
        - x_min, x_max, y_min, y_max (float): Boundary limits (defaults 0..11).
        """,
        args_schema=CheckBoundsInput,
    )

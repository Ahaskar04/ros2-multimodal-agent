#!/usr/bin/env python3

"""
Motion Tools for TurtleSim

This module contains tools for controlling turtle movement including
linear movement, rotation, and curved trajectories. Every tool takes an
optional `turtle_name` so the agent can drive a specific turtle by name;
leaving it empty targets the default turtle (turtle1).
"""


from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

from turtlesim_agent.tools.math_tools import (
    compute_duration_from_circular_angle_and_angular_velocity,
    compute_linear_and_angular_velocity_from_radius,
)

TURTLE_NAME_FIELD = Field(
    default="",
    description="Name of the turtle to control (e.g. 'turtle1', 'turtle2'). "
    "Leave empty to target the default turtle.",
)


def _pose_str(node, turtle_name):
    name = turtle_name or node.default_turtle
    p = node.get_pose(turtle_name)
    if p is None:
        return f"Pose data for {name} not received yet."
    return f"Current Pose of {name}: x={p.x:.2f}, y={p.y:.2f}, yaw={p.theta:.2f} rad"


def make_move_linear_tool(node):
    class MoveInput(BaseModel):
        distance: float = Field(
            description="Linear distance in centimeters. Positive for forward, negative for backward."
        )
        turtle_name: str = TURTLE_NAME_FIELD

    async def inner(distance: float, turtle_name: str = "") -> str:
        if distance != 0.0:
            result = await node.move_linear(turtle_name, distance)
            if not result:
                name = turtle_name or node.default_turtle
                return f"{name} hit a boundary during movement and was stopped."
        return _pose_str(node, turtle_name)

    return StructuredTool.from_function(
        func=inner,
        coroutine=inner,
        name="move_linear",
        description="""
        Moves a turtle forward or backward by the specified distance.

        Args:
        - distance (float): Linear distance in centimeters. Positive forward, negative backward.
        - turtle_name (str): Which turtle to move. Empty = default turtle.
        """,
        args_schema=MoveInput,
    )


def make_rotate_tool(node):
    class RotateInput(BaseModel):
        angle: float = Field(
            description="Relative rotation in radians. Positive turns left (CCW), negative turns right (CW)."
        )
        turtle_name: str = TURTLE_NAME_FIELD

    def inner(angle: float, turtle_name: str = "") -> str:
        if angle != 0.0:
            node.rotate_angle(turtle_name, angle)
        return _pose_str(node, turtle_name)

    return StructuredTool.from_function(
        func=inner,
        name="rotate_robot",
        description="""
        Rotates a turtle by the specified angle (in radians).

        Args:
        - angle (float): Relative rotation in radians. Positive left (CCW), negative right (CW).
        - turtle_name (str): Which turtle to rotate. Empty = default turtle.
        """,
        args_schema=RotateInput,
    )


def make_move_non_linear_tool(node):
    class MoveCurveInput(BaseModel):
        linear_velocity: float = Field(
            description="Forward/backward speed in cm/s. Positive forward."
        )
        angular_velocity: float = Field(
            description="Rotational speed in rad/s. Positive turns left (CCW)."
        )
        duration_sec: float = Field(description="Duration of movement in seconds.")
        turtle_name: str = TURTLE_NAME_FIELD

    async def inner(
        linear_velocity: float,
        angular_velocity: float,
        duration_sec: float,
        turtle_name: str = "",
    ) -> str:
        result = await node.move_non_linear(
            turtle_name,
            duration_sec,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )
        if not result:
            name = turtle_name or node.default_turtle
            return f"{name} hit a boundary during curved movement and was stopped."
        return _pose_str(node, turtle_name)

    return StructuredTool.from_function(
        func=inner,
        coroutine=inner,
        name="move_non_linear",
        description="""
        Moves a turtle in a curved trajectory (arcs/spirals).

        Args:
        - linear_velocity (cm/s): Forward/backward speed. Positive forward.
        - angular_velocity (rad/s): Rotational speed. Positive left (CCW).
        - duration_sec (s): How long the motion lasts.
        - turtle_name (str): Which turtle to move. Empty = default turtle.
        """,
        args_schema=MoveCurveInput,
    )


def make_teleport_absolute_tool(node):
    class TeleportAbsInput(BaseModel):
        x: float = Field(description="Target X position (0.0-11.0)")
        y: float = Field(description="Target Y position (0.0-11.0)")
        yaw: float = Field(description="Target orientation in radians")
        turtle_name: str = TURTLE_NAME_FIELD

    async def inner(x: float, y: float, yaw: float, turtle_name: str = "") -> str:
        r, g, b, width, off = node.get_pen_state(turtle_name)
        # Lift the pen so the jump leaves no trail, then restore prior pen state.
        await node.call_set_pen_async(turtle_name, r, g, b, width, True)
        await node.call_teleport_absolute_async(turtle_name, x, y, yaw)
        await node.call_set_pen_async(turtle_name, r, g, b, width, off)
        name = turtle_name or node.default_turtle
        return f"{name} teleported to ({x}, {y}) with orientation {yaw} rad."

    return StructuredTool.from_function(
        func=inner,
        coroutine=inner,
        name="teleport_absolute",
        description="""
        Teleports a turtle to an absolute position/orientation instantly without
        drawing a trail. The pen is auto-lifted before and restored after, so you
        don't need set_pen_up_down for the jump itself.
        Example: {'x': 5.0, 'y': 5.0, 'yaw': 1.57, 'turtle_name': 'turtle2'}.

        Args:
        - x (float): Target X position (0.0-11.0).
        - y (float): Target Y position (0.0-11.0).
        - yaw (float): Target yaw in radians.
        - turtle_name (str): Which turtle to teleport. Empty = default turtle.
        """,
        args_schema=TeleportAbsInput,
    )


def make_teleport_relative_tool(node):
    class TeleportRelInput(BaseModel):
        linear: float = Field(description="Linear distance to teleport forward")
        angular: float = Field(description="Angular rotation in radians")
        turtle_name: str = TURTLE_NAME_FIELD

    async def inner(linear: float, angular: float, turtle_name: str = "") -> str:
        r, g, b, width, off = node.get_pen_state(turtle_name)
        await node.call_set_pen_async(turtle_name, r, g, b, width, True)
        await node.call_teleport_relative_async(turtle_name, linear, angular)
        await node.call_set_pen_async(turtle_name, r, g, b, width, off)
        name = turtle_name or node.default_turtle
        return f"{name} teleported {linear} units forward and rotated {angular} rad."

    return StructuredTool.from_function(
        func=inner,
        coroutine=inner,
        name="teleport_relative",
        description="""
        Teleports a turtle relative to its current pose instantly without a trail.

        Args:
        - linear (float): Linear distance to teleport forward from current position.
        - angular (float): Angular rotation in radians from current orientation.
        - turtle_name (str): Which turtle to teleport. Empty = default turtle.
        """,
        args_schema=TeleportRelInput,
    )


def make_move_on_arc_tool(node):
    class ArcMoveInput(BaseModel):
        radius: float = Field(description="Radius of the circular path in centimeters")
        circular_angle: float = Field(
            description="Angle to rotate along the arc in radians"
        )
        turtle_name: str = TURTLE_NAME_FIELD

    async def inner(radius: float, circular_angle: float, turtle_name: str = "") -> str:
        linear_velocity, angular_velocity = (
            compute_linear_and_angular_velocity_from_radius.invoke({"radius": radius})
        )
        duration = compute_duration_from_circular_angle_and_angular_velocity.invoke(
            {"circular_angle": circular_angle, "angular_velocity": angular_velocity}
        )
        if circular_angle < 0:
            angular_velocity = -angular_velocity

        result = await node.move_non_linear(
            turtle_name,
            duration,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )
        if not result:
            name = turtle_name or node.default_turtle
            return f"{name} hit a boundary during arc movement and was stopped."
        return _pose_str(node, turtle_name)

    return StructuredTool.from_function(
        func=inner,
        coroutine=inner,
        name="move_on_arc",
        description="""
        Moves a turtle along a circular arc using the specified radius and angle.
        Example: to draw a semi-circle from left to right, face north and set
        circular_angle to -3.14; from right to left, set it to +3.14.

        Args:
        - radius (float): radius of the circular path in centimeters.
        - circular_angle (float): angle to rotate along the arc in radians.
        - turtle_name (str): Which turtle to move. Empty = default turtle.
        """,
        args_schema=ArcMoveInput,
    )

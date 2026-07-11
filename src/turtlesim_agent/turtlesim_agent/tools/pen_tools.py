#!/usr/bin/env python3

"""
Pen Tools for TurtleSim

This module contains tools for controlling a turtle's drawing pen,
including color, width, and up/down state. Each tool takes an optional
`turtle_name`; leaving it empty targets the default turtle (turtle1).
"""

from langchain.tools import StructuredTool
from pydantic.v1 import BaseModel, Field

TURTLE_NAME_FIELD = Field(
    default="",
    description="Name of the turtle whose pen to set (e.g. 'turtle1', 'turtle2'). "
    "Leave empty to target the default turtle.",
)


def make_set_pen_color_width_tool(node):
    class SetPenColorWidthInput(BaseModel):
        r: int = Field(description="Red color component (0-255)")
        g: int = Field(description="Green color component (0-255)")
        b: int = Field(description="Blue color component (0-255)")
        width: int = Field(description="Pen width (1-10)")
        turtle_name: str = TURTLE_NAME_FIELD

    async def inner(r: int, g: int, b: int, width: int, turtle_name: str = "") -> str:
        _, _, _, _, current_off = node.get_pen_state(turtle_name)
        await node.call_set_pen_async(turtle_name, r, g, b, width, current_off)
        name = turtle_name or node.default_turtle
        pen_state = "up (not drawing)" if current_off else "down (drawing)"
        return (
            f"{name} pen color set to RGB({r}, {g}, {b}) with width {width}. "
            f"Pen is {pen_state}."
        )

    return StructuredTool.from_function(
        func=inner,
        coroutine=inner,
        name="set_pen_color_width",
        description="""
        Sets a turtle's pen color and width while keeping its current up/down state.

        Args:
        - r (int): Red (0-255).
        - g (int): Green (0-255).
        - b (int): Blue (0-255).
        - width (int): Pen width (1-10).
        - turtle_name (str): Which turtle's pen. Empty = default turtle.
        """,
        args_schema=SetPenColorWidthInput,
    )


def make_set_pen_up_down_tool(node):
    class SetPenUpDownInput(BaseModel):
        pen_up: bool = Field(
            description="Set pen up (True) to stop drawing, or down (False) to start drawing"
        )
        turtle_name: str = TURTLE_NAME_FIELD

    async def inner(pen_up: bool, turtle_name: str = "") -> str:
        r, g, b, width, _ = node.get_pen_state(turtle_name)
        await node.call_set_pen_async(turtle_name, r, g, b, width, pen_up)
        name = turtle_name or node.default_turtle
        if pen_up:
            return (
                f"{name} pen lifted up - it will not draw while moving. "
                f"Color: RGB({r}, {g}, {b}), Width: {width}"
            )
        return (
            f"{name} pen put down - it will draw while moving. "
            f"Color: RGB({r}, {g}, {b}), Width: {width}"
        )

    return StructuredTool.from_function(
        func=inner,
        coroutine=inner,
        name="set_pen_up_down",
        description="""
        Sets a turtle's pen up (stop drawing) or down (start drawing) while keeping
        its current color and width.

        Args:
        - pen_up (bool): True lifts the pen (stop drawing), False lowers it (start drawing).
        - turtle_name (str): Which turtle's pen. Empty = default turtle.
        """,
        args_schema=SetPenUpDownInput,
    )

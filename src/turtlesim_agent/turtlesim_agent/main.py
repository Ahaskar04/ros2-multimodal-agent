#!/usr/bin/env python3
"""
Main entry point for the TurtleSim agent with LangChain integration.
This module initializes the ROS2 node and connects it with a language-based agent
that can interpret natural language commands to control one or more turtles.
"""

import asyncio
import threading

import rclpy
from langchain.agents import AgentExecutor
from langchain_core.callbacks import BaseCallbackHandler
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor

from turtlesim_agent.chat_entrypoint import chat_invoke
from turtlesim_agent.llms import create_agent
from turtlesim_agent.prompts import build_prompt
from turtlesim_agent.tools.all_tools import make_all_tools
from turtlesim_agent.turtlesim_node import TurtleSimAgent


class ReasoningCallback(BaseCallbackHandler):
    """Prints the agent's reasoning and tool calls so the LLM's thinking is
    visible during the demo."""

    def on_llm_end(self, response, **kwargs):
        for generation in response.generations:
            for gen in generation:
                text = getattr(gen, "text", "") or ""
                if text.strip():
                    print(f"\n🧠 {text.strip()}")

    def on_agent_action(self, action, **kwargs):
        print(f"🔧 {action.tool}({action.tool_input})")

    def on_agent_finish(self, finish, **kwargs):
        output = finish.return_values.get("output", "")
        if output:
            print(f"\n✅ {output}")


def spin_node_thread(node):
    """Run the ROS2 node in a separate thread to allow concurrent execution."""
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()


def main():
    """Initialize and run the TurtleSim agent with LangChain integration."""
    rclpy.init()
    node = TurtleSimAgent()

    spin_thread = threading.Thread(target=spin_node_thread, args=(node,), daemon=True)
    spin_thread.start()

    if not node.wait_for_pose(timeout=5.0):
        node.get_logger().error("Initial pose not received")
        node.destroy_node()
        rclpy.shutdown()
        return

    # Spawn the extra turtles (if any) before building the prompt, so the agent's
    # system prompt knows exactly how many turtles exist and their names.
    asyncio.run(node.setup_additional_turtles())
    node.get_logger().info(f"Active turtles: {', '.join(node.turtle_names)}")

    tools = make_all_tools(node)
    prompt = build_prompt(node.turtle_names)
    agent = create_agent(
        model_name=node.agent_model, tools=tools, temperature=0.0, prompt=prompt
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=50,
        handle_parsing_errors=True,
        callbacks=[ReasoningCallback()],
    )
    chat_invoke(
        interface=node.interface,
        agent_executor=agent_executor,
        node=node,
        spin_thread=spin_thread,
    )


if __name__ == "__main__":
    main()

import math
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_srvs.srv import Empty
from turtlesim.msg import Pose
from turtlesim.srv import Kill, SetPen, Spawn, TeleportAbsolute, TeleportRelative

from turtlesim_agent.utils import normalize_angle

TWIST_ANGULAR = 0.8
TWIST_VELOCITY = 1.0
ROTATION_ERROR_THRESHOLD = 0.034  # radians
DISTANCE_ERROR_THRESHOLD = 0.1  # centimeters
PUBLISH_RATE = 0.05


class TurtleHandle:
    """Holds all per-turtle state: pose, publisher, pen state, service clients."""

    def __init__(self, name):
        self.name = name
        self.pub = None
        self.current_pose = None
        self.pose_ready = threading.Event()

        # Pen state (defaults match turtlesim's white pen, width 3, drawing on)
        self.pen_r = 255
        self.pen_g = 255
        self.pen_b = 255
        self.pen_width = 3
        self.pen_off = False

        # Per-turtle service clients
        self.set_pen_client = None
        self.teleport_abs_client = None
        self.teleport_rel_client = None


class TurtleSimAgent(Node):
    def __init__(self):
        super().__init__("turtlesim_agent")
        self.declare_parameter("interface", "cli")
        self.declare_parameter("agent_model", "gpt-5.1")
        self.declare_parameter("num_turtles", 1)

        self.interface = (
            self.get_parameter("interface").get_parameter_value().string_value
        )
        self.agent_model = (
            self.get_parameter("agent_model").get_parameter_value().string_value
        )
        self.num_turtles = (
            self.get_parameter("num_turtles").get_parameter_value().integer_value
        )
        if self.num_turtles < 1:
            self.num_turtles = 1

        # Per-turtle registry, keyed by name (insertion order preserved).
        self.turtles = {}

        # Global (simulation-wide) service clients
        self.reset_client = self.create_client(Empty, "/reset")
        self.clear_client = self.create_client(Empty, "/clear")
        self.kill_client = self.create_client(Kill, "/kill")
        self.spawn_client = self.create_client(Spawn, "/spawn")

        # turtle1 always exists in turtlesim by default. Register handles for all
        # configured turtles up front (before the executor spins) so creating
        # publishers/subscriptions/clients is not raced against spinning. The
        # extra turtles are actually spawned later via setup_additional_turtles().
        self._register_turtle("turtle1")
        for i in range(2, self.num_turtles + 1):
            self._register_turtle(f"turtle{i}")

    # ===== Turtle registry helpers =====
    def _register_turtle(self, name):
        """Create the publisher, subscription, and service clients for a turtle."""
        handle = TurtleHandle(name)
        handle.pub = self.create_publisher(Twist, f"/{name}/cmd_vel", 10)
        self.create_subscription(Pose, f"/{name}/pose", self._make_pose_cb(name), 10)
        handle.set_pen_client = self.create_client(SetPen, f"/{name}/set_pen")
        handle.teleport_abs_client = self.create_client(
            TeleportAbsolute, f"/{name}/teleport_absolute"
        )
        handle.teleport_rel_client = self.create_client(
            TeleportRelative, f"/{name}/teleport_relative"
        )
        self.turtles[name] = handle
        return handle

    def _make_pose_cb(self, name):
        def cb(msg):
            handle = self.turtles.get(name)
            if handle is not None:
                handle.current_pose = msg
                handle.pose_ready.set()

        return cb

    @property
    def default_turtle(self):
        """Name of the first registered turtle (turtle1)."""
        return next(iter(self.turtles))

    @property
    def turtle_names(self):
        return list(self.turtles.keys())

    def _resolve(self, name):
        """Return the handle for `name`, or the default turtle if name is empty."""
        if not name:
            return self.turtles[self.default_turtle]
        return self.turtles.get(name)

    # ===== Pose / pen accessors =====
    def wait_for_pose(self, name=None, timeout=1.0):
        handle = self._resolve(name)
        if handle is None:
            return False
        return handle.pose_ready.wait(timeout=timeout)

    def get_pose(self, name=None):
        handle = self._resolve(name)
        return handle.current_pose if handle else None

    def get_pen_state(self, name=None):
        """Return (r, g, b, width, off) for the given turtle."""
        h = self._resolve(name)
        if h is None:
            return (255, 255, 255, 3, False)
        return (h.pen_r, h.pen_g, h.pen_b, h.pen_width, h.pen_off)

    def get_pen_info(self, name=None):
        h = self._resolve(name)
        if h is None:
            return None
        return {
            "r": h.pen_r,
            "g": h.pen_g,
            "b": h.pen_b,
            "width": h.pen_width,
            "off": h.pen_off,
            "color_rgb": (h.pen_r, h.pen_g, h.pen_b),
            "is_drawing": not h.pen_off,
        }

    def get_all_poses(self):
        """Return {name: (x, y, theta)} for every turtle with a known pose."""
        result = {}
        for name, h in self.turtles.items():
            if h.current_pose is not None:
                p = h.current_pose
                result[name] = (p.x, p.y, p.theta)
        return result

    def _stop_robot(self, handle):
        """Send stop commands to bring a turtle to a complete halt."""
        stop_cmd = Twist()
        for _ in range(5):
            handle.pub.publish(stop_cmd)
            time.sleep(PUBLISH_RATE)

    # ===== Boundary check =====
    def is_within_bounds(self, name=None, x_min=0, x_max=11, y_min=0, y_max=11):
        h = self._resolve(name)
        if h is None or h.current_pose is None:
            return False
        x, y = h.current_pose.x, h.current_pose.y
        return (x_min < x < x_max) and (y_min < y < y_max)

    # ===== Movement methods =====
    def rotate_angle(self, name, angle_rad):
        h = self._resolve(name)
        if h is None:
            return False
        if not h.pose_ready.wait(timeout=1.0):
            self.get_logger().warning(f"No pose for {name}, cannot rotate")
            return False

        start_yaw = h.current_pose.theta
        target_yaw = normalize_angle(start_yaw + angle_rad)
        angular_velocity = TWIST_ANGULAR if angle_rad > 0 else -TWIST_ANGULAR

        while rclpy.ok():
            diff = normalize_angle(target_yaw - h.current_pose.theta)
            if abs(diff) < ROTATION_ERROR_THRESHOLD:
                break
            twist = Twist()
            twist.angular.z = angular_velocity
            h.pub.publish(twist)
            time.sleep(PUBLISH_RATE)

        self._stop_robot(h)
        return True

    async def move_linear(self, name, distance_m):
        h = self._resolve(name)
        if h is None:
            return False
        if not h.pose_ready.wait(timeout=1.0):
            self.get_logger().warning(f"No pose for {name}, cannot move")
            return False

        start_x, start_y = h.current_pose.x, h.current_pose.y
        linear_velocity = TWIST_VELOCITY if distance_m > 0 else -TWIST_VELOCITY

        while rclpy.ok():
            # In multi-turtle mode we never reset the whole sim (that would wipe
            # other turtles); we just stop this turtle at the boundary.
            if not self.is_within_bounds(name):
                self.get_logger().warning(f"{name} hit a boundary; stopping")
                self._stop_robot(h)
                return False

            dx = h.current_pose.x - start_x
            dy = h.current_pose.y - start_y
            moved = math.sqrt(dx**2 + dy**2)
            if abs(distance_m) - moved < DISTANCE_ERROR_THRESHOLD:
                break

            twist = Twist()
            twist.linear.x = linear_velocity
            h.pub.publish(twist)
            time.sleep(PUBLISH_RATE)

        self._stop_robot(h)
        return True

    async def move_non_linear(
        self,
        name,
        duration_sec,
        linear_velocity=TWIST_VELOCITY,
        angular_velocity=TWIST_ANGULAR,
    ):
        h = self._resolve(name)
        if h is None:
            return False
        if not h.pose_ready.wait(timeout=1.0):
            self.get_logger().warning(f"No pose for {name}, cannot move")
            return False

        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time < duration_sec):
            if not self.is_within_bounds(name):
                self.get_logger().warning(f"{name} hit a boundary; stopping")
                self._stop_robot(h)
                return False
            twist = Twist()
            twist.linear.x = linear_velocity
            twist.angular.z = angular_velocity
            h.pub.publish(twist)
            time.sleep(PUBLISH_RATE)

        self._stop_robot(h)
        return True

    # ===== Service client helpers =====
    def wait_for_service(self, client):
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting...")

    async def call_reset_async(self):
        """Reset the simulation. NOTE: turtlesim's reset removes all spawned
        turtles and leaves only turtle1; prefer clear() in multi-turtle mode."""
        self.wait_for_service(self.reset_client)
        future = self.reset_client.call_async(Empty.Request())
        await future
        if future.result() is not None:
            self.get_logger().info("Reset the simulation")
            t1 = self.turtles.get("turtle1")
            if t1 is not None:
                t1.pen_r, t1.pen_g, t1.pen_b = 0, 0, 255
                t1.pen_width, t1.pen_off = 3, False
        else:
            self.get_logger().error("Failed to reset simulation")

    async def call_clear_async(self):
        self.wait_for_service(self.clear_client)
        future = self.clear_client.call_async(Empty.Request())
        await future
        if future.result() is not None:
            self.get_logger().info("Cleared the screen")
        else:
            self.get_logger().error("Failed to clear screen")

    async def call_kill_async(self, name):
        self.wait_for_service(self.kill_client)
        request = Kill.Request()
        request.name = name
        future = self.kill_client.call_async(request)
        await future
        if future.result() is not None:
            self.turtles.pop(name, None)
            self.get_logger().info(f"Killed turtle: {name}")
        else:
            self.get_logger().error(f"Failed to kill turtle: {name}")

    async def call_spawn_async(self, x, y, theta, name=""):
        self.wait_for_service(self.spawn_client)
        request = Spawn.Request()
        request.x = float(x)
        request.y = float(y)
        request.theta = float(theta)
        request.name = name
        future = self.spawn_client.call_async(request)
        await future
        if future.result() is not None:
            spawned = future.result().name
            self.get_logger().info(f"Spawned turtle: {spawned}")
            return spawned
        else:
            self.get_logger().error("Failed to spawn turtle")
            return None

    async def call_set_pen_async(self, name, r, g, b, width, off):
        h = self._resolve(name)
        if h is None:
            return
        self.wait_for_service(h.set_pen_client)
        request = SetPen.Request()
        request.r = int(r)
        request.g = int(g)
        request.b = int(b)
        request.width = int(width)
        request.off = bool(off)
        future = h.set_pen_client.call_async(request)
        await future
        if future.result() is not None:
            h.pen_r, h.pen_g, h.pen_b = int(r), int(g), int(b)
            h.pen_width, h.pen_off = int(width), bool(off)
        else:
            self.get_logger().error(f"Failed to set pen for {name}")

    async def call_teleport_absolute_async(self, name, x, y, theta):
        h = self._resolve(name)
        if h is None:
            return
        self.wait_for_service(h.teleport_abs_client)
        request = TeleportAbsolute.Request()
        request.x = float(x)
        request.y = float(y)
        request.theta = float(theta)
        future = h.teleport_abs_client.call_async(request)
        await future

    async def call_teleport_relative_async(self, name, linear, angular):
        h = self._resolve(name)
        if h is None:
            return
        self.wait_for_service(h.teleport_rel_client)
        request = TeleportRelative.Request()
        request.linear = float(linear)
        request.angular = float(angular)
        future = h.teleport_rel_client.call_async(request)
        await future

    # ===== Multi-turtle startup =====
    def _initial_positions(self, n):
        """Compute spawn positions for turtles. turtle1 keeps its default center
        position; the rest are spread across the canvas so they don't overlap."""
        positions = {1: (5.544, 5.544, 0.0)}
        if n > 1:
            margin = 1.5
            span = 11.0 - 2 * margin
            for i in range(2, n + 1):
                frac = (i - 1) / n
                x = margin + span * frac
                y = 2.0
                positions[i] = (x, y, 0.0)
        return positions

    async def setup_additional_turtles(self):
        """Spawn turtles 2..N (turtle1 already exists) and wait for their poses."""
        if self.num_turtles <= 1:
            return
        positions = self._initial_positions(self.num_turtles)
        for i in range(2, self.num_turtles + 1):
            name = f"turtle{i}"
            x, y, theta = positions[i]
            await self.call_spawn_async(x, y, theta, name)

        # Wait until every turtle has reported a pose.
        for name, h in self.turtles.items():
            if not h.pose_ready.wait(timeout=5.0):
                self.get_logger().warning(f"No pose received for {name}")

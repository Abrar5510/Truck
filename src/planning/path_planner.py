"""
Path Planning Module

Implements hierarchical path planning including:
- Global planning (A* on road network)
- Behavioral planning (FSM-based)
- Local motion planning (Frenet frame optimization)

Author: Self-Driving Truck Project
Date: 2025-11-17
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import heapq


class BehaviorState(Enum):
    """Behavioral states for decision making"""
    LANE_KEEP = "lane_keep"
    LANE_CHANGE_LEFT = "lane_change_left"
    LANE_CHANGE_RIGHT = "lane_change_right"
    FOLLOW_VEHICLE = "follow_vehicle"
    MERGE = "merge"
    STOP = "stop"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class Waypoint:
    """Single waypoint in path"""
    x: float
    y: float
    theta: float  # Heading (radians)
    v: float  # Target velocity (m/s)
    kappa: float = 0.0  # Curvature (1/m)


@dataclass
class Trajectory:
    """Complete trajectory"""
    waypoints: List[Waypoint]
    cost: float
    behavior: BehaviorState


class FrenetFrame:
    """Frenet frame coordinate system for path planning"""

    def __init__(self, reference_path: List[Tuple[float, float]]):
        self.reference_path = np.array(reference_path)
        self.s_values = self._compute_arc_lengths()

    def _compute_arc_lengths(self) -> np.ndarray:
        """Compute arc length along reference path"""
        s = np.zeros(len(self.reference_path))

        for i in range(1, len(self.reference_path)):
            dx = self.reference_path[i, 0] - self.reference_path[i-1, 0]
            dy = self.reference_path[i, 1] - self.reference_path[i-1, 1]
            s[i] = s[i-1] + math.sqrt(dx**2 + dy**2)

        return s

    def cartesian_to_frenet(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert Cartesian (x, y) to Frenet (s, d)

        Args:
            x, y: Cartesian coordinates

        Returns:
            (s, d): Longitudinal and lateral Frenet coordinates
        """
        # Find closest point on reference path
        distances = np.sqrt(
            (self.reference_path[:, 0] - x)**2 +
            (self.reference_path[:, 1] - y)**2
        )
        closest_idx = np.argmin(distances)

        # Get s coordinate
        s = self.s_values[closest_idx]

        # Compute d (lateral offset)
        if closest_idx < len(self.reference_path) - 1:
            # Vector along path
            dx = self.reference_path[closest_idx + 1, 0] - self.reference_path[closest_idx, 0]
            dy = self.reference_path[closest_idx + 1, 1] - self.reference_path[closest_idx, 1]

            # Normalize
            length = math.sqrt(dx**2 + dy**2)
            if length > 0:
                dx /= length
                dy /= length

            # Vector from path to point
            vx = x - self.reference_path[closest_idx, 0]
            vy = y - self.reference_path[closest_idx, 1]

            # Lateral offset (cross product)
            d = vx * (-dy) + vy * dx
        else:
            d = distances[closest_idx]

        return s, d

    def frenet_to_cartesian(self, s: float, d: float) -> Tuple[float, float, float]:
        """
        Convert Frenet (s, d) to Cartesian (x, y, theta)

        Args:
            s: Longitudinal coordinate
            d: Lateral offset

        Returns:
            (x, y, theta): Cartesian position and heading
        """
        # Find closest s value
        idx = np.searchsorted(self.s_values, s)
        idx = min(idx, len(self.reference_path) - 2)

        # Interpolate position on reference path
        s_frac = (s - self.s_values[idx]) / (self.s_values[idx + 1] - self.s_values[idx] + 1e-6)
        s_frac = np.clip(s_frac, 0, 1)

        x_ref = (1 - s_frac) * self.reference_path[idx, 0] + s_frac * self.reference_path[idx + 1, 0]
        y_ref = (1 - s_frac) * self.reference_path[idx, 1] + s_frac * self.reference_path[idx + 1, 1]

        # Compute heading of reference path
        dx = self.reference_path[idx + 1, 0] - self.reference_path[idx, 0]
        dy = self.reference_path[idx + 1, 1] - self.reference_path[idx, 1]
        theta_ref = math.atan2(dy, dx)

        # Add lateral offset
        x = x_ref - d * math.sin(theta_ref)
        y = y_ref + d * math.cos(theta_ref)

        return x, y, theta_ref


class QuinticPolynomial:
    """Quintic polynomial for trajectory generation"""

    def __init__(self, x0: float, v0: float, a0: float,
                 x1: float, v1: float, a1: float, T: float):
        """
        Generate quintic polynomial from boundary conditions

        Args:
            x0, v0, a0: Initial position, velocity, acceleration
            x1, v1, a1: Final position, velocity, acceleration
            T: Time duration
        """
        self.a0 = x0
        self.a1 = v0
        self.a2 = a0 / 2.0

        # Solve for remaining coefficients
        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ])
        b = np.array([
            x1 - self.a0 - self.a1*T - self.a2*T**2,
            v1 - self.a1 - 2*self.a2*T,
            a1 - 2*self.a2
        ])

        coeffs = np.linalg.solve(A, b)
        self.a3, self.a4, self.a5 = coeffs

    def calc_point(self, t: float) -> float:
        """Calculate position at time t"""
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + \
               self.a4*t**4 + self.a5*t**5

    def calc_first_derivative(self, t: float) -> float:
        """Calculate velocity at time t"""
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2 + \
               4*self.a4*t**3 + 5*self.a5*t**4

    def calc_second_derivative(self, t: float) -> float:
        """Calculate acceleration at time t"""
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3

    def calc_third_derivative(self, t: float) -> float:
        """Calculate jerk at time t"""
        return 6*self.a3 + 24*self.a4*t + 60*self.a5*t**2


class MotionPlanner:
    """Local motion planner using Frenet frame optimization"""

    def __init__(
        self,
        max_speed: float = 30.0,
        max_accel: float = 2.5,
        max_jerk: float = 2.0,
        dt: float = 0.1
    ):
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.max_jerk = max_jerk
        self.dt = dt

        # Cost weights
        self.k_jerk = 0.1
        self.k_time = 1.0
        self.k_diff = 1.0
        self.k_lat = 1.0

    def generate_trajectory(
        self,
        frenet: FrenetFrame,
        s0: float, d0: float, v0: float,
        target_speed: float,
        target_d: List[float],
        T_range: Tuple[float, float] = (4.0, 6.0)
    ) -> List[Trajectory]:
        """
        Generate candidate trajectories in Frenet frame

        Args:
            frenet: Frenet frame reference
            s0, d0: Current Frenet position
            v0: Current velocity
            target_speed: Desired speed
            target_d: List of target lateral positions to sample
            T_range: Time horizon range (min, max)

        Returns:
            List of candidate trajectories
        """
        trajectories = []

        # Sample time horizons
        T_samples = np.linspace(T_range[0], T_range[1], 3)

        for T in T_samples:
            for d_target in target_d:
                # Longitudinal trajectory
                s1 = s0 + v0 * T + 0.5 * self.max_accel * T**2
                s_poly = QuinticPolynomial(s0, v0, 0.0, s1, target_speed, 0.0, T)

                # Lateral trajectory
                d_poly = QuinticPolynomial(d0, 0.0, 0.0, d_target, 0.0, 0.0, T)

                # Generate waypoints
                waypoints = []
                valid = True

                for t in np.arange(0, T, self.dt):
                    s = s_poly.calc_point(t)
                    d = d_poly.calc_point(t)
                    v = s_poly.calc_first_derivative(t)
                    a = s_poly.calc_second_derivative(t)
                    jerk = s_poly.calc_third_derivative(t)

                    # Check constraints
                    if v < 0 or v > self.max_speed:
                        valid = False
                        break
                    if abs(a) > self.max_accel:
                        valid = False
                        break
                    if abs(jerk) > self.max_jerk:
                        valid = False
                        break

                    # Convert to Cartesian
                    x, y, theta = frenet.frenet_to_cartesian(s, d)

                    waypoints.append(Waypoint(x, y, theta, v))

                if valid and waypoints:
                    # Calculate cost
                    cost = self._calculate_cost(s_poly, d_poly, T, d_target, d0)
                    behavior = self._determine_behavior(d_target, d0)

                    trajectories.append(Trajectory(waypoints, cost, behavior))

        # Sort by cost
        trajectories.sort(key=lambda t: t.cost)

        return trajectories

    def _calculate_cost(
        self,
        s_poly: QuinticPolynomial,
        d_poly: QuinticPolynomial,
        T: float,
        d_target: float,
        d0: float
    ) -> float:
        """Calculate trajectory cost"""
        # Jerk cost (comfort)
        jerk_s = s_poly.calc_third_derivative(T/2)
        jerk_d = d_poly.calc_third_derivative(T/2)
        jerk_cost = self.k_jerk * (jerk_s**2 + jerk_d**2)

        # Time cost (efficiency)
        time_cost = self.k_time * T

        # Lateral deviation cost
        lat_cost = self.k_lat * abs(d_target)

        # Difference from current lateral position
        diff_cost = self.k_diff * abs(d_target - d0)

        total_cost = jerk_cost + time_cost + lat_cost + diff_cost

        return total_cost

    def _determine_behavior(self, d_target: float, d0: float) -> BehaviorState:
        """Determine behavior based on lateral movement"""
        d_diff = d_target - d0

        if abs(d_diff) < 0.5:
            return BehaviorState.LANE_KEEP
        elif d_diff > 0.5:
            return BehaviorState.LANE_CHANGE_LEFT
        elif d_diff < -0.5:
            return BehaviorState.LANE_CHANGE_RIGHT
        else:
            return BehaviorState.LANE_KEEP


class BehavioralPlanner:
    """High-level behavioral decision making"""

    def __init__(self, lane_width: float = 3.5):
        self.lane_width = lane_width
        self.current_state = BehaviorState.LANE_KEEP

    def decide_behavior(
        self,
        current_lane: int,
        ego_velocity: float,
        surrounding_vehicles: List[dict]
    ) -> Tuple[BehaviorState, List[float]]:
        """
        Decide driving behavior and target lateral positions

        Args:
            current_lane: Current lane index (0 = center)
            ego_velocity: Ego vehicle velocity
            surrounding_vehicles: List of detected vehicles with positions

        Returns:
            (behavior_state, target_d_values)
        """
        # Default: stay in lane
        target_d = [current_lane * self.lane_width]

        # Check for lane change opportunities
        left_clear = self._is_lane_clear(current_lane - 1, surrounding_vehicles)
        right_clear = self._is_lane_clear(current_lane + 1, surrounding_vehicles)

        # Check if following slow vehicle
        front_vehicle = self._get_front_vehicle(current_lane, surrounding_vehicles)

        if front_vehicle and front_vehicle['velocity'] < ego_velocity - 5.0:
            # Slow vehicle ahead, consider lane change
            if left_clear:
                target_d.append((current_lane - 1) * self.lane_width)
                self.current_state = BehaviorState.LANE_CHANGE_LEFT
            elif right_clear:
                target_d.append((current_lane + 1) * self.lane_width)
                self.current_state = BehaviorState.LANE_CHANGE_RIGHT
            else:
                self.current_state = BehaviorState.FOLLOW_VEHICLE
        else:
            self.current_state = BehaviorState.LANE_KEEP

        return self.current_state, target_d

    def _is_lane_clear(self, lane: int, vehicles: List[dict]) -> bool:
        """Check if lane is clear for lane change"""
        for vehicle in vehicles:
            if vehicle['lane'] == lane:
                if abs(vehicle['distance']) < 30.0:  # 30m safety margin
                    return False
        return True

    def _get_front_vehicle(self, lane: int, vehicles: List[dict]) -> Optional[dict]:
        """Get front vehicle in same lane"""
        front_vehicles = [v for v in vehicles if v['lane'] == lane and v['distance'] > 0]
        if front_vehicles:
            return min(front_vehicles, key=lambda v: v['distance'])
        return None


class PathPlanner:
    """
    Complete path planning system

    Combines global, behavioral, and local planning
    """

    def __init__(self, config: dict):
        self.config = config
        self.motion_planner = MotionPlanner(
            max_speed=config.get('max_speed', 30.0),
            max_accel=config.get('max_accel', 2.5),
            max_jerk=config.get('max_jerk', 2.0)
        )
        self.behavioral_planner = BehavioralPlanner(
            lane_width=config.get('lane_width', 3.5)
        )

    def plan(
        self,
        reference_path: List[Tuple[float, float]],
        current_state: dict,
        obstacles: List[dict]
    ) -> Optional[Trajectory]:
        """
        Generate optimal trajectory

        Args:
            reference_path: Reference path as list of (x, y) points
            current_state: Dict with 'x', 'y', 'v', 'lane'
            obstacles: List of detected obstacles

        Returns:
            Best trajectory or None
        """
        # Create Frenet frame
        frenet = FrenetFrame(reference_path)

        # Convert current position to Frenet
        s0, d0 = frenet.cartesian_to_frenet(current_state['x'], current_state['y'])

        # Behavioral planning
        behavior, target_d_list = self.behavioral_planner.decide_behavior(
            current_state['lane'],
            current_state['v'],
            obstacles
        )

        # Generate candidate trajectories
        trajectories = self.motion_planner.generate_trajectory(
            frenet=frenet,
            s0=s0,
            d0=d0,
            v0=current_state['v'],
            target_speed=self.config.get('cruise_speed', 25.0),
            target_d=target_d_list
        )

        # Check for collisions
        valid_trajectories = [t for t in trajectories if not self._has_collision(t, obstacles)]

        if valid_trajectories:
            return valid_trajectories[0]  # Return lowest cost trajectory
        else:
            # Emergency: generate emergency stop trajectory
            return self._generate_emergency_trajectory(frenet, s0, d0, current_state['v'])

    def _has_collision(self, trajectory: Trajectory, obstacles: List[dict]) -> bool:
        """Check if trajectory collides with obstacles"""
        # Simplified collision check
        # In practice, would check for overlap of vehicle bounding box with obstacles
        for waypoint in trajectory.waypoints:
            for obstacle in obstacles:
                dist = math.sqrt(
                    (waypoint.x - obstacle['x'])**2 +
                    (waypoint.y - obstacle['y'])**2
                )
                if dist < 5.0:  # 5m safety margin
                    return True
        return False

    def _generate_emergency_trajectory(
        self,
        frenet: FrenetFrame,
        s0: float,
        d0: float,
        v0: float
    ) -> Trajectory:
        """Generate emergency stop trajectory"""
        T = max(v0 / 5.0, 2.0)  # Time to stop
        waypoints = []

        for t in np.arange(0, T, 0.1):
            v = v0 * (1 - t/T)
            s = s0 + v0*t - 0.5*(v0/T)*t**2
            d = d0

            x, y, theta = frenet.frenet_to_cartesian(s, d)
            waypoints.append(Waypoint(x, y, theta, v))

        return Trajectory(waypoints, float('inf'), BehaviorState.EMERGENCY_STOP)


if __name__ == "__main__":
    print("Path Planning Module")
    print("=" * 50)

    # Example usage
    config = {
        'max_speed': 30.0,
        'max_accel': 2.5,
        'max_jerk': 2.0,
        'cruise_speed': 25.0,
        'lane_width': 3.5
    }

    planner = PathPlanner(config)

    # Create simple reference path (straight road)
    reference_path = [(i * 1.0, 0.0) for i in range(500)]

    current_state = {
        'x': 0.0,
        'y': 0.0,
        'v': 20.0,
        'lane': 0
    }

    obstacles = []

    trajectory = planner.plan(reference_path, current_state, obstacles)

    if trajectory:
        print(f"Generated trajectory with {len(trajectory.waypoints)} waypoints")
        print(f"Behavior: {trajectory.behavior.value}")
        print(f"Cost: {trajectory.cost:.2f}")
        print(f"First waypoint: ({trajectory.waypoints[0].x:.2f}, {trajectory.waypoints[0].y:.2f})")
    else:
        print("No valid trajectory found!")

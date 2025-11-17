"""
Vehicle Control Module

Implements hierarchical control for autonomous truck:
- High-level: Model Predictive Control (MPC)
- Low-level: PID control for steering and throttle/brake

Author: Self-Driving Truck Project
Date: 2025-11-17
"""

import numpy as np
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class ControlCommand:
    """Control commands for vehicle"""
    steering: float  # Steering angle in radians
    throttle: float  # Throttle [0, 1]
    brake: float  # Brake [0, 1]
    gear: str = "D"  # D, R, N, P


@dataclass
class VehicleParameters:
    """Vehicle physical parameters"""
    wheelbase: float = 4.5  # m
    max_steering_angle: float = np.radians(35)  # rad
    max_acceleration: float = 2.5  # m/s²
    max_deceleration: float = 8.0  # m/s²
    vehicle_length: float = 7.0  # m
    vehicle_width: float = 2.5  # m


class PIDController:
    """PID controller implementation"""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_min: float = -1.0, output_max: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max

        self.integral = 0.0
        self.prev_error = 0.0
        self.dt = 0.01  # Default timestep

    def update(self, error: float, dt: Optional[float] = None) -> float:
        """
        Update PID controller

        Args:
            error: Current error
            dt: Time step (optional)

        Returns:
            Control output
        """
        if dt is not None:
            self.dt = dt

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * self.dt
        i_term = self.ki * self.integral

        # Derivative term
        d_term = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error

        # Calculate output
        output = p_term + i_term + d_term

        # Clamp output
        output = np.clip(output, self.output_min, self.output_max)

        # Anti-windup
        if output == self.output_max or output == self.output_min:
            self.integral -= error * self.dt  # Prevent integral windup

        return output

    def reset(self):
        """Reset controller state"""
        self.integral = 0.0
        self.prev_error = 0.0


class StanleyController:
    """
    Stanley lateral controller for path tracking

    Combines heading error and cross-track error
    """

    def __init__(self, k: float = 1.0, ks: float = 0.5, wheelbase: float = 4.5):
        self.k = k  # Cross-track error gain
        self.ks = ks  # Softening constant
        self.wheelbase = wheelbase

    def compute_steering(
        self,
        front_axle_pos: Tuple[float, float],
        heading: float,
        path_points: List[Tuple[float, float, float]],
        velocity: float
    ) -> float:
        """
        Compute steering angle using Stanley controller

        Args:
            front_axle_pos: Front axle position (x, y)
            heading: Current heading angle (radians)
            path_points: List of (x, y, theta) path points
            velocity: Current velocity (m/s)

        Returns:
            Steering angle (radians)
        """
        # Find nearest path point
        fx, fy = front_axle_pos
        min_dist = float('inf')
        target_idx = 0

        for i, (px, py, _) in enumerate(path_points):
            dist = math.sqrt((fx - px)**2 + (fy - py)**2)
            if dist < min_dist:
                min_dist = dist
                target_idx = i

        # Get target point
        tx, ty, t_theta = path_points[target_idx]

        # Heading error
        theta_e = self.normalize_angle(t_theta - heading)

        # Cross-track error
        # Calculate perpendicular distance from front axle to path
        path_vec_x = math.cos(t_theta)
        path_vec_y = math.sin(t_theta)

        to_point_x = fx - tx
        to_point_y = fy - ty

        # Cross-track error (positive = left of path)
        cross_track_error = -(to_point_x * path_vec_y - to_point_y * path_vec_x)

        # Cross-track steering
        theta_d = math.atan2(self.k * cross_track_error, self.ks + velocity)

        # Total steering angle
        delta = theta_e + theta_d

        return delta

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class PurePursuitController:
    """
    Pure Pursuit lateral controller

    Geometric path tracking algorithm
    """

    def __init__(self, lookahead_gain: float = 0.5,
                 lookahead_min: float = 5.0,
                 lookahead_max: float = 20.0,
                 wheelbase: float = 4.5):
        self.lookahead_gain = lookahead_gain
        self.lookahead_min = lookahead_min
        self.lookahead_max = lookahead_max
        self.wheelbase = wheelbase

    def compute_steering(
        self,
        position: Tuple[float, float],
        heading: float,
        path_points: List[Tuple[float, float]],
        velocity: float
    ) -> float:
        """
        Compute steering angle using Pure Pursuit

        Args:
            position: Current position (x, y)
            heading: Current heading (radians)
            path_points: List of (x, y) path points
            velocity: Current velocity (m/s)

        Returns:
            Steering angle (radians)
        """
        # Compute lookahead distance
        lookahead = np.clip(
            self.lookahead_gain * velocity,
            self.lookahead_min,
            self.lookahead_max
        )

        # Find lookahead point
        cx, cy = position
        min_dist_diff = float('inf')
        target_idx = 0

        for i, (px, py) in enumerate(path_points):
            dist = math.sqrt((px - cx)**2 + (py - cy)**2)
            dist_diff = abs(dist - lookahead)

            if dist_diff < min_dist_diff and dist >= lookahead * 0.8:
                min_dist_diff = dist_diff
                target_idx = i

        # Get target point
        tx, ty = path_points[target_idx]

        # Transform to vehicle frame
        dx = tx - cx
        dy = ty - cy

        # Rotate to vehicle frame
        target_x = dx * math.cos(-heading) - dy * math.sin(-heading)
        target_y = dx * math.sin(-heading) + dy * math.cos(-heading)

        # Compute steering angle
        alpha = math.atan2(target_y, target_x)
        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), lookahead)

        return delta


class LongitudinalController:
    """Longitudinal controller for speed control"""

    def __init__(self, vehicle_params: VehicleParameters):
        self.vehicle_params = vehicle_params

        # PID for velocity control
        self.velocity_pid = PIDController(
            kp=0.3, ki=0.05, kd=0.02,
            output_min=-1.0, output_max=1.0
        )

    def compute_throttle_brake(
        self,
        current_velocity: float,
        target_velocity: float,
        dt: float
    ) -> Tuple[float, float]:
        """
        Compute throttle and brake commands

        Args:
            current_velocity: Current velocity (m/s)
            target_velocity: Desired velocity (m/s)
            dt: Time step (s)

        Returns:
            (throttle, brake) both in [0, 1]
        """
        error = target_velocity - current_velocity

        # PID output
        output = self.velocity_pid.update(error, dt)

        # Split into throttle/brake
        if output > 0:
            throttle = min(output, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(-output, 1.0)

        return throttle, brake


class MPCController:
    """
    Model Predictive Controller for trajectory tracking

    Simplified MPC implementation (full version would use optimization library)
    """

    def __init__(
        self,
        vehicle_params: VehicleParameters,
        horizon: int = 20,
        dt: float = 0.1
    ):
        self.vehicle_params = vehicle_params
        self.horizon = horizon
        self.dt = dt

        # Cost weights
        self.Q = np.diag([10.0, 10.0, 1.0, 1.0])  # State cost [x, y, theta, v]
        self.R = np.diag([1.0, 1.0])  # Control cost [delta, a]

    def compute_control(
        self,
        current_state: np.ndarray,
        reference_trajectory: List[np.ndarray]
    ) -> Tuple[float, float]:
        """
        Compute optimal control using MPC

        Args:
            current_state: [x, y, theta, v]
            reference_trajectory: List of reference states

        Returns:
            (steering, acceleration)
        """
        # Simplified MPC: Use first-order approximation
        # In practice, would use cvxpy or casadi for optimization

        if not reference_trajectory:
            return 0.0, 0.0

        # Take first reference point
        ref_state = reference_trajectory[0]

        # State error
        error = ref_state - current_state

        # Simple proportional control (approximation of MPC)
        # Lateral error
        lateral_error = error[0] * math.sin(current_state[2]) - \
                       error[1] * math.cos(current_state[2])

        # Heading error
        heading_error = self.normalize_angle(error[2])

        # Steering (lateral + heading)
        steering = -0.5 * lateral_error + 1.0 * heading_error
        steering = np.clip(steering, -self.vehicle_params.max_steering_angle,
                          self.vehicle_params.max_steering_angle)

        # Velocity error
        velocity_error = error[3]

        # Acceleration
        acceleration = 0.5 * velocity_error
        acceleration = np.clip(acceleration,
                              -self.vehicle_params.max_deceleration,
                              self.vehicle_params.max_acceleration)

        return steering, acceleration

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class VehicleController:
    """
    High-level vehicle controller

    Combines lateral and longitudinal control
    """

    def __init__(
        self,
        vehicle_params: Optional[VehicleParameters] = None,
        control_mode: str = 'stanley'  # 'stanley', 'pure_pursuit', 'mpc'
    ):
        self.vehicle_params = vehicle_params or VehicleParameters()
        self.control_mode = control_mode

        # Initialize controllers
        if control_mode == 'stanley':
            self.lateral_controller = StanleyController(
                k=1.0, ks=0.5,
                wheelbase=self.vehicle_params.wheelbase
            )
        elif control_mode == 'pure_pursuit':
            self.lateral_controller = PurePursuitController(
                lookahead_gain=0.5,
                wheelbase=self.vehicle_params.wheelbase
            )
        elif control_mode == 'mpc':
            self.lateral_controller = MPCController(
                vehicle_params=self.vehicle_params
            )
        else:
            raise ValueError(f"Unknown control mode: {control_mode}")

        self.longitudinal_controller = LongitudinalController(self.vehicle_params)

        # Emergency stop flag
        self.emergency_stop = False

    def compute_control(
        self,
        current_state: dict,
        trajectory: List[dict],
        dt: float = 0.01
    ) -> ControlCommand:
        """
        Compute control commands

        Args:
            current_state: Dict with 'x', 'y', 'theta', 'v'
            trajectory: List of trajectory points with 'x', 'y', 'theta', 'v'
            dt: Control timestep

        Returns:
            ControlCommand object
        """
        if self.emergency_stop:
            return ControlCommand(
                steering=0.0,
                throttle=0.0,
                brake=1.0
            )

        # Extract path points for lateral control
        if self.control_mode in ['stanley', 'pure_pursuit']:
            if self.control_mode == 'stanley':
                # Front axle position
                front_x = current_state['x'] + self.vehicle_params.wheelbase * \
                          math.cos(current_state['theta'])
                front_y = current_state['y'] + self.vehicle_params.wheelbase * \
                          math.sin(current_state['theta'])

                path_points = [(p['x'], p['y'], p['theta']) for p in trajectory]

                steering = self.lateral_controller.compute_steering(
                    (front_x, front_y),
                    current_state['theta'],
                    path_points,
                    current_state['v']
                )
            else:  # pure_pursuit
                path_points = [(p['x'], p['y']) for p in trajectory]

                steering = self.lateral_controller.compute_steering(
                    (current_state['x'], current_state['y']),
                    current_state['theta'],
                    path_points,
                    current_state['v']
                )

            # Get target velocity from first trajectory point
            target_v = trajectory[0]['v'] if trajectory else current_state['v']

            # Longitudinal control
            throttle, brake = self.longitudinal_controller.compute_throttle_brake(
                current_state['v'],
                target_v,
                dt
            )

        else:  # MPC
            current_state_vec = np.array([
                current_state['x'],
                current_state['y'],
                current_state['theta'],
                current_state['v']
            ])

            ref_trajectory = [
                np.array([p['x'], p['y'], p['theta'], p['v']])
                for p in trajectory
            ]

            steering, acceleration = self.lateral_controller.compute_control(
                current_state_vec,
                ref_trajectory
            )

            # Convert acceleration to throttle/brake
            if acceleration > 0:
                throttle = min(acceleration / self.vehicle_params.max_acceleration, 1.0)
                brake = 0.0
            else:
                throttle = 0.0
                brake = min(-acceleration / self.vehicle_params.max_deceleration, 1.0)

        # Clamp steering
        steering = np.clip(
            steering,
            -self.vehicle_params.max_steering_angle,
            self.vehicle_params.max_steering_angle
        )

        return ControlCommand(
            steering=steering,
            throttle=throttle,
            brake=brake
        )

    def activate_emergency_stop(self):
        """Activate emergency stop"""
        self.emergency_stop = True

    def deactivate_emergency_stop(self):
        """Deactivate emergency stop"""
        self.emergency_stop = False


if __name__ == "__main__":
    print("Vehicle Control Module")
    print("=" * 50)

    # Example usage
    vehicle_params = VehicleParameters()
    controller = VehicleController(vehicle_params, control_mode='stanley')

    current_state = {
        'x': 0.0,
        'y': 0.0,
        'theta': 0.0,
        'v': 15.0
    }

    # Simple straight trajectory
    trajectory = [
        {'x': i * 1.0, 'y': 0.0, 'theta': 0.0, 'v': 20.0}
        for i in range(1, 50)
    ]

    command = controller.compute_control(current_state, trajectory, dt=0.01)

    print(f"Control Commands:")
    print(f"  Steering: {np.degrees(command.steering):.2f}°")
    print(f"  Throttle: {command.throttle:.2f}")
    print(f"  Brake: {command.brake:.2f}")

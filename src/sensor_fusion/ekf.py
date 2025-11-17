"""
Extended Kalman Filter for Sensor Fusion

Fuses data from GPS, IMU, wheel encoders, camera, LiDAR, and radar
for accurate state estimation.

Author: Self-Driving Truck Project
Date: 2025-11-17
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class SensorType(Enum):
    """Sensor types for fusion"""
    GPS = "gps"
    IMU = "imu"
    WHEEL_ENCODER = "wheel_encoder"
    CAMERA = "camera"
    LIDAR = "lidar"
    RADAR = "radar"


@dataclass
class VehicleState:
    """Vehicle state vector"""
    x: float  # Position X (meters)
    y: float  # Position Y (meters)
    theta: float  # Heading angle (radians)
    v: float  # Velocity (m/s)
    a: float  # Acceleration (m/s²)
    omega: float  # Yaw rate (rad/s)
    beta: float = 0.0  # Trailer angle (rad)

    def to_vector(self) -> np.ndarray:
        """Convert to numpy array"""
        return np.array([self.x, self.y, self.theta, self.v, self.a, self.omega, self.beta])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'VehicleState':
        """Create from numpy array"""
        return cls(
            x=vec[0], y=vec[1], theta=vec[2],
            v=vec[3], a=vec[4], omega=vec[5],
            beta=vec[6] if len(vec) > 6 else 0.0
        )


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for vehicle state estimation

    State vector: [x, y, theta, v, a, omega, beta]
    """

    def __init__(
        self,
        dt: float = 0.01,
        wheelbase: float = 4.5,
        trailer_length: float = 11.0
    ):
        self.dt = dt
        self.wheelbase = wheelbase
        self.trailer_length = trailer_length

        # State dimension
        self.n_states = 7

        # Initialize state
        self.x = np.zeros(self.n_states)  # State vector
        self.P = np.eye(self.n_states) * 10.0  # Covariance matrix

        # Process noise covariance Q
        self.Q = np.diag([
            0.1,   # x position
            0.1,   # y position
            0.01,  # theta (heading)
            0.5,   # velocity
            0.3,   # acceleration
            0.1,   # yaw rate
            0.05,  # trailer angle
        ])

        # Measurement noise covariances (will be updated per sensor)
        self.R = {
            SensorType.GPS: np.diag([0.5, 0.5]),  # x, y
            SensorType.IMU: np.diag([0.01, 0.05]),  # theta, omega
            SensorType.WHEEL_ENCODER: np.array([[0.1]]),  # velocity
            SensorType.CAMERA: np.diag([0.3, 0.3]),  # relative position
            SensorType.LIDAR: np.diag([0.1, 0.1]),  # obstacle position
            SensorType.RADAR: np.diag([0.2, 0.3]),  # distance, velocity
        }

    def predict(self, control_input: Optional[np.ndarray] = None) -> VehicleState:
        """
        Prediction step of EKF

        Args:
            control_input: [steering_angle, acceleration] (optional)

        Returns:
            Predicted state
        """
        # Extract current state
        x, y, theta, v, a, omega, beta = self.x

        # State transition (bicycle model with trailer)
        if control_input is not None:
            delta, a_cmd = control_input
            a = a_cmd  # Update acceleration from control
        else:
            delta = 0.0

        # Predict next state using motion model
        x_next = x + v * np.cos(theta) * self.dt
        y_next = y + v * np.sin(theta) * self.dt
        theta_next = theta + omega * self.dt
        v_next = v + a * self.dt
        a_next = a  # Assume constant acceleration
        omega_next = (v / self.wheelbase) * np.tan(delta)

        # Trailer dynamics
        beta_dot = (v / self.trailer_length) * np.sin(theta - beta)
        beta_next = beta + beta_dot * self.dt

        # Update state
        self.x = np.array([x_next, y_next, theta_next, v_next, a_next, omega_next, beta_next])

        # Compute Jacobian of state transition
        F = self.compute_state_jacobian(v, theta, omega, beta, delta)

        # Update covariance
        self.P = F @ self.P @ F.T + self.Q

        return VehicleState.from_vector(self.x)

    def compute_state_jacobian(
        self,
        v: float,
        theta: float,
        omega: float,
        beta: float,
        delta: float
    ) -> np.ndarray:
        """Compute Jacobian of state transition function"""
        dt = self.dt

        F = np.eye(self.n_states)

        # Partial derivatives
        F[0, 2] = -v * np.sin(theta) * dt  # dx/dtheta
        F[0, 3] = np.cos(theta) * dt        # dx/dv

        F[1, 2] = v * np.cos(theta) * dt    # dy/dtheta
        F[1, 3] = np.sin(theta) * dt        # dy/dv

        F[2, 5] = dt                        # dtheta/domega

        F[3, 4] = dt                        # dv/da

        # Trailer angle dynamics
        F[6, 2] = (v / self.trailer_length) * np.cos(theta - beta) * dt
        F[6, 3] = (1 / self.trailer_length) * np.sin(theta - beta) * dt
        F[6, 6] = 1 - (v / self.trailer_length) * np.cos(theta - beta) * dt

        return F

    def update_gps(self, measurement: np.ndarray):
        """
        Update step with GPS measurement

        Args:
            measurement: [x, y] position
        """
        # Measurement model: H maps state to measurement
        H = np.zeros((2, self.n_states))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y

        # Predicted measurement
        z_pred = H @ self.x

        # Innovation
        y = measurement - z_pred

        # Innovation covariance
        S = H @ self.P @ H.T + self.R[SensorType.GPS]

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.n_states)
        self.P = (I - K @ H) @ self.P

    def update_imu(self, measurement: np.ndarray):
        """
        Update step with IMU measurement

        Args:
            measurement: [theta, omega] heading and yaw rate
        """
        H = np.zeros((2, self.n_states))
        H[0, 2] = 1.0  # theta
        H[1, 5] = 1.0  # omega

        z_pred = H @ self.x
        y = measurement - z_pred

        # Normalize angle difference
        y[0] = self.normalize_angle(y[0])

        S = H @ self.P @ H.T + self.R[SensorType.IMU]
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.x[2] = self.normalize_angle(self.x[2])  # Normalize theta

        I = np.eye(self.n_states)
        self.P = (I - K @ H) @ self.P

    def update_wheel_encoder(self, measurement: float):
        """
        Update step with wheel encoder measurement

        Args:
            measurement: velocity (m/s)
        """
        H = np.zeros((1, self.n_states))
        H[0, 3] = 1.0  # velocity

        z_pred = H @ self.x
        y = np.array([measurement]) - z_pred

        S = H @ self.P @ H.T + self.R[SensorType.WHEEL_ENCODER]
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(self.n_states)
        self.P = (I - K @ H) @ self.P

    def update_camera_lane(self, lateral_offset: float, heading_error: float):
        """
        Update step with camera lane detection

        Args:
            lateral_offset: Lateral offset from lane center (meters)
            heading_error: Heading error relative to lane (radians)
        """
        # This is a simplified model
        # In practice, would need to transform lane measurements to global frame
        measurement = np.array([lateral_offset, heading_error])

        # Simplified measurement model
        H = np.zeros((2, self.n_states))
        H[0, 1] = 1.0  # Approximate lateral offset as y
        H[1, 2] = 1.0  # Heading error as theta

        z_pred = H @ self.x
        y = measurement - z_pred

        y[1] = self.normalize_angle(y[1])

        S = H @ self.P @ H.T + self.R[SensorType.CAMERA]
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y

        I = np.eye(self.n_states)
        self.P = (I - K @ H) @ self.P

    def get_state(self) -> VehicleState:
        """Get current state estimate"""
        return VehicleState.from_vector(self.x)

    def get_covariance(self) -> np.ndarray:
        """Get current covariance matrix"""
        return self.P.copy()

    def get_position_uncertainty(self) -> Tuple[float, float]:
        """Get position uncertainty (std dev in x, y)"""
        return np.sqrt(self.P[0, 0]), np.sqrt(self.P[1, 1])

    def get_heading_uncertainty(self) -> float:
        """Get heading uncertainty (std dev)"""
        return np.sqrt(self.P[2, 2])

    @staticmethod
    def normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def reset(self, initial_state: Optional[VehicleState] = None):
        """Reset filter to initial state"""
        if initial_state:
            self.x = initial_state.to_vector()
        else:
            self.x = np.zeros(self.n_states)

        self.P = np.eye(self.n_states) * 10.0


class SensorFusionSystem:
    """
    High-level sensor fusion system

    Manages multiple EKFs and sensor synchronization
    """

    def __init__(self, update_rate: float = 100.0):
        self.dt = 1.0 / update_rate
        self.ekf = ExtendedKalmanFilter(dt=self.dt)

        # Sensor timestamps
        self.last_update = {
            SensorType.GPS: 0.0,
            SensorType.IMU: 0.0,
            SensorType.WHEEL_ENCODER: 0.0,
            SensorType.CAMERA: 0.0,
        }

        # Current time
        self.current_time = 0.0

    def update(
        self,
        sensor_data: Dict[SensorType, Tuple[float, np.ndarray]],
        control_input: Optional[np.ndarray] = None
    ) -> VehicleState:
        """
        Update sensor fusion with new sensor data

        Args:
            sensor_data: Dict mapping SensorType to (timestamp, measurement)
            control_input: Control inputs [steering, acceleration]

        Returns:
            Fused vehicle state
        """
        # Prediction step
        predicted_state = self.ekf.predict(control_input)

        # Update step for each sensor
        for sensor_type, (timestamp, measurement) in sensor_data.items():
            if timestamp > self.last_update[sensor_type]:
                if sensor_type == SensorType.GPS:
                    self.ekf.update_gps(measurement)
                elif sensor_type == SensorType.IMU:
                    self.ekf.update_imu(measurement)
                elif sensor_type == SensorType.WHEEL_ENCODER:
                    self.ekf.update_wheel_encoder(measurement[0])
                elif sensor_type == SensorType.CAMERA:
                    self.ekf.update_camera_lane(measurement[0], measurement[1])

                self.last_update[sensor_type] = timestamp

        self.current_time += self.dt

        return self.ekf.get_state()

    def get_state_with_uncertainty(self) -> Tuple[VehicleState, np.ndarray]:
        """Get state estimate with covariance"""
        return self.ekf.get_state(), self.ekf.get_covariance()


if __name__ == "__main__":
    print("Extended Kalman Filter - Sensor Fusion")
    print("=" * 50)

    # Example usage
    fusion = SensorFusionSystem(update_rate=100.0)

    # Simulate sensor data
    sensor_data = {
        SensorType.GPS: (0.01, np.array([10.0, 5.0])),
        SensorType.IMU: (0.01, np.array([0.1, 0.05])),
        SensorType.WHEEL_ENCODER: (0.01, np.array([15.0])),
    }

    state = fusion.update(sensor_data, control_input=np.array([0.0, 0.5]))

    print(f"Fused State:")
    print(f"  Position: ({state.x:.2f}, {state.y:.2f})")
    print(f"  Heading: {np.degrees(state.theta):.2f}°")
    print(f"  Velocity: {state.v:.2f} m/s")
    print(f"  Yaw Rate: {np.degrees(state.omega):.2f}°/s")

    pos_unc = fusion.ekf.get_position_uncertainty()
    print(f"\nUncertainty:")
    print(f"  Position: ±{pos_unc[0]:.2f}m (x), ±{pos_unc[1]:.2f}m (y)")
    print(f"  Heading: ±{np.degrees(fusion.ekf.get_heading_uncertainty()):.2f}°")

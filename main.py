#!/usr/bin/env python3
"""
Main entry point for Self-Driving Truck System

This script initializes and runs the complete autonomous driving pipeline
including perception, planning, and control.

Usage:
    python main.py --config config/vehicle_config.yaml
    python main.py --simulation --scenario highway_cruise
    python main.py --camera /dev/video0 --lidar eth0

Author: Self-Driving Truck Project
Date: 2025-11-17
"""

import argparse
import sys
import time
import signal
from pathlib import Path
import yaml
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from src.computer_vision.models.lane_detection import LaneDetector, LaneConfig
    from src.computer_vision.models.object_detection import ObjectDetector, DetectionConfig
    from src.computer_vision.models.traffic_sign_recognition import TrafficSignRecognizer, SignRecognitionConfig
    from src.sensor_fusion.ekf import SensorFusionSystem, SensorType, VehicleState
    from src.planning.path_planner import PathPlanner
    from src.control.controller import VehicleController, VehicleParameters
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class AutonomousTruckSystem:
    """Main autonomous truck system orchestrator"""

    def __init__(self, config_path: str = "config/vehicle_config.yaml"):
        """
        Initialize the autonomous truck system

        Args:
            config_path: Path to vehicle configuration file
        """
        print("=" * 70)
        print("Self-Driving Truck System - Initializing")
        print("=" * 70)

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize components
        print("\n[1/6] Initializing perception modules...")
        self._init_perception()

        print("[2/6] Initializing sensor fusion...")
        self._init_sensor_fusion()

        print("[3/6] Initializing path planner...")
        self._init_planner()

        print("[4/6] Initializing vehicle controller...")
        self._init_controller()

        print("[5/6] Setting up safety monitors...")
        self._init_safety()

        print("[6/6] Starting main loop...")

        # System state
        self.running = False
        self.emergency_stop = False

        # Performance metrics
        self.frame_count = 0
        self.start_time = time.time()

        print("\n" + "=" * 70)
        print("System initialized successfully!")
        print("=" * 70 + "\n")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'vehicle': {
                'dimensions': {
                    'cab': {'wheelbase': 4.5, 'length': 7.0, 'width': 2.5}
                }
            },
            'control': {
                'frequencies': {
                    'perception': 30,
                    'planning': 10,
                    'control': 100
                }
            },
            'planning': {
                'max_speed': 30.0,
                'max_accel': 2.5,
                'max_jerk': 2.0,
                'cruise_speed': 25.0,
                'lane_width': 3.5
            }
        }

    def _init_perception(self):
        """Initialize perception modules"""
        self.lane_detector = LaneDetector(config=LaneConfig())
        print("  ✓ Lane detector ready")

        self.object_detector = ObjectDetector(config=DetectionConfig())
        print("  ✓ Object detector ready")

        self.sign_recognizer = TrafficSignRecognizer(config=SignRecognitionConfig())
        print("  ✓ Traffic sign recognizer ready")

    def _init_sensor_fusion(self):
        """Initialize sensor fusion system"""
        update_rate = self.config.get('control', {}).get('frequencies', {}).get('sensor_fusion', 100)
        self.sensor_fusion = SensorFusionSystem(update_rate=update_rate)
        print(f"  ✓ Sensor fusion initialized ({update_rate} Hz)")

    def _init_planner(self):
        """Initialize path planner"""
        planning_config = self.config.get('planning', {})
        self.path_planner = PathPlanner(planning_config)
        print("  ✓ Path planner ready")

    def _init_controller(self):
        """Initialize vehicle controller"""
        wheelbase = self.config.get('vehicle', {}).get('dimensions', {}).get('cab', {}).get('wheelbase', 4.5)
        self.vehicle_params = VehicleParameters(wheelbase=wheelbase)
        self.controller = VehicleController(
            vehicle_params=self.vehicle_params,
            control_mode='stanley'
        )
        print("  ✓ Vehicle controller ready (Stanley)")

    def _init_safety(self):
        """Initialize safety monitoring systems"""
        self.safety_monitor = {
            'last_perception': time.time(),
            'last_planning': time.time(),
            'last_control': time.time(),
            'sensor_health': True,
            'system_health': True
        }
        print("  ✓ Safety monitors active")

    def process_frame(self, frame: np.ndarray, sensor_data: dict) -> dict:
        """
        Process single frame through the pipeline

        Args:
            frame: Camera image
            sensor_data: Dict with sensor measurements

        Returns:
            Dict with perception results
        """
        results = {}

        # Lane detection
        lanes = self.lane_detector.detect(frame)
        results['lanes'] = lanes

        # Object detection
        objects = self.object_detector.detect(frame)
        results['objects'] = objects

        # Extract traffic sign ROIs from object detections
        sign_bboxes = [obj.bbox for obj in objects if 'SIGN' in obj.class_name]
        if sign_bboxes:
            signs = self.sign_recognizer.recognize(frame, sign_bboxes)
            results['signs'] = signs
        else:
            results['signs'] = []

        return results

    def update_state(self, perception_results: dict, sensor_data: dict):
        """
        Update vehicle state using sensor fusion

        Args:
            perception_results: Results from perception pipeline
            sensor_data: Raw sensor measurements
        """
        # Prepare sensor data for fusion
        fusion_data = {}

        if 'gps' in sensor_data:
            fusion_data[SensorType.GPS] = (
                sensor_data['gps']['timestamp'],
                np.array([sensor_data['gps']['x'], sensor_data['gps']['y']])
            )

        if 'imu' in sensor_data:
            fusion_data[SensorType.IMU] = (
                sensor_data['imu']['timestamp'],
                np.array([sensor_data['imu']['theta'], sensor_data['imu']['omega']])
            )

        if 'wheel_encoder' in sensor_data:
            fusion_data[SensorType.WHEEL_ENCODER] = (
                sensor_data['wheel_encoder']['timestamp'],
                np.array([sensor_data['wheel_encoder']['velocity']])
            )

        # Fuse sensor data
        control_input = sensor_data.get('control_input', None)
        self.vehicle_state = self.sensor_fusion.update(fusion_data, control_input)

        return self.vehicle_state

    def plan_trajectory(self, vehicle_state: VehicleState, perception_results: dict):
        """
        Plan optimal trajectory

        Args:
            vehicle_state: Current vehicle state
            perception_results: Perception outputs

        Returns:
            Planned trajectory
        """
        # Create reference path (simplified - normally from HD map)
        reference_path = [(i * 1.0, 0.0) for i in range(500)]

        # Prepare current state
        current_state = {
            'x': vehicle_state.x,
            'y': vehicle_state.y,
            'v': vehicle_state.v,
            'lane': 0  # Center lane
        }

        # Convert detected objects to obstacles
        obstacles = []
        for obj in perception_results.get('objects', []):
            if obj.distance:
                obstacles.append({
                    'x': vehicle_state.x + obj.distance * np.cos(vehicle_state.theta),
                    'y': vehicle_state.y + obj.distance * np.sin(vehicle_state.theta),
                    'v': 0.0,  # Assume stationary
                    'lane': 0
                })

        # Plan trajectory
        trajectory = self.path_planner.plan(reference_path, current_state, obstacles)

        return trajectory

    def compute_control(self, vehicle_state: VehicleState, trajectory):
        """
        Compute control commands

        Args:
            vehicle_state: Current state
            trajectory: Planned trajectory

        Returns:
            Control commands
        """
        if trajectory is None or self.emergency_stop:
            self.controller.activate_emergency_stop()
        else:
            self.controller.deactivate_emergency_stop()

        # Prepare current state
        current_state = {
            'x': vehicle_state.x,
            'y': vehicle_state.y,
            'theta': vehicle_state.theta,
            'v': vehicle_state.v
        }

        # Prepare trajectory
        if trajectory:
            trajectory_points = [
                {
                    'x': wp.x,
                    'y': wp.y,
                    'theta': wp.theta,
                    'v': wp.v
                }
                for wp in trajectory.waypoints
            ]
        else:
            trajectory_points = []

        # Compute control
        control_command = self.controller.compute_control(
            current_state,
            trajectory_points,
            dt=0.01
        )

        return control_command

    def run(self, simulation: bool = False):
        """
        Main system loop

        Args:
            simulation: Whether to run in simulation mode
        """
        print("\nStarting autonomous driving system...")
        print("Press Ctrl+C to stop\n")

        self.running = True

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

        try:
            while self.running:
                loop_start = time.time()

                # Simulated sensor data (replace with actual sensor interfaces)
                if simulation:
                    frame, sensor_data = self._get_simulated_data()
                else:
                    frame, sensor_data = self._get_sensor_data()

                # Perception
                perception_results = self.process_frame(frame, sensor_data)

                # State estimation
                vehicle_state = self.update_state(perception_results, sensor_data)

                # Planning (lower frequency)
                if self.frame_count % 3 == 0:  # 10 Hz planning
                    trajectory = self.plan_trajectory(vehicle_state, perception_results)
                    self.current_trajectory = trajectory

                # Control
                control_command = self.compute_control(
                    vehicle_state,
                    getattr(self, 'current_trajectory', None)
                )

                # Execute control (send to vehicle actuators)
                if not simulation:
                    self._execute_control(control_command)

                # Update metrics
                self.frame_count += 1

                # Print status
                if self.frame_count % 30 == 0:  # Every 1 second at 30 FPS
                    self._print_status(vehicle_state, control_command)

                # Maintain loop rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, 1.0/30.0 - elapsed)  # 30 Hz
                time.sleep(sleep_time)

        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._shutdown()

    def _get_simulated_data(self):
        """Get simulated sensor data"""
        # Create dummy frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Simulated sensor data
        sensor_data = {
            'gps': {
                'timestamp': time.time(),
                'x': self.frame_count * 0.5,
                'y': 0.0
            },
            'imu': {
                'timestamp': time.time(),
                'theta': 0.0,
                'omega': 0.0
            },
            'wheel_encoder': {
                'timestamp': time.time(),
                'velocity': 20.0
            }
        }

        return frame, sensor_data

    def _get_sensor_data(self):
        """Get real sensor data (placeholder)"""
        # TODO: Implement actual sensor interfaces
        print("Warning: Using simulated data. Implement real sensor interfaces.")
        return self._get_simulated_data()

    def _execute_control(self, control_command):
        """Execute control commands on vehicle (placeholder)"""
        # TODO: Implement actual vehicle interface (CAN bus, etc.)
        pass

    def _print_status(self, state: VehicleState, control):
        """Print system status"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        print(f"[{elapsed:.1f}s] "
              f"Pos: ({state.x:.1f}, {state.y:.1f}), "
              f"Heading: {np.degrees(state.theta):.1f}°, "
              f"Speed: {state.v:.1f} m/s, "
              f"Steering: {np.degrees(control.steering):.1f}°, "
              f"FPS: {fps:.1f}")

    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C"""
        print("\n\nShutdown signal received...")
        self.running = False

    def _shutdown(self):
        """Graceful shutdown"""
        print("\nShutting down autonomous truck system...")
        print(f"Total frames processed: {self.frame_count}")
        print(f"Average FPS: {self.frame_count / (time.time() - self.start_time):.2f}")
        print("Goodbye!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Self-Driving Truck System")
    parser.add_argument('--config', type=str, default='config/vehicle_config.yaml',
                       help='Path to vehicle configuration file')
    parser.add_argument('--simulation', action='store_true',
                       help='Run in simulation mode')
    parser.add_argument('--scenario', type=str, default='highway_cruise',
                       help='Simulation scenario to run')

    args = parser.parse_args()

    # Create and run system
    system = AutonomousTruckSystem(config_path=args.config)
    system.run(simulation=args.simulation)


if __name__ == "__main__":
    main()

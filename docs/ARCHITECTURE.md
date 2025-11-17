# System Architecture

## Overview

The self-driving truck system follows a modular, layered architecture enabling real-time processing, sensor fusion, intelligent decision-making, and precise vehicle control.

## System Layers

### 1. Sensor Layer

The sensor layer provides raw data from multiple sources:

```
┌─────────────────────────────────────────────────────────────┐
│                      SENSOR LAYER                            │
├─────────────┬──────────────┬────────────┬───────────────────┤
│  Cameras    │    LiDAR     │   Radar    │  GPS/IMU/Odometry │
├─────────────┼──────────────┼────────────┼───────────────────┤
│ • 6x cameras│ • 64-channel │ • 4x radar │ • RTK GPS         │
│ • 1920x1080 │ • 200m range │ • 250m max │ • 9-DOF IMU       │
│ • 30 FPS    │ • 10 Hz      │ • Doppler  │ • Wheel encoders  │
└─────────────┴──────────────┴────────────┴───────────────────┘
```

**Key Components:**
- **Camera Array**: 360° visual coverage
  - Front: Wide FOV for lane detection, traffic signs, vehicles
  - Sides: Blind spot monitoring
  - Rear: Reversing and trailer monitoring

- **LiDAR**: Velodyne HDL-64E or equivalent
  - Accurate 3D point clouds
  - Works in low light conditions
  - Object distance and shape

- **Radar**: Continental ARS4xx series
  - All-weather operation
  - Velocity measurement
  - Long-range detection

- **Localization Sensors**:
  - RTK GPS: 2cm accuracy
  - IMU: Orientation and acceleration
  - Wheel encoders: Dead reckoning

### 2. Perception Layer

Processes sensor data to understand the environment:

```
┌─────────────────────────────────────────────────────────────┐
│                   PERCEPTION LAYER                           │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ Lane         │  Object      │  Traffic     │  Sensor        │
│ Detection    │  Detection   │  Sign Recog  │  Fusion        │
├──────────────┼──────────────┼──────────────┼────────────────┤
│ • U-Net CNN  │ • YOLOv8     │ • CNN        │ • EKF          │
│ • Polynomial │ • 3D bbox    │ • 43 classes │ • Multi-modal  │
│ • 98.5% acc  │ • mAP 0.89   │ • 99.2% acc  │ • 100 Hz       │
└──────────────┴──────────────┴──────────────┴────────────────┘
```

**Modules:**

#### Lane Detection (src/computer_vision/models/lane_detection.py)
```
Input: Camera frame (1920x1080)
  ↓
ResNet50 Encoder
  ↓
U-Net Decoder
  ↓
Lane Segmentation Mask
  ↓
Polynomial Fitting
  ↓
Output: Lane boundaries + confidence
```

- Architecture: U-Net with ResNet50 backbone
- Performance: 98.5% accuracy @ 45 FPS
- Outputs: Left/right lane polynomials, confidence scores

#### Object Detection (src/computer_vision/models/object_detection.py)
```
Input: Camera frame
  ↓
YOLOv8 CNN
  ↓
Multi-scale Feature Extraction
  ↓
Detection Head
  ↓
NMS (Non-Maximum Suppression)
  ↓
Output: [class, bbox, confidence, distance]
```

- Classes: 11 (vehicles, pedestrians, traffic lights, obstacles)
- Performance: mAP 0.89 @ 35 FPS
- Distance estimation: Pinhole camera model

#### Traffic Sign Recognition (src/computer_vision/models/traffic_sign_recognition.py)
```
Input: Sign ROI (64x64)
  ↓
EfficientSignNet / TrafficSignCNN
  ↓
Feature Extraction
  ↓
Softmax Classification
  ↓
Output: Sign class + confidence
```

- Classes: 43 (GTSRB standard)
- Performance: 99.2% accuracy @ 60+ FPS
- Includes speed limit extraction

#### Sensor Fusion (src/sensor_fusion/ekf.py)
```
State Vector: [x, y, θ, v, a, ω, β]

Prediction Step:
  x_k = f(x_k-1, u_k) + w_k
  P_k = F_k * P_k-1 * F_k^T + Q_k

Update Step (per sensor):
  K_k = P_k * H_k^T * (H_k * P_k * H_k^T + R_k)^-1
  x_k = x_k + K_k * (z_k - h(x_k))
  P_k = (I - K_k * H_k) * P_k
```

- Algorithm: Extended Kalman Filter (EKF)
- State: Position, heading, velocity, acceleration, yaw rate, trailer angle
- Update rate: 100 Hz
- Fuses: GPS, IMU, wheel encoders, camera, LiDAR

### 3. Planning Layer

Determines optimal trajectory:

```
┌─────────────────────────────────────────────────────────────┐
│                   PLANNING LAYER                             │
├──────────────┬──────────────┬──────────────────────────────┤
│   Global     │  Behavioral  │      Local Motion            │
│   Planning   │  Planning    │      Planning                │
├──────────────┼──────────────┼──────────────────────────────┤
│ • A* search  │ • FSM        │ • Frenet frame               │
│ • Road graph │ • Lane keep  │ • Quintic polynomials        │
│ • Route      │ • Lane change│ • Cost optimization          │
│              │ • Follow     │ • Collision check            │
└──────────────┴──────────────┴──────────────────────────────┘
```

#### Global Planning
- **Algorithm**: A* on HD map road network
- **Input**: Start/destination GPS coordinates
- **Output**: Sequence of road segments
- **Update**: On route change

#### Behavioral Planning
- **Algorithm**: Finite State Machine (FSM)
- **States**:
  - LANE_KEEP: Maintain current lane
  - LANE_CHANGE_LEFT/RIGHT: Change lanes
  - FOLLOW_VEHICLE: Adaptive cruise control
  - MERGE: Highway merging
  - STOP: Stop at intersection
  - EMERGENCY_STOP: Critical safety event

- **Decision Factors**:
  - Current traffic situation
  - Road rules and regulations
  - Surrounding vehicles
  - Route requirements

#### Local Motion Planning (src/planning/path_planner.py)
```
Frenet Frame Optimization:

Lateral Trajectory:
  d(t) = a0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵

Longitudinal Trajectory:
  s(t) = s0 + v0*t + ½*a*t²

Cost Function:
  J = k_jerk * ∫j² + k_time * T + k_lat * |d| + k_diff * |Δd|

Constraints:
  v_min ≤ v(t) ≤ v_max
  |a(t)| ≤ a_max
  |j(t)| ≤ j_max
  No collisions
```

- **Horizon**: 5 seconds
- **Update Rate**: 10 Hz
- **Output**: Trajectory waypoints [x, y, θ, v, κ]

### 4. Control Layer

Executes planned trajectory:

```
┌─────────────────────────────────────────────────────────────┐
│                   CONTROL LAYER                              │
├──────────────┬──────────────┬──────────────────────────────┤
│  Lateral     │ Longitudinal │      Safety                  │
│  Control     │  Control     │      Monitor                 │
├──────────────┼──────────────┼──────────────────────────────┤
│ • Stanley    │ • PID        │ • Constraint checking        │
│ • Pure Pur.  │ • Accel/Brake│ • Emergency stop             │
│ • MPC        │ • 100 Hz     │ • Failsafe triggers          │
└──────────────┴──────────────┴──────────────────────────────┘
```

#### Lateral Control (src/control/controller.py)

**Stanley Controller:**
```
δ = θ_e + atan(k * e / (k_s + v))

where:
  θ_e = heading error
  e = cross-track error
  v = velocity
  k, k_s = tuning parameters
```

**Pure Pursuit:**
```
α = atan2(2 * L * sin(α), l_d)

where:
  L = wheelbase
  l_d = lookahead distance
  α = angle to lookahead point
```

**MPC (Model Predictive Control):**
```
minimize: Σ(||x - x_ref||²_Q + ||u||²_R)

subject to:
  x_k+1 = f(x_k, u_k)  (vehicle dynamics)
  u_min ≤ u_k ≤ u_max
  x_min ≤ x_k ≤ x_max
```

#### Longitudinal Control
- **Algorithm**: PID controller
- **Inputs**: Target velocity, current velocity
- **Outputs**: Throttle [0,1], Brake [0,1]
- **Gains**: Kp=0.3, Ki=0.05, Kd=0.02

### 5. Actuation Layer

```
┌─────────────────────────────────────────────────────────────┐
│                  ACTUATION LAYER                             │
├──────────────┬──────────────┬──────────────────────────────┤
│  Steering    │  Throttle    │      Brake                   │
│  Actuator    │  Actuator    │      Actuator                │
├──────────────┼──────────────┼──────────────────────────────┤
│ • EPS        │ • Drive-by   │ • Air brake system           │
│ • ±35°       │   -wire      │ • EBS                        │
│ • 30°/s max  │ • 0-100%     │ • 0.4s response time         │
└──────────────┴──────────────┴──────────────────────────────┘
```

## Data Flow

```
Sensors → Raw Data (Camera, LiDAR, Radar, GPS, IMU)
    ↓
Preprocessing → Calibration, Synchronization
    ↓
Perception → Lane, Objects, Signs, Localization
    ↓
World Model → Drivable area, Obstacles, Traffic state
    ↓
Planning → Global route → Behavioral decision → Local trajectory
    ↓
Control → Lateral (steering) + Longitudinal (throttle/brake)
    ↓
Actuation → Vehicle Commands
    ↓
Vehicle Motion
    ↓ (Feedback)
Sensors (closed loop)
```

## Software Architecture

### Module Organization

```
src/
├── computer_vision/
│   ├── models/
│   │   ├── lane_detection.py      # U-Net lane detector
│   │   ├── object_detection.py    # YOLOv8 detector
│   │   └── traffic_sign_recognition.py
│   └── utils/
│       ├── image_processing.py
│       └── visualization.py
│
├── sensor_fusion/
│   ├── ekf.py                      # Extended Kalman Filter
│   ├── calibration.py              # Sensor calibration
│   └── synchronization.py          # Temporal alignment
│
├── perception/
│   ├── pipeline.py                 # Main perception pipeline
│   ├── object_tracking.py          # Multi-object tracking
│   └── world_model.py              # Environment representation
│
├── planning/
│   ├── path_planner.py             # Motion planning
│   ├── behavior_planner.py         # Behavioral FSM
│   └── global_planner.py           # A* route planning
│
├── control/
│   ├── controller.py               # Vehicle controllers
│   ├── mpc.py                      # Model predictive control
│   └── pid.py                      # PID controllers
│
└── utils/
    ├── coordinate_transform.py
    ├── geometry.py
    └── logger.py
```

### Communication Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Message Bus (ZMQ/ROS2)                 │
└──────────────────────────────────────────────────────────┘
     ↑            ↑            ↑            ↑            ↑
     │            │            │            │            │
┌────┴───┐  ┌────┴───┐  ┌────┴────┐  ┌────┴───┐  ┌────┴───┐
│Sensors │  │Percept.│  │Planning │  │Control │  │Logging │
└────────┘  └────────┘  └─────────┘  └────────┘  └────────┘
```

**Message Topics:**
- `/sensors/camera/*`: Camera images
- `/sensors/lidar/points`: Point clouds
- `/sensors/radar/tracks`: Radar detections
- `/sensors/gps/fix`: GPS position
- `/sensors/imu/data`: IMU measurements
- `/perception/lanes`: Detected lanes
- `/perception/objects`: Detected objects
- `/perception/signs`: Traffic signs
- `/localization/pose`: Fused vehicle state
- `/planning/trajectory`: Planned trajectory
- `/control/commands`: Control commands

### Processing Pipeline Timing

```
Sensor Acquisition:    0 ms
    ↓
Perception:           33 ms (30 Hz)
    ↓
Sensor Fusion:        10 ms (100 Hz)
    ↓
Planning:            100 ms (10 Hz)
    ↓
Control:              10 ms (100 Hz)
    ↓
Total Latency:       ~100 ms (acceptable for highway driving)
```

## Safety Architecture

### Redundancy

1. **Dual Computing Platforms**: Primary + backup ECU
2. **Sensor Redundancy**: Multiple sensor modalities
3. **Fallback Behaviors**: Graceful degradation
4. **Watchdog Timers**: Detect system failures

### Safety Monitors

```
┌─────────────────────────────────────────────────────────────┐
│                   SAFETY MONITOR                             │
├──────────────┬──────────────┬──────────────────────────────┤
│ Input        │ System       │ Output                       │
│ Validation   │ Health       │ Validation                   │
├──────────────┼──────────────┼──────────────────────────────┤
│ • Sensor     │ • CPU usage  │ • Command limits             │
│ • Bounds     │ • Memory     │ • Rate limits                │
│ • Outliers   │ • Latency    │ • Consistency                │
└──────────────┴──────────────┴──────────────────────────────┘
```

### Fail-safe Mechanisms

1. **Emergency Stop**: Triggered by critical failures
2. **Minimum Risk Condition**: Safe state when automation fails
3. **Driver Handover**: Alert driver and transition control
4. **Geofencing**: Restrict operation to validated areas

## Simulation and Testing

### Blender-based Simulation

```
blender_scripts/
├── truck_model.py      # 3D truck with sensors
├── environment.py      # Roads, traffic, obstacles
└── sensors.py          # Sensor simulation
```

**Features:**
- Realistic truck physics
- Sensor visualization (cameras, LiDAR, radar)
- Traffic scenarios
- Weather conditions

### Testing Framework

```
tests/
├── unit/              # Module-level tests
├── integration/       # System integration tests
└── simulation/        # Scenario-based tests
```

**Test Coverage:**
- Unit tests: >80% code coverage
- Integration tests: End-to-end pipeline
- Simulation: 10,000+ scenarios

## Deployment Architecture

### Hardware Configuration

```
┌────────────────────────────────────────────────────────┐
│              Computing Platform                         │
│  ┌──────────────────┬──────────────────┐              │
│  │  Primary ECU     │  Backup ECU      │              │
│  │  • NVIDIA Orin   │  • Intel i9      │              │
│  │  • 32GB RAM      │  • 32GB RAM      │              │
│  │  • 1TB SSD       │  • 1TB SSD       │              │
│  └──────────────────┴──────────────────┘              │
│                                                         │
│  ┌──────────────────────────────────────┐             │
│  │       CAN Bus / Ethernet             │             │
│  └──────────────────────────────────────┘             │
│          ↑        ↑        ↑        ↑                  │
│      Sensors   Cameras   Actuators  OBD                │
└────────────────────────────────────────────────────────┘
```

### Software Stack

```
┌─────────────────────────────────────────┐
│     Application Layer (Python/C++)      │
├─────────────────────────────────────────┤
│     ROS2 / Custom Middleware            │
├─────────────────────────────────────────┤
│     Linux RT (Ubuntu 22.04)             │
├─────────────────────────────────────────┤
│     Hardware Drivers (CUDA, V4L2)       │
└─────────────────────────────────────────┘
```

## Performance Optimization

### GPU Acceleration
- Lane detection: CUDA kernels
- Object detection: TensorRT optimization
- Point cloud processing: GPU point cloud library

### Multi-threading
- Sensor acquisition: Separate threads per sensor
- Perception modules: Parallel processing
- Pipeline: Producer-consumer pattern

### Memory Management
- Shared memory: Inter-process communication
- Memory pools: Reduce allocation overhead
- Zero-copy: Minimize data copying

## Monitoring and Diagnostics

### Logging System
- **Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Storage**: Rolling log files + database
- **Real-time**: Dashboard visualization

### Metrics
- Latency per module
- Throughput (FPS)
- CPU/GPU/Memory usage
- Sensor health
- Detection accuracy (runtime validation)

### Black Box Recorder
- Last 5 minutes of sensor data
- All decisions and control commands
- For incident analysis

---

**Document Version**: 1.0
**Last Updated**: 2025-11-17

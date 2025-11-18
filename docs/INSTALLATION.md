# Installation Guide

Complete guide for setting up the Self-Driving Truck system on your development machine or deployment hardware.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Software Prerequisites](#software-prerequisites)
3. [Installation Steps](#installation-steps)
4. [Hardware Setup](#hardware-setup)
5. [Configuration](#configuration)
6. [Verification](#verification)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Development Environment

**Minimum:**
- CPU: Intel Core i7 or AMD Ryzen 7
- RAM: 16GB
- GPU: NVIDIA GTX 1660 Ti (6GB VRAM)
- Storage: 256GB SSD
- OS: Ubuntu 20.04 or 22.04 LTS

**Recommended:**
- CPU: Intel Core i9 or AMD Ryzen 9
- RAM: 32GB
- GPU: NVIDIA RTX 3080 or better (10GB+ VRAM)
- Storage: 1TB NVMe SSD
- OS: Ubuntu 22.04 LTS

### Deployment Hardware

**Computing Platform:**
- NVIDIA Jetson AGX Orin (64GB) or
- Industrial PC with RTX 4090 GPU
- 32GB RAM minimum
- 1TB SSD for data storage
- Dual Ethernet ports
- CAN bus interface

**Sensors:**
- 6x cameras (1920x1080 @ 30 FPS minimum)
- 1x 64-channel LiDAR (Velodyne HDL-64E or equivalent)
- 4x long-range radar (Continental ARS4xx series)
- RTK GPS receiver
- 9-DOF IMU
- Wheel speed sensors

## Software Prerequisites

### 1. Operating System

Install Ubuntu 22.04 LTS:
```bash
# Download from: https://ubuntu.com/download/desktop
# Create bootable USB and install
```

### 2. System Updates

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential cmake git wget curl
```

### 3. Python 3.10+

```bash
# Check Python version
python3 --version  # Should be 3.10 or higher

# Install Python development tools
sudo apt install -y python3-pip python3-dev python3-venv
```

### 4. NVIDIA Drivers and CUDA

```bash
# Check if NVIDIA GPU is detected
lspci | grep -i nvidia

# Install NVIDIA drivers
sudo apt install -y nvidia-driver-535

# Reboot
sudo reboot

# Verify installation
nvidia-smi

# Install CUDA Toolkit 12.0
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify CUDA installation
nvcc --version
```

### 5. cuDNN

```bash
# Download cuDNN from NVIDIA website (requires login)
# https://developer.nvidia.com/cudnn

# Install cuDNN
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-8.9.0.131/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install -y libcudnn8 libcudnn8-dev
```

### 6. Additional Dependencies

```bash
# OpenCV dependencies
sudo apt install -y libopencv-dev python3-opencv

# Blender dependencies (for 3D modeling)
sudo apt install -y blender

# PCL (Point Cloud Library) for LiDAR
sudo apt install -y libpcl-dev

# Video codec libraries
sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libswscale-dev

# Visualization
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

## Installation Steps

### 1. Clone Repository

```bash
cd ~/
git clone https://github.com/Abrar5510/Truck.git
cd Truck
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install Python Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### 4. Download Pre-trained Models

```bash
# Create models directory
mkdir -p data/models

# Download pre-trained weights (when available)
# python scripts/download_models.py
```

### 5. Install ROS2 (Optional, for robot integration)

```bash
# Add ROS2 apt repository
sudo apt install -y software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# Add repository to sources list
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble
sudo apt update
sudo apt install -y ros-humble-desktop

# Source ROS2
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 6. Build C++ Extensions (if any)

```bash
# Navigate to C++ directory
cd src/cpp_extensions

# Build with CMake
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

cd ../../..
```

## Hardware Setup

### 1. Sensor Installation

#### Cameras
- **Front Center**: Mount on windshield, centered, pointing forward
- **Front Left/Right**: Mount on front corners at 30° angle
- **Side Left/Right**: Mount on mirrors or side panels
- **Rear Center**: Mount on rear panel

**Camera Calibration:**
```bash
# Run camera calibration tool
python tools/calibrate_cameras.py --checkerboard 9x6 --square-size 0.025
```

#### LiDAR
- Mount on roof, centered, 3.6m above ground
- Ensure 360° unobstructed view
- Connect via Ethernet

**LiDAR Setup:**
```bash
# Configure LiDAR IP address
sudo ip addr add 192.168.1.10/24 dev eth0

# Test connection
ping 192.168.1.201  # Default Velodyne IP

# Visualize LiDAR data
python tools/visualize_lidar.py
```

#### Radar
- Mount radars on front, rear, and sides
- Angle according to specifications
- Connect via CAN bus

#### GPS/IMU
- Mount GPS antenna on roof with clear sky view
- Mount IMU near vehicle center of mass
- Connect via USB or serial

### 2. Computing Platform Setup

```bash
# Install real-time kernel (for low latency)
sudo apt install -y linux-lowlatency
sudo reboot

# Configure CPU governor for performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable unnecessary services
sudo systemctl disable bluetooth
sudo systemctl disable cups
```

### 3. CAN Bus Setup

```bash
# Install CAN utilities
sudo apt install -y can-utils

# Configure CAN interface
sudo ip link set can0 type can bitrate 500000
sudo ip link set up can0

# Test CAN bus
candump can0

# Send test message
cansend can0 123#DEADBEEF
```

## Configuration

### 1. Vehicle Configuration

Edit `config/vehicle_config.yaml`:
```yaml
vehicle:
  dimensions:
    cab:
      length: 7.0  # Your truck length in meters
      width: 2.5
      wheelbase: 4.5
```

### 2. Sensor Configuration

Edit `config/camera_config.yaml`:
```yaml
cameras:
  front_center:
    device: "/dev/video0"
    resolution: [1920, 1080]
    fps: 30
    calibration_file: "data/calibration/front_center.yaml"
```

### 3. Model Configuration

Edit `config/model_config.yaml`:
```yaml
models:
  lane_detection:
    model_path: "data/models/lane_detector.pth"
    confidence_threshold: 0.5

  object_detection:
    model_path: "data/models/yolov8.pth"
    confidence_threshold: 0.5
    nms_threshold: 0.45
```

## Verification

### 1. Test Individual Modules

```bash
# Activate virtual environment
source venv/bin/activate

# Test lane detection
python -m src.computer_vision.models.lane_detection

# Test object detection
python -m src.computer_vision.models.object_detection

# Test sensor fusion
python -m src.sensor_fusion.ekf

# Test path planning
python -m src.planning.path_planner

# Test control
python -m src.control.controller
```

### 2. Run Unit Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_lane_detection.py
```

### 3. Sensor Health Check

```bash
# Check all sensors
python tools/sensor_check.py

# Expected output:
# ✓ Front camera: OK
# ✓ LiDAR: OK
# ✓ GPS: OK (Fix quality: RTK)
# ✓ IMU: OK
```

### 4. End-to-End Test

```bash
# Run complete pipeline on test data
python scripts/run_pipeline.py --input data/test_videos/highway.mp4 --visualize

# Or run in simulation
python scripts/run_simulation.py --scenario highway_cruise
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution:**
```python
# Reduce batch size in config
# Use gradient checkpointing
# Enable mixed precision training
```

### Issue: Camera Not Detected

```bash
# Check camera devices
ls /dev/video*

# Check camera info
v4l2-ctl --list-devices

# Test camera
ffplay /dev/video0
```

### Issue: LiDAR Not Responding

```bash
# Check network connection
sudo tcpdump -i eth0 port 2368

# Verify IP configuration
ifconfig eth0

# Restart LiDAR
sudo systemctl restart velodyne
```

### Issue: High Latency

```bash
# Check CPU frequency
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq

# Monitor process
top -H

# Use real-time priorities
sudo nice -n -20 python main.py
```

### Issue: Import Errors

```bash
# Verify installation
pip list | grep torch
pip list | grep opencv

# Reinstall package
pip install --force-reinstall opencv-python
```

## Performance Tuning

### 1. GPU Optimization

```bash
# Set GPU to maximum performance
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 350  # Set power limit to 350W

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### 2. CPU Optimization

```bash
# Use all cores
export OMP_NUM_THREADS=$(nproc)

# Pin threads to cores
taskset -c 0-15 python main.py
```

### 3. Memory Optimization

```bash
# Increase swap space
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## Next Steps

After successful installation:

1. **Calibrate Sensors**: Run calibration procedures for all sensors
2. **Collect Data**: Gather training data for your specific environment
3. **Train Models**: Fine-tune pre-trained models on your data
4. **Validate**: Test in simulation before on-road deployment
5. **Safety Review**: Complete safety validation checklist

## Support

For issues and questions:
- GitHub Issues: https://github.com/Abrar5510/Truck/issues
- Documentation: See `docs/` folder
- API Reference: See `docs/API.md`

---

**Last Updated**: 2025-11-17
**Version**: 1.0

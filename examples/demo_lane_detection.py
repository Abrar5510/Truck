#!/usr/bin/env python3
"""
Demo: Lane Detection

Simple demonstration of the lane detection module.

Usage:
    python examples/demo_lane_detection.py
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.computer_vision.models.lane_detection import LaneDetector, LaneConfig


def create_demo_image():
    """Create a synthetic road image with lanes"""
    height, width = 720, 1280
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Road surface (dark gray)
    image[:, :] = (60, 60, 60)

    # Draw lane markings
    lane_positions = [width//4, width//2, 3*width//4]

    for x_pos in lane_positions:
        # Perspective effect - lanes converge at horizon
        horizon_y = height // 3
        bottom_y = height

        # Calculate lane width at different y positions
        for y in range(horizon_y, bottom_y, 20):
            # Perspective scaling
            scale = (y - horizon_y) / (bottom_y - horizon_y)
            x_offset = int(x_pos + (width//2 - x_pos) * (1 - scale))

            # Draw dashed line
            if y % 40 < 20:  # Dashed pattern
                cv2.line(
                    image,
                    (x_offset, y),
                    (x_offset, min(y + 15, bottom_y)),
                    (255, 255, 255),
                    3
                )

    return image


def main():
    print("=" * 60)
    print("Lane Detection Demo")
    print("=" * 60)

    # Create lane detector
    config = LaneConfig()
    detector = LaneDetector(config=config)
    print(f"\n✓ Lane detector initialized")
    print(f"  - Input size: {config.image_width}x{config.image_height}")
    print(f"  - Backbone: {config.backbone}")
    print(f"  - Confidence threshold: {config.confidence_threshold}")

    # Create demo image
    print(f"\n✓ Creating demo road image...")
    image = create_demo_image()

    # Detect lanes
    print(f"\n✓ Running lane detection...")
    lanes = detector.detect(image)

    print(f"\n✓ Detection complete!")
    print(f"  - Detected {len(lanes)} lane(s)")

    for i, lane in enumerate(lanes):
        print(f"\n  Lane {i+1}:")
        print(f"    Type: {lane.lane_type}")
        print(f"    Confidence: {lane.confidence:.3f}")
        print(f"    Points: {len(lane.points)}")
        print(f"    Polynomial: {lane.polynomial}")

    # Calculate steering angle
    steering_angle = detector.get_steering_angle(lanes)
    print(f"\n✓ Steering angle: {np.degrees(steering_angle):.2f}°")

    # Visualize
    print(f"\n✓ Generating visualization...")
    result_image = detector.visualize(image, lanes, show_points=True)

    # Save result
    output_path = Path(__file__).parent / "lane_detection_demo.jpg"
    cv2.imwrite(str(output_path), result_image)
    print(f"\n✓ Result saved to: {output_path}")

    # Display (if display available)
    try:
        cv2.imshow("Lane Detection Demo", result_image)
        print(f"\n✓ Displaying result (press any key to close)...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print(f"\n  (Display not available in this environment)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

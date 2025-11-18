"""
Lane Detection Module

This module implements a deep learning-based lane detection system using
a U-Net architecture with ResNet50 backbone for semantic segmentation.

Author: Self-Driving Truck Project
Date: 2025-11-17
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class LaneConfig:
    """Configuration for lane detection model"""
    image_height: int = 360
    image_width: int = 640
    num_classes: int = 2  # Background and lane
    backbone: str = 'resnet50'
    pretrained: bool = True
    confidence_threshold: float = 0.5
    poly_degree: int = 3  # Polynomial degree for lane fitting


class ResNetEncoder(nn.Module):
    """ResNet50 encoder for feature extraction"""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet50(pretrained=pretrained)

        # Extract encoder layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

    def forward(self, x):
        """Forward pass returning multi-scale features"""
        x0 = self.relu(self.bn1(self.conv1(x)))
        x1 = self.layer1(self.maxpool(x0))  # 1/4
        x2 = self.layer2(x1)  # 1/8
        x3 = self.layer3(x2)  # 1/16
        x4 = self.layer4(x3)  # 1/32

        return x1, x2, x3, x4


class DecoderBlock(nn.Module):
    """Decoder block with skip connections"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )

        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels,
            kernel_size=3, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        """Forward pass with skip connection"""
        x = self.upsample(x)

        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        return x


class LaneDetectionModel(nn.Module):
    """U-Net architecture for lane segmentation"""

    def __init__(self, config: LaneConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = ResNetEncoder(pretrained=config.pretrained)

        # Decoder
        self.decoder4 = DecoderBlock(2048, 1024, 512)
        self.decoder3 = DecoderBlock(512, 512, 256)
        self.decoder2 = DecoderBlock(256, 256, 128)
        self.decoder1 = DecoderBlock(128, 0, 64)

        # Final classification head
        self.final_conv = nn.Conv2d(64, config.num_classes, kernel_size=1)

        # Upsampling to original size
        self.final_upsample = nn.Upsample(
            size=(config.image_height, config.image_width),
            mode='bilinear',
            align_corners=True
        )

    def forward(self, x):
        """Forward pass"""
        # Encode
        x1, x2, x3, x4 = self.encoder(x)

        # Decode with skip connections
        d4 = self.decoder4(x4, x3)
        d3 = self.decoder3(d4, x2)
        d2 = self.decoder2(d3, x1)
        d1 = self.decoder1(d2, None)

        # Final output
        out = self.final_conv(d1)
        out = self.final_upsample(out)

        return out


@dataclass
class Lane:
    """Represents a detected lane"""
    points: np.ndarray  # (N, 2) array of (x, y) points
    polynomial: np.ndarray  # Polynomial coefficients
    confidence: float
    lane_type: str  # 'left', 'right', 'center'


class LaneDetector:
    """
    High-level lane detection interface

    Example:
        detector = LaneDetector(model_path='lane_model.pth')
        lanes = detector.detect(image)
        result_img = detector.visualize(image, lanes)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[LaneConfig] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.config = config or LaneConfig()
        self.device = device

        # Initialize model
        self.model = LaneDetectionModel(self.config).to(device)

        # Load weights if provided
        if model_path:
            self.load_weights(model_path)

        self.model.eval()

        # Image normalization parameters (ImageNet stats)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    def load_weights(self, path: str):
        """Load pre-trained model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded model weights from {path}")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Resize
        img = cv2.resize(
            image,
            (self.config.image_width, self.config.image_height)
        )

        # Convert to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # To tensor and normalize
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.to(self.device)
        img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor.unsqueeze(0)

    def postprocess(self, segmentation: torch.Tensor) -> List[Lane]:
        """
        Extract lane polynomials from segmentation mask

        Args:
            segmentation: (1, H, W) tensor with lane probabilities

        Returns:
            List of Lane objects
        """
        # Convert to numpy
        mask = segmentation.squeeze().cpu().numpy()
        mask = (mask > self.config.confidence_threshold).astype(np.uint8) * 255

        lanes = []

        # Find connected components (each lane)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        for label in range(1, num_labels):  # Skip background (0)
            # Get points for this lane
            lane_mask = (labels == label).astype(np.uint8)

            # Filter small components
            if stats[label, cv2.CC_STAT_AREA] < 100:
                continue

            # Extract lane points using sliding window
            lane_points = self._extract_lane_points(lane_mask)

            if len(lane_points) < 10:
                continue

            # Fit polynomial
            try:
                x = lane_points[:, 0]
                y = lane_points[:, 1]

                # Fit polynomial: x = f(y) for better vertical stability
                poly = np.polyfit(y, x, self.config.poly_degree)

                # Determine lane type based on position
                avg_x = np.mean(x)
                if avg_x < self.config.image_width * 0.4:
                    lane_type = 'left'
                elif avg_x > self.config.image_width * 0.6:
                    lane_type = 'right'
                else:
                    lane_type = 'center'

                # Calculate confidence based on fit quality
                x_fit = np.polyval(poly, y)
                residual = np.mean((x - x_fit) ** 2)
                confidence = np.exp(-residual / 100)

                lanes.append(Lane(
                    points=lane_points,
                    polynomial=poly,
                    confidence=confidence,
                    lane_type=lane_type
                ))

            except np.linalg.LinAlgError:
                continue

        return lanes

    def _extract_lane_points(self, lane_mask: np.ndarray) -> np.ndarray:
        """Extract lane center points using sliding window"""
        height, width = lane_mask.shape
        window_height = 20
        num_windows = height // window_height

        points = []

        for i in range(num_windows):
            y_start = height - (i + 1) * window_height
            y_end = height - i * window_height

            window = lane_mask[y_start:y_end, :]

            # Find x positions where lane exists
            x_positions = np.where(window.sum(axis=0) > 0)[0]

            if len(x_positions) > 0:
                x_center = int(np.mean(x_positions))
                y_center = (y_start + y_end) // 2
                points.append([x_center, y_center])

        return np.array(points)

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[Lane]:
        """
        Detect lanes in image

        Args:
            image: Input image (H, W, 3) in BGR format

        Returns:
            List of detected Lane objects
        """
        # Preprocess
        img_tensor = self.preprocess(image)

        # Inference
        output = self.model(img_tensor)
        segmentation = torch.softmax(output, dim=1)[:, 1, :, :]  # Lane class

        # Postprocess
        lanes = self.postprocess(segmentation)

        return lanes

    def visualize(
        self,
        image: np.ndarray,
        lanes: List[Lane],
        show_points: bool = False
    ) -> np.ndarray:
        """
        Visualize detected lanes on image

        Args:
            image: Original image
            lanes: Detected lanes
            show_points: Whether to show individual lane points

        Returns:
            Image with lanes drawn
        """
        result = image.copy()
        height, width = image.shape[:2]

        # Scale factors for original image size
        scale_x = width / self.config.image_width
        scale_y = height / self.config.image_height

        colors = {
            'left': (255, 0, 0),    # Blue
            'right': (0, 255, 0),    # Green
            'center': (0, 255, 255)  # Yellow
        }

        for lane in lanes:
            color = colors.get(lane.lane_type, (255, 255, 255))

            # Draw polynomial curve
            y_vals = np.linspace(0, self.config.image_height - 1, 100)
            x_vals = np.polyval(lane.polynomial, y_vals)

            # Scale to original image size
            pts = np.column_stack([x_vals * scale_x, y_vals * scale_y]).astype(np.int32)

            # Filter valid points
            valid = (pts[:, 0] >= 0) & (pts[:, 0] < width) & \
                    (pts[:, 1] >= 0) & (pts[:, 1] < height)
            pts = pts[valid]

            # Draw curve
            if len(pts) > 1:
                cv2.polylines(result, [pts], False, color, 3)

            # Draw points if requested
            if show_points:
                scaled_points = (lane.points * [scale_x, scale_y]).astype(np.int32)
                for pt in scaled_points:
                    cv2.circle(result, tuple(pt), 3, color, -1)

            # Draw lane type label
            if len(pts) > 0:
                label_pos = tuple(pts[0])
                cv2.putText(
                    result,
                    f"{lane.lane_type} ({lane.confidence:.2f})",
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        return result

    def get_steering_angle(self, lanes: List[Lane]) -> float:
        """
        Calculate steering angle based on detected lanes

        Args:
            lanes: Detected lanes

        Returns:
            Steering angle in radians (negative = left, positive = right)
        """
        # Find left and right lanes
        left_lane = None
        right_lane = None

        for lane in lanes:
            if lane.lane_type == 'left' and (left_lane is None or lane.confidence > left_lane.confidence):
                left_lane = lane
            elif lane.lane_type == 'right' and (right_lane is None or lane.confidence > right_lane.confidence):
                right_lane = lane

        if left_lane is None and right_lane is None:
            return 0.0  # No lanes detected

        # Calculate desired position (center of lane)
        image_center = self.config.image_width / 2
        horizon_y = self.config.image_height * 0.7  # Look ahead point

        if left_lane and right_lane:
            # Both lanes visible
            left_x = np.polyval(left_lane.polynomial, horizon_y)
            right_x = np.polyval(right_lane.polynomial, horizon_y)
            lane_center = (left_x + right_x) / 2
        elif left_lane:
            # Only left lane
            left_x = np.polyval(left_lane.polynomial, horizon_y)
            lane_center = left_x + 1.8 / 3.5 * self.config.image_width  # Assume lane width
        else:
            # Only right lane
            right_x = np.polyval(right_lane.polynomial, horizon_y)
            lane_center = right_x - 1.8 / 3.5 * self.config.image_width

        # Calculate error and steering angle
        error = lane_center - image_center
        steering_angle = np.arctan(error / (self.config.image_height * 0.5))

        return steering_angle


if __name__ == "__main__":
    # Example usage
    print("Lane Detection Module")
    print("=" * 50)

    # Create detector
    config = LaneConfig()
    detector = LaneDetector(config=config)

    print(f"Model initialized on {detector.device}")
    print(f"Input size: {config.image_width}x{config.image_height}")
    print(f"Ready for lane detection!")

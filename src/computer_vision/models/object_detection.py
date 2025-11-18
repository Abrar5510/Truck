"""
Object Detection Module

YOLOv8-based real-time object detection for autonomous driving.
Detects vehicles, pedestrians, cyclists, traffic lights, and obstacles.

Author: Self-Driving Truck Project
Date: 2025-11-17
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ObjectClass(Enum):
    """Object classes for detection"""
    CAR = 0
    TRUCK = 1
    BUS = 2
    MOTORCYCLE = 3
    BICYCLE = 4
    PEDESTRIAN = 5
    TRAFFIC_LIGHT_RED = 6
    TRAFFIC_LIGHT_YELLOW = 7
    TRAFFIC_LIGHT_GREEN = 8
    TRAFFIC_SIGN = 9
    OBSTACLE = 10


@dataclass
class Detection:
    """Represents a detected object"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    distance: Optional[float] = None  # Estimated distance in meters
    velocity: Optional[float] = None  # Relative velocity in m/s


@dataclass
class DetectionConfig:
    """Configuration for object detection"""
    image_size: int = 640
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.45
    max_detections: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Camera parameters for distance estimation
    focal_length: float = 1000.0  # pixels
    sensor_height: float = 0.006  # meters (typical)

    # Known object heights for distance estimation (meters)
    object_heights: Dict[int, float] = None

    def __post_init__(self):
        if self.object_heights is None:
            self.object_heights = {
                ObjectClass.CAR.value: 1.5,
                ObjectClass.TRUCK.value: 3.0,
                ObjectClass.BUS.value: 3.2,
                ObjectClass.MOTORCYCLE.value: 1.2,
                ObjectClass.BICYCLE.value: 1.5,
                ObjectClass.PEDESTRIAN.value: 1.7,
                ObjectClass.TRAFFIC_LIGHT_RED.value: 0.3,
                ObjectClass.TRAFFIC_LIGHT_YELLOW.value: 0.3,
                ObjectClass.TRAFFIC_LIGHT_GREEN.value: 0.3,
                ObjectClass.TRAFFIC_SIGN.value: 0.6,
            }


class Conv(nn.Module):
    """Standard convolution with batch norm and activation"""

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block"""

    def __init__(self, in_channels, out_channels, shortcut=True, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels, out_channels, 3, 1, 1, groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions"""

    def __init__(self, in_channels, out_channels, num_blocks=1, shortcut=False, groups=1, expansion=0.5):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = Conv(in_channels, 2 * hidden_channels, 1, 1)
        self.conv2 = Conv((2 + num_blocks) * hidden_channels, out_channels, 1)
        self.bottlenecks = nn.ModuleList(
            Bottleneck(hidden_channels, hidden_channels, shortcut, groups, 1.0) for _ in range(num_blocks)
        )

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.bottlenecks)
        return self.conv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = Conv(in_channels, hidden_channels, 1, 1)
        self.conv2 = Conv(hidden_channels * 4, out_channels, 1, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        return self.conv2(torch.cat([x, y1, y2, self.maxpool(y2)], 1))


class DetectionHead(nn.Module):
    """YOLOv8 detection head"""

    def __init__(self, num_classes=11, in_channels=(256, 512, 1024)):
        super().__init__()
        self.num_classes = num_classes
        self.num_outputs = num_classes + 4  # classes + bbox

        self.stems = nn.ModuleList(
            Conv(in_ch, in_ch, 3, 1, 1) for in_ch in in_channels
        )

        self.cls_convs = nn.ModuleList(
            nn.Sequential(
                Conv(in_ch, in_ch, 3, 1, 1),
                Conv(in_ch, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, num_classes, 1)
            ) for in_ch in in_channels
        )

        self.box_convs = nn.ModuleList(
            nn.Sequential(
                Conv(in_ch, in_ch, 3, 1, 1),
                Conv(in_ch, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, 4, 1)
            ) for in_ch in in_channels
        )

    def forward(self, features):
        """Forward pass"""
        outputs = []
        for i, feat in enumerate(features):
            x = self.stems[i](feat)
            cls_output = self.cls_convs[i](x)
            box_output = self.box_convs[i](x)
            outputs.append(torch.cat([box_output, cls_output], 1))
        return outputs


class YOLOv8(nn.Module):
    """YOLOv8 Model for Object Detection"""

    def __init__(self, num_classes=11):
        super().__init__()
        self.num_classes = num_classes

        # Backbone
        self.conv1 = Conv(3, 64, 3, 2, 1)
        self.conv2 = Conv(64, 128, 3, 2, 1)
        self.c2f1 = C2f(128, 128, 3)

        self.conv3 = Conv(128, 256, 3, 2, 1)
        self.c2f2 = C2f(256, 256, 6)

        self.conv4 = Conv(256, 512, 3, 2, 1)
        self.c2f3 = C2f(512, 512, 6)

        self.conv5 = Conv(512, 1024, 3, 2, 1)
        self.c2f4 = C2f(1024, 1024, 3)
        self.sppf = SPPF(1024, 1024, 5)

        # Neck (FPN)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.c2f5 = C2f(1024 + 512, 512, 3)
        self.c2f6 = C2f(512 + 256, 256, 3)

        self.conv6 = Conv(256, 256, 3, 2, 1)
        self.c2f7 = C2f(256 + 512, 512, 3)

        self.conv7 = Conv(512, 512, 3, 2, 1)
        self.c2f8 = C2f(512 + 1024, 1024, 3)

        # Detection head
        self.head = DetectionHead(num_classes, (256, 512, 1024))

    def forward(self, x):
        """Forward pass"""
        # Backbone
        x = self.conv1(x)
        x = self.c2f1(self.conv2(x))

        x1 = self.c2f2(self.conv3(x))  # P3
        x2 = self.c2f3(self.conv4(x1))  # P4
        x3 = self.sppf(self.c2f4(self.conv5(x2)))  # P5

        # Neck - Top-down
        p5 = x3
        p4 = self.c2f5(torch.cat([self.upsample(p5), x2], 1))
        p3 = self.c2f6(torch.cat([self.upsample(p4), x1], 1))

        # Bottom-up
        p4 = self.c2f7(torch.cat([self.conv6(p3), p4], 1))
        p5 = self.c2f8(torch.cat([self.conv7(p4), p5], 1))

        # Detection head
        outputs = self.head([p3, p4, p5])

        return outputs


class ObjectDetector:
    """
    High-level object detection interface

    Example:
        detector = ObjectDetector(model_path='yolov8.pth')
        detections = detector.detect(image)
        result_img = detector.visualize(image, detections)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[DetectionConfig] = None
    ):
        self.config = config or DetectionConfig()

        # Initialize model
        self.model = YOLOv8(num_classes=len(ObjectClass)).to(self.config.device)

        if model_path:
            self.load_weights(model_path)

        self.model.eval()

        # Class names
        self.class_names = [cls.name for cls in ObjectClass]

        # Colors for visualization (BGR)
        self.colors = self._generate_colors(len(ObjectClass))

    def load_weights(self, path: str):
        """Load pre-trained weights"""
        checkpoint = torch.load(path, map_location=self.config.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded model weights from {path}")

    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each class"""
        np.random.seed(42)
        return [(int(c[0]), int(c[1]), int(c[2]))
                for c in np.random.randint(0, 255, size=(num_classes, 3))]

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, float]:
        """Preprocess image for detection"""
        height, width = image.shape[:2]

        # Calculate scale
        scale = self.config.image_size / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)

        # Resize
        resized = cv2.resize(image, (new_width, new_height))

        # Pad to square
        padded = np.full((self.config.image_size, self.config.image_size, 3), 114, dtype=np.uint8)
        padded[:new_height, :new_width] = resized

        # Convert to tensor
        img_tensor = torch.from_numpy(padded).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.config.device)

        return img_tensor, scale

    def non_max_suppression(self, predictions: torch.Tensor) -> List[Detection]:
        """Apply NMS to predictions"""
        detections = []

        for pred in predictions:
            # pred shape: [num_anchors, 4 + num_classes]
            boxes = pred[:, :4]
            scores = pred[:, 4:].max(1)
            class_ids = pred[:, 4:].argmax(1)

            # Filter by confidence
            mask = scores.values > self.config.confidence_threshold
            boxes = boxes[mask]
            scores = scores.values[mask]
            class_ids = class_ids[mask]

            if len(boxes) == 0:
                continue

            # Convert to x1, y1, x2, y2
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            # NMS
            keep = torchvision.ops.nms(boxes, scores, self.config.nms_threshold)

            for idx in keep[:self.config.max_detections]:
                detections.append(Detection(
                    bbox=boxes[idx].cpu().numpy(),
                    confidence=float(scores[idx]),
                    class_id=int(class_ids[idx]),
                    class_name=self.class_names[int(class_ids[idx])]
                ))

        return detections

    def estimate_distance(self, detection: Detection, image_height: int) -> float:
        """
        Estimate distance to object using pinhole camera model

        Distance = (Real_Height * Focal_Length) / Pixel_Height
        """
        bbox_height = detection.bbox[3] - detection.bbox[1]

        if detection.class_id in self.config.object_heights:
            real_height = self.config.object_heights[detection.class_id]
            distance = (real_height * self.config.focal_length) / bbox_height
            return distance

        return None

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        Detect objects in image

        Args:
            image: Input image in BGR format

        Returns:
            List of Detection objects
        """
        orig_height, orig_width = image.shape[:2]

        # Preprocess
        img_tensor, scale = self.preprocess(image)

        # Inference
        outputs = self.model(img_tensor)

        # Process outputs
        # Concatenate multi-scale predictions
        predictions = []
        for output in outputs:
            batch_size, _, height, width = output.shape
            output = output.view(batch_size, -1, height * width).permute(0, 2, 1)
            predictions.append(output)

        predictions = torch.cat(predictions, dim=1)

        # Apply NMS
        detections = self.non_max_suppression(predictions)

        # Scale boxes back to original image
        for det in detections:
            det.bbox = det.bbox / scale
            det.bbox[0] = np.clip(det.bbox[0], 0, orig_width)
            det.bbox[1] = np.clip(det.bbox[1], 0, orig_height)
            det.bbox[2] = np.clip(det.bbox[2], 0, orig_width)
            det.bbox[3] = np.clip(det.bbox[3], 0, orig_height)

            # Estimate distance
            det.distance = self.estimate_distance(det, orig_height)

        return detections

    def visualize(
        self,
        image: np.ndarray,
        detections: List[Detection],
        show_distance: bool = True
    ) -> np.ndarray:
        """
        Visualize detections on image

        Args:
            image: Original image
            detections: List of detections
            show_distance: Whether to show estimated distance

        Returns:
            Image with drawn detections
        """
        result = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)
            color = self.colors[det.class_id]

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            label = f"{det.class_name} {det.confidence:.2f}"
            if show_distance and det.distance is not None:
                label += f" {det.distance:.1f}m"

            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                result,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                result,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )

        return result

    def get_critical_objects(
        self,
        detections: List[Detection],
        distance_threshold: float = 50.0
    ) -> List[Detection]:
        """
        Filter detections for critical objects (close or high priority)

        Args:
            detections: All detections
            distance_threshold: Maximum distance to consider (meters)

        Returns:
            Filtered list of critical detections
        """
        critical = []

        priority_classes = [
            ObjectClass.PEDESTRIAN.value,
            ObjectClass.BICYCLE.value,
            ObjectClass.MOTORCYCLE.value,
            ObjectClass.TRAFFIC_LIGHT_RED.value
        ]

        for det in detections:
            # High priority classes are always critical
            if det.class_id in priority_classes:
                critical.append(det)
                continue

            # Close objects are critical
            if det.distance and det.distance < distance_threshold:
                critical.append(det)

        return critical


if __name__ == "__main__":
    # Example usage
    print("Object Detection Module - YOLOv8")
    print("=" * 50)

    config = DetectionConfig()
    detector = ObjectDetector(config=config)

    print(f"Model initialized on {config.device}")
    print(f"Number of classes: {len(ObjectClass)}")
    print(f"Confidence threshold: {config.confidence_threshold}")
    print(f"NMS threshold: {config.nms_threshold}")
    print("Ready for object detection!")

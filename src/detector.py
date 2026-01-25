"""
Object Detection Module using YOLOv8

This module handles detection of:
- Cars, trucks, buses
- Pedestrians
- Cyclists
- Traffic lights/signs

HOW YOLO WORKS (Simple Explanation):
====================================
1. Image is divided into a grid (e.g., 20x20)
2. Each cell predicts: "Is there an object centered here?"
3. If yes, predict: bounding box (x, y, width, height) + class + confidence
4. Many predictions are made, then filtered by confidence
5. Overlapping boxes for same object are merged (Non-Max Suppression)

Think of it like asking 400 people (grid cells) to each look at their 
small region and report what they see. Then we combine their answers.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time

# We'll use ultralytics YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")


class ObjectDetector:
    """
    YOLOv8-based object detector for autonomous driving.
    
    Attributes:
        model: The YOLOv8 model
        confidence_threshold: Minimum confidence to keep a detection
        classes_of_interest: Which object classes we care about
    
    Example:
        >>> detector = ObjectDetector()
        >>> frame = cv2.imread("road.jpg")
        >>> detections = detector.detect(frame)
        >>> print(f"Found {len(detections)} objects")
    """
    
    # COCO class names that are relevant for driving
    # YOLO is trained on COCO dataset with 80 classes
    DRIVING_CLASSES = {
        0: 'person',      # Pedestrians - CRITICAL
        1: 'bicycle',     # Cyclists
        2: 'car',         # Most common
        3: 'motorcycle',  # Two-wheelers
        5: 'bus',         # Large vehicles
        7: 'truck',       # Large vehicles
        9: 'traffic light',
        11: 'stop sign',
    }
    
    # Colors for each class (BGR format for OpenCV)
    CLASS_COLORS = {
        'person': (0, 0, 255),        # Red - high priority
        'bicycle': (255, 165, 0),     # Orange
        'car': (0, 255, 0),           # Green
        'motorcycle': (255, 165, 0),  # Orange
        'bus': (255, 255, 0),         # Cyan
        'truck': (255, 255, 0),       # Cyan
        'traffic light': (0, 255, 255), # Yellow
        'stop sign': (0, 0, 255),     # Red
    }
    
    def __init__(
        self, 
        model_path: str = "yolov8m.pt",  # medium model - good balance
        confidence_threshold: float = 0.5,
        device: str = "auto"  # "cuda", "cpu", or "auto"
    ):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to YOLO weights or model name (auto-downloads)
            confidence_threshold: Minimum confidence (0-1) to keep detection
            device: Where to run inference
        
        Model sizes (speed vs accuracy trade-off):
            - yolov8n.pt: Nano - fastest, least accurate
            - yolov8s.pt: Small - fast, decent accuracy  
            - yolov8m.pt: Medium - balanced (WE USE THIS)
            - yolov8l.pt: Large - slower, more accurate
            - yolov8x.pt: Extra-large - slowest, most accurate
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = device
        
        # Load the model
        if YOLO_AVAILABLE:
            print(f"Loading YOLOv8 model: {model_path}")
            self.model = YOLO(model_path)
            print(f"Model loaded successfully!")
        else:
            print("YOLOv8 not available - using mock detector for demo")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: BGR image (OpenCV format) as numpy array
        
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - class_name: What type of object
                - confidence: How sure we are (0-1)
                - class_id: Numeric class ID
        
        Example output:
            [
                {'bbox': [100, 200, 300, 400], 'class_name': 'car', 
                 'confidence': 0.92, 'class_id': 2},
                {'bbox': [500, 300, 550, 450], 'class_name': 'person', 
                 'confidence': 0.87, 'class_id': 0}
            ]
        """
        if self.model is None:
            # Return mock data for testing without model
            return self._mock_detect(frame)
        
        # Run inference
        # verbose=False suppresses YOLO's printing
        results = self.model(frame, verbose=False, conf=self.confidence_threshold)
        
        detections = []
        
        # Process results
        # results[0] contains detections for first (only) image
        for result in results:
            boxes = result.boxes  # Bounding boxes
            
            for box in boxes:
                # Get class ID and check if it's relevant for driving
                class_id = int(box.cls[0])
                
                if class_id not in self.DRIVING_CLASSES:
                    continue  # Skip irrelevant classes (like 'couch', 'cat', etc.)
                
                # Extract bounding box coordinates
                # box.xyxy gives [x1, y1, x2, y2] format
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence score
                confidence = float(box.conf[0])
                
                # Get class name
                class_name = self.DRIVING_CLASSES[class_id]
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class_name': class_name,
                    'confidence': confidence,
                    'class_id': class_id
                })
        
        return detections
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """
        Detect objects in multiple frames (more efficient than one-by-one).
        
        This is useful for video processing where we can batch frames.
        
        Args:
            frames: List of BGR images
        
        Returns:
            List of detection lists (one per frame)
        """
        all_detections = []
        
        if self.model is None:
            return [self._mock_detect(f) for f in frames]
        
        # YOLO can process batches natively
        results = self.model(frames, verbose=False, conf=self.confidence_threshold)
        
        for result in results:
            frame_detections = []
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                if class_id not in self.DRIVING_CLASSES:
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_name = self.DRIVING_CLASSES[class_id]
                
                frame_detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class_name': class_name,
                    'confidence': confidence,
                    'class_id': class_id
                })
            
            all_detections.append(frame_detections)
        
        return all_detections
    
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get the display color for a class."""
        return self.CLASS_COLORS.get(class_name, (128, 128, 128))
    
    def _mock_detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Mock detection for testing without model.
        Returns fake but realistic-looking detections.
        """
        h, w = frame.shape[:2]
        
        # Generate some fake detections in typical locations
        return [
            {
                'bbox': [int(w*0.3), int(h*0.4), int(w*0.5), int(h*0.7)],
                'class_name': 'car',
                'confidence': 0.92,
                'class_id': 2
            },
            {
                'bbox': [int(w*0.6), int(h*0.45), int(w*0.75), int(h*0.65)],
                'class_name': 'car', 
                'confidence': 0.87,
                'class_id': 2
            },
        ]


# Quick test when run directly
if __name__ == "__main__":
    print("Testing ObjectDetector...")
    
    # Create detector
    detector = ObjectDetector(confidence_threshold=0.5)
    
    # Create a test image (black image for demo)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Run detection
    start = time.time()
    detections = detector.detect(test_image)
    elapsed = time.time() - start
    
    print(f"\nDetection took: {elapsed*1000:.1f} ms")
    print(f"Found {len(detections)} objects:")
    for det in detections:
        print(f"  - {det['class_name']}: {det['confidence']:.2f}")

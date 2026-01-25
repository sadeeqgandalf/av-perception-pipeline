"""
Visualization Module

This module handles drawing detections on video frames:
- Bounding boxes around detected objects
- Lane line overlays
- Information dashboard
- Warning indicators

The goal is to create CLEAR, PROFESSIONAL visualizations
that demonstrate the system's capabilities to recruiters.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class Visualizer:
    """
    Draws detection results on video frames.
    
    Creates professional visualizations with:
    - Color-coded bounding boxes
    - Confidence scores
    - Lane overlays
    - Stats dashboard
    
    Example:
        >>> viz = Visualizer()
        >>> annotated = viz.draw_frame(frame, detections, lanes, metrics)
        >>> cv2.imshow("Result", annotated)
    """
    
    # Professional color scheme (BGR format)
    COLORS = {
        'primary': (255, 165, 0),     # Orange
        'success': (0, 255, 0),        # Green
        'warning': (0, 255, 255),      # Yellow
        'danger': (0, 0, 255),         # Red
        'info': (255, 255, 0),         # Cyan
        'lane_fill': (0, 100, 0),      # Dark green
        'text_bg': (30, 30, 30),       # Dark gray
    }
    
    # Object colors (match detector.py)
    OBJECT_COLORS = {
        'person': (0, 0, 255),         # Red - pedestrians are critical
        'bicycle': (0, 165, 255),      # Orange
        'car': (0, 255, 0),            # Green
        'motorcycle': (0, 165, 255),   # Orange
        'bus': (255, 255, 0),          # Cyan
        'truck': (255, 255, 0),        # Cyan
        'traffic light': (0, 255, 255), # Yellow
        'stop sign': (0, 0, 255),      # Red
    }
    
    def __init__(self, show_dashboard: bool = True):
        """
        Initialize visualizer.
        
        Args:
            show_dashboard: Whether to show stats panel
        """
        self.show_dashboard = show_dashboard
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        lane_data: Dict,
        metrics: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Draw all visualizations on a frame.
        
        Args:
            frame: Original BGR image
            detections: List of object detections
            lane_data: Lane detection results
            metrics: Performance metrics (fps, etc.)
        
        Returns:
            Annotated frame with all visualizations
        """
        # Create a copy so we don't modify original
        output = frame.copy()
        
        # 1. Draw lane overlay (first, so objects appear on top)
        output = self.draw_lane_overlay(output, lane_data)
        
        # 2. Draw object bounding boxes
        output = self.draw_detections(output, detections)
        
        # 3. Draw dashboard if enabled
        if self.show_dashboard and metrics:
            output = self.draw_dashboard(output, detections, lane_data, metrics)
        
        # 4. Draw warnings if needed
        output = self.draw_warnings(output, detections, lane_data)
        
        return output
    
    def draw_detections(
        self, 
        frame: np.ndarray, 
        detections: List[Dict]
    ) -> np.ndarray:
        """
        Draw bounding boxes around detected objects.
        
        Each box includes:
        - Colored rectangle
        - Class label
        - Confidence percentage
        """
        output = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Get color for this class
            color = self.OBJECT_COLORS.get(class_name, (128, 128, 128))
            
            # Draw bounding box
            thickness = 2 if class_name != 'person' else 3
            cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label = f"{class_name}: {confidence*100:.0f}%"
            
            # Calculate label position and background
            font_scale = 0.5
            font_thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(
                label, self.font, font_scale, font_thickness
            )
            
            # Draw label background
            label_y = max(y1 - 10, label_h + 10)
            cv2.rectangle(
                output,
                (x1, label_y - label_h - 5),
                (x1 + label_w + 5, label_y + 5),
                color,
                -1  # Filled
            )
            
            # Draw label text (white on colored background)
            cv2.putText(
                output, label,
                (x1 + 2, label_y),
                self.font, font_scale, (255, 255, 255), font_thickness
            )
            
            # Add distance indicator for close objects (simple heuristic)
            box_height = y2 - y1
            if box_height > frame.shape[0] * 0.3:  # Object takes >30% of frame height
                self._draw_distance_warning(output, x1, y2, "CLOSE")
        
        return output
    
    def draw_lane_overlay(
        self, 
        frame: np.ndarray, 
        lane_data: Dict
    ) -> np.ndarray:
        """
        Draw lane lines and fill the detected lane area.
        
        Creates a semi-transparent overlay showing:
        - Left lane line (green/blue)
        - Right lane line (green/blue)
        - Filled lane area (semi-transparent green)
        """
        output = frame.copy()
        
        left_points = lane_data.get('left_points', [])
        right_points = lane_data.get('right_points', [])
        
        # Draw lane fill if both lanes detected
        if left_points and right_points:
            # Create polygon points for fill
            # Left points top-to-bottom, right points bottom-to-top
            polygon_points = left_points + right_points[::-1]
            
            if len(polygon_points) >= 3:
                pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
                
                # Create overlay for transparency
                overlay = output.copy()
                cv2.fillPoly(overlay, [pts], self.COLORS['lane_fill'])
                
                # Blend with original (alpha = 0.3 for subtle effect)
                output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
        
        # Draw lane lines
        line_color = self.COLORS['success']  # Green
        
        if left_points and len(left_points) >= 2:
            pts = np.array(left_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [pts], False, line_color, 3)
        
        if right_points and len(right_points) >= 2:
            pts = np.array(right_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [pts], False, line_color, 3)
        
        # Draw center offset indicator
        offset = lane_data.get('lane_center_offset', 0)
        if abs(offset) > 50:  # Significant offset
            self._draw_offset_indicator(output, offset)
        
        return output
    
    def draw_dashboard(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        lane_data: Dict,
        metrics: Dict
    ) -> np.ndarray:
        """
        Draw a professional stats dashboard on the frame.
        
        Shows:
        - FPS (inference speed)
        - Object counts
        - Lane status
        - Confidence levels
        """
        output = frame.copy()
        height, width = output.shape[:2]
        
        # Dashboard dimensions
        dash_width = 220
        dash_height = 140
        margin = 10
        
        # Dashboard background (top-right corner)
        x1 = width - dash_width - margin
        y1 = margin
        x2 = width - margin
        y2 = y1 + dash_height
        
        # Draw semi-transparent background
        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLORS['text_bg'], -1)
        output = cv2.addWeighted(overlay, 0.7, output, 0.3, 0)
        
        # Draw border
        cv2.rectangle(output, (x1, y1), (x2, y2), self.COLORS['primary'], 2)
        
        # Draw title
        cv2.putText(
            output, "PERCEPTION STATUS",
            (x1 + 10, y1 + 25),
            self.font, 0.5, self.COLORS['primary'], 1
        )
        
        # Draw horizontal line under title
        cv2.line(output, (x1 + 5, y1 + 35), (x2 - 5, y1 + 35), self.COLORS['primary'], 1)
        
        # Stats line positions
        line_y = y1 + 55
        line_spacing = 22
        
        # FPS
        fps = metrics.get('fps', 0)
        fps_color = self.COLORS['success'] if fps >= 25 else self.COLORS['warning']
        cv2.putText(
            output, f"FPS: {fps:.1f}",
            (x1 + 10, line_y),
            self.font, 0.45, fps_color, 1
        )
        
        # Object count
        line_y += line_spacing
        obj_count = len(detections)
        cv2.putText(
            output, f"Objects: {obj_count}",
            (x1 + 10, line_y),
            self.font, 0.45, (255, 255, 255), 1
        )
        
        # Lane status
        line_y += line_spacing
        left_ok = lane_data.get('left_detected', False)
        right_ok = lane_data.get('right_detected', False)
        
        if left_ok and right_ok:
            lane_status = "BOTH"
            lane_color = self.COLORS['success']
        elif left_ok or right_ok:
            lane_status = "PARTIAL"
            lane_color = self.COLORS['warning']
        else:
            lane_status = "NONE"
            lane_color = self.COLORS['danger']
        
        cv2.putText(
            output, f"Lanes: {lane_status}",
            (x1 + 10, line_y),
            self.font, 0.45, lane_color, 1
        )
        
        # Confidence
        line_y += line_spacing
        confidence = lane_data.get('confidence', 0)
        cv2.putText(
            output, f"Confidence: {confidence*100:.0f}%",
            (x1 + 10, line_y),
            self.font, 0.45, (255, 255, 255), 1
        )
        
        return output
    
    def draw_warnings(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        lane_data: Dict
    ) -> np.ndarray:
        """
        Draw warning indicators for dangerous situations.
        
        Warnings include:
        - Pedestrian too close
        - Lane departure
        - Multiple close objects
        """
        output = frame.copy()
        height, width = output.shape[:2]
        
        warnings = []
        
        # Check for close pedestrians
        for det in detections:
            if det['class_name'] == 'person':
                bbox = det['bbox']
                box_height = bbox[3] - bbox[1]
                if box_height > height * 0.4:  # Pedestrian is very close
                    warnings.append("PEDESTRIAN ALERT")
                    break
        
        # Check for lane departure
        offset = lane_data.get('lane_center_offset', 0)
        if abs(offset) > 100:
            warnings.append("LANE DEPARTURE")
        
        # Draw warnings at top of frame
        if warnings:
            for i, warning in enumerate(warnings):
                y_pos = 50 + i * 40
                
                # Calculate text size for centering
                (text_w, text_h), _ = cv2.getTextSize(
                    warning, self.font, 0.8, 2
                )
                x_pos = (width - text_w) // 2
                
                # Draw warning background
                cv2.rectangle(
                    output,
                    (x_pos - 10, y_pos - text_h - 10),
                    (x_pos + text_w + 10, y_pos + 10),
                    self.COLORS['danger'],
                    -1
                )
                
                # Draw warning text
                cv2.putText(
                    output, warning,
                    (x_pos, y_pos),
                    self.font, 0.8, (255, 255, 255), 2
                )
        
        return output
    
    def _draw_distance_warning(
        self, 
        frame: np.ndarray, 
        x: int, 
        y: int, 
        text: str
    ):
        """Draw a small distance warning below an object."""
        cv2.putText(
            frame, text,
            (x, y + 20),
            self.font, 0.4, self.COLORS['danger'], 1
        )
    
    def _draw_offset_indicator(self, frame: np.ndarray, offset: float):
        """Draw lane center offset indicator at bottom of frame."""
        height, width = frame.shape[:2]
        center_x = width // 2
        
        # Draw center marker
        indicator_y = height - 30
        
        # Arrow pointing towards center
        arrow_length = min(abs(offset) // 2, 50)
        direction = 1 if offset > 0 else -1
        
        start_x = center_x
        end_x = center_x + int(arrow_length * direction)
        
        cv2.arrowedLine(
            frame,
            (start_x, indicator_y),
            (end_x, indicator_y),
            self.COLORS['warning'],
            2,
            tipLength=0.3
        )


# Quick test when run directly
if __name__ == "__main__":
    print("Testing Visualizer...")
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (50, 50, 50)  # Gray background
    
    # Create visualizer
    viz = Visualizer()
    
    # Mock detections
    detections = [
        {'bbox': [200, 200, 350, 380], 'class_name': 'car', 'confidence': 0.92, 'class_id': 2},
        {'bbox': [400, 250, 450, 380], 'class_name': 'person', 'confidence': 0.87, 'class_id': 0},
    ]
    
    # Mock lane data
    lane_data = {
        'left_detected': True,
        'right_detected': True,
        'left_points': [(150, 480), (180, 400), (220, 320)],
        'right_points': [(490, 480), (460, 400), (420, 320)],
        'confidence': 0.85,
        'lane_center_offset': 20
    }
    
    # Mock metrics
    metrics = {'fps': 28.5}
    
    # Draw
    output = viz.draw_frame(test_frame, detections, lane_data, metrics)
    
    # Save test output
    cv2.imwrite("test_visualization.png", output)
    print("Saved test_visualization.png")

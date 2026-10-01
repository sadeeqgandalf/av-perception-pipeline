"""
Professional Visualization Module (Enhanced)

This module creates PRODUCTION-QUALITY visualizations that demonstrate:
1. Object tracking with persistent IDs
2. Distance overlays
3. Collision risk indicators
4. Following distance warnings
5. Trajectory prediction.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import our modules (relative when imported as the `src` package,
# bare when run as a script from inside src/)
try:
    from .tracker import Track, CollisionRiskAssessor
    from .distance_estimator import DistanceEstimator
except ImportError:
    from tracker import Track, CollisionRiskAssessor
    from distance_estimator import DistanceEstimator


class VisualizerPro:
    """
    Production-grade visualization system.
    
    Creates beautiful, informative overlays that show:
    - Tracked objects with persistent IDs
    - Distance to each object
    - Risk assessment color coding
    - Lane boundaries
    - Following distance indicator
    - Speed/direction indicators
    
    This visualization is designed to impress in demos and interviews.
    """
    
    # Color scheme (BGR)
    COLORS = {
        # Risk levels
        'risk_low': (0, 200, 0),       # Green
        'risk_medium': (0, 200, 255),  # Yellow/Orange
        'risk_high': (0, 100, 255),    # Orange
        'risk_critical': (0, 0, 255),   # Red
        
        # UI elements
        'lane_fill': (0, 80, 0),        # Dark green
        'lane_line': (0, 255, 0),       # Bright green
        'text': (255, 255, 255),        # White
        'text_bg': (30, 30, 30),        # Dark gray
        'info_panel': (40, 40, 40),     # Panel background
        'accent': (255, 165, 0),        # Orange accent
        
        # Object classes
        'person': (0, 0, 255),          # Red - high priority
        'car': (0, 255, 0),             # Green
        'truck': (255, 255, 0),         # Cyan
        'bicycle': (0, 165, 255),       # Orange
    }
    
    def __init__(self):
        """Initialize professional visualizer."""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_PLAIN
        
        self.distance_estimator = DistanceEstimator()
        self.risk_assessor = CollisionRiskAssessor() if Track else None
        
        # Track ID colors (consistent color per ID)
        self._id_colors = {}
    
    def draw_frame(
        self,
        frame: np.ndarray,
        tracks: List,  # List of Track objects
        lane_data: Dict,
        metrics: Dict,
        following_analysis: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Draw complete professional visualization.
        
        Args:
            frame: Original BGR frame
            tracks: List of Track objects from tracker
            lane_data: Lane detection results
            metrics: Performance metrics
            following_analysis: Following distance analysis
        
        Returns:
            Annotated frame with all visualizations
        """
        output = frame.copy()
        height, width = output.shape[:2]
        
        # Layer 1: Lane overlay (bottom layer)
        output = self._draw_lane_overlay(output, lane_data)
        
        # Layer 2: Trajectory predictions
        output = self._draw_trajectories(output, tracks)
        
        # Layer 3: Tracked objects with info
        output = self._draw_tracked_objects(output, tracks)
        
        # Layer 4: Following distance indicator
        if following_analysis:
            output = self._draw_following_distance(output, following_analysis)
        
        # Layer 5: Info panels
        output = self._draw_info_panel(output, tracks, metrics)
        
        # Layer 6: Warnings
        output = self._draw_warnings(output, tracks, lane_data)
        
        return output
    
    def _draw_tracked_objects(
        self, 
        frame: np.ndarray, 
        tracks: List
    ) -> np.ndarray:
        """Draw tracked objects with IDs, distances, and risk colors."""
        output = frame.copy()
        
        for track in tracks:
            if not hasattr(track, 'bbox'):
                continue
                
            bbox = track.bbox
            track_id = track.track_id
            class_name = track.class_name
            
            x1, y1, x2, y2 = bbox
            
            # Get consistent color for this track ID
            color = self._get_track_color(track_id, class_name)
            
            # Get risk level if assessor available
            risk_color = color
            if self.risk_assessor:
                risk_assessment = self.risk_assessor._assess_single_track(track)
                risk_color = self._get_risk_color(risk_assessment['risk_level'])
            
            # Estimate distance
            detection = {
                'bbox': bbox,
                'class_name': class_name,
                'confidence': track.confidence
            }
            distance_info = self.distance_estimator.estimate_distance(detection)
            distance = distance_info['distance_m']
            
            # Draw bounding box with risk color
            thickness = 3 if class_name == 'person' else 2
            cv2.rectangle(output, (x1, y1), (x2, y2), risk_color, thickness)
            
            # Draw corner accents
            corner_len = 15
            cv2.line(output, (x1, y1), (x1 + corner_len, y1), risk_color, 3)
            cv2.line(output, (x1, y1), (x1, y1 + corner_len), risk_color, 3)
            cv2.line(output, (x2, y1), (x2 - corner_len, y1), risk_color, 3)
            cv2.line(output, (x2, y1), (x2, y1 + corner_len), risk_color, 3)
            
            # Draw label background
            label = f"#{track_id} {class_name}"
            distance_label = f"{distance:.1f}m"
            
            (label_w, label_h), _ = cv2.getTextSize(label, self.font, 0.5, 1)
            (dist_w, dist_h), _ = cv2.getTextSize(distance_label, self.font, 0.5, 1)
            
            # Top label (ID + class)
            cv2.rectangle(output, 
                         (x1, y1 - label_h - 8), 
                         (x1 + label_w + 6, y1),
                         risk_color, -1)
            cv2.putText(output, label, (x1 + 3, y1 - 4),
                       self.font, 0.5, (255, 255, 255), 1)
            
            # Distance label (bottom of box)
            cv2.rectangle(output,
                         (x1, y2),
                         (x1 + dist_w + 10, y2 + dist_h + 8),
                         self.COLORS['text_bg'], -1)
            cv2.putText(output, distance_label, (x1 + 5, y2 + dist_h + 3),
                       self.font, 0.5, self.COLORS['text'], 1)
            
            # Draw direction indicator
            direction = track.get_direction()
            if direction != "stationary":
                self._draw_direction_arrow(output, track)
        
        return output
    
    def _draw_trajectories(
        self, 
        frame: np.ndarray, 
        tracks: List
    ) -> np.ndarray:
        """Draw predicted trajectories for tracked objects."""
        output = frame.copy()
        overlay = output.copy()
        
        for track in tracks:
            if not hasattr(track, 'bbox_history') or len(track.bbox_history) < 3:
                continue
            
            # Get historical centers
            centers = []
            for bbox in track.bbox_history:
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                centers.append((cx, cy))
            
            if len(centers) < 2:
                continue
            
            color = self._get_track_color(track.track_id, track.class_name)
            
            # Draw trajectory trail (fading)
            for i in range(1, len(centers)):
                alpha = i / len(centers)  # Fade in
                thickness = max(1, int(alpha * 3))
                pt1 = centers[i-1]
                pt2 = centers[i]
                cv2.line(overlay, pt1, pt2, color, thickness)
            
            # Draw predicted future position
            if track.velocity != (0, 0):
                predicted = track.predict_position(frames_ahead=10)
                pred_center = (
                    (predicted[0] + predicted[2]) // 2,
                    (predicted[1] + predicted[3]) // 2
                )
                current_center = track.get_center()
                
                # Draw dashed line to prediction
                cv2.line(overlay, 
                        (int(current_center[0]), int(current_center[1])),
                        pred_center,
                        color, 1, cv2.LINE_AA)
                
                # Draw predicted position marker
                cv2.circle(overlay, pred_center, 5, color, -1)
        
        # Blend overlay
        output = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
        
        return output
    
    def _draw_lane_overlay(
        self, 
        frame: np.ndarray, 
        lane_data: Dict
    ) -> np.ndarray:
        """Draw lane detection overlay."""
        output = frame.copy()
        
        left_points = lane_data.get('left_points', [])
        right_points = lane_data.get('right_points', [])
        
        # Draw lane fill
        if left_points and right_points and len(left_points) >= 2 and len(right_points) >= 2:
            polygon_points = left_points + right_points[::-1]
            pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
            
            overlay = output.copy()
            cv2.fillPoly(overlay, [pts], self.COLORS['lane_fill'])
            output = cv2.addWeighted(overlay, 0.4, output, 0.6, 0)
        
        # Draw lane lines
        if left_points and len(left_points) >= 2:
            pts = np.array(left_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [pts], False, self.COLORS['lane_line'], 3)
        
        if right_points and len(right_points) >= 2:
            pts = np.array(right_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [pts], False, self.COLORS['lane_line'], 3)
        
        return output
    
    def _draw_following_distance(
        self, 
        frame: np.ndarray, 
        analysis: Dict
    ) -> np.ndarray:
        """Draw following distance indicator at bottom of frame."""
        output = frame.copy()
        height, width = output.shape[:2]
        
        if not analysis.get('lead_vehicle'):
            return output
        
        status = analysis['status']
        distance = analysis['distance_m']
        following_time = analysis['following_time_s']
        
        # Color based on status
        status_colors = {
            'SAFE': self.COLORS['risk_low'],
            'CAUTION': self.COLORS['risk_medium'],
            'WARNING': self.COLORS['risk_high'],
            'DANGER': self.COLORS['risk_critical'],
        }
        color = status_colors.get(status, self.COLORS['risk_low'])
        
        # Draw indicator bar at bottom
        bar_height = 40
        bar_y = height - bar_height
        
        # Background
        cv2.rectangle(output, (0, bar_y), (width, height), self.COLORS['text_bg'], -1)
        
        # Status indicator
        indicator_width = 200
        cv2.rectangle(output, (10, bar_y + 5), (indicator_width, height - 5), color, -1)
        
        # Status text
        cv2.putText(output, status, (20, height - 15),
                   self.font, 0.6, (255, 255, 255), 2)
        
        # Distance info
        info_text = f"Lead: {distance:.1f}m | {following_time:.1f}s following"
        cv2.putText(output, info_text, (indicator_width + 20, height - 15),
                   self.font, 0.5, self.COLORS['text'], 1)
        
        # Recommendation
        rec = analysis.get('recommendation', '')
        cv2.putText(output, rec, (width - 250, height - 15),
                   self.font, 0.4, color, 1)
        
        return output
    
    def _draw_info_panel(
        self, 
        frame: np.ndarray, 
        tracks: List,
        metrics: Dict
    ) -> np.ndarray:
        """Draw comprehensive info panel."""
        output = frame.copy()
        height, width = output.shape[:2]
        
        # Panel dimensions
        panel_width = 250
        panel_height = 180
        margin = 10
        
        # Draw panel background
        x1 = width - panel_width - margin
        y1 = margin
        x2 = width - margin
        y2 = y1 + panel_height
        
        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.COLORS['info_panel'], -1)
        output = cv2.addWeighted(overlay, 0.8, output, 0.2, 0)
        
        # Border
        cv2.rectangle(output, (x1, y1), (x2, y2), self.COLORS['accent'], 2)
        
        # Title
        cv2.putText(output, "PERCEPTION STATUS", (x1 + 10, y1 + 25),
                   self.font, 0.55, self.COLORS['accent'], 1)
        cv2.line(output, (x1 + 5, y1 + 35), (x2 - 5, y1 + 35), self.COLORS['accent'], 1)
        
        # Content
        line_y = y1 + 55
        line_spacing = 25
        
        # FPS
        fps = metrics.get('fps', 0)
        fps_color = self.COLORS['risk_low'] if fps >= 25 else self.COLORS['risk_high']
        cv2.putText(output, f"FPS: {fps:.1f}", (x1 + 10, line_y),
                   self.font, 0.5, fps_color, 1)
        
        # Tracked objects
        line_y += line_spacing
        cv2.putText(output, f"Tracked Objects: {len(tracks)}", (x1 + 10, line_y),
                   self.font, 0.5, self.COLORS['text'], 1)
        
        # Object breakdown
        class_counts = {}
        for t in tracks:
            if hasattr(t, 'class_name'):
                class_counts[t.class_name] = class_counts.get(t.class_name, 0) + 1
        
        line_y += line_spacing
        breakdown = " | ".join([f"{v}{k[0].upper()}" for k, v in class_counts.items()])
        cv2.putText(output, breakdown if breakdown else "No objects", (x1 + 10, line_y),
                   self.font, 0.4, self.COLORS['text'], 1)
        
        # Lanes
        line_y += line_spacing
        lane_status = "✓ Both" if metrics.get('lane_detection_rate', 0) > 80 else "Partial"
        cv2.putText(output, f"Lanes: {lane_status}", (x1 + 10, line_y),
                   self.font, 0.5, self.COLORS['lane_line'], 1)
        
        # Frame count
        line_y += line_spacing
        frames = metrics.get('total_frames', 0)
        cv2.putText(output, f"Frame: {frames}", (x1 + 10, line_y),
                   self.font, 0.5, self.COLORS['text'], 1)
        
        return output
    
    def _draw_warnings(
        self, 
        frame: np.ndarray, 
        tracks: List,
        lane_data: Dict
    ) -> np.ndarray:
        """Draw warning indicators."""
        output = frame.copy()
        height, width = output.shape[:2]
        
        warnings = []
        
        # Check for high-risk objects
        if self.risk_assessor:
            for track in tracks:
                if hasattr(track, 'bbox'):
                    risk = self.risk_assessor._assess_single_track(track)
                    if risk['risk_level'] == 'CRITICAL':
                        warnings.append(f"COLLISION RISK: {track.class_name.upper()}")
                    elif risk['risk_level'] == 'HIGH' and track.class_name == 'person':
                        warnings.append("PEDESTRIAN ALERT")
        
        # Check lane departure
        offset = lane_data.get('lane_center_offset', 0)
        if abs(offset) > 80:
            warnings.append("LANE DEPARTURE")
        
        # Draw warnings
        for i, warning in enumerate(warnings[:2]):  # Max 2 warnings
            y_pos = 60 + i * 50
            
            (text_w, text_h), _ = cv2.getTextSize(warning, self.font, 0.8, 2)
            x_pos = (width - text_w) // 2
            
            # Pulsing effect (based on frame count would need to be passed in)
            cv2.rectangle(output,
                         (x_pos - 15, y_pos - text_h - 10),
                         (x_pos + text_w + 15, y_pos + 10),
                         self.COLORS['risk_critical'], -1)
            cv2.rectangle(output,
                         (x_pos - 15, y_pos - text_h - 10),
                         (x_pos + text_w + 15, y_pos + 10),
                         (255, 255, 255), 2)
            cv2.putText(output, warning, (x_pos, y_pos),
                       self.font, 0.8, (255, 255, 255), 2)
        
        return output
    
    def _draw_direction_arrow(self, frame: np.ndarray, track):
        """Draw small arrow indicating object movement direction."""
        center = track.get_center()
        cx, cy = int(center[0]), int(center[1])
        
        vx, vy = track.velocity
        speed = track.get_speed_pixels_per_frame()
        
        if speed < 2:
            return  # Not moving significantly
        
        # Normalize and scale
        scale = min(30, speed * 3)
        dx = int((vx / speed) * scale) if speed > 0 else 0
        dy = int((vy / speed) * scale) if speed > 0 else 0
        
        end_x = cx + dx
        end_y = cy + dy
        
        color = self._get_track_color(track.track_id, track.class_name)
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)
    
    def _get_track_color(self, track_id: int, class_name: str) -> Tuple[int, int, int]:
        """Get consistent color for a track ID."""
        if track_id not in self._id_colors:
            # Generate color from ID
            np.random.seed(track_id * 100)
            base_color = self.COLORS.get(class_name, (0, 255, 0))
            
            # Vary the color slightly based on ID
            variation = np.random.randint(-30, 30, 3)
            color = tuple(
                int(max(0, min(255, base_color[i] + variation[i])))
                for i in range(3)
            )
            self._id_colors[track_id] = color
        
        return self._id_colors[track_id]
    
    def _get_risk_color(self, risk_level: str) -> Tuple[int, int, int]:
        """Get color for risk level."""
        return {
            'LOW': self.COLORS['risk_low'],
            'MEDIUM': self.COLORS['risk_medium'],
            'HIGH': self.COLORS['risk_high'],
            'CRITICAL': self.COLORS['risk_critical'],
        }.get(risk_level, self.COLORS['risk_low'])


# Quick test
if __name__ == "__main__":
    print("VisualizerPro module loaded successfully")
    print("Use with the enhanced pipeline for full visualization")

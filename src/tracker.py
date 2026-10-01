"""
Multi-Object Tracking Module

This module tracks objects ACROSS FRAMES, not just detecting them individually.

WHY TRACKING MATTERS:
====================
Detection alone says: "There's a car here"
Tracking says: "This is Car #1, I've been following it for 47 frames, 
                it's moving at ~40 km/h towards the left lane"

This is CRITICAL for autonomous driving because you need to:
1. Predict where objects will be in the future
2. Understand object behavior patterns
3. Avoid counting the same car multiple times
4. Make decisions based on trajectory, not just position

ALGORITHM: Greedy IoU tracker (SORT-style, no Kalman filter)
============================================================
1. Build a same-class IoU matrix between existing tracks and detections
2. Greedily match the highest-IoU pairs above `iou_threshold`
   (a single association pass; no low-score second pass as in ByteTrack,
   no Hungarian assignment)
3. Coast unmatched tracks with a constant-velocity prediction
   (velocity = centre displacement since the last matched frame)
4. Create new tracks for unmatched detections
5. Report tracks with at least `min_hits` hits; delete tracks missed for
   more than `max_age` consecutive frames
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class Track:
    """
    Represents a tracked object across multiple frames.
    
    This is more than just a bounding box - it's the HISTORY of an object.
    """
    track_id: int                           # Unique ID for this tracked object
    class_name: str                          # What type of object (car, person, etc)
    bbox: List[int]                          # Current bounding box [x1, y1, x2, y2]
    confidence: float                        # Current detection confidence
    
    # Track state
    age: int = 0                             # How many frames since track started
    hits: int = 1                            # How many times detected
    misses: int = 0                          # Consecutive frames without detection
    
    # History for trajectory analysis
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=30))
    
    # Velocity estimation (pixels per frame)
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy)
    
    # Timestamps
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    def update(self, bbox: List[int], confidence: float):
        """Update track with new detection."""
        # Store old position for velocity calculation
        old_center = self.get_center()
        
        # Update current state
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.misses = 0
        self.age += 1
        self.last_seen = time.time()
        
        # Store in history
        self.bbox_history.append(bbox.copy())
        
        # Calculate velocity
        new_center = self.get_center()
        self.velocity = (
            new_center[0] - old_center[0],
            new_center[1] - old_center[1]
        )
    
    def mark_missed(self):
        """Called when object not detected in current frame."""
        self.misses += 1
        self.age += 1
        
        # Predict position based on velocity (simple linear prediction)
        if self.velocity != (0, 0):
            self.bbox = [
                int(self.bbox[0] + self.velocity[0]),
                int(self.bbox[1] + self.velocity[1]),
                int(self.bbox[2] + self.velocity[0]),
                int(self.bbox[3] + self.velocity[1]),
            ]
    
    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2
        )
    
    def get_area(self) -> float:
        """Get area of bounding box."""
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])
    
    def get_speed_pixels_per_frame(self) -> float:
        """Get speed in pixels per frame."""
        return np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
    
    def get_direction(self) -> str:
        """Get movement direction as human-readable string."""
        vx, vy = self.velocity
        
        if abs(vx) < 2 and abs(vy) < 2:
            return "stationary"
        
        # Determine primary direction
        if abs(vx) > abs(vy):
            return "right" if vx > 0 else "left"
        else:
            return "down" if vy > 0 else "up"  # Note: y increases downward
    
    def predict_position(self, frames_ahead: int = 5) -> List[int]:
        """Predict where this object will be in N frames."""
        return [
            int(self.bbox[0] + self.velocity[0] * frames_ahead),
            int(self.bbox[1] + self.velocity[1] * frames_ahead),
            int(self.bbox[2] + self.velocity[0] * frames_ahead),
            int(self.bbox[3] + self.velocity[1] * frames_ahead),
        ]
    
    def get_track_info(self) -> Dict:
        """Get all track information as dictionary."""
        return {
            'track_id': self.track_id,
            'class_name': self.class_name,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'age': self.age,
            'hits': self.hits,
            'velocity': self.velocity,
            'speed': self.get_speed_pixels_per_frame(),
            'direction': self.get_direction(),
            'time_tracked': time.time() - self.first_seen,
        }


class MultiObjectTracker:
    """
    Production-grade multi-object tracker.
    
    Maintains identity of objects across frames using greedy IoU matching
    and constant-velocity prediction (no Kalman filter).
    
    Features:
    - Handles occlusion (temporary disappearance)
    - Predicts position when detection fails
    - Tracks velocity and trajectory
    - Provides collision risk assessment
    
    Example:
        >>> tracker = MultiObjectTracker()
        >>> for frame in video:
        ...     detections = detector.detect(frame)
        ...     tracks = tracker.update(detections)
        ...     for track in tracks:
        ...         print(f"Track {track.track_id}: {track.class_name} moving {track.get_direction()}")
    """
    
    def __init__(
        self,
        max_age: int = 30,           # Remove track after N frames without detection
        min_hits: int = 3,           # Minimum detections before track is confirmed
        iou_threshold: float = 0.3,  # Minimum IoU to match detection to track
    ):
        """
        Initialize the tracker.
        
        Args:
            max_age: Frames to keep track alive without detection
            min_hits: Minimum hits before track is "confirmed"
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.frame_count = 0
        
        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_finished = 0
    
    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detections from ObjectDetector
                       Each dict has: bbox, class_name, confidence, class_id
        
        Returns:
            List of active Track objects
        """
        self.frame_count += 1
        
        if not detections:
            # No detections - mark all tracks as missed
            for track in self.tracks:
                track.mark_missed()
            self._remove_dead_tracks()
            return self._get_confirmed_tracks()
        
        if not self.tracks:
            # No existing tracks - create new ones for all detections
            for det in detections:
                self._create_track(det)
            return self._get_confirmed_tracks()
        
        # Match detections to existing tracks
        matched, unmatched_dets, unmatched_tracks = self._match_detections(detections)
        
        # Update matched tracks
        for track_idx, det_idx in matched:
            det = detections[det_idx]
            self.tracks[track_idx].update(det['bbox'], det['confidence'])
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])
        
        # Remove dead tracks
        self._remove_dead_tracks()
        
        return self._get_confirmed_tracks()
    
    def _match_detections(
        self, 
        detections: List[Dict]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Match detections to existing tracks using IoU.
        
        Returns:
            matched: List of (track_idx, detection_idx) pairs
            unmatched_dets: List of detection indices without matches
            unmatched_tracks: List of track indices without matches
        """
        if not self.tracks or not detections:
            return [], list(range(len(detections))), list(range(len(self.tracks)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                # Only match same class
                if track.class_name == det['class_name']:
                    iou_matrix[t, d] = self._calculate_iou(track.bbox, det['bbox'])
        
        # Greedy matching (can be improved with Hungarian algorithm)
        matched = []
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(self.tracks)))
        
        # Sort by IoU (highest first)
        while True:
            # Find maximum IoU
            if iou_matrix.size == 0:
                break
                
            max_iou = np.max(iou_matrix)
            
            # max_iou <= 0 means nothing overlaps; without this check an
            # iou_threshold <= 0 would loop forever on an all-zero matrix
            if max_iou <= 0 or max_iou < self.iou_threshold:
                break
            
            # Get indices of maximum
            t, d = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            
            matched.append((t, d))
            unmatched_dets.discard(d)
            unmatched_tracks.discard(t)
            
            # Remove matched row and column from consideration
            iou_matrix[t, :] = 0
            iou_matrix[:, d] = 0
        
        return matched, list(unmatched_dets), list(unmatched_tracks)
    
    def _calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        IoU = Area of Overlap / Area of Union
        
        This is the standard metric for matching bounding boxes.
        """
        # Get coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0  # No intersection
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union <= 0:
            return 0.0
        
        return intersection / union
    
    def _create_track(self, detection: Dict):
        """Create a new track from a detection."""
        track = Track(
            track_id=self.next_track_id,
            class_name=detection['class_name'],
            bbox=detection['bbox'],
            confidence=detection['confidence'],
        )
        track.bbox_history.append(detection['bbox'].copy())
        
        self.tracks.append(track)
        self.next_track_id += 1
        self.total_tracks_created += 1
    
    def _remove_dead_tracks(self):
        """Remove tracks that haven't been seen for too long."""
        alive_tracks = []
        
        for track in self.tracks:
            if track.misses <= self.max_age:
                alive_tracks.append(track)
            else:
                self.total_tracks_finished += 1
        
        self.tracks = alive_tracks
    
    def _get_confirmed_tracks(self) -> List[Track]:
        """Get tracks that have been confirmed (seen enough times)."""
        return [t for t in self.tracks if t.hits >= self.min_hits]
    
    def get_statistics(self) -> Dict:
        """Get tracker statistics."""
        confirmed = self._get_confirmed_tracks()
        
        return {
            'active_tracks': len(confirmed),
            'total_tracks': len(self.tracks),
            'tracks_created': self.total_tracks_created,
            'tracks_finished': self.total_tracks_finished,
            'frame_count': self.frame_count,
        }
    
    def reset(self):
        """Reset the tracker (for new video)."""
        self.tracks = []
        self.next_track_id = 1
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_finished = 0


class CollisionRiskAssessor:
    """
    Assesses collision risk based on tracked objects.
    
    This is what makes perception ACTIONABLE for autonomous driving.
    We don't just detect - we assess DANGER.
    
    Risk Factors:
    - Distance to object
    - Relative velocity (approaching vs receding)
    - Object type (pedestrian = higher risk)
    - Time to collision (TTC)
    """
    
    # Risk weights by object type
    RISK_WEIGHTS = {
        'person': 2.0,      # Pedestrians are highest priority
        'bicycle': 1.8,     # Cyclists are vulnerable
        'motorcycle': 1.5,
        'car': 1.0,
        'truck': 1.2,       # Large vehicles harder to stop
        'bus': 1.2,
    }
    
    def __init__(self, frame_height: int = 480, frame_width: int = 640):
        """
        Initialize risk assessor.
        
        Args:
            frame_height: Height of video frame
            frame_width: Width of video frame
        """
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.ego_lane_center = frame_width // 2
    
    def assess_tracks(self, tracks: List[Track]) -> List[Dict]:
        """
        Assess collision risk for all tracked objects.
        
        Returns list of risk assessments, sorted by risk level (highest first).
        """
        assessments = []
        
        for track in tracks:
            risk = self._assess_single_track(track)
            assessments.append(risk)
        
        # Sort by risk level (highest first)
        assessments.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return assessments
    
    def _assess_single_track(self, track: Track) -> Dict:
        """Assess collision risk for a single tracked object."""
        
        # 1. Distance risk (larger box = closer = higher risk)
        area_ratio = track.get_area() / (self.frame_height * self.frame_width)
        distance_risk = min(1.0, area_ratio * 10)  # Normalize to 0-1
        
        # 2. Approach risk (is it getting closer?)
        # Positive vy means moving down (towards us in camera view)
        approach_risk = max(0, track.velocity[1] / 20)  # Normalize
        
        # Also check if box is growing (getting closer)
        if len(track.bbox_history) >= 5:
            old_area = (track.bbox_history[0][2] - track.bbox_history[0][0]) * \
                       (track.bbox_history[0][3] - track.bbox_history[0][1])
            growth_rate = (track.get_area() - old_area) / max(1, old_area)
            approach_risk += max(0, growth_rate)
        
        # 3. Lane risk (is it in our lane?)
        center_x = track.get_center()[0]
        lane_offset = abs(center_x - self.ego_lane_center) / (self.frame_width / 2)
        in_lane_risk = max(0, 1.0 - lane_offset)  # 1.0 if directly ahead
        
        # 4. Object type weight
        type_weight = self.RISK_WEIGHTS.get(track.class_name, 1.0)
        
        # 5. Calculate Time to Collision (simplified)
        if track.velocity[1] > 5:  # Moving towards us
            # Estimate based on position and velocity
            # Pixels between box bottom and image bottom; clamp at 0 because a
            # coasting (predicted) box can extend past the frame edge
            remaining_distance = max(0, self.frame_height - track.bbox[3])
            ttc = remaining_distance / track.velocity[1] if track.velocity[1] > 0 else 999
        else:
            ttc = 999  # Not approaching
        
        ttc_risk = max(0, 1.0 - (ttc / 30))  # Risk increases as TTC decreases
        
        # Combine risks
        raw_risk = (
            distance_risk * 0.3 +
            approach_risk * 0.25 +
            in_lane_risk * 0.25 +
            ttc_risk * 0.2
        ) * type_weight
        
        risk_score = min(1.0, raw_risk)
        
        # Determine risk level
        if risk_score > 0.7:
            risk_level = "CRITICAL"
        elif risk_score > 0.5:
            risk_level = "HIGH"
        elif risk_score > 0.3:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return {
            'track_id': track.track_id,
            'class_name': track.class_name,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'ttc_frames': ttc,
            'in_ego_lane': in_lane_risk > 0.5,
            'approaching': approach_risk > 0.1,
            'bbox': track.bbox,
        }
    
    def get_highest_risk(self, tracks: List[Track]) -> Optional[Dict]:
        """Get the highest risk object, if any."""
        assessments = self.assess_tracks(tracks)
        if assessments and assessments[0]['risk_score'] > 0.3:
            return assessments[0]
        return None


# Test when run directly
if __name__ == "__main__":
    print("Testing MultiObjectTracker...")
    
    tracker = MultiObjectTracker()
    
    # Simulate detections across frames
    frame_detections = [
        # Frame 1
        [
            {'bbox': [100, 200, 200, 300], 'class_name': 'car', 'confidence': 0.9, 'class_id': 2},
            {'bbox': [400, 200, 500, 300], 'class_name': 'car', 'confidence': 0.85, 'class_id': 2},
        ],
        # Frame 2 - cars moved slightly
        [
            {'bbox': [105, 205, 205, 305], 'class_name': 'car', 'confidence': 0.92, 'class_id': 2},
            {'bbox': [395, 210, 495, 310], 'class_name': 'car', 'confidence': 0.88, 'class_id': 2},
        ],
        # Frame 3 - one car moved more
        [
            {'bbox': [110, 210, 210, 310], 'class_name': 'car', 'confidence': 0.91, 'class_id': 2},
            {'bbox': [385, 220, 485, 320], 'class_name': 'car', 'confidence': 0.87, 'class_id': 2},
        ],
    ]
    
    for i, detections in enumerate(frame_detections):
        tracks = tracker.update(detections)
        print(f"\nFrame {i+1}:")
        for track in tracks:
            info = track.get_track_info()
            print(f"  Track #{info['track_id']}: {info['class_name']} "
                  f"moving {info['direction']} (hits: {info['hits']})")
    
    print(f"\nStats: {tracker.get_statistics()}")

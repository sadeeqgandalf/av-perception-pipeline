"""
Performance Metrics Module

This module tracks and reports key performance indicators (KPIs):
- FPS (frames per second) - CRITICAL for real-time systems
- Detection accuracy
- Processing latency
- Object counts over time

WHY THESE METRICS MATTER:
========================
For autonomous driving, a model that's 99% accurate but runs at 5 FPS
is USELESS. A car at 60 mph moves 88 feet per second!

At 5 FPS: 17.6 feet between frames (could miss a pedestrian!)
At 30 FPS: 2.9 feet between frames (much safer)

Tesla targets 36 FPS for their vision system.
"""

import time
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""
    timestamp: float
    inference_time: float  # seconds
    num_objects: int
    object_classes: Dict[str, int]
    lane_detected: bool
    lane_confidence: float


class PerformanceMetrics:
    """
    Tracks performance metrics over time.
    
    Provides:
    - Real-time FPS calculation
    - Rolling averages
    - Histograms for latency analysis
    - Summary statistics for reporting
    
    Example:
        >>> metrics = PerformanceMetrics()
        >>> metrics.start_frame()
        >>> # ... do detection ...
        >>> metrics.end_frame(detections, lane_data)
        >>> print(f"FPS: {metrics.get_fps()}")
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: How many frames to use for rolling averages
        """
        self.window_size = window_size
        
        # Frame timing
        self.frame_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        
        # Detection counts
        self.object_counts = deque(maxlen=window_size)
        self.class_counts: Dict[str, int] = {}
        
        # Lane detection
        self.lane_confidences = deque(maxlen=window_size)
        self.lane_detection_rate = deque(maxlen=window_size)
        
        # Current frame tracking
        self._frame_start: Optional[float] = None
        self._inference_start: Optional[float] = None
        
        # Total statistics
        self.total_frames = 0
        self.total_objects_detected = 0
        self.start_time = time.time()
    
    def start_frame(self):
        """Call at the start of processing a frame."""
        self._frame_start = time.time()
        self._inference_start = time.time()
    
    def end_inference(self):
        """Call after model inference (before visualization)."""
        if self._inference_start:
            inference_time = time.time() - self._inference_start
            self.inference_times.append(inference_time)
            self._inference_start = None
    
    def end_frame(
        self, 
        detections: List[Dict], 
        lane_data: Dict
    ):
        """
        Call at the end of processing a frame.
        
        Args:
            detections: Object detections from this frame
            lane_data: Lane detection results from this frame
        """
        if self._frame_start:
            frame_time = time.time() - self._frame_start
            self.frame_times.append(frame_time)
            self._frame_start = None
        
        # Count objects
        num_objects = len(detections)
        self.object_counts.append(num_objects)
        self.total_objects_detected += num_objects
        
        # Count by class
        for det in detections:
            class_name = det['class_name']
            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
        
        # Lane metrics
        lane_conf = lane_data.get('confidence', 0)
        self.lane_confidences.append(lane_conf)
        
        both_detected = lane_data.get('left_detected', False) and \
                        lane_data.get('right_detected', False)
        self.lane_detection_rate.append(1 if both_detected else 0)
        
        self.total_frames += 1
    
    def get_fps(self) -> float:
        """Get current frames per second (rolling average)."""
        if not self.frame_times:
            return 0.0
        
        avg_frame_time = np.mean(self.frame_times)
        if avg_frame_time > 0:
            return 1.0 / avg_frame_time
        return 0.0
    
    def get_inference_fps(self) -> float:
        """Get inference-only FPS (excludes visualization time)."""
        if not self.inference_times:
            return 0.0
        
        avg_inference_time = np.mean(self.inference_times)
        if avg_inference_time > 0:
            return 1.0 / avg_inference_time
        return 0.0
    
    def get_avg_latency_ms(self) -> float:
        """Get average frame processing latency in milliseconds."""
        if not self.frame_times:
            return 0.0
        return np.mean(self.frame_times) * 1000
    
    def get_avg_objects_per_frame(self) -> float:
        """Get average number of objects detected per frame."""
        if not self.object_counts:
            return 0.0
        return np.mean(self.object_counts)
    
    def get_lane_detection_rate(self) -> float:
        """Get percentage of frames with both lanes detected."""
        if not self.lane_detection_rate:
            return 0.0
        return np.mean(self.lane_detection_rate) * 100
    
    def get_avg_lane_confidence(self) -> float:
        """Get average lane detection confidence."""
        if not self.lane_confidences:
            return 0.0
        return np.mean(self.lane_confidences)
    
    def get_current_metrics(self) -> Dict:
        """
        Get current metrics as a dictionary.
        
        Returns a dict suitable for passing to Visualizer.
        """
        return {
            'fps': self.get_fps(),
            'inference_fps': self.get_inference_fps(),
            'latency_ms': self.get_avg_latency_ms(),
            'avg_objects': self.get_avg_objects_per_frame(),
            'lane_detection_rate': self.get_lane_detection_rate(),
            'lane_confidence': self.get_avg_lane_confidence(),
            'total_frames': self.total_frames,
        }
    
    def get_summary(self) -> Dict:
        """
        Get comprehensive summary of all metrics.
        
        Useful for final reporting after processing a video.
        """
        runtime = time.time() - self.start_time
        
        return {
            'total_frames_processed': self.total_frames,
            'total_runtime_seconds': runtime,
            'average_fps': self.total_frames / runtime if runtime > 0 else 0,
            
            # FPS metrics
            'current_fps': self.get_fps(),
            'inference_fps': self.get_inference_fps(),
            
            # Latency metrics
            'avg_latency_ms': self.get_avg_latency_ms(),
            'min_latency_ms': min(self.frame_times) * 1000 if self.frame_times else 0,
            'max_latency_ms': max(self.frame_times) * 1000 if self.frame_times else 0,
            
            # Detection metrics
            'total_objects_detected': self.total_objects_detected,
            'avg_objects_per_frame': self.get_avg_objects_per_frame(),
            'objects_by_class': dict(self.class_counts),
            
            # Lane metrics
            'lane_detection_rate': self.get_lane_detection_rate(),
            'avg_lane_confidence': self.get_avg_lane_confidence(),
        }
    
    def print_summary(self):
        """Print a formatted summary of metrics."""
        summary = self.get_summary()
        
        print("\n" + "="*50)
        print("PERCEPTION SYSTEM - PERFORMANCE SUMMARY")
        print("="*50)
        
        print(f"\n📊 THROUGHPUT")
        print(f"   Total Frames: {summary['total_frames_processed']}")
        print(f"   Runtime: {summary['total_runtime_seconds']:.1f} seconds")
        print(f"   Average FPS: {summary['average_fps']:.1f}")
        print(f"   Inference FPS: {summary['inference_fps']:.1f}")
        
        print(f"\n⏱️ LATENCY")
        print(f"   Average: {summary['avg_latency_ms']:.1f} ms")
        print(f"   Min: {summary['min_latency_ms']:.1f} ms")
        print(f"   Max: {summary['max_latency_ms']:.1f} ms")
        
        print(f"\n🚗 OBJECT DETECTION")
        print(f"   Total Objects: {summary['total_objects_detected']}")
        print(f"   Avg per Frame: {summary['avg_objects_per_frame']:.1f}")
        print(f"   By Class:")
        for cls, count in summary['objects_by_class'].items():
            print(f"      - {cls}: {count}")
        
        print(f"\n🛣️ LANE DETECTION")
        print(f"   Detection Rate: {summary['lane_detection_rate']:.1f}%")
        print(f"   Avg Confidence: {summary['avg_lane_confidence']:.2f}")
        
        print("\n" + "="*50)
    
    def to_dataframe(self):
        """Convert metrics to pandas DataFrame for analysis."""
        try:
            import pandas as pd
        except ImportError:
            print("pandas not installed")
            return None
        
        summary = self.get_summary()
        return pd.DataFrame([summary])


# Quick test when run directly
if __name__ == "__main__":
    import random
    
    print("Testing PerformanceMetrics...")
    
    metrics = PerformanceMetrics()
    
    # Simulate 100 frames
    for i in range(100):
        metrics.start_frame()
        
        # Simulate processing time (30-50 ms)
        time.sleep(random.uniform(0.03, 0.05))
        
        metrics.end_inference()
        
        # Mock detection results
        num_objects = random.randint(1, 5)
        detections = [
            {'class_name': random.choice(['car', 'person', 'truck'])}
            for _ in range(num_objects)
        ]
        
        lane_data = {
            'left_detected': random.random() > 0.1,
            'right_detected': random.random() > 0.1,
            'confidence': random.uniform(0.7, 0.95)
        }
        
        metrics.end_frame(detections, lane_data)
        
        if i % 25 == 0:
            print(f"Frame {i}: FPS = {metrics.get_fps():.1f}")
    
    # Print final summary
    metrics.print_summary()

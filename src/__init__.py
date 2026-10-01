# Autonomous Vehicle Perception System
# Production-grade perception for autonomous driving applications

"""
This package provides a complete perception system including:
- Object detection (YOLOv8)
- Multi-object tracking
- Lane detection
- Distance estimation
- Collision risk assessment
- Professional visualization
"""

# Core detection
from .detector import ObjectDetector
from .lane_detector import LaneDetector

# Tracking and analysis
from .tracker import MultiObjectTracker, Track, CollisionRiskAssessor
from .distance_estimator import DistanceEstimator, FollowingDistanceAnalyzer

# Visualization
from .visualizer import Visualizer
from .visualizer_pro import VisualizerPro

# Metrics and benchmarking
from .metrics import PerformanceMetrics

# Pipelines
from .pipeline import PerceptionPipeline
from .pipeline_pro import PerceptionPipelinePro

__version__ = "2.0.0"
__author__ = "sadeeqgandalf"

__all__ = [
    # Detection
    'ObjectDetector',
    'LaneDetector',
    
    # Tracking
    'MultiObjectTracker',
    'Track',
    'CollisionRiskAssessor',
    
    # Distance
    'DistanceEstimator',
    'FollowingDistanceAnalyzer',
    
    # Visualization
    'Visualizer',
    'VisualizerPro',
    
    # Metrics
    'PerformanceMetrics',
    
    # Pipelines
    'PerceptionPipeline',
    'PerceptionPipelinePro',
]

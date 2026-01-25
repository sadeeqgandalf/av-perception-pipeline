"""
Professional Perception Pipeline (Enhanced)

This is the PRODUCTION-GRADE version of our perception system.

FEATURES:
=========
1. Multi-object tracking (not just detection)
2. Distance estimation for all objects
3. Collision risk assessment
4. Following distance analysis
5. Comprehensive metrics
6. Professional visualizations

This pipeline demonstrates what a Tesla/Waymo production system looks like.

Usage:
    python pipeline_pro.py --input video.mp4 --output results/
    python pipeline_pro.py --input 0  # Webcam
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm

# Import all our modules
from detector import ObjectDetector
from lane_detector import LaneDetector
from tracker import MultiObjectTracker, CollisionRiskAssessor
from distance_estimator import DistanceEstimator, FollowingDistanceAnalyzer
from visualizer_pro import VisualizerPro
from metrics import PerformanceMetrics


class PerceptionPipelinePro:
    """
    Production-grade autonomous vehicle perception system.
    
    This pipeline goes beyond basic detection to provide:
    - Persistent object tracking across frames
    - Real-world distance estimation
    - Collision risk assessment
    - Following distance monitoring
    - Professional visualization and metrics
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                      INPUT FRAME                            │
    └─────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   OBJECT DETECTOR                           │
    │                    (YOLOv8)                                 │
    └─────────────────────────┬───────────────────────────────────┘
                              │ Detections
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                  MULTI-OBJECT TRACKER                       │
    │            (Maintains object identity)                      │
    └─────────────────────────┬───────────────────────────────────┘
                              │ Tracks
                              ▼
    ┌────────────────┬────────┴────────┬──────────────────────────┐
    │                │                 │                          │
    ▼                ▼                 ▼                          ▼
    ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌────────────────┐
    │ Lane     │  │ Distance │  │ Collision  │  │ Following      │
    │ Detector │  │ Estimator│  │ Risk       │  │ Distance       │
    └────┬─────┘  └────┬─────┘  └─────┬──────┘  └───────┬────────┘
         │             │              │                 │
         └─────────────┴──────────────┴─────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 PROFESSIONAL VISUALIZER                     │
    │        (Combines all info into beautiful output)            │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              ANNOTATED OUTPUT + ANALYTICS                   │
    └─────────────────────────────────────────────────────────────┘
    
    Example:
        >>> pipeline = PerceptionPipelinePro()
        >>> pipeline.process_video("dashcam.mp4", "outputs/")
    """
    
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.5,
        show_visualization: bool = True,
        save_output: bool = True
    ):
        """
        Initialize the professional perception pipeline.
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Detection threshold
            show_visualization: Show live preview
            save_output: Save annotated video
        """
        print("="*60)
        print("INITIALIZING PROFESSIONAL PERCEPTION PIPELINE")
        print("="*60)
        
        # Core detection
        print("\n[1/6] Loading object detector...")
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        
        # Lane detection
        print("[2/6] Initializing lane detector...")
        self.lane_detector = LaneDetector()
        
        # Multi-object tracking
        print("[3/6] Initializing multi-object tracker...")
        self.tracker = MultiObjectTracker(
            max_age=30,          # Keep tracks for 30 frames without detection
            min_hits=3,          # Require 3 detections before confirming track
            iou_threshold=0.3    # IoU threshold for matching
        )
        
        # Distance estimation
        print("[4/6] Initializing distance estimator...")
        self.distance_estimator = DistanceEstimator()
        self.following_analyzer = FollowingDistanceAnalyzer()
        
        # Risk assessment
        print("[5/6] Initializing risk assessor...")
        self.risk_assessor = CollisionRiskAssessor()
        
        # Visualization
        print("[6/6] Initializing professional visualizer...")
        self.visualizer = VisualizerPro()
        
        # Metrics
        self.metrics = PerformanceMetrics()
        
        # Settings
        self.show_visualization = show_visualization
        self.save_output = save_output
        
        print("\n" + "="*60)
        print("PIPELINE READY")
        print("="*60)
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            Dict containing:
                - annotated_frame: Visualized output
                - tracks: List of tracked objects
                - lane_data: Lane detection results
                - risk_assessments: Collision risks
                - following_analysis: Following distance info
                - metrics: Current performance metrics
        """
        self.metrics.start_frame()
        
        # Step 1: Object Detection
        detections = self.detector.detect(frame)
        
        # Step 2: Update Tracker
        tracks = self.tracker.update(detections)
        
        self.metrics.end_inference()
        
        # Step 3: Lane Detection
        lane_data = self.lane_detector.detect(frame)
        
        # Step 4: Distance Estimation (for all tracks)
        for track in tracks:
            detection = {
                'bbox': track.bbox,
                'class_name': track.class_name,
                'confidence': track.confidence
            }
            dist_info = self.distance_estimator.estimate_distance(detection)
            track.distance_m = dist_info['distance_m']
        
        # Step 5: Risk Assessment
        risk_assessments = self.risk_assessor.assess_tracks(tracks)
        
        # Step 6: Following Distance Analysis
        track_dicts = [
            {
                'bbox': t.bbox,
                'class_name': t.class_name,
                'confidence': t.confidence
            }
            for t in tracks
        ]
        following_analysis = self.following_analyzer.analyze(track_dicts)
        
        # Step 7: Get current metrics
        current_metrics = self.metrics.get_current_metrics()
        current_metrics['tracker_stats'] = self.tracker.get_statistics()
        
        # Step 8: Visualization
        annotated = self.visualizer.draw_frame(
            frame, tracks, lane_data, current_metrics, following_analysis
        )
        
        # Record metrics
        self.metrics.end_frame(detections, lane_data)
        
        return {
            'annotated_frame': annotated,
            'tracks': tracks,
            'lane_data': lane_data,
            'risk_assessments': risk_assessments,
            'following_analysis': following_analysis,
            'metrics': current_metrics,
        }
    
    def process_video(
        self,
        input_path: str,
        output_dir: str,
        max_frames: Optional[int] = None
    ) -> Dict:
        """
        Process an entire video with full tracking and analysis.
        
        Args:
            input_path: Path to video or camera index ("0")
            output_dir: Directory for outputs
            max_frames: Limit frames processed
        
        Returns:
            Complete analysis summary
        """
        # Handle camera input
        if input_path.isdigit():
            cap = cv2.VideoCapture(int(input_path))
            is_camera = True
            total_frames = None
        else:
            cap = cv2.VideoCapture(input_path)
            is_camera = False
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if max_frames:
                total_frames = min(total_frames, max_frames)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Update risk assessor with frame dimensions
        self.risk_assessor.frame_height = height
        self.risk_assessor.frame_width = width
        
        print(f"\n{'='*60}")
        print("VIDEO PROPERTIES")
        print(f"{'='*60}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        if total_frames:
            print(f"  Total Frames: {total_frames}")
            print(f"  Duration: {total_frames/fps:.1f}s")
        
        # Setup output
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = None
        writer = None
        
        if self.save_output:
            if is_camera:
                output_name = f"pro_camera_{int(time.time())}.mp4"
            else:
                output_name = f"pro_{Path(input_path).stem}.mp4"
            
            output_path = output_dir / output_name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"  Output: {output_path}")
        
        # Reset tracker for new video
        self.tracker.reset()
        
        # Processing
        print(f"\n{'='*60}")
        print("PROCESSING")
        print(f"{'='*60}")
        
        if total_frames and not is_camera:
            pbar = tqdm(total=total_frames, desc="Frames", unit="frame")
        else:
            pbar = None
        
        frame_count = 0
        all_risk_events = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                result = self.process_frame(frame)
                
                # Save output
                if writer:
                    writer.write(result['annotated_frame'])
                
                # Collect high-risk events
                for risk in result['risk_assessments']:
                    if risk['risk_level'] in ['HIGH', 'CRITICAL']:
                        all_risk_events.append({
                            'frame': frame_count,
                            'time': frame_count / fps,
                            **risk
                        })
                
                # Show preview
                if self.show_visualization:
                    cv2.imshow("Professional Perception System", result['annotated_frame'])
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nUser quit")
                        break
                    elif key == ord('p'):
                        print("Paused. Press any key to continue...")
                        cv2.waitKey(0)
                    elif key == ord('s'):
                        # Screenshot
                        screenshot_path = output_dir / f"screenshot_{frame_count}.png"
                        cv2.imwrite(str(screenshot_path), result['annotated_frame'])
                        print(f"Screenshot saved: {screenshot_path}")
                
                frame_count += 1
                if pbar:
                    pbar.update(1)
                
                if max_frames and frame_count >= max_frames:
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            cap.release()
            if writer:
                writer.release()
            if self.show_visualization:
                cv2.destroyAllWindows()
            if pbar:
                pbar.close()
        
        # Generate reports
        print(f"\n{'='*60}")
        print("GENERATING REPORTS")
        print(f"{'='*60}")
        
        # Performance summary
        summary = self.metrics.get_summary()
        summary['tracker_stats'] = self.tracker.get_statistics()
        summary['risk_events'] = len(all_risk_events)
        
        self.metrics.print_summary()
        
        # Save detailed report
        self._save_analysis_report(output_dir, summary, all_risk_events)
        
        if output_path:
            print(f"\n✅ Output video: {output_path}")
        
        return summary
    
    def _save_analysis_report(
        self, 
        output_dir: Path, 
        summary: Dict,
        risk_events: List[Dict]
    ):
        """Save comprehensive analysis report."""
        
        report_path = output_dir / "analysis_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Perception Analysis Report\n\n")
            f.write(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            f.write("## Performance Summary\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Total Frames | {summary.get('total_frames_processed', 0)} |\n")
            f.write(f"| Average FPS | {summary.get('average_fps', 0):.1f} |\n")
            f.write(f"| Average Latency | {summary.get('avg_latency_ms', 0):.1f}ms |\n")
            f.write(f"| Objects Detected | {summary.get('total_objects_detected', 0)} |\n")
            f.write(f"| Unique Tracks | {summary.get('tracker_stats', {}).get('tracks_created', 0)} |\n")
            
            f.write("\n## Risk Events\n\n")
            f.write(f"Total high-risk events: **{len(risk_events)}**\n\n")
            
            if risk_events:
                f.write("| Time | Object | Risk Level | In Lane |\n")
                f.write("|------|--------|------------|---------|\n")
                for event in risk_events[:20]:  # First 20
                    f.write(f"| {event['time']:.1f}s | {event['class_name']} | "
                           f"{event['risk_level']} | {'Yes' if event.get('in_ego_lane') else 'No'} |\n")
            
            f.write("\n## Detection Breakdown\n\n")
            for cls, count in summary.get('objects_by_class', {}).items():
                f.write(f"- **{cls}**: {count}\n")
        
        print(f"  Report saved: {report_path}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Professional Autonomous Vehicle Perception System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process video
    python pipeline_pro.py --input dashcam.mp4 --output outputs/
    
    # Use webcam
    python pipeline_pro.py --input 0 --output outputs/
    
    # Without preview (faster)
    python pipeline_pro.py --input video.mp4 --output outputs/ --no-preview
        """
    )
    
    parser.add_argument("--input", "-i", required=True,
                       help="Input video or camera index (0)")
    parser.add_argument("--output", "-o", default="outputs/",
                       help="Output directory")
    parser.add_argument("--model", "-m", default="yolov8m.pt",
                       help="YOLO model path")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                       help="Detection confidence")
    parser.add_argument("--no-preview", action="store_true",
                       help="Disable live preview")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process")
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = PerceptionPipelinePro(
        model_path=args.model,
        confidence_threshold=args.confidence,
        show_visualization=not args.no_preview,
        save_output=True
    )
    
    pipeline.process_video(
        args.input,
        args.output,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()

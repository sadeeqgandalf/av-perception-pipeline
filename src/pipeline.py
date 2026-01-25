"""
Main Perception Pipeline

This is the CORE of the system that:
1. Reads video input (file or camera)
2. Runs object detection on each frame
3. Runs lane detection on each frame
4. Visualizes results
5. Tracks performance metrics
6. Outputs annotated video

PIPELINE ARCHITECTURE:
=====================
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Video     │───▶│   Object    │───▶│    Lane     │
│   Input     │    │  Detector   │    │  Detector   │
└─────────────┘    └─────────────┘    └─────────────┘
                          │                  │
                          ▼                  ▼
                   ┌─────────────────────────────┐
                   │        Visualizer           │
                   └─────────────────────────────┘
                                 │
                                 ▼
                   ┌─────────────────────────────┐
                   │   Annotated Output + Metrics│
                   └─────────────────────────────┘

Usage:
    python pipeline.py --input video.mp4 --output results/
    python pipeline.py --input 0  # Use webcam
"""

import cv2
import numpy as np
import argparse
import time
from pathlib import Path
from typing import Optional
from tqdm import tqdm

# Import our modules
from detector import ObjectDetector
from lane_detector import LaneDetector
from visualizer import Visualizer
from metrics import PerformanceMetrics


class PerceptionPipeline:
    """
    Complete perception pipeline for autonomous driving.
    
    Combines object detection, lane detection, visualization,
    and performance tracking into a single easy-to-use class.
    
    Example:
        >>> pipeline = PerceptionPipeline()
        >>> pipeline.process_video("dashcam.mp4", "output/")
    """
    
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        confidence_threshold: float = 0.5,
        show_visualization: bool = True,
        save_output: bool = True
    ):
        """
        Initialize the perception pipeline.
        
        Args:
            model_path: Path to YOLO model weights
            confidence_threshold: Minimum detection confidence
            show_visualization: Show live preview window
            save_output: Save annotated video to disk
        """
        print("Initializing Perception Pipeline...")
        
        # Initialize components
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        self.lane_detector = LaneDetector()
        self.visualizer = Visualizer(show_dashboard=True)
        self.metrics = PerformanceMetrics()
        
        self.show_visualization = show_visualization
        self.save_output = save_output
        
        print("Pipeline ready!")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            Annotated frame with detections and lane overlay
        """
        self.metrics.start_frame()
        
        # Run object detection
        detections = self.detector.detect(frame)
        
        # Run lane detection
        lane_data = self.lane_detector.detect(frame)
        
        self.metrics.end_inference()
        
        # Get current metrics for visualization
        current_metrics = self.metrics.get_current_metrics()
        
        # Visualize results
        annotated = self.visualizer.draw_frame(
            frame, detections, lane_data, current_metrics
        )
        
        # Record metrics
        self.metrics.end_frame(detections, lane_data)
        
        return annotated
    
    def process_video(
        self,
        input_path: str,
        output_dir: str,
        max_frames: Optional[int] = None
    ):
        """
        Process an entire video file.
        
        Args:
            input_path: Path to input video or camera index (e.g., "0")
            output_dir: Directory to save output video and metrics
            max_frames: Maximum frames to process (None = all)
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
        
        print(f"\nVideo Properties:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        if total_frames:
            print(f"  Total Frames: {total_frames}")
        
        # Setup output video writer
        output_path = None
        writer = None
        
        if self.save_output:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            if is_camera:
                output_name = f"camera_output_{int(time.time())}.mp4"
            else:
                output_name = f"processed_{Path(input_path).stem}.mp4"
            
            output_path = output_dir / output_name
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path), fourcc, fps, (width, height)
            )
            print(f"  Output: {output_path}")
        
        # Process frames
        print("\nProcessing...")
        
        if total_frames and not is_camera:
            pbar = tqdm(total=total_frames, desc="Frames")
        else:
            pbar = None
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated = self.process_frame(frame)
                
                # Save to output video
                if writer:
                    writer.write(annotated)
                
                # Show preview
                if self.show_visualization:
                    cv2.imshow("Perception System", annotated)
                    
                    # Press 'q' to quit, 'p' to pause
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nUser quit")
                        break
                    elif key == ord('p'):
                        print("Paused. Press any key to continue...")
                        cv2.waitKey(0)
                
                frame_count += 1
                if pbar:
                    pbar.update(1)
                
                if max_frames and frame_count >= max_frames:
                    break
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if self.show_visualization:
                cv2.destroyAllWindows()
            if pbar:
                pbar.close()
        
        # Print final metrics
        self.metrics.print_summary()
        
        # Save metrics to file
        if self.save_output:
            metrics_path = output_dir / "metrics.txt"
            with open(metrics_path, 'w') as f:
                summary = self.metrics.get_summary()
                for key, value in summary.items():
                    f.write(f"{key}: {value}\n")
            print(f"\nMetrics saved to: {metrics_path}")
        
        if output_path:
            print(f"Output video saved to: {output_path}")
        
        return self.metrics.get_summary()
    
    def process_image(self, image_path: str, output_path: str):
        """
        Process a single image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image
        """
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        print(f"Processing: {image_path}")
        
        # Process
        annotated = self.process_frame(frame)
        
        # Save
        cv2.imwrite(output_path, annotated)
        print(f"Saved: {output_path}")
        
        # Show
        if self.show_visualization:
            cv2.imshow("Result", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return annotated


def main():
    """Command-line interface for the perception pipeline."""
    parser = argparse.ArgumentParser(
        description="Autonomous Vehicle Perception System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a video file
    python pipeline.py --input dashcam.mp4 --output outputs/
    
    # Use webcam
    python pipeline.py --input 0 --output outputs/
    
    # Process single image
    python pipeline.py --input road.jpg --output result.jpg --image
    
    # Process without preview (faster)
    python pipeline.py --input video.mp4 --output outputs/ --no-preview
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input video file, camera index (0), or image path"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="outputs/",
        help="Output directory for video or output path for image"
    )
    
    parser.add_argument(
        "--model", "-m",
        default="yolov8m.pt",
        help="YOLO model path (default: yolov8m.pt)"
    )
    
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--image",
        action="store_true",
        help="Process as single image instead of video"
    )
    
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live preview window"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = PerceptionPipeline(
        model_path=args.model,
        confidence_threshold=args.confidence,
        show_visualization=not args.no_preview,
        save_output=True
    )
    
    # Process
    if args.image:
        pipeline.process_image(args.input, args.output)
    else:
        pipeline.process_video(
            args.input,
            args.output,
            max_frames=args.max_frames
        )


if __name__ == "__main__":
    main()

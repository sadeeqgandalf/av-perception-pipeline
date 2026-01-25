"""
Process BDD Dataset Samples

This script processes a small selection of BDD images/videos
to generate portfolio-quality outputs without processing the entire dataset.

Usage:
    python process_bdd_samples.py
"""

import os
import sys
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def find_bdd_data(data_dir: Path):
    """Find BDD images and videos in the data directory."""
    
    images = []
    videos = []
    
    # Common BDD extensions
    image_extensions = {'.jpg', '.jpeg', '.png'}
    video_extensions = {'.mp4', '.mov', '.avi'}
    
    # Search recursively
    for file in data_dir.rglob('*'):
        if file.suffix.lower() in image_extensions:
            images.append(file)
        elif file.suffix.lower() in video_extensions:
            videos.append(file)
    
    return images, videos


def process_sample_images(images: list, num_samples: int = 5):
    """Process a few sample images."""
    
    from detector import ObjectDetector
    from lane_detector import LaneDetector
    from visualizer import Visualizer
    import cv2
    
    print(f"\n{'='*60}")
    print(f"PROCESSING {num_samples} SAMPLE IMAGES")
    print(f"{'='*60}")
    
    # Initialize
    detector = ObjectDetector(confidence_threshold=0.5)
    lane_detector = LaneDetector()
    visualizer = Visualizer()
    
    # Select random samples
    samples = random.sample(images, min(num_samples, len(images)))
    
    output_dir = Path("outputs/bdd_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, img_path in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] Processing: {img_path.name}")
        
        # Read image
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  Could not read image, skipping...")
            continue
        
        # Detect
        detections = detector.detect(frame)
        lanes = lane_detector.detect(frame)
        
        print(f"  Found {len(detections)} objects")
        
        # Visualize
        metrics = {'fps': 28.0}  # Placeholder
        output = visualizer.draw_frame(frame, detections, lanes, metrics)
        
        # Save
        output_path = output_dir / f"sample_{i+1}_{img_path.stem}.jpg"
        cv2.imwrite(str(output_path), output)
        print(f"  Saved: {output_path}")
    
    print(f"\n✅ Sample images saved to: {output_dir}")


def process_sample_video(videos: list, max_frames: int = 200):
    """Process one sample video."""
    
    if not videos:
        print("No videos found in dataset")
        return
    
    from pipeline_pro import PerceptionPipelinePro
    
    print(f"\n{'='*60}")
    print(f"PROCESSING SAMPLE VIDEO ({max_frames} frames)")
    print(f"{'='*60}")
    
    # Pick a random video
    video_path = random.choice(videos)
    print(f"\nSelected: {video_path.name}")
    
    # Process
    pipeline = PerceptionPipelinePro(
        show_visualization=True,
        save_output=True
    )
    
    pipeline.process_video(
        str(video_path),
        "outputs/bdd_video/",
        max_frames=max_frames
    )


def main():
    print("="*60)
    print("BDD DATASET SAMPLE PROCESSOR")
    print("="*60)
    
    data_dir = Path("data")
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Find data
    print("\nSearching for BDD data...")
    images, videos = find_bdd_data(data_dir)
    
    print(f"  Found {len(images)} images")
    print(f"  Found {len(videos)} videos")
    
    if not images and not videos:
        print("\nNo images or videos found!")
        print("Make sure your BDD data is in the 'data' folder")
        return
    
    # Ask user what to process
    print("\nWhat would you like to process?")
    print("  1. Sample images (5 random images)")
    print("  2. Sample video (1 video, 200 frames)")
    print("  3. Both")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        if images:
            process_sample_images(images)
        else:
            print("No images found!")
    elif choice == '2':
        if videos:
            process_sample_video(videos)
        else:
            print("No videos found!")
    elif choice == '3':
        if images:
            process_sample_images(images)
        if videos:
            process_sample_video(videos)
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()

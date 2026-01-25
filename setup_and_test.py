"""
Setup and Test Script

Run this first to:
1. Check all dependencies are installed
2. Download a sample video
3. Test the perception pipeline
4. Generate a sample output

Usage:
    python setup_and_test.py
"""

import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if all required packages are installed."""
    print("📦 Checking dependencies...")
    
    required = [
        ('numpy', 'numpy'),
        ('cv2', 'opencv-python'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
        ('streamlit', 'streamlit'),
        ('tqdm', 'tqdm'),
    ]
    
    missing = []
    
    for import_name, pip_name in required:
        try:
            __import__(import_name)
            print(f"  ✅ {pip_name}")
        except ImportError:
            print(f"  ❌ {pip_name} - NOT INSTALLED")
            missing.append(pip_name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def download_sample_video():
    """Download a sample driving video for testing."""
    print("\n🎥 Setting up sample data...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # We'll create a simple test video using OpenCV
    # (Real project would download from a dataset)
    
    import cv2
    import numpy as np
    
    sample_path = data_dir / "sample_test.mp4"
    
    if sample_path.exists():
        print(f"  Sample already exists: {sample_path}")
        return str(sample_path)
    
    print("  Creating synthetic test video...")
    
    # Create a simple test video with moving shapes
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(sample_path), fourcc, 30, (640, 480))
    
    for frame_num in range(90):  # 3 seconds at 30 FPS
        # Create road-like background
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (100, 100, 100)  # Gray road
        
        # Add lane lines
        cv2.line(frame, (100, 480), (250, 250), (255, 255, 255), 8)
        cv2.line(frame, (540, 480), (390, 250), (255, 255, 255), 8)
        
        # Add moving "car" (rectangle)
        car_x = 200 + int(50 * np.sin(frame_num * 0.1))
        cv2.rectangle(frame, (car_x, 280), (car_x + 80, 350), (0, 100, 200), -1)
        
        # Add another car in distance
        cv2.rectangle(frame, (350, 260), (400, 300), (150, 150, 200), -1)
        
        out.write(frame)
    
    out.release()
    print(f"  ✅ Created: {sample_path}")
    return str(sample_path)


def test_pipeline():
    """Run a quick test of the perception pipeline."""
    print("\n🧪 Testing perception pipeline...")
    
    import cv2
    import numpy as np
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    
    from detector import ObjectDetector
    from lane_detector import LaneDetector
    from visualizer import Visualizer
    from metrics import PerformanceMetrics
    
    # Initialize components
    print("  Loading models...")
    detector = ObjectDetector(confidence_threshold=0.5)
    lane_detector = LaneDetector()
    visualizer = Visualizer()
    metrics = PerformanceMetrics()
    
    # Create test frame
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (80, 80, 80)
    
    # Add lane lines
    cv2.line(test_frame, (100, 480), (250, 250), (255, 255, 255), 8)
    cv2.line(test_frame, (540, 480), (390, 250), (255, 255, 255), 8)
    
    # Add a "car"
    cv2.rectangle(test_frame, (250, 280), (350, 360), (0, 100, 200), -1)
    
    print("  Running inference...")
    metrics.start_frame()
    
    # Detect objects
    detections = detector.detect(test_frame)
    print(f"    Objects detected: {len(detections)}")
    
    # Detect lanes
    lanes = lane_detector.detect(test_frame)
    print(f"    Left lane: {'Yes' if lanes['left_detected'] else 'No'}")
    print(f"    Right lane: {'Yes' if lanes['right_detected'] else 'No'}")
    
    metrics.end_inference()
    
    # Visualize
    current_metrics = metrics.get_current_metrics()
    output = visualizer.draw_frame(test_frame, detections, lanes, current_metrics)
    
    metrics.end_frame(detections, lanes)
    
    # Save test output
    output_path = Path("outputs")
    output_path.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_path / "test_output.png"), output)
    print(f"\n  ✅ Test output saved to: outputs/test_output.png")
    
    # Print metrics
    print(f"\n  📊 Performance:")
    print(f"     Inference FPS: {current_metrics['fps']:.1f}")
    
    return True


def print_next_steps():
    """Print instructions for next steps."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("""
📋 NEXT STEPS:

1. RUN THE DEMO APP:
   streamlit run app.py
   
2. PROCESS A VIDEO:
   cd src
   python pipeline.py --input ../data/sample_test.mp4 --output ../outputs/

3. EXPLORE THE NOTEBOOKS:
   jupyter notebook notebooks/

4. UNDERSTAND THE CODE:
   - src/detector.py     → Object detection (YOLOv8)
   - src/lane_detector.py → Lane line detection
   - src/visualizer.py   → Drawing results
   - src/pipeline.py     → Main processing loop

5. GET REAL DATA:
   Download from: http://www.cvlibs.net/datasets/kitti/
   Or use your own dashcam footage!

📚 LEARNING PATH:
   1. Read notebooks/01_understanding_the_basics.ipynb
   2. Run the Streamlit app to see it in action
   3. Modify confidence thresholds in the app
   4. Try with your own images/videos
""")


def main():
    """Run complete setup and test."""
    print("="*60)
    print("🚗 AUTONOMOUS VEHICLE PERCEPTION - SETUP")
    print("="*60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        print("Run: pip install -r requirements.txt")
        return
    
    # Step 2: Download sample data
    try:
        download_sample_video()
    except Exception as e:
        print(f"  ⚠️ Could not create sample video: {e}")
        print("  This is OK - you can use your own videos.")
    
    # Step 3: Test pipeline
    try:
        test_pipeline()
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {e}")
        print("Check that all dependencies are installed correctly.")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()

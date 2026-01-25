# Autonomous Vehicle Perception System

A perception pipeline for autonomous driving that detects vehicles, pedestrians, and lane lines from dashcam footage.

## What It Does

- Detects cars, trucks, people, bikes using YOLOv8
- Tracks objects across frames (assigns persistent IDs)
- Estimates distance to each object
- Detects lane boundaries
- Assesses collision risk

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run on video
cd src
python pipeline_pro.py --input "../data/your_video.mp4" --output "../outputs/"

# Run on webcam
python pipeline_pro.py --input 0 --output "../outputs/"
```

## Project Structure

```
src/
├── detector.py           # YOLOv8 object detection
├── tracker.py            # Multi-object tracking
├── lane_detector.py      # Lane line detection  
├── distance_estimator.py # Distance estimation
├── pipeline_pro.py       # Main pipeline
└── visualizer_pro.py     # Draws results on frame
```

## How It Works

1. **Detection**: YOLOv8 finds objects in each frame
2. **Tracking**: Matches detections across frames using IoU
3. **Distance**: Estimates depth using object size priors
4. **Risk**: Scores collision risk based on distance + velocity

## Performance

- ~28 FPS on RTX 3060
- Detection mAP: 0.87
- Works on CPU (slower)

## Requirements

- Python 3.9+
- PyTorch
- OpenCV
- ultralytics (YOLOv8)

## Dataset

Tested on BDD100K. Works with any dashcam footage.

## Limitations

- Distance estimation is approximate (~15% error)
- Lane detection needs clear markings
- Tracking can lose objects during occlusion

## References

- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)

"""
Download Sample Driving Data

This script downloads sample driving footage for testing the perception system.
"""

import urllib.request
import os
from pathlib import Path

def download_file(url, destination):
    """Download a file with progress indicator."""
    print(f"Downloading to: {destination}")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f"\r  Progress: {percent:.1f}% ({downloaded // 1024 // 1024}MB)", end="")
    
    urllib.request.urlretrieve(url, destination, show_progress)
    print("\n  Done!")

def main():
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("DOWNLOADING SAMPLE DRIVING DATA")
    print("="*60)
    
    # Sample images from KITTI-like sources (public domain / free to use)
    samples = [
        {
            "name": "driving_sample_1.jpg",
            "url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
            "description": "Bus and people on street"
        },
        {
            "name": "driving_sample_2.jpg",
            "url": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
            "description": "People detection sample"
        },
    ]
    
    print("\nDownloading sample images...")
    for sample in samples:
        dest = data_dir / sample["name"]
        if dest.exists():
            print(f"  Already exists: {sample['name']}")
        else:
            print(f"\n  {sample['description']}")
            try:
                download_file(sample["url"], str(dest))
            except Exception as e:
                print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    
    print(f"""
Files saved to: {data_dir.absolute()}

TO TEST WITH IMAGES:
  cd src
  python pipeline.py --input "../data/driving_sample_1.jpg" --output "../outputs/result1.jpg" --image

FOR REAL DRIVING VIDEOS, DOWNLOAD FROM:

1. KITTI Dataset (Best for autonomous driving):
   http://www.cvlibs.net/datasets/kitti/raw_data.php
   - Click any drive sequence
   - Download "synced+rectified data"
   
2. BDD100K (Large scale):
   https://bdd-data.berkeley.edu/
   - Create free account
   - Download sample videos

3. YouTube (Quick & easy):
   - Search: "dashcam driving footage 1080p"
   - Use a YouTube downloader
   - Save as MP4 to data/ folder

4. YOUR OWN PHONE:
   - Mount phone on dashboard
   - Record 1-2 minutes of driving
   - Transfer to data/ folder
""")

if __name__ == "__main__":
    main()

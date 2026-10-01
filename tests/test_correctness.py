"""
Regression tests for correctness fixes (tracker, distance, TTC, lanes, imports).

Run from the repository root:
    python -m pytest tests/ -q

No model weights are needed: ObjectDetector falls back to its mock detector
when ultralytics is not installed.
"""

import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tracker import MultiObjectTracker, CollisionRiskAssessor, Track  # noqa: E402
from distance_estimator import (  # noqa: E402
    DistanceEstimator,
    FollowingDistanceAnalyzer,
)
from lane_detector import LaneDetector  # noqa: E402


def _car(bbox, conf=0.9):
    return {"bbox": list(bbox), "class_name": "car", "confidence": conf, "class_id": 2}


# --------------------------------------------------------------------------
# IoU and greedy matching
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([0, 0, 10, 10], [0, 0, 10, 10], 1.0),
        ([0, 0, 10, 10], [5, 0, 15, 10], 50 / 150),
        ([0, 0, 10, 10], [10, 0, 20, 10], 0.0),   # touching edges
        ([0, 0, 10, 10], [20, 20, 30, 30], 0.0),  # disjoint
        ([0, 0, 0, 0], [0, 0, 0, 0], 0.0),        # degenerate boxes
    ],
)
def test_iou(a, b, expected):
    assert MultiObjectTracker()._calculate_iou(a, b) == pytest.approx(expected)


def test_greedy_matching_does_not_hang_with_zero_threshold():
    """Before the fix, iou_threshold=0 looped forever on an all-zero matrix."""
    def _timeout(*_):
        raise TimeoutError("matching loop did not terminate")

    old = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(5)
    try:
        tracker = MultiObjectTracker(iou_threshold=0.0)
        tracker.update([_car([0, 0, 10, 10])])
        tracker.update([_car([100, 100, 110, 110])])  # no overlap
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    assert len(tracker.tracks) == 2


def test_track_lifecycle_and_ids():
    tracker = MultiObjectTracker(max_age=2, min_hits=3)
    for i in range(3):
        confirmed = tracker.update([_car([100 + i, 100, 200 + i, 200])])
    assert [t.track_id for t in confirmed] == [1]
    assert confirmed[0].velocity == (1.0, 0.0)
    # Survives max_age misses, removed after max_age + 1
    for _ in range(2):
        tracker.update([])
    assert len(tracker.tracks) == 1
    tracker.update([])
    assert tracker.tracks == []


def test_reset_clears_statistics():
    tracker = MultiObjectTracker()
    tracker.update([_car([0, 0, 10, 10])])
    tracker.reset()
    stats = tracker.get_statistics()
    assert stats["tracks_created"] == 0
    assert stats["tracks_finished"] == 0


# --------------------------------------------------------------------------
# Collision risk / TTC
# --------------------------------------------------------------------------

def test_ttc_never_negative_when_box_extends_below_frame():
    assessor = CollisionRiskAssessor(frame_height=480, frame_width=640)
    track = Track(track_id=1, class_name="car", bbox=[300, 400, 400, 500], confidence=0.9)
    track.velocity = (0.0, 10.0)
    result = assessor._assess_single_track(track)
    assert result["ttc_frames"] >= 0
    assert 0.0 <= result["risk_score"] <= 1.0


def test_ttc_not_approaching_is_sentinel():
    assessor = CollisionRiskAssessor()
    track = Track(track_id=1, class_name="car", bbox=[300, 200, 400, 300], confidence=0.9)
    track.velocity = (0.0, -3.0)
    assert assessor._assess_single_track(track)["ttc_frames"] == 999


def test_ttc_pixel_formula():
    assessor = CollisionRiskAssessor(frame_height=480)
    track = Track(track_id=1, class_name="car", bbox=[300, 300, 400, 400], confidence=0.9)
    track.velocity = (0.0, 10.0)
    # (480 - 400) px / 10 px per frame = 8 frames
    assert assessor._assess_single_track(track)["ttc_frames"] == pytest.approx(8.0)


# --------------------------------------------------------------------------
# Distance estimation
# --------------------------------------------------------------------------

def test_pinhole_size_formula_person():
    est = DistanceEstimator()
    # distance = real_height * f / pixel_height = 1.7 * 500 / 85 = 10 m
    assert est._estimate_from_size(85, 30, "person") == pytest.approx(10.0)


def test_flat_ground_perspective_formula():
    est = DistanceEstimator()
    horizon = 480 * 0.4
    y = 312
    expected = 500.0 * 1.2 / (y - horizon)  # f * H / (y - horizon) = 5 m
    assert est.ground_distances[y] == pytest.approx(expected)


def test_tiny_and_zero_boxes_do_not_divide_by_zero():
    est = DistanceEstimator()
    for bbox in ([100, 100, 100, 100], [100, 100, 105, 105]):
        out = est.estimate_distance({"bbox": bbox, "class_name": "car", "confidence": 0.5})
        assert np.isfinite(out["distance_m"]) and out["distance_m"] > 0


def test_following_distance_with_stopped_ego_vehicle():
    """ego_speed_mps=0.0 used to be replaced by the 15 m/s default."""
    analyzer = FollowingDistanceAnalyzer(assumed_speed_mps=15.0)
    result = analyzer.analyze([_car([270, 250, 370, 330])], ego_speed_mps=0.0)
    assert result["following_time_s"] == 999
    assert result["status"] == "SAFE"


def test_following_distance_uses_actual_frame_width():
    """A car dead-ahead in a 1280-wide frame must be the lead vehicle."""
    analyzer = FollowingDistanceAnalyzer()
    result = analyzer.analyze([_car([590, 400, 690, 470])], frame_width=1280)
    assert result["lead_vehicle"] == "car"


def test_following_distance_default_width_unchanged():
    analyzer = FollowingDistanceAnalyzer()
    assert analyzer.analyze([_car([270, 250, 370, 330])])["lead_vehicle"] == "car"
    assert analyzer.analyze([_car([590, 250, 630, 330])])["lead_vehicle"] is None


# --------------------------------------------------------------------------
# Lane detector
# --------------------------------------------------------------------------

def _road(h=480, w=640):
    import cv2

    frame = np.full((h, w, 3), 90, np.uint8)
    cv2.line(frame, (int(w * 0.15), h), (int(w * 0.4), int(h * 0.55)), (255, 255, 255), 8)
    cv2.line(frame, (int(w * 0.85), h), (int(w * 0.6), int(h * 0.55)), (255, 255, 255), 8)
    return frame


def test_separate_lines_accepts_both_opencv_layouts():
    det = LaneDetector()
    segs = np.array([[100, 180, 180, 60], [540, 180, 460, 60]], dtype=np.int32)
    for lines in (segs.reshape(-1, 1, 4), segs):  # OpenCV 4.x and 5.x shapes
        left, right = det._separate_lines(lines, (192, 640))
        assert len(left) == 1 and len(right) == 1


@pytest.mark.parametrize("frame", [
    _road(),
    _road(720, 1280),
    np.zeros((480, 640, 3), np.uint8),  # no lines at all
])
def test_lane_detect_runs(frame):
    out = LaneDetector().detect(frame)
    assert set(out) >= {"left_detected", "right_detected", "lane_center_offset", "confidence"}


# --------------------------------------------------------------------------
# Imports / pipeline smoke test
# --------------------------------------------------------------------------

def test_src_package_imports():
    """app.py does `from src.detector import ...`; that used to raise
    ModuleNotFoundError (caught by app.py, which silently fell back to mock
    mode). Run in a subprocess so src/ is not on sys.path."""
    code = (
        "import src; from src.detector import ObjectDetector; "
        "from src.visualizer_pro import VisualizerPro; VisualizerPro(); "
        "print(src.__author__)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == "sadeeqgandalf"


def test_pipeline_pro_frame_smoke():
    import pipeline_pro

    pipe = pipeline_pro.PerceptionPipelinePro(
        model_path="yolov8n.pt", show_visualization=False, save_output=False
    )
    if pipe.detector.model is not None:
        pytest.skip("ultralytics installed; smoke test targets the mock detector")
    for _ in range(4):
        result = pipe.process_frame(_road(720, 1280))
    assert result["annotated_frame"].shape == (720, 1280, 3)
    assert len(result["tracks"]) == 2


def test_pipeline_pro_risk_assessors_use_video_size(tmp_path):
    """process_video set frame_width but left ego_lane_center at 320 and did
    not touch the visualizer's own assessor, so a car dead-ahead in 1280-wide
    video was reported as out of the ego lane."""
    import cv2
    import pipeline_pro

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    if not writer.isOpened():
        pytest.skip("mp4v encoder unavailable in this OpenCV build")
    for _ in range(3):
        writer.write(_road(720, 1280))
    writer.release()

    pipe = pipeline_pro.PerceptionPipelinePro(
        model_path="yolov8n.pt", show_visualization=False, save_output=False
    )
    pipe.process_video(str(video), str(tmp_path / "out"), max_frames=3)
    for assessor in (pipe.risk_assessor, pipe.visualizer.risk_assessor):
        assert (assessor.frame_width, assessor.frame_height) == (1280, 720)
        assert assessor.ego_lane_center == 640
        dead_ahead = Track(track_id=1, class_name="car", bbox=[590, 400, 690, 470], confidence=0.9)
        assert assessor._assess_single_track(dead_ahead)["in_ego_lane"]

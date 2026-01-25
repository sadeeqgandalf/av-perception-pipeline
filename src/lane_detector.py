"""
Lane Detection Module

This module detects lane lines on the road using a combination of:
1. Classical computer vision (edge detection, color filtering)
2. Polynomial curve fitting
3. Temporal smoothing across frames

HOW LANE DETECTION WORKS (Simple Explanation):
=============================================
1. REGION OF INTEREST: Focus on bottom half of image (where road is)
2. COLOR FILTERING: Find yellow and white pixels (lane colors)
3. EDGE DETECTION: Find sharp changes in brightness (lane edges)
4. LINE DETECTION: Use Hough Transform to find straight lines
5. CURVE FITTING: Fit a polynomial to detected points
6. SMOOTHING: Average with previous frames to reduce jitter

Think of it like this:
- You're looking for the "gutters" of the road
- They're usually white or yellow
- They form smooth curves, not random zigzags
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import deque


class LaneDetector:
    """
    Lane line detector for autonomous driving.
    
    Detects left and right lane boundaries and fits polynomial curves.
    Uses temporal smoothing for stable output.
    
    Attributes:
        history_length: How many frames to average for smoothing
        left_lane_history: Recent left lane detections
        right_lane_history: Recent right lane detections
    
    Example:
        >>> lane_detector = LaneDetector()
        >>> frame = cv2.imread("road.jpg")
        >>> lanes = lane_detector.detect(frame)
        >>> print(f"Left lane detected: {lanes['left_detected']}")
    """
    
    def __init__(self, history_length: int = 5):
        """
        Initialize lane detector.
        
        Args:
            history_length: Number of frames to average for smoothing
                          Higher = smoother but slower to react to changes
        """
        self.history_length = history_length
        
        # Store recent detections for smoothing
        # deque automatically removes oldest when full
        self.left_lane_history = deque(maxlen=history_length)
        self.right_lane_history = deque(maxlen=history_length)
        
        # Lane detection parameters (tuned for typical dashcam)
        self.canny_low = 50      # Lower edge detection threshold
        self.canny_high = 150    # Upper edge detection threshold
        self.hough_threshold = 50  # Min votes for line detection
        
    def detect(self, frame: np.ndarray) -> Dict:
        """
        Detect lane lines in a frame.
        
        Args:
            frame: BGR image (OpenCV format)
        
        Returns:
            Dictionary containing:
                - left_detected: bool, was left lane found?
                - right_detected: bool, was right lane found?
                - left_points: List of (x, y) points for left lane
                - right_points: List of (x, y) points for right lane
                - left_poly: Polynomial coefficients for left lane
                - right_poly: Polynomial coefficients for right lane
                - lane_center_offset: How far car is from lane center (pixels)
                - confidence: Overall detection confidence (0-1)
        """
        height, width = frame.shape[:2]
        
        # Step 1: Define region of interest (bottom portion of image)
        roi_top = int(height * 0.6)  # Start from 60% down
        roi = frame[roi_top:, :]
        
        # Step 2: Convert to different color spaces for detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
        
        # Step 3: Create masks for white and yellow lane lines
        white_mask = self._detect_white_lines(roi, gray)
        yellow_mask = self._detect_yellow_lines(roi, hls)
        
        # Combine masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # Step 4: Edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # Combine edges with color mask
        lane_edges = cv2.bitwise_and(edges, combined_mask)
        
        # Step 5: Hough Line Transform to find lines
        lines = cv2.HoughLinesP(
            lane_edges,
            rho=1,              # Distance resolution in pixels
            theta=np.pi/180,    # Angle resolution in radians
            threshold=self.hough_threshold,
            minLineLength=40,   # Minimum line length
            maxLineGap=100      # Maximum gap between line segments
        )
        
        # Step 6: Separate lines into left and right based on slope
        left_lines, right_lines = self._separate_lines(lines, roi.shape)
        
        # Step 7: Fit polynomials to lane points
        left_poly, left_points = self._fit_lane_polynomial(left_lines, roi.shape)
        right_poly, right_points = self._fit_lane_polynomial(right_lines, roi.shape)
        
        # Step 8: Apply temporal smoothing
        if left_poly is not None:
            self.left_lane_history.append(left_poly)
            left_poly = self._smooth_polynomial(self.left_lane_history)
        
        if right_poly is not None:
            self.right_lane_history.append(right_poly)
            right_poly = self._smooth_polynomial(self.right_lane_history)
        
        # Step 9: Generate output points (adjusted to full frame coordinates)
        left_detected = left_poly is not None
        right_detected = right_poly is not None
        
        # Generate lane points in full frame coordinates
        if left_detected:
            left_points = self._generate_lane_points(left_poly, roi.shape, roi_top)
        else:
            left_points = []
            
        if right_detected:
            right_points = self._generate_lane_points(right_poly, roi.shape, roi_top)
        else:
            right_points = []
        
        # Calculate lane center offset
        lane_center_offset = self._calculate_center_offset(
            left_poly, right_poly, width
        )
        
        # Calculate confidence based on detection quality
        confidence = self._calculate_confidence(
            left_detected, right_detected, 
            len(left_lines) if left_lines else 0,
            len(right_lines) if right_lines else 0
        )
        
        return {
            'left_detected': left_detected,
            'right_detected': right_detected,
            'left_points': left_points,
            'right_points': right_points,
            'left_poly': left_poly,
            'right_poly': right_poly,
            'lane_center_offset': lane_center_offset,
            'confidence': confidence
        }
    
    def _detect_white_lines(self, roi: np.ndarray, gray: np.ndarray) -> np.ndarray:
        """
        Detect white lane lines using brightness threshold.
        
        White lines are simply very bright pixels.
        """
        # Threshold: pixels brighter than 200 (out of 255)
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        return white_mask
    
    def _detect_yellow_lines(self, roi: np.ndarray, hls: np.ndarray) -> np.ndarray:
        """
        Detect yellow lane lines using HLS color space.
        
        HLS = Hue, Lightness, Saturation
        Yellow has specific hue range and high saturation.
        """
        # Yellow color range in HLS
        lower_yellow = np.array([15, 30, 115])
        upper_yellow = np.array([35, 204, 255])
        
        yellow_mask = cv2.inRange(hls, lower_yellow, upper_yellow)
        return yellow_mask
    
    def _separate_lines(
        self, 
        lines: Optional[np.ndarray], 
        shape: Tuple[int, int]
    ) -> Tuple[List, List]:
        """
        Separate detected lines into left and right lane lines.
        
        Uses slope to determine which side:
        - Negative slope = left lane (going up-left in image coords)
        - Positive slope = right lane (going up-right in image coords)
        
        Note: In image coordinates, y increases downward!
        """
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
        
        height, width = shape[:2]
        center_x = width // 2
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip nearly horizontal lines (not lane lines)
            if abs(y2 - y1) < 10:
                continue
            
            # Calculate slope
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Add small value to avoid division by zero
            
            # Filter by slope magnitude (lane lines are typically 30-70 degrees)
            if abs(slope) < 0.5 or abs(slope) > 2.0:
                continue
            
            # Negative slope and on left side = left lane
            if slope < 0 and x1 < center_x and x2 < center_x:
                left_lines.append(line[0])
            # Positive slope and on right side = right lane
            elif slope > 0 and x1 > center_x and x2 > center_x:
                right_lines.append(line[0])
        
        return left_lines, right_lines
    
    def _fit_lane_polynomial(
        self, 
        lines: List, 
        shape: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], List]:
        """
        Fit a 2nd degree polynomial to lane line points.
        
        Why polynomial? Because lanes curve!
        y = ax² + bx + c
        
        Returns (coefficients, points_used)
        """
        if not lines or len(lines) < 2:
            return None, []
        
        # Collect all points from detected lines
        all_x = []
        all_y = []
        
        for x1, y1, x2, y2 in lines:
            all_x.extend([x1, x2])
            all_y.extend([y1, y2])
        
        if len(all_x) < 3:
            return None, []
        
        try:
            # Fit polynomial: x = f(y) because lanes are more vertical
            # This avoids issues with vertical lines
            coefficients = np.polyfit(all_y, all_x, deg=2)
            return coefficients, list(zip(all_x, all_y))
        except np.linalg.LinAlgError:
            return None, []
    
    def _smooth_polynomial(self, history: deque) -> np.ndarray:
        """
        Average polynomial coefficients over recent frames.
        
        This reduces jitter and makes lane lines stable.
        """
        if not history:
            return None
        
        # Stack all coefficient arrays and take mean
        coeffs = np.array(list(history))
        return np.mean(coeffs, axis=0)
    
    def _generate_lane_points(
        self, 
        poly: np.ndarray, 
        roi_shape: Tuple[int, int],
        roi_top: int
    ) -> List[Tuple[int, int]]:
        """
        Generate points along the lane curve for visualization.
        
        Args:
            poly: Polynomial coefficients [a, b, c]
            roi_shape: Shape of the ROI
            roi_top: Y offset of ROI in full frame
        
        Returns:
            List of (x, y) points in full frame coordinates
        """
        height = roi_shape[0]
        
        # Generate y values from top to bottom of ROI
        y_values = np.linspace(0, height - 1, num=10)
        
        # Calculate x values using polynomial
        x_values = np.polyval(poly, y_values)
        
        # Convert to full frame coordinates
        points = []
        for x, y in zip(x_values, y_values):
            full_y = int(y + roi_top)  # Adjust for ROI offset
            points.append((int(x), full_y))
        
        return points
    
    def _calculate_center_offset(
        self, 
        left_poly: Optional[np.ndarray],
        right_poly: Optional[np.ndarray],
        frame_width: int
    ) -> float:
        """
        Calculate how far the car is from lane center.
        
        Positive = car is right of center
        Negative = car is left of center
        
        Assumes camera is mounted at car center.
        """
        if left_poly is None or right_poly is None:
            return 0.0
        
        # Calculate lane positions at bottom of frame (where car is)
        bottom_y = 100  # Arbitrary y value near bottom of ROI
        
        left_x = np.polyval(left_poly, bottom_y)
        right_x = np.polyval(right_poly, bottom_y)
        
        lane_center = (left_x + right_x) / 2
        frame_center = frame_width / 2
        
        offset = frame_center - lane_center
        return offset
    
    def _calculate_confidence(
        self, 
        left_detected: bool,
        right_detected: bool,
        left_line_count: int,
        right_line_count: int
    ) -> float:
        """
        Calculate overall detection confidence.
        
        Based on:
        - Whether both lanes were detected
        - How many line segments support each lane
        """
        base_confidence = 0.0
        
        if left_detected:
            base_confidence += 0.4
        if right_detected:
            base_confidence += 0.4
        
        # Bonus for having multiple supporting lines
        line_confidence = min(0.2, (left_line_count + right_line_count) / 20)
        
        return min(1.0, base_confidence + line_confidence)
    
    def reset(self):
        """Clear lane history (use when scene changes dramatically)."""
        self.left_lane_history.clear()
        self.right_lane_history.clear()


# Quick test when run directly
if __name__ == "__main__":
    print("Testing LaneDetector...")
    
    # Create detector
    detector = LaneDetector()
    
    # Create a test image with fake lane lines
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw some white "lane lines"
    cv2.line(test_image, (200, 480), (280, 300), (255, 255, 255), 10)
    cv2.line(test_image, (440, 480), (360, 300), (255, 255, 255), 10)
    
    # Run detection
    result = detector.detect(test_image)
    
    print(f"\nResults:")
    print(f"  Left lane detected: {result['left_detected']}")
    print(f"  Right lane detected: {result['right_detected']}")
    print(f"  Confidence: {result['confidence']:.2f}")
    print(f"  Center offset: {result['lane_center_offset']:.1f} pixels")

import cv2
from typing import Dict, List, Set, Optional
from .geometry import Point
from .tracking import TrackingState

class LineCounter:
    def __init__(self):
        self.counted_ids: Set[int] = set()
        self.line_x_position = 0.5  # Vertical line at 50% of width
        self.line_y_position = 0.5  # Horizontal line at 50% of height
        self.tracking_state = TrackingState()
        self.counts = {'line1': 0, 'line2': 0}
        self.latest_crossings = {'line1': None, 'line2': None}

    def draw_counting_lines(self, frame: cv2.Mat) -> cv2.Mat:
        """Draw the counting lines on the frame"""
        height, width = frame.shape[:2]
        
        # Draw vertical counting line (yellow)
        line_x = int(width * self.line_x_position)
        cv2.line(frame, (line_x, 0), (line_x, height), (0, 255, 255), 2)
        
        # Draw horizontal separator line (white)
        line_y = int(height * self.line_y_position)
        cv2.line(frame, (0, line_y), (width, line_y), (255, 255, 255), 2)
        
        # Add labels
        cv2.putText(frame, "Line 1", (10, int(height * 0.25)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Line 2", (10, int(height * 0.75)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def update_counts(self, detections: List[Dict]) -> None:
        """Update part counts based on detected objects crossing the line"""
        for detection in detections:
            track_id = detection['track_id']
            current_pos = self._get_detection_center(detection)
            
            if self.tracking_state.has_previous_position(track_id):
                prev_pos = self.tracking_state.get_previous_position(track_id)
                if self._has_crossed_line(prev_pos, current_pos):
                    self._process_line_crossing(track_id, detection, current_pos)
            
            self.tracking_state.update_position(track_id, current_pos)

    def _get_detection_center(self, detection: Dict) -> Point:
        """Calculate center point of detection box"""
        x1, y1, x2, y2 = detection['box']
        return Point((x1 + x2) / 2, (y1 + y2) / 2)

    def _has_crossed_line(self, prev_pos: Point, current_pos: Point) -> bool:
        """Check if movement between points crosses the vertical counting line"""
        return (prev_pos.x < self.line_x_position and current_pos.x >= self.line_x_position) or \
               (prev_pos.x > self.line_x_position and current_pos.x <= self.line_x_position)

    def _process_line_crossing(self, track_id: int, detection: Dict, position: Point) -> None:
        """Process a line crossing event"""
        if track_id not in self.counted_ids:
            # Determine which line based on y-position
            line_key = 'line1' if position.y < self.line_y_position else 'line2'
            
            # Update counts and store crossing information
            self.counts[line_key] += 1
            self.latest_crossings[line_key] = {
                'class_name': detection['class_name'],
                'timestamp': cv2.getTickCount()
            }
            
            self.counted_ids.add(track_id)

    def get_counts(self) -> Dict[str, int]:
        """Get current counts for both lines"""
        return self.counts.copy()

    def get_latest_crossings(self) -> Dict[str, Optional[Dict]]:
        """Get information about the latest crossings for each line"""
        return self.latest_crossings.copy()

    def reset(self) -> None:
        """Reset all counting data"""
        self.counted_ids.clear()
        self.tracking_state.reset()
        self.counts = {'line1': 0, 'line2': 0}
        self.latest_crossings = {'line1': None, 'line2': None}
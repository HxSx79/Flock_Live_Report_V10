import cv2
from ultralytics import YOLO
from typing import Dict, List
from .config import Config
from .line_counter import LineCounter
from .production_tracker import ProductionTracker

class ObjectDetector:
    def __init__(self):
        self.config = Config()
        self.model = YOLO(self.config.model_path)
        self.model.conf = self.config.confidence_threshold
        self.names = self.model.model.names
        self.line_counter = LineCounter()
        self.production_tracker = ProductionTracker()

    def process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        if frame is None:
            return frame
            
        frame = cv2.resize(frame, (self.config.frame_width, self.config.frame_height))
        
        # Draw counting lines first
        frame = self.line_counter.draw_counting_lines(frame)
        
        # Run detection
        results = self.model.track(frame, persist=True)
        detections = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = self.names[class_id]
                x1, y1, x2, y2 = box
                
                detection = {
                    'class_name': class_name,
                    'track_id': int(track_id),
                    'box': box
                }
                detections.append(detection)
                
                # Draw detection box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_name}', (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update line counter with new detections
        self.line_counter.update_counts(detections)
        
        # Update production tracker
        self.production_tracker.update_production(
            self.line_counter.get_counts(),
            self.line_counter.get_latest_crossings()
        )

        return frame
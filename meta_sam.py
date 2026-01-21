#!/usr/bin/env python3
"""
Computer Vision Based Box Detection
Uses traditional CV methods to detect cardboard boxes
Optimized for 30 FPS with processing only every 30th frame
"""

import cv2
import numpy as np
import time
from datetime import datetime
import argparse
from collections import defaultdict, deque

# Try to import YOLO for person detection only
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available. Run: pip install ultralytics")
    exit(1)


class CVBoxDetector:
    """Computer Vision based box detector - detects rectangular cardboard boxes"""
    
    def __init__(self, process_interval=30):
        print(f"\n{'='*60}")
        print("INITIALIZING CV BOX DETECTOR")
        print(f"{'='*60}")
        
        # Load YOLO for person detection only
        print("Loading YOLO11n for person detection...")
        self.yolo = YOLO("yolo11n.pt")
        print("✅ YOLO loaded")
        
        self.process_interval = process_interval
        self.frame_count = 0
        self.detection_log = []
        self.fps_history = deque(maxlen=30)
        
        # CV parameters for box detection
        self.min_box_area = 2000  # Minimum area for a box
        self.max_box_area = 100000  # Maximum area for a box
        self.min_aspect_ratio = 0.3  # Min width/height ratio
        self.max_aspect_ratio = 3.0  # Max width/height ratio
        self.edge_threshold1 = 50
        self.edge_threshold2 = 150
        
        # Matching parameters
        self.iou_threshold = 0.2
        self.vertical_tolerance = 1.5
        self.horizontal_tolerance = 0.9
        
        # Cache last detection to display between processing frames
        self.last_persons = []
        self.last_boxes = []
        self.last_matches = {}
        
        print(f"Processing every {process_interval} frames")
        print(f"Box detection: Area {self.min_box_area}-{self.max_box_area}")
        print(f"{'='*60}\n")
    
    def detect_boxes_cv(self, frame, persons):
        """
        Detect cardboard boxes using computer vision
        Looks for rectangular contours in the carrying zones of detected persons
        """
        boxes = []
        
        # For each person, analyze their carrying zone
        for person_idx, person in enumerate(persons):
            px1, py1, px2, py2 = person["bbox"]
            person_height = py2 - py1
            person_width = px2 - px1
            
            # Define carrying zone (lower body area with margins)
            zone_top = max(0, py1 + int(person_height * 0.3))
            zone_bottom = min(frame.shape[0], py2 + int(person_height * 0.5))
            person_center_x = (px1 + px2) // 2
            horizontal_range = int(person_width * self.horizontal_tolerance)
            zone_left = max(0, person_center_x - horizontal_range)
            zone_right = min(frame.shape[1], person_center_x + horizontal_range)
            
            # Extract carrying zone ROI
            roi = frame[zone_top:zone_bottom, zone_left:zone_right]
            
            if roi.size == 0:
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Apply GaussianBlur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, self.edge_threshold1, self.edge_threshold2)
            
            # Morphological operations to close gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contours
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if area < self.min_box_area or area > self.max_box_area:
                    continue
                
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Convert back to full frame coordinates
                bx1 = zone_left + x
                by1 = zone_top + y
                bx2 = bx1 + w
                by2 = by1 + h
                
                # Check aspect ratio (boxes are somewhat rectangular)
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                    continue
                
                # Check rectangularity (how much the contour fills its bounding box)
                rectangularity = area / (w * h) if (w * h) > 0 else 0
                if rectangularity < 0.6:  # Should be fairly rectangular
                    continue
                
                # Approximate polygon to check if it's box-like
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Boxes typically have 4-8 vertices when approximated
                if len(approx) < 4 or len(approx) > 12:
                    continue
                
                # Calculate confidence based on rectangularity and area
                confidence = min(1.0, rectangularity * 0.5 + (area / self.max_box_area) * 0.5)
                
                boxes.append({
                    "bbox": [bx1, by1, bx2, by2],
                    "area": area,
                    "conf": confidence,
                    "aspect_ratio": aspect_ratio,
                    "rectangularity": rectangularity,
                    "vertices": len(approx),
                    "person_idx": person_idx  # Which person's zone this came from
                })
        
        return boxes
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        xi_min = max(x1_min, x2_min)
        yi_min = max(y1_min, y2_min)
        xi_max = min(x1_max, x2_max)
        yi_max = min(y1_max, y2_max)
        
        if xi_max < xi_min or yi_max < yi_min:
            return 0.0
        
        intersection = (xi_max - xi_min) * (yi_max - yi_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def match_boxes_to_persons(self, persons, boxes):
        """Match detected boxes to persons"""
        person_box_map = defaultdict(list)
        
        for box in boxes:
            box_bbox = box["bbox"]
            box_center_x = (box_bbox[0] + box_bbox[2]) / 2
            box_center_y = (box_bbox[1] + box_bbox[3]) / 2
            
            best_person = None
            best_score = 0
            
            for person_idx, person in enumerate(persons):
                px1, py1, px2, py2 = person["bbox"]
                person_width = px2 - px1
                person_height = py2 - py1
                person_center_x = (px1 + px2) / 2
                
                # Calculate IoU
                iou = self.calculate_iou(person["bbox"], box_bbox)
                
                # Check spatial proximity
                horizontal_dist = abs(box_center_x - person_center_x)
                horizontal_range = person_width * self.horizontal_tolerance
                
                # Check vertical position (in carrying zone)
                carrying_zone_top = py1 + (person_height * 0.3)
                carrying_zone_bottom = py2 + (person_height * (self.vertical_tolerance - 1.0))
                in_zone = carrying_zone_top < box_center_y < carrying_zone_bottom
                
                # Calculate match score
                if iou > self.iou_threshold or (horizontal_dist < horizontal_range and in_zone):
                    # Prefer the person from whose zone this box was detected
                    zone_bonus = 0.3 if box.get("person_idx") == person_idx else 0
                    proximity_score = 1.0 - (horizontal_dist / horizontal_range) if horizontal_range > 0 else 0
                    score = iou * 0.4 + proximity_score * 0.4 + zone_bonus + box["conf"] * 0.2
                    
                    if score > best_score:
                        best_score = score
                        best_person = person_idx
            
            if best_person is not None:
                person_box_map[best_person].append({
                    **box,
                    "match_score": best_score
                })
        
        return person_box_map
    
    def detect_persons(self, frame):
        """Detect persons using YOLO"""
        results = self.yolo(frame, conf=0.4, imgsz=416, verbose=False)[0]
        
        persons = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # Person class
                xyxy = box.xyxy[0].cpu().numpy()
                bbox = [int(x) for x in xyxy]
                conf = float(box.conf[0])
                persons.append({"bbox": bbox, "conf": conf})
        
        return persons
    
    def process_frame(self, frame, force_process=False):
        """Process frame - only do heavy processing every N frames"""
        start_time = time.time()
        
        # Always detect persons (fast)
        persons = self.detect_persons(frame)
        
        # Only do box detection every N frames
        should_process = (self.frame_count % self.process_interval == 0) or force_process
        
        if should_process:
            # Heavy processing: detect boxes
            boxes = self.detect_boxes_cv(frame, persons)
            matches = self.match_boxes_to_persons(persons, boxes)
            
            # Cache results
            self.last_persons = persons
            self.last_boxes = boxes
            self.last_matches = matches
            
            # Log
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            print(f"\n{'='*60}")
            print(f"FRAME {self.frame_count} | {timestamp} | PROCESSED")
            print(f"{'='*60}")
            print(f"Persons: {len(persons)} | Boxes detected: {len(boxes)}")
            
            for person_idx, person in enumerate(persons):
                px1, py1, px2, py2 = person["bbox"]
                held_boxes = matches.get(person_idx, [])
                
                print(f"\nPerson {person_idx + 1}:")
                print(f"  BBox: [{px1}, {py1}] -> [{px2}, {py2}]")
                print(f"  Size: {px2-px1}x{py2-py1}")
                print(f"  Boxes held: {len(held_boxes)}")
                
                for box_idx, box in enumerate(held_boxes):
                    bx1, by1, bx2, by2 = box["bbox"]
                    print(f"    Box {box_idx + 1}:")
                    print(f"      BBox: [{bx1}, {by1}] -> [{bx2}, {by2}]")
                    print(f"      Size: {bx2-bx1}x{by2-by1}")
                    print(f"      Conf: {box['conf']:.3f}")
                    print(f"      Area: {box['area']}")
                    print(f"      Aspect: {box['aspect_ratio']:.2f}")
                    print(f"      Match: {box['match_score']:.3f}")
            
            elapsed = (time.time() - start_time) * 1000
            print(f"\nProcessing: {elapsed:.1f}ms")
            print(f"{'='*60}")
            
            self.detection_log.append({
                "frame": self.frame_count,
                "timestamp": timestamp,
                "persons": len(persons),
                "boxes": len(boxes),
                "matched": sum(len(b) for b in matches.values()),
                "processing_time_ms": elapsed
            })
        else:
            # Use cached results but update persons
            self.last_persons = persons
            elapsed = (time.time() - start_time) * 1000
        
        # Calculate FPS
        fps = 1000.0 / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)
        
        self.frame_count += 1
        
        return self.last_persons, self.last_boxes, self.last_matches
    
    def draw_detections(self, frame, persons, boxes, matches):
        """Draw detections on frame"""
        display = frame.copy()
        h, w = display.shape[:2]
        
        # Draw persons with carrying zones
        person_colors = [
            (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255)
        ]
        
        for person_idx, person in enumerate(persons):
            px1, py1, px2, py2 = person["bbox"]
            color = person_colors[person_idx % len(person_colors)]
            
            # Person box
            cv2.rectangle(display, (px1, py1), (px2, py2), color, 2)
            
            # Carrying zone
            person_height = py2 - py1
            person_width = px2 - px1
            person_center_x = (px1 + px2) // 2
            
            zone_top = py1 + int(person_height * 0.3)
            zone_bottom = py2 + int(person_height * (self.vertical_tolerance - 1.0))
            horizontal_range = int(person_width * self.horizontal_tolerance)
            zone_left = person_center_x - horizontal_range
            zone_right = person_center_x + horizontal_range
            
            # Draw zone (semi-transparent)
            overlay = display.copy()
            zone_color = tuple([int(c * 0.5) for c in color])
            cv2.rectangle(overlay, (zone_left, zone_top), (zone_right, zone_bottom),
                         zone_color, -1)
            display = cv2.addWeighted(overlay, 0.15, display, 0.85, 0)
            cv2.rectangle(display, (zone_left, zone_top), (zone_right, zone_bottom),
                         color, 1, cv2.LINE_AA)
            
            # Label
            held_boxes = matches.get(person_idx, [])
            label = f"Person {person_idx + 1}"
            if held_boxes:
                label += f" | {len(held_boxes)} box(es)"
            
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display, (px1, py1 - th - 10), (px1 + tw + 6, py1), color, -1)
            cv2.putText(display, label, (px1 + 3, py1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw boxes
        matched_box_ids = set()
        for person_idx, person_boxes in matches.items():
            color = person_colors[person_idx % len(person_colors)]
            for box in person_boxes:
                matched_box_ids.add(id(box))
                bx1, by1, bx2, by2 = box["bbox"]
                
                # Box rectangle (thick for matched)
                cv2.rectangle(display, (bx1, by1), (bx2, by2), color, 3)
                
                # Box label
                label = f"Box {box['conf']:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(display, (bx1, by1 - th - 6), (bx1 + tw + 4, by1), color, -1)
                cv2.putText(display, label, (bx1 + 2, by1 - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw unmatched boxes in gray
        for box in boxes:
            if id(box) not in matched_box_ids:
                bx1, by1, bx2, by2 = box["bbox"]
                cv2.rectangle(display, (bx1, by1), (bx2, by2), (128, 128, 128), 1)
                label = f"? {box['conf']:.2f}"
                cv2.putText(display, label, (bx1, by1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        
        # Info overlay
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        next_process = self.process_interval - (self.frame_count % self.process_interval)
        
        info = [
            f"Frame: {self.frame_count}",
            f"FPS: {avg_fps:.1f}",
            f"Persons: {len(persons)}",
            f"Boxes: {len(boxes)}",
            f"Matched: {sum(len(b) for b in matches.values())}",
            f"Next scan: {next_process}f",
            f"Time: {datetime.now().strftime('%H:%M:%S')}"
        ]
        
        overlay = display.copy()
        cv2.rectangle(overlay, (5, 5), (300, 230), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)
        
        y = 30
        for text in info:
            cv2.putText(display, text, (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y += 30
        
        return display
    
    def print_summary(self):
        """Print summary"""
        if not self.detection_log:
            return
        
        print(f"\n{'='*60}")
        print("DETECTION SUMMARY")
        print(f"{'='*60}")
        
        total = len(self.detection_log)
        avg_persons = sum(log["persons"] for log in self.detection_log) / total
        avg_boxes = sum(log["boxes"] for log in self.detection_log) / total
        avg_matched = sum(log["matched"] for log in self.detection_log) / total
        avg_time = sum(log["processing_time_ms"] for log in self.detection_log) / total
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        
        print(f"Processed frames: {total}")
        print(f"Total frames: {self.frame_count}")
        print(f"Process ratio: 1/{self.process_interval}")
        print(f"Avg persons/frame: {avg_persons:.2f}")
        print(f"Avg boxes/frame: {avg_boxes:.2f}")
        print(f"Avg matched/frame: {avg_matched:.2f}")
        print(f"Avg processing time: {avg_time:.1f}ms")
        print(f"Avg FPS: {avg_fps:.1f}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='CV Box Detection Tester')
    parser.add_argument('--rtsp', type=str,
                       default="rtsp://fasspay:fasspay2025@192.168.122.127:554/stream1",
                       help='RTSP stream URL')
    parser.add_argument('--interval', type=int, default=30,
                       help='Process every N frames (default: 30)')
    parser.add_argument('--no-display', action='store_true',
                       help='Headless mode')
    args = parser.parse_args()
    
    # Initialize detector
    detector = CVBoxDetector(process_interval=args.interval)
    
    # Connect to stream
    print(f"Connecting to: {args.rtsp}")
    cap = cv2.VideoCapture(args.rtsp)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Failed to open stream")
        return
    
    print("✅ Stream connected")
    print(f"Processing interval: Every {args.interval} frames")
    print("\nControls:")
    print("  Q - Quit")
    print("  SPACE - Pause/Resume")
    print("  S - Save frame")
    print("  P - Force process current frame\n")
    
    if not args.no_display:
        cv2.namedWindow("CV Box Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CV Box Detection", 1280, 720)
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Stream lost, retrying...")
                    time.sleep(2)
                    cap = cv2.VideoCapture(args.rtsp)
                    continue
                
                # Process frame
                persons, boxes, matches = detector.process_frame(frame)
                
                if not args.no_display:
                    display = detector.draw_detections(frame, persons, boxes, matches)
                    cv2.imshow("CV Box Detection", display)
            
            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    print(f"\n{'PAUSED' if paused else 'RESUMED'}\n")
                elif key == ord('s'):
                    filename = f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, display)
                    print(f"\n✅ Saved: {filename}\n")
                elif key == ord('p'):
                    print("\n🔄 Force processing...\n")
                    persons, boxes, matches = detector.process_frame(frame, force_process=True)
    
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user")
    
    finally:
        detector.print_summary()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CV BOX DETECTION - CARDBOARD BOX DETECTOR")
    print("Uses computer vision to detect boxes")
    print("Optimized for 30 FPS with periodic processing")
    print("="*60 + "\n")
    main()
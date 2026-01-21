#!/usr/bin/env python3
"""
Enhanced CCTV with Improved Multi-Person Entry/Exit Tracking
Key improvements:
1. Zone-based tracking (entry/middle/exit zones)
2. Individual person state machines
3. Confidence-based direction confirmation
4. Better handling of simultaneous movements
"""

from collections import defaultdict, deque
import cv2
import sys
import os
import time
import threading
from queue import Queue, Empty
import numpy as np
import gc
import argparse
from datetime import datetime

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Configuration
PROCESS_RESOLUTION = 720
YOLO_IMG_SIZE = 320
MAX_DRAWN_PERSONS = 10

# Enhanced tracking parameters
ZONE_RATIO = 0.2  # Top/bottom 20% are entry/exit zones
MIN_MOVEMENT = 40  # Minimum pixels to register movement
DIRECTION_FRAMES = 4  # Frames needed to confirm direction
POSITION_HISTORY = 10  # Frames to keep for tracking

# Box detection
BOX_COLOR_LOWER = np.array([10, 50, 50])
BOX_COLOR_UPPER = np.array([30, 255, 255])
MIN_BOX_AREA = 1000
MAX_BOX_AREA = 50000

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("❌ YOLO required: pip install ultralytics")
    sys.exit(1)

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max < xi_min or yi_max < yi_min:
        return 0.0
    
    intersection = (xi_max - xi_min) * (yi_max - yi_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0

def is_box_held(person_bbox, box_bbox):
    iou = calculate_iou(person_bbox, box_bbox)
    px1, py1, px2, py2 = person_bbox
    bx1, by1, bx2, by2 = box_bbox
    
    box_cx = (bx1 + bx2) / 2
    box_cy = (by1 + by2) / 2
    
    in_x = px1 < box_cx < px2
    in_y = py1 < box_cy < py2
    carry_zone = box_cy > py1 + (py2 - py1) * 0.3
    
    return iou > 0.3 or (in_x and in_y and carry_zone)

class EnhancedDirectionTracker:
    """Multi-person zone-based direction tracker"""
    
    def __init__(self, frame_height):
        self.frame_height = frame_height
        self.entry_y = int(frame_height * ZONE_RATIO)
        self.exit_y = int(frame_height * (1 - ZONE_RATIO))
        
        # Per-person tracking state
        self.tracks = defaultdict(lambda: {
            'positions': deque(maxlen=POSITION_HISTORY),
            'directions': deque(maxlen=DIRECTION_FRAMES),
            'zone_transitions': [],
            'confirmed_dir': None,
            'last_event': None,
            'event_triggered': False
        })
        
        print(f"✅ Zone Tracker: Entry(0-{self.entry_y}) Middle({self.entry_y}-{self.exit_y}) Exit({self.exit_y}-{frame_height})")
    
    def get_zone(self, y):
        if y < self.entry_y:
            return 'ENTRY'
        elif y > self.exit_y:
            return 'EXIT'
        return 'MIDDLE'
    
    def update(self, person_id, bbox):
        track = self.tracks[person_id]
        x1, y1, x2, y2 = bbox
        
        # Use bottom of bbox for more stable tracking
        tracking_y = y2
        zone = self.get_zone(tracking_y)
        
        # Store position
        track['positions'].append({
            'y': tracking_y,
            'zone': zone,
            'time': time.time(),
            'bbox': bbox
        })
        
        # Need at least 3 positions for direction
        if len(track['positions']) < 3:
            return {
                'direction': 'UNKNOWN',
                'zone': zone,
                'event': None,
                'confidence': 0.0
            }
        
        # Calculate movement
        positions = list(track['positions'])
        old_pos = positions[-DIRECTION_FRAMES if len(positions) >= DIRECTION_FRAMES else 0]
        new_pos = positions[-1]
        
        dy = new_pos['y'] - old_pos['y']
        
        # Determine direction
        if abs(dy) > MIN_MOVEMENT:
            direction = 'EXITING' if dy > 0 else 'ENTERING'
            track['directions'].append(direction)
            
            # Confirm if consistent
            if len(track['directions']) >= DIRECTION_FRAMES:
                recent = list(track['directions'])
                if all(d == recent[0] for d in recent):
                    track['confirmed_dir'] = recent[0]
        
        # Detect zone transitions
        if len(positions) >= 2:
            prev_zone = positions[-2]['zone']
            curr_zone = positions[-1]['zone']
            
            if prev_zone != curr_zone:
                transition = f"{prev_zone}->{curr_zone}"
                track['zone_transitions'].append(transition)
        
        # Event detection with state machine
        event = None
        
        if track['confirmed_dir'] and not track['event_triggered']:
            # Check zone transitions for events
            if len(track['zone_transitions']) >= 2:
                transitions = track['zone_transitions'][-2:]
                
                # Complete exit: MIDDLE->EXIT
                if track['confirmed_dir'] == 'EXITING':
                    if 'MIDDLE->EXIT' in transitions:
                        event = 'EXITED'
                        track['event_triggered'] = True
                        track['last_event'] = 'EXITED'
                
                # Complete entry: ENTRY->MIDDLE
                elif track['confirmed_dir'] == 'ENTERING':
                    if 'ENTRY->MIDDLE' in transitions:
                        event = 'ENTERED'
                        track['event_triggered'] = True
                        track['last_event'] = 'ENTERED'
        
        # Reset event flag when direction changes
        if track['confirmed_dir'] and len(track['directions']) >= 2:
            if track['directions'][-1] != track['directions'][-2]:
                track['event_triggered'] = False
        
        # Calculate confidence
        confidence = 0.0
        if track['confirmed_dir']:
            confidence = len([d for d in track['directions'] if d == track['confirmed_dir']]) / DIRECTION_FRAMES
        
        return {
            'direction': track['confirmed_dir'] or 'STATIONARY',
            'zone': zone,
            'event': event,
            'confidence': min(confidence, 1.0),
            'movement': dy
        }
    
    def cleanup(self, active_ids, max_age=5.0):
        current = time.time()
        to_remove = []
        
        for pid, track in self.tracks.items():
            if pid not in active_ids and track['positions']:
                last_seen = track['positions'][-1]['time']
                if current - last_seen > max_age:
                    to_remove.append(pid)
        
        for pid in to_remove:
            del self.tracks[pid]

class BoxDetector:
    def __init__(self):
        self.box_classes = {
            24: "backpack", 26: "handbag", 28: "suitcase",
            39: "bottle", 63: "laptop", 67: "phone"
        }
    
    def detect_color_boxes(self, frame, person_bbox):
        px1, py1, px2, py2 = person_bbox
        h, w = frame.shape[:2]
        
        x1 = max(0, px1 - 20)
        y1 = max(0, py1 - 20)
        x2 = min(w, px2 + 20)
        y2 = min(h, py2 + 20)
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return []
        
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BOX_COLOR_LOWER, BOX_COLOR_UPPER)
        
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_BOX_AREA < area < MAX_BOX_AREA:
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                aspect = bw / bh if bh > 0 else 0
                if 0.3 < aspect < 3.0:
                    boxes.append([x1 + x, y1 + y, x1 + x + bw, y1 + y + bh])
        
        return boxes
    
    def detect_boxes(self, frame, persons, yolo_boxes, scale=1.0):
        results = {}
        
        for idx, person in enumerate(persons):
            bbox = person["bbox"]
            if scale != 1.0:
                bbox = [int(c / scale) for c in bbox]
            
            held = []
            
            # YOLO boxes
            for box in yolo_boxes:
                box_bbox = box["bbox"]
                if scale != 1.0:
                    box_bbox = [int(c / scale) for c in box_bbox]
                
                if is_box_held(bbox, box_bbox):
                    held.append({
                        "bbox": box_bbox,
                        "name": self.box_classes.get(box["class"], "object"),
                        "conf": box["conf"]
                    })
            
            # Color detection
            color_boxes = self.detect_color_boxes(frame, bbox)
            for cb in color_boxes:
                if not any(calculate_iou(cb, h["bbox"]) > 0.4 for h in held):
                    held.append({
                        "bbox": cb,
                        "name": "cardboard_box",
                        "conf": 0.7
                    })
            
            results[idx] = {
                "count": len(held),
                "boxes": held
            }
        
        return results

class PersonTracker:
    def __init__(self, frame_height):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = 20
        
        self.names = {}
        self.bboxes = {}
        self.box_counts = {}
        self.directions = {}
        
        self.direction_tracker = EnhancedDirectionTracker(frame_height)
        self.box_detector = BoxDetector()
        
        print("✅ PersonTracker initialized")
    
    def register(self, centroid, bbox, box_count=0):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.names[self.next_id] = "Unknown"
        self.bboxes[self.next_id] = bbox
        self.box_counts[self.next_id] = box_count
        self.next_id += 1
    
    def deregister(self, oid):
        for d in [self.objects, self.disappeared, self.names, self.bboxes, 
                  self.box_counts, self.directions]:
            d.pop(oid, None)
    
    def update(self, detections, box_results, scale=1.0):
        if not detections:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return
        
        # Convert to original coordinates
        curr_centroids = []
        curr_bboxes = []
        curr_counts = []
        
        for idx, det in enumerate(detections):
            bbox_proc = det["bbox"]
            bbox_orig = [int(c / scale) for c in bbox_proc]
            
            cx = (bbox_orig[0] + bbox_orig[2]) // 2
            cy = (bbox_orig[1] + bbox_orig[3]) // 2
            
            curr_centroids.append((cx, cy))
            curr_bboxes.append(bbox_orig)
            curr_counts.append(box_results.get(idx, {}).get("count", 0))
        
        # Match or register
        if not self.objects:
            for i in range(len(curr_centroids)):
                self.register(curr_centroids[i], curr_bboxes[i], curr_counts[i])
        else:
            matched_curr = set()
            matched_exist = set()
            
            for oid in list(self.objects.keys()):
                exist_cent = self.objects[oid]
                best_idx = -1
                best_dist = float('inf')
                
                for j, curr_cent in enumerate(curr_centroids):
                    if j in matched_curr:
                        continue
                    
                    dist = np.sqrt((exist_cent[0] - curr_cent[0])**2 + 
                                  (exist_cent[1] - curr_cent[1])**2)
                    
                    if dist < 150 and dist < best_dist:
                        best_dist = dist
                        best_idx = j
                
                if best_idx != -1:
                    self.objects[oid] = curr_centroids[best_idx]
                    self.disappeared[oid] = 0
                    self.bboxes[oid] = curr_bboxes[best_idx]
                    self.box_counts[oid] = curr_counts[best_idx]
                    matched_curr.add(best_idx)
                    matched_exist.add(oid)
                else:
                    self.disappeared[oid] += 1
            
            for j in range(len(curr_centroids)):
                if j not in matched_curr:
                    self.register(curr_centroids[j], curr_bboxes[j], curr_counts[j])
            
            for oid in list(self.objects.keys()):
                if oid not in matched_exist:
                    self.disappeared[oid] += 1
                    if self.disappeared[oid] > self.max_disappeared:
                        self.deregister(oid)
        
        # Update directions
        for oid in list(self.objects.keys()):
            if oid in self.bboxes:
                result = self.direction_tracker.update(oid, self.bboxes[oid])
                self.directions[oid] = result
        
        self.direction_tracker.cleanup(set(self.objects.keys()))
        
        return self.objects

class AIWorker:
    def __init__(self, detector, tracker):
        self.detector = detector
        self.tracker = tracker
        self.box_detector = BoxDetector()
        
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.running = True
        self.avg_time = 0
    
    def start(self):
        threading.Thread(target=self._worker, daemon=True).start()
    
    def submit(self, frame):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
        
        orig_shape = frame.shape[:2]
        
        if frame.shape[1] > PROCESS_RESOLUTION:
            scale = PROCESS_RESOLUTION / frame.shape[1]
            h = int(frame.shape[0] * scale)
            small = cv2.resize(frame, (PROCESS_RESOLUTION, h))
        else:
            small = frame.copy()
            scale = 1.0
        
        self.frame_queue.put((small, orig_shape, scale))
    
    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
    
    def _worker(self):
        print("🚀 AI Worker running")
        
        while self.running:
            try:
                data = self.frame_queue.get(timeout=1)
                frame, orig_shape, scale = data
                
                start = time.time()
                
                # YOLO detection
                results = self.detector(frame, conf=0.3, imgsz=YOLO_IMG_SIZE, verbose=False)[0]
                
                persons = []
                boxes = []
                
                for box in results.boxes:
                    cls = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    bbox = [int(x) for x in xyxy]
                    
                    if cls == 0:
                        persons.append({"bbox": bbox, "conf": float(box.conf[0])})
                    elif cls in [24, 26, 28, 39, 63, 67]:
                        boxes.append({"bbox": bbox, "class": cls, "conf": float(box.conf[0])})
                
                # Box detection
                box_results = self.box_detector.detect_boxes(frame, persons, boxes, scale)
                
                elapsed = (time.time() - start) * 1000
                self.avg_time = self.avg_time * 0.9 + elapsed * 0.1
                
                result = {
                    "persons": persons,
                    "box_results": box_results,
                    "time": elapsed,
                    "scale": scale
                }
                
                if self.result_queue.full():
                    self.result_queue.get()
                self.result_queue.put(result)
                
            except Empty:
                continue
            except Exception as e:
                print(f"AI Error: {e}")

class EnhancedCCTV:
    def __init__(self, rtsp_url, headless=False):
        self.rtsp_url = rtsp_url
        self.headless = headless
        
        print("⏳ Initializing...")
        
        # Get frame dimensions first
        cap = cv2.VideoCapture(rtsp_url)
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot connect to stream")
            sys.exit(1)
        
        frame_height = frame.shape[0]
        cap.release()
        
        # Initialize components
        self.tracker = PersonTracker(frame_height)
        self.detector = YOLO("yolo11n.pt")
        self.ai_worker = AIWorker(self.detector, self.tracker)
        
        self.running = True
        self.latest_result = None
        self.events_log = []
        
        print("\n" + "="*60)
        print("ENHANCED MULTI-PERSON TRACKING CCTV")
        print("="*60)
        print(f"Frame height: {frame_height}px")
        print(f"Entry zone: 0-{int(frame_height * ZONE_RATIO)}px")
        print(f"Exit zone: {int(frame_height * (1-ZONE_RATIO))}-{frame_height}px")
        print("="*60)
    
    def log_event(self, person_id, event, direction, boxes):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        name = self.tracker.names.get(person_id, "Unknown")
        
        msg = f"🚪 [{timestamp}] Person {person_id} ({name}) {event} - {direction} with {boxes} box(es)"
        print(msg)
        
        self.events_log.append({
            'time': timestamp,
            'person_id': person_id,
            'name': name,
            'event': event,
            'direction': direction,
            'boxes': boxes
        })
    
    def start(self):
        print(f"\n📡 Connecting to: {self.rtsp_url}")
        
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.ai_worker.start()
        
        last_submit = 0
        last_events = {}
        
        if not self.headless:
            cv2.namedWindow("Enhanced CCTV", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Enhanced CCTV", 960, 540)
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Reconnecting...")
                    time.sleep(1)
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue
                
                now = time.time()
                
                # Submit for AI (3 FPS)
                if now - last_submit > 0.33:
                    self.ai_worker.submit(frame.copy())
                    last_submit = now
                
                # Get results
                res = self.ai_worker.get_result()
                if res:
                    self.latest_result = res
                    self.tracker.update(res["persons"], res["box_results"], res["scale"])
                    
                    # Check for events
                    for oid in list(self.tracker.objects.keys()):
                        if oid in self.tracker.directions:
                            dir_info = self.tracker.directions[oid]
                            event = dir_info.get('event')
                            
                            if event and last_events.get(oid) != event:
                                boxes = self.tracker.box_counts.get(oid, 0)
                                self.log_event(oid, event, dir_info['direction'], boxes)
                                last_events[oid] = event
                
                # Display
                if not self.headless:
                    display = self._draw(frame)
                    cv2.imshow("Enhanced CCTV", display)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
        except KeyboardInterrupt:
            print("\n🛑 Stopped")
        finally:
            cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
    
    def _draw(self, frame):
        disp = frame.copy()
        h, w = disp.shape[:2]
        
        # Zones
        entry_y = self.tracker.direction_tracker.entry_y
        exit_y = self.tracker.direction_tracker.exit_y
        
        cv2.line(disp, (0, entry_y), (w, entry_y), (0, 255, 0), 2)
        cv2.putText(disp, "ENTRY", (10, entry_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.line(disp, (0, exit_y), (w, exit_y), (0, 0, 255), 2)
        cv2.putText(disp, "EXIT", (10, exit_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # People
        for oid, bbox in list(self.tracker.bboxes.items())[:MAX_DRAWN_PERSONS]:
            x1, y1, x2, y2 = bbox
            boxes = self.tracker.box_counts.get(oid, 0)
            dir_info = self.tracker.directions.get(oid, {})
            
            direction = dir_info.get('direction', 'STATIONARY')
            zone = dir_info.get('zone', '?')
            conf = dir_info.get('confidence', 0.0)
            
            # Color based on direction
            if direction == 'EXITING':
                color = (0, 0, 255)
            elif direction == 'ENTERING':
                color = (0, 255, 0)
            else:
                color = (255, 165, 0)
            
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            
            label = f"P{oid} {direction}"
            if boxes > 0:
                label += f" 📦x{boxes}"
            
            cv2.putText(disp, label, (x1, max(y1 - 10, 30)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            cv2.putText(disp, f"{zone} ({conf:.1f})", (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Arrow
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            if direction == 'ENTERING':
                cv2.arrowedLine(disp, (cx, cy + 30), (cx, cy - 30), 
                              color, 3, tipLength=0.3)
            elif direction == 'EXITING':
                cv2.arrowedLine(disp, (cx, cy - 30), (cx, cy + 30), 
                              color, 3, tipLength=0.3)
        
        # Stats
        total = len(self.tracker.objects)
        total_boxes = sum(self.tracker.box_counts.values())
        
        cv2.putText(disp, f"People: {total} | Boxes: {total_boxes}", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(disp, f"AI: {self.ai_worker.avg_time:.0f}ms", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return disp

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced Multi-Person CCTV')
    parser.add_argument('--rtsp', type=str, 
                       default="rtsp://fasspay:fasspay2025@192.168.122.127:554/stream1")
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    
    if not YOLO_AVAILABLE:
        print("❌ Install YOLO: pip install ultralytics")
        sys.exit(1)
    
    try:
        app = EnhancedCCTV(args.rtsp, args.headless)
        app.start()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
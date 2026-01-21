import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO
import torch
import json
import os

# RTSP stream URL
DEFAULT_RTSP = "rtsp://fasspay:fasspay2025@192.168.122.127:554/stream1"
CONFIG_FILE = "door_config.json"

class BoxCounter:
    def __init__(self):
        # Initialize YOLOv11 model
        print("Loading YOLOv11 model...")
        try:
            self.model = YOLO('yolov8n.pt')  # Auto-downloads if not exists
            print("✓ YOLOv11 model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying to download model...")
            self.model = YOLO('yolov8n.pt')
        
        # Define classes for detection
        self.person_class_id = 0
        self.box_class_ids = [24, 26, 27, 28, 39]  # handbag, tie, suitcase, frisbee, bottle
        
        # Door region coordinates - will load from file or set manually
        self.door_region = None
        self.exit_line = None
        
        # Tracking variables
        self.track_history = {}
        
        # Counters
        self.person_exit_count = 0
        self.box_exit_count = 0
        self.session_person_count = 0
        self.session_box_count = 0
        
        # FPS calculation
        self.prev_time = time.time()
        self.fps = 0
        self.frame_count = 0
        
        # Colors for visualization
        self.colors = {
            'person': (0, 255, 0),      # Green
            'box': (255, 100, 0),       # Orange
            'door': (0, 165, 255),      # Blue-Orange
            'line': (0, 0, 255),        # Red
            'track': (255, 255, 0),     # Cyan
            'text': (255, 255, 255),    # White
            'panel': (40, 40, 40)       # Dark gray
        }
        
        # Load saved configuration
        self.load_config()
    
    def load_config(self):
        """Load door configuration from file if exists"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    self.door_region = tuple(config['door_region'])
                    self.exit_line = config['exit_line']
                    print(f"✓ Loaded door configuration from {CONFIG_FILE}")
                    print(f"  Door region: {self.door_region}")
                    return True
            except Exception as e:
                print(f"Error loading config: {e}")
        return False
    
    def save_config(self):
        """Save door configuration to file"""
        if self.door_region:
            config = {
                'door_region': list(self.door_region),
                'exit_line': self.exit_line
            }
            try:
                with open(CONFIG_FILE, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"✓ Configuration saved to {CONFIG_FILE}")
            except Exception as e:
                print(f"Error saving config: {e}")
    
    def set_door_region_interactive(self, frame):
        """Interactive door region selection with better UI"""
        print("\n" + "="*60)
        print("DOOR REGION SETUP")
        print("="*60)
        print("Instructions:")
        print("1. Click and drag to define the door area")
        print("2. Press 'ENTER' or 'SPACE' to confirm")
        print("3. Press 'C' to cancel and use default")
        print("="*60)
        
        # Create a smaller window for setup
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Resize for better viewing if too large
        if w > 800 or h > 600:
            scale = min(800/w, 600/h)
            new_w, new_h = int(w * scale), int(h * scale)
            display_frame = cv2.resize(display_frame, (new_w, new_h))
        
        roi_start = None
        roi_end = None
        drawing = False
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal roi_start, roi_end, drawing
            
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_start = (x, y)
                roi_end = (x, y)
                drawing = True
                
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                roi_end = (x, y)
                
            elif event == cv2.EVENT_LBUTTONUP:
                roi_end = (x, y)
                drawing = False
        
        # Create window with reasonable size
        cv2.namedWindow("Set Door Region", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Set Door Region", 800, 600)
        cv2.setMouseCallback("Set Door Region", mouse_callback)
        
        while True:
            # Create display copy
            temp_frame = display_frame.copy()
            h_display, w_display = temp_frame.shape[:2]
            
            # Draw instructions
            cv2.putText(temp_frame, "Drag to select door area", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(temp_frame, "Press ENTER to confirm, C to cancel", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw current selection
            if roi_start and roi_end:
                # Make sure coordinates are in correct order
                x1, y1 = roi_start
                x2, y2 = roi_end
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                # Draw rectangle
                cv2.rectangle(temp_frame, (x1, y1), (x2, y2), self.colors['door'], 2)
                
                # Draw size info
                width = x2 - x1
                height = y2 - y1
                size_text = f"Size: {width}x{height}"
                cv2.putText(temp_frame, size_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['door'], 1)
            
            # Draw help text
            help_text = [
                "Tip: Make sure door area covers the exit path",
                "The exit line will be at the right edge of this box"
            ]
            for i, text in enumerate(help_text):
                cv2.putText(temp_frame, text, (10, h_display - 60 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Set Door Region", temp_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13 or key == 32:  # ENTER or SPACE
                if roi_start and roi_end:
                    # Scale back to original frame coordinates if needed
                    if w != w_display or h != h_display:
                        scale_x = w / w_display
                        scale_y = h / h_display
                        x1, y1 = roi_start
                        x2, y2 = roi_end
                        x1, x2 = int(min(x1, x2) * scale_x), int(max(x1, x2) * scale_x)
                        y1, y2 = int(min(y1, y2) * scale_y), int(max(y1, y2) * scale_y)
                    else:
                        x1, y1 = roi_start
                        x2, y2 = roi_end
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                    
                    self.door_region = (x1, y1, x2 - x1, y2 - y1)
                    self.exit_line = x2 - 20  # Exit line 20px from right edge
                    
                    print(f"\n✓ Door region set:")
                    print(f"  Location: ({x1}, {y1}) to ({x2}, {y2})")
                    print(f"  Size: {x2 - x1} x {y2 - y1}")
                    print(f"  Exit line at: x = {self.exit_line}")
                    
                    # Save configuration
                    self.save_config()
                    
                    cv2.destroyWindow("Set Door Region")
                    return True
            
            elif key == ord('c') or key == 27:  # C or ESC
                print("\n✗ Door setup cancelled")
                cv2.destroyWindow("Set Door Region")
                return False
        
        cv2.destroyWindow("Set Door Region")
        return False
    
    def detect_objects(self, frame):
        """Detect persons and boxes using YOLOv11"""
        try:
            # Run inference with tracking
            results = self.model.track(
                frame, 
                persist=True,
                classes=[self.person_class_id] + self.box_class_ids,
                conf=0.25,  # Lower confidence for better detection
                iou=0.5,
                verbose=False,
                tracker="bytetrack.yaml"  # Use ByteTrack for better tracking
            )
            
            detections = []
            
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                
                if boxes.id is not None:  # Tracking available
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        class_id = int(boxes.cls[i])
                        track_id = int(boxes.id[i])
                        conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                        
                        label = self.model.names[class_id]
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'label': label,
                            'class_id': class_id,
                            'track_id': track_id,
                            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'confidence': conf,
                            'area': (x2 - x1) * (y2 - y1)
                        })
            
            return detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def is_inside_door(self, center):
        """Check if a point is inside the door region"""
        if self.door_region is None:
            return False
        
        x, y, w, h = self.door_region
        center_x, center_y = center
        
        return (x <= center_x <= x + w) and (y <= center_y <= y + h)
    
    def update_tracking(self, detections):
        """Update object tracking and check for exits"""
        current_frame_persons = {}
        current_frame_boxes = {}
        
        # Separate persons and boxes
        for det in detections:
            track_id = det.get('track_id')
            if track_id is None:
                continue
                
            if det['class_id'] == self.person_class_id:
                current_frame_persons[track_id] = det
            elif det['class_id'] in self.box_class_ids:
                current_frame_boxes[track_id] = det
        
        # Update or create track history for persons
        for track_id, person in current_frame_persons.items():
            if track_id not in self.track_history:
                # New person detected
                self.track_history[track_id] = {
                    'type': 'person',
                    'positions': deque(maxlen=20),
                    'exited': False,
                    'boxes_carried': set(),
                    'entry_time': time.time(),
                    'exit_time': None,
                    'last_seen': time.time(),
                    'total_boxes': 0
                }
                print(f"👤 New person detected: ID {track_id}")
            
            # Update positions and timing
            self.track_history[track_id]['positions'].append(person['center'])
            self.track_history[track_id]['last_seen'] = time.time()
            
            # Check if person is in door area
            in_door_area = self.is_inside_door(person['center'])
            
            if in_door_area:
                # Find boxes this person might be carrying
                person_boxes = []
                person_bbox = person['bbox']
                
                for box_id, box in current_frame_boxes.items():
                    if self.is_carrying(person_bbox, box['bbox']):
                        person_boxes.append(box_id)
                        self.track_history[track_id]['boxes_carried'].add(box_id)
                
                # Check for exit
                if not self.track_history[track_id]['exited'] and self.exit_line:
                    center_x, _ = person['center']
                    if center_x > self.exit_line:
                        # Person has exited!
                        self.track_history[track_id]['exited'] = True
                        self.track_history[track_id]['exit_time'] = time.time()
                        
                        # Count boxes carried
                        boxes_carried = len(self.track_history[track_id]['boxes_carried'])
                        self.track_history[track_id]['total_boxes'] = boxes_carried
                        
                        # Update counters
                        self.person_exit_count += 1
                        self.session_person_count += 1
                        self.box_exit_count += boxes_carried
                        self.session_box_count += boxes_carried
                        
                        # Calculate time in view
                        time_in_view = time.time() - self.track_history[track_id]['entry_time']
                        
                        print(f"\n🚪 EXIT DETECTED!")
                        print(f"   Person ID: {track_id}")
                        print(f"   Boxes carried: {boxes_carried}")
                        print(f"   Time in view: {time_in_view:.1f}s")
                        print(f"   Session total: {self.session_person_count} people, {self.session_box_count} boxes")
        
        # Update box tracks
        for track_id, box in current_frame_boxes.items():
            if track_id not in self.track_history:
                self.track_history[track_id] = {
                    'type': 'box',
                    'positions': deque(maxlen=20),
                    'carried_by': None,
                    'last_seen': time.time()
                }
            
            self.track_history[track_id]['positions'].append(box['center'])
            self.track_history[track_id]['last_seen'] = time.time()
        
        # Clean up old tracks (not seen for 3 seconds)
        current_time = time.time()
        tracks_to_remove = []
        
        for track_id, history in self.track_history.items():
            if current_time - history['last_seen'] > 3.0:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_history[track_id]
    
    def is_carrying(self, person_bbox, box_bbox):
        """Check if person is carrying a box"""
        px1, py1, px2, py2 = person_bbox
        bx1, by1, bx2, by2 = box_bbox
        
        # Calculate centers
        person_center_x = (px1 + px2) // 2
        person_center_y = (py1 + py2) // 2
        box_center_x = (bx1 + bx2) // 2
        box_center_y = (by1 + by2) // 2
        
        # Check if box overlaps with upper body of person
        # Upper body is roughly top 60% of person bbox
        upper_body_top = py1
        upper_body_bottom = py1 + int((py2 - py1) * 0.6)
        
        # Box should be within person's width + margin
        margin = 30
        width_overlap = not (bx2 < px1 - margin or bx1 > px2 + margin)
        
        # Box should be near person's upper body
        height_near = upper_body_top - margin <= by1 <= upper_body_bottom + margin
        
        return width_overlap and height_near
    
    def draw_visualization(self, frame, detections):
        """Draw visual elements on frame with improved layout"""
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - self.prev_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1 / time_diff)
        self.prev_time = current_time
        self.frame_count += 1
        
        # Create a copy for drawing
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # Draw door region if set
        if self.door_region:
            x, y, door_w, door_h = self.door_region
            # Draw semi-transparent door area
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (x, y), (x + door_w, y + door_h), self.colors['door'], -1)
            cv2.addWeighted(overlay, 0.1, display_frame, 0.9, 0, display_frame)
            
            # Draw door border
            cv2.rectangle(display_frame, (x, y), (x + door_w, y + door_h), self.colors['door'], 2)
            
            # Draw door label
            cv2.putText(display_frame, "DOOR", (x + 5, y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['door'], 2)
            
            # Draw exit line
            if self.exit_line:
                cv2.line(display_frame, (self.exit_line, y), 
                        (self.exit_line, y + door_h), self.colors['line'], 3)
                cv2.putText(display_frame, "EXIT", (self.exit_line + 5, y + 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['line'], 2)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            track_id = det.get('track_id')
            class_id = det['class_id']
            conf = det['confidence']
            
            # Choose color and label
            if class_id == self.person_class_id:
                color = self.colors['person']
                obj_type = "PERSON"
            else:
                color = self.colors['box']
                obj_type = "BOX"
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if track_id:
                label_text = f"{obj_type} #{track_id}"
            else:
                label_text = f"{obj_type}"
            
            # Label background
            (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1 - text_height - 5),
                         (x1 + text_width + 5, y1), color, -1)
            
            # Label text
            cv2.putText(display_frame, label_text, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw center point
            center_x, center_y = det['center']
            cv2.circle(display_frame, (center_x, center_y), 3, color, -1)
            
            # If in door area, highlight
            if self.is_inside_door((center_x, center_y)):
                cv2.circle(display_frame, (center_x, center_y), 8, (0, 255, 255), 2)
        
        # Draw track trails
        for track_id, history in self.track_history.items():
            if len(history['positions']) > 1:
                points = list(history['positions'])
                for i in range(1, len(points)):
                    if points[i-1] and points[i]:
                        color = self.colors['person'] if history['type'] == 'person' else self.colors['box']
                        cv2.line(display_frame, points[i-1], points[i], color, 2)
        
        # ========== DRAW CONTROL PANEL ==========
        panel_width = 350
        panel_height = h
        
        # Create semi-transparent panel
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:] = self.colors['panel']
        
        # Add panel to left side
        display_frame[0:panel_height, 0:panel_width] = cv2.addWeighted(
            display_frame[0:panel_height, 0:panel_width], 0.3,
            panel, 0.7, 0
        )
        
        y_offset = 30
        line_height = 30
        
        # Title
        cv2.putText(display_frame, "BOX COUNTER SYSTEM", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += line_height + 10
        
        # FPS
        cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        y_offset += line_height
        
        # Session Counters
        cv2.putText(display_frame, "SESSION COUNTERS:", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(display_frame, f"People Exited: {self.session_person_count}", (25, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        y_offset += line_height
        
        cv2.putText(display_frame, f"Boxes Exited: {self.session_box_count}", (25, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        y_offset += line_height + 10
        
        # Total Counters (persistent)
        cv2.putText(display_frame, "TOTAL COUNTERS:", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(display_frame, f"Total People: {self.person_exit_count}", (25, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        y_offset += line_height
        
        cv2.putText(display_frame, f"Total Boxes: {self.box_exit_count}", (25, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        y_offset += line_height + 10
        
        # Current Status
        people_at_door = sum(1 for det in detections 
                           if det['class_id'] == self.person_class_id and 
                           self.is_inside_door(det['center']))
        
        cv2.putText(display_frame, "CURRENT STATUS:", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_offset += line_height
        
        cv2.putText(display_frame, f"People at door: {people_at_door}", (25, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        y_offset += line_height
        
        active_tracks = len([h for h in self.track_history.values() if h['type'] == 'person'])
        cv2.putText(display_frame, f"Active persons: {active_tracks}", (25, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        y_offset += line_height + 20
        
        # Controls Info
        cv2.putText(display_frame, "CONTROLS:", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_offset += line_height
        
        controls = [
            "Q - Quit",
            "R - Reset session",
            "D - Set door area",
            "S - Save totals",
            "P - Pause/resume",
            "C - Clear all data"
        ]
        
        for control in controls:
            cv2.putText(display_frame, control, (25, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_offset += line_height - 5
        
        # Legend
        y_offset += 10
        cv2.putText(display_frame, "LEGEND:", (15, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y_offset += line_height
        
        legend_items = [
            ("Green box", "Person"),
            ("Orange box", "Box/Object"),
            ("Blue area", "Door zone"),
            ("Red line", "Exit line"),
            ("Trail", "Movement path")
        ]
        
        for color_text, desc in legend_items:
            cv2.putText(display_frame, f"● {desc}", (25, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_offset += line_height - 10
        
        # System status
        status_text = f"Frame: {self.frame_count}"
        cv2.putText(display_frame, status_text, (15, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return display_frame

def main():
    print("="*70)
    print("🚪 BOX COUNTER SYSTEM - Door Exit Monitoring")
    print("="*70)
    
    # Initialize system
    counter = BoxCounter()
    
    # Connect to RTSP
    print(f"\n📹 Connecting to RTSP stream...")
    print(f"   URL: {DEFAULT_RTSP}")
    
    # Try different connection methods
    cap = None
    for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(DEFAULT_RTSP, backend)
        if cap.isOpened():
            print(f"   ✓ Connected (backend: {backend})")
            break
        if cap:
            cap.release()
    
    if not cap or not cap.isOpened():
        print("   ✗ Failed to connect to RTSP stream")
        print("\n🔧 Troubleshooting:")
        print("   1. Check if camera is online")
        print("   2. Verify IP address: 192.168.122.127")
        print("   3. Test with VLC: rtsp://fasspay:fasspay2025@192.168.122.127:554/stream1")
        return
    
    # Get stream info
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"   Resolution: {w}x{h}")
    print(f"   FPS: {fps if fps > 0 else 'Auto'}")
    
    # Check if door region is already configured
    if counter.door_region is None:
        print("\n📍 No door configuration found")
        print("   Need to set up door area...")
        
        # Get a frame for setup
        ret, frame = cap.read()
        if not ret:
            print("   ✗ Could not read frame for setup")
            cap.release()
            return
        
        # Interactive door setup
        setup_success = counter.set_door_region_interactive(frame)
        if not setup_success:
            print("\n⚠️  Using default door area")
            # Set default door area (center 30% of frame)
            door_w, door_h = int(w * 0.3), int(h * 0.4)
            door_x = (w - door_w) // 2
            door_y = (h - door_h) // 2
            counter.door_region = (door_x, door_y, door_w, door_h)
            counter.exit_line = door_x + door_w - 20
            print(f"   Default door: {counter.door_region}")
    else:
        print(f"\n✅ Using saved door configuration")
        print(f"   Door area: {counter.door_region}")
    
    print("\n" + "="*70)
    print("✅ SYSTEM READY - Starting monitoring...")
    print("="*70)
    print("\n📋 Controls:")
    print("   Q - Quit application")
    print("   R - Reset session counters")
    print("   D - Redefine door area")
    print("   S - Save total counters to file")
    print("   P - Pause/Resume detection")
    print("   C - Clear all data (totals too)")
    print("   ESC - Exit")
    print("\n🎯 Monitoring door exits with box counting...")
    
    # Create main window
    window_name = "Box Counter - Door Exit Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1200, w + 350), min(800, h))
    
    paused = False
    last_save_time = 0
    
    while True:
        if not paused:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("\n⚠️  Frame read error, reconnecting...")
                time.sleep(1)
                cap.release()
                cap = cv2.VideoCapture(DEFAULT_RTSP)
                continue
            
            # Detect objects
            detections = counter.detect_objects(frame)
            
            # Update tracking
            counter.update_tracking(detections)
            
            # Draw visualization
            display_frame = counter.draw_visualization(frame, detections)
        else:
            # When paused, show paused overlay
            display_frame = frame.copy()
            cv2.putText(display_frame, "PAUSED", (w//2 - 100, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.putText(display_frame, "Press 'P' to resume", (w//2 - 150, h//2 + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show frame
        cv2.imshow(window_name, display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1 if not paused else 100) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or ESC
            print("\n👋 Exiting application...")
            break
            
        elif key == ord('r'):
            counter.session_person_count = 0
            counter.session_box_count = 0
            counter.track_history.clear()
            print("\n🔄 Session counters reset!")
            
        elif key == ord('d'):
            paused = True
            cv2.destroyWindow(window_name)
            ret, frame = cap.read()
            if ret:
                setup_success = counter.set_door_region_interactive(frame)
                if setup_success:
                    counter.save_config()
            # Recreate window
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(1200, w + 350), min(800, h))
            paused = False
            
        elif key == ord('s'):
            # Save totals to file
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"box_counter_totals_{timestamp}.txt"
            with open(filename, 'w') as f:
                f.write("="*50 + "\n")
                f.write("BOX COUNTER SYSTEM - TOTALS REPORT\n")
                f.write("="*50 + "\n\n")
                f.write(f"Report Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total People Exited: {counter.person_exit_count}\n")
                f.write(f"Total Boxes Exited: {counter.box_exit_count}\n")
                f.write(f"Session People: {counter.session_person_count}\n")
                f.write(f"Session Boxes: {counter.session_box_count}\n")
                f.write(f"Door Region: {counter.door_region}\n")
                f.write(f"Exit Line: x={counter.exit_line}\n")
                f.write("\n" + "="*50 + "\n")
            
            print(f"\n💾 Totals saved to: {filename}")
            last_save_time = time.time()
            
        elif key == ord('p'):
            paused = not paused
            status = "PAUSED" if paused else "RESUMED"
            print(f"\n⏸️  {status}")
            
        elif key == ord('c'):
            confirm = input("\n⚠️  Are you sure you want to clear ALL data? (y/n): ").lower()
            if confirm == 'y':
                counter.person_exit_count = 0
                counter.box_exit_count = 0
                counter.session_person_count = 0
                counter.session_box_count = 0
                counter.track_history.clear()
                if os.path.exists(CONFIG_FILE):
                    os.remove(CONFIG_FILE)
                print("🗑️  All data cleared!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Final report
    print("\n" + "="*70)
    print("📊 FINAL REPORT")
    print("="*70)
    print(f"Total People Exited: {counter.person_exit_count}")
    print(f"Total Boxes Carried Out: {counter.box_exit_count}")
    print(f"This Session: {counter.session_person_count} people, {counter.session_box_count} boxes")
    print(f"Total Frames Processed: {counter.frame_count}")
    print("="*70)
    print("Thank you for using Box Counter System!")
    
    # Auto-save on exit
    if counter.person_exit_count > 0:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"box_counter_final_{timestamp}.txt"
        with open(filename, 'w') as f:
            f.write(f"Final totals: {counter.person_exit_count} people, {counter.box_exit_count} boxes\n")
        print(f"Final totals saved to: {filename}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
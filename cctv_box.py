#!/usr/bin/env python3
"""
Enhanced CCTV System with Meta SAM for Box Detection
- Uses SAM for precise box segmentation
- Tracks direction (entering/exiting)
- Counts boxes per person with direction
- Better box detection accuracy
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
import json
import csv

# Suppress warnings
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Memory optimization settings
YOLO_USE_GPU = True
INSIGHTFACE_USE_GPU = True
MAX_FACES_PER_FRAME = 1
PROCESS_RESOLUTION = 1080
FACE_DETECTION_SIZE = (416, 416)
YOLO_IMG_SIZE = 416

# Display settings
MAX_DRAWN_PERSONS = 10

# Box detection settings
BOX_IOU_THRESHOLD = 0.3
BOX_CLASSES = [24, 25, 26, 28, 39, 41, 56, 58, 63, 64, 65, 66, 67, 73]

# Try to import Meta SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
    print("✅ Meta SAM available")
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️  Install SAM: pip install git+https://github.com/facebookresearch/segment-anything.git")

# Try to import Grounding DINO
try:
    import groundingdino.datasets.transforms as T
    from groundingdino.models import build_model
    from groundingdino.util import box_ops
    from groundingdino.util.inference import predict
    import torch
    from PIL import Image
    GROUNDING_DINO_AVAILABLE = True
    print("✅ Grounding DINO available")
except ImportError:
    GROUNDING_DINO_AVAILABLE = False
    print("⚠️  Grounding DINO not installed")

# YOLOv11
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Run: pip install ultralytics")

# InsightFace
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("Warning: insightface not installed. Run: pip install insightface onnxruntime-gpu")

# Configuration
USE_META_SAM = True
USE_DIRECTION_TRACKING = True
BOX_COUNT_HISTORY_LENGTH = 10
DIRECTION_THRESHOLD = 30

def clear_gpu_memory():
    """Clear GPU cache to free memory"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass
    gc.collect()

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    
    if xi_max < xi_min or yi_max < yi_min:
        return 0.0
    
    intersection_area = (xi_max - xi_min) * (yi_max - yi_min)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def is_box_held_by_person(person_bbox, box_bbox, iou_threshold=BOX_IOU_THRESHOLD):
    """Determine if a box is being held by a person"""
    iou = calculate_iou(person_bbox, box_bbox)
    
    px1, py1, px2, py2 = person_bbox
    bx1, by1, bx2, by2 = box_bbox
    
    box_center_x = (bx1 + bx2) / 2
    box_center_y = (by1 + by2) / 2
    
    in_person_x = px1 < box_center_x < px2
    in_person_y = py1 < box_center_y < py2
    
    person_lower_y = py1 + (py2 - py1) * 0.3
    in_carrying_zone = box_center_y > person_lower_y
    
    return iou > iou_threshold or (in_person_x and in_person_y and in_carrying_zone)

class BoxDetectionModule:
    """Basic box detection module"""
    def __init__(self):
        self.box_classes = BOX_CLASSES
        self.iou_threshold = BOX_IOU_THRESHOLD
        self.person_box_history = defaultdict(list)
        self.history_length = 5
        
    def detect_boxes_for_persons(self, persons, boxes, frame=None):
        """Match boxes to persons and count them"""
        results = {}
        
        for i, person in enumerate(persons):
            person_bbox = person["bbox"]
            held_boxes = []
            
            for box in boxes:
                if is_box_held_by_person(person_bbox, box["bbox"], self.iou_threshold):
                    held_boxes.append({
                        "bbox": box["bbox"],
                        "class": box["class"],
                        "conf": box["conf"],
                        "class_name": self._get_box_class_name(box["class"])
                    })
            
            results[i] = {
                "box_count": len(held_boxes),
                "boxes": held_boxes,
                "status": "holding_boxes" if len(held_boxes) > 0 else "no_boxes"
            }
        
        return results
    
    def _get_box_class_name(self, class_id):
        """Get human-readable name for box class"""
        class_names = {
            24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase",
            39: "bottle", 41: "cup", 56: "chair", 58: "plant",
            63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
            67: "phone", 73: "book", 999: "box"
        }
        return class_names.get(class_id, f"object_{class_id}")
    
    def get_stable_box_count(self, person_id, current_count):
        """Get stabilized box count using temporal averaging"""
        self.person_box_history[person_id].append(current_count)
        
        if len(self.person_box_history[person_id]) > self.history_length:
            self.person_box_history[person_id].pop(0)
        
        counts = self.person_box_history[person_id]
        if not counts:
            return 0
        
        count_freq = defaultdict(int)
        for c in counts:
            count_freq[c] += 1
        
        return max(count_freq.items(), key=lambda x: x[1])[0]
    
    def clear_history(self, person_id):
        """Clear history for a person who left"""
        if person_id in self.person_box_history:
            del self.person_box_history[person_id]

class FaceRecognitionSystem:
    """Face recognition system (simplified version)"""
    def __init__(self, embeddings_file="known_embeddings.npz"):
        self.embeddings_file = embeddings_file
        self.known_encodings = []
        self.known_names = []
        self.enabled = False
        self.training_active = False
        self.unknown_collections = {}
        self.unknown_counter = 1
        
        if INSIGHTFACE_AVAILABLE:
            try:
                self.app = FaceAnalysis(name='buffalo_l')
                self.app.prepare(ctx_id=0, det_size=FACE_DETECTION_SIZE)
                self.enabled = True
            except Exception as e:
                print(f"❌ Failed to initialize InsightFace: {e}")
    
    def identify_faces_fast(self, frame, person_bboxes):
        """Identify faces in bounding boxes"""
        if not self.enabled or not person_bboxes:
            return []
        
        results = []
        
        for bbox in person_bboxes[:MAX_FACES_PER_FRAME]:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            cx1, cy1 = max(0, x1-20), max(0, y1-20)
            cx2, cy2 = min(w, x2+20), min(h, y2+20)
            
            person_img = frame[cy1:cy2, cx1:cx2]
            if person_img.size == 0:
                continue
            
            try:
                faces = self.app.get(person_img)
                if not faces:
                    continue
                
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                target_face = faces[0]
                embedding = target_face.normed_embedding
                
                best_name = "Unknown"
                best_score = 0.0
                
                if self.known_encodings:
                    scores = np.dot(self.known_encodings, embedding)
                    best_idx = np.argmax(scores)
                    best_score = scores[best_idx]
                    
                    if best_score > 0.45:
                        best_name = self.known_names[best_idx]
                
                results.append({
                    "name": best_name,
                    "bbox": bbox,
                    "confidence": float(best_score),
                    "status": "known" if best_name != "Unknown" else "unknown"
                })
                
            except Exception as e:
                continue
        
        return results
    
    def toggle_training(self):
        """Toggle training mode"""
        self.training_active = not self.training_active
        return self.training_active

class PersonTracker:
    """Original PersonTracker class"""
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.person_names = {}
        self.person_confidences = {}
        self.entry_times = {}
        self.has_exited = set()
        self.recognition_locked = {}
        self.last_bbox = {}
        self.person_box_counts = {}
        self.box_detection_module = BoxDetectionModule()
    
    def register(self, centroid, name="Unknown", bbox=None, box_count=0, confidence=0.0):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.person_names[self.next_id] = name
        self.person_confidences[self.next_id] = confidence
        self.entry_times[self.next_id] = datetime.now()
        self.recognition_locked[self.next_id] = (name != "Unknown")
        self.person_box_counts[self.next_id] = box_count
        if bbox is not None:
            self.last_bbox[self.next_id] = bbox
        self.next_id += 1
    
    def deregister(self, object_id):
        keys_to_delete = [
            'objects', 'disappeared', 'person_names', 'person_confidences',
            'entry_times', 'recognition_locked', 'last_bbox', 'person_box_counts'
        ]
        
        for key in keys_to_delete:
            attr = getattr(self, key, {})
            if isinstance(attr, dict) and object_id in attr:
                del attr[object_id]
        
        self.box_detection_module.clear_history(object_id)
    
    def update(self, detections, faces, box_results):
        """Update tracker"""
        input_centroids = []
        input_names = []
        input_confidences = []
        input_bboxes = []
        input_should_lock = []
        input_box_counts = []
        
        for idx, det in enumerate(detections):
            bbox = det["bbox"]
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            input_centroids.append((cx, cy))
            input_bboxes.append(bbox)
            
            box_count = box_results.get(idx, {}).get("box_count", 0)
            input_box_counts.append(box_count)
            
            name = "Unknown"
            should_lock = False
            confidence = 0.0
            
            for face in faces:
                fx1, fy1, fx2, fy2 = face["bbox"]
                bx1, by1, bx2, by2 = bbox
                
                if (fx1 >= bx1 and fy1 >= by1 and fx2 <= bx2 and fy2 <= by2):
                    name = face["name"]
                    should_lock = face.get("should_lock", False)
                    confidence = face.get("confidence", 0.0)
                    break
            
            input_names.append(name)
            input_confidences.append(confidence)
            input_should_lock.append(should_lock)
        
        if len(input_centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects
        
        input_centroids = np.array(input_centroids)
        
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_names[i], input_bboxes[i], 
                            input_box_counts[i], input_confidences[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))
            
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > 200:
                    continue
                
                obj_id = object_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.disappeared[obj_id] = 0
                self.last_bbox[obj_id] = input_bboxes[col]
                
                raw_box_count = input_box_counts[col]
                stable_count = self.box_detection_module.get_stable_box_count(obj_id, raw_box_count)
                self.person_box_counts[obj_id] = stable_count
                
                if not self.recognition_locked.get(obj_id, False):
                    if input_names[col] != "Unknown":
                        self.person_names[obj_id] = input_names[col]
                        self.person_confidences[obj_id] = input_confidences[col]
                
                used_rows.add(row)
                used_cols.add(col)
            
            for i in range(len(input_centroids)):
                if i not in used_cols:
                    self.register(input_centroids[i], input_names[i], input_bboxes[i],
                                input_box_counts[i], input_confidences[i])
            
            for i in range(len(object_ids)):
                if i not in used_rows:
                    obj_id = object_ids[i]
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)
        
        return self.objects

class DirectionTracker:
    """Track person movement direction (entering/exiting)"""
    def __init__(self, exit_line_ratio=0.7):
        self.person_positions = defaultdict(lambda: deque(maxlen=10))
        self.directions = {}
        self.exit_line_ratio = exit_line_ratio
        self.frame_height = None
    
    def update(self, person_id, bbox, frame_height):
        """Update position history and calculate direction"""
        self.frame_height = frame_height
        
        x1, y1, x2, y2 = bbox
        center_y = (y1 + y2) / 2
        self.person_positions[person_id].append(center_y)
        
        if len(self.person_positions[person_id]) >= 3:
            positions = list(self.person_positions[person_id])
            if len(positions) >= 2:
                dy = positions[-1] - positions[0]
                
                if abs(dy) > DIRECTION_THRESHOLD:
                    if dy > 0:
                        self.directions[person_id] = "exiting"
                    else:
                        self.directions[person_id] = "entering"
        
        exit_line = frame_height * self.exit_line_ratio
        if center_y > exit_line and person_id in self.directions:
            if self.directions[person_id] == "exiting":
                return "exited"
        
        return self.directions.get(person_id, "stationary")
    
    def get_direction(self, person_id):
        return self.directions.get(person_id, "stationary")

class MetaSAMBoxDetector:
    """Meta Segment Anything Model for precise box detection"""
    def __init__(self):
        if not SAM_AVAILABLE:
            print("❌ SAM not available. Using fallback detection.")
            self.available = False
            return
        
        self.available = True
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        
        try:
            import torch
            sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
            if torch.cuda.is_available():
                sam.to(device="cuda")
                print("✅ SAM loaded on GPU")
            else:
                print("⚠️  SAM using CPU")
            
            self.predictor = SamPredictor(sam)
            print("✅ Meta SAM initialized")
        except Exception as e:
            print(f"❌ Failed to load SAM: {e}")
            self.available = False
    
    def detect_boxes_with_sam(self, frame, person_bboxes):
        if not self.available or not person_bboxes:
            return {}
        
        results = {}
        h, w = frame.shape[:2]
        self.predictor.set_image(frame)
        
        for idx, person_bbox in enumerate(person_bboxes):
            px1, py1, px2, py2 = person_bbox
            
            prompt_points = []
            prompt_labels = []
            
            center_x = (px1 + px2) // 2
            lower_y = py2 - (py2 - py1) // 4
            
            prompt_points.append([center_x, lower_y])
            prompt_labels.append(1)
            
            margin = 20
            for dx in [-margin, 0, margin]:
                for dy in [-margin, 0, margin]:
                    if dx == 0 and dy == 0:
                        continue
                    prompt_points.append([center_x + dx, lower_y + dy])
                    prompt_labels.append(1)
            
            prompt_points = np.array(prompt_points)
            prompt_labels = np.array(prompt_labels)
            
            try:
                masks, scores, _ = self.predictor.predict(
                    point_coords=prompt_points,
                    point_labels=prompt_labels,
                    multimask_output=True
                )
                
                valid_masks = []
                for mask_idx, mask in enumerate(masks):
                    y_indices, x_indices = np.where(mask)
                    if len(x_indices) == 0 or len(y_indices) == 0:
                        continue
                    
                    bx1 = np.min(x_indices)
                    by1 = np.min(y_indices)
                    bx2 = np.max(x_indices)
                    by2 = np.max(y_indices)
                    
                    if self._is_box_near_person(person_bbox, (bx1, by1, bx2, by2)):
                        area = (bx2 - bx1) * (by2 - by1)
                        if 1000 < area < 50000:
                            valid_masks.append({
                                "bbox": [bx1, by1, bx2, by2],
                                "mask": mask,
                                "score": scores[mask_idx],
                                "area": area
                            })
                
                if valid_masks:
                    best_mask = max(valid_masks, key=lambda x: x["score"])
                    results[idx] = {
                        "boxes": [best_mask["bbox"]],
                        "confidence": float(best_mask["score"]),
                        "count": 1
                    }
                else:
                    results[idx] = {"boxes": [], "confidence": 0.0, "count": 0}
                    
            except Exception as e:
                print(f"SAM detection error: {e}")
                results[idx] = {"boxes": [], "confidence": 0.0, "count": 0}
        
        return results
    
    def _is_box_near_person(self, person_bbox, box_bbox):
        px1, py1, px2, py2 = person_bbox
        bx1, by1, bx2, by2 = box_bbox
        
        person_center_x = (px1 + px2) / 2
        person_center_y = (py1 + py2) / 2
        box_center_x = (bx1 + bx2) / 2
        box_center_y = (by1 + by2) / 2
        
        horiz_dist = abs(person_center_x - box_center_x)
        vertical_dist = box_center_y - person_center_y
        
        return (vertical_dist > 0 and 
                horiz_dist < (px2 - px1) * 0.8 and
                vertical_dist < (py2 - py1) * 1.5)

class EnhancedBoxDetectionModule:
    """Enhanced box detection with Meta SAM and direction tracking"""
    def __init__(self):
        self.sam_detector = MetaSAMBoxDetector() if USE_META_SAM else None
        self.direction_tracker = DirectionTracker() if USE_DIRECTION_TRACKING else None
        self.person_box_history = defaultdict(lambda: deque(maxlen=BOX_COUNT_HISTORY_LENGTH))
        self.entry_exit_log = []
    
    def detect_all_boxes(self, frame, persons, yolo_boxes):
        results = {}
        person_bboxes = [p["bbox"] for p in persons]
        
        for idx, person in enumerate(persons):
            person_bbox = person["bbox"]
            held_boxes = []
            
            for box in yolo_boxes:
                if is_box_held_by_person(person_bbox, box["bbox"]):
                    held_boxes.append({
                        "bbox": box["bbox"],
                        "class": box["class"],
                        "conf": box["conf"],
                        "source": "yolo"
                    })
            
            if self.sam_detector and self.sam_detector.available:
                sam_results = self.sam_detector.detect_boxes_with_sam(frame, [person_bbox])
                
                if idx in sam_results and sam_results[idx]["count"] > 0:
                    for box_bbox in sam_results[idx]["boxes"]:
                        is_duplicate = False
                        for existing in held_boxes:
                            iou = calculate_iou(box_bbox, existing["bbox"])
                            if iou > 0.3:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            held_boxes.append({
                                "bbox": box_bbox,
                                "class": 999,
                                "conf": sam_results[idx]["confidence"],
                                "source": "sam"
                            })
            
            results[idx] = {
                "box_count": len(held_boxes),
                "boxes": held_boxes,
                "status": "holding_boxes" if len(held_boxes) > 0 else "no_boxes"
            }
        
        return results
    
    def update_direction(self, person_id, bbox, frame_height, box_count):
        if not self.direction_tracker:
            return "stationary"
        
        direction = self.direction_tracker.update(person_id, bbox, frame_height)
        
        self.person_box_history[person_id].append({
            "timestamp": datetime.now(),
            "box_count": box_count,
            "direction": direction,
            "bbox": bbox
        })
        
        if direction == "exited":
            exit_event = {
                "person_id": person_id,
                "timestamp": datetime.now(),
                "event": "exit",
                "box_count": box_count
            }
            self.entry_exit_log.append(exit_event)
            print(f"🚪 Person {person_id} EXITED with {box_count} boxes")
        
        elif direction == "entering":
            entry_event = {
                "person_id": person_id,
                "timestamp": datetime.now(),
                "event": "entry",
                "box_count": box_count
            }
            self.entry_exit_log.append(entry_event)
            print(f"🚪 Person {person_id} ENTERED with {box_count} boxes")
        
        return direction

class EnhancedPersonTracker(PersonTracker):
    def __init__(self, max_disappeared=30):
        super().__init__(max_disappeared)
        self.enhanced_box_detector = EnhancedBoxDetectionModule()
        self.person_directions = {}
        self.has_exited = set()
    
    def update(self, detections, faces, box_results, frame=None):
        super().update(detections, faces, box_results)
        
        for obj_id in list(self.objects.keys()):
            if obj_id in self.last_bbox:
                bbox = self.last_bbox[obj_id]
                box_count = self.person_box_counts.get(obj_id, 0)
                
                if frame is not None:
                    frame_height = frame.shape[0]
                    direction = self.enhanced_box_detector.update_direction(
                        obj_id, bbox, frame_height, box_count
                    )
                    self.person_directions[obj_id] = direction
        
        return self.objects

class AIWorker:
    def __init__(self, detector, face_system, tracker):
        self.detector = detector
        self.face_system = face_system
        self.tracker = tracker
        self.box_detector = EnhancedBoxDetectionModule()
        self.frame_queue = Queue(maxsize=3)
        self.result_queue = Queue(maxsize=3)
        self.running = True
        self.avg_time_ms = 0
    
    def start(self):
        threading.Thread(target=self._worker, daemon=True).start()
    
    def submit_frame(self, frame):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
        self.frame_queue.put(frame)
    
    def get_result(self):
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None
    
    def _worker(self):
        print("🚀 AI Worker Started")
        frame_count = 0
        
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
                start_time = time.time()
                
                h, w = frame.shape[:2]
                scale_factor = 1.0
                
                if w > PROCESS_RESOLUTION:
                    scale_factor = PROCESS_RESOLUTION / w
                    new_h = int(h * scale_factor)
                    process_frame = cv2.resize(frame, (PROCESS_RESOLUTION, new_h))
                else:
                    process_frame = frame
                
                results = self.detector(process_frame, conf=0.4, imgsz=YOLO_IMG_SIZE, verbose=False)[0]
                
                detections = {"persons": [], "boxes": []}
                
                for box in results.boxes:
                    cls = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    bbox = [int(x) for x in xyxy]
                    
                    det = {
                        "bbox": bbox,
                        "class": cls,
                        "conf": float(box.conf[0])
                    }
                    
                    if cls == 0:
                        detections["persons"].append(det)
                    elif cls in BOX_CLASSES:
                        detections["boxes"].append(det)
                
                box_results = self.box_detector.detect_all_boxes(
                    process_frame,
                    detections["persons"],
                    detections["boxes"]
                )
                
                persons_needing_recognition = []
                
                for p in detections["persons"]:
                    bbox = p["bbox"]
                    cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                    
                    needs_fr = True
                    for obj_id, tracked_centroid in self.tracker.objects.items():
                        dist = np.sqrt((cx - tracked_centroid[0])**2 + (cy - tracked_centroid[1])**2)
                        
                        if dist < 100 and self.tracker.recognition_locked.get(obj_id, False):
                            needs_fr = False
                            break
                    
                    if needs_fr:
                        persons_needing_recognition.append(bbox)
                
                person_bboxes = persons_needing_recognition[:MAX_FACES_PER_FRAME]
                faces = self.face_system.identify_faces_fast(process_frame, person_bboxes)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    clear_gpu_memory()
                
                elapsed = (time.time() - start_time) * 1000
                self.avg_time_ms = self.avg_time_ms * 0.9 + elapsed * 0.1
                
                result = {
                    "detections": detections,
                    "faces": faces,
                    "box_results": box_results,
                    "process_time": elapsed,
                    "scale_factor": scale_factor,
                    "frame_shape": process_frame.shape[:2],
                    "process_frame": process_frame
                }
                
                if self.result_queue.full():
                    self.result_queue.get()
                self.result_queue.put(result)
                
            except Empty:
                continue
            except Exception as e:
                print(f"AI Error: {e}")
                clear_gpu_memory()

class EnhancedInventoryDoorCCTV:
    def __init__(self, rtsp_url, headless=False):
        self.rtsp_url = rtsp_url
        self.log_file = "inventory_log.csv"
        self.events_log_file = "entry_exit_events.csv"
        self.headless = headless
        
        self.face_system = FaceRecognitionSystem()
        self.tracker = EnhancedPersonTracker()
        
        print("Loading YOLO...")
        self.detector = YOLO("yolo11n.pt")
        clear_gpu_memory()
        
        self.ai_worker = AIWorker(self.detector, self.face_system, self.tracker)
        
        self.running = True
        self.latest_result = None
        self.pending_registrations = []
        
        self._init_logs()
    
    def _init_logs(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("Timestamp,Person_ID,Name,Box_Count,Box_Details\n")
        
        if not os.path.exists(self.events_log_file):
            with open(self.events_log_file, "w") as f:
                f.write("Timestamp,Person_ID,Name,Event,Direction,Box_Count\n")
    
    def _log_event(self, person_id, name, event, direction, box_count):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"{timestamp},{person_id},{name},{event},{direction},{box_count}\n"
        
        with open(self.events_log_file, "a") as f:
            f.write(log_entry)
        
        print(f"📝 EVENT: {event.upper()} | Person {person_id} ({name}) | "
              f"Direction: {direction} | Boxes: {box_count}")
    
    def start(self):
        print(f"Connecting to {self.rtsp_url}...")
        
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.ai_worker.start()
        
        last_submit = 0
        last_ai_update = 0
        
        if not self.headless:
            cv2.namedWindow("Enhanced CCTV", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Enhanced CCTV", 1280, 720)
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Stream lost, retrying...")
                    time.sleep(2)
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue
                
                now = time.time()
                
                if now - last_submit > 0.1:
                    self.ai_worker.submit_frame(frame.copy())
                    last_submit = now
                
                res = self.ai_worker.get_result()
                if res:
                    self.latest_result = res
                    self._process_logic(res)
                    last_ai_update = now
                
                if not self.headless:
                    display_frame = self._draw_overlay(frame)
                    
                    ai_age = now - last_ai_update if last_ai_update > 0 else 0
                    fps_text = f"AI: {self.ai_worker.avg_time_ms:.1f}ms (age: {ai_age:.2f}s)"
                    cv2.putText(display_frame, fps_text, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow("Enhanced CCTV", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord('t'):
                        self.face_system.toggle_training()
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord('t'):
                        self.face_system.toggle_training()
                        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
    
    def _process_logic(self, res):
        detections = res["detections"]["persons"]
        faces = res["faces"]
        box_results = res.get("box_results", {})
        
        self.tracker.update(detections, faces, box_results, res.get("process_frame"))
        
        for obj_id in list(self.tracker.objects.keys()):
            if obj_id in self.tracker.person_directions:
                direction = self.tracker.person_directions[obj_id]
                
                if direction == "exited" and obj_id not in self.tracker.has_exited:
                    name = self.tracker.person_names.get(obj_id, "Unknown")
                    box_count = self.tracker.person_box_counts.get(obj_id, 0)
                    
                    self._log_event(obj_id, name, "exit", direction, box_count)
                    self.tracker.has_exited.add(obj_id)
                    
                elif direction == "entering" and obj_id in self.tracker.has_exited:
                    self.tracker.has_exited.remove(obj_id)
    
    def _draw_overlay(self, frame):
        if self.latest_result:
            h, w = frame.shape[:2]
            scale = self.latest_result["scale_factor"]
            
            if scale != 1.0:
                display_h, display_w = self.latest_result["frame_shape"]
                display_frame = cv2.resize(frame, (display_w, display_h))
            else:
                display_frame = frame
            
            tracked_items = list(self.tracker.objects.items())
            tracked_items.sort(key=lambda x: x[0], reverse=True)
            tracked_items = tracked_items[:MAX_DRAWN_PERSONS]
            
            for obj_id, centroid in tracked_items:
                if obj_id in self.tracker.last_bbox:
                    x1, y1, x2, y2 = self.tracker.last_bbox[obj_id]
                    name = self.tracker.person_names.get(obj_id, "Unknown")
                    box_count = self.tracker.person_box_counts.get(obj_id, 0)
                    confidence = self.tracker.person_confidences.get(obj_id, 0.0)
                    
                    if name != "Unknown":
                        color = (0, 255, 0)
                        label = f"{name} [{confidence:.2f}] ID:{obj_id}"
                    else:
                        color = (0, 165, 255)
                        label = f"Person ID:{obj_id}"
                    
                    if box_count > 0:
                        label += f" 📦x{box_count}"
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    direction = self.tracker.person_directions.get(obj_id, "stationary")
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    if direction == "entering":
                        cv2.arrowedLine(display_frame,
                                       (center_x, center_y + 20),
                                       (center_x, center_y - 20),
                                       (0, 255, 0), 2, tipLength=0.3)
                        dir_text = "ENTERING"
                        dir_color = (0, 255, 0)
                    elif direction == "exiting":
                        cv2.arrowedLine(display_frame,
                                       (center_x, center_y - 20),
                                       (center_x, center_y + 20),
                                       (0, 0, 255), 2, tipLength=0.3)
                        dir_text = "EXITING"
                        dir_color = (0, 0, 255)
                    else:
                        dir_text = "STATIONARY"
                        dir_color = (255, 255, 0)
                    
                    cv2.putText(display_frame, dir_text,
                               (x1, y1 - 40 if y1 > 40 else y2 + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, dir_color, 1)
            
            if hasattr(self.tracker, 'enhanced_box_detector') and \
               hasattr(self.tracker.enhanced_box_detector, 'direction_tracker'):
                
                exit_line_y = int(h * self.tracker.enhanced_box_detector.direction_tracker.exit_line_ratio)
                cv2.line(display_frame, (0, exit_line_y), (w, exit_line_y),
                        (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(display_frame, "EXIT LINE", (w - 100, exit_line_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            total_tracked = len(self.tracker.objects)
            locked_count = sum(1 for locked in self.tracker.recognition_locked.values() if locked)
            total_boxes = sum(self.tracker.person_box_counts.values())
            
            cv2.putText(display_frame, f"Tracked: {total_tracked} | Locked: {locked_count} | Boxes: {total_boxes}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            return display_frame
        
        return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced CCTV with Meta SAM')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--no-sam', action='store_true', help='Disable Meta SAM')
    parser.add_argument('--no-direction', action='store_true', help='Disable direction tracking')
    args = parser.parse_args()
    
    if args.no_sam:
        USE_META_SAM = False
    if args.no_direction:
        USE_DIRECTION_TRACKING = False
    
    RTSP_URL = "rtsp://fasspay:fasspay2025@192.168.122.127:554/stream1"
    
    print("\n" + "="*60)
    print("ENHANCED CCTV WITH BOX DETECTION & DIRECTION TRACKING")
    print("="*60)
    print(f"Meta SAM: {'ENABLED' if USE_META_SAM else 'DISABLED'}")
    print(f"Direction Tracking: {'ENABLED' if USE_DIRECTION_TRACKING else 'DISABLED'}")
    print(f"Headless Mode: {args.headless}")
    print("="*60 + "\n")
    
    if USE_META_SAM and SAM_AVAILABLE and not os.path.exists("sam_vit_h_4b8939.pth"):
        print("⚠️  Download SAM model:")
        print("   https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print("   Place in current directory")
        USE_META_SAM = False
    
    app = EnhancedInventoryDoorCCTV(RTSP_URL, headless=args.headless)
    app.start()
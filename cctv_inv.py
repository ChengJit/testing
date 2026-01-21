#!/usr/bin/env python3
"""
CCTV System with Box Detection, Counting & Similarity Scores
- Shows confidence/similarity scores next to names
- Limits max drawn boxes on screen to 10
- Detects person holding boxes with count
"""

from collections import defaultdict
import csv
import json
from datetime import datetime
import cv2
import sys
import os
import time
import threading
from queue import Queue, Empty
import numpy as np
import gc
import argparse

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
MAX_DRAWN_PERSONS = 10  # Limit boxes drawn on screen

# Box detection settings
BOX_IOU_THRESHOLD = 0.3
BOX_CLASSES = [24, 25, 26, 28, 39, 41, 56, 58, 63, 64, 65, 66, 67, 73]  # Extended to detect more "box-like" objects
# 24: backpack, 25: umbrella, 26: handbag, 28: suitcase
# 39: bottle, 41: cup, 56: chair (sometimes held), 58: potted plant
# 63: laptop, 64: mouse, 65: remote, 66: keyboard, 67: cell phone
# 73: book
# Note: COCO dataset doesn't have "cardboard box" - need custom training for that

# Alternative: Detect based on shape/appearance
DETECT_GENERIC_BOXES = True  # Enable generic box detection based on color/shape
BOX_COLOR_LOWER = np.array([10, 50, 50])   # Brown/tan color range (HSV)
BOX_COLOR_UPPER = np.array([30, 255, 255])  # Covers cardboard brown tones

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

# Torch for GPU memory management
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def clear_gpu_memory():
    """Clear GPU cache to free memory"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
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
    """Module for detecting and counting boxes held by persons"""
    
    def __init__(self):
        self.box_classes = BOX_CLASSES
        self.iou_threshold = BOX_IOU_THRESHOLD
        self.person_box_history = defaultdict(list)
        self.history_length = 5
        
    def detect_generic_boxes(self, frame, person_bbox):
        """
        Detect generic brown/cardboard boxes using color and contour detection
        This supplements YOLO detection for items not in COCO dataset
        """
        if not DETECT_GENERIC_BOXES:
            return []
        
        px1, py1, px2, py2 = person_bbox
        
        # Expand search area slightly around person
        h, w = frame.shape[:2]
        margin = 30
        search_x1 = max(0, px1 - margin)
        search_y1 = max(0, py1 - margin)
        search_x2 = min(w, px2 + margin)
        search_y2 = min(h, py2 + margin)
        
        person_region = frame[search_y1:search_y2, search_x1:search_x2]
        
        if person_region.size == 0:
            return []
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)
        
        # Create mask for brown/tan/cardboard colors
        mask = cv2.inRange(hsv, BOX_COLOR_LOWER, BOX_COLOR_UPPER)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (must be significant size)
            if area < 2000:  # Minimum box area in pixels
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (boxes are usually squarish or rectangular)
            aspect_ratio = float(w) / h if h > 0 else 0
            if aspect_ratio < 0.3 or aspect_ratio > 3.0:
                continue
            
            # Convert back to frame coordinates
            box_x1 = search_x1 + x
            box_y1 = search_y1 + y
            box_x2 = search_x1 + x + w
            box_y2 = search_y1 + y + h
            
            detected_boxes.append({
                "bbox": [box_x1, box_y1, box_x2, box_y2],
                "class": 999,  # Custom class for generic boxes
                "conf": 0.8,  # Confidence based on color match
                "class_name": "box"
            })
        
        return detected_boxes
        
    def detect_boxes_for_persons(self, persons, boxes, frame=None):
        """Match boxes to persons and count them"""
        results = {}
        
        for i, person in enumerate(persons):
            person_bbox = person["bbox"]
            held_boxes = []
            
            # Check YOLO-detected objects
            for box in boxes:
                box_bbox = box["bbox"]
                
                if is_box_held_by_person(person_bbox, box_bbox, self.iou_threshold):
                    held_boxes.append({
                        "bbox": box_bbox,
                        "class": box["class"],
                        "conf": box["conf"],
                        "class_name": self._get_box_class_name(box["class"])
                    })
            
            # Also check for generic brown boxes if frame provided
            if frame is not None and DETECT_GENERIC_BOXES:
                generic_boxes = self.detect_generic_boxes(frame, person_bbox)
                
                # Filter out duplicates (boxes already detected by YOLO)
                for gen_box in generic_boxes:
                    gb_bbox = gen_box["bbox"]
                    is_duplicate = False
                    
                    for existing in held_boxes:
                        ex_bbox = existing["bbox"]
                        iou = calculate_iou(gb_bbox, ex_bbox)
                        if iou > 0.5:  # High overlap means duplicate
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        held_boxes.append(gen_box)
            
            results[i] = {
                "box_count": len(held_boxes),
                "boxes": held_boxes,
                "status": "holding_boxes" if len(held_boxes) > 0 else "no_boxes"
            }
        
        return results
    
    def _get_box_class_name(self, class_id):
        """Get human-readable name for box class"""
        class_names = {
            24: "backpack",
            25: "umbrella", 
            26: "handbag",
            28: "suitcase",
            39: "bottle",
            41: "cup",
            56: "chair",
            58: "plant",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "phone",
            73: "book",
            999: "box"  # Generic cardboard box
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
    """Memory-optimized recognition using InsightFace"""

    def __init__(self, embeddings_file="known_embeddings.npz"):
        self.embeddings_file = embeddings_file
        self.known_encodings = []
        self.known_names = []
        self.enabled = False
        self.recognition_log = []
        
        self.training_active = False
        self.auto_training_on_unknown = True
        self.unknown_collections = {}
        self.unknown_counter = 1
        self.samples_needed = 5
        self.face_cache = {}
        self.cache_timeout = 2.0
        self.last_collection_time = {}

        if not INSIGHTFACE_AVAILABLE:
            print("❌ InsightFace not available. Face recognition disabled.")
            return

        print("⏳ Initializing InsightFace (Buffalo_L for speed)...")
        try:
            if INSIGHTFACE_USE_GPU:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("   Using GPU for InsightFace")
            else:
                providers = ['CPUExecutionProvider']
                print("   Using CPU for InsightFace (memory saving mode)")
            
            self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=FACE_DETECTION_SIZE)
            import onnxruntime
            print(f"🔎 ONNX Runtime Devices: {onnxruntime.get_available_providers()}")
            self.enabled = True
            self.load_embeddings()
            
            clear_gpu_memory()
            
        except Exception as e:
            print(f"❌ Failed to initialize InsightFace: {e}")
            print("   Try setting INSIGHTFACE_USE_GPU = False")
            self.enabled = False

    def load_embeddings(self):
        """Load InsightFace embeddings"""
        if os.path.exists(self.embeddings_file):
            try:
                data = np.load(self.embeddings_file, allow_pickle=True)
                self.known_encodings = list(data["encodings"])
                self.known_names = list(data["names"])
                print(f"✅ Loaded {len(self.known_names)} InsightFace embeddings")
                
                if self.known_encodings:
                    print(f"   Embedding Shape: {self.known_encodings[0].shape}")
                return True
            except Exception as e:
                print(f"❌ Error loading embeddings: {e}")
                self.known_encodings = []
                self.known_names = []
        else:
            print("ℹ️  No existing embeddings found. Starting fresh.")
        return False

    def save_embeddings(self):
        """Save embeddings to disk"""
        try:
            np.savez_compressed(self.embeddings_file, 
                           encodings=np.array(self.known_encodings), 
                           names=np.array(self.known_names))
            print(f"💾 Saved {len(self.known_names)} embeddings.")
            return True
        except Exception as e:
            print(f"❌ Error saving embeddings: {e}")
            return False

    def toggle_training(self):
        """Toggle training mode"""
        self.training_active = not self.training_active
        if self.training_active:
            print("\n" + "="*40)
            print("🎯 TRAINING MODE: ON")
            print(f"Collecting {self.samples_needed} samples for unknown faces.")
            print("="*40 + "\n")
        else:
            print("\n🛑 TRAINING MODE: OFF\n")
            self.unknown_collections.clear()
        return self.training_active

    def identify_faces_fast(self, frame, person_bboxes):
        """Perform recognition on specific crops"""
        if not self.enabled or not person_bboxes:
            return []

        results = []
        current_time = time.time()
        
        person_bboxes = person_bboxes[:MAX_FACES_PER_FRAME]
        
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            
            h, w = frame.shape[:2]
            m = 20
            cx1, cy1 = max(0, x1-m), max(0, y1-m)
            cx2, cy2 = min(w, x2+m), min(h, y2+m)
            
            person_img = frame[cy1:cy2, cx1:cx2]
            if person_img.size == 0: 
                continue

            ph, pw = person_img.shape[:2]
            if ph < 100 or pw < 100:
                continue

            try:
                faces = self.app.get(person_img)
                
                if not faces:
                    continue
                
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                target_face = faces[0]
                embedding = target_face.normed_embedding

                match_found = False
                best_name = "Unknown"
                best_score = 0.0
                should_lock = False

                if self.known_encodings:
                    scores = np.dot(self.known_encodings, embedding)
                    best_idx = np.argmax(scores)
                    best_score = scores[best_idx]

                    if best_score > 0.45:
                        best_name = self.known_names[best_idx]
                        match_found = True
                        should_lock = (best_score >= 0.8)
                        
                        results.append({
                            "name": best_name,
                            "bbox": bbox,
                            "confidence": float(best_score),
                            "status": "known",
                            "should_lock": should_lock
                        })
                
                if not match_found:
                    if self.auto_training_on_unknown and not self.training_active:
                        self.training_active = True
                        print("\n" + "="*40)
                        print("🎯 AUTO TRAINING MODE: ON")
                        print(f"Unknown face detected! Collecting {self.samples_needed} samples.")
                        print("="*40 + "\n")
                    
                    if self.training_active:
                        unknown_id = self._handle_unknown_collection(embedding, person_img, current_time)
                        
                        collection_len = len(self.unknown_collections.get(unknown_id, []))
                        status = "ready_to_register" if collection_len >= self.samples_needed else "collecting"
                        
                        results.append({
                            "name": unknown_id,
                            "bbox": bbox,
                            "confidence": float(best_score),
                            "status": status,
                            "samples_count": collection_len,
                            "should_lock": False
                        })
                    
            except Exception as e:
                print(f"Face recognition error: {e}")
                continue

        return results

    def _handle_unknown_collection(self, embedding, face_img, current_time):
        """Logic to group unknown faces together during training with time-based throttling"""
        collection_interval = 0.5
        
        for uid, collection in self.unknown_collections.items():
            if not collection: continue
            existing_embs = [item[0] for item in collection]
            avg_emb = np.mean(existing_embs, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            
            sim = np.dot(avg_emb, embedding)
            if sim > 0.5:
                last_time = self.last_collection_time.get(uid, 0)
                if current_time - last_time < collection_interval:
                    return uid
                
                if len(collection) < self.samples_needed:
                    collection.append((embedding, face_img))
                    self.last_collection_time[uid] = current_time
                    print(f"📸 Collected sample {len(collection)}/{self.samples_needed} for {uid}")
                return uid
        
        new_uid = f"Unknown_{self.unknown_counter}"
        self.unknown_counter += 1
        self.unknown_collections[new_uid] = [(embedding, face_img)]
        self.last_collection_time[new_uid] = current_time
        print(f"👤 New unknown person detected: {new_uid}")
        return new_uid

    def discard_unknown(self, unknown_id):
        """Discard/skip an unknown face collection"""
        if unknown_id in self.unknown_collections:
            del self.unknown_collections[unknown_id]
            if unknown_id in self.last_collection_time:
                del self.last_collection_time[unknown_id]
            print(f"🗑️  Discarded {unknown_id}")
            return True
        return False
    
    def register_unknown(self, unknown_id, name):
        """Finalize registration of an unknown ID"""
        if unknown_id not in self.unknown_collections:
            return False, "ID not found"
        
        collection = self.unknown_collections[unknown_id]
        if not collection:
            return False, "No samples"
            
        embeddings = [x[0] for x in collection]
        avg_emb = np.mean(embeddings, axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)
        
        self.known_encodings.append(avg_emb)
        self.known_names.append(name)
        self.save_embeddings()
        
        del self.unknown_collections[unknown_id]
        
        if not os.path.exists("known_faces"): 
            os.makedirs("known_faces")
        
        person_folder = os.path.join("known_faces", name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
            print(f"📁 Created folder: {person_folder}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, (_, face_img) in enumerate(collection):
            img_filename = f"{name}_{timestamp}_{idx+1}.jpg"
            img_path = os.path.join(person_folder, img_filename)
            cv2.imwrite(img_path, face_img)
            print(f"   💾 Saved: {img_filename}")
        
        print(f"✅ Registered {name} with {len(collection)} images in {person_folder}")
        
        return True, f"Registered {name}"

    def create_montage(self, unknown_id):
        """Create visual montage for registration UI"""
        if unknown_id not in self.unknown_collections: 
            return None
        frames = [x[1] for x in self.unknown_collections[unknown_id]]
        if not frames: 
            return None
        
        thumb_size = 100
        thumbs = [cv2.resize(f, (thumb_size, thumb_size)) for f in frames]
        montage = np.hstack(thumbs)
        
        max_montage_width = 600
        if montage.shape[1] > max_montage_width:
            scale = max_montage_width / montage.shape[1]
            new_h = int(montage.shape[0] * scale)
            montage = cv2.resize(montage, (max_montage_width, new_h))
        
        return montage

class PersonTracker:
    """Smart centroid tracker with face recognition caching and confidence tracking"""
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.person_names = {}
        self.person_confidences = {}  # NEW: Track confidence scores
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
        self.person_confidences[self.next_id] = confidence  # Store confidence
        self.entry_times[self.next_id] = datetime.now()
        self.recognition_locked[self.next_id] = (name != "Unknown")
        self.person_box_counts[self.next_id] = box_count
        if bbox is not None:
            self.last_bbox[self.next_id] = bbox
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.person_names[object_id]
        if object_id in self.person_confidences:
            del self.person_confidences[object_id]
        if object_id in self.entry_times: 
            del self.entry_times[object_id]
        if object_id in self.recognition_locked:
            del self.recognition_locked[object_id]
        if object_id in self.last_bbox:
            del self.last_bbox[object_id]
        if object_id in self.person_box_counts:
            del self.person_box_counts[object_id]
        
        self.box_detection_module.clear_history(object_id)

    def update(self, detections, faces, box_results):
        """Update tracker with smart recognition caching and box counting"""
        input_centroids = []
        input_names = []
        input_confidences = []  # NEW: Track confidences
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
                if input_should_lock[i] and input_names[i] != "Unknown":
                    self.recognition_locked[self.next_id - 1] = True
                    print(f"🔒 Locked tracking: ID {self.next_id - 1} = {input_names[i]} (confidence: {input_confidences[i]:.2f})")
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
                        
                        if input_should_lock[col]:
                            self.recognition_locked[obj_id] = True
                            print(f"🔒 Locked tracking: ID {obj_id} = {input_names[col]} (confidence: {input_confidences[col]:.2f})")

                used_rows.add(row)
                used_cols.add(col)

            for i in range(len(input_centroids)):
                if i not in used_cols:
                    self.register(input_centroids[i], input_names[i], input_bboxes[i], 
                                input_box_counts[i], input_confidences[i])
                    if input_should_lock[i] and input_names[i] != "Unknown":
                        self.recognition_locked[self.next_id - 1] = True
                        print(f"🔒 Locked tracking: ID {self.next_id - 1} = {input_names[i]} (confidence: {input_confidences[i]:.2f})")

            for i in range(len(object_ids)):
                if i not in used_rows:
                    obj_id = object_ids[i]
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)

        return self.objects
    
    def get_persons_needing_recognition(self):
        """Return list of tracked person IDs that need face recognition"""
        need_recognition = []
        for obj_id in self.objects.keys():
            if not self.recognition_locked.get(obj_id, False) and self.disappeared[obj_id] == 0:
                if obj_id in self.last_bbox:
                    need_recognition.append((obj_id, self.last_bbox[obj_id]))
        return need_recognition

class AIWorker:
    """Background AI processing thread with box detection"""

    def __init__(self, detector, face_system, tracker):
        self.detector = detector
        self.face_system = face_system
        self.tracker = tracker
        self.box_detector = BoxDetectionModule()
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
        print("🚀 AI Worker Started (with Box Detection)")
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

                box_results = self.box_detector.detect_boxes_for_persons(
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
                    "frame_shape": process_frame.shape[:2]
                }
                
                if self.result_queue.full(): 
                    self.result_queue.get()
                self.result_queue.put(result)

            except Empty:
                continue
            except Exception as e:
                print(f"AI Error: {e}")
                clear_gpu_memory()

class OptimizedInventoryDoorCCTV:
    def __init__(self, rtsp_url, headless=False):
        self.rtsp_url = rtsp_url
        self.log_file = "inventory_log.csv"
        self.headless = headless
        
        self.face_system = FaceRecognitionSystem()
        self.tracker = PersonTracker()
        
        print("Loading YOLO...")
        device = 0 if YOLO_USE_GPU else 'cpu'
        self.detector = YOLO("yolo11n.pt")
        if not YOLO_USE_GPU:
            print("   Using CPU for YOLO (memory saving mode)")
        
        clear_gpu_memory()
        
        self.ai_worker = AIWorker(self.detector, self.face_system, self.tracker)
        
        self.running = True
        self.latest_result = None
        self.pending_registrations = []

        self.exit_y_ratio = 0.7
        
        self._init_log()

    def _init_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("Timestamp,Person_ID,Name,Box_Count,Box_Details,Snapshot\n")

    def start(self):
        print(f"Connecting to {self.rtsp_url}...")
        print(f"Configuration: YOLO GPU={YOLO_USE_GPU}, InsightFace GPU={INSIGHTFACE_USE_GPU}")
        print(f"Processing Resolution: {PROCESS_RESOLUTION}p, Max Faces: {MAX_FACES_PER_FRAME}")
        print(f"Face Detection Size: {FACE_DETECTION_SIZE}")
        print(f"Box Detection: ENABLED (IOU Threshold: {BOX_IOU_THRESHOLD})")
        print(f"Max Drawn Persons: {MAX_DRAWN_PERSONS}")
        print(f"Headless Mode: {self.headless}")
        
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.ai_worker.start()
        
        last_submit = 0
        last_ai_update = 0
        display_fps = 0
        display_frame_count = 0
        display_start = time.time()
        
        if not self.headless:
            cv2.namedWindow("CCTV with Box Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("CCTV with Box Detection", 1280, 720)

        frame_count = 0
        start_time = time.time()

        print("\n" + "="*50)
        if self.headless:
            print("⚡ HEADLESS MODE ACTIVE - Performance Monitoring")
        else:
            print("🖥️  DISPLAY MODE ACTIVE - High Refresh Rate Display")
        print("="*50 + "\n")

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("Stream lost, retrying...")
                    time.sleep(2)
                    cap = cv2.VideoCapture(self.rtsp_url)
                    continue

                now = time.time()
                
                # Submit frames for AI processing at slower rate (10 FPS)
                if now - last_submit > 0.1:
                    self.ai_worker.submit_frame(frame.copy())
                    last_submit = now

                # Check for AI results (non-blocking)
                res = self.ai_worker.get_result()
                if res:
                    self.latest_result = res
                    self._process_logic(res)
                    last_ai_update = now

                frame_count += 1
                if self.headless and frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"[HEADLESS] Frames: {frame_count:4d} | FPS: {fps:5.1f} | AI Latency: {self.ai_worker.avg_time_ms:5.1f}ms")

                # ALWAYS draw and display at camera frame rate (not limited by AI)
                if not self.headless:
                    display_frame = self._draw_overlay(frame)
                    
                    # Calculate display FPS
                    display_frame_count += 1
                    if display_frame_count % 30 == 0:
                        display_elapsed = time.time() - display_start
                        display_fps = 30 / display_elapsed
                        display_start = time.time()
                        display_frame_count = 0
                    
                    # Add display FPS and AI update info
                    ai_age = now - last_ai_update if last_ai_update > 0 else 0
                    fps_text = f"Display: {display_fps:.1f} FPS | AI: {self.ai_worker.avg_time_ms:.1f}ms (age: {ai_age:.2f}s)"
                    cv2.putText(display_frame, fps_text, (10, display_frame.shape[0] - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    cv2.imshow("CCTV with Box Detection", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): 
                        break
                    if key == ord('t'): 
                        self.face_system.toggle_training()
                    if key == 13:
                        self._handle_registration()
                    if key == 27:
                        self._handle_skip()
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord('t'): 
                        self.face_system.toggle_training()
                        
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
        finally:
            cap.release()
            if not self.headless:
                cv2.destroyAllWindows()
            print(f"\n📊 Final Stats: {frame_count} frames processed")

    def _process_logic(self, res):
        """Update tracker and check for exits with box detection"""
        detections = res["detections"]["persons"]
        faces = res["faces"]
        box_results = res.get("box_results", {})
        
        self.tracker.update(detections, faces, box_results)

    def _draw_overlay(self, frame):
        """Draws bounding boxes and info with box detection visualization - LIMITED TO MAX_DRAWN_PERSONS"""
        if self.latest_result:
            h, w = frame.shape[:2]
            scale = self.latest_result["scale_factor"]
            
            if scale != 1.0:
                display_h, display_w = self.latest_result["frame_shape"]
                display_frame = cv2.resize(frame, (display_w, display_h))
            else:
                display_frame = frame

            # Sort tracked persons by ID (newest first) and limit to MAX_DRAWN_PERSONS
            tracked_items = list(self.tracker.objects.items())
            tracked_items.sort(key=lambda x: x[0], reverse=True)  # Sort by ID descending
            tracked_items = tracked_items[:MAX_DRAWN_PERSONS]  # Limit displayed persons
            
            drawn_count = 0
            for obj_id, centroid in tracked_items:
                if obj_id in self.tracker.last_bbox:
                    x1, y1, x2, y2 = self.tracker.last_bbox[obj_id]
                    name = self.tracker.person_names.get(obj_id, "Unknown")
                    is_locked = self.tracker.recognition_locked.get(obj_id, False)
                    box_count = self.tracker.person_box_counts.get(obj_id, 0)
                    confidence = self.tracker.person_confidences.get(obj_id, 0.0)
                    
                    # Color coding
                    if is_locked and name != "Unknown":
                        color = (0, 255, 0)  # Green
                        label = f"{name} [{confidence:.2f}] ID:{obj_id} 🔒"
                    elif name != "Unknown":
                        color = (255, 255, 0)  # Yellow
                        label = f"{name}? [{confidence:.2f}] ID:{obj_id}"
                    else:
                        color = (0, 165, 255)  # Orange
                        label = f"Person ID:{obj_id}"
                    
                    # Add box count to label
                    if box_count > 0:
                        label += f" 📦x{box_count}"
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    drawn_count += 1

            # Draw detected boxes (including generic brown boxes)
            for box_det in self.latest_result["detections"]["boxes"]:
                bx1, by1, bx2, by2 = box_det["bbox"]
                box_class = box_det["class"]
                box_conf = box_det["conf"]
                
                cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (255, 255, 0), 1)
                
                class_names = {
                    24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase",
                    39: "bottle", 41: "cup", 56: "chair", 58: "plant",
                    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
                    67: "phone", 73: "book", 999: "box"
                }
                class_name = class_names.get(box_class, f"obj_{box_class}")
                label = f"{class_name} {box_conf:.2f}"
                cv2.putText(display_frame, label, (bx1, by1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            # Also draw generic detected boxes from results
            if "box_results" in self.latest_result:
                for person_idx, box_result in self.latest_result["box_results"].items():
                    for held_box in box_result.get("boxes", []):
                        if held_box.get("class") == 999:  # Generic box
                            bx1, by1, bx2, by2 = held_box["bbox"]
                            cv2.rectangle(display_frame, (bx1, by1), (bx2, by2), (0, 140, 255), 2)  # Orange for generic
                            label = f"BOX {held_box['conf']:.2f}"
                            cv2.putText(display_frame, label, (bx1, by1-5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)

            # Face boxes for training/registration
            for f in self.latest_result["faces"]:
                fx1, fy1, fx2, fy2 = f["bbox"]
                
                if f["status"] == "ready_to_register":
                    cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
                    label = f"{f['name']} [READY]"
                    cv2.putText(display_frame, label, (fx1, fy1-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    if f["name"] not in [x[0] for x in self.pending_registrations]:
                        montage = self.face_system.create_montage(f["name"])
                        if montage is not None:
                            self.pending_registrations.append((f["name"], montage))
                            
                elif f["status"] == "collecting":
                    cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (0, 165, 255), 2)
                    label = f"Collecting {f['samples_count']}/5"
                    cv2.putText(display_frame, label, (fx1, fy1-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

            # Stats overlay
            cv2.putText(display_frame, f"AI: {self.ai_worker.avg_time_ms:.1f}ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Tracking stats
            locked_count = sum(1 for locked in self.tracker.recognition_locked.values() if locked)
            total_tracked = len(self.tracker.objects)
            total_boxes = sum(self.tracker.person_box_counts.values())
            
            cv2.putText(display_frame, f"Tracked: {total_tracked} | Shown: {drawn_count}/{MAX_DRAWN_PERSONS} | Locked: {locked_count} | Boxes: {total_boxes}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            if self.face_system.training_active:
                mode_text = "TRAINING MODE (AUTO)" if self.face_system.auto_training_on_unknown else "TRAINING MODE"
                cv2.putText(display_frame, mode_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif self.face_system.auto_training_on_unknown:
                cv2.putText(display_frame, "AUTO TRAINING: READY", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)

            if self.pending_registrations:
                uid, montage = self.pending_registrations[0]
                mh, mw = montage.shape[:2]
                
                dh, dw = display_frame.shape[:2]
                
                y_pos = 120
                x_pos = 10
                
                if y_pos + mh <= dh and x_pos + mw <= dw:
                    display_frame[y_pos:y_pos+mh, x_pos:x_pos+mw] = montage
                    cv2.putText(display_frame, "ENTER=Register | ESC=Skip", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                else:
                    cv2.putText(display_frame, f"{uid} READY - ENTER=Register | ESC=Skip", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            return display_frame
        
        return cv2.resize(frame, (PROCESS_RESOLUTION, int(frame.shape[0]*(PROCESS_RESOLUTION/frame.shape[1])))) if frame.shape[1] > PROCESS_RESOLUTION else frame

    def _handle_registration(self):
        """Handle face registration with name input"""
        if self.pending_registrations:
            uid, _ = self.pending_registrations[0]
            try:
                import pyautogui
                new_name = pyautogui.prompt(text=f'Enter name for {uid}', title='Face Registration', default='')
            except:
                print(f"\nEnter name for {uid}: ", end='')
                new_name = input().strip()
            
            if new_name:
                self.face_system.register_unknown(uid, new_name)
                self.pending_registrations.pop(0)
            else:
                print("⚠️  Registration cancelled (no name entered)")
    
    def _handle_skip(self):
        """Handle skipping/discarding a face registration"""
        if self.pending_registrations:
            uid, _ = self.pending_registrations[0]
            self.face_system.discard_unknown(uid)
            self.pending_registrations.pop(0)
            print(f"⏭️  Skipped registration for {uid}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inventory Door CCTV System with Box Detection')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no display window)')
    args = parser.parse_args()
    
    # CONFIGURATION
    RTSP_URL = "rtsp://fasspay:fasspay2025@192.168.122.127:554/stream1"
    
    print("\n" + "="*50)
    print("CCTV SYSTEM WITH BOX DETECTION & SIM SCORES")
    print("="*50)
    print(f"YOLO GPU: {YOLO_USE_GPU}")
    print(f"InsightFace GPU: {INSIGHTFACE_USE_GPU}")
    print(f"Processing Resolution: {PROCESS_RESOLUTION}p")
    print(f"YOLO Image Size: {YOLO_IMG_SIZE}px")
    print(f"Face Detection Size: {FACE_DETECTION_SIZE}")
    print(f"Max Faces Per Frame: {MAX_FACES_PER_FRAME}")
    print(f"Max Drawn Persons: {MAX_DRAWN_PERSONS}")
    print(f"Box Detection: ENABLED")
    print(f"Box IOU Threshold: {BOX_IOU_THRESHOLD}")
    print(f"Box Classes: {BOX_CLASSES}")
    print(f"Headless Mode: {args.headless}")
    print("="*50 + "\n")
    
    if args.headless:
        print("⚡ HEADLESS MODE - No display window (optimal performance)")
        print("   Press 'q' to exit (window must have focus)")
        print("   Press 't' to toggle training mode")
        print("   Press Ctrl+C for emergency exit")
        print("   Monitoring AI latency...\n")
    
    app = OptimizedInventoryDoorCCTV(RTSP_URL, headless=args.headless)
    app.start()
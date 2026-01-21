#!/usr/bin/env python3
"""
MEMORY-OPTIMIZED Inventory Door CCTV System with HEADLESS MODE
Solutions for CUDA OOM:
1. Use CPU for one of the models
2. Reduce model sizes
3. Process fewer faces simultaneously
4. Clear GPU cache between operations
5. Run headless for better performance
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
YOLO_USE_GPU = True  # Set to False to use CPU for YOLO
INSIGHTFACE_USE_GPU = True  # Set to False to use CPU for InsightFace
MAX_FACES_PER_FRAME = 1  # Limit faces processed per frame (reduced from 2)
PROCESS_RESOLUTION = 1080  # Reduced from 1080 for speed
FACE_DETECTION_SIZE = (416, 416)  # Reduced from (320, 320) for speed
YOLO_IMG_SIZE = 416  # Smaller YOLO input size for speed (default is 640)

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

class FaceRecognitionSystem:
    """Memory-optimized recognition using InsightFace"""

    def __init__(self, embeddings_file="known_embeddings.npz"):
        self.embeddings_file = embeddings_file
        self.known_encodings = []
        self.known_names = []
        self.enabled = False
        self.recognition_log = []
        
        # Training/Collection state
        self.training_active = False
        self.auto_training_on_unknown = True  # NEW: Auto-enable training for unknowns
        self.unknown_collections = {}
        self.unknown_counter = 1
        self.samples_needed = 5
        self.face_cache = {}
        self.cache_timeout = 2.0
        self.last_collection_time = {}  # Track when we last collected for each unknown

        if not INSIGHTFACE_AVAILABLE:
            print("❌ InsightFace not available. Face recognition disabled.")
            return

        print("⏳ Initializing InsightFace (Buffalo_L for speed)...")
        try:
            # Choose execution provider based on config
            if INSIGHTFACE_USE_GPU:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print("   Using GPU for InsightFace")
            else:
                providers = ['CPUExecutionProvider']
                print("   Using CPU for InsightFace (memory saving mode)")
            
            # Use buffalo_s (smaller, faster model) instead of buffalo_l
            self.app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=FACE_DETECTION_SIZE)
            import onnxruntime
            print(f"🔎 ONNX Runtime Devices: {onnxruntime.get_available_providers()}")
            self.enabled = True
            self.load_embeddings()
            
            # Clear any initialization memory
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
        """Perform recognition on specific crops - HIGH RESOLUTION for accuracy"""
        if not self.enabled or not person_bboxes:
            return []

        results = []
        current_time = time.time()
        
        # Limit number of faces to process
        person_bboxes = person_bboxes[:MAX_FACES_PER_FRAME]
        
        # Iterate over detected persons
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            
            # Larger margin for better face context
            h, w = frame.shape[:2]
            m = 20  # Reduced margin from 40 to 20 pixels for speed
            cx1, cy1 = max(0, x1-m), max(0, y1-m)
            cx2, cy2 = min(w, x2+m), min(h, y2+m)
            
            person_img = frame[cy1:cy2, cx1:cx2]
            if person_img.size == 0: 
                continue

            # Ensure person crop is at least 200x200 for good face detection
            ph, pw = person_img.shape[:2]
            if ph < 100 or pw < 100:
                continue

            try:
                # InsightFace Detection
                faces = self.app.get(person_img)
                
                if not faces:
                    continue
                
                # Take largest face
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                target_face = faces[0]
                embedding = target_face.normed_embedding

                match_found = False
                best_name = "Unknown"
                best_score = 0.0
                should_lock = False  # NEW: Track if confidence is high enough to lock

                # Compare with KNOWN faces
                if self.known_encodings:
                    scores = np.dot(self.known_encodings, embedding)
                    best_idx = np.argmax(scores)
                    best_score = scores[best_idx]

                    if best_score > 0.45:  # Lower threshold for buffalo_s (was 0.40)
                        best_name = self.known_names[best_idx]
                        match_found = True
                        
                        # Only lock if confidence >= 0.5
                        should_lock = (best_score >= 0.8)
                        
                        results.append({
                            "name": best_name,
                            "bbox": bbox,
                            "confidence": float(best_score),
                            "status": "known",
                            "should_lock": should_lock  # NEW: Pass locking decision
                        })
                
                # If NOT known - AUTO-ENABLE TRAINING or collect if training active
                if not match_found:
                    # Auto-enable training when unknown detected
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
                            "should_lock": False  # Never lock unknown faces
                        })
                    
            except Exception as e:
                print(f"Face recognition error: {e}")
                continue

        return results

    def _handle_unknown_collection(self, embedding, face_img, current_time):
        """Logic to group unknown faces together during training with time-based throttling"""
        collection_interval = 0.5  # Minimum 0.5 seconds between collections for same person
        
        for uid, collection in self.unknown_collections.items():
            if not collection: continue
            existing_embs = [item[0] for item in collection]
            avg_emb = np.mean(existing_embs, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            
            sim = np.dot(avg_emb, embedding)
            if sim > 0.5:
                # Check time throttle
                last_time = self.last_collection_time.get(uid, 0)
                if current_time - last_time < collection_interval:
                    return uid  # Skip collection, too soon
                
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
        
        # Create person-specific folder
        if not os.path.exists("known_faces"): 
            os.makedirs("known_faces")
        
        person_folder = os.path.join("known_faces", name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
            print(f"📁 Created folder: {person_folder}")
        
        # Save all collected samples to person's folder
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
        
        # Create smaller thumbnails to fit in reduced resolution
        thumb_size = 100  # Reduced from 150
        thumbs = [cv2.resize(f, (thumb_size, thumb_size)) for f in frames]
        montage = np.hstack(thumbs)
        
        # Ensure montage fits in frame (max width check)
        max_montage_width = 600  # Safe width for 640p display
        if montage.shape[1] > max_montage_width:
            scale = max_montage_width / montage.shape[1]
            new_h = int(montage.shape[0] * scale)
            montage = cv2.resize(montage, (max_montage_width, new_h))
        
        return montage

class PersonTracker:
    """Smart centroid tracker with face recognition caching"""
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.person_names = {}
        self.entry_times = {}
        self.has_exited = set()
        self.recognition_locked = {}  # NEW: Track if person has been recognized
        self.last_bbox = {}  # NEW: Store last bounding box for each tracked person

    def register(self, centroid, name="Unknown", bbox=None):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.person_names[self.next_id] = name
        self.entry_times[self.next_id] = datetime.now()
        self.recognition_locked[self.next_id] = (name != "Unknown")  # Lock if recognized
        if bbox is not None:
            self.last_bbox[self.next_id] = bbox
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.person_names[object_id]
        if object_id in self.entry_times: 
            del self.entry_times[object_id]
        if object_id in self.recognition_locked:
            del self.recognition_locked[object_id]
        if object_id in self.last_bbox:
            del self.last_bbox[object_id]

    def update(self, detections, faces):
        """Update tracker with smart recognition caching"""
        input_centroids = []
        input_names = []
        input_bboxes = []
        input_should_lock = []  # NEW: Track which recognitions should be locked
        
        for det in detections:
            bbox = det["bbox"]
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            input_centroids.append((cx, cy))
            input_bboxes.append(bbox)
            
            # Default to Unknown
            name = "Unknown"
            should_lock = False
            
            # Try to match with recognized faces
            for face in faces:
                fx1, fy1, fx2, fy2 = face["bbox"]
                bx1, by1, bx2, by2 = bbox
                
                # Check if face bbox overlaps with person bbox
                if (fx1 >= bx1 and fy1 >= by1 and fx2 <= bx2 and fy2 <= by2):
                    name = face["name"]
                    should_lock = face.get("should_lock", False)
                    break
            
            input_names.append(name)
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
                self.register(input_centroids[i], input_names[i], input_bboxes[i])
                # Lock immediately if confidence is high enough
                if input_should_lock[i] and input_names[i] != "Unknown":
                    self.recognition_locked[self.next_id - 1] = True
                    print(f"🔒 Locked tracking: ID {self.next_id - 1} = {input_names[i]} (high confidence)")
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
                
                # SMART UPDATE: Only update name if not locked OR if new high-confidence recognition found
                if not self.recognition_locked.get(obj_id, False):
                    # Not locked yet, update if we got a recognition
                    if input_names[col] != "Unknown":
                        self.person_names[obj_id] = input_names[col]
                        
                        # Lock only if confidence >= 0.6 (increased from 0.5)
                        if input_should_lock[col]:
                            self.recognition_locked[obj_id] = True
                            print(f"🔒 Locked tracking: ID {obj_id} = {input_names[col]} (confidence >= 0.6)")
                # If locked, keep existing name (don't overwrite with "Unknown")

                used_rows.add(row)
                used_cols.add(col)

            for i in range(len(input_centroids)):
                if i not in used_cols:
                    self.register(input_centroids[i], input_names[i], input_bboxes[i])
                    # Lock immediately if confidence is high enough
                    if input_should_lock[i] and input_names[i] != "Unknown":
                        self.recognition_locked[self.next_id - 1] = True
                        print(f"🔒 Locked tracking: ID {self.next_id - 1} = {input_names[i]} (high confidence)")

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
            # Need recognition if not locked (unknown) and still visible
            if not self.recognition_locked.get(obj_id, False) and self.disappeared[obj_id] == 0:
                if obj_id in self.last_bbox:
                    need_recognition.append((obj_id, self.last_bbox[obj_id]))
        return need_recognition

class AIWorker:
    """Background AI processing thread - MEMORY OPTIMIZED with FASTER updates"""

    def __init__(self, detector, face_system, tracker):
        self.detector = detector
        self.face_system = face_system
        self.tracker = tracker  # NEW: Reference to tracker for smart FR
        self.frame_queue = Queue(maxsize=3)  # Increased queue size
        self.result_queue = Queue(maxsize=3)  # Increased queue size
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

                # Resolution cap - use PROCESS_RESOLUTION
                h, w = frame.shape[:2]
                scale_factor = 1.0
                
                if w > PROCESS_RESOLUTION:
                    scale_factor = PROCESS_RESOLUTION / w
                    new_h = int(h * scale_factor)
                    process_frame = cv2.resize(frame, (PROCESS_RESOLUTION, new_h))
                else:
                    process_frame = frame
                
                # YOLO Detect with smaller image size for speed
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
                    elif cls in [24, 25, 26, 28]:
                        detections["boxes"].append(det)

                # Face Recognition - SMART: Only run on persons needing recognition
                persons_needing_recognition = []
                
                # Filter persons: only run FR on those not already locked
                for p in detections["persons"]:
                    bbox = p["bbox"]
                    cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                    
                    # Check if this bbox matches any locked person in tracker
                    needs_fr = True
                    for obj_id, tracked_centroid in self.tracker.objects.items():
                        # Calculate distance between detection and tracked object
                        dist = np.sqrt((cx - tracked_centroid[0])**2 + (cy - tracked_centroid[1])**2)
                        
                        # If close to a tracked object and that object is locked, skip FR
                        if dist < 100 and self.tracker.recognition_locked.get(obj_id, False):
                            needs_fr = False
                            break
                    
                    if needs_fr:
                        persons_needing_recognition.append(bbox)
                
                # Limit to MAX_FACES_PER_FRAME
                person_bboxes = persons_needing_recognition[:MAX_FACES_PER_FRAME]
                
                faces = self.face_system.identify_faces_fast(process_frame, person_bboxes)

                # Periodic GPU memory cleanup (less frequent for speed)
                frame_count += 1
                if frame_count % 100 == 0:  # Every 100 frames instead of 30
                    clear_gpu_memory()

                elapsed = (time.time() - start_time) * 1000
                self.avg_time_ms = self.avg_time_ms * 0.9 + elapsed * 0.1
                
                result = {
                    "detections": detections,
                    "faces": faces,
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
        
        # Init Subsystems
        self.face_system = FaceRecognitionSystem()
        self.tracker = PersonTracker()
        
        print("Loading YOLO...")
        device = 0 if YOLO_USE_GPU else 'cpu'
        self.detector = YOLO("yolo11n.pt")
        if not YOLO_USE_GPU:
            print("   Using CPU for YOLO (memory saving mode)")
        
        # Clear memory after loading
        clear_gpu_memory()
        
        self.ai_worker = AIWorker(self.detector, self.face_system, self.tracker)  # Pass tracker
        
        self.running = True
        self.latest_result = None
        self.pending_registrations = []

        self.exit_y_ratio = 0.7
        
        self._init_log()

    def _init_log(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w") as f:
                f.write("Timestamp,Person_ID,Name,Boxes,Snapshot\n")

    def start(self):
        print(f"Connecting to {self.rtsp_url}...")
        print(f"Configuration: YOLO GPU={YOLO_USE_GPU}, InsightFace GPU={INSIGHTFACE_USE_GPU}")
        print(f"Processing Resolution: {PROCESS_RESOLUTION}p, Max Faces: {MAX_FACES_PER_FRAME}")
        print(f"Face Detection Size: {FACE_DETECTION_SIZE}")
        print(f"Headless Mode: {self.headless}")
        
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Request higher resolution from camera if available
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.ai_worker.start()
        
        last_submit = 0
        
        if not self.headless:
            cv2.namedWindow("CCTV Memory Optimized", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("CCTV Memory Optimized", 1280, 720)

        frame_count = 0
        start_time = time.time()

        print("\n" + "="*50)
        if self.headless:
            print("⚡ HEADLESS MODE ACTIVE - Performance Monitoring")
        else:
            print("🖥️  DISPLAY MODE ACTIVE")
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
                if now - last_submit > 0.1:
                    self.ai_worker.submit_frame(frame)
                    last_submit = now

                res = self.ai_worker.get_result()
                if res:
                    self.latest_result = res
                    self._process_logic(res)

                # Performance monitoring in headless mode
                frame_count += 1
                if self.headless and frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    print(f"[HEADLESS] Frames: {frame_count:4d} | FPS: {fps:5.1f} | AI Latency: {self.ai_worker.avg_time_ms:5.1f}ms")

                if not self.headless:
                    display_frame = self._draw_overlay(frame)
                    cv2.imshow("CCTV Memory Optimized", display_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'): 
                        break
                    if key == ord('t'): 
                        self.face_system.toggle_training()
                    if key == 13:  # Enter key
                        self._handle_registration()
                    if key == 27:  # ESC key
                        self._handle_skip()
                else:
                    # In headless mode, minimal key checking
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
        """Update tracker and check for exits with smart FR filtering"""
        detections = res["detections"]["persons"]
        faces = res["faces"]
        
        # First, update tracker to get current state
        self.tracker.update(detections, faces)
        
        # Now filter detections to only run FR on unlocked persons in next frame
        # This is handled in the AI worker by checking tracker state

    def _draw_overlay(self, frame):
        """Draws bounding boxes and info with smart tracking display - NO DUPLICATES"""
        if self.latest_result:
            h, w = frame.shape[:2]
            scale = self.latest_result["scale_factor"]
            
            if scale != 1.0:
                display_h, display_w = self.latest_result["frame_shape"]
                display_frame = cv2.resize(frame, (display_w, display_h))
            else:
                display_frame = frame

            # Draw ONLY tracked persons (no duplicates!)
            for obj_id, centroid in self.tracker.objects.items():
                if obj_id in self.tracker.last_bbox:
                    x1, y1, x2, y2 = self.tracker.last_bbox[obj_id]
                    name = self.tracker.person_names.get(obj_id, "Unknown")
                    is_locked = self.tracker.recognition_locked.get(obj_id, False)
                    
                    # Color coding
                    if is_locked and name != "Unknown":
                        color = (0, 255, 0)  # Green - recognized and locked (confidence >= 0.6)
                        label = f"{name} [ID:{obj_id}] 🔒"
                    elif name != "Unknown":
                        color = (255, 255, 0)  # Yellow - recognized but not locked (confidence < 0.6)
                        label = f"{name}? [ID:{obj_id}]"
                    else:
                        color = (0, 165, 255)  # Orange - needs recognition
                        label = f"Person [ID:{obj_id}]"
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)  # Reduced font size

            # Only draw face boxes for training/registration (not for regular recognition)
            for f in self.latest_result["faces"]:
                fx1, fy1, fx2, fy2 = f["bbox"]
                
                if f["status"] == "ready_to_register":
                    # Draw face box for registration
                    cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (0, 255, 255), 2)
                    label = f"{f['name']} [READY]"
                    cv2.putText(display_frame, label, (fx1, fy1-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  # Reduced font
                    
                    if f["name"] not in [x[0] for x in self.pending_registrations]:
                        montage = self.face_system.create_montage(f["name"])
                        if montage is not None:
                            self.pending_registrations.append((f["name"], montage))
                            
                elif f["status"] == "collecting":
                    # Draw face box during collection
                    cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (0, 165, 255), 2)
                    label = f"Collecting {f['samples_count']}/5"
                    cv2.putText(display_frame, label, (fx1, fy1-30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)  # Reduced font

            # Stats and status - reduced font sizes
            cv2.putText(display_frame, f"AI: {self.ai_worker.avg_time_ms:.1f}ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)  # Reduced font
            
            # Show tracking stats
            locked_count = sum(1 for locked in self.tracker.recognition_locked.values() if locked)
            total_tracked = len(self.tracker.objects)
            cv2.putText(display_frame, f"Tracked: {total_tracked} | Locked: {locked_count}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)  # Reduced font
            
            if self.face_system.training_active:
                mode_text = "TRAINING MODE (AUTO)" if self.face_system.auto_training_on_unknown else "TRAINING MODE"
                cv2.putText(display_frame, mode_text, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)  # Reduced font
            elif self.face_system.auto_training_on_unknown:
                cv2.putText(display_frame, "AUTO TRAINING: READY", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)  # Reduced font

            if self.pending_registrations:
                uid, montage = self.pending_registrations[0]
                mh, mw = montage.shape[:2]
                
                # Ensure montage fits within display frame
                dh, dw = display_frame.shape[:2]
                
                # Position montage safely
                y_pos = 120
                x_pos = 10
                
                # Check if montage fits, if not skip display or resize
                if y_pos + mh <= dh and x_pos + mw <= dw:
                    display_frame[y_pos:y_pos+mh, x_pos:x_pos+mw] = montage
                    cv2.putText(display_frame, "ENTER=Register | ESC=Skip", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)  # Reduced font
                else:
                    # Montage too large, just show text
                    cv2.putText(display_frame, f"{uid} READY - ENTER=Register | ESC=Skip", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)  # Reduced font

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
                # Fallback if pyautogui not available
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
    parser = argparse.ArgumentParser(description='Inventory Door CCTV System')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no display window)')
    args = parser.parse_args()
    
    # CONFIGURATION
    RTSP_URL = "rtsp://fasspay:fasspay2025@192.168.122.127:554/stream1"
    
    print("\n" + "="*50)
    print("HIGH RESOLUTION RECOGNITION SETTINGS")
    print("="*50)
    print(f"YOLO GPU: {YOLO_USE_GPU}")
    print(f"InsightFace GPU: {INSIGHTFACE_USE_GPU}")
    print(f"Processing Resolution: {PROCESS_RESOLUTION}p")
    print(f"YOLO Image Size: {YOLO_IMG_SIZE}px")
    print(f"Face Detection Size: {FACE_DETECTION_SIZE}")
    print(f"Max Faces Per Frame: {MAX_FACES_PER_FRAME}")
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
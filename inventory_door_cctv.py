#!/usr/bin/env python3
"""
MEMORY-OPTIMIZED Inventory Door CCTV System
Solutions for CUDA OOM:
1. Use CPU for one of the models
2. Reduce model sizes
3. Process fewer faces simultaneously
4. Clear GPU cache between operations
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

# Suppress warnings
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Memory optimization settings
YOLO_USE_GPU = True  # Set to False to use CPU for YOLO
INSIGHTFACE_USE_GPU = True  # Set to False to use CPU for InsightFace
MAX_FACES_PER_FRAME = 2  # Limit faces processed per frame
PROCESS_RESOLUTION = 1080  # Higher resolution for better face recognition (was 720)
FACE_DETECTION_SIZE = (320, 320)  # Higher face detection resolution (was 320x320)

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
        self.unknown_collections = {}
        self.unknown_counter = 1
        self.samples_needed = 5
        self.face_cache = {}
        self.cache_timeout = 2.0

        if not INSIGHTFACE_AVAILABLE:
            print("❌ InsightFace not available. Face recognition disabled.")
            return

        print("⏳ Initializing InsightFace (Buffalo_L)...")
        try:
            # Choose execution provider based on config
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
            print("🎓 TRAINING MODE: ON")
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
        
        # Limit number of faces to process
        person_bboxes = person_bboxes[:MAX_FACES_PER_FRAME]
        
        # Iterate over detected persons
        for bbox in person_bboxes:
            x1, y1, x2, y2 = bbox
            
            # Larger margin for better face context
            h, w = frame.shape[:2]
            m = 40  # Increased margin from 20 to 40 pixels
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

                # Compare with KNOWN faces
                if self.known_encodings:
                    scores = np.dot(self.known_encodings, embedding)
                    best_idx = np.argmax(scores)
                    best_score = scores[best_idx]

                    if best_score > 0.40:  # Slightly lower threshold for higher res (was 0.45)
                        best_name = self.known_names[best_idx]
                        match_found = True
                        
                        results.append({
                            "name": best_name,
                            "bbox": bbox,
                            "confidence": float(best_score),
                            "status": "known"
                        })
                
                # If NOT known and TRAINING is active
                if not match_found and self.training_active:
                    unknown_id = self._handle_unknown_collection(embedding, person_img)
                    
                    collection_len = len(self.unknown_collections.get(unknown_id, []))
                    status = "ready_to_register" if collection_len >= self.samples_needed else "collecting"
                    
                    results.append({
                        "name": unknown_id,
                        "bbox": bbox,
                        "confidence": float(best_score),
                        "status": status,
                        "samples_count": collection_len
                    })
                    
            except Exception as e:
                print(f"Face recognition error: {e}")
                continue

        return results

    def _handle_unknown_collection(self, embedding, face_img):
        """Logic to group unknown faces together during training"""
        for uid, collection in self.unknown_collections.items():
            if not collection: continue
            existing_embs = [item[0] for item in collection]
            avg_emb = np.mean(existing_embs, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb)
            
            sim = np.dot(avg_emb, embedding)
            if sim > 0.5:
                if len(collection) < self.samples_needed:
                    collection.append((embedding, face_img))
                    print(f"📸 Collected sample {len(collection)}/{self.samples_needed} for {uid}")
                return uid
        
        new_uid = f"Unknown_{self.unknown_counter}"
        self.unknown_counter += 1
        self.unknown_collections[new_uid] = [(embedding, face_img)]
        print(f"👤 New unknown person detected: {new_uid}")
        return new_uid

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
        ref_img = collection[0][1]
        cv2.imwrite(f"known_faces/{name}.jpg", ref_img)
        
        return True, f"Registered {name}"

    def create_montage(self, unknown_id):
        """Create visual montage for registration UI"""
        if unknown_id not in self.unknown_collections: 
            return None
        frames = [x[1] for x in self.unknown_collections[unknown_id]]
        if not frames: 
            return None
        
        thumbs = [cv2.resize(f, (150, 150)) for f in frames]
        return np.hstack(thumbs)

class PersonTracker:
    """Simple centroid tracker"""
    def __init__(self, max_disappeared=30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.person_names = {}
        self.entry_times = {}
        self.has_exited = set()

    def register(self, centroid, name="Unknown"):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.person_names[self.next_id] = name
        self.entry_times[self.next_id] = datetime.now()
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.person_names[object_id]
        if object_id in self.entry_times: 
            del self.entry_times[object_id]

    def update(self, detections, faces):
        input_centroids = []
        input_names = []
        
        for det in detections:
            bbox = det["bbox"]
            cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
            input_centroids.append((cx, cy))
            
            name = "Unknown"
            for face in faces:
                fx1, fy1, fx2, fy2 = face["bbox"]
                if fx1 == bbox[0] and fy1 == bbox[1]:
                    name = face["name"]
            input_names.append(name)

        if len(input_centroids) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.array(input_centroids)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_names[i])
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
                
                if self.person_names[obj_id] == "Unknown" and input_names[col] != "Unknown":
                    self.person_names[obj_id] = input_names[col]

                used_rows.add(row)
                used_cols.add(col)

            for i in range(len(input_centroids)):
                if i not in used_cols:
                    self.register(input_centroids[i], input_names[i])

            for i in range(len(object_ids)):
                if i not in used_rows:
                    obj_id = object_ids[i]
                    self.disappeared[obj_id] += 1
                    if self.disappeared[obj_id] > self.max_disappeared:
                        self.deregister(obj_id)

        return self.objects

class AIWorker:
    """Background AI processing thread - MEMORY OPTIMIZED with FASTER updates"""

    def __init__(self, detector, face_system):
        self.detector = detector
        self.face_system = face_system
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
                
                # YOLO Detect (higher confidence for better person detection)
                results = self.detector(process_frame, conf=0.4, verbose=False)[0]  # Lowered from 0.5 to catch more

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

                # Face Recognition - limit to MAX_FACES_PER_FRAME
                person_bboxes = [p["bbox"] for p in detections["persons"][:MAX_FACES_PER_FRAME]]
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
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.log_file = "inventory_log.csv"
        
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
        
        self.ai_worker = AIWorker(self.detector, self.face_system)
        
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
        
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Request higher resolution from camera if available
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        self.ai_worker.start()
        
        last_submit = 0
        
        cv2.namedWindow("CCTV Memory Optimized", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("CCTV Memory Optimized", 1280, 720)  # Larger display window

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

            display_frame = self._draw_overlay(frame)
            
            cv2.imshow("CCTV Memory Optimized", display_frame)
            
            key = cv2.waitKey(1) & 0xFF  # Fast display refresh
            if key == ord('q'): 
                break
            if key == ord('t'): 
                self.face_system.toggle_training()
            if key == 13: 
                self._handle_registration()

        cap.release()
        cv2.destroyAllWindows()

    def _process_logic(self, res):
        """Update tracker and check for exits"""
        detections = res["detections"]["persons"]
        faces = res["faces"]
        self.tracker.update(detections, faces)

    def _draw_overlay(self, frame):
        """Draws bounding boxes and info"""
        if self.latest_result:
            h, w = frame.shape[:2]
            scale = self.latest_result["scale_factor"]
            
            if scale != 1.0:
                display_h, display_w = self.latest_result["frame_shape"]
                display_frame = cv2.resize(frame, (display_w, display_h))
            else:
                display_frame = frame

            for p in self.latest_result["detections"]["persons"]:
                x1, y1, x2, y2 = p["bbox"]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = "Person"
                color = (0, 255, 0)
                
                for f in self.latest_result["faces"]:
                    # Extract coordinates
                    px1, py1, px2, py2 = p["bbox"]
                    fx1, fy1, fx2, fy2 = f["bbox"]
                    
                    # Check if the face is inside the person box
                    if fx1 >= px1 and fy1 >= py1 and fx2 <= px2 and fy2 <= py2:
                        if f["status"] == "known":
                            label = f"{f['name']} ({f['confidence']:.2f})"
                            color = (0, 255, 0)
                        elif f["status"] == "ready_to_register":
                            label = f"{f['name']} [READY]"
                            color = (0, 255, 255)
                            if f["name"] not in [x[0] for x in self.pending_registrations]:
                                montage = self.face_system.create_montage(f["name"])
                                if montage is not None:
                                    self.pending_registrations.append((f["name"], montage))
                        elif f["status"] == "collecting":
                            label = f"Collecting {f['samples_count']}/5"
                            color = (0, 165, 255)

                cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.putText(display_frame, f"AI: {self.ai_worker.avg_time_ms:.1f}ms", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if self.face_system.training_active:
                cv2.putText(display_frame, "TRAINING MODE", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if self.pending_registrations:
                uid, montage = self.pending_registrations[0]
                mh, mw = montage.shape[:2]
                display_frame[100:100+mh, 10:10+mw] = montage
                cv2.putText(display_frame, "PRESS ENTER TO REGISTER", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            return display_frame
        
        return cv2.resize(frame, (PROCESS_RESOLUTION, int(frame.shape[0]*(PROCESS_RESOLUTION/frame.shape[1])))) if frame.shape[1] > PROCESS_RESOLUTION else frame

    def _handle_registration(self):
        if self.pending_registrations:
            uid, _ = self.pending_registrations[0]
            # This creates a real popup window on top of everything
            new_name = pyautogui.prompt(text=f'Enter name for {uid}', title='Face Registration' , default='')
            
            if new_name:
                self.face_system.register_unknown(uid, new_name)
                self.pending_registrations.pop(0)

if __name__ == "__main__":
    # CONFIGURATION
    RTSP_URL = "rtsp://fasspay:fasspay2025@192.168.122.127:554/stream1"
    
    print("\n" + "="*50)
    print("HIGH RESOLUTION RECOGNITION SETTINGS")
    print("="*50)
    print(f"YOLO GPU: {YOLO_USE_GPU}")
    print(f"InsightFace GPU: {INSIGHTFACE_USE_GPU}")
    print(f"Processing Resolution: {PROCESS_RESOLUTION}p")
    print(f"Face Detection Size: {FACE_DETECTION_SIZE}")
    print(f"Max Faces Per Frame: {MAX_FACES_PER_FRAME}")
    print("="*50 + "\n")
    
    app = OptimizedInventoryDoorCCTV(RTSP_URL)
    app.start()
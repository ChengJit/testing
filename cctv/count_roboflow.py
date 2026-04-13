#!/usr/bin/env python3
"""
Roboflow Box Counter - Use pre-trained model from Roboflow!
"""

import cv2
import os
import time

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


def download_model():
    """Download pre-trained box model from Roboflow."""
    print("Downloading pre-trained box model from Roboflow...")

    try:
        from roboflow import Roboflow

        # Using a popular cardboard box detection model
        # This one has 4304 images trained!
        rf = Roboflow(api_key="")  # Public models don't need key

        project = rf.universe("carboard-box").project("carboard-box")
        version = project.version(4)
        model = version.model

        # Download YOLO weights
        dataset = version.download("yolov8")
        print(f"Downloaded to: {dataset.location}")
        return dataset.location

    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying alternative download method...")
        return None


def download_model_direct():
    """Direct download of YOLO weights."""
    import urllib.request
    import zipfile

    weights_dir = os.path.join(os.path.dirname(__file__), "roboflow_model")
    weights_file = os.path.join(weights_dir, "best.pt")

    if os.path.exists(weights_file):
        print("Model already downloaded!")
        return weights_file

    os.makedirs(weights_dir, exist_ok=True)

    # Try multiple box detection models
    models = [
        ("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt", "yolov8n.pt"),
    ]

    print("Downloading YOLOv8 base model...")
    url, fname = models[0]
    path = os.path.join(weights_dir, fname)
    urllib.request.urlretrieve(url, path)
    print(f"Downloaded: {path}")

    return path


class RoboflowCounter:
    def __init__(self):
        self.model = None
        self.roi = None
        self.baseline = 0
        self.last_count = 0
        self.confidence = 0.25

        # Tracking
        self.tracked_boxes = []
        self.min_hits = 3
        self.max_age = 8
        self.next_id = 0

    def load(self):
        """Load model."""
        try:
            from ultralytics import YOLO

            # First try our trained model
            local_model = os.path.join(os.path.dirname(__file__), "box_model.pt")
            if os.path.exists(local_model):
                print(f"Using your trained model: {local_model}")
                self.model = YOLO(local_model)
                self.custom = True
                return True

            # Try roboflow inference
            try:
                from inference import get_model
                print("Loading Roboflow model...")
                self.model = get_model(model_id="carboard-box/4")
                self.roboflow = True
                print("Roboflow model ready!")
                return True
            except:
                pass

            # Fallback to YOLOv8 with COCO (has 'box' class... kind of)
            print("Using YOLOv8 base model...")
            self.model = YOLO("yolov8n.pt")
            self.custom = False
            print("Model ready! (Using general detector)")
            return True

        except Exception as e:
            print(f"Error: {e}")
            return False

    def detect(self, frame):
        """Detect boxes."""
        if self.model is None:
            return []

        h, w = frame.shape[:2]

        # Apply ROI
        if self.roi:
            x1 = int(self.roi[0] * w)
            y1 = int(self.roi[1] * h)
            x2 = int(self.roi[2] * w)
            y2 = int(self.roi[3] * h)
            work = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            work = frame
            offset = (0, 0)

        detections = []

        if hasattr(self, 'roboflow') and self.roboflow:
            # Roboflow inference
            results = self.model.infer(work, confidence=self.confidence)
            for pred in results[0].predictions:
                cx, cy = pred.x, pred.y
                bw, bh = pred.width, pred.height
                x1 = int(cx - bw/2) + offset[0]
                y1 = int(cy - bh/2) + offset[1]
                x2 = int(cx + bw/2) + offset[0]
                y2 = int(cy + bh/2) + offset[1]
                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'score': pred.confidence,
                    'label': pred.class_name
                })
        else:
            # YOLO inference
            results = self.model(work, conf=self.confidence, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    detections.append({
                        'x1': x1 + offset[0],
                        'y1': y1 + offset[1],
                        'x2': x2 + offset[0],
                        'y2': y2 + offset[1],
                        'score': conf,
                        'label': self.model.names.get(cls, 'box')
                    })

        # Update tracking
        detections = self._update_tracking(detections)

        confirmed = len([d for d in detections if d.get('confirmed')])
        self.last_count = confirmed

        return detections

    def _iou(self, b1, b2):
        x1 = max(b1['x1'], b2['x1'])
        y1 = max(b1['y1'], b2['y1'])
        x2 = min(b1['x2'], b2['x2'])
        y2 = min(b1['y2'], b2['y2'])
        if x2 <= x1 or y2 <= y1:
            return 0
        inter = (x2-x1) * (y2-y1)
        a1 = (b1['x2']-b1['x1']) * (b1['y2']-b1['y1'])
        a2 = (b2['x2']-b2['x1']) * (b2['y2']-b2['y1'])
        return inter / (a1 + a2 - inter)

    def _update_tracking(self, detections):
        for tb in self.tracked_boxes:
            tb['age'] += 1

        matched_t = set()
        matched_d = set()

        for i, det in enumerate(detections):
            best_iou, best_j = 0, None
            for j, tb in enumerate(self.tracked_boxes):
                if j in matched_t:
                    continue
                iou = self._iou(det, tb)
                if iou > best_iou and iou > 0.25:
                    best_iou, best_j = iou, j

            if best_j is not None:
                tb = self.tracked_boxes[best_j]
                s = 0.7
                tb['x1'] = int(tb['x1']*s + det['x1']*(1-s))
                tb['y1'] = int(tb['y1']*s + det['y1']*(1-s))
                tb['x2'] = int(tb['x2']*s + det['x2']*(1-s))
                tb['y2'] = int(tb['y2']*s + det['y2']*(1-s))
                tb['hits'] += 1
                tb['age'] = 0
                matched_t.add(best_j)
                matched_d.add(i)

        for i, det in enumerate(detections):
            if i not in matched_d:
                self.tracked_boxes.append({
                    'id': self.next_id,
                    'x1': det['x1'], 'y1': det['y1'],
                    'x2': det['x2'], 'y2': det['y2'],
                    'hits': 1, 'age': 0,
                    'score': det['score'],
                    'label': det.get('label', 'box')
                })
                self.next_id += 1

        self.tracked_boxes = [tb for tb in self.tracked_boxes if tb['age'] <= self.max_age]

        result = []
        for tb in self.tracked_boxes:
            result.append({
                'x1': tb['x1'], 'y1': tb['y1'],
                'x2': tb['x2'], 'y2': tb['y2'],
                'score': tb['score'],
                'label': tb.get('label', 'box'),
                'hits': tb['hits'],
                'confirmed': tb['hits'] >= self.min_hits
            })
        return result

    def set_roi(self, frame):
        h, w = frame.shape[:2]
        scale = min(1280/w, 720/h, 1.0)
        if scale < 1.0:
            display = cv2.resize(frame, (int(w*scale), int(h*scale)))
        else:
            display = frame.copy()

        pts = []
        drawing = False
        temp = display.copy()

        def mouse(event, x, y, flags, param):
            nonlocal pts, drawing, temp
            if event == cv2.EVENT_LBUTTONDOWN:
                pts = [(x,y)]
                drawing = True
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp = display.copy()
                cv2.rectangle(temp, pts[0], (x,y), (0,255,255), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                pts.append((x,y))
                drawing = False

        win = "Draw ROI - ENTER confirm, ESC skip"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, mouse)

        while True:
            cv2.imshow(win, temp)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and len(pts) == 2:
                break
            elif key == 27:
                cv2.destroyWindow(win)
                return

        cv2.destroyWindow(win)
        dh, dw = display.shape[:2]
        self.roi = (
            min(pts[0][0], pts[1][0])/dw,
            min(pts[0][1], pts[1][1])/dh,
            max(pts[0][0], pts[1][0])/dw,
            max(pts[0][1], pts[1][1])/dh,
        )
        print("ROI set!")

    def draw(self, frame, detections):
        display = frame.copy()
        h, w = display.shape[:2]

        if self.roi:
            rx1, ry1 = int(self.roi[0]*w), int(self.roi[1]*h)
            rx2, ry2 = int(self.roi[2]*w), int(self.roi[3]*h)
            cv2.rectangle(display, (rx1,ry1), (rx2,ry2), (255,255,0), 2)

        confirmed = 0
        for d in detections:
            if d.get('confirmed'):
                confirmed += 1
                cv2.rectangle(display, (d['x1'],d['y1']), (d['x2'],d['y2']), (0,255,0), 2)
                cv2.putText(display, str(confirmed), (d['x1']+5, d['y1']+25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            else:
                cv2.rectangle(display, (d['x1'],d['y1']), (d['x2'],d['y2']), (0,200,255), 1)

        cv2.rectangle(display, (10,10), (200,80), (0,0,0), -1)
        cv2.putText(display, f"Boxes: {self.last_count}", (20,50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

        cv2.rectangle(display, (0,h-25), (w,h), (40,40,40), -1)
        cv2.putText(display, "C:Lock R:ROI +/-:Conf [/]:Hits Q:Quit",
                   (10, h-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

        return display


def run(camera_ip):
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2?rtsp_transport=tcp"

    print("=" * 50)
    print("  ROBOFLOW BOX COUNTER")
    print("=" * 50)

    # Install roboflow inference if needed
    print("\nChecking dependencies...")
    try:
        from inference import get_model
        print("Roboflow inference ready!")
    except:
        print("Installing roboflow inference...")
        os.system("pip install inference")

    counter = RoboflowCounter()
    if not counter.load():
        return

    print(f"\nConnecting to {camera_ip}...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("No connect!")
        return

    print("Connected!")

    cap.grab()
    ret, frame = cap.retrieve()
    if ret:
        counter.set_roi(frame)

    print("\nControls: C=Lock R=ROI +/-=Confidence [/]=Hits Q=Quit")

    win = "Roboflow Counter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    error_count = 0

    try:
        while True:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                error_count += 1
                if error_count > 30:
                    print("Reconnecting...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                    error_count = 0
                continue
            error_count = 0

            detections = counter.detect(frame)
            display = counter.draw(frame, detections)

            dh, dw = display.shape[:2]
            scale = min(1280/dw, 720/dh, 1.0)
            if scale < 1.0:
                display = cv2.resize(display, (int(dw*scale), int(dh*scale)))

            cv2.imshow(win, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                counter.set_roi(frame)
            elif key == ord('+') or key == ord('='):
                counter.confidence = min(0.9, counter.confidence + 0.05)
                print(f"Confidence: {counter.confidence:.2f}")
            elif key == ord('-'):
                counter.confidence = max(0.1, counter.confidence - 0.05)
                print(f"Confidence: {counter.confidence:.2f}")
            elif key == ord('['):
                counter.min_hits = max(1, counter.min_hits - 1)
                print(f"Min hits: {counter.min_hits}")
            elif key == ord(']'):
                counter.min_hits = min(10, counter.min_hits + 1)
                print(f"Min hits: {counter.min_hits}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python count_roboflow.py <camera_ip>")
        print("Example: python count_roboflow.py .129")
    else:
        run(sys.argv[1])

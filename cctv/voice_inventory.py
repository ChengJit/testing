#!/usr/bin/env python3
"""
VOICE INVENTORY SYSTEM - "Hey Tapo" Edition!

Say: "Hey Tapo, stocking D1V1450000001"
System: Tracks box, saves location!

Features:
- Voice commands from Tapo camera mic
- Box tracking with GroundingDINO
- Zone grid overlay
- SQLite database for inventory
- Speaker feedback (if supported)
"""

import cv2
import numpy as np
import os
import sys
import time
import sqlite3
import threading
import queue
import subprocess
import tempfile
from datetime import datetime

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Add GroundingDINO path
GDINO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GroundingDINO")
sys.path.insert(0, GDINO_PATH)


class InventoryDatabase:
    """SQLite database for inventory tracking."""

    def __init__(self, db_path="inventory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT NOT NULL,
                zone TEXT NOT NULL,
                box_x1 INTEGER,
                box_y1 INTEGER,
                box_x2 INTEGER,
                box_y2 INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'stocked'
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT,
                sku TEXT,
                zone TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def stock_item(self, sku, zone, box_coords=None):
        """Add item to inventory."""
        cursor = self.conn.cursor()

        # Check if SKU already exists
        cursor.execute("SELECT zone FROM inventory WHERE sku = ? AND status = 'stocked'", (sku,))
        existing = cursor.fetchone()

        if existing:
            # Update location
            if box_coords:
                cursor.execute("""
                    UPDATE inventory
                    SET zone = ?, box_x1 = ?, box_y1 = ?, box_x2 = ?, box_y2 = ?, timestamp = ?
                    WHERE sku = ? AND status = 'stocked'
                """, (zone, box_coords[0], box_coords[1], box_coords[2], box_coords[3],
                      datetime.now(), sku))
            else:
                cursor.execute("""
                    UPDATE inventory SET zone = ?, timestamp = ? WHERE sku = ? AND status = 'stocked'
                """, (zone, datetime.now(), sku))
            action = "moved"
        else:
            # New item
            if box_coords:
                cursor.execute("""
                    INSERT INTO inventory (sku, zone, box_x1, box_y1, box_x2, box_y2)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (sku, zone, box_coords[0], box_coords[1], box_coords[2], box_coords[3]))
            else:
                cursor.execute("INSERT INTO inventory (sku, zone) VALUES (?, ?)", (sku, zone))
            action = "stocked"

        # Log activity
        cursor.execute("INSERT INTO activity_log (action, sku, zone) VALUES (?, ?, ?)",
                      (action, sku, zone))
        self.conn.commit()
        return action

    def pick_item(self, sku):
        """Remove item from inventory."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT zone FROM inventory WHERE sku = ? AND status = 'stocked'", (sku,))
        result = cursor.fetchone()

        if result:
            cursor.execute("UPDATE inventory SET status = 'picked' WHERE sku = ? AND status = 'stocked'", (sku,))
            cursor.execute("INSERT INTO activity_log (action, sku, zone) VALUES (?, ?, ?)",
                          ("picked", sku, result[0]))
            self.conn.commit()
            return result[0]
        return None

    def find_item(self, sku):
        """Find item location."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT zone, box_x1, box_y1, box_x2, box_y2
            FROM inventory WHERE sku = ? AND status = 'stocked'
        """, (sku,))
        return cursor.fetchone()

    def get_zone_contents(self, zone):
        """Get all items in a zone."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT sku FROM inventory WHERE zone = ? AND status = 'stocked'", (zone,))
        return [row[0] for row in cursor.fetchall()]

    def get_all_stocked(self):
        """Get all stocked items."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT sku, zone FROM inventory WHERE status = 'stocked'")
        return cursor.fetchall()


class VoiceSpeaker:
    """Speak responses - laptop speaker for now, Tapo speaker later."""

    def __init__(self):
        self.engine = None
        self._init_engine()

    def _init_engine(self):
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 1.0)
            print("Voice speaker ready!")
        except Exception as e:
            print(f"Voice speaker error: {e}")
            self.engine = None

    def say(self, text):
        """Speak text."""
        print(f"[SPEAK] {text}")
        if self.engine:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except:
                pass

    def acknowledge(self):
        """Say wake word acknowledgement."""
        self.say("Hi! How can I help?")

    def confirm_stock(self, sku, zone):
        """Confirm stocking."""
        self.say(f"Stocked {sku} in zone {zone}")

    def confirm_pick(self, sku, zone):
        """Confirm picking."""
        self.say(f"Picked {sku} from zone {zone}")

    def item_found(self, sku, zone):
        """Say item location."""
        self.say(f"{sku} is in zone {zone}")

    def not_found(self, sku):
        """Item not found."""
        self.say(f"{sku} not found in inventory")


class VoiceListener:
    """Listen for voice commands from Tapo camera mic."""

    def __init__(self, camera_ip, wake_word="hey tapo", speaker=None):
        self.camera_ip = camera_ip
        self.wake_word = wake_word.lower()
        self.command_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.speaker = speaker

    def start(self):
        """Start listening thread."""
        self.running = True
        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        print(f"Voice listener started! Say '{self.wake_word}' + command")

    def stop(self):
        """Stop listening."""
        self.running = False

    def _listen_loop(self):
        """Continuously listen for commands."""
        try:
            import speech_recognition as sr
        except ImportError:
            print("Installing speech_recognition...")
            os.system("pip install SpeechRecognition")
            import speech_recognition as sr

        recognizer = sr.Recognizer()
        url = f"rtsp://fasspay:fasspay2025@{self.camera_ip}:554/stream1"

        while self.running:
            try:
                # Record audio chunk from camera
                audio_file = self._capture_audio(url, duration=4)
                if not audio_file:
                    continue

                # Recognize speech
                with sr.AudioFile(audio_file) as source:
                    audio = recognizer.record(source)

                try:
                    text = recognizer.recognize_google(audio).lower()
                    print(f"Heard: {text}")

                    # Check for wake word
                    if self.wake_word in text:
                        # Acknowledge wake word!
                        if self.speaker:
                            self.speaker.acknowledge()

                        # Extract command after wake word
                        parts = text.split(self.wake_word)
                        if len(parts) > 1:
                            command = parts[1].strip()
                            self.command_queue.put(command)
                            print(f"Command: {command}")

                except sr.UnknownValueError:
                    pass  # No speech detected
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")

                # Cleanup
                try:
                    os.remove(audio_file)
                except:
                    pass

            except Exception as e:
                print(f"Listen error: {e}")
                time.sleep(1)

    def _capture_audio(self, url, duration=4):
        """Capture audio from RTSP stream."""
        temp_file = tempfile.mktemp(suffix=".wav")

        cmd = [
            "ffmpeg", "-y",
            "-rtsp_transport", "tcp",
            "-i", url,
            "-t", str(duration),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            temp_file
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=duration + 5)
            if os.path.exists(temp_file) and os.path.getsize(temp_file) > 1000:
                return temp_file
        except:
            pass

        return None

    def get_command(self):
        """Get next command from queue (non-blocking)."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None


class ZoneGrid:
    """Overlay zone grid on camera view."""

    def __init__(self, rows=3, cols=4):
        self.rows = rows
        self.cols = cols
        self.zone_names = self._generate_zone_names()

    def _generate_zone_names(self):
        """Generate zone names like A1, A2, B1, B2, etc."""
        names = {}
        for r in range(self.rows):
            for c in range(self.cols):
                row_letter = chr(65 + r)  # A, B, C...
                col_num = c + 1
                names[(r, c)] = f"{row_letter}{col_num}"
        return names

    def get_zone_at_point(self, x, y, frame_w, frame_h):
        """Get zone name for a point."""
        col = int(x / frame_w * self.cols)
        row = int(y / frame_h * self.rows)
        col = min(col, self.cols - 1)
        row = min(row, self.rows - 1)
        return self.zone_names.get((row, col), "?")

    def get_zone_for_box(self, box, frame_w, frame_h):
        """Get zone name for a box (using center)."""
        cx = (box['x1'] + box['x2']) // 2
        cy = (box['y1'] + box['y2']) // 2
        return self.get_zone_at_point(cx, cy, frame_w, frame_h)

    def draw_grid(self, frame, highlight_zone=None):
        """Draw zone grid overlay."""
        h, w = frame.shape[:2]
        cell_w = w // self.cols
        cell_h = h // self.rows

        # Draw grid lines
        for i in range(1, self.cols):
            x = i * cell_w
            cv2.line(frame, (x, 0), (x, h), (100, 100, 100), 1)

        for i in range(1, self.rows):
            y = i * cell_h
            cv2.line(frame, (0, y), (w, y), (100, 100, 100), 1)

        # Draw zone labels
        for (r, c), name in self.zone_names.items():
            x = c * cell_w + 10
            y = r * cell_h + 30

            # Highlight active zone
            if highlight_zone and name == highlight_zone:
                cv2.rectangle(frame,
                             (c * cell_w, r * cell_h),
                             ((c + 1) * cell_w, (r + 1) * cell_h),
                             (0, 255, 255), 3)

            cv2.putText(frame, name, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

        return frame


class VoiceInventorySystem:
    """Main voice-controlled inventory system."""

    def __init__(self, camera_ip):
        self.camera_ip = camera_ip
        self.db = InventoryDatabase()
        self.speaker = VoiceSpeaker()
        self.voice = VoiceListener(camera_ip, speaker=self.speaker)
        self.grid = ZoneGrid(rows=3, cols=4)

        # Detection
        self.detector = None
        self.tracked_boxes = []

        # State
        self.pending_sku = None
        self.highlight_zone = None
        self.last_message = ""
        self.message_time = 0

    def load_detector(self):
        """Load GroundingDINO detector."""
        try:
            from groundingdino.util.inference import load_model
            import torch

            print("Loading GroundingDINO...")
            config = os.path.join(GDINO_PATH, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
            weights = os.path.join(GDINO_PATH, "weights", "groundingdino_swint_ogc.pth")

            if not os.path.exists(weights):
                print("Downloading weights...")
                os.makedirs(os.path.dirname(weights), exist_ok=True)
                import urllib.request
                url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
                urllib.request.urlretrieve(url, weights)

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.detector = load_model(config, weights, device=self.device)
            print("Detector ready!")
            return True

        except Exception as e:
            print(f"Detector error: {e}")
            return False

    def detect_boxes(self, frame):
        """Detect boxes in frame."""
        if self.detector is None:
            return []

        import torch
        from PIL import Image
        import groundingdino.datasets.transforms as T
        from groundingdino.util.inference import predict

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor, _ = transform(image_pil, None)

        boxes, logits, phrases = predict(
            model=self.detector,
            image=image_tensor,
            caption="cardboard box . brown box . package",
            box_threshold=0.25,
            text_threshold=0.25,
            device=self.device
        )

        h, w = frame.shape[:2]
        detections = []
        for box, score in zip(boxes, logits):
            cx, cy, bw, bh = box.tolist()
            x1 = int((cx - bw/2) * w)
            y1 = int((cy - bh/2) * h)
            x2 = int((cx + bw/2) * w)
            y2 = int((cy + bh/2) * h)
            detections.append({
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'score': float(score)
            })

        return detections

    def parse_command(self, command):
        """Parse voice command."""
        command = command.lower().strip()

        # "stocking D1V1450000001"
        if "stocking" in command or "stock" in command:
            words = command.replace("stocking", "").replace("stock", "").strip().split()
            if words:
                sku = words[0].upper().replace(" ", "")
                return ("stock", sku)

        # "picking D1V1450000001"
        if "picking" in command or "pick" in command:
            words = command.replace("picking", "").replace("pick", "").strip().split()
            if words:
                sku = words[0].upper().replace(" ", "")
                return ("pick", sku)

        # "find D1V1450000001"
        if "find" in command or "where" in command or "locate" in command:
            words = command.replace("find", "").replace("where", "").replace("locate", "").strip().split()
            if words:
                sku = words[0].upper().replace(" ", "")
                return ("find", sku)

        return None

    def set_message(self, msg):
        """Set status message."""
        self.last_message = msg
        self.message_time = time.time()
        print(f">> {msg}")

    def process_command(self, action, sku, boxes, frame_shape):
        """Process a voice command."""
        h, w = frame_shape[:2]

        if action == "stock":
            self.pending_sku = sku
            self.set_message(f"Place {sku} in a zone...")

            # Find nearest box to assign
            if boxes:
                # Use first detected box for now
                box = boxes[0]
                zone = self.grid.get_zone_for_box(box, w, h)
                result = self.db.stock_item(sku, zone, (box['x1'], box['y1'], box['x2'], box['y2']))
                self.highlight_zone = zone
                self.set_message(f"{sku} {result} in zone {zone}!")
                self.speaker.confirm_stock(sku, zone)
                self.pending_sku = None
            else:
                self.speaker.say(f"No box detected. Please show the box.")

        elif action == "pick":
            zone = self.db.pick_item(sku)
            if zone:
                self.set_message(f"Picked {sku} from zone {zone}")
                self.highlight_zone = zone
                self.speaker.confirm_pick(sku, zone)
            else:
                self.set_message(f"{sku} not found in inventory!")
                self.speaker.not_found(sku)

        elif action == "find":
            result = self.db.find_item(sku)
            if result:
                zone = result[0]
                self.highlight_zone = zone
                self.set_message(f"{sku} is in zone {zone}")
                self.speaker.item_found(sku, zone)
            else:
                self.set_message(f"{sku} not found!")
                self.speaker.not_found(sku)

    def draw_overlay(self, frame, boxes):
        """Draw everything on frame."""
        h, w = frame.shape[:2]

        # Draw zone grid
        self.grid.draw_grid(frame, self.highlight_zone)

        # Draw detected boxes
        for i, box in enumerate(boxes):
            zone = self.grid.get_zone_for_box(box, w, h)
            cv2.rectangle(frame, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 255, 0), 2)
            cv2.putText(frame, f"{i+1}:{zone}", (box['x1'], box['y1'] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Status panel
        cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.putText(frame, "VOICE INVENTORY", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Say: 'Hey Tapo, stocking [SKU]'", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show message
        if self.last_message and time.time() - self.message_time < 5:
            cv2.putText(frame, self.last_message, (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show inventory count
        items = self.db.get_all_stocked()
        cv2.putText(frame, f"Inventory: {len(items)} items", (w - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Help bar
        cv2.rectangle(frame, (0, h - 30), (w, h), (40, 40, 40), -1)
        cv2.putText(frame, "Voice: 'Hey Tapo, stocking/picking/find [SKU]' | M=Manual | L=List | Q=Quit",
                   (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return frame

    def run(self):
        """Run the voice inventory system."""
        url = f"rtsp://fasspay:fasspay2025@{self.camera_ip}:554/stream2?rtsp_transport=tcp"

        print("=" * 60)
        print("  VOICE INVENTORY SYSTEM - 'Hey Tapo' Edition!")
        print("=" * 60)

        if not self.load_detector():
            print("Warning: Running without box detection")

        print(f"\nConnecting to camera {self.camera_ip}...")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            print("Failed to connect!")
            return

        print("Connected!")

        # Start voice listener
        self.voice.start()

        print("\nVoice Commands:")
        print("  'Hey Tapo, stocking D1V1450000001' - Add item to inventory")
        print("  'Hey Tapo, picking D1V1450000001'  - Remove item")
        print("  'Hey Tapo, find D1V1450000001'     - Locate item")
        print("\nKeyboard:")
        print("  M - Manual SKU entry")
        print("  L - List inventory")
        print("  Q - Quit")

        win = "Voice Inventory"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        boxes = []
        last_detect = 0
        detect_interval = 2.0
        error_count = 0

        try:
            while True:
                cap.grab()
                ret, frame = cap.retrieve()
                if not ret:
                    error_count += 1
                    if error_count > 30:
                        cap.release()
                        time.sleep(1)
                        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                        error_count = 0
                    continue
                error_count = 0

                # Detect boxes periodically
                if self.detector and time.time() - last_detect > detect_interval:
                    boxes = self.detect_boxes(frame)
                    last_detect = time.time()

                # Check for voice commands
                command = self.voice.get_command()
                if command:
                    parsed = self.parse_command(command)
                    if parsed:
                        action, sku = parsed
                        self.process_command(action, sku, boxes, frame.shape)

                # Clear highlight after 3 seconds
                if self.highlight_zone and time.time() - self.message_time > 3:
                    self.highlight_zone = None

                # Draw overlay
                display = self.draw_overlay(frame.copy(), boxes)

                # Resize for display
                dh, dw = display.shape[:2]
                scale = min(1280 / dw, 720 / dh, 1.0)
                if scale < 1.0:
                    display = cv2.resize(display, (int(dw * scale), int(dh * scale)))

                cv2.imshow(win, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('m'):
                    # Manual SKU entry
                    sku = input("Enter SKU: ").strip().upper()
                    if sku and boxes:
                        box = boxes[0]
                        h, w = frame.shape[:2]
                        zone = self.grid.get_zone_for_box(box, w, h)
                        self.db.stock_item(sku, zone, (box['x1'], box['y1'], box['x2'], box['y2']))
                        self.set_message(f"Stocked {sku} in {zone}")
                elif key == ord('l'):
                    # List inventory
                    items = self.db.get_all_stocked()
                    print("\n=== INVENTORY ===")
                    for sku, zone in items:
                        print(f"  {sku} -> {zone}")
                    print(f"Total: {len(items)} items\n")

        finally:
            self.voice.stop()
            cap.release()
            cv2.destroyAllWindows()


def main():
    if len(sys.argv) < 2:
        print("Voice Inventory System")
        print("=" * 40)
        print("\nUsage: python voice_inventory.py <camera_ip>")
        print("Example: python voice_inventory.py .129")
        print("\nCommands:")
        print("  'Hey Tapo, stocking [SKU]' - Add item")
        print("  'Hey Tapo, picking [SKU]'  - Remove item")
        print("  'Hey Tapo, find [SKU]'     - Locate item")
    else:
        camera_ip = sys.argv[1]
        if camera_ip.startswith('.'):
            camera_ip = f"192.168.122{camera_ip}"

        system = VoiceInventorySystem(camera_ip)
        system.run()


if __name__ == "__main__":
    main()

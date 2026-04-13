#!/usr/bin/env python3
"""
Florence-2 Box Counter
=======================

Microsoft's Florence-2 vision model - excellent for detailed detection.
Can detect individual objects even when stacked.

Features:
- Zero-shot detection with text prompts
- Very accurate for small objects
- Can count and describe objects
"""

import cv2
import numpy as np
import time
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import os

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"


@dataclass
class FlorenceDetection:
    """Detection from Florence-2."""
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    score: float

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


class FlorenceCounter:
    """
    Count boxes using Florence-2 model.
    """

    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.roi = None
        self.last_count = 0
        self.baseline = 0

        # Detection settings
        self.prompt = "<OD>"  # Object Detection task
        self.min_area = 500
        self.max_area = 50000

    def load(self) -> bool:
        """Load Florence-2 model."""
        try:
            # Disable flash_attn requirement
            import sys

            # Create a fake flash_attn module to bypass import check
            import types
            fake_flash_attn = types.ModuleType('flash_attn')
            fake_flash_attn.flash_attn_func = None
            fake_flash_attn.flash_attn_varlen_func = None
            sys.modules['flash_attn'] = fake_flash_attn
            sys.modules['flash_attn.flash_attn_interface'] = fake_flash_attn

            import torch

            print("Loading Florence-2 (this may take a moment)...")

            # Check GPU
            if torch.cuda.is_available():
                self.device = "cuda"
                dtype = torch.float16
                print(f"  Using GPU: {torch.cuda.get_device_name()}")
            else:
                self.device = "cpu"
                dtype = torch.float32
                print("  Using CPU (slower)")

            # Monkey-patch to fix forced_bos_token_id error
            import transformers
            original_getattr = None

            def patched_config_getattr(self, name):
                if name == 'forced_bos_token_id':
                    return None
                if original_getattr:
                    return original_getattr(self, name)
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

            from transformers import AutoProcessor, AutoModelForCausalLM

            model_id = "microsoft/Florence-2-base"

            # Load processor first
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )

            # Try to patch the config class before model loading
            try:
                import importlib
                import sys
                # Clear any cached modules
                modules_to_clear = [k for k in sys.modules if 'florence' in k.lower()]
                for mod in modules_to_clear:
                    del sys.modules[mod]
            except:
                pass

            # Set environment to disable flash attention
            os.environ["FLASH_ATTENTION_SKIP_CUDA_BUILD"] = "TRUE"

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

            self.model = self.model.to(self.device)
            self.model.eval()

            print("  Florence-2 ready!")
            return True

        except ImportError as e:
            print(f"Florence-2 requires: pip install transformers torch timm einops")
            print(f"Error: {e}")
            return False
        except Exception as e:
            print(f"Error loading Florence-2: {e}")
            print("\nThis is a known transformers compatibility issue.")
            print("Try downgrading transformers:")
            print("  pip install transformers==4.40.0")
            return False

    def detect(self, frame: np.ndarray, task: str = "<OD>") -> List[FlorenceDetection]:
        """
        Detect objects using Florence-2.

        Tasks:
        - <OD> : Object Detection
        - <DENSE_REGION_CAPTION> : Detailed region captions
        - <CAPTION_TO_PHRASE_GROUNDING> : Find specific objects
        """
        if self.model is None:
            return []

        import torch
        from PIL import Image

        h, w = frame.shape[:2]

        # Apply ROI
        if self.roi:
            x1 = int(self.roi[0] * w)
            y1 = int(self.roi[1] * h)
            x2 = int(self.roi[2] * w)
            y2 = int(self.roi[3] * h)
            work_frame = frame[y1:y2, x1:x2]
            offset = (x1, y1)
            work_h, work_w = work_frame.shape[:2]
        else:
            work_frame = frame
            offset = (0, 0)
            work_h, work_w = h, w

        # Resize for faster processing
        work_h, work_w = work_frame.shape[:2]
        max_size = 512
        scale = min(max_size / work_w, max_size / work_h, 1.0)
        if scale < 1.0:
            new_w = int(work_w * scale)
            new_h = int(work_h * scale)
            work_frame = cv2.resize(work_frame, (new_w, new_h))

        # Convert to PIL
        image = Image.fromarray(cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB))

        # Process
        inputs = self.processor(text=task, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        # Decode
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Parse results
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )

        detections = []

        if task == "<OD>" and "<OD>" in parsed:
            result = parsed["<OD>"]
            bboxes = result.get("bboxes", [])
            labels = result.get("labels", [])

            for bbox, label in zip(bboxes, labels):
                bx1, by1, bx2, by2 = [int(c) for c in bbox]

                # Scale back and add offset
                x1 = int(bx1 / scale) + offset[0]
                y1 = int(by1 / scale) + offset[1]
                x2 = int(bx2 / scale) + offset[0]
                y2 = int(by2 / scale) + offset[1]

                area = (x2 - x1) * (y2 - y1)
                if self.min_area <= area <= self.max_area:
                    detections.append(FlorenceDetection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        label=label, score=0.9
                    ))

        elif task == "<DENSE_REGION_CAPTION>" and "<DENSE_REGION_CAPTION>" in parsed:
            result = parsed["<DENSE_REGION_CAPTION>"]
            bboxes = result.get("bboxes", [])
            labels = result.get("labels", [])

            for bbox, label in zip(bboxes, labels):
                bx1, by1, bx2, by2 = [int(c) for c in bbox]
                x1 = int(bx1 / scale) + offset[0]
                y1 = int(by1 / scale) + offset[1]
                x2 = int(bx2 / scale) + offset[0]
                y2 = int(by2 / scale) + offset[1]

                area = (x2 - x1) * (y2 - y1)
                if self.min_area <= area <= self.max_area:
                    # Filter for boxes
                    label_lower = label.lower()
                    if any(kw in label_lower for kw in ["box", "carton", "package", "cardboard"]):
                        detections.append(FlorenceDetection(
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            label=label, score=0.9
                        ))

        return detections

    def detect_boxes(self, frame: np.ndarray) -> List[FlorenceDetection]:
        """Detect specifically cardboard boxes."""
        if self.model is None:
            return []

        import torch
        from PIL import Image

        h, w = frame.shape[:2]

        # Apply ROI
        if self.roi:
            x1 = int(self.roi[0] * w)
            y1 = int(self.roi[1] * h)
            x2 = int(self.roi[2] * w)
            y2 = int(self.roi[3] * h)
            work_frame = frame[y1:y2, x1:x2]
            offset = (x1, y1)
        else:
            work_frame = frame
            offset = (0, 0)

        # Resize for faster processing
        work_h, work_w = work_frame.shape[:2]
        max_size = 512
        scale = min(max_size / work_w, max_size / work_h, 1.0)
        if scale < 1.0:
            new_w = int(work_w * scale)
            new_h = int(work_h * scale)
            work_frame = cv2.resize(work_frame, (new_w, new_h))

        image = Image.fromarray(cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB))

        # Use phrase grounding to find boxes
        task = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = "cardboard box"

        inputs = self.processor(
            text=task + prompt,
            images=image,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height)
        )

        detections = []

        if "<CAPTION_TO_PHRASE_GROUNDING>" in parsed:
            result = parsed["<CAPTION_TO_PHRASE_GROUNDING>"]
            bboxes = result.get("bboxes", [])

            for bbox in bboxes:
                bx1, by1, bx2, by2 = [int(c) for c in bbox]
                x1 = int(bx1 / scale) + offset[0]
                y1 = int(by1 / scale) + offset[1]
                x2 = int(bx2 / scale) + offset[0]
                y2 = int(by2 / scale) + offset[1]

                area = (x2 - x1) * (y2 - y1)
                if self.min_area <= area <= self.max_area:
                    detections.append(FlorenceDetection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        label="box", score=0.9
                    ))

        return detections

    def count(self, frame: np.ndarray, use_grounding: bool = True) -> Tuple[int, List[FlorenceDetection]]:
        """Count boxes."""
        if use_grounding:
            boxes = self.detect_boxes(frame)
        else:
            boxes = self.detect(frame, "<OD>")
            # Filter for box-like objects
            boxes = [b for b in boxes if any(
                kw in b.label.lower() for kw in ["box", "carton", "package", "card"]
            )]

        self.last_count = len(boxes)
        return self.last_count, boxes

    def set_roi_interactive(self, frame: np.ndarray):
        """Draw ROI."""
        roi_pts = []
        drawing = False
        temp = frame.copy()

        def mouse_cb(event, x, y, flags, param):
            nonlocal roi_pts, drawing, temp
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_pts = [(x, y)]
                drawing = True
            elif event == cv2.EVENT_MOUSEMOVE and drawing:
                temp = frame.copy()
                cv2.rectangle(temp, roi_pts[0], (x, y), (0, 255, 255), 2)
            elif event == cv2.EVENT_LBUTTONUP:
                roi_pts.append((x, y))
                drawing = False

        win = "Draw ROI - ENTER to confirm"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, mouse_cb)

        while True:
            cv2.imshow(win, temp)
            key = cv2.waitKey(30) & 0xFF
            if key == 13 and len(roi_pts) == 2:
                break
            elif key == 27:
                cv2.destroyWindow(win)
                return

        cv2.destroyWindow(win)

        h, w = frame.shape[:2]
        self.roi = (
            min(roi_pts[0][0], roi_pts[1][0]) / w,
            min(roi_pts[0][1], roi_pts[1][1]) / h,
            max(roi_pts[0][0], roi_pts[1][0]) / w,
            max(roi_pts[0][1], roi_pts[1][1]) / h,
        )
        print(f"ROI: {self.roi}")

    def draw_overlay(self, frame: np.ndarray, detections: List[FlorenceDetection]) -> np.ndarray:
        """Draw detections."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw ROI
        if self.roi:
            rx1 = int(self.roi[0] * w)
            ry1 = int(self.roi[1] * h)
            rx2 = int(self.roi[2] * w)
            ry2 = int(self.roi[3] * h)
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

        # Draw detections
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
        ]

        for i, det in enumerate(detections):
            color = colors[i % len(colors)]
            cv2.rectangle(display, (det.x1, det.y1), (det.x2, det.y2), color, 2)

            # Number
            cv2.putText(display, str(i + 1), (det.x1 + 5, det.y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Label (if not just "box")
            if det.label and det.label.lower() != "box":
                cv2.putText(display, det.label[:15], (det.x1, det.y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Count panel
        cv2.rectangle(display, (10, 10), (220, 80), (0, 0, 0), -1)
        cv2.putText(display, f"Boxes: {len(detections)}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if self.baseline > 0:
            change = len(detections) - self.baseline
            color = (0, 255, 0) if change >= 0 else (0, 0, 255)
            cv2.putText(display, f"Change: {change:+d}", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return display


def run_florence_counter(camera_ip: str):
    """Run Florence-2 counter."""
    if camera_ip.startswith('.'):
        camera_ip = f"192.168.122{camera_ip}"
    url = f"rtsp://fasspay:fasspay2025@{camera_ip}:554/stream2?rtsp_transport=tcp"

    print("=" * 50)
    print("  Florence-2 Box Counter")
    print("=" * 50)

    counter = FlorenceCounter()
    if not counter.load():
        print("\nInstall requirements:")
        print("  pip install transformers torch timm einops")
        return

    print(f"\nConnecting to {camera_ip}...")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Failed to connect!")
        return

    print("Connected!")

    # Get frame for ROI
    cap.grab()
    ret, frame = cap.retrieve()
    if ret:
        counter.set_roi_interactive(frame)

    print("\nControls:")
    print("  B - Set baseline")
    print("  R - Redraw ROI")
    print("  G - Toggle grounding mode (specific vs general)")
    print("  S - Screenshot")
    print("  Q - Quit")
    print("-" * 50)

    window = "Florence-2 Box Counter"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    import threading

    detections = []
    detecting = False
    detect_frame = None
    detect_result = None
    detect_interval = 3.0
    use_grounding = True
    last_detect = 0

    def detect_thread():
        nonlocal detect_result, detecting
        if detect_frame is not None:
            mode = "grounding" if use_grounding else "general"
            print(f"Detecting ({mode})...", end=" ", flush=True)
            start = time.time()
            count, dets = counter.count(detect_frame, use_grounding)
            elapsed = time.time() - start
            print(f"{count} boxes ({elapsed:.1f}s)")
            detect_result = dets
        detecting = False

    try:
        while True:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                continue

            # Start detection in background
            if not detecting and time.time() - last_detect >= detect_interval:
                detect_frame = frame.copy()
                detecting = True
                last_detect = time.time()
                threading.Thread(target=detect_thread, daemon=True).start()

            # Update detections when ready
            if detect_result is not None:
                detections = detect_result
                detect_result = None

            display = counter.draw_overlay(frame, detections)

            # Mode indicator
            mode = "GROUNDING" if use_grounding else "GENERAL"
            cv2.putText(display, f"Mode: {mode}", (display.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            h = display.shape[0]
            cv2.rectangle(display, (0, h - 30), (display.shape[1], h), (40, 40, 40), -1)
            cv2.putText(display, "B:Baseline R:ROI G:Mode S:Screenshot Q:Quit",
                       (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow(window, display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                counter.baseline = counter.last_count
                print(f"Baseline: {counter.baseline}")
            elif key == ord('r'):
                counter.set_roi_interactive(frame)
            elif key == ord('g'):
                use_grounding = not use_grounding
                print(f"Mode: {'Grounding' if use_grounding else 'General'}")
            elif key == ord('s'):
                fname = f"florence_{int(time.time())}.jpg"
                cv2.imwrite(fname, display)
                print(f"Saved: {fname}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python florence_counter.py <camera_ip>")
    else:
        run_florence_counter(sys.argv[1])

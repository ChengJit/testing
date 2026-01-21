#!/usr/bin/env python3
"""
YOLO Box Detection Training System
Complete system to collect data and train a custom model for cardboard box detection
"""

import cv2
import os
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
import yaml

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("❌ Install ultralytics: pip install ultralytics")
    exit(1)


class BoxTrainingDataCollector:
    """Collect training data for box detection"""
    
    def __init__(self, output_dir="box_training_data", dataset_name="cardboard_boxes"):
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        
        # Create directory structure directly in train/val format
        self.train_images_dir = self.output_dir / "train" / "images"
        self.train_labels_dir = self.output_dir / "train" / "labels"
        self.val_images_dir = self.output_dir / "val" / "images"
        self.val_labels_dir = self.output_dir / "val" / "labels"
        
        self.train_images_dir.mkdir(parents=True, exist_ok=True)
        self.train_labels_dir.mkdir(parents=True, exist_ok=True)
        self.val_images_dir.mkdir(parents=True, exist_ok=True)
        self.val_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Track saved images for auto-split
        self.saved_images = []
        
        # State
        self.current_frame = None
        self.boxes = []  # List of boxes being annotated
        self.current_box = None  # Box being drawn
        self.drawing = False
        self.image_count = 0
        
        # Mouse callback state
        self.start_point = None
        
        print(f"\n{'='*60}")
        print("BOX TRAINING DATA COLLECTOR")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        print(f"Dataset name: {self.dataset_name}")
        print(f"{'='*60}\n")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_box = [x, y, x, y]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_box = [self.start_point[0], self.start_point[1], x, y]
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point:
                # Ensure box has minimum size
                x1, y1 = self.start_point
                if abs(x - x1) > 10 and abs(y - y1) > 10:
                    # Normalize coordinates (ensure x1 < x2, y1 < y2)
                    box = [
                        min(x1, x),
                        min(y1, y),
                        max(x1, x),
                        max(y1, y)
                    ]
                    self.boxes.append(box)
                    print(f"✅ Box added: {box}")
                self.current_box = None
                self.start_point = None
    
    def draw_boxes(self, frame):
        """Draw all annotated boxes on frame"""
        display = frame.copy()
        
        # Draw saved boxes
        for idx, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, f"Box {idx+1}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw current box being drawn
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Add instructions
        instructions = [
            "INSTRUCTIONS:",
            "- Click & drag to draw box",
            "- U: Undo last box",
            "- S: Save annotations",
            "- C: Clear all boxes",
            "- SPACE: Next frame",
            "- Q: Quit",
            f"Boxes: {len(self.boxes)} | Saved: {self.image_count}"
        ]
        
        y = 30
        for text in instructions:
            cv2.putText(display, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y += 25
        
        return display
    
    def save_annotations(self, frame):
        """Save current frame and annotations in YOLO format"""
        if len(self.boxes) == 0:
            print("⚠️ No boxes to save!")
            return False
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_name = f"box_{timestamp}.jpg"
        label_name = f"box_{timestamp}.txt"
        
        # Auto-split: 80% train, 20% val
        import random
        is_train = random.random() < 0.8
        
        if is_train:
            image_path = self.train_images_dir / image_name
            label_path = self.train_labels_dir / label_name
            split = "TRAIN"
        else:
            image_path = self.val_images_dir / image_name
            label_path = self.val_labels_dir / label_name
            split = "VAL"
        
        # Save image
        cv2.imwrite(str(image_path), frame)
        
        # Save labels in YOLO format: class x_center y_center width height (normalized)
        h, w = frame.shape[:2]
        
        with open(label_path, 'w') as f:
            for box in self.boxes:
                x1, y1, x2, y2 = box
                
                # Calculate center and dimensions (normalized)
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Class 0 for "box"
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        self.image_count += 1
        self.saved_images.append((image_name, split))
        print(f"✅ Saved [{split}]: {image_name} with {len(self.boxes)} box(es)")
        
        # Clear boxes for next image
        self.boxes = []
        
        return True
    
    def collect_from_stream(self, rtsp_url, window_width=1280, window_height=720):
        """Collect training data from RTSP stream"""
        print(f"Connecting to: {rtsp_url}")
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Failed to open stream")
            return
        
        print("✅ Stream connected")
        print("\nStart annotating boxes!")
        print("Click and drag to draw boxes around cardboard boxes\n")
        
        cv2.namedWindow("Box Annotation Tool", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Box Annotation Tool", window_width, window_height)
        cv2.setMouseCallback("Box Annotation Tool", self.mouse_callback)
        
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("⚠️ Stream lost, retrying...")
                        time.sleep(2)
                        cap = cv2.VideoCapture(rtsp_url)
                        continue
                    
                    self.current_frame = frame
                
                display = self.draw_boxes(self.current_frame)
                cv2.imshow("Box Annotation Tool", display)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                    if paused:
                        print("\n⏸️  PAUSED - Annotate boxes")
                    else:
                        print("\n▶️  RESUMED")
                elif key == ord('s'):
                    if paused:
                        self.save_annotations(self.current_frame)
                    else:
                        print("⚠️ Pause first (SPACE) before saving")
                elif key == ord('u'):
                    if self.boxes:
                        removed = self.boxes.pop()
                        print(f"↩️  Undone: {removed}")
                    else:
                        print("⚠️ No boxes to undo")
                elif key == ord('c'):
                    if self.boxes:
                        print(f"🗑️  Cleared {len(self.boxes)} box(es)")
                        self.boxes = []
                    
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"\n✅ Collection complete!")
            print(f"   Images saved: {self.image_count}")
            print(f"   Location: {self.output_dir}")
    
    def create_dataset_yaml(self):
        """Create dataset.yaml for YOLO training"""
        
        # Count images in train/val
        train_images = list(self.train_images_dir.glob("*.jpg"))
        val_images = list(self.val_images_dir.glob("*.jpg"))
        
        train_count = len(train_images)
        val_count = len(val_images)
        total = train_count + val_count
        
        if total == 0:
            print("⚠️ No images found!")
            return None
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Train: {train_count} images")
        print(f"   Val: {val_count} images")
        print(f"   Total: {total} images")
        
        # Create YAML
        yaml_path = self.output_dir / "dataset.yaml"
        
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'names': {
                0: 'box'
            },
            'nc': 1  # number of classes
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"✅ Dataset YAML created: {yaml_path}")
        
        return yaml_path


class BoxModelTrainer:
    """Train YOLO model for box detection"""
    
    def __init__(self, dataset_yaml, model_name="yolo11n.pt"):
        self.dataset_yaml = dataset_yaml
        self.model_name = model_name
        
        print(f"\n{'='*60}")
        print("BOX MODEL TRAINER")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_yaml}")
        print(f"Base model: {model_name}")
        print(f"{'='*60}\n")
    
    def train(self, epochs=100, imgsz=640, batch=16, device='0'):
        """Train the model"""
        print("🚀 Starting training...\n")
        
        # Clear GPU memory before training
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            
            # Get GPU info
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU Memory: {total_mem:.2f}GB")
            
            # Auto-adjust batch size for Jetson
            if total_mem < 8:  # Jetson with < 8GB
                print("⚠️ Detected Jetson device with limited memory")
                if batch > 8:
                    batch = 4
                    print(f"   Auto-adjusting batch size to {batch}")
                if imgsz > 416:
                    imgsz = 416
                    print(f"   Auto-adjusting image size to {imgsz}")
        
        # Load base model
        model = YOLO(self.model_name)
        
        print(f"\n📊 Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch}")
        print(f"   Device: {device}")
        print()
        
        # Train with Jetson-optimized settings
        try:
            results = model.train(
                data=str(self.dataset_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                project='box_detection_training',
                name='cardboard_box_detector',
                exist_ok=True,
                patience=20,  # Early stopping
                save=True,
                plots=True,
                workers=2,  # Reduce workers for Jetson
                cache=False,  # Don't cache images to save memory
                amp=True,  # Use automatic mixed precision
                verbose=True
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "NVML_SUCCESS" in str(e):
                print("\n❌ GPU Out of Memory!")
                print("\n💡 Try these solutions:")
                print("   1. Reduce batch size: --batch 2")
                print("   2. Reduce image size: --imgsz 320")
                print("   3. Use CPU: --device cpu (slower but works)")
                print("   4. Close other programs to free GPU memory")
                print("\nExample:")
                print(f"   python {__file__} --mode train --batch 2 --imgsz 320\n")
                return None
            else:
                raise e
        
        print("\n✅ Training complete!")
        print(f"📁 Results saved in: box_detection_training/cardboard_box_detector")
        
        return results
    
    def evaluate(self, model_path):
        """Evaluate trained model"""
        print(f"\n📊 Evaluating model: {model_path}")
        
        model = YOLO(model_path)
        metrics = model.val()
        
        print("\n📈 Metrics:")
        print(f"   mAP50: {metrics.box.map50:.4f}")
        print(f"   mAP50-95: {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall: {metrics.box.mr:.4f}")
        
        return metrics
    
    def test_inference(self, model_path, rtsp_url):
        """Test the trained model on live stream"""
        print(f"\n🎥 Testing inference on stream...")
        
        model = YOLO(model_path)
        
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Failed to open stream")
            return
        
        print("✅ Stream connected")
        print("Press Q to quit\n")
        
        cv2.namedWindow("Box Detection Test", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Box Detection Test", 1280, 720)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference
                results = model(frame, conf=0.25, verbose=False)[0]
                
                # Draw results
                display = frame.copy()
                
                box_count = 0
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Box {conf:.2f}"
                    cv2.putText(display, label, (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    box_count += 1
                
                # Add info
                cv2.putText(display, f"Boxes detected: {box_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                cv2.imshow("Box Detection Test", display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='YOLO Box Detection Training System')
    parser.add_argument('--mode', choices=['collect', 'train', 'test'], required=True,
                       help='Mode: collect data, train model, or test inference')
    parser.add_argument('--rtsp', type=str,
                       default="rtsp://fasspay:fasspay2025@192.168.122.127:554/stream1",
                       help='RTSP stream URL')
    parser.add_argument('--data-dir', type=str, default='box_training_data',
                       help='Training data directory')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       help='Base model for training or trained model for testing')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--window-width', type=int, default=1280,
                       help='Window width (default: 1280)')
    parser.add_argument('--window-height', type=int, default=720,
                       help='Window height (default: 720)')
    parser.add_argument('--device', type=str, default='0',
                       help='Device (0 for GPU, cpu for CPU)')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        print("\n" + "="*60)
        print("MODE: DATA COLLECTION")
        print("="*60)
        print("\nInstructions:")
        print("1. Press SPACE to pause video")
        print("2. Click and drag to draw boxes around cardboard boxes")
        print("3. Press S to save the annotated frame")
        print("4. Press SPACE to resume and capture more frames")
        print("5. Collect at least 100-200 images for good results")
        print("="*60 + "\n")
        
        collector = BoxTrainingDataCollector(output_dir=args.data_dir)
        collector.collect_from_stream(args.rtsp, args.window_width, args.window_height)
        
        # Create dataset YAML
        print("\n📦 Creating dataset configuration...")
        yaml_path = collector.create_dataset_yaml()
        
        if yaml_path:
            print(f"\n✅ Ready for training!")
            print(f"   Run: python {__file__} --mode train --data-dir {args.data_dir}")
    
    elif args.mode == 'train':
        print("\n" + "="*60)
        print("MODE: TRAINING")
        print("="*60 + "\n")
        
        dataset_yaml = Path(args.data_dir) / "dataset.yaml"
        
        if not dataset_yaml.exists():
            print(f"❌ Dataset YAML not found: {dataset_yaml}")
            print("   Run data collection first!")
            return
        
        trainer = BoxModelTrainer(dataset_yaml, model_name=args.model)
        trainer.train(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device
        )
        
        print("\n✅ Training complete!")
        print("   Best model: box_detection_training/cardboard_box_detector/weights/best.pt")
        print(f"\n   Test it: python {__file__} --mode test --model box_detection_training/cardboard_box_detector/weights/best.pt")
    
    elif args.mode == 'test':
        print("\n" + "="*60)
        print("MODE: TESTING")
        print("="*60 + "\n")
        
        if not Path(args.model).exists():
            print(f"❌ Model not found: {args.model}")
            return
        
        trainer = BoxModelTrainer(None)
        trainer.test_inference(args.model, args.rtsp)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("YOLO BOX DETECTION TRAINING SYSTEM")
    print("="*60)
    print("\nWorkflow:")
    print("1. Collect data:  --mode collect")
    print("2. Train model:   --mode train")
    print("3. Test model:    --mode test --model path/to/best.pt")
    print("="*60 + "\n")
    
    main()
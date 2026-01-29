"""
Tkinter GUI for the Inventory Door Monitor.
Embeds the CCTV feed and provides controls for collection, training, and review.
"""

import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, List

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    raise ImportError("Tkinter is required for GUI mode")

try:
    from PIL import Image, ImageTk
except ImportError:
    raise ImportError("Pillow is required for GUI mode: pip install Pillow")

from .config import Config
from .app import InventoryMonitor, ProcessingResult
from .detectors.box import TrainingDataCollector

logger = logging.getLogger(__name__)


@dataclass
class ReviewItem:
    """A low-confidence detection queued for manual review."""
    image: np.ndarray           # BGR person crop
    identity: Optional[str]
    confidence: float
    box_count: int
    track_id: int
    timestamp: float


class InventoryGUI:
    """Main Tkinter GUI for the Inventory Monitor."""

    def __init__(self, config: Config):
        self.config = config
        self.monitor: Optional[InventoryMonitor] = None

        # State
        self.review_queue: List[ReviewItem] = []
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._feed_visible = True
        self._collection_active = False
        self._training_in_progress = False
        self._gallery_photos: List[ImageTk.PhotoImage] = []
        self._review_photos: List[ImageTk.PhotoImage] = []

        # Build UI
        self.root = tk.Tk()
        self.root.title("Inventory Monitor")
        self.root.geometry("1280x800")
        self.root.minsize(900, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()

    # ------------------------------------------------------------------ layout

    def _build_layout(self):
        """Build the main layout with PanedWindow."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top paned window: feed + sidebar
        self.paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: feed canvas
        self.feed_frame = ttk.LabelFrame(self.paned, text="CCTV Feed")
        self.canvas = tk.Canvas(self.feed_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.paned.add(self.feed_frame, weight=3)

        # Right: sidebar
        sidebar = ttk.Frame(self.paned, width=220)
        self._build_sidebar(sidebar)
        self.paned.add(sidebar, weight=1)

        # Bottom: review panel
        self._build_review_panel(main_frame)

        # Status bar
        self._build_status_bar(main_frame)

    def _build_sidebar(self, parent: ttk.Frame):
        """Build the control sidebar."""
        pad = dict(padx=6, pady=3, sticky="ew")

        ttk.Label(parent, text="Controls", font=("", 11, "bold")).grid(
            row=0, column=0, pady=(6, 10), sticky="w", padx=6)

        # Collection button
        self.btn_collection = ttk.Button(
            parent, text="Start Collection", command=self._toggle_collection)
        self.btn_collection.grid(row=1, column=0, **pad)

        # Train button
        self.btn_train = ttk.Button(
            parent, text="Train Model", command=self._train_model)
        self.btn_train.grid(row=2, column=0, **pad)

        # Feed toggle
        self.btn_feed = ttk.Button(
            parent, text="Hide Feed", command=self._toggle_feed)
        self.btn_feed.grid(row=3, column=0, **pad)

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=4, column=0, sticky="ew", pady=8, padx=6)

        # Overlay checkboxes
        self.var_zones = tk.BooleanVar(value=self.config.display.show_zones)
        self.var_stats = tk.BooleanVar(value=self.config.display.show_stats)
        self.var_boxes = tk.BooleanVar(value=self.config.display.show_boxes)

        ttk.Checkbutton(parent, text="Zones", variable=self.var_zones,
                        command=self._update_overlays).grid(row=5, column=0, **pad)
        ttk.Checkbutton(parent, text="Stats", variable=self.var_stats,
                        command=self._update_overlays).grid(row=6, column=0, **pad)
        ttk.Checkbutton(parent, text="Boxes", variable=self.var_boxes,
                        command=self._update_overlays).grid(row=7, column=0, **pad)

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=8, column=0, sticky="ew", pady=8, padx=6)

        # Collection gallery label
        ttk.Label(parent, text="Collection Gallery").grid(
            row=9, column=0, sticky="w", padx=6)

        # Scrollable gallery
        gallery_frame = ttk.Frame(parent)
        gallery_frame.grid(row=10, column=0, sticky="nsew", padx=6, pady=4)
        parent.rowconfigure(10, weight=1)
        parent.columnconfigure(0, weight=1)

        self.gallery_canvas = tk.Canvas(gallery_frame, width=180, bg="#2b2b2b")
        gallery_scrollbar = ttk.Scrollbar(
            gallery_frame, orient=tk.VERTICAL, command=self.gallery_canvas.yview)
        self.gallery_inner = ttk.Frame(self.gallery_canvas)

        self.gallery_canvas.configure(yscrollcommand=gallery_scrollbar.set)
        gallery_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.gallery_canvas.create_window(
            (0, 0), window=self.gallery_inner, anchor="nw")
        self.gallery_inner.bind(
            "<Configure>",
            lambda e: self.gallery_canvas.configure(
                scrollregion=self.gallery_canvas.bbox("all")))

    def _build_review_panel(self, parent: ttk.Frame):
        """Build the horizontal review queue panel."""
        review_frame = ttk.LabelFrame(parent, text="Review Queue", height=140)
        review_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        review_frame.pack_propagate(False)

        self.review_canvas = tk.Canvas(review_frame, height=120, bg="#1e1e1e")
        review_scroll = ttk.Scrollbar(
            review_frame, orient=tk.HORIZONTAL, command=self.review_canvas.xview)
        self.review_inner = ttk.Frame(self.review_canvas)

        self.review_canvas.configure(xscrollcommand=review_scroll.set)
        review_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.review_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.review_canvas.create_window(
            (0, 0), window=self.review_inner, anchor="nw")
        self.review_inner.bind(
            "<Configure>",
            lambda e: self.review_canvas.configure(
                scrollregion=self.review_canvas.bbox("all")))

    def _build_status_bar(self, parent: ttk.Frame):
        """Build the bottom status bar."""
        self.status_bar = ttk.Frame(parent)
        self.status_bar.pack(fill=tk.X, padx=4, pady=(0, 4))

        self.lbl_fps = ttk.Label(self.status_bar, text="FPS: --")
        self.lbl_fps.pack(side=tk.LEFT, padx=(4, 12))

        self.lbl_ai = ttk.Label(self.status_bar, text="AI: --")
        self.lbl_ai.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_collection = ttk.Label(self.status_bar, text="Collection: 0")
        self.lbl_collection.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_review = ttk.Label(self.status_bar, text="Review: 0")
        self.lbl_review.pack(side=tk.LEFT, padx=(0, 12))

    # --------------------------------------------------------------- frame loop

    def _update_frame(self):
        """Called periodically to process and display one frame."""
        if not self.monitor or not self.monitor.running:
            return

        frame = self.monitor.process_one_frame()

        if frame is not None:
            # Check for low-confidence faces -> review queue
            if self.monitor.last_ai_result:
                self._check_review_queue(
                    self.monitor.last_ai_result,
                    self.monitor.tracker.get_confirmed_tracks(),
                    frame,
                )

            # Draw overlay and show
            if self._feed_visible:
                display = self.monitor._draw_overlay(frame)
                self._show_frame(display)

            # Update status bar
            self._update_status()

        self.root.after(33, self._update_frame)  # ~30fps

    def _show_frame(self, frame: np.ndarray):
        """Convert BGR frame to PhotoImage and display on canvas."""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        # Resize to fit canvas
        frame = cv2.resize(frame, (canvas_w, canvas_h),
                           interpolation=cv2.INTER_LINEAR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(image=pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _update_status(self):
        """Update status bar labels."""
        if not self.monitor:
            return

        elapsed = time.time() - self.monitor.start_time
        fps = self.monitor.frame_count / elapsed if elapsed > 0 else 0
        self.lbl_fps.config(text=f"FPS: {fps:.1f}")

        if self.monitor.last_ai_result:
            self.lbl_ai.config(
                text=f"AI: {self.monitor.last_ai_result.processing_time_ms:.0f}ms")

        if self.monitor.training_collector:
            count = self.monitor.training_collector.sample_count
            self.lbl_collection.config(text=f"Collection: {count}")

        self.lbl_review.config(text=f"Review: {len(self.review_queue)}")

    # ---------------------------------------------------------------- controls

    def _toggle_collection(self):
        """Start or stop training data collection."""
        if not self.monitor:
            return

        if self._collection_active:
            # Stop
            if self.monitor.training_collector:
                self.monitor.training_collector.stop()
                self.monitor.training_collector = None
                self.monitor.ai_worker.training_collector = None
            self._collection_active = False
            self.btn_collection.config(text="Start Collection")
            logger.info("Training data collection DISABLED")
        else:
            # Start
            collector = TrainingDataCollector(
                output_dir=self.config.training.output_dir,
                capture_interval=self.config.training.capture_interval,
                max_samples=self.config.training.max_samples,
                save_full_frame=self.config.training.save_full_frame,
                negative_ratio=self.config.training.negative_ratio,
            )
            collector.start()
            self.monitor.training_collector = collector
            self.monitor.ai_worker.training_collector = collector
            self._collection_active = True
            self.btn_collection.config(text="Stop Collection")
            logger.info("Training data collection ENABLED")

    def _train_model(self):
        """Run face retraining in a background thread."""
        if self._training_in_progress or not self.monitor:
            return

        self._training_in_progress = True
        self.btn_train.config(text="Training...", state=tk.DISABLED)

        def _do_train():
            try:
                count = self.monitor.face_recognizer.retrain_all()
                logger.info(f"Retrained {count} faces")
            except Exception as e:
                logger.error(f"Training failed: {e}")
            finally:
                self.root.after(0, self._on_train_done)

        threading.Thread(target=_do_train, daemon=True).start()

    def _on_train_done(self):
        """Re-enable train button after training completes."""
        self._training_in_progress = False
        self.btn_train.config(text="Train Model", state=tk.NORMAL)

    def _toggle_feed(self):
        """Show or hide the CCTV feed canvas."""
        if self._feed_visible:
            self.canvas.delete("all")
            self.canvas.config(bg="black")
            self.btn_feed.config(text="Show Feed")
            self._feed_visible = False
        else:
            self.btn_feed.config(text="Hide Feed")
            self._feed_visible = True

    def _update_overlays(self):
        """Sync overlay checkboxes to monitor display settings."""
        if self.monitor:
            self.monitor.show_zones = self.var_zones.get()
            self.monitor.show_stats = self.var_stats.get()
            self.monitor.show_boxes = self.var_boxes.get()

    # ------------------------------------------------------------- review queue

    def _check_review_queue(
        self,
        result: ProcessingResult,
        tracks: dict,
        frame: np.ndarray,
    ):
        """Check for low-confidence face detections and add to review queue."""
        threshold = self.config.review.threshold
        max_size = self.config.review.max_queue_size

        for track_id, (name, conf) in result.faces.items():
            if conf >= threshold:
                continue
            if len(self.review_queue) >= max_size:
                break
            # Skip if this track is already in queue
            if any(r.track_id == track_id for r in self.review_queue):
                continue

            # Crop person from frame
            if track_id not in tracks:
                continue
            track = tracks[track_id]
            x1, y1, x2, y2 = track.bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2].copy()

            if crop.size == 0:
                continue

            box_count = result.boxes.get(track_id, 0)
            item = ReviewItem(
                image=crop,
                identity=name,
                confidence=conf,
                box_count=box_count,
                track_id=track_id,
                timestamp=time.time(),
            )
            self.review_queue.append(item)
            self._render_review_item(item, len(self.review_queue) - 1)

    def _render_review_item(self, item: ReviewItem, index: int):
        """Render a review card in the review panel."""
        card = ttk.Frame(self.review_inner, relief="ridge", borderwidth=1)
        card.pack(side=tk.LEFT, padx=4, pady=4)

        # Thumbnail
        thumb = cv2.resize(item.image, (80, 100), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(image=pil_img)
        self._review_photos.append(photo)  # prevent GC

        lbl_img = ttk.Label(card, image=photo)
        lbl_img.pack(padx=2, pady=2)

        # Info
        name_str = item.identity or "Unknown"
        ttk.Label(card, text=name_str, font=("", 9, "bold")).pack()
        ttk.Label(card, text=f"{item.confidence:.0%} | {item.box_count} box").pack()

        # Buttons
        btn_frame = ttk.Frame(card)
        btn_frame.pack(pady=2)
        ttk.Button(btn_frame, text="Correct", width=7,
                   command=lambda i=index: self._accept_review(i)).pack(
            side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="Fix", width=5,
                   command=lambda i=index: self._fix_review(i)).pack(
            side=tk.LEFT, padx=1)

    def _accept_review(self, index: int):
        """Accept detection as correct, remove card."""
        if index >= len(self.review_queue):
            return
        item = self.review_queue.pop(index)
        logger.info(f"Review accepted: track {item.track_id} as '{item.identity}'")
        self._rebuild_review_panel()

    def _fix_review(self, index: int):
        """Open a dialog to correct the detection."""
        if index >= len(self.review_queue):
            return
        item = self.review_queue[index]

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Fix Detection - Track {item.track_id}")
        dialog.geometry("300x180")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Correct Name:").pack(pady=(12, 2))
        name_var = tk.StringVar(value=item.identity or "")
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=25)
        name_entry.pack(pady=2)
        name_entry.focus_set()

        has_box_var = tk.BooleanVar(value=item.box_count > 0)
        ttk.Checkbutton(dialog, text="Has Box", variable=has_box_var).pack(pady=6)

        def _save():
            corrected_name = name_var.get().strip()
            has_box = has_box_var.get()
            if corrected_name:
                logger.info(
                    f"Review corrected: track {item.track_id} -> "
                    f"'{corrected_name}', has_box={has_box}")
            if index < len(self.review_queue):
                self.review_queue.pop(index)
            dialog.destroy()
            self._rebuild_review_panel()

        ttk.Button(dialog, text="Save", command=_save).pack(pady=8)

    def _rebuild_review_panel(self):
        """Clear and re-render all review cards."""
        for widget in self.review_inner.winfo_children():
            widget.destroy()
        self._review_photos.clear()
        for i, item in enumerate(self.review_queue):
            self._render_review_item(item, i)

    # ----------------------------------------------------------- run / shutdown

    def run(self):
        """Create the monitor, start processing, enter mainloop."""
        self.config.headless = True  # GUI handles display
        self.monitor = InventoryMonitor(self.config)

        if not self.monitor.start():
            logger.error("Failed to start monitor")
            self.root.destroy()
            return

        self.root.after(100, self._update_frame)
        self.root.mainloop()

    def _on_close(self):
        """Gracefully stop monitor and close window."""
        if self.monitor:
            self.monitor.stop()
            self.monitor = None
        self.root.destroy()

"""
Tkinter GUI for the Inventory Door Monitor.
Embeds the CCTV feed and provides controls for data collection, training,
review-queue correction, and a manifest browser for curating training data.
"""

import json
import logging
import os
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
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

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ReviewItem:
    """A low-confidence detection queued for manual review."""
    image: np.ndarray           # BGR person crop
    identity: Optional[str]
    confidence: float
    box_count: int
    track_id: int
    timestamp: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_to_photo(bgr: np.ndarray, size: tuple = None) -> ImageTk.PhotoImage:
    """Convert a BGR numpy array to an ImageTk.PhotoImage."""
    if size is not None:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(image=Image.fromarray(rgb))


# ---------------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------------

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

        # Keep references so Tk doesn't garbage-collect images
        self._gallery_photos: List[ImageTk.PhotoImage] = []
        self._review_photos: List[ImageTk.PhotoImage] = []
        self._manifest_photos: List[ImageTk.PhotoImage] = []

        # Gallery refresh tracking
        self._gallery_last_count = 0
        self._gallery_max_thumbs = 50  # keep sidebar light

        # Build UI
        self.root = tk.Tk()
        self.root.title("Inventory Monitor")
        self.root.geometry("1400x900")
        self.root.minsize(1000, 650)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()

    # ================================================================== layout

    def _build_layout(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # ----- top area: feed + sidebar
        self.paned = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: feed
        self.feed_frame = ttk.LabelFrame(self.paned, text="CCTV Feed")
        self.canvas = tk.Canvas(self.feed_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.paned.add(self.feed_frame, weight=3)

        # Right: sidebar
        sidebar = ttk.Frame(self.paned, width=260)
        self._build_sidebar(sidebar)
        self.paned.add(sidebar, weight=1)

        # ----- bottom: review panel
        self._build_review_panel(main)

        # ----- status bar
        self._build_status_bar(main)

    # ------------------------------------------------------------ sidebar

    def _build_sidebar(self, parent: ttk.Frame):
        pad = dict(padx=6, pady=3, sticky="ew")
        parent.columnconfigure(0, weight=1)

        row = 0

        # --- Controls header
        ttk.Label(parent, text="Controls", font=("", 11, "bold")).grid(
            row=row, column=0, pady=(6, 8), sticky="w", padx=6)
        row += 1

        # Collection toggle
        self.btn_collection = ttk.Button(
            parent, text="Start Collection", command=self._toggle_collection)
        self.btn_collection.grid(row=row, column=0, **pad); row += 1

        # Train Model
        self.btn_train = ttk.Button(
            parent, text="Train Model", command=self._train_model)
        self.btn_train.grid(row=row, column=0, **pad); row += 1

        # Hide/Show Feed
        self.btn_feed = ttk.Button(
            parent, text="Hide Feed", command=self._toggle_feed)
        self.btn_feed.grid(row=row, column=0, **pad); row += 1

        # Browse collected samples (opens manifest viewer)
        self.btn_browse = ttk.Button(
            parent, text="Browse Samples", command=self._open_manifest_browser)
        self.btn_browse.grid(row=row, column=0, **pad); row += 1

        # Export YOLO dataset
        self.btn_export = ttk.Button(
            parent, text="Export YOLO Dataset", command=self._export_dataset)
        self.btn_export.grid(row=row, column=0, **pad); row += 1

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky="ew", pady=8, padx=6); row += 1

        # --- Overlay toggles
        self.var_zones = tk.BooleanVar(value=self.config.display.show_zones)
        self.var_stats = tk.BooleanVar(value=self.config.display.show_stats)
        self.var_boxes = tk.BooleanVar(value=self.config.display.show_boxes)

        for text, var in [("Zones", self.var_zones),
                          ("Stats", self.var_stats),
                          ("Boxes", self.var_boxes)]:
            ttk.Checkbutton(parent, text=text, variable=var,
                            command=self._update_overlays).grid(
                                row=row, column=0, **pad)
            row += 1

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky="ew", pady=8, padx=6); row += 1

        # --- Collection config
        ttk.Label(parent, text="Collection Settings",
                  font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=6); row += 1

        # Capture interval slider
        ttk.Label(parent, text="Capture every N frames:").grid(
            row=row, column=0, sticky="w", padx=6); row += 1
        self.var_capture_interval = tk.IntVar(
            value=self.config.training.capture_interval)
        ttk.Spinbox(parent, from_=1, to=120, width=6,
                     textvariable=self.var_capture_interval).grid(
            row=row, column=0, sticky="w", padx=6, pady=2); row += 1

        # Negative ratio slider
        ttk.Label(parent, text="Negative sample ratio:").grid(
            row=row, column=0, sticky="w", padx=6); row += 1
        self.var_neg_ratio = tk.DoubleVar(
            value=self.config.training.negative_ratio)
        ttk.Spinbox(parent, from_=0.0, to=1.0, increment=0.1, width=6,
                     textvariable=self.var_neg_ratio).grid(
            row=row, column=0, sticky="w", padx=6, pady=2); row += 1

        # Save full frames checkbox
        self.var_full_frame = tk.BooleanVar(
            value=self.config.training.save_full_frame)
        ttk.Checkbutton(parent, text="Save full frames",
                        variable=self.var_full_frame).grid(
            row=row, column=0, sticky="w", padx=6, pady=2); row += 1

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky="ew", pady=8, padx=6); row += 1

        # --- Live collection gallery
        ttk.Label(parent, text="Recent Captures").grid(
            row=row, column=0, sticky="w", padx=6); row += 1

        gallery_frame = ttk.Frame(parent)
        gallery_frame.grid(row=row, column=0, sticky="nsew", padx=6, pady=4)
        parent.rowconfigure(row, weight=1)

        self.gallery_canvas = tk.Canvas(gallery_frame, width=220, bg="#2b2b2b")
        gallery_sb = ttk.Scrollbar(
            gallery_frame, orient=tk.VERTICAL,
            command=self.gallery_canvas.yview)
        self.gallery_inner = ttk.Frame(self.gallery_canvas)

        self.gallery_canvas.configure(yscrollcommand=gallery_sb.set)
        gallery_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.gallery_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.gallery_canvas.create_window(
            (0, 0), window=self.gallery_inner, anchor="nw")
        self.gallery_inner.bind(
            "<Configure>",
            lambda _: self.gallery_canvas.configure(
                scrollregion=self.gallery_canvas.bbox("all")))

    # ---------------------------------------------------------- review panel

    def _build_review_panel(self, parent: ttk.Frame):
        review_frame = ttk.LabelFrame(parent, text="Review Queue", height=150)
        review_frame.pack(fill=tk.X, padx=4, pady=(0, 2))
        review_frame.pack_propagate(False)

        self.review_canvas = tk.Canvas(review_frame, height=130, bg="#1e1e1e")
        review_scroll = ttk.Scrollbar(
            review_frame, orient=tk.HORIZONTAL,
            command=self.review_canvas.xview)
        self.review_inner = ttk.Frame(self.review_canvas)

        self.review_canvas.configure(xscrollcommand=review_scroll.set)
        review_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.review_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.review_canvas.create_window(
            (0, 0), window=self.review_inner, anchor="nw")
        self.review_inner.bind(
            "<Configure>",
            lambda _: self.review_canvas.configure(
                scrollregion=self.review_canvas.bbox("all")))

    # ----------------------------------------------------------- status bar

    def _build_status_bar(self, parent: ttk.Frame):
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, padx=4, pady=(0, 4))

        self.lbl_fps = ttk.Label(bar, text="FPS: --")
        self.lbl_fps.pack(side=tk.LEFT, padx=(4, 12))

        self.lbl_ai = ttk.Label(bar, text="AI: --")
        self.lbl_ai.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_collection = ttk.Label(bar, text="Collection: 0")
        self.lbl_collection.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_pos_neg = ttk.Label(bar, text="Pos/Neg: -/-")
        self.lbl_pos_neg.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_review = ttk.Label(bar, text="Review: 0")
        self.lbl_review.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_tracks = ttk.Label(bar, text="Tracks: 0")
        self.lbl_tracks.pack(side=tk.LEFT, padx=(0, 12))

    # ================================================================ frame loop

    def _update_frame(self):
        """Called via root.after to process and display one frame."""
        if not self.monitor or not self.monitor.running:
            return

        try:
            frame = self.monitor.process_one_frame()
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.root.after(100, self._update_frame)
            return

        if frame is not None:
            # Review queue check
            if self.monitor.last_ai_result:
                try:
                    self._check_review_queue(
                        self.monitor.last_ai_result,
                        self.monitor.tracker.get_confirmed_tracks(),
                        frame,
                    )
                except Exception:
                    pass

            # Draw + show
            if self._feed_visible:
                try:
                    display = self.monitor._draw_overlay(frame)
                    self._show_frame(display)
                except Exception:
                    pass

            # Status
            self._update_status()

            # Periodically refresh gallery
            self._maybe_refresh_gallery()

        self.root.after(33, self._update_frame)

    def _show_frame(self, frame: np.ndarray):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        self._photo = _bgr_to_photo(frame, (cw, ch))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._photo)

    def _update_status(self):
        if not self.monitor:
            return

        elapsed = time.time() - self.monitor.start_time
        fps = self.monitor.frame_count / elapsed if elapsed > 0 else 0
        self.lbl_fps.config(text=f"FPS: {fps:.1f}")

        if self.monitor.last_ai_result:
            self.lbl_ai.config(
                text=f"AI: {self.monitor.last_ai_result.processing_time_ms:.0f}ms")

        collector = self.monitor.training_collector
        if collector:
            count = collector._sample_counter
            self.lbl_collection.config(text=f"Collection: {count}")
            # Pos/neg from manifest (cheap: only recount every ~2s via frame count)
            if self.monitor.frame_count % 60 == 0:
                try:
                    pos, neg = collector._count_label_balance()
                    self.lbl_pos_neg.config(text=f"Pos/Neg: {pos}/{neg}")
                except Exception:
                    pass

        tracks = self.monitor.tracker.get_confirmed_tracks()
        self.lbl_tracks.config(text=f"Tracks: {len(tracks)}")
        self.lbl_review.config(text=f"Review: {len(self.review_queue)}")

    # ====================================================== collection gallery

    def _maybe_refresh_gallery(self):
        """Add new thumbnails to the sidebar gallery from collected samples."""
        collector = self.monitor.training_collector if self.monitor else None
        if not collector:
            return
        current = collector._sample_counter
        if current <= self._gallery_last_count:
            return
        # Only refresh every 10 new samples to avoid thrashing
        if (current - self._gallery_last_count) < 10 and current > 10:
            return

        self._gallery_last_count = current
        self._refresh_gallery(collector)

    def _refresh_gallery(self, collector: TrainingDataCollector):
        """Reload the last N thumbnails from the images directory."""
        for w in self.gallery_inner.winfo_children():
            w.destroy()
        self._gallery_photos.clear()

        images_dir = collector._images_dir
        if not images_dir.exists():
            return

        # Get most-recent images (skip _full variants)
        files = sorted(
            [f for f in images_dir.iterdir()
             if f.suffix == ".jpg" and "_full" not in f.name],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:self._gallery_max_thumbs]

        for fpath in files:
            try:
                img = cv2.imread(str(fpath))
                if img is None:
                    continue
                photo = _bgr_to_photo(img, (100, 80))
                self._gallery_photos.append(photo)

                card = ttk.Frame(self.gallery_inner)
                card.pack(anchor="w", pady=2, padx=2)
                ttk.Label(card, image=photo).pack(side=tk.LEFT, padx=2)
                label = "pos" if fpath.name.startswith("pos") else "neg"
                ttk.Label(card, text=f"{fpath.stem}\n[{label}]",
                          font=("", 7)).pack(side=tk.LEFT, padx=2)
            except Exception:
                continue

    # =========================================================== controls

    def _toggle_collection(self):
        if not self.monitor:
            return

        if self._collection_active:
            if self.monitor.training_collector:
                self.monitor.training_collector.stop()
                self.monitor.training_collector = None
                self.monitor.ai_worker.training_collector = None
            self._collection_active = False
            self.btn_collection.config(text="Start Collection")
            logger.info("Training data collection DISABLED")
        else:
            collector = TrainingDataCollector(
                output_dir=self.config.training.output_dir,
                capture_interval=self.var_capture_interval.get(),
                max_samples=self.config.training.max_samples,
                save_full_frame=self.var_full_frame.get(),
                negative_ratio=self.var_neg_ratio.get(),
            )
            collector.start()
            self.monitor.training_collector = collector
            self.monitor.ai_worker.training_collector = collector
            self._collection_active = True
            self.btn_collection.config(text="Stop Collection")
            logger.info("Training data collection ENABLED")

    def _train_model(self):
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
        self._training_in_progress = False
        self.btn_train.config(text="Train Model", state=tk.NORMAL)

    def _toggle_feed(self):
        if self._feed_visible:
            self.canvas.delete("all")
            self.canvas.config(bg="black")
            self.btn_feed.config(text="Show Feed")
            self._feed_visible = False
        else:
            self.btn_feed.config(text="Hide Feed")
            self._feed_visible = True

    def _update_overlays(self):
        if self.monitor:
            self.monitor.show_zones = self.var_zones.get()
            self.monitor.show_stats = self.var_stats.get()
            self.monitor.show_boxes = self.var_boxes.get()

    # =========================================================== review queue

    def _check_review_queue(self, result: ProcessingResult, tracks: dict,
                            frame: np.ndarray):
        threshold = self.config.review.threshold
        max_size = self.config.review.max_queue_size

        for track_id, (name, conf) in result.faces.items():
            if conf >= threshold:
                continue
            if len(self.review_queue) >= max_size:
                break
            if any(r.track_id == track_id for r in self.review_queue):
                continue
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
                image=crop, identity=name, confidence=conf,
                box_count=box_count, track_id=track_id,
                timestamp=time.time())
            self.review_queue.append(item)
            self._render_review_item(item, len(self.review_queue) - 1)

    def _render_review_item(self, item: ReviewItem, index: int):
        card = ttk.Frame(self.review_inner, relief="ridge", borderwidth=1)
        card.pack(side=tk.LEFT, padx=4, pady=4)

        photo = _bgr_to_photo(item.image, (80, 100))
        self._review_photos.append(photo)

        ttk.Label(card, image=photo).pack(padx=2, pady=2)

        name_str = item.identity or "Unknown"
        ttk.Label(card, text=name_str, font=("", 9, "bold")).pack()
        ttk.Label(card, text=f"{item.confidence:.0%} | {item.box_count} box").pack()

        bf = ttk.Frame(card)
        bf.pack(pady=2)
        ttk.Button(bf, text="Correct", width=7,
                   command=lambda i=index: self._accept_review(i)).pack(
            side=tk.LEFT, padx=1)
        ttk.Button(bf, text="Fix", width=5,
                   command=lambda i=index: self._fix_review(i)).pack(
            side=tk.LEFT, padx=1)

    def _accept_review(self, index: int):
        if index >= len(self.review_queue):
            return
        item = self.review_queue.pop(index)
        logger.info(f"Review accepted: track {item.track_id} as '{item.identity}'")
        self._update_manifest_verified(item.track_id, True)
        self._rebuild_review_panel()

    def _fix_review(self, index: int):
        if index >= len(self.review_queue):
            return
        item = self.review_queue[index]

        dialog = tk.Toplevel(self.root)
        dialog.title(f"Fix Detection - Track {item.track_id}")
        dialog.geometry("320x260")
        dialog.transient(self.root)
        dialog.grab_set()

        # Thumbnail
        photo = _bgr_to_photo(item.image, (100, 120))
        self._review_photos.append(photo)
        ttk.Label(dialog, image=photo).pack(pady=(8, 4))

        ttk.Label(dialog, text="Correct Name:").pack(pady=(4, 2))
        name_var = tk.StringVar(value=item.identity or "")
        entry = ttk.Entry(dialog, textvariable=name_var, width=25)
        entry.pack(pady=2)
        entry.focus_set()

        has_box_var = tk.BooleanVar(value=item.box_count > 0)
        ttk.Checkbutton(dialog, text="Carrying a box",
                        variable=has_box_var).pack(pady=4)

        box_count_var = tk.IntVar(value=item.box_count)
        cf = ttk.Frame(dialog)
        cf.pack(pady=2)
        ttk.Label(cf, text="Box count:").pack(side=tk.LEFT)
        ttk.Spinbox(cf, from_=0, to=10, width=4,
                     textvariable=box_count_var).pack(side=tk.LEFT, padx=4)

        def _save():
            corrected_name = name_var.get().strip()
            has_box = has_box_var.get()
            box_cnt = box_count_var.get()
            logger.info(
                f"Review corrected: track {item.track_id} -> "
                f"'{corrected_name}', has_box={has_box}, boxes={box_cnt}")
            self._save_correction(item, corrected_name, has_box, box_cnt)
            if index < len(self.review_queue):
                self.review_queue.pop(index)
            dialog.destroy()
            self._rebuild_review_panel()

        ttk.Button(dialog, text="Save", command=_save).pack(pady=8)

    def _rebuild_review_panel(self):
        for w in self.review_inner.winfo_children():
            w.destroy()
        self._review_photos.clear()
        for i, item in enumerate(self.review_queue):
            self._render_review_item(item, i)

    # =========================================================== manifest helpers

    def _get_manifest_path(self) -> Path:
        return Path(self.config.training.output_dir) / "manifest.jsonl"

    def _update_manifest_verified(self, track_id: int, verified: bool):
        """Mark all manifest entries for a track_id as verified."""
        path = self._get_manifest_path()
        if not path.exists():
            return
        try:
            lines = path.read_text().splitlines()
            new_lines = []
            for line in lines:
                entry = json.loads(line)
                if entry.get("track_id") == track_id:
                    entry["verified"] = verified
                new_lines.append(json.dumps(entry))
            path.write_text("\n".join(new_lines) + "\n")
        except Exception as e:
            logger.error(f"Manifest update error: {e}")

    def _save_correction(self, item: ReviewItem, name: str,
                         has_box: bool, box_count: int):
        """Append a correction entry to the manifest."""
        path = self._get_manifest_path()
        correction = {
            "path": None,
            "label": "has_box" if has_box else "no_box",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "track_id": item.track_id,
            "identity": name,
            "box_count": box_count,
            "methods": ["manual_correction"],
            "confidences": [item.confidence],
            "verified": True,
            "correction": True,
        }
        # Also save the crop image
        images_dir = Path(self.config.training.output_dir) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        prefix = "pos" if has_box else "neg"
        filename = f"{prefix}_fix_{item.track_id}_{int(time.time())}"
        img_path = images_dir / f"{filename}.jpg"
        cv2.imwrite(str(img_path), item.image)
        correction["path"] = str(img_path)

        try:
            with open(path, "a") as f:
                f.write(json.dumps(correction) + "\n")
        except Exception as e:
            logger.error(f"Correction save error: {e}")

    # =========================================================== manifest browser

    def _open_manifest_browser(self):
        """Open a Toplevel window to browse, verify, and delete collected samples."""
        path = self._get_manifest_path()
        if not path.exists():
            messagebox.showinfo("Browse Samples",
                                "No manifest found. Start collecting data first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Training Data Browser")
        win.geometry("900x600")
        win.transient(self.root)

        # Toolbar
        toolbar = ttk.Frame(win)
        toolbar.pack(fill=tk.X, padx=4, pady=4)

        filter_var = tk.StringVar(value="all")
        ttk.Label(toolbar, text="Filter:").pack(side=tk.LEFT, padx=4)
        for val, text in [("all", "All"), ("has_box", "Has Box"),
                          ("no_box", "No Box"), ("unverified", "Unverified")]:
            ttk.Radiobutton(toolbar, text=text, variable=filter_var,
                            value=val).pack(side=tk.LEFT, padx=2)

        ttk.Button(toolbar, text="Refresh",
                   command=lambda: _load()).pack(side=tk.RIGHT, padx=4)
        ttk.Button(toolbar, text="Delete Selected",
                   command=lambda: _delete_selected()).pack(side=tk.RIGHT, padx=4)
        ttk.Button(toolbar, text="Verify Selected",
                   command=lambda: _verify_selected()).pack(side=tk.RIGHT, padx=4)

        # Treeview for manifest entries
        cols = ("label", "identity", "boxes", "methods", "verified", "time")
        tree = ttk.Treeview(win, columns=cols, show="headings", selectmode="extended")
        for col, w in [("label", 80), ("identity", 100), ("boxes", 50),
                       ("methods", 120), ("verified", 70), ("time", 150)]:
            tree.heading(col, text=col.title())
            tree.column(col, width=w)

        tree_scroll = ttk.Scrollbar(win, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=tree_scroll.set)
        tree.pack(fill=tk.BOTH, expand=True, padx=4, side=tk.LEFT)
        tree_scroll.pack(fill=tk.Y, side=tk.LEFT)

        # Preview panel
        preview_frame = ttk.LabelFrame(win, text="Preview", width=250)
        preview_frame.pack(fill=tk.Y, side=tk.RIGHT, padx=4, pady=4)
        preview_frame.pack_propagate(False)
        preview_label = ttk.Label(preview_frame)
        preview_label.pack(expand=True)
        preview_info = ttk.Label(preview_frame, text="", wraplength=230)
        preview_info.pack(pady=4)

        entries: List[dict] = []
        _preview_photo = [None]  # mutable ref for GC protection

        def _load():
            nonlocal entries
            tree.delete(*tree.get_children())
            entries = []
            try:
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    entries.append(json.loads(line))
            except Exception as e:
                logger.error(f"Manifest read error: {e}")
                return

            filt = filter_var.get()
            for i, e in enumerate(entries):
                if filt == "has_box" and e.get("label") != "has_box":
                    continue
                if filt == "no_box" and e.get("label") != "no_box":
                    continue
                if filt == "unverified" and e.get("verified", False):
                    continue
                tree.insert("", tk.END, iid=str(i), values=(
                    e.get("label", "?"),
                    e.get("identity", "?"),
                    e.get("box_count", 0),
                    ",".join(e.get("methods", [])),
                    "Yes" if e.get("verified") else "No",
                    e.get("timestamp", ""),
                ))

        def _on_select(_event):
            sel = tree.selection()
            if not sel:
                return
            idx = int(sel[0])
            entry = entries[idx]
            img_path = entry.get("path")
            if img_path and Path(img_path).exists():
                img = cv2.imread(img_path)
                if img is not None:
                    _preview_photo[0] = _bgr_to_photo(img, (220, 180))
                    preview_label.config(image=_preview_photo[0])
            info = (f"Track: {entry.get('track_id')}\n"
                    f"Label: {entry.get('label')}\n"
                    f"Identity: {entry.get('identity')}\n"
                    f"Boxes: {entry.get('box_count')}\n"
                    f"Verified: {entry.get('verified')}")
            preview_info.config(text=info)

        tree.bind("<<TreeviewSelect>>", _on_select)

        def _verify_selected():
            sel = tree.selection()
            if not sel:
                return
            for s in sel:
                idx = int(s)
                entries[idx]["verified"] = True
            _write_back()
            _load()

        def _delete_selected():
            sel = tree.selection()
            if not sel:
                return
            if not messagebox.askyesno("Delete",
                                       f"Delete {len(sel)} sample(s)?"):
                return
            indices = sorted([int(s) for s in sel], reverse=True)
            for idx in indices:
                e = entries[idx]
                # Delete image file
                img_path = e.get("path")
                if img_path:
                    p = Path(img_path)
                    if p.exists():
                        p.unlink()
                    # Delete label file
                    lbl = Path(self.config.training.output_dir) / "labels" / (
                        p.stem + ".txt")
                    if lbl.exists():
                        lbl.unlink()
                entries.pop(idx)
            _write_back()
            _load()

        def _write_back():
            try:
                with open(path, "w") as f:
                    for e in entries:
                        f.write(json.dumps(e) + "\n")
            except Exception as ex:
                logger.error(f"Manifest write error: {ex}")

        filter_var.trace_add("write", lambda *_: _load())
        _load()

    # =========================================================== dataset export

    def _export_dataset(self):
        """Export verified samples as a YOLO-format dataset ready for training."""
        manifest_path = self._get_manifest_path()
        if not manifest_path.exists():
            messagebox.showinfo("Export", "No manifest found.")
            return

        dest = filedialog.askdirectory(title="Select export directory")
        if not dest:
            return

        def _do_export():
            try:
                dest_path = Path(dest)
                img_dir = dest_path / "images" / "train"
                lbl_dir = dest_path / "labels" / "train"
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)

                entries = []
                for line in manifest_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    entries.append(json.loads(line))

                # Only export verified samples
                exported = 0
                for e in entries:
                    if not e.get("verified", False):
                        continue
                    src_img = e.get("path")
                    if not src_img or not Path(src_img).exists():
                        continue

                    src = Path(src_img)
                    dst_img = img_dir / src.name
                    # Copy image
                    import shutil
                    shutil.copy2(str(src), str(dst_img))

                    # Copy label if exists
                    src_lbl = Path(self.config.training.output_dir) / "labels" / (
                        src.stem + ".txt")
                    if src_lbl.exists():
                        shutil.copy2(str(src_lbl), str(lbl_dir / src_lbl.name))
                    else:
                        # Create label from manifest info
                        label = e.get("label", "no_box")
                        lbl_path = lbl_dir / (src.stem + ".txt")
                        if label == "has_box":
                            # Full-image bounding box as fallback
                            lbl_path.write_text("0 0.5 0.5 1.0 1.0\n")
                        else:
                            lbl_path.write_text("")  # empty = no objects
                    exported += 1

                # Write data.yaml
                yaml_path = dest_path / "data.yaml"
                yaml_path.write_text(
                    f"path: {dest_path.resolve()}\n"
                    f"train: images/train\n"
                    f"val: images/train\n"
                    f"\n"
                    f"nc: 1\n"
                    f"names:\n"
                    f"  0: box\n"
                )

                self.root.after(0, lambda: messagebox.showinfo(
                    "Export Complete",
                    f"Exported {exported} verified samples to:\n{dest_path}\n\n"
                    f"Train with:\n"
                    f"  yolo detect train data={yaml_path} model=yolo11n.pt epochs=50"))

            except Exception as ex:
                logger.error(f"Export error: {ex}")
                self.root.after(0, lambda: messagebox.showerror(
                    "Export Error", str(ex)))

        threading.Thread(target=_do_export, daemon=True).start()

    # =========================================================== run / shutdown

    def run(self):
        self.config.headless = True
        self.monitor = InventoryMonitor(self.config)

        if not self.monitor.start():
            logger.error("Failed to start monitor")
            self.root.destroy()
            return

        self.root.after(100, self._update_frame)
        self.root.mainloop()

    def _on_close(self):
        if self.monitor:
            self.monitor.stop()
            self.monitor = None
        self.root.destroy()

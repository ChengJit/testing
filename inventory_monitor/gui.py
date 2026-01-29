"""
Simplified Tkinter GUI for the Inventory Door Monitor.
Focused on: live feed, entry/exit detection with boxes, face verification, and live training.
"""

import logging
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict, Set

import cv2
import numpy as np

try:
    import tkinter as tk
    from tkinter import ttk, simpledialog
except ImportError:
    raise ImportError("Tkinter is required for GUI mode")

try:
    from PIL import Image, ImageTk
except ImportError:
    raise ImportError("Pillow is required for GUI mode: pip install Pillow")

from .config import Config
from .app import InventoryMonitor, ProcessingResult

logger = logging.getLogger(__name__)


def _bgr_to_photo(bgr: np.ndarray, size: tuple = None) -> ImageTk.PhotoImage:
    """Convert a BGR numpy array to an ImageTk.PhotoImage."""
    if size is not None:
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(image=Image.fromarray(rgb))


class InventoryGUI:
    """Simplified GUI: live feed + face verification + entry/exit with boxes."""

    # Don't re-prompt the same track within this many seconds
    VERIFY_COOLDOWN = 15.0
    # Face confidence below this triggers verification popup
    VERIFY_THRESHOLD = 0.60

    def __init__(self, config: Config):
        self.config = config
        self.monitor: Optional[InventoryMonitor] = None

        # State
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._training_in_progress = False
        self._verify_dialog_open = False

        # Track which track_ids we've already prompted for (cooldown)
        self._verified_tracks: Dict[int, float] = {}
        # Track IDs the user has explicitly confirmed (don't ask again)
        self._confirmed_tracks: Set[int] = set()

        # Build UI
        self.root = tk.Tk()
        self.root.title("Door Monitor")
        self.root.geometry("1280x780")
        self.root.minsize(800, 500)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_layout()

    # ================================================================ layout

    def _build_layout(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # Top: feed + sidebar
        top = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        top.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Left: CCTV feed
        feed_frame = ttk.LabelFrame(top, text="CCTV Feed")
        self.canvas = tk.Canvas(feed_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        top.add(feed_frame, weight=4)

        # Right: controls
        sidebar = ttk.Frame(top, width=220)
        self._build_sidebar(sidebar)
        top.add(sidebar, weight=1)

        # Bottom: status bar
        self._build_status_bar(main)

    def _build_sidebar(self, parent: ttk.Frame):
        pad = dict(padx=6, pady=4, sticky="ew")
        parent.columnconfigure(0, weight=1)
        row = 0

        ttk.Label(parent, text="Controls", font=("", 11, "bold")).grid(
            row=row, column=0, pady=(6, 8), sticky="w", padx=6)
        row += 1

        # Train faces button
        self.btn_train = ttk.Button(
            parent, text="Retrain Faces", command=self._train_faces)
        self.btn_train.grid(row=row, column=0, **pad)
        row += 1

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky="ew", pady=8, padx=6)
        row += 1

        # Overlay toggles
        self.var_zones = tk.BooleanVar(value=self.config.display.show_zones)
        self.var_stats = tk.BooleanVar(value=self.config.display.show_stats)
        self.var_boxes = tk.BooleanVar(value=self.config.display.show_boxes)

        for text, var in [("Show Door Line", self.var_zones),
                          ("Show Stats", self.var_stats),
                          ("Show Boxes", self.var_boxes)]:
            ttk.Checkbutton(parent, text=text, variable=var,
                            command=self._update_overlays).grid(
                                row=row, column=0, **pad)
            row += 1

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky="ew", pady=8, padx=6)
        row += 1

        # Known faces list
        ttk.Label(parent, text="Known Faces", font=("", 10, "bold")).grid(
            row=row, column=0, sticky="w", padx=6)
        row += 1

        self.faces_listbox = tk.Listbox(parent, height=8, font=("", 9))
        self.faces_listbox.grid(row=row, column=0, sticky="nsew", padx=6, pady=4)
        parent.rowconfigure(row, weight=1)
        row += 1

        ttk.Button(parent, text="Refresh List",
                   command=self._refresh_faces_list).grid(row=row, column=0, **pad)
        row += 1

    def _build_status_bar(self, parent: ttk.Frame):
        bar = ttk.Frame(parent)
        bar.pack(fill=tk.X, padx=4, pady=(0, 4))

        self.lbl_fps = ttk.Label(bar, text="FPS: --")
        self.lbl_fps.pack(side=tk.LEFT, padx=(4, 12))

        self.lbl_ai = ttk.Label(bar, text="AI: --")
        self.lbl_ai.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_tracks = ttk.Label(bar, text="Tracks: 0")
        self.lbl_tracks.pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_entries = ttk.Label(bar, text="In: 0 | Out: 0")
        self.lbl_entries.pack(side=tk.LEFT, padx=(0, 12))

    # ================================================================ frame loop

    def _update_frame(self):
        if not self.monitor or not self.monitor.running:
            return

        try:
            frame = self.monitor.process_one_frame()
        except Exception as e:
            logger.error(f"Frame error: {e}")
            self.root.after(100, self._update_frame)
            return

        if frame is not None:
            # Check for faces needing verification
            if self.monitor.last_ai_result and not self._verify_dialog_open:
                self._check_face_verification(
                    self.monitor.last_ai_result, frame)

            # Draw + show
            try:
                display = self.monitor._draw_overlay(frame)
                self._show_frame(display)
            except Exception:
                pass

            self._update_status()

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

        tracks = self.monitor.tracker.get_confirmed_tracks()
        self.lbl_tracks.config(text=f"Tracks: {len(tracks)}")

        stats = self.monitor.event_manager.get_statistics()
        self.lbl_entries.config(
            text=f"In: {stats['total_entries']} | Out: {stats['total_exits']}")

    # ================================================================ face verification

    def _check_face_verification(self, result: ProcessingResult,
                                 frame: np.ndarray):
        """Check if any detected face needs user verification."""
        now = time.time()
        tracks = self.monitor.tracker.get_confirmed_tracks()

        for track_id, (name, conf) in result.faces.items():
            # Skip if already confirmed by user
            if track_id in self._confirmed_tracks:
                continue

            # Skip if identity is locked with high confidence
            track = tracks.get(track_id)
            if track and track.identity_locked:
                continue

            # Skip if recently prompted
            last_time = self._verified_tracks.get(track_id, 0)
            if now - last_time < self.VERIFY_COOLDOWN:
                continue

            # Trigger verification if confidence below threshold
            if conf < self.VERIFY_THRESHOLD:
                self._verified_tracks[track_id] = now

                # Get person crop for the dialog
                if track_id not in tracks:
                    continue
                bbox = tracks[track_id].bbox
                x1, y1, x2, y2 = bbox
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue

                # Also check if face recognizer has unknown face data
                box_count = result.boxes.get(track_id, 0)
                self._open_verify_dialog(
                    track_id, name, conf, crop, box_count)
                break  # one dialog at a time

    def _open_verify_dialog(self, track_id: int, suggested_name: str,
                            confidence: float, crop: np.ndarray,
                            box_count: int):
        """Open a dialog asking user to verify/correct face identity."""
        self._verify_dialog_open = True

        dialog = tk.Toplevel(self.root)
        dialog.title("Who is this?")
        dialog.geometry("360x380")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        # Prevent closing without answering
        dialog.protocol("WM_DELETE_WINDOW", lambda: self._dismiss_verify(dialog, track_id))

        # Person crop
        photo = _bgr_to_photo(crop, (160, 200))
        self._verify_photo = photo  # prevent GC
        ttk.Label(dialog, image=photo).pack(pady=(10, 5))

        # Info
        if suggested_name and suggested_name != "Unknown":
            question = f"Is this {suggested_name}? (confidence: {confidence:.0%})"
        else:
            question = "Unknown person detected"
        ttk.Label(dialog, text=question, font=("", 10)).pack(pady=4)

        box_text = f"Carrying {box_count} box(es)" if box_count > 0 else "No boxes"
        ttk.Label(dialog, text=box_text, font=("", 9)).pack(pady=2)

        # Name input
        ttk.Label(dialog, text="Enter name (or correct):").pack(pady=(8, 2))
        name_var = tk.StringVar(value=suggested_name if suggested_name != "Unknown" else "")
        entry = ttk.Entry(dialog, textvariable=name_var, width=25, font=("", 11))
        entry.pack(pady=4)
        entry.focus_set()
        entry.select_range(0, tk.END)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)

        def _confirm():
            name = name_var.get().strip()
            if name:
                self._save_and_train(track_id, name, crop)
                self._confirmed_tracks.add(track_id)
            self._verify_dialog_open = False
            dialog.destroy()

        def _skip():
            self._verify_dialog_open = False
            dialog.destroy()

        ttk.Button(btn_frame, text="Confirm", command=_confirm, width=12).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="Skip", command=_skip, width=12).pack(
            side=tk.LEFT, padx=4)

        # Enter key confirms
        entry.bind("<Return>", lambda e: _confirm())

    def _dismiss_verify(self, dialog, track_id):
        """Handle dialog close (X button)."""
        self._verify_dialog_open = False
        dialog.destroy()

    def _save_and_train(self, track_id: int, name: str, crop: np.ndarray):
        """Save face crop to known_faces/<name>/ and retrain embeddings."""
        # Save the crop image
        person_dir = self.config.faces_dir / name
        person_dir.mkdir(parents=True, exist_ok=True)
        img_path = person_dir / f"{name}_{int(time.time())}_{track_id}.jpg"
        cv2.imwrite(str(img_path), crop)
        logger.info(f"Saved face crop: {img_path}")

        # Also save any collected unknown face images from the recognizer
        unknown_data = self.monitor.face_recognizer.unknown_faces.get(track_id)
        if unknown_data and unknown_data.images:
            for i, img in enumerate(unknown_data.images):
                extra_path = person_dir / f"{name}_{int(time.time())}_{track_id}_f{i}.jpg"
                cv2.imwrite(str(extra_path), img)
            logger.info(f"Saved {len(unknown_data.images)} additional face crops for {name}")
            # Clear collected data
            del self.monitor.face_recognizer.unknown_faces[track_id]

        # Update track identity immediately
        tracks = self.monitor.tracker.get_confirmed_tracks()
        track = tracks.get(track_id)
        if track:
            track.set_identity(name, 1.0, lock=True)

        # Retrain in background so detection keeps running
        self._retrain_background()

    def _retrain_background(self):
        """Retrain face embeddings in a background thread."""
        if self._training_in_progress:
            return

        self._training_in_progress = True
        self.btn_train.config(text="Training...", state=tk.DISABLED)

        def _do():
            try:
                count = self.monitor.face_recognizer.retrain_all()
                logger.info(f"Retrained {count} faces")
            except Exception as e:
                logger.error(f"Training failed: {e}")
            finally:
                self.root.after(0, self._on_train_done)

        threading.Thread(target=_do, daemon=True).start()

    def _on_train_done(self):
        self._training_in_progress = False
        self.btn_train.config(text="Retrain Faces", state=tk.NORMAL)
        self._refresh_faces_list()

    # ================================================================ controls

    def _train_faces(self):
        self._retrain_background()

    def _update_overlays(self):
        if self.monitor:
            self.monitor.show_zones = self.var_zones.get()
            self.monitor.show_stats = self.var_stats.get()
            self.monitor.show_boxes = self.var_boxes.get()

    def _refresh_faces_list(self):
        """Refresh the known faces listbox."""
        self.faces_listbox.delete(0, tk.END)
        if self.config.faces_dir.exists():
            for person_dir in sorted(self.config.faces_dir.iterdir()):
                if person_dir.is_dir():
                    count = len(list(person_dir.glob("*.jpg")) +
                                list(person_dir.glob("*.jpeg")) +
                                list(person_dir.glob("*.png")))
                    self.faces_listbox.insert(tk.END, f"{person_dir.name} ({count} imgs)")

    # ================================================================ run / shutdown

    def run(self):
        self.config.headless = True
        self.monitor = InventoryMonitor(self.config)

        if not self.monitor.start():
            logger.error("Failed to start monitor")
            self.root.destroy()
            return

        self._refresh_faces_list()
        self.root.after(100, self._update_frame)
        self.root.mainloop()

    def _on_close(self):
        if self.monitor:
            self.monitor.stop()
            self.monitor = None
        self.root.destroy()

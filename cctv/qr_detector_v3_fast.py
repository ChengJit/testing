"""
QR Detector V3: Fastest - Downscale + Smart Upscale
Downscales full frame for box detection, upscales individual crops for QR.
Best for 3K/4K cameras.
"""

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class QRDetectorV3:
    """
    Fastest QR detection for high-res cameras.

    Strategy:
    1. Use box detections from YOLO (already optimized)
    2. Crop each box region from FULL resolution frame
    3. Enhance crop (contrast, sharpen) for better QR reading
    4. Parallel decode with pyzbar
    """

    def __init__(self, padding=30, target_crop_size=300, max_workers=4, enhance=True):
        """
        Args:
            padding: Extra pixels around box
            target_crop_size: Resize crops to this size for consistent detection
            max_workers: Parallel threads
            enhance: Apply image enhancement for better QR detection
        """
        self.padding = padding
        self.target_crop_size = target_crop_size
        self.max_workers = max_workers
        self.enhance = enhance
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Try pyzbar
        self.use_pyzbar = False
        try:
            from pyzbar import pyzbar
            self.pyzbar = pyzbar
            self.use_pyzbar = True
        except ImportError:
            pass

    def detect_in_frame(self, frame, box_detections):
        """
        Fast QR detection optimized for 3K cameras.
        """
        if not box_detections:
            return []

        h, w = frame.shape[:2]
        qr_codes = []

        # Prepare crops from full-res frame
        crops = []
        for det in box_detections:
            x1 = max(0, det['x1'] - self.padding)
            y1 = max(0, det['y1'] - self.padding)
            x2 = min(w, det['x2'] + self.padding)
            y2 = min(h, det['y2'] + self.padding)

            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] > 20 and crop.shape[1] > 20:
                crops.append((crop.copy(), x1, y1, x2 - x1, y2 - y1, det))

        if not crops:
            return []

        # Process in parallel
        futures = {}
        for crop_data in crops:
            future = self.executor.submit(self._process_crop_fast, *crop_data[:5])
            futures[future] = crop_data[5]  # det

        # Collect results
        for future in as_completed(futures):
            det = futures[future]
            try:
                results = future.result()
                for qr in results:
                    qr['box_detection'] = det
                    qr_codes.append(qr)
            except Exception as e:
                pass

        return qr_codes

    def _process_crop_fast(self, crop, offset_x, offset_y, orig_w, orig_h):
        """Process single crop with optimizations."""
        results = []

        # Resize to target size for consistent detection
        crop_h, crop_w = crop.shape[:2]
        scale_x = self.target_crop_size / crop_w
        scale_y = self.target_crop_size / crop_h
        scale = min(scale_x, scale_y, 2.0)  # Don't upscale more than 2x

        if scale != 1.0:
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Enhance image for better QR detection
        if self.enhance:
            crop = self._enhance_for_qr(crop)

        # Try multiple detection methods
        results = self._multi_detect(crop, scale, offset_x, offset_y)

        return results

    def _enhance_for_qr(self, image):
        """Enhance image for better QR code detection."""
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Adaptive threshold to handle varying lighting
        # This helps with QR codes in shadows or bright areas
        enhanced = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Convert back to BGR for pyzbar compatibility
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def _multi_detect(self, crop, scale, offset_x, offset_y):
        """Try multiple detection methods."""
        results = []

        # Method 1: pyzbar on enhanced image
        if self.use_pyzbar:
            try:
                decoded = self.pyzbar.decode(crop)
                for obj in decoded:
                    data = obj.data.decode('utf-8')
                    rect = obj.rect

                    # Convert back to original frame coordinates
                    cx = int(rect.left / scale + rect.width / scale / 2 + offset_x)
                    cy = int(rect.top / scale + rect.height / scale / 2 + offset_y)

                    pts = np.array([
                        [rect.left / scale + offset_x, rect.top / scale + offset_y],
                        [(rect.left + rect.width) / scale + offset_x, rect.top / scale + offset_y],
                        [(rect.left + rect.width) / scale + offset_x, (rect.top + rect.height) / scale + offset_y],
                        [rect.left / scale + offset_x, (rect.top + rect.height) / scale + offset_y]
                    ], dtype=np.int32)

                    results.append({
                        'data': data,
                        'center': (cx, cy),
                        'points': pts
                    })

                if results:
                    return results
            except:
                pass

        # Method 2: Try on original (non-enhanced) grayscale
        if self.use_pyzbar and self.enhance:
            try:
                # Get original crop before enhancement
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
                decoded = self.pyzbar.decode(gray)
                for obj in decoded:
                    data = obj.data.decode('utf-8')
                    rect = obj.rect
                    cx = int(rect.left / scale + rect.width / scale / 2 + offset_x)
                    cy = int(rect.top / scale + rect.height / scale / 2 + offset_y)
                    pts = np.array([
                        [rect.left / scale + offset_x, rect.top / scale + offset_y],
                        [(rect.left + rect.width) / scale + offset_x, rect.top / scale + offset_y],
                        [(rect.left + rect.width) / scale + offset_x, (rect.top + rect.height) / scale + offset_y],
                        [rect.left / scale + offset_x, (rect.top + rect.height) / scale + offset_y]
                    ], dtype=np.int32)
                    results.append({
                        'data': data,
                        'center': (cx, cy),
                        'points': pts
                    })
                if results:
                    return results
            except:
                pass

        # Method 3: OpenCV QR detector (fallback)
        try:
            qr = cv2.QRCodeDetector()
            data, points, _ = qr.detectAndDecode(crop)
            if data and points is not None:
                pts = points[0]
                cx = int(np.mean(pts[:, 0]) / scale + offset_x)
                cy = int(np.mean(pts[:, 1]) / scale + offset_y)
                pts = (pts / scale + [offset_x, offset_y]).astype(np.int32)
                results.append({
                    'data': data,
                    'center': (cx, cy),
                    'points': pts
                })
        except:
            pass

        return results

    def shutdown(self):
        """Shutdown thread pool."""
        self.executor.shutdown(wait=False)

"""
QR Detector V2: Multi-threaded scanning
Scans QR codes in parallel using thread pool.
"""

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

class QRDetectorV2:
    """Multi-threaded QR detection - parallel scanning of box regions."""

    def __init__(self, padding=20, min_crop_size=100, max_workers=4):
        """
        Args:
            padding: Extra pixels around box
            min_crop_size: Minimum crop size
            max_workers: Number of parallel threads
        """
        self.padding = padding
        self.min_crop_size = min_crop_size
        self.max_workers = max_workers
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
        Detect QR codes in parallel across box regions.

        Args:
            frame: Full camera frame
            box_detections: List of box dicts

        Returns:
            List of QR detections
        """
        if not box_detections:
            return []

        h, w = frame.shape[:2]
        qr_codes = []

        # Prepare crops
        crops = []
        for det in box_detections:
            x1 = max(0, det['x1'] - self.padding)
            y1 = max(0, det['y1'] - self.padding)
            x2 = min(w, det['x2'] + self.padding)
            y2 = min(h, det['y2'] + self.padding)

            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] > 10 and crop.shape[1] > 10:
                crops.append((crop, x1, y1, det))

        # Submit all crops to thread pool
        futures = {}
        for crop, x1, y1, det in crops:
            future = self.executor.submit(self._process_crop, crop, x1, y1)
            futures[future] = det

        # Collect results
        for future in as_completed(futures):
            det = futures[future]
            try:
                results = future.result()
                for qr in results:
                    qr['box_detection'] = det
                    qr_codes.append(qr)
            except:
                pass

        return qr_codes

    def _process_crop(self, crop, offset_x, offset_y):
        """Process single crop in thread."""
        crop_h, crop_w = crop.shape[:2]

        # Upscale if needed
        scale = 1.0
        if crop_w < self.min_crop_size or crop_h < self.min_crop_size:
            scale = self.min_crop_size / min(crop_w, crop_h)
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_LINEAR)

        results = []

        # Try pyzbar
        if self.use_pyzbar:
            try:
                decoded = self.pyzbar.decode(crop)
                for obj in decoded:
                    data = obj.data.decode('utf-8')
                    rect = obj.rect
                    pts = np.array([
                        [rect.left, rect.top],
                        [rect.left + rect.width, rect.top],
                        [rect.left + rect.width, rect.top + rect.height],
                        [rect.left, rect.top + rect.height]
                    ], dtype=np.float32)

                    # Adjust back to full frame coords
                    cx = int(rect.left / scale + rect.width / scale / 2 + offset_x)
                    cy = int(rect.top / scale + rect.height / scale / 2 + offset_y)
                    pts = (pts / scale + [offset_x, offset_y]).astype(np.int32)

                    results.append({
                        'data': data,
                        'center': (cx, cy),
                        'points': pts
                    })
                if results:
                    return results
            except:
                pass

        # Fallback: OpenCV
        try:
            qr_detector = cv2.QRCodeDetector()
            retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(crop)
            if retval and points is not None:
                for i, data in enumerate(decoded_info):
                    if data:
                        pts = points[i].astype(np.float32)
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

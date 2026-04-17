"""
QR Detector V1: Crop-based scanning
Scans QR codes only within detected box regions instead of full frame.
"""

import cv2
import numpy as np

class QRDetectorV1:
    """Crop-based QR detection - scan only inside box regions."""

    def __init__(self, padding=20, min_crop_size=100):
        """
        Args:
            padding: Extra pixels around box to include in crop
            min_crop_size: Minimum crop size (upscale if smaller)
        """
        self.padding = padding
        self.min_crop_size = min_crop_size
        self.qr_detector = cv2.QRCodeDetector()

        # Try pyzbar (better accuracy)
        self.use_pyzbar = False
        try:
            from pyzbar import pyzbar
            self.pyzbar = pyzbar
            self.use_pyzbar = True
        except ImportError:
            pass

    def detect_in_frame(self, frame, box_detections):
        """
        Detect QR codes only within box regions.

        Args:
            frame: Full camera frame
            box_detections: List of box dicts with x1,y1,x2,y2

        Returns:
            List of QR detections with data, center, points
        """
        qr_codes = []
        h, w = frame.shape[:2]

        for det in box_detections:
            # Get box region with padding
            x1 = max(0, det['x1'] - self.padding)
            y1 = max(0, det['y1'] - self.padding)
            x2 = min(w, det['x2'] + self.padding)
            y2 = min(h, det['y2'] + self.padding)

            # Crop region
            crop = frame[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]

            if crop_h < 10 or crop_w < 10:
                continue

            # Upscale small crops for better QR detection
            scale = 1.0
            if crop_w < self.min_crop_size or crop_h < self.min_crop_size:
                scale = self.min_crop_size / min(crop_w, crop_h)
                crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_LINEAR)

            # Detect QR in crop
            qr_result = self._detect_qr(crop)

            # Adjust coordinates back to full frame
            for qr in qr_result:
                # Scale points back
                qr['center'] = (
                    int(qr['center'][0] / scale + x1),
                    int(qr['center'][1] / scale + y1)
                )
                qr['points'] = (qr['points'] / scale + [x1, y1]).astype(np.int32)
                qr['box_detection'] = det  # Link to source box
                qr_codes.append(qr)

        return qr_codes

    def _detect_qr(self, image):
        """Detect QR codes in image."""
        results = []

        # Try pyzbar first
        if self.use_pyzbar:
            try:
                decoded = self.pyzbar.decode(image)
                for obj in decoded:
                    data = obj.data.decode('utf-8')
                    rect = obj.rect
                    pts = np.array([
                        [rect.left, rect.top],
                        [rect.left + rect.width, rect.top],
                        [rect.left + rect.width, rect.top + rect.height],
                        [rect.left, rect.top + rect.height]
                    ], dtype=np.float32)
                    cx = rect.left + rect.width // 2
                    cy = rect.top + rect.height // 2
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
            retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(image)
            if retval and points is not None:
                for i, data in enumerate(decoded_info):
                    if data:
                        pts = points[i].astype(np.float32)
                        cx = int(np.mean(pts[:, 0]))
                        cy = int(np.mean(pts[:, 1]))
                        results.append({
                            'data': data,
                            'center': (cx, cy),
                            'points': pts
                        })
        except:
            pass

        return results

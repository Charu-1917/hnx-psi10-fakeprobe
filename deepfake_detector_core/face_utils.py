"""
Face Detection, Cropping, and Tracking Utilities for FakeProbe-X.
Employs MTCNN / OpenCV face localization with margin-padded face extraction and temporal consistency tracking.
"""

import cv2
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List


class FaceDetector:
    """Robust face detection and spatial cropping engine."""

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._mtcnn = None
        self._init_mtcnn()

    def _init_mtcnn(self):
        try:
            from facenet_pytorch import MTCNN
            self._mtcnn = MTCNN(
                keep_all=True,
                select_largest=True,
                post_process=False,
                device=self.device
            )
        except Exception:
            self._mtcnn = None

    def detect_faces(self, image_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[float]]:
        """
        Detect face bounding boxes and detection confidence scores.
        Returns:
            boxes: list of [x1, y1, x2, y2]
            probs: list of confidence floats
        """
        if self._mtcnn is not None:
            try:
                pil_img = Image.fromarray(image_rgb)
                boxes, probs = self._mtcnn.detect(pil_img)
                if boxes is not None and len(boxes) > 0:
                    valid_boxes = []
                    valid_probs = []
                    for box, prob in zip(boxes, probs):
                        if prob is not None and prob >= 0.70:
                            valid_boxes.append(box.astype(int))
                            valid_probs.append(float(prob))
                    return valid_boxes, valid_probs
            except Exception:
                pass

        return [], []

    def crop_primary_face(
        self,
        image_rgb: np.ndarray,
        target_size: int = 224,
        margin: float = 0.25
    ) -> Tuple[np.ndarray, Optional[list], float]:
        """
        Detect and crop the primary face with contextual boundary margin.
        If no face is detected, gracefully center-crops a square to prevent geometric distortion.
        Returns:
            cropped_face_rgb: (target_size, target_size, 3)
            box: [x1, y1, x2, y2] or None
            face_confidence: float (1.0 for detected face, 0.0 for fallback)
        """
        h, w, _ = image_rgb.shape
        boxes, probs = self.detect_faces(image_rgb)

        if len(boxes) > 0:
            box = boxes[0]
            conf = probs[0]
            x1, y1, x2, y2 = box

            bw = x2 - x1
            bh = y2 - y1
            # Expand with margin
            pad_x = int(bw * margin)
            pad_y = int(bh * margin)

            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            crop = image_rgb[cy1:cy2, cx1:cx2]
            if crop.size > 0 and crop.shape[0] >= 16 and crop.shape[1] >= 16:
                resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
                return resized, [cx1, cy1, cx2, cy2], conf

        # Fallback: Square center-crop with aspect ratio preservation
        side = min(h, w)
        sy = (h - side) // 2
        sx = (w - side) // 2
        center_crop = image_rgb[sy:sy + side, sx:sx + side]
        resized = cv2.resize(center_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return resized, None, 0.0

    def track_faces_across_frames(
        self,
        frames: List[np.ndarray]
    ) -> dict:
        """
        Analyze face spatial stability and presence across sequential video frames.
        Returns face presence rate, bounding box jitter, and scale variation.
        """
        detected_boxes = []
        confs = []

        for frame in frames:
            boxes, probs = self.detect_faces(frame)
            if len(boxes) > 0:
                detected_boxes.append(boxes[0])
                confs.append(probs[0])
            else:
                detected_boxes.append(None)
                confs.append(0.0)

        num_frames = len(frames)
        valid_count = sum(1 for b in detected_boxes if b is not None)
        detection_rate = valid_count / num_frames if num_frames > 0 else 0.0

        centers = []
        scales = []
        for box in detected_boxes:
            if box is not None:
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                area = max(1, (x2 - x1) * (y2 - y1))
                centers.append((cx, cy))
                scales.append(np.sqrt(area))

        if len(centers) >= 2:
            center_diffs = [
                np.sqrt((centers[i][0] - centers[i-1][0])**2 + (centers[i][1] - centers[i-1][1])**2)
                for i in range(1, len(centers))
            ]
            jitter_score = float(np.mean(center_diffs))
            scale_variation = float(np.std(scales) / (np.mean(scales) + 1e-6))
        else:
            jitter_score = 0.0
            scale_variation = 0.0

        # Spatial consistency score: high presence, low abrupt jitter
        presence_score = detection_rate
        stability_score = max(0.0, 1.0 - (jitter_score / 100.0))
        face_consistency_score = float(0.6 * presence_score + 0.4 * stability_score)

        return {
            "detection_rate": detection_rate,
            "detected_count": valid_count,
            "total_frames": num_frames,
            "jitter_score": jitter_score,
            "scale_variation": scale_variation,
            "face_consistency_score": face_consistency_score,
            "boxes_per_frame": detected_boxes,
        }

import math

class SyncEvidenceCalibrator:
    """
    MVP heuristic calibrator for SyncNet evidence.
    Converts raw offset/confidence into a continuous [0, 1] desync score.
    
    IMPORTANT: The output sync_desync_score is NOT a scientifically validated probability.
    It is merely a heuristic translation of offset frames for use in MVP fusion.
    """
    def __init__(
        self,
        offset_tolerance_frames: int = 2,
        reliable_confidence_threshold: float = 3.0,
        max_offset_penalty: int = 15
    ):
        # 1. offset_tolerance_frames: small offsets (e.g. 1-2 frames) yield low desync scores.
        self.offset_tolerance_frames = offset_tolerance_frames
        
        # 2. reliable_confidence_threshold: minimum confidence to trust the offset.
        self.reliable_confidence_threshold = reliable_confidence_threshold
        
        # 3. max_offset_penalty: offsets >= this value yield a maximum 1.0 desync score.
        self.max_offset_penalty = max_offset_penalty

    def calibrate(self, offset_frames: int, confidence: float, fps: float = None) -> dict:
        """
        Calibrates the raw SyncNet evidence.
        """
        # 1. Validate inputs
        if offset_frames is None or not isinstance(offset_frames, (int, float)) or not math.isfinite(offset_frames):
            raise ValueError("offset_frames must be a finite number")
            
        if confidence is None or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
            raise ValueError("confidence must be a valid, finite number")
            
        if fps is not None and fps <= 0:
            raise ValueError("fps must be > 0")

        # 2. Evaluate reliability
        reliable = confidence >= self.reliable_confidence_threshold
        
        # 3. Heuristic calculation
        abs_offset = abs(offset_frames)
        
        if not reliable:
            # Low confidence -> We cannot trust the offset.
            # Output None to indicate UNAVAILABLE EVIDENCE.
            sync_desync_score = None
        else:
            if abs_offset <= self.offset_tolerance_frames:
                # Small offset -> mapped linearly from 0.0 to 0.2
                if self.offset_tolerance_frames > 0:
                    sync_desync_score = (abs_offset / self.offset_tolerance_frames) * 0.2
                else:
                    sync_desync_score = 0.0
            else:
                # Larger offset -> progressively higher desync score from 0.2 to 1.0
                clamped_offset = min(abs_offset, self.max_offset_penalty)
                penalty_range = self.max_offset_penalty - self.offset_tolerance_frames
                if penalty_range > 0:
                    progress = (clamped_offset - self.offset_tolerance_frames) / penalty_range
                    sync_desync_score = 0.2 + (progress * 0.8)
                else:
                    sync_desync_score = 1.0

        # Ensure bounds [0, 1] if not None
        if sync_desync_score is not None:
            sync_desync_score = max(0.0, min(1.0, sync_desync_score))
        
        # 4. Optional ms calculation
        offset_ms = None
        if fps is not None:
            offset_ms = (offset_frames / fps) * 1000.0
            
        return {
            "sync_desync_score": float(sync_desync_score) if sync_desync_score is not None else None,
            "offset_frames": int(offset_frames),
            "offset_ms": float(offset_ms) if offset_ms is not None else None,
            "confidence": float(confidence),
            "reliable": bool(reliable),
            "evidence": [
                f"Offset: {int(offset_frames)} frames",
                f"Confidence: {confidence:.3f} (Threshold: {self.reliable_confidence_threshold})",
                f"Reliable: {bool(reliable)}"
            ]
        }

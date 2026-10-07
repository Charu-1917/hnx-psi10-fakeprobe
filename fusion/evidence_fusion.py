from typing import Optional, Dict, Any

class SyncEvidenceCalibrator:
    """
    MVP heuristic interface for calibrating SyncNet raw offset/confidence
    into a continuous [0, 1] desynchronization probability.
    """
    def __init__(self, offset_tolerance: int = 1, confidence_threshold: float = 3.0):
        self.offset_tolerance = offset_tolerance
        self.confidence_threshold = confidence_threshold

    def calibrate(self, offset_frames: int, confidence: float) -> float:
        """
        Converts raw SyncNet evidence into a manipulation probability [0, 1].
        This is an MVP heuristic and NOT scientifically validated.
        """
        if confidence < self.confidence_threshold:
            # Low confidence might mean dubbed or fully unsynchronized speech.
            # We map this to 1.0 (highly likely manipulated) for MVP.
            return 1.0
        
        if abs(offset_frames) > self.offset_tolerance:
            # High confidence but outside tolerance -> Desynchronized (manipulated).
            return 1.0
            
        # High confidence, within offset tolerance -> Synchronized (authentic).
        return 0.0

class EvidenceFusionEngine:
    """
    Decision-level evidence fusion engine.
    Combines independent modality manipulation probabilities into a final score.
    """
    def __init__(
        self, 
        w_visual: float = 0.40, 
        w_audio: float = 0.30, 
        w_sync: float = 0.30,
        threshold_authentic: float = 0.30,
        threshold_manipulated: float = 0.70,
        sync_calibrator: Optional[SyncEvidenceCalibrator] = None
    ):
        self.w_visual = w_visual
        self.w_audio = w_audio
        self.w_sync = w_sync
        self.threshold_authentic = threshold_authentic
        self.threshold_manipulated = threshold_manipulated
        self.sync_calibrator = sync_calibrator or SyncEvidenceCalibrator()

    def fuse(
        self,
        visual_score: Optional[float] = None,
        audio_score: Optional[float] = None,
        sync_offset_frames: Optional[int] = None,
        sync_confidence: Optional[float] = None,
        fps: float = 25.0
    ) -> Dict[str, Any]:
        """
        Fuses multimodal forensic evidence into a single manipulation score.
        Missing modalities are handled by renormalizing the weights over available inputs.
        """
        # Bounds checks
        if visual_score is not None and not (0.0 <= visual_score <= 1.0):
            raise ValueError(f"visual_score {visual_score} outside [0, 1]")
        if audio_score is not None and not (0.0 <= audio_score <= 1.0):
            raise ValueError(f"audio_score {audio_score} outside [0, 1]")

        sync_desync_score = None
        sync_evidence = None

        if sync_offset_frames is not None and sync_confidence is not None:
            sync_desync_score = self.sync_calibrator.calibrate(sync_offset_frames, sync_confidence)
            sync_evidence = {
                "offset_frames": sync_offset_frames,
                "offset_ms": (sync_offset_frames / fps) * 1000.0,
                "confidence": sync_confidence,
                "reliable": sync_confidence >= self.sync_calibrator.confidence_threshold
            }

        available_weights = 0.0
        total_score = 0.0

        if visual_score is not None:
            available_weights += self.w_visual
            total_score += visual_score * self.w_visual

        if audio_score is not None:
            available_weights += self.w_audio
            total_score += audio_score * self.w_audio

        if sync_desync_score is not None:
            available_weights += self.w_sync
            total_score += sync_desync_score * self.w_sync

        if available_weights == 0.0:
            raise ValueError("No usable modality exists. Provide at least one valid score.")

        manipulation_score = total_score / available_weights

        epsilon = 1e-9
        if manipulation_score < self.threshold_authentic - epsilon:
            decision = "LIKELY_AUTHENTIC"
        elif manipulation_score >= self.threshold_manipulated - epsilon:
            decision = "LIKELY_MANIPULATED"
        else:
            decision = "UNCERTAIN"

        return {
            "manipulation_score": float(manipulation_score),
            "decision": decision,
            "modalities": {
                "visual_score": float(visual_score) if visual_score is not None else None,
                "audio_score": float(audio_score) if audio_score is not None else None,
                "sync_desync_score": float(sync_desync_score) if sync_desync_score is not None else None
            },
            "sync_evidence": sync_evidence
        }

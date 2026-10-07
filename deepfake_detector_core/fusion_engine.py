"""
FakeProbe-X Adaptive Evidence Fusion Engine and Three-Way Decision System.
Dynamically weights multi-modal forensic signals based on real-time reliability estimates,
evaluates cross-modal consistency, and renders evidence-backed REAL / FAKE / UNCERTAIN verdicts.
"""

import numpy as np
from typing import List, Optional
from .types import (
    DetectorResult,
    DecisionVerdict,
    ReliabilityLevel,
    ForensicEvidence,
    UnifiedForensicReport,
    QualityResult,
    QualityLevel
)
from .reliability import ReliabilityEstimator
from configs.forensic_config import DEFAULT_CONFIG


class AdaptiveEvidenceFusionEngine:
    """Central FakeProbe-X reliability-aware fusion architecture."""

    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config
        self.reliability_estimator = ReliabilityEstimator(config=config)

    def fuse_image_evidence(
        self,
        image_result: DetectorResult,
        sample_name: str = "image_sample"
    ) -> UnifiedForensicReport:
        """
        Fuse detector prediction, frequency anomalies, and input quality into an explainable image report.
        """
        evidence_list = []
        reasons = []

        # 1. Main Neural Detector Evidence
        img_rel = image_result.reliability
        img_prob = image_result.probability_fake
        w_img = self.config.fusion_weight_image_detector

        evidence_list.append(ForensicEvidence(
            name="CoAtNet-0 5-Channel Classifier",
            modality="image",
            score=img_prob,
            weight=w_img,
            reliability=img_rel,
            description="5-channel spatial and compression residual classifier",
            details={"raw_logit": image_result.raw_score}
        ))

        # 2. Frequency Forensic Evidence
        freq_anomaly = image_result.evidence.get("frequency_anomaly_score", 0.5)
        w_freq = self.config.fusion_weight_frequency_forensic
        evidence_list.append(ForensicEvidence(
            name="Fourier Frequency Domain Spectrum",
            modality="frequency",
            score=freq_anomaly,
            weight=w_freq,
            reliability=0.75 if image_result.quality.is_acceptable else 0.40,
            description="Radial power spectrum decay and high-frequency energy ratio analysis",
            details={
                "spectral_slope": image_result.evidence.get("spectral_slope"),
                "high_frequency_ratio": image_result.evidence.get("high_frequency_ratio")
            }
        ))

        # 3. Weighted Fusion Calculation
        total_weight = 0.0
        weighted_sum = 0.0
        for ev in evidence_list:
            effective_w = ev.weight * ev.reliability
            weighted_sum += ev.score * effective_w
            total_weight += effective_w

        if total_weight > 0:
            fused_fake_prob = float(weighted_sum / total_weight)
        else:
            fused_fake_prob = img_prob

        fused_real_prob = float(1.0 - fused_fake_prob)

        # 4. Reliability & Uncertainty Estimation
        rel_score, rel_level, uncertainty, unc_reasons = self.reliability_estimator.estimate_reliability(
            quality=image_result.quality,
            probabilities=[img_prob, freq_anomaly],
            agreement_score=1.0 - abs(img_prob - freq_anomaly),
            coverage_score=1.0 if image_result.evidence.get("face_detected") else 0.70
        )

        # 5. Three-Way Decision Logic
        if not image_result.quality.is_acceptable or image_result.quality.quality_level == QualityLevel.LOW:
            final_verdict = DecisionVerdict.UNCERTAIN
            reasons.append(f"Low input image quality: {', '.join(image_result.quality.issues)}")
        elif uncertainty >= self.config.high_uncertainty_threshold or rel_score < self.config.min_reliability_threshold:
            final_verdict = DecisionVerdict.UNCERTAIN
            reasons.extend(unc_reasons)
        elif self.config.uncertainty_real_boundary <= fused_fake_prob <= self.config.uncertainty_fake_boundary:
            final_verdict = DecisionVerdict.UNCERTAIN
            reasons.append(f"Ambiguous artifact score ({fused_fake_prob:.3f}) inside uncertain deadband [{self.config.uncertainty_real_boundary:.2f}, {self.config.uncertainty_fake_boundary:.2f}]")
        elif fused_fake_prob > self.config.uncertainty_fake_boundary:
            final_verdict = DecisionVerdict.FAKE
            reasons.append(f"Elevated spatial/compression anomaly score ({fused_fake_prob*100:.1f}%) detected by CoAtNet-5ch")
            if freq_anomaly > 0.60:
                reasons.append(f"Unnatural high-frequency spectral roll-off ({image_result.evidence.get('spectral_slope', 0):.2f})")
        else:
            final_verdict = DecisionVerdict.REAL
            reasons.append(f"Consistent natural photographic statistics (Real probability: {fused_real_prob*100:.1f}%)")

        confidence = float(abs(fused_fake_prob - 0.50) * 2.0)

        # Summary text
        summary = (
            f"Verdict: {final_verdict.value} | Confidence: {confidence*100:.1f}% | "
            f"Reliability: {rel_level.value} ({rel_score:.2f}) | Quality: {image_result.quality.quality_level.value}"
        )

        return UnifiedForensicReport(
            sample_name=sample_name,
            modality="image",
            final_verdict=final_verdict,
            final_fake_probability=fused_fake_prob,
            final_real_probability=fused_real_prob,
            confidence=confidence,
            reliability=rel_score,
            reliability_level=rel_level,
            uncertainty_score=uncertainty,
            quality=image_result.quality,
            detector_results=[image_result],
            evidence_list=evidence_list,
            agreement_score=1.0 - abs(img_prob - freq_anomaly),
            reasons=reasons,
            summary_text=summary,
            visual_artifacts={
                "ela_map": image_result.evidence.get("ela_map"),
                "fft_map": image_result.evidence.get("fft_map"),
                "face_box": image_result.evidence.get("face_box"),
                "processed_crop_rgb": image_result.evidence.get("processed_crop_rgb"),
            }
        )

    def fuse_audio_evidence(
        self,
        audio_result: DetectorResult,
        sample_name: str = "audio_sample"
    ) -> UnifiedForensicReport:
        """
        Fuse acoustic spoofing probability, SNR, and signal duration into an explainable audio report.
        """
        evidence_list = []
        reasons = []

        aud_prob = audio_result.probability_fake
        aud_rel = audio_result.reliability

        evidence_list.append(ForensicEvidence(
            name="Whisper-Base + XGBoost Acoustic Classifier",
            modality="audio",
            score=aud_prob,
            weight=1.0,
            reliability=aud_rel,
            description="512-dimensional Whisper encoder embedding classifier",
            details={"raw_margin": audio_result.raw_score}
        ))

        rel_score, rel_level, uncertainty, unc_reasons = self.reliability_estimator.estimate_reliability(
            quality=audio_result.quality,
            probabilities=[aud_prob],
            agreement_score=1.0,
            coverage_score=1.0
        )

        fused_fake_prob = aud_prob
        fused_real_prob = 1.0 - aud_prob

        # Decision
        if not audio_result.quality.is_acceptable or audio_result.quality.quality_level == QualityLevel.LOW:
            final_verdict = DecisionVerdict.UNCERTAIN
            reasons.append(f"Acoustic signal too degraded: {', '.join(audio_result.quality.issues)}")
        elif uncertainty >= self.config.high_uncertainty_threshold or rel_score < self.config.min_reliability_threshold:
            final_verdict = DecisionVerdict.UNCERTAIN
            reasons.extend(unc_reasons)
        elif self.config.uncertainty_real_boundary <= fused_fake_prob <= self.config.uncertainty_fake_boundary:
            final_verdict = DecisionVerdict.UNCERTAIN
            reasons.append(f"Acoustic spoof score ({fused_fake_prob:.3f}) inside uncertain deadband")
        elif fused_fake_prob > self.config.uncertainty_fake_boundary:
            final_verdict = DecisionVerdict.FAKE
            reasons.append(f"Synthetic speech patterns detected (Spoof probability: {fused_fake_prob*100:.1f}%)")
        else:
            final_verdict = DecisionVerdict.REAL
            reasons.append(f"Natural human vocal tract acoustics (Bonafide probability: {fused_real_prob*100:.1f}%)")

        confidence = float(abs(fused_fake_prob - 0.50) * 2.0)
        summary = (
            f"Verdict: {final_verdict.value} | Confidence: {confidence*100:.1f}% | "
            f"Reliability: {rel_level.value} ({rel_score:.2f}) | Quality: {audio_result.quality.quality_level.value}"
        )

        return UnifiedForensicReport(
            sample_name=sample_name,
            modality="audio",
            final_verdict=final_verdict,
            final_fake_probability=fused_fake_prob,
            final_real_probability=fused_real_prob,
            confidence=confidence,
            reliability=rel_score,
            reliability_level=rel_level,
            uncertainty_score=uncertainty,
            quality=audio_result.quality,
            detector_results=[audio_result],
            evidence_list=evidence_list,
            agreement_score=1.0,
            reasons=reasons,
            summary_text=summary,
        )

    def fuse_video_evidence(
        self,
        video_result: DetectorResult,
        sample_name: str = "video_sample"
    ) -> UnifiedForensicReport:
        """
        Fuse multimodal visual pooled features, acoustic classifier, temporal timeline, and face tracking.
        """
        evidence_list = []
        reasons = []

        vid_prob = video_result.probability_fake
        vid_rel = video_result.reliability
        has_audio = video_result.evidence.get("has_audio", False)
        face_track = video_result.evidence.get("face_tracking", {})
        suspicious_frames = video_result.evidence.get("suspicious_frames", [])
        frame_probs = video_result.evidence.get("frame_fake_probabilities", [])

        # 1. Multimodal Model Evidence
        evidence_list.append(ForensicEvidence(
            name="Multimodal Fusion Model (Visual+Acoustic)",
            modality="multimodal",
            score=vid_prob,
            weight=0.50,
            reliability=vid_rel,
            description="Statistical pooling of CoAtNet visual frames + Whisper acoustic embedding with XGBoost",
            details={"raw_fused_score": video_result.raw_score}
        ))

        # 2. Temporal Consistency Evidence
        if len(frame_probs) > 0:
            temporal_mean = float(np.mean(frame_probs))
            temporal_std = float(np.std(frame_probs))
            evidence_list.append(ForensicEvidence(
                name="Frame-by-Frame Temporal Consistency",
                modality="temporal",
                score=temporal_mean,
                weight=0.30,
                reliability=0.85,
                description=f"Sequential frame analysis across {len(frame_probs)} frames (std deviation: {temporal_std:.3f})",
                details={"suspicious_frame_count": len(suspicious_frames)}
            ))
        else:
            temporal_mean = vid_prob

        # 3. Audio Track / Cross-modal Agreement
        audio_res_dict = video_result.evidence.get("audio_result")
        if has_audio and audio_res_dict is not None:
            aud_prob = audio_res_dict.get("probability_fake", 0.5)
            evidence_list.append(ForensicEvidence(
                name="Isolated Speech Track Classifier",
                modality="audio",
                score=aud_prob,
                weight=0.20,
                reliability=audio_res_dict.get("reliability", 0.7),
                description="Whisper-Base speech spoof verification",
                details={}
            ))
            # Cross-modal agreement
            cross_modal_diff = abs(temporal_mean - aud_prob)
            agreement_score = float(max(0.0, 1.0 - cross_modal_diff))

            if cross_modal_diff > 0.45:
                reasons.append(
                    f"Cross-modal discrepancy: Visual spoof score is {temporal_mean*100:.1f}% while Audio spoof score is {aud_prob*100:.1f}%"
                )
        else:
            agreement_score = 1.0
            reasons.append("Audio track unavailable — relying on visual and temporal stream forensics")

        # 4. Weighted Fusion Calculation
        total_weight = 0.0
        weighted_sum = 0.0
        for ev in evidence_list:
            effective_w = ev.weight * ev.reliability
            weighted_sum += ev.score * effective_w
            total_weight += effective_w

        if total_weight > 0:
            fused_fake_prob = float(weighted_sum / total_weight)
        else:
            fused_fake_prob = vid_prob

        fused_real_prob = float(1.0 - fused_fake_prob)

        # 5. Reliability & Uncertainty
        rel_score, rel_level, uncertainty, unc_reasons = self.reliability_estimator.estimate_reliability(
            quality=video_result.quality,
            probabilities=[vid_prob, temporal_mean],
            agreement_score=agreement_score,
            coverage_score=face_track.get("detection_rate", 0.8)
        )

        # 6. Three-Way Decision Logic
        if not video_result.quality.is_acceptable or video_result.quality.quality_level == QualityLevel.LOW:
            final_verdict = DecisionVerdict.UNCERTAIN
            reasons.append(f"Low video quality: {', '.join(video_result.quality.issues)}")
        elif uncertainty >= self.config.high_uncertainty_threshold or rel_score < self.config.min_reliability_threshold:
            final_verdict = DecisionVerdict.UNCERTAIN
            reasons.extend(unc_reasons)
        elif self.config.uncertainty_real_boundary <= fused_fake_prob <= self.config.uncertainty_fake_boundary:
            final_verdict = DecisionVerdict.UNCERTAIN
            reasons.append(f"Inconclusive video manipulation signature ({fused_fake_prob:.3f}) inside uncertain deadband")
        elif fused_fake_prob >= self.config.video_decision_threshold:
            final_verdict = DecisionVerdict.FAKE
            reasons.append(f"Video deepfake artifacts confirmed across {len(suspicious_frames)} temporal frames")
        else:
            final_verdict = DecisionVerdict.REAL
            reasons.append(f"Consistent temporal and facial dynamics (Real probability: {fused_real_prob*100:.1f}%)")

        confidence = float(abs(fused_fake_prob - 0.50) * 2.0)
        summary = (
            f"Verdict: {final_verdict.value} | Confidence: {confidence*100:.1f}% | "
            f"Reliability: {rel_level.value} ({rel_score:.2f}) | Quality: {video_result.quality.quality_level.value}"
        )

        return UnifiedForensicReport(
            sample_name=sample_name,
            modality="video",
            final_verdict=final_verdict,
            final_fake_probability=fused_fake_prob,
            final_real_probability=fused_real_prob,
            confidence=confidence,
            reliability=rel_score,
            reliability_level=rel_level,
            uncertainty_score=uncertainty,
            quality=video_result.quality,
            detector_results=[video_result],
            evidence_list=evidence_list,
            agreement_score=agreement_score,
            reasons=reasons,
            summary_text=summary,
            visual_artifacts={
                "suspicious_frames": suspicious_frames,
                "frame_timestamps": video_result.evidence.get("frame_timestamps"),
                "frame_fake_probabilities": frame_probs,
                "sampled_frames_rgb": video_result.evidence.get("sampled_frames_rgb"),
                "processed_face_frames_rgb": video_result.evidence.get("processed_face_frames_rgb"),
            }
        )

"""
Reliability Estimation and Uncertainty Quantification for FakeProbe-X.
Quantifies input quality, detector agreement, coverage, and epistemic uncertainty to prevent overconfident false classifications.
"""

import numpy as np
from typing import List, Tuple
from .types import ReliabilityLevel, QualityResult
from configs.forensic_config import DEFAULT_CONFIG


class ReliabilityEstimator:
    """Estimates the forensic reliability and uncertainty of detector predictions."""

    def __init__(self, config=DEFAULT_CONFIG):
        self.config = config

    def estimate_reliability(
        self,
        quality: QualityResult,
        probabilities: List[float],
        agreement_score: float = 1.0,
        coverage_score: float = 1.0
    ) -> Tuple[float, ReliabilityLevel, float, List[str]]:
        """
        Calculate composite reliability and heuristic uncertainty.
        Returns:
            reliability: float in [0.0, 1.0]
            reliability_level: HIGH, MEDIUM, LOW
            uncertainty_score: float in [0.0, 1.0]
            uncertainty_reasons: list of explanatory strings
        """
        uncertainty_reasons = []

        # 1. Quality Component
        quality_score = float(np.clip(quality.quality_score, 0.0, 1.0))
        if quality_score < 0.45:
            uncertainty_reasons.append(f"Suboptimal input quality ({quality.quality_level.value})")

        # 2. Confidence / Boundary Distance Component
        if len(probabilities) > 0:
            avg_prob = float(np.mean(probabilities))
            # Distance from 0.5 boundary scaled to [0, 1]
            boundary_dist = abs(avg_prob - 0.50) * 2.0
            confidence_score = float(np.clip(boundary_dist, 0.0, 1.0))

            if self.config.uncertainty_real_boundary <= avg_prob <= self.config.uncertainty_fake_boundary:
                uncertainty_reasons.append(
                    f"Prediction probability ({avg_prob:.3f}) falls in the ambiguous boundary zone [{self.config.uncertainty_real_boundary:.2f}, {self.config.uncertainty_fake_boundary:.2f}]"
                )
        else:
            confidence_score = 0.0
            uncertainty_reasons.append("No valid detector probability available")

        # 3. Agreement Component
        agreement = float(np.clip(agreement_score, 0.0, 1.0))
        if agreement < 0.65:
            uncertainty_reasons.append(f"High disagreement between forensic detectors (agreement: {agreement*100:.1f}%)")

        # 4. Coverage Component (e.g. face presence rate, audio presence)
        coverage = float(np.clip(coverage_score, 0.0, 1.0))
        if coverage < 0.50:
            uncertainty_reasons.append(f"Limited modality/facial coverage ({coverage*100:.1f}%)")

        # Composite Reliability Formula
        wq = self.config.quality_weight
        wc = self.config.confidence_weight
        wa = self.config.agreement_weight
        wcov = self.config.coverage_weight

        reliability = (
            wq * quality_score +
            wc * confidence_score +
            wa * agreement +
            wcov * coverage
        )
        reliability = float(np.clip(reliability, 0.0, 1.0))

        # Heuristic Uncertainty: Inverse of confidence, quality, and agreement
        score_uncertainty = 1.0 - confidence_score
        quality_uncertainty = 1.0 - quality_score
        disagreement_uncertainty = 1.0 - agreement
        coverage_uncertainty = 1.0 - coverage

        uncertainty_score = (
            0.40 * score_uncertainty +
            0.30 * quality_uncertainty +
            0.20 * disagreement_uncertainty +
            0.10 * coverage_uncertainty
        )
        uncertainty_score = float(np.clip(uncertainty_score, 0.0, 1.0))

        # Categorize Level
        if reliability >= 0.70 and uncertainty_score <= 0.35:
            reliability_level = ReliabilityLevel.HIGH
        elif reliability >= 0.40 and uncertainty_score <= 0.60:
            reliability_level = ReliabilityLevel.MEDIUM
        else:
            reliability_level = ReliabilityLevel.LOW

        return reliability, reliability_level, uncertainty_score, uncertainty_reasons

    @staticmethod
    def calculate_detector_agreement(fake_probabilities: List[float]) -> float:
        """
        Calculate consensus score across multiple detector probabilities.
        Returns 1.0 if all agree (all high fake or all low fake), decreasing to 0.0 for split votes.
        """
        if len(fake_probabilities) <= 1:
            return 1.0

        probs = np.array(fake_probabilities)
        # Binarize at 0.5
        preds = (probs >= 0.5).astype(int)
        majority_count = max(np.sum(preds == 1), np.sum(preds == 0))
        agreement = majority_count / len(preds)

        # Penalize when probabilities are polar opposites (e.g., 0.95 vs 0.05)
        spread = float(np.max(probs) - np.min(probs))
        adjusted_agreement = agreement * max(0.2, 1.0 - (spread / 1.5))
        return float(np.clip(adjusted_agreement, 0.0, 1.0))

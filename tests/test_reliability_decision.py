"""
Unit Tests for Reliability Estimator, Adaptive Fusion, and Three-Way Decision Engine.
"""

import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepfake_detector_core import (
    ReliabilityEstimator,
    AdaptiveEvidenceFusionEngine,
    DecisionVerdict,
    QualityResult,
    QualityLevel,
    DetectorResult,
)


def test_reliability_calculation_bounds():
    re = ReliabilityEstimator()
    quality = QualityResult(
        modality="image",
        quality_score=0.90,
        quality_level=QualityLevel.GOOD,
        is_acceptable=True,
    )
    rel, level, unc, reasons = re.estimate_reliability(
        quality=quality,
        probabilities=[0.95],
        agreement_score=1.0,
        coverage_score=1.0
    )
    assert 0.0 <= rel <= 1.0
    assert 0.0 <= unc <= 1.0
    assert level.value in ["HIGH", "MEDIUM", "LOW"]


def test_uncertainty_trigger_on_borderline_probabilities():
    re = ReliabilityEstimator()
    quality = QualityResult(
        modality="image",
        quality_score=0.85,
        quality_level=QualityLevel.GOOD,
        is_acceptable=True,
    )
    # Borderline 0.51 probability
    rel, level, unc, reasons = re.estimate_reliability(
        quality=quality,
        probabilities=[0.51],
        agreement_score=1.0,
        coverage_score=1.0
    )
    assert any("ambiguous" in r.lower() or "boundary" in r.lower() for r in reasons)


def test_three_way_verdict_decision():
    fusion = AdaptiveEvidenceFusionEngine()
    quality_good = QualityResult(
        modality="image",
        quality_score=0.90,
        quality_level=QualityLevel.GOOD,
        is_acceptable=True
    )
    det_fake = DetectorResult(
        model="TestModel",
        modality="image",
        prediction=DecisionVerdict.FAKE,
        raw_score=2.5,
        probability_fake=0.92,
        probability_real=0.08,
        confidence=0.84,
        reliability=0.85,
        quality=quality_good,
        evidence={"frequency_anomaly_score": 0.88, "face_detected": True}
    )
    report_fake = fusion.fuse_image_evidence(det_fake, sample_name="test_fake")
    assert report_fake.final_verdict == DecisionVerdict.FAKE

    # Borderline detector result
    det_borderline = DetectorResult(
        model="TestModel",
        modality="image",
        prediction=DecisionVerdict.UNCERTAIN,
        raw_score=0.05,
        probability_fake=0.51,
        probability_real=0.49,
        confidence=0.02,
        reliability=0.60,
        quality=quality_good,
        evidence={"frequency_anomaly_score": 0.50, "face_detected": True}
    )
    report_borderline = fusion.fuse_image_evidence(det_borderline, sample_name="test_borderline")
    assert report_borderline.final_verdict == DecisionVerdict.UNCERTAIN


if __name__ == "__main__":
    test_reliability_calculation_bounds()
    test_uncertainty_trigger_on_borderline_probabilities()
    test_three_way_verdict_decision()
    print("[PASS] All reliability and decision tests passed.")

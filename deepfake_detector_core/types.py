"""
Standardized Data Types and Enums for FakeProbe-X.
Defines common schemas for detector outputs, quality analysis, forensic evidence, and final reports.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np


class DecisionVerdict(str, Enum):
    REAL = "REAL"
    FAKE = "FAKE"
    UNCERTAIN = "UNCERTAIN"


class QualityLevel(str, Enum):
    GOOD = "GOOD"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ReliabilityLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass
class QualityResult:
    modality: str
    quality_score: float                  # [0.0, 1.0]
    quality_level: QualityLevel          # GOOD, MEDIUM, LOW
    is_acceptable: bool                  # True if sufficient for reliable analysis
    issues: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "quality_score": round(self.quality_score, 4),
            "quality_level": self.quality_level.value,
            "is_acceptable": self.is_acceptable,
            "issues": self.issues,
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.metrics.items()},
        }


@dataclass
class ForensicEvidence:
    name: str
    modality: str
    score: float                         # [0.0, 1.0] indicating anomaly / fake likelihood
    weight: float                        # dynamic weight assigned during fusion
    reliability: float                   # [0.0, 1.0] reliability of this specific evidence
    description: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorResult:
    """Standardized return structure for every modality detector."""
    model: str
    modality: str
    prediction: DecisionVerdict          # REAL, FAKE, UNCERTAIN
    raw_score: float                     # Unbounded raw logit or native classifier score
    probability_fake: float              # [0.0, 1.0]
    probability_real: float              # [0.0, 1.0]
    confidence: float                    # [0.0, 1.0] distance from uncertainty boundary
    reliability: float                   # [0.0, 1.0] input quality & coverage weighted
    quality: QualityResult
    evidence: dict[str, Any] = field(default_factory=dict)
    detail_label: str = ""
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "modality": self.modality,
            "prediction": self.prediction.value,
            "detail_label": self.detail_label,
            "raw_score": round(self.raw_score, 4),
            "probability_fake": round(self.probability_fake, 4),
            "probability_real": round(self.probability_real, 4),
            "confidence": round(self.confidence, 4),
            "reliability": round(self.reliability, 4),
            "quality": self.quality.to_dict(),
            "evidence_keys": list(self.evidence.keys()),
            "error_message": self.error_message,
        }


@dataclass
class UnifiedForensicReport:
    """Complete explainable forensic report for a media sample."""
    sample_name: str
    modality: str
    final_verdict: DecisionVerdict       # REAL, FAKE, UNCERTAIN
    final_fake_probability: float        # [0.0, 1.0]
    final_real_probability: float        # [0.0, 1.0]
    confidence: float                    # [0.0, 1.0]
    reliability: float                   # [0.0, 1.0]
    reliability_level: ReliabilityLevel  # HIGH, MEDIUM, LOW
    uncertainty_score: float             # [0.0, 1.0]
    quality: QualityResult
    detector_results: list[DetectorResult] = field(default_factory=list)
    evidence_list: list[ForensicEvidence] = field(default_factory=list)
    agreement_score: float = 1.0         # [0.0, 1.0] degree of consensus across detectors
    reasons: list[str] = field(default_factory=list)
    summary_text: str = ""
    visual_artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sample_name": self.sample_name,
            "modality": self.modality,
            "final_verdict": self.final_verdict.value,
            "final_fake_probability": round(self.final_fake_probability, 4),
            "final_real_probability": round(self.final_real_probability, 4),
            "confidence": round(self.confidence, 4),
            "reliability": round(self.reliability, 4),
            "reliability_level": self.reliability_level.value,
            "uncertainty_score": round(self.uncertainty_score, 4),
            "quality": self.quality.to_dict(),
            "agreement_score": round(self.agreement_score, 4),
            "reasons": self.reasons,
            "summary_text": self.summary_text,
            "detector_summaries": [d.to_dict() for d in self.detector_results],
        }

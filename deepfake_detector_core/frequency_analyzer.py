"""
Frequency-Domain Forensic Analyzer for FakeProbe-X.
Extracts 2D Fourier spectra, radial power distribution, high-frequency energy ratio,
and spectral decay anomalies characteristic of generative synthesis artifacts.
"""

import cv2
import numpy as np
from typing import Tuple


class FrequencyForensicAnalyzer:
    """Extracts frequency-domain forensic evidence from images and video frames."""

    def __init__(self, high_freq_cutoff: float = 0.65):
        self.high_freq_cutoff = high_freq_cutoff

    def compute_fft_spectrum(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 2D shifted FFT complex spectrum and normalized log-magnitude map.
        """
        if len(image_rgb.shape) == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image_rgb.astype(np.float32)

        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))

        mag_max = magnitude.max()
        if mag_max > 0:
            norm_mag = magnitude / mag_max
        else:
            norm_mag = magnitude

        return fft_shift, norm_mag.astype(np.float32)

    def compute_radial_profile(self, magnitude_spectrum: np.ndarray, num_bins: int = 50) -> np.ndarray:
        """
        Compute azimuthally averaged radial power distribution from center (DC) to Nyquist.
        """
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = min(cy, cx)

        bins = np.linspace(0, max_r, num_bins + 1)
        radial_profile = np.zeros(num_bins, dtype=np.float32)

        for i in range(num_bins):
            mask = (r >= bins[i]) & (r < bins[i + 1])
            if np.any(mask):
                radial_profile[i] = float(np.mean(magnitude_spectrum[mask]))

        return radial_profile

    def analyze_frequency_domain(self, image_rgb: np.ndarray) -> dict:
        """
        Perform complete frequency-domain forensic analysis.
        Returns anomaly scores and structured forensic evidence.
        """
        h, w = image_rgb.shape[:2]
        fft_shift, norm_mag = self.compute_fft_spectrum(image_rgb)

        power_spectrum = np.abs(fft_shift) ** 2
        total_power = float(np.sum(power_spectrum) + 1e-10)

        # High-frequency mask
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = min(cy, cx)
        high_freq_mask = r >= (self.high_freq_cutoff * max_r)

        high_freq_power = float(np.sum(power_spectrum[high_freq_mask]))
        high_freq_ratio = float(high_freq_power / total_power)

        # Radial profile & spectral slope estimation (log(power) vs log(frequency))
        radial = self.compute_radial_profile(norm_mag, num_bins=32)
        freq_indices = np.arange(1, len(radial) + 1, dtype=np.float32)

        # Natural images exhibit power-law decay ~ 1/f^alpha (alpha ~ 1.5 - 2.5)
        # Synthetic generation / upsampling often disrupts this slope
        valid_indices = radial > 1e-6
        if np.sum(valid_indices) > 5:
            log_f = np.log(freq_indices[valid_indices])
            log_p = np.log(radial[valid_indices] + 1e-6)
            # Linear fit: log(P) = -alpha * log(f) + c
            coeffs = np.polyfit(log_f, log_p, 1)
            slope = float(coeffs[0])
        else:
            slope = -1.5

        # Heuristic anomaly score: deviation from natural slope (-1.8) and abnormal high-frequency spikes
        slope_deviation = abs(slope - (-1.8))
        anomaly_from_slope = min(1.0, slope_deviation / 2.0)
        anomaly_from_hf = min(1.0, max(0.0, (high_freq_ratio - 0.15) / 0.35))

        frequency_anomaly_score = float(np.clip(0.6 * anomaly_from_slope + 0.4 * anomaly_from_hf, 0.0, 1.0))

        return {
            "frequency_anomaly_score": frequency_anomaly_score,
            "high_frequency_ratio": high_freq_ratio,
            "spectral_slope": slope,
            "radial_profile": radial.tolist(),
            "norm_magnitude_map": norm_mag,
            "evidence_summary": f"Spectral slope: {slope:.2f}, High-freq energy ratio: {high_freq_ratio*100:.1f}%",
        }

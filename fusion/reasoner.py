import cv2
import numpy as np

class ForensicEvidenceReasoner:
    """
    Forensic Evidence Reasoner
    Extracts suspicious regions and constructs deterministic evidence statements based on
    modality scores and maps.
    """
    def __init__(self, anomaly_threshold=0.5, min_area_pixels=100):
        # MVP heuristics, not scientifically validated parameters
        self.anomaly_threshold = anomaly_threshold
        self.min_area_pixels = min_area_pixels

    def generate_evidence(self, payload: dict, media_type: str, anomaly_map=None, reliability_map=None) -> dict:
        suspicious_regions = []
        evidence = payload.get("evidence", [])
        if evidence is None:
            evidence = []

        # 1. VISUAL EVIDENCE & REGION EXTRACTION
        if anomaly_map is not None and isinstance(anomaly_map, np.ndarray) and anomaly_map.ndim == 2:
            try:
                # Thresholding mask
                mask = (anomaly_map >= self.anomaly_threshold).astype(np.uint8) * 255
                
                # Connected Components
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                
                valid_reliability = (
                    reliability_map is not None 
                    and isinstance(reliability_map, np.ndarray) 
                    and reliability_map.shape == anomaly_map.shape
                )
                
                for i in range(1, num_labels): # ignore background (0)
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area >= self.min_area_pixels:
                        x = int(stats[i, cv2.CC_STAT_LEFT])
                        y = int(stats[i, cv2.CC_STAT_TOP])
                        width = int(stats[i, cv2.CC_STAT_WIDTH])
                        height = int(stats[i, cv2.CC_STAT_HEIGHT])
                        
                        region_mask = (labels == i)
                        mean_anomaly = float(np.mean(anomaly_map[region_mask]))
                        max_anomaly = float(np.max(anomaly_map[region_mask]))
                        
                        region_info = {
                            "x": x,
                            "y": y,
                            "width": width,
                            "height": height,
                            "area": int(area),
                            "mean_anomaly": round(mean_anomaly, 4),
                            "max_anomaly": round(max_anomaly, 4)
                        }
                        
                        if valid_reliability:
                            mean_reliability = float(np.mean(reliability_map[region_mask]))
                            region_info["mean_reliability"] = round(mean_reliability, 4)
                            
                        suspicious_regions.append(region_info)
                        
            except Exception:
                # Silently fail region extraction without crashing
                pass

        # Textual Visual Evidence
        if payload.get("modalities", {}).get("visual_score") is not None:
            n_regions = len(suspicious_regions)
            if media_type == "video":
                prefix = "Representative-frame v"
            else:
                prefix = "V"
                
            if n_regions > 0:
                evidence.append(f"{prefix}isual analysis identified {n_regions} anomalous region(s).")
            else:
                evidence.append(f"{prefix}isual analysis identified no significant anomalous regions.")
        else:
            evidence.append("Visual analysis was not performed or was unavailable.")

        # 2. AUDIO EVIDENCE
        audio_score = payload.get("modalities", {}).get("audio_score")
        if audio_score is not None:
            evidence.append(f"Audio forensic analysis produced an AASIST spoof score of {audio_score:.2f}.")

        # 3. SYNC EVIDENCE (Video only)
        if media_type == "video":
            sync_evidence = payload.get("sync_evidence", {})
            if sync_evidence and sync_evidence.get("reliable") is not None:
                if sync_evidence.get("reliable") is True:
                    offset_ms = sync_evidence.get("offset_ms")
                    if offset_ms is not None:
                        evidence.append(f"Audio-video synchronization analysis found an offset of {offset_ms:.1f} ms.")
                else:
                    evidence.append("Audio-video synchronization evidence was unavailable because model confidence was below the configured reliability threshold.")

        # 4. CONCLUSION EVIDENCE
        decision = payload.get("decision", "UNCERTAIN")
        evidence.append(f"The final fusion result is {decision}.")

        # Update and return payload
        payload["suspicious_regions"] = suspicious_regions
        payload["evidence"] = evidence

        return payload

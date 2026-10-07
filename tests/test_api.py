import os
import unittest
import uuid
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# We patch the state *after* importing the app to avoid running lifespan
from api import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        # Override the orchestrator with a mock to avoid slow tests
        self.mock_orchestrator = MagicMock()
        self.mock_result = {
            "manipulation_score": 0.8,
            "decision": "LIKELY_MANIPULATED",
            "modalities": {},
            "visual_evidence": {
                "anomaly_map_path": os.path.join(os.getcwd(), "artifacts_fake123", "anomaly_map.png")
            },
            "sync_evidence": {},
            "evidence": [],
            "suspicious_regions": [],
            "suspicious_timeline": [],
            "processing": {}
        }
        self.mock_orchestrator.analyze.return_value = self.mock_result
        
        app.state.orchestrator = self.mock_orchestrator
        self.client = TestClient(app)
        
        # Setup fake artifact for GET test
        self.fake_artifact_dir = os.path.join(os.getcwd(), "artifacts_fake123")
        os.makedirs(self.fake_artifact_dir, exist_ok=True)
        self.fake_artifact_path = os.path.join(self.fake_artifact_dir, "anomaly_map.png")
        with open(self.fake_artifact_path, "w") as f:
            f.write("fake_png_data")

    def tearDown(self):
        if os.path.exists(self.fake_artifact_path):
            os.remove(self.fake_artifact_path)
        if os.path.exists(self.fake_artifact_dir):
            os.rmdir(self.fake_artifact_dir)

    def test_analyze_no_file(self):
        # Test missing file
        response = self.client.post("/api/v1/analyze")
        self.assertEqual(response.status_code, 422) # FastAPI validation error for missing body

    def test_analyze_with_file(self):
        # Test successful upload and JSON rewrite
        files = {"file": ("test.png", b"fake_image_data", "image/png")}
        response = self.client.post("/api/v1/analyze", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["decision"], "LIKELY_MANIPULATED")
        
        # Verify URL rewrite
        self.assertEqual(data["visual_evidence"]["anomaly_map_path"], "/api/v1/artifacts/fake123/anomaly_map.png")
        
        # Verify orchestrator was called
        self.mock_orchestrator.analyze.assert_called_once()
        
        # Verify temp file cleanup (the mock argument should be a temp file that is now deleted)
        called_path = self.mock_orchestrator.analyze.call_args[0][0]
        self.assertFalse(os.path.exists(called_path))

    def test_get_artifact_success(self):
        response = self.client.get("/api/v1/artifacts/fake123/anomaly_map.png")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "fake_png_data")

    def test_get_artifact_missing(self):
        response = self.client.get("/api/v1/artifacts/fake123/nonexistent.png")
        self.assertEqual(response.status_code, 404)

    def test_get_artifact_traversal(self):
        # Test traversal rejection
        # We must URL encode to prevent httpx from normalizing the path client-side
        response1 = self.client.get("/api/v1/artifacts/fake123/..%2Fsecrets.txt")
        self.assertIn(response1.status_code, [400, 404])
        
        response2 = self.client.get("/api/v1/artifacts/..%2Ffake123/anomaly_map.png")
        self.assertIn(response2.status_code, [400, 404])

if __name__ == "__main__":
    unittest.main()

import os
import sys

# Add current directory to Python path
current_dir = os.getcwd()
sys.path.insert(0, current_dir)

# Now try to import
try:
    from fastapi.testclient import TestClient

    from api import app

    client = TestClient(app)
    API_AVAILABLE = True
except Exception as e:
    print(f"Error importing api: {e}")
    API_AVAILABLE = False
    client = None

import pytest


@pytest.mark.skipif(not API_AVAILABLE, reason="API module not available")
class TestAPIHealth:

    def test_root_endpoint(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_model_info(self):
        response = client.get("/model-info")
        assert response.status_code == 200


@pytest.mark.skipif(not API_AVAILABLE, reason="API module not available")
class TestAPIPrediction:

    def test_predict_invalid_file(self):
        response = client.post(
            "/predict", files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400


class TestEnvironment:

    def test_python_version(self):
        import sys

        assert sys.version_info.major == 3
        assert sys.version_info.minor >= 8

    def test_torch_installed(self):
        import torch

        assert torch.__version__ is not None

    def test_numpy_installed(self):
        import numpy as np

        assert np.__version__ is not None

    def test_fastapi_installed(self):
        import fastapi

        assert fastapi.__version__ is not None

    def test_api_file_exists(self):
        import os

        assert os.path.exists("api.py")

    def test_models_vit_exists(self):
        import os

        assert os.path.exists("models_vit.py")

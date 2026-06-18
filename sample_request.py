"""
Sample script to test the API
"""

import sys
from pathlib import Path

import requests

API_URL = "http://localhost:8000"


def test_health():
    response = requests.get(f"{API_URL}/health")
    print(f"Health: {response.json()}")


def test_model_info():
    response = requests.get(f"{API_URL}/model-info")
    print(f"Model Info: {response.json()}")


def predict_image(image_path: str):
    if not Path(image_path).exists():
        print(f"Error: File {image_path} not found")
        return

    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/png")}
        response = requests.post(f"{API_URL}/predict", files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"\nImage: {image_path}")
        print(f"Prediction: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        print(f"Error: {response.status_code}")


if __name__ == "__main__":
    try:
        test_health()
        test_model_info()
    except requests.exceptions.ConnectionError:
        print("API is not running. Start with: uvicorn api:app --reload")
        sys.exit(1)

    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        print("\nUsage: python sample_request.py <image_path>")

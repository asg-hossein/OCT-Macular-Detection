"""Pytest configuration and fixtures"""

import pytest
from pathlib import Path
import torch
from PIL import Image
import numpy as np


@pytest.fixture
def sample_image():
    """Create a sample OCT image for testing"""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array, mode="RGB")


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary image file"""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode="RGB")
    img_path = tmp_path / "test_image.png"
    img.save(img_path)
    return img_path


@pytest.fixture
def device():
    """Get available device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
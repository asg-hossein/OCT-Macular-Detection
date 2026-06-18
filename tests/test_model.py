import pytest
import torch
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestSimpleModel:

    def test_torch_available(self):
        assert torch is not None
        x = torch.tensor([1, 2, 3])
        assert x.sum() == 6

    def test_numpy_available(self):
        import numpy as np
        arr = np.array([1, 2, 3])
        assert arr.sum() == 6


class TestModelFiles:

    def test_api_file_exists(self):
        assert os.path.exists(os.path.join(project_root, "api.py"))

    def test_models_vit_exists(self):
        assert os.path.exists(os.path.join(project_root, "models_vit.py"))

    @pytest.mark.skip(reason="weights file is not in repository")
    def test_weights_file_exists(self):
        weights_path = os.path.join(project_root, "weights", "RETFound_oct_weights.pth")
        assert os.path.exists(weights_path)


class TestTransform:

    def test_transform_import(self):
        from torchvision import transforms
        assert transforms is not None

    def test_basic_transform(self, sample_image):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        tensor = transform(sample_image)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32


class TestModelOutput:

    def test_model_import(self):
        import models_vit
        assert models_vit is not None

    def test_model_creation(self):
        import models_vit as models
        model = models.__dict__["RETFound_mae"](
            img_size=224, num_classes=2, drop_path_rate=0, global_pool=True
        )
        assert model is not None

    def test_model_forward_shape(self):
        import models_vit as models
        model = models.__dict__["RETFound_mae"](
            img_size=224, num_classes=2, drop_path_rate=0, global_pool=True
        )
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        assert output.shape == (1, 2)

    def test_model_softmax_output(self):
        import models_vit as models
        model = models.__dict__["RETFound_mae"](
            img_size=224, num_classes=2, drop_path_rate=0, global_pool=True
        )
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
            probs = torch.nn.functional.softmax(output, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), rtol=1e-5)
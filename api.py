"""
RETFound OCT Image Classification API
FastAPI service for macular hole detection using RETFound model
"""

import io
import os
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from torchvision import transforms

import models_vit as models

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_PATH = (
    Path(__file__).parent / "output_dir" / "OIMHS_finetune" / "checkpoint-best.pth"
)
WEIGHTS_PATH = Path(__file__).parent / "weights" / "RETFound_oct_weights.pth"
INPUT_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ["Normal", "Macular Hole"]

# Image transforms (same as in training)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose(
    [
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)

# ----------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
model = None


def load_model():
    """Load RETFound model with pretrained weights"""
    global model

    # Build model
    model = models.__dict__["RETFound_mae"](
        img_size=INPUT_SIZE,
        num_classes=NUM_CLASSES,
        drop_path_rate=0,
        global_pool=True,
    )

    # Load pretrained weights if available
    if WEIGHTS_PATH.exists():
        print(f"Loading pretrained weights from {WEIGHTS_PATH}")
        checkpoint = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        else:
            checkpoint_model = checkpoint

        # Clean up keys
        checkpoint_model = {
            k.replace("backbone.", ""): v for k, v in checkpoint_model.items()
        }
        checkpoint_model = {
            k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()
        }
        checkpoint_model = {
            k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()
        }

        model.load_state_dict(checkpoint_model, strict=False)
        print("Loaded RETFound pretrained weights")

    # Load fine-tuned weights if available
    if MODEL_PATH.exists():
        print(f"Loading fine-tuned weights from {MODEL_PATH}")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print("Loaded fine-tuned weights")

    model.to(device)
    model.eval()
    print("Model ready for inference")


@torch.no_grad()
def predict_image(image: Image.Image) -> Dict:
    """Run prediction on a single PIL image"""
    global model
    if model is None:
        load_model()

    # Transform image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Forward pass
    output = model(img_tensor)
    probabilities = F.softmax(output, dim=1)[0]
    predicted_class = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class].item()

    return {
        "class_id": predicted_class,
        "class_name": CLASS_NAMES[predicted_class],
        "confidence": round(confidence, 4),
        "probabilities": {
            CLASS_NAMES[0]: round(probabilities[0].item(), 4),
            CLASS_NAMES[1]: round(probabilities[1].item(), 4),
        },
    }


# ----------------------------------------------------------------------
# FastAPI App
# ----------------------------------------------------------------------
app = FastAPI(
    title="RETFound OCT Classification API",
    description="API for Macular Hole detection in OCT images",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    return {"message": "RETFound OCT Classification API", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict macular hole from OCT image"""
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read and convert image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run prediction
        result = predict_image(image)

        return {"success": True, **result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Batch prediction for multiple OCT images"""
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            result = predict_image(image)
            results.append({"filename": file.filename, "success": True, **result})
        except Exception as e:
            results.append(
                {"filename": file.filename, "success": False, "error": str(e)}
            )

    return {"results": results}


@app.get("/model-info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "RETFound_mae",
        "input_size": INPUT_SIZE,
        "num_classes": NUM_CLASSES,
        "classes": CLASS_NAMES,
        "device": str(device),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

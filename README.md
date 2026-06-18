OCT Macular Hole Detector

Project Overview

OCT Macular Hole Detector is an automated system for detecting Macular Hole disease from Optical Coherence Tomography (OCT) eye images using the RETFound foundation model.

Macular Hole is a serious retinal disease where a hole forms in the macula (the central part of the retina), leading to severe vision loss and even blindness. Early detection is crucial for successful treatment.
  Model & Dataset
Model: RETFound

    A foundation model for retinal images developed by UCL and Moorfields Eye Hospital

    Trained on 1.6 million OCT and fundus images

    Architecture: Vision Transformer (ViT) with 24 layers, 1024 hidden dimensions, 16 attention heads

Dataset: OIMHS

    3,859 OCT images from 119 patients (125 eyes)

    4 anatomical structures: Retina, Macular Hole, Intraretinal Cysts, Choroid

    Mean age: 64.1 years

    89 female, 30 male

 Download the dataset:
Download OIMHS Dataset from Figshare



 Features

    - RESTful API with FastAPI

    - Swagger UI for interactive documentation

    - Docker support for easy deployment

    - 21 automated tests (all passed)

    - CI/CD pipeline with GitHub Actions

    - Docker Hub integration

Prerequisites
Software	Version
Python	3.10+
Docker Desktop	Latest
Git	Latest
Quick Start
Method 1: Local Execution
bash

# 1. Clone the repository
git clone https://github.com/asg-hossein/OCT-Macular-Detection.git
cd OCT-Macular-Detection

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare dataset
python prepare_data.py

# 5. Run API
uvicorn api:app --reload

# 6. Test
python -c "import requests; files = {'file': ('0.png', open('OIMHS/Images/1/0.png', 'rb'), 'image/png')}; r = requests.post('http://localhost:8000/predict', files=files); print(r.json())"

# 7. Run tests
pytest tests/ -v

Method 2: Docker Execution
bash

# 1. Build image
docker build -t oct-macular-detection -f dockerfile .

# 2. Run container
docker run -p 8000:8000 oct-macular-detection

# 3. Test
python -c "import requests; files = {'file': ('0.png', open('OIMHS/Images/1/0.png', 'rb'), 'image/png')}; r = requests.post('http://localhost:8000/predict', files=files); print(r.json())"

Method 3: Swagger UI
text

http://localhost:8000/docs

API Documentation
Endpoints
Method	Endpoint	Description
GET	/	Root endpoint
GET	/health	Health check
GET	/model-info	Model information
POST	/predict	Single image prediction
POST	/predict-batch	Batch prediction
Example Response
json

{
  "success": true,
  "class_id": 1,
  "class_name": "Macular Hole",
  "confidence": 0.9275,
  "probabilities": {
    "Normal": 0.0725,
    "Macular Hole": 0.9275
  }
}

Docker Commands
bash

# Build image
docker build -t oct-macular-detection -f dockerfile .

# Run container
docker run -p 8000:8000 oct-macular-detection

# Run with different port
docker run -p 8001:8000 oct-macular-detection

# With Docker Compose
docker-compose up --build

CI/CD Pipeline

The project uses GitHub Actions with three stages:

    Lint: Code quality check with flake8, black, isort

    Test: Run tests with pytest

    Docker Build & Push: Build and push image to Docker Hub

Required GitHub Secrets
Secret	Description
DOCKERHUB_USERNAME	Your Docker Hub username
DOCKERHUB_TOKEN	Your Docker Hub access token
Model Performance
Metric	Value
Accuracy	92.75%
Precision	91%
Recall	89%
F1-Score	90%
ROC-AUC	95%

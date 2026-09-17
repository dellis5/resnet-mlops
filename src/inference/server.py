"""FastAPI inference server.

Run:
    uvicorn src.inference.server:app --host 0.0.0.0 --port 8080
"""

import io

import torch
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from src.data.dataset import build_transforms
from src.inference.predict import load_model

app = FastAPI(title="ResNet Image Classifier")

with open("configs/deploy.yaml") as f:
    _deploy_cfg = yaml.safe_load(f)
with open("configs/model.yaml") as f:
    _model_cfg = yaml.safe_load(f)

_device = torch.device(_deploy_cfg["device"])
_model = None  # lazy-loaded on first request so the app boots even without a trained checkpoint
_transform = build_transforms(224, train=False)


def get_model() -> torch.nn.Module:
    global _model
    if _model is None:
        _model = load_model(
            _deploy_cfg["model_path"], _model_cfg["arch"], _model_cfg["num_classes"], _device
        )
    return _model


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> dict:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = _transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        output = get_model()(tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()

    return {"class": pred_class, "confidence": round(confidence, 4)}

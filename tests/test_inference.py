import torch
from fastapi.testclient import TestClient

from src.training.train import build_model


def test_model_forward_pass_for_inference():
    model = build_model(arch="resnet18", num_classes=5, pretrained=False, freeze_backbone=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 5)


def test_health_endpoint():
    from src.inference.server import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

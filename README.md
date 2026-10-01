# resnet-mlops

End-to-end MLOps pipeline for a ResNet-based image classifier: data versioning (DVC),
training, evaluation, and containerized/K8s deployment.

Trained on the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
(37 breeds, ~7,400 images) using a pretrained ResNet18, reaching 90.2% validation accuracy.

## Layout

- `configs/` — YAML configs for data, model, training, and deployment
- `data/` — DVC-tracked raw and processed datasets (not in Git)
- `models/` — DVC-tracked checkpoints and promoted models (not in Git)
- `src/data/` — dataset loading and preprocessing
- `src/training/` — training loop, model builder, callbacks
- `src/inference/` — CLI prediction and FastAPI serving
- `pipelines/dvc.yaml` — reproducible DVC pipeline (preprocess -> train)
- `deployment/` — Dockerfiles, docker-compose, and Kubernetes manifests
- `tests/` — pytest unit tests

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data workflow

1. Put raw images under `data/raw/<class_name>/*.jpg`.
2. `dvc add data/raw` then commit the generated `data/raw.dvc` to Git.
3. `dvc repro pipelines/dvc.yaml` to preprocess and train, or run stages manually:
   ```bash
   python -m src.data.preprocess
   python -m src.training.train
   ```
4. `dvc push` to upload data/model artifacts to the configured remote.

## Serving

```bash
uvicorn src.inference.server:app --reload --port 8080
# or
docker compose -f deployment/docker-compose.yaml up --build
```

## Tests

```bash
pytest
```

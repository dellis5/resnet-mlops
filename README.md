WHERE TO ACCESS:

the whole project lives at C:\Users\danie\resnet-mlops; the actual dataset/model binary files are tracked by DVC and physically stored at C:\Users\danie\dvc-storage.

The readme.txt covers:

Exact paths to every key file (code, configs, data, trained model)
Step-by-step commands for: classifying a photo, running the web API, retraining, running tests, and saving new data/model versions with DVC+Git
A quick-reference cheat sheet at the bottom

# resnet-mlops

End-to-end MLOps pipeline for a ResNet-based image classifier: data versioning (DVC),
training, evaluation, and containerized/K8s deployment.

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

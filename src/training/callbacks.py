"""Early stopping and checkpointing helpers."""

from pathlib import Path

import torch


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score: float | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        improved = self.best_score is None or (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def save_checkpoint(model: torch.nn.Module, checkpoint_dir: str, filename: str) -> str:
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    torch.save(model.state_dict(), path)
    return str(path)

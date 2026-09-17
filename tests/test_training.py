import torch

from src.training.callbacks import EarlyStopping
from src.training.engine import evaluate, train_one_epoch
from src.training.train import build_model


def _make_loader():
    images = torch.randn(8, 3, 32, 32)
    labels = torch.randint(0, 2, (8,))
    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=4)


def test_build_model_output_shape():
    model = build_model(arch="resnet18", num_classes=2, pretrained=False, freeze_backbone=False)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 2)


def test_train_and_eval_one_epoch_runs():
    model = build_model(arch="resnet18", num_classes=2, pretrained=False, freeze_backbone=False)
    loader = _make_loader()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    loss = train_one_epoch(model, loader, optimizer, criterion, device)
    assert loss >= 0

    val_loss, val_acc = evaluate(model, loader, criterion, device)
    assert val_loss >= 0
    assert 0 <= val_acc <= 1


def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=2, mode="max")
    scores = [0.5, 0.4, 0.3]
    for score in scores:
        stopper.step(score)
    assert stopper.should_stop

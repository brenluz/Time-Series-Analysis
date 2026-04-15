import copy

import torch
import torch.nn as nn
from torch import optim


def train_batch(model: nn.Module,
                X_tensor: torch.Tensor,
                y_tensor: torch.Tensor,
                epochs: int,
                lr: float = 0.001,
                patience: int = 30,
                clip: float = 1.0,
                device: torch.device | str | None = None) -> nn.Module:
    """
    Full-batch training with early stopping and best-weight restoration.

    Trains the model on the entire dataset as one batch per epoch.
    Saves the best model state whenever training loss improves, and
    restores it at the end — so the returned model is always the
    best-seen checkpoint, not the final epoch.

    A ReduceLROnPlateau scheduler halves the learning rate when progress
    stalls, allowing fine-tuning after the initial fast descent.

    Parameters
    ----------
    model     : PyTorch model to train (modified in-place and returned).
    X_tensor  : Input tensor for the model.
    y_tensor  : Target tensor.
    epochs    : Maximum number of training epochs.
    lr        : Initial Adam learning rate.
    patience  : Early-stopping patience (epochs without improvement).
    clip      : Gradient clipping max norm.

    Returns
    -------
    model : The model loaded with its best weights.
    """
    if device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    X_tensor = X_tensor.to(device)
    y_tensor = y_tensor.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=patience // 2, min_lr=1e-6
    )
    loss_fn    = nn.MSELoss()
    best_loss  = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    counter    = 0

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(X_tensor), y_tensor)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step(loss.item())

        if loss.item() < best_loss - 1e-6:
            best_loss  = loss.item()
            best_state = copy.deepcopy(model.state_dict())
            counter    = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_state)
    return model
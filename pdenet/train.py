"""
Training script for PDE-Net.

Implements layer-wise training as described in the PDE-Net paper:
1. Warm-up: train with frozen filters (initial guess for coefficients/NN)
2. Layer-wise: train 1 block, then 2 blocks, ..., then n_blocks
3. All parameters shared across layers
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import PDENet, get_frozen_filters


def layerwise_train(
    model,
    train_batches,
    n_epochs_per_layer=2000,
    lr=0.01,
    dt=0.015,
    device='cpu',
    warmup_epochs=500,
    freeze_all_filters=True,
    verbose=True,
):
    """
    Layer-wise training of PDE-Net.

    Args:
        model: PDENet instance
        train_batches: list of lists of (input, target) mini-batches,
                       train_batches[k] for training with k+1 blocks
        n_epochs_per_layer: number of optimization steps per layer
        lr: learning rate for L-BFGS
        device: torch device
        warmup_epochs: number of warm-up steps (frozen filters)
        verbose: print progress

    Returns:
        history: dict of training losses
    """
    model = model.to(device)
    history = {'layer': [], 'loss': []}

    # Warm-up: train coefficients with frozen filters
    if warmup_epochs > 0 and len(train_batches) > 0:
        if verbose:
            print("=" * 60)
            print("WARM-UP: training coefficients with frozen filters")
            print("=" * 60)

        # Set up frozen filters (replace trainable filters with frozen ones)
        freeze_filters(model.block)

        # Train on first layer's data (1 δt-block)
        batch = train_batches[0][0]
        inputs = torch.from_numpy(batch[0]).float().unsqueeze(1).to(device)
        targets = torch.from_numpy(batch[1]).float().unsqueeze(1).to(device)

        loss = train_with_lbfgs(
            model, inputs, targets, n_steps=1, dt=dt,
            max_iter=warmup_epochs, lr=lr, verbose=verbose,
            desc="Warm-up"
        )
        history['layer'].append(0)
        history['loss'].append(loss)

        if not freeze_all_filters:
            # Unfreeze filters for joint training
            unfreeze_filters(model.block)
        else:
            if verbose:
                print("  Filters remain frozen (Frozen-PDE-Net mode)")

    # Layer-wise training
    for layer_idx in range(len(train_batches)):
        n_steps = layer_idx + 1  # number of δt-blocks

        if verbose:
            print("=" * 60)
            print(f"Layer {n_steps}: training {n_steps} δt-block(s)")
            print("=" * 60)

        layer_losses = []

        for batch_idx, batch in enumerate(train_batches[layer_idx]):
            inputs = torch.from_numpy(batch[0]).float().unsqueeze(1).to(device)
            targets = torch.from_numpy(batch[1]).float().unsqueeze(1).to(device)

            loss = train_with_lbfgs(
                model, inputs, targets, n_steps=n_steps, dt=dt,
                max_iter=n_epochs_per_layer // max(1, len(train_batches[layer_idx])),
                lr=lr, verbose=verbose,
                desc=f"Layer {n_steps}, batch {batch_idx + 1}"
            )
            layer_losses.append(loss)

        avg_loss = np.mean(layer_losses) if layer_losses else 0
        history['layer'].append(n_steps)
        history['loss'].append(avg_loss)

        if verbose:
            print(f"  Layer {n_steps} avg loss: {avg_loss:.6e}")

    return history


def train_with_lbfgs(model, inputs, targets, n_steps=1, dt=0.015, max_iter=2000,
                     lr=0.1, verbose=True, desc=""):
    """Train model with L-BFGS optimizer."""
    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_iter,
        max_eval=max_iter * 2,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=50,
        line_search_fn='strong_wolfe',
    )

    def closure():
        optimizer.zero_grad()
        outputs = model(inputs, dt=dt, n_steps=n_steps)
        loss = nn.MSELoss()(outputs, targets)
        loss.backward()
        return loss

    final_loss = None

    for i in range(max_iter):
        loss = optimizer.step(closure)

        if loss is None:
            # L-BFGS converged
            break

        final_loss = loss.item()

        if verbose and i % max(1, max_iter // 10) == 0:
            print(f"  [{desc}] iter {i}: loss = {final_loss:.6e}")

    if verbose:
        print(f"  [{desc}] Final loss: {final_loss:.6e}")

    return final_loss


def freeze_filters(block):
    """Replace trainable constrained filters with frozen (base) filters."""
    for name, module in block.named_modules():
        if isinstance(module, ConstrainedConv2d):
            if module.theta is not None:
                module.theta.requires_grad_(False)


def unfreeze_filters(block):
    """Restore trainability of constrained filters."""
    for name, module in block.named_modules():
        if isinstance(module, ConstrainedConv2d):
            if module.theta is not None:
                module.theta.requires_grad_(True)


# Import locally to avoid circular imports
from .model import ConstrainedConv2d

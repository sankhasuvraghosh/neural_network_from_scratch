# Neural Network From Scratch (NumPy)

A minimal feed-forward neural network implemented from scratch in NumPy, trained on the classic **spiral dataset** (via `nnfs`) for multi-class classification. This follows the architecture style popularized by the "Neural Networks from Scratch" (NNFS) book/course, with a dense layer, ReLU activation, dropout, softmax + categorical cross-entropy loss, and a choice of optimizers (SGD, Adagrad, RMSProp, Adam).

## Features

- **Dense (fully connected) layer** — `layer_dense`
  - Forward/backward pass with gradient computation
  - Supports L1 and L2 regularization on both weights and biases
- **Activations**
  - `relu` — ReLU activation with forward/backward pass
  - `softmax_activation` — Softmax with forward/backward pass
- **Loss**
  - `layer_entropy_loss` — Categorical cross-entropy loss
  - `activation_loss_entropy` — Combined softmax + cross-entropy for a faster, more numerically stable backward pass
- **Regularization**
  - `Dropout` — Inverted dropout layer, correctly disabled during evaluation (`training=False`)
- **Optimizers** (each with learning rate decay support)
  - `sgd_optimizer` — SGD with optional momentum
  - `adagrad_optimizer`
  - `rmsprop_optimizer`
  - `adam_optimizer`

## Requirements

```
numpy
matplotlib
nnfs
```

Install with:

```bash
pip install numpy matplotlib nnfs
```

## Usage

Run the script directly:

```bash
python nn_from_scratch.py
```

By default it will:

1. Generate a spiral dataset with 100 samples per class across 3 classes (`nnfs.datasets.spiral_data`).
2. Build a small network:
   - `Dense(2 → 64)` → `ReLU` → `Dropout(0.1)` → `Dense(64 → 3)` → `Softmax + Cross-Entropy`
3. Train for 10,001 epochs using the Adam optimizer (`lr=0.02`, `decay=5e-4`), printing epoch, accuracy, loss, and current learning rate every 100 epochs.
4. Evaluate the trained model on a freshly generated spiral test set — with dropout disabled — and print final test accuracy.

## Architecture Overview

```
Input (2 features)
   │
   ▼
Dense Layer (2 → 64)   [L2 regularization: 5e-4 on weights & biases]
   │
   ▼
ReLU Activation
   │
   ▼
Dropout (rate=0.1)     [active only when training=True]
   │
   ▼
Dense Layer (64 → 3)   [L2 regularization: 5e-4 on weights & biases]
   │
   ▼
Softmax + Categorical Cross-Entropy Loss
```

## Configuration

Key hyperparameters you can tune at the top of the script:

| Parameter | Location | Default |
|---|---|---|
| Samples per class / number of classes | `spiral_data(samples=100, classes=3)` | 100 / 3 |
| Hidden layer size | `layer_dense(2, 64, ...)` | 64 |
| Dropout rate | `Dropout(0.1)` | 0.1 (drops 10%) |
| L2 regularization strength | `w_regularizer_l2`, `b_regularizer_l2` | 5e-4 |
| Optimizer | `adam_optimizer(...)` | Adam |
| Learning rate | `learning_rate=0.02` | 0.02 |
| Learning rate decay | `decay=5e-4` | 5e-4 |
| Training epochs | `for epoch in range(10001)` | 10,001 |

Swap `adam_optimizer` for `sgd_optimizer`, `adagrad_optimizer`, or `rmsprop_optimizer` to compare optimizer performance — all share the same `pre_update_parameters()` / `update_parameters(layer)` / `post_update_parameters()` interface.

## Fixed Since Last Version

- **Dropout eval-mode bug**: `Dropout.forward` previously set `self.output = inputs.copy()` when `training=False` but then fell through and unconditionally overwrote it with the masked/dropped-out version — meaning dropout was still applied at test time. A `return` was added after the `training=False` branch so inference now correctly passes inputs through untouched.

## Notes / Remaining Quirks

- **RMSProp/Adagrad naming inconsistency**: these two optimizers use `layer.biases_cache` while `adam_optimizer` uses `layer.bias_cache`. This is harmless internally (each optimizer manages its own attribute name) but means you can't freely switch optimizers mid-training on the same layer instance without re-initializing it.
- **No train/validation split during training**: the "test" evaluation at the end uses a newly sampled spiral dataset rather than a held-out split of the original data.
- **Global variable reuse**: `predictions`/`accuracy` are recomputed and overwritten during both the training loop and the test block — fine for a script, but worth namespacing if extended into functions/classes.

## Extending

Some natural next steps:
- Add a validation loop every N epochs to monitor generalization during training.
- Save/load trained weights (`dense1.weights`, `dense1.biases`, etc.) via `np.savez`.
- Add a plotting step with `matplotlib` to visualize decision boundaries (the import is already present but unused).
- Refactor layers/optimizers into separate modules for reuse across projects.

## License

No license specified — add one (e.g., MIT) if you plan to share or reuse this code.

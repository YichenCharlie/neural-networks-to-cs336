# Neural Networks from Scratch — A Path to Stanford CS336

Hands-on neural network implementation with PyTorch, progressing from foundational models to language modeling and ML systems.

The focus is on independently writing training and evaluation code, understanding tensor operations and gradients, and consolidating notebook experiments into standalone programs. The foundational exercises use PyTorch autograd and built-in layers; “from scratch” refers to implementing the model workflows rather than building an automatic differentiation engine.

## Stage A — Neural Network Foundations

Implemented linear regression, binary classification, and an XOR MLP to study gradients, manual parameter updates, SGD, nonlinear activations, and model organization with `nn.Module`.

The final exercise combines training, prediction, and weight saving/loading in a standalone script. Restored model outputs were verified against the original model.

| Component | Configuration / Result |
|---|---|
| Architecture | Linear(2, 4) → ReLU → Linear(4, 1) |
| Trainable parameters | 17 |
| Loss | Binary cross-entropy with logits |
| Optimizer | SGD, learning rate 0.1 |
| Training steps | 3,000 |
| Final training loss | 0.005162 |
| Training accuracy | 4/4 |

![Learned XOR decision boundary](figures/stage_a/xor_decision_boundary.png)

The background shows the predicted probability of class 1, and the black contour marks the 0.5 decision boundary. Results describe the four training samples, not generalization performance.

## Stage B — Fashion-MNIST Classification

Built a complete image-classification workflow with mini-batch training, separate validation, checkpoint selection, and held-out test evaluation. The implementation was developed in notebooks and consolidated into a standalone Python script.

| Component | Configuration |
|---|---|
| Dataset | Fashion-MNIST, 10 classes |
| Split | 54,000 training / 6,000 validation / 10,000 test |
| Input | 1 × 28 × 28 grayscale image |
| Architecture | Flatten → Linear(784, 128) → ReLU → Linear(128, 10) |
| Trainable parameters | 101,770 |
| Loss | Cross-entropy |
| Optimizer | SGD, learning rate 0.1 |
| Batch size / Epochs | 32 / 5 |
| Model selection | Highest validation accuracy |
| Execution | CPU |

**Standalone script result:** the selected checkpoint achieved **86.41% test accuracy** on all 10,000 test images.

| Metric | Result |
|---|---:|
| Selected epoch | 5 |
| Validation accuracy | 86.80% |
| Test accuracy | 86.41% |
| Test cross-entropy loss | 0.373224 |

The script reloads the selected checkpoint into a new model before testing. Training and validation losses are aggregated by sample count, including partial final batches.

An earlier notebook run reached 86.75% test accuracy. These are separate training runs: the data split uses seed 42, while model initialization and training order are not fully seeded. Correct and incorrect predictions were also visualized to inspect model behavior.

[Stage B learning notes (Chinese)](docs/stage_b_notes.md)

## Run the Experiments

From the repository root, with the Python environment activated:

### Stage A

```bash
python scripts/stage_a_xor.py
python scripts/plot_stage_a_xor.py
```

The training script saves weights to `notebooks/stage_a/checkpoints/stage_a_final.pt`. The plotting script loads those weights and saves the decision-boundary figure without retraining.

### Stage B

```bash
python scripts/stage_b_fashion_mnist.py
```

The script downloads or reuses Fashion-MNIST in `data/`, trains for five epochs, saves the best validation checkpoint to `checkpoints/stageb_fashion_mnist_best.pt`, and evaluates the restored model on the official test set.

Running the script again starts a new training run and may overwrite that checkpoint. Exact metrics can vary across runs.

## Learning Approach

- Implement core model and training logic independently.
- Use notebooks to inspect tensors, gradients, and intermediate outputs.
- Consolidate completed workflows into standalone Python scripts.
- Check tensor shapes, sample counts, metric aggregation, and checkpoint restoration.
- Prioritize understanding and correctness over benchmark scores.

## Next Direction

Build a small character-level language model to learn text encoding, next-token prediction, and autoregressive generation, then progress toward Stanford CS336 assignments on language modeling and systems.

## Tools

Python · PyTorch · torchvision · Jupyter · NumPy · Matplotlib

[Stanford CS336: Language Modeling from Scratch](https://cs336.stanford.edu/)
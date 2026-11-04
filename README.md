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

## Stage C — Character-Level Bigram Language Model

Built a language model for `"hello world"` using an **8 × 8 trainable lookup table**. Each row represents the current character; its eight logits score the possible next characters. Each SGD step samples two text windows, each containing four next-character targets. Training and final evaluation use the same text so that every prediction can be checked by hand.

| Standalone run | Result |
|---|---:|
| Trainable parameters | 64 |
| SGD steps / learning rate | 1,000 / 0.1 |
| Full-text cross-entropy, before → after training | **2.0672 → 0.6473** |
| Correct next-character predictions given the original inputs | 7/10 |

The 7/10 result illustrates the model's context limit. In the training text, `l` has three distinct successors and `o` has two. A bigram model gives the same scores every time it sees the same character, so a fixed `argmax` must miss at least three of the ten targets.

The script also generates text from a single starting character. Each sampled character becomes the next input. One run produced `heldrlo wo wordeweldl`: the initial `h` plus 20 sampled characters.

[Stage C learning notes (Chinese)](docs/stage_c_notes.md)

## Run the Experiments

From the repository root, with the Python environment activated:

```bash
# Stage A: train the XOR model, then plot its decision boundary
python scripts/stage_a_xor.py
python scripts/plot_stage_a_xor.py

# Stage B: train, select a checkpoint, and evaluate on the test set
python scripts/stage_b_fashion_mnist.py

# Stage C: train the bigram model and generate text
python scripts/stage_c_bigram.py
```

Stage A saves weights to `notebooks/stage_a/checkpoints/stage_a_final.pt`; the plotting script loads them without retraining. Stage B downloads or reuses Fashion-MNIST in `data/` and writes the selected checkpoint to `checkpoints/stageb_fashion_mnist_best.pt`. Stage C uses an inline training string and requires no dataset download.

Reported metrics come from individual runs. Initialization, training order, sampled windows, and generated text can vary.

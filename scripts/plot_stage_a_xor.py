from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = ROOT / "notebooks/stage_a/checkpoints/stage_a_final.pt"
OUTPUT = ROOT / "figures/stage_a/xor_decision_boundary.png"


class XORmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 4)
        self.layer2 = nn.Linear(4, 1)

    def forward(self, x):
        return self.layer2(torch.relu(self.layer1(x)))


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}\n"
            "Check where the training script saved stage_a_final.pt."
        )

    model = XORmodel()
    state = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    x = torch.tensor(
        [[-1, -1], [-1, 1], [1, -1], [1, 1]],
        dtype=torch.float32,
    )
    y = torch.tensor([0, 1, 1, 0])

    # 在平面上建立密集网格，观察模型在不同位置的预测。
    axis = np.linspace(-1.6, 1.6, 301)
    xx, yy = np.meshgrid(axis, axis)
    grid = torch.from_numpy(
        np.column_stack([xx.ravel(), yy.ravel()])
    ).float()

    with torch.inference_mode():
        surface = torch.sigmoid(model(grid)).reshape(xx.shape).numpy()
        probabilities = torch.sigmoid(model(x)).squeeze(1)
        predictions = (probabilities >= 0.5).long()
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(x).squeeze(1), y.float()
        ).item()

    correct = (predictions == y).sum().item()
    parameter_count = sum(p.numel() for p in model.parameters())

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
    }):
        fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)

        shading = ax.pcolormesh(
            xx, yy, surface,
            cmap="RdBu_r",
            vmin=0,
            vmax=1,
            shading="auto",
            rasterized=True,
        )

        if surface.min() < 0.5 < surface.max():
            boundary = ax.contour(
                xx, yy, surface,
                levels=[0.5],
                colors="#202020",
                linewidths=1.7,
            )
            ax.clabel(boundary, fmt={0.5: "p = 0.5"}, fontsize=9)

        styles = [
            (0, "o", "#2166ac"),
            (1, "s", "#b2182b"),
        ]
        for label, marker, color in styles:
            points = x[y == label].numpy()
            ax.scatter(
                points[:, 0], points[:, 1],
                s=130,
                marker=marker,
                c=color,
                edgecolors="white",
                linewidths=1.8,
                label=f"Class {label}",
                zorder=3,
            )

        for point, probability in zip(x.numpy(), probabilities.tolist()):
            ax.annotate(
                f"p={probability:.3f}",
                xy=point,
                xytext=(0, 13),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.9,
                },
                zorder=4,
            )

        ax.set(
            xlabel=r"$x_1$",
            ylabel=r"$x_2$",
            xlim=(-1.6, 1.6),
            ylim=(-1.6, 1.6),
            title=(
                "XOR classification with a two-layer MLP\n"
                f"2 → 4 → 1  |  {parameter_count} parameters  |  "
                f"Training accuracy: {correct}/4"
            ),
        )
        ax.set_aspect("equal")
        ax.set_xticks([-1.5, -1, 0, 1, 1.5])
        ax.set_yticks([-1.5, -1, 0, 1, 1.5])
        ax.legend(loc="lower center", ncol=2, framealpha=0.95)

        colorbar = fig.colorbar(shading, ax=ax, pad=0.03, shrink=0.85)
        colorbar.set_label("Predicted probability of class 1")

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT, bbox_inches="tight")
        plt.close(fig)

    print("Checkpoint:", CHECKPOINT)
    print(f"Training BCE loss: {loss:.6f}")
    print("Probabilities:", probabilities.tolist())
    print("Predictions:", predictions.tolist())
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
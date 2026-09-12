"""Generate the `probably` project logo.

A series of beta distributions with peaks shifting left to right and
decreasing height, in rainbow colors — a small nod to the package's focus
on probability distributions.

Renders two variants so the wordmark stays legible on either background:
`logo.png` (dark wordmark, for light themes) and `logo-dark.png` (light
wordmark, for dark themes).

Run with:

    python examples/logo.py

Writes examples/assets/logo.png and examples/assets/logo-dark.png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from scipy import stats

BETA_PARAMS = [
    (2, 8),
    (3, 6),
    (4, 4),
    (5, 3.5),
    (7, 3),
    (9, 2.5),
    (12, 2),
]

MAX_HEIGHT = 0.6

x = np.linspace(0, 1, 400)
colors = colormaps["rainbow"](np.linspace(1, 0, len(BETA_PARAMS)))
heights = np.linspace(MAX_HEIGHT, 0.09, len(BETA_PARAMS))


def render(wordmark_color: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    for (a, b), color, height in zip(BETA_PARAMS, colors, heights, strict=True):
        pdf = stats.beta.pdf(x, a, b)
        y = pdf / pdf.max() * height
        ax.plot(x, y, color=color, linewidth=3, solid_capstyle="round")
        ax.fill_between(x, y, color=color, alpha=0.15)

    ax.text(
        0.5,
        -0.06,
        "Pr(obably)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=68,
        fontweight="bold",
        color=wordmark_color,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, MAX_HEIGHT * 1.05)
    ax.axis("off")
    fig.tight_layout(pad=0.1)

    output_path = Path(__file__).parent / "assets" / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


for color, filename in (("#0b0b0b", "logo.png"), ("#f5f5f5", "logo-dark.png")):
    print(f"wrote {render(color, filename)}")

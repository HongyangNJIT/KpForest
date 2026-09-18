# =========================================================================
#   (c) Copyright 2026
#   All rights reserved
#   Programs written by Hongyang Zhang
#   Department of Computer Science
#   New Jersey Institute of Technology
#   University Heights, Newark, NJ 07102, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================

"""Draw Figure 3 from the predictions computed by the test workflow."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


def plot_figure3(predictions, destination):
    """Save the paper-style figure without changing any prediction values."""
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame.event_onset_time, utc=True, errors="raise")
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    observed, point, lower, upper = (frame[column].to_numpy() for column in (
        "observed_max_kp", "predicted_max_kp",
        "predicted_max_kp_lower", "predicted_max_kp_upper"))
    if frame.empty or not np.isfinite([observed, point, lower, upper]).all():
        raise ValueError("Figure 3 requires finite predictions and observations.")
    if not ((lower <= point) & (point <= upper)).all():
        raise ValueError("Each prediction must lie within its interval.")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    style = {
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 17, "axes.labelsize": 17,
        "xtick.labelsize": 17, "ytick.labelsize": 17,
        "axes.grid": False, "figure.dpi": 300, "savefig.dpi": 300,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    }
    with plt.rc_context(style):
        x = np.arange(len(frame))
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.errorbar(x, point, yerr=[point - lower, upper - point], fmt="none",
                    ecolor="#AAAAAA", elinewidth=3.5, capsize=8,
                    capthick=3.5, alpha=.85, zorder=10)
        ax.scatter(x, point, marker="D", s=120, color="#0055AA",
                   edgecolors="white", linewidth=1.5, zorder=20)
        ax.scatter(x, observed, marker="o", s=180, facecolors="none",
                   edgecolors="#FF8800", linewidth=2.5, zorder=22)
        ax.set_xticks(x, frame.date.dt.strftime("%Y-%m-%d"), rotation=90, ha="center")
        ax.set_yticks(np.arange(10))
        ax.set_ylim(-.5, 9.5)
        ax.set_xlabel("CME Onset Time", fontsize=24, labelpad=20)
        ax.set_ylabel("Max Kp", fontsize=24, labelpad=20)
        ax.grid(True, linestyle=":", alpha=.4, axis="y")
        ax.spines[["right", "top"]].set_visible(False)
        handles = [
            Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="#FF8800",
                   markeredgewidth=2.5, markersize=12, linewidth=0, label="Observed Value"),
            Line2D([], [], marker="D", markerfacecolor="#0055AA", markeredgecolor="white",
                   markeredgewidth=1.5, markersize=10, linewidth=0, label="Predicted Value"),
            Line2D([], [], marker="|", color="#AAAAAA", markersize=12, linewidth=2,
                   label="Prediction Interval"),
        ]
        ax.legend(handles=handles, loc="lower left", framealpha=.9, edgecolor="none",
                  prop={"family": ["Arial", "DejaVu Sans"], "size": 17})
        fig.tight_layout()
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer())
        width, height = (pixels / 300 + 1e-9 for pixels in (4107, 2907))
        if tight.width > width or tight.height > height:
            plt.close(fig)
            raise ValueError("The Figure 3 canvas would clip the plotted content.")
        box = Bbox.from_bounds(tight.x0 - (width - tight.width) / 2,
                               tight.y0 - (height - tight.height) / 2,
                               width, height)
        fig.savefig(destination, dpi=300, bbox_inches=box, pad_inches=0)
        plt.close(fig)
    return destination


def plot_main():
    """Plot fresh model predictions, or an explicitly supplied prediction CSV."""
    import argparse
    import sys

    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path,
                        help="Existing predictions.csv; omitted means run the supplied model.")
    parser.add_argument("--output", type=Path, default=root / "figure3.png")
    args = parser.parse_args()
    if args.predictions is None:
        from test import main as run_test

        predictions = run_test([])
    else:
        predictions = pd.read_csv(args.predictions)
    plot_figure3(predictions, args.output)
    print(f"Figure 3 saved as {args.output.name}.", file=sys.stderr)


if __name__ == "__main__":
    plot_main()

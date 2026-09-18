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

"""Evaluate the supplied model and print the 24-event paper example."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "model code"))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from model import load_model, predict



def evaluate(observed, prediction):
    """MAE/MRE use the median; EC includes both interval endpoints."""
    y = np.asarray(observed, dtype=float)
    if y.ndim != 1 or len(y) != len(prediction) or not np.isfinite(y).all():
        raise ValueError("Each prediction requires a finite observed Max Kp.")
    if len(y) < 2 or not np.any(y != 0):
        raise ValueError("Scoring requires at least two events and a nonzero target.")
    nonzero = y != 0
    lower, point, upper = prediction.T
    return {
        "MAE": float(mean_absolute_error(y, point)),
        "MRE": float(np.mean(np.abs(point[nonzero] - y[nonzero]) / np.abs(y[nonzero]))),
        "R2": float(r2_score(y, point)),
        "EC": float(np.mean((y >= lower) & (y <= upper)) * 100),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "pre-trained model and weights")
    parser.add_argument("--data", type=Path, default=ROOT / "testing data/test.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    forest, metadata, quantiles = load_model(args.model)
    frame = pd.read_csv(args.data)
    values = predict(forest, metadata, quantiles, frame)
    observed = frame[metadata["target_col"]].to_numpy(dtype=float)
    metrics = evaluate(observed, values)
    times = pd.to_datetime(frame["observedTime"], utc=True, errors="raise")
    if times.isna().any():
        raise ValueError("Each test event requires an onset time.")
    predictions = pd.DataFrame({
        "event_onset_time": times.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "predicted_max_kp": values[:, 1],
        "predicted_max_kp_lower": values[:, 0],
        "predicted_max_kp_upper": values[:, 2],
        "observed_max_kp": observed,
    })
    args.output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output / "predictions.csv", index=False)
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for row in predictions.itertuples(index=False):
        print(f"Event onset time: {row.event_onset_time}  "
              f"Predicted Max Kp: {row.predicted_max_kp:.4f}  "
              f"Predicted Max Kp range: {row.predicted_max_kp_lower:.4f} to "
              f"{row.predicted_max_kp_upper:.4f}  "
              f"Observed Max Kp: {row.observed_max_kp:.2f}")
    print(f"\nMAE: {metrics['MAE']:.4f}")
    print(f"MRE: {metrics['MRE']:.4f}")
    print(f"R^2: {metrics['R2']:.4f}")
    print(f"EC: {metrics['EC']:.2f}%")
    return predictions


if __name__ == "__main__":
    from plot_figure3 import plot_figure3

    plot_figure3(main(), ROOT / "figure3.png")
    print("Figure 3 saved as figure3.png in the repository root.", file=sys.stderr)

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

"""Load the fitted forest and convert residual predictions to Max Kp."""
import pickle
import warnings
from pathlib import Path

import numpy as np
from quantile_forest import RandomForestQuantileRegressor
from sklearn.exceptions import InconsistentVersionWarning


def load_model(directory):
    directory = Path(directory)
    # The bundle uses a plain dictionary and a native forest, with no project classes.
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        with (directory / "model.pkl").open("rb") as stream:
            bundle = pickle.load(stream)
    if not isinstance(bundle, dict) or bundle.get("format_version") != 1:
        raise ValueError("Unsupported model bundle format.")
    forest, metadata = bundle["forest"], bundle["metadata"]
    features = metadata["features"]
    if not isinstance(forest, RandomForestQuantileRegressor):
        raise ValueError("The model must be a fitted RandomForestQuantileRegressor.")
    if len(features) != forest.n_features_in_ or len(set(features)) != len(features):
        raise ValueError("The feature list does not match the fitted forest.")
    if metadata["method"] != "zero_residual" or metadata["kp_bounds"] != [0, 9]:
        raise ValueError("This program requires the residual Kp model with bounds [0, 9].")
    lo, hi = map(float, bundle["quantiles"])
    if not 0 < lo < 0.5 < hi < 1:
        raise ValueError("Interval quantiles must satisfy 0 < lower < 0.5 < upper < 1.")
    return forest, metadata, (lo, hi)


def predict(forest, metadata, quantiles, frame):
    """Return lower, median and upper Kp; observations are not model inputs."""
    features = metadata["features"]
    baseline = metadata["baseline_col"]
    required = features + [baseline]
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Provide nonempty data with unique column names.")
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Missing input columns: {missing}")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("All model inputs must be finite numeric values.")
    residuals = forest.predict(frame[features].to_numpy(),
                               quantiles=[quantiles[0], 0.5, quantiles[1]])
    # Add the Scoreboard baseline to every quantile, then apply Kp bounds.
    result = np.clip(residuals + frame[baseline].to_numpy()[:, None], 0, 9)
    if not np.isfinite(result).all() or (np.diff(result, axis=1) < 0).any():
        raise ValueError("The model returned invalid or unordered predictions.")
    return result

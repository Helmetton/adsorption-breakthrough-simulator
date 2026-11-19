# -*- coding: utf-8 -*-
from __future__ import annotations

from io import BytesIO
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from encoding_utils import safe_read_csv


try:
    import joblib
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
except Exception:  # pragma: no cover - optional dependency
    joblib = None
    ExtraTreesRegressor = None
    HistGradientBoostingRegressor = None
    RandomForestRegressor = None
    LinearRegression = None
    MLPRegressor = None
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None
    train_test_split = None


try:
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None


TARGET_COLUMN = "C_out_C0"
BASE_FEATURE_COLUMNS = [
    "time_min",
    "C0_mg_L",
    "C_breakthrough_ratio",
    "flow_ml_min",
    "flow_m3_h",
    "Q_m3_h",
    "bed_height_cm",
    "bed_height_m",
    "column_diameter_cm",
    "column_diameter_m",
    "temperature_C",
    "epsilon",
    "bulk_density_g_cm3",
    "bulk_density_kg_m3",
    "bed_volume_cm3",
    "bed_volume_m3",
    "carbon_mass_g",
    "carbon_mass_kg",
    "EBCT_min",
    "KH_L_g",
    "K_H_used",
    "KH_T_L_g",
    "kLDF_1_min",
    "kLDF_used",
    "kLDF_used_1_min",
    "Dax_cm2_min",
    "Dax_used",
    "Dax_T_cm2_min",
    "scale_mode",
    "sorbent_profile",
    "pollutant",
]


def _sklearn_available() -> bool:
    return all(
        item is not None
        for item in [
            joblib,
            RandomForestRegressor,
            ExtraTreesRegressor,
            HistGradientBoostingRegressor,
            LinearRegression,
            train_test_split,
        ]
    )


def create_model_bundle(model, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "metadata": metadata,
        "bundle_version": "1.0",
    }


def save_joblib_to_bytes(obj) -> bytes:
    if joblib is None:
        raise RuntimeError("joblib is not available.")
    buffer = BytesIO()
    joblib.dump(obj, buffer, compress=3)
    return buffer.getvalue()


def save_model_bundle_to_bytes(model, metadata: dict[str, Any]) -> bytes:
    return save_joblib_to_bytes(create_model_bundle(model, metadata))


def metadata_to_bytes(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")


def prepare_ml_dataset(df_or_path) -> tuple[pd.DataFrame, pd.Series, list[str], str]:
    df = safe_read_csv(df_or_path) if isinstance(df_or_path, str) else df_or_path.copy()
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"В датасете нет целевого столбца {TARGET_COLUMN}.")

    available = [col for col in BASE_FEATURE_COLUMNS if col in df.columns]
    if not available:
        raise ValueError("Не найдены подходящие признаки для обучения ML-модели.")

    features = df[available].copy()
    features = pd.get_dummies(features, dummy_na=True)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0.0)
    return features, target, list(features.columns), TARGET_COLUMN


def _metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_baseline_models(df_or_path, output_dir="ml_models",
                          dataset_source="synthetic_dynamic_sorption_dataset.csv"):
    """Compatibility API; no row-wise scientific training route remains."""
    from ml_adsorber import load_ml_dataset, train_models
    df = load_ml_dataset(df_or_path) if isinstance(df_or_path, (str, Path)) else df_or_path.copy()
    return train_models(df, "curve", dataset_filename=dataset_source, output_dir=output_dir)


def optimize_model_with_optuna(df_or_path, model_type="random_forest", n_trials=30):
    """Use inner grouped validation; the outer test is excluded from Optuna."""
    from ml_adsorber import load_ml_dataset, train_models
    df = load_ml_dataset(df_or_path) if isinstance(df_or_path, (str, Path)) else df_or_path.copy()
    return train_models(df, "curve", use_optuna=True, n_trials=n_trials)


def train_breakthrough_model(
    df_or_path,
    output_dir: str = "ml_models",
    dataset_source: str = "synthetic_dynamic_sorption_dataset.csv",
    use_optuna: bool = False,
    n_trials: int = 30,
    target_transform: str = "log1p",
) -> dict[str, Any]:
    """Train a one-row-per-case breakthrough-time model.

    The target is breakthrough_time_min. Time-resolved outputs such as
    time_min, C_out_C0 and C_out_mg_L are intentionally excluded from features.
    """
    from ml_adsorber import load_ml_dataset, train_models

    df = load_ml_dataset(df_or_path) if isinstance(df_or_path, str) else df_or_path.copy()
    result = train_models(
        df,
        task_type="breakthrough",
        dataset_filename=dataset_source,
        output_dir=output_dir,
        use_optuna=use_optuna,
        n_trials=n_trials,
        target_transform=target_transform,
    )
    metadata = result.get("metadata") or {}
    forbidden = {"time_min", "C_out_C0", "C_out_mg_L"}
    feature_columns = set(metadata.get("feature_columns") or [])
    if forbidden & feature_columns:
        return {
            "success": False,
            "message": f"Breakthrough feature_columns contain forbidden curve columns: {sorted(forbidden & feature_columns)}",
            "metrics": result.get("metrics", {}),
            "metadata": metadata,
        }
    return result

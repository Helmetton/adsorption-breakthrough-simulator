# -*- coding: utf-8 -*-
"""ML utilities for breakthrough-curve learning and adsorber screening.

The mechanistic simulator remains the source of physical synthetic datasets.
This module only trains surrogate models and never changes the transport model.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from encoding_utils import safe_read_csv
from model import DEFAULT_BREAKTHROUGH_RATIO, interpret_breakthrough_curve
from parameter_provenance import dataset_parameter_provenance


try:
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception:  # pragma: no cover - optional dependency
    joblib = None
    ColumnTransformer = None
    ExtraTreesRegressor = None
    HistGradientBoostingRegressor = None
    RandomForestRegressor = None
    permutation_importance = None
    LinearRegression = None
    mean_absolute_error = None
    mean_squared_error = None
    r2_score = None
    GroupShuffleSplit = None
    Pipeline = None
    OneHotEncoder = None
    StandardScaler = None


try:
    import optuna
except Exception:  # pragma: no cover - optional dependency
    optuna = None


PROGRAM_VERSION = "dynamic_adsorption_streamlit_ml_v1"

CANONICAL_ALIASES = {
    "flow": ["flow", "flow_ml_min", "Q", "Q_m3_h", "flow_m3_h"],
    "height": ["height", "H", "bed_height_cm", "bed_height_m", "H_m"],
    "diameter": ["diameter", "D", "column_diameter_cm", "column_diameter_m", "D_col_m"],
    "temperature": ["temperature", "temperature_C", "T", "T_C"],
    "pressure_bar": ["pressure_bar", "P_bar", "P_in_bar"],
    "bulk_density": ["bulk_density", "bulk_density_g_cm3", "bulk_density_kg_m3"],
    "carbon_mass": ["carbon_mass", "carbon_mass_g", "carbon_mass_kg"],
    "KH_used_L_g": ["KH_used_L_g", "K_H_L_g", "KH_L_g", "K_H_used", "KH_T_L_g"],
    "kLDF_used_1_min": ["kLDF_used_1_min", "kLDF_1_min", "kLDF_used", "kLDF_T_1_min"],
    "Dax_used_cm2_min": ["Dax_used_cm2_min", "Dax_cm2_min", "Dax_used", "Dax_T_cm2_min"],
    "adsorbent_profile": ["adsorbent_profile", "adsorbent_profile_name", "sorbent_profile"],
    "isotherm_model": ["isotherm_model"],
    "scale_mode": ["scale_mode"],
}

CURVE_FEATURES = [
    "time_min",
    "C0_mg_L",
    "C_breakthrough_ratio",
    "flow",
    "height",
    "diameter",
    "temperature",
    "pressure_bar",
    "epsilon",
    "bulk_density",
    "carbon_mass",
    "EBCT_min",
    "KH_used_L_g",
    "kLDF_used_1_min",
    "Dax_used_cm2_min",
    "isotherm_model",
    "adsorbent_profile",
    "scale_mode",
]

BREAKTHROUGH_FEATURES = [col for col in CURVE_FEATURES if col != "time_min"]
CATEGORICAL_FEATURES = ["isotherm_model", "adsorbent_profile", "scale_mode"]
CURVE_TARGET = "C_out_C0"
BREAKTHROUGH_TARGET = "breakthrough_time_min"


def sklearn_available() -> bool:
    return all(
        item is not None
        for item in [
            joblib,
            ColumnTransformer,
            GroupShuffleSplit,
            Pipeline,
            OneHotEncoder,
            LinearRegression,
            RandomForestRegressor,
            ExtraTreesRegressor,
            HistGradientBoostingRegressor,
        ]
    )


def load_ml_dataset(path_or_file, file_type: str | None = None) -> pd.DataFrame:
    if isinstance(path_or_file, (str, os.PathLike)):
        path = str(path_or_file)
        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)
        return safe_read_csv(path)
    name = getattr(path_or_file, "name", "")
    if file_type == "xlsx" or name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path_or_file)
    return pd.read_csv(path_or_file, encoding="utf-8-sig")


def _coalesce_column(df: pd.DataFrame, target: str, aliases: list[str]) -> None:
    if target in df.columns:
        return
    for alias in aliases:
        if alias in df.columns:
            df[target] = df[alias]
            return


def normalize_scale_mode(value) -> str:
    text = str(value).strip().lower()
    if any(token in text for token in ("lab", "лаборатор")):
        return "laboratory"
    if any(token in text for token in ("engineer", "інженер", "инженер")):
        return "engineering"
    if any(token in text for token in ("industr", "промис", "промыш")):
        return "industrial"
    return text


def _as_bool_series(series: pd.Series) -> pd.Series:
    """Parse booleans robustly from bool, numeric and string CSV values."""
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "reached", "found", "да", "так", "истина"}
    false_values = {"false", "0", "no", "n", "not_found", "none", "nan", "", "нет", "ні", "ложь"}
    parsed = normalized.map(lambda value: True if value in true_values else False if value in false_values else np.nan)
    if parsed.isna().any():
        numeric = pd.to_numeric(series, errors="coerce")
        parsed = parsed.where(parsed.notna(), numeric.fillna(0).astype(float) != 0.0)
    return parsed.fillna(False).astype(bool)


def canonicalize_ml_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for target, aliases in CANONICAL_ALIASES.items():
        _coalesce_column(result, target, aliases)
    if "scale_mode" in result:
        result["scale_mode"] = result["scale_mode"].map(normalize_scale_mode)
    if "C_breakthrough_ratio" not in result.columns and {"C_breakthrough_mg_L", "C0_mg_L"}.issubset(result.columns):
        result["C_breakthrough_ratio"] = (
            pd.to_numeric(result["C_breakthrough_mg_L"], errors="coerce")
            / pd.to_numeric(result["C0_mg_L"], errors="coerce").replace(0, np.nan)
        )
    if "diameter" not in result.columns:
        if "column_area_cm2" in result.columns:
            area = pd.to_numeric(result["column_area_cm2"], errors="coerce")
            result["diameter"] = np.sqrt(4.0 * area / math.pi)
        elif "column_area_m2" in result.columns:
            area = pd.to_numeric(result["column_area_m2"], errors="coerce")
            result["diameter"] = np.sqrt(4.0 * area / math.pi)
    return result


def _derive_breakthrough_times(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    required = {"case_id", "time_min", "C_out_C0"}
    if not required.issubset(result.columns):
        return result
    target_was_present = BREAKTHROUGH_TARGET in result.columns
    found_was_present = "breakthrough_found" in result.columns
    if not target_was_present:
        result[BREAKTHROUGH_TARGET] = np.nan
    if not found_was_present:
        result["breakthrough_found"] = False
    for case_id, group in result.groupby("case_id"):
        ordered = group.sort_values("time_min")
        time = pd.to_numeric(ordered["time_min"], errors="coerce").to_numpy(dtype=float)
        ratio = pd.to_numeric(ordered["C_out_C0"], errors="coerce").to_numpy(dtype=float)
        if len(time) == 0:
            continue
        interpretation = interpret_breakthrough_curve(
            time,
            ratio,
            c0_mg_l=1.0,
            breakthrough_ratio=DEFAULT_BREAKTHROUGH_RATIO,
        )
        breakthrough_found = interpretation.breakthrough_bound_type == "exact"
        if (
            breakthrough_found
            and (
                not target_was_present
                or result.loc[result["case_id"] == case_id, BREAKTHROUGH_TARGET].isna().all()
            )
        ):
            result.loc[result["case_id"] == case_id, BREAKTHROUGH_TARGET] = (
                interpretation.breakthrough_time_min
            )
        if not found_was_present:
            result.loc[result["case_id"] == case_id, "breakthrough_found"] = breakthrough_found
        if not breakthrough_found:
            result.loc[result["case_id"] == case_id, "breakthrough_found"] = False
            result.loc[result["case_id"] == case_id, BREAKTHROUGH_TARGET] = np.nan
    if not found_was_present:
        result.attrs["breakthrough_found_reconstructed"] = True
    return result


def validate_ml_dataset(df: pd.DataFrame, dataset_mode: str = "synthetic") -> dict[str, Any]:
    canonical = canonicalize_ml_columns(df)
    required = ["case_id", "time_min", "C_out_C0", "C0_mg_L"]
    if dataset_mode == "experimental":
        required = ["case_id", "time_min", "C_out_C0"]
    missing = [col for col in required if col not in canonical.columns]
    optional = [
        col
        for col in [
            "C_out_mg_L",
            "flow",
            "height",
            "diameter",
            "temperature",
            "EBCT_min",
            "carbon_mass",
            "KH_used_L_g",
            "kLDF_used_1_min",
            "Dax_used_cm2_min",
            "breakthrough_time_min",
            "adsorbent_profile",
            "isotherm_model",
        ]
        if col in canonical.columns
    ]
    warnings = []
    if "case_id" in canonical.columns:
        n_cases = int(canonical["case_id"].nunique())
    else:
        n_cases = 0
    if "breakthrough_time_min" not in canonical.columns:
        warnings.append("breakthrough_time_min is missing; breakthrough model training will be unavailable.")
    if "C_breakthrough_ratio" not in canonical.columns:
        warnings.append("C_breakthrough_ratio is missing; it will be unavailable as an ML feature.")
    if n_cases < 3:
        warnings.append("Dataset has fewer than 3 unique case_id values; group train/test split may be unstable.")
    return {
        "missing_required_columns": missing,
        "optional_columns_found": optional,
        "number_of_cases": n_cases,
        "number_of_rows": int(len(canonical)),
        "warnings": warnings,
        "canonical_df": canonical,
    }


def validate_dataset_for_breakthrough_training(
    df: pd.DataFrame,
    required_service_time_h: float | None = None,
    training_mode: str = "validated",
) -> dict[str, Any]:
    canonical = canonicalize_ml_columns(df)
    canonical = _derive_breakthrough_times(canonical)
    errors: list[str] = []
    warnings: list[str] = []
    recommended_action: list[str] = []
    required_columns = ["case_id", "C0_mg_L", "flow", "height", "diameter", "temperature"]
    missing_columns = [col for col in required_columns if col not in canonical.columns]
    if missing_columns:
        errors.append("missing_required_features: " + ", ".join(missing_columns))
        recommended_action.append("Regenerate or upload a dataset with C0, flow, height, diameter and temperature columns.")
    if "case_id" not in canonical.columns:
        return {
            "dataset_ready": False,
            "warnings": warnings,
            "errors": errors or ["case_id is missing"],
            "recommended_action": recommended_action or ["Add case_id to the dataset."],
            "report": {},
            "canonical_df": canonical,
        }
    number_of_rows = int(len(canonical))
    number_of_cases = int(canonical["case_id"].nunique())
    duplicated_rows = int(canonical.duplicated(["case_id", "time_min"]).sum()) if "time_min" in canonical.columns else 0
    if duplicated_rows:
        warnings.append(f"duplicated_case_time_rows: {duplicated_rows}")
    if "breakthrough_found" not in canonical.columns or BREAKTHROUGH_TARGET not in canonical.columns:
        errors.append("breakthrough_found/breakthrough_time_min cannot be reconstructed")
        recommended_action.append("Generate a dataset with time_min and C_out_C0 curves, or include breakthrough_found and breakthrough_time_min.")
    found_by_case = (
        _as_bool_series(canonical.groupby("case_id")["breakthrough_found"].first())
        if "breakthrough_found" in canonical.columns
        else pd.Series(dtype=bool)
    )
    found_cases = set(found_by_case[found_by_case].index)
    not_found_cases = set(found_by_case[~found_by_case].index)
    case_targets = canonical[canonical["case_id"].isin(found_cases)].groupby("case_id")[BREAKTHROUGH_TARGET].first() if BREAKTHROUGH_TARGET in canonical.columns else pd.Series(dtype=float)
    case_targets = pd.to_numeric(case_targets, errors="coerce").dropna()
    n_found = int(case_targets.shape[0])
    n_not_found = int(len(not_found_cases))
    share_found = float(n_found / max(n_found + n_not_found, 1))
    unique_targets = int(case_targets.round(8).nunique()) if not case_targets.empty else 0
    critical_columns = [col for col in required_columns if col in canonical.columns]
    critical_nan = {col: int(pd.to_numeric(canonical[col], errors="coerce").isna().sum()) for col in critical_columns}
    critical_nan = {col: count for col, count in critical_nan.items() if count > 0}
    if critical_nan:
        errors.append("nan_in_critical_columns: " + ", ".join(f"{col}={count}" for col, count in critical_nan.items()))
    mode = "demo" if str(training_mode).lower().startswith("demo") else "validated"
    min_valid_cases = 100 if mode == "demo" else 1000
    recommended_valid_cases = 300 if mode == "demo" else 1000
    if n_found < 100:
        errors.append("too_few_valid_breakthrough_cases")
        recommended_action.append("Generate at least 300 valid breakthrough cases for demo mode or 1000+ for validated selection.")
    elif n_found < min_valid_cases:
        errors.append("too_few_valid_breakthrough_cases")
        recommended_action.append(f"Generate at least {min_valid_cases} valid breakthrough cases for {mode} training mode.")
    elif n_found < recommended_valid_cases:
        warnings.append("fewer_than_recommended_valid_breakthrough_cases")
        recommended_action.append(f"Increase valid breakthrough cases to at least {recommended_valid_cases} for a more stable model.")
    if unique_targets < 10:
        errors.append("too_few_distinct_target_values")
        recommended_action.append("Expand H/D/Q/t_end ranges so breakthrough_time_min has at least 10 distinct values.")
    if share_found < 0.70:
        warnings.append("share_breakthrough_found_below_0_70")
        recommended_action.append("Increase t_end_min or adjust ranges so 70-80% of cases reach breakthrough.")
    target_stats = {
        "min": float(case_targets.min()) if not case_targets.empty else np.nan,
        "median": float(case_targets.median()) if not case_targets.empty else np.nan,
        "p90": float(case_targets.quantile(0.90)) if not case_targets.empty else np.nan,
        "p95": float(case_targets.quantile(0.95)) if not case_targets.empty else np.nan,
        "max": float(case_targets.max()) if not case_targets.empty else np.nan,
        "unit": "min",
    }
    required_time_min = None if required_service_time_h is None else float(required_service_time_h) * 60.0
    if required_time_min is not None and math.isfinite(target_stats["max"]):
        if target_stats["max"] < required_time_min:
            errors.append("target_max_below_required_time")
            recommended_action.append("Generate a dataset with t_end and breakthrough targets above the required service time.")
        elif target_stats["max"] < 1.5 * required_time_min:
            warnings.append("target_max_less_than_1_5_required_time")
        if target_stats["p90"] < required_time_min and target_stats["p95"] < required_time_min:
            warnings.append("too_few_long_breakthrough_cases_near_required_time")
            recommended_action.append("Generate more long-service cases near the required time.")
    scale_counts = canonical.groupby("case_id")["scale_mode"].first().value_counts().to_dict() if "scale_mode" in canonical.columns else {}
    if len(scale_counts) > 1 and min(scale_counts.values()) / max(sum(scale_counts.values()), 1) < 0.10:
        warnings.append("scale_mode_distribution_is_imbalanced")
    report = {
        "number_of_rows": number_of_rows,
        "number_of_cases": number_of_cases,
        "number_of_breakthrough_found_cases": n_found,
        "number_of_not_found_cases": n_not_found,
        "share_breakthrough_found": share_found,
        "unique_target_values": unique_targets,
        "duplicated_case_id_time_rows": duplicated_rows,
        "critical_nan": critical_nan,
        "target_stats": target_stats,
        "feature_ranges": feature_ranges(canonical[[col for col in BREAKTHROUGH_FEATURES if col in canonical.columns]]) if number_of_rows else {},
        "scale_mode_counts": scale_counts,
        "breakthrough_found_reconstructed": bool(canonical.attrs.get("breakthrough_found_reconstructed", False)),
        "dataset_quality_mode": "normal" if mode == "validated" and n_found >= 1000 else "demo" if mode == "demo" and n_found >= 100 else "invalid",
        "training_mode": mode,
        "minimum_valid_breakthrough_cases": min_valid_cases,
        "recommended_valid_breakthrough_cases": recommended_valid_cases,
    }
    if not recommended_action and not errors:
        recommended_action.append("Dataset is usable for breakthrough-model training.")
    return {
        "dataset_ready": len(errors) == 0,
        "warnings": warnings,
        "errors": errors,
        "recommended_action": recommended_action,
        "report": report,
        "canonical_df": canonical,
    }


def _target_to_model_space(y: pd.Series | np.ndarray, target_transform: str) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if target_transform == "log1p":
        return np.log1p(np.maximum(values, 0.0))
    return values


def _target_from_model_space(y: pd.Series | np.ndarray, target_transform: str) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if target_transform == "log1p":
        return np.expm1(values)
    return values


def _metrics(y_true, y_pred, task_type: str = "curve") -> dict[str, float | str]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if not np.isfinite(y_pred).all():
        return {"MAE": float("inf"), "RMSE": float("inf"), "R2": float("nan"),
                "MedianAE": float("inf"), "nonfinite_predictions": int((~np.isfinite(y_pred)).sum())}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
    median_ae = float(np.median(np.abs(y_true - y_pred)))
    result: dict[str, float | str] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MedianAE": median_ae}
    if task_type == "curve":
        clipped = np.clip(y_pred, 0.0, 1.0)
        result["raw_RMSE"] = rmse
        result["raw_MAE"] = mae
        result["raw_R2"] = r2
        result["clipped_RMSE"] = float(np.sqrt(mean_squared_error(y_true, clipped)))
        result["clipped_MAE"] = float(mean_absolute_error(y_true, clipped))
        result["clipped_R2"] = float(r2_score(y_true, clipped))
        result["raw_predictions_outside_0_1"] = int(((y_pred < 0) | (y_pred > 1)).sum())
        result["clipping_policy"] = "reported_only_0_1_no_hidden_clipping"
        denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        mask = denom > 1e-12
        result["SMAPE_percent"] = (
            float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100.0)
            if mask.any()
            else float("nan")
        )
    elif task_type == "breakthrough":
        valid_for_mape = y_true > 60.0
        if valid_for_mape.sum() >= 3:
            result["MAPE_percent_filtered_y_gt_60_min"] = float(
                np.mean(np.abs((y_true[valid_for_mape] - y_pred[valid_for_mape]) / y_true[valid_for_mape])) * 100.0
            )
        else:
            result["MAPE_percent_filtered_y_gt_60_min"] = "not reliable"
        result["RMSE_log1p"] = float(np.sqrt(mean_squared_error(np.log1p(np.maximum(y_true, 0.0)), np.log1p(np.maximum(y_pred, 0.0)))))
    return result


def _feature_frame(df: pd.DataFrame, task_type: str) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], str, dict[str, Any]]:
    canonical = canonicalize_ml_columns(df)
    info: dict[str, Any] = {"warnings": []}
    if task_type == "curve":
        target = CURVE_TARGET
        features = [col for col in CURVE_FEATURES if col in canonical.columns]
        data = canonical.dropna(subset=["case_id", target]).copy()
    elif task_type == "breakthrough":
        target = BREAKTHROUGH_TARGET
        canonical = _derive_breakthrough_times(canonical)
        info["breakthrough_found_reconstructed"] = bool(canonical.attrs.get("breakthrough_found_reconstructed", False))
        if target not in canonical.columns:
            raise ValueError(f"Target column {target} is missing.")
        if "breakthrough_found" in canonical.columns:
            found_by_case = _as_bool_series(canonical.groupby("case_id")["breakthrough_found"].first())
            valid_cases = set(found_by_case[found_by_case].index)
            excluded_cases = int(len(found_by_case) - len(valid_cases))
            canonical = canonical[canonical["case_id"].isin(valid_cases)].copy()
            info["excluded_not_found_cases_count"] = excluded_cases
            info["n_breakthrough_found_cases"] = int(len(valid_cases))
            info["n_breakthrough_not_found_cases"] = int(excluded_cases)
            info["used_breakthrough_found_filter"] = True
            if info["breakthrough_found_reconstructed"]:
                info["warnings"].append("breakthrough_found reconstructed from C_out/C0 time curves.")
        else:
            info["excluded_not_found_cases_count"] = 0
            info["n_breakthrough_found_cases"] = int(canonical["case_id"].nunique()) if "case_id" in canonical.columns else 0
            info["n_breakthrough_not_found_cases"] = 0
            info["used_breakthrough_found_filter"] = False
            info["warnings"].append("breakthrough_found is missing; breakthrough-time training uses all derived case targets.")
        features = [col for col in BREAKTHROUGH_FEATURES if col in canonical.columns]
        data = canonical.dropna(subset=["case_id", target]).copy()
        sort_cols = ["case_id"]
        data = data.sort_values(sort_cols).groupby("case_id", as_index=False).first()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    if target not in data.columns:
        raise ValueError(f"Target column {target} is missing.")
    if "case_id" not in data.columns:
        raise ValueError("case_id is required for group train/test split.")
    if not features:
        raise ValueError("No usable feature columns found.")
    X = data[features].copy()
    y = pd.to_numeric(data[target], errors="coerce")
    keep = y.notna()
    X = X.loc[keep]
    y = y.loc[keep]
    groups = data.loc[keep, "case_id"]
    return X, y, groups, features, target, info


def _build_pipeline(model, X: pd.DataFrame) -> Pipeline:
    categorical = [col for col in X.columns if col in CATEGORICAL_FEATURES or X[col].dtype == "object"]
    numeric = [col for col in X.columns if col not in categorical]
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def _candidate_models(task_type: str = "breakthrough") -> dict[str, Any]:
    if task_type == "curve":
        return {
            "linear_regression": LinearRegression(),
            "baseline_random_forest": RandomForestRegressor(
                n_estimators=60, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=80, max_depth=16, min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=80, max_depth=16, min_samples_leaf=3, random_state=42, n_jobs=-1
            ),
            "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
        }
    return {
        "linear_regression": LinearRegression(),
        "baseline_random_forest": RandomForestRegressor(
            n_estimators=80, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=160, max_depth=24, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=160, max_depth=24, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=42),
    }


def _group_split(X: pd.DataFrame, y: pd.Series, groups: pd.Series, test_size: float = 0.25):
    unique_cases = pd.Series(groups).drop_duplicates()
    if len(unique_cases) < 3:
        raise ValueError("Too few case_id groups; splitting curve rows is forbidden.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    return next(splitter.split(X, y, groups))


def feature_ranges(X: pd.DataFrame) -> dict[str, dict[str, Any]]:
    result = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            values = pd.to_numeric(X[col], errors="coerce")
            result[col] = {
                "min": float(values.min()),
                "median": float(values.median()),
                "max": float(values.max()),
            }
        else:
            result[col] = {"values": sorted(map(str, pd.Series(X[col]).dropna().unique()[:50]))}
    return result


def infer_feature_units(X: pd.DataFrame, feature_columns: list[str], scale_values: list[str] | None = None, source_df: pd.DataFrame | None = None) -> dict[str, str]:
    units: dict[str, str] = {}
    scale_text = " ".join(scale_values or []).lower()
    lab_like = any(token in scale_text for token in ["lab", "лабо", "лаборатор", "laboratory"])
    for col in feature_columns:
        unit = "-"
        if col == "C0_mg_L":
            unit = "mg/L"
        elif col == "C_breakthrough_ratio":
            unit = "-"
        elif col == "flow":
            unit = "ml/min" if lab_like else "m3/h"
        elif col == "height":
            unit = "cm" if lab_like else "m"
        elif col == "diameter":
            unit = "cm" if lab_like else "m"
        elif col == "temperature":
            unit = "degC"
        elif col == "pressure_bar":
            unit = "bar"
        elif col == "bulk_density":
            values = pd.to_numeric(X[col], errors="coerce") if col in X.columns else pd.Series(dtype=float)
            unit = "g/cm3" if values.dropna().max() < 10 else "kg/m3"
        elif col == "carbon_mass":
            values = pd.to_numeric(X[col], errors="coerce") if col in X.columns else pd.Series(dtype=float)
            unit = "kg" if not lab_like or values.dropna().max() < 1 else "g"
        elif col == "EBCT_min":
            unit = "min"
        elif col == "KH_used_L_g":
            unit = "L/g"
        elif col == "kLDF_used_1_min":
            unit = "1/min"
        elif col == "Dax_used_cm2_min":
            unit = "cm2/min"
        units[col] = unit
    # Prefer the actual source column's unit to legacy magnitude heuristics.
    source_alias_units = {
        "flow": [("flow_ml_min", "ml/min"), ("Q_m3_h", "m3/h"), ("flow_m3_h", "m3/h")],
        "height": [("bed_height_cm", "cm"), ("height_cm", "cm"), ("bed_height_m", "m"), ("height_m", "m")],
        "diameter": [("column_diameter_cm", "cm"), ("diameter_cm", "cm"), ("column_diameter_m", "m"), ("diameter_m", "m")],
        "carbon_mass": [("carbon_mass_g", "g"), ("carbon_mass_kg", "kg")],
        "bulk_density": [("bulk_density_g_cm3", "g/cm3"), ("bulk_density_kg_m3", "kg/m3")],
    }
    if source_df is not None:
        for feature, aliases in source_alias_units.items():
            if feature not in feature_columns:
                continue
            for alias, unit in aliases:
                if alias in source_df:
                    if feature not in source_df or np.allclose(
                        pd.to_numeric(source_df[feature], errors="coerce"),
                        pd.to_numeric(source_df[alias], errors="coerce"), equal_nan=True):
                        units[feature] = unit
                        break
    return units


def normalize_units_for_ml(
    candidate_df: pd.DataFrame,
    metadata: dict[str, Any] | None,
    scale_mode: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert candidate features to the same units used when the ML model was trained."""
    metadata = metadata or {}
    feature_columns = list(metadata.get("feature_columns") or [])
    feature_units = dict(metadata.get("feature_units") or {})
    normalized = normalize_candidate_features(candidate_df, feature_columns)
    if "scale_mode" in normalized:
        allowed = (metadata.get("feature_ranges") or {}).get("scale_mode", {}).get("values", [])
        trained_labels = {normalize_scale_mode(value): value for value in allowed}
        normalized["scale_mode"] = normalized["scale_mode"].map(
            lambda value: trained_labels.get(normalize_scale_mode(value), normalize_scale_mode(value)))
    diagnostics: dict[str, Any] = {
        "feature_units": feature_units,
        "candidate_model_units": {},
        "candidate_display_units": {},
        "detected_unit_mismatches": [],
        "applied_unit_conversions": [],
    }
    scale_text = str(scale_mode or "").lower()
    lab_like = any(token in scale_text for token in ["lab", "лабо", "лаборатор", "laboratory"])

    if "carbon_mass_g" in candidate_df.columns:
        diagnostics["candidate_display_units"]["carbon_mass_g"] = "g"
    if "carbon_mass_kg" in candidate_df.columns:
        diagnostics["candidate_display_units"]["carbon_mass_kg"] = "kg"

    if "carbon_mass" in feature_columns:
        target_unit = feature_units.get("carbon_mass")
        if not target_unit:
            ranges = (metadata.get("feature_ranges") or {}).get("carbon_mass") or {}
            try:
                model_max = float(ranges.get("max"))
                candidate_min = float(pd.to_numeric(normalized.get("carbon_mass"), errors="coerce").min())
                if model_max < 0.1 and candidate_min > 1.0:
                    target_unit = "kg"
                    diagnostics["detected_unit_mismatches"].append("legacy_carbon_mass_kg_vs_g")
                elif model_max > 1.0:
                    target_unit = "g"
            except Exception:
                target_unit = "g" if lab_like else "kg"
        diagnostics["candidate_model_units"]["carbon_mass"] = target_unit
        if target_unit == "kg":
            if "carbon_mass_g" in candidate_df.columns:
                normalized["carbon_mass"] = pd.to_numeric(candidate_df["carbon_mass_g"], errors="coerce") / 1000.0
                diagnostics["applied_unit_conversions"].append("carbon_mass_g -> carbon_mass kg")
            elif "carbon_mass_kg" in candidate_df.columns:
                normalized["carbon_mass"] = pd.to_numeric(candidate_df["carbon_mass_kg"], errors="coerce")
            elif "carbon_mass" in normalized.columns:
                values = pd.to_numeric(normalized["carbon_mass"], errors="coerce")
                if lab_like:
                    normalized["carbon_mass"] = values / 1000.0
                    diagnostics["applied_unit_conversions"].append("legacy carbon_mass g -> kg")
        elif target_unit == "g":
            if "carbon_mass_kg" in candidate_df.columns:
                normalized["carbon_mass"] = pd.to_numeric(candidate_df["carbon_mass_kg"], errors="coerce") * 1000.0
                diagnostics["applied_unit_conversions"].append("carbon_mass_kg -> carbon_mass g")
    return normalized, diagnostics


def _cross_validate_selected_model(
    model_name: str,
    target_transform: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    task_type: str,
    n_splits: int = 3,
) -> dict[str, Any]:
    unique_cases = pd.Series(groups).nunique()
    if unique_cases < 30 or GroupShuffleSplit is None:
        return {"available": False, "warning": "Dataset is too small for stable group cross-validation."}
    base_name = str(model_name).split(":", 1)[-1]
    if base_name.startswith("optuna_"):
        base_name = base_name.replace("optuna_", "", 1)
    factory = _candidate_models(task_type)
    if base_name not in factory:
        return {"available": False, "warning": "Selected model cannot be cross-validated by factory."}
    splitter = GroupShuffleSplit(n_splits=int(n_splits), test_size=0.25, random_state=123)
    rmse_values = []
    r2_values = []
    mae_values = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        pipeline = _build_pipeline(factory[base_name], X)
        pipeline.fit(X.iloc[train_idx], _target_to_model_space(y.iloc[train_idx], target_transform))
        pred = _target_from_model_space(pipeline.predict(X.iloc[test_idx]), target_transform)
        metric = _metrics(y.iloc[test_idx], pred, task_type=task_type)
        rmse_values.append(float(metric["RMSE"]))
        mae_values.append(float(metric["MAE"]))
        r2_values.append(float(metric["R2"]))
    return {
        "available": True,
        "n_splits": int(n_splits),
        "mean_RMSE": float(np.mean(rmse_values)),
        "std_RMSE": float(np.std(rmse_values, ddof=1)) if len(rmse_values) > 1 else 0.0,
        "mean_MAE": float(np.mean(mae_values)),
        "std_MAE": float(np.std(mae_values, ddof=1)) if len(mae_values) > 1 else 0.0,
        "mean_R2": float(np.mean(r2_values)),
        "std_R2": float(np.std(r2_values, ddof=1)) if len(r2_values) > 1 else 0.0,
    }


def _model_family(model_name: str) -> str:
    return "linear" if "linear_regression" in str(model_name) else "tree"


def _breakthrough_quality_status(
    model_name: str,
    metric: dict[str, Any],
    predictions: np.ndarray,
    target_max: float,
    n_found_cases: int,
    dataset_quality_mode: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    pred = np.asarray(predictions, dtype=float)
    finite = np.isfinite(pred)
    r2 = float(metric.get("R2", float("nan")))
    rmse = float(metric.get("RMSE", float("inf")))
    mae = float(metric.get("MAE", float("inf")))
    role = "baseline" if "linear_regression" in str(model_name) or "baseline_random_forest" in str(model_name) else "candidate"
    if role == "baseline":
        reasons.append("baseline_only")
    if n_found_cases < 100:
        reasons.append("too_few_breakthrough_cases")
    elif dataset_quality_mode != "demo" and n_found_cases < 1000:
        reasons.append("demo_dataset_not_validated_for_selection")
    r2_threshold = 0.75 if dataset_quality_mode == "normal" else 0.40
    if not math.isfinite(r2) or r2 < r2_threshold:
        reasons.append("poor_r2")
    if not math.isfinite(rmse) or (math.isfinite(target_max) and target_max > 0 and rmse > 0.25 * target_max):
        reasons.append("poor_rmse")
    if not math.isfinite(mae) or (math.isfinite(target_max) and target_max > 0 and mae > 0.20 * target_max):
        reasons.append("poor_mae")
    if len(pred) == 0 or not finite.all():
        reasons.append("non_finite_predictions")
    else:
        if float(np.nanmin(pred)) < 0:
            reasons.append("negative_predictions")
        if math.isfinite(target_max) and target_max > 0 and float(np.nanmax(pred)) > 1.5 * target_max:
            reasons.append("prediction_above_target_domain")
    hard_reasons = [reason for reason in reasons if not str(reason).startswith("demo_mode_")]
    can_use = len(hard_reasons) == 0
    return {
        "model": model_name,
        "role": role,
        "selection_quality_status": "ready" if can_use else "failed",
        "can_be_used_for_selection": bool(can_use),
        "quality_reasons": reasons,
        "model_family": _model_family(model_name),
        "extrapolation_allowed": False,
    }


def train_models(
    df: pd.DataFrame,
    task_type: str,
    dataset_filename: str = "dataset",
    output_dir: str = "ml_models",
    use_optuna: bool = False,
    n_trials: int = 30,
    target_transform: str | None = None,
    training_mode: str = "validated",
    generation_provenance: dict | None = None,
) -> dict[str, Any]:
    if not sklearn_available():
        return {"success": False, "message": "scikit-learn/joblib is not available.", "metrics": {}}
    dataset_quality = {}
    if task_type == "breakthrough":
        dataset_validation = validate_dataset_for_breakthrough_training(df, training_mode=training_mode)
        dataset_quality = dataset_validation.get("report", {})
        if not dataset_validation.get("dataset_ready"):
            return {
                "success": False,
                "message": "Dataset is insufficient for breakthrough-model training.",
                "metrics": {},
                "dataset_quality": dataset_quality,
                "errors": dataset_validation.get("errors", []),
                "warnings": dataset_validation.get("warnings", []),
                "recommended_action": dataset_validation.get("recommended_action", []),
            }
    try:
        X, y, groups, feature_columns, target_column, training_info = _feature_frame(df, task_type)
    except Exception as exc:
        return {"success": False, "message": str(exc), "metrics": {}}
    if len(X) < 10:
        return {"success": False, "message": "At least 10 rows are required for training.", "metrics": {}}
    from ml_validation import grouped_train_validation_test
    train_idx, validation_idx, test_idx = grouped_train_validation_test(groups)
    development_idx = np.sort(np.r_[train_idx, validation_idx])
    X_train, X_validation = X.iloc[train_idx], X.iloc[validation_idx]
    y_train, y_validation = y.iloc[train_idx], y.iloc[validation_idx]
    metrics = {}
    fitted_models = {}
    prediction_cache: dict[str, np.ndarray] = {}
    model_transforms: dict[str, str] = {}
    transform_candidates = ["none", "log1p"] if task_type == "breakthrough" else [target_transform or "none"]
    for transform in transform_candidates:
        y_train_model = _target_to_model_space(y_train, transform)
        for name, model in _candidate_models(task_type).items():
            metric_name = f"{transform}:{name}" if task_type == "breakthrough" else name
            pipeline = _build_pipeline(model, X_train)
            pipeline.fit(X_train, y_train_model)
            pred_model = pipeline.predict(X_validation)
            pred = _target_from_model_space(pred_model, transform)
            metrics[metric_name] = _metrics(y_validation, pred, task_type=task_type)
            fitted_models[metric_name] = pipeline
            prediction_cache[metric_name] = pred
            model_transforms[metric_name] = transform
    best_name_raw = sorted(metrics, key=lambda name: (metrics[name]["RMSE"], 0 if name.endswith(":linear_regression") or name == "linear_regression" else 1))[0]
    best_name = best_name_raw
    quality_by_model: dict[str, Any] = {}
    target_max_for_quality = float(pd.to_numeric(y.iloc[development_idx], errors="coerce").max())
    n_found_cases = int(training_info.get("n_breakthrough_found_cases", pd.Series(groups).nunique()))
    dataset_quality_mode = str((dataset_quality or {}).get("dataset_quality_mode", "normal"))
    if task_type == "breakthrough":
        for name in metrics:
            quality_by_model[name] = _breakthrough_quality_status(
                name,
                metrics[name],
                prediction_cache.get(name, np.array([])),
                target_max_for_quality,
                n_found_cases,
                dataset_quality_mode,
            )
        ready_candidates = [
            name for name, quality in quality_by_model.items()
            if quality.get("can_be_used_for_selection")
        ]
        nonlinear_candidates = [
            name for name, quality in quality_by_model.items()
            if quality.get("role") == "candidate"
        ]
        if ready_candidates:
            best_name = sorted(ready_candidates, key=lambda name: (metrics[name]["RMSE"], -metrics[name].get("R2", -999.0)))[0]
        elif nonlinear_candidates:
            best_name = sorted(nonlinear_candidates, key=lambda name: (metrics[name]["RMSE"], -metrics[name].get("R2", -999.0)))[0]
        else:
            best_name = best_name_raw
    target_transform = model_transforms.get(best_name, target_transform or "none")
    best_model = fitted_models[best_name]
    best_base_name = best_name.split(":", 1)[-1]
    if use_optuna and optuna is not None and best_base_name in {"random_forest", "extra_trees", "hist_gradient_boosting"}:
        optimized = optimize_with_optuna(X_train, y_train, X_validation, y_validation, X_train, best_base_name, int(n_trials), task_type, target_transform)
        if optimized.get("success") and optimized["metrics"]["RMSE"] <= metrics[best_name]["RMSE"]:
            best_name = f"{target_transform}:optuna_{best_base_name}" if task_type == "breakthrough" else f"optuna_{best_base_name}"
            best_model = optimized["model"]
            metrics[best_name] = optimized["metrics"]
            prediction_cache[best_name] = np.asarray(optimized.get("predictions", []), dtype=float)
            model_transforms[best_name] = target_transform
            if task_type == "breakthrough":
                quality_by_model[best_name] = _breakthrough_quality_status(
                    best_name,
                    metrics[best_name],
                    prediction_cache.get(best_name, np.array([])),
                    target_max_for_quality,
                    n_found_cases,
                    dataset_quality_mode,
                )
    selection_quality = quality_by_model.get(best_name, {"can_be_used_for_selection": task_type != "breakthrough", "quality_reasons": [], "model_family": _model_family(best_name), "extrapolation_allowed": False})
    if task_type == "breakthrough":
        for name, quality in quality_by_model.items():
            metrics[name]["selection_quality_status"] = quality.get("selection_quality_status")
            metrics[name]["can_be_used_for_selection"] = bool(quality.get("can_be_used_for_selection"))
            metrics[name]["quality_reasons"] = ", ".join(quality.get("quality_reasons", []))
            metrics[name]["role"] = quality.get("role")
            metrics[name]["target_transform"] = model_transforms.get(name)
        selection_ready = bool(selection_quality.get("can_be_used_for_selection"))
        selection_block_reason = "" if selection_ready else "Poor breakthrough model quality: " + ", ".join(selection_quality.get("quality_reasons", []))
    else:
        selection_ready = False
        selection_block_reason = "Curve model is not used for direct adsorber selection."
    cross_validation_metrics = _cross_validate_selected_model(
        best_name,
        target_transform,
        X.iloc[development_idx],
        y.iloc[development_idx],
        groups.iloc[development_idx],
        task_type,
    )
    # Selection is now frozen. Refit on development groups and touch the outer test once.
    best_model.fit(X.iloc[development_idx], _target_to_model_space(y.iloc[development_idx], target_transform))
    test_prediction = _target_from_model_space(best_model.predict(X.iloc[test_idx]), target_transform)
    final_test_metrics = _metrics(y.iloc[test_idx], test_prediction, task_type=task_type)
    if task_type == "breakthrough":
        test_quality = _breakthrough_quality_status(best_name, final_test_metrics, test_prediction,
            target_max_for_quality, n_found_cases, dataset_quality_mode)
        if not test_quality["can_be_used_for_selection"]:
            selection_ready = False
            selection_block_reason = "Final held-out test failed: " + ", ".join(test_quality["quality_reasons"])
    from model import MODEL_VERSION
    source_versions = sorted(map(str, df.get("mechanistic_model_id", pd.Series(dtype=str)).dropna().unique()))
    from scientific_provenance import resolve_dataset_provenance, dataframe_hash, DATAFRAME_HASH_ALGORITHM
    verified_provenance = resolve_dataset_provenance(df, generation_provenance)
    source_identified = bool(verified_provenance.get("generation_source"))
    if task_type == "breakthrough" and not source_identified:
        selection_ready = False
        selection_block_reason = "Dataset origin cannot be verified against a compatible generation manifest; selection is blocked."
    for metric in metrics.values():
        metric["evaluation_split"] = "validation"
    from scientific_provenance import current_provenance
    os.makedirs(output_dir, exist_ok=True)
    baseline_metrics = {
        name: metric for name, metric in metrics.items()
        if "linear_regression" in name or "baseline_random_forest" in name
    }
    candidate_model_metrics = {
        name: metric for name, metric in metrics.items()
        if name not in baseline_metrics
    }
    excluded_models_with_reasons = {
        name: metric.get("quality_reasons", "")
        for name, metric in metrics.items()
        if not bool(metric.get("can_be_used_for_selection", task_type != "breakthrough"))
    }
    model_file = "curve_model.joblib" if task_type == "curve" else "breakthrough_model.joblib"
    metadata_file = "curve_model_metadata.json" if task_type == "curve" else "breakthrough_model_metadata.json"
    bundle_file = "curve_model_bundle.joblib" if task_type == "curve" else "breakthrough_model_bundle.joblib"
    report_file = "curve_model_report.json" if task_type == "curve" else "breakthrough_model_report.json"
    metrics_file = "curve_model_metrics.csv" if task_type == "curve" else "breakthrough_model_metrics.csv"
    dataset_quality_file = "dataset_quality_report.json"
    ranges_for_metadata = feature_ranges(X.iloc[development_idx])
    scale_values = []
    if "scale_mode" in X.columns:
        scale_values = [str(value) for value in pd.Series(X["scale_mode"]).dropna().unique()[:20]]
    feature_units = infer_feature_units(X, feature_columns, scale_values, source_df=df)
    sorption_parameter_stats = {
        key: ranges_for_metadata.get(key)
        for key in ["KH_used_L_g", "kLDF_used_1_min", "Dax_used_cm2_min"]
        if key in ranges_for_metadata
    }
    metadata = {
        "model_type": best_name,
        "baseline_best_model_type": best_name_raw,
        "task_type": task_type,
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "number_of_rows": int(len(X)),
        "number_of_cases": int(pd.Series(groups).nunique()),
        "feature_columns": feature_columns,
        "feature_units": feature_units,
        "target_column": target_column,
        "target_unit": "min" if task_type == "breakthrough" else "C/C0",
        "target_transform": target_transform,
        "target_transform_candidates": transform_candidates,
        "selected_target_transform": target_transform,
        "feature_importance_target_space": target_transform,
        "feature_importance_scope": "descriptive_development_refit_not_independent_validation_or_causality",
        "recommended_use": "adsorber_selection" if task_type == "breakthrough" else "curve_analysis",
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "candidate_model_metrics": candidate_model_metrics,
        "excluded_models_with_reasons": excluded_models_with_reasons,
        "selection_model_metrics": final_test_metrics,
        "final_test_metrics": final_test_metrics,
        "metric_scope": "candidate_table_validation_selected_summary_final_test",
        "validation_scope": "synthetic_surrogate_accuracy_not_independent_experimental_validation",
        "validation_metrics": metrics.get(best_name, {}),
        "cross_validation_metrics": cross_validation_metrics,
        "cross_validation_scope": "development_only_post_selection_diagnostic_not_outer_test",
        "selection_ready": selection_ready,
        "suitable_for_adsorber_selection": bool(selection_ready and task_type == "breakthrough"),
        "selection_block_reason": selection_block_reason,
        "selected_model": best_name if selection_ready else None,
        "why_selected": (
            f"{best_name} passed the nonlinear model quality gate and had the lowest RMSE among ready candidates."
            if selection_ready
            else ""
        ),
        "why_not_linear_regression": "LinearRegression is used only as a baseline and is not selected as the final adsorber-selection model.",
        "model_family": selection_quality.get("model_family", _model_family(best_name)),
        "extrapolation_allowed": bool(selection_quality.get("extrapolation_allowed", False)),
        "quality_reasons": selection_quality.get("quality_reasons", []),
        "train_case_ids_count": int(pd.Series(groups.iloc[train_idx]).nunique()),
        "test_case_ids_count": int(pd.Series(groups.iloc[test_idx]).nunique()),
        "validation_case_ids_count": int(pd.Series(groups.iloc[validation_idx]).nunique()),
        "split_case_ids": {key: list(map(str, pd.Series(groups.iloc[idx]).unique())) for key, idx in [("train", train_idx), ("validation", validation_idx), ("test", test_idx)]},
        "dataset_filename": dataset_filename,
        "source_model_versions": source_versions,
        "dataset_provenance_status": "identified_current_simulator" if source_identified else "unverified_or_obsolete_origin",
        "program_version": PROGRAM_VERSION,
        "dataset_quality": dataset_quality,
        "training_mode": str(training_mode),
        "feature_ranges": ranges_for_metadata,
        "sorption_parameter_source": "training_dataset",
        "sorption_parameter_stats": sorption_parameter_stats,
        "sorption_parameter_provenance": dataset_parameter_provenance(df),
        "target_stats": {
            "min": float(pd.to_numeric(y.iloc[development_idx], errors="coerce").min()),
            "median": float(pd.to_numeric(y.iloc[development_idx], errors="coerce").median()),
            "max": float(pd.to_numeric(y.iloc[development_idx], errors="coerce").max()),
            "p90": float(pd.to_numeric(y.iloc[development_idx], errors="coerce").quantile(0.90)),
            "p95": float(pd.to_numeric(y.iloc[development_idx], errors="coerce").quantile(0.95)),
            "unit": "min",
        },
        "training_warnings": training_info.get("warnings", []),
        "excluded_not_found_cases_count": training_info.get("excluded_not_found_cases_count", 0),
        "n_breakthrough_found_cases": training_info.get("n_breakthrough_found_cases", 0),
        "n_breakthrough_not_found_cases": training_info.get("n_breakthrough_not_found_cases", 0),
        "used_only_found_breakthrough_cases": bool(training_info.get("used_breakthrough_found_filter", False)),
        "breakthrough_found_reconstructed": bool(training_info.get("breakthrough_found_reconstructed", False)),
        "used_breakthrough_found_filter": training_info.get("used_breakthrough_found_filter", False),
        "saved_model_file": model_file,
        "saved_metadata_file": metadata_file,
        "saved_bundle_file": bundle_file if task_type == "breakthrough" else None,
        "saved_report_file": report_file,
        "saved_metrics_file": metrics_file,
        "saved_dataset_quality_file": dataset_quality_file,
    }
    from model import DEFAULT_N_CELLS
    metadata["training_environment"] = current_provenance()
    metadata.update(verified_provenance)
    metadata["dataset_dataframe_sha256"] = dataframe_hash(df)
    metadata["dataset_dataframe_sha256_algorithm"] = DATAFRAME_HASH_ALGORITHM
    metadata["dataset_sha256"] = metadata["dataset_dataframe_sha256"]
    metadata["dataset_sha256_legacy_alias_of"] = "dataset_dataframe_sha256"
    if source_identified:
        metadata["dataset_csv_sha256"] = verified_provenance["generation_source"]["dataset_csv_sha256"]
        metadata["dataset_csv_sha256_algorithm"] = "SHA-256(file_bytes)"
    metadata_hash_payload = {
        "task_type": metadata.get("task_type"),
        "target_column": metadata.get("target_column"),
        "model_type": metadata.get("model_type"),
        "feature_columns": metadata.get("feature_columns"),
        "feature_units": metadata.get("feature_units"),
        "target_stats": metadata.get("target_stats"),
        "training_date": metadata.get("training_date"),
        "dataset_filename": metadata.get("dataset_filename"),
    }
    metadata["metadata_hash"] = hashlib.sha256(
        json.dumps(metadata_hash_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    metadata["model_id"] = f"{task_type}_{metadata['metadata_hash']}"
    if joblib is not None:
        joblib.dump(best_model, os.path.join(output_dir, model_file), compress=3)
        if task_type == "breakthrough":
            joblib.dump(
                {"model": best_model, "metadata": metadata, "bundle_version": "1.0"},
                os.path.join(output_dir, bundle_file),
                compress=3,
            )
    with open(os.path.join(output_dir, metadata_file), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, report_file), "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_model": metadata.get("selected_model"),
                "why_selected": metadata.get("why_selected"),
                "why_not_linear_regression": metadata.get("why_not_linear_regression"),
                "quality_status": "ready" if metadata.get("selection_ready") else "failed",
                "selection_block_reason": metadata.get("selection_block_reason"),
                "allowed_prediction_range": metadata.get("target_stats"),
                "target_max_h": (metadata.get("target_stats") or {}).get("max", 0) / 60.0 if (metadata.get("target_stats") or {}).get("max") is not None else None,
                "suitable_for_required_time": None,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "model"}).to_csv(
        os.path.join(output_dir, metrics_file),
        index=False,
        encoding="utf-8-sig",
    )
    if task_type == "breakthrough":
        with open(os.path.join(output_dir, dataset_quality_file), "w", encoding="utf-8") as f:
            json.dump(dataset_quality, f, ensure_ascii=False, indent=2)
    importances = compute_feature_importance(best_model, X_validation, y_validation, feature_columns,
                                            target_transform=target_transform)
    return {
        "success": True,
        "message": "Training finished.",
        "task_type": task_type,
        "best_model_name": best_name,
        "best_model": best_model,
        "metrics": metrics,
        "metadata": metadata,
        "feature_importance": importances,
        "candidate_model_quality": quality_by_model,
        "model_path": os.path.join(output_dir, model_file),
        "metadata_path": os.path.join(output_dir, metadata_file),
        "bundle_path": os.path.join(output_dir, bundle_file),
        "warnings": training_info.get("warnings", []),
    }


def optimize_with_optuna(X_train, y_train, X_test, y_test, X_all, model_name: str, n_trials: int, task_type: str, target_transform: str) -> dict[str, Any]:
    if optuna is None:
        return {"success": False, "message": "Optuna is not installed."}

    def objective(trial):
        if "extra_trees" in model_name:
            model = ExtraTreesRegressor(
                n_estimators=trial.suggest_int("n_estimators", 60, 260),
                max_depth=trial.suggest_int("max_depth", 3, 40),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                random_state=42,
                n_jobs=-1,
            )
        elif "hist_gradient" in model_name:
            model = HistGradientBoostingRegressor(
                max_iter=trial.suggest_int("max_iter", 60, 260),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 8, 64),
                l2_regularization=trial.suggest_float("l2_regularization", 1e-8, 1e-1, log=True),
                random_state=42,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 60, 260),
                max_depth=trial.suggest_int("max_depth", 3, 40),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 12),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                random_state=42,
                n_jobs=-1,
            )
        pipeline = _build_pipeline(model, X_all)
        pipeline.fit(X_train, _target_to_model_space(y_train, target_transform))
        pred = _target_from_model_space(pipeline.predict(X_test), target_transform)
        return float(np.sqrt(mean_squared_error(y_test, pred)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    if "extra_trees" in model_name:
        best_estimator = ExtraTreesRegressor(random_state=42, n_jobs=-1, **study.best_params)
    elif "hist_gradient" in model_name:
        best_estimator = HistGradientBoostingRegressor(random_state=42, **study.best_params)
    else:
        best_estimator = RandomForestRegressor(random_state=42, n_jobs=-1, **study.best_params)
    best_pipeline = _build_pipeline(best_estimator, X_all)
    best_pipeline.fit(X_train, _target_to_model_space(y_train, target_transform))
    pred = _target_from_model_space(best_pipeline.predict(X_test), target_transform)
    return {
        "success": True,
        "model": best_pipeline,
        "metrics": _metrics(y_test, pred, task_type=task_type),
        "best_params": study.best_params,
        "predictions": pred,
    }


def compute_feature_importance(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, feature_columns: list[str], target_transform: str = "none") -> pd.DataFrame:
    """Descriptive importance; score in the estimator's actual target space.

    The selected estimator was refitted on development data, so these values
    are not an independent validation or evidence of physical causality.
    """
    try:
        fitted_model = model.named_steps.get("model")
        if hasattr(fitted_model, "feature_importances_"):
            preprocess = model.named_steps["preprocess"]
            names = preprocess.get_feature_names_out()
            values = fitted_model.feature_importances_
            return pd.DataFrame({"feature": names, "importance": values}).sort_values("importance", ascending=False).head(10)
    except Exception:
        pass
    if permutation_importance is None:
        return pd.DataFrame()
    try:
        perm = permutation_importance(model, X_test, _target_to_model_space(y_test, target_transform), n_repeats=5, random_state=42)
        return pd.DataFrame({"feature": feature_columns, "importance": perm.importances_mean}).sort_values("importance", ascending=False).head(10)
    except Exception:
        return pd.DataFrame()


def align_features_for_model(candidate_df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    """Return candidate features in the exact order expected by a saved model."""
    feature_columns = list((metadata or {}).get("feature_columns") or [])
    if not feature_columns:
        raise ValueError("Model metadata does not contain feature_columns.")
    normalized = normalize_candidate_features(candidate_df, feature_columns)
    missing = [col for col in feature_columns if col not in normalized.columns]
    if missing:
        raise ValueError(f"columns are missing: {missing}")
    return normalized[feature_columns].copy()


def missing_features_for_model(candidate_df: pd.DataFrame, metadata: dict[str, Any]) -> list[str]:
    feature_columns = list((metadata or {}).get("feature_columns") or [])
    normalized = normalize_candidate_features(candidate_df, feature_columns)
    return [col for col in feature_columns if col not in normalized.columns]


def predict_with_metadata(model, candidate_df: pd.DataFrame, metadata: dict[str, Any]) -> np.ndarray:
    normalized, _diagnostics = normalize_units_for_ml(candidate_df, metadata)
    aligned = align_features_for_model(normalized, metadata)
    raw_prediction = model.predict(aligned)
    return _target_from_model_space(raw_prediction, (metadata or {}).get("target_transform", "none"))


MODEL_DOMAIN_POLICY = "empirical_development_minmax_closed_no_profile_boundary_tolerance"
DERIVED_DOMAIN_FEATURES = {"bed_volume", "carbon_mass", "EBCT_min", "superficial_velocity", "interstitial_velocity", "pressure_drop_bar", "H_to_D_ratio"}


def candidate_domain_mask_and_report(feature_df, metadata, strict_domain=True):
    """Common UI/export rule, applied only after conversion to model units."""
    mask = pd.Series(True, index=feature_df.index)
    rows = []
    for feature, bounds in (metadata or {}).get("feature_ranges", {}).items():
        if feature not in feature_df or not isinstance(bounds, dict):
            continue
        if "values" in bounds:
            inside = feature_df[feature].notna() & feature_df[feature].astype(str).isin(map(str, bounds["values"]))
            model_range = ", ".join(map(str, bounds["values"]))
            generated_range = ", ".join(feature_df[feature].astype(str).unique())
            problem = "unknown_category"
        elif "min" in bounds and "max" in bounds:
            values = pd.to_numeric(feature_df[feature], errors="coerce")
            inside = np.isfinite(values) & values.between(float(bounds["min"]), float(bounds["max"]), inclusive="both")
            model_range = f"{float(bounds['min']):.8g}-{float(bounds['max']):.8g}"
            generated_range = f"{values.min():.8g}-{values.max():.8g}"
            problem = "derived from D,H,rho" if feature in DERIVED_DOMAIN_FEATURES else "outside requested/generated range"
        else:
            continue
        if strict_domain:
            mask &= inside
        if not inside.all():
            rows.append({"parameter": feature, "model_range": model_range, "generated_range": generated_range,
                         "model_unit": (metadata or {}).get("feature_units", {}).get(feature),
                         "problem": problem, "rejected_candidates": int((~inside).sum()),
                         "note": "derived_feature" if feature in DERIVED_DOMAIN_FEATURES else "input_feature",
                         "domain_policy": MODEL_DOMAIN_POLICY})
    return mask, pd.DataFrame(rows)


def selection_model_errors(metadata):
    from scientific_provenance import compatibility_errors
    errors = compatibility_errors(metadata)
    if metadata.get("task_type") != "breakthrough" or metadata.get("target_column") != BREAKTHROUGH_TARGET:
        errors.append("A breakthrough model is required, not a curve model.")
    if "time_min" in metadata.get("feature_columns", []):
        errors.append("time_min cannot be an adsorber-selection feature.")
    if metadata.get("selection_ready") is not True or metadata.get("suitable_for_adsorber_selection") is not True:
        errors.append("The model did not pass the selection quality gate.")
    return errors


def sanitize_breakthrough_predictions(predicted_min, metadata):
    target_max = float((metadata.get("target_stats") or {}).get("max", float("nan")))
    statuses = []
    for value in predicted_min:
        if not np.isfinite(value) or value < 0:
            statuses.append("invalid")
        elif value > 365 * 24 * 60:
            statuses.append("absurd_prediction")
        elif np.isfinite(target_max) and value > 1.2 * target_max:
            statuses.append("outside_target_range")
        else:
            statuses.append("valid")
    return statuses


def predict_selection_candidates(model, raw_feature_df, metadata):
    """Main route: never predict an unverified or out-of-domain construction."""
    errors = selection_model_errors(metadata)
    if errors:
        raise ValueError("; ".join(errors))
    normalized, units = normalize_units_for_ml(raw_feature_df, metadata, "laboratory")
    missing = missing_features_for_model(normalized, metadata)
    if missing:
        raise ValueError(f"Missing selection features: {missing}")
    domain_mask, domain_report = candidate_domain_mask_and_report(normalized, metadata, True)
    predictions = pd.Series(np.nan, index=normalized.index, dtype=float)
    statuses = pd.Series("outside_feature_range", index=normalized.index, dtype=object)
    if domain_mask.any():
        values = predict_with_metadata(model, normalized.loc[domain_mask], metadata)
        predictions.loc[domain_mask] = values
        statuses.loc[domain_mask] = sanitize_breakthrough_predictions(values, metadata)
    return normalized, domain_mask, predictions, statuses, domain_report, units


def normalize_candidate_features(candidate_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    normalized = candidate_df.copy()
    aliases = {
        "flow": ["flow_ml_min", "Q_ml_min", "Q"],
        "height": ["H_cm", "bed_height_cm"],
        "diameter": ["D_cm", "column_diameter_cm"],
        "temperature": ["temperature_C", "T_C"],
        "carbon_mass": ["carbon_mass_g"],
    }
    for target, source_names in aliases.items():
        if target in feature_columns and target not in normalized.columns:
            for source_name in source_names:
                if source_name in normalized.columns:
                    normalized[target] = normalized[source_name]
                    break
    return normalized


def _model_filename(model_file) -> str:
    return str(getattr(model_file, "name", "") or model_file or "")


def _infer_feature_columns_from_model(model) -> list[str]:
    for obj in [
        model,
        getattr(model, "named_steps", {}).get("preprocess") if hasattr(model, "named_steps") else None,
    ]:
        if obj is not None and hasattr(obj, "feature_names_in_"):
            return [str(col) for col in list(obj.feature_names_in_)]
    return []


def _legacy_metadata_for_model(model, filename: str) -> dict[str, Any] | None:
    feature_columns = _infer_feature_columns_from_model(model)
    lower_name = filename.lower()
    if "breakthrough_model" in lower_name and "time_min" not in feature_columns:
        task_type = "breakthrough"
        target_column = BREAKTHROUGH_TARGET
    elif "curve_model" in lower_name or "time_min" in feature_columns:
        task_type = "curve"
        target_column = CURVE_TARGET
    else:
        return None
    return {
        "model_type": "legacy_loaded_model",
        "task_type": task_type,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "training_date": None,
        "metrics": {},
        "number_of_rows": None,
        "number_of_cases": None,
        "train_case_ids_count": None,
        "test_case_ids_count": None,
        "dataset_filename": None,
        "program_version": "legacy_without_metadata",
        "feature_ranges": {},
        "legacy_without_metadata": True,
    }


def model_load_warnings(model, metadata: dict[str, Any], model_filename: str = "", metadata_filename: str = "") -> list[str]:
    warnings: list[str] = []
    from scientific_provenance import provenance_is_current
    if not provenance_is_current(metadata):
        warnings.append("Model provenance is incompatible with the current laboratory data or numerical core; retraining is required.")
        metadata["selection_ready"] = False
        metadata["suitable_for_adsorber_selection"] = False
        metadata["selection_block_reason"] = warnings[-1]
    task_type = (metadata or {}).get("task_type")
    feature_columns = list((metadata or {}).get("feature_columns") or [])
    pipeline_features = _infer_feature_columns_from_model(model)
    lower_model_name = (model_filename or "").lower()

    if not task_type:
        warnings.append("metadata.json не содержит task_type. Рекомендуется переобучить модель и скачать model_bundle.joblib.")
    if pipeline_features and feature_columns and list(pipeline_features) != list(feature_columns):
        warnings.append(
            "feature_columns из metadata не совпадают с признаками pipeline. "
            "Проверьте, что .joblib и metadata.json относятся к одной модели."
        )
    if task_type == "curve" and "breakthrough" in lower_model_name:
        warnings.append("metadata task_type=curve, но имя файла похоже на breakthrough-модель.")
    if task_type == "breakthrough" and "time_min" in feature_columns:
        warnings.append("metadata task_type=breakthrough, но feature_columns содержат time_min.")
    return warnings


def load_saved_model(model_file, metadata_file=None) -> dict[str, Any]:
    if joblib is None:
        return {"success": False, "message": "joblib is not available."}
    filename = _model_filename(model_file)
    metadata_filename = _model_filename(metadata_file) if metadata_file is not None else ""
    loaded_object = joblib.load(model_file)
    if isinstance(loaded_object, dict) and "model" in loaded_object:
        metadata = loaded_object.get("metadata") or {}
        if not metadata:
            return {"success": False, "message": "metadata.json пустой. Переобучите модель и скачайте model_bundle.joblib."}
        model = loaded_object["model"]
        return {
            "success": True,
            "model": model,
            "metadata": metadata,
            "bundle_version": loaded_object.get("bundle_version"),
            "message": "bundle loaded",
            "loaded_model_filename": filename,
            "loaded_metadata_filename": "embedded in bundle",
            "warnings": model_load_warnings(model, metadata, filename, "embedded in bundle"),
        }

    model = loaded_object
    metadata = None
    if metadata_file is not None:
        try:
            if hasattr(metadata_file, "read"):
                metadata = json.load(metadata_file)
            else:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
        except Exception:
            return {"success": False, "message": "metadata.json пустой. Переобучите модель и скачайте model_bundle.joblib."}
        if not metadata:
            return {"success": False, "message": "metadata.json пустой. Переобучите модель и скачайте model_bundle.joblib."}
    else:
        metadata = _legacy_metadata_for_model(model, filename)
        if metadata is None:
            return {
                "success": False,
                "message": "metadata.json не загружен. Нельзя определить тип модели. Рекомендуется загрузить model_bundle.joblib.",
            }
        return {
            "success": True,
            "model": model,
            "metadata": metadata,
            "legacy_warning": "Модель загружена в legacy-режиме без metadata. Рекомендуется переобучить и скачать model_bundle.joblib.",
            "loaded_model_filename": filename,
            "loaded_metadata_filename": "",
            "warnings": model_load_warnings(model, metadata, filename, ""),
        }
    return {
        "success": True,
        "model": model,
        "metadata": metadata,
        "loaded_model_filename": filename,
        "loaded_metadata_filename": metadata_filename,
        "warnings": model_load_warnings(model, metadata, filename, metadata_filename),
    }


def make_candidate_adsorbers(
    n_candidates: int,
    ranges: dict[str, tuple[float, float]],
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_seed))
    data = {}
    for name, (lo, hi) in ranges.items():
        data[name] = rng.uniform(float(lo), float(hi), int(n_candidates))
    return pd.DataFrame(data)


def _range_values(values: list[float], lo: float, hi: float) -> list[float]:
    selected = [float(value) for value in values if float(lo) <= float(value) <= float(hi)]
    if selected:
        return selected
    midpoint = 0.5 * (float(lo) + float(hi))
    return [float(np.clip(midpoint, float(lo), float(hi)))]


def _rounded_uniform(rng: np.random.Generator, lo: float, hi: float, n: int, step: float) -> np.ndarray:
    values = rng.uniform(float(lo), float(hi), int(n))
    rounded = np.round(values / float(step)) * float(step)
    return np.clip(rounded, float(lo), float(hi))


def _grid_values(lo: float, hi: float, step: float) -> np.ndarray:
    count = max(int(math.floor((float(hi) - float(lo)) / float(step))) + 1, 1)
    values = float(lo) + np.arange(count, dtype=float) * float(step)
    values = values[values <= float(hi) + 1e-12]
    if len(values) == 0:
        values = np.array([0.5 * (float(lo) + float(hi))], dtype=float)
    return values


def make_engineering_candidate_adsorbers(
    n_candidates: int,
    scale_mode: str,
    ranges: dict[str, tuple[float, float]],
    random_seed: int = 42,
    method: str = "engineering_grid_random",
    model_feature_ranges: dict | None = None,
    strict_domain: bool = True,
) -> pd.DataFrame:
    """Generate rounded, engineering-like adsorber candidates.

    The legacy make_candidate_adsorbers() remains available for purely
    mathematical continuous random sampling.
    """
    rng = np.random.default_rng(int(random_seed))
    n = int(n_candidates)
    sample_n = n
    scale_text = str(scale_mode).lower()
    is_lab = "lab" in scale_text or "лаборатор" in scale_text or "лаборатор" in scale_text

    def limit_to_model_domain(result: pd.DataFrame) -> pd.DataFrame:
        # Legacy arguments remain accepted. Model-domain checks belong downstream,
        # after candidate features have been converted to the model's units.
        limited = result.head(n).reset_index(drop=True)
        limited.attrs["pre_domain_generated_count"] = int(len(result))
        return limited

    if method == "latin_hypercube_continuous":
        try:
            from scipy.stats import qmc

            names = list(ranges.keys())
            sampler = qmc.LatinHypercube(d=len(names), seed=int(random_seed))
            unit = sampler.random(sample_n)
            data = {}
            for i, name in enumerate(names):
                lo, hi = ranges[name]
                data[name] = float(lo) + unit[:, i] * (float(hi) - float(lo))
            return limit_to_model_domain(pd.DataFrame(data))
        except Exception:
            return limit_to_model_domain(make_candidate_adsorbers(sample_n, ranges, random_seed=random_seed))

    if method == "random_continuous":
        return limit_to_model_domain(make_candidate_adsorbers(sample_n, ranges, random_seed=random_seed))

    flow_lo, flow_hi = ranges.get("flow", (1.0, 20.0))
    h_lo, h_hi = ranges.get("height", (2.5, 30.0) if is_lab else (0.5, 3.0))
    d_lo, d_hi = ranges.get("diameter", (1.5, 5.0) if is_lab else (0.05, 0.5))
    t_lo, t_hi = ranges.get("temperature", (10.0, 30.0))
    eps_lo, eps_hi = ranges.get("epsilon", (0.35, 0.45))
    rho_lo, rho_hi = ranges.get("bulk_density", (0.24, 0.31) if is_lab else (490.0, 530.0))
    dp_lo, dp_hi = ranges.get("dp_mm", (0.8, 2.0))

    if is_lab:
        diameter_values = _range_values([1.5, 2.0, 2.5, 3.0, 3.2, 4.0, 5.0], d_lo, d_hi)
        height_step = 0.5
        flow_step = 0.1 if flow_hi <= 20.0 else 0.5
        rho_step = 0.001
    else:
        diameter_values = _range_values([0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00, 1.20, 1.50, 2.00, 2.50, 3.00], d_lo, d_hi)
        is_industrial = "industrial" in scale_text or "пром" in scale_text or "Ð¿Ñ€Ð¾Ð¼" in scale_text
        height_step = 0.1 if is_industrial else 0.05
        flow_step = 0.1 if is_industrial else (0.001 if flow_hi <= 0.1 else 0.01)
        rho_step = 5.0

    dp_values = _range_values([0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0], dp_lo, dp_hi)

    if method == "discrete_grid":
        h_values = _grid_values(h_lo, h_hi, height_step)
        flow_values = _grid_values(flow_lo, flow_hi, flow_step)
        eps_values = _grid_values(eps_lo, eps_hi, 0.01)
        rho_values = _grid_values(rho_lo, rho_hi, rho_step)
        rows = []
        for d in diameter_values:
            for h in h_values:
                for flow in flow_values:
                    for eps in eps_values:
                        for rho in rho_values:
                            rows.append(
                                {
                                    "flow": float(flow),
                                    "height": float(h),
                                    "diameter": float(d),
                                    "temperature": float(round(0.5 * (t_lo + t_hi), 1)),
                                    "epsilon": float(eps),
                                    "bulk_density": float(rho),
                                    "dp_mm": float(dp_values[0]),
                                }
                            )
        result = pd.DataFrame(rows)
        return limit_to_model_domain(result)

    return limit_to_model_domain(pd.DataFrame(
        {
            "flow": _rounded_uniform(rng, flow_lo, flow_hi, sample_n, flow_step),
            "height": _rounded_uniform(rng, h_lo, h_hi, sample_n, height_step),
            "diameter": rng.choice(np.array(diameter_values, dtype=float), size=sample_n, replace=True),
            "temperature": _rounded_uniform(rng, t_lo, t_hi, sample_n, 0.5),
            "epsilon": _rounded_uniform(rng, eps_lo, eps_hi, sample_n, 0.01),
            "bulk_density": _rounded_uniform(rng, rho_lo, rho_hi, sample_n, rho_step),
            "dp_mm": rng.choice(np.array(dp_values, dtype=float), size=sample_n, replace=True),
        }
    ))


def flag_extrapolation(row: pd.Series, feature_ranges: dict[str, Any]) -> str:
    warnings = []
    for col, value in row.items():
        ranges = feature_ranges.get(col)
        if not ranges or "min" not in ranges:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric < float(ranges["min"]) or numeric > float(ranges["max"]):
            warnings.append(col)
    return "" if not warnings else "outside training range: " + ", ".join(warnings[:6])

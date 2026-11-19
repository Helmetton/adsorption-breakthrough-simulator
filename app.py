# -*- coding: utf-8 -*-
from __future__ import annotations

from startup_probe import StartupProbe
_startup_probe = StartupProbe()

from io import BytesIO
from itertools import product
from pathlib import Path
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import zipfile

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), ".matplotlib_cache"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import qmc

from adsorbents import ADSORBENT_PROFILES
from dependency_manager import check_dependencies, install_missing_dependencies
from encoding_utils import clean_dataframe_text, fix_mojibake, safe_read_csv
from fit_all_experiments import fit_all_experiments
from industrial_model import (
    ergun_pressure_drop,
    industrial_adsorber_diagnostics,
    validate_industrial_inputs,
)
from mass_balance import calculate_mass_balance, summarize_mass_balance
from selection_verification import verify_candidate, verification_export_bytes
from ml_training import train_baseline_models
from ml_training import metadata_to_bytes, save_joblib_to_bytes, save_model_bundle_to_bytes
from ml_adsorber import (
    align_features_for_model,
    candidate_domain_mask_and_report as shared_candidate_domain_mask_and_report,
    MODEL_DOMAIN_POLICY,
    flag_extrapolation,
    load_ml_dataset,
    load_saved_model,
    make_candidate_adsorbers,
    make_engineering_candidate_adsorbers,
    missing_features_for_model,
    normalize_units_for_ml,
    normalize_scale_mode,
    predict_with_metadata,
    sklearn_available,
    train_models as train_adsorber_models,
    validate_dataset_for_breakthrough_training,
    validate_ml_dataset,
)
from plotting import PLOTLY_AVAILABLE, plot_breakthrough_interactive
from operating_regime import check_operating_regime, load_calibration_ranges
from recommended_ranges import ADSORBER_SELECTION_RANGES, INDUSTRIAL_SCALE_PRESETS, RECOMMENDED_DATASET_RANGES
from sorption_literature_profiles import (
    SORPTION_PARAMETER_PROFILES,
    resolve_sorption_parameters,
)
from temperature_correction import (
    DEFAULT_D_M_REF_M2_S,
    DEFAULT_EA_KLDF_KJ_MOL,
    apply_temperature_adaptation,
    experiment_temperature_warnings,
    laboratory_temperature_range,
    load_experiment_temperature_map,
    resolve_temperature_correction_parameters,
)
from translations import HELP_SECTIONS, LANGUAGES, tr
from ui_presentation import PresentationUI, figure_snapshot, figure_from_snapshot
from ui_saved_plots import saved_plot_png
from ui_sections import experiment_description, render_experiment_overview, dataset_counts, render_dataset_preview, render_parameter_source, render_help, render_quality_overview
from parameter_provenance import (parameter_origin, parameter_origins, provenance_record,
                                  effective_input_payload, temperature_lineage,
                                  sampled_parameter_payload, training_stat_origins,
                                  dataset_parameter_provenance, geometry_records)
from ui_parameter_provenance import parameter_source_table, origin_description
from ui_startup_loading import tracked_tabs, load_dataset_file, cached_breakthrough_quality
from unit_conversions import industrial_to_column_params, verify_unit_conversions
from validation_models import validate_industrial_input, validate_simulation_input
from model import (
    DEFAULT_BREAKTHROUGH_RATIO,
    DEFAULT_N_CELLS,
    MODEL_VERSION,
    ColumnParams,
    SorptionParams,
    bootstrap_fit_uncertainty,
    breakthrough_status,
    column_area_from_diameter,
    experiment_breakthrough_concentration,
    first_breakthrough_time,
    fit_model_to_experiment,
    interpret_breakthrough_curve,
    load_experiments_csv,
    make_column_from_experiment,
    run_simulation,
    simulate_breakthrough,
    simulate_breakthrough_full,
)


_startup_probe.mark("imports_complete")
DATA_PATH = "cleaned_dynamic_sorption_data.csv"
st = PresentationUI(st)
SUMMARY_PATH = "dynamic_sorption_experiment_summary.csv"
DEFAULT_PROFILE_KEY = "chemviron_natural"


def download_button_persistent(label: str, data, file_name: str, mime: str, key: str):
    try:
        return st.download_button(
            label,
            data=data,
            file_name=file_name,
            mime=mime,
            key=key,
            on_click="ignore",
        )
    except TypeError:
        return st.download_button(
            label,
            data=data,
            file_name=file_name,
            mime=mime,
            key=key,
        )


def ml_best_metrics(metadata: dict) -> dict:
    if (metadata or {}).get("final_test_metrics"):
        return metadata["final_test_metrics"]
    metrics = (metadata or {}).get("metrics") or {}
    model_type = (metadata or {}).get("model_type")
    if model_type and isinstance(metrics.get(model_type), dict):
        return metrics[model_type]
    metric_rows = [value for value in metrics.values() if isinstance(value, dict)]
    if not metric_rows:
        return {}
    return sorted(metric_rows, key=lambda row: row.get("RMSE", float("inf")))[0]


def ensure_model_identity(metadata: dict | None) -> dict:
    metadata = dict(metadata or {})
    if metadata.get("model_id") and metadata.get("metadata_hash"):
        return metadata
    payload = {
        "task_type": metadata.get("task_type"),
        "target_column": metadata.get("target_column"),
        "model_type": metadata.get("model_type"),
        "feature_columns": metadata.get("feature_columns"),
        "feature_units": metadata.get("feature_units"),
        "target_stats": metadata.get("target_stats"),
        "training_date": metadata.get("training_date"),
        "dataset_filename": metadata.get("dataset_filename"),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]
    metadata.setdefault("metadata_hash", digest)
    metadata.setdefault("model_id", f"{metadata.get('task_type', 'model')}_{digest}")
    return metadata


def ml_model_summary_dataframe(
    metadata: dict | None,
    model_filename: str = "",
    metadata_filename: str = "",
    source: str = "",
) -> pd.DataFrame:
    metadata = ensure_model_identity(metadata)
    feature_columns = list(metadata.get("feature_columns") or [])
    best_metrics = ml_best_metrics(metadata)
    suitable = (
        metadata.get("task_type") == "breakthrough"
        and "time_min" not in feature_columns
        and metadata.get("selection_ready") is True
    )
    return pd.DataFrame(
        [
            {
                "loaded_model_filename": model_filename or "-",
                "loaded_metadata_filename": metadata_filename or "-",
                "source": source or "-",
                "model_id": metadata.get("model_id") or "-",
                "metadata_hash": metadata.get("metadata_hash") or "-",
                "task_type": metadata.get("task_type") or "-",
                "target_column": metadata.get("target_column") or "-",
                "model_type": metadata.get("model_type") or "-",
                "feature_columns": ", ".join(map(str, feature_columns)) or "-",
                "number_of_features": len(feature_columns),
                "RMSE": best_metrics.get("RMSE"),
                "MAE": best_metrics.get("MAE"),
                "R2": best_metrics.get("R2"),
                "selection_ready": metadata.get("selection_ready"),
                "selection_block_reason": metadata.get("selection_block_reason"),
                "model_family": metadata.get("model_family"),
                "suitable_for_adsorber_selection": bool(suitable),
            }
        ]
    )


def validate_model_for_adsorber_selection(metadata: dict | None) -> bool:
    from scientific_provenance import provenance_is_current
    metadata = ensure_model_identity(metadata)
    suitable_flag = metadata.get("suitable_for_adsorber_selection")
    suitable = True if suitable_flag is None else bool(suitable_flag)
    feature_columns = list(metadata.get("feature_columns") or [])
    return (
        metadata.get("task_type") == "breakthrough"
        and metadata.get("target_column") == "breakthrough_time_min"
        and metadata.get("selection_ready") is True
        and suitable
        and "time_min" not in feature_columns
        and provenance_is_current(metadata)
        and "linear_regression" not in str(metadata.get("model_type", ""))
    )


def get_ml_model_registry() -> dict:
    registry = st.session_state.get("ml_model_registry")
    if not isinstance(registry, dict):
        registry = {}
        st.session_state["ml_model_registry"] = registry
    return registry


def get_registry_model(task_type: str) -> dict:
    item = get_ml_model_registry().get(task_type)
    return item if isinstance(item, dict) else {}


def set_registry_model(
    task_type: str,
    model,
    metadata: dict,
    metrics,
    bundle_bytes: bytes | None = None,
    model_bytes: bytes | None = None,
    metadata_bytes: bytes | None = None,
    metrics_bytes: bytes | None = None,
    source: str = "trained_in_current_session",
) -> dict:
    metadata = ensure_model_identity(metadata)
    ready = validate_model_for_adsorber_selection(metadata) if task_type == "breakthrough" else False
    item = {
        "model": model,
        "metadata": metadata,
        "metrics": metrics,
        "bundle_bytes": bundle_bytes,
        "model_bytes": model_bytes,
        "metadata_bytes": metadata_bytes,
        "metrics_bytes": metrics_bytes,
        "source": source,
        "ready": bool(ready),
    }
    get_ml_model_registry()[task_type] = item
    return item


def set_active_breakthrough_model_from_registry() -> bool:
    item = get_registry_model("breakthrough")
    metadata = item.get("metadata") or {}
    if not item.get("model") or not validate_model_for_adsorber_selection(metadata):
        return False
    if (st.session_state.get("active_ml_metadata") or {}).get("model_id") != metadata.get("model_id"):
        for key in ["ml_all_candidates_df", "ml_candidates_df", "ml_verification_candidates_df",
                    "ml_rejected_candidates_df", "ml_rejection_reasons_df", "ml_selection_diagnostics_df",
                    "ml_candidate_domain_df", "ml_schema_diagnostics_df", "ml_unit_diagnostics",
                    "ml_selection_input_snapshot"]:
            st.session_state.pop(key, None)
    st.session_state["active_ml_model"] = item["model"]
    st.session_state["active_ml_metadata"] = metadata
    st.session_state["active_ml_task_type"] = "breakthrough"
    st.session_state["active_ml_source"] = item.get("source") or "trained_in_current_session"
    st.session_state["active_ml_model_key"] = "breakthrough"
    return True


def bytes_size_mb(data) -> float:
    if data is None:
        return 0.0
    try:
        return len(data) / (1024.0 * 1024.0)
    except TypeError:
        return 0.0


def save_registry_models_to_local_folder(folder: str = "models") -> list[str]:
    os.makedirs(folder, exist_ok=True)
    saved: list[str] = []
    for task_type, item in get_ml_model_registry().items():
        if not isinstance(item, dict):
            continue
        files = {
            f"{task_type}_model_bundle.joblib": item.get("bundle_bytes"),
            f"{task_type}_model.joblib": item.get("model_bytes"),
            f"{task_type}_metadata.json": item.get("metadata_bytes"),
            f"{task_type}_metrics.csv": item.get("metrics_bytes"),
        }
        for filename, data in files.items():
            if data:
                path = os.path.join(folder, filename)
                with open(path, "wb") as handle:
                    handle.write(data)
                saved.append(path)
    return saved


def render_active_selection_message(lang: str) -> None:
    metadata = st.session_state.get("active_ml_metadata")
    if isinstance(metadata, dict) and validate_model_for_adsorber_selection(metadata):
        st.success(tr(lang, "ml_active_breakthrough_ready"))
    else:
        st.warning(tr(lang, "ml_train_breakthrough_first"))


def model_metrics_column_config(task_type: str, lang: str) -> dict:
    return {
        "RMSE": st.column_config.NumberColumn(tr(lang, "ui_field_RMSE_min" if task_type == "breakthrough" else "ui_field_RMSE_C0"), help=tr(lang, "ui_metric_help")),
        "MAE": st.column_config.NumberColumn(tr(lang, "ui_field_MAE_min" if task_type == "breakthrough" else "ui_field_MAE_C0"), help=tr(lang, "ui_metric_help")),
    }


def render_active_ml_model_card(lang: str) -> None:
    metadata = st.session_state.get("active_ml_metadata")
    source = st.session_state.get("active_ml_source", "none")
    st.markdown(f"#### {tr(lang, 'ml_active_selection_model')}")
    st.caption(tr(lang, "scientific_ml_scope"))
    if isinstance(metadata, dict) and metadata:
        if metadata.get("task_type") == "curve":
            st.error(tr(lang, "ml_wrong_model_for_selection"))
        st.dataframe(
            ml_model_summary_dataframe(metadata, source=source),
            width="stretch",
            hide_index=True,
        )
        if source == "trained_in_current_session":
            st.success(tr(lang, "ml_using_trained_session_model"))
        elif source == "uploaded_model":
            st.info(tr(lang, "ml_using_uploaded_model"))
    else:
        st.info(tr(lang, "ml_no_active_model"))


def render_sync_active_breakthrough_button(lang: str, key: str) -> None:
    item = get_registry_model("breakthrough")
    metadata = ensure_model_identity(item.get("metadata") if item else {})
    active_metadata = ensure_model_identity(st.session_state.get("active_ml_metadata"))
    if not item or not validate_model_for_adsorber_selection(metadata):
        return
    if active_metadata.get("model_id") != metadata.get("model_id"):
        if st.button(tr(lang, "ml_sync_active_with_latest_training"), key=key):
            if set_active_breakthrough_model_from_registry():
                st.success(tr(lang, "ml_breakthrough_session_model_active"))
                st.rerun()
            else:
                st.error(tr(lang, "ml_no_valid_breakthrough_in_session"))


def render_session_models_registry(lang: str, show_downloads: bool = True, key_prefix: str = "registry") -> None:
    registry = get_ml_model_registry()
    if not registry:
        return
    st.markdown(f"#### {tr(lang, 'ml_models_trained_current_session')}")
    for task_type, title_key in [
        ("breakthrough", "ml_breakthrough_model_for_selection"),
        ("curve", "ml_curve_model_for_analysis"),
    ]:
        item = registry.get(task_type)
        if not isinstance(item, dict) or not item.get("metadata"):
            continue
        metadata = item.get("metadata") or {}
        st.markdown(f"##### {tr(lang, title_key)}")
        st.dataframe(
            ml_model_summary_dataframe(metadata, source=item.get("source", "trained_in_current_session")),
            width="stretch",
            hide_index=True,
        )
        best_metrics = ml_best_metrics(metadata)
        target_stats = metadata.get("target_stats") or {}
        if task_type == "breakthrough":
            metric_row = {
                "target": "breakthrough_time_min",
                "units": "min",
                "selected_model": metadata.get("model_type"),
                "RMSE_min": best_metrics.get("RMSE"),
                "MAE_min": best_metrics.get("MAE"),
                "R2_time": best_metrics.get("R2"),
                "target_max_h": (target_stats.get("max") / 60.0) if target_stats.get("max") is not None else None,
                "selection_ready": metadata.get("selection_ready"),
            }
        else:
            metric_row = {
                "target": "C_out_C0",
                "units": "dimensionless",
                "selected_model": metadata.get("model_type"),
                "RMSE_C/C0": best_metrics.get("RMSE"),
                "MAE_C/C0": best_metrics.get("MAE"),
                "R2_curve": best_metrics.get("R2"),
                "selection_ready": False,
            }
        st.dataframe(pd.DataFrame([metric_row]), width="stretch", hide_index=True)
        if task_type == "breakthrough":
            if item.get("ready"):
                if st.button(tr(lang, "ml_make_active_for_selection"), key=f"{key_prefix}_make_active_{task_type}"):
                    if set_active_breakthrough_model_from_registry():
                        st.success(tr(lang, "ml_breakthrough_session_model_active"))
                    else:
                        st.error(tr(lang, "ml_no_valid_breakthrough_in_session"))
            else:
                st.error(tr(lang, "ml_model_not_selection_ready"))
        else:
            st.info(tr(lang, "ml_curve_not_direct_selection"))
        if show_downloads:
            if task_type == "curve" and not item.get("bundle_bytes"):
                st.warning(tr(lang, "ml_curve_bundle_not_prepared"))
                if st.button(tr(lang, "ml_prepare_curve_download"), key=f"{key_prefix}_prepare_curve_download"):
                    item["model_bytes"] = save_joblib_to_bytes(item["model"])
                    item["bundle_bytes"] = save_model_bundle_to_bytes(item["model"], metadata)
                    st.success(tr(lang, "ml_curve_download_prepared"))
                    st.rerun()
                continue
            bundle_bytes = item.get("bundle_bytes") or b""
            model_bytes = item.get("model_bytes") or b""
            metadata_bytes = item.get("metadata_bytes") or b""
            metrics_bytes = item.get("metrics_bytes") or b""
            bundle_mb = bytes_size_mb(bundle_bytes)
            st.caption(f"{task_type}_model_bundle.joblib: {bundle_mb:.1f} MB")
            if bundle_mb > 200.0:
                st.warning(tr(lang, "ml_bundle_too_large_warning"))
            cols = st.columns(4)
            with cols[0]:
                download_button_persistent(
                    f"{tr(lang, 'ml_download_bundle')} ({task_type})",
                    bundle_bytes,
                    f"{task_type}_model_bundle.joblib",
                    "application/octet-stream",
                    f"{key_prefix}_download_registry_{task_type}_bundle",
                )
            with cols[1]:
                download_button_persistent(
                    f"{tr(lang, 'ml_download_model')} ({task_type})",
                    model_bytes,
                    f"{task_type}_model.joblib",
                    "application/octet-stream",
                    f"{key_prefix}_download_registry_{task_type}_model",
                )
            with cols[2]:
                download_button_persistent(
                    f"{tr(lang, 'ml_download_metadata')} ({task_type})",
                    metadata_bytes,
                    f"{task_type}_metadata.json",
                    "application/json",
                    f"{key_prefix}_download_registry_{task_type}_metadata",
                )
            with cols[3]:
                download_button_persistent(
                    f"metrics.csv ({task_type})",
                    metrics_bytes,
                    f"{task_type}_metrics.csv",
                    "text/csv",
                    f"{key_prefix}_download_registry_{task_type}_metrics",
                )


def set_ml_dataset_for_training(df: pd.DataFrame, filename: str, source: str, dataset_mode: str = "synthetic") -> dict:
    validation = validate_ml_dataset(df, dataset_mode=dataset_mode)
    st.session_state["ml_dataset_df"] = validation["canonical_df"]
    st.session_state["ml_dataset_filename"] = filename
    st.session_state["ml_dataset_source"] = source
    from scientific_provenance import resolve_dataset_provenance
    verified = resolve_dataset_provenance(df)
    st.session_state["ml_dataset_metadata"] = {
        "source": source, "declared_model_versions": sorted(map(str, df.get("mechanistic_model_id", pd.Series(dtype=str)).dropna().unique())),
        "sorption_parameter_provenance": dataset_parameter_provenance(df),
        **verified,
    }
    st.session_state["ml_dataset_df"].attrs["generation_provenance"] = verified
    return validation


def case_count(df: pd.DataFrame) -> int:
    return int(df["case_id"].nunique()) if isinstance(df, pd.DataFrame) and "case_id" in df.columns else 0


USER_REQUEST_FEATURES = {
    "C0_mg_L",
    "C_breakthrough_ratio",
    "flow",
    "height",
    "diameter",
    "temperature",
    "epsilon",
    "bulk_density",
    "KH_used_L_g",
    "kLDF_used_1_min",
    "Dax_used_cm2_min",
    "scale_mode",
}

DERIVED_CANDIDATE_FEATURES = {
    "bed_volume",
    "carbon_mass",
    "EBCT_min",
    "superficial_velocity",
    "interstitial_velocity",
    "pressure_drop_bar",
    "H_to_D_ratio",
}


def outside_training_features(feature_row: dict, metadata: dict | None) -> list[str]:
    feature_ranges = (metadata or {}).get("feature_ranges", {}) or {}
    outside = []
    for col, value in feature_row.items():
        ranges = feature_ranges.get(col)
        if not ranges or "min" not in ranges or "max" not in ranges:
            continue
        try:
            numeric = float(value)
            lo = float(ranges["min"])
            hi = float(ranges["max"])
        except (TypeError, ValueError):
            continue
        if numeric < lo or numeric > hi:
            outside.append(col)
    return outside


def metadata_numeric_range(metadata: dict | None, feature: str) -> dict:
    return ((metadata or {}).get("feature_ranges") or {}).get(feature) or {}


def metadata_range_value(metadata: dict | None, feature: str, fallback: float, key: str = "median") -> float:
    ranges = metadata_numeric_range(metadata, feature)
    for candidate_key in (key, "median", "min", "max"):
        try:
            value = float(ranges.get(candidate_key))
            if math.isfinite(value):
                return value
        except (TypeError, ValueError):
            continue
    return float(fallback)


def model_range_tuple(metadata: dict | None, feature: str, fallback: tuple[float, float]) -> tuple[float, float]:
    ranges = metadata_numeric_range(metadata, feature)
    try:
        lo = float(ranges.get("min"))
        hi = float(ranges.get("max"))
        if math.isfinite(lo) and math.isfinite(hi) and lo <= hi:
            return lo, hi
    except (TypeError, ValueError):
        pass
    return fallback


def validate_selection_request_against_model(
    metadata: dict | None,
    requested_ranges: dict[str, tuple[float, float]],
    required_time_min: float,
    selection_scale: str,
    parameter_source_values: dict,
    allow_strong_extrapolation: bool = False,
) -> dict:
    metadata = metadata or {}
    errors: list[str] = []
    warnings: list[str] = []
    domain_rows: list[dict] = []
    if metadata.get("task_type") != "breakthrough":
        errors.append("metadata task_type != breakthrough")
    target_stats = metadata.get("target_stats") or {}
    try:
        target_max = float(target_stats.get("max"))
        target_median = float(target_stats.get("median"))
    except (TypeError, ValueError):
        target_max = float("nan")
        target_median = float("nan")
    if not math.isfinite(target_max) or target_max <= 0:
        errors.append(tr(st.session_state.get("language", "ru"), "ml_missing_target_stats"))
    else:
        domain_rows.append(
            {
                "parameter": "breakthrough_time_min",
                "model_range": f"max {target_max:.3g} min ({target_max / 60.0:.3g} h)",
                "requested_range": f"{required_time_min:.3g} min ({required_time_min / 60.0:.3g} h)",
                "status": "inside" if required_time_min <= target_max else "outside",
            }
        )
        if required_time_min > 1.2 * target_max:
            errors.append(
                tr(st.session_state.get("language", "ru"), "ml_required_time_outside_domain").format(
                    required=f"{required_time_min / 60.0:.3g}",
                    maximum=f"{target_max / 60.0:.3g}",
                )
            )
        elif required_time_min > target_max:
            warnings.append(
                tr(st.session_state.get("language", "ru"), "ml_required_time_caution").format(
                    maximum=f"{target_max / 60.0:.3g}"
                )
            )
        elif required_time_min * 1.5 > target_max:
            warnings.append(
                tr(st.session_state.get("language", "ru"), "ml_target_range_too_narrow").format(
                    maximum=f"{target_max / 60.0:.3g}",
                    required=f"{required_time_min / 60.0:.3g}",
                )
            )

    feature_ranges = metadata.get("feature_ranges") or {}
    for feature, requested in requested_ranges.items():
        ranges = feature_ranges.get(feature)
        if not ranges or "min" not in ranges or "max" not in ranges:
            warnings.append(tr(st.session_state.get("language", "ru"), "ml_model_missing_feature_range").format(feature=feature))
            continue
        try:
            model_min = float(ranges["min"])
            model_max = float(ranges["max"])
            req_min, req_max = float(requested[0]), float(requested[1])
        except (TypeError, ValueError):
            continue
        status = "inside" if req_min >= model_min and req_max <= model_max else "outside"
        domain_rows.append(
            {
                "parameter": feature,
                "model_range": f"{model_min:.4g}-{model_max:.4g}",
                "requested_range": f"{req_min:.4g}-{req_max:.4g}",
                "status": status,
            }
        )
        far_low = req_min < model_min - 0.25 * max(abs(model_min), 1e-12)
        far_high = req_max > model_max + 0.25 * max(abs(model_max), 1e-12)
        if (far_low or far_high) and not allow_strong_extrapolation:
            if feature in USER_REQUEST_FEATURES:
                errors.append(tr(st.session_state.get("language", "ru"), "ml_requested_range_outside_domain").format(feature=feature))
            else:
                warnings.append(tr(st.session_state.get("language", "ru"), "ml_derived_feature_outside").format(feature=feature))

    for feature in ["KH_used_L_g", "kLDF_used_1_min", "Dax_used_cm2_min"]:
        value = parameter_source_values.get(feature)
        ranges = feature_ranges.get(feature)
        if value is None or not ranges or "min" not in ranges or "max" not in ranges:
            continue
        try:
            numeric = float(value)
            model_min = float(ranges["min"])
            model_max = float(ranges["max"])
        except (TypeError, ValueError):
            continue
        status = "inside" if model_min <= numeric <= model_max else "outside"
        domain_rows.append(
            {
                "parameter": feature,
                "model_range": f"{model_min:.4g}-{model_max:.4g}",
                "requested_range": f"{numeric:.4g}",
                "status": status,
            }
        )
        if status == "outside" and not allow_strong_extrapolation:
            errors.append(tr(st.session_state.get("language", "ru"), "ml_sorption_params_outside_domain").format(feature=feature))

    scale_values = (feature_ranges.get("scale_mode") or {}).get("values") or []
    if scale_values and normalize_scale_mode(selection_scale) not in set(map(normalize_scale_mode, scale_values)) and not allow_strong_extrapolation:
        domain_rows.append(
            {
                "parameter": "scale_mode",
                "model_range": ", ".join(map(str, scale_values)),
                "requested_range": str(selection_scale),
                "status": "outside",
            }
        )
        errors.append(tr(st.session_state.get("language", "ru"), "ml_scale_outside_domain"))

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "domain_df": pd.DataFrame(domain_rows),
        "target_max_min": target_max,
        "target_median_min": target_median,
    }


def sanitize_breakthrough_predictions(predicted_min: np.ndarray, metadata: dict | None) -> list[str]:
    target_stats = (metadata or {}).get("target_stats") or {}
    try:
        target_max = float(target_stats.get("max"))
    except (TypeError, ValueError):
        target_max = float("nan")
    statuses = []
    for value in predicted_min:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            statuses.append("invalid")
            continue
        if not math.isfinite(numeric) or numeric < 0:
            statuses.append("invalid")
        elif numeric > 365.0 * 24.0 * 60.0:
            statuses.append("absurd_prediction")
        elif math.isfinite(target_max) and target_max > 0 and numeric > 1.2 * target_max:
            statuses.append("outside_target_range")
        else:
            statuses.append("valid")
    return statuses


def candidate_domain_mask_and_report(feature_df: pd.DataFrame, metadata: dict | None, strict_domain: bool) -> tuple[pd.Series, pd.DataFrame]:
    return shared_candidate_domain_mask_and_report(feature_df, metadata or {}, strict_domain)


def selection_diagnostics_dataframe(
    all_candidates_df: pd.DataFrame,
    n_generated: int,
    n_prediction_errors: int,
    required_time_min: float,
) -> pd.DataFrame:
    if all_candidates_df.empty:
        prediction_values = pd.Series(dtype=float)
    else:
        prediction_values = pd.to_numeric(
            all_candidates_df.get("predicted_breakthrough_time_min", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
    best_predicted_time_min = float(prediction_values.max()) if not prediction_values.empty else np.nan
    return pd.DataFrame(
        [
            {
                "n_candidates_generated": int(n_generated),
                "n_predictions_success": int(len(prediction_values)),
                "n_prediction_errors": int(n_prediction_errors),
                "n_pass_geometry": int(all_candidates_df.get("passes_geometry", pd.Series(dtype=bool)).sum()) if not all_candidates_df.empty else 0,
                "n_pass_EBCT": int(all_candidates_df.get("passes_EBCT", pd.Series(dtype=bool)).sum()) if not all_candidates_df.empty else 0,
                "n_pass_velocity": int(all_candidates_df.get("passes_velocity", pd.Series(dtype=bool)).sum()) if not all_candidates_df.empty else 0,
                "n_pass_pressure": int(all_candidates_df.get("passes_pressure_limit", pd.Series(dtype=bool)).sum()) if not all_candidates_df.empty else 0,
                "n_pass_required_time": int(all_candidates_df.get("passes_required_time", pd.Series(dtype=bool)).sum()) if not all_candidates_df.empty else 0,
                "n_filtered_by_required_time": int((~all_candidates_df.get("passes_required_time", pd.Series(dtype=bool))).sum()) if not all_candidates_df.empty else 0,
                "n_filtered_by_mass": int((~all_candidates_df.get("passes_mass_limit", pd.Series(dtype=bool))).sum()) if not all_candidates_df.empty else 0,
                "n_filtered_by_pressure": int((~all_candidates_df.get("passes_pressure_limit", pd.Series(dtype=bool))).sum()) if not all_candidates_df.empty else 0,
                "n_outside_training_range": int(all_candidates_df.get("outside_training_range", pd.Series(dtype=bool)).sum()) if not all_candidates_df.empty else 0,
                "predicted_time_min_min": float(prediction_values.min()) if not prediction_values.empty else np.nan,
                "predicted_time_min_median": float(prediction_values.median()) if not prediction_values.empty else np.nan,
                "predicted_time_min_max": best_predicted_time_min,
                "required_time_min": float(required_time_min),
                "best_predicted_time_min": best_predicted_time_min,
                "best_predicted_time_h": best_predicted_time_min / 60.0 if math.isfinite(best_predicted_time_min) else np.nan,
                "best_predicted_time_days": best_predicted_time_min / 1440.0 if math.isfinite(best_predicted_time_min) else np.nan,
            }
        ]
    )


def sort_constrained_candidates(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    result = df.copy()
    sort_columns = [
        "failed_constraints_count",
        "prediction_unreliable",
        "outside_training_range",
        "predicted_breakthrough_time_h",
        "carbon_mass",
    ]
    ascending = [True, True, True, False, True]
    sort_columns = [col for col in sort_columns if col in result.columns]
    ascending = ascending[: len(sort_columns)]
    return result.sort_values(sort_columns, ascending=ascending)


def rejection_reasons_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty or "reject_reason" not in df.columns:
        return pd.DataFrame()
    counts: dict[str, int] = {}
    for value in df["reject_reason"].fillna("").astype(str):
        for part in [item.strip() for item in value.split(";") if item.strip()]:
            counts[part] = counts.get(part, 0) + 1
    return pd.DataFrame(
        [{"reason": reason, "count": count} for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]
    )


def plot_prediction_histogram(all_candidates_df: pd.DataFrame, required_time_h: float, lang: str):
    values = pd.to_numeric(all_candidates_df.get("predicted_breakthrough_time_h"), errors="coerce").dropna()
    if values.empty:
        return None
    if PLOTLY_AVAILABLE:
        import plotly.express as px

        fig = px.histogram(
            x=values,
            nbins=35,
            labels={"x": tr(lang, "predicted_breakthrough_time_h")},
            title=tr(lang, "ml_prediction_distribution"),
        )
        fig.add_vline(
            x=float(required_time_h),
            line_dash="dash",
            line_color="red",
            annotation_text=tr(lang, "required_service_time_h"),
        )
        return fig
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(values, bins=35, color="#4C78A8", alpha=0.82)
    ax.axvline(float(required_time_h), color="#D62728", linestyle="--", linewidth=1.6, label=tr(lang, "required_service_time_h"))
    ax.set_xlabel(tr(lang, "predicted_breakthrough_time_h"))
    ax.set_ylabel("count")
    ax.set_title(tr(lang, "ml_prediction_distribution"))
    ax.legend()
    return fig


def display_prediction_histogram(all_candidates_df: pd.DataFrame, required_time_h: float, lang: str) -> None:
    fig = plot_prediction_histogram(all_candidates_df, required_time_h, lang)
    if fig is None:
        return
    if PLOTLY_AVAILABLE and hasattr(fig, "to_plotly_json"):
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.pyplot(fig)
        plt.close(fig)


def calculate_pressure_drop_ergun_for_candidate(
    scale_mode: str,
    flow_value: float,
    diameter_value: float,
    height_value: float,
    dp_mm: float,
    epsilon: float,
    temperature_C: float,
    particle_sphericity: float = 0.85,
) -> tuple[float, str]:
    try:
        if scale_mode == SCALE_LABELS[0]:
            q_m3_s = float(flow_value) * 1e-6 / 60.0
            d_m = float(diameter_value) / 100.0
            h_m = float(height_value) / 100.0
            status = "diagnostic_only_gravity_flow"
        else:
            q_m3_s = float(flow_value) / 3600.0
            d_m = float(diameter_value)
            h_m = float(height_value)
            status = "calculated"
        pressure = ergun_pressure_drop(
            Q_m3_s=q_m3_s,
            D_col_m=d_m,
            H_m=h_m,
            dp_m=float(dp_mm) / 1000.0,
            epsilon=float(epsilon),
            T_C=float(temperature_C),
            particle_sphericity=float(particle_sphericity),
        )
        return float(pressure["DeltaP_bar"]), status
    except Exception:
        return float("nan"), "pressure_calculation_failed"


def engineering_summary_for_candidate(row: pd.Series, scale_mode: str) -> str:
    if scale_mode == SCALE_LABELS[0]:
        return (
            f"D = {row['diameter']:.1f} см, H = {row['height']:.1f} см, "
            f"Q = {row['flow']:.2f} мл/мин, m ≈ {row['carbon_mass']:.1f} г, "
            f"EBCT = {row['EBCT_min']:.1f} мин"
        )
    return (
        f"D = {row['diameter']:.2f} м, H = {row['height']:.2f} м, "
        f"Q = {row['flow']:.2f} м³/ч, m ≈ {row['carbon_mass']:.1f} кг, "
        f"EBCT = {row['EBCT_min']:.1f} мин"
    )


def format_adsorber_candidates_for_display(df: pd.DataFrame, scale_mode: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    display = df.copy()
    if scale_mode == SCALE_LABELS[0]:
        formats = {
            "diameter": 1,
            "height": 1,
            "flow": 2,
            "temperature_C": 1,
            "epsilon": 2,
            "bulk_density": 3,
            "bed_volume": 1,
            "carbon_mass": 1,
            "superficial_velocity": 2,
            "interstitial_velocity": 2,
            "H_to_D_ratio": 2,
            "pressure_drop_bar": 4,
            "EBCT_min": 1,
            "predicted_breakthrough_time_h": 2,
            "predicted_breakthrough_time_days": 3,
            "safety_margin": 2,
        }
    else:
        formats = {
            "diameter": 2,
            "height": 2,
            "flow": 3,
            "temperature_C": 1,
            "epsilon": 2,
            "bulk_density": 0,
            "bed_volume": 3,
            "carbon_mass": 1,
            "superficial_velocity": 2,
            "interstitial_velocity": 2,
            "H_to_D_ratio": 2,
            "pressure_drop_bar": 3,
            "EBCT_min": 1,
            "predicted_breakthrough_time_h": 2,
            "predicted_breakthrough_time_days": 3,
            "safety_margin": 2,
        }
    for col, digits in formats.items():
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce").round(digits)
    preferred_columns = [
        "candidate_id",
        "engineering_summary",
        "decision_status",
        "reject_reason",
        "predicted_breakthrough_time_h_display",
        "prediction_status",
        "required_time_h",
        "outside_training_range",
        "prediction_unreliable",
        "failed_constraints_count",
        "diameter",
        "height",
        "flow",
        "temperature_C",
        "epsilon",
        "bulk_density",
        "dp_mm",
        "bed_volume",
        "carbon_mass",
        "EBCT_min",
        "superficial_velocity",
        "interstitial_velocity",
        "H_to_D_ratio",
        "pressure_drop_bar",
        "pressure_check_status",
        "predicted_breakthrough_time_days",
        "safety_margin",
        "passes_geometry",
        "passes_EBCT",
        "passes_velocity",
        "passes_pressure_limit",
        "passes_required_time",
        "extrapolation_warning",
    ]
    display = display[[col for col in preferred_columns if col in display.columns]]
    if scale_mode == SCALE_LABELS[0]:
        display = display.rename(
            columns={
                "diameter": "D_cm",
                "height": "H_cm",
                "flow": "flow_ml_min",
                "bulk_density": "bulk_density_g_cm3",
                "bed_volume": "bed_volume_cm3",
                "carbon_mass": "carbon_mass_g",
                "superficial_velocity": "superficial_velocity_cm_min",
                "interstitial_velocity": "interstitial_velocity_cm_min",
            }
        )
    else:
        display = display.rename(
            columns={
                "diameter": "D_m",
                "height": "H_m",
                "flow": "Q_m3_h",
                "bulk_density": "bulk_density_kg_m3",
                "bed_volume": "bed_volume_m3",
                "carbon_mass": "carbon_mass_kg",
                "superficial_velocity": "superficial_velocity_m_h",
                "interstitial_velocity": "interstitial_velocity_m_h",
            }
        )
    return display


st.set_page_config(
    page_title=tr(st.session_state.get("language", "uk"), "app_title"),
    layout="wide",
)
st.markdown(
    "<style>" + Path(__file__).with_name("ui_theme.css").read_text(encoding="utf-8") + "</style>",
    unsafe_allow_html=True,
)


def load_experiment_data() -> dict[str, list[dict]]:
    from scientific_provenance import file_hash
    return _load_experiment_data_at_revision(file_hash(DATA_PATH))


@st.cache_data
def _load_experiment_data_at_revision(content_sha256):
    return load_experiments_csv(DATA_PATH)


def load_summary_data() -> pd.DataFrame:
    from scientific_provenance import file_hash
    return _load_summary_data_at_revision(file_hash(SUMMARY_PATH))


@st.cache_data
def _load_summary_data_at_revision(content_sha256):
    return safe_read_csv(SUMMARY_PATH)


def figure_to_png_bytes(fig) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def make_figures_zip(figures_dir: str) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if os.path.isdir(figures_dir):
            for filename in sorted(os.listdir(figures_dir)):
                if filename.lower().endswith(".png"):
                    path = os.path.join(figures_dir, filename)
                    archive.write(path, arcname=filename)
    buffer.seek(0)
    return buffer.getvalue()


def results_to_dataframe(simulation: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_min": simulation["time_min"],
            "C_out_mg_L": simulation["C_out_mg_L"],
            "C_out_C0": simulation["C_out_C0"],
        }
    )


def experimental_rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows).copy()
    for column in ["time_min", "C_out_mg_L", "C_out_C0", "C0_mg_L"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "C_out_C0" not in df.columns or df["C_out_C0"].isna().all():
        if {"C_out_mg_L", "C0_mg_L"}.issubset(df.columns):
            df["C_out_C0"] = df["C_out_mg_L"] / df["C0_mg_L"].replace(0, np.nan)
    return df.sort_values("time_min").reset_index(drop=True)


def validation_split_indices(
    df: pd.DataFrame,
    mode: str,
    holdout_fraction: float = 0.25,
    seed: int = 42,
    manual_times: list[float] | None = None,
    block_name: str = "middle",
    breakthrough_ratio: float = 0.05,
) -> tuple[list[int], list[int]]:
    n_points = len(df)
    all_indices = list(range(n_points))
    if n_points < 5:
        raise ValueError("too_few_points_for_holdout")
    if mode == "manual":
        selected = set(float(value) for value in (manual_times or []))
        hidden = [idx for idx, value in enumerate(df["time_min"].to_numpy(dtype=float)) if float(value) in selected]
    elif mode == "random":
        rng = np.random.default_rng(int(seed))
        n_hidden = max(1, int(round(float(holdout_fraction) * n_points)))
        hidden = sorted(map(int, rng.choice(all_indices, size=min(n_hidden, n_points - 4), replace=False)))
    elif mode == "block":
        if block_name == "early":
            hidden = all_indices[: max(1, int(round(0.25 * n_points)))]
        elif block_name == "late":
            hidden = all_indices[-max(1, int(round(0.25 * n_points))) :]
        elif block_name == "near_breakthrough":
            ratios = df["C_out_C0"].to_numpy(dtype=float)
            hidden = sorted(np.argsort(np.abs(ratios - float(breakthrough_ratio)))[: max(1, int(round(0.25 * n_points)))].tolist())
        else:
            start = max(0, int(round(0.375 * n_points)))
            count = max(1, int(round(0.25 * n_points)))
            hidden = all_indices[start : start + count]
    else:
        hidden = [n_points // 2]
    hidden = sorted(set(int(idx) for idx in hidden if 0 <= int(idx) < n_points))
    train = [idx for idx in all_indices if idx not in hidden]
    if len(hidden) == 0:
        raise ValueError("no_hidden_points_selected")
    if len(train) < 4:
        raise ValueError("too_few_train_points_for_holdout")
    return train, hidden


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, scale: float | None = None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "nRMSE_percent": np.nan,
            "R2_Q2": np.nan,
            "sMAPE_percent": np.nan,
            "max_abs_error": np.nan,
            "n_points": 0,
        }
    residual = y_pred - y_true
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    value_range = float(scale) if scale is not None else float(np.nanmax(y_true) - np.nanmin(y_true))
    nrmse = float(100.0 * rmse / value_range) if value_range > 1e-12 else np.nan
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_values = np.where(denom > 1e-12, np.abs(residual) / denom, 0.0)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = np.nan if ss_tot <= 1e-30 else float(1.0 - ss_res / ss_tot)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "nRMSE_percent": nrmse,
        "R2_Q2": r2 if np.isfinite(r2) else None,
        "R2": r2 if np.isfinite(r2) else None,
        "sMAPE_percent": float(100.0 * np.mean(smape_values)),
        "max_abs_error": float(np.max(np.abs(residual))),
        "n_points": int(len(y_true)),
    }


def validation_status_from_metrics(metrics_validation: dict, breakthrough_error: dict | None = None) -> tuple[str, str]:
    nrmse = metrics_validation.get("nRMSE_percent")
    r2 = metrics_validation.get("R2_Q2")
    rmse = metrics_validation.get("RMSE")
    bt_percent = None
    if breakthrough_error:
        bt_percent = breakthrough_error.get("breakthrough_time_error_percent")
    good_bt = bt_percent is None or (math.isfinite(float(bt_percent)) and abs(float(bt_percent)) <= 15.0)
    if np.isfinite(nrmse) and np.isfinite(r2) and nrmse <= 15.0 and r2 >= 0.8 and good_bt:
        return "good", "experiment_validation_good_text"
    if np.isfinite(nrmse) and np.isfinite(r2) and nrmse <= 25.0 and r2 >= 0.6:
        return "acceptable", "experiment_validation_acceptable_text"
    if not np.isfinite(nrmse) or not np.isfinite(r2):
        if np.isfinite(rmse) and rmse <= 0.10 and good_bt:
            return "good", "experiment_validation_good_text"
        if np.isfinite(rmse) and rmse <= 0.20:
            return "acceptable", "experiment_validation_acceptable_text"
    return "poor", "experiment_validation_poor_text"


def plot_holdout_validation(
    experiment_id: str,
    mode_label: str,
    full_df: pd.DataFrame,
    train_indices: list[int],
    hidden_indices: list[int],
    model_time: np.ndarray | None,
    model_ratio: np.ndarray | None,
    breakthrough_ratio: float,
    hidden_predictions: pd.DataFrame | None = None,
    all_point_model_ratio: np.ndarray | None = None,
    lang: str = "ru",
):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        full_df["time_min"],
        full_df["C_out_C0"],
        "o",
        color="0.65",
        markersize=4,
        label=tr(lang, "validation_all_experimental_points"),
    )
    train_df = full_df.iloc[train_indices]
    hidden_df = full_df.iloc[hidden_indices]
    ax.plot(train_df["time_min"], train_df["C_out_C0"], "o", color="tab:blue", label=tr(lang, "validation_calibration_points"))
    ax.plot(hidden_df["time_min"], hidden_df["C_out_C0"], "o", color="tab:red", label=tr(lang, "validation_hidden_points"))
    if model_time is not None and model_ratio is not None:
        ax.plot(model_time, model_ratio, "-", color="tab:orange", linewidth=2, label=tr(lang, "validation_model_train_curve"))
    if hidden_predictions is not None and not hidden_predictions.empty:
        ax.plot(
            hidden_predictions["time_min"],
            hidden_predictions["C_pred_C0"],
            "x",
            color="black",
            markersize=7,
            label=tr(lang, "validation_predicted_hidden_points"),
        )
    if all_point_model_ratio is not None and model_time is not None:
        ax.plot(model_time, all_point_model_ratio, "--", color="tab:green", linewidth=1.4, label=tr(lang, "validation_model_all_points"))
    ax.axhline(
        breakthrough_ratio,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=tr(lang, "breakthrough_boundary_label").format(value=f"{breakthrough_ratio:.3f}"),
    )
    ax.set_xlabel(tr(lang, "time_min"))
    ax.set_ylabel("C/C0")
    ax.set_title(f"{experiment_id} | {mode_label}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def run_single_holdout_validation(
    experiment_id: str,
    rows: list[dict],
    train_indices: list[int],
    hidden_indices: list[int],
    epsilon: float,
    min_time_for_fit: float | None,
    n_cells: int,
    fit_initial_concentration: bool,
    breakthrough_ratio: float,
    compare_full_fit: bool,
    mode_label: str,
    lang: str,
) -> dict:
    full_df = experimental_rows_to_dataframe(rows)
    train_rows = [dict(rows[idx]) for idx in train_indices]
    hidden_df = full_df.iloc[hidden_indices].copy()
    fitted, fit_metrics = fit_model_to_experiment(
        train_rows,
        epsilon=epsilon,
        min_time_for_fit=min_time_for_fit,
        n_cells=n_cells,
        fit_initial_concentration=fit_initial_concentration,
    )
    column = make_column_from_experiment(rows, epsilon=epsilon)
    model_time = np.linspace(0.0, float(full_df["time_min"].max()), 240)
    model_conc = simulate_breakthrough(model_time, column, fitted, n_cells=n_cells)
    model_ratio = model_conc / max(column.c0_mg_l, 1e-30)
    train_time = full_df.iloc[train_indices]["time_min"].to_numpy(dtype=float)
    train_exp = full_df.iloc[train_indices]["C_out_C0"].to_numpy(dtype=float)
    train_pred = simulate_breakthrough(train_time, column, fitted, n_cells=n_cells) / max(column.c0_mg_l, 1e-30)
    hidden_time = hidden_df["time_min"].to_numpy(dtype=float)
    hidden_exp = hidden_df["C_out_C0"].to_numpy(dtype=float)
    hidden_pred = simulate_breakthrough(hidden_time, column, fitted, n_cells=n_cells) / max(column.c0_mg_l, 1e-30)
    residuals_df = pd.DataFrame(
        {
            "time_min": hidden_time,
            "C_exp_C0": hidden_exp,
            "C_pred_C0": hidden_pred,
            "residual": hidden_pred - hidden_exp,
            "abs_error": np.abs(hidden_pred - hidden_exp),
            "squared_error": (hidden_pred - hidden_exp) ** 2,
        }
    )
    train_metrics = regression_metrics(train_exp, train_pred)
    validation_metrics = regression_metrics(hidden_exp, hidden_pred)
    exp_bt = breakthrough_status(full_df["time_min"].to_numpy(dtype=float), full_df["C_out_C0"].to_numpy(dtype=float), breakthrough_ratio)
    model_bt = breakthrough_status(model_time, model_ratio, breakthrough_ratio)
    breakthrough_error = None
    if exp_bt.get("time_min") is not None and model_bt.get("time_min") is not None:
        exp_time = float(exp_bt["time_min"])
        model_time_bt = float(model_bt["time_min"])
        breakthrough_error = {
            "t_breakthrough_exp_min": exp_time,
            "t_breakthrough_model_min": model_time_bt,
            "breakthrough_time_error_min": model_time_bt - exp_time,
            "breakthrough_time_error_percent": 100.0 * (model_time_bt - exp_time) / max(exp_time, 1e-30),
        }
    all_point_model_ratio = None
    if compare_full_fit:
        all_fitted, _ = fit_model_to_experiment(
            rows,
            epsilon=epsilon,
            min_time_for_fit=min_time_for_fit,
            n_cells=n_cells,
            fit_initial_concentration=fit_initial_concentration,
        )
        all_point_model_ratio = simulate_breakthrough(model_time, column, all_fitted, n_cells=n_cells) / max(column.c0_mg_l, 1e-30)
    status, recommendation_key = validation_status_from_metrics(validation_metrics, breakthrough_error)
    fig = plot_holdout_validation(
        experiment_id,
        mode_label,
        full_df,
        train_indices,
        hidden_indices,
        model_time,
        model_ratio,
        breakthrough_ratio,
        hidden_predictions=residuals_df,
        all_point_model_ratio=all_point_model_ratio,
        lang=lang,
    )
    report = {
        "experiment_id": experiment_id,
        "validation_mode": mode_label,
        "hidden_points": residuals_df[["time_min", "C_exp_C0", "C_pred_C0", "residual"]].to_dict(orient="records"),
        "fitted_parameters_train_only": {
            "KH_L_g": float(fitted.KH_L_g),
            "kLDF_1_min": float(fitted.kLDF_1_min),
            "Dax_cm2_min": float(fitted.Dax_cm2_min),
            "c_initial_mg_L": float(fitted.c_initial_mg_l),
        },
        "metrics_train": train_metrics,
        "metrics_validation": validation_metrics,
        "breakthrough_error": breakthrough_error,
        "breakthrough_status_experiment": exp_bt,
        "breakthrough_status_model": model_bt,
        "validation_status": status,
        "recommendation": tr(lang, recommendation_key),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {
        "report": report,
        "residuals_df": residuals_df,
        "metrics_train": train_metrics,
        "metrics_validation": validation_metrics,
        "figure": fig,
        "plot_png": figure_to_png_bytes(fig),
        "fitted": fitted,
        "fit_metrics": fit_metrics,
    }


def run_leave_one_out_validation(
    experiment_id: str,
    rows: list[dict],
    epsilon: float,
    min_time_for_fit: float | None,
    n_cells: int,
    fit_initial_concentration: bool,
    breakthrough_ratio: float,
    lang: str,
) -> dict:
    full_df = experimental_rows_to_dataframe(rows)
    column = make_column_from_experiment(rows, epsilon=epsilon)
    residual_rows: list[dict] = []
    fitted_params: list[dict] = []
    progress = st.progress(0.0)
    for idx in range(len(rows)):
        train_indices = [j for j in range(len(rows)) if j != idx]
        train_rows = [dict(rows[j]) for j in train_indices]
        try:
            fitted, _metrics = fit_model_to_experiment(
                train_rows,
                epsilon=epsilon,
                min_time_for_fit=min_time_for_fit,
                n_cells=n_cells,
                fit_initial_concentration=fit_initial_concentration,
            )
            time_i = float(full_df.iloc[idx]["time_min"])
            exp_i = float(full_df.iloc[idx]["C_out_C0"])
            pred_i = float(simulate_breakthrough([time_i], column, fitted, n_cells=n_cells)[0] / max(column.c0_mg_l, 1e-30))
            residual_rows.append(
                {
                    "point_index": idx,
                    "time_min": time_i,
                    "C_exp_C0": exp_i,
                    "C_pred_C0": pred_i,
                    "residual": pred_i - exp_i,
                    "abs_error": abs(pred_i - exp_i),
                    "squared_error": (pred_i - exp_i) ** 2,
                }
            )
            fitted_params.append({"point_index": idx, "KH_L_g": fitted.KH_L_g, "kLDF_1_min": fitted.kLDF_1_min, "Dax_cm2_min": fitted.Dax_cm2_min})
        finally:
            progress.progress(min(1.0, float(idx + 1) / max(len(rows), 1)))
    progress.empty()
    residuals_df = pd.DataFrame(residual_rows)
    metrics_validation = regression_metrics(residuals_df["C_exp_C0"].to_numpy(dtype=float), residuals_df["C_pred_C0"].to_numpy(dtype=float))
    status, recommendation_key = validation_status_from_metrics(metrics_validation, None)
    fig = plot_holdout_validation(
        experiment_id,
        "Leave-one-out",
        full_df,
        list(range(len(rows))),
        list(range(len(rows))),
        None,
        None,
        breakthrough_ratio,
        hidden_predictions=residuals_df,
        lang=lang,
    )
    report = {
        "experiment_id": experiment_id,
        "validation_mode": "leave_one_out",
        "hidden_points": residuals_df[["time_min", "C_exp_C0", "C_pred_C0", "residual"]].to_dict(orient="records"),
        "fitted_parameters_train_only": fitted_params,
        "metrics_train": {},
        "metrics_validation": metrics_validation,
        "breakthrough_error": None,
        "validation_status": status,
        "recommendation": tr(lang, recommendation_key),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {
        "report": report,
        "residuals_df": residuals_df,
        "metrics_train": {},
        "metrics_validation": metrics_validation,
        "figure": fig,
        "plot_png": figure_to_png_bytes(fig),
    }


def plot_manual_holdout_validation(
    experiment_id: str,
    full_df: pd.DataFrame,
    train_indices: list[int],
    hidden_indices: list[int],
    model_time: np.ndarray,
    model_ratio: np.ndarray,
    breakthrough_ratio: float,
    all_point_model_ratio: np.ndarray | None = None,
    lang: str = "ru",
):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        full_df["time_min"],
        full_df["C_out_C0"],
        "o-",
        color="0.65",
        linewidth=1.0,
        markersize=4,
        label=tr(lang, "manual_validation_experimental_curve_all_points"),
    )
    train_df = full_df.iloc[train_indices]
    hidden_df = full_df.iloc[hidden_indices]
    ax.plot(
        train_df["time_min"],
        train_df["C_out_C0"],
        "o",
        color="tab:blue",
        markersize=5,
        label=tr(lang, "points_used_for_calibration"),
    )
    ax.plot(
        hidden_df["time_min"],
        hidden_df["C_out_C0"],
        "o",
        color="tab:red",
        markersize=6,
        label=tr(lang, "excluded_points_for_validation"),
    )
    ax.plot(
        model_time,
        model_ratio,
        "-",
        color="tab:orange",
        linewidth=2.2,
        label=tr(lang, "manual_validation_model_remaining_points"),
    )
    if all_point_model_ratio is not None:
        ax.plot(
            model_time,
            all_point_model_ratio,
            "--",
            color="tab:green",
            linewidth=1.6,
            label=tr(lang, "validation_model_all_points"),
        )
    ax.axhline(
        breakthrough_ratio,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label=tr(lang, "breakthrough_boundary_label").format(value=f"{breakthrough_ratio:.3f}"),
    )
    ax.set_xlabel(tr(lang, "manual_validation_x_axis"))
    ax.set_ylabel("C/C0")
    ax.set_title(tr(lang, "manual_validation_plot_title").format(experiment_id=experiment_id))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def run_manual_calibration_holdout_validation(
    experiment_id: str,
    rows: list[dict],
    hidden_indices: list[int],
    epsilon: float,
    min_time_for_fit: float | None,
    n_cells: int,
    fit_initial_concentration: bool,
    breakthrough_ratio: float,
    compare_full_fit: bool,
    lang: str,
) -> dict:
    full_df = experimental_rows_to_dataframe(rows)
    all_indices = list(range(len(full_df)))
    hidden_indices = sorted(set(int(index) for index in hidden_indices))
    if any(index < 0 or index >= len(full_df) for index in hidden_indices):
        raise ValueError("Invalid hidden point index")
    train_indices = [index for index in all_indices if index not in hidden_indices]
    if not hidden_indices:
        raise ValueError(tr(lang, "select_at_least_one_hidden_point"))
    if len(train_indices) < 4:
        raise ValueError(tr(lang, "not_enough_train_points_after_exclusion"))

    train_rows = [dict(rows[index]) for index in train_indices]
    fitted, _fit_metrics = fit_model_to_experiment(
        train_rows,
        epsilon=epsilon,
        min_time_for_fit=min_time_for_fit,
        n_cells=n_cells,
        fit_initial_concentration=fit_initial_concentration,
    )
    if not _fit_metrics["success"]:
        raise RuntimeError(_fit_metrics["message"])
    column = make_column_from_experiment(rows, epsilon=epsilon)
    model_time = np.linspace(0.0, float(full_df["time_min"].max()), 300)
    model_conc = simulate_breakthrough(model_time, column, fitted, n_cells=n_cells)
    model_ratio = model_conc / max(column.c0_mg_l, 1e-30)

    measured_interval = model_time >= float(full_df["time_min"].min())
    exp_interp = np.interp(
        model_time[measured_interval],
        full_df["time_min"].to_numpy(dtype=float),
        full_df["C_out_C0"].to_numpy(dtype=float),
    )
    scale = max(float(np.nanmax(full_df["C_out_C0"]) - np.nanmin(full_df["C_out_C0"])), 0.05)
    curve_residual = model_ratio[measured_interval] - exp_interp
    rmse_curve = float(np.sqrt(np.mean(curve_residual**2)))
    curve_similarity = float(np.clip(100.0 * (1.0 - rmse_curve / scale), 0.0, 100.0))

    hidden_df = full_df.iloc[hidden_indices].copy()
    hidden_time = hidden_df["time_min"].to_numpy(dtype=float)
    hidden_exp = hidden_df["C_out_C0"].to_numpy(dtype=float)
    hidden_pred = simulate_breakthrough(hidden_time, column, fitted, n_cells=n_cells) / max(column.c0_mg_l, 1e-30)
    hidden_residual = hidden_pred - hidden_exp
    rmse_hidden = float(np.sqrt(np.mean(hidden_residual**2)))
    mae_hidden = float(np.mean(np.abs(hidden_residual)))
    max_abs_error_hidden = float(np.max(np.abs(hidden_residual)))
    hidden_similarity = float(np.clip(100.0 * (1.0 - rmse_hidden / scale), 0.0, 100.0))
    metrics_validation = regression_metrics(hidden_exp, hidden_pred, scale=scale)
    fit_times = np.asarray(_fit_metrics["times_min"])
    fit_mask = fit_times >= float(_fit_metrics["fit_start_time_min"])
    fitted_train_indices = [index for index in train_indices
                            if float(full_df.iloc[index]["time_min"]) >= float(_fit_metrics["fit_start_time_min"])]
    metrics_train = regression_metrics(
        np.asarray(_fit_metrics["observed_mg_L"])[fit_mask] / column.c0_mg_l,
        np.asarray(_fit_metrics["predicted_mg_L"])[fit_mask] / column.c0_mg_l,
        scale=scale,
    )

    residuals_df = pd.DataFrame(
        {
            "time_min": hidden_time,
            "C_exp_C0": hidden_exp,
            "C_pred_C0": hidden_pred,
            "residual": hidden_residual,
            "abs_error": np.abs(hidden_residual),
            "abs_error_percent_of_scale": 100.0 * np.abs(hidden_residual) / max(scale, 1e-30),
        }
    )
    if metrics_validation["nRMSE_percent"] <= 15.0 and (metrics_validation["R2"] is None or metrics_validation["R2"] >= 0.8):
        validation_status = "good"
        recommendation_key = "validation_curve_good"
    elif metrics_validation["nRMSE_percent"] <= 25.0 and (metrics_validation["R2"] is None or metrics_validation["R2"] >= 0.6):
        validation_status = "acceptable"
        recommendation_key = "validation_curve_acceptable"
    else:
        validation_status = "poor"
        recommendation_key = "validation_curve_poor"

    all_point_model_ratio = None
    if compare_full_fit:
        all_fitted, all_fit_metrics = fit_model_to_experiment(
            rows,
            epsilon=epsilon,
            min_time_for_fit=min_time_for_fit,
            n_cells=n_cells,
            fit_initial_concentration=fit_initial_concentration,
        )
        if not all_fit_metrics["success"]:
            raise RuntimeError(all_fit_metrics["message"])
        all_point_model_ratio = simulate_breakthrough(model_time, column, all_fitted, n_cells=n_cells) / max(column.c0_mg_l, 1e-30)

    fig = plot_manual_holdout_validation(
        experiment_id,
        full_df,
        fitted_train_indices,
        hidden_indices,
        model_time,
        model_ratio,
        breakthrough_ratio,
        all_point_model_ratio=all_point_model_ratio,
        lang=lang,
    )
    metrics_df = pd.DataFrame(
        [
            {"metric": "curve_similarity_percent", "value": curve_similarity},
            {"metric": "hidden_points_similarity_percent", "value": hidden_similarity},
            {"metric": "RMSE_curve", "value": rmse_curve},
            {"metric": "RMSE_hidden", "value": rmse_hidden},
            {"metric": "MAE_hidden", "value": mae_hidden},
            {"metric": "max_abs_error_hidden", "value": max_abs_error_hidden},
            {"metric": "number_of_hidden_points", "value": len(hidden_indices)},
            {"metric": "number_of_train_points", "value": len(fitted_train_indices)},
            {"metric": "number_of_retained_points", "value": len(train_indices)},
        ]
    )
    primary_rows = [{"metric": f"{scope}_{metric}", "value": values.get(metric)}
                    for scope, values in [("train", metrics_train), ("hidden", metrics_validation)]
                    for metric in ["RMSE", "MAE", "R2", "n_points"]]
    metrics_df = pd.concat([pd.DataFrame(primary_rows), metrics_df], ignore_index=True)
    report = {
        "experiment_id": experiment_id,
        "hidden_point_indices": hidden_indices,
        "hidden_times_min": [float(full_df.iloc[index]["time_min"]) for index in hidden_indices],
        "metrics_train": metrics_train,
        "metrics_validation": metrics_validation,
        "validation_scope": "internal_hidden_points_of_one_experiment_not_independent_validation",
        "fit_configuration": {"n_cells": n_cells, "model_version": _fit_metrics.get("model_version"), "small_signal_policy": "retain_algebraic_unclipped", "fit_start_time_min": float(_fit_metrics["fit_start_time_min"])},
        "train_point_indices": fitted_train_indices,
        "retained_point_indices": train_indices,
        "fitted_parameters_train_only": {
            "KH_L_g": float(fitted.KH_L_g),
            "kLDF_1_min": float(fitted.kLDF_1_min),
            "Dax_cm2_min": float(fitted.Dax_cm2_min),
            "c_initial_mg_L": float(fitted.c_initial_mg_l),
        },
        "curve_similarity_percent": curve_similarity,
        "hidden_similarity_percent": hidden_similarity,
        "RMSE_curve": rmse_curve,
        "RMSE_hidden": rmse_hidden,
        "MAE_hidden": mae_hidden,
        "max_abs_error_hidden": max_abs_error_hidden,
        "validation_status": validation_status,
        "recommendation": tr(lang, recommendation_key),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return {
        "report": report,
        "residuals_df": residuals_df,
        "metrics_df": metrics_df,
        "fig": fig,
        "plot_png": figure_to_png_bytes(fig),
    }


def df_ui(df: pd.DataFrame, lang: str, column_key_map: dict[str, str]) -> pd.DataFrame:
    return df.rename(
        columns={
            column: tr(lang, key)
            for column, key in column_key_map.items()
            if column in df.columns
        }
    )


def profile_label(profile_key: str, lang: str = "ru") -> str:
    profile = ADSORBENT_PROFILES[profile_key]
    return fix_mojibake(tr(lang, profile.get("name_key", profile_key)))


def passport_bulk_density_range(profile_key: str) -> tuple[float | None, float | None]:
    profile = ADSORBENT_PROFILES[profile_key]
    return profile.get("passport_bulk_density_min"), profile.get("passport_bulk_density_max")


def experimental_bed_density_range(profile_key: str) -> tuple[float | None, float | None]:
    profile = ADSORBENT_PROFILES[profile_key]
    return profile.get("experimental_bed_density_min"), profile.get("experimental_bed_density_max")


def density_range_text(min_value: float | None, max_value: float | None, lang: str = "ru") -> str:
    if min_value is None or max_value is None:
        return "—"
    unit = "g/cm³" if lang == "en" else "г/см³"
    return f"{min_value:.2f}–{max_value:.2f} {unit}"


def required_diameter_range_for_passport_density(
    mass_g: float,
    height_cm: float,
    profile_key: str,
) -> tuple[float | None, float | None]:
    rho_min, rho_max = passport_bulk_density_range(profile_key)
    if rho_min is None or rho_max is None or height_cm <= 0:
        return None, None
    diameter_for_rho_max = math.sqrt(4.0 * mass_g / (math.pi * rho_max * height_cm))
    diameter_for_rho_min = math.sqrt(4.0 * mass_g / (math.pi * rho_min * height_cm))
    return float(diameter_for_rho_max), float(diameter_for_rho_min)


def render_adsorbent_card(profile_key: str, lang: str) -> None:
    profile = ADSORBENT_PROFILES[profile_key]
    rows = [
        (tr(lang, "col_type"), tr(lang, profile.get("type_key", "")) if profile.get("type_key") else ""),
        (tr(lang, "col_manufacturer"), fix_mojibake(profile.get("manufacturer"))),
        (tr(lang, "col_country"), tr(lang, profile.get("country_key", "")) if profile.get("country_key") else ""),
        (tr(lang, "moisture"), fix_mojibake(profile.get("moisture"))),
        (tr(lang, "ash"), fix_mojibake(profile.get("ash"))),
        (tr(lang, "iodine_number"), fix_mojibake(profile.get("iodine_number"))),
        (tr(lang, "passport_bulk_density"), density_range_text(profile.get("passport_bulk_density_min"), profile.get("passport_bulk_density_max"), lang)),
        (
            tr(lang, "experimental_bed_density"),
            density_range_text(profile.get("experimental_bed_density_min"), profile.get("experimental_bed_density_max"), lang),
        ),
        ("BET surface area", fix_mojibake(profile.get("bet_surface_area"))),
        (tr(lang, "hardness"), fix_mojibake(profile.get("hardness"))),
        ("pH", fix_mojibake(profile.get("ph"))),
        (tr(lang, "methylene_blue_adsorption"), fix_mojibake(profile.get("methylene_blue_adsorption"))),
        (tr(lang, "col_comment"), tr(lang, profile.get("comment_key", "")) if profile.get("comment_key") else ""),
    ]
    table = pd.DataFrame(
        [{tr(lang, "col_characteristic"): name, tr(lang, "col_value"): value or "—"} for name, value in rows]
    )
    st.markdown(f"**{profile_label(profile_key, lang)}**")
    st.dataframe(clean_dataframe_text(table), width="stretch", hide_index=True)

    passport_min, passport_max = passport_bulk_density_range(profile_key)
    if passport_min is not None and passport_max is not None:
        st.info(
            f"{tr(lang, 'passport_bulk_density')}: "
            f"{passport_min:.2f}–{passport_max:.2f} г/см³"
        )
    experimental_min, experimental_max = experimental_bed_density_range(profile_key)
    if experimental_min is not None and experimental_max is not None:
        st.info(
            f"{tr(lang, 'experimental_bed_density')}: "
            f"{experimental_min:.2f}–{experimental_max:.2f} г/см³"
        )
        if profile.get("experimental_bed_density_comment_key"):
            st.caption(tr(lang, profile["experimental_bed_density_comment_key"]))


def calculate_bed_geometry(diameter_cm: float, height_cm: float, mass_g: float) -> dict:
    area_cm2 = column_area_from_diameter(diameter_cm)
    bed_volume_cm3 = area_cm2 * height_cm
    bulk_density = mass_g / bed_volume_cm3 if bed_volume_cm3 > 0 else np.nan
    return {
        "area_cm2": area_cm2,
        "bed_volume_cm3": bed_volume_cm3,
        "bulk_density_g_cm3": bulk_density,
    }


def experiment_bed_diagnostics(rows: list[dict]) -> pd.DataFrame:
    first = rows[0]
    height = float(first["bed_height_cm"])
    mass = float(first["carbon_mass_g"])
    area = float(first["column_area_cm2"])
    d_equiv = math.sqrt(4.0 * area / math.pi)
    loading_density = mass / (area * height)
    return pd.DataFrame(
        [
            {
                "bed_height_cm": height,
                "carbon_mass_g": mass,
                "column_area_cm2 из Excel": area,
                "D_equiv из Excel, см": d_equiv,
                "расчётная плотность загрузки, г/см³": loading_density,
            }
        ]
    )


def render_density_check(profile_key: str, diameter_cm: float, height_cm: float, mass_g: float, lang: str) -> dict:
    geometry = calculate_bed_geometry(diameter_cm, height_cm, mass_g)
    st.dataframe(
        pd.DataFrame(
            [
                {
                    f"{tr(lang, 'col_column_area')}, см²": geometry["area_cm2"],
                    f"{tr(lang, 'col_bed_volume')}, см³": geometry["bed_volume_cm3"],
                    f"{tr(lang, 'col_carbon_mass')}, г": mass_g,
                    f"{tr(lang, 'col_bed_loading_density')}, г/см³": geometry[
                        "bulk_density_g_cm3"
                    ],
                    tr(lang, "col_passport_range"): density_range_text(
                        *passport_bulk_density_range(profile_key), lang
                    ),
                    tr(lang, "col_experimental_range"): density_range_text(
                        *experimental_bed_density_range(profile_key), lang
                    ),
                }
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    passport_min, passport_max = passport_bulk_density_range(profile_key)
    experimental_min, experimental_max = experimental_bed_density_range(profile_key)
    bulk_density = geometry["bulk_density_g_cm3"]
    in_passport = (
        passport_min is not None and passport_max is not None and passport_min <= bulk_density <= passport_max
    )
    in_experimental = (
        experimental_min is not None
        and experimental_max is not None
        and experimental_min <= bulk_density <= experimental_max
    )

    if profile_key != "custom" and passport_min is not None and passport_max is not None:
        if in_passport:
            st.success(tr(lang, "density_matches_passport"))
        elif in_experimental:
            st.info(tr(lang, "density_matches_experiment"))
        else:
            st.warning(tr(lang, "density_outside_ranges"))

        d_min, d_max = required_diameter_range_for_passport_density(
            mass_g,
            height_cm,
            profile_key,
        )
        if d_min is not None and d_max is not None:
            st.info(
                tr(lang, "expected_diameter_hint").format(
                    rho_min=passport_min,
                    rho_max=passport_max,
                    d_min=d_min,
                    d_max=d_max,
                )
            )
    return geometry


def build_breakthrough_figure(
    time_min: np.ndarray,
    model_ratio: np.ndarray,
    breakthrough_ratio: float,
    model_breakthrough_time_min: float | None,
    experiment_time: np.ndarray | None = None,
    experiment_ratio: np.ndarray | None = None,
    experiment_breakthrough_time_min: float | None = None,
    experiment_breakthrough_label: str | None = None,
    experiment_breakthrough_status: str | None = None,
    model_breakthrough_label: str | None = None,
    model_breakthrough_status: str | None = None,
    fit_start_time_min: float | None = None,
    startup_end_time_min: float | None = None,
    title: str = "",
    show_experiment_line: bool = True,
    lang: str = "ru",
):
    fig, ax = plt.subplots(figsize=(9, 5))

    if experiment_time is not None and experiment_ratio is not None:
        if show_experiment_line:
            ax.plot(
                experiment_time,
                experiment_ratio,
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=tr(lang, "experimental_line"),
            )
        ax.plot(
            experiment_time,
            experiment_ratio,
            "o",
            markersize=4,
            label=tr(lang, "experimental_points"),
        )

    ax.plot(time_min, model_ratio, "-", linewidth=2, label=tr(lang, "model_curve"))
    ax.axhline(
        breakthrough_ratio,
        color="tab:red",
        linestyle="--",
        linewidth=1.4,
        label=tr(lang, "breakthrough_boundary_label").format(value=f"{breakthrough_ratio:.3f}"),
    )

    if experiment_breakthrough_time_min is not None:
        ax.axvline(
            experiment_breakthrough_time_min,
            color="tab:green",
            linestyle=":",
            linewidth=1.6,
            label=(
                f"{tr(lang, 'breakthrough_time_experiment')}: {experiment_breakthrough_label}"
                if experiment_breakthrough_label
                else f"{tr(lang, 'breakthrough_time_experiment')} = {experiment_breakthrough_time_min:.2f} мин"
            ),
        )
    elif experiment_breakthrough_status == "not_reached":
        ax.text(
            0.02,
            0.93,
            f"{tr(lang, 'experiment')}: {tr(lang, 'breakthrough_not_reached')}",
            transform=ax.transAxes,
            fontsize=8,
            color="tab:green",
            va="top",
        )

    if model_breakthrough_time_min is not None:
        ax.axvline(
            model_breakthrough_time_min,
            color="tab:orange",
            linestyle="-.",
            linewidth=1.6,
            label=(
                f"{tr(lang, 'breakthrough_time_model')}: {model_breakthrough_label}"
                if model_breakthrough_label
                else f"{tr(lang, 'breakthrough_time_model')} = {model_breakthrough_time_min:.2f} мин"
            ),
        )
    elif model_breakthrough_status == "not_reached":
        ax.text(
            0.02,
            0.86,
            f"{tr(lang, 'model_curve')}: {tr(lang, 'breakthrough_not_reached')}",
            transform=ax.transAxes,
            fontsize=8,
            color="tab:orange",
            va="top",
        )

    if fit_start_time_min is not None and fit_start_time_min > 0:
        ax.axvline(
            fit_start_time_min,
            color="tab:gray",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label=tr(lang, "fit_start_label").format(value=f"{fit_start_time_min:.0f}"),
        )
    if startup_end_time_min is not None:
        ax.axvline(
            startup_end_time_min,
            color="tab:purple",
            linestyle=":",
            linewidth=1.0,
            alpha=0.8,
            label=tr(lang, "startup_end_label").format(value=f"{startup_end_time_min:.2f}"),
        )

    ax.set_xlabel(tr(lang, "time_min"))
    ax.set_ylabel("C/C0")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def animate_curve(
    placeholder,
    time_min: np.ndarray,
    model_ratio: np.ndarray,
    breakthrough_ratio: float,
    breakthrough_time_min: float | None,
    duration_s: int,
):
    frame_count = min(max(len(time_min), 20), 90)
    frame_indexes = np.unique(np.linspace(1, len(time_min), frame_count).astype(int))
    delay = duration_s / max(len(frame_indexes), 1)

    for index in frame_indexes:
        fig = build_breakthrough_figure(
            time_min[:index],
            model_ratio[:index],
            breakthrough_ratio,
            breakthrough_time_min if time_min[index - 1] >= (breakthrough_time_min or np.inf) else None,
            title="Анимация рассчитанной кривой прорыва",
        )
        placeholder.pyplot(fig, clear_figure=True)
        plt.close(fig)
        time.sleep(delay)


def metrics_dataframe(
    model_breakthrough_time_min: float | None,
    flow_ml_min: float,
    rmse: float | None = None,
    experiment_breakthrough_time_min: float | None = None,
    r2: float | None = None,
    model_breakthrough_status: str | None = None,
    model_breakthrough_bound_type: str | None = None,
    experiment_breakthrough_status: str | None = None,
    experiment_breakthrough_bound_type: str | None = None,
) -> pd.DataFrame:
    model_volume = (
        None
        if model_breakthrough_time_min is None or model_breakthrough_bound_type != "exact"
        else flow_ml_min * model_breakthrough_time_min
    )
    time_error = (
        None
        if (
            model_breakthrough_time_min is None
            or experiment_breakthrough_time_min is None
            or model_breakthrough_bound_type != "exact"
            or experiment_breakthrough_bound_type != "exact"
        )
        else model_breakthrough_time_min - experiment_breakthrough_time_min
    )
    return pd.DataFrame(
        [
            {
                "t прорыва, эксперимент, мин": experiment_breakthrough_time_min,
                "статус t, эксперимент": experiment_breakthrough_status,
                "тип значения t, эксперимент": experiment_breakthrough_bound_type,
                "t прорыва, модель, мин": model_breakthrough_time_min,
                "статус t, модель": model_breakthrough_status,
                "тип значения t, модель": model_breakthrough_bound_type,
                "ошибка t прорыва, мин": time_error,
                "V_b модели = Q * t_b, мл": model_volume,
                "RMSE, мг/л": rmse,
                "R²": r2,
            }
        ]
    )


def range_input(label: str, min_key: str, max_key: str, default_min: float, default_max: float, **kwargs) -> tuple[float, float]:
    col_min, col_max = st.columns(2)
    with col_min:
        if min_key in st.session_state:
            min_value = st.number_input(f"{label} min", key=min_key, **kwargs)
        else:
            min_value = st.number_input(f"{label} min", value=default_min, key=min_key, **kwargs)
    with col_max:
        if max_key in st.session_state:
            max_value = st.number_input(f"{label} max", key=max_key, **kwargs)
        else:
            max_value = st.number_input(f"{label} max", value=default_max, key=max_key, **kwargs)
    return float(min_value), float(max_value)


def validate_ranges(ranges: dict[str, tuple[float, float]]) -> list[str]:
    errors = []
    for name, (min_value, max_value) in ranges.items():
        if min_value > max_value:
            errors.append(f"{name}: minimum больше maximum.")
    return errors


def apply_industrial_preset(scale_mode: str) -> None:
    preset = INDUSTRIAL_SCALE_PRESETS[scale_mode]
    for name, value in preset.items():
        st.session_state[f"ind_{name}"] = value


def selected_experiment_parameters(experiment_id: str) -> dict[str, float]:
    if os.path.exists("fitted_experiment_parameters.csv"):
        try:
            fitted_df = safe_read_csv("fitted_experiment_parameters.csv")
            match = fitted_df[fitted_df["experiment_id"] == experiment_id]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "KH_ref_L_g": float(row["KH_L_g"]),
                    "kLDF_ref_1_min": float(row["kLDF_1_min"]),
                    "Dax_ref_cm2_min": float(row["Dax_cm2_min"]),
                    "T_ref_C": float(row.get("T_ref_C", row.get("T_exp_C", row.get("temperature_C", 20.0))) or 20.0),
                }
        except Exception:
            pass
    experiments = load_experiment_data()
    rows = experiments[experiment_id]
    fitted, _metrics = fit_model_to_experiment(rows, fit_initial_concentration=False)
    return {
        "KH_ref_L_g": fitted.KH_L_g,
        "kLDF_ref_1_min": fitted.kLDF_1_min,
        "Dax_ref_cm2_min": fitted.Dax_cm2_min,
        "T_ref_C": float(rows[0].get("temperature_C") or 20.0),
    }


def fitted_temperature_dataframe() -> pd.DataFrame | None:
    if not os.path.exists("fitted_experiment_parameters.csv"):
        return None
    try:
        return safe_read_csv("fitted_experiment_parameters.csv")
    except Exception:
        return None


def temperature_reference_from_params(params: dict[str, float], fallback: float = 20.0) -> float:
    for key in ("T_ref_C", "T_exp_C", "temperature_C"):
        value = params.get(key)
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_float):
            return value_float
    return float(fallback)


def make_temperature_adaptation_dataframe(adaptation: dict) -> pd.DataFrame:
    keys = [
        "T_ref_C",
        "T_sim_C",
        "KH_ref_L_g",
        "KH_used_L_g",
        "KH_temperature_factor",
        "kLDF_ref_1_min",
        "kLDF_used_1_min",
        "kLDF_temperature_factor",
        "Dax_ref_cm2_min",
        "Dax_used_cm2_min",
        "Dax_temperature_factor",
        "temperature_correction_source",
        "temperature_correction_confidence",
    ]
    return pd.DataFrame(
        [
            {
                "parameter": key,
                "value": format_diagnostic_value(adaptation.get(key)),
            }
            for key in keys
        ]
    )


def make_sorption_parameter_source_dataframe(params: dict, lang: str = "ru", usage_context=None) -> pd.DataFrame:
    return parameter_source_table(params, lang, format_diagnostic_value,
                                  usage_context or {"solver": "henry_ldf"})


def temperature_adaptation_messages(adaptation: dict) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    if adaptation.get("temperature_correction_source") == "literature_engineering_default":
        messages.append(("warning", "temperature_engineering_default_warning"))
    if abs(float(adaptation["T_sim_C"]) - float(adaptation["T_ref_C"])) <= 3.0:
        messages.append(("info", "temperature_close_to_calibration"))
    else:
        messages.append(("warning", "temperature_differs_from_calibration"))
    t_min, t_max = laboratory_temperature_range()
    if t_min is not None and t_max is not None:
        t_sim = float(adaptation["T_sim_C"])
        if t_sim < t_min or t_sim > t_max:
            messages.append(("warning", "temperature_outside_lab_range"))
    return messages


def unit_samples(method: str, n_cases: int, n_dims: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if method == "Latin Hypercube Sampling":
        sampler = qmc.LatinHypercube(d=n_dims, seed=seed)
        return sampler.random(n_cases)
    if method == "grid":
        levels = max(2, int(math.ceil(n_cases ** (1.0 / n_dims))))
        axis = np.linspace(0.0, 1.0, levels)
        grid = np.array(list(product(axis, repeat=n_dims)), dtype=float)
        if len(grid) < n_cases:
            repeats = int(math.ceil(n_cases / len(grid)))
            grid = np.tile(grid, (repeats, 1))
        rng.shuffle(grid)
        return grid[:n_cases]
    return rng.random((n_cases, n_dims))


def scale_samples(unit: np.ndarray, ranges: list[tuple[float, float]]) -> np.ndarray:
    mins = np.array([item[0] for item in ranges], dtype=float)
    maxs = np.array([item[1] for item in ranges], dtype=float)
    return mins + unit * (maxs - mins)


def generate_ml_dataset_from_ranges(
    profile_key: str,
    ranges: dict[str, tuple[float, float]],
    number_of_cases: int,
    time_step_min: float,
    random_seed: int,
    method: str,
    density_source: str,
) -> pd.DataFrame:
    profile = ADSORBENT_PROFILES[profile_key]
    dimension_names = [
        "C0_mg_L",
        "C_breakthrough_ratio",
        "flow_ml_min",
        "bed_height_cm",
        "column_diameter_cm",
        "bulk_density_g_cm3",
        "epsilon",
        "temperature_C",
        "KH_L_g",
        "kLDF_1_min",
        "Dax_cm2_min",
        "t_end_min",
    ]
    sample_ranges = [ranges[name] for name in dimension_names]
    samples = scale_samples(
        unit_samples(method, number_of_cases, len(dimension_names), random_seed),
        sample_ranges,
    )

    rows: list[dict] = []
    for case_index, values in enumerate(samples, start=1):
        params = dict(zip(dimension_names, values))
        area_cm2 = column_area_from_diameter(params["column_diameter_cm"])
        bed_volume_cm3 = area_cm2 * params["bed_height_cm"]
        mass_g = params["bulk_density_g_cm3"] * bed_volume_cm3
        c_break_mg_l = params["C_breakthrough_ratio"] * params["C0_mg_L"]
        t_end = max(float(params["t_end_min"]), time_step_min)
        times = np.arange(0.0, t_end + 0.5 * time_step_min, time_step_min)

        column = ColumnParams(
            length_cm=float(params["bed_height_cm"]),
            area_cm2=float(area_cm2),
            mass_g=float(mass_g),
            epsilon=float(params["epsilon"]),
            flow_ml_min=float(params["flow_ml_min"]),
            c0_mg_l=float(params["C0_mg_L"]),
        )
        sorption = SorptionParams(
            KH_L_g=float(params["KH_L_g"]),
            kLDF_1_min=float(params["kLDF_1_min"]),
            Dax_cm2_min=float(params["Dax_cm2_min"]),
        )
        concentrations = simulate_breakthrough(times, column, sorption, n_cells=DEFAULT_N_CELLS)
        ratios = concentrations / params["C0_mg_L"]
        breakthrough_time = first_breakthrough_time(
            times,
            ratios,
            float(params["C_breakthrough_ratio"]),
        )
        breakthrough_volume = (
            None if breakthrough_time is None else params["flow_ml_min"] * breakthrough_time
        )

        for time_min, concentration, ratio in zip(times, concentrations, ratios):
            rows.append(
                {
                    "case_id": case_index,
                    "time_min": float(time_min),
                    "C0_mg_L": float(params["C0_mg_L"]),
                    "C_breakthrough_ratio": float(params["C_breakthrough_ratio"]),
                    "C_breakthrough_mg_L": float(c_break_mg_l),
                    "temperature_C": float(params["temperature_C"]),
                    "bed_height_cm": float(params["bed_height_cm"]),
                    "column_diameter_cm": float(params["column_diameter_cm"]),
                    "column_area_cm2": float(area_cm2),
                    "bed_volume_cm3": float(bed_volume_cm3),
                    "bulk_density_g_cm3": float(params["bulk_density_g_cm3"]),
                    "bulk_density_source": density_source,
                    "carbon_mass_g": float(mass_g),
                    "epsilon": float(params["epsilon"]),
                    "flow_ml_min": float(params["flow_ml_min"]),
                    "K_H_L_g": float(params["KH_L_g"]),
                    "kLDF_1_min": float(params["kLDF_1_min"]),
                    "Dax_cm2_min": float(params["Dax_cm2_min"]),
                    "breakthrough_time_min": breakthrough_time,
                    "breakthrough_volume_ml": breakthrough_volume,
                    "C_out_mg_L": float(concentration),
                    "C_out_C0": float(ratio),
                    "adsorbent_profile_key": profile_key,
                    "adsorbent_profile_name": profile.get("name"),
                    "adsorbent_type": profile.get("type"),
                    "manufacturer": profile.get("manufacturer"),
                    "country": profile.get("country"),
                    "bulk_density_profile_min_g_cm3": profile.get("passport_bulk_density_min"),
                    "bulk_density_profile_max_g_cm3": profile.get("passport_bulk_density_max"),
                    "passport_bulk_density_min_g_cm3": profile.get("passport_bulk_density_min"),
                    "passport_bulk_density_max_g_cm3": profile.get("passport_bulk_density_max"),
                    "experimental_bed_density_min_g_cm3": profile.get("experimental_bed_density_min"),
                    "experimental_bed_density_max_g_cm3": profile.get("experimental_bed_density_max"),
                }
            )

    return pd.DataFrame(rows)


def generate_industrial_ml_dataset(
    ranges: dict[str, tuple[float, float]],
    number_of_cases: int,
    time_step_h: float,
    random_seed: int,
    method: str,
    scaleup_mode: str,
    kLDF_mode: str,
) -> pd.DataFrame:
    dimension_names = [
        "Q_m3_h",
        "C0_mg_L",
        "C_breakthrough_ratio",
        "T_C",
        "D_col_m",
        "H_m",
        "dp_mm",
        "epsilon",
        "bulk_density_kg_m3",
        "particle_sphericity",
        "KH_ref_L_g",
        "kLDF_ref_1_min",
        "Dax_ref_cm2_min",
        "T_ref_C",
        "deltaH_ads_kJ_mol",
        "Ea_kLDF_kJ_mol",
        "D_m_ref_m2_s",
        "t_end_h",
    ]
    samples = scale_samples(
        unit_samples(method, number_of_cases, len(dimension_names), random_seed),
        [ranges[name] for name in dimension_names],
    )

    rows: list[dict] = []
    for case_index, values in enumerate(samples, start=1):
        params = dict(zip(dimension_names, values))
        c_break_mg_l = params["C_breakthrough_ratio"] * params["C0_mg_L"]
        validation = validate_industrial_inputs(
            params["Q_m3_h"],
            params["C0_mg_L"],
            c_break_mg_l,
            params["T_C"],
            params["D_col_m"],
            params["H_m"],
            params["dp_mm"],
            params["epsilon"],
            params["bulk_density_kg_m3"],
            params["particle_sphericity"],
        )
        if validation["errors"]:
            continue

        diagnostics = industrial_adsorber_diagnostics(
            Q_m3_h=params["Q_m3_h"],
            D_col_m=params["D_col_m"],
            H_m=params["H_m"],
            dp_mm=params["dp_mm"],
            epsilon=params["epsilon"],
            bulk_density_kg_m3=params["bulk_density_kg_m3"],
            T_C=params["T_C"],
            D_m_ref_m2_s=params["D_m_ref_m2_s"],
            T_ref_C=params["T_ref_C"],
            KH_ref_L_g=params["KH_ref_L_g"],
            kLDF_ref_1_min=params["kLDF_ref_1_min"],
            Dax_ref_cm2_min=params["Dax_ref_cm2_min"],
            deltaH_ads_kJ_mol=params["deltaH_ads_kJ_mol"],
            Ea_kLDF_kJ_mol=params["Ea_kLDF_kJ_mol"],
            Dax_mode="инженерная оценка через скорость и размер частиц",
            kLDF_mode=kLDF_mode,
            particle_sphericity=params["particle_sphericity"],
            C0_mg_L=params["C0_mg_L"],
        )
        column = industrial_to_column_params(
            params["Q_m3_h"],
            params["D_col_m"],
            params["H_m"],
            params["bulk_density_kg_m3"],
            params["epsilon"],
            params["C0_mg_L"],
        )
        sorption = SorptionParams(
            KH_L_g=float(diagnostics["KH_T_L_g"]),
            kLDF_1_min=float(diagnostics["kLDF_used_1_min"]),
            Dax_cm2_min=float(diagnostics["Dax_T_cm2_min"]),
        )
        t_end_min = max(float(params["t_end_h"]) * 60.0, time_step_h * 60.0)
        time_step_min = max(float(time_step_h) * 60.0, 1e-6)
        times_min = np.arange(0.0, t_end_min + 0.5 * time_step_min, time_step_min)
        concentrations = simulate_breakthrough(times_min, column, sorption, n_cells=DEFAULT_N_CELLS)
        ratios = concentrations / params["C0_mg_L"]
        breakthrough = first_breakthrough_time(
            times_min,
            ratios,
            float(params["C_breakthrough_ratio"]),
        )
        breakthrough_volume_ml = None if breakthrough is None else column.flow_ml_min * breakthrough

        for time_min, concentration, ratio in zip(times_min, concentrations, ratios):
            rows.append(
                {
                    "case_id": case_index,
                    "time_min": float(time_min),
                    "time_h": float(time_min / 60.0),
                    "C0_mg_L": float(params["C0_mg_L"]),
                    "C_breakthrough_ratio": float(params["C_breakthrough_ratio"]),
                    "C_breakthrough_mg_L": float(c_break_mg_l),
                    "Q_m3_h": float(params["Q_m3_h"]),
                    "T_C": float(params["T_C"]),
                    "D_col_m": float(params["D_col_m"]),
                    "H_m": float(params["H_m"]),
                    "dp_mm": float(params["dp_mm"]),
                    "epsilon": float(params["epsilon"]),
                    "bulk_density_kg_m3": float(params["bulk_density_kg_m3"]),
                    "particle_sphericity": float(params["particle_sphericity"]),
                    "EBCT_min": diagnostics["EBCT_min"],
                    "superficial_velocity_m_h": diagnostics["superficial_velocity_m_h"],
                    "interstitial_velocity_m_h": diagnostics["interstitial_velocity_m_h"],
                    "Re_p": diagnostics["Re_p"],
                    "Sc": diagnostics["Sc"],
                    "Sh": diagnostics["Sh"],
                    "k_f_m_s": diagnostics["k_f_m_s"],
                    "pressure_drop_bar": diagnostics["pressure_drop_bar"],
                    "bed_volume_m3": diagnostics["bed_volume_m3"],
                    "carbon_mass_kg": diagnostics["carbon_mass_kg"],
                    "KH_T_L_g": diagnostics["KH_T_L_g"],
                    "kLDF_T_1_min": diagnostics["kLDF_T_1_min"],
                    "kLDF_mode": diagnostics["kLDF_mode"],
                    "kLDF_ref_1_min": diagnostics["kLDF_ref_1_min"],
                    "kLDF_Arrhenius_1_min": diagnostics["kLDF_Arrhenius_1_min"],
                    "kLDF_film_1_min": diagnostics["kLDF_film_1_min"],
                    "kLDF_used_1_min": diagnostics["kLDF_used_1_min"],
                    "Dax_T_cm2_min": diagnostics["Dax_T_cm2_min"],
                    "scaleup_mode": scaleup_mode,
                    "extrapolation_warning": "industrial scale-up is an engineering extrapolation",
                    "breakthrough_time_min": breakthrough,
                    "breakthrough_volume_ml": breakthrough_volume_ml,
                    "C_out_mg_L": float(concentration),
                    "C_out_C0": float(ratio),
                }
            )
    return pd.DataFrame(rows)


def experiment_label(experiment_id: str) -> str:
    rows = experiments[experiment_id]
    label_lang = st.session_state.get("language", "uk")
    return experiment_description(rows, label_lang, record=list(experiments).index(experiment_id) + 1)


SCALE_LABELS = ["Лабораторный", "Инженерный", "Промышленный"]
SCALE_DESCRIPTIONS = {
    "Лабораторный": "параметры близки к реальным лабораторным опытам",
    "Инженерный": "умеренная экстраполяция диапазонов",
    "Промышленный": "сценарный расчёт больших адсорберов",
}
SCALE_TRANSLATION_KEYS = {
    "Лабораторный": "scale_laboratory",
    "Инженерный": "scale_engineering",
    "Промышленный": "scale_industrial",
}
SCALE_INFO_KEYS = {
    "Лабораторный": "scale_laboratory_info",
    "Инженерный": "scale_engineering_info",
    "Промышленный": "scale_industrial_info",
}
SCALE_TO_PRESET = {
    "Лабораторный": "лабораторный валидированный",
    "Инженерный": "расширенный инженерный",
    "Промышленный": "промышленный scale-up",
}
SCALE_TO_DATASET = {
    "Лабораторный": "laboratory_validated_dataset",
    "Инженерный": "engineering_extended_dataset",
    "Промышленный": "industrial_scaleup_dataset",
}
CHEMVIRON_START_PARAMS = {
    "KH_L_g": 0.048,
    "kLDF_1_min": 0.014,
    "Dax_cm2_min": 0.00026,
    "T_ref_C": 20.0,
}
AUTO_TIME_LIMITS_MIN = {
    "Лабораторный": (300.0, 1000.0),
    "Инженерный": (48.0 * 60.0, 7.0 * 24.0 * 60.0),
    "Промышленный": (30.0 * 24.0 * 60.0, 180.0 * 24.0 * 60.0),
}
MIN_TARGET_DISPLAY_TIME_MIN = {
    SCALE_LABELS[0]: 2.0,
    SCALE_LABELS[1]: 30.0,
    SCALE_LABELS[2]: 60.0,
}
CURVE_DISPLAY_OPTIONS = [
    "until_breakthrough_plus_20",
    "until_c_c0_05",
    "until_c_c0_08",
    "until_c_c0_095",
    "manual_time",
]
CURVE_TARGET_RATIOS = {
    "until_c_c0_05": 0.50,
    "until_c_c0_08": 0.80,
    "until_c_c0_095": 0.95,
}
WORK_MODE_KEYS = {
    "Одна симуляция": "single_simulation",
    "Генерация серии экспериментов для ML": "ml_generation",
}
PARAMETER_SOURCE_KEYS = {
    "Авто: лучший доступный источник": "auto_best_available",
    "Последняя калибровка": "source_last_calibration",
    "Лучший эксперимент": "source_best_experiment",
    "Профиль выбранного угля": "source_adsorbent_profile",
    "Литературный типовой профиль": "source_literature_generic",
    "Ручной ввод": "manual_input",
}
PARAMETER_SOURCE_MODE_KEYS = {
    "Авто: лучший доступный источник": "auto_best_available",
    "Последняя калибровка": "last_calibration",
    "Лучший эксперимент": "best_experiment",
    "Профиль выбранного угля": "adsorbent_profile",
    "Литературный типовой профиль": "literature_generic",
    "Ручной ввод": "manual",
}
PRESSURE_MODE_KEYS = {
    "Атмосферное": "pressure_atmospheric",
    "Задать давление на входе": "pressure_inlet",
    "Задать перепад давления": "pressure_delta",
}


ENGINEERING_DIAGNOSTIC_ROWS = [
    ("Число Рейнольдса", "Re_p", "Re_p", "–", "Характеризует гидродинамический режим обтекания частиц."),
    ("Число Шмидта", "Sc", "Sc", "–", "Отношение вязкостного переноса к молекулярной диффузии."),
    ("Число Шервуда", "Sh", "Sh", "–", "Характеризует интенсивность внешнего массообмена."),
    ("Число Пекле по частице", "Pe_p", "Pe_p", "–", "Сравнивает конвективный перенос с диффузионным на масштабе частицы."),
    ("Число Пекле по слою (поверхностная скорость)", "Pe_L_superficial", "Pe_L_superficial", "–", "Сравнивает конвективный перенос с осевой дисперсией по длине слоя."),
    ("Число Пекле по слою (межпоровая скорость)", "Pe_L_interstitial", "Pe_L_interstitial", "–", "Аналогично Pe_L, но с использованием межпоровой скорости."),
    ("Коэффициент внешнего массообмена", "k_f", "k_f_m_s", "м/с", "Эффективность переноса красителя из жидкости к поверхности сорбента."),
    ("Давление на входе", "P_in", "P_in_bar", "бар", "Давление, заданное или рассчитанное на входе адсорбера."),
    ("Давление на выходе", "P_out", "P_out_bar", "бар", "Давление на выходе адсорбера; по умолчанию атмосферное."),
    ("Потеря давления", "ΔP", "pressure_drop_bar", "бар", "Перепад давления на адсорбере по уравнению Эргуна."),
    ("Доступный перепад давления", "ΔP_available", "available_deltaP_bar", "бар", "Перепад давления, заданный пользователем для проверки реализуемости расхода."),
    ("Проверка давления", "ΔP_required ≤ ΔP_available", "pressure_feasible", "–", "Показывает, достаточно ли заданного перепада давления для текущего расхода."),
    ("Источник свойств воды", "–", "water_properties_source", "–", "Показывает, использовался ли CoolProp или fallback-аппроксимация."),
    ("Константа Генри при T", "K_H(T)", "KH_T_L_g", "л/г", "Линейный коэффициент равновесной сорбции при данной температуре."),
    ("k_LDF (калиброванное)", "kLDF_ref", "kLDF_ref_1_min", "1/мин", "Эффективный кинетический коэффициент, полученный из калибровки."),
    ("k_LDF (Arrhenius)", "kLDF_Arrhenius", "kLDF_Arrhenius_1_min", "1/мин", "Температурно скорректированное значение k_LDF."),
    ("k_LDF по внешнему массообмену", "kLDF_film", "kLDF_film_1_min", "1/мин", "Приближённая оценка через внешний массообмен. Только диагностическая оценка."),
    ("Использованное k_LDF", "kLDF_used", "kLDF_used_1_min", "1/мин", "Значение, которое реально использовано в симуляции."),
    ("Режим выбора k_LDF", "kLDF_mode", "kLDF_mode", "–", "Показывает, откуда взят коэффициент: калибровка, Arrhenius, film или комбинированно."),
    ("Осевая дисперсия при T", "D_ax(T)", "Dax_T_cm2_min", "см²/мин", "Эффективный коэффициент осевой дисперсии в слое."),
]


def preset_for_scale(scale_mode: str) -> dict:
    try:
        return INDUSTRIAL_SCALE_PRESETS[SCALE_TO_PRESET[scale_mode]]
    except KeyError:
        preset_keys = list(INDUSTRIAL_SCALE_PRESETS.keys())
        scale_index = SCALE_LABELS.index(scale_mode) if scale_mode in SCALE_LABELS else 0
        return INDUSTRIAL_SCALE_PRESETS[preset_keys[min(scale_index, len(preset_keys) - 1)]]


def dataset_ranges_for_scale(scale_mode: str) -> dict:
    ranges = dict(RECOMMENDED_DATASET_RANGES[SCALE_TO_DATASET[scale_mode]])
    if "C0_mg_L" in ranges:
        ranges["C0_mg_L"] = (0.1, 10.0)
    return ranges


def best_calibrated_parameters(lang: str = "ru") -> tuple[dict[str, float], str]:
    path = "fitted_experiment_parameters.csv"
    if not os.path.exists(path):
        return CHEMVIRON_START_PARAMS.copy(), tr(lang, "best_params_file_missing")
    try:
        df = safe_read_csv(path)
        if df.empty:
            return CHEMVIRON_START_PARAMS.copy(), tr(lang, "best_params_table_empty")
        candidates = df.copy()
        if "calibration_mode" in candidates.columns:
            clean = candidates[candidates["calibration_mode"].astype(str) == "clean_bed"]
            if not clean.empty:
                candidates = clean
        if "breakthrough_status_exp" in candidates.columns:
            candidates["_bt_preferred"] = (
                candidates["breakthrough_status_exp"].astype(str) == "reached_by_interpolation"
            ).astype(int)
        else:
            candidates["_bt_preferred"] = 0
        rmse_col = "RMSE_C0" if "RMSE_C0" in candidates.columns else "RMSE/C0"
        r2_col = "R2" if "R2" in candidates.columns else "R_squared"
        candidates[rmse_col] = pd.to_numeric(candidates[rmse_col], errors="coerce").fillna(np.inf)
        candidates[r2_col] = pd.to_numeric(candidates[r2_col], errors="coerce").fillna(-np.inf)
        candidates = candidates.sort_values(["_bt_preferred", rmse_col, r2_col], ascending=[False, True, False])
        row = candidates.iloc[0]
        params = {
            "KH_L_g": float(row["KH_L_g"]),
            "kLDF_1_min": float(row["kLDF_1_min"]),
            "Dax_cm2_min": float(row["Dax_cm2_min"]),
        }
        exp_id = row.get("experiment_id", tr(lang, "best_experiment_fallback_name"))
        params["experiment_id"] = exp_id
        params["T_ref_C"] = temperature_reference_from_params(
            {
                "T_ref_C": row.get("T_ref_C"),
                "T_exp_C": row.get("T_exp_C"),
                "temperature_C": row.get("temperature_C"),
            }
        )
        return params, tr(lang, "best_params_used").format(experiment_id=exp_id)
    except Exception as exc:
        return CHEMVIRON_START_PARAMS.copy(), tr(lang, "best_params_failed").format(error=exc)


def resolve_auto_sorption_parameters(
    source: str,
    profile_key: str,
    pollutant: str,
    lang: str = "ru",
) -> tuple[dict[str, float], str, str]:
    mode = PARAMETER_SOURCE_MODE_KEYS.get(source, "auto_best_available")
    fitted_df = fitted_temperature_dataframe()
    params = resolve_sorption_parameters(
        profile_key,
        pollutant,
        mode,
        fitted_df=fitted_df,
        last_calibration=st.session_state.get("last_calibration"),
    )
    source_key = str(params.get("parameter_source", "generic_fallback"))
    level = "success" if source_key in {"last_calibration", "best_experiment"} else "warning"
    if source_key == "last_calibration":
        message = tr(lang, "latest_calibration_used").format(
            experiment_id=params.get("experiment_id", tr(lang, "current_experiment"))
        )
    elif source_key == "best_experiment":
        message = tr(lang, "best_experiment_params_used").format(
            experiment_id=params.get("experiment_id", tr(lang, "current_experiment"))
        )
    elif source_key == "chemviron_profile_start":
        message = tr(lang, "adsorbent_profile_params_used")
    elif source_key == "literature_generic":
        message = tr(lang, "literature_profile_params_used")
    elif source_key == "manual":
        message = tr(lang, "manual_input")
        level = "info"
    else:
        message = tr(lang, "generic_fallback_params_used")
    params["T_ref_C"] = temperature_reference_from_params(params, CHEMVIRON_START_PARAMS["T_ref_C"])
    return params, message, level


def apply_simulator_preset(scale_mode: str) -> None:
    preset = preset_for_scale(scale_mode)
    prefix = "sim"
    if scale_mode == SCALE_LABELS[0]:
        st.session_state[f"{prefix}_flow_lab"] = preset["Q_m3_h"] * 1_000_000.0 / 60.0
        st.session_state[f"{prefix}_height_lab"] = preset["H_m"] * 100.0
        st.session_state[f"{prefix}_diameter_lab"] = preset["D_col_m"] * 100.0
        st.session_state[f"{prefix}_bulk_density_lab"] = preset["bulk_density_kg_m3"] / 1000.0
    else:
        st.session_state[f"{prefix}_flow_si"] = preset["Q_m3_h"]
        st.session_state[f"{prefix}_height_si"] = preset["H_m"]
        st.session_state[f"{prefix}_diameter_si"] = preset["D_col_m"]
        st.session_state[f"{prefix}_bulk_density_si"] = preset["bulk_density_kg_m3"]
    st.session_state[f"{prefix}_C0"] = preset["C0_mg_L"]
    st.session_state[f"{prefix}_C_break"] = preset["C_breakthrough_mg_L"]
    st.session_state[f"{prefix}_T_C"] = preset["T_C"]
    st.session_state[f"{prefix}_epsilon"] = preset["epsilon"]
    st.session_state[f"{prefix}_dp_mm"] = preset["dp_mm"]
    st.session_state[f"{prefix}_particle_sphericity"] = preset["particle_sphericity"]
    st.session_state[f"{prefix}_T_ref_C"] = preset["T_ref_C"]
    st.session_state[f"{prefix}_deltaH"] = preset["deltaH_ads_kJ_mol"]
    st.session_state[f"{prefix}_Ea"] = preset["Ea_kLDF_kJ_mol"]
    st.session_state[f"{prefix}_D_m_ref"] = preset["D_m_ref_m2_s"]
    st.session_state[f"{prefix}_n_points"] = int(preset["n_points"])
    st.session_state[f"{prefix}_n_cells"] = int(preset["n_cells"])
    st.session_state[f"{prefix}_manual_time_h"] = preset["t_end_h"]
    st.session_state[f"{prefix}_KH"] = preset["KH_ref_L_g"]
    st.session_state[f"{prefix}_kLDF"] = preset["kLDF_ref_1_min"]
    st.session_state[f"{prefix}_Dax"] = preset["Dax_ref_cm2_min"]


def build_column_and_diagnostics(
    scale_mode: str,
    C0_mg_L: float,
    flow_value: float,
    height_value: float,
    diameter_value: float,
    epsilon: float,
    bulk_density_value: float,
    temperature_C: float,
    dp_mm: float,
    particle_sphericity: float,
    KH_ref: float,
    kLDF_ref: float,
    Dax_ref: float,
    T_ref_C: float,
    deltaH_ads: float,
    Ea_kLDF: float,
    D_m_ref: float,
    Dax_mode: str,
    kLDF_mode: str,
    pressure_mode: str = "Атмосферное",
    P_in_bar: float = 1.01325,
    P_out_bar: float = 1.01325,
    deltaP_available_bar: float | None = None,
    apply_pressure_KH_correction: bool = False,
    deltaV_ads_cm3_mol: float = 0.0,
    temperature_adaptation: dict | None = None,
) -> tuple[ColumnParams, SorptionParams, dict, list[str]]:
    KH_used = float(temperature_adaptation.get("KH_used_L_g", KH_ref)) if temperature_adaptation else float(KH_ref)
    kLDF_used = float(temperature_adaptation.get("kLDF_used_1_min", kLDF_ref)) if temperature_adaptation else float(kLDF_ref)
    Dax_used = float(temperature_adaptation.get("Dax_used_cm2_min", Dax_ref)) if temperature_adaptation else float(Dax_ref)
    if scale_mode == SCALE_LABELS[0]:
        area_cm2 = column_area_from_diameter(diameter_value)
        bed_volume_cm3 = area_cm2 * height_value
        mass_g = bulk_density_value * bed_volume_cm3
        column = ColumnParams(
            length_cm=float(height_value),
            area_cm2=float(area_cm2),
            mass_g=float(mass_g),
            epsilon=float(epsilon),
            flow_ml_min=float(flow_value),
            c0_mg_l=float(C0_mg_L),
        )
        sorption = SorptionParams(KH_used, kLDF_used, Dax_used)
        ebct_min = bed_volume_cm3 / max(float(flow_value), 1e-30)
        superficial_cm_min = float(flow_value) / max(area_cm2, 1e-30)
        interstitial_cm_min = superficial_cm_min / max(float(epsilon), 1e-30)
        if Dax_used > 0:
            pe_l_superficial = superficial_cm_min * float(height_value) / Dax_used
            pe_l_interstitial = interstitial_cm_min * float(height_value) / Dax_used
        else:
            pe_l_superficial = float("inf")
            pe_l_interstitial = float("inf")
        warnings = []
        extrapolation = bool(float(C0_mg_L) > 10.0)
        if extrapolation:
            warnings.append("C0 > 10 мг/л: линейная изотерма Генри может быть недостаточной; расчёт помечен как extrapolation.")
        diagnostics = {
            "bed_volume_cm3": bed_volume_cm3,
            "bed_volume_m3": bed_volume_cm3 / 1_000_000.0,
            "carbon_mass_g": mass_g,
            "carbon_mass_kg": mass_g / 1000.0,
            "bed_loading_density_g_cm3": bulk_density_value,
            "EBCT_min": ebct_min,
            "superficial_velocity_cm_min": superficial_cm_min,
            "superficial_velocity_m_h": superficial_cm_min * 0.6,
            "interstitial_velocity_cm_min": interstitial_cm_min,
            "Pe_L_superficial": pe_l_superficial,
            "Pe_L_interstitial": pe_l_interstitial,
            "KH_T_L_g": KH_used,
            "kLDF_used_1_min": kLDF_used,
            "Dax_T_cm2_min": Dax_used,
            "extrapolation": extrapolation,
            "pressure_mode": pressure_mode,
            "P_in_bar": float(P_in_bar),
            "P_out_bar": float(P_out_bar),
            "available_deltaP_bar": deltaP_available_bar,
            "required_deltaP_bar": None,
            "pressure_feasible": True,
            "pressure_warning": "",
            "warnings": warnings,
        }
        if temperature_adaptation:
            diagnostics.update(temperature_adaptation)
        return column, sorption, diagnostics, []

    Q_m3_h = float(flow_value)
    D_col_m = float(diameter_value)
    H_m = float(height_value)
    c_break_dummy = 0.05 * float(C0_mg_L)
    validation = validate_industrial_inputs(
        Q_m3_h,
        C0_mg_L,
        c_break_dummy,
        temperature_C,
        D_col_m,
        H_m,
        dp_mm,
        epsilon,
        bulk_density_value,
        particle_sphericity,
    )
    diagnostics = industrial_adsorber_diagnostics(
        Q_m3_h=Q_m3_h,
        D_col_m=D_col_m,
        H_m=H_m,
        dp_mm=dp_mm,
        epsilon=epsilon,
        bulk_density_kg_m3=bulk_density_value,
        T_C=temperature_C,
        D_m_ref_m2_s=D_m_ref,
        T_ref_C=T_ref_C,
        KH_ref_L_g=KH_ref,
        kLDF_ref_1_min=kLDF_ref,
        Dax_ref_cm2_min=Dax_ref,
        deltaH_ads_kJ_mol=deltaH_ads,
        Ea_kLDF_kJ_mol=Ea_kLDF,
        Dax_mode=Dax_mode,
        kLDF_mode=kLDF_mode,
        particle_sphericity=particle_sphericity,
        C0_mg_L=C0_mg_L,
        pressure_mode=pressure_mode,
        P_in_bar=P_in_bar,
        P_out_bar=P_out_bar,
        deltaP_available_bar=deltaP_available_bar,
        apply_pressure_KH_correction=apply_pressure_KH_correction,
        deltaV_ads_cm3_mol=deltaV_ads_cm3_mol,
    )
    if temperature_adaptation:
        diagnostics.update(temperature_adaptation)
        diagnostics["KH_T_L_g"] = KH_used
        diagnostics["kLDF_used_1_min"] = kLDF_used
        diagnostics["Dax_T_cm2_min"] = Dax_used
    diagnostics["extrapolation"] = bool(float(C0_mg_L) > 10.0)
    column = industrial_to_column_params(Q_m3_h, D_col_m, H_m, bulk_density_value, epsilon, C0_mg_L)
    sorption = SorptionParams(
        KH_L_g=KH_used,
        kLDF_1_min=kLDF_used,
        Dax_cm2_min=Dax_used,
    )
    return column, sorption, diagnostics, validation["errors"] + validation["warnings"]


def simulate_for_time(
    t_end_min: float,
    n_points: int,
    column: ColumnParams,
    sorption: SorptionParams,
    n_cells: int,
    breakthrough_ratio: float,
) -> dict:
    t_end_min = max(float(t_end_min), 1e-6)
    n_points = max(3, int(n_points))
    times = np.linspace(0.0, t_end_min, n_points)
    concentrations = simulate_breakthrough(times, column, sorption, n_cells=int(n_cells))
    ratios = concentrations / column.c0_mg_l
    status = breakthrough_status(times, ratios, breakthrough_ratio)
    return {
        "time_min": times,
        "C_out_mg_L": concentrations,
        "C_out_C0": ratios,
        "breakthrough": status,
        "breakthrough_found": status["status"] != "not_reached",
        "t_end_actual_min": float(t_end_min),
    }


def simulate_with_time_mode(
    curve_display_mode: str,
    scale_mode: str,
    manual_t_end_min: float,
    n_points: int,
    column: ColumnParams,
    sorption: SorptionParams,
    n_cells: int,
    breakthrough_ratio: float,
) -> dict:
    initial_t_end, max_t_end = AUTO_TIME_LIMITS_MIN[scale_mode]

    def annotate_result(
        result: dict,
        target_ratio: float | None,
        target_status: dict,
        target_reached: bool,
    ) -> dict:
        result["target_ratio"] = target_ratio
        result["target_time_min"] = target_status.get("time_min")
        result["target_reached"] = target_reached
        result["curve_display_mode"] = curve_display_mode
        result["breakthrough_time_min"] = result["breakthrough"].get("time_min")
        if curve_display_mode == "until_breakthrough_plus_20":
            result["curve_comment"] = "на графике показана ранняя часть кривой"
        elif curve_display_mode == "manual_time":
            result["curve_comment"] = "показано ручное время моделирования"
        else:
            result["curve_comment"] = "показана расширенная кривая до заданного C/C0"
        return result

    if curve_display_mode == "manual_time":
        result = simulate_for_time(manual_t_end_min, n_points, column, sorption, n_cells, breakthrough_ratio)
        result["auto_time_message"] = "Использовано ручное время расчёта."
        manual_target = breakthrough_ratio
        target_status = breakthrough_status(result["time_min"], result["C_out_C0"], manual_target)
        return annotate_result(
            result,
            manual_target,
            target_status,
            target_status["status"] != "not_reached",
        )

    if curve_display_mode == "until_breakthrough_plus_20":
        target_ratio = breakthrough_ratio
        multiplier = 1.2
    else:
        target_ratio = CURVE_TARGET_RATIOS[curve_display_mode]
        multiplier = 1.05

    t_end = initial_t_end
    last_result = None
    while t_end <= max_t_end + 1e-9:
        last_result = simulate_for_time(t_end, n_points, column, sorption, n_cells, breakthrough_ratio)
        target_status = breakthrough_status(last_result["time_min"], last_result["C_out_C0"], target_ratio)
        target_time = target_status.get("time_min")
        if target_time is not None:
            if curve_display_mode == "until_breakthrough_plus_20":
                final_t = min(max_t_end, multiplier * float(target_time))
            else:
                min_display = MIN_TARGET_DISPLAY_TIME_MIN.get(scale_mode, 2.0)
                final_t = min(max_t_end, max(multiplier * float(target_time), min_display))
            final_result = simulate_for_time(final_t, n_points, column, sorption, n_cells, breakthrough_ratio)
            final_target_status = breakthrough_status(final_result["time_min"], final_result["C_out_C0"], target_ratio)
            if curve_display_mode == "until_breakthrough_plus_20":
                final_result["auto_time_message"] = f"Авто-время: расчёт выполнен до 1.2 * t_b = {final_t:.2f} мин."
            else:
                final_result["auto_time_message"] = f"Расчёт выполнен до C/C0 = {target_ratio:.2f}: {final_t:.2f} мин."
            return annotate_result(final_result, target_ratio, final_target_status, True)
        if t_end >= max_t_end:
            break
        t_end = min(t_end * 2.0, max_t_end)

    assert last_result is not None
    target_status = breakthrough_status(last_result["time_min"], last_result["C_out_C0"], target_ratio)
    last_result["auto_time_message"] = "Целевой уровень C/C0 не достигнут за максимальное время расчёта."
    return annotate_result(last_result, target_ratio, target_status, False)


def plot_simulation_curve(
    time_min: np.ndarray,
    ratio: np.ndarray,
    breakthrough_ratio: float,
    breakthrough: dict,
    scale_mode: str,
    title: str,
    ebct_min: float | None = None,
    lang: str = "ru",
):
    max_time = float(np.nanmax(time_min)) if len(time_min) else 0.0
    if scale_mode == "Промышленный" and max_time >= 24.0 * 60.0:
        factor = 24.0 * 60.0
        x_label = tr(lang, "time_days")
        unit = x_label.split(",")[-1].strip()
    elif scale_mode != SCALE_LABELS[0] or max_time > 600.0:
        factor = 60.0
        x_label = tr(lang, "time_h")
        unit = x_label.split(",")[-1].strip()
    else:
        factor = 1.0
        x_label = tr(lang, "time_min")
        unit = x_label.split(",")[-1].strip()
    x = time_min / factor
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, ratio, "-", linewidth=2, label=tr(lang, "model_curve"))
    ax.axhline(
        breakthrough_ratio,
        color="tab:red",
        linestyle="--",
        linewidth=1.4,
        label=tr(lang, "breakthrough_boundary_label").format(value=f"{breakthrough_ratio:.3f}"),
    )
    tb = breakthrough.get("time_min")
    if tb is not None:
        tb_plot = float(tb) / factor
        label = f"{tr(lang, 'breakthrough_time')}: {breakthrough.get('label', '')}"
        if breakthrough.get("status") == "reached_by_interpolation":
            label = f"{tr(lang, 'breakthrough_time')} = {tb_plot:.2f} {unit}"
        elif breakthrough.get("status") == "already_above_at_first_sample":
            label = f"{tr(lang, 'breakthrough_time')} ≤ {tb_plot:.2f} {unit}"
        ax.axvline(tb_plot, color="tab:orange", linestyle="-.", linewidth=1.5, label=label)
    else:
        ax.text(0.02, 0.92, tr(lang, "breakthrough_not_reached"), transform=ax.transAxes, color="tab:orange", va="top")
    if ebct_min is not None and ebct_min > 0:
        ax.axvline(
            ebct_min / factor,
            color="tab:gray",
            linestyle=":",
            linewidth=1.1,
            label=tr(lang, "ebct_label").format(value=f"{ebct_min:.2f}"),
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("C/C0")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def format_diagnostic_value(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "да" if bool(value) else "нет"
    if isinstance(value, str):
        return fix_mojibake(value) or "—"
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(value_float):
        return str(value_float)
    abs_value = abs(value_float)
    if abs_value == 0:
        return "0"
    if abs_value < 1e-4 or abs_value >= 1e3:
        return f"{value_float:.3e}"
    if abs_value < 0.01:
        return f"{value_float:.3g}"
    if abs_value < 1:
        return f"{value_float:.4f}".rstrip("0").rstrip(".")
    return f"{value_float:.4g}"


ENGINEERING_ROW_LOCALIZATION = {
    "en": {
        "Re_p": ("Reynolds number", "Characterizes hydrodynamic flow around particles."),
        "Sc": ("Schmidt number", "Ratio of momentum diffusivity to molecular diffusivity."),
        "Sh": ("Sherwood number", "Characterizes external mass transfer intensity."),
        "Pe_p": ("Particle Peclet number", "Compares convection with diffusion at the particle scale."),
        "Pe_L_superficial": ("Bed Peclet number (superficial velocity)", "Compares convection with axial dispersion along bed length."),
        "Pe_L_interstitial": ("Bed Peclet number (interstitial velocity)", "Similar to Pe_L, but based on interstitial velocity."),
        "k_f_m_s": ("External mass-transfer coefficient", "Transfer efficiency from liquid to adsorbent surface."),
        "P_in_bar": ("Inlet pressure", "Pressure specified or calculated at the adsorber inlet."),
        "P_out_bar": ("Outlet pressure", "Adsorber outlet pressure; atmospheric by default."),
        "pressure_drop_bar": ("Pressure drop", "Pressure drop across the adsorber from the Ergun equation."),
        "available_deltaP_bar": ("Available pressure drop", "Pressure drop provided by the user for flow feasibility checking."),
        "pressure_feasible": ("Pressure feasibility", "Shows whether available pressure drop is sufficient for current flow."),
        "water_properties_source": ("Water properties source", "Shows whether CoolProp or fallback approximation was used."),
        "KH_T_L_g": ("Henry constant at T", "Linear equilibrium adsorption coefficient at this temperature."),
        "kLDF_ref_1_min": ("k_LDF (calibrated)", "Effective kinetic coefficient obtained from calibration."),
        "kLDF_Arrhenius_1_min": ("k_LDF (Arrhenius)", "Temperature-corrected k_LDF value."),
        "kLDF_film_1_min": ("k_LDF from film transfer", "Approximate estimate from external mass transfer; diagnostic only."),
        "kLDF_used_1_min": ("Used k_LDF", "The value actually used in the simulation."),
        "kLDF_mode": ("k_LDF selection mode", "Shows whether the coefficient came from calibration, Arrhenius, film estimate or combined mode."),
        "Dax_T_cm2_min": ("Axial dispersion at T", "Effective axial dispersion coefficient in the bed."),
    },
    "uk": {
        "Re_p": ("Число Рейнольдса", "Характеризує гідродинамічний режим обтікання частинок."),
        "Sc": ("Число Шмідта", "Відношення в’язкісного переносу до молекулярної дифузії."),
        "Sh": ("Число Шервуда", "Характеризує інтенсивність зовнішнього масообміну."),
        "Pe_p": ("Число Пекле за частинкою", "Порівнює конвективний перенос із дифузійним на масштабі частинки."),
        "Pe_L_superficial": ("Число Пекле за шаром (поверхнева швидкість)", "Порівнює конвективний перенос з осьовою дисперсією за довжиною шару."),
        "Pe_L_interstitial": ("Число Пекле за шаром (міжпорова швидкість)", "Аналогічно Pe_L, але з використанням міжпорової швидкості."),
        "k_f_m_s": ("Коефіцієнт зовнішнього масообміну", "Ефективність переносу барвника з рідини до поверхні сорбенту."),
        "P_in_bar": ("Тиск на вході", "Тиск, заданий або розрахований на вході адсорбера."),
        "P_out_bar": ("Тиск на виході", "Тиск на виході адсорбера; за замовчуванням атмосферний."),
        "pressure_drop_bar": ("Втрата тиску", "Перепад тиску на адсорбері за рівнянням Ергуна."),
        "available_deltaP_bar": ("Доступний перепад тиску", "Перепад тиску, заданий користувачем для перевірки реалізовності витрати."),
        "pressure_feasible": ("Перевірка тиску", "Показує, чи достатній заданий перепад тиску для поточної витрати."),
        "water_properties_source": ("Джерело властивостей води", "Показує, чи використано CoolProp або fallback-апроксимацію."),
        "KH_T_L_g": ("Константа Генрі при T", "Лінійний коефіцієнт рівноважної сорбції за цієї температури."),
        "kLDF_ref_1_min": ("k_LDF (каліброване)", "Ефективний кінетичний коефіцієнт, отриманий з калібрування."),
        "kLDF_Arrhenius_1_min": ("k_LDF (Arrhenius)", "Температурно скориговане значення k_LDF."),
        "kLDF_film_1_min": ("k_LDF за зовнішнім масообміном", "Наближена оцінка через зовнішній масообмін; лише діагностична."),
        "kLDF_used_1_min": ("Використане k_LDF", "Значення, яке реально використано в симуляції."),
        "kLDF_mode": ("Режим вибору k_LDF", "Показує, звідки взято коефіцієнт: калібрування, Arrhenius, film або комбіновано."),
        "Dax_T_cm2_min": ("Осьова дисперсія при T", "Ефективний коефіцієнт осьової дисперсії в шарі."),
    },
}


def localized_engineering_row(lang: str, key: str, default_title: str, default_explanation: str) -> tuple[str, str]:
    return ENGINEERING_ROW_LOCALIZATION.get(lang, {}).get(key, (default_title, default_explanation))


def interpret_engineering_indicator(key: str, value, diagnostics: dict, lang: str = "ru") -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        value_float = None
    if key == "Re_p" and value_float is not None:
        if lang == "en":
            if value_float < 1:
                return "very calm flow"
            if value_float <= 10:
                return "low Reynolds number; transition to moderate particle flow"
            return "more intensive particle flow"
        if lang == "uk":
            if value_float < 1:
                return "дуже спокійний потік"
            if value_float <= 10:
                return "низький Reynolds, перехід до помірного обтікання"
            return "інтенсивніше обтікання частинок"
        if value_float < 1:
            return "очень спокойный поток"
        if value_float <= 10:
            return "низкий Reynolds, переход к умеренному обтеканию"
        return "интенсивнее обтекание частиц"
    if key == "Sc" and value_float is not None:
        if lang == "en":
            return "Sc >> 1: diffusion is much slower than momentum transfer" if value_float > 100 else "moderate ratio of viscous and diffusive transfer"
        if lang == "uk":
            return "Sc >> 1: дифузія помітно повільніша за перенос імпульсу" if value_float > 100 else "помірне співвідношення в’язкісного й дифузійного переносу"
        return "Sc >> 1: диффузия заметно медленнее переноса импульса" if value_float > 100 else "умеренное соотношение вязкостного и диффузионного переноса"
    if key == "Sh" and value_float is not None:
        if lang == "en":
            return "high Sh: intensive external mass transfer" if value_float > 20 else "moderate external mass transfer intensity"
        if lang == "uk":
            return "високий Sh: інтенсивний зовнішній масообмін" if value_float > 20 else "помірна інтенсивність зовнішнього масообміну"
        return "высокий Sh: интенсивный внешний массообмен" if value_float > 20 else "умеренная интенсивность внешнего массообмена"
    if key in {"Pe_p", "Pe_L_superficial", "Pe_L_interstitial"} and value_float is not None:
        if lang == "en":
            if value_float > 100:
                return "very high Pe: convection dominates"
            if value_float < 1:
                return "low Pe: diffusion or dispersion is important"
            return "convection and dispersion/diffusion are comparable"
        if lang == "uk":
            if value_float > 100:
                return "Pe дуже велике: конвективний перенос домінує"
            if value_float < 1:
                return "низьке Pe: помітна роль дифузії або дисперсії"
            return "конвекція та дисперсія/дифузія співмірні за роллю"
        if value_float > 100:
            return "Pe очень большой: конвективный перенос доминирует"
        if value_float < 1:
            return "низкий Pe: заметна роль диффузии или дисперсии"
        return "конвекция и дисперсия/диффузия сопоставимы по роли"
    if key == "pressure_drop_bar" and value_float is not None:
        if lang == "en":
            if value_float < 0.1:
                return "low pressure drop"
            if value_float > 2:
                return "high pressure drop"
            return "moderate pressure drop"
        if lang == "uk":
            if value_float < 0.1:
                return "мала втрата тиску"
            if value_float > 2:
                return "висока втрата тиску"
            return "помірна втрата тиску"
        if value_float < 0.1:
            return "малая потеря давления"
        if value_float > 2:
            return "высокая потеря давления"
        return "умеренная потеря давления"
    if key == "kLDF_film_1_min" and value_float is not None:
        ref = diagnostics.get("kLDF_ref_1_min")
        try:
            ref_float = float(ref)
            if ref_float > 0 and value_float / ref_float > 10:
                if lang == "en":
                    return "kLDF_film >> kLDF_ref: external transfer is not the main limiting step"
                if lang == "uk":
                    return "kLDF_film >> kLDF_ref: зовнішній масообмін не є головною лімітуючою ланкою"
                return "kLDF_film >> kLDF_ref: внешний массообмен не является главным лимитирующим звеном"
        except (TypeError, ValueError):
            pass
        if lang == "en":
            return "diagnostic estimate of external mass transfer"
        if lang == "uk":
            return "діагностична оцінка зовнішнього масообміну"
        return "диагностическая оценка внешнего массообмена"
    if key == "pressure_feasible":
        if lang == "en":
            return "available pressure drop is sufficient" if bool(value) else "available pressure drop is insufficient for current Q"
        if lang == "uk":
            return "заданого перепаду тиску достатньо" if bool(value) else "заданого перепаду тиску недостатньо для поточного Q"
        return "заданного перепада давления достаточно" if bool(value) else "заданного перепада давления недостаточно для текущего Q"
    if key == "water_properties_source":
        if lang == "en":
            return "CoolProp accounts for T and P" if value == "CoolProp" else "fallback accounts for temperature, but not pressure"
        if lang == "uk":
            return "CoolProp враховує T і P" if value == "CoolProp" else "fallback враховує температуру, але не тиск"
        return "CoolProp учитывает T и P" if value == "CoolProp" else "fallback учитывает температуру, но не давление"
    if lang == "en":
        return "diagnostic parameter"
    if lang == "uk":
        return "діагностичний параметр"
    return "диагностический параметр"


def make_engineering_diagnostics_dataframe(diagnostics: dict, lang: str = "ru") -> pd.DataFrame:
    rows = []
    for title, symbol, key, unit, explanation in ENGINEERING_DIAGNOSTIC_ROWS:
        if key == "available_deltaP_bar" and diagnostics.get(key) is None:
            continue
        title, explanation = localized_engineering_row(lang, key, title, explanation)
        rows.append(
            {
                tr(lang, "col_indicator"): title,
                tr(lang, "col_symbol"): symbol,
                tr(lang, "col_value"): format_diagnostic_value(diagnostics.get(key)),
                tr(lang, "col_unit"): unit,
                tr(lang, "col_short_explanation"): explanation,
                tr(lang, "col_interpretation"): interpret_engineering_indicator(key, diagnostics.get(key), diagnostics, lang),
            }
        )
    return pd.DataFrame(rows)


def make_basic_diagnostics_dataframe(
    column: ColumnParams,
    diagnostics: dict,
    breakthrough: dict,
    result: dict,
    curve_display_label: str,
) -> pd.DataFrame:
    tb = breakthrough.get("time_min")
    bed_volume = diagnostics.get("bed_volume_m3", diagnostics.get("bed_volume_cm3"))
    carbon_mass = diagnostics.get("carbon_mass_kg", diagnostics.get("carbon_mass_g"))
    superficial_velocity = diagnostics.get(
        "superficial_velocity_m_h",
        diagnostics.get("superficial_velocity_cm_min"),
    )
    breakthrough_volume = None if tb is None else column.flow_ml_min * float(tb)
    row = {
        "bed_volume": bed_volume,
        "carbon_mass": carbon_mass,
        "EBCT_min": diagnostics.get("EBCT_min"),
        "superficial_velocity": superficial_velocity,
        "breakthrough_time_min": result.get("breakthrough_time_min"),
        "breakthrough_volume": breakthrough_volume,
        "breakthrough_found": result.get("breakthrough_found"),
        "t_end_actual_min": result.get("t_end_actual_min"),
    }
    if diagnostics.get("pressure_drop_bar") is not None:
        row["pressure_drop_bar"] = diagnostics.get("pressure_drop_bar")
    return pd.DataFrame(
        [row]
    )


def make_operating_regime_dataframe(regime: dict, lang: str = "ru") -> pd.DataFrame:
    preferred_keys = [
        "bed_volume_ml",
        "void_volume_ml",
        "bed_volume_m3",
        "void_volume_m3",
        "EBCT_min",
        "void_residence_time_min",
        "superficial_velocity_cm_min",
        "interstitial_velocity_cm_min",
        "superficial_velocity_m_h",
        "interstitial_velocity_m_h",
        "bed_volumes_per_min",
        "bed_volumes_per_hour",
        "estimated_carbon_mass_g",
        "estimated_carbon_mass_kg",
        "dye_loading_mg_min",
        "dye_loading_g_h",
        "extrapolation_level",
    ]
    df = pd.DataFrame(
        [
            {tr(lang, "col_indicator"): key, tr(lang, "col_value"): regime.get(key)}
            for key in preferred_keys
            if key in regime
        ]
    )
    if not df.empty:
        value_column = tr(lang, "col_value")
        df[value_column] = df[value_column].astype(str)
    return df



def append_curve_rows(
    rows: list[dict],
    case_id: int,
    result: dict,
    case_meta: dict,
) -> None:
    for time_min, concentration, ratio in zip(
        result["time_min"],
        result["C_out_mg_L"],
        result["C_out_C0"],
    ):
        rows.append(
            {
                "case_id": case_id,
                "time_min": float(time_min),
                "C_out_mg_L": float(concentration),
                "C_out_C0": float(ratio),
                **case_meta,
            }
        )


def render_mass_balance_block(
    time_min,
    column: ColumnParams,
    sorption: SorptionParams,
    n_cells: int,
    key_prefix: str,
    lang: str = "ru",
) -> tuple[pd.DataFrame, dict]:
    full = simulate_breakthrough_full(time_min, column, sorption, n_cells=int(n_cells))
    balance_df = calculate_mass_balance(
        full["time_min"],
        full["C_profile_mg_L"],
        full["q_profile_mg_g"],
        full["C_out_mg_L"],
        column,
        sorption,
    )
    summary = summarize_mass_balance(balance_df)
    st.markdown(f"#### {tr(lang, 'mass_balance_check')}")
    st.dataframe(pd.DataFrame([summary]), width="stretch", hide_index=True)
    if not summary.get("mass_balance_ok", False):
        st.warning(tr(lang, "mass_balance_error_high"))

    fig_mb, ax1 = plt.subplots(figsize=(9.5, 4.8))
    ax1.plot(balance_df["time_min"], balance_df["M_in_mg"], label="M_in")
    ax1.plot(balance_df["time_min"], balance_df["M_out_mg"], label="M_out")
    ax1.plot(balance_df["time_min"], balance_df["M_liquid_mg"], label="M_liquid")
    ax1.plot(balance_df["time_min"], balance_df["M_adsorbed_mg"], label="M_adsorbed")
    ax1.set_xlabel(tr(lang, "time_min"))
    ax1.set_ylabel(tr(lang, "mass_dye_mg"))
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(
        balance_df["time_min"],
        balance_df["mass_balance_error_percent"],
        color="tab:red",
        linestyle="--",
        label=tr(lang, "balance_error_percent"),
    )
    ax2.set_ylabel(tr(lang, "balance_error_percent"))
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig_mb.tight_layout()
    st.pyplot(fig_mb)
    st.download_button(
        tr(lang, "download_mass_balance_csv"),
        data=balance_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{key_prefix}_mass_balance.csv",
        mime="text/csv",
        key=f"{key_prefix}_download_mass_balance",
    )
    plt.close(fig_mb)
    return balance_df, summary



_startup_probe.install(globals())
_startup_probe.mark("definitions_complete")
experiments = load_experiment_data()

language_labels = list(LANGUAGES.values())
language_by_label = {label: key for key, label in LANGUAGES.items()}
current_language = st.session_state.get("language", "uk")
current_language_label = LANGUAGES.get(current_language, LANGUAGES["uk"])


def update_interface_language():
    st.session_state["language"] = language_by_label[st.session_state["language_selector"]]


selected_language_label = st.selectbox(
    tr(current_language, "language_label"),
    language_labels,
    index=language_labels.index(current_language_label),
    key="language_selector",
    on_change=update_interface_language,
)
lang = language_by_label[selected_language_label]
st.session_state["language"] = lang

if st.sidebar.button(tr(lang, "clear_interface_cache"), key="clear_streamlit_cache"):
    st.cache_data.clear()
    st.session_state.clear()
    st.rerun()

st.title(tr(lang, "app_title"))
st.caption(tr(lang, "app_subtitle"))
with st.expander(tr(lang, "ui_glossary")):
    st.markdown(tr(lang, "ui_glossary_body"))

tab_calibration, tab_simulator, tab_ml_module, tab_help = tracked_tabs(st,
    [
        tr(lang, "tab_calibration"),
        tr(lang, "tab_simulator"),
        tr(lang, "tab_ml_module"),
        tr(lang, "tab_help"),
    ], key="ui_main_tabs",
)
# Keep this data-dependent numeric widget only while its ML preview is absent.
# Avoid writing its Session State key in a run that instantiates the widget.
if not tab_ml_module.open and "ml_quality_required_h" in st.session_state:
    st.session_state["ml_quality_required_h"] = st.session_state["ml_quality_required_h"]


_startup_probe.mark("calibration_begin")
with tab_calibration:
    st.subheader(tr(lang, "calibration_title"))
    st.caption(tr(lang, "ui_calibration_workflow"))
    left, right = st.columns([0.46, 0.54])

    with left:
        experiment_ids = list(experiments.keys())
        default_experiment = "22.04.26_8.5cm_18.0g_1_7"
        default_index = experiment_ids.index(default_experiment) if default_experiment in experiment_ids else 0
        experiment_id = st.selectbox(
            tr(lang, "experiment"),
            experiment_ids,
            index=default_index,
            format_func=experiment_label,
            key="fit_experiment_id",
        )
        epsilon_fit = st.number_input(
            tr(lang, "porosity"),
            0.05,
            0.95,
            0.40,
            0.01,
            key="fit_epsilon",
            help=tr(lang, "ui_porosity_help"),
        )
        fit_start_mode = st.radio(
            tr(lang, "fit_start_mode"),
            ["automatic", "manual"],
            index=0,
            format_func=lambda value: tr(lang, f"fit_start_mode_{value}"),
            key="fit_start_mode",
        )
        if fit_start_mode == "manual":
            min_time_for_fit = st.number_input(
                tr(lang, "fit_start_time"),
                0.0,
                500.0,
                15.0,
                1.0,
                key="fit_min_time",
                help=tr(lang, "fit_start_manual_help"),
            )
        else:
            min_time_for_fit = None
            preview_rows = experiments[experiment_id]
            preview_column = make_column_from_experiment(preview_rows, epsilon=epsilon_fit)
            preview_breakthrough = interpret_breakthrough_curve(
                [row["time_min"] for row in preview_rows],
                [row["C_out_mg_L"] / preview_column.c0_mg_l for row in preview_rows],
                c0_mg_l=preview_column.c0_mg_l,
                breakthrough_ratio=DEFAULT_BREAKTHROUGH_RATIO,
                original_breakthrough_concentration_mg_l=experiment_breakthrough_concentration(preview_rows),
            )
            preview_fit_start = (
                preview_breakthrough.first_below_threshold_time_min
                if preview_breakthrough.startup_transient_detected
                else min(float(row["time_min"]) for row in preview_rows)
            )
            st.caption(
                tr(lang, "fit_start_automatic_value").format(
                    value=f"{float(preview_fit_start):.2f}"
                )
            )
        n_cells_fit = st.slider(
            tr(lang, "model_cells"),
            20,
            1280,
            DEFAULT_N_CELLS,
            10,
            key="fit_n_cells",
            help=tr(lang, "scientific_grid_policy"),
        )
        legacy_calibration_modes = {
            "Чистый слой: C_initial = 0": "clean_bed",
            "Диагностический режим: подбирать C_initial": "diagnostic_fit_C_initial",
        }
        if st.session_state.get("fit_calibration_mode") in legacy_calibration_modes:
            st.session_state["fit_calibration_mode"] = legacy_calibration_modes[st.session_state["fit_calibration_mode"]]
        calibration_mode = st.selectbox(
            tr(lang, "calibration_mode"),
            [
                "clean_bed",
                "diagnostic_fit_C_initial",
            ],
            index=0,
            key="fit_calibration_mode",
            format_func=lambda value: tr(lang, "ui_mode_diagnostic" if value == "diagnostic_fit_C_initial" else "ui_mode_clean"),
            help=tr(lang, "ui_calibration_mode_help"),
        )
        calibration_mode_label = tr(lang, "ui_mode_diagnostic" if calibration_mode == "diagnostic_fit_C_initial" else "ui_mode_clean")
        fit_initial_concentration = calibration_mode == "diagnostic_fit_C_initial"
        show_experiment_line = st.checkbox(
            tr(lang, "show_experimental_line"),
            value=True,
            key="fit_show_experiment_line",
            help=tr(lang, "ui_experimental_line_help"),
        )
        with st.expander(tr(lang, "advanced_parameters"), expanded=False):
            check_fit_mass_balance = st.checkbox(
                tr(lang, "check_fit_mass_balance"),
                value=False,
                key="fit_check_mass_balance",
            )
            run_bootstrap = st.checkbox(
                tr(lang, "bootstrap_uncertainty"),
                value=False,
                key="fit_run_bootstrap",
            )
            n_bootstrap = st.number_input(
                tr(lang, "ui_bootstrap_count"),
                min_value=5,
                max_value=200,
                value=30,
                step=5,
                key="fit_n_bootstrap",
                disabled=not run_bootstrap,
            )
        run_fit = st.button(tr(lang, "run_calibration"), type="primary", key="fit_run")

        with st.expander(tr(lang, "bed_loading_diagnostics"), expanded=False):
            try:
                summary = load_summary_data()
                st.dataframe(summary[summary["experiment_id"] == experiment_id], width="stretch")
            except Exception:
                st.info(tr(lang, "summary_unavailable"))

            st.markdown(f"#### {tr(lang, 'bed_loading_diagnostics')}")
            st.dataframe(
                experiment_bed_diagnostics(experiments[experiment_id]),
                width="stretch",
                hide_index=True,
            )

        with st.expander(tr(lang, "scientific_validation"), expanded=False):
            run_all_fits = st.button(
                tr(lang, "run_all_experiments"),
                key="fit_all_run",
                type="secondary",
            )
            sort_fit_table_by = st.selectbox(
                tr(lang, "sort_table_by"),
                ["RMSE/C0", "R²", "ошибка t прорыва"],
                key="fit_all_sort_by",
            )

            if run_all_fits:
                with st.spinner(tr(lang, "running_all_fits")):
                    st.session_state["fit_all_results"] = fit_all_experiments(
                        data_path=DATA_PATH,
                        output_csv="fitted_experiment_parameters.csv",
                        figures_dir=os.path.join("figures", "fitted_experiments"),
                        comparison_path=os.path.join("figures", "comparison_all_experiments.png"),
                        epsilon=epsilon_fit,
                        min_time_for_fit=min_time_for_fit,
                        fit_n_cells=n_cells_fit,
                        plot_n_cells=n_cells_fit,
                        calibration_mode=calibration_mode,
                    )

            fit_all_results = st.session_state.get("fit_all_results")
            if fit_all_results is None and os.path.exists("fitted_experiment_parameters.csv"):
                try:
                    fit_all_results = safe_read_csv("fitted_experiment_parameters.csv")
                except Exception:
                    fit_all_results = None

            if fit_all_results is not None:
                sort_map = {
                    "RMSE/C0": ("RMSE_C0", True),
                    "R²": ("R2", False),
                    "ошибка t прорыва": ("breakthrough_time_error_min", True),
                }
                sort_column, ascending = sort_map[sort_fit_table_by]
                table_to_show = fit_all_results.copy()
                if sort_column == "breakthrough_time_error_min":
                    table_to_show["_abs_t_error"] = table_to_show[sort_column].abs()
                    table_to_show = table_to_show.sort_values("_abs_t_error", ascending=True).drop(
                        columns=["_abs_t_error"]
                    )
                else:
                    table_to_show = table_to_show.sort_values(sort_column, ascending=ascending)
                st.dataframe(table_to_show, width="stretch", height=460)

                classification_columns = {
                    "use_for_calibration",
                    "use_for_validation",
                    "exclude_from_ml_ranges",
                    "exclude_reason",
                }
                if classification_columns.issubset(set(fit_all_results.columns)):
                    st.markdown(f"#### {tr(lang, 'automatic_experiment_classification')}")
                    list_columns = [
                        "experiment_id",
                        "RMSE_C0",
                        "R2",
                        "breakthrough_time_exp_min",
                        "breakthrough_time_model_min",
                        "breakthrough_status_exp",
                        "breakthrough_status_model",
                        "exclude_reason",
                    ]
                    available_columns = [
                        column for column in list_columns if column in fit_all_results.columns
                    ]
                    calibration_cases = fit_all_results[
                        fit_all_results["use_for_calibration"] == True
                    ][available_columns]
                    validation_cases = fit_all_results[
                        fit_all_results["use_for_validation"] == True
                    ][available_columns]
                    excluded_cases = fit_all_results[
                        fit_all_results["exclude_from_ml_ranges"] == True
                    ][available_columns]

                    class_tab_cal, class_tab_val, class_tab_exc = st.tabs(
                        [
                            tr(lang, "calibration_cases"),
                            tr(lang, "validation_cases"),
                            tr(lang, "excluded_cases"),
                        ]
                    )
                    with class_tab_cal:
                        st.dataframe(calibration_cases, width="stretch", height=260, hide_index=True)
                    with class_tab_val:
                        st.dataframe(validation_cases, width="stretch", height=260, hide_index=True)
                    with class_tab_exc:
                        st.dataframe(excluded_cases, width="stretch", height=260, hide_index=True)

                comparison_path = os.path.join("figures", "comparison_all_experiments.png")
                figures_dir = os.path.join("figures", "fitted_experiments")
                st.markdown(f"#### {tr(lang, 'calibration_graphs')}")
                figure_files = []
                if os.path.isdir(figures_dir):
                    figure_files = sorted(
                        filename
                        for filename in os.listdir(figures_dir)
                        if filename.lower().endswith(".png")
                    )
                if figure_files:
                    selected_figure = st.selectbox(
                        tr(lang, "show_experiment_graph"),
                        figure_files,
                        key="fit_selected_figure",
                    )
                    try:
                        localized_plot = saved_plot_png([Path(selected_figure).stem], DATA_PATH, "fitted_experiment_parameters.csv", lang)
                        st.image(localized_plot, caption=tr(lang, "ui_saved_graph_note"), width="stretch")
                    except (KeyError, IndexError, ValueError, OSError) as exc:
                        st.warning(tr(lang, "ui_saved_graph_missing"))
                        with st.expander(tr(lang, "ui_details")):
                            st.caption(str(exc))
                            st.image(os.path.join(figures_dir, selected_figure), width="stretch")
                prepare_saved_plots = st.button(tr(lang, "ui_prepare_saved_plots"), key="fit_prepare_localized_plots")
                if prepare_saved_plots or st.button(tr(lang, "open_comparison_plot"), key="fit_show_comparison_plot"):
                    try:
                        with st.spinner(tr(lang, "ui_prepare_saved_plots")):
                            localized_comparison = saved_plot_png(list(experiments), DATA_PATH, "fitted_experiment_parameters.csv", lang)
                        st.session_state["ui_comparison_plot"] = {"lang": lang, "data": localized_comparison}
                        localized_archive = BytesIO()
                        with zipfile.ZipFile(localized_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                            for saved_experiment_id in experiments:
                                archive.writestr(f"{saved_experiment_id}_{lang}.png", saved_plot_png([saved_experiment_id], DATA_PATH, "fitted_experiment_parameters.csv", lang))
                        st.session_state["ui_saved_plots_archive"] = {"lang": lang, "data": localized_archive.getvalue()}
                    except (KeyError, IndexError, ValueError, OSError) as exc:
                        st.warning(tr(lang, "ui_saved_graph_missing"))
                        with st.expander(tr(lang, "ui_details")):
                            st.caption(str(exc))
                localized_comparison = st.session_state.get("ui_comparison_plot", {})
                if localized_comparison.get("lang") == lang:
                    st.image(localized_comparison["data"], caption=tr(lang, "ui_saved_graph_note"), width="stretch")
                    st.download_button(
                        tr(lang, "download_comparison_png"),
                        data=localized_comparison["data"], file_name=f"comparison_all_experiments_{lang}.png",
                        mime="image/png", key="fit_download_comparison_png",
                    )
                    localized_archive = st.session_state.get("ui_saved_plots_archive", {})
                    if localized_archive.get("lang") == lang:
                        st.download_button(tr(lang, "download_figures_zip"), data=localized_archive["data"], file_name=f"fitted_experiment_figures_{lang}.zip", mime="application/zip", key="fit_download_localized_zip")
                if os.path.exists(comparison_path):
                    with st.expander(tr(lang, "ui_details")), open(comparison_path, "rb") as f:
                        st.download_button(
                            tr(lang, "download_comparison_png"),
                            data=f.read(),
                            file_name="comparison_all_experiments.png",
                            mime="image/png",
                            key="fit_download_original_comparison_png",
                        )
                if os.path.exists("fitted_experiment_parameters.csv"):
                    with open("fitted_experiment_parameters.csv", "rb") as f:
                        st.download_button(
                            tr(lang, "download_fitted_parameters_csv"),
                            data=f.read(),
                            file_name="fitted_experiment_parameters.csv",
                            mime="text/csv",
                            key="fit_all_download_csv",
                        )
                with st.expander(tr(lang, "ui_details")):
                    st.caption(tr(lang, "ui_original_plots_note"))
                    st.download_button(
                        tr(lang, "download_figures_zip"),
                        data=make_figures_zip(figures_dir),
                        file_name="fitted_experiment_figures.zip",
                        mime="application/zip",
                        key="fit_all_download_zip",
                    )
                st.caption(tr(lang, "figures_saved_note"))

    with right:
        if run_fit:
            rows = experiments[experiment_id]
            with st.spinner(tr(lang, "ui_fit_running")):
                fitted, metrics = fit_model_to_experiment(
                    rows,
                    epsilon=epsilon_fit,
                    min_time_for_fit=min_time_for_fit,
                    n_cells=n_cells_fit,
                    fit_initial_concentration=fit_initial_concentration,
                    breakthrough_ratio=DEFAULT_BREAKTHROUGH_RATIO,
                )
            if not metrics["success"]:
                st.error(tr(lang, "scientific_optimizer_failed") + ": " + metrics["message"])
                st.stop()

            temperature_values = [
                float(row.get("temperature_C"))
                for row in rows
                if row.get("temperature_C") is not None
                and math.isfinite(float(row.get("temperature_C")))
            ]
            T_exp_C = float(np.median(temperature_values)) if temperature_values else None
            T_ref_C = T_exp_C if T_exp_C is not None else 20.0
            st.session_state["last_calibration"] = {
                "experiment_id": experiment_id,
                "n_cells": int(n_cells_fit),
                "model_version": metrics["model_version"],
                "KH_L_g": fitted.KH_L_g,
                "kLDF_1_min": fitted.kLDF_1_min,
                "Dax_cm2_min": fitted.Dax_cm2_min,
                "T_exp_C": T_exp_C,
                "T_ref_C": T_ref_C,
                "temperature_reference_assumed": T_exp_C is None,
                "KH_ref_L_g": fitted.KH_L_g,
                "kLDF_ref_1_min": fitted.kLDF_1_min,
                "Dax_ref_cm2_min": fitted.Dax_cm2_min,
                "fit_start_time_min": metrics.get("fit_start_time_min"),
                "fit_start_source": metrics.get("fit_start_source"),
                "startup_transient_detected": metrics.get("startup_transient_detected"),
                "startup_end_time_min": metrics.get("startup_end_time_min"),
                "breakthrough_ratio_used": DEFAULT_BREAKTHROUGH_RATIO,
                "breakthrough_interpretation": metrics.get("breakthrough_interpretation", {}),
            }
            st.session_state.pop("manual_holdout_validation_report", None)
            st.session_state.pop("equilibrium_comparison", None)

            column = make_column_from_experiment(rows, epsilon=epsilon_fit)
            exp_time = np.array([row["time_min"] for row in rows], dtype=float)
            exp_conc = np.array([row["C_out_mg_L"] for row in rows], dtype=float)
            exp_ratio = exp_conc / column.c0_mg_l
            breakthrough_ratio = DEFAULT_BREAKTHROUGH_RATIO
            model_time = np.linspace(0, float(np.max(exp_time)), 240)
            model_conc = simulate_breakthrough(model_time, column, fitted, n_cells=n_cells_fit)
            model_ratio = model_conc / column.c0_mg_l
            model_breakthrough = breakthrough_status(model_time, model_ratio, breakthrough_ratio)
            experiment_breakthrough = breakthrough_status(exp_time, exp_ratio, breakthrough_ratio)
            model_breakthrough_time = model_breakthrough["time_min"]
            experiment_breakthrough_time = experiment_breakthrough["time_min"]

            fig = build_breakthrough_figure(
                model_time,
                model_ratio,
                breakthrough_ratio,
                model_breakthrough_time,
                experiment_time=exp_time,
                experiment_ratio=exp_ratio,
                experiment_breakthrough_time_min=experiment_breakthrough_time,
                experiment_breakthrough_label=experiment_breakthrough["label"],
                experiment_breakthrough_status=experiment_breakthrough["status"],
                model_breakthrough_label=model_breakthrough["label"],
                model_breakthrough_status=model_breakthrough["status"],
                fit_start_time_min=metrics.get("fit_start_time_min"),
                startup_end_time_min=metrics.get("startup_end_time_min"),
                title=f"{experiment_id} | {calibration_mode_label}",
                show_experiment_line=show_experiment_line,
                lang=lang,
            )
            st.pyplot(fig)
            st.session_state["ui_current_calibration_plot"] = {"experiment_id": experiment_id, "plot": figure_snapshot(fig)}

            params_df = pd.DataFrame(
                [
                    {
                        "K_H, л/г": fitted.KH_L_g,
                        "k_LDF, 1/мин": fitted.kLDF_1_min,
                        "D_ax, см²/мин": fitted.Dax_cm2_min,
                        "T_exp_C": T_exp_C,
                        "T_ref_C": T_ref_C,
                        "temperature_reference_assumed": T_exp_C is None,
                        "KH_ref_L_g": fitted.KH_L_g,
                        "kLDF_ref_1_min": fitted.kLDF_1_min,
                        "Dax_ref_cm2_min": fitted.Dax_cm2_min,
                        "C_initial, мг/л": fitted.c_initial_mg_l,
                        "Режим калибровки": calibration_mode,
                        "Начало fit, мин": metrics.get("fit_start_time_min"),
                        "Источник начала fit": metrics.get("fit_start_source"),
                        "Startup transient": metrics.get("startup_transient_detected"),
                        "Конец startup, мин": metrics.get("startup_end_time_min"),
                        "Критерий прорыва C/C0": DEFAULT_BREAKTHROUGH_RATIO,
                        "RMSE, мг/л": metrics["rmse_mg_L"],
                        "RMSE/C0": metrics["rmse_C0"],
                        "R²": metrics.get("r2"),
                    }
                ]
            )
            st.dataframe(params_df, width="stretch")
            st.dataframe(
                metrics_dataframe(
                    model_breakthrough_time,
                    column.flow_ml_min,
                    metrics["rmse_mg_L"],
                    experiment_breakthrough_time_min=experiment_breakthrough_time,
                    r2=metrics.get("r2"),
                    model_breakthrough_status=model_breakthrough.get("status"),
                    model_breakthrough_bound_type=model_breakthrough.get("breakthrough_bound_type"),
                    experiment_breakthrough_status=experiment_breakthrough.get("status"),
                    experiment_breakthrough_bound_type=experiment_breakthrough.get("breakthrough_bound_type"),
                ),
                width="stretch",
            )

            if check_fit_mass_balance:
                render_mass_balance_block(model_time, column, fitted, n_cells_fit, "fit", lang)

            if run_bootstrap:
                with st.spinner(tr(lang, "ui_bootstrap_running")):
                    uncertainty = bootstrap_fit_uncertainty(
                        rows,
                        epsilon=epsilon_fit,
                        min_time_for_fit=min_time_for_fit,
                        n_cells=n_cells_fit,
                        n_bootstrap=int(n_bootstrap),
                        fit_initial_concentration=fit_initial_concentration,
                    )
                st.markdown(f"#### {tr(lang, 'ui_bootstrap_title')}")
                st.caption(
                    tr(lang, "ui_bootstrap_results").format(success=uncertainty['n_success'], failed=uncertainty['n_failed'])
                )
                if uncertainty["confidence_intervals"]:
                    ci_df = pd.DataFrame(uncertainty["confidence_intervals"]).T.reset_index()
                    ci_df = ci_df.rename(columns={"index": "parameter"})
                    st.dataframe(ci_df, width="stretch", hide_index=True)
                if uncertainty["correlation_matrix"]:
                    corr_df = pd.DataFrame(uncertainty["correlation_matrix"])
                    st.dataframe(corr_df, width="stretch")

            csv_df = pd.DataFrame(
                {
                    "time_min": model_time,
                    "C_model_mg_L": model_conc,
                    "C_model_C0": model_ratio,
                }
            )
            st.download_button(
                tr(lang, "download_png"),
                data=figure_to_png_bytes(fig),
                file_name=f"{experiment_id}_fit.png",
                mime="image/png",
                key="fit_download_png",
            )
            st.download_button(
                tr(lang, "download_csv"),
                data=csv_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"{experiment_id}_fit_curve.csv",
                mime="text/csv",
                key="fit_download_csv",
            )
            plt.close(fig)
        else:
            cached_calibration = st.session_state.get("ui_current_calibration_plot", {})
            if cached_calibration.get("experiment_id") == experiment_id:
                cached_figure = figure_from_snapshot(cached_calibration["plot"], lang)
                st.pyplot(cached_figure)
                plt.close(cached_figure)
                st.caption(tr(lang, "ui_validation_next"))
            else:
                render_experiment_overview(st, experiment_id, experiments[experiment_id], lang)
                st.caption(tr(lang, "select_experiment_and_run"))

    st.markdown(f"#### {tr(lang, 'manual_holdout_validation_title')}")
    last_calibration = st.session_state.get("last_calibration")
    current_calibration_ready = (
        isinstance(last_calibration, dict)
        and last_calibration.get("experiment_id") == experiment_id
        and last_calibration.get("model_version") == MODEL_VERSION
    )
    if not current_calibration_ready:
        st.caption(tr(lang, "manual_holdout_run_calibration_first"))
    else:
        if last_calibration.get("breakthrough_interpretation", {}).get("crossing_interpretation") == "late_upward_crossing_after_initial_exceedance":
            st.warning(tr(lang, "equilibrium_late_crossing"))
        st.caption(tr(lang, "manual_holdout_validation_caption"))
        st.info(tr(lang, "scientific_holdout_scope"))
        st.caption(tr(lang, "scientific_signal_policy"))
        manual_rows = experiments[experiment_id]
        manual_df = experimental_rows_to_dataframe(manual_rows)
        editor_df = pd.DataFrame(
            {
                "exclude": False,
                "point_index": list(range(len(manual_df))),
                "time_min": manual_df["time_min"].to_numpy(dtype=float),
                "C_out_C0": manual_df["C_out_C0"].to_numpy(dtype=float),
                "C_out_mg_L": manual_df["C_out_mg_L"].to_numpy(dtype=float),
            }
        )
        hidden_selection_key = f"ui_manual_hidden_indices_{experiment_id}"
        saved_hidden_indices = st.session_state.get(hidden_selection_key, [])
        for point_index in saved_hidden_indices:
            if 0 <= point_index < len(editor_df):
                editor_df.loc[point_index, "exclude"] = True
        edited_points = st.data_editor(
            editor_df,
            key=f"manual_holdout_points_{experiment_id}",
            width="stretch",
            hide_index=True,
            disabled=["point_index", "time_min", "C_out_C0", "C_out_mg_L"],
            column_config={
                "exclude": st.column_config.CheckboxColumn(tr(lang, "select_hidden_points")),
            },
        )
        st.session_state[hidden_selection_key] = edited_points.loc[edited_points["exclude"].astype(bool), "point_index"].astype(int).to_list()
        manual_col_1, manual_col_2 = st.columns(2)
        with manual_col_1:
            manual_compare_full_fit = st.checkbox(
                tr(lang, "validation_compare_full_fit"),
                value=False,
                key="manual_holdout_compare_full_fit",
            )
        with manual_col_2:
            manual_breakthrough_ratio = DEFAULT_BREAKTHROUGH_RATIO
            st.metric(
                tr(lang, "validation_breakthrough_ratio"),
                f"{manual_breakthrough_ratio:.4f}",
            )
        if st.button(tr(lang, "run_manual_holdout_validation"), type="primary", key="run_manual_holdout_validation"):
            hidden_indices = (
                edited_points.loc[edited_points["exclude"].astype(bool), "point_index"]
                .astype(int)
                .to_list()
            )
            if not hidden_indices:
                st.error(tr(lang, "select_at_least_one_hidden_point"))
            elif len(manual_df) - len(hidden_indices) < 4:
                st.error(tr(lang, "not_enough_train_points_after_exclusion"))
            else:
                try:
                    with st.status(tr(lang, "validation_running"), expanded=True) as status:
                        status.write(tr(lang, "validation_fit_train_only"))
                        manual_result = run_manual_calibration_holdout_validation(
                            experiment_id,
                            manual_rows,
                            hidden_indices,
                            epsilon=float(epsilon_fit),
                            min_time_for_fit=min_time_for_fit,
                            n_cells=int(n_cells_fit),
                            fit_initial_concentration=bool(fit_initial_concentration),
                            breakthrough_ratio=float(manual_breakthrough_ratio),
                            compare_full_fit=bool(manual_compare_full_fit),
                            lang=lang,
                        )
                        st.session_state["manual_holdout_validation_report"] = manual_result["report"]
                        st.session_state["manual_holdout_hidden_errors_csv"] = manual_result["residuals_df"].to_csv(index=False).encode("utf-8-sig")
                        st.session_state["manual_holdout_validation_plot_png"] = manual_result["plot_png"]
                        st.session_state["ui_manual_holdout_plot"] = figure_snapshot(manual_result["fig"])
                        st.session_state["manual_holdout_metrics_df"] = manual_result["metrics_df"]
                        status.update(label=tr(lang, "manual_holdout_validation_complete"), state="complete")
                except Exception as exc:
                    st.error(tr(lang, "validation_failed"))
                    with st.expander(tr(lang, "show_technical_details")):
                        st.exception(exc)

        manual_report = st.session_state.get("manual_holdout_validation_report")
        if isinstance(manual_report, dict) and manual_report.get("experiment_id") == experiment_id:
            status_value = manual_report.get("validation_status", "unknown")
            st.metric(tr(lang, "curve_similarity_percent"), f"{manual_report.get('curve_similarity_percent', float('nan')):.1f}%")
            st.metric(tr(lang, "hidden_points_similarity_percent"), f"{manual_report.get('hidden_similarity_percent', float('nan')):.1f}%")
            if status_value == "good":
                st.success(tr(lang, "validation_curve_good"))
            elif status_value == "acceptable":
                st.warning(tr(lang, "validation_curve_acceptable"))
            else:
                st.error(tr(lang, "validation_curve_poor"))
            plot_png = st.session_state.get("manual_holdout_validation_plot_png")
            plot_spec = st.session_state.get("ui_manual_holdout_plot")
            if plot_spec:
                localized_figure = figure_from_snapshot(plot_spec, lang)
                plot_png = figure_to_png_bytes(localized_figure)
                plt.close(localized_figure)
            if plot_png:
                st.image(plot_png, caption=tr(lang, "manual_holdout_validation_title"))
            metrics_df = st.session_state.get("manual_holdout_metrics_df")
            if isinstance(metrics_df, pd.DataFrame):
                st.dataframe(metrics_df, width="stretch", hide_index=True)
            hidden_errors_csv = st.session_state.get("manual_holdout_hidden_errors_csv")
            if hidden_errors_csv:
                hidden_errors_df = pd.read_csv(BytesIO(hidden_errors_csv))
                st.dataframe(hidden_errors_df, width="stretch", hide_index=True)
            downloads = st.columns(3)
            with downloads[0]:
                download_button_persistent(
                    tr(lang, "download_manual_validation_report"),
                    json.dumps(manual_report, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                    "manual_holdout_validation_report.json",
                    "application/json",
                    "download_manual_holdout_report_json",
                )
            with downloads[1]:
                if hidden_errors_csv:
                    download_button_persistent(
                        tr(lang, "download_hidden_points_errors"),
                        hidden_errors_csv,
                        "hidden_points_errors.csv",
                        "text/csv",
                        "download_manual_holdout_errors_csv",
                    )
            with downloads[2]:
                if plot_png:
                    download_button_persistent(
                        tr(lang, "download_manual_validation_plot"),
                        plot_png,
                        "manual_holdout_validation_plot.png",
                        "image/png",
                        "download_manual_holdout_plot_png",
                    )

    if current_calibration_ready:
        with st.expander(tr(lang, "equilibrium_comparison_title"), expanded=False):
            st.caption(tr(lang, "equilibrium_comparison_note"))
            st.caption(tr(lang, "equilibrium_nested_seed"))
            if st.button(tr(lang, "equilibrium_compare_run"), key="run_equilibrium_comparison"):
                try:
                    from equilibrium_comparison import compare_equilibria
                    selected_hidden = edited_points.loc[edited_points["exclude"].astype(bool), "point_index"].astype(int).tolist()
                    if not selected_hidden:
                        st.error(tr(lang, "select_at_least_one_hidden_point"))
                    elif len(manual_rows) - len(selected_hidden) < 4:
                        st.error(tr(lang, "not_enough_train_points_after_exclusion"))
                    else:
                        with st.spinner(tr(lang, "equilibrium_compare_running")):
                            comparison = compare_equilibria(manual_rows, selected_hidden,
                                epsilon=float(epsilon_fit), n_cells=int(n_cells_fit),
                                fit_start=float(last_calibration["fit_start_time_min"]),
                                fit_initial_concentration=bool(fit_initial_concentration))
                        # Keep presentation records, not fitted estimators; the
                        # calibration and active breakthrough pointers are untouched.
                        st.session_state["equilibrium_comparison"] = {
                            "experiment_id": experiment_id, "hidden_indices": selected_hidden,
                            "n_cells": int(n_cells_fit), "epsilon": float(epsilon_fit),
                            "fit_start_time_min": float(last_calibration["fit_start_time_min"]),
                            "fit_initial_concentration": bool(fit_initial_concentration),
                            "models": {name: {"parameters": item["parameters"],
                                "train_metrics": item["train_metrics"], "hidden_metrics": item["hidden_metrics"],
                                "stability": item["stability"].to_dict(orient="records"),
                                "curve": item["curve"].to_dict(orient="list"),
                                "residuals": item["residuals"].to_dict(orient="list")}
                                for name, item in comparison.items()}}
                except Exception as exc:
                    st.error(tr(lang, "validation_failed"))
                    st.exception(exc)
            comparison = st.session_state.get("equilibrium_comparison", {})
            if comparison.get("experiment_id") == experiment_id:
                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(manual_df["time_min"], manual_df["C_out_C0"], "o", color="gray", label=tr(lang, "validation_all_experimental_points"))
                hidden_plot = manual_df.iloc[comparison["hidden_indices"]]
                retained_plot = manual_df.drop(index=comparison["hidden_indices"])
                ax.scatter(retained_plot["time_min"], retained_plot["C_out_C0"], color="#1678b0", zorder=3, label=tr(lang, "points_used_for_calibration"))
                ax.scatter(hidden_plot["time_min"], hidden_plot["C_out_C0"], color="red", zorder=4, label=tr(lang, "excluded_points_for_validation"))
                metrics_rows = []
                for name, item in comparison["models"].items():
                    ax.plot(item["curve"]["time_min"], item["curve"]["model_C0"], label=name)
                    metrics_rows.append({"model": name,
                        tr(lang, "equilibrium_train_rmse"): item["train_metrics"]["RMSE_C0"],
                        tr(lang, "equilibrium_hidden_rmse"): item["hidden_metrics"]["RMSE_C0"],
                        "R² (hidden)": item["hidden_metrics"]["R2"]})
                    if any(row["at_bounds"] for row in item["stability"]):
                        st.warning(tr(lang, "equilibrium_boundary_warning"))
                ax.set(xlabel="time_min", ylabel="C/C₀", title=comparison["experiment_id"])
                ax.grid(alpha=.25)
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                st.dataframe(pd.DataFrame(metrics_rows), hide_index=True)
                with st.expander(tr(lang, "show_technical_details")):
                    st.json(comparison)
                download_button_persistent(tr(lang, "equilibrium_download"),
                    json.dumps(comparison, ensure_ascii=False, indent=2).encode("utf-8"),
                    "equilibrium_comparison.json", "application/json", "download_equilibrium_comparison")

    with st.expander(tr(lang, "advanced_experiment_validation"), expanded=False):
        st.caption(tr(lang, "advanced_experiment_validation_caption"))
        validation_rows = experiments[experiment_id]
        validation_df = experimental_rows_to_dataframe(validation_rows)
        validation_default_mode = "leave_one_out" if len(validation_df) <= 50 else "random"
        mode_options = ["leave_one_out", "random", "manual", "block"]
        validation_mode = st.radio(
            tr(lang, "validation_mode"),
            mode_options,
            index=mode_options.index(validation_default_mode),
            format_func=lambda value: tr(lang, f"validation_mode_{value}"),
            horizontal=True,
            key="experiment_validation_mode",
        )
        val_col_1, val_col_2, val_col_3 = st.columns(3)
        with val_col_1:
            validation_breakthrough_ratio = DEFAULT_BREAKTHROUGH_RATIO
            st.metric(
                tr(lang, "validation_breakthrough_ratio"),
                f"{validation_breakthrough_ratio:.4f}",
            )
        with val_col_2:
            validation_seed = st.number_input(
                "random_seed",
                min_value=0,
                value=42,
                step=1,
                key="experiment_validation_seed",
                disabled=validation_mode != "random",
            )
        with val_col_3:
            compare_full_fit = st.checkbox(
                tr(lang, "validation_compare_full_fit"),
                value=False,
                key="experiment_validation_compare_full_fit",
                disabled=validation_mode == "leave_one_out",
            )

        holdout_fraction = 0.25
        manual_times: list[float] = []
        block_name = "middle"
        if validation_mode == "random":
            holdout_fraction = st.slider(
                tr(lang, "validation_holdout_fraction"),
                min_value=0.1,
                max_value=0.5,
                value=0.25,
                step=0.05,
                key="experiment_validation_holdout_fraction",
            )
        elif validation_mode == "manual":
            time_values = [float(value) for value in validation_df["time_min"].dropna().to_list()]
            manual_times = st.multiselect(
                tr(lang, "validation_manual_hidden_times"),
                options=time_values,
                default=time_values[len(time_values) // 2 : len(time_values) // 2 + 1] if time_values else [],
                format_func=lambda value: f"{float(value):.3g} min",
                key="experiment_validation_manual_times",
            )
        elif validation_mode == "block":
            block_name = st.selectbox(
                tr(lang, "validation_block"),
                ["early", "middle", "late", "near_breakthrough"],
                format_func=lambda value: tr(lang, f"validation_block_{value}"),
                key="experiment_validation_block",
            )
        if validation_mode == "leave_one_out" and len(validation_df) > 50:
            st.warning(tr(lang, "validation_loo_heavy_warning").format(count=len(validation_df)))

        if st.button(tr(lang, "run_experiment_validation"), type="primary", key="run_experiment_holdout_validation"):
            try:
                with st.status(tr(lang, "validation_running"), expanded=True) as status:
                    status.write(tr(lang, "validation_split_points"))
                    if validation_mode == "leave_one_out":
                        result = run_leave_one_out_validation(
                            experiment_id,
                            validation_rows,
                            epsilon=float(epsilon_fit),
                            min_time_for_fit=min_time_for_fit,
                            n_cells=int(n_cells_fit),
                            fit_initial_concentration=bool(fit_initial_concentration),
                            breakthrough_ratio=float(validation_breakthrough_ratio),
                            lang=lang,
                        )
                    else:
                        train_indices, hidden_indices = validation_split_indices(
                            validation_df,
                            validation_mode,
                            holdout_fraction=float(holdout_fraction),
                            seed=int(validation_seed),
                            manual_times=manual_times,
                            block_name=block_name,
                            breakthrough_ratio=float(validation_breakthrough_ratio),
                        )
                        status.write(tr(lang, "validation_fit_train_only"))
                        result = run_single_holdout_validation(
                            experiment_id,
                            validation_rows,
                            train_indices,
                            hidden_indices,
                            epsilon=float(epsilon_fit),
                            min_time_for_fit=min_time_for_fit,
                            n_cells=int(n_cells_fit),
                            fit_initial_concentration=bool(fit_initial_concentration),
                            breakthrough_ratio=float(validation_breakthrough_ratio),
                            compare_full_fit=bool(compare_full_fit),
                            mode_label=tr(lang, f"validation_mode_{validation_mode}"),
                            lang=lang,
                        )
                    report = result["report"]
                    st.session_state["experiment_validation_report"] = report
                    st.session_state["experiment_validation_metrics_csv"] = result["residuals_df"].to_csv(index=False).encode("utf-8-sig")
                    st.session_state["experiment_validation_plot_png"] = result["plot_png"]
                    st.session_state["ui_advanced_validation_plot"] = figure_snapshot(result["figure"])
                    status.update(label=tr(lang, "validation_complete"), state="complete")
            except Exception as exc:
                st.error(tr(lang, "validation_failed"))
                with st.expander(tr(lang, "show_technical_details")):
                    st.exception(exc)

        validation_report = st.session_state.get("experiment_validation_report")
        if isinstance(validation_report, dict):
            status_value = validation_report.get("validation_status", "unknown")
            if status_value == "good":
                st.success(tr(lang, "validation_status_good"))
            elif status_value == "acceptable":
                st.info(tr(lang, "validation_status_acceptable"))
            else:
                st.error(tr(lang, "validation_status_poor"))
            st.write(validation_report.get("recommendation", ""))
            summary_df = pd.DataFrame(
                [
                    {"metric_group": tr(lang, "validation_train_error"), **(validation_report.get("metrics_train") or {})},
                    {"metric_group": tr(lang, "validation_hidden_error"), **(validation_report.get("metrics_validation") or {})},
                ]
            )
            st.dataframe(summary_df, width="stretch", hide_index=True)
            if validation_report.get("breakthrough_error"):
                st.markdown(f"##### {tr(lang, 'validation_breakthrough_error')}")
                st.dataframe(pd.DataFrame([validation_report["breakthrough_error"]]), width="stretch", hide_index=True)
            else:
                st.info(tr(lang, "validation_breakthrough_not_calculated"))
            plot_png = st.session_state.get("experiment_validation_plot_png")
            plot_spec = st.session_state.get("ui_advanced_validation_plot")
            if plot_spec:
                localized_figure = figure_from_snapshot(plot_spec, lang)
                plot_png = figure_to_png_bytes(localized_figure)
                plt.close(localized_figure)
            if plot_png:
                st.image(plot_png, caption=tr(lang, "experiment_holdout_validation"))
            if validation_report.get("hidden_points"):
                st.markdown(f"##### {tr(lang, 'validation_residuals')}")
                st.dataframe(pd.DataFrame(validation_report["hidden_points"]), width="stretch", hide_index=True)
            downloads = st.columns(3)
            with downloads[0]:
                download_button_persistent(
                    tr(lang, "download_validation_report_json"),
                    json.dumps(validation_report, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
                    "experiment_validation_report.json",
                    "application/json",
                    "download_experiment_validation_json",
                )
            with downloads[1]:
                metrics_csv = st.session_state.get("experiment_validation_metrics_csv")
                if metrics_csv:
                    download_button_persistent(
                        tr(lang, "download_validation_metrics_csv"),
                        metrics_csv,
                        "experiment_validation_metrics.csv",
                        "text/csv",
                        "download_experiment_validation_csv",
                    )
            with downloads[2]:
                if plot_png:
                    download_button_persistent(
                        tr(lang, "download_validation_plot_png"),
                        plot_png,
                        "experiment_validation_plot.png",
                        "image/png",
                        "download_experiment_validation_png",
                    )


_startup_probe.mark("simulator_begin")
with tab_simulator:
    st.subheader(tr(lang, "simulator_title"))
    st.caption(tr(lang, "ui_simulation_workflow"))
    with st.expander(tr(lang, "help_parameters_not_universal_title")):
        st.markdown(tr(lang, "universal_params_warning"))

    top_col_1, top_col_2 = st.columns(2)
    with top_col_1:
        work_mode = st.radio(
            tr(lang, "operation_mode"),
            ["Одна симуляция", "Генерация серии экспериментов для ML"],
            horizontal=True,
            key="sim_work_mode",
            format_func=lambda value: tr(lang, WORK_MODE_KEYS[value]),
        )
    with top_col_2:
        scale_mode = st.selectbox(
            tr(lang, "scale"),
            SCALE_LABELS,
            key="sim_scale_mode",
            format_func=lambda value: tr(lang, SCALE_TRANSLATION_KEYS[value]),
        )
        st.caption(tr(lang, SCALE_INFO_KEYS[scale_mode]))

    profile_key = st.selectbox(
        tr(lang, "adsorbent"),
        list(ADSORBENT_PROFILES.keys()),
        index=list(ADSORBENT_PROFILES.keys()).index(DEFAULT_PROFILE_KEY),
        format_func=lambda value: profile_label(value, lang),
        key="sim_profile_key",
    )
    pollutant = st.text_input(tr(lang, "pollutant"), value=tr(lang, "methylene_blue"), key="sim_pollutant")

    with st.expander(tr(lang, "selected_adsorbent_characteristics"), expanded=False):
        render_adsorbent_card(profile_key, lang)

    parameter_source = st.selectbox(
        tr(lang, "model_parameter_source"),
        [
            "Авто: лучший доступный источник",
            "Последняя калибровка",
            "Лучший эксперимент",
            "Профиль выбранного угля",
            "Литературный типовой профиль",
            "Ручной ввод",
        ],
        key="sim_parameter_source",
        format_func=lambda value: tr(lang, PARAMETER_SOURCE_KEYS[value]),
    )
    auto_params, source_message, source_level = resolve_auto_sorption_parameters(
        parameter_source,
        profile_key,
        pollutant,
        lang,
    )
    render_parameter_source(
        st, auto_params, experiments, source_message, source_level,
        make_sorption_parameter_source_dataframe(auto_params, lang), lang,
    )
    if auto_params.get("parameter_source") == "literature_generic":
        st.warning(tr(lang, "literature_profile_warning"))

    preset = preset_for_scale(scale_mode)
    if st.button(tr(lang, "apply_recommended_values"), key="sim_apply_preset"):
        apply_simulator_preset(scale_mode)
        st.success(tr(lang, "recommended_values_applied").format(scale=tr(lang, SCALE_TRANSLATION_KEYS[scale_mode])))

    if work_mode == "Одна симуляция":
        main_left, main_right = st.columns([0.38, 0.62])
        with main_left:
            st.markdown(f"#### {tr(lang, 'main_parameters')}")
            C0_mg_L = st.number_input(
                tr(lang, "c0_mg_l"),
                min_value=0.001,
                max_value=100.0,
                value=float(preset["C0_mg_L"]),
                key="sim_C0",
                help=tr(lang, "c0_help"),
            )
            if C0_mg_L > 10.0:
                st.warning(
                    tr(lang, "help_henry_isotherm_body")
                )

            breakthrough_ratio = DEFAULT_BREAKTHROUGH_RATIO
            C_break_mg_L = breakthrough_ratio * C0_mg_L
            st.info(
                tr(lang, "scientific_breakthrough_criterion").format(
                    ratio=f"{breakthrough_ratio:.2f}",
                    concentration=f"{C_break_mg_L:.4g}",
                )
            )

            T_C = st.number_input(
                tr(lang, "temperature_c"),
                min_value=0.0,
                max_value=100.0,
                value=float(preset["T_C"]),
                key="sim_T_C",
                help=tr(lang, "temperature_help"),
            )

            st.markdown(f"#### {tr(lang, 'pressure_mode')}")
            pressure_mode = st.selectbox(
                tr(lang, "pressure_mode"),
                list(PRESSURE_MODE_KEYS.keys()),
                index=0,
                key="sim_pressure_mode",
                format_func=lambda value: tr(lang, PRESSURE_MODE_KEYS[value]),
                help=tr(lang, "pressure_help"),
            )
            st.caption(tr(lang, "pressure_note"))
            if pressure_mode == "Атмосферное":
                P_in_bar = st.number_input(
                    tr(lang, "pressure_bar"),
                    min_value=0.1,
                    max_value=10.0,
                    value=1.01325,
                    format="%.5f",
                    key="sim_pressure_atm_bar",
                    disabled=True,
                    help=tr(lang, "pressure_help"),
                )
                P_out_bar = 1.01325
                deltaP_available_bar = None
            elif pressure_mode == "Задать давление на входе":
                P_in_bar = st.number_input(
                    tr(lang, "inlet_pressure_bar"),
                    min_value=0.1,
                    max_value=10.0,
                    value=1.01325,
                    format="%.5f",
                    key="sim_P_in_bar",
                    help=tr(lang, "pressure_help"),
                )
                P_out_bar = st.number_input(
                    tr(lang, "outlet_pressure_bar"),
                    min_value=0.1,
                    max_value=10.0,
                    value=1.01325,
                    format="%.5f",
                    key="sim_P_out_bar",
                    help=tr(lang, "pressure_help"),
                )
                deltaP_available_bar = max(float(P_in_bar) - float(P_out_bar), 0.0)
            else:
                deltaP_available_bar = st.number_input(
                    tr(lang, "available_delta_p_bar"),
                    min_value=0.0,
                    max_value=10.0,
                    value=0.5,
                    format="%.5f",
                    key="sim_deltaP_available_bar",
                    help=tr(lang, "pressure_help"),
                )
                P_out_bar = st.number_input(
                    tr(lang, "outlet_pressure_bar"),
                    min_value=0.1,
                    max_value=10.0,
                    value=1.01325,
                    format="%.5f",
                    key="sim_deltaP_P_out_bar",
                    help=tr(lang, "pressure_help"),
                )
                P_in_bar = float(P_out_bar) + float(deltaP_available_bar)

            if scale_mode == SCALE_LABELS[0]:
                flow_value = st.number_input(tr(lang, "flow_ml_min"), min_value=1e-6, value=float(preset["Q_m3_h"] * 1_000_000.0 / 60.0), key="sim_flow_lab", help=tr(lang, "flow_help"))
                height_value = st.number_input(tr(lang, "bed_height_cm"), min_value=0.001, value=float(preset["H_m"] * 100.0), key="sim_height_lab", help=tr(lang, "height_help"))
                diameter_value = st.number_input(tr(lang, "column_diameter_cm"), min_value=0.001, value=float(preset["D_col_m"] * 100.0), key="sim_diameter_lab", help=tr(lang, "diameter_help"))
            else:
                flow_value = st.number_input(tr(lang, "flow_m3_h"), min_value=1e-9, value=float(preset["Q_m3_h"]), format="%.6f", key="sim_flow_si", help=tr(lang, "flow_help"))
                height_value = st.number_input(tr(lang, "bed_height_m"), min_value=0.001, value=float(preset["H_m"]), format="%.4f", key="sim_height_si", help=tr(lang, "height_help"))
                diameter_value = st.number_input(tr(lang, "column_diameter_m"), min_value=0.001, value=float(preset["D_col_m"]), format="%.4f", key="sim_diameter_si", help=tr(lang, "diameter_help"))

            parameter_source_mode = PARAMETER_SOURCE_MODE_KEYS.get(parameter_source, "auto_best_available")
            if parameter_source_mode != "manual":
                st.session_state["sim_KH"] = auto_params["KH_L_g"]
                st.session_state["sim_kLDF"] = auto_params["kLDF_1_min"]
                st.session_state["sim_Dax"] = auto_params["Dax_cm2_min"]

            if parameter_source_mode == "manual":
                KH_ref = st.number_input(tr(lang, "kh_label"), min_value=1e-9, value=float(auto_params["KH_L_g"]), format="%.8f", key="sim_KH", help=tr(lang, "help_henry_isotherm_body"))
                kLDF_ref = st.number_input(tr(lang, "kldf_label"), min_value=1e-9, value=float(auto_params["kLDF_1_min"]), format="%.8f", key="sim_kLDF", help=tr(lang, "help_ldf_body"))
                Dax_ref = st.number_input(tr(lang, "dax_label"), min_value=1e-12, value=float(auto_params["Dax_cm2_min"]), format="%.10f", key="sim_Dax", help=tr(lang, "help_axial_dispersion_body"))
            else:
                KH_ref = float(st.session_state["sim_KH"])
                kLDF_ref = float(st.session_state["sim_kLDF"])
                Dax_ref = float(st.session_state["sim_Dax"])

            curve_display_mode = st.selectbox(
                tr(lang, "curve_display_mode"),
                CURVE_DISPLAY_OPTIONS,
                key="sim_curve_display_mode",
                format_func=lambda value: tr(lang, value),
            )
            st.caption(tr(lang, "curve_mode_note"))
            manual_time_h = st.number_input(tr(lang, "time_h"), min_value=0.001, value=float(preset["t_end_h"]), key="sim_manual_time_h")

            with st.expander(tr(lang, "advanced_parameters"), expanded=False):
                epsilon = st.number_input(tr(lang, "porosity"), min_value=0.05, max_value=0.95, value=float(preset["epsilon"]), format="%.4f", key="sim_epsilon")
                if scale_mode == SCALE_LABELS[0]:
                    bulk_density = st.number_input("Плотность загрузки слоя, г/см³", min_value=0.001, value=float(preset["bulk_density_kg_m3"] / 1000.0), format="%.4f", key="sim_bulk_density_lab")
                else:
                    bulk_density = st.number_input("Насыпная плотность / плотность загрузки, кг/м³", min_value=1.0, value=float(preset["bulk_density_kg_m3"]), key="sim_bulk_density_si")
                dp_mm = st.number_input("Диаметр частиц dp, мм", min_value=0.001, value=float(preset["dp_mm"]), format="%.4f", key="sim_dp_mm")
                particle_sphericity = st.number_input("Сферичность частиц", min_value=0.5, max_value=1.0, value=float(preset["particle_sphericity"]), step=0.01, format="%.2f", key="sim_particle_sphericity")
                if parameter_source_mode != "manual":
                    KH_ref = st.number_input(tr(lang, "kh_label"), min_value=1e-9, value=KH_ref, format="%.8f", key="sim_KH_advanced", help=tr(lang, "help_henry_isotherm_body"))
                    kLDF_ref = st.number_input(tr(lang, "kldf_label"), min_value=1e-9, value=kLDF_ref, format="%.8f", key="sim_kLDF_advanced", help=tr(lang, "help_ldf_body"))
                    Dax_ref = st.number_input(tr(lang, "dax_label"), min_value=1e-12, value=Dax_ref, format="%.10f", key="sim_Dax_advanced", help=tr(lang, "help_axial_dispersion_body"))
                auto_temperature_parameters = resolve_temperature_correction_parameters(
                    profile_key,
                    pollutant,
                    fitted_temperature_dataframe(),
                )
                T_ref_C = temperature_reference_from_params(auto_params, float(preset["T_ref_C"]))
                if auto_temperature_parameters.get("source") == "estimated_from_own_calibrations":
                    deltaH_ads = float(auto_temperature_parameters["deltaH_ads_kJ_mol"])
                    Ea_kLDF = float(auto_temperature_parameters["Ea_kLDF_kJ_mol"])
                    D_m_ref = float(auto_temperature_parameters["D_m_ref_m2_s"])
                else:
                    deltaH_ads = float(auto_params.get("deltaH_ads_kJ_mol", auto_temperature_parameters["deltaH_ads_kJ_mol"]))
                    Ea_kLDF = float(auto_params.get("Ea_kLDF_kJ_mol", auto_temperature_parameters["Ea_kLDF_kJ_mol"]))
                    D_m_ref = float(auto_params.get("D_m_ref_m2_s", auto_temperature_parameters["D_m_ref_m2_s"]))
                temperature_correction_source = str(auto_temperature_parameters["source"])
                temperature_correction_confidence = str(auto_temperature_parameters["confidence"])
                Dax_temperature_mode = "proportional_to_diffusion"
                Dax_mode = "температурная поправка к D_ax_ref"
                kLDF_mode = "температурная поправка Arrhenius"
                temperature_sensitivity_deltaH_values = None

                with st.expander(tr(lang, "temperature_adaptation_settings"), expanded=False):
                    override_temperature = st.checkbox(
                        tr(lang, "override_temperature_parameters"),
                        value=False,
                        key="sim_override_temperature_parameters",
                    )
                    scenario_options = [
                        "auto",
                        "exothermic",
                        "weak",
                        "endothermic",
                        "manual",
                        "sensitivity",
                    ]
                    scenario_key_map = {
                        "auto": "temperature_scenario_auto",
                        "exothermic": "temperature_scenario_exothermic",
                        "weak": "temperature_scenario_weak",
                        "endothermic": "temperature_scenario_endothermic",
                        "manual": "temperature_scenario_manual",
                        "sensitivity": "temperature_scenario_sensitivity",
                    }
                    if override_temperature:
                        temperature_scenario = st.selectbox(
                            tr(lang, "temperature_scenario"),
                            scenario_options,
                            index=0,
                            key="sim_temperature_scenario",
                            format_func=lambda value: tr(lang, scenario_key_map[value]),
                        )
                        if temperature_scenario == "exothermic":
                            deltaH_ads = -20.0
                            temperature_correction_source = "manual_override"
                            temperature_correction_confidence = "user_defined"
                        elif temperature_scenario == "weak":
                            deltaH_ads = 0.0
                            temperature_correction_source = "manual_override"
                            temperature_correction_confidence = "user_defined"
                        elif temperature_scenario == "endothermic":
                            deltaH_ads = 20.0
                            temperature_correction_source = "manual_override"
                            temperature_correction_confidence = "user_defined"
                        elif temperature_scenario == "manual":
                            deltaH_ads = st.number_input("deltaH_ads, kJ/mol", min_value=-60.0, max_value=60.0, value=deltaH_ads, key="sim_deltaH_manual")
                            temperature_correction_source = "manual_override"
                            temperature_correction_confidence = "user_defined"
                        elif temperature_scenario == "sensitivity":
                            temperature_sensitivity_deltaH_values = [-30, -20, -10, 0, 10, 20, 30]
                            temperature_correction_source = "manual_override"
                            temperature_correction_confidence = "user_defined"
                        Ea_kLDF = st.number_input("Ea_kLDF, kJ/mol", min_value=0.0, max_value=100.0, value=Ea_kLDF, key="sim_Ea_override")
                        D_m_ref = st.number_input("D_m_ref, m²/s", min_value=1e-12, value=D_m_ref, format="%.2e", key="sim_D_m_ref_override")
                        T_ref_C = st.number_input("T_ref, °C", min_value=-20.0, max_value=100.0, value=T_ref_C, key="sim_T_ref_override")
                        Dax_temperature_mode = st.selectbox(
                            tr(lang, "dax_mode"),
                            ["proportional_to_diffusion", "no_correction", "engineering_estimate"],
                            key="sim_Dax_temperature_mode",
                        )
                    else:
                        st.number_input("T_ref, °C", value=T_ref_C, disabled=True, key="sim_T_ref_auto_display")
                        st.number_input("deltaH_ads, kJ/mol", value=deltaH_ads, disabled=True, key="sim_deltaH_auto_display")
                        st.number_input("Ea_kLDF, kJ/mol", value=Ea_kLDF, disabled=True, key="sim_Ea_auto_display")
                        st.number_input("D_m_ref, m²/s", value=D_m_ref, format="%.2e", disabled=True, key="sim_Dm_auto_display")
                apply_pressure_KH_correction = st.checkbox(
                    tr(lang, "pressure_kh_correction"),
                    value=False,
                    key="sim_apply_pressure_KH",
                    help="Не включайте без экспериментально обоснованного ΔV_ads. В базовом режиме давление не меняет K_H.",
                )
                deltaV_ads_cm3_mol = 0.0
                if apply_pressure_KH_correction:
                    st.warning(tr(lang, "pressure_correction_warning"))
                    deltaV_ads_cm3_mol = st.number_input(
                        "ΔV_ads, см³/моль",
                        value=0.0,
                        format="%.6f",
                        key="sim_deltaV_ads",
                        help="Экспериментальный объёмный параметр для поправки K_H(P).",
                    )
                n_cells = st.number_input(tr(lang, "model_cells_label"), min_value=20, max_value=1280, value=int((st.session_state.get("last_calibration") or {}).get("n_cells", preset["n_cells"])), step=10, key="sim_n_cells")
                n_points = st.number_input(tr(lang, "points_count"), min_value=20, max_value=5000, value=int(preset["n_points"]), step=10, key="sim_n_points")

            if scale_mode == SCALE_LABELS[0]:
                area_for_temperature_cm2 = column_area_from_diameter(float(diameter_value))
                superficial_velocity_for_temperature_m_s = (
                    float(flow_value) / max(area_for_temperature_cm2, 1e-30)
                ) * 0.01 / 60.0
            else:
                area_for_temperature_m2 = math.pi * float(diameter_value) ** 2 / 4.0
                superficial_velocity_for_temperature_m_s = (
                    float(flow_value) / 3600.0
                ) / max(area_for_temperature_m2, 1e-30)
            temperature_adaptation = apply_temperature_adaptation(
                KH_ref_L_g=float(KH_ref),
                kLDF_ref_1_min=float(kLDF_ref),
                Dax_ref_cm2_min=float(Dax_ref),
                T_ref_C=float(T_ref_C),
                T_sim_C=float(T_C),
                deltaH_ads_kJ_mol=float(deltaH_ads),
                Ea_kLDF_kJ_mol=float(Ea_kLDF),
                D_m_ref_m2_s=float(D_m_ref),
                Dax_temperature_mode=str(Dax_temperature_mode),
                source=str(temperature_correction_source),
                confidence=str(temperature_correction_confidence),
                superficial_velocity_m_s=float(superficial_velocity_for_temperature_m_s),
                dp_m=float(dp_mm) / 1000.0,
            )
            temperature_records = parameter_origins(auto_temperature_parameters) if auto_temperature_parameters.get("source") == "estimated_from_own_calibrations" else parameter_origins(auto_params)
            effective_values = {
                "KH_L_g": float(KH_ref), "kLDF_1_min": float(kLDF_ref), "Dax_cm2_min": float(Dax_ref),
                "T_ref_C": float(T_ref_C), "T_sim_C": float(T_C),
                "deltaH_ads_kJ_mol": float(deltaH_ads), "Ea_kLDF_kJ_mol": float(Ea_kLDF),
                "D_m_ref_m2_s": float(D_m_ref),
            }
            user_parameters = {"T_sim_C"}
            if parameter_source_mode == "manual":
                user_parameters.update({"KH_L_g", "kLDF_1_min", "Dax_cm2_min"})
            if override_temperature:
                user_parameters.update({"Ea_kLDF_kJ_mol", "D_m_ref_m2_s", "T_ref_C"})
                if temperature_scenario in {"exothermic", "weak", "endothermic", "manual"}:
                    user_parameters.add("deltaH_ads_kJ_mol")
            inherited_temperature = {key: temperature_records[key] for key in ("deltaH_ads_kJ_mol", "Ea_kLDF_kJ_mol", "D_m_ref_m2_s") if key in temperature_records}
            effective_params = effective_input_payload(auto_params, effective_values, user_parameters, inherited_temperature)
            if "T_ref_C" not in auto_params and not override_temperature:
                effective_params["parameter_provenance"]["T_ref_C"] = provenance_record("default", method="preset_default", assignment_path="preset.T_ref_C")
            effective_params["engineering_estimate_available"] = True
            effective_params["parameter_provenance"].update(geometry_records(scale_mode, "user"))
            temperature_adaptation = temperature_lineage(temperature_adaptation, effective_params)
            st.session_state["sim_effective_parameter_payload"] = effective_params
            with st.expander(tr(lang, "provenance_effective_parameters")):
                st.dataframe(make_sorption_parameter_source_dataframe(effective_params, lang,
                             {"solver": "henry_ldf", "Dax_temperature_mode": Dax_temperature_mode}), width="stretch", hide_index=True)
                st.json(temperature_adaptation["parameter_provenance"])
            st.info(tr(lang, "temperature_adaptation_auto"))
            st.dataframe(
                make_temperature_adaptation_dataframe(temperature_adaptation),
                width="stretch",
                hide_index=True,
            )
            for level, message_key in temperature_adaptation_messages(temperature_adaptation):
                if level == "warning":
                    st.warning(tr(lang, message_key))
                else:
                    st.info(tr(lang, message_key))

            calibration_ranges = load_calibration_ranges()
            operating_regime = check_operating_regime(
                scale_mode=scale_mode,
                Q=float(flow_value),
                H=float(height_value),
                D=float(diameter_value),
                epsilon=float(epsilon),
                C0=float(C0_mg_L),
                C_breakthrough=float(C_break_mg_L),
                bulk_density=float(bulk_density),
                calibration_ranges=calibration_ranges,
                temperature_C=float(T_C),
            )
            st.markdown(f"#### {tr(lang, 'operating_regime_diagnostics')}")
            st.dataframe(make_operating_regime_dataframe(operating_regime, lang), width="stretch", hide_index=True)
            for message in operating_regime.get("info", []):
                st.info(message)
            for message in operating_regime.get("warnings", []):
                st.warning(message)
            chart_options = ["static", "interactive"] if PLOTLY_AVAILABLE else ["static"]
            if PLOTLY_AVAILABLE:
                pass
            else:
                st.warning(tr(lang, "plotly_missing"))
            chart_type = st.selectbox(
                tr(lang, "chart_type"),
                chart_options,
                key="sim_chart_type",
                format_func=lambda value: tr(lang, "chart_interactive" if value == "interactive" else "chart_static"),
            )
            check_mass_balance = st.checkbox(
                tr(lang, "run_mass_balance_check"),
                value=False,
                key="sim_check_mass_balance",
            )
            run_sim = st.button(tr(lang, "run_simulation"), type="primary", key="sim_run")

        with main_right:
            if run_sim and C_break_mg_L < C0_mg_L:
                input_check = validate_simulation_input(
                    C0_mg_L=float(C0_mg_L),
                    C_breakthrough_mg_L=float(C_break_mg_L),
                    flow=float(flow_value),
                    height=float(height_value),
                    diameter=float(diameter_value),
                    temperature_C=float(T_C),
                    epsilon=float(epsilon),
                    bulk_density=float(bulk_density),
                    KH_L_g=float(KH_ref),
                    kLDF_1_min=float(kLDF_ref),
                    Dax_cm2_min=float(Dax_ref),
                    n_cells=int(n_cells),
                    n_points=int(n_points),
                )
                if scale_mode != SCALE_LABELS[0]:
                    industrial_check = validate_industrial_input(
                        Q_m3_h=float(flow_value),
                        D_col_m=float(diameter_value),
                        H_m=float(height_value),
                        dp_mm=float(dp_mm),
                        epsilon=float(epsilon),
                        bulk_density_kg_m3=float(bulk_density),
                        particle_sphericity=float(particle_sphericity),
                        T_C=float(T_C),
                    )
                    input_check["errors"].extend(industrial_check["errors"])
                    input_check["warnings"].extend(industrial_check["warnings"])
                for message in input_check["warnings"]:
                    st.warning(message)
                if input_check["errors"]:
                    for message in input_check["errors"]:
                        st.error(message)
                    st.stop()
                column, sorption, diagnostics, validation_messages = build_column_and_diagnostics(
                    scale_mode, C0_mg_L, flow_value, height_value, diameter_value, float(epsilon), float(bulk_density), T_C,
                    float(dp_mm), float(particle_sphericity), float(KH_ref), float(kLDF_ref), float(Dax_ref),
                    float(T_ref_C), float(deltaH_ads), float(Ea_kLDF), float(D_m_ref), Dax_mode, kLDF_mode,
                    pressure_mode=pressure_mode,
                    P_in_bar=float(P_in_bar),
                    P_out_bar=float(P_out_bar),
                    deltaP_available_bar=None if deltaP_available_bar is None else float(deltaP_available_bar),
                    apply_pressure_KH_correction=bool(apply_pressure_KH_correction),
                    deltaV_ads_cm3_mol=float(deltaV_ads_cm3_mol),
                    temperature_adaptation=temperature_adaptation,
                )
                diagnostics.update(operating_regime)
                for message in validation_messages:
                    st.warning(message)
                for message in diagnostics.get("warnings", []):
                    st.warning(message)
                result = simulate_with_time_mode(
                    curve_display_mode,
                    scale_mode,
                    float(manual_time_h) * 60.0,
                    int(n_points),
                    column,
                    sorption,
                    int(n_cells),
                    float(breakthrough_ratio),
                )
                if not result.get("target_reached", False) and curve_display_mode != "manual_time":
                    st.warning(tr(lang, "target_not_reached"))
                tb_calc = result["breakthrough"].get("time_min")
                ebct_value = diagnostics.get("EBCT_min")
                void_time = diagnostics.get("void_residence_time_min")
                if tb_calc is not None and ebct_value is not None and float(tb_calc) < float(ebct_value):
                    st.warning(tr(lang, "tb_less_than_ebct_warning"))
                if tb_calc is not None and void_time is not None and float(tb_calc) < float(void_time):
                    st.warning(tr(lang, "tb_less_than_void_warning"))
                if diagnostics.get("EBCT_min", 999.0) < 1.0 or diagnostics.get("bed_volumes_per_min", 0.0) > 1.0:
                    st.warning(tr(lang, "flow_too_high_warning"))
                st.caption(result["auto_time_message"])
                fig = plot_simulation_curve(
                    result["time_min"], result["C_out_C0"], float(breakthrough_ratio), result["breakthrough"], scale_mode,
                    title=f"{tr(lang, SCALE_TRANSLATION_KEYS[scale_mode])}: {pollutant} / {profile_label(profile_key, lang)}", ebct_min=float(diagnostics.get("EBCT_min", 0.0)),
                    lang=lang,
                )
                if chart_type == "interactive" and PLOTLY_AVAILABLE:
                    plotly_fig = plot_breakthrough_interactive(
                        result["time_min"],
                        result["C_out_mg_L"],
                        result["C_out_C0"],
                        float(breakthrough_ratio),
                        breakthrough_time_min=result["breakthrough"].get("time_min"),
                        ebct_min=float(diagnostics.get("EBCT_min", 0.0)),
                        title=f"{tr(lang, SCALE_TRANSLATION_KEYS[scale_mode])}: {pollutant}",
                        lang=lang,
                    )
                    st.plotly_chart(plotly_fig, use_container_width=True)
                else:
                    st.pyplot(fig)
                st.caption(tr(lang, "curve_mode_note"))
                if temperature_sensitivity_deltaH_values:
                    sensitivity_fig, sensitivity_ax = plt.subplots(figsize=(9, 5))
                    for deltaH_case in temperature_sensitivity_deltaH_values:
                        sensitivity_adaptation = apply_temperature_adaptation(
                            KH_ref_L_g=float(KH_ref),
                            kLDF_ref_1_min=float(kLDF_ref),
                            Dax_ref_cm2_min=float(Dax_ref),
                            T_ref_C=float(T_ref_C),
                            T_sim_C=float(T_C),
                            deltaH_ads_kJ_mol=float(deltaH_case),
                            Ea_kLDF_kJ_mol=float(Ea_kLDF),
                            D_m_ref_m2_s=float(D_m_ref),
                            Dax_temperature_mode=str(Dax_temperature_mode),
                            source="manual_override",
                            confidence="user_defined",
                            superficial_velocity_m_s=float(superficial_velocity_for_temperature_m_s),
                            dp_m=float(dp_mm) / 1000.0,
                        )
                        sensitivity_params = effective_input_payload(effective_params, {"deltaH_ads_kJ_mol": float(deltaH_case)},
                            inherited={"deltaH_ads_kJ_mol": provenance_record("default", method="preset_default", assignment_path="temperature_sensitivity_deltaH_values")})
                        sensitivity_adaptation = temperature_lineage(sensitivity_adaptation, sensitivity_params)
                        sensitivity_sorption = SorptionParams(
                            float(sensitivity_adaptation["KH_used_L_g"]),
                            float(sensitivity_adaptation["kLDF_used_1_min"]),
                            float(sensitivity_adaptation["Dax_used_cm2_min"]),
                        )
                        sensitivity_conc = simulate_breakthrough(
                            result["time_min"],
                            column,
                            sensitivity_sorption,
                            n_cells=int(n_cells),
                        )
                        sensitivity_ax.plot(
                            result["time_min"],
                            sensitivity_conc / float(C0_mg_L),
                            linewidth=1.4,
                            label=f"ΔH_ads = {deltaH_case:+.0f} kJ/mol",
                        )
                    sensitivity_ax.axhline(
                        float(breakthrough_ratio),
                        color="tab:red",
                        linestyle="--",
                        linewidth=1.2,
                        label="C/C0 breakthrough",
                    )
                    sensitivity_ax.set_xlabel(tr(lang, "time_min"))
                    sensitivity_ax.set_ylabel("C/C0")
                    sensitivity_ax.grid(True, alpha=0.25)
                    sensitivity_ax.legend(fontsize=8)
                    st.pyplot(sensitivity_fig)
                    plt.close(sensitivity_fig)
                st.markdown(f"#### {tr(lang, 'results_table')}")
                st.dataframe(
                    make_basic_diagnostics_dataframe(
                        column,
                        diagnostics,
                        result["breakthrough"],
                        result,
                        tr(lang, curve_display_mode),
                    ),
                    width="stretch",
                    hide_index=True,
                )
                if scale_mode != SCALE_LABELS[0]:
                    st.markdown(f"#### {tr(lang, 'engineering_diagnostics')}")
                    hydro_df = make_engineering_diagnostics_dataframe(diagnostics, lang)
                    st.dataframe(hydro_df, width="stretch", hide_index=True)
                    with st.expander(tr(lang, "engineering_indicators_help")):
                        st.markdown(
                            tr(lang, "help_engineering_diagnostics_body")
                        )
                curve_df = pd.DataFrame({"time_min": result["time_min"], "C_out_mg_L": result["C_out_mg_L"], "C_out_C0": result["C_out_C0"]})
                if check_mass_balance:
                    render_mass_balance_block(result["time_min"], column, sorption, int(n_cells), "simulation", lang)
                st.download_button(tr(lang, "download_csv"), data=curve_df.to_csv(index=False).encode("utf-8-sig"), file_name="simulated_breakthrough_curve.csv", mime="text/csv", key="sim_download_csv")
                st.download_button(tr(lang, "download_png"), data=figure_to_png_bytes(fig), file_name="simulated_breakthrough_curve.png", mime="image/png", key="sim_download_png")
                plt.close(fig)
            else:
                st.caption(tr(lang, "set_parameters_and_run"))

    else:
        st.markdown(f"#### {tr(lang, 'ml_generation')}")
        validation_report = st.session_state.get("manual_holdout_validation_report") or st.session_state.get("experiment_validation_report")
        st.markdown(f"##### {tr(lang, 'experiment_validation_state')}")
        if isinstance(validation_report, dict):
            status_value = validation_report.get("validation_status", "unknown")
            metrics_validation = validation_report.get("metrics_validation") or {}
            state_df = pd.DataFrame(
                [
                    {
                        "experiment_id": validation_report.get("experiment_id"),
                        "validation_mode": validation_report.get("validation_mode"),
                        "validation_status": status_value,
                        "RMSE_validation": metrics_validation.get("RMSE"),
                        "nRMSE_percent": metrics_validation.get("nRMSE_percent"),
                        "R2_Q2": metrics_validation.get("R2_Q2"),
                        "curve_similarity_percent": validation_report.get("curve_similarity_percent"),
                        "hidden_similarity_percent": validation_report.get("hidden_similarity_percent"),
                        "created_at": validation_report.get("created_at"),
                    }
                ]
            )
            st.dataframe(state_df, width="stretch", hide_index=True)
            if status_value == "poor":
                st.warning(tr(lang, "experiment_validation_poor_generation_warning"))
            elif status_value == "good":
                st.success(tr(lang, "experiment_validation_good_for_generation"))
            else:
                st.info(tr(lang, "experiment_validation_acceptable_for_generation"))
        else:
            st.warning(tr(lang, "experiment_validation_recommended_before_generation"))
            st.checkbox(
                tr(lang, "allow_generation_without_holdout_validation"),
                value=True,
                key="allow_generation_without_holdout_validation",
            )
        ranges = dataset_ranges_for_scale(scale_mode)
        regen_request = st.session_state.pop("ml_auto_regenerate_request", None)
        auto_generate_now = bool(st.session_state.pop("ml_auto_generate_now", False))
        if isinstance(regen_request, dict):
            st.session_state["unified_ml_generation_type"] = "balanced_breakthrough_dataset"
            st.session_state["unified_ml_cases"] = int(regen_request.get("target_valid_cases", 1000))
            if scale_mode == SCALE_LABELS[0]:
                st.session_state["ml_C0_min"] = max(float(st.session_state.get("ml_C0_min", 2.0)), 2.0)
            st.info(tr(lang, "auto_regeneration_settings_applied"))
        ml_col_1, ml_col_2, ml_col_3 = st.columns(3)
        with ml_col_1:
            target_valid_cases = st.number_input(
                tr(lang, "target_valid_breakthrough_cases"),
                min_value=100,
                max_value=5000,
                value=1000,
                step=100,
                key="unified_ml_cases",
                help=tr(lang, "ml_recommended_cases_help"),
            )
            if int(target_valid_cases) < 500:
                st.warning(tr(lang, "ml_fewer_than_demo_cases_warning"))
            st.caption(tr(lang, "valid_breakthrough_cases_explanation"))
        with ml_col_2:
            ml_generation_type = st.selectbox(
                tr(lang, "candidate_generation_type"),
                ["balanced_breakthrough_dataset", "engineering_realistic", "latin_hypercube_continuous", "random_continuous"],
                index=0,
                format_func=lambda value: tr(lang, f"dataset_generation_{value}"),
                key="unified_ml_generation_type",
            )
            if ml_generation_type not in {"balanced_breakthrough_dataset", "engineering_realistic"}:
                st.warning(tr(lang, "candidate_generation_continuous_warning"))
        with ml_col_3:
            random_seed = st.number_input("random_seed", min_value=0, value=42, step=1, key="unified_ml_seed")
            target_selection_time_h = st.number_input(
                tr(lang, "target_selection_time_h"),
                min_value=0.1,
                value=2.0 if scale_mode == SCALE_LABELS[0] else 24.0,
                step=0.5,
                key="unified_ml_target_selection_time_h",
                help=tr(lang, "target_selection_time_help"),
            )

        st.markdown(f"##### {tr(lang, 'main_ranges')}")
        if scale_mode == SCALE_LABELS[0]:
            C0_range = range_input(tr(lang, "c0_mg_l"), "ml_C0_min", "ml_C0_max", 2.0, 10.0, min_value=0.001, max_value=100.0, format="%.4f")
            if float(C0_range[0]) < 1.0:
                st.warning(tr(lang, "ml_low_c0_breakthrough_warning"))
            cbr_range = (DEFAULT_BREAKTHROUGH_RATIO, DEFAULT_BREAKTHROUGH_RATIO)
            st.metric("C/C0", f"{DEFAULT_BREAKTHROUGH_RATIO:.2f}")
            flow_range = range_input(tr(lang, "flow_ml_min"), "ml_Q_lab_min", "ml_Q_lab_max", *ranges["flow_ml_min"], min_value=1e-6, format="%.4f")
            height_range = range_input(tr(lang, "bed_height_cm"), "ml_H_lab_min", "ml_H_lab_max", *ranges["bed_height_cm"], min_value=0.001, format="%.4f")
            diameter_range = range_input(tr(lang, "column_diameter_cm"), "ml_D_lab_min", "ml_D_lab_max", *ranges["column_diameter_cm"], min_value=0.001, format="%.4f")
            temp_range = range_input("T, °C", "ml_T_min", "ml_T_max", *ranges["temperature_C"], min_value=0.0, max_value=100.0, format="%.2f")
        else:
            industrial_ranges = RECOMMENDED_DATASET_RANGES["industrial_scaleup_dataset"]
            C0_range = range_input(tr(lang, "c0_mg_l"), "ml_C0_si_min", "ml_C0_si_max", 0.1, 10.0, min_value=0.001, max_value=100.0, format="%.4f")
            cbr_range = (DEFAULT_BREAKTHROUGH_RATIO, DEFAULT_BREAKTHROUGH_RATIO)
            st.metric("C/C0", f"{DEFAULT_BREAKTHROUGH_RATIO:.2f}")
            flow_range = range_input(tr(lang, "flow_m3_h"), "ml_Q_si_min", "ml_Q_si_max", *industrial_ranges["Q_m3_h"], min_value=1e-9, format="%.6f")
            height_range = range_input(tr(lang, "bed_height_m"), "ml_H_si_min", "ml_H_si_max", *industrial_ranges["H_m"], min_value=0.001, format="%.4f")
            diameter_range = range_input(tr(lang, "column_diameter_m"), "ml_D_si_min", "ml_D_si_max", *industrial_ranges["D_col_m"], min_value=0.001, format="%.4f")
            temp_range = range_input("T, °C", "ml_T_si_min", "ml_T_si_max", *industrial_ranges["T_C"], min_value=0.0, max_value=100.0, format="%.2f")

        with st.expander(tr(lang, "advanced_ranges"), expanded=False):
            profile_sorption_defaults = resolve_sorption_parameters(
                profile_key,
                pollutant,
                "adsorbent_profile",
                fitted_df=fitted_temperature_dataframe(),
                last_calibration=None,
            )
            fixed_profile_values = st.checkbox(
                tr(lang, "fixed_profile_values"),
                value=False,
                key="unified_ml_fixed_profile_values",
            )
            kh_default_range = (
                profile_sorption_defaults.get("KH_L_g"),
                profile_sorption_defaults.get("KH_L_g"),
            ) if fixed_profile_values else (
                profile_sorption_defaults.get("KH_range_min_L_g") or ranges.get("KH_L_g", (0.005, 0.10))[0],
                profile_sorption_defaults.get("KH_range_max_L_g") or ranges.get("KH_L_g", (0.005, 0.10))[1],
            )
            qmax_default_range = (
                profile_sorption_defaults.get("qmax_mg_g"),
                profile_sorption_defaults.get("qmax_mg_g"),
            ) if fixed_profile_values else (
                profile_sorption_defaults.get("qmax_range_min_mg_g") or 80.0,
                profile_sorption_defaults.get("qmax_range_max_mg_g") or 350.0,
            )
            b_default = float(profile_sorption_defaults.get("b_L_mg") or 0.00025)
            b_default_range = (b_default, b_default) if fixed_profile_values else (0.5 * b_default, 1.5 * b_default)
            deltaH_default = float(profile_sorption_defaults.get("deltaH_ads_kJ_mol") or 20.0)
            Ea_default = float(profile_sorption_defaults.get("Ea_kLDF_kJ_mol") or 20.0)
            profile_raw = SORPTION_PARAMETER_PROFILES.get(profile_key, {})
            deltaH_profile_range = profile_raw.get("deltaH_range_kJ_mol", (-20.0, 35.0))
            Ea_profile_range = profile_raw.get("Ea_kLDF_range_kJ_mol", (10.0, 35.0))
            deltaH_default_range = (deltaH_default, deltaH_default) if fixed_profile_values else deltaH_profile_range
            Ea_default_range = (Ea_default, Ea_default) if fixed_profile_values else Ea_profile_range
            if scale_mode == SCALE_LABELS[0]:
                density_range = range_input("Плотность загрузки, г/см³", "ml_rho_lab_min", "ml_rho_lab_max", *ranges["bulk_density_g_cm3"], min_value=0.001, format="%.4f")
                dp_range = range_input("dp, мм", "ml_dp_lab_min", "ml_dp_lab_max", 0.8, 2.0, min_value=0.001, format="%.4f")
                epsilon_range = range_input("epsilon", "ml_eps_min", "ml_eps_max", *ranges["epsilon"], min_value=0.05, max_value=0.95, format="%.4f")
                KH_range = range_input(tr(lang, "kh_label"), "ml_KH_min", "ml_KH_max", *kh_default_range, min_value=1e-9, format="%.8f")
                kLDF_range = range_input(tr(lang, "kldf_label"), "ml_kLDF_min", "ml_kLDF_max", *ranges["kLDF_1_min"], min_value=1e-9, format="%.8f")
                Dax_range = range_input(tr(lang, "dax_label"), "ml_Dax_min", "ml_Dax_max", *ranges["Dax_cm2_min"], min_value=1e-12, format="%.10f")
            else:
                industrial_ranges = RECOMMENDED_DATASET_RANGES["industrial_scaleup_dataset"]
                density_range = range_input("Плотность загрузки, кг/м³", "ml_rho_si_min", "ml_rho_si_max", *industrial_ranges["bulk_density_kg_m3"], min_value=1.0, format="%.2f")
                dp_range = range_input("dp, мм", "ml_dp_si_min", "ml_dp_si_max", *industrial_ranges["dp_mm"], min_value=0.001, format="%.4f")
                epsilon_range = range_input("epsilon", "ml_eps_si_min", "ml_eps_si_max", *industrial_ranges["epsilon"], min_value=0.05, max_value=0.95, format="%.4f")
                KH_range = range_input(tr(lang, "kh_label"), "ml_KH_si_min", "ml_KH_si_max", *kh_default_range, min_value=1e-9, format="%.8f")
                kLDF_range = range_input(tr(lang, "kldf_label"), "ml_kLDF_si_min", "ml_kLDF_si_max", *industrial_ranges["kLDF_ref_1_min"], min_value=1e-9, format="%.8f")
                Dax_range = range_input(tr(lang, "dax_label"), "ml_Dax_si_min", "ml_Dax_si_max", *industrial_ranges["Dax_ref_cm2_min"], min_value=1e-12, format="%.10f")
            qmax_range = range_input("qmax, мг/г", "ml_qmax_min", "ml_qmax_max", *qmax_default_range, min_value=1e-9, format="%.4f")
            b_range = range_input("b, л/мг", "ml_b_min", "ml_b_max", *b_default_range, min_value=1e-12, format="%.10f")
            ml_auto_temperature_parameters = resolve_temperature_correction_parameters(profile_key, pollutant, fitted_temperature_dataframe())
            ml_Dm_ref_default = float(ml_auto_temperature_parameters["D_m_ref_m2_s"])
            ml_T_ref_default = temperature_reference_from_params(CHEMVIRON_START_PARAMS, 20.0)
            deltaH_range = range_input("deltaH_ads, кДж/моль", "ml_dH_min", "ml_dH_max", *deltaH_default_range, format="%.2f")
            Ea_range = range_input("Ea_kLDF, кДж/моль", "ml_Ea_min", "ml_Ea_max", *Ea_default_range, min_value=0.0, format="%.2f")
            time_step = st.number_input("Шаг времени, мин", min_value=0.1, value=5.0 if scale_mode == SCALE_LABELS[0] else 60.0, key="unified_ml_time_step")
            ml_n_cells = int((st.session_state.get("last_calibration") or {}).get("n_cells", DEFAULT_N_CELLS))
            st.metric(tr(lang, "model_cells"), ml_n_cells)
            st.caption(tr(lang, "scientific_grid_policy"))
            check_ml_mass_balance = st.checkbox(
                tr(lang, "mass_balance_first_cases"),
                value=False,
                key="unified_ml_check_mass_balance",
            )
            ml_mass_balance_n = st.number_input(
                "N случаев для проверки баланса",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="unified_ml_mass_balance_n",
                disabled=not check_ml_mass_balance,
            )

        ml_ranges = {"C0_mg_L": C0_range, "C_breakthrough_ratio": cbr_range, "flow": flow_range, "height": height_range, "diameter": diameter_range, "temperature_C": temp_range, "bulk_density": density_range, "dp_mm": dp_range, "epsilon": epsilon_range, "KH_L_g": KH_range, "kLDF_1_min": kLDF_range, "Dax_cm2_min": Dax_range, "qmax_mg_g": qmax_range, "b_L_mg": b_range, "deltaH_ads_kJ_mol": deltaH_range, "Ea_kLDF_kJ_mol": Ea_range}
        range_errors = validate_ranges(ml_ranges)
        for error in range_errors:
            st.error(error)

        generate_ml = st.button(tr(lang, "generate_dataset"), type="primary", key="unified_ml_generate")
        if (generate_ml or auto_generate_now) and not range_errors:
            dimension_names = list(ml_ranges.keys())
            sampling_method = "Latin Hypercube Sampling" if ml_generation_type == "latin_hypercube_continuous" else "random uniform"
            balanced_generation = ml_generation_type == "balanced_breakthrough_dataset"
            target_valid_count = int(target_valid_cases)
            max_attempts = target_valid_count * (10 if balanced_generation else 1)
            samples = scale_samples(unit_samples(sampling_method, int(max_attempts), len(dimension_names), int(random_seed)), [ml_ranges[name] for name in dimension_names])
            if ml_generation_type in {"balanced_breakthrough_dataset", "engineering_realistic"}:
                engineering_geometry = make_engineering_candidate_adsorbers(
                    int(max_attempts),
                    scale_mode,
                    {
                        "flow": flow_range,
                        "height": height_range,
                        "diameter": diameter_range,
                        "temperature": temp_range,
                        "epsilon": epsilon_range,
                        "bulk_density": density_range,
                        "dp_mm": dp_range,
                    },
                    random_seed=int(random_seed),
                    method="engineering_grid_random",
                )
                for geom_name in ["flow", "height", "diameter", "temperature", "bulk_density", "dp_mm", "epsilon"]:
                    if geom_name in dimension_names:
                        samples[:, dimension_names.index(geom_name)] = engineering_geometry[geom_name].to_numpy(dtype=float)
                if "temperature_C" in dimension_names:
                    samples[:, dimension_names.index("temperature_C")] = engineering_geometry["temperature"].to_numpy(dtype=float)
            rows: list[dict] = []
            mass_balance_summaries: list[dict] = []
            rejected_case_summaries: list[dict] = []
            valid_breakthrough_cases = 0
            accepted_case_id = 0
            bin_counts = {"short": 0, "medium": 0, "long": 0}
            bin_quotas = {
                "short": int(round(0.25 * target_valid_count)),
                "medium": int(round(0.50 * target_valid_count)),
            }
            bin_quotas["long"] = max(target_valid_count - bin_quotas["short"] - bin_quotas["medium"], 0)
            target_time_min = float(target_selection_time_h) * 60.0
            if scale_mode == SCALE_LABELS[0]:
                balanced_bin_edges = (30.0, 120.0, max(300.0, 1.5 * target_time_min))
                t_end_limit_min = max(300.0, 1.5 * target_time_min)
            else:
                balanced_bin_edges = (0.5 * target_time_min, target_time_min, 2.0 * target_time_min)
                t_end_limit_min = max(1.5 * target_time_min, 24.0 * 60.0)

            def breakthrough_bin(time_min: float) -> str:
                if time_min < balanced_bin_edges[0]:
                    return "short"
                if time_min < balanced_bin_edges[1]:
                    return "medium"
                return "long"

            progress = st.progress(0.0)
            for attempt_index, values in enumerate(samples, start=1):
                if balanced_generation and valid_breakthrough_cases >= target_valid_count:
                    break
                p = dict(zip(dimension_names, values))
                try:
                    if scale_mode == SCALE_LABELS[0]:
                        ml_area_cm2 = column_area_from_diameter(float(p["diameter"]))
                        ml_superficial_velocity_m_s = (
                            float(p["flow"]) / max(ml_area_cm2, 1e-30)
                        ) * 0.01 / 60.0
                        ml_Dax_temperature_mode = "proportional_to_diffusion"
                        ml_Dax_mode = "температурная поправка к D_ax_ref"
                    else:
                        ml_area_m2 = math.pi * float(p["diameter"]) ** 2 / 4.0
                        ml_superficial_velocity_m_s = (
                            float(p["flow"]) / 3600.0
                        ) / max(ml_area_m2, 1e-30)
                        ml_Dax_temperature_mode = "engineering_estimate"
                        ml_Dax_mode = "инженерная оценка через скорость и размер частиц"
                    ml_temperature_adaptation = apply_temperature_adaptation(
                        KH_ref_L_g=float(p["KH_L_g"]),
                        kLDF_ref_1_min=float(p["kLDF_1_min"]),
                        Dax_ref_cm2_min=float(p["Dax_cm2_min"]),
                        T_ref_C=float(ml_T_ref_default),
                        T_sim_C=float(p["temperature_C"]),
                        deltaH_ads_kJ_mol=float(p["deltaH_ads_kJ_mol"]),
                        Ea_kLDF_kJ_mol=float(p["Ea_kLDF_kJ_mol"]),
                        D_m_ref_m2_s=float(ml_Dm_ref_default),
                        Dax_temperature_mode=ml_Dax_temperature_mode,
                        source=str(ml_auto_temperature_parameters["source"]),
                        confidence=str(ml_auto_temperature_parameters["confidence"]),
                        superficial_velocity_m_s=float(ml_superficial_velocity_m_s),
                        dp_m=float(p["dp_mm"]) / 1000.0,
                    )
                    sampled_params = sampled_parameter_payload(p, ml_ranges, profile_sorption_defaults, sampling_method, int(random_seed))
                    sampled_params["T_ref_C"] = float(ml_T_ref_default)
                    sampled_params["T_sim_C"] = float(p["temperature_C"])
                    sampled_params["D_m_ref_m2_s"] = float(ml_Dm_ref_default)
                    sampled_params["engineering_estimate_available"] = True
                    sampled_params["parameter_provenance"].update({
                        "T_ref_C": provenance_record("default", method="preset_default", assignment_path="temperature_reference_from_params(CHEMVIRON_START_PARAMS, 20.0)"),
                        "T_sim_C": provenance_record("derived", method="range_sampling", configured_range=list(temp_range), dependencies=["configured_range.temperature_C"]),
                        "D_m_ref_m2_s": parameter_origin(ml_auto_temperature_parameters, "D_m_ref_m2_s"),
                        **geometry_records(scale_mode, "derived"),
                    })
                    ml_temperature_adaptation = temperature_lineage(ml_temperature_adaptation, sampled_params)
                    column, sorption, diagnostics, validation_messages = build_column_and_diagnostics(
                        scale_mode, float(p["C0_mg_L"]), float(p["flow"]), float(p["height"]), float(p["diameter"]), float(p["epsilon"]), float(p["bulk_density"]), float(p["temperature_C"]), float(p["dp_mm"]), 0.85, float(p["KH_L_g"]), float(p["kLDF_1_min"]), float(p["Dax_cm2_min"]), float(ml_T_ref_default), float(p["deltaH_ads_kJ_mol"]), float(p["Ea_kLDF_kJ_mol"]), float(ml_Dm_ref_default), ml_Dax_mode, "температурная поправка Arrhenius",
                        temperature_adaptation=ml_temperature_adaptation,
                    )
                    br = float(p["C_breakthrough_ratio"])
                    if balanced_generation:
                        prelim = simulate_for_time(t_end_limit_min, 160, column, sorption, int(ml_n_cells), br)
                    else:
                        prelim = simulate_with_time_mode("until_breakthrough_plus_20", scale_mode, 60.0, 160, column, sorption, int(ml_n_cells), br)
                    t_end_actual = float(prelim["t_end_actual_min"])
                    times = np.arange(0.0, t_end_actual + 0.5 * float(time_step), float(time_step))
                    if len(times) < 3:
                        times = np.linspace(0.0, t_end_actual, 3)
                    concentrations = simulate_breakthrough(times, column, sorption, n_cells=int(ml_n_cells))
                    ratios = concentrations / float(p["C0_mg_L"])
                    bt = breakthrough_status(times, ratios, br)
                    breakthrough_found = bt["status"] != "not_reached"
                    breakthrough_time_min = None if bt.get("time_min") is None else float(bt.get("time_min"))
                    simulation_status = "breakthrough_reached" if breakthrough_found else "no_breakthrough_within_t_end"
                    if balanced_generation:
                        accept_case = False
                        if breakthrough_found and breakthrough_time_min is not None:
                            tb_bin = breakthrough_bin(float(breakthrough_time_min))
                            if bin_counts[tb_bin] < bin_quotas[tb_bin] or attempt_index > target_valid_count * 3:
                                accept_case = True
                                bin_counts[tb_bin] += 1
                        if not accept_case:
                            rejected_case_summaries.append(
                                {
                                    "attempt_id": attempt_index,
                       "breakthrough_found": bool(breakthrough_found),
                        "n_cells": int(ml_n_cells),
                        "mechanistic_model_id": MODEL_VERSION,
                                    "breakthrough_time_min": breakthrough_time_min,
                                    "simulation_status": simulation_status,
                                }
                            )
                            progress.progress(min(valid_breakthrough_cases / max(target_valid_count, 1), 0.999))
                            continue
                    accepted_case_id += 1
                    if breakthrough_found:
                        valid_breakthrough_cases += 1
                    t_min_lab, t_max_lab = laboratory_temperature_range()
                    temperature_extrapolation_flag = False
                    if t_min_lab is not None and t_max_lab is not None:
                        temperature_extrapolation_flag = (
                            float(p["temperature_C"]) < float(t_min_lab)
                            or float(p["temperature_C"]) > float(t_max_lab)
                        )
                    if check_ml_mass_balance and accepted_case_id <= int(ml_mass_balance_n):
                        full = simulate_breakthrough_full(times, column, sorption, n_cells=int(ml_n_cells))
                        balance_df = calculate_mass_balance(
                            full["time_min"],
                            full["C_profile_mg_L"],
                            full["q_profile_mg_g"],
                            full["C_out_mg_L"],
                            column,
                            sorption,
                        )
                        mass_balance_summaries.append(
                            {"case_id": accepted_case_id, **summarize_mass_balance(balance_df)}
                        )
                    case_meta = {
                        "n_cells": int(ml_n_cells),
                        "mechanistic_model_id": MODEL_VERSION,
                        "scale_mode": scale_mode,
                        "candidate_generation_type": ml_generation_type,
                        "adsorbent_profile_key": profile_key,
                        "adsorbent_profile_name": profile_label(profile_key, lang),
                        "sorbent_profile": profile_label(profile_key, lang),
                        "pollutant": pollutant,
                        "parameter_source": profile_sorption_defaults.get("parameter_source"),
                        "parameter_confidence": profile_sorption_defaults.get("parameter_confidence"),
                        "isotherm_model": profile_sorption_defaults.get("isotherm_model"),
                        "C0_mg_L": float(p["C0_mg_L"]),
                        "C_breakthrough_ratio": br,
                        "C_breakthrough_mg_L": br * float(p["C0_mg_L"]),
                        "Q": float(p["flow"]),
                        "H": float(p["height"]),
                        "D": float(p["diameter"]),
                        "T": float(p["temperature_C"]),
                        "bed_volume": diagnostics.get("bed_volume_m3"),
                        "carbon_mass": diagnostics.get("carbon_mass_kg"),
                        "EBCT_min": diagnostics.get("EBCT_min"),
                        "superficial_velocity": diagnostics.get(
                            "superficial_velocity_m_h",
                            diagnostics.get("superficial_velocity_cm_min"),
                        ),
                        "pressure_drop_bar": diagnostics.get("pressure_drop_bar"),
                        "K_H_used": float(sorption.KH_L_g),
                        "kLDF_used": float(sorption.kLDF_1_min),
                        "Dax_used": float(sorption.Dax_cm2_min),
                        "KH_L_g": float(p["KH_L_g"]),
                        "KH_range_min_L_g": kh_default_range[0],
                        "KH_range_max_L_g": kh_default_range[1],
                        "qmax_mg_g": float(p["qmax_mg_g"]),
                        "qmax_range_min_mg_g": qmax_default_range[0],
                        "qmax_range_max_mg_g": qmax_default_range[1],
                        "b_L_mg": float(p["b_L_mg"]),
                        "T_ref_C": ml_temperature_adaptation.get("T_ref_C"),
                        "T_sim_C": ml_temperature_adaptation.get("T_sim_C"),
                        "temperature_correction_mode": ml_temperature_adaptation.get("temperature_correction_mode"),
                        "temperature_correction_source": ml_temperature_adaptation.get("temperature_correction_source"),
                        "temperature_correction_confidence": ml_temperature_adaptation.get("temperature_correction_confidence"),
                        "deltaH_ads_kJ_mol": ml_temperature_adaptation.get("deltaH_ads_kJ_mol"),
                        "Ea_kLDF_kJ_mol": ml_temperature_adaptation.get("Ea_kLDF_kJ_mol"),
                        "KH_ref_L_g": ml_temperature_adaptation.get("KH_ref_L_g"),
                        "KH_used_L_g": ml_temperature_adaptation.get("KH_used_L_g"),
                        "KH_temperature_factor": ml_temperature_adaptation.get("KH_temperature_factor"),
                        "kLDF_ref_1_min": ml_temperature_adaptation.get("kLDF_ref_1_min"),
                        "kLDF_used_1_min": ml_temperature_adaptation.get("kLDF_used_1_min"),
                        "kLDF_temperature_factor": ml_temperature_adaptation.get("kLDF_temperature_factor"),
                        "Dax_ref_cm2_min": ml_temperature_adaptation.get("Dax_ref_cm2_min"),
                        "Dax_used_cm2_min": ml_temperature_adaptation.get("Dax_used_cm2_min"),
                        "Dax_temperature_factor": ml_temperature_adaptation.get("Dax_temperature_factor"),
                        "temperature_extrapolation_flag": bool(temperature_extrapolation_flag),
                        "breakthrough_time_min": breakthrough_time_min,
                        "breakthrough_found": bool(breakthrough_found),
                        "t_end_actual_min": t_end_actual,
                        "simulation_status": simulation_status,
                        "warning": "; ".join([*(validation_messages or []), *diagnostics.get("warnings", [])]),
                    }
                    first_curve_row = len(rows)
                    append_curve_rows(rows, accepted_case_id, {"time_min": times, "C_out_mg_L": concentrations, "C_out_C0": ratios}, case_meta)
                    # One lineage capsule per case, not a large copy at every time point.
                    rows[first_curve_row]["parameter_provenance_json"] = json.dumps(ml_temperature_adaptation["parameter_provenance"], ensure_ascii=False, sort_keys=True)
                except Exception as exc:
                    rejected_case_summaries.append({"attempt_id": attempt_index, "simulation_status": "case_skipped", "warning": f"case skipped: {exc}"})
                    if not balanced_generation:
                        rows.append({"case_id": attempt_index, "scale_mode": scale_mode, "sorbent_profile": profile_label(profile_key, lang), "pollutant": pollutant, "simulation_status": "case_skipped", "warning": f"case skipped: {exc}"})
                if balanced_generation:
                    progress.progress(min(valid_breakthrough_cases / max(target_valid_count, 1), 1.0))
                else:
                    progress.progress(attempt_index / int(max_attempts))
            ml_df = pd.DataFrame(rows)
            if not ml_df.empty:
                from scientific_provenance import record_generation_manifest
                generation_dir = Path("generated_datasets")
                generation_dir.mkdir(exist_ok=True)
                generation_name = f"session_{time.time_ns()}"
                dataset_path = generation_dir / f"{generation_name}.csv"
                ml_df.to_csv(dataset_path, index=False, encoding="utf-8-sig")
                ml_df.attrs["generation_provenance"] = record_generation_manifest(
                    ml_df, dataset_path, generation_dir / f"{generation_name}_manifest.json",
                    "streamlit_physical_generator", ranges=ml_ranges,
                    sorption_source="configured_ranges_sampling",
                    sorption_parameter_provenance=dataset_parameter_provenance(ml_df),
                )
            st.session_state["unified_ml_rejected_cases"] = pd.DataFrame(rejected_case_summaries)
            st.session_state["unified_ml_df"] = ml_df
            set_ml_dataset_for_training(
                ml_df,
                filename="generated_in_current_session",
                source="generated_in_current_session",
                dataset_mode="synthetic",
            )
            st.session_state["unified_ml_mass_balance_summary"] = pd.DataFrame(mass_balance_summaries)
            st.success(tr(lang, "rows_generated").format(count=len(ml_df)))
            if balanced_generation:
                st.info(
                    tr(lang, "balanced_generation_summary").format(
                        valid=valid_breakthrough_cases,
                        target=target_valid_count,
                        attempts=attempt_index,
                    )
                )
                if valid_breakthrough_cases < target_valid_count:
                    st.warning(tr(lang, "balanced_generation_incomplete"))
            st.success(tr(lang, "ml_dataset_auto_sent"))

        ml_df = st.session_state.get("unified_ml_df")
        if isinstance(ml_df, pd.DataFrame) and not ml_df.empty:
            st.markdown(f"#### {tr(lang, 'preview_first_rows')}")
            st.dataframe(ml_df.head(20), width="stretch")
            case_cols = ["case_id", "scale_mode", "candidate_generation_type", "adsorbent_profile_key", "adsorbent_profile_name", "sorbent_profile", "pollutant", "parameter_source", "parameter_confidence", "isotherm_model", "C0_mg_L", "C_breakthrough_ratio", "Q", "H", "D", "T", "T_ref_C", "T_sim_C", "temperature_correction_source", "temperature_correction_confidence", "bed_volume", "carbon_mass", "EBCT_min", "superficial_velocity", "pressure_drop_bar", "KH_L_g", "KH_range_min_L_g", "KH_range_max_L_g", "qmax_mg_g", "qmax_range_min_mg_g", "qmax_range_max_mg_g", "b_L_mg", "KH_ref_L_g", "KH_used_L_g", "KH_temperature_factor", "kLDF_ref_1_min", "kLDF_used_1_min", "kLDF_temperature_factor", "Dax_ref_cm2_min", "Dax_used_cm2_min", "Dax_temperature_factor", "temperature_extrapolation_flag", "breakthrough_time_min", "breakthrough_found", "t_end_actual_min", "simulation_status"]
            case_cols = [col for col in case_cols if col in ml_df.columns]
            cases_df = ml_df[case_cols].drop_duplicates("case_id")
            st.markdown(f"#### {tr(lang, 'case_parameters')}")
            st.dataframe(cases_df, width="stretch", height=300)
            numeric_cases = sorted(pd.to_numeric(ml_df["case_id"], errors="coerce").dropna().astype(int).unique())
            if numeric_cases:
                case_id = st.selectbox(tr(lang, "case_id_for_plot"), numeric_cases, key="unified_ml_case_plot")
                case_df = ml_df[ml_df["case_id"] == case_id]
                first_case = case_df.iloc[0]
                bt = {"time_min": None if pd.isna(first_case.get("breakthrough_time_min")) else float(first_case.get("breakthrough_time_min")), "status": "reached_by_interpolation" if bool(first_case.get("breakthrough_found")) else "not_reached", "label": ""}
                fig = plot_simulation_curve(case_df["time_min"].to_numpy(dtype=float), case_df["C_out_C0"].to_numpy(dtype=float), float(first_case["C_breakthrough_ratio"]), bt, str(first_case.get("scale_mode", scale_mode)), title=f"ML case_id = {case_id}", ebct_min=None if pd.isna(first_case.get("EBCT_min")) else float(first_case.get("EBCT_min")), lang=lang)
                st.pyplot(fig)
                plt.close(fig)
            mb_summary_df = st.session_state.get("unified_ml_mass_balance_summary")
            if isinstance(mb_summary_df, pd.DataFrame) and not mb_summary_df.empty:
                st.markdown(f"#### {tr(lang, 'mass_balance_check')}")
                st.dataframe(mb_summary_df, width="stretch", hide_index=True)
            download_button_persistent(
                tr(lang, "download_ml_dataset"),
                data=ml_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="synthetic_dynamic_sorption_dataset.csv",
                mime="text/csv",
                key="unified_ml_download",
            )


_startup_probe.mark("ml_begin")
with tab_ml_module:
    st.subheader(tr(lang, "ml_module_title"))
    st.caption(tr(lang, "ui_ml_workflow"))
    st.caption(tr(lang, "ui_metric_help"))
    st.warning(tr(lang, "joblib_security_warning"))
    data_block, train_block, load_block, select_block = st.tabs(
        [
            tr(lang, "ml_load_data"),
            tr(lang, "ml_train_models"),
            tr(lang, "ml_load_saved_model"),
            tr(lang, "ml_adsorber_selection"),
        ]
    )

    with data_block:
        st.markdown(f"#### {tr(lang, 'ml_load_data')}")
        current_session_df = st.session_state.get("unified_ml_df")
        if not isinstance(current_session_df, pd.DataFrame) or current_session_df.empty:
            current_session_df = st.session_state.get("ml_dataset")
        if isinstance(current_session_df, pd.DataFrame) and not current_session_df.empty:
            st.success(tr(lang, "ml_current_session_dataset_ready"))
            if st.button(tr(lang, "ml_use_current_session_dataset"), key="ml_use_current_session_dataset"):
                set_ml_dataset_for_training(
                    current_session_df,
                    filename="generated_in_current_session",
                    source="generated_in_current_session",
                    dataset_mode="synthetic",
                )
                st.success(tr(lang, "ml_current_session_dataset_selected"))
            if "ml_dataset_df" not in st.session_state and tab_ml_module.open:
                set_ml_dataset_for_training(
                    current_session_df,
                    filename="generated_in_current_session",
                    source="generated_in_current_session",
                    dataset_mode="synthetic",
                )
        dataset_mode = st.selectbox(
            tr(lang, "ml_dataset_mode"),
            ["synthetic", "experimental"],
            format_func=lambda value: tr(lang, f"ml_dataset_{value}"),
            key="ml_dataset_mode",
        )
        source_options = ["current_session", "existing_synthetic", "upload_csv", "upload_xlsx"]
        default_source_index = 0 if isinstance(current_session_df, pd.DataFrame) and not current_session_df.empty else 1
        source_mode = st.radio(
            tr(lang, "ml_data_source"),
            source_options,
            index=default_source_index,
            format_func=lambda value: tr(lang, f"ml_source_{value}"),
            horizontal=True,
            key="ml_data_source",
        )
        ml_df = None
        dataset_filename = "synthetic_dynamic_sorption_dataset.csv"
        if source_mode == "current_session":
            if isinstance(current_session_df, pd.DataFrame) and not current_session_df.empty:
                if tab_ml_module.open:
                    ml_df = current_session_df
                dataset_filename = "generated_in_current_session"
                st.success(tr(lang, "ml_current_session_dataset_found"))
            else:
                st.warning(tr(lang, "ml_no_current_session_dataset"))
        elif source_mode == "existing_synthetic":
            if os.path.exists(dataset_filename):
                if tab_ml_module.open:
                    ml_df = load_dataset_file(dataset_filename)
            else:
                st.error(tr(lang, "ml_existing_dataset_missing"))
        else:
            uploaded_dataset = st.file_uploader(
                tr(lang, "ml_upload_dataset"),
                type=["csv"] if source_mode == "upload_csv" else ["xlsx", "xls"],
                key="ml_dataset_upload",
            )
            if uploaded_dataset is not None and tab_ml_module.open:
                dataset_filename = uploaded_dataset.name
                ml_df = load_ml_dataset(uploaded_dataset, "xlsx" if source_mode == "upload_xlsx" else "csv")
        if ml_df is not None:
            validation = set_ml_dataset_for_training(
                ml_df,
                filename=dataset_filename,
                source=source_mode,
                dataset_mode=dataset_mode,
            )
            validation_table = pd.DataFrame(
                [
                    {"item": "number_of_rows", "value": str(validation["number_of_rows"])},
                    {"item": "number_of_cases", "value": str(validation["number_of_cases"])},
                    {"item": "optional_columns_found", "value": ", ".join(validation["optional_columns_found"])},
                ]
            )
            dataset_counts(st, validation["number_of_rows"], validation["number_of_cases"], lang)
            with st.expander(tr(lang, "ui_field_optional_columns_found")):
                st.code(", ".join(validation["optional_columns_found"]), language=None)
            if validation["missing_required_columns"]:
                st.error(
                    tr(lang, "ml_missing_required_columns").format(
                        columns=", ".join(validation["missing_required_columns"])
                    )
                )
            for warning in validation["warnings"]:
                st.warning(warning)
            render_dataset_preview(st, validation["canonical_df"], lang)
            required_for_quality_h = st.number_input(
                tr(lang, "required_service_time_h"),
                min_value=0.1,
                value=float(st.session_state.get("ml_quality_required_h", 2.0)),
                key="ml_quality_required_h",
            )
            training_mode = st.selectbox(
                tr(lang, "ml_training_mode"),
                ["validated", "demo"],
                index=0,
                format_func=lambda value: tr(lang, f"ml_training_mode_{value}"),
                key="ml_training_mode",
            )
            breakthrough_quality = cached_breakthrough_quality(
                validation["canonical_df"],
                required_service_time_h=float(required_for_quality_h),
                training_mode=training_mode,
            )
            st.session_state["breakthrough_dataset_validation"] = breakthrough_quality
            st.markdown(f"#### {tr(lang, 'ml_dataset_quality_check')}")
            report = breakthrough_quality.get("report", {})
            target_stats = report.get("target_stats", {}) if isinstance(report, dict) else {}
            render_quality_overview(st, report, lang)
            with st.expander(tr(lang, "ui_details")):
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "number_of_rows": report.get("number_of_rows"),
                                "number_of_cases": report.get("number_of_cases"),
                                "number_of_breakthrough_found_cases": report.get("number_of_breakthrough_found_cases"),
                                "number_of_not_found_cases": report.get("number_of_not_found_cases"),
                                "share_breakthrough_found": report.get("share_breakthrough_found"),
                                "target_min": target_stats.get("min"),
                                "target_median": target_stats.get("median"),
                                "target_p90": target_stats.get("p90"),
                                "target_p95": target_stats.get("p95"),
                                "target_max": target_stats.get("max"),
                                "target_max_h": (target_stats.get("max") / 60.0) if target_stats.get("max") is not None and pd.notna(target_stats.get("max")) else None,
                                "dataset_quality_mode": report.get("dataset_quality_mode"),
                                "dataset_ready": breakthrough_quality.get("dataset_ready"),
                            }
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            st.caption(tr(lang, "ui_cases_help"))
            st.info(tr(lang, "ui_structure_vs_coverage"))
            if breakthrough_quality.get("dataset_ready"):
                st.success(tr(lang, "ml_dataset_ready_for_training"))
            else:
                st.error(tr(lang, "ml_dataset_not_enough_for_training"))
            if report.get("breakthrough_found_reconstructed"):
                st.info(tr(lang, "ml_breakthrough_found_reconstructed"))
            for message in breakthrough_quality.get("errors", []):
                st.error(message)
            for message in breakthrough_quality.get("warnings", []):
                st.warning(message)
            if breakthrough_quality.get("recommended_action"):
                st.info("; ".join(map(str, breakthrough_quality.get("recommended_action", []))))
            if "too_few_valid_breakthrough_cases" in set(map(str, breakthrough_quality.get("errors", []))):
                if st.button(tr(lang, "auto_generate_valid_cases"), key="ml_auto_generate_valid_cases"):
                    min_required = int((report or {}).get("minimum_valid_breakthrough_cases") or 1000)
                    st.session_state["ml_auto_regenerate_request"] = {
                        "target_valid_cases": max(min_required, 1000),
                    }
                    st.session_state["ml_auto_generate_now"] = True
                    st.rerun()

    with train_block:
        st.markdown(f"#### {tr(lang, 'ml_train_models')}")
        render_active_ml_model_card(lang)
        render_sync_active_breakthrough_button(lang, "train_sync_active_breakthrough")
        loaded_df = st.session_state.get("ml_dataset_df")
        if not isinstance(loaded_df, pd.DataFrame) or loaded_df.empty:
            st.info(tr(lang, "ml_load_dataset_first"))
        elif not sklearn_available():
            st.error(tr(lang, "ml_sklearn_missing"))
        else:
            st.info(tr(lang, "ml_what_to_choose_body"))
            model_choice = st.radio(
                tr(lang, "ml_model_choice"),
                ["both", "breakthrough", "curve"],
                index=1,
                format_func=lambda value: tr(lang, f"ml_choice_{value}"),
                key="ml_model_choice",
            )
            if model_choice == "both":
                st.success(tr(lang, "ml_both_choice_info"))
                st.warning(tr(lang, "ml_curve_heavy_warning"))
            elif model_choice == "breakthrough":
                st.success(tr(lang, "ml_breakthrough_choice_info"))
            else:
                st.info(tr(lang, "ml_curve_choice_info"))
            breakthrough_dataset_validation = st.session_state.get("breakthrough_dataset_validation")
            training_mode = st.session_state.get("ml_training_mode", "validated")
            breakthrough_training_blocked = False
            if model_choice in {"both", "breakthrough"}:
                breakthrough_dataset_validation = cached_breakthrough_quality(
                    loaded_df,
                    training_mode=str(training_mode),
                )
                st.session_state["breakthrough_dataset_validation"] = breakthrough_dataset_validation
                if not breakthrough_dataset_validation.get("dataset_ready"):
                    breakthrough_training_blocked = True
                    st.error(tr(lang, "ml_dataset_not_enough_for_training"))
                    for message in breakthrough_dataset_validation.get("errors", []):
                        st.error(message)
                    if breakthrough_dataset_validation.get("recommended_action"):
                        st.info("; ".join(map(str, breakthrough_dataset_validation.get("recommended_action", []))))
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            tr(lang, "ml_compare_model"): tr(lang, "ml_breakthrough_model_title"),
                            tr(lang, "ml_compare_predicts"): "breakthrough_time_min",
                            tr(lang, "ml_compare_needs_time"): tr(lang, "no"),
                            tr(lang, "ml_compare_use"): tr(lang, "ml_use_adsorber_selection"),
                        },
                        {
                            tr(lang, "ml_compare_model"): tr(lang, "ml_curve_model_title"),
                            tr(lang, "ml_compare_predicts"): "C_out/C0(t)",
                            tr(lang, "ml_compare_needs_time"): tr(lang, "yes"),
                            tr(lang, "ml_compare_use"): tr(lang, "ml_use_curve_analysis"),
                        },
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            use_optuna = st.checkbox(tr(lang, "ml_use_optuna"), value=False, key="ml_use_optuna")
            n_trials = st.number_input("n_trials", min_value=5, max_value=100, value=30, step=5, key="ml_n_trials")
            if model_choice in {"both", "breakthrough"}:
                st.info(tr(lang, "ml_auto_target_transform"))
            if use_optuna:
                st.warning(tr(lang, "ml_optuna_warning"))

            def run_training(task_type: str) -> None:
                with st.spinner(tr(lang, "ml_training_in_progress")):
                    try:
                        result = train_adsorber_models(
                            loaded_df,
                            task_type=task_type,
                            dataset_filename=st.session_state.get("ml_dataset_filename", "dataset"),
                            output_dir="ml_models",
                            use_optuna=use_optuna,
                            n_trials=int(n_trials),
                            target_transform=None if task_type == "breakthrough" else "none",
                            training_mode=str(training_mode),
                            generation_provenance=st.session_state.get("ml_dataset_metadata"),
                        )
                    except Exception as exc:
                        result = {"success": False, "message": tr(lang, "ml_training_failed"), "details": repr(exc)}
                if not result.get("success"):
                    st.error(result.get("message", "training failed"))
                    if result.get("details"):
                        with st.expander(tr(lang, "show_technical_details")):
                            st.code(result["details"], language="text")
                else:
                    model = result["best_model"]
                    metadata = result["metadata"]
                    metrics_df = pd.DataFrame(result["metrics"]).T.reset_index().rename(columns={"index": "model"})
                    metadata_bytes = metadata_to_bytes(metadata)
                    metrics_bytes = metrics_df.to_csv(index=False).encode("utf-8-sig")
                    if task_type == "curve":
                        model_bytes = b""
                        bundle_bytes = b""
                    else:
                        model_bytes = save_joblib_to_bytes(model)
                        bundle_bytes = save_model_bundle_to_bytes(model, metadata)
                    registry_item = set_registry_model(
                        task_type,
                        model,
                        metadata,
                        metrics_df,
                        bundle_bytes=bundle_bytes,
                        model_bytes=model_bytes,
                        metadata_bytes=metadata_bytes,
                        metrics_bytes=metrics_bytes,
                        source="trained_in_current_session",
                    )

                    st.session_state[f"ml_{task_type}_model"] = model
                    st.session_state[f"ml_{task_type}_metadata"] = metadata
                    st.session_state["trained_ml_debug_task_type"] = task_type
                    st.session_state["trained_ml_feature_importance"] = result.get("feature_importance")
                    if task_type == "breakthrough":
                        st.session_state["trained_breakthrough_model"] = model
                        st.session_state["trained_breakthrough_metadata"] = metadata
                        st.session_state["trained_breakthrough_bundle_bytes"] = bundle_bytes
                        if registry_item.get("ready") is True:
                            set_active_breakthrough_model_from_registry()
                    else:
                        st.session_state["trained_curve_model"] = model
                        st.session_state["trained_curve_metadata"] = metadata
                        st.session_state.pop("trained_curve_bundle_bytes", None)
                    st.success(result["message"])
                    if task_type == "breakthrough":
                        if metadata.get("selection_ready") is True:
                            st.success(tr(lang, "ml_model_selected_for_selection"))
                        else:
                            st.error(tr(lang, "ml_model_trained_but_not_active"))
                            st.warning(str(metadata.get("selection_block_reason") or tr(lang, "ml_model_quality_gate_failed")))
                            if st.session_state.get("active_ml_model") is not None:
                                st.info(tr(lang, "ml_active_model_kept_because_new_model_bad"))
                    else:
                        st.info(tr(lang, "ml_curve_trained_not_active"))
                    for warning in result.get("warnings", []):
                        st.warning(warning)
                    st.dataframe(metrics_df, width="stretch", hide_index=True, column_config=model_metrics_column_config(task_type, lang))
                    if task_type == "breakthrough":
                        target_stats = metadata.get("target_stats") or {}
                        found = int(metadata.get("n_breakthrough_found_cases", 0) or 0)
                        not_found = int(metadata.get("n_breakthrough_not_found_cases", 0) or 0)
                        total_cases = max(found + not_found, 1)
                        st.markdown(f"#### {tr(lang, 'ml_target_distribution')}")
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "target_min_min": target_stats.get("min"),
                                        "target_median_min": target_stats.get("median"),
                                        "target_p90_min": target_stats.get("p90"),
                                        "target_p95_min": target_stats.get("p95"),
                                        "target_max_min": target_stats.get("max"),
                                        "number_of_found_cases": found,
                                        "number_of_not_found_cases": not_found,
                                        "share_found_cases": found / total_cases,
                                        "selection_ready": metadata.get("selection_ready"),
                                        "selection_block_reason": metadata.get("selection_block_reason"),
                                    }
                                ]
                            ),
                            width="stretch",
                            hide_index=True,
                        )
                        if metadata.get("breakthrough_found_reconstructed"):
                            st.info(tr(lang, "ml_breakthrough_found_reconstructed"))
                        st.markdown(f"#### {tr(lang, 'ml_selected_model')}")
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "selected_model": metadata.get("selected_model") or metadata.get("model_type"),
                                        "model_type": metadata.get("model_type"),
                                        "selected_target_transform": metadata.get("selected_target_transform"),
                                        "selection_ready": metadata.get("selection_ready"),
                                        "selection_block_reason": metadata.get("selection_block_reason"),
                                        "why_selected": metadata.get("why_selected") or tr(lang, "ml_model_not_selected_low_quality"),
                                        "why_not_linear_regression": metadata.get("why_not_linear_regression"),
                                        "target_max_h": (target_stats.get("max") / 60.0) if target_stats.get("max") is not None and pd.notna(target_stats.get("max")) else None,
                                    }
                                ]
                            ),
                            width="stretch",
                            hide_index=True,
                        )
                        st.info(tr(lang, "ml_linear_regression_baseline_only"))
                        st.caption(tr(lang, "ml_choose_model_manually_disabled"))
                    if isinstance(result.get("feature_importance"), pd.DataFrame) and not result["feature_importance"].empty:
                        st.markdown(f"#### {tr(lang, 'ml_feature_importance')}")
                        st.caption(tr(lang, "equilibrium_importance_scope"))
                        st.dataframe(result["feature_importance"], width="stretch", hide_index=True)

            if model_choice == "both":
                train_button_label = tr(lang, "ml_train_both_button")
            elif model_choice == "breakthrough":
                train_button_label = tr(lang, "ml_train_recommended_breakthrough_button")
            else:
                train_button_label = tr(lang, "ml_train_optional_curve_button")
            if st.button(train_button_label, type="primary", key="ml_train_selected_model_button", disabled=breakthrough_training_blocked):
                if model_choice == "both":
                    run_training("breakthrough")
                    run_training("curve")
                else:
                    run_training(model_choice)

            render_session_models_registry(lang, show_downloads=True, key_prefix="train")

    with load_block:
        st.markdown(f"#### {tr(lang, 'ml_load_saved_model')}")
        render_active_ml_model_card(lang)
        render_sync_active_breakthrough_button(lang, "load_sync_active_breakthrough")
        render_session_models_registry(lang, show_downloads=True, key_prefix="load")
        if st.button(tr(lang, "ml_use_breakthrough_from_session"), key="load_use_breakthrough_from_session"):
            if set_active_breakthrough_model_from_registry():
                st.success(tr(lang, "ml_breakthrough_session_model_active"))
            else:
                st.error(tr(lang, "ml_no_valid_breakthrough_in_session"))
        local_col1, local_col2 = st.columns(2)
        with local_col1:
            if st.button(tr(lang, "ml_save_models_local"), key="ml_save_models_local"):
                saved_paths = save_registry_models_to_local_folder("models")
                if saved_paths:
                    st.success("; ".join(saved_paths))
                else:
                    st.warning(tr(lang, "ml_no_valid_breakthrough_in_session"))
        with local_col2:
            if st.button(tr(lang, "ml_load_breakthrough_local"), key="ml_load_breakthrough_local"):
                local_bundle_path = os.path.join("models", "breakthrough_model_bundle.joblib")
                if not os.path.exists(local_bundle_path):
                    st.error(local_bundle_path)
                else:
                    try:
                        loaded_model = load_saved_model(local_bundle_path, None)
                    except Exception as exc:
                        loaded_model = {"success": False, "message": tr(lang, "ml_model_load_failed"), "details": repr(exc)}
                    if loaded_model.get("success"):
                        store_metadata = loaded_model.get("metadata") or {}
                        if validate_model_for_adsorber_selection(store_metadata):
                            set_registry_model(
                                "breakthrough",
                                loaded_model["model"],
                                store_metadata,
                                store_metadata.get("metrics") or {},
                                source="local_models_folder",
                            )
                            if set_active_breakthrough_model_from_registry():
                                st.success(tr(lang, "ml_breakthrough_session_model_active"))
                        else:
                            st.error(tr(lang, "ml_model_not_selection_ready"))
                    else:
                        st.error(loaded_model.get("message", "load failed"))
                        if loaded_model.get("details"):
                            with st.expander(tr(lang, "show_technical_details")):
                                st.code(loaded_model["details"], language="text")
        st.info(tr(lang, "ml_upload_breakthrough_bundle_hint"))

        def store_loaded_model_state(loaded_model: dict) -> dict:
            metadata = loaded_model.get("metadata") or {}
            task = metadata.get("task_type")
            st.session_state["loaded_ml_model"] = loaded_model["model"]
            st.session_state["loaded_ml_metadata"] = metadata
            st.session_state["loaded_ml_task_type"] = task
            st.session_state["loaded_ml_model_filename"] = loaded_model.get("loaded_model_filename", "")
            st.session_state["loaded_ml_metadata_filename"] = loaded_model.get("loaded_metadata_filename", "")
            st.session_state["loaded_ml_warnings"] = loaded_model.get("warnings", [])
            st.session_state[f"ml_{task}_model"] = loaded_model["model"]
            st.session_state[f"ml_{task}_metadata"] = metadata
            if task in {"curve", "breakthrough"}:
                set_registry_model(
                    task,
                    loaded_model["model"],
                    metadata,
                    metadata.get("metrics") or {},
                    source="uploaded_model",
                )
            return metadata

        def render_loaded_model_diagnostics() -> None:
            metadata = st.session_state.get("loaded_ml_metadata")
            if not isinstance(metadata, dict) or not metadata:
                return
            st.markdown(f"#### {tr(lang, 'ml_loaded_model_diagnostics')}")
            st.dataframe(
                ml_model_summary_dataframe(
                    metadata,
                    model_filename=st.session_state.get("loaded_ml_model_filename", ""),
                    metadata_filename=st.session_state.get("loaded_ml_metadata_filename", ""),
                    source="uploaded_model",
                ),
                width="stretch",
                hide_index=True,
            )
            for warning in st.session_state.get("loaded_ml_warnings", []):
                st.warning(warning)
            if metadata.get("task_type") == "breakthrough":
                if metadata.get("selection_ready") is not True:
                    st.error(tr(lang, "ml_model_not_selection_ready"))
                    st.warning(str(metadata.get("selection_block_reason") or tr(lang, "ml_model_quality_gate_failed")))
                elif st.button(tr(lang, "ml_use_loaded_for_selection"), key="ml_use_loaded_for_selection"):
                    if set_active_breakthrough_model_from_registry():
                        st.success(tr(lang, "ml_loaded_model_is_active"))
                    else:
                        st.error(tr(lang, "ml_no_valid_breakthrough_in_session"))
            else:
                st.info(tr(lang, "ml_loaded_not_suitable_for_selection"))

        st.markdown(f"**{tr(lang, 'ml_upload_breakthrough_bundle_hint')}**")
        bundle_upload = st.file_uploader(
            tr(lang, "ml_upload_breakthrough_bundle"),
            type=["joblib"],
            key="ml_upload_bundle",
        )
        if bundle_upload is not None and st.button(tr(lang, "ml_load_bundle_button"), key="ml_load_bundle_button"):
            try:
                loaded_model = load_saved_model(bundle_upload, None)
            except Exception as exc:
                loaded_model = {"success": False, "message": tr(lang, "ml_model_load_failed"), "details": repr(exc)}
            if loaded_model.get("success"):
                metadata = loaded_model.get("metadata") or {}
                task = metadata.get("task_type")
                if task not in {"curve", "breakthrough"}:
                    st.error(tr(lang, "ml_metadata_task_missing"))
                else:
                    store_loaded_model_state(loaded_model)
                    st.success(tr(lang, "ml_model_loaded"))
                    if task == "curve":
                        st.warning(tr(lang, "ml_curve_model_not_for_selection"))
                st.json(metadata)
            else:
                st.error(loaded_model.get("message", "load failed"))
                if loaded_model.get("details"):
                    with st.expander(tr(lang, "show_technical_details")):
                        st.code(loaded_model["details"], language="text")

        with st.expander(tr(lang, "ml_legacy_load_mode"), expanded=False):
            model_upload = st.file_uploader(tr(lang, "ml_upload_joblib"), type=["joblib"], key="ml_upload_joblib")
            metadata_upload = st.file_uploader(tr(lang, "ml_upload_metadata"), type=["json"], key="ml_upload_metadata")
            if model_upload is not None and st.button(tr(lang, "ml_load_model_button"), key="ml_load_model_button"):
                try:
                    loaded_model = load_saved_model(model_upload, metadata_upload)
                except Exception as exc:
                    loaded_model = {"success": False, "message": tr(lang, "ml_model_load_failed"), "details": repr(exc)}
                if loaded_model.get("success"):
                    metadata = loaded_model.get("metadata") or {}
                    task = metadata.get("task_type")
                    if task not in {"curve", "breakthrough"}:
                        st.error(tr(lang, "ml_metadata_task_missing"))
                    else:
                        store_loaded_model_state(loaded_model)
                        st.success(tr(lang, "ml_model_loaded"))
                        if loaded_model.get("legacy_warning"):
                            st.warning(loaded_model["legacy_warning"])
                        if task == "curve":
                            st.warning(tr(lang, "ml_curve_model_not_for_selection"))
                    st.json(metadata)
                else:
                    st.error(loaded_model.get("message", "load failed"))
                    if loaded_model.get("details"):
                        with st.expander(tr(lang, "show_technical_details")):
                            st.code(loaded_model["details"], language="text")

        render_loaded_model_diagnostics()

        if st.button(tr(lang, "ml_clear_loaded_model"), key="ml_clear_loaded_model"):
            for state_key in [
                "loaded_ml_model",
                "loaded_ml_metadata",
                "loaded_ml_task_type",
                "loaded_ml_model_filename",
                "loaded_ml_metadata_filename",
                "loaded_ml_warnings",
                "trained_breakthrough_model",
                "trained_breakthrough_metadata",
                "trained_breakthrough_bundle_bytes",
                "trained_curve_model",
                "trained_curve_metadata",
                "active_ml_model",
                "active_ml_metadata",
                "active_ml_source",
                "active_ml_task_type",
                "active_ml_model_key",
            ]:
                st.session_state.pop(state_key, None)
            st.rerun()

    with select_block:
        st.markdown(f"#### {tr(lang, 'ml_adsorber_selection')}")
        st.info(tr(lang, "ml_required_breakthrough_model"))
        active_model = st.session_state.get("active_ml_model")
        active_metadata = st.session_state.get("active_ml_metadata")
        active_source = st.session_state.get("active_ml_source", "none")
        breakthrough_model = active_model
        breakthrough_metadata = active_metadata

        render_active_ml_model_card(lang)
        render_sync_active_breakthrough_button(lang, "selection_sync_active_breakthrough")
        if not validate_model_for_adsorber_selection(active_metadata) and get_registry_model("breakthrough"):
            if st.button(tr(lang, "ml_use_breakthrough_from_session"), key="selection_use_breakthrough_from_session"):
                if set_active_breakthrough_model_from_registry():
                    st.success(tr(lang, "ml_breakthrough_session_model_active"))
                    st.rerun()
                else:
                    st.error(tr(lang, "ml_no_valid_breakthrough_in_session"))
        if isinstance(active_metadata, dict) and active_metadata:
            if st.button(tr(lang, "ml_clear_active_model"), key="ml_clear_active_model"):
                for state_key in [
                    "active_ml_model",
                    "active_ml_metadata",
                    "active_ml_source",
                    "active_ml_task_type",
                    "active_ml_model_key",
                ]:
                    st.session_state.pop(state_key, None)
                st.rerun()

        if breakthrough_model is None:
            st.warning(tr(lang, "ml_breakthrough_model_required"))
        elif not validate_model_for_adsorber_selection(breakthrough_metadata):
            if (breakthrough_metadata or {}).get("task_type") == "curve":
                st.error(tr(lang, "ml_wrong_model_for_selection"))
            elif (breakthrough_metadata or {}).get("task_type") != "breakthrough":
                st.error(tr(lang, "ml_breakthrough_task_required"))
            else:
                from scientific_provenance import provenance_is_current
                if not provenance_is_current(breakthrough_metadata or {}):
                    st.error(tr(lang, "scientific_stale_model"))
                st.error(tr(lang, "ml_model_not_selection_ready"))
                st.warning(str((breakthrough_metadata or {}).get("selection_block_reason") or tr(lang, "ml_model_quality_gate_failed")))
        else:
            st.caption(tr(lang, "ml_selection_caption"))
            selection_scale = st.selectbox(
                tr(lang, "ml_selection_scale"),
                SCALE_LABELS,
                index=0,
                format_func=lambda value: tr(lang, SCALE_TRANSLATION_KEYS[value]),
                key="ml_selection_scale",
            )
            selection_key = "laboratory" if selection_scale == SCALE_LABELS[0] else "engineering" if selection_scale == SCALE_LABELS[1] else "industrial"
            selection_ranges = ADSORBER_SELECTION_RANGES[selection_key]
            selection_range_mode = st.selectbox(
                tr(lang, "ml_selection_range_mode"),
                ["validated_model_range", "extended_engineering_range"],
                index=0,
                format_func=lambda value: tr(lang, f"ml_selection_range_{value}"),
                key="ml_selection_range_mode",
            )
            allow_strong_extrapolation = False
            if selection_range_mode == "extended_engineering_range":
                st.warning(tr(lang, "ml_extended_range_warning"))
                allow_strong_extrapolation = st.checkbox(
                    tr(lang, "ml_allow_strong_extrapolation"),
                    value=False,
                    key="ml_allow_strong_extrapolation",
                )
            ml_domain_mode = st.selectbox(
                tr(lang, "ml_domain_mode"),
                ["strict", "engineering_extrapolation"],
                index=0,
                format_func=lambda value: tr(lang, f"ml_domain_{value}"),
                key="ml_domain_mode",
            )
            strict_ml_domain = ml_domain_mode == "strict"
            show_continuous_generation = st.checkbox(
                tr(lang, "candidate_generation_show_continuous_debug"),
                value=False,
                key="candidate_generation_show_continuous_debug",
            )
            candidate_generation_options = ["engineering_grid_random", "discrete_grid"]
            if show_continuous_generation:
                candidate_generation_options.extend(["latin_hypercube_continuous", "random_continuous"])
            if st.session_state.get("candidate_generation_method") not in candidate_generation_options:
                st.session_state["candidate_generation_method"] = "engineering_grid_random"
            candidate_generation_method = st.selectbox(
                tr(lang, "candidate_generation_type"),
                candidate_generation_options,
                index=0,
                format_func=lambda value: tr(lang, f"candidate_generation_{value}"),
                key="candidate_generation_method",
            )
            if candidate_generation_method in {"latin_hypercube_continuous", "random_continuous"}:
                st.warning(tr(lang, "candidate_generation_continuous_warning"))
            if selection_scale == SCALE_LABELS[0]:
                st.info(tr(lang, "ml_lab_selection_note"))
            elif selection_scale == SCALE_LABELS[1]:
                st.info(tr(lang, "scale_engineering_info"))
            else:
                st.info(tr(lang, "scale_industrial_info"))
            render_active_selection_message(lang)
            if selection_range_mode == "validated_model_range":
                default_flow_range = model_range_tuple(breakthrough_metadata, "flow", selection_ranges.get("flow_ml_min", selection_ranges.get("Q_m3_h", (1.0, 20.0))))
                default_height_range = model_range_tuple(breakthrough_metadata, "height", selection_ranges.get("H_cm", selection_ranges.get("H_m", (2.5, 30.0))))
                default_diameter_range = model_range_tuple(breakthrough_metadata, "diameter", (min(selection_ranges.get("standard_D_cm", [1.5])), max(selection_ranges.get("standard_D_cm", [5.0]))) if selection_scale == SCALE_LABELS[0] else selection_ranges.get("D_m", (0.05, 0.5)))
                default_temperature_range = model_range_tuple(breakthrough_metadata, "temperature", (10.0, 30.0))
                default_epsilon_range = model_range_tuple(breakthrough_metadata, "epsilon", (0.35, 0.45))
                default_density_range = model_range_tuple(breakthrough_metadata, "bulk_density", selection_ranges.get("rho_lab_g_cm3", (490.0, 530.0)))
            else:
                default_flow_range = selection_ranges.get("flow_ml_min", selection_ranges.get("Q_m3_h", (1.0, 20.0)))
                default_height_range = selection_ranges.get("H_cm", selection_ranges.get("H_m", (2.5, 30.0)))
                default_diameter_range = (min(selection_ranges.get("standard_D_cm", [1.5])), max(selection_ranges.get("standard_D_cm", [5.0]))) if selection_scale == SCALE_LABELS[0] else selection_ranges.get("D_m", (0.05, 0.5))
                default_temperature_range = (10.0, 30.0)
                default_epsilon_range = (0.35, 0.45)
                default_density_range = selection_ranges.get("rho_lab_g_cm3", (490.0, 530.0))
            target_stats = (breakthrough_metadata or {}).get("target_stats") or {}
            target_max_min = float(target_stats.get("max", 0.0) or 0.0)
            if target_max_min > 0:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "target_min_min": target_stats.get("min"),
                                "target_median_min": target_stats.get("median"),
                                "target_max_min": target_stats.get("max"),
                                "target_max_h": target_max_min / 60.0,
                            }
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
            if "sel_required_h" not in st.session_state:
                if selection_scale == SCALE_LABELS[0] and target_max_min > 0:
                    default_required_h = max(0.1, min(2.0, 0.8 * target_max_min / 60.0))
                elif target_max_min > 0:
                    default_required_h = max(0.1, min(24.0, 0.8 * target_max_min / 60.0))
                else:
                    default_required_h = 2.0 if selection_scale == SCALE_LABELS[0] else 24.0
            else:
                default_required_h = float(st.session_state.get("sel_required_h", 2.0))
            def apply_model_safe_ranges() -> None:
                feature_to_keys = {
                    "flow": ("sel_flow_min", "sel_flow_max"),
                    "height": ("sel_h_min", "sel_h_max"),
                    "diameter": ("sel_d_min", "sel_d_max"),
                    "temperature": ("sel_temp_min", "sel_temp_max"),
                    "epsilon": ("sel_eps_min", "sel_eps_max"),
                    "bulk_density": ("sel_rho_min", "sel_rho_max"),
                }
                for feature, (min_key, max_key) in feature_to_keys.items():
                    ranges = metadata_numeric_range(breakthrough_metadata, feature)
                    try:
                        lo = float(ranges.get("min"))
                        hi = float(ranges.get("max"))
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(lo) and math.isfinite(hi) and lo <= hi:
                        st.session_state[min_key] = lo
                        st.session_state[max_key] = hi
                st.session_state["sel_C0"] = metadata_range_value(breakthrough_metadata, "C0_mg_L", 5.0)
                st.session_state["sel_cbr"] = DEFAULT_BREAKTHROUGH_RATIO
                if target_max_min > 0:
                    current_required = float(st.session_state.get("sel_required_h", default_required_h))
                    st.session_state["sel_required_h"] = max(0.1, min(current_required, 0.8 * target_max_min / 60.0))

            st.button(
                tr(lang, "ml_apply_model_safe_ranges"),
                key="ml_apply_model_safe_ranges",
                on_click=apply_model_safe_ranges,
            )
            sel_col1, sel_col2, sel_col3 = st.columns(3)
            with sel_col1:
                sel_C0 = st.number_input("C0_mg_L", min_value=0.001, value=metadata_range_value(breakthrough_metadata, "C0_mg_L", 5.0) if strict_ml_domain else 5.0, key="sel_C0")
                sel_cbr = st.number_input("C_breakthrough_ratio", min_value=0.001, max_value=0.99, value=0.05, key="sel_cbr")
                sel_required_h = st.number_input(tr(lang, "required_service_time_h"), min_value=0.1, value=float(default_required_h), key="sel_required_h")
            with sel_col2:
                if selection_scale == SCALE_LABELS[0]:
                    sel_flow_min, sel_flow_max = range_input("flow_ml_min", "sel_flow_min", "sel_flow_max", *default_flow_range, min_value=1e-6, format="%.4f")
                else:
                    sel_flow_min, sel_flow_max = range_input("Q_m3_h", "sel_flow_min", "sel_flow_max", *default_flow_range, min_value=1e-6, format="%.4f")
                sel_temp_min, sel_temp_max = range_input("temperature_C", "sel_temp_min", "sel_temp_max", *default_temperature_range, format="%.2f")
                mass_label = "maximum_carbon_mass_g" if selection_scale == SCALE_LABELS[0] else "maximum_carbon_mass_kg"
                sel_max_mass = st.number_input(mass_label, min_value=0.0, value=0.0, key="sel_max_mass")
            with sel_col3:
                if selection_scale == SCALE_LABELS[0]:
                    standard_diameters = selection_ranges["standard_D_cm"]
                    sel_d_min, sel_d_max = range_input("D_cm", "sel_d_min", "sel_d_max", *default_diameter_range, min_value=0.001, format="%.3f")
                    sel_h_min, sel_h_max = range_input("H_cm", "sel_h_min", "sel_h_max", *default_height_range, min_value=0.001, format="%.3f")
                else:
                    sel_d_min, sel_d_max = range_input("D_m", "sel_d_min", "sel_d_max", *default_diameter_range, min_value=0.001, format="%.3f")
                    sel_h_min, sel_h_max = range_input("H_m", "sel_h_min", "sel_h_max", *default_height_range, min_value=0.001, format="%.3f")
                n_candidates = st.number_input("n_candidates", min_value=100, max_value=10000, value=1000, step=100, key="sel_n_candidates")
            with st.expander(tr(lang, "advanced_parameters"), expanded=False):
                sel_eps_min, sel_eps_max = range_input("epsilon", "sel_eps_min", "sel_eps_max", *default_epsilon_range, min_value=0.05, max_value=0.95, format="%.3f")
                if selection_scale == SCALE_LABELS[0]:
                    density_source = st.selectbox(
                        tr(lang, "density_source"),
                        ["lab_loading", "passport_bulk_density", "manual"],
                        index=0,
                        format_func=lambda value: tr(lang, f"density_source_{value}"),
                        key="sel_density_source",
                    )
                    if density_source == "lab_loading":
                        sel_rho_min, sel_rho_max = default_density_range if selection_range_mode == "validated_model_range" else selection_ranges["rho_lab_g_cm3"]
                        st.caption(f"{sel_rho_min:.3f}–{sel_rho_max:.3f} г/см³")
                    elif density_source == "passport_bulk_density":
                        sel_rho_min, sel_rho_max = selection_ranges["rho_passport_g_cm3"]
                        st.caption(f"{sel_rho_min:.3f}–{sel_rho_max:.3f} г/см³")
                    else:
                        sel_rho_min, sel_rho_max = range_input("bulk_density_g_cm3", "sel_rho_min", "sel_rho_max", *default_density_range, min_value=0.001, format="%.4f")
                else:
                    sel_rho_min, sel_rho_max = range_input("bulk_density_kg_m3", "sel_rho_min", "sel_rho_max", *default_density_range, min_value=1.0, format="%.2f")
                sel_dp_min, sel_dp_max = range_input("dp_mm", "sel_dp_min", "sel_dp_max", 0.8, 2.0, min_value=0.001, format="%.4f")
                sel_pressure_max = st.number_input("maximum_pressure_drop_bar", min_value=0.0, value=2.0, key="sel_pressure_max")
                sel_ebct_min, sel_ebct_max = range_input("EBCT_min", "sel_ebct_min", "sel_ebct_max", *selection_ranges["EBCT_min"], min_value=0.0, format="%.2f")
            if target_max_min > 0:
                if float(sel_required_h) > target_max_min / 60.0:
                    st.error(tr(lang, "ml_required_time_exceeds_target").format(maximum=f"{target_max_min / 60.0:.3g}"))
                elif float(sel_required_h) >= 24.0 and target_max_min < 24.0 * 60.0:
                    st.warning(tr(lang, "ml_model_not_for_24h").format(maximum=f"{target_max_min / 60.0:.3g}"))
                else:
                    st.info(
                        tr(lang, "ml_required_time_inside_target").format(
                            required=f"{float(sel_required_h):.3g}",
                            maximum=f"{target_max_min / 60.0:.3g}",
                        )
                    )
            if st.button(tr(lang, "ml_sync_sorption_to_active_model"), key="ml_sync_sorption_to_active_model"):
                st.session_state["ml_sorption_source_for_selection"] = "training_median"
                st.rerun()
            ml_sorption_source = st.selectbox(
                tr(lang, "ml_sorption_source_for_selection"),
                ["training_median", "current_calibration", "best_experiment", "adsorbent_profile", "manual"],
                index=0,
                format_func=lambda value: tr(lang, f"ml_sorption_source_{value}"),
                key="ml_sorption_source_for_selection",
            )
            training_param_values = {
                "KH_used_L_g": metadata_range_value(breakthrough_metadata, "KH_used_L_g", CHEMVIRON_START_PARAMS["KH_L_g"]),
                "kLDF_used_1_min": metadata_range_value(breakthrough_metadata, "kLDF_used_1_min", CHEMVIRON_START_PARAMS["kLDF_1_min"]),
                "Dax_used_cm2_min": metadata_range_value(breakthrough_metadata, "Dax_used_cm2_min", CHEMVIRON_START_PARAMS["Dax_cm2_min"]),
                "isotherm_model": "Henry",
            }
            training_param_values["parameter_provenance"] = training_stat_origins(
                breakthrough_metadata, training_param_values,
                {"KH_used_L_g": CHEMVIRON_START_PARAMS["KH_L_g"], "kLDF_used_1_min": CHEMVIRON_START_PARAMS["kLDF_1_min"], "Dax_used_cm2_min": CHEMVIRON_START_PARAMS["Dax_cm2_min"]},
            )
            if ml_sorption_source == "training_median":
                selection_sorption_params = training_param_values
            elif ml_sorption_source == "manual":
                manual_col1, manual_col2, manual_col3 = st.columns(3)
                with manual_col1:
                    manual_kh = st.number_input("KH_used_L_g", min_value=1e-12, value=float(training_param_values["KH_used_L_g"]), format="%.8f", key="ml_selection_manual_kh")
                with manual_col2:
                    manual_kldf = st.number_input("kLDF_used_1_min", min_value=1e-12, value=float(training_param_values["kLDF_used_1_min"]), format="%.8f", key="ml_selection_manual_kldf")
                with manual_col3:
                    manual_dax = st.number_input("Dax_used_cm2_min", min_value=0.0, value=float(training_param_values["Dax_used_cm2_min"]), format="%.10f", key="ml_selection_manual_dax")
                selection_sorption_params = {
                    "KH_used_L_g": float(manual_kh),
                    "kLDF_used_1_min": float(manual_kldf),
                    "Dax_used_cm2_min": float(manual_dax),
                    "isotherm_model": "Henry",
                    "parameter_provenance": {key: provenance_record("user", method="user_input", input_field=key) for key in ("KH_used_L_g", "kLDF_used_1_min", "Dax_used_cm2_min")},
                }
            else:
                resolved = resolve_sorption_parameters(
                    profile_key,
                    pollutant,
                    "auto_best_available" if ml_sorption_source == "current_calibration" else ml_sorption_source,
                    fitted_df=fitted_temperature_dataframe(),
                    last_calibration=st.session_state.get("last_calibration"),
                )
                selection_sorption_params = {
                    "KH_used_L_g": float(resolved["KH_L_g"]),
                    "kLDF_used_1_min": float(resolved["kLDF_1_min"]),
                    "Dax_used_cm2_min": float(resolved["Dax_cm2_min"]),
                    "isotherm_model": resolved.get("isotherm_model", "Henry"),
                    "parameter_provenance": {target: parameter_origin(resolved, reference) for target, reference in (("KH_used_L_g", "KH_L_g"), ("kLDF_used_1_min", "kLDF_1_min"), ("Dax_used_cm2_min", "Dax_cm2_min"))},
                }
            st.session_state["active_ml_sorption_params"] = selection_sorption_params
            sorption_rows = []
            for param_name in ["KH_used_L_g", "kLDF_used_1_min", "Dax_used_cm2_min"]:
                origin_label, origin_comment = origin_description(parameter_origin(selection_sorption_params, param_name), lang)
                param_range = metadata_numeric_range(breakthrough_metadata, param_name)
                value = float(selection_sorption_params[param_name])
                try:
                    range_min = float(param_range.get("min"))
                    range_max = float(param_range.get("max"))
                    status = "inside" if range_min <= value <= range_max else "outside"
                    training_range = f"{range_min:.4g}-{range_max:.4g}"
                except (TypeError, ValueError):
                    status = "unknown"
                    training_range = "-"
                sorption_rows.append(
                    {
                        "parameter": param_name,
                        "selection_value": value,
                        "training_range": training_range,
                        "source": tr(lang, f"ml_sorption_source_{ml_sorption_source}"),
                        tr(lang, "provenance_col_origin"): origin_label,
                        tr(lang, "col_comment"): origin_comment,
                        "status": status,
                    }
                )
            st.dataframe(pd.DataFrame(sorption_rows), width="stretch", hide_index=True)
            if float(sel_required_h) >= 24.0 and selection_scale == SCALE_LABELS[0]:
                st.warning(tr(lang, "ml_long_service_lab_warning"))

            def build_candidate_prediction_row(candidate_row: pd.Series, candidate_id: int = 0) -> tuple[dict, dict, list[str]]:
                flow_value = float(candidate_row["flow"])
                height_value = float(candidate_row["height"])
                diameter_value = float(candidate_row["diameter"])
                temperature_value = float(candidate_row["temperature"])
                epsilon_value = float(candidate_row["epsilon"])
                bulk_density_value = float(candidate_row["bulk_density"])
                dp_value = float(candidate_row["dp_mm"])
                column, sorption, diagnostics, _messages = build_column_and_diagnostics(
                    selection_scale,
                    float(sel_C0),
                    flow_value,
                    height_value,
                    diameter_value,
                    epsilon_value,
                    bulk_density_value,
                    temperature_value,
                    dp_value,
                    0.85,
                    float(selection_sorption_params["KH_used_L_g"]),
                    float(selection_sorption_params["kLDF_used_1_min"]),
                    float(selection_sorption_params["Dax_used_cm2_min"]),
                    20.0,
                    20.0,
                    20.0,
                    5e-10,
                    "температурная поправка к D_ax_ref",
                    "температурная поправка Arrhenius",
                )
                if selection_scale == SCALE_LABELS[0]:
                    pressure_drop_bar, pressure_check_status = calculate_pressure_drop_ergun_for_candidate(
                        selection_scale,
                        flow_value,
                        diameter_value,
                        height_value,
                        dp_value,
                        epsilon_value,
                        temperature_value,
                    )
                    bed_volume_display = float(diagnostics.get("bed_volume_cm3", 0.0))
                    carbon_mass_display = float(diagnostics.get("carbon_mass_g", 0.0))
                    carbon_mass_g = carbon_mass_display
                    carbon_mass_kg = carbon_mass_display / 1000.0
                    bed_volume_cm3 = bed_volume_display
                    bed_volume_m3 = bed_volume_display / 1_000_000.0
                    superficial_display = float(diagnostics.get("superficial_velocity_cm_min", 0.0))
                    interstitial_display = float(diagnostics.get("interstitial_velocity_cm_min", 0.0))
                else:
                    pressure_drop_bar, pressure_check_status = calculate_pressure_drop_ergun_for_candidate(
                        selection_scale,
                        flow_value,
                        diameter_value,
                        height_value,
                        dp_value,
                        epsilon_value,
                        temperature_value,
                    )
                    bed_volume_display = float(diagnostics.get("bed_volume_m3", 0.0))
                    carbon_mass_display = float(diagnostics.get("carbon_mass_kg", 0.0))
                    carbon_mass_kg = carbon_mass_display
                    carbon_mass_g = carbon_mass_display * 1000.0
                    bed_volume_m3 = bed_volume_display
                    bed_volume_cm3 = bed_volume_display * 1_000_000.0
                    superficial_display = float(diagnostics.get("superficial_velocity_m_h", 0.0))
                    interstitial_display = float(diagnostics.get("interstitial_velocity_m_h", 0.0))
                h_to_d_ratio = height_value / max(diameter_value, 1e-30)
                feature_row = {
                    "C0_mg_L": float(sel_C0),
                    "C_breakthrough_ratio": float(sel_cbr),
                    "flow": flow_value,
                    "height": height_value,
                    "diameter": diameter_value,
                    "temperature": temperature_value,
                    "pressure_bar": 1.01325,
                    "epsilon": epsilon_value,
                    "bulk_density": bulk_density_value,
                    "carbon_mass": carbon_mass_g if selection_scale == SCALE_LABELS[0] else carbon_mass_kg,
                    "carbon_mass_g": carbon_mass_g,
                    "carbon_mass_kg": carbon_mass_kg,
                    "bed_volume_cm3": bed_volume_cm3,
                    "bed_volume_m3": bed_volume_m3,
                    "EBCT_min": float(diagnostics.get("EBCT_min", 0.0)),
                    "KH_used_L_g": float(selection_sorption_params["KH_used_L_g"]),
                    "kLDF_used_1_min": float(selection_sorption_params["kLDF_used_1_min"]),
                    "Dax_used_cm2_min": float(selection_sorption_params["Dax_used_cm2_min"]),
                    "isotherm_model": selection_sorption_params["isotherm_model"],
                    "adsorbent_profile": profile_label(profile_key, lang),
                    "scale_mode": selection_scale,
                }
                display_row = {
                    "candidate_id": int(candidate_id),
                    "scale_mode": selection_scale,
                    "flow": flow_value,
                    "height": height_value,
                    "diameter": diameter_value,
                    "temperature_C": temperature_value,
                    "epsilon": epsilon_value,
                    "bulk_density": bulk_density_value,
                    "dp_mm": dp_value,
                    "bed_volume": bed_volume_display,
                    "carbon_mass": carbon_mass_display,
                    "superficial_velocity": superficial_display,
                    "interstitial_velocity": interstitial_display,
                    "H_to_D_ratio": h_to_d_ratio,
                    "EBCT_min": feature_row["EBCT_min"],
                    "pressure_drop_bar": pressure_drop_bar,
                    "pressure_check_status": pressure_check_status,
                    "column": column,
                    "sorption": sorption,
                    "diagnostics": diagnostics,
                }
                return feature_row, display_row, outside_training_features(feature_row, breakthrough_metadata or {})

            if st.button(tr(lang, "ml_check_current_inputs"), key="ml_check_current_inputs"):
                midpoint_candidate = pd.Series(
                    {
                        "flow": 0.5 * (float(sel_flow_min) + float(sel_flow_max)),
                        "height": 0.5 * (float(sel_h_min) + float(sel_h_max)),
                        "diameter": 0.5 * (float(sel_d_min) + float(sel_d_max)),
                        "temperature": 0.5 * (float(sel_temp_min) + float(sel_temp_max)),
                        "epsilon": 0.5 * (float(sel_eps_min) + float(sel_eps_max)),
                        "bulk_density": 0.5 * (float(sel_rho_min) + float(sel_rho_max)),
                        "dp_mm": 0.5 * (float(sel_dp_min) + float(sel_dp_max)),
                    }
                )
                feature_row, _display_row, outside = build_candidate_prediction_row(midpoint_candidate, -1)
                feature_df_raw = pd.DataFrame([feature_row])
                feature_df, unit_diagnostics = normalize_units_for_ml(feature_df_raw, breakthrough_metadata or {}, selection_scale)
                outside = outside_training_features(feature_df.iloc[0].to_dict(), breakthrough_metadata or {})
                missing = missing_features_for_model(feature_df, breakthrough_metadata or {})
                st.markdown(f"#### {tr(lang, 'ml_prediction_test')}")
                st.dataframe(feature_df, width="stretch", hide_index=True)
                if unit_diagnostics.get("applied_unit_conversions"):
                    st.info(tr(lang, "unit_conversion_applied") + ": " + ", ".join(unit_diagnostics["applied_unit_conversions"]))
                if missing:
                    st.error(tr(lang, "ml_missing_model_features").format(columns=", ".join(missing)))
                else:
                    try:
                        predicted_test_min = float(predict_with_metadata(breakthrough_model, feature_df, breakthrough_metadata or {})[0])
                        st.success(f"{tr(lang, 'predicted_breakthrough_time_h')}: {predicted_test_min / 60.0:.3g}")
                    except Exception as exc:
                        st.error(tr(lang, "ml_prediction_failed"))
                        with st.expander(tr(lang, "show_technical_details")):
                            st.code(repr(exc), language="text")
                if outside:
                    st.warning(tr(lang, "ml_outside_training_features").format(features=", ".join(outside)))

            if st.button(tr(lang, "ml_generate_candidates"), type="primary", key="ml_generate_candidates"):
                with st.status(tr(lang, "adsorber_selection_running"), expanded=True) as status:
                    progress_bar = st.progress(0.0)
                    try:
                        status.write(tr(lang, "generating_engineering_candidates"))
                        ranges_for_candidates = {
                            "flow": (sel_flow_min, sel_flow_max),
                            "height": (sel_h_min, sel_h_max),
                            "diameter": (sel_d_min, sel_d_max),
                            "temperature": (sel_temp_min, sel_temp_max),
                            "epsilon": (sel_eps_min, sel_eps_max),
                            "bulk_density": (sel_rho_min, sel_rho_max),
                            "dp_mm": (sel_dp_min, sel_dp_max),
                        }
                        requested_ranges_for_model = {
                            "C0_mg_L": (float(sel_C0), float(sel_C0)),
                            "C_breakthrough_ratio": (float(sel_cbr), float(sel_cbr)),
                            "flow": (float(sel_flow_min), float(sel_flow_max)),
                            "height": (float(sel_h_min), float(sel_h_max)),
                            "diameter": (float(sel_d_min), float(sel_d_max)),
                            "temperature": (float(sel_temp_min), float(sel_temp_max)),
                            "epsilon": (float(sel_eps_min), float(sel_eps_max)),
                            "bulk_density": (float(sel_rho_min), float(sel_rho_max)),
                        }
                        preflight = validate_selection_request_against_model(
                            breakthrough_metadata,
                            requested_ranges_for_model,
                            float(sel_required_h) * 60.0,
                            selection_scale,
                            selection_sorption_params,
                            allow_strong_extrapolation=bool(allow_strong_extrapolation or not strict_ml_domain),
                        )
                        st.session_state["ml_selection_domain_df"] = preflight["domain_df"]
                        if not preflight["ok"]:
                            for state_key in [
                                "ml_all_candidates_df",
                                "ml_candidates_df",
                                "ml_verification_candidates_df",
                                "ml_rejected_candidates_df",
                                "ml_rejection_reasons_df",
                                "ml_selection_diagnostics_df",
                            ]:
                                st.session_state.pop(state_key, None)
                            status.update(label=tr(lang, "adsorber_selection_error"), state="error")
                            st.error(tr(lang, "ml_selection_domain_error_title"))
                            for message in preflight["errors"]:
                                st.error(message)
                            if isinstance(preflight["domain_df"], pd.DataFrame) and not preflight["domain_df"].empty:
                                st.dataframe(preflight["domain_df"], width="stretch", hide_index=True)
                            st.stop()
                        for message in preflight["warnings"]:
                            st.warning(message)
                        candidates = make_engineering_candidate_adsorbers(
                            int(n_candidates),
                            selection_scale,
                            ranges_for_candidates,
                            random_seed=42,
                            method=candidate_generation_method,
                            model_feature_ranges=None,
                            strict_domain=False,
                        )
                        generator_domain_attrs = dict(getattr(candidates, "attrs", {}) or {})
                        generated_count_total = int(generator_domain_attrs.get("pre_domain_generated_count", len(candidates)))
                        if candidates.empty:
                            candidate_domain_df = pd.DataFrame()
                            if "carbon_mass_generated_min" in generator_domain_attrs:
                                candidate_domain_df = pd.DataFrame(
                                    [
                                        {
                                            "parameter": "carbon_mass",
                                            "model_range": f"{float(generator_domain_attrs.get('carbon_mass_model_min')):.4g}-{float(generator_domain_attrs.get('carbon_mass_model_max')):.4g}",
                                            "generated_range": f"{float(generator_domain_attrs.get('carbon_mass_generated_min')):.4g}-{float(generator_domain_attrs.get('carbon_mass_generated_max')):.4g}",
                                            "problem": "derived from D,H,rho",
                                            "rejected_candidates": int(generator_domain_attrs.get("carbon_mass_filtered_count", generated_count_total)),
                                            "note": "derived_feature",
                                        }
                                    ]
                                )
                            st.session_state["ml_candidate_domain_df"] = candidate_domain_df
                            st.session_state["ml_selection_diagnostics_df"] = pd.DataFrame(
                                [
                                    {
                                        "n_candidates_generated": int(generated_count_total),
                                        "n_predictions_success": 0,
                                        "n_prediction_errors": 0,
                                        "n_outside_training_range": int(generated_count_total),
                                        "n_candidates_in_ml_domain": 0,
                                        "n_candidates_filtered_by_ml_domain": int(generated_count_total),
                                        "required_time_min": float(sel_required_h) * 60.0,
                                    }
                                ]
                            )
                            for state_key in [
                                "ml_all_candidates_df",
                                "ml_candidates_df",
                                "ml_verification_candidates_df",
                                "ml_rejected_candidates_df",
                                "ml_rejection_reasons_df",
                            ]:
                                st.session_state.pop(state_key, None)
                            progress_bar.progress(1.0)
                            status.update(label=tr(lang, "adsorber_selection_complete"), state="complete")
                            st.warning(tr(lang, "ml_no_candidates_in_domain"))
                            st.info(tr(lang, "ml_no_candidates_in_domain_hint"))
                            if not candidate_domain_df.empty:
                                st.dataframe(candidate_domain_df, width="stretch", hide_index=True)
                            st.stop()
                        progress_bar.progress(0.15)
                        status.write(tr(lang, "calculating_candidate_features"))
                        feature_rows: list[dict] = []
                        display_rows: list[dict] = []
                        outside_rows: list[list[str]] = []
                        for row_number, (idx, row) in enumerate(candidates.iterrows(), start=1):
                            feature_row, display_row, outside_features = build_candidate_prediction_row(row, int(idx))
                            feature_rows.append(feature_row)
                            display_rows.append(display_row)
                            outside_rows.append(outside_features)
                            if row_number % 50 == 0 or row_number == len(candidates):
                                progress_bar.progress(0.15 + 0.35 * row_number / max(len(candidates), 1))
                        feature_df_raw = pd.DataFrame(feature_rows)
                        status.write(tr(lang, "normalizing_ml_units"))
                        feature_df, unit_diagnostics = normalize_units_for_ml(
                            feature_df_raw,
                            breakthrough_metadata or {},
                            selection_scale,
                        )
                        outside_rows = [
                            outside_training_features(row.to_dict(), breakthrough_metadata or {})
                            for _, row in feature_df.iterrows()
                        ]
                        feature_ranges_for_schema = (breakthrough_metadata or {}).get("feature_ranges", {}) or {}
                        feature_units_for_schema = (breakthrough_metadata or {}).get("feature_units", {}) or {}
                        schema_rows = []
                        conversions_text = "; ".join(map(str, unit_diagnostics.get("applied_unit_conversions") or []))
                        for feature_name in list((breakthrough_metadata or {}).get("feature_columns") or []):
                            ranges = feature_ranges_for_schema.get(feature_name) or {}
                            before_series = pd.to_numeric(feature_df_raw[feature_name], errors="coerce") if feature_name in feature_df_raw.columns else pd.Series(dtype=float)
                            after_series = pd.to_numeric(feature_df[feature_name], errors="coerce") if feature_name in feature_df.columns else pd.Series(dtype=float)
                            display_unit = ""
                            if feature_name == "carbon_mass":
                                if "carbon_mass_g" in feature_df_raw.columns:
                                    display_unit = "g"
                                elif "carbon_mass_kg" in feature_df_raw.columns:
                                    display_unit = "kg"
                            schema_rows.append(
                                {
                                    "feature": feature_name,
                                    "model_unit": feature_units_for_schema.get(feature_name, ""),
                                    "candidate_display_unit": display_unit,
                                    "conversion_applied": conversions_text if feature_name == "carbon_mass" else "",
                                    "model_min": ranges.get("min"),
                                    "model_median": ranges.get("median"),
                                    "model_max": ranges.get("max"),
                                    "candidate_min_before": float(before_series.min()) if before_series.notna().any() else None,
                                    "candidate_max_before": float(before_series.max()) if before_series.notna().any() else None,
                                    "candidate_min_after": float(after_series.min()) if after_series.notna().any() else None,
                                    "candidate_max_after": float(after_series.max()) if after_series.notna().any() else None,
                                }
                            )
                        st.session_state["ml_schema_diagnostics_df"] = pd.DataFrame(schema_rows)
                        st.session_state["ml_unit_diagnostics"] = unit_diagnostics
                        missing_features = missing_features_for_model(feature_df, breakthrough_metadata or {})
                        if missing_features:
                            if missing_features == ["time_min"] and (breakthrough_metadata or {}).get("task_type") == "curve":
                                raise ValueError(tr(lang, "ml_using_curve_model_for_selection"))
                            raise ValueError(
                                tr(lang, "ml_missing_model_features").format(columns=", ".join(missing_features))
                            )
                        domain_mask, candidate_domain_df = candidate_domain_mask_and_report(
                            feature_df,
                            breakthrough_metadata or {},
                            strict_domain=bool(strict_ml_domain),
                        )
                        if "carbon_mass_generated_min" in generator_domain_attrs:
                            candidate_domain_df = pd.concat(
                                [
                                    candidate_domain_df,
                                    pd.DataFrame(
                                        [
                                            {
                                                "parameter": "carbon_mass",
                                                "model_range": f"{float(generator_domain_attrs.get('carbon_mass_model_min')):.4g}-{float(generator_domain_attrs.get('carbon_mass_model_max')):.4g}",
                                                "generated_range": f"{float(generator_domain_attrs.get('carbon_mass_generated_min')):.4g}-{float(generator_domain_attrs.get('carbon_mass_generated_max')):.4g}",
                                                "problem": "derived from D,H,rho",
                                                "rejected_candidates": int(generator_domain_attrs.get("carbon_mass_filtered_count", 0)),
                                                "note": "derived_feature",
                                            }
                                        ]
                                    ),
                                ],
                                ignore_index=True,
                            ).drop_duplicates(subset=["parameter", "model_range", "generated_range", "problem"], keep="first")
                        st.session_state["ml_candidate_domain_df"] = candidate_domain_df
                        st.session_state["ml_candidates_generated_count"] = int(generated_count_total)
                        st.session_state["ml_candidates_in_domain_count"] = int(domain_mask.sum())
                        if isinstance(candidate_domain_df, pd.DataFrame) and not candidate_domain_df.empty:
                            derived_problems = candidate_domain_df[candidate_domain_df.get("note").astype(str).eq("derived_feature")]
                            if not derived_problems.empty:
                                st.warning(tr(lang, "ml_carbon_mass_derived_warning"))
                                st.info(tr(lang, "ml_candidates_in_domain").format(count=int(domain_mask.sum())))
                        if strict_ml_domain and int(domain_mask.sum()) == 0:
                            diagnostics_df = pd.DataFrame(
                                [
                                    {
                                        "n_candidates_generated": int(generated_count_total),
                                        "n_predictions_success": 0,
                                        "n_prediction_errors": 0,
                                        "n_outside_training_range": int(len(feature_df)),
                                        "n_candidates_in_ml_domain": 0,
                                        "n_candidates_filtered_by_ml_domain": int(generated_count_total),
                                        "required_time_min": float(sel_required_h) * 60.0,
                                    }
                                ]
                            )
                            for state_key in [
                                "ml_all_candidates_df",
                                "ml_candidates_df",
                                "ml_verification_candidates_df",
                                "ml_rejected_candidates_df",
                                "ml_rejection_reasons_df",
                            ]:
                                st.session_state.pop(state_key, None)
                            st.session_state["ml_selection_diagnostics_df"] = diagnostics_df
                            st.session_state["ml_selection_required_h"] = float(sel_required_h)
                            st.session_state["ml_last_selection_scale"] = selection_scale
                            progress_bar.progress(1.0)
                            status.update(label=tr(lang, "adsorber_selection_complete"), state="complete")
                            st.warning(tr(lang, "ml_no_candidates_in_domain"))
                            st.info(tr(lang, "ml_no_candidates_in_domain_hint"))
                            if isinstance(candidate_domain_df, pd.DataFrame) and not candidate_domain_df.empty:
                                st.dataframe(candidate_domain_df, width="stretch", hide_index=True)
                            st.stop()
                        prediction_indices = list(feature_df.index[domain_mask]) if strict_ml_domain else list(feature_df.index)
                        feature_df_for_prediction = feature_df.loc[prediction_indices].reset_index(drop=True)
                        normalized_feature_rows = feature_df.to_dict("records")
                        feature_rows_for_prediction = [normalized_feature_rows[int(i)] for i in prediction_indices]
                        display_rows_for_prediction = [display_rows[int(i)] for i in prediction_indices]
                        outside_rows_for_prediction = [outside_rows[int(i)] for i in prediction_indices]
                        status.write(tr(lang, "ml_prediction_breakthrough_time"))
                        predicted_values = predict_with_metadata(breakthrough_model, feature_df_for_prediction, breakthrough_metadata or {})
                        prediction_statuses = sanitize_breakthrough_predictions(predicted_values, breakthrough_metadata or {})
                        progress_bar.progress(0.75)
                        status.write(tr(lang, "filtering_candidates"))
                        metadata_ranges = (breakthrough_metadata or {}).get("feature_ranges", {})
                        rows = []
                        required_time_min = float(sel_required_h) * 60.0
                        h_to_d_min, h_to_d_max = selection_ranges["H_to_D_ratio"]
                        ebct_min_limit, ebct_max_limit = float(sel_ebct_min), float(sel_ebct_max)
                        if selection_scale == SCALE_LABELS[0]:
                            velocity_min, velocity_max = selection_ranges["superficial_velocity_cm_min"]
                        else:
                            velocity_min, velocity_max = selection_ranges["superficial_velocity_m_h"]
                        target_stats = (breakthrough_metadata or {}).get("target_stats") or {}
                        max_training_breakthrough_min = None
                        try:
                            max_training_breakthrough_min = float(target_stats.get("max"))
                        except (TypeError, ValueError):
                            max_training_breakthrough_min = None
                        for display_row, feature_row, outside_features, predicted_min, prediction_status in zip(display_rows_for_prediction, feature_rows_for_prediction, outside_rows_for_prediction, predicted_values, prediction_statuses):
                            predicted_min = float(predicted_min)
                            predicted_h = predicted_min / 60.0
                            pressure_drop_bar = display_row["pressure_drop_bar"]
                            pressure_value = float(pressure_drop_bar) if pressure_drop_bar is not None and pd.notna(pressure_drop_bar) else float("nan")
                            safety_margin = predicted_min / max(required_time_min, 1e-30)
                            passes_geometry = bool(h_to_d_min <= display_row["H_to_D_ratio"] <= h_to_d_max and display_row["bed_volume"] > 0 and display_row["carbon_mass"] > 0)
                            passes_EBCT = bool(ebct_min_limit <= display_row["EBCT_min"] <= ebct_max_limit)
                            passes_velocity = bool(velocity_min <= display_row["superficial_velocity"] <= velocity_max)
                            passes_required_time = predicted_min >= required_time_min
                            passes_mass_limit = bool(sel_max_mass <= 0 or display_row["carbon_mass"] <= float(sel_max_mass))
                            passes_pressure_limit = bool(
                                selection_scale == SCALE_LABELS[0]
                                or not math.isfinite(pressure_value)
                                or pressure_value <= float(sel_pressure_max)
                            )
                            outside_training_range = bool(outside_features)
                            prediction_unreliable = prediction_status != "valid"
                            passes_core_engineering = bool(
                                passes_geometry
                                and passes_EBCT
                                and passes_velocity
                                and passes_required_time
                                and passes_mass_limit
                                and passes_pressure_limit
                            )
                            reject_reasons = []
                            if not passes_required_time:
                                reject_reasons.append(tr(lang, "reject_required_time"))
                            if not passes_mass_limit:
                                reject_reasons.append(tr(lang, "reject_mass_limit"))
                            if not passes_pressure_limit:
                                reject_reasons.append(tr(lang, "reject_pressure_limit"))
                            if outside_training_range:
                                reject_reasons.append(tr(lang, "reject_outside_training_range") + ": " + ", ".join(outside_features[:8]))
                            if prediction_unreliable:
                                reject_reasons.append(tr(lang, "reject_prediction_unreliable"))
                            if not passes_geometry:
                                reject_reasons.append(tr(lang, "reject_geometry"))
                            if not passes_EBCT:
                                reject_reasons.append(tr(lang, "reject_ebct"))
                            if not passes_velocity:
                                reject_reasons.append(tr(lang, "reject_velocity"))
                            failed_constraints_count = int(
                                (not passes_geometry)
                                + (not passes_EBCT)
                                + (not passes_velocity)
                                + (not passes_required_time)
                                + (not passes_mass_limit)
                                + (not passes_pressure_limit)
                                + outside_training_range
                                + prediction_unreliable
                            )
                            if passes_core_engineering and not outside_training_range and not prediction_unreliable:
                                decision_status = tr(lang, "decision_suitable")
                            elif passes_core_engineering and outside_training_range and not prediction_unreliable:
                                decision_status = tr(lang, "decision_needs_physical_check")
                            else:
                                decision_status = tr(lang, "decision_rejected")
                            row_payload = {
                                "task_type": "breakthrough",
                                "target": "breakthrough_time_min",
                                "model_id": (breakthrough_metadata or {}).get("model_id"),
                                "bundle_source": st.session_state.get("active_ml_source"),
                                "domain_policy": MODEL_DOMAIN_POLICY,
                                "breakthrough_ratio": float(sel_cbr),
                                "target_unit": "min",
                                "sorption_parameter_source": ml_sorption_source,
                                "parameter_provenance_json": json.dumps(selection_sorption_params.get("parameter_provenance", {}), ensure_ascii=False, sort_keys=True),
                                **{k: v for k, v in display_row.items() if k not in {"column", "sorption", "diagnostics"}},
                                "predicted_breakthrough_time_min": predicted_min,
                                "predicted_breakthrough_time_h": predicted_h,
                                "predicted_breakthrough_time_h_display": f"{predicted_h:.3g}" if prediction_status == "valid" else tr(lang, "prediction_unreliable_display"),
                                "predicted_breakthrough_time_days": predicted_min / 1440.0,
                                "required_time_h": float(sel_required_h),
                                "decision_status": decision_status,
                                "failed_constraints_count": failed_constraints_count,
                                "passes_geometry": passes_geometry,
                                "passes_EBCT": passes_EBCT,
                                "passes_velocity": passes_velocity,
                                "passes_required_time": passes_required_time,
                                "passes_mass_limit": passes_mass_limit,
                                "passes_pressure_limit": passes_pressure_limit,
                                "outside_training_range": outside_training_range,
                                "prediction_unreliable": prediction_unreliable,
                                "prediction_status": prediction_status,
                                "reject_reason": "; ".join(reject_reasons) if reject_reasons else "",
                                "safety_margin": safety_margin,
                                "extrapolation_warning": flag_extrapolation(pd.Series(feature_row), metadata_ranges),
                                "balanced_score": display_row["carbon_mass"] + 0.01 * display_row["bed_volume"] + 0.1 * (pressure_value if math.isfinite(pressure_value) else 0.0) - 10.0 * safety_margin,
                                **{f"feature_{k}": v for k, v in feature_row.items()},
                            }
                            row_payload["engineering_summary"] = engineering_summary_for_candidate(pd.Series(row_payload), selection_scale)
                            rows.append(row_payload)
                        all_candidates_df = pd.DataFrame(rows)
                        suitable_mask = (
                            all_candidates_df["passes_geometry"]
                            & all_candidates_df["passes_EBCT"]
                            & all_candidates_df["passes_velocity"]
                            & all_candidates_df["passes_required_time"]
                            & all_candidates_df["passes_mass_limit"]
                            & all_candidates_df["passes_pressure_limit"]
                            & ~all_candidates_df["outside_training_range"]
                            & ~all_candidates_df["prediction_unreliable"]
                        )
                        verification_mask = (
                            all_candidates_df["passes_geometry"]
                            & all_candidates_df["passes_EBCT"]
                            & all_candidates_df["passes_velocity"]
                            & all_candidates_df["passes_required_time"]
                            & all_candidates_df["passes_mass_limit"]
                            & all_candidates_df["passes_pressure_limit"]
                            & all_candidates_df["outside_training_range"]
                            & ~all_candidates_df["prediction_unreliable"]
                        )
                        rejected_mask = ~(suitable_mask | verification_mask)
                        status.write(tr(lang, "ranking_candidates"))
                        suitable_candidates_df = all_candidates_df[suitable_mask].sort_values("balanced_score").head(50)
                        verification_candidates_df = sort_constrained_candidates(all_candidates_df[verification_mask]).head(20)
                        rejected_candidates_df = sort_constrained_candidates(all_candidates_df[rejected_mask]).head(20)
                        status.write(tr(lang, "forming_selection_report"))
                        diagnostics_df = selection_diagnostics_dataframe(
                            all_candidates_df,
                            int(generated_count_total),
                            0,
                            required_time_min,
                        )
                        diagnostics_df["n_candidates_in_ml_domain"] = int(len(feature_df_for_prediction))
                        diagnostics_df["n_candidates_filtered_by_ml_domain"] = int(generated_count_total - len(feature_df_for_prediction))
                        st.session_state["ml_all_candidates_df"] = all_candidates_df
                        st.session_state["ml_candidates_df"] = suitable_candidates_df
                        st.session_state["ml_verification_candidates_df"] = verification_candidates_df
                        st.session_state["ml_rejected_candidates_df"] = rejected_candidates_df
                        st.session_state["ml_rejection_reasons_df"] = rejection_reasons_dataframe(all_candidates_df)
                        st.session_state["ml_selection_diagnostics_df"] = diagnostics_df
                        st.session_state["ml_selection_required_h"] = float(sel_required_h)
                        st.session_state["ml_last_selection_scale"] = selection_scale
                        st.session_state["ml_selection_input_snapshot"] = {
                            "model_id": (breakthrough_metadata or {}).get("model_id"),
                            "model_version": MODEL_VERSION,
                            "selection_scale": selection_scale,
                            "selection_key": selection_key,
                            "candidate_generation_method": candidate_generation_method,
                            "C0_mg_L": float(sel_C0),
                            "C_breakthrough_ratio": float(sel_cbr),
                            "required_service_time_h": float(sel_required_h),
                            "flow_min": float(sel_flow_min),
                            "flow_max": float(sel_flow_max),
                            "temperature_min": float(sel_temp_min),
                            "temperature_max": float(sel_temp_max),
                            "D_min": float(sel_d_min),
                            "D_max": float(sel_d_max),
                            "H_min": float(sel_h_min),
                            "H_max": float(sel_h_max),
                            "max_mass": float(sel_max_mass),
                            "n_candidates": int(n_candidates),
                        }
                        progress_bar.progress(1.0)
                        status.update(label=tr(lang, "adsorber_selection_complete"), state="complete")
                        if not suitable_candidates_df.empty:
                            st.success(tr(lang, "ml_candidates_found").format(count=len(suitable_candidates_df)))
                        else:
                            st.warning(tr(lang, "ml_selection_no_suitable"))
                    except Exception as exc:
                        status.update(label=tr(lang, "adsorber_selection_error"), state="error")
                        st.error(str(exc))
                        with st.expander(tr(lang, "show_technical_details")):
                            st.code(repr(exc), language="text")
            candidates_df = st.session_state.get("ml_candidates_df")
            all_candidates_df = st.session_state.get("ml_all_candidates_df")
            verification_candidates_df = st.session_state.get("ml_verification_candidates_df")
            rejected_candidates_df = st.session_state.get("ml_rejected_candidates_df")
            rejection_reasons_df = st.session_state.get("ml_rejection_reasons_df")
            diagnostics_df = st.session_state.get("ml_selection_diagnostics_df")
            candidate_domain_df = st.session_state.get("ml_candidate_domain_df")
            schema_diagnostics_df = st.session_state.get("ml_schema_diagnostics_df")
            unit_diagnostics = st.session_state.get("ml_unit_diagnostics") or {}
            display_scale = st.session_state.get("ml_last_selection_scale", selection_scale)
            input_snapshot = st.session_state.get("ml_selection_input_snapshot")
            if isinstance(input_snapshot, dict) and input_snapshot:
                with st.expander(tr(lang, "ml_selection_input_snapshot"), expanded=False):
                    st.dataframe(pd.DataFrame([input_snapshot]), width="stretch", hide_index=True)
            if isinstance(diagnostics_df, pd.DataFrame) and not diagnostics_df.empty:
                st.markdown(f"#### {tr(lang, 'ml_selection_diagnostics')}")
                st.dataframe(diagnostics_df, width="stretch", hide_index=True)
            if isinstance(candidate_domain_df, pd.DataFrame) and not candidate_domain_df.empty:
                st.markdown(f"#### {tr(lang, 'ml_candidate_domain_diagnostics')}")
                st.info(tr(lang, "ml_not_user_requested_feature"))
                st.dataframe(candidate_domain_df, width="stretch", hide_index=True)
            if isinstance(schema_diagnostics_df, pd.DataFrame) and not schema_diagnostics_df.empty:
                with st.expander(tr(lang, "ml_schema_diagnostics"), expanded=False):
                    conversions = unit_diagnostics.get("applied_unit_conversions") or []
                    mismatches = unit_diagnostics.get("detected_unit_mismatches") or []
                    if mismatches:
                        st.warning(tr(lang, "unit_mismatch_detected") + ": " + ", ".join(map(str, mismatches)))
                    if conversions:
                        st.success(tr(lang, "unit_conversion_applied") + ": " + ", ".join(map(str, conversions)))
                    if any("carbon_mass_g" in str(item) and "kg" in str(item) for item in conversions):
                        st.info(tr(lang, "carbon_mass_model_kg_display_g"))
                    st.dataframe(schema_diagnostics_df, width="stretch", hide_index=True)
            if isinstance(rejection_reasons_df, pd.DataFrame) and not rejection_reasons_df.empty:
                st.markdown(f"#### {tr(lang, 'ml_rejection_reason_counts')}")
                st.dataframe(
                    rejection_reasons_df.rename(
                        columns={"reason": tr(lang, "reason"), "count": tr(lang, "count")}
                    ),
                    width="stretch",
                    hide_index=True,
                )
            if isinstance(all_candidates_df, pd.DataFrame) and not all_candidates_df.empty:
                display_prediction_histogram(
                    all_candidates_df,
                    float(st.session_state.get("ml_selection_required_h", sel_required_h)),
                    lang,
                )
            if isinstance(candidates_df, pd.DataFrame) and not candidates_df.empty:
                recommended_candidate = candidates_df.sort_values("balanced_score").iloc[0]
                st.markdown(f"#### {tr(lang, 'ml_recommended_adsorber')}")
                recommended_payload = format_adsorber_candidates_for_display(
                    pd.DataFrame([recommended_candidate]),
                    display_scale,
                )
                st.dataframe(recommended_payload, width="stretch", hide_index=True)
                st.info(tr(lang, "ml_recommended_adsorber_reason"))
                top_alternatives = candidates_df.sort_values("balanced_score").head(3)
                if len(top_alternatives) > 1:
                    st.markdown(f"#### {tr(lang, 'ml_top3_alternatives')}")
                    st.dataframe(
                        format_adsorber_candidates_for_display(top_alternatives, display_scale),
                        width="stretch",
                        height=180,
                    )
                st.markdown(f"#### {tr(lang, 'ml_suitable_candidates')}")
                display_df = format_adsorber_candidates_for_display(candidates_df, display_scale)
                st.dataframe(
                    display_df,
                    width="stretch",
                    height=350,
                )
            if isinstance(verification_candidates_df, pd.DataFrame) and not verification_candidates_df.empty:
                st.markdown(f"#### {tr(lang, 'ml_verification_candidates')}")
                st.info(tr(lang, "ml_verification_candidates_info"))
                st.dataframe(format_adsorber_candidates_for_display(verification_candidates_df, display_scale), width="stretch", height=350)
                candidates_df = pd.concat([candidates_df if isinstance(candidates_df, pd.DataFrame) else pd.DataFrame(), verification_candidates_df], ignore_index=True)
            if (
                (not isinstance(st.session_state.get("ml_candidates_df"), pd.DataFrame) or st.session_state.get("ml_candidates_df").empty)
                and isinstance(all_candidates_df, pd.DataFrame)
                and not all_candidates_df.empty
            ):
                st.warning(tr(lang, "ml_selection_no_suitable"))
                best_predicted = pd.to_numeric(all_candidates_df["predicted_breakthrough_time_h"], errors="coerce").max()
                required_h = float(st.session_state.get("ml_selection_required_h", sel_required_h))
                if best_predicted < required_h:
                    st.info(
                        tr(lang, "ml_max_predicted_less_required").format(
                            predicted=f"{best_predicted:.3g}",
                            required=f"{required_h:.3g}",
                        )
                    )
                else:
                    st.info(tr(lang, "ml_time_ok_but_other_constraints"))
                if isinstance(rejection_reasons_df, pd.DataFrame) and not rejection_reasons_df.empty:
                    top_reason = rejection_reasons_df.sort_values("count", ascending=False).iloc[0]
                    st.warning(
                        tr(lang, "ml_main_rejection_reason").format(
                            reason=str(top_reason.get("reason")),
                            count=int(top_reason.get("count", 0)),
                        )
                    )
                st.info(tr(lang, "ml_expand_dataset_or_ranges"))
            if isinstance(rejected_candidates_df, pd.DataFrame) and not rejected_candidates_df.empty:
                st.markdown(f"#### {tr(lang, 'ml_rejected_candidates')}")
                st.dataframe(format_adsorber_candidates_for_display(rejected_candidates_df, display_scale), width="stretch", height=350)
                if not isinstance(candidates_df, pd.DataFrame) or candidates_df.empty:
                    candidates_df = rejected_candidates_df
            if isinstance(candidates_df, pd.DataFrame) and not candidates_df.empty:
                candidate_id = st.selectbox("candidate_id", candidates_df["candidate_id"].tolist(), key="ml_candidate_verify_id")
                curve_model = st.session_state.get("trained_curve_model") or st.session_state.get("ml_curve_model")
                curve_metadata = st.session_state.get("trained_curve_metadata") or st.session_state.get("ml_curve_metadata")
                if curve_model is not None and isinstance(curve_metadata, dict) and curve_metadata.get("task_type") == "curve":
                    if st.button(tr(lang, "ml_check_curve_shape"), key="ml_check_curve_shape"):
                        candidate = candidates_df[candidates_df["candidate_id"] == candidate_id].iloc[0]
                        max_time_min = max(float(candidate["predicted_breakthrough_time_min"]) * 1.5, float(sel_required_h) * 60.0, 60.0)
                        time_grid = np.linspace(0.0, max_time_min, 160)
                        curve_rows = []
                        for t_value in time_grid:
                            row_data = {
                                key.replace("feature_", ""): candidate[key]
                                for key in candidate.index
                                if str(key).startswith("feature_")
                            }
                            row_data["time_min"] = float(t_value)
                            curve_rows.append(row_data)
                        curve_df = pd.DataFrame(curve_rows)
                        missing_curve = missing_features_for_model(curve_df, curve_metadata)
                        if missing_curve:
                            st.error(tr(lang, "ml_missing_model_features").format(columns=", ".join(missing_curve)))
                        else:
                            try:
                                curve_pred = np.clip(predict_with_metadata(curve_model, curve_df, curve_metadata), 0.0, 1.2)
                                curve_bt = breakthrough_status(time_grid, curve_pred, float(sel_cbr))
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.plot(time_grid, curve_pred, linewidth=2.0, label=tr(lang, "model_curve"))
                                ax.axhline(float(sel_cbr), linestyle="--", color="red", label=tr(lang, "breakthrough_boundary"))
                                if curve_bt.get("time_min") is not None:
                                    ax.axvline(float(curve_bt["time_min"]), linestyle=":", color="black", label="t_b")
                                ax.set_xlabel(tr(lang, "time_min"))
                                ax.set_ylabel("C/C0")
                                ax.legend()
                                st.pyplot(fig)
                                plt.close(fig)
                                st.dataframe(
                                    pd.DataFrame(
                                        [
                                            {
                                                "breakthrough_model_t_b_min": float(candidate["predicted_breakthrough_time_min"]),
                                                "curve_model_t_b_min": curve_bt.get("time_min"),
                                            }
                                        ]
                                    ),
                                    width="stretch",
                                    hide_index=True,
                                )
                            except Exception as exc:
                                st.error(tr(lang, "ml_prediction_failed"))
                                with st.expander(tr(lang, "show_technical_details")):
                                    st.code(repr(exc), language="text")
                if st.button(tr(lang, "ml_verify_with_simulator"), key="ml_verify_with_simulator"):
                    candidate = candidates_df[candidates_df["candidate_id"] == candidate_id].iloc[0]
                    try:
                        sim_result = verify_candidate(candidate, breakthrough_metadata or {})
                    except (ValueError, OSError, KeyError, RuntimeError) as exc:
                        st.error(tr(lang, "ml_verification_alignment_failed"))
                        st.code(str(exc), language="text")
                    else:
                        st.session_state["ml_last_mechanistic_verification"] = sim_result
                        st.caption(tr(lang, "ml_verification_source_grid"))
                        fig = plot_simulation_curve(
                            sim_result["time_min"], sim_result["C_out_C0"],
                            sim_result["configuration"]["breakthrough_ratio"], sim_result["breakthrough"],
                            str(candidate["scale_mode"]), title=tr(lang, "ml_physical_verification"),
                            ebct_min=float(candidate["EBCT_min"]), lang=lang,
                        )
                        st.pyplot(fig)
                        st.dataframe(pd.DataFrame([sim_result["record"]])[[
                            "ml_breakthrough_time_min", "mechanistic_breakthrough_time_min",
                            "mechanistic_bound_type", "mechanistic_observation_bound_min",
                            "ML_minus_mechanistic_min", "mechanistic_passes_required_time",
                        ]], width="stretch", hide_index=True)
                        with st.expander(tr(lang, "show_technical_details")):
                            st.json(sim_result["configuration"])
                            st.json(json.loads(sim_result["record"]["effective_parameters_json"]))
                        download_button_persistent(
                            tr(lang, "ml_download_aligned_verification"), verification_export_bytes(sim_result),
                            "adsorber_mechanistic_verification.csv", "text/csv", "ml_aligned_verification_csv")
                        plt.close(fig)


_startup_probe.mark("help_begin")
with tab_help:
    st.subheader(tr(lang, "help_title"))
    render_help(st, lang)

    st.markdown(f"### {tr(lang, 'unit_check')}")
    unit_check = verify_unit_conversions()
    st.dataframe(pd.DataFrame([{k: v for k, v in unit_check.items() if k != "warnings"}]), width="stretch", hide_index=True)
    for warning in unit_check.get("warnings", []):
        st.info(fix_mojibake(warning))

    st.markdown(f"### {tr(lang, 'library_status')}")
    dependency_status = pd.DataFrame(check_dependencies())
    st.dataframe(
        dependency_status[
            [
                "library",
                "import_name",
                "package_name",
                "available",
                "required",
                "purpose",
                "fallback",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    missing_optional = dependency_status[
        (~dependency_status["available"]) & (~dependency_status["required"])
    ]
    if not missing_optional.empty:
        st.warning(tr(lang, "install_optional_warning"))
        if st.button(
            tr(lang, "install_missing_optional"),
            key="install_missing_optional_dependencies",
        ):
            with st.spinner(tr(lang, "installing_optional")):
                install_result = install_missing_dependencies(only_optional=True)
            st.code(install_result.get("logs") or tr(lang, "no_install_actions"), language="text")
            summary_df = pd.DataFrame(
                [
                    {
                        "installed": ", ".join(install_result["installed"]) or "-",
                        "already_available": ", ".join(install_result["already_available"]) or "-",
                        "failed_optional": ", ".join(install_result["failed_optional"]) or "-",
                        "failed_required": ", ".join(install_result["failed_required"]) or "-",
                    }
                ]
            )
            st.dataframe(summary_df, width="stretch", hide_index=True)
            if install_result["failed_optional"]:
                st.warning(tr(lang, "optional_install_failed"))
            st.success(tr(lang, "install_finished_restart"))
    else:
        st.success(tr(lang, "all_optional_available"))

    st.info(tr(lang, "windows_pyd_ml_note"))
    if st.button(tr(lang, "text_integrity_check_button"), key="run_text_integrity_check"):
        result = subprocess.run(
            [sys.executable, "check_text_integrity.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        output = "\n".join(part for part in [result.stdout, result.stderr] if part)
        if result.returncode == 0:
            st.success(tr(lang, "text_integrity_ok"))
        else:
            st.error(tr(lang, "text_integrity_failed"))
        st.code(output or tr(lang, "text_integrity_no_output"), language="text")

    st.markdown(f"### {tr(lang, 'scientific_verification_tools')}")
    st.markdown(tr(lang, "scientific_verification_tools_body"))

_startup_probe.finish(st)

# -*- coding: utf-8 -*-
"""Starting sorption parameters for adsorbent/pollutant pairs.

These profiles are engineering starting points, not validated constants for a
specific carbon. User calibration data must always override literature defaults.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from parameter_provenance import parameter_origins, provenance_record, derived_record, declared_profile_ratio


DEFAULT_KLDF_1_MIN = 0.014
DEFAULT_DAX_CM2_MIN = 0.00026
DEFAULT_FALLBACK_KH_L_G = 0.05
DEFAULT_FALLBACK_QMAX_MG_G = 200.0
DEFAULT_FALLBACK_DELTAH_KJ_MOL = 20.0
DEFAULT_FALLBACK_EA_KJ_MOL = 20.0
DEFAULT_D_M_REF_M2_S = 5e-10


SORPTION_PARAMETER_PROFILES = {
    "chemviron_natural": {
        "pollutant": "methylene_blue",
        "preferred_isotherm": "Henry",
        "KH_default_L_g": 0.048,
        "KH_range_L_g": (0.02, 0.10),
        "qmax_default_mg_g": 200.0,
        "qmax_range_mg_g": (180.0, 230.0),
        "b_default_L_mg": 0.048 / 200.0,
        "parameter_provenance": {"b_L_mg": declared_profile_ratio("chemviron_natural", 0.048, 200.0)},
        "deltaH_default_kJ_mol": 20.0,
        "deltaH_range_kJ_mol": (-20.0, 35.0),
        "Ea_kLDF_default_kJ_mol": 20.0,
        "Ea_kLDF_range_kJ_mol": (10.0, 35.0),
        "D_m_ref_m2_s": DEFAULT_D_M_REF_M2_S,
        "source_type": "user_adsorbent_profile_plus_calibration_start",
        "confidence": "medium_if_calibrated_else_low",
        "notes_key": "sorption_profile_chemviron_notes",
        "references": [
            "User passport: methylene blue adsorption 180-230 mg/g",
            "User calibration data should override this profile",
        ],
    },
    "coconut_generic": {
        "pollutant": "methylene_blue",
        "preferred_isotherm": "Langmuir",
        "KH_default_L_g": 0.05,
        "KH_range_L_g": (0.01, 0.20),
        "qmax_default_mg_g": 250.0,
        "qmax_range_mg_g": (150.0, 350.0),
        "b_default_L_mg": 0.05 / 250.0,
        "parameter_provenance": {"b_L_mg": declared_profile_ratio("coconut_generic", 0.05, 250.0)},
        "deltaH_default_kJ_mol": 20.0,
        "deltaH_range_kJ_mol": (-20.0, 35.0),
        "Ea_kLDF_default_kJ_mol": 20.0,
        "Ea_kLDF_range_kJ_mol": (10.0, 35.0),
        "D_m_ref_m2_s": DEFAULT_D_M_REF_M2_S,
        "source_type": "literature_generic",
        "confidence": "low",
        "notes_key": "sorption_profile_coconut_notes",
        "references": [
            "Generic literature values for methylene blue adsorption on coconut-shell activated carbon"
        ],
    },
    "wood_generic": {
        "pollutant": "methylene_blue",
        "preferred_isotherm": "Langmuir",
        "KH_default_L_g": 0.04,
        "KH_range_L_g": (0.005, 0.15),
        "qmax_default_mg_g": 180.0,
        "qmax_range_mg_g": (80.0, 300.0),
        "b_default_L_mg": 0.04 / 180.0,
        "parameter_provenance": {"b_L_mg": declared_profile_ratio("wood_generic", 0.04, 180.0)},
        "deltaH_default_kJ_mol": 20.0,
        "deltaH_range_kJ_mol": (-20.0, 35.0),
        "Ea_kLDF_default_kJ_mol": 20.0,
        "Ea_kLDF_range_kJ_mol": (10.0, 35.0),
        "D_m_ref_m2_s": DEFAULT_D_M_REF_M2_S,
        "source_type": "literature_generic",
        "confidence": "low",
        "notes_key": "sorption_profile_wood_notes",
    },
    "coal_generic": {
        "pollutant": "methylene_blue",
        "preferred_isotherm": "Langmuir",
        "KH_default_L_g": 0.05,
        "KH_range_L_g": (0.005, 0.20),
        "qmax_default_mg_g": 220.0,
        "qmax_range_mg_g": (100.0, 350.0),
        "b_default_L_mg": 0.05 / 220.0,
        "parameter_provenance": {"b_L_mg": declared_profile_ratio("coal_generic", 0.05, 220.0)},
        "deltaH_default_kJ_mol": 20.0,
        "deltaH_range_kJ_mol": (-20.0, 35.0),
        "Ea_kLDF_default_kJ_mol": 20.0,
        "Ea_kLDF_range_kJ_mol": (10.0, 35.0),
        "D_m_ref_m2_s": DEFAULT_D_M_REF_M2_S,
        "source_type": "literature_generic",
        "confidence": "low",
        "notes_key": "sorption_profile_coal_notes",
    },
}


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _best_experiment_row(fitted_df: pd.DataFrame | None) -> pd.Series | None:
    if fitted_df is None or fitted_df.empty:
        return None
    required = {"KH_L_g", "kLDF_1_min", "Dax_cm2_min"}
    if not required.issubset(fitted_df.columns):
        return None
    candidates = fitted_df.copy()
    from model import MODEL_VERSION
    if "model_version" not in candidates:
        return None
    candidates = candidates[candidates["model_version"].eq(MODEL_VERSION)]
    if "optimizer_success" in candidates:
        candidates = candidates[candidates["optimizer_success"].astype(str).str.lower().isin({"true", "1"})]
    if "calibration_mode" in candidates.columns:
        clean = candidates[candidates["calibration_mode"].astype(str).eq("clean_bed")]
        if not clean.empty:
            candidates = clean
    if "breakthrough_status_exp" in candidates.columns:
        candidates = candidates[candidates["breakthrough_status_exp"].astype(str).eq("reached_by_interpolation")]
    if "breakthrough_status_model" in candidates.columns:
        candidates = candidates[candidates["breakthrough_status_model"].astype(str).eq("reached_by_interpolation")]
    for flag_col in ("nonphysical_initial_flag", "exclude_from_ml_ranges"):
        if flag_col in candidates.columns:
            candidates = candidates[~candidates[flag_col].astype(str).str.lower().isin({"true", "1", "yes", "y"})]
    if candidates.empty:
        return None
    candidates["_rmse_rank"] = pd.to_numeric(candidates.get("RMSE_C0"), errors="coerce")
    if candidates["_rmse_rank"].isna().all() and "RMSE/C0" in candidates.columns:
        candidates["_rmse_rank"] = pd.to_numeric(candidates["RMSE/C0"], errors="coerce")
    candidates["_rmse_rank"] = candidates["_rmse_rank"].fillna(float("inf"))
    candidates["_r2_rank"] = pd.to_numeric(candidates.get("R2"), errors="coerce").fillna(-float("inf"))
    candidates["_KH"] = pd.to_numeric(candidates["KH_L_g"], errors="coerce")
    candidates["_kLDF"] = pd.to_numeric(candidates["kLDF_1_min"], errors="coerce")
    candidates["_Dax"] = pd.to_numeric(candidates["Dax_cm2_min"], errors="coerce")
    candidates = candidates[
        (candidates["_r2_rank"] >= 0.90)
        & (candidates["_rmse_rank"] <= 0.03)
        & candidates["_KH"].between(0.001, 0.5)
        & candidates["_kLDF"].between(1e-4, 10.0)
        & candidates["_Dax"].between(1e-6, 100.0)
    ]
    candidates = candidates.sort_values(["_rmse_rank", "_r2_rank"], ascending=[True, False])
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _profile_payload(profile_key: str, source: str, confidence: str) -> dict[str, Any]:
    profile = SORPTION_PARAMETER_PROFILES.get(profile_key) or {}
    declared_profile = bool(profile)
    if not profile:
        profile = {
            "pollutant": "methylene_blue",
            "preferred_isotherm": "Henry",
            "KH_default_L_g": DEFAULT_FALLBACK_KH_L_G,
            "KH_range_L_g": (0.005, 0.20),
            "qmax_default_mg_g": DEFAULT_FALLBACK_QMAX_MG_G,
            "qmax_range_mg_g": (80.0, 350.0),
            "b_default_L_mg": DEFAULT_FALLBACK_KH_L_G / DEFAULT_FALLBACK_QMAX_MG_G,
            "deltaH_default_kJ_mol": DEFAULT_FALLBACK_DELTAH_KJ_MOL,
            "deltaH_range_kJ_mol": (-20.0, 35.0),
            "Ea_kLDF_default_kJ_mol": DEFAULT_FALLBACK_EA_KJ_MOL,
            "Ea_kLDF_range_kJ_mol": (10.0, 35.0),
            "D_m_ref_m2_s": DEFAULT_D_M_REF_M2_S,
            "notes_key": "sorption_profile_generic_notes",
        }
    kh_range = profile.get("KH_range_L_g", (None, None))
    qmax_range = profile.get("qmax_range_mg_g", (None, None))
    payload = {
        "isotherm_model": profile.get("preferred_isotherm", "Henry"),
        "KH_L_g": float(profile.get("KH_default_L_g", DEFAULT_FALLBACK_KH_L_G)),
        "kLDF_1_min": DEFAULT_KLDF_1_MIN,
        "Dax_cm2_min": DEFAULT_DAX_CM2_MIN,
        "qmax_mg_g": float(profile.get("qmax_default_mg_g", DEFAULT_FALLBACK_QMAX_MG_G)),
        "b_L_mg": float(profile.get("b_default_L_mg", DEFAULT_FALLBACK_KH_L_G / DEFAULT_FALLBACK_QMAX_MG_G)),
        "deltaH_ads_kJ_mol": float(profile.get("deltaH_default_kJ_mol", DEFAULT_FALLBACK_DELTAH_KJ_MOL)),
        "Ea_kLDF_kJ_mol": float(profile.get("Ea_kLDF_default_kJ_mol", DEFAULT_FALLBACK_EA_KJ_MOL)),
        "D_m_ref_m2_s": float(profile.get("D_m_ref_m2_s", DEFAULT_D_M_REF_M2_S)),
        "KH_range_min_L_g": kh_range[0],
        "KH_range_max_L_g": kh_range[1],
        "qmax_range_min_mg_g": qmax_range[0],
        "qmax_range_max_mg_g": qmax_range[1],
        "parameter_source": source,
        "parameter_confidence": confidence,
        "parameter_notes": profile.get("notes_key", "sorption_profile_generic_notes"),
    }
    fields = {
        "KH_L_g": "KH_default_L_g", "qmax_mg_g": "qmax_default_mg_g",
        "b_L_mg": "b_default_L_mg", "deltaH_ads_kJ_mol": "deltaH_default_kJ_mol",
        "Ea_kLDF_kJ_mol": "Ea_kLDF_default_kJ_mol", "D_m_ref_m2_s": "D_m_ref_m2_s",
    }
    records = {}
    documented = parameter_origins(profile)
    for key, field in fields.items():
        if declared_profile and field in profile:
            records[key] = documented.get(key) or provenance_record(
                "profile", method="profile_declared", profile_key=profile_key, profile_field=field,
                profile_reference_notes=profile.get("references"),
            )
        else:
            records[key] = provenance_record("default", method="fallback", assignment_path="_profile_payload", missing_field=field)
    for key, constant in (("kLDF_1_min", "DEFAULT_KLDF_1_MIN"), ("Dax_cm2_min", "DEFAULT_DAX_CM2_MIN")):
        records[key] = provenance_record("default", method="constant_default", default_name=constant)
    if records["b_L_mg"]["origin"] == "default":
        defaults = {key: provenance_record("default", method="constant_default", default_name=key)
                    for key in ("DEFAULT_FALLBACK_KH_L_G", "DEFAULT_FALLBACK_QMAX_MG_G")}
        records["b_L_mg"] = derived_record("b_L_mg = DEFAULT_FALLBACK_KH_L_G / DEFAULT_FALLBACK_QMAX_MG_G", defaults, defaults,
                                         dependency_values={"DEFAULT_FALLBACK_KH_L_G": DEFAULT_FALLBACK_KH_L_G,
                                                            "DEFAULT_FALLBACK_QMAX_MG_G": DEFAULT_FALLBACK_QMAX_MG_G})
    payload["parameter_provenance"] = records
    return payload


def _calibration_payload(row_or_dict: Any, source: str, confidence: str) -> dict[str, Any]:
    def get(*names: str, default: float | None = None) -> float | None:
        for name in names:
            if isinstance(row_or_dict, dict):
                value = row_or_dict.get(name)
            else:
                value = row_or_dict.get(name, None)
            parsed = _finite_float(value)
            if parsed is not None:
                return parsed
        return default

    kh = get("KH_L_g", "KH_ref_L_g", default=DEFAULT_FALLBACK_KH_L_G)
    qmax = get("qmax_mg_g", default=DEFAULT_FALLBACK_QMAX_MG_G)
    payload = {
        "isotherm_model": "Henry",
        "KH_L_g": kh,
        "kLDF_1_min": get("kLDF_1_min", "kLDF_ref_1_min", default=DEFAULT_KLDF_1_MIN),
        "Dax_cm2_min": get("Dax_cm2_min", "Dax_ref_cm2_min", default=DEFAULT_DAX_CM2_MIN),
        "qmax_mg_g": qmax,
        "b_L_mg": None if kh is None or qmax in (None, 0) else kh / qmax,
        "deltaH_ads_kJ_mol": get("deltaH_ads_kJ_mol", default=DEFAULT_FALLBACK_DELTAH_KJ_MOL),
        "Ea_kLDF_kJ_mol": get("Ea_kLDF_kJ_mol", default=DEFAULT_FALLBACK_EA_KJ_MOL),
        "D_m_ref_m2_s": get("D_m_ref_m2_s", default=DEFAULT_D_M_REF_M2_S),
        "T_ref_C": get("T_ref_C", "T_exp_C", "temperature_C", default=20.0),
        "T_exp_C": get("T_exp_C", "T_ref_C", "temperature_C", default=20.0),
        "KH_range_min_L_g": None,
        "KH_range_max_L_g": None,
        "qmax_range_min_mg_g": None,
        "qmax_range_max_mg_g": None,
        "parameter_source": source,
        "parameter_confidence": confidence,
        "parameter_notes": "sorption_profile_calibrated_notes",
    }
    aliases = {
        "KH_L_g": ("KH_L_g", "KH_ref_L_g"),
        "kLDF_1_min": ("kLDF_1_min", "kLDF_ref_1_min"),
        "Dax_cm2_min": ("Dax_cm2_min", "Dax_ref_cm2_min"),
        "qmax_mg_g": ("qmax_mg_g",), "deltaH_ads_kJ_mol": ("deltaH_ads_kJ_mol",),
        "Ea_kLDF_kJ_mol": ("Ea_kLDF_kJ_mol",), "D_m_ref_m2_s": ("D_m_ref_m2_s",),
        "T_ref_C": ("T_ref_C", "T_exp_C", "temperature_C"),
        "T_exp_C": ("T_exp_C", "T_ref_C", "temperature_C"),
    }
    records = {}
    documented = parameter_origins(row_or_dict)
    experiment_id = row_or_dict.get("experiment_id")
    for key, names in aliases.items():
        field = next((name for name in names if _finite_float(row_or_dict.get(name)) is not None), None)
        if field is None:
            records[key] = provenance_record("default", method="fallback", assignment_path="_calibration_payload", missing_fields=list(names))
        elif key in documented:
            records[key] = documented[key]
        elif field in documented:
            records[key] = documented[field]
        elif key in {"KH_L_g", "kLDF_1_min", "Dax_cm2_min"}:
            records[key] = provenance_record("fitted", method="calibration_fit", experiment_id=experiment_id, record_field=field)
        else:
            records[key] = provenance_record("unknown", method="stored_record", experiment_id=experiment_id, record_field=field)
        if key in {"T_ref_C", "T_exp_C"} and str(row_or_dict.get("temperature_reference_assumed", "")).lower() in {"true", "1"}:
            records[key] = provenance_record("default", method="fallback", assignment_path="calibration_temperature_reference")
    records["b_L_mg"] = derived_record("b_L_mg = KH_L_g / qmax_mg_g", ("KH_L_g", "qmax_mg_g"), records,
                                       dependency_values={"KH_L_g": kh, "qmax_mg_g": qmax},
                                       defined_when="KH_L_g is not None and qmax_mg_g not in (None, 0)")
    payload["parameter_provenance"] = records
    return payload


def resolve_sorption_parameters(
    adsorbent_profile_key: str,
    pollutant: str,
    parameter_source_mode: str,
    fitted_df: pd.DataFrame | None = None,
    last_calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mode = parameter_source_mode
    pollutant_key = "methylene_blue" if pollutant in {"метиленовый синий", "methylene_blue", "methylene blue"} else pollutant

    def from_last() -> dict[str, Any] | None:
        from model import MODEL_VERSION
        if last_calibration and last_calibration.get("model_version") == MODEL_VERSION:
            payload = _calibration_payload(last_calibration, "last_calibration", "low_to_medium")
            payload["experiment_id"] = last_calibration.get("experiment_id")
            return payload
        return None

    def from_best() -> dict[str, Any] | None:
        row = _best_experiment_row(fitted_df)
        if row is None:
            return None
        confidence = "low_to_medium"
        payload = _calibration_payload(row, "best_experiment", confidence)
        payload["experiment_id"] = row.get("experiment_id")
        return payload

    if mode == "last_calibration":
        return from_last() or _profile_payload(adsorbent_profile_key, "generic_fallback", "low")
    if mode == "best_experiment":
        return from_best() or _profile_payload(adsorbent_profile_key, "generic_fallback", "low")
    if mode == "adsorbent_profile":
        source = "chemviron_profile_start" if adsorbent_profile_key == "chemviron_natural" else "literature_generic"
        confidence = "low_to_medium" if adsorbent_profile_key == "chemviron_natural" else "low"
        return _profile_payload(adsorbent_profile_key, source, confidence)
    if mode == "literature_generic":
        key = adsorbent_profile_key if adsorbent_profile_key in SORPTION_PARAMETER_PROFILES else "coal_generic"
        return _profile_payload(key, "literature_generic", "low")
    if mode == "manual":
        return _profile_payload(adsorbent_profile_key, "manual", "user_defined")

    if mode == "auto_best_available":
        result = from_last()
        if result is not None:
            return result
        result = from_best()
        if result is not None:
            return result
        if adsorbent_profile_key == "chemviron_natural":
            return _profile_payload(adsorbent_profile_key, "chemviron_profile_start", "low_to_medium")
        if adsorbent_profile_key in SORPTION_PARAMETER_PROFILES:
            return _profile_payload(adsorbent_profile_key, "literature_generic", "low")

    return _profile_payload(adsorbent_profile_key, "generic_fallback", "low")

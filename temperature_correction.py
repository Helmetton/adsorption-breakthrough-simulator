# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pandas as pd

from encoding_utils import safe_read_csv
from industrial_model import (
    R_GAS_J_MOL_K,
    arrhenius_correction,
    henry_temperature_correction,
    molecular_diffusion_temperature_correction,
    water_properties,
)
from unit_conversions import m2_s_to_cm2_min
from parameter_provenance import provenance_record


DEFAULT_T_REF_C = 20.0
DEFAULT_DELTAH_ADS_KJ_MOL = 20.0
DEFAULT_EA_KLDF_KJ_MOL = 20.0
DEFAULT_D_M_REF_M2_S = 5e-10


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def load_experiment_temperature_map(
    fitted_path: str = "fitted_experiment_parameters.csv",
    data_path: str = "cleaned_dynamic_sorption_data.csv",
) -> dict[str, float]:
    """Return experiment_id -> reference experiment temperature in deg C."""
    source_path = fitted_path if os.path.exists(fitted_path) else data_path
    if not os.path.exists(source_path):
        return {}

    df = safe_read_csv(source_path)
    if "experiment_id" not in df.columns:
        return {}

    if "T_ref_C" in df.columns:
        temperature_col = "T_ref_C"
    elif "T_exp_C" in df.columns:
        temperature_col = "T_exp_C"
    elif "temperature_C" in df.columns:
        temperature_col = "temperature_C"
    else:
        return {
            str(experiment_id): DEFAULT_T_REF_C
            for experiment_id in df["experiment_id"].dropna().unique()
        }

    result: dict[str, float] = {}
    for experiment_id, group in df.groupby("experiment_id"):
        values = pd.to_numeric(group[temperature_col], errors="coerce").dropna()
        if values.empty:
            result[str(experiment_id)] = DEFAULT_T_REF_C
        else:
            result[str(experiment_id)] = float(values.median())
    return result


def experiment_temperature_warnings(
    fitted_path: str = "fitted_experiment_parameters.csv",
    data_path: str = "cleaned_dynamic_sorption_data.csv",
) -> list[str]:
    source_path = fitted_path if os.path.exists(fitted_path) else data_path
    if not os.path.exists(source_path):
        return ["Температура эксперимента не найдена. Использовано T_ref = 20 °C."]
    df = safe_read_csv(source_path)
    if "experiment_id" not in df.columns:
        return ["Температура эксперимента не найдена. Использовано T_ref = 20 °C."]
    if "T_ref_C" in df.columns:
        temperature_col = "T_ref_C"
    elif "T_exp_C" in df.columns:
        temperature_col = "T_exp_C"
    elif "temperature_C" in df.columns:
        temperature_col = "temperature_C"
    else:
        return ["Температура эксперимента не найдена. Использовано T_ref = 20 °C."]

    warnings: list[str] = []
    for experiment_id, group in df.groupby("experiment_id"):
        values = pd.to_numeric(group[temperature_col], errors="coerce").dropna()
        if values.empty:
            warnings.append(f"{experiment_id}: температура эксперимента не найдена. Использовано T_ref = 20 °C.")
        elif float(values.max() - values.min()) > 1.0:
            warnings.append(f"{experiment_id}: температура внутри эксперимента меняется; используется медианное значение.")
    return warnings


def laboratory_temperature_range() -> tuple[float | None, float | None]:
    temperature_map = load_experiment_temperature_map()
    values = [value for value in temperature_map.values() if math.isfinite(value)]
    if not values:
        return None, None
    return float(min(values)), float(max(values))


def _quality_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if "calibration_mode" in df.columns:
        mask &= df["calibration_mode"].astype(str).eq("clean_bed")
    if "RMSE_C0" in df.columns:
        mask &= pd.to_numeric(df["RMSE_C0"], errors="coerce").le(0.05)
    elif "RMSE/C0" in df.columns:
        mask &= pd.to_numeric(df["RMSE/C0"], errors="coerce").le(0.05)
    if "R2" in df.columns:
        r2 = pd.to_numeric(df["R2"], errors="coerce")
        mask &= (r2.ge(0.70) | r2.isna())
    return mask


def _estimate_deltaH_from_calibrations(df: pd.DataFrame) -> float | None:
    required = {"KH_L_g"}
    if not required.issubset(df.columns):
        return None
    if "T_ref_C" in df.columns:
        t_col = "T_ref_C"
    elif "T_exp_C" in df.columns:
        t_col = "T_exp_C"
    elif "temperature_C" in df.columns:
        t_col = "temperature_C"
    else:
        return None

    candidates = df[_quality_mask(df)].copy()
    candidates["T_K"] = pd.to_numeric(candidates[t_col], errors="coerce") + 273.15
    candidates["KH"] = pd.to_numeric(candidates["KH_L_g"], errors="coerce")
    candidates = candidates[(candidates["T_K"] > 0) & (candidates["KH"] > 0)]
    if len(candidates) < 3:
        return None
    if float(candidates["T_K"].max() - candidates["T_K"].min()) < 5.0:
        return None

    x = 1.0 / candidates["T_K"].to_numpy(dtype=float)
    y = np.log(candidates["KH"].to_numpy(dtype=float))
    try:
        slope, _intercept = np.polyfit(x, y, 1)
    except Exception:
        return None
    deltaH_kJ_mol = -float(slope) * R_GAS_J_MOL_K / 1000.0
    if not math.isfinite(deltaH_kJ_mol) or abs(deltaH_kJ_mol) > 60.0:
        return None
    return deltaH_kJ_mol


def resolve_temperature_correction_parameters(
    profile_key: str,
    pollutant: str,
    fitted_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Resolve engineering temperature correction parameters.

    The defaults are intentionally conservative literature-engineering values,
    not measured constants for Chemviron Carbon.
    """
    if fitted_df is None and os.path.exists("fitted_experiment_parameters.csv"):
        try:
            fitted_df = safe_read_csv("fitted_experiment_parameters.csv")
        except Exception:
            fitted_df = None

    if fitted_df is not None and not fitted_df.empty:
        estimated_deltaH = _estimate_deltaH_from_calibrations(fitted_df)
        if estimated_deltaH is not None:
            return {
                "deltaH_ads_kJ_mol": float(estimated_deltaH),
                "Ea_kLDF_kJ_mol": DEFAULT_EA_KLDF_KJ_MOL,
                "D_m_ref_m2_s": DEFAULT_D_M_REF_M2_S,
                "source": "estimated_from_own_calibrations",
                "confidence": "medium",
                "parameter_provenance": {
                    "deltaH_ads_kJ_mol": provenance_record("derived", method="calibration_temperature_regression",
                        formula="deltaH_ads_kJ_mol = -slope(ln(KH), 1/T_K) * R / 1000",
                        dependencies=["calibration.KH_L_g", "calibration.temperature"],
                        assignment_path="_estimate_deltaH_from_calibrations"),
                    "Ea_kLDF_kJ_mol": provenance_record("default", method="constant_default", default_name="DEFAULT_EA_KLDF_KJ_MOL"),
                    "D_m_ref_m2_s": provenance_record("default", method="constant_default", default_name="DEFAULT_D_M_REF_M2_S"),
                },
            }

    return {
        "deltaH_ads_kJ_mol": DEFAULT_DELTAH_ADS_KJ_MOL,
        "Ea_kLDF_kJ_mol": DEFAULT_EA_KLDF_KJ_MOL,
        "D_m_ref_m2_s": DEFAULT_D_M_REF_M2_S,
        "source": "literature_engineering_default",
        "confidence": "low",
        "parameter_provenance": {
            key: provenance_record("default", method="constant_default", default_name=name)
            for key, name in (("deltaH_ads_kJ_mol", "DEFAULT_DELTAH_ADS_KJ_MOL"),
                              ("Ea_kLDF_kJ_mol", "DEFAULT_EA_KLDF_KJ_MOL"),
                              ("D_m_ref_m2_s", "DEFAULT_D_M_REF_M2_S"))
        },
    }


def apply_temperature_adaptation(
    KH_ref_L_g: float,
    kLDF_ref_1_min: float,
    Dax_ref_cm2_min: float,
    T_ref_C: float,
    T_sim_C: float,
    deltaH_ads_kJ_mol: float,
    Ea_kLDF_kJ_mol: float,
    D_m_ref_m2_s: float,
    Dax_temperature_mode: str = "proportional_to_diffusion",
    source: str = "literature_engineering_default",
    confidence: str = "low",
    superficial_velocity_m_s: float | None = None,
    dp_m: float | None = None,
) -> dict[str, Any]:
    KH_used = henry_temperature_correction(KH_ref_L_g, T_ref_C, T_sim_C, deltaH_ads_kJ_mol)
    kLDF_used = arrhenius_correction(kLDF_ref_1_min, T_ref_C, T_sim_C, Ea_kLDF_kJ_mol)

    mu_ref = float(water_properties(T_ref_C)["mu_water_Pa_s"])
    mu_T = float(water_properties(T_sim_C)["mu_water_Pa_s"])
    D_m_used = molecular_diffusion_temperature_correction(
        D_m_ref_m2_s,
        T_ref_C,
        T_sim_C,
        mu_ref,
        mu_T,
    )

    if Dax_temperature_mode == "no_correction":
        Dax_used = float(Dax_ref_cm2_min)
    elif (
        Dax_temperature_mode == "engineering_estimate"
        and superficial_velocity_m_s is not None
        and dp_m is not None
    ):
        Dax_used = m2_s_to_cm2_min(0.5 * float(superficial_velocity_m_s) * float(dp_m) + D_m_used)
    else:
        Dax_used = float(Dax_ref_cm2_min) * D_m_used / max(float(D_m_ref_m2_s), 1e-30)

    return {
        "temperature_correction_mode": "auto",
        "KH_ref_L_g": float(KH_ref_L_g),
        "KH_used_L_g": float(KH_used),
        "KH_temperature_factor": float(KH_used) / max(float(KH_ref_L_g), 1e-30),
        "kLDF_ref_1_min": float(kLDF_ref_1_min),
        "kLDF_used_1_min": float(kLDF_used),
        "kLDF_temperature_factor": float(kLDF_used) / max(float(kLDF_ref_1_min), 1e-30),
        "Dax_ref_cm2_min": float(Dax_ref_cm2_min),
        "Dax_used_cm2_min": float(Dax_used),
        "Dax_temperature_factor": float(Dax_used) / max(float(Dax_ref_cm2_min), 1e-30),
        "T_ref_C": float(T_ref_C),
        "T_sim_C": float(T_sim_C),
        "deltaH_ads_kJ_mol": float(deltaH_ads_kJ_mol),
        "Ea_kLDF_kJ_mol": float(Ea_kLDF_kJ_mol),
        "D_m_ref_m2_s": float(D_m_ref_m2_s),
        "D_m_used_m2_s": float(D_m_used),
        "Dax_temperature_mode": Dax_temperature_mode,
        "temperature_correction_source": source,
        "temperature_correction_confidence": confidence,
    }
